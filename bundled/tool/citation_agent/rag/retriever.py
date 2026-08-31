"""Two-stage BM25/dense retrieval, RRF fusion, and passage reranking."""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace

from ..claim_analysis import ClaimProfile
from ..config import EmbeddingConfig, RagConfig
from ..schemas import EvidenceSnippet
from .index import RagIndex
from .models import RagDocument, RagPassageHit, RagRetrievalResult, RagRetrievalStats
from .reranker import PassageReranker


class RagRetriever:
    def __init__(
        self,
        *,
        workspace_path: str,
        config: RagConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        logger=None,
    ):
        self.config = config or RagConfig.from_env()
        self.embedding_config = embedding_config or EmbeddingConfig.from_env()
        self.logger = logger
        self.index = RagIndex(
            workspace_path=workspace_path,
            config=self.config,
            embedding_config=self.embedding_config,
            logger=logger,
        )

    def retrieve(
        self,
        *,
        claim: str,
        queries: list[str],
        profile: ClaimProfile,
        documents: list[RagDocument],
        rebuild_index: bool = False,
        document_hydrator: Callable[[list[RagDocument]], list[RagDocument]] | None = None,
    ) -> RagRetrievalResult:
        total_started = time.perf_counter()
        if not documents:
            return RagRetrievalResult(
                stats=RagRetrievalStats(fallback_reason="RAG source contains no documents")
            )
        documents_by_id = {document.document_id: document for document in documents}
        metadata_documents = [replace(document, pdf_path=None) for document in documents]
        metadata_started = time.perf_counter()
        sync = self.index.sync(metadata_documents, rebuild_index=rebuild_index)
        metadata_sync_ms = elapsed_ms(metadata_started)
        allowed_ids = {document.document_id for document in documents}
        retrieval_queries = list(dict.fromkeys([claim, *queries]))[:12]

        paper_started = time.perf_counter()
        paper_lexical = self.index.paper_lexical_rank(
            retrieval_queries, allowed_ids, self.config.paper_lexical_top_k
        )
        paper_dense = self.index.paper_dense_rank(
            retrieval_queries, allowed_ids, self.config.paper_dense_top_k
        )
        paper_fused = reciprocal_rank_fusion(
            [(paper_lexical, self.config.bm25_weight), (paper_dense, self.config.dense_weight)],
            self.config.rrf_k,
        )
        paper_retrieval_ms = elapsed_ms(paper_started)
        if not paper_fused:
            return RagRetrievalResult(
                stats=replace(
                    sync,
                    paper_lexical_hits=len(paper_lexical),
                    paper_dense_hits=len(paper_dense),
                    metadata_sync_ms=metadata_sync_ms,
                    paper_retrieval_ms=paper_retrieval_ms,
                    total_ms=elapsed_ms(total_started),
                    fallback_reason=sync.fallback_reason or "RAG paper retrieval returned no matches",
                )
            )
        paper_ids = [identity for identity, _score in paper_fused[: self.config.paper_fused_top_k]]
        paper_score = dict(paper_fused)
        paper_set = set(paper_ids)

        pdf_requested_ids = paper_ids[: max(self.config.pdf_index_top_k, 0)]
        pdf_candidates = [
            documents_by_id[identity]
            for identity in pdf_requested_ids
            if identity in documents_by_id
        ]
        pdf_started = time.perf_counter()
        hydration_error: str | None = None
        if document_hydrator and pdf_candidates:
            try:
                pdf_candidates = document_hydrator(pdf_candidates)
            except Exception as exc:  # pylint: disable=broad-except
                hydration_error = f"PDF hydration failed: {type(exc).__name__}: {exc}"
                if self.logger:
                    self.logger(f"{hydration_error}; abstract-only RAG retained.")
                pdf_candidates = []
        hydrated_documents = [document for document in pdf_candidates if document.pdf_path]
        pdf_sync = self.index.sync(hydrated_documents) if hydrated_documents else RagRetrievalStats()
        pdf_index_ms = elapsed_ms(pdf_started)

        passage_started = time.perf_counter()
        passage_lexical = self.index.chunk_lexical_rank(
            retrieval_queries, paper_set, self.config.passage_lexical_top_k
        )
        passage_dense = self.index.chunk_dense_rank(
            retrieval_queries, paper_set, self.config.passage_dense_top_k
        )
        passage_fused = reciprocal_rank_fusion(
            [(passage_lexical, self.config.bm25_weight), (passage_dense, self.config.dense_weight)],
            self.config.rrf_k,
        )[: self.config.passage_fused_top_k]
        chunks = self.index.load_chunks(identity for identity, _score in passage_fused)
        lexical_positions = {identity: rank for rank, identity in enumerate(passage_lexical, start=1)}
        dense_positions = {identity: rank for rank, identity in enumerate(passage_dense, start=1)}
        hits = [
            RagPassageHit(
                chunk=chunks[chunk_id],
                fused_score=score,
                lexical_rank=lexical_positions.get(chunk_id),
                dense_rank=dense_positions.get(chunk_id),
            )
            for chunk_id, score in passage_fused
            if chunk_id in chunks
        ]
        passage_retrieval_ms = elapsed_ms(passage_started)

        reranker_started = time.perf_counter()
        reranker = PassageReranker(self.config, self.embedding_config, logger=self.logger)
        reranked = reranker.rerank(claim, hits, profile)
        reranker_ms = elapsed_ms(reranker_started)
        selected_hits = diversify_passages(
            reranked or hits,
            max_per_document=self.config.max_passages_per_paper,
        )
        by_document: dict[str, list[RagPassageHit]] = defaultdict(list)
        for hit in selected_hits:
            by_document[hit.chunk.document_id].append(hit)
        indexed_documents_by_id = self.index.load_documents(paper_ids)

        ranked_documents = sorted(
            paper_ids,
            key=lambda identity: (
                -max(
                    [hit.reranker_score or hit.fused_score for hit in by_document.get(identity, [])]
                    or [0.0]
                ),
                -paper_score.get(identity, 0.0),
                identity,
            ),
        )
        suggestions = []
        for document_id in ranked_documents:
            document = indexed_documents_by_id.get(document_id)
            if document is None:
                continue
            document_hits = by_document.get(document_id, [])[: self.config.max_passages_per_paper]
            passages = [hit_to_evidence(hit) for hit in document_hits]
            best_score = max([hit.reranker_score or hit.fused_score for hit in document_hits] or [0.0])
            confidence = min(0.82, 0.50 + min(best_score, 1.0) * 0.32)
            retrieval_mode = "BM25+Dense+RRF"
            if not paper_dense and not passage_dense:
                retrieval_mode = "BM25-only fallback"
            reason = (
                f"RAG retrieval: {retrieval_mode}; paper_rank={paper_ids.index(document_id) + 1}; "
                f"evidence_passages={len(passages)}; cross_encoder={'yes' if reranker.used_cross_encoder else 'fallback'}."
            )
            suggestions.append(
                document.to_suggestion(
                    evidence_passages=passages,
                    confidence=confidence,
                    reason=reason,
                )
            )

        fallback_reasons = [
            item
            for item in [sync.fallback_reason, hydration_error, reranker.fallback_reason]
            if item
        ]
        stats = replace(
            sync,
            paper_lexical_hits=len(paper_lexical),
            paper_dense_hits=len(paper_dense),
            paper_fused_hits=len(paper_ids),
            passage_lexical_hits=len(passage_lexical),
            passage_dense_hits=len(passage_dense),
            passage_fused_hits=len(passage_fused),
            reranked_passages=len(reranked),
            embedding_used=bool(paper_dense or passage_dense),
            reranker_used=reranker.used_cross_encoder,
            pdf_documents_requested=len(pdf_requested_ids),
            pdf_documents_hydrated=len(hydrated_documents),
            documents_indexed=sync.documents_indexed + pdf_sync.documents_indexed,
            documents_reused=sync.documents_reused + pdf_sync.documents_reused,
            documents_failed=sync.documents_failed + pdf_sync.documents_failed,
            chunks_indexed=sync.chunks_indexed + pdf_sync.chunks_indexed,
            metadata_sync_ms=metadata_sync_ms,
            pdf_index_ms=pdf_index_ms,
            paper_retrieval_ms=paper_retrieval_ms,
            passage_retrieval_ms=passage_retrieval_ms,
            reranker_ms=reranker_ms,
            total_ms=elapsed_ms(total_started),
            fallback_reason="; ".join(fallback_reasons) or None,
        )
        return RagRetrievalResult(
            suggestions=suggestions[: self.config.paper_fused_top_k],
            stats=stats,
        )


def reciprocal_rank_fusion(
    rankings: list[tuple[list[str], float]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in rankings:
        for rank, identity in enumerate(ranking, start=1):
            scores[identity] += max(weight, 0.0) / (max(rrf_k, 1) + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def diversify_passages(
    hits: list[RagPassageHit],
    *,
    max_per_document: int,
) -> list[RagPassageHit]:
    counts: dict[str, int] = defaultdict(int)
    selected: list[RagPassageHit] = []
    seen_parents: set[tuple[str, str]] = set()
    for hit in hits:
        identity = hit.chunk.document_id
        if counts[identity] >= max(max_per_document, 1):
            continue
        parent_key = (identity, hit.chunk.parent_id)
        if parent_key in seen_parents and counts[identity] > 0:
            continue
        seen_parents.add(parent_key)
        counts[identity] += 1
        selected.append(hit)
    return selected


def hit_to_evidence(hit: RagPassageHit) -> EvidenceSnippet:
    chunk = hit.chunk
    page = chunk.page_start
    section = chunk.section
    if chunk.page_start and chunk.page_end and chunk.page_end != chunk.page_start:
        section = f"{section} (pp. {chunk.page_start}-{chunk.page_end})"
    return EvidenceSnippet(
        page=page,
        section=section,
        text=chunk.text,
        source="abstract" if chunk.chunk_type == "abstract" else "pdf",
        score=hit.reranker_score or hit.fused_score,
        quality_score=chunk.quality_score,
        quality_flags=list(chunk.quality_flags),
    )


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
