"""Cross-encoder reranking with a deterministic local fallback."""
from __future__ import annotations

import math
import re
from dataclasses import replace

from ..claim_analysis import ClaimProfile
from ..config import EmbeddingConfig, RagConfig
from ..tools.adapters import execute_reranker
from .models import RagPassageHit


class PassageReranker:
    def __init__(self, config: RagConfig, embedding_config: EmbeddingConfig, logger=None):
        self.config = config
        self.embedding_config = embedding_config
        self.logger = logger
        self.used_cross_encoder = False
        self.fallback_reason: str | None = None

    def rerank(
        self,
        query: str,
        hits: list[RagPassageHit],
        profile: ClaimProfile,
    ) -> list[RagPassageHit]:
        candidates = hits[: self.config.reranker_input_k]
        if not candidates:
            return []
        model_scores = self._cross_encoder_scores(query, candidates)
        normalized_model = min_max(model_scores) if model_scores else []
        required = profile.required_concept_terms | profile.foundational_concept_terms
        query_terms = terms(query)
        rescored: list[tuple[float, RagPassageHit]] = []
        for index, hit in enumerate(candidates):
            chunk_terms = terms(hit.chunk.text)
            lexical = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            required_coverage = len(required & chunk_terms) / max(len(required), 1) if required else 0.0
            model_score = normalized_model[index] if index < len(normalized_model) else lexical
            final = (
                0.70 * model_score
                + 0.15 * normalize_rrf(hit.fused_score)
                + 0.10 * hit.chunk.quality_score
                + 0.05 * required_coverage
            )
            rescored.append((final, replace(hit, reranker_score=round(final, 6))))
        rescored.sort(key=lambda item: (-item[0], item[1].chunk.document_id, item[1].chunk.ordinal))
        return [hit for _score, hit in rescored[: self.config.reranker_top_k]]

    def _cross_encoder_scores(self, query: str, hits: list[RagPassageHit]) -> list[float]:
        if not self.config.reranker_enabled:
            self.fallback_reason = "cross-encoder disabled"
            return []
        if self.embedding_config.model_provider != "fastembed":
            self.fallback_reason = "cross-encoder requires fastembed provider"
            return []
        if self.embedding_config.auto_download != "always":
            self.fallback_reason = "cross-encoder model download is not allowed by current policy"
            return []
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            encoder = TextCrossEncoder(
                model_name=self.config.reranker_model_id,
                cache_dir=self.embedding_config.cache_dir,
            )
            texts = [hit.chunk.text for hit in hits]
            scores = execute_reranker(
                query,
                texts,
                lambda: [float(value) for value in encoder.rerank(query, texts)],
            )
            self.used_cross_encoder = len(scores) == len(hits)
            return scores if self.used_cross_encoder else []
        except Exception as exc:  # pylint: disable=broad-except
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger(f"RAG cross-encoder unavailable; deterministic reranker used: {self.fallback_reason}")
            return []


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_+\-]{1,}", text.lower()))


def min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def normalize_rrf(value: float) -> float:
    return max(0.0, min(value * 60.0, 1.0))
