"""Evidence passage retrieval and support judging."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .bibliography import tokenize
from .cache import CachedPdfPassage, EvidenceJudgeCache, PdfPassageCache, stable_text_hash
from .claim_analysis import ClaimProfile, build_claim_profile, score_text_against_claim
from .config import EmbeddingConfig, ModelConfig
from .context import ContextItem, ContextManager, policy_for_stage
from .embedding import rank_evidence_passages
from .embedding.embedder import BaseEmbedder, EmbeddingUnavailable, create_embedder
from .schemas import CitationSuggestion, EvidenceSnippet
from .tools.adapters import execute_litellm_completion, execute_pdf_reader


VERDICT_ORDER = {
    "direct_support": 5,
    "foundational_support": 4,
    "partial_support": 3,
    "related_only": 2,
    "metadata_match": 1,
    "insufficient_evidence": 0,
    "contradiction": 0,
}

VERDICT_CONFIDENCE_CAPS = {
    "direct_support": 0.95,
    "foundational_support": 0.86,
    "partial_support": 0.78,
    "related_only": 0.58,
    "metadata_match": 0.45,
    "insufficient_evidence": 0.35,
    "contradiction": 0.25,
}


@dataclass(frozen=True)
class Passage:
    text: str
    source: str
    section: str | None = None
    page: int | None = None
    score: float = 0.0
    quality_score: float = 1.0
    is_own_claim: bool | None = None
    is_related_work_summary: bool | None = None
    supports_core_mechanism: bool | None = None
    citation_density: float | None = None
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceQuality:
    quality_score: float
    discard: bool
    is_own_claim: bool | None
    is_related_work_summary: bool | None
    supports_core_mechanism: bool | None
    citation_density: float
    flags: tuple[str, ...] = ()


class EvidenceRetriever:
    """Collect short passages from abstract and optional local PDFs."""

    def __init__(
        self,
        max_passages: int = 5,
        logger: Callable[[str], None] | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ):
        self.max_passages = max(max_passages, 1)
        self.logger = logger
        self.embedding_config = embedding_config or EmbeddingConfig.from_env()
        self._embedder: BaseEmbedder | None = None
        self._skip_log_counts: dict[str, int] = {}
        self._pdf_cache = PdfPassageCache()

    def _get_embedder(self) -> BaseEmbedder | None:
        if not self.embedding_config.enabled:
            return None
        if self._embedder is not None:
            return self._embedder
        try:
            self._embedder = create_embedder(self.embedding_config)
        except EmbeddingUnavailable as exc:
            self._log(f"Embedding model unavailable; lexical fallback used: {exc}")
            return None
        return self._embedder

    def retrieve(
        self,
        suggestion: CitationSuggestion,
        profile: ClaimProfile,
        claim: str | None = None,
    ) -> list[EvidenceSnippet]:
        passages: list[Passage] = []
        passages.extend(self._metadata_passages(suggestion, profile))
        rag_passages = self._indexed_passages(suggestion, profile)
        passages.extend(rag_passages)
        if not any(passage.source == "pdf" for passage in rag_passages):
            passages.extend(self._pdf_passages(suggestion, profile))
        passages = dedupe_passages(passages)
        query = evidence_query_text(claim=claim, profile=profile)
        passages = rank_evidence_passages(
            query=query,
            passages=passages,
            config=self.embedding_config,
            logger=self._log,
            embedder=self._get_embedder(),
        )
        passages = sorted(passages, key=passage_sort_key)
        return [
            EvidenceSnippet(
                page=passage.page,
                section=passage.section,
                text=passage.text,
                source=passage.source if passage.source in {"metadata", "abstract", "pdf"} else "unknown",
                score=round(passage.score, 4),
                quality_score=round(passage.quality_score, 4),
                is_own_claim=passage.is_own_claim,
                is_related_work_summary=passage.is_related_work_summary,
                supports_core_mechanism=passage.supports_core_mechanism,
                citation_density=round(passage.citation_density, 4) if passage.citation_density is not None else None,
                quality_flags=list(passage.quality_flags),
            )
            for passage in passages[: self.max_passages]
        ]

    def _indexed_passages(
        self, suggestion: CitationSuggestion, profile: ClaimProfile
    ) -> list[Passage]:
        """Reuse passages selected by the RAG retriever instead of reparsing the PDF."""
        passages: list[Passage] = []
        for snippet in suggestion.evidence_passages:
            text = str(snippet.text or "").strip()
            if not text:
                continue
            quality = analyze_evidence_quality(
                text,
                source=snippet.source,
                suggestion=suggestion,
                profile=profile,
            )
            if quality.discard:
                continue
            passages.append(
                Passage(
                    text=text,
                    source=snippet.source,
                    section=snippet.section,
                    page=snippet.page,
                    score=snippet.score or score_text_against_claim(profile, text),
                    quality_score=snippet.quality_score or quality.quality_score,
                    is_own_claim=snippet.is_own_claim
                    if snippet.is_own_claim is not None
                    else quality.is_own_claim,
                    is_related_work_summary=snippet.is_related_work_summary
                    if snippet.is_related_work_summary is not None
                    else quality.is_related_work_summary,
                    supports_core_mechanism=snippet.supports_core_mechanism
                    if snippet.supports_core_mechanism is not None
                    else quality.supports_core_mechanism,
                    citation_density=snippet.citation_density
                    if snippet.citation_density is not None
                    else quality.citation_density,
                    quality_flags=tuple(snippet.quality_flags or quality.flags),
                )
            )
        return passages

    def _metadata_passages(
        self, suggestion: CitationSuggestion, profile: ClaimProfile
    ) -> list[Passage]:
        texts: list[tuple[str, str]] = []
        if suggestion.evidence.text:
            section = suggestion.evidence.section or "Metadata"
            source = "abstract" if "abstract" in section.lower() else "metadata"
            texts.append((source, suggestion.evidence.text))

        metadata = " ".join(
            part
            for part in [
                suggestion.title,
                " ".join(suggestion.authors),
                suggestion.venue or "",
                suggestion.year or "",
            ]
            if part
        )
        texts.append(("metadata", metadata))

        passages: list[Passage] = []
        for source, text in texts:
            for chunk in split_passages(text):
                quality = analyze_evidence_quality(chunk, source=source, suggestion=suggestion, profile=profile)
                if quality.discard:
                    self._log_skip(
                        "metadata_low_quality",
                        "Metadata evidence skipped as low-quality passage: "
                        f"title={suggestion.title!r}, flags={quality.flags}, sample={chunk[:120]!r}",
                    )
                    continue
                if quality.is_related_work_summary and not quality.is_own_claim:
                    self._log_skip(
                        "metadata_external_reference",
                        "Metadata evidence skipped as external reference discussion: "
                        f"title={suggestion.title!r}, flags={quality.flags}",
                    )
                    continue
                score = score_text_against_claim(profile, chunk)
                if score > 0:
                    adjusted_score = score * max(quality.quality_score, 0.1)
                    passages.append(
                        Passage(
                            text=chunk,
                            source=source,
                            section="Abstract" if source == "abstract" else "Metadata",
                            score=adjusted_score,
                            quality_score=quality.quality_score,
                            is_own_claim=quality.is_own_claim,
                            is_related_work_summary=quality.is_related_work_summary,
                            supports_core_mechanism=quality.supports_core_mechanism,
                            citation_density=quality.citation_density,
                            quality_flags=tuple(quality.flags),
                        )
                    )
        return passages

    def _pdf_passages(
        self, suggestion: CitationSuggestion, profile: ClaimProfile
    ) -> list[Passage]:
        if not suggestion.pdf_path:
            return []

        pdf_path = os.path.expanduser(suggestion.pdf_path)
        if not os.path.exists(pdf_path):
            self._log(f"PDF evidence skipped for {suggestion.title}: file not found: {pdf_path}")
            return []

        cached_passages = self._pdf_cache.get(pdf_path)
        if cached_passages is not None:
            raw_passages = cached_passages
            self._log(f"PDF passage cache hit for {suggestion.title}: {len(raw_passages)} passage(s).")
        else:
            try:
                raw_passages = execute_pdf_reader(
                    pdf_path,
                    lambda: self._extract_pdf_passages(pdf_path, suggestion),
                )
            except Exception as exc:  # pylint: disable=broad-except
                self._log(
                    f"PDF tool failed for {suggestion.title}; abstract/metadata fallback retained: "
                    f"{type(exc).__name__}: {exc}"
                )
                raw_passages = []
            self._pdf_cache.put(pdf_path, raw_passages)

        passages: list[Passage] = []
        for cached in raw_passages:
            chunk = cached.text
            quality = analyze_evidence_quality(chunk, source="pdf", suggestion=suggestion, profile=profile)
            if quality.discard:
                self._log_skip(
                    "pdf_low_quality_cached",
                    "PDF evidence skipped as low-quality cached passage: "
                    f"title={suggestion.title!r}, page={cached.page}, flags={quality.flags}, sample={chunk[:120]!r}",
                )
                continue
            if quality.is_related_work_summary and not quality.is_own_claim:
                self._log_skip(
                    "pdf_external_reference",
                    "PDF evidence skipped as external reference discussion: "
                    f"title={suggestion.title!r}, page={cached.page}, flags={quality.flags}, sample={chunk[:120]!r}",
                )
                continue
            score = score_text_against_claim(profile, chunk)
            if score <= 0:
                continue
            adjusted_score = score * max(quality.quality_score, 0.1)
            passages.append(
                Passage(
                    text=chunk,
                    source="pdf",
                    section="PDF",
                    page=cached.page,
                    score=adjusted_score,
                    quality_score=quality.quality_score,
                    is_own_claim=quality.is_own_claim,
                    is_related_work_summary=quality.is_related_work_summary,
                    supports_core_mechanism=quality.supports_core_mechanism,
                    citation_density=quality.citation_density,
                    quality_flags=tuple(quality.flags),
                )
            )
        return passages

    def _extract_pdf_passages(
        self,
        pdf_path: str,
        suggestion: CitationSuggestion,
    ) -> list[CachedPdfPassage]:
        try:
            import fitz  # type: ignore
        except ImportError:
            self._log("PDF evidence skipped: PyMuPDF/fitz is not installed in the v2 virtualenv.")
            return []

        passages: list[CachedPdfPassage] = []
        try:
            with fitz.open(pdf_path) as document:
                for page_index, page in enumerate(document):
                    text = page.get_text("text")
                    for chunk in split_passages(text):
                        quality = analyze_evidence_quality(chunk, source="pdf", suggestion=suggestion)
                        if quality.discard:
                            self._log_skip(
                                "pdf_low_quality",
                                "PDF evidence skipped as low-quality passage: "
                                f"title={suggestion.title!r}, page={page_index + 1}, flags={quality.flags}, sample={chunk[:120]!r}",
                            )
                            continue
                        passages.append(CachedPdfPassage(page=page_index + 1, text=chunk))
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"PDF evidence read failed for {suggestion.title}: {type(exc).__name__}: {exc}")
            return []

        return passages

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _log_skip(self, category: str, message: str, limit: int = 8) -> None:
        count = self._skip_log_counts.get(category, 0)
        if count < limit:
            self._log(message)
        elif count == limit:
            self._log(f"Further {category} evidence skip logs suppressed.")
        self._skip_log_counts[category] = count + 1


class EvidenceJudge:
    """Judge whether retrieved passages support the selected claim."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        logger: Callable[[str], None] | None = None,
        force_refresh: bool = False,
    ):
        self.model_config = model_config or ModelConfig.from_env()
        self.logger = logger
        self.force_refresh = force_refresh
        self._cache = EvidenceJudgeCache()
        self._context_reports: list[dict[str, Any]] = []
        self.last_batch_stats: dict[str, Any] = {
            "llm_enabled": bool(self.model_config.use_llm and self.model_config.api_key),
            "remote_candidates": 0,
            "cache_hits": 0,
            "local_fallback_candidates": 0,
            "error": None,
            "context_reports": [],
        }

    def judge(
        self,
        claim: str,
        profile: ClaimProfile,
        suggestion: CitationSuggestion,
        passages: list[EvidenceSnippet],
        allow_llm: bool = True,
    ) -> CitationSuggestion:
        if not passages:
            return suggestion.model_copy(
                update={
                    "verdict": "insufficient_evidence",
                    "confidence": min(suggestion.confidence, 0.35),
                    "evidence": EvidenceSnippet(),
                    "evidence_passages": [],
                    "reason": "未找到摘要或 PDF passage，不能确认该文献支持选中文段。",
                }
            )

        if allow_llm and self.model_config.use_llm and self.model_config.api_key:
            try:
                return self._judge_with_litellm(claim, profile, suggestion, passages)
            except Exception as exc:  # pylint: disable=broad-except
                self._log(
                    f"LLM evidence judge failed for {suggestion.title}; "
                    f"fallback to local rules: {type(exc).__name__}: {exc}"
                )

        return self._judge_with_rules(profile, suggestion, passages)

    def judge_batch(
        self,
        claim: str,
        profile: ClaimProfile,
        items: list[tuple[CitationSuggestion, list[EvidenceSnippet]]],
    ) -> list[CitationSuggestion]:
        self._context_reports = []
        if not items:
            self.last_batch_stats = {
                "llm_enabled": bool(self.model_config.use_llm and self.model_config.api_key),
                "remote_candidates": 0,
                "cache_hits": 0,
                "local_fallback_candidates": 0,
                "error": None,
                "context_reports": [],
            }
            return []

        if not self.model_config.use_llm or not self.model_config.api_key:
            self.last_batch_stats = {
                "llm_enabled": False,
                "remote_candidates": 0,
                "cache_hits": 0,
                "local_fallback_candidates": len(items),
                "error": None,
                "context_reports": [],
            }
            return [
                self.judge(claim, profile, suggestion, passages, allow_llm=False)
                for suggestion, passages in items
            ]

        llm_items = [
            (suggestion, passages)
            for suggestion, passages in items
            if passages
        ]
        no_passage_count = len(items) - len(llm_items)
        if not llm_items:
            self.last_batch_stats = {
                "llm_enabled": True,
                "remote_candidates": 0,
                "cache_hits": 0,
                "local_fallback_candidates": len(items),
                "error": None,
                "context_reports": [],
            }
            return [
                self.judge(claim, profile, suggestion, passages, allow_llm=False)
                for suggestion, passages in items
            ]

        cached_by_id, uncached_items = (
            ({}, llm_items)
            if self.force_refresh
            else self._cached_judgments(claim, profile, llm_items)
        )
        if self.force_refresh:
            self._log(f"Evidence judge cache bypassed for {len(llm_items)} candidate(s).")
        if self.logger and cached_by_id:
            self._log(f"Evidence judge cache hit for {len(cached_by_id)} candidate(s).")

        try:
            judged_by_id = self._judge_batch_with_litellm(claim, profile, uncached_items) if uncached_items else {}
            judged_by_id.update(cached_by_id)
        except Exception as exc:  # pylint: disable=broad-except
            self.last_batch_stats = {
                "llm_enabled": True,
                "remote_candidates": 0,
                "remote_attempted_candidates": len(uncached_items),
                "cache_hits": len(cached_by_id),
                "local_fallback_candidates": len(items),
                "error": f"{type(exc).__name__}: {exc}",
                "context_reports": list(self._context_reports),
            }
            self._log(
                "Batch LLM evidence judge failed; fallback to local rules: "
                f"{type(exc).__name__}: {exc}"
            )
            return [
                self.judge(claim, profile, suggestion, passages, allow_llm=False)
                for suggestion, passages in items
            ]

        judged: list[CitationSuggestion] = []
        local_fallback_count = no_passage_count
        for suggestion, passages in items:
            if not passages:
                judged.append(self.judge(claim, profile, suggestion, passages, allow_llm=False))
                continue
            key = suggestion_batch_key(suggestion)
            if key in judged_by_id:
                judged.append(judged_by_id[key])
            else:
                local_fallback_count += 1
                judged.append(self.judge(claim, profile, suggestion, passages, allow_llm=False))
        self.last_batch_stats = {
            "llm_enabled": True,
            "remote_candidates": len(uncached_items),
            "remote_attempted_candidates": len(uncached_items),
            "cache_hits": len(cached_by_id),
            "local_fallback_candidates": local_fallback_count,
            "error": None,
            "context_reports": list(self._context_reports),
        }
        return judged

    def _cached_judgments(
        self,
        claim: str,
        profile: ClaimProfile,
        items: list[tuple[CitationSuggestion, list[EvidenceSnippet]]],
    ) -> tuple[dict[str, CitationSuggestion], list[tuple[CitationSuggestion, list[EvidenceSnippet]]]]:
        cached: dict[str, CitationSuggestion] = {}
        uncached: list[tuple[CitationSuggestion, list[EvidenceSnippet]]] = []
        for suggestion, passages in items:
            cache_key = evidence_judge_cache_key(
                self.model_config.model,
                claim,
                profile,
                suggestion,
                passages,
            )
            raw = self._cache.get(cache_key)
            if raw:
                judged = self._suggestion_from_cached_judgment(suggestion, passages, raw)
                if judged:
                    cached[suggestion_batch_key(suggestion)] = judged
                    continue
            uncached.append((suggestion, passages))
        return cached, uncached

    def _suggestion_from_cached_judgment(
        self,
        suggestion: CitationSuggestion,
        passages: list[EvidenceSnippet],
        raw: dict[str, Any],
    ) -> CitationSuggestion | None:
        verdict = raw.get("verdict")
        if verdict not in VERDICT_ORDER or verdict == "metadata_match":
            return None
        evidence_index = raw.get("evidence_index")
        if not isinstance(evidence_index, int) or not 0 <= evidence_index < len(passages):
            evidence_index = 0
        confidence = cap_confidence_for_verdict(
            clamp_float(raw.get("confidence"), default=suggestion.confidence),
            verdict,
        )
        return suggestion.model_copy(
            update={
                "verdict": verdict,
                "confidence": confidence,
                "evidence": passages[stable_primary_evidence_index(passages, evidence_index)],
                "evidence_passages": passages,
                "reason": str(raw.get("reason") or "缓存的 LLM 证据判断。"),
            }
        )

    def _judge_with_litellm(
        self,
        claim: str,
        profile: ClaimProfile,
        suggestion: CitationSuggestion,
        passages: list[EvidenceSnippet],
    ) -> CitationSuggestion:
        passage_payload = [
            {
                "index": index,
                "source": passage.source,
                "page": passage.page,
                "section": passage.section,
                "text": passage.text,
                "quality": {
                    "quality_score": passage.quality_score,
                    "is_own_claim": passage.is_own_claim,
                    "is_related_work_summary": passage.is_related_work_summary,
                    "supports_core_mechanism": passage.supports_core_mechanism,
                    "citation_density": passage.citation_density,
                    "flags": passage.quality_flags,
                },
            }
            for index, passage in enumerate(passages)
        ]
        system_prompt = (
            "You verify citation evidence. Return only valid JSON with keys: "
            "verdict, confidence, reason, evidence_index. verdict must be one of "
            "direct_support, foundational_support, partial_support, related_only, contradiction, insufficient_evidence. "
            "Use direct_support only when a passage explicitly supports the claim's meaning; "
            "use foundational_support when the paper establishes a foundational method family, "
            "representation, or differentiable/neural mechanism required by the claim, but does "
            "not by itself prove every downstream/domain-specific part of the claim. "
            "Do not use foundational_support for an ordinary application/system paper merely "
            "because it uses or mentions the required concepts; the passages should show broad "
            "method-family, survey/review, paradigm, foundational representation, or original "
            "formulation relevance. use related_only when it only shares topic words. A candidate "
            "must support both the claim's subject scope and predicate relation. For broad claims "
            "about a research field or recent methods, a single system paper is not direct_support "
            "unless its own passage explicitly makes that broader field-level claim. Important: do "
            "not treat a passage as support for this paper if the passage is merely describing or "
            "citing another paper inside related work. Also ignore citation-heavy related-work "
            "survey passages that mainly summarize prior papers instead of the current paper's own "
            "contribution. Only use evidence that reflects this paper's own abstract, method, findings, "
            "or contribution. Confidence means citation-support confidence; insufficient_evidence "
            "must use a low confidence value. Ignore reference-list entries, figure/table captions, "
            "architecture labels, OCR fragments, or keyword streams rather than coherent prose."
        )
        context_packet = ContextManager(
            policy_for_stage(
                "evidence_verification",
                evidence_budget=self.model_config.context_evidence_budget_tokens,
            ),
            logger=self.logger,
        ).build_json_messages(
            system_prompt=system_prompt,
            items=[
                ContextItem(
                    "claim", claim, priority="required", source="editor_selection",
                    purpose="define_the_citation_claim", max_tokens=800,
                ),
                ContextItem(
                    "claim_interpretation",
                    claim_interpretation_payload(profile),
                    priority="required",
                    source="claim_analysis",
                    purpose="preserve_required_mechanisms_and_scope",
                    max_tokens=1600,
                ),
                ContextItem(
                    "paper",
                    {
                        "title": suggestion.title,
                        "authors": suggestion.authors,
                        "year": suggestion.year,
                        "venue": suggestion.venue,
                    },
                    priority="required",
                    source="candidate_metadata",
                    purpose="identify_the_paper_being_judged",
                    max_tokens=700,
                    protected_keys=("title", "year"),
                ),
                ContextItem(
                    "passages",
                    passage_payload,
                    priority="compressible",
                    source="abstract_and_pdf_reader",
                    purpose="verify_claim_support_from_paper_evidence",
                    min_tokens=1800,
                    max_tokens=5600,
                    min_items=len(passage_payload),
                    protected_keys=("index", "source", "page", "section"),
                ),
            ],
        )
        self._context_reports.append(context_packet.report.to_dict())
        kwargs: dict[str, Any] = {
            "model": self.model_config.model,
            "messages": context_packet.messages,
            "api_key": self.model_config.api_key,
            "timeout": self.model_config.evidence_timeout_seconds,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self.model_config.api_base:
            kwargs["api_base"] = self.model_config.api_base

        response = execute_litellm_completion(kwargs)
        data = parse_json_object(response.choices[0].message.content)
        verdict = data.get("verdict")
        if verdict not in VERDICT_ORDER or verdict == "metadata_match":
            verdict = "insufficient_evidence"

        evidence_index = data.get("evidence_index")
        if not isinstance(evidence_index, int) or not 0 <= evidence_index < len(passages):
            evidence_index = 0

        confidence = cap_confidence_for_verdict(
            clamp_float(data.get("confidence"), default=suggestion.confidence),
            verdict,
        )
        stable_evidence_index = stable_primary_evidence_index(passages, evidence_index)
        verdict, confidence, reason = apply_claim_scope_guard(
            claim=claim,
            profile=profile,
            suggestion=suggestion,
            passages=passages,
            verdict=verdict,
            confidence=confidence,
            reason=str(data.get("reason") or "LLM 已根据 evidence passages 进行支持性判断。"),
        )
        return suggestion.model_copy(
            update={
                "verdict": verdict,
                "confidence": confidence,
                "evidence": passages[stable_evidence_index],
                "evidence_passages": passages,
                "reason": reason,
            }
        )

    def _judge_batch_with_litellm(
        self,
        claim: str,
        profile: ClaimProfile,
        items: list[tuple[CitationSuggestion, list[EvidenceSnippet]]],
    ) -> dict[str, CitationSuggestion]:
        candidates = []
        for candidate_index, (suggestion, passages) in enumerate(items):
            candidates.append(
                {
                    "candidate_id": suggestion_batch_key(suggestion),
                    "candidate_index": candidate_index,
                    "paper": {
                        "title": suggestion.title,
                        "authors": suggestion.authors,
                        "year": suggestion.year,
                        "venue": suggestion.venue,
                    },
                    "passages": [
                        {
                            "index": passage_index,
                            "source": passage.source,
                            "page": passage.page,
                            "section": passage.section,
                            "text": passage.text,
                            "quality": {
                                "quality_score": passage.quality_score,
                                "is_own_claim": passage.is_own_claim,
                                "is_related_work_summary": passage.is_related_work_summary,
                                "supports_core_mechanism": passage.supports_core_mechanism,
                                "citation_density": passage.citation_density,
                                "flags": passage.quality_flags,
                            },
                        }
                        for passage_index, passage in enumerate(passages)
                    ],
                }
            )

        system_prompt = (
            "You verify citation evidence for multiple candidate papers. Return only valid JSON. "
            "JSON schema: {\"results\":[{\"candidate_id\": string, \"verdict\": string, "
            "\"confidence\": number, \"reason\": string, \"evidence_index\": number}]}. "
            "verdict must be one of direct_support, foundational_support, partial_support, related_only, "
            "contradiction, insufficient_evidence. Use direct_support only when the paper's own abstract, "
            "method, finding, or contribution explicitly supports both the claim's subject scope and "
            "predicate relation. Use foundational_support when the paper establishes a foundational method "
            "family, representation, or differentiable/neural mechanism required by the claim, but does not "
            "by itself prove every downstream/domain-specific part of the claim. If a paper only matches "
            "context/domain words but misses required concepts, use insufficient_evidence or related_only, "
            "not foundational_support. The claim_interpretation lists required_concepts and "
            "excluded_context_concepts; use it to avoid recommending ordinary domain papers that do not use "
            "the required mechanism. Reserve foundational_support for broad method-family, survey/review, "
            "paradigm, foundational representation, or original formulation evidence. For broad field claims, "
            "a single system paper is not direct_support unless its own passage explicitly makes that broader "
            "claim. Do not use related-work sentences that merely describe or cite another paper. Ignore "
            "citation-heavy related-work summaries, reference-list entries, figure/table captions, OCR "
            "fragments, and keyword streams. Confidence means citation-support confidence; "
            "insufficient_evidence must have low confidence. Judge every candidate independently."
        )
        context_packet = ContextManager(
            policy_for_stage(
                "evidence_verification",
                evidence_budget=self.model_config.context_evidence_budget_tokens,
            ),
            logger=self.logger,
        ).build_json_messages(
            system_prompt=system_prompt,
            items=[
                ContextItem(
                    "claim", claim, priority="required", source="editor_selection",
                    purpose="define_the_citation_claim", max_tokens=800,
                ),
                ContextItem(
                    "claim_interpretation",
                    claim_interpretation_payload(profile),
                    priority="required",
                    source="claim_analysis",
                    purpose="preserve_required_mechanisms_and_scope",
                    max_tokens=1600,
                ),
                ContextItem(
                    "candidates",
                    candidates,
                    priority="compressible",
                    source="candidate_metadata_and_evidence",
                    purpose="judge_each_candidate_against_the_claim",
                    min_tokens=max(2400, len(candidates) * 220),
                    max_tokens=max(3000, self.model_config.context_evidence_budget_tokens - 2200),
                    min_items=len(candidates),
                    protected_keys=(
                        "candidate_id",
                        "candidate_index",
                        "index",
                        "source",
                        "page",
                        "section",
                        "title",
                        "year",
                    ),
                ),
            ],
        )
        self._context_reports.append(context_packet.report.to_dict())
        kwargs: dict[str, Any] = {
            "model": self.model_config.model,
            "messages": context_packet.messages,
            "api_key": self.model_config.api_key,
            "timeout": self.model_config.evidence_timeout_seconds,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self.model_config.api_base:
            kwargs["api_base"] = self.model_config.api_base

        response = execute_litellm_completion(kwargs)
        data = parse_json_object(response.choices[0].message.content)
        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Batch evidence judge response missing results list.")

        items_by_id = {
            suggestion_batch_key(suggestion): (suggestion, passages)
            for suggestion, passages in items
        }
        judged: dict[str, CitationSuggestion] = {}
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("candidate_id") or "")
            if candidate_id not in items_by_id:
                continue
            suggestion, passages = items_by_id[candidate_id]
            verdict = raw.get("verdict")
            if verdict not in VERDICT_ORDER or verdict == "metadata_match":
                verdict = "insufficient_evidence"
            evidence_index = raw.get("evidence_index")
            if not isinstance(evidence_index, int) or not 0 <= evidence_index < len(passages):
                evidence_index = 0
            confidence = cap_confidence_for_verdict(
                clamp_float(raw.get("confidence"), default=suggestion.confidence),
                verdict,
            )
            stable_evidence_index = stable_primary_evidence_index(passages, evidence_index)
            verdict, confidence, reason = apply_claim_scope_guard(
                claim=claim,
                profile=profile,
                suggestion=suggestion,
                passages=passages,
                verdict=verdict,
                confidence=confidence,
                reason=str(raw.get("reason") or "LLM 已根据 evidence passages 进行批量支持性判断。"),
            )
            judged[candidate_id] = suggestion.model_copy(
                update={
                    "verdict": verdict,
                    "confidence": confidence,
                    "evidence": passages[stable_evidence_index],
                    "evidence_passages": passages,
                    "reason": reason,
                }
            )
            self._cache.put(
                evidence_judge_cache_key(self.model_config.model, claim, profile, suggestion, passages),
                {
                    "verdict": verdict,
                    "confidence": confidence,
                    "reason": reason,
                    "evidence_index": stable_evidence_index,
                },
            )
        return judged

    def _judge_with_rules(
        self,
        profile: ClaimProfile,
        suggestion: CitationSuggestion,
        passages: list[EvidenceSnippet],
    ) -> CitationSuggestion:
        best = passages[0]
        best_text = best.text or ""
        evidence_lower = best_text.lower()
        evidence_terms = tokenize(best_text)
        title_terms = tokenize(suggestion.title)
        all_evidence_terms = tokenize(
            " ".join(
                part
                for part in [
                    suggestion.title,
                    suggestion.evidence.text or "",
                    *[passage.text or "" for passage in passages[:3]],
                ]
                if part
            )
        )
        phrase_hits = sum(1 for phrase in profile.phrases if phrase in evidence_lower)
        claim_hits = len(profile.terms & evidence_terms)
        must_hits = len(profile.must_include_terms & evidence_terms)
        must_coverage = must_hits / max(len(profile.must_include_terms), 1)
        claim_coverage = claim_hits / max(len(profile.terms), 1)
        required_hits = len(profile.required_concept_terms & all_evidence_terms)
        foundational_hits = len(profile.foundational_concept_terms & all_evidence_terms)
        required_coverage = required_hits / max(len(profile.required_concept_terms), 1)
        foundational_coverage = foundational_hits / max(len(profile.foundational_concept_terms), 1)
        excluded_hits = len(profile.excluded_concept_terms & all_evidence_terms)
        support_hits = sum(
            1
            for support_term in profile.support_terms
            if support_term in evidence_terms or support_term in evidence_lower
        )
        title_hits = len(profile.terms & title_terms)

        has_required_mechanism = required_coverage >= 0.3 or foundational_coverage >= 0.3
        if excluded_hits and not has_required_mechanism:
            verdict = "insufficient_evidence"
            confidence = 0.28
            reason = "候选主要命中上下文/领域词，但没有覆盖待引句要求的核心机制概念。"
        elif profile.claim_kind == "field_level_background" and has_required_mechanism and has_foundational_scope_evidence(
            profile, suggestion, passages
        ):
            verdict = "foundational_support"
            confidence = 0.74
            reason = "候选证据覆盖了待引句要求的核心方法族/机制，可作为领域背景或奠基性支持。"
        elif (phrase_hits >= 2 and support_hits >= 2) or (support_hits >= 2 and must_coverage >= 0.45):
            verdict = "direct_support"
            confidence = 0.82
            reason = "摘要/PDF passage 同时覆盖研究对象和关键谓词，局部规则判断为直接支持。"
        elif (phrase_hits >= 1 and support_hits >= 1) or (support_hits >= 1 and must_coverage >= 0.3):
            verdict = "partial_support"
            confidence = 0.68
            reason = "摘要/PDF passage 覆盖了部分研究对象和谓词，局部规则判断为部分支持。"
        elif claim_coverage >= 0.35 or title_hits >= 2:
            verdict = "related_only"
            confidence = 0.5
            reason = "候选文献主题相关，但 passage 尚未明确表达选中文段的完整含义。"
        else:
            verdict = "insufficient_evidence"
            confidence = 0.34
            reason = "未找到能支持选中文段核心含义的摘要/PDF passage。"

        if suggestion.citation_key and verdict in {"direct_support", "foundational_support", "partial_support"}:
            confidence += 0.05
        verdict, confidence, reason = apply_claim_scope_guard(
            claim=profile.claim,
            profile=profile,
            suggestion=suggestion,
            passages=passages,
            verdict=verdict,
            confidence=min(confidence, 0.92),
            reason=reason,
        )

        return suggestion.model_copy(
            update={
                "verdict": verdict,
                "confidence": confidence,
                "evidence": best,
                "evidence_passages": passages,
                "reason": reason,
            }
        )

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


def split_passages(text: str, max_chars: int = 700) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    rough = re.split(r"(?<=[.!?])\s+", normalized)
    passages: list[str] = []
    buffer = ""
    for sentence in rough:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            passages.append(buffer)
        buffer = sentence[:max_chars]
    if buffer:
        passages.append(buffer)
    return passages


def passage_sort_key(passage: Passage) -> tuple[float, int, int, str, str]:
    """Stable ordering for evidence snippets.

    Scores are often tied. Without deterministic tie-breakers, Zotero/PDF
    ordering noise can make the UI show different passages between runs.
    """
    source_priority = {
        "abstract": 0,
        "pdf": 1,
        "metadata": 2,
        "unknown": 3,
    }.get(passage.source, 4)
    page = passage.page if passage.page is not None else 0
    section = passage.section or ""
    text = re.sub(r"\s+", " ", passage.text or "").strip().lower()
    quality = getattr(passage, "quality_score", 1.0)
    return (-round(passage.score, 6), -round(quality, 6), source_priority, page, section.lower(), text)


def dedupe_passages(passages: list[Passage]) -> list[Passage]:
    best: dict[tuple[str, int | None, str], Passage] = {}
    for passage in passages:
        normalized = re.sub(r"\s+", " ", passage.text or "").strip().lower()
        if not normalized:
            continue
        key = (passage.source, passage.page, normalized)
        previous = best.get(key)
        if previous is None or passage.score > previous.score:
            best[key] = passage
    return list(best.values())


def evidence_query_text(claim: str | None, profile: ClaimProfile) -> str:
    parts = [
        claim or "",
        profile.claim_focus,
        " ".join(profile.phrases),
        " ".join(sorted(profile.required_concept_terms)),
        " ".join(sorted(profile.foundational_concept_terms)),
        " ".join(sorted(profile.support_terms)),
        " ".join(sorted(profile.terms)),
    ]
    return " ".join(part for part in parts if part).strip()


def evidence_snippet_sort_key(snippet: EvidenceSnippet) -> tuple[float, float, int, int, str, str]:
    score = snippet.score if snippet.score is not None else 0.0
    quality = snippet.quality_score if snippet.quality_score is not None else 1.0
    source_priority = {
        "abstract": 0,
        "pdf": 1,
        "metadata": 2,
        "unknown": 3,
    }.get(snippet.source, 4)
    page = snippet.page if snippet.page is not None else 0
    section = snippet.section or ""
    text = re.sub(r"\s+", " ", snippet.text or "").strip().lower()
    return (-round(score, 6), -round(quality, 6), source_priority, page, section.lower(), text)


def stable_primary_evidence_index(
    passages: list[EvidenceSnippet],
    llm_index: int,
) -> int:
    """Prefer the LLM's cited passage only when it ties the deterministic best.

    Remote model calls can choose different evidence_index values across runs.
    This keeps the displayed primary evidence stable while still allowing the
    model to select among genuinely equal top-ranked passages.
    """
    if not passages:
        return 0

    best_key = min(evidence_snippet_sort_key(passage) for passage in passages)
    selected_key = evidence_snippet_sort_key(passages[llm_index])
    if selected_key == best_key:
        return llm_index

    return min(range(len(passages)), key=lambda index: evidence_snippet_sort_key(passages[index]))


def suggestion_batch_key(suggestion: CitationSuggestion) -> str:
    if suggestion.doi:
        return f"doi:{suggestion.doi.lower().strip()}"
    if suggestion.citation_key:
        return f"citekey:{suggestion.citation_key.lower().strip()}"
    return f"paper:{suggestion.paper_id.lower().strip()}"


def evidence_judge_cache_key(
    model: str,
    claim: str,
    profile: ClaimProfile,
    suggestion: CitationSuggestion,
    passages: list[EvidenceSnippet],
) -> str:
    payload = {
        "version": 5,
        "model": model,
        "claim": re.sub(r"\s+", " ", claim).strip(),
        "claim_interpretation": claim_interpretation_payload(profile),
        "candidate": suggestion_batch_key(suggestion),
        "title": suggestion.title,
        "passages": [
            {
                "source": passage.source,
                "page": passage.page,
                "section": passage.section,
                "text": re.sub(r"\s+", " ", passage.text or "").strip(),
                "quality_score": passage.quality_score,
                "quality_flags": passage.quality_flags,
            }
            for passage in passages
        ],
    }
    return stable_text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def claim_interpretation_payload(profile: ClaimProfile) -> dict[str, Any]:
    return {
        "claim_focus": profile.claim_focus,
        "claim_kind": profile.claim_kind,
        "required_concepts": sorted(profile.required_concept_terms),
        "foundational_concepts": sorted(profile.foundational_concept_terms),
        "excluded_context_concepts": sorted(profile.excluded_concept_terms | profile.exclude_terms),
        "support_terms": sorted(profile.support_terms),
        "note": (
            "A valid citation should support the required concepts/mechanism. "
            "Papers that only match excluded_context_concepts or broad domain terms are not enough."
        ),
    }


FOUNDATIONAL_BROAD_MARKERS = (
    "survey",
    "review",
    "overview",
    "taxonomy",
    "benchmark",
    "tutorial",
    "foundational",
    "seminal",
    "paradigm",
    "method family",
    "class of methods",
    "line of work",
)

FOUNDATIONAL_ORIGINALITY_MARKERS = (
    "introduce",
    "introduces",
    "introduced",
    "present",
    "presents",
    "propose",
    "proposes",
    "proposed",
    "formulate",
    "formulates",
    "formulated",
    "represent",
    "represents",
    "represented",
)

METHOD_FAMILY_MARKERS = (
    "model",
    "models",
    "representation",
    "representations",
    "field",
    "fields",
    "rendering",
    "optimization",
    "formulation",
    "framework",
    "paradigm",
    "mapping",
)


def has_foundational_scope_evidence(
    profile: ClaimProfile,
    suggestion: CitationSuggestion,
    passages: list[EvidenceSnippet],
) -> bool:
    """Return True when evidence has a broad/foundational role, not just topic overlap."""

    text = " ".join(
        part
        for part in [
            suggestion.title,
            suggestion.venue or "",
            suggestion.evidence.text or "",
            *[passage.text or "" for passage in passages[:5]],
        ]
        if part
    )
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if not lowered:
        return False

    evidence_terms = tokenize(lowered)
    required_terms = profile.required_concept_terms | profile.foundational_concept_terms
    required_hits = required_terms & evidence_terms
    required_coverage = len(required_hits) / max(len(required_terms), 1)
    if required_terms and required_coverage < 0.3:
        return False

    if any(marker in lowered for marker in FOUNDATIONAL_BROAD_MARKERS):
        return True

    has_originality = any(marker in evidence_terms for marker in FOUNDATIONAL_ORIGINALITY_MARKERS)
    has_method_family = any(marker in evidence_terms for marker in METHOD_FAMILY_MARKERS)
    if has_originality and has_method_family and required_coverage >= 0.35:
        return True

    if re.search(
        r"\b(?:field|family|class|line|paradigm|formulation|representation)s?\s+of\b",
        lowered,
    ):
        return True

    return False


def apply_claim_scope_guard(
    claim: str,
    suggestion: CitationSuggestion,
    passages: list[EvidenceSnippet],
    verdict: str,
    confidence: float,
    reason: str,
    profile: ClaimProfile | None = None,
) -> tuple[str, float, str]:
    """Downgrade confident judgments when evidence does not cover claim anchors."""
    if verdict not in {"direct_support", "foundational_support", "partial_support"}:
        return verdict, confidence, reason

    profile = profile or build_claim_profile(claim)
    evidence_text = " ".join(
        part
        for part in [
            suggestion.title,
            suggestion.evidence.text or "",
            *[passage.text or "" for passage in passages[:5]],
        ]
        if part
    )
    evidence_terms = tokenize(evidence_text)
    required_terms = profile.required_concept_terms | profile.foundational_concept_terms
    if required_terms:
        required_hits = required_terms & evidence_terms
        required_coverage = len(required_hits) / max(len(required_terms), 1)
        min_required_coverage = 0.3 if profile.claim_kind == "field_level_background" else 0.18
        if not required_hits or required_coverage < min_required_coverage:
            guarded_reason = (
                f"{reason} 核心机制校验：候选证据没有覆盖 citation contract 中的必要概念 "
                f"({', '.join(sorted(required_terms))})，因此不能作为推荐引用。"
            )
            return "insufficient_evidence", min(confidence, 0.35), guarded_reason

    if verdict == "foundational_support" and not has_foundational_scope_evidence(
        profile, suggestion, passages
    ):
        guarded_reason = (
            f"{reason} 奠基性校验：候选证据更像普通应用/系统论文，未显示其作为方法族、"
            "综述、范式或基础表示的支持作用，因此降为诊断候选。"
        )
        return "partial_support", min(confidence, 0.62), guarded_reason

    if not profile.must_include_terms:
        return verdict, confidence, reason

    missing_anchor_ratio = 1.0 - (
        len(profile.must_include_terms & evidence_terms) / max(len(profile.must_include_terms), 1)
    )
    lexical_score = score_text_against_claim(profile, evidence_text)

    if missing_anchor_ratio >= 0.75 and lexical_score < 0.35:
        guarded_reason = (
            f"{reason} 作用域校验：候选证据没有覆盖选中文段的核心主题词 "
            f"({', '.join(sorted(profile.must_include_terms))})，因此不能作为强支持。"
        )
        return "insufficient_evidence", min(confidence, 0.35), guarded_reason

    return verdict, confidence, reason


def analyze_evidence_quality(
    text: str,
    source: str = "pdf",
    suggestion: CitationSuggestion | None = None,
    profile: ClaimProfile | None = None,
) -> EvidenceQuality:
    """Score a passage's usefulness as direct evidence for the current paper.

    v13 keeps the existing hard filters, but also exposes quality dimensions
    to ranking, LLM verification, UI, trace and future eval metrics.
    """
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return EvidenceQuality(
            quality_score=0.0,
            discard=True,
            is_own_claim=None,
            is_related_work_summary=None,
            supports_core_mechanism=None,
            citation_density=0.0,
            flags=("empty",),
        )

    flags: list[str] = []
    lowered = normalized.lower()
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", normalized)
    citation_density = estimate_citation_density(normalized)
    own_claim = looks_like_own_claim(normalized)
    related_summary = looks_like_related_work_summary(normalized, suggestion)
    supports_core = supports_core_mechanism_text(normalized, profile) if profile else None

    if looks_like_reference_entry(normalized):
        flags.append("reference_list")
    if looks_like_figure_or_table_caption(lowered):
        flags.append("figure_or_table_caption")
    if looks_like_toc_or_front_matter(lowered):
        flags.append("toc_or_front_matter")
    if source == "pdf" and len(words) < 12:
        flags.append("too_short")
    sentence_count = len(re.findall(r"[.!?]", normalized))
    if source == "pdf" and sentence_count == 0 and len(words) > 25:
        flags.append("no_sentence_boundary")
    if source == "pdf" and looks_like_ocr_or_diagram_word_stream(normalized, words):
        flags.append("ocr_or_diagram_stream")
    if related_summary:
        flags.append("related_work_summary")
    if own_claim:
        flags.append("own_claim")
    if supports_core is True:
        flags.append("supports_core_mechanism")
    elif supports_core is False and profile and profile.required_concept_terms:
        flags.append("missing_core_mechanism")
    if citation_density >= 0.32:
        flags.append("citation_dense")

    hard_discard_flags = {
        "reference_list",
        "figure_or_table_caption",
        "toc_or_front_matter",
        "too_short",
        "no_sentence_boundary",
        "ocr_or_diagram_stream",
    }
    discard = any(flag in hard_discard_flags for flag in flags)
    if related_summary and not own_claim:
        discard = True

    quality = 1.0
    if citation_density >= 0.18:
        quality -= min(0.35, citation_density)
    if related_summary:
        quality -= 0.45
    if own_claim:
        quality += 0.12
    if supports_core is True:
        quality += 0.10
    elif supports_core is False and profile and profile.required_concept_terms:
        quality -= 0.22
    if source == "metadata":
        quality -= 0.18
    if source == "abstract":
        quality += 0.04
    if source == "pdf":
        quality += 0.02
    if discard:
        quality = min(quality, 0.12)

    return EvidenceQuality(
        quality_score=max(0.0, min(round(quality, 4), 1.0)),
        discard=discard,
        is_own_claim=own_claim,
        is_related_work_summary=related_summary,
        supports_core_mechanism=supports_core,
        citation_density=round(citation_density, 4),
        flags=tuple(flags),
    )


def estimate_citation_density(text: str) -> float:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)
    if not words:
        return 0.0
    bracket_numbers = sum(
        len(re.findall(r"\d+", group))
        for group in re.findall(r"\[[0-9,\-\s]{1,40}\]", text)
    )
    author_year = len(re.findall(r"\([A-Z][A-Za-z\-]+(?: et al\.)?,?\s+20\d{2}[a-z]?\)", text))
    et_al = len(re.findall(r"\bet al\.", text, flags=re.IGNORECASE))
    return (bracket_numbers + author_year + et_al) / max(len(words), 1)


def looks_like_own_claim(text: str) -> bool:
    lowered = text.lower()
    own_markers = (
        "we propose",
        "we present",
        "we introduce",
        "we develop",
        "we demonstrate",
        "we show",
        "we formulate",
        "our method",
        "our approach",
        "our system",
        "this paper proposes",
        "this paper presents",
        "this work proposes",
        "this work presents",
        "in this paper, we",
    )
    return any(marker in lowered for marker in own_markers)


def looks_like_related_work_summary(
    text: str,
    suggestion: CitationSuggestion | None = None,
) -> bool:
    candidate_mentions = 0
    if suggestion:
        title_tokens = title_signature_tokens(suggestion.title)
        lowered = text.lower()
        candidate_mentions = sum(1 for token in title_tokens if token in lowered)
        if is_external_reference_discussion(text, suggestion):
            return True
    return looks_like_citation_heavy_related_work(text, candidate_mentions)


def supports_core_mechanism_text(text: str, profile: ClaimProfile | None) -> bool | None:
    if not profile:
        return None
    required_terms = profile.required_concept_terms | profile.foundational_concept_terms
    if not required_terms:
        required_terms = profile.must_include_terms
    if not required_terms:
        required_terms = profile.support_terms | {
            term
            for term in profile.terms
            if term not in {"existing", "recent", "generally", "usually", "typically", "methods", "systems"}
        }
    if not required_terms:
        return None
    evidence_terms = tokenize(text)
    hits = required_terms & evidence_terms
    coverage = len(hits) / max(len(required_terms), 1)
    return bool(hits and coverage >= 0.18)


def looks_like_toc_or_front_matter(lowered: str) -> bool:
    front_markers = (
        "contents",
        "table of contents",
        "list of figures",
        "list of tables",
        "publication date",
        "submitted to",
        "all rights reserved",
        "arxiv preprint",
        "anonymous submission",
    )
    marker_hits = sum(1 for marker in front_markers if marker in lowered)
    if marker_hits >= 2:
        return True
    if lowered.startswith(("abstract", "keywords")):
        return False
    return False


def is_low_quality_evidence_passage(text: str, source: str = "pdf") -> bool:
    return analyze_evidence_quality(text, source=source).discard


def looks_like_reference_entry(text: str) -> bool:
    lowered = text.lower()
    if len(re.findall(r"\[\d+\]\s+[A-Z][A-Za-z\-]+", text)) >= 2:
        return True
    if re.search(r"\barxiv:\d{4}\.\d{4,5}", lowered):
        citation_years = len(re.findall(r"\((?:19|20)\d{2}\)|\b(?:19|20)\d{2}\b", lowered))
        author_initials = len(re.findall(r"\b[A-Z]\.,", text))
        has_abstract_marker = bool(re.search(r"\babstract\b|\bwe\s+(present|propose|introduce|develop)\b", lowered))
        if not has_abstract_marker and (citation_years >= 2 or author_initials >= 3):
            return True
    if re.search(r"\b(ieee transactions|proceedings of|conference on|journal of)\b", lowered):
        citation_years = len(re.findall(r"\((?:19|20)\d{2}\)|\b(?:19|20)\d{2}\b", lowered))
        author_initials = len(re.findall(r"\b[A-Z]\.,", text))
        if citation_years >= 2 or author_initials >= 4:
            return True
    if len(re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.", text)) >= 3:
        return True
    return False


def looks_like_figure_or_table_caption(lowered: str) -> bool:
    caption_markers = (
        "fig.",
        "figure",
        "table",
        "core components",
        "agent side",
        "server side",
        "left)",
        "right)",
        "pipeline",
        "architecture",
    )
    marker_hits = sum(1 for marker in caption_markers if marker in lowered)
    diagram_terms = (
        "tracking",
        "local mapping",
        "loop closure",
        "pose optimization",
        "map refinement",
        "data stream",
        "sub-map",
        "merging",
    )
    diagram_hits = sum(1 for term in diagram_terms if term in lowered)
    return marker_hits >= 2 or (marker_hits >= 1 and diagram_hits >= 3)


def looks_like_ocr_or_diagram_word_stream(text: str, words: list[str]) -> bool:
    short_words = sum(1 for word in words if len(word) <= 2)
    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    titlecase_ratio = sum(1 for word in words if word[:1].isupper()) / max(len(words), 1)
    punctuation_count = len(re.findall(r"[,.;:!?]", text))
    connectors = len(re.findall(r"\b(?:and|or|but|because|which|that|where|while|although|therefore)\b", text.lower()))

    if short_words >= 6 and punctuation_count <= 2:
        return True
    if uppercase_words >= 4 and connectors <= 2:
        return True
    if titlecase_ratio > 0.55 and punctuation_count <= 3 and connectors <= 2:
        return True
    return False


def is_external_reference_discussion(text: str, suggestion: CitationSuggestion) -> bool:
    """Return True for passages that mainly describe another cited paper.

    PDF evidence often includes related-work sentences such as
    "MAGIC-SLAM [38] extends ...". If the current candidate is GRAND-SLAM,
    that passage should not be used as evidence for GRAND-SLAM's own claim.
    This is intentionally conservative: it skips citation-heavy third-person
    snippets only when they do not look like the candidate's own abstract or
    contribution statement.
    """
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return False

    lowered = normalized.lower()
    self_markers = {
        "we ",
        "our ",
        "this paper",
        "this work",
        "we propose",
        "we present",
        "we introduce",
        "we develop",
        "we demonstrate",
    }
    if any(marker in lowered for marker in self_markers):
        return False

    title_tokens = title_signature_tokens(suggestion.title)
    candidate_mentions = sum(1 for token in title_tokens if token in lowered)
    citation_like = bool(
        re.search(r"\[[0-9,\-\s]{1,20}\]", normalized)
        or re.search(r"\b[A-Z][A-Za-z\-]+ et al\.", normalized)
        or re.search(r"\([A-Z][A-Za-z\-]+(?: et al\.)?,?\s+20\d{2}\)", normalized)
    )
    if not citation_like:
        return False

    if looks_like_citation_heavy_related_work(normalized, candidate_mentions):
        return True

    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    leading_system = re.match(
        r"^(?:e\.?\s*)?([A-Z][A-Za-z0-9]*(?:[- ][A-Z0-9][A-Za-z0-9]*){0,3})\s*(?:\[[0-9,\-\s]+\])?\s+",
        first_sentence,
    )
    if leading_system:
        system_name = leading_system.group(1).lower().replace(" ", "-")
        candidate_title_lower = suggestion.title.lower()
        if system_name not in candidate_title_lower and candidate_mentions < 2:
            return True

    reference_verbs = (
        "extends",
        "proposes",
        "introduced",
        "presents",
        "developed",
        "uses",
        "leverages",
        "incorporates",
        "focuses on",
    )
    if citation_like and any(verb in lowered for verb in reference_verbs) and candidate_mentions < 2:
        return True

    return False


def looks_like_citation_heavy_related_work(text: str, candidate_mentions: int) -> bool:
    """Detect related-work survey passages that mostly summarize prior papers."""
    lowered = text.lower()
    citation_groups = re.findall(r"\[[0-9,\-\s]{1,30}\]", text)
    citation_numbers = sum(len(re.findall(r"\d+", group)) for group in citation_groups)
    author_year_refs = len(
        re.findall(r"\([A-Z][A-Za-z\-]+(?: et al\.)?,?\s+20\d{2}\)", text)
    )
    citation_load = citation_numbers + author_year_refs
    if citation_load < 4:
        return False

    related_markers = (
        "inspired by",
        "there has been",
        "recent progress",
        "progress in",
        "these efforts",
        "have led to",
        "development of",
        "a comprehensive survey",
        "can be found",
        "latest neural methods",
        "network-based methods",
        "prior work",
        "previous work",
        "existing methods",
        "recent methods",
        "comprehensive dense slam systems",
    )
    marker_hits = sum(1 for marker in related_markers if marker in lowered)
    if marker_hits >= 2 and candidate_mentions < 2:
        return True

    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)
    if citation_load >= 8 and sentence_count <= 5 and marker_hits >= 1 and candidate_mentions < 3:
        return True

    return False


def title_signature_tokens(title: str) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9\-]{3,}", title.lower())
    return {
        token
        for token in tokens
        if token
        not in {
            "with",
            "from",
            "using",
            "through",
            "local",
            "global",
            "large-scale",
            "consistent",
            "optimization",
            "dense",
            "photo",
            "realistic",
            "photorealistic",
            "real-time",
            "online",
            "visual",
            "mapping",
            "tracking",
        }
    }


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


def cap_confidence_for_verdict(confidence: float, verdict: str) -> float:
    cap = VERDICT_CONFIDENCE_CAPS.get(verdict, 0.5)
    return min(confidence, cap)
