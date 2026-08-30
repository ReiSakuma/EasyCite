"""Shared backend schemas for EasyCite."""
from __future__ import annotations

from typing import Literal
from typing import Any

from pydantic import BaseModel, Field


ClaimType = Literal[
    "background",
    "related_work_summary",
    "method_claim",
    "empirical_result",
    "definition",
    "common_knowledge",
    "writing_only",
    "other",
]


class CitationDetectionResult(BaseModel):
    citation_needed: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    claim_type: ClaimType
    source: Literal["llm", "heuristic"]


class EvidenceSnippet(BaseModel):
    page: int | None = None
    section: str | None = None
    text: str | None = None
    source: Literal["metadata", "abstract", "pdf", "unknown"] = "unknown"
    score: float | None = None
    quality_score: float | None = None
    is_own_claim: bool | None = None
    is_related_work_summary: bool | None = None
    supports_core_mechanism: bool | None = None
    citation_density: float | None = None
    quality_flags: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    run_id: str
    stage: str
    message: str
    candidate_count_before: int | None = None
    candidate_count_after: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    related_candidate_keys: list[str] = Field(default_factory=list)


class CitationSuggestion(BaseModel):
    paper_id: str
    title: str
    citation_key: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    verdict: Literal[
        "metadata_match",
        "foundational_support",
        "direct_support",
        "partial_support",
        "related_only",
        "contradiction",
        "insufficient_evidence",
    ] = "metadata_match"
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["current_bib", "zotero", "online", "mock"]
    evidence: EvidenceSnippet = Field(default_factory=EvidenceSnippet)
    evidence_passages: list[EvidenceSnippet] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    reason: str


class AuditClaimResult(CitationDetectionResult):
    suggestions: list[CitationSuggestion] = Field(default_factory=list)
    diagnostic_suggestions: list[CitationSuggestion] = Field(default_factory=list)
    target_trace: list[str] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    agent_state: dict[str, Any] = Field(default_factory=dict)
    agent_run_path: str | None = None
    recommended_rewrite: str | None = None
    tool_warnings: list[str] = Field(default_factory=list)
