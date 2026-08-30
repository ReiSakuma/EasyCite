"""Project-local feedback memory for EasyCite recommendations."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .bibliography import tokenize
from .cache import default_cache_dir, stable_text_hash
from .schemas import CitationSuggestion
from .memory_lifecycle.service import MemoryService


FeedbackAction = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class FeedbackRecord:
    action: FeedbackAction
    claim: str
    context: str
    paper_identity: str
    title: str
    citation_key: str | None
    source: str
    created_at: float


@dataclass(frozen=True)
class FeedbackAdjustment:
    identity: str
    action: FeedbackAction
    delta: float
    similarity: float
    reason: str


@dataclass(frozen=True)
class FeedbackRankingResult:
    suggestions: list[CitationSuggestion]
    adjustments: dict[str, FeedbackAdjustment] = field(default_factory=dict)
    summary: str = ""


class CandidateFeedbackMemory:
    """Persist accepted/rejected candidate feedback under the current project."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._available = True

    @classmethod
    def for_workspace(cls, workspace_path: str, memory_path: str | None = None) -> "CandidateFeedbackMemory":
        if memory_path:
            return cls(Path(memory_path))
        if workspace_path:
            return cls(Path(workspace_path) / ".citation-agent" / "memory.sqlite3")
        return cls(default_cache_dir() / "memory.sqlite3")

    def record_feedback(
        self,
        *,
        workspace_path: str,
        claim: str,
        context: str,
        suggestion: CitationSuggestion,
        action: FeedbackAction,
    ) -> dict[str, Any]:
        if action not in {"accepted", "rejected"}:
            raise ValueError(f"Unsupported feedback action: {action}")
        identity = suggestion_identity(suggestion)
        now = time.time()
        payload = {
            "workspace_path": workspace_path,
            "claim": claim,
            "context": context,
            "paper_identity": identity,
            "title": suggestion.title,
            "citation_key": suggestion.citation_key,
            "source": suggestion.source,
            "action": action,
            "created_at": now,
        }
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.execute(
                    """
                    INSERT INTO candidate_feedback(
                        workspace_hash, claim_hash, paper_identity, action,
                        title, citation_key, source, claim_text, context_text, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_text_hash(workspace_path or "<no-workspace>"),
                        stable_text_hash(claim),
                        identity,
                        action,
                        suggestion.title,
                        suggestion.citation_key,
                        suggestion.source,
                        claim,
                        context,
                        now,
                    ),
                )
            # Keep the legacy feedback table readable while also recording a
            # governed project-memory event with provenance and scope.
            MemoryService.for_workspace(
                workspace_path,
                str(self.db_path),
            ).put(
                memory_type="candidate_feedback",
                scope="project",
                content=f"{action} candidate: {suggestion.title}",
                payload={"action": action, "paper_identity": identity, "citation_key": suggestion.citation_key},
                source="explicit_user_feedback",
                provenance={"claim_hash": stable_text_hash(claim), "context_hash": stable_text_hash(context)},
                confidence=0.35,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._available = False
            return {**payload, "recorded": False, "error": f"{type(exc).__name__}: {exc}"}
        return {**payload, "recorded": True}

    def clear(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                row = conn.execute("SELECT COUNT(*) FROM candidate_feedback").fetchone()
                count = int(row[0]) if row else 0
                conn.execute("DELETE FROM candidate_feedback")
        except Exception as exc:  # pylint: disable=broad-except
            self._available = False
            return {"cleared": False, "count": 0, "error": f"{type(exc).__name__}: {exc}"}
        return {"cleared": True, "count": count, "path": str(self.db_path)}

    def load_recent(self, limit: int = 500) -> list[FeedbackRecord]:
        if not self._available:
            return []
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                rows = conn.execute(
                    """
                    SELECT action, claim_text, context_text, paper_identity, title,
                           citation_key, source, created_at
                    FROM candidate_feedback
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (max(limit, 1),),
                ).fetchall()
        except Exception:  # pylint: disable=broad-except
            self._available = False
            return []
        return [
            FeedbackRecord(
                action=str(action),  # type: ignore[arg-type]
                claim=str(claim_text or ""),
                context=str(context_text or ""),
                paper_identity=str(paper_identity or ""),
                title=str(title or ""),
                citation_key=str(citation_key) if citation_key else None,
                source=str(source or ""),
                created_at=float(created_at or 0),
            )
            for action, claim_text, context_text, paper_identity, title, citation_key, source, created_at in rows
            if action in {"accepted", "rejected"} and paper_identity
        ]

    def rank_with_feedback(
        self,
        *,
        suggestions: list[CitationSuggestion],
        claim: str,
        context: str,
        positive_boost: float,
        negative_penalty: float,
        min_similarity: float,
    ) -> FeedbackRankingResult:
        if not suggestions:
            return FeedbackRankingResult(suggestions=[])
        records = self.load_recent()
        if not records:
            return FeedbackRankingResult(suggestions=suggestions, summary="memory enabled, no feedback yet")

        claim_terms = feedback_terms(claim, context)
        now = time.time()
        adjustments: dict[str, FeedbackAdjustment] = {}
        adjusted: list[CitationSuggestion] = []
        for suggestion in suggestions:
            identity = suggestion_identity(suggestion)
            matched = [
                record
                for record in records
                if record.paper_identity == identity
                or (
                    suggestion.citation_key
                    and record.citation_key
                    and suggestion.citation_key.lower() == record.citation_key.lower()
                )
            ]
            if not matched:
                adjusted.append(suggestion)
                continue

            score = 0.0
            best_similarity = 0.0
            accepted_hits = 0
            rejected_hits = 0
            for record in matched:
                similarity = jaccard_similarity(claim_terms, feedback_terms(record.claim, record.context))
                if similarity < min_similarity:
                    continue
                age_days = max((now - record.created_at) / 86400.0, 0.0)
                recency_weight = max(0.35, 1.0 / (1.0 + age_days / 90.0))
                best_similarity = max(best_similarity, similarity)
                if record.action == "accepted":
                    accepted_hits += 1
                    score += positive_boost * similarity * recency_weight
                else:
                    rejected_hits += 1
                    score -= negative_penalty * similarity * recency_weight

            if abs(score) < 0.001:
                adjusted.append(suggestion)
                continue

            delta = max(-negative_penalty, min(positive_boost, score))
            new_confidence = min(1.0, max(0.0, suggestion.confidence + delta))
            action: FeedbackAction = "accepted" if delta > 0 else "rejected"
            reason = (
                f"Memory feedback: {action}, delta={delta:+.2f}, "
                f"similar_claim={best_similarity:.2f}, accepted={accepted_hits}, rejected={rejected_hits}."
            )
            adjustments[identity] = FeedbackAdjustment(
                identity=identity,
                action=action,
                delta=round(delta, 4),
                similarity=round(best_similarity, 4),
                reason=reason,
            )
            adjusted.append(
                suggestion.model_copy(
                    update={
                        "confidence": new_confidence,
                        "reason": f"{suggestion.reason} {reason}",
                    }
                )
            )

        if adjustments:
            adjusted = sorted(
                adjusted,
                key=lambda item: (
                    verdict_priority(item),
                    -round(item.confidence, 6),
                    normalize_title(item.title),
                    (item.citation_key or item.paper_id).lower(),
                ),
            )
        summary = (
            f"memory enabled, records={len(records)}, adjustments={len(adjustments)}"
        )
        return FeedbackRankingResult(
            suggestions=adjusted,
            adjustments=adjustments,
            summary=summary,
        )

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_hash TEXT NOT NULL,
                claim_hash TEXT NOT NULL,
                paper_identity TEXT NOT NULL,
                action TEXT NOT NULL,
                title TEXT NOT NULL,
                citation_key TEXT,
                source TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                context_text TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_feedback_identity ON candidate_feedback(paper_identity)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_feedback_created ON candidate_feedback(created_at)"
        )


def record_candidate_feedback(params: dict[str, Any]) -> dict[str, Any]:
    suggestion = CitationSuggestion.model_validate(params.get("suggestion") or {})
    memory = CandidateFeedbackMemory.for_workspace(
        str(params.get("workspace_path") or ""),
        str(params.get("memory_path") or "") or None,
    )
    return memory.record_feedback(
        workspace_path=str(params.get("workspace_path") or ""),
        claim=str(params.get("claim") or ""),
        context=str(params.get("context") or ""),
        suggestion=suggestion,
        action=str(params.get("action") or "accepted"),  # type: ignore[arg-type]
    )


def clear_candidate_feedback(params: dict[str, Any]) -> dict[str, Any]:
    memory = CandidateFeedbackMemory.for_workspace(
        str(params.get("workspace_path") or ""),
        str(params.get("memory_path") or "") or None,
    )
    return memory.clear()


def feedback_terms(claim: str, context: str) -> set[str]:
    terms = tokenize(f"{claim} {context}")
    return {
        term
        for term in terms
        if len(term) >= 4
        and term
        not in {
            "paper",
            "papers",
            "result",
            "results",
            "method",
            "methods",
            "system",
            "systems",
            "using",
            "with",
            "from",
            "that",
            "this",
            "these",
            "those",
        }
    }


def jaccard_similarity(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def suggestion_identity(suggestion: CitationSuggestion) -> str:
    doi = normalize_doi(suggestion.doi)
    if doi:
        return f"doi:{doi}"
    title = normalize_title(suggestion.title)
    if title:
        return f"title:{title}"
    if suggestion.citation_key:
        return f"citekey:{suggestion.citation_key.lower()}"
    return f"paper:{suggestion.paper_id.lower()}"


def normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    lowered = doi.lower().strip()
    lowered = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", lowered)
    lowered = re.sub(r"^doi:\s*", "", lowered)
    return lowered.rstrip("./")


def normalize_title(title: str) -> str:
    lowered = title.lower()
    lowered = re.sub(r"\{|\}|\\[a-zA-Z]+\s*", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(token for token in lowered.split() if token)


def verdict_priority(suggestion: CitationSuggestion) -> int:
    order = {
        "direct_support": -5,
        "foundational_support": -4,
        "partial_support": -3,
        "related_only": -2,
        "metadata_match": -1,
        "insufficient_evidence": 0,
        "contradiction": 0,
    }
    return order.get(suggestion.verdict, 99)


def render_feedback_summary(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
