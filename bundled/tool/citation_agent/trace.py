"""Trace helpers for observable EasyCite runs."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from .schemas import CitationSuggestion, TraceEvent


class TraceRecorder:
    """Collect coarse run events and per-candidate lifecycle notes."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self._events: list[TraceEvent] = []
        self._candidate_notes: dict[str, list[str]] = defaultdict(list)
        self._stage_started_at: dict[str, float] = {}

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def start(self, stage: str) -> None:
        self._stage_started_at[stage] = time.perf_counter()

    def record(
        self,
        stage: str,
        message: str,
        *,
        before: int | None = None,
        after: int | None = None,
        error: str | None = None,
        candidates: Iterable[CitationSuggestion] | None = None,
        identity: Callable[[CitationSuggestion], str] | None = None,
    ) -> None:
        started_at = self._stage_started_at.pop(stage, None)
        latency_ms = None
        if started_at is not None:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        candidate_keys = []
        if candidates is not None:
            candidate_keys = [
                suggestion.citation_key or suggestion.paper_id
                for suggestion in candidates
            ][:20]
        self._events.append(
            TraceEvent(
                run_id=self.run_id,
                stage=stage,
                message=message,
                candidate_count_before=before,
                candidate_count_after=after,
                latency_ms=latency_ms,
                error=error,
                related_candidate_keys=candidate_keys,
            )
        )
        if candidates is not None and identity is not None:
            for suggestion in candidates:
                self.note_candidate(identity(suggestion), stage, message)

    def note_candidate(self, candidate_id: str, stage: str, message: str) -> None:
        rendered = f"{stage}: {message}"
        notes = self._candidate_notes[candidate_id]
        if rendered not in notes:
            notes.append(rendered)

    def notes_for(self, candidate_id: str) -> list[str]:
        return list(self._candidate_notes.get(candidate_id, []))

    def attach_candidate_trace(
        self,
        suggestions: list[CitationSuggestion],
        identity: Callable[[CitationSuggestion], str],
    ) -> list[CitationSuggestion]:
        traced = []
        for suggestion in suggestions:
            notes = self.notes_for(identity(suggestion))
            if not notes:
                traced.append(suggestion)
                continue
            merged = [*suggestion.trace]
            for note in notes:
                if note not in merged:
                    merged.append(note)
            traced.append(suggestion.model_copy(update={"trace": merged}))
        return traced


def summarize_candidates(suggestions: list[CitationSuggestion], limit: int = 5) -> str:
    if not suggestions:
        return "none"
    return ", ".join(
        f"{suggestion.title} (@{suggestion.citation_key or '<none>'})"
        for suggestion in suggestions[:limit]
    )


def event_dicts(events: list[TraceEvent]) -> list[dict[str, Any]]:
    return [event.model_dump() for event in events]
