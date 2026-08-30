"""Serializable node-state types for EasyCite's graph-shaped run records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict


class EasyCiteState(TypedDict, total=False):
    """Checkpoint-ready state shape shared with a future LangGraph StateGraph."""

    run_id: str
    request: dict[str, Any]
    runtime: dict[str, Any]
    claim_profile: dict[str, Any]
    retrieval_plan: dict[str, Any]
    candidate_pool: dict[str, Any]
    verification: dict[str, Any]
    output: dict[str, Any]
    trace: dict[str, Any]
    graph: dict[str, Any]


@dataclass(frozen=True)
class AgentNodeRecord:
    """One observable workflow node derived from a TraceEvent."""

    name: str
    stage: str
    kind: str
    message: str
    candidate_count_before: int | None
    candidate_count_after: int | None
    latency_ms: float | None
    checkpoint: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
