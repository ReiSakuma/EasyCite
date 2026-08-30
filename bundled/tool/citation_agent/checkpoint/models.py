"""Serializable checkpoint contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RunStatus = Literal["running", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class CheckpointRecord:
    run_id: str
    request_hash: str
    graph_version: str
    schema_version: str
    current_node: str
    completed_nodes: tuple[str, ...] = ()
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    pending_actions: tuple[str, ...] = ()
    tool_idempotency_keys: tuple[str, ...] = ()
    status: RunStatus = "running"
    cancellation_state: str = "not_requested"
    last_error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_hash": self.request_hash,
            "graph_version": self.graph_version,
            "schema_version": self.schema_version,
            "current_node": self.current_node,
            "completed_nodes": list(self.completed_nodes),
            "state_snapshot": self.state_snapshot,
            "pending_actions": list(self.pending_actions),
            "tool_idempotency_keys": list(self.tool_idempotency_keys),
            "status": self.status,
            "cancellation_state": self.cancellation_state,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }
