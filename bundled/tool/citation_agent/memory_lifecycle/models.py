"""Stable data contracts for lifecycle-managed memory records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MemoryScope = Literal["session", "project", "user", "bad_case"]
MemoryStatus = Literal["active", "superseded", "invalid"]


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    memory_type: str
    scope: MemoryScope
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float | None = None
    version: int = 1
    status: MemoryStatus = "active"
    evidence_refs: tuple[str, ...] = ()
    correction_of: str | None = None

    def is_active(self, now: float) -> bool:
        return self.status == "active" and (
            self.expires_at is None or self.expires_at > now
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "memory_type": self.memory_type,
            "scope": self.scope,
            "content": self.content,
            "payload": self.payload,
            "source": self.source,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "version": self.version,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "correction_of": self.correction_of,
        }
