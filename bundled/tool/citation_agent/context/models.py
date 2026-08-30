"""Serializable models used by the Context Manager."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ContextPriority = Literal["required", "compressible", "droppable"]


@dataclass(frozen=True)
class ContextItem:
    """One named piece of context with an explicit budget policy."""

    name: str
    value: Any
    priority: ContextPriority = "compressible"
    source: str = "runtime"
    purpose: str = "model_context"
    min_tokens: int = 0
    max_tokens: int | None = None
    min_items: int = 0
    protected_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextItemReport:
    name: str
    priority: ContextPriority
    source: str
    purpose: str
    original_tokens: int
    final_tokens: int
    action: Literal["kept", "compacted", "dropped"]
    reason: str


@dataclass(frozen=True)
class BudgetReport:
    stage: str
    budget_tokens: int
    original_tokens: int
    final_tokens: int
    system_tokens: int
    budget_exceeded: bool
    items: tuple[ContextItemReport, ...] = field(default_factory=tuple)

    @property
    def pruned_items(self) -> list[str]:
        return [item.name for item in self.items if item.action == "dropped"]

    @property
    def compacted_items(self) -> list[str]:
        return [item.name for item in self.items if item.action == "compacted"]

    def summary(self) -> str:
        return (
            f"Context budget stage={self.stage}, input_tokens_estimated="
            f"{self.original_tokens}->{self.final_tokens}, budget={self.budget_tokens}, "
            f"compacted_items={self.compacted_items or ['none']}, "
            f"pruned_items={self.pruned_items or ['none']}, "
            f"budget_exceeded={self.budget_exceeded}."
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "input_tokens_estimated": self.final_tokens,
                "input_tokens_estimated_before": self.original_tokens,
                "budget": self.budget_tokens,
                "pruned_items": self.pruned_items,
                "compacted_items": self.compacted_items,
            }
        )
        return value


@dataclass(frozen=True)
class ContextPacket:
    """The final model messages and their auditable budget report."""

    stage: str
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    report: BudgetReport
