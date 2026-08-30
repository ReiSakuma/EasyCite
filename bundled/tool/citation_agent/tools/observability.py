"""Request-local collection of tool observations."""
from __future__ import annotations

from contextvars import ContextVar

from .base import ToolObservation


_OBSERVATIONS: ContextVar[tuple[ToolObservation, ...]] = ContextVar(
    "citation_agent_tool_observations",
    default=(),
)


def reset_tool_observations() -> None:
    _OBSERVATIONS.set(())


def record_tool_observation(observation: ToolObservation) -> None:
    _OBSERVATIONS.set((*_OBSERVATIONS.get(), observation))


def get_tool_observations() -> list[dict]:
    return [observation.to_dict() for observation in _OBSERVATIONS.get()]
