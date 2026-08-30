"""Registration and discovery for versioned EasyCite tools."""
from __future__ import annotations

from typing import Any

from .base import ToolDefinition, ToolHealth
from .errors import ToolExecutionError


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition, *, replace: bool = False) -> None:
        if definition.name in self._tools and not replace:
            raise ToolExecutionError(
                "configuration_error",
                f"Tool {definition.name!r} is already registered.",
                tool_name=definition.name,
            )
        self._tools[definition.name] = definition

    def resolve(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolExecutionError(
                "configuration_error",
                f"Tool {name!r} is not registered.",
                tool_name=name,
            ) from exc

    def list(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in sorted(self._tools)]

    def find_by_capability(self, capability: str) -> list[ToolDefinition]:
        return [tool for tool in self.list() if capability in tool.capabilities]

    def health(self, context: dict[str, Any]) -> dict[str, ToolHealth]:
        results: dict[str, ToolHealth] = {}
        for tool in self.list():
            if tool.health_check is None:
                continue
            try:
                results[tool.name] = tool.health_check(context)
            except Exception as exc:  # pylint: disable=broad-except
                results[tool.name] = ToolHealth(
                    status="error",
                    summary=f"Health check failed: {type(exc).__name__}: {exc}",
                )
        return results
