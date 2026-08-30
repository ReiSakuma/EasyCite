"""Unified tool contracts, registry, adapters, and reliable execution."""

from .adapters import (
    build_default_registry,
    execute_better_bibtex_export,
    execute_embedding,
    execute_litellm_completion,
    execute_pdf_reader,
    execute_zotero_search,
)
from .base import (
    IdempotencyPolicy,
    PermissionPolicy,
    RetryPolicy,
    TimeoutPolicy,
    ToolDefinition,
    ToolObservation,
    ToolResult,
)
from .errors import ToolErrorCategory, ToolExecutionError
from .executor import ToolExecutor, get_default_executor
from .observability import get_tool_observations, reset_tool_observations
from .registry import ToolRegistry

__all__ = [
    "IdempotencyPolicy",
    "PermissionPolicy",
    "RetryPolicy",
    "TimeoutPolicy",
    "ToolDefinition",
    "ToolErrorCategory",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolObservation",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "execute_better_bibtex_export",
    "execute_embedding",
    "execute_litellm_completion",
    "execute_pdf_reader",
    "execute_zotero_search",
    "get_default_executor",
    "get_tool_observations",
    "reset_tool_observations",
]
