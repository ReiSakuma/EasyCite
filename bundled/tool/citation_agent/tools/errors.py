"""Structured failure categories for external and host tools."""
from __future__ import annotations

import json
import socket
import urllib.error
from typing import Any, Literal


ToolErrorCategory = Literal[
    "configuration_error",
    "dependency_missing",
    "authentication_error",
    "connection_error",
    "timeout",
    "rate_limited",
    "invalid_input",
    "invalid_output",
    "not_found",
    "permission_denied",
    "internal_error",
]


class ToolExecutionError(RuntimeError):
    """A business-safe tool error with a searchable category."""

    def __init__(
        self,
        category: ToolErrorCategory,
        message: str,
        *,
        retryable: bool = False,
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.tool_name = tool_name
        self.details = details or {}


def classify_exception(exc: BaseException) -> tuple[ToolErrorCategory, bool]:
    if isinstance(exc, ToolExecutionError):
        return exc.category, exc.retryable
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout", True
    if "timeout" in name or "timed out" in message or "time out" in message:
        return "timeout", True
    if "ratelimit" in name or "rate limit" in message or "too many requests" in message:
        return "rate_limited", True
    if "authentication" in name or "unauthorized" in message or "invalid api key" in message:
        return "authentication_error", False
    if "connection" in name or "connection refused" in message:
        return "connection_error", True
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "dependency_missing", False
    if isinstance(exc, PermissionError):
        return "permission_denied", False
    if isinstance(exc, FileNotFoundError):
        return "not_found", False
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {401, 403}:
            return "authentication_error", False
        if exc.code == 404:
            return "not_found", False
        if exc.code == 429:
            return "rate_limited", True
        return "connection_error", 500 <= exc.code < 600
    if isinstance(exc, (urllib.error.URLError, ConnectionError, ConnectionRefusedError, OSError)):
        return "connection_error", True
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_output", False
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "invalid_input", False
    return "internal_error", False
