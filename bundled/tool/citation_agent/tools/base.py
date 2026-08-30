"""Core contracts for EasyCite tools."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from .errors import ToolErrorCategory


ToolStatus = Literal["ok", "warning", "error", "disabled"]
ToolHandler = Callable[[dict[str, Any]], Any]
OutputValidator = Callable[[Any], bool]
HealthCheck = Callable[[dict[str, Any]], "ToolHealth"]


@dataclass(frozen=True)
class TimeoutPolicy:
    seconds: float = 20.0


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0
    initial_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    retryable_categories: tuple[ToolErrorCategory, ...] = (
        "connection_error",
        "timeout",
        "rate_limited",
    )


@dataclass(frozen=True)
class IdempotencyPolicy:
    enabled: bool = False
    key_field: str = "idempotency_key"
    require_key: bool = False


@dataclass(frozen=True)
class PermissionPolicy:
    required_permission: str | None = None
    allow_by_default: bool = True


@dataclass(frozen=True)
class ToolHealth:
    status: ToolStatus
    summary: str
    details: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    version: str
    description: str
    input_schema: dict[str, type | tuple[type, ...]] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()
    output_validator: OutputValidator | None = None
    read_only: bool = True
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    idempotency_policy: IdempotencyPolicy = field(default_factory=IdempotencyPolicy)
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    fallback_chain: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    handler: ToolHandler | None = None
    health_check: HealthCheck | None = None

    @property
    def mutating(self) -> bool:
        return not self.read_only

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "read_only": self.read_only,
            "required_fields": list(self.required_fields),
            "timeout_seconds": self.timeout_policy.seconds,
            "max_retries": self.retry_policy.max_retries,
            "idempotent": self.idempotency_policy.enabled,
            "permission": self.permission_policy.required_permission,
            "fallback_chain": list(self.fallback_chain),
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class ToolObservation:
    tool_call_id: str
    tool_name: str
    tool_version: str
    success: bool
    attempt: int
    latency_ms: float
    fallback: str | None = None
    error_category: ToolErrorCategory | None = None
    error_message: str | None = None
    input_validated: bool = False
    output_validated: bool = False
    idempotency_key: str | None = None
    cache_hit: bool = False
    output_summary: str = ""

    @classmethod
    def create(cls, tool_name: str, tool_version: str, **kwargs: Any) -> "ToolObservation":
        return cls(
            tool_call_id=kwargs.pop("tool_call_id", uuid.uuid4().hex),
            tool_name=tool_name,
            tool_version=tool_version,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    value: Any
    observation: ToolObservation

    @property
    def success(self) -> bool:
        return self.observation.success
