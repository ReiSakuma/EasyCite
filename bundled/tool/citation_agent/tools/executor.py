"""Reliable execution with validation, deadlines, retries, fallback, and idempotency."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

from .base import ToolDefinition, ToolObservation, ToolResult
from .errors import ToolExecutionError, classify_exception
from .observability import record_tool_observation
from .registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._idempotency_cache: dict[tuple[str, str], Any] = {}
        self._failure_counts: dict[str, int] = {}
        self._circuits_open_until: dict[str, float] = {}
        self._lock = threading.RLock()

    def execute(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        *,
        operation: Callable[[dict[str, Any]], Any] | None = None,
        permissions: set[str] | None = None,
        timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
        allow_fallback: bool = True,
    ) -> ToolResult:
        definition = self.registry.resolve(tool_name)
        tool_call_id = uuid.uuid4().hex
        input_validated = False
        try:
            self._validate_input(definition, inputs)
            input_validated = True
            self._check_permission(definition, permissions or set())
            self._check_circuit(definition)
        except Exception as exc:
            category, retryable = classify_exception(exc)
            record_tool_observation(
                ToolObservation.create(
                    tool_name,
                    definition.version,
                    tool_call_id=tool_call_id,
                    success=False,
                    attempt=0,
                    latency_ms=0.0,
                    error_category=category,
                    error_message=str(exc),
                    input_validated=input_validated,
                    output_validated=False,
                )
            )
            if isinstance(exc, ToolExecutionError):
                raise
            raise ToolExecutionError(
                category,
                str(exc),
                retryable=retryable,
                tool_name=tool_name,
            ) from exc

        key = idempotency_key or str(inputs.get(definition.idempotency_policy.key_field) or "")
        if definition.idempotency_policy.require_key and not key:
            raise ToolExecutionError(
                "invalid_input",
                f"Tool {tool_name} requires an idempotency key.",
                tool_name=tool_name,
            )
        if definition.idempotency_policy.enabled and key:
            with self._lock:
                cached = self._idempotency_cache.get((tool_name, key))
            if cached is not None:
                observation = ToolObservation.create(
                    tool_name,
                    definition.version,
                    tool_call_id=tool_call_id,
                    success=True,
                    attempt=0,
                    latency_ms=0.0,
                    input_validated=True,
                    output_validated=True,
                    idempotency_key=key,
                    cache_hit=True,
                    output_summary=summarize_output(cached),
                )
                record_tool_observation(observation)
                return ToolResult(cached, observation)

        handler = operation or definition.handler
        if handler is None:
            raise ToolExecutionError(
                "configuration_error",
                f"Tool {tool_name} has no execution handler.",
                tool_name=tool_name,
            )

        attempts = max(definition.retry_policy.max_retries, 0) + 1
        last_error: ToolExecutionError | None = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                value = self._call_with_deadline(
                    handler,
                    inputs,
                    timeout_seconds or definition.timeout_policy.seconds,
                )
                self._validate_output(definition, value)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                observation = ToolObservation.create(
                    tool_name,
                    definition.version,
                    tool_call_id=tool_call_id,
                    success=True,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    input_validated=True,
                    output_validated=True,
                    idempotency_key=key or None,
                    output_summary=summarize_output(value),
                )
                self._record_success(definition.name)
                if definition.idempotency_policy.enabled and key:
                    with self._lock:
                        self._idempotency_cache[(tool_name, key)] = value
                record_tool_observation(observation)
                return ToolResult(value, observation)
            except Exception as exc:  # pylint: disable=broad-except
                category, retryable = classify_exception(exc)
                if isinstance(exc, ToolExecutionError):
                    message = str(exc)
                else:
                    message = f"{type(exc).__name__}: {exc}"
                last_error = ToolExecutionError(
                    category,
                    message,
                    retryable=retryable,
                    tool_name=tool_name,
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                record_tool_observation(
                    ToolObservation.create(
                        tool_name,
                        definition.version,
                        tool_call_id=tool_call_id,
                        success=False,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        error_category=category,
                        error_message=message,
                        input_validated=True,
                        output_validated=False,
                        idempotency_key=key or None,
                    )
                )
                should_retry = (
                    attempt < attempts
                    and retryable
                    and category in definition.retry_policy.retryable_categories
                )
                if should_retry:
                    delay = definition.retry_policy.initial_backoff_seconds * (
                        definition.retry_policy.backoff_multiplier ** (attempt - 1)
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
                break

        self._record_failure(definition.name, last_error.category if last_error else "internal_error")
        if allow_fallback:
            fallback = self._execute_fallbacks(definition, inputs, permissions, tool_call_id)
            if fallback is not None:
                return fallback
        assert last_error is not None
        raise last_error

    def _execute_fallbacks(
        self,
        definition: ToolDefinition,
        inputs: dict[str, Any],
        permissions: set[str],
        parent_call_id: str,
    ) -> ToolResult | None:
        for fallback_name in definition.fallback_chain:
            try:
                result = self.execute(
                    fallback_name,
                    inputs,
                    permissions=permissions,
                    allow_fallback=False,
                )
            except ToolExecutionError:
                continue
            observation = ToolObservation.create(
                definition.name,
                definition.version,
                tool_call_id=parent_call_id,
                success=True,
                attempt=result.observation.attempt,
                latency_ms=result.observation.latency_ms,
                fallback=fallback_name,
                input_validated=True,
                output_validated=True,
                output_summary=summarize_output(result.value),
            )
            record_tool_observation(observation)
            return ToolResult(result.value, observation)
        return None

    @staticmethod
    def _call_with_deadline(
        handler: Callable[[dict[str, Any]], Any],
        inputs: dict[str, Any],
        timeout_seconds: float,
    ) -> Any:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="citation-tool")
        future = pool.submit(handler, inputs)
        try:
            return future.result(timeout=max(float(timeout_seconds), 0.001))
        except FutureTimeoutError as exc:
            future.cancel()
            raise ToolExecutionError(
                "timeout",
                f"Tool deadline exceeded after {timeout_seconds:.2f}s.",
                retryable=True,
            ) from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _validate_input(definition: ToolDefinition, inputs: dict[str, Any]) -> None:
        if not isinstance(inputs, dict):
            raise ToolExecutionError("invalid_input", "Tool input must be an object.", tool_name=definition.name)
        missing = [field for field in definition.required_fields if field not in inputs]
        if missing:
            raise ToolExecutionError(
                "invalid_input",
                f"Missing required tool input(s): {', '.join(missing)}.",
                tool_name=definition.name,
            )
        for field, expected_type in definition.input_schema.items():
            if field in inputs and not isinstance(inputs[field], expected_type):
                raise ToolExecutionError(
                    "invalid_input",
                    f"Tool input {field!r} has type {type(inputs[field]).__name__}; expected {expected_type}.",
                    tool_name=definition.name,
                )

    @staticmethod
    def _validate_output(definition: ToolDefinition, value: Any) -> None:
        if definition.output_validator and not definition.output_validator(value):
            raise ToolExecutionError(
                "invalid_output",
                f"Tool {definition.name} returned an invalid output contract.",
                tool_name=definition.name,
            )

    @staticmethod
    def _check_permission(definition: ToolDefinition, permissions: set[str]) -> None:
        policy = definition.permission_policy
        if policy.required_permission and policy.required_permission not in permissions and not policy.allow_by_default:
            raise ToolExecutionError(
                "permission_denied",
                f"Tool {definition.name} requires permission {policy.required_permission!r}.",
                tool_name=definition.name,
            )

    def _check_circuit(self, definition: ToolDefinition) -> None:
        with self._lock:
            open_until = self._circuits_open_until.get(definition.name, 0.0)
        if open_until > time.monotonic():
            raise ToolExecutionError(
                "connection_error",
                f"Tool circuit is temporarily open for {definition.name}.",
                retryable=True,
                tool_name=definition.name,
            )

    def _record_success(self, name: str) -> None:
        with self._lock:
            self._failure_counts[name] = 0
            self._circuits_open_until.pop(name, None)

    def _record_failure(self, name: str, category: str) -> None:
        if category not in {"connection_error", "timeout", "rate_limited"}:
            return
        with self._lock:
            count = self._failure_counts.get(name, 0) + 1
            self._failure_counts[name] = count
            if count >= 3:
                self._circuits_open_until[name] = time.monotonic() + 20.0


def summarize_output(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple, set, dict)):
        return f"{type(value).__name__}[{len(value)}]"
    text = str(value).replace("\n", " ")
    return text[:160]


_DEFAULT_EXECUTOR: ToolExecutor | None = None


def get_default_executor() -> ToolExecutor:
    global _DEFAULT_EXECUTOR  # pylint: disable=global-statement
    if _DEFAULT_EXECUTOR is None:
        from .adapters import build_default_registry

        _DEFAULT_EXECUTOR = ToolExecutor(build_default_registry())
    return _DEFAULT_EXECUTOR
