"""Reliability and contract tests for the v17 Tool Registry."""
from __future__ import annotations

import pathlib
import sys
import time

import pytest


def _setup_path() -> None:
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.tools.adapters import build_default_registry  # noqa: E402
from citation_agent.tools.base import (  # noqa: E402
    RetryPolicy,
    TimeoutPolicy,
    ToolDefinition,
)
from citation_agent.tools.errors import ToolExecutionError, classify_exception  # noqa: E402
from citation_agent.tools.executor import ToolExecutor  # noqa: E402
from citation_agent.tools.observability import (  # noqa: E402
    get_tool_observations,
    reset_tool_observations,
)
from citation_agent.tools.registry import ToolRegistry  # noqa: E402


def make_registry(*definitions: ToolDefinition) -> ToolRegistry:
    registry = ToolRegistry()
    for definition in definitions:
        registry.register(definition)
    return registry


def test_tool_executor_validates_input_and_successful_output() -> None:
    definition = ToolDefinition(
        name="test.echo",
        version="1.0.0",
        description="echo",
        input_schema={"text": str},
        required_fields=("text",),
        output_validator=lambda value: isinstance(value, str),
        handler=lambda inputs: inputs["text"],
    )
    executor = ToolExecutor(make_registry(definition))

    with pytest.raises(ToolExecutionError) as error:
        executor.execute("test.echo", {"text": 3})
    assert error.value.category == "invalid_input"

    result = executor.execute("test.echo", {"text": "ok"})
    assert result.value == "ok"
    assert result.observation.output_validated is True


def test_tool_executor_retries_connection_error_then_succeeds() -> None:
    attempts = 0

    def flaky(_inputs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return ["recovered"]

    definition = ToolDefinition(
        name="test.retry",
        version="1.0.0",
        description="retry",
        output_validator=lambda value: isinstance(value, list),
        retry_policy=RetryPolicy(max_retries=1, initial_backoff_seconds=0),
        handler=flaky,
    )
    reset_tool_observations()
    result = ToolExecutor(make_registry(definition)).execute("test.retry", {})

    assert result.value == ["recovered"]
    assert attempts == 2
    observations = get_tool_observations()
    assert observations[0]["error_category"] == "connection_error"
    assert observations[-1]["success"] is True
    assert observations[-1]["attempt"] == 2


def test_tool_executor_classifies_timeout_and_invalid_output() -> None:
    timeout_tool = ToolDefinition(
        name="test.timeout",
        version="1.0.0",
        description="timeout",
        timeout_policy=TimeoutPolicy(0.01),
        handler=lambda _inputs: time.sleep(0.05),
    )
    invalid_tool = ToolDefinition(
        name="test.invalid",
        version="1.0.0",
        description="invalid",
        output_validator=lambda value: isinstance(value, list),
        handler=lambda _inputs: "not-a-list",
    )
    executor = ToolExecutor(make_registry(timeout_tool, invalid_tool))

    with pytest.raises(ToolExecutionError) as timeout_error:
        executor.execute("test.timeout", {})
    assert timeout_error.value.category == "timeout"

    with pytest.raises(ToolExecutionError) as output_error:
        executor.execute("test.invalid", {})
    assert output_error.value.category == "invalid_output"


def test_tool_executor_uses_declared_fallback() -> None:
    fallback = ToolDefinition(
        name="test.local",
        version="1.0.0",
        description="fallback",
        input_schema={"fallback_candidates": list},
        required_fields=("fallback_candidates",),
        output_validator=lambda value: isinstance(value, list),
        handler=lambda inputs: inputs["fallback_candidates"],
    )
    primary = ToolDefinition(
        name="test.remote",
        version="1.0.0",
        description="primary",
        output_validator=lambda value: isinstance(value, list),
        fallback_chain=("test.local",),
        handler=lambda _inputs: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    result = ToolExecutor(make_registry(primary, fallback)).execute(
        "test.remote",
        {"fallback_candidates": ["from-current-bib"]},
    )

    assert result.value == ["from-current-bib"]
    assert result.observation.fallback == "test.local"


def test_bibtex_writer_is_permission_guarded_and_idempotent(tmp_path) -> None:
    reset_tool_observations()
    executor = ToolExecutor(build_default_registry())
    bib_path = tmp_path / "references.bib"
    inputs = {
        "path": str(bib_path),
        "entries": [
            {
                "citation_key": "smith2025",
                "bibtex": "@article{smith2025,\n  title={A Paper},\n  year={2025}\n}",
            }
        ],
        "idempotency_key": "run-1:smith2025",
    }

    with pytest.raises(ToolExecutionError) as denied:
        executor.execute("bibtex.write", inputs)
    assert denied.value.category == "permission_denied"
    assert get_tool_observations()[-1]["attempt"] == 0
    assert get_tool_observations()[-1]["error_category"] == "permission_denied"

    first = executor.execute(
        "bibtex.write",
        inputs,
        permissions={"workspace_bibtex_write"},
    )
    second = executor.execute(
        "bibtex.write",
        inputs,
        permissions={"workspace_bibtex_write"},
    )

    assert first.value["appended"] == ["smith2025"]
    assert second.observation.cache_hit is True
    assert bib_path.read_text(encoding="utf-8").count("@article{smith2025") == 1


def test_registry_supports_capability_discovery() -> None:
    registry = build_default_registry()
    names = {tool.name for tool in registry.find_by_capability("citation_retrieval")}
    assert {"current_bib.search", "zotero.search"}.issubset(names)


def test_provider_specific_timeout_is_classified_without_sdk_coupling() -> None:
    class DeepseekTimeout(Exception):
        pass

    assert classify_exception(DeepseekTimeout("request timed out")) == ("timeout", True)
