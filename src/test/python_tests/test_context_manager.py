"""Tests for v16 budgeted LLM context construction."""
from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace


def _setup_path() -> None:
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.config import ModelConfig  # noqa: E402
from citation_agent.context import (  # noqa: E402
    ContextItem,
    ContextManager,
    ContextPolicy,
    estimate_text_tokens,
)
from citation_agent.query_planner import call_litellm_query_planner  # noqa: E402


def test_context_manager_is_deterministic_and_respects_budget() -> None:
    manager = ContextManager(ContextPolicy(stage="test", token_budget=220))
    items = [
        ContextItem("claim", "A required scientific claim.", priority="required"),
        ContextItem(
            "paragraph_context",
            "Background context about mapping and optimization. " * 80,
            priority="compressible",
            min_tokens=30,
            max_tokens=90,
        ),
        ContextItem(
            "debug_log",
            "Repeated low-value observation. " * 100,
            priority="droppable",
        ),
    ]

    first = manager.build_json_messages(system_prompt="Return JSON.", items=items)
    second = manager.build_json_messages(system_prompt="Return JSON.", items=items)

    assert first.messages == second.messages
    assert first.report.to_dict() == second.report.to_dict()
    assert first.payload["claim"] == "A required scientific claim."
    assert "debug_log" not in first.payload
    assert "[truncated]" in first.payload["paragraph_context"]
    assert first.report.final_tokens <= first.report.budget_tokens


def test_context_manager_preserves_nested_schema_and_identifiers() -> None:
    candidates = [
        {
            "candidate_id": f"paper-{index}",
            "title": f"Paper {index}",
            "passages": [
                {"index": 0, "source": "pdf", "text": "Evidence sentence. " * 120}
            ],
        }
        for index in range(3)
    ]
    packet = ContextManager(
        ContextPolicy(stage="evidence_verification", token_budget=520)
    ).build_json_messages(
        system_prompt="Judge each candidate and return JSON.",
        items=[
            ContextItem("claim", "Claim.", priority="required"),
            ContextItem(
                "candidates",
                candidates,
                priority="compressible",
                min_tokens=300,
                max_tokens=430,
                min_items=3,
                protected_keys=("candidate_id", "title", "index", "source"),
            ),
        ],
    )

    assert [item["candidate_id"] for item in packet.payload["candidates"]] == [
        "paper-0",
        "paper-1",
        "paper-2",
    ]
    assert all(item["passages"][0]["text"] for item in packet.payload["candidates"])
    assert packet.report.final_tokens <= packet.report.budget_tokens


def test_context_manager_reports_impossible_required_overflow() -> None:
    packet = ContextManager(ContextPolicy(stage="test", token_budget=64)).build_json_messages(
        system_prompt="System instruction.",
        items=[
            ContextItem("required", "Cannot be removed. " * 100, priority="required"),
        ],
    )

    assert packet.payload["required"].startswith("Cannot be removed.")
    assert packet.report.budget_exceeded is True
    item_report = packet.report.to_dict()["items"][0]
    assert item_report["reason"] == "required_item_preserved_despite_budget_overflow"


def test_context_manager_deduplicates_lists_and_reports_provenance() -> None:
    packet = ContextManager(ContextPolicy(stage="test", token_budget=300)).build_json_messages(
        system_prompt="Return JSON.",
        items=[
            ContextItem(
                "observations",
                ["same observation", "same observation", "different observation"],
                source="tool_results",
                purpose="avoid_repeated_tool_context",
            )
        ],
    )

    assert packet.payload["observations"] == ["same observation", "different observation"]
    report = packet.report.to_dict()
    assert report["input_tokens_estimated"] == report["final_tokens"]
    assert report["budget"] == 300
    assert report["items"][0]["source"] == "tool_results"
    assert report["items"][0]["purpose"] == "avoid_repeated_tool_context"
    assert report["items"][0]["action"] == "compacted"


def test_query_planner_uses_context_manager_and_returns_budget_report(monkeypatch) -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        content = json.dumps(
            {
                "queries": ["neural mapping"],
                "concepts": ["neural representation"],
                "required_concepts": ["differentiable representation"],
                "foundational_concepts": ["neural radiance field"],
                "excluded_concepts": ["geometric map"],
                "claim_focus": "neural mapping",
                "reason": "test",
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))
    config = ModelConfig(
        use_llm=True,
        provider="test",
        model="test-model",
        api_key="test-key",
        api_base=None,
        context_query_budget_tokens=700,
    )
    plan = call_litellm_query_planner(
        claim="Recent neural representations changed mapping.",
        context="Long paragraph context. " * 500,
        base_queries=["neural mapping", "differentiable rendering"],
        config=config,
    )

    assert plan.source == "llm"
    assert plan.context_report["stage"] == "query_planning"
    assert plan.context_report["final_tokens"] <= 700
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["claim"] == "Recent neural representations changed mapping."
    assert payload["max_queries"] == 10


def test_token_estimator_counts_cjk_conservatively() -> None:
    assert estimate_text_tokens("citation context") > 0
    assert estimate_text_tokens("科研引用") >= 4
