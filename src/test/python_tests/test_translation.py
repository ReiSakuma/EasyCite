"""Tests for on-demand evidence translation."""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import patch


def _setup_path():
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.config import ModelConfig  # noqa: E402
from citation_agent.translation import (  # noqa: E402
    split_translation_texts,
    normalize_translation_texts,
    translate_evidence,
)


def test_translate_evidence_reports_unavailable_without_model_config():
    result = translate_evidence(
        {
            "texts": [
                {
                    "id": "claim",
                    "text": "Existing multi-agent Gaussian SLAM methods rely on optimization.",
                }
            ]
        },
        model_config=ModelConfig(
            use_llm=False,
            provider="openai",
            model="gpt-4o-mini",
            api_key=None,
            api_base=None,
        ),
    )

    assert result["available"] is False
    assert "未配置模型 API" in result["reason"]
    assert result["translations"] == []


def test_normalize_translation_texts_deduplicates_and_limits_empty_values():
    texts = normalize_translation_texts(
        [
            {"id": "a", "text": "  First   text. "},
            {"id": "a", "text": "Duplicate id."},
            {"id": "b", "text": ""},
            {"id": "c", "text": "Second text."},
        ]
    )

    assert texts == [
        {"id": "a", "text": "First text."},
        {"id": "c", "text": "Second text."},
    ]


def test_normalize_translation_texts_accepts_object_items():
    item = type(
        "RequestObject",
        (),
        {
            "id": "claim",
            "text": " Existing  methods   rely on optimization. ",
        },
    )()

    texts = normalize_translation_texts([item])

    assert texts == [
        {"id": "claim", "text": "Existing methods rely on optimization."},
    ]


def test_translate_evidence_returns_partial_results_when_batch_times_out():
    config = ModelConfig(
        use_llm=True,
        provider="deepseek",
        model="deepseek-chat",
        api_key="test-key",
        api_base=None,
        timeout_seconds=20,
        translation_timeout_seconds=60,
    )

    def _fake_translate(texts, _config):
        if len(texts) > 1:
            raise TimeoutError("batch timeout")
        if texts[0]["id"] == "passage-1":
            raise TimeoutError("single timeout")
        return [{"id": texts[0]["id"], "zh": f"中文：{texts[0]['text']}"}]

    with patch("citation_agent.translation.call_litellm_translation", _fake_translate):
        result = translate_evidence(
            {
                "texts": [
                    {"id": "claim", "text": "Claim."},
                    {"id": "passage-1", "text": "Slow passage."},
                    {"id": "passage-2", "text": "Fast passage."},
                ]
            },
            model_config=config,
        )

    assert result["available"] is True
    assert "部分生成" in result["reason"]
    assert result["translations"] == [
        {"id": "claim", "zh": "中文：Claim."},
        {"id": "passage-2", "zh": "中文：Fast passage."},
    ]


def test_translate_evidence_splits_long_passage_and_combines_original_id():
    config = ModelConfig(
        use_llm=True,
        provider="deepseek",
        model="deepseek-chat",
        api_key="test-key",
        api_base=None,
        timeout_seconds=20,
        translation_timeout_seconds=60,
    )
    long_text = "Sentence about Gaussian SLAM optimization. " * 80
    chunk_items, chunk_groups = split_translation_texts(
        [{"id": "passage-0", "text": long_text}]
    )

    assert len(chunk_items) > 1
    assert chunk_groups["passage-0"] == [item["id"] for item in chunk_items]

    def _fake_translate(texts, _config):
        return [{"id": item["id"], "zh": f"中文：{item['id']}"} for item in texts]

    with patch("citation_agent.translation.call_litellm_translation", _fake_translate):
        result = translate_evidence(
            {"texts": [{"id": "passage-0", "text": long_text}]},
            model_config=config,
        )

    assert result["available"] is True
    assert result["translations"][0]["id"] == "passage-0"
    assert "passage-0__part1" in result["translations"][0]["zh"]
