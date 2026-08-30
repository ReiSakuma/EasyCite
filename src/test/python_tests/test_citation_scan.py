"""Tests for background citation-needed scan semantic ranking."""
from __future__ import annotations

import pathlib
import sys


def _setup_path():
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.citation_scan import rank_citation_scan_sentences  # noqa: E402


def test_rank_citation_scan_reports_disabled_when_embedding_disabled(monkeypatch):
    monkeypatch.delenv("CITATION_AGENT_EMBEDDING_ENABLED", raising=False)

    result = rank_citation_scan_sentences(
        {
            "sentences": [
                {
                    "id": "s1",
                    "text": "Existing methods generally rely on offline optimization.",
                }
            ]
        }
    )

    assert result["available"] is False
    assert result["scores"] == []
    assert "disabled" in result["reason"]


def test_rank_citation_scan_scores_sentences_with_hash_embedding(monkeypatch):
    monkeypatch.setenv("CITATION_AGENT_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("CITATION_AGENT_EMBEDDING_MODEL_PROVIDER", "hash")

    result = rank_citation_scan_sentences(
        {
            "sentences": [
                {
                    "id": "s1",
                    "text": "Existing methods generally rely on offline optimization and need citations.",
                },
                {
                    "id": "s2",
                    "text": "We implement the parser using a small state machine.",
                },
            ]
        }
    )

    assert result["available"] is True
    assert result["provider"] == "hash"
    assert [score["id"] for score in result["scores"]] == ["s1", "s2"]
    assert all(isinstance(score["score"], float) for score in result["scores"])
    assert all(score["matched_prototype"] for score in result["scores"])
