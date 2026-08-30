from __future__ import annotations

import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL_DIR = os.fsdecode(_PROJECT_ROOT / "bundled" / "tool")
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from citation_agent.memory import CandidateFeedbackMemory  # noqa: E402
from citation_agent.schemas import CitationSuggestion, EvidenceSnippet  # noqa: E402


def _suggestion(title: str, key: str, confidence: float = 0.7) -> CitationSuggestion:
    return CitationSuggestion(
        paper_id=key,
        title=title,
        citation_key=key,
        authors=["Ada Smith"],
        year="2025",
        verdict="direct_support",
        confidence=confidence,
        source="zotero",
        evidence=EvidenceSnippet(source="abstract", text="mock", score=1.0),
        reason="mock",
    )


def test_feedback_memory_boosts_accepted_candidate_for_similar_claim(tmp_path):
    memory = CandidateFeedbackMemory.for_workspace(
        str(tmp_path),
        str(tmp_path / "memory.sqlite3"),
    )
    accepted = _suggestion("Neural Mapping for Dense SLAM", "neural2025", confidence=0.6)
    baseline = _suggestion("Geometric Mapping for Dense SLAM", "geom2025", confidence=0.62)

    memory.record_feedback(
        workspace_path=str(tmp_path),
        claim="Neural differentiable mapping jointly optimizes geometry appearance and pose.",
        context="Dense SLAM paper.",
        suggestion=accepted,
        action="accepted",
    )

    result = memory.rank_with_feedback(
        suggestions=[baseline, accepted],
        claim="Recent neural differentiable mapping optimizes geometry appearance and camera pose.",
        context="Dense SLAM introduction.",
        positive_boost=0.08,
        negative_penalty=0.18,
        min_similarity=0.2,
    )

    assert result.suggestions[0].citation_key == "neural2025"
    assert result.adjustments
    assert "Memory feedback" in result.suggestions[0].reason


def test_feedback_memory_demotes_rejected_candidate_and_can_clear(tmp_path):
    memory = CandidateFeedbackMemory.for_workspace(
        str(tmp_path),
        str(tmp_path / "memory.sqlite3"),
    )
    rejected = _suggestion("Weakly Related Gaussian SLAM", "weak2025", confidence=0.8)
    strong = _suggestion("Direct Neural Mapping Paper", "direct2025", confidence=0.75)

    memory.record_feedback(
        workspace_path=str(tmp_path),
        claim="Neural differentiable mapping optimizes geometry appearance and pose.",
        context="Dense SLAM paper.",
        suggestion=rejected,
        action="rejected",
    )

    result = memory.rank_with_feedback(
        suggestions=[rejected, strong],
        claim="Neural differentiable mapping optimizes geometry appearance and pose.",
        context="Dense SLAM paper.",
        positive_boost=0.08,
        negative_penalty=0.18,
        min_similarity=0.2,
    )

    assert result.suggestions[0].citation_key == "direct2025"
    cleared = memory.clear()
    assert cleared["cleared"] is True
    assert cleared["count"] == 1
    assert memory.load_recent() == []
