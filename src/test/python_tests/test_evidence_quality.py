"""Tests for V13 evidence passage quality scoring."""
from __future__ import annotations

import pathlib
import sys


def _setup_path():
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.claim_analysis import build_claim_profile  # noqa: E402
from citation_agent.config import EmbeddingConfig  # noqa: E402
from citation_agent.evidence import EvidenceRetriever, analyze_evidence_quality  # noqa: E402
from citation_agent.schemas import CitationSuggestion, EvidenceSnippet  # noqa: E402


def _suggestion(title: str = "GRAND-SLAM: Local Optimization for Multi-Agent Gaussian SLAM") -> CitationSuggestion:
    return CitationSuggestion(
        paper_id="zotero:test",
        title=title,
        citation_key="test2026",
        authors=["Ada Smith"],
        year="2026",
        source="zotero",
        confidence=0.68,
        evidence=EvidenceSnippet(source="abstract", text=""),
        reason="metadata match",
    )


def test_reference_list_passage_is_discarded():
    text = (
        "Grand-slam: Local optimization for globally consistent large-scale multi-agent gaussian slam. "
        "arXiv preprint arXiv:2506.18885. Tian, Y., Chang, Y., Arias, F.H., Nieto-Granda, C., "
        "How, J.P., and Carlone, L. (2022). Kimera-multi: Robust distributed dense metric-semantic SLAM."
    )

    quality = analyze_evidence_quality(text, source="pdf", suggestion=_suggestion())

    assert quality.discard
    assert "reference_list" in quality.flags
    assert quality.quality_score <= 0.12


def test_diagram_or_caption_stream_is_discarded():
    text = (
        "AGENT SIDE Agent i j k Tracking & Local Mapping Loop Closure Detection 3D Gaussian Map "
        "Representation Pose Optimization & Map Refinement Data Stream SERVER SIDE Inter-Agent "
        "Loop Closure Semantic Extraction Sub-map Merging & Global Map Refinement Fig. 3."
    )

    quality = analyze_evidence_quality(text, source="pdf", suggestion=_suggestion())

    assert quality.discard
    assert {"figure_or_table_caption", "ocr_or_diagram_stream"} & set(quality.flags)


def test_related_work_summary_about_external_paper_is_discarded():
    text = (
        "MAGIC-SLAM [38] extends 3DGS SLAM to support loop closure by leveraging a pretrained "
        "vision-language model, while CCM-SLAM [39] and CVI-SLAM [40] use a central server for "
        "map fusion and global optimization."
    )

    quality = analyze_evidence_quality(text, source="pdf", suggestion=_suggestion())

    assert quality.discard
    assert quality.is_related_work_summary is True
    assert quality.is_own_claim is False


def test_own_contribution_passage_is_kept_and_rewarded():
    claim = "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization."
    profile = build_claim_profile(claim)
    text = (
        "In this paper, we present GRAND-SLAM, a multi-agent Gaussian SLAM system that performs "
        "local optimization for globally consistent maps and avoids expensive offline global "
        "optimization while preserving map consistency."
    )

    quality = analyze_evidence_quality(text, source="abstract", suggestion=_suggestion(), profile=profile)

    assert not quality.discard
    assert quality.is_own_claim is True
    assert quality.supports_core_mechanism is True
    assert quality.quality_score >= 0.9


def test_evidence_retriever_emits_quality_metadata():
    claim = "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization."
    profile = build_claim_profile(claim)
    suggestion = _suggestion()
    suggestion.evidence = EvidenceSnippet(
        source="abstract",
        section="Abstract",
        text=(
            "In this paper, we present a multi-agent Gaussian SLAM method with local optimization "
            "that avoids offline global optimization for shared map consistency."
        ),
    )
    retriever = EvidenceRetriever(
        max_passages=2,
        embedding_config=EmbeddingConfig(enabled=False),
    )

    snippets = retriever.retrieve(suggestion, profile, claim=claim)

    assert snippets
    assert snippets[0].quality_score is not None
    assert snippets[0].quality_score > 0.7
    assert snippets[0].is_own_claim is True
    assert "own_claim" in snippets[0].quality_flags
