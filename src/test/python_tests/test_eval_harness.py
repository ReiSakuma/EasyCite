from __future__ import annotations

import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL_DIR = os.fsdecode(_PROJECT_ROOT / "bundled" / "tool")
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from citation_agent.config import SourceConfig
from citation_agent.eval_harness import (
    EvalCase,
    build_report,
    run_case,
    summarize,
)
from citation_agent.recommender import RecommendationBundle
from citation_agent.schemas import CitationDetectionResult, CitationSuggestion, EvidenceSnippet, TraceEvent


def _suggestion(
    *,
    title: str,
    citation_key: str,
    verdict: str = "direct_support",
    evidence_text: str = "neural radiance field differentiable rendering camera pose",
) -> CitationSuggestion:
    return CitationSuggestion(
        paper_id=citation_key,
        title=title,
        citation_key=citation_key,
        authors=["Ada Smith"],
        year="2024",
        verdict=verdict,
        confidence=0.9,
        source="zotero",
        evidence=EvidenceSnippet(source="abstract", text=evidence_text, score=1.0),
        reason="mock",
    )


def test_run_case_computes_recall_and_injects_target_trace(monkeypatch):
    class FakeDetector:
        def detect(self, _params):
            return CitationDetectionResult(
                citation_needed=True,
                reason="mock",
                confidence=0.9,
                claim_type="related_work_summary",
                source="heuristic",
            )

    captured_targets = []

    class FakeRecommender:
        def __init__(self, source_config):
            captured_targets.extend(source_config.debug_target_papers)

        def recommend_with_diagnostics(self, _params):
            return RecommendationBundle(
                suggestions=[
                    _suggestion(
                        title="NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
                        citation_key="mildenhallNeRF2020",
                    )
                ],
                diagnostic_suggestions=[],
                target_trace=["final_recommendations: FOUND target='NeRF'."],
                trace_events=[
                    TraceEvent(
                        run_id="run-1",
                        stage="final_recommendations",
                        message="mock",
                    )
                ],
            )

    monkeypatch.setattr("citation_agent.eval_harness.CitationNeedDetector", lambda: FakeDetector())
    monkeypatch.setattr("citation_agent.eval_harness.CitationRecommender", FakeRecommender)

    result = run_case(
        EvalCase(
            id="nerf_001",
            claim="Recent neural representations changed mapping.",
            gold_papers=["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"],
            expected_evidence_keywords=["neural radiance field", "differentiable rendering"],
        ),
        default_workspace_path="/tmp/project",
        source_config=SourceConfig(
            max_suggestions=5,
            zotero_enabled=False,
            zotero_endpoint=None,
        ),
    )

    assert result.target_recall_at_5 is True
    assert result.diagnostic_recall_at_10 is True
    assert result.end_to_end_success is True
    assert result.evidence_keyword_hit_rate == 1.0
    assert captured_targets == [
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    ]


def test_summarize_reports_forbidden_and_unsupported_failures():
    result = run_case_result_fixture(
        forbidden_recommended=True,
        unsupported_main_recommendations=1,
        end_to_end_success=False,
    )

    metrics = summarize([result])

    assert metrics["forbidden_recommendation_rate"] == 1.0
    assert metrics["unsupported_recommendation_rate"] == 1.0
    assert metrics["end_to_end_success_rate"] == 0.0


def test_build_report_includes_failed_case_ids():
    passed = run_case_result_fixture(id="pass", end_to_end_success=True)
    failed = run_case_result_fixture(id="fail", end_to_end_success=False)

    report = build_report([passed, failed])

    assert report["schema_version"] == "v10.eval_report.1"
    assert report["failed_cases"] == ["fail"]
    assert report["metrics"]["total"] == 2


def run_case_result_fixture(**overrides):
    from citation_agent.eval_harness import CandidateEvalSummary, CaseResult

    data = {
        "id": "case",
        "citation_needed_expected": True,
        "citation_needed_predicted": True,
        "top_recommendations": [
            CandidateEvalSummary(
                title="Forbidden Paper",
                citation_key="forbidden",
                verdict="insufficient_evidence",
                confidence=0.2,
                source="zotero",
                evidence_text="",
            )
        ],
        "diagnostic_candidates": [],
        "target_recall_at_5": False,
        "diagnostic_recall_at_10": False,
        "forbidden_recommended": False,
        "unsupported_main_recommendations": 0,
        "evidence_keyword_hit_rate": 0.0,
        "end_to_end_success": False,
        "failure_reasons": ["gold_or_acceptable_missing_from_top5"],
        "latency_ms": 10.0,
        "trace_run_id": "run",
        "trace_stages": ["start", "stop"],
        "target_trace": [],
    }
    data.update(overrides)
    return CaseResult(**data)
