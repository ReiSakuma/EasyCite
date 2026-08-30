"""Deterministic replay of frozen recommendation outcomes."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from citation_agent.eval_harness import (
    CandidateEvalSummary,
    CaseResult,
    build_report,
    matches_any,
)


def run_replay_file(path: pathlib.Path) -> dict[str, Any]:
    """Evaluate frozen case outcomes without Zotero, PDF, embedding or LLM."""
    results: list[CaseResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            payload = json.loads(line)
            results.append(_result_from_fixture(payload, path, number))
    report = build_report(results)
    report["mode"] = "replay"
    report["fixture"] = str(path)
    report["replay_contract"] = {
        "schema_version": "v19.eval_fixture.1",
        "external_calls": 0,
        "deterministic": True,
    }
    return report


def _result_from_fixture(payload: dict[str, Any], path: pathlib.Path, number: int) -> CaseResult:
    case_id = str(payload.get("id") or f"fixture-{number}")
    target_papers = [*payload.get("gold_papers", []), *payload.get("acceptable_papers", [])]
    top = [_summary(item) for item in payload.get("top_recommendations", [])]
    diagnostic = [_summary(item) for item in payload.get("diagnostic_candidates", [])]
    top_titles = [item.title for item in top]
    all_titles = [*top_titles, *(item.title for item in diagnostic)]
    expected = bool(payload.get("citation_needed", True))
    predicted = bool(payload.get("citation_needed_predicted", expected))
    forbidden = any(matches_any(item.title, payload.get("forbidden_papers", [])) for item in top)
    unsupported = sum(item.verdict in {"insufficient_evidence", "contradiction"} for item in top)
    keyword_text = " ".join(item.evidence_text for item in [*top, *diagnostic])
    keywords = payload.get("expected_evidence_keywords", [])
    hits = sum(str(keyword).lower() in keyword_text.lower() for keyword in keywords)
    keyword_rate = round(hits / len(keywords), 4) if keywords else 1.0
    target_top = bool(target_papers) and any(matches_any(title, target_papers) for title in top_titles)
    target_diag = target_top or (bool(target_papers) and any(matches_any(title, target_papers) for title in all_titles))
    failures: list[str] = []
    if expected != predicted:
        failures.append("citation_needed_mismatch")
    if expected and target_papers and not target_top:
        failures.append("gold_or_acceptable_missing_from_top5")
    if expected and target_papers and not target_diag:
        failures.append("gold_or_acceptable_missing_from_diagnostics")
    if forbidden:
        failures.append("forbidden_paper_recommended")
    if unsupported:
        failures.append("unsupported_main_recommendation")
    return CaseResult(
        id=case_id,
        citation_needed_expected=expected,
        citation_needed_predicted=predicted,
        top_recommendations=top,
        diagnostic_candidates=diagnostic[:10],
        target_recall_at_5=target_top,
        diagnostic_recall_at_10=target_diag,
        forbidden_recommended=forbidden,
        unsupported_main_recommendations=unsupported,
        evidence_keyword_hit_rate=keyword_rate,
        end_to_end_success=not failures,
        failure_reasons=failures,
        latency_ms=float(payload.get("latency_ms", 0.0)),
        trace_run_id=payload.get("trace_run_id"),
        trace_stages=list(payload.get("trace_stages", ["replay"])),
        target_trace=list(payload.get("target_trace", [])),
    )


def _summary(payload: dict[str, Any]) -> CandidateEvalSummary:
    return CandidateEvalSummary(
        title=str(payload.get("title", "")),
        citation_key=payload.get("citation_key"),
        verdict=str(payload.get("verdict", "metadata_match")),
        confidence=float(payload.get("confidence", 0.0)),
        source=str(payload.get("source", "fixture")),
        evidence_text=str(payload.get("evidence_text", ""))[:500],
    )
