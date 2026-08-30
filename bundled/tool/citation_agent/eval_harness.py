"""Regression eval harness for EasyCite recommendation quality."""
from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from .config import SourceConfig
from .detection import CitationNeedDetector
from .recommender import CitationRecommender
from .schemas import CitationSuggestion, TraceEvent


SUPPORTIVE_VERDICTS = {
    "foundational_support",
    "direct_support",
    "partial_support",
    "related_only",
    "metadata_match",
}
UNSUPPORTED_VERDICTS = {"insufficient_evidence", "contradiction"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    claim: str
    context: str = ""
    citation_needed: bool = True
    gold_papers: list[str] = field(default_factory=list)
    acceptable_papers: list[str] = field(default_factory=list)
    forbidden_papers: list[str] = field(default_factory=list)
    expected_evidence_keywords: list[str] = field(default_factory=list)
    workspace_path: str = ""
    file_path: str = ""
    existing_citations: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class CandidateEvalSummary:
    title: str
    citation_key: str | None
    verdict: str
    confidence: float
    source: str
    evidence_text: str


@dataclass(frozen=True)
class CaseResult:
    id: str
    citation_needed_expected: bool
    citation_needed_predicted: bool
    top_recommendations: list[CandidateEvalSummary]
    diagnostic_candidates: list[CandidateEvalSummary]
    target_recall_at_5: bool
    diagnostic_recall_at_10: bool
    forbidden_recommended: bool
    unsupported_main_recommendations: int
    evidence_keyword_hit_rate: float
    end_to_end_success: bool
    failure_reasons: list[str]
    latency_ms: float
    trace_run_id: str | None
    trace_stages: list[str]
    target_trace: list[str]


def normalize_text(text: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    )


def matches_any(text: str, targets: Iterable[str]) -> bool:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    for target in targets:
        normalized_target = normalize_text(target)
        if not normalized_target:
            continue
        if normalized_target in normalized_text or normalized_text in normalized_target:
            return True
        target_terms = set(normalized_target.split())
        text_terms = set(normalized_text.split())
        if len(target_terms) >= 2 and len(target_terms & text_terms) / len(target_terms) >= 0.7:
            return True
    return False


def load_cases(path: pathlib.Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            allowed = set(EvalCase.__dataclass_fields__)
            cases.append(EvalCase(**{key: value for key, value in payload.items() if key in allowed}))
    return cases


def load_case_files(paths: Iterable[pathlib.Path]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in paths:
        cases.extend(load_cases(path))
    return cases


def run_eval_cases(
    cases: list[EvalCase],
    *,
    default_workspace_path: str,
    source_config: SourceConfig | None = None,
    trace_targets: bool = True,
) -> dict[str, Any]:
    config = source_config or SourceConfig.from_env()
    results = [
        run_case(
            case,
            default_workspace_path=default_workspace_path,
            source_config=config,
            trace_targets=trace_targets,
        )
        for case in cases
    ]
    return build_report(results)


def run_case(
    case: EvalCase,
    *,
    default_workspace_path: str,
    source_config: SourceConfig,
    trace_targets: bool = True,
) -> CaseResult:
    workspace_path = case.workspace_path or default_workspace_path
    params: dict[str, Any] = {
        "workspace_path": workspace_path,
        "file_path": case.file_path,
        "line_start": 0,
        "line_end": 0,
        "claim": case.claim,
        "context": case.context or case.claim,
        "existing_citations": case.existing_citations,
        "force_refresh": True,
    }
    target_papers = [*case.gold_papers, *case.acceptable_papers]
    run_config = source_config
    if trace_targets and target_papers:
        run_config = replace(source_config, debug_target_papers=target_papers)

    started = time.perf_counter()
    detection = CitationNeedDetector().detect(params)
    suggestions: list[CitationSuggestion] = []
    diagnostics: list[CitationSuggestion] = []
    trace_events: list[TraceEvent] = []
    target_trace: list[str] = []
    if detection.citation_needed:
        bundle = CitationRecommender(run_config).recommend_with_diagnostics(params)
        suggestions = bundle.suggestions
        diagnostics = bundle.diagnostic_suggestions
        trace_events = bundle.trace_events
        target_trace = bundle.target_trace
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    top_recommendations = [candidate_summary(item) for item in suggestions[:5]]
    diagnostic_candidates = [candidate_summary(item) for item in diagnostics[:10]]
    top_titles = [item.title for item in top_recommendations]
    diagnostic_titles = [item.title for item in diagnostic_candidates]
    target_recall_at_5 = bool(target_papers) and any(matches_any(title, target_papers) for title in top_titles)
    diagnostic_recall_at_10 = target_recall_at_5 or (
        bool(target_papers) and any(matches_any(title, target_papers) for title in diagnostic_titles)
    )
    forbidden_recommended = any(matches_any(title, case.forbidden_papers) for title in top_titles)
    unsupported_count = sum(
        1 for suggestion in suggestions if suggestion.verdict in UNSUPPORTED_VERDICTS
    )
    evidence_keyword_hit_rate = compute_evidence_keyword_hit_rate(
        suggestions,
        case.expected_evidence_keywords,
    )
    failure_reasons = infer_failure_reasons(
        case=case,
        citation_needed_predicted=detection.citation_needed,
        target_recall_at_5=target_recall_at_5,
        diagnostic_recall_at_10=diagnostic_recall_at_10,
        forbidden_recommended=forbidden_recommended,
        unsupported_count=unsupported_count,
        trace_events=trace_events,
        target_trace=target_trace,
    )
    end_to_end_success = not failure_reasons
    return CaseResult(
        id=case.id,
        citation_needed_expected=case.citation_needed,
        citation_needed_predicted=detection.citation_needed,
        top_recommendations=top_recommendations,
        diagnostic_candidates=diagnostic_candidates,
        target_recall_at_5=target_recall_at_5,
        diagnostic_recall_at_10=diagnostic_recall_at_10,
        forbidden_recommended=forbidden_recommended,
        unsupported_main_recommendations=unsupported_count,
        evidence_keyword_hit_rate=evidence_keyword_hit_rate,
        end_to_end_success=end_to_end_success,
        failure_reasons=failure_reasons,
        latency_ms=latency_ms,
        trace_run_id=trace_events[0].run_id if trace_events else None,
        trace_stages=[event.stage for event in trace_events],
        target_trace=target_trace,
    )


def candidate_summary(suggestion: CitationSuggestion) -> CandidateEvalSummary:
    evidence_text = suggestion.evidence.text or ""
    if not evidence_text and suggestion.evidence_passages:
        evidence_text = suggestion.evidence_passages[0].text or ""
    return CandidateEvalSummary(
        title=suggestion.title,
        citation_key=suggestion.citation_key,
        verdict=suggestion.verdict,
        confidence=round(suggestion.confidence, 4),
        source=suggestion.source,
        evidence_text=evidence_text[:500],
    )


def compute_evidence_keyword_hit_rate(
    suggestions: list[CitationSuggestion],
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 1.0
    evidence_text = " ".join(
        part
        for suggestion in suggestions
        for part in [
            suggestion.reason,
            suggestion.evidence.text or "",
            " ".join(passage.text or "" for passage in suggestion.evidence_passages),
        ]
        if part
    )
    normalized = normalize_text(evidence_text)
    hits = sum(1 for keyword in expected_keywords if normalize_text(keyword) in normalized)
    return round(hits / len(expected_keywords), 4)


def infer_failure_reasons(
    *,
    case: EvalCase,
    citation_needed_predicted: bool,
    target_recall_at_5: bool,
    diagnostic_recall_at_10: bool,
    forbidden_recommended: bool,
    unsupported_count: int,
    trace_events: list[TraceEvent],
    target_trace: list[str],
) -> list[str]:
    reasons: list[str] = []
    if case.citation_needed != citation_needed_predicted:
        reasons.append("citation_needed_mismatch")
    if case.citation_needed and case.gold_papers and not target_recall_at_5:
        reasons.append("gold_or_acceptable_missing_from_top5")
    if case.citation_needed and case.gold_papers and not diagnostic_recall_at_10:
        reasons.append("gold_or_acceptable_missing_from_diagnostics")
    if forbidden_recommended:
        reasons.append("forbidden_paper_recommended")
    if unsupported_count > 0:
        reasons.append("unsupported_main_recommendation")
    if any(event.error for event in trace_events):
        reasons.append("trace_contains_error")
    if any("timeout" in (event.message + " " + (event.error or "")).lower() for event in trace_events):
        reasons.append("llm_or_tool_timeout")
    if target_trace and any("not found" in line for line in target_trace[-3:]):
        reasons.append("target_missing_late_stage")
    return reasons


def build_report(results: list[CaseResult]) -> dict[str, Any]:
    return {
        "schema_version": "v10.eval_report.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": summarize(results),
        "failed_cases": [
            result.id
            for result in results
            if not result.end_to_end_success
        ],
        "cases": [case_result_to_dict(result) for result in results],
    }


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}
    expected_positive = [result for result in results if result.citation_needed_expected]
    predicted_positive = [result for result in results if result.citation_needed_predicted]
    true_positive = [
        result
        for result in results
        if result.citation_needed_expected and result.citation_needed_predicted
    ]
    citation_correct = sum(
        result.citation_needed_expected == result.citation_needed_predicted
        for result in results
    )
    final_recommendation_count = sum(len(result.top_recommendations) for result in results)
    unsupported = sum(result.unsupported_main_recommendations for result in results)
    tool_or_llm_timeouts = sum(
        1
        for result in results
        if "llm_or_tool_timeout" in result.failure_reasons
    )
    trace_errors = sum(
        1
        for result in results
        if "trace_contains_error" in result.failure_reasons
    )
    return {
        "total": total,
        "citation_needed_accuracy": ratio(citation_correct, total),
        "citation_needed_precision": ratio(len(true_positive), len(predicted_positive)),
        "citation_needed_recall": ratio(len(true_positive), len(expected_positive)),
        "citation_needed_f1": f1(
            ratio(len(true_positive), len(predicted_positive)),
            ratio(len(true_positive), len(expected_positive)),
        ),
        "target_recall_at_5": ratio(
            sum(result.target_recall_at_5 for result in results if result.citation_needed_expected),
            len(expected_positive),
        ),
        "diagnostic_recall_at_10": ratio(
            sum(result.diagnostic_recall_at_10 for result in results if result.citation_needed_expected),
            len(expected_positive),
        ),
        "forbidden_recommendation_rate": ratio(
            sum(result.forbidden_recommended for result in results),
            total,
        ),
        "unsupported_recommendation_rate": ratio(unsupported, final_recommendation_count),
        "end_to_end_success_rate": ratio(
            sum(result.end_to_end_success for result in results),
            total,
        ),
        "average_evidence_keyword_hit_rate": round(
            sum(result.evidence_keyword_hit_rate for result in results) / total,
            4,
        ),
        "average_latency_ms": round(
            sum(result.latency_ms for result in results) / total,
            2,
        ),
        "llm_timeout_rate": ratio(tool_or_llm_timeouts, total),
        "tool_failure_rate": ratio(trace_errors, total),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def case_result_to_dict(result: CaseResult) -> dict[str, Any]:
    return asdict(result)


def write_report(report: dict[str, Any], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "# EasyCite Eval Report",
        "",
        f"- schema: `{report.get('schema_version', '')}`",
        f"- created_at: `{report.get('created_at', '')}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    failed = report.get("failed_cases", [])
    lines.extend(["", "## Failed Cases", ""])
    if failed:
        for case_id in failed:
            lines.append(f"- `{case_id}`")
    else:
        lines.append("No failed cases.")
    lines.extend(["", "## Case Summary", ""])
    for case in report.get("cases", []):
        status = "PASS" if case.get("end_to_end_success") else "FAIL"
        lines.append(f"- `{case.get('id')}`: {status}; reasons={case.get('failure_reasons')}")
    return "\n".join(lines) + "\n"
