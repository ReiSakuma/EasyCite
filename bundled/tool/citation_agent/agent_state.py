"""Agent-state snapshots and run-record persistence for EasyCite v9."""
from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .agent_nodes import graph_from_trace_events
from .schemas import CitationSuggestion, TraceEvent


def build_agent_state_snapshot(
    *,
    params: dict[str, Any],
    run_id: str,
    model_snapshot: dict[str, Any],
    source_snapshot: dict[str, Any],
    claim_profile: Any,
    retrieval_plan: Any,
    candidate_counts: dict[str, int],
    final: list[CitationSuggestion],
    diagnostics: list[CitationSuggestion],
    target_trace: list[str],
    trace_events: list[TraceEvent],
    context_reports: list[dict[str, Any]] | None = None,
    tool_observations: list[dict[str, Any]] | None = None,
    memory_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact LangGraph runtime state snapshot.

    v9 runs the recommender through a real LangGraph StateGraph by default and
    persists the run as explicit graph nodes and edges: request -> profile ->
    plan -> pools -> verification -> output -> trace.
    """

    return {
        "schema_version": "v20.agent_state.1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": {
            "workspace_path": str(params.get("workspace_path") or ""),
            "file_path": str(params.get("file_path") or ""),
            "line_start": params.get("line_start"),
            "line_end": params.get("line_end"),
            "claim": str(params.get("claim") or ""),
            "context": str(params.get("context") or ""),
            "existing_citations": list(params.get("existing_citations") or []),
            "force_refresh": bool(params.get("force_refresh")),
        },
        "runtime": {
            "model": model_snapshot,
            "sources": source_snapshot,
        },
        "claim_profile": {
            "claim_kind": getattr(claim_profile, "claim_kind", ""),
            "retrieval_intent": getattr(claim_profile, "retrieval_intent", ""),
            "claim_focus": getattr(claim_profile, "claim_focus", ""),
            "phrases": list(getattr(claim_profile, "phrases", []) or []),
            "query_phrases": list(getattr(claim_profile, "query_phrases", []) or []),
            "support_terms": sorted(getattr(claim_profile, "support_terms", set()) or []),
            "must_include_terms": sorted(getattr(claim_profile, "must_include_terms", set()) or []),
            "required_concept_terms": sorted(getattr(claim_profile, "required_concept_terms", set()) or []),
            "foundational_concept_terms": sorted(
                getattr(claim_profile, "foundational_concept_terms", set()) or []
            ),
            "excluded_concept_terms": sorted(getattr(claim_profile, "excluded_concept_terms", set()) or []),
        },
        "retrieval_plan": {
            "source": getattr(retrieval_plan, "source", ""),
            "queries": list(getattr(retrieval_plan, "queries", []) or []),
            "concepts": list(getattr(retrieval_plan, "concepts", []) or []),
            "required_concepts": list(getattr(retrieval_plan, "required_concepts", []) or []),
            "foundational_concepts": list(getattr(retrieval_plan, "foundational_concepts", []) or []),
            "excluded_concepts": list(getattr(retrieval_plan, "excluded_concepts", []) or []),
            "claim_focus": getattr(retrieval_plan, "claim_focus", ""),
            "reason": getattr(retrieval_plan, "reason", ""),
        },
        "candidate_pool": {
            "counts": candidate_counts,
        },
        "verification": {
            "final_recommendations": [candidate_summary(item) for item in final],
            "diagnostic_candidates": [candidate_summary(item) for item in diagnostics],
            "target_trace": target_trace,
        },
        "output": {
            "recommendation_count": len(final),
            "diagnostic_count": len(diagnostics),
            "citation_keys": [
                item.citation_key
                for item in final
                if item.citation_key
            ],
        },
        "trace": {
            "events": [event.model_dump() for event in trace_events],
        },
        "context_engineering": {
            "schema_version": "v16.context.1",
            "reports": list(context_reports or []),
            "total_calls": len(context_reports or []),
            "estimated_tokens_before": sum(
                int(report.get("original_tokens", 0) or 0)
                for report in (context_reports or [])
            ),
            "estimated_tokens_after": sum(
                int(report.get("final_tokens", 0) or 0)
                for report in (context_reports or [])
            ),
        },
        "tool_reliability": {
            "schema_version": "v17.tools.1",
            "observations": list(tool_observations or []),
            "total_calls": len(tool_observations or []),
            "successful_calls": sum(
                1 for item in (tool_observations or []) if item.get("success")
            ),
            "failed_calls": sum(
                1 for item in (tool_observations or []) if not item.get("success")
            ),
            "fallback_calls": sum(
                1 for item in (tool_observations or []) if item.get("fallback")
            ),
            "error_categories": sorted(
                {
                    str(item.get("error_category"))
                    for item in (tool_observations or [])
                    if item.get("error_category")
                }
            ),
        },
        "memory": {
            "schema_version": "v18.memory.1",
            **(memory_snapshot or {"total": 0, "by_scope": {}, "recalled": 0}),
        },
        "graph": graph_from_trace_events(trace_events),
        "checkpoint": {
            "schema_version": "v20.checkpoint.1",
            "run_id": run_id,
            "status": "completed",
            "resume_supported": True,
            "secrets_persisted": False,
        },
    }


def save_agent_run_record(
    params: dict[str, Any],
    state: dict[str, Any],
    logger: Callable[[str], None] | None = None,
) -> str | None:
    """Persist the latest agent-state snapshot under .citation-agent/runs."""

    workspace_path = str(params.get("workspace_path") or "").strip()
    file_path = str(params.get("file_path") or "").strip()
    if workspace_path:
        root = pathlib.Path(workspace_path)
    elif file_path:
        root = pathlib.Path(file_path).parent
    else:
        return None

    try:
        run_dir = root / ".citation-agent" / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(state.get("run_id") or "unknown")
        record_path = run_dir / f"{safe_file_name(run_id)}.json"
        record_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        latest_path = run_dir / "latest.json"
        latest_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "path": str(record_path),
                    "created_at": state.get("created_at"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if logger:
            logger(f"EasyCite run record saved: {record_path}")
        return str(record_path)
    except OSError as exc:
        if logger:
            logger(f"EasyCite run record save failed: {type(exc).__name__}: {exc}")
        return None


def candidate_summary(suggestion: CitationSuggestion) -> dict[str, Any]:
    evidence = suggestion.evidence
    return {
        "paper_id": suggestion.paper_id,
        "title": suggestion.title,
        "citation_key": suggestion.citation_key,
        "source": suggestion.source,
        "year": suggestion.year,
        "verdict": suggestion.verdict,
        "confidence": suggestion.confidence,
        "reason": suggestion.reason,
        "evidence": {
            "source": evidence.source,
            "page": evidence.page,
            "section": evidence.section,
            "score": evidence.score,
            "text": evidence.text,
        },
        "trace": suggestion.trace,
    }


def safe_file_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:120]
