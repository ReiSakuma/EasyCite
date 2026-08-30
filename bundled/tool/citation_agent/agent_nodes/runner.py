"""Graph metadata for EasyCite v9 LangGraph orchestration.

The default runtime now executes a real LangGraph StateGraph. This module
renders trace events as graph nodes, edges, and checkpoint metadata for saved
run records and developer inspection.
"""
from __future__ import annotations

from typing import Any

from citation_agent.schemas import TraceEvent

from .state import AgentNodeRecord


STAGE_TO_NODE: dict[str, tuple[str, str]] = {
    "start": ("start", "control"),
    "llm_config": ("configure_runtime", "runtime"),
    "graph_runtime": ("compile_stategraph", "runtime"),
    "claim_analysis": ("analyze_claim", "reasoning"),
    "query_planning": ("plan_retrieval", "reasoning"),
    "current_bib_search": ("recall_current_bib", "retrieval"),
    "zotero_retrieval": ("recall_zotero", "retrieval"),
    "zotero_forced_target_probe": ("debug_target_probe", "diagnostics"),
    "title_doi_dedupe": ("dedupe_candidates", "filter"),
    "scope_filter": ("scope_filter", "filter"),
    "metadata_rank": ("metadata_rank", "rank"),
    "metadata_diagnostics": ("metadata_diagnostics", "diagnostics"),
    "evidence_shortlist": ("select_evidence_targets", "rank"),
    "evidence_verification": ("verify_evidence", "reasoning"),
    "evidence_rank": ("rank_verified_evidence", "rank"),
    "final_recommendations": ("calibrate_recommendations", "output"),
    "stop": ("prepare_review_payload", "control"),
}

GRAPH_EDGES: list[dict[str, str]] = [
    {"source": "start", "target": "configure_runtime"},
    {"source": "configure_runtime", "target": "compile_stategraph"},
    {"source": "compile_stategraph", "target": "analyze_claim"},
    {"source": "analyze_claim", "target": "plan_retrieval"},
    {"source": "plan_retrieval", "target": "recall_current_bib"},
    {"source": "plan_retrieval", "target": "recall_zotero"},
    {"source": "recall_zotero", "target": "debug_target_probe"},
    {"source": "recall_current_bib", "target": "dedupe_candidates"},
    {"source": "recall_zotero", "target": "dedupe_candidates"},
    {"source": "debug_target_probe", "target": "dedupe_candidates"},
    {"source": "dedupe_candidates", "target": "scope_filter"},
    {"source": "scope_filter", "target": "metadata_rank"},
    {"source": "metadata_rank", "target": "select_evidence_targets"},
    {"source": "metadata_rank", "target": "metadata_diagnostics"},
    {"source": "select_evidence_targets", "target": "verify_evidence"},
    {"source": "verify_evidence", "target": "rank_verified_evidence"},
    {"source": "rank_verified_evidence", "target": "calibrate_recommendations"},
    {"source": "metadata_diagnostics", "target": "calibrate_recommendations"},
    {"source": "calibrate_recommendations", "target": "prepare_review_payload"},
]

CHECKPOINT_NODES = {
    "analyze_claim",
    "plan_retrieval",
    "recall_zotero",
    "dedupe_candidates",
    "select_evidence_targets",
    "verify_evidence",
    "prepare_review_payload",
}


def graph_from_trace_events(events: list[TraceEvent]) -> dict[str, Any]:
    """Render trace events as a graph snapshot."""

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        name, kind = STAGE_TO_NODE.get(event.stage, (event.stage, "custom"))
        node = AgentNodeRecord(
            name=name,
            stage=event.stage,
            kind=kind,
            message=event.message,
            candidate_count_before=event.candidate_count_before,
            candidate_count_after=event.candidate_count_after,
            latency_ms=event.latency_ms,
            checkpoint=name in CHECKPOINT_NODES,
            error=event.error,
        ).to_dict()
        if name in seen:
            node["repeat"] = True
        seen.add(name)
        nodes.append(node)

    return {
        "schema_version": "v9.graph.1",
        "framework": "langgraph",
        "runtime": "langgraph-stategraph",
        "nodes": nodes,
        "edges": GRAPH_EDGES,
        "checkpoint_policy": {
            "checkpoint_after": sorted(CHECKPOINT_NODES),
            "storage": ".citation-agent/runs/{run_id}.json",
            "replay_status": "trace_replay_ready",
        },
    }
