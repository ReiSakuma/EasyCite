"""True LangGraph StateGraph orchestration for EasyCite v9."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from citation_agent.agent_state import build_agent_state_snapshot, save_agent_run_record
from citation_agent.bibliography import BibRepository
from citation_agent.claim_analysis import ClaimProfile, build_claim_profile
from citation_agent.config import ModelConfig, SourceConfig
from citation_agent.query_planner import RetrievalQueryPlan
from citation_agent.recommender import (
    RecommendationBundle,
    ScopeFilteredCandidate,
    append_target_diagnostics,
    apply_feedback_memory_ranking,
    build_evidence_shortlist,
    dedupe_suggestions,
    diagnostic_reason_for_verified_suggestion,
    is_recommendable_suggestion,
    mark_diagnostic_suggestion,
    metadata_score,
    partition_excluded_scope_candidates,
    planned_retrieval_terms,
    record_target_rank_stage,
    record_target_stage,
    rerank_by_claim_profile,
    sort_verified_suggestions,
    suggestion_identity,
)
from citation_agent.schemas import CitationSuggestion
from citation_agent.trace import TraceRecorder, summarize_candidates
from citation_agent.memory_lifecycle.service import MemoryService
from citation_agent.cache import stable_text_hash
from citation_agent.checkpoint.run_store import RunStore
from citation_agent.tools.adapters import execute_zotero_search
from citation_agent.tools.observability import get_tool_observations, reset_tool_observations

import citation_agent.recommender as recommender_module


class RecommendationGraphState(TypedDict, total=False):
    params: dict[str, Any]
    source_config: SourceConfig
    logger: Callable[[str], None] | None
    claim: str
    context: str
    force_refresh: bool
    trace: TraceRecorder
    target_trace: list[str]
    debug_targets: list[str]
    model_config: ModelConfig
    model_snapshot: dict[str, Any]
    source_snapshot: dict[str, Any]
    max_suggestions: int
    metadata_limit: int
    abstract_limit: int
    profile: ClaimProfile
    retrieval_plan: RetrievalQueryPlan
    bib_suggestions: list[CitationSuggestion]
    zotero_suggestions: list[CitationSuggestion]
    forced_target_probe_suggestions: list[CitationSuggestion]
    suggestions: list[CitationSuggestion]
    deduped: list[CitationSuggestion]
    scoped: list[CitationSuggestion]
    scope_filtered: list[ScopeFilteredCandidate]
    ranked: list[CitationSuggestion]
    metadata_diagnostics: list[CitationSuggestion]
    shortlist: list[CitationSuggestion]
    verified: list[CitationSuggestion]
    sorted_verified: list[CitationSuggestion]
    final: list[CitationSuggestion]
    diagnostic_suggestions: list[CitationSuggestion]
    candidate_counts: dict[str, int]
    agent_state: dict[str, Any]
    agent_run_path: str | None
    bundle: RecommendationBundle
    context_reports: list[dict[str, Any]]
    tool_observations: list[dict[str, Any]]
    checkpoint_store: RunStore
    checkpoint_completed_nodes: list[str]
    checkpoint_run_id: str
    resume_record: Any


def run_langgraph_recommendation(
    *,
    params: dict[str, Any],
    source_config: SourceConfig,
    logger: Callable[[str], None] | None = None,
) -> RecommendationBundle:
    """Run citation recommendation through a LangGraph StateGraph."""

    graph = _build_graph()
    workspace_path = str(params.get("workspace_path") or "")
    request_hash = stable_text_hash(json.dumps(params, ensure_ascii=False, sort_keys=True, default=str))
    checkpoint_store = RunStore.for_workspace(
        workspace_path,
        str(params.get("checkpoint_path") or "") or None,
    )
    resume_record = None
    resume_run_id = str(params.get("resume_run_id") or "").strip()
    if resume_run_id:
        resume_record = checkpoint_store.load(resume_run_id)
        if resume_record is None:
            raise RuntimeError(f"Checkpoint {resume_run_id!r} was not found or has expired")
        if resume_record.graph_version != checkpoint_store.graph_version:
            raise RuntimeError("Checkpoint graph version is incompatible with this runtime")
    state: RecommendationGraphState = {
        "params": params,
        "source_config": source_config,
        "logger": logger,
        "checkpoint_store": checkpoint_store,
        "checkpoint_completed_nodes": [],
        "checkpoint_run_id": str(params.get("run_id") or request_hash[:24]),
        "resume_record": resume_record,
    }
    result = graph.invoke(state)
    bundle = result.get("bundle")
    if not isinstance(bundle, RecommendationBundle):
        raise RuntimeError("LangGraph workflow completed without a RecommendationBundle")
    return bundle


def _build_graph():
    workflow = StateGraph(RecommendationGraphState)
    nodes = {
        "initialize": initialize_node,
        "validate_request": validate_request_node,
        "analyze_claim": analyze_claim_node,
        "plan_retrieval": plan_retrieval_node,
        "recall_current_bib": recall_current_bib_node,
        "recall_zotero": recall_zotero_node,
        "dedupe_candidates": dedupe_candidates_node,
        "scope_filter": scope_filter_node,
        "metadata_rank": metadata_rank_node,
        "select_evidence_targets": select_evidence_targets_node,
        "verify_evidence": verify_evidence_node,
        "rank_verified_evidence": rank_verified_evidence_node,
        "finalize_recommendations": finalize_recommendations_node,
        "persist_run": persist_run_node,
    }
    for name, handler in nodes.items():
        workflow.add_node(name, _checkpointed(name, handler))

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "validate_request")
    workflow.add_conditional_edges(
        "validate_request",
        should_continue_after_validation,
        {"continue": "analyze_claim", "stop": "persist_run"},
    )
    workflow.add_edge("analyze_claim", "plan_retrieval")
    workflow.add_edge("plan_retrieval", "recall_current_bib")
    workflow.add_edge("recall_current_bib", "recall_zotero")
    workflow.add_edge("recall_zotero", "dedupe_candidates")
    workflow.add_edge("dedupe_candidates", "scope_filter")
    workflow.add_edge("scope_filter", "metadata_rank")
    workflow.add_edge("metadata_rank", "select_evidence_targets")
    workflow.add_edge("select_evidence_targets", "verify_evidence")
    workflow.add_edge("verify_evidence", "rank_verified_evidence")
    workflow.add_edge("rank_verified_evidence", "finalize_recommendations")
    workflow.add_edge("finalize_recommendations", "persist_run")
    workflow.add_edge("persist_run", END)
    return workflow.compile()


def _checkpointed(name: str, handler: Callable[[RecommendationGraphState], RecommendationGraphState]):
    """Persist a compact checkpoint after every graph node."""
    def wrapped(state: RecommendationGraphState) -> RecommendationGraphState:
        result = handler(state)
        store = result.get("checkpoint_store")
        trace = result.get("trace")
        if store is None or trace is None:
            return result
        completed = [*result.get("checkpoint_completed_nodes", []), name]
        snapshot = {
            "run_id": trace.run_id,
            "claim": result.get("claim", ""),
            "context": result.get("context", ""),
            "candidate_counts": result.get("candidate_counts", {}),
            "current_node": name,
            "completed_nodes": completed,
        }
        store.save(
            run_id=trace.run_id,
            request_hash=stable_text_hash(json.dumps(result.get("params", {}), sort_keys=True, default=str)),
            current_node=name,
            completed_nodes=completed,
            state_snapshot=snapshot,
            status="completed" if name == "persist_run" else "running",
        )
        return {**result, "checkpoint_completed_nodes": completed, "checkpoint_run_id": trace.run_id}
    return wrapped


def initialize_node(state: RecommendationGraphState) -> RecommendationGraphState:
    params = state["params"]
    if not params.get("_preserve_tool_observations"):
        reset_tool_observations()
    source_config = state["source_config"]
    claim = str(params.get("claim", "")).strip()
    context = str(params.get("context") or "")
    force_refresh = bool(params.get("force_refresh")) and state.get("resume_record") is None
    trace = TraceRecorder()
    trace.record(
        "start",
        f"Citation recommendation run started. runtime=langgraph, force_refresh={force_refresh}, "
        f"resumed={bool(state.get('resume_record'))}.",
        before=0,
        after=0,
    )
    model_config = ModelConfig.from_env()
    trace.record(
        "llm_config",
        (
            "LLM config snapshot: "
            f"use_llm={model_config.use_llm}, "
            f"api_key={'set' if model_config.api_key else 'missing'}, "
            f"provider={model_config.provider}, model={model_config.model}, "
            f"api_base={'set' if model_config.api_base else 'empty'}, "
            f"timeout={model_config.timeout_seconds}s, "
            f"evidence_timeout={model_config.evidence_timeout_seconds}s, "
            f"evidence_max_llm_candidates={model_config.evidence_max_llm_candidates}, "
            f"context_budgets=query:{model_config.context_query_budget_tokens}/"
            f"evidence:{model_config.context_evidence_budget_tokens}/"
            f"translation:{model_config.context_translation_budget_tokens}."
        ),
        before=0,
        after=0,
    )
    if state.get("resume_record"):
        trace.record(
            "checkpoint_resume",
            "Resuming from checkpoint: "
            f"node={state['resume_record'].current_node}, "
            f"completed={list(state['resume_record'].completed_nodes)}.",
        )
    trace.record(
        "graph_runtime",
        "LangGraph StateGraph runtime is active: initialize -> validate -> analyze -> retrieve -> verify -> finalize.",
        before=0,
        after=0,
    )
    model_snapshot = {
        "use_llm": model_config.use_llm,
        "api_key": "set" if model_config.api_key else "missing",
        "provider": model_config.provider,
        "model": model_config.model,
        "api_base": "set" if model_config.api_base else "empty",
        "timeout_seconds": model_config.timeout_seconds,
        "evidence_timeout_seconds": model_config.evidence_timeout_seconds,
        "evidence_max_llm_candidates": model_config.evidence_max_llm_candidates,
        "context_query_budget_tokens": model_config.context_query_budget_tokens,
        "context_evidence_budget_tokens": model_config.context_evidence_budget_tokens,
        "context_translation_budget_tokens": model_config.context_translation_budget_tokens,
    }
    source_snapshot = {
        "max_suggestions": source_config.max_suggestions,
        "zotero_enabled": source_config.zotero_enabled,
        "zotero_endpoint": source_config.zotero_endpoint or "",
        "zotero_collection_keys": source_config.zotero_collection_keys,
        "max_metadata_candidates": source_config.max_metadata_candidates,
        "max_abstract_candidates": source_config.max_abstract_candidates,
        "max_evidence_passages": source_config.max_evidence_passages,
        "enable_evidence_verification": source_config.enable_evidence_verification,
        "agent_runtime": source_config.agent_runtime,
        "memory_enabled": source_config.memory_enabled,
        "force_refresh": force_refresh,
    }
    target_trace: list[str] = []
    if source_config.debug_target_papers:
        target_trace.append(
            "目标文献追踪已启用：" + "; ".join(source_config.debug_target_papers)
        )
    return {
        **state,
        "claim": claim,
        "context": context,
        "force_refresh": force_refresh,
        "trace": trace,
        "model_config": model_config,
        "model_snapshot": model_snapshot,
        "source_snapshot": source_snapshot,
        "target_trace": target_trace,
        "debug_targets": source_config.debug_target_papers,
        "max_suggestions": max(source_config.max_suggestions, 0),
        "bib_suggestions": [],
        "zotero_suggestions": [],
        "forced_target_probe_suggestions": [],
        "suggestions": [],
        "diagnostic_suggestions": [],
        "context_reports": [],
        "tool_observations": [],
    }


def validate_request_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    logger = state.get("logger")
    claim = state["claim"]
    max_suggestions = state["max_suggestions"]
    if not claim:
        _log(logger, "Citation recommender skipped: empty claim.")
        trace.record("stop", "Skipped because claim is empty.", before=0, after=0)
        return {**state, "final": [], "diagnostic_suggestions": []}
    if max_suggestions == 0:
        _log(logger, "Citation recommender skipped: max_suggestions=0.")
        trace.record("stop", "Skipped because max_suggestions=0.", before=0, after=0)
        return {**state, "final": [], "diagnostic_suggestions": []}

    source_config = state["source_config"]
    metadata_limit = max(
        max_suggestions,
        min(source_config.max_metadata_candidates, 50),
    )
    abstract_limit = max(
        max_suggestions,
        min(source_config.max_abstract_candidates, metadata_limit),
    )
    return {**state, "metadata_limit": metadata_limit, "abstract_limit": abstract_limit}


def should_continue_after_validation(state: RecommendationGraphState) -> str:
    return "continue" if state.get("claim") and state.get("max_suggestions", 0) > 0 else "stop"


def analyze_claim_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    trace.start("claim_analysis")
    profile = build_claim_profile(state["claim"], context=state["context"])
    trace.record(
        "claim_analysis",
        (
            f"Claim profile built: kind={profile.claim_kind}, "
            f"intent={profile.retrieval_intent}, phrases={profile.phrases[:5]}."
        ),
    )
    return {**state, "profile": profile}


def plan_retrieval_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    profile = state["profile"]
    local_queries = profile.query_phrases
    trace.start("query_planning")
    retrieval_plan = recommender_module.plan_retrieval_queries(
        claim=state["claim"],
        context=state["context"],
        base_queries=local_queries,
        model_config=state["model_config"],
        logger=state.get("logger"),
        force_refresh=state["force_refresh"],
    )
    context_reports = list(state.get("context_reports", []))
    if retrieval_plan.context_report:
        context_reports.append(retrieval_plan.context_report)
        trace.record(
            "context_query_planning",
            context_report_summary(retrieval_plan.context_report),
        )
    profile = replace(
        profile,
        query_phrases=retrieval_plan.queries,
        retrieval_terms=planned_retrieval_terms(retrieval_plan.concepts, retrieval_plan.queries),
        required_concept_terms=planned_retrieval_terms(retrieval_plan.required_concepts, []),
        foundational_concept_terms=planned_retrieval_terms(retrieval_plan.foundational_concepts, []),
        excluded_concept_terms=planned_retrieval_terms(retrieval_plan.excluded_concepts, []),
        claim_focus=retrieval_plan.claim_focus,
    )
    trace.record(
        "query_planning",
        (
            f"Retrieval plan source={retrieval_plan.source}, "
            f"queries={retrieval_plan.queries[:8]}, "
            f"required={retrieval_plan.required_concepts[:8]}, "
            f"reason={retrieval_plan.reason or '<empty>'}."
        ),
    )
    _log(
        state.get("logger"),
        "Claim-aware retrieval profile: "
        f"phrases={profile.phrases}, support_terms={sorted(profile.support_terms)}, "
        f"kind={profile.claim_kind}, intent={profile.retrieval_intent}, "
        f"must_include={sorted(profile.must_include_terms)}, "
        f"exclude={sorted(profile.exclude_terms)}, local_queries={local_queries}, "
        f"retrieval_terms={sorted(profile.retrieval_terms)}, "
        f"required_concepts={sorted(profile.required_concept_terms)}, "
        f"foundational_concepts={sorted(profile.foundational_concept_terms)}, "
        f"excluded_concepts={sorted(profile.excluded_concept_terms)}, "
        f"claim_focus={profile.claim_focus!r}, "
        f"retrieval_plan_source={retrieval_plan.source}, queries={retrieval_plan.queries}.",
    )
    return {
        **state,
        "profile": profile,
        "retrieval_plan": retrieval_plan,
        "context_reports": context_reports,
    }


def recall_current_bib_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    params = state["params"]
    trace.start("current_bib_search")
    bib_suggestions = BibRepository(
        workspace_path=str(params.get("workspace_path") or ""),
        file_path=str(params.get("file_path") or ""),
    ).search(state["claim"], limit=state["metadata_limit"])
    suggestions = [*state.get("suggestions", []), *bib_suggestions]
    trace.record(
        "current_bib_search",
        f"Current BibTeX search returned {len(bib_suggestions)} candidate(s): {summarize_candidates(bib_suggestions)}.",
        before=0,
        after=len(bib_suggestions),
        candidates=bib_suggestions,
        identity=suggestion_identity,
    )
    record_target_stage(
        state["target_trace"],
        "current_bib_search",
        bib_suggestions,
        state["debug_targets"],
        note=f"returned={len(bib_suggestions)}",
    )
    _log(state.get("logger"), f"Current BibTeX candidate search returned {len(bib_suggestions)} item(s).")
    return {**state, "bib_suggestions": bib_suggestions, "suggestions": suggestions}


def recall_zotero_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    source_config = state["source_config"]
    suggestions = list(state.get("suggestions", []))
    forced_target_probe_suggestions: list[CitationSuggestion] = []
    zotero_suggestions: list[CitationSuggestion] = []
    _log(
        state.get("logger"),
        "Zotero config: "
        f"enabled={source_config.zotero_enabled}, "
        f"endpoint={source_config.zotero_endpoint or '<empty>'}, "
        f"metadata_limit={state['metadata_limit']}.",
    )
    if not source_config.zotero_enabled:
        _log(state.get("logger"), "Zotero search skipped: citation-agent.zoteroEnabled=false.")
    elif not source_config.zotero_endpoint:
        _log(state.get("logger"), "Zotero search skipped: citation-agent.zoteroEndpoint is empty.")
    else:
        trace.start("zotero_retrieval")
        zotero_client = recommender_module.ZoteroClient(
            endpoint=source_config.zotero_endpoint,
            timeout_seconds=source_config.zotero_timeout_seconds,
            collection_keys=source_config.zotero_collection_keys,
            logger=state.get("logger"),
        )
        zotero_suggestions = execute_zotero_search(
            zotero_client,
            claim=state["claim"],
            limit=state["metadata_limit"],
            query_phrases=state["retrieval_plan"].queries,
            metadata_limit=state["metadata_limit"],
            fallback_candidates=state.get("bib_suggestions", []),
        )
        before = len(suggestions)
        suggestions.extend(zotero_suggestions)
        trace.record(
            "zotero_retrieval",
            (
                f"Zotero retrieval returned {len(zotero_suggestions)} candidate(s) "
                f"from collection scope={source_config.zotero_collection_keys or ['<all>']}: "
                f"{summarize_candidates(zotero_suggestions)}."
            ),
            before=before,
            after=len(suggestions),
            candidates=zotero_suggestions,
            identity=suggestion_identity,
        )
        record_target_stage(
            state["target_trace"],
            "zotero_retrieval_queries",
            zotero_suggestions,
            state["debug_targets"],
            note=f"returned={len(zotero_suggestions)}",
        )
        if state["debug_targets"]:
            forced_target_probe_suggestions = execute_zotero_search(
                zotero_client,
                claim=state["claim"],
                limit=state["metadata_limit"],
                query_phrases=state["debug_targets"],
                metadata_limit=state["metadata_limit"],
                fallback_candidates=[],
            )
            forced_target_probe_suggestions = [
                suggestion.model_copy(
                    update={
                        "reason": (
                            f"{suggestion.reason} 目标追踪：由 debugTargetPapers 旁路探测召回；"
                            "该探测不会改变主推荐结果。"
                        )
                    }
                )
                for suggestion in forced_target_probe_suggestions
            ]
            trace.record(
                "forced_debug_probe",
                (
                    f"Debug target probe returned {len(forced_target_probe_suggestions)} candidate(s); "
                    "probe is diagnostic-only and does not change main ranking."
                ),
                before=0,
                after=len(forced_target_probe_suggestions),
                candidates=forced_target_probe_suggestions,
                identity=suggestion_identity,
            )
            record_target_stage(
                state["target_trace"],
                "zotero_forced_target_probe",
                forced_target_probe_suggestions,
                state["debug_targets"],
                note=(
                    f"targets={state['debug_targets']}, returned={len(forced_target_probe_suggestions)}, "
                    "probe_only=true"
                ),
            )
        _log(state.get("logger"), f"Zotero candidate search returned {len(zotero_suggestions)} usable item(s).")
    return {
        **state,
        "zotero_suggestions": zotero_suggestions,
        "forced_target_probe_suggestions": forced_target_probe_suggestions,
        "suggestions": suggestions,
    }


def dedupe_candidates_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    suggestions = state.get("suggestions", [])
    trace.start("title_doi_dedupe")
    deduped = dedupe_suggestions(suggestions, logger=state.get("logger"))
    trace.record(
        "title_doi_dedupe",
        f"Title/DOI dedupe kept {len(deduped)} of {len(suggestions)} candidate(s).",
        before=len(suggestions),
        after=len(deduped),
        candidates=deduped,
        identity=suggestion_identity,
    )
    record_target_stage(
        state["target_trace"],
        "after_title_doi_dedupe",
        deduped,
        state["debug_targets"],
        note=f"kept={len(deduped)}, before={len(suggestions)}",
    )
    _log(state.get("logger"), f"Total candidate suggestions after title/DOI dedupe: {len(deduped)}.")
    return {**state, "deduped": deduped}


def scope_filter_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    trace.start("scope_filter")
    scoped, scope_filtered = partition_excluded_scope_candidates(
        state["deduped"],
        state["profile"],
        logger=state.get("logger"),
    )
    trace.record(
        "scope_filter",
        f"Scope filter kept {len(scoped)} and removed {len(scope_filtered)} candidate(s).",
        before=len(state["deduped"]),
        after=len(scoped),
        candidates=scoped,
        identity=suggestion_identity,
    )
    for filtered in scope_filtered:
        trace.note_candidate(
            suggestion_identity(filtered.suggestion),
            "scope_filter_removed",
            filtered.reason,
        )
    record_target_stage(
        state["target_trace"],
        "scope_filter_kept",
        scoped,
        state["debug_targets"],
        note=f"kept={len(scoped)}, removed={len(scope_filtered)}",
    )
    record_target_stage(
        state["target_trace"],
        "scope_filter_removed",
        [item.suggestion for item in scope_filtered],
        state["debug_targets"],
        note="; ".join(item.reason for item in scope_filtered) or "none",
    )
    if len(scoped) != len(state["deduped"]):
        _log(
            state.get("logger"),
            f"Scope-filtered candidates: kept={len(scoped)}, removed={len(state['deduped']) - len(scoped)}.",
        )
    return {**state, "scoped": scoped, "scope_filtered": scope_filtered}


def metadata_rank_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    trace.start("metadata_rank")
    ranked = rerank_by_claim_profile(state["scoped"], state["profile"])
    trace.record(
        "metadata_rank",
        f"Metadata ranking completed for {len(ranked)} candidate(s); top={summarize_candidates(ranked)}.",
        before=len(state["scoped"]),
        after=len(ranked),
        candidates=ranked[:10],
        identity=suggestion_identity,
    )
    for index, suggestion in enumerate(ranked, start=1):
        if index <= 20:
            trace.note_candidate(
                suggestion_identity(suggestion),
                "metadata_rank",
                f"rank={index}, metadata_score={metadata_score(suggestion, state['profile']):.3f}",
            )
    record_target_rank_stage(state["target_trace"], "metadata_rank", ranked, state["debug_targets"])
    _log(
        state.get("logger"),
        "Metadata-ranked candidates: " + ", ".join(suggestion.title for suggestion in ranked[:10]),
    )
    return {**state, "ranked": ranked}


def select_evidence_targets_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    trace.start("evidence_shortlist")
    shortlist = build_evidence_shortlist(
        ranked=state["ranked"],
        profile=state["profile"],
        abstract_limit=state["abstract_limit"],
        max_extra=state["max_suggestions"],
        logger=state.get("logger"),
    )
    metadata_diagnostics = state["ranked"][state["abstract_limit"] : state["abstract_limit"] + state["max_suggestions"]]
    trace.record(
        "evidence_shortlist",
        f"Evidence shortlist selected {len(shortlist)} candidate(s); abstract_limit={state['abstract_limit']}.",
        before=len(state["ranked"]),
        after=len(shortlist),
        candidates=shortlist,
        identity=suggestion_identity,
    )
    for suggestion in metadata_diagnostics:
        trace.note_candidate(
            suggestion_identity(suggestion),
            "metadata_diagnostic",
            "Identified by metadata rank but outside the evidence verification shortlist.",
        )
    record_target_stage(
        state["target_trace"],
        "evidence_shortlist",
        shortlist,
        state["debug_targets"],
        note=f"shortlist={len(shortlist)}, abstract_limit={state['abstract_limit']}",
    )
    record_target_rank_stage(state["target_trace"], "metadata_diagnostics", metadata_diagnostics, state["debug_targets"])
    _log(
        state.get("logger"),
        f"Evidence verification shortlist: {len(shortlist)} item(s), max_passages={state['source_config'].max_evidence_passages}.",
    )
    return {**state, "shortlist": shortlist, "metadata_diagnostics": metadata_diagnostics}


def verify_evidence_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    source_config = state["source_config"]
    model_config = state["model_config"]
    if source_config.enable_evidence_verification:
        trace.start("evidence_verification")
        verified = recommender_module.verify_evidence(
            claim=state["claim"],
            profile=state["profile"],
            suggestions=state["shortlist"],
            max_passages=source_config.max_evidence_passages,
            logger=state.get("logger"),
            force_refresh=state["force_refresh"],
        )
        evidence_stats = getattr(recommender_module.verify_evidence, "last_stats", {})
        context_reports = [
            *state.get("context_reports", []),
            *list(evidence_stats.get("context_reports") or []),
        ]
        for index, report in enumerate(evidence_stats.get("context_reports") or [], start=1):
            trace.record(
                f"context_evidence_verification_{index}",
                context_report_summary(report),
            )
        trace.record(
            "evidence_verification",
            (
                f"Evidence verification completed for {len(verified)} candidate(s). "
                f"LLM enabled={'yes' if model_config.use_llm and model_config.api_key else 'no'}, "
                f"remote={evidence_stats.get('remote_candidates', 0)}, "
                f"cache_hits={evidence_stats.get('cache_hits', 0)}, "
                f"local_fallback={evidence_stats.get('local_fallback_candidates', 0)}, "
                f"skipped_by_limit={evidence_stats.get('skipped_by_limit', 0)}, "
                f"max_llm_candidates={model_config.evidence_max_llm_candidates}."
            ),
            before=len(state["shortlist"]),
            after=len(verified),
            candidates=verified,
            identity=suggestion_identity,
        )
    else:
        verified = state["shortlist"]
        context_reports = list(state.get("context_reports", []))
        _log(state.get("logger"), "Evidence verification skipped by configuration.")
        trace.record(
            "evidence_verification",
            (
                "Evidence verification skipped by configuration. "
                f"LLM requested={'yes' if model_config.use_llm and model_config.api_key else 'no'}."
            ),
            before=len(state["shortlist"]),
            after=len(verified),
            candidates=verified,
            identity=suggestion_identity,
        )
    return {**state, "verified": verified, "context_reports": context_reports}


def rank_verified_evidence_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    trace.start("evidence_rank")
    sorted_verified = sort_verified_suggestions(state["verified"])
    memory_result = apply_feedback_memory_ranking(
        suggestions=sorted_verified,
        params=state["params"],
        source_config=state["source_config"],
        logger=state.get("logger"),
    )
    sorted_verified = memory_result.suggestions
    trace.record(
        "evidence_rank",
        (
            f"Evidence ranking completed; top={summarize_candidates(sorted_verified)}. "
            f"{memory_result.summary}"
        ),
        before=len(state["verified"]),
        after=len(sorted_verified),
        candidates=sorted_verified[:10],
        identity=suggestion_identity,
    )
    for index, suggestion in enumerate(sorted_verified, start=1):
        adjustment = memory_result.adjustments.get(suggestion_identity(suggestion))
        trace.note_candidate(
            suggestion_identity(suggestion),
            "evidence_rank",
            (
                f"rank={index}, verdict={suggestion.verdict}, "
                f"confidence={suggestion.confidence:.2f}"
                + (f", {adjustment.reason}" if adjustment else "")
            ),
        )
    record_target_rank_stage(state["target_trace"], "evidence_rank", sorted_verified, state["debug_targets"])
    _log(
        state.get("logger"),
        "Evidence-ranked candidates: "
        + ", ".join(
            f"{suggestion.verdict}:{suggestion.confidence:.2f}:{suggestion.title}"
            for suggestion in sorted_verified[:10]
        ),
    )
    return {**state, "sorted_verified": sorted_verified}


def finalize_recommendations_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    recommendable = [
        suggestion
        for suggestion in state["sorted_verified"]
        if is_recommendable_suggestion(state["profile"], suggestion)
    ]
    final = recommendable[: state["max_suggestions"]]
    trace.record(
        "final_recommendations",
        f"Final recommendation kept {len(final)} candidate(s) from {len(recommendable)} recommendable candidate(s).",
        before=len(state["sorted_verified"]),
        after=len(final),
        candidates=final,
        identity=suggestion_identity,
    )
    for suggestion in final:
        trace.note_candidate(
            suggestion_identity(suggestion),
            "final_recommendation",
            "Entered the main recommendation list.",
        )
    record_target_stage(
        state["target_trace"],
        "final_recommendations",
        final,
        state["debug_targets"],
        note=f"returned={len(final)}",
    )
    final_ids = {suggestion_identity(suggestion) for suggestion in final}
    diagnostic_limit = max(
        state["max_suggestions"] * 2,
        state["source_config"].max_abstract_candidates,
        8,
    )
    diagnostic_suggestions = [
        mark_diagnostic_suggestion(
            suggestion,
            diagnostic_reason_for_verified_suggestion(suggestion),
        )
        for suggestion in state["sorted_verified"]
        if suggestion_identity(suggestion) not in final_ids
    ][:diagnostic_limit]
    diagnostic_ids = {suggestion_identity(suggestion) for suggestion in diagnostic_suggestions}
    for filtered in state["scope_filtered"]:
        identity = suggestion_identity(filtered.suggestion)
        if identity in final_ids or identity in diagnostic_ids:
            continue
        diagnostic_suggestions.append(mark_diagnostic_suggestion(filtered.suggestion, filtered.reason))
        diagnostic_ids.add(identity)
        if len(diagnostic_suggestions) >= diagnostic_limit:
            break
    for suggestion in state["metadata_diagnostics"]:
        identity = suggestion_identity(suggestion)
        if identity in final_ids or identity in diagnostic_ids:
            continue
        diagnostic_suggestions.append(
            mark_diagnostic_suggestion(
                suggestion,
                "该候选已在元数据召回中识别，但未进入证据验证 shortlist；可提高 maxAbstractCandidates 观察其 PDF/摘要证据。",
            )
        )
        diagnostic_ids.add(identity)
        if len(diagnostic_suggestions) >= diagnostic_limit:
            break
    if state["debug_targets"]:
        diagnostic_suggestions = append_target_diagnostics(
            diagnostic_suggestions=diagnostic_suggestions,
            final_ids=final_ids,
            candidate_sources=[
                *state["sorted_verified"],
                *state["metadata_diagnostics"],
                *state["ranked"],
                *[item.suggestion for item in state["scope_filtered"]],
                *state["forced_target_probe_suggestions"],
            ],
            targets=state["debug_targets"],
        )
    record_target_stage(
        state["target_trace"],
        "diagnostic_suggestions",
        diagnostic_suggestions,
        state["debug_targets"],
        note=f"returned={len(diagnostic_suggestions)}",
    )
    for suggestion in diagnostic_suggestions:
        trace.note_candidate(
            suggestion_identity(suggestion),
            "diagnostic_suggestion",
            "Kept as a diagnostic candidate for review but not placed in the main recommendation list.",
        )
    final = trace.attach_candidate_trace(final, suggestion_identity)
    diagnostic_suggestions = trace.attach_candidate_trace(diagnostic_suggestions, suggestion_identity)
    candidate_counts = {
        "current_bib_candidates": len(state["bib_suggestions"]),
        "forced_debug_candidates": len(state["forced_target_probe_suggestions"]),
        "all_recalled_candidates": len(state["suggestions"]),
        "deduped_candidates": len(state["deduped"]),
        "scoped_candidates": len(state["scoped"]),
        "scope_filtered_candidates": len(state["scope_filtered"]),
        "metadata_ranked_candidates": len(state["ranked"]),
        "evidence_shortlist": len(state["shortlist"]),
        "verified_candidates": len(state["verified"]),
        "recommendable_candidates": len(recommendable),
        "final_recommendations": len(final),
        "diagnostic_candidates": len(diagnostic_suggestions),
    }
    trace.record(
        "stop",
        f"Run completed: final={len(final)}, diagnostic={len(diagnostic_suggestions)}, run_id={trace.run_id}.",
        before=len(state["suggestions"]),
        after=len(final),
    )
    return {
        **state,
        "final": final,
        "diagnostic_suggestions": diagnostic_suggestions,
        "candidate_counts": candidate_counts,
    }


def persist_run_node(state: RecommendationGraphState) -> RecommendationGraphState:
    trace = state["trace"]
    final = state.get("final", [])
    diagnostic_suggestions = state.get("diagnostic_suggestions", [])
    candidate_counts = state.get(
        "candidate_counts",
        {
            "current_bib_candidates": len(state.get("bib_suggestions", [])),
            "forced_debug_candidates": len(state.get("forced_target_probe_suggestions", [])),
            "all_recalled_candidates": len(state.get("suggestions", [])),
            "deduped_candidates": len(state.get("deduped", [])),
            "scoped_candidates": len(state.get("scoped", [])),
            "scope_filtered_candidates": len(state.get("scope_filtered", [])),
            "metadata_ranked_candidates": len(state.get("ranked", [])),
            "evidence_shortlist": len(state.get("shortlist", [])),
            "verified_candidates": len(state.get("verified", [])),
            "recommendable_candidates": len(final),
            "final_recommendations": len(final),
            "diagnostic_candidates": len(diagnostic_suggestions),
        },
    )
    profile = state.get("profile") or build_claim_profile(state.get("claim", ""), context=state.get("context", ""))
    retrieval_plan = state.get("retrieval_plan")
    if retrieval_plan is None:
        retrieval_plan = recommender_module.plan_retrieval_queries(
            claim=state.get("claim", ""),
            context=state.get("context", ""),
            base_queries=[],
            model_config=state["model_config"],
            logger=state.get("logger"),
            force_refresh=state.get("force_refresh", False),
        )
    tool_observations = get_tool_observations()
    for observation in tool_observations:
        trace.record("tool_call", recommender_module.format_tool_observation(observation))
    agent_state = build_agent_state_snapshot(
        params=state["params"],
        run_id=trace.run_id,
        model_snapshot=state["model_snapshot"],
        source_snapshot=state["source_snapshot"],
        claim_profile=profile,
        retrieval_plan=retrieval_plan,
        candidate_counts=candidate_counts,
        final=final,
        diagnostics=diagnostic_suggestions,
        target_trace=state["target_trace"],
        trace_events=trace.events,
        context_reports=state.get("context_reports", []),
        tool_observations=tool_observations,
        memory_snapshot=MemoryService.for_workspace(
            str(state["params"].get("workspace_path") or ""),
            state["source_config"].memory_path,
        ).stats(),
    )
    graph = agent_state.get("graph", {})
    graph["framework"] = "langgraph"
    graph["runtime"] = "langgraph-stategraph"
    graph["compiled"] = True
    graph["entrypoint"] = "initialize"
    agent_state["graph"] = graph
    agent_run_path = save_agent_run_record(state["params"], agent_state, logger=state.get("logger"))
    if state["debug_targets"]:
        _log(state.get("logger"), "EasyCite target trace:\n" + "\n".join(state["target_trace"]))
    _log(state.get("logger"), f"Total candidate suggestions after evidence ranking: {len(final)}.")
    if diagnostic_suggestions:
        _log(
            state.get("logger"),
            "Diagnostic candidates retained: "
            + ", ".join(suggestion.title for suggestion in diagnostic_suggestions[:5]),
        )
    bundle = RecommendationBundle(
        suggestions=final,
        diagnostic_suggestions=diagnostic_suggestions,
        target_trace=state["target_trace"],
        trace_events=trace.events,
        agent_state=agent_state,
        agent_run_path=agent_run_path,
        tool_warnings=recommender_module.tool_warnings_from_observations(tool_observations),
    )
    return {
        **state,
        "tool_observations": tool_observations,
        "agent_state": agent_state,
        "agent_run_path": agent_run_path,
        "bundle": bundle,
    }


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger:
        logger(message)


def context_report_summary(report: dict[str, Any]) -> str:
    compacted = [
        str(item.get("name"))
        for item in report.get("items", [])
        if item.get("action") == "compacted"
    ]
    dropped = [
        str(item.get("name"))
        for item in report.get("items", [])
        if item.get("action") == "dropped"
    ]
    return (
        f"Context budget stage={report.get('stage', '<unknown>')}, "
        f"estimated_tokens={report.get('original_tokens', 0)}->"
        f"{report.get('final_tokens', 0)}/{report.get('budget_tokens', 0)}, "
        f"compacted={compacted or ['none']}, dropped={dropped or ['none']}, "
        f"budget_exceeded={bool(report.get('budget_exceeded'))}."
    )
