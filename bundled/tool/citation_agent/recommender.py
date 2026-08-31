"""Citation candidate recommendation orchestration."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .agent_state import build_agent_state_snapshot, save_agent_run_record
from .bibliography import BibRepository, tokenize
from .claim_analysis import (
    ClaimProfile,
    build_claim_profile,
    scope_conflict_reason,
    score_text_against_claim,
)
from .config import ModelConfig, SourceConfig
from .evidence import EvidenceJudge, EvidenceRetriever, VERDICT_ORDER
from .memory import CandidateFeedbackMemory, FeedbackRankingResult
from .query_planner import plan_retrieval_queries
from .schemas import CitationSuggestion, TraceEvent
from .trace import TraceRecorder, summarize_candidates
from .tools.adapters import execute_zotero_search
from .tools.observability import get_tool_observations, reset_tool_observations
from .zotero import ZoteroClient


@dataclass(frozen=True)
class RecommendationBundle:
    suggestions: list[CitationSuggestion]
    diagnostic_suggestions: list[CitationSuggestion]
    target_trace: list[str] = field(default_factory=list)
    trace_events: list[TraceEvent] = field(default_factory=list)
    agent_state: dict[str, Any] = field(default_factory=dict)
    agent_run_path: str | None = None
    tool_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScopeFilteredCandidate:
    suggestion: CitationSuggestion
    reason: str


class CitationRecommender:
    """Search available local sources for candidate citations."""

    def __init__(
        self,
        source_config: SourceConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self.source_config = source_config or SourceConfig.from_env()
        self.logger = logger

    def recommend(self, params: dict[str, Any]) -> list[CitationSuggestion]:
        return self.recommend_with_diagnostics(params).suggestions

    def recommend_with_diagnostics(self, params: dict[str, Any]) -> RecommendationBundle:
        if not params.get("_preserve_tool_observations"):
            reset_tool_observations()
        if self.source_config.agent_runtime == "classic":
            return self._recommend_with_diagnostics_classic(params)
        try:
            from .agent_graph.runner import run_langgraph_recommendation

            return run_langgraph_recommendation(
                params=params,
                source_config=self.source_config,
                logger=self._log,
            )
        except Exception as exc:
            self._log(
                "EasyCite LangGraph runtime unavailable; "
                f"falling back to classic runtime: {type(exc).__name__}: {exc}"
            )
            return self._recommend_with_diagnostics_classic(params)

    def _recommend_with_diagnostics_classic(self, params: dict[str, Any]) -> RecommendationBundle:
        claim = str(params.get("claim", "")).strip()
        force_refresh = bool(params.get("force_refresh"))
        trace = TraceRecorder()
        trace.record(
            "start",
            f"Citation recommendation run started. runtime=classic, force_refresh={force_refresh}.",
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
            "max_suggestions": self.source_config.max_suggestions,
            "zotero_enabled": self.source_config.zotero_enabled,
            "zotero_endpoint": self.source_config.zotero_endpoint or "",
            "zotero_collection_keys": self.source_config.zotero_collection_keys,
            "max_metadata_candidates": self.source_config.max_metadata_candidates,
            "max_abstract_candidates": self.source_config.max_abstract_candidates,
            "max_evidence_passages": self.source_config.max_evidence_passages,
            "enable_evidence_verification": self.source_config.enable_evidence_verification,
            "memory_enabled": self.source_config.memory_enabled,
            "force_refresh": force_refresh,
        }
        target_trace: list[str] = []
        context_reports: list[dict[str, Any]] = []
        debug_targets = self.source_config.debug_target_papers
        if debug_targets:
            target_trace.append(
                "目标文献追踪已启用："
                + "; ".join(debug_targets)
            )
        if not claim:
            self._log("Citation recommender skipped: empty claim.")
            trace.record("stop", "Skipped because claim is empty.", before=0, after=0)
            return RecommendationBundle(
                suggestions=[],
                diagnostic_suggestions=[],
                target_trace=target_trace,
                trace_events=trace.events,
            )

        max_suggestions = max(self.source_config.max_suggestions, 0)
        if max_suggestions == 0:
            self._log("Citation recommender skipped: max_suggestions=0.")
            trace.record("stop", "Skipped because max_suggestions=0.", before=0, after=0)
            return RecommendationBundle(
                suggestions=[],
                diagnostic_suggestions=[],
                target_trace=target_trace,
                trace_events=trace.events,
            )

        context = str(params.get("context") or "")
        trace.start("claim_analysis")
        profile = build_claim_profile(claim, context=context)
        trace.record(
            "claim_analysis",
            (
                f"Claim profile built: kind={profile.claim_kind}, "
                f"intent={profile.retrieval_intent}, phrases={profile.phrases[:5]}."
            ),
        )
        local_queries = profile.query_phrases
        trace.start("query_planning")
        retrieval_plan = plan_retrieval_queries(
            claim=claim,
            context=context,
            base_queries=local_queries,
            model_config=model_config,
            logger=self._log,
            force_refresh=force_refresh,
        )
        if retrieval_plan.context_report:
            context_reports.append(retrieval_plan.context_report)
            trace.record(
                "context_query_planning",
                format_context_report(retrieval_plan.context_report),
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
        self._log(
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
            f"retrieval_plan_source={retrieval_plan.source}, queries={retrieval_plan.queries}."
        )

        metadata_limit = max(
            max_suggestions,
            min(self.source_config.max_metadata_candidates, 50),
        )
        abstract_limit = max(
            max_suggestions,
            min(self.source_config.max_abstract_candidates, metadata_limit),
        )

        suggestions: list[CitationSuggestion] = []
        forced_target_probe_suggestions: list[CitationSuggestion] = []
        trace.start("current_bib_search")
        bib_suggestions = BibRepository(
            workspace_path=str(params.get("workspace_path") or ""),
            file_path=str(params.get("file_path") or ""),
        ).search(claim, limit=metadata_limit)
        suggestions.extend(bib_suggestions)
        trace.record(
            "current_bib_search",
            f"Current BibTeX search returned {len(bib_suggestions)} candidate(s): {summarize_candidates(bib_suggestions)}.",
            before=0,
            after=len(bib_suggestions),
            candidates=bib_suggestions,
            identity=suggestion_identity,
        )
        record_target_stage(
            target_trace,
            "current_bib_search",
            bib_suggestions,
            debug_targets,
            note=f"returned={len(bib_suggestions)}",
        )
        self._log(
            f"Current BibTeX candidate search returned {len(bib_suggestions)} item(s)."
        )

        self._log(
            "Zotero config: "
            f"enabled={self.source_config.zotero_enabled}, "
            f"endpoint={self.source_config.zotero_endpoint or '<empty>'}, "
            f"metadata_limit={metadata_limit}."
        )
        if not self.source_config.zotero_enabled:
            self._log("Zotero search skipped: citation-agent.zoteroEnabled=false.")
        elif not self.source_config.zotero_endpoint:
            self._log("Zotero search skipped: citation-agent.zoteroEndpoint is empty.")
        else:
            trace.start("zotero_retrieval")
            zotero_client = ZoteroClient(
                endpoint=self.source_config.zotero_endpoint,
                timeout_seconds=self.source_config.zotero_timeout_seconds,
                collection_keys=self.source_config.zotero_collection_keys,
                logger=self._log,
            )
            zotero_suggestions = execute_zotero_search(
                zotero_client,
                claim=claim,
                limit=metadata_limit,
                query_phrases=retrieval_plan.queries,
                metadata_limit=metadata_limit,
                fallback_candidates=bib_suggestions,
            )
            suggestions.extend(zotero_suggestions)
            trace.record(
                "zotero_retrieval",
                (
                    f"Zotero retrieval returned {len(zotero_suggestions)} candidate(s) "
                    f"from collection scope={self.source_config.zotero_collection_keys or ['<all>']}: "
                    f"{summarize_candidates(zotero_suggestions)}."
                ),
                before=len(suggestions) - len(zotero_suggestions),
                after=len(suggestions),
                candidates=zotero_suggestions,
                identity=suggestion_identity,
            )
            record_target_stage(
                target_trace,
                "zotero_retrieval_queries",
                zotero_suggestions,
                debug_targets,
                note=f"returned={len(zotero_suggestions)}",
            )
            if debug_targets:
                forced_target_probe_suggestions = execute_zotero_search(
                    zotero_client,
                    claim=claim,
                    limit=metadata_limit,
                    query_phrases=debug_targets,
                    metadata_limit=metadata_limit,
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
                    target_trace,
                    "zotero_forced_target_probe",
                    forced_target_probe_suggestions,
                    debug_targets,
                    note=(
                        f"targets={debug_targets}, returned={len(forced_target_probe_suggestions)}, "
                        "probe_only=true"
                    ),
                )
            self._log(
                f"Zotero candidate search returned {len(zotero_suggestions)} usable item(s)."
            )

        trace.start("title_doi_dedupe")
        deduped = dedupe_suggestions(suggestions, logger=self._log)
        trace.record(
            "title_doi_dedupe",
            f"Title/DOI dedupe kept {len(deduped)} of {len(suggestions)} candidate(s).",
            before=len(suggestions),
            after=len(deduped),
            candidates=deduped,
            identity=suggestion_identity,
        )
        record_target_stage(
            target_trace,
            "after_title_doi_dedupe",
            deduped,
            debug_targets,
            note=f"kept={len(deduped)}, before={len(suggestions)}",
        )
        self._log(f"Total candidate suggestions after title/DOI dedupe: {len(deduped)}.")
        trace.start("scope_filter")
        scoped, scope_filtered = partition_excluded_scope_candidates(deduped, profile, logger=self._log)
        trace.record(
            "scope_filter",
            f"Scope filter kept {len(scoped)} and removed {len(scope_filtered)} candidate(s).",
            before=len(deduped),
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
            target_trace,
            "scope_filter_kept",
            scoped,
            debug_targets,
            note=f"kept={len(scoped)}, removed={len(scope_filtered)}",
        )
        record_target_stage(
            target_trace,
            "scope_filter_removed",
            [item.suggestion for item in scope_filtered],
            debug_targets,
            note="; ".join(item.reason for item in scope_filtered) or "none",
        )
        if len(scoped) != len(deduped):
            self._log(
                "Scope-filtered candidates: "
                f"kept={len(scoped)}, removed={len(deduped) - len(scoped)}."
            )

        trace.start("metadata_rank")
        ranked = rerank_by_claim_profile(scoped, profile)
        trace.record(
            "metadata_rank",
            f"Metadata ranking completed for {len(ranked)} candidate(s); top={summarize_candidates(ranked)}.",
            before=len(scoped),
            after=len(ranked),
            candidates=ranked[:10],
            identity=suggestion_identity,
        )
        for index, suggestion in enumerate(ranked, start=1):
            if index <= 20:
                trace.note_candidate(
                    suggestion_identity(suggestion),
                    "metadata_rank",
                    f"rank={index}, metadata_score={metadata_score(suggestion, profile):.3f}",
                )
        record_target_rank_stage(target_trace, "metadata_rank", ranked, debug_targets)
        self._log(
            "Metadata-ranked candidates: "
            + ", ".join(suggestion.title for suggestion in ranked[:10])
        )
        trace.start("evidence_shortlist")
        shortlist = build_evidence_shortlist(
            ranked=ranked,
            profile=profile,
            abstract_limit=abstract_limit,
            max_extra=max_suggestions,
            logger=self._log,
        )
        metadata_diagnostics = ranked[abstract_limit : abstract_limit + max_suggestions]
        trace.record(
            "evidence_shortlist",
            f"Evidence shortlist selected {len(shortlist)} candidate(s); abstract_limit={abstract_limit}.",
            before=len(ranked),
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
            target_trace,
            "evidence_shortlist",
            shortlist,
            debug_targets,
            note=f"shortlist={len(shortlist)}, abstract_limit={abstract_limit}",
        )
        record_target_rank_stage(target_trace, "metadata_diagnostics", metadata_diagnostics, debug_targets)
        self._log(
            f"Evidence verification shortlist: {len(shortlist)} item(s), "
            f"max_passages={self.source_config.max_evidence_passages}."
        )
        if self.source_config.enable_evidence_verification:
            trace.start("evidence_verification")
            verified = verify_evidence(
                claim=claim,
                profile=profile,
                suggestions=shortlist,
                max_passages=self.source_config.max_evidence_passages,
                logger=self._log,
                force_refresh=force_refresh,
            )
            evidence_stats = getattr(verify_evidence, "last_stats", {})
            new_context_reports = list(evidence_stats.get("context_reports") or [])
            context_reports.extend(new_context_reports)
            for index, report in enumerate(new_context_reports, start=1):
                trace.record(
                    f"context_evidence_verification_{index}",
                    format_context_report(report),
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
                before=len(shortlist),
                after=len(verified),
                candidates=verified,
                identity=suggestion_identity,
            )
        else:
            verified = shortlist
            self._log("Evidence verification skipped by configuration.")
            trace.record(
                "evidence_verification",
                (
                    "Evidence verification skipped by configuration. "
                    f"LLM requested={'yes' if model_config.use_llm and model_config.api_key else 'no'}."
                ),
                before=len(shortlist),
                after=len(verified),
                candidates=verified,
                identity=suggestion_identity,
            )

        trace.start("evidence_rank")
        sorted_verified = sort_verified_suggestions(verified)
        memory_result = apply_feedback_memory_ranking(
            suggestions=sorted_verified,
            params=params,
            source_config=self.source_config,
            logger=self._log,
        )
        sorted_verified = memory_result.suggestions
        trace.record(
            "evidence_rank",
            (
                f"Evidence ranking completed; top={summarize_candidates(sorted_verified)}. "
                f"{memory_result.summary}"
            ),
            before=len(verified),
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
        record_target_rank_stage(target_trace, "evidence_rank", sorted_verified, debug_targets)
        self._log(
            "Evidence-ranked candidates: "
            + ", ".join(
                f"{suggestion.verdict}:{suggestion.confidence:.2f}:{suggestion.title}"
                for suggestion in sorted_verified[:10]
            )
        )
        recommendable = [
            suggestion
            for suggestion in sorted_verified
            if is_recommendable_suggestion(profile, suggestion)
        ]
        final = recommendable[:max_suggestions]
        trace.record(
            "final_recommendations",
            f"Final recommendation kept {len(final)} candidate(s) from {len(recommendable)} recommendable candidate(s).",
            before=len(sorted_verified),
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
            target_trace,
            "final_recommendations",
            final,
            debug_targets,
            note=f"returned={len(final)}",
        )
        final_ids = {suggestion_identity(suggestion) for suggestion in final}
        diagnostic_limit = max(
            max_suggestions * 2,
            self.source_config.max_abstract_candidates,
            8,
        )
        diagnostic_suggestions = [
            mark_diagnostic_suggestion(
                suggestion,
                diagnostic_reason_for_verified_suggestion(suggestion),
            )
            for suggestion in sorted_verified
            if suggestion_identity(suggestion) not in final_ids
        ]
        diagnostic_suggestions = diagnostic_suggestions[:diagnostic_limit]
        diagnostic_ids = {suggestion_identity(suggestion) for suggestion in diagnostic_suggestions}
        for filtered in scope_filtered:
            identity = suggestion_identity(filtered.suggestion)
            if identity in final_ids or identity in diagnostic_ids:
                continue
            diagnostic_suggestions.append(
                mark_diagnostic_suggestion(filtered.suggestion, filtered.reason)
            )
            diagnostic_ids.add(identity)
            if len(diagnostic_suggestions) >= diagnostic_limit:
                break
        for suggestion in metadata_diagnostics:
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
        if debug_targets:
            diagnostic_suggestions = append_target_diagnostics(
            diagnostic_suggestions=diagnostic_suggestions,
            final_ids=final_ids,
                candidate_sources=[
                    *sorted_verified,
                    *metadata_diagnostics,
                    *ranked,
                    *[item.suggestion for item in scope_filtered],
                    *forced_target_probe_suggestions,
                ],
                targets=debug_targets,
            )
        record_target_stage(
            target_trace,
            "diagnostic_suggestions",
            diagnostic_suggestions,
            debug_targets,
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
            "current_bib_candidates": len(bib_suggestions),
            "forced_debug_candidates": len(forced_target_probe_suggestions),
            "all_recalled_candidates": len(suggestions),
            "deduped_candidates": len(deduped),
            "scoped_candidates": len(scoped),
            "scope_filtered_candidates": len(scope_filtered),
            "metadata_ranked_candidates": len(ranked),
            "evidence_shortlist": len(shortlist),
            "verified_candidates": len(verified),
            "recommendable_candidates": len(recommendable),
            "final_recommendations": len(final),
            "diagnostic_candidates": len(diagnostic_suggestions),
        }
        trace.record(
            "stop",
            (
                f"Run completed: final={len(final)}, "
                f"diagnostic={len(diagnostic_suggestions)}, run_id={trace.run_id}."
            ),
            before=len(suggestions),
            after=len(final),
        )
        tool_observations = get_tool_observations()
        for observation in tool_observations:
            trace.record("tool_call", format_tool_observation(observation))
        agent_state = build_agent_state_snapshot(
            params=params,
            run_id=trace.run_id,
            model_snapshot=model_snapshot,
            source_snapshot=source_snapshot,
            claim_profile=profile,
            retrieval_plan=retrieval_plan,
            candidate_counts=candidate_counts,
            final=final,
            diagnostics=diagnostic_suggestions,
            target_trace=target_trace,
            trace_events=trace.events,
            context_reports=context_reports,
            tool_observations=tool_observations,
        )
        agent_run_path = save_agent_run_record(params, agent_state, logger=self._log)
        if debug_targets:
            self._log("EasyCite target trace:\n" + "\n".join(target_trace))
        self._log(f"Total candidate suggestions after evidence ranking: {len(final)}.")
        if diagnostic_suggestions:
            self._log(
                "Diagnostic candidates retained: "
                + ", ".join(suggestion.title for suggestion in diagnostic_suggestions[:5])
            )
        return RecommendationBundle(
            suggestions=final,
            diagnostic_suggestions=diagnostic_suggestions,
            target_trace=target_trace,
            trace_events=trace.events,
            agent_state=agent_state,
            agent_run_path=agent_run_path,
            tool_warnings=tool_warnings_from_observations(tool_observations),
        )

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


def format_context_report(report: dict[str, Any]) -> str:
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


def format_tool_observation(observation: dict[str, Any]) -> str:
    return (
        f"tool_call_id={observation.get('tool_call_id')}, "
        f"tool={observation.get('tool_name')}@{observation.get('tool_version')}, "
        f"success={observation.get('success')}, attempt={observation.get('attempt')}, "
        f"latency_ms={observation.get('latency_ms')}, "
        f"fallback={observation.get('fallback') or 'none'}, "
        f"error_category={observation.get('error_category') or 'none'}, "
        f"output_validated={observation.get('output_validated')}."
    )


def tool_warnings_from_observations(observations: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    latest_by_tool: dict[str, dict[str, Any]] = {}
    for observation in observations:
        tool_name = str(observation.get("tool_name") or "unknown")
        latest_by_tool[tool_name] = observation
        fallback = observation.get("fallback")
        if fallback:
            warning = f"{tool_name} 不可用，已降级到 {fallback}。"
            if warning not in warnings:
                warnings.append(warning)
    for tool_name, observation in latest_by_tool.items():
        if observation.get("success"):
            continue
        category = observation.get("error_category") or "internal_error"
        warning = f"{tool_name} 调用失败（{category}），系统已保留可用的本地降级流程。"
        if warning not in warnings:
            warnings.append(warning)
    return warnings


def planned_retrieval_terms(concepts: list[str], queries: list[str]) -> set[str]:
    """Convert planned query concepts into weak metadata-ranking terms."""

    terms: set[str] = set()
    for text in [*concepts, *queries[:12]]:
        terms.update(tokenize(text))
    return terms


def append_target_suggestions(
    shortlist: list[CitationSuggestion],
    ranked: list[CitationSuggestion],
    targets: list[str],
) -> list[CitationSuggestion]:
    identities = {suggestion_identity(suggestion) for suggestion in shortlist}
    appended = list(shortlist)
    for match in target_matches_in_suggestions(ranked, targets):
        suggestion = match["suggestion"]
        identity = suggestion_identity(suggestion)
        if identity in identities:
            continue
        appended.append(
            suggestion.model_copy(
                update={
                    "reason": (
                        f"{suggestion.reason} 目标追踪：该文献未进入常规 evidence shortlist，"
                        "但因 debugTargetPapers 被追加验证。"
                    )
                }
            )
        )
        identities.add(identity)
    return appended


def build_evidence_shortlist(
    ranked: list[CitationSuggestion],
    profile: ClaimProfile,
    abstract_limit: int,
    max_extra: int,
    logger: Callable[[str], None] | None = None,
) -> list[CitationSuggestion]:
    """Build evidence shortlist from top metadata rank plus strong contract hits.

    Metadata rank is a fast coarse sort. It should not be the only gate before
    expensive evidence reading, because foundational papers can have sparse
    metadata for the exact paragraph domain. This adds a small, deterministic
    second lane for candidates that strongly match the LLM-derived citation
    contract even when their metadata rank is lower.
    """

    base = list(ranked[:abstract_limit])
    identities = {suggestion_identity(suggestion) for suggestion in base}
    extra_budget = max(max_extra, 0)
    if extra_budget == 0:
        return base

    scored_extra: list[tuple[float, CitationSuggestion, str]] = []
    for suggestion in ranked[abstract_limit:]:
        identity = suggestion_identity(suggestion)
        if identity in identities:
            continue
        score, reason = contract_shortlist_score(suggestion, profile)
        if score < 0.42:
            continue
        scored_extra.append((score, suggestion, reason))

    scored_extra.sort(
        key=lambda item: (
            -round(item[0], 6),
            metadata_ranking_key(item[1], profile),
        )
    )
    extras: list[CitationSuggestion] = []
    for score, suggestion, reason in scored_extra[:extra_budget]:
        extras.append(
            suggestion.model_copy(
                update={
                    "reason": (
                        f"{suggestion.reason} Contract shortlist: {reason} "
                        f"(contract_score={score:.2f})."
                    )
                }
            )
        )
        identities.add(suggestion_identity(suggestion))

    if logger and extras:
        logger(
            "Evidence shortlist contract extras: "
            + ", ".join(f"{suggestion.title}" for suggestion in extras)
        )
    return [*base, *extras]


def contract_shortlist_score(
    suggestion: CitationSuggestion,
    profile: ClaimProfile,
) -> tuple[float, str]:
    required_terms = profile.required_concept_terms | profile.foundational_concept_terms
    contract_source = "llm"
    if not required_terms:
        required_terms = fallback_contract_terms(profile)
        contract_source = "claim"
    if not required_terms:
        return 0.0, "no required/foundational/fallback contract terms"

    text = " ".join(
        part
        for part in [
            suggestion.title,
            suggestion.venue or "",
            suggestion.evidence.text or "",
        ]
        if part
    )
    terms = tokenize(text)
    required_hits = required_terms & terms
    min_hits = 3 if len(required_terms) >= 5 else 2
    if len(required_hits) < min_hits:
        return 0.0, "too few contract term hits"

    excluded_terms = profile.excluded_concept_terms | profile.exclude_terms
    if excluded_terms and excluded_terms & terms and not required_hits:
        return 0.0, "only background/context terms matched"

    coverage = len(required_hits) / max(len(required_terms), 1)
    foundational_hits = profile.foundational_concept_terms & terms
    retrieval_hits = profile.retrieval_terms & terms
    score = coverage
    score += min(0.25, len(required_hits) * 0.04)
    score += min(0.2, len(foundational_hits) * 0.06)
    score += min(0.12, len(retrieval_hits) * 0.02)
    reason = (
        "低排名候选强命中 citation contract："
        f"source={contract_source}, required_hits={', '.join(sorted(required_hits)) or 'none'}"
    )
    return min(score, 1.0), reason


def fallback_contract_terms(profile: ClaimProfile) -> set[str]:
    """Build a conservative citation contract when LLM retrieval planning fails.

    This is intentionally generic: it uses terms extracted from the selected
    claim itself, not paper names or domain-specific hard-coded targets.
    """

    candidates = set(profile.retrieval_terms) | set(profile.must_include_terms)
    candidates -= set(profile.exclude_terms)
    candidates -= {
        "claim",
        "claims",
        "design",
        "field",
        "fields",
        "general",
        "generally",
        "problem",
        "recent",
        "recently",
        "space",
        "system",
        "systems",
        "work",
        "works",
    }
    return {term for term in candidates if len(term) >= 4}


def append_target_diagnostics(
    diagnostic_suggestions: list[CitationSuggestion],
    final_ids: set[str],
    candidate_sources: list[CitationSuggestion],
    targets: list[str],
) -> list[CitationSuggestion]:
    identities = {suggestion_identity(suggestion) for suggestion in diagnostic_suggestions}
    appended = list(diagnostic_suggestions)
    for match in target_matches_in_suggestions(candidate_sources, targets):
        suggestion = match["suggestion"]
        identity = suggestion_identity(suggestion)
        if identity in final_ids or identity in identities:
            continue
        appended.append(
            mark_diagnostic_suggestion(
                suggestion,
                "目标追踪：该文献被 debugTargetPapers 旁路命中，仅保留在诊断区；不会改变主推荐候选。",
            )
        )
        identities.add(identity)
    return appended


def record_target_stage(
    trace: list[str],
    stage: str,
    suggestions: list[CitationSuggestion],
    targets: list[str],
    note: str = "",
) -> None:
    if not targets:
        return
    matches = target_matches_in_suggestions(suggestions, targets)
    if matches:
        rendered = "; ".join(render_target_match(match) for match in matches)
        trace.append(f"{stage}: FOUND {rendered}. {note}".strip())
    else:
        trace.append(f"{stage}: not found among {len(suggestions)} candidate(s). {note}".strip())


def record_target_rank_stage(
    trace: list[str],
    stage: str,
    suggestions: list[CitationSuggestion],
    targets: list[str],
) -> None:
    if not targets:
        return
    matches = target_matches_in_suggestions(suggestions, targets)
    if matches:
        rendered = []
        for match in matches:
            rank = suggestions.index(match["suggestion"]) + 1
            suggestion = match["suggestion"]
            rendered.append(
                f"rank={rank}, target={match['target']!r}, "
                f"title={suggestion.title!r}, verdict={suggestion.verdict}, "
                f"confidence={suggestion.confidence:.2f}"
            )
        trace.append(f"{stage}: FOUND " + "; ".join(rendered))
    else:
        trace.append(f"{stage}: not found among {len(suggestions)} ranked candidate(s).")


def target_matches_in_suggestions(
    suggestions: list[CitationSuggestion],
    targets: list[str],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for suggestion in suggestions:
        for target in targets:
            if suggestion_matches_target(suggestion, target):
                key = (suggestion_identity(suggestion), target.lower())
                if key in seen:
                    continue
                seen.add(key)
                matches.append({"target": target, "suggestion": suggestion})
    return matches


def suggestion_matches_target(suggestion: CitationSuggestion, target: str) -> bool:
    normalized_target = normalize_target_text(target)
    if not normalized_target:
        return False
    haystack_parts = [
        suggestion.title,
        suggestion.citation_key or "",
        suggestion.paper_id,
        suggestion.doi or "",
        suggestion.url or "",
        " ".join(suggestion.authors),
    ]
    normalized_haystack = normalize_target_text(" ".join(haystack_parts))
    if normalized_target in normalized_haystack:
        return True

    target_terms = set(tokenize(target))
    haystack_terms = set(tokenize(" ".join(haystack_parts)))
    if len(target_terms) >= 2:
        return len(target_terms & haystack_terms) / len(target_terms) >= 0.7
    return bool(target_terms & haystack_terms)


def normalize_target_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", lowered)
    lowered = re.sub(r"^doi:\s*", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def render_target_match(match: dict[str, Any]) -> str:
    suggestion: CitationSuggestion = match["suggestion"]
    return (
        f"target={match['target']!r}, title={suggestion.title!r}, "
        f"key=@{suggestion.citation_key or '<none>'}, source={suggestion.source}, "
        f"verdict={suggestion.verdict}, confidence={suggestion.confidence:.2f}"
    )


def dedupe_suggestions(
    suggestions: list[CitationSuggestion],
    logger: Callable[[str], None] | None = None,
) -> list[CitationSuggestion]:
    deduped: list[CitationSuggestion] = []
    for suggestion in suggestions:
        match_index = next(
            (
                index
                for index, existing in enumerate(deduped)
                if is_same_article(existing, suggestion)
            ),
            None,
        )
        if match_index is None:
            deduped.append(suggestion)
            continue
        existing = deduped[match_index]
        chosen = choose_better_duplicate(existing, suggestion)
        if logger:
            logger(
                "Duplicate citation candidate merged: "
                f"kept={chosen.title!r} @{chosen.citation_key or '<no-key>'}, "
                f"dropped={suggestion.title!r} @{suggestion.citation_key or '<no-key>'}"
            )
        deduped[match_index] = chosen
    return deduped


def filter_excluded_scope_candidates(
    suggestions: list[CitationSuggestion],
    profile: ClaimProfile,
    logger: Callable[[str], None] | None = None,
) -> list[CitationSuggestion]:
    return partition_excluded_scope_candidates(suggestions, profile, logger=logger)[0]


def partition_excluded_scope_candidates(
    suggestions: list[CitationSuggestion],
    profile: ClaimProfile,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[CitationSuggestion], list[ScopeFilteredCandidate]]:
    if not profile.exclude_terms:
        return suggestions, []

    kept: list[CitationSuggestion] = []
    removed: list[ScopeFilteredCandidate] = []
    for suggestion in suggestions:
        reason = suggestion_scope_conflict_reason(suggestion, profile)
        if reason:
            if logger:
                logger(
                    "Candidate removed by claim scope exclusion: "
                    f"title={suggestion.title!r}, reason={reason}"
                )
            removed.append(
                ScopeFilteredCandidate(
                    suggestion=suggestion,
                    reason=f"范围过滤：{reason}",
                )
            )
            continue
        kept.append(suggestion)
    return kept, removed


def violates_exclusion_scope(suggestion: CitationSuggestion, profile: ClaimProfile) -> bool:
    """Filter candidates dominated by topic anchors found only in surrounding context."""
    return suggestion_scope_conflict_reason(suggestion, profile) is not None


def suggestion_scope_conflict_reason(
    suggestion: CitationSuggestion,
    profile: ClaimProfile,
) -> str | None:
    identity_text = " ".join(
        part
        for part in [
            suggestion.title,
            suggestion.citation_key or "",
            suggestion.paper_id,
        ]
        if part
    )
    identity_terms = tokenize(identity_text)
    required_terms = profile.required_concept_terms | profile.foundational_concept_terms
    excluded_terms = profile.excluded_concept_terms | profile.exclude_terms
    if excluded_terms and excluded_terms & identity_terms and required_terms and not (required_terms & identity_terms):
        return (
            "候选主要命中上下文/背景概念，但缺少待引用句要求的核心机制概念："
            f"background={', '.join(sorted(excluded_terms & identity_terms))}; "
            f"required={', '.join(sorted(required_terms))}"
        )
    if not profile.exclude_terms:
        return None
    return scope_conflict_reason(profile, identity_text)


def is_same_article(first: CitationSuggestion, second: CitationSuggestion) -> bool:
    first_doi = normalize_doi(first.doi)
    second_doi = normalize_doi(second.doi)
    if first_doi and second_doi and first_doi == second_doi:
        return True

    first_title = normalize_title(first.title)
    second_title = normalize_title(second.title)
    if first_title and second_title and first_title == second_title:
        return True

    if titles_are_near_duplicates(first_title, second_title):
        return True

    if first.citation_key and second.citation_key:
        return first.citation_key.lower() == second.citation_key.lower()

    return False


def suggestion_identity(suggestion: CitationSuggestion) -> str:
    doi = normalize_doi(suggestion.doi)
    if doi:
        return f"doi:{doi}"

    title_key = normalize_title(suggestion.title)
    if title_key:
        return f"title:{title_key}"

    if suggestion.citation_key:
        return f"citekey:{suggestion.citation_key.lower()}"

    return f"paper:{suggestion.paper_id.lower()}"


def normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    lowered = doi.lower().strip()
    lowered = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", lowered)
    lowered = re.sub(r"^doi:\s*", "", lowered)
    return lowered.rstrip("./")


def normalize_title(title: str) -> str:
    lowered = title.lower()
    lowered = re.sub(r"\{|\}|\\[a-zA-Z]+\s*", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    tokens = [token for token in lowered.split() if token]
    return " ".join(tokens)


TITLE_TRAILING_GENERIC_TERMS = {
    "article",
    "draft",
    "paper",
    "preprint",
    "study",
    "system",
    "systems",
    "method",
    "methods",
    "approach",
    "approaches",
    "framework",
}


def title_signature_tokens(normalized_title: str) -> set[str]:
    tokens = {
        token
        for token in normalized_title.split()
        if len(token) > 2 and token not in TITLE_TRAILING_GENERIC_TERMS
    }
    return tokens


def titles_are_near_duplicates(first_title: str, second_title: str) -> bool:
    first_terms = title_signature_tokens(first_title)
    second_terms = title_signature_tokens(second_title)
    if len(first_terms) < 3 or len(second_terms) < 3:
        return False

    overlap = len(first_terms & second_terms)
    containment = overlap / min(len(first_terms), len(second_terms))
    jaccard = overlap / len(first_terms | second_terms)
    return containment >= 0.9 or jaccard >= 0.82


def choose_better_duplicate(
    first: CitationSuggestion,
    second: CitationSuggestion,
) -> CitationSuggestion:
    def score(suggestion: CitationSuggestion) -> tuple[int, int, int, float, str, str]:
        return (
            1 if suggestion.source == "current_bib" else 0,
            1 if suggestion.citation_key else 0,
            1 if suggestion.pdf_path else 0,
            suggestion.confidence,
            normalize_title(suggestion.title),
            (suggestion.citation_key or suggestion.paper_id).lower(),
        )

    chosen = max([first, second], key=score)
    alternate = second if chosen is first else first
    chosen_evidence_is_metadata = chosen.evidence.source == "metadata"
    alternate_has_content = alternate.evidence.source in {"abstract", "pdf"}
    return chosen.model_copy(
        update={
            "citation_key": chosen.citation_key or alternate.citation_key,
            "authors": chosen.authors or alternate.authors,
            "year": chosen.year or alternate.year,
            "venue": chosen.venue or alternate.venue,
            "doi": chosen.doi or alternate.doi,
            "url": chosen.url or alternate.url,
            "pdf_path": chosen.pdf_path or alternate.pdf_path,
            "evidence": alternate.evidence
            if chosen_evidence_is_metadata and alternate_has_content
            else chosen.evidence,
            "evidence_passages": dedupe_evidence_snippets(
                [*chosen.evidence_passages, *alternate.evidence_passages]
            ),
        }
    )


def dedupe_evidence_snippets(snippets: list) -> list:
    seen: set[tuple[str, int | None, str]] = set()
    result = []
    for snippet in snippets:
        key = (
            str(snippet.source),
            snippet.page,
            re.sub(r"\s+", " ", str(snippet.text or "")).strip().lower(),
        )
        if not key[2] or key in seen:
            continue
        seen.add(key)
        result.append(snippet)
    return result


def mark_diagnostic_suggestion(
    suggestion: CitationSuggestion,
    diagnostic_reason: str,
) -> CitationSuggestion:
    return suggestion.model_copy(
        update={
            "reason": f"{suggestion.reason} 诊断：{diagnostic_reason}",
        }
    )


def is_recommendable_suggestion(profile: ClaimProfile, suggestion: CitationSuggestion) -> bool:
    if suggestion.verdict in {"direct_support", "foundational_support"}:
        return True
    if profile.claim_kind == "field_level_background":
        return False
    return suggestion.verdict in {"partial_support", "related_only", "metadata_match"}


def diagnostic_reason_for_verified_suggestion(suggestion: CitationSuggestion) -> str:
    if suggestion.verdict == "insufficient_evidence":
        return "证据验证结论为证据不足，因此不进入主推荐区。"
    if suggestion.verdict == "contradiction":
        return "证据验证发现可能矛盾，因此不进入主推荐区。"
    return "已完成证据验证，但综合 verdict/置信度排序未进入最终推荐前五。"


def rerank_by_claim_profile(
    suggestions: list[CitationSuggestion],
    profile: ClaimProfile,
) -> list[CitationSuggestion]:
    return sorted(
        suggestions,
        key=lambda suggestion: metadata_ranking_key(suggestion, profile),
    )


def metadata_ranking_key(
    suggestion: CitationSuggestion,
    profile: ClaimProfile,
) -> tuple[float, int, str, str]:
    return (
        -round(metadata_score(suggestion, profile), 6),
        source_priority(suggestion.source),
        normalize_title(suggestion.title),
        (suggestion.citation_key or suggestion.paper_id).lower(),
    )


def metadata_score(suggestion: CitationSuggestion, profile: ClaimProfile) -> float:
    text = " ".join(
        part
        for part in [
            suggestion.title,
            " ".join(suggestion.authors),
            suggestion.venue or "",
            suggestion.evidence.text or "",
        ]
        if part
    )
    score = score_text_against_claim(profile, text)
    text_terms = set(re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text.lower()))
    if profile.must_include_terms and not (profile.must_include_terms & text_terms):
        score -= 0.45
    excluded_hits = profile.exclude_terms & text_terms
    if excluded_hits:
        must_hits = profile.must_include_terms & text_terms
        penalty_unit = 0.2 if must_hits else 0.35
        score -= min(0.8, penalty_unit * len(excluded_hits))
    if suggestion.citation_key:
        score += 0.05
    if suggestion.source == "current_bib":
        score += 0.03
    return score


def verify_evidence(
    claim: str,
    profile: ClaimProfile,
    suggestions: list[CitationSuggestion],
    max_passages: int,
    logger: Callable[[str], None] | None = None,
    force_refresh: bool = False,
) -> list[CitationSuggestion]:
    retriever = EvidenceRetriever(max_passages=max_passages, logger=logger)
    judge = EvidenceJudge(logger=logger, force_refresh=force_refresh)
    max_llm_candidates = max(judge.model_config.evidence_max_llm_candidates, 0)
    evidence_items: list[tuple[CitationSuggestion, Any]] = []
    for suggestion in suggestions:
        evidence_items.append((suggestion, retriever.retrieve(suggestion, profile, claim=claim)))

    llm_items = evidence_items[:max_llm_candidates]
    local_items = evidence_items[max_llm_candidates:]
    if logger and judge.model_config.use_llm and judge.model_config.api_key and llm_items:
        logger(f"Batch LLM evidence judge requested for {len(llm_items)} candidate(s).")

    verified = judge.judge_batch(claim, profile, llm_items) if llm_items else []
    batch_stats = dict(judge.last_batch_stats)
    if local_items and logger and judge.model_config.use_llm and judge.model_config.api_key:
        logger(
            "LLM evidence judge skipped for "
            f"{len(local_items)} candidate(s): max_llm_candidates={max_llm_candidates}; "
            "local rules used."
        )
    verified.extend(
        judge.judge(claim, profile, suggestion, passages, allow_llm=False)
        for suggestion, passages in local_items
    )
    verify_evidence.last_stats = {
        "llm_enabled": bool(judge.model_config.use_llm and judge.model_config.api_key),
        "remote_candidates": int(batch_stats.get("remote_candidates", 0) or 0),
        "remote_attempted_candidates": int(batch_stats.get("remote_attempted_candidates", 0) or 0),
        "cache_hits": int(batch_stats.get("cache_hits", 0) or 0),
        "local_fallback_candidates": int(batch_stats.get("local_fallback_candidates", 0) or 0) + len(local_items),
        "skipped_by_limit": len(local_items),
        "error": batch_stats.get("error"),
        "context_reports": list(batch_stats.get("context_reports") or []),
    }

    for judged in verified:
        if logger:
            logger(
                "Evidence verdict: "
                f"{judged.verdict} confidence={judged.confidence:.2f} "
                f"title={judged.title!r}"
            )
    return verified


def sort_verified_suggestions(
    suggestions: list[CitationSuggestion],
) -> list[CitationSuggestion]:
    return sorted(
        suggestions,
        key=lambda suggestion: (
            -VERDICT_ORDER.get(suggestion.verdict, 0),
            -round(suggestion.confidence, 6),
            -int(bool(suggestion.citation_key)),
            source_priority(suggestion.source),
            normalize_title(suggestion.title),
            (suggestion.citation_key or suggestion.paper_id).lower(),
        ),
    )


def apply_feedback_memory_ranking(
    *,
    suggestions: list[CitationSuggestion],
    params: dict[str, Any],
    source_config: SourceConfig,
    logger: Callable[[str], None] | None = None,
) -> FeedbackRankingResult:
    if not source_config.memory_enabled:
        return FeedbackRankingResult(suggestions=suggestions, summary="memory disabled")
    memory = CandidateFeedbackMemory.for_workspace(
        str(params.get("workspace_path") or ""),
        source_config.memory_path,
    )
    result = memory.rank_with_feedback(
        suggestions=suggestions,
        claim=str(params.get("claim") or ""),
        context=str(params.get("context") or ""),
        positive_boost=max(0.0, min(source_config.memory_positive_boost, 0.25)),
        negative_penalty=max(0.0, min(source_config.memory_negative_penalty, 0.5)),
        min_similarity=max(0.0, min(source_config.memory_min_similarity, 1.0)),
    )
    if logger and result.adjustments:
        logger(
            "EasyCite memory adjusted candidates: "
            + "; ".join(adjustment.reason for adjustment in result.adjustments.values())
        )
    return result


def source_priority(source: str) -> int:
    return {
        "current_bib": 0,
        "zotero": 1,
        "online": 2,
        "mock": 3,
    }.get(source, 4)
