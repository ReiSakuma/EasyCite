"""LLM-assisted retrieval query planning for citation search."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .cache import RetrievalPlanCache, stable_text_hash
from .claim_analysis import dedupe_preserve_order
from .config import ModelConfig
from .context import ContextItem, ContextManager, policy_for_stage
from .tools.adapters import execute_litellm_completion


@dataclass(frozen=True)
class RetrievalQueryPlan:
    """Search queries and concepts used before citation evidence judging."""

    queries: list[str]
    concepts: list[str] = field(default_factory=list)
    required_concepts: list[str] = field(default_factory=list)
    foundational_concepts: list[str] = field(default_factory=list)
    excluded_concepts: list[str] = field(default_factory=list)
    claim_focus: str = ""
    reason: str = ""
    source: str = "local"
    context_report: dict = field(default_factory=dict, compare=False, repr=False)


def plan_retrieval_queries(
    claim: str,
    context: str,
    base_queries: list[str],
    model_config: ModelConfig | None = None,
    logger: Callable[[str], None] | None = None,
    force_refresh: bool = False,
) -> RetrievalQueryPlan:
    """Return local queries plus optional LLM-expanded retrieval queries.

    The planner is deliberately upstream of evidence judging: it may improve
    recall by proposing synonyms, acronyms, and canonical method-family names,
    but it does not decide which paper is valid evidence.
    """

    config = model_config or ModelConfig.from_env()
    local_queries = sanitize_queries(base_queries)
    if not config.use_llm or not config.api_key:
        return RetrievalQueryPlan(queries=local_queries, source="local", reason="LLM query planner disabled.")

    cache = RetrievalPlanCache()
    cache_key = retrieval_plan_cache_key(
        model=config.model,
        claim=claim,
        context=context,
        base_queries=local_queries,
    )
    cached = None if force_refresh else cache.get(cache_key)
    if cached:
        plan = retrieval_plan_from_mapping(cached)
        if plan:
            plan = RetrievalQueryPlan(
                queries=plan.queries,
                concepts=plan.concepts,
                required_concepts=plan.required_concepts,
                foundational_concepts=plan.foundational_concepts,
                excluded_concepts=plan.excluded_concepts,
                claim_focus=plan.claim_focus,
                reason=f"LLM query planner cache hit. Original reason: {plan.reason}",
                source="llm_cache",
                context_report={},
            )
            if logger:
                logger(
                    "LLM query planner cache hit: "
                    f"queries={plan.queries[:16]}, required={plan.required_concepts[:8]}, "
                    f"foundational={plan.foundational_concepts[:8]}, excluded={plan.excluded_concepts[:8]}."
                )
            return plan
    if force_refresh and logger:
        logger("LLM query planner cache bypassed by force_refresh=true.")

    try:
        expanded = call_litellm_query_planner(
            claim=claim,
            context=context,
            base_queries=local_queries,
            config=config,
            logger=logger,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if logger:
            logger(f"LLM query planner failed; local queries used: {type(exc).__name__}: {exc}")
        return RetrievalQueryPlan(
            queries=local_queries,
            source="local",
            reason=f"LLM query planner failed: {type(exc).__name__}: {exc}",
        )

    queries = sanitize_queries([*expanded.queries, *local_queries])
    concepts = sanitize_queries(expanded.concepts, max_query_words=6)
    required_concepts = sanitize_queries(expanded.required_concepts, max_query_words=6)
    foundational_concepts = sanitize_queries(expanded.foundational_concepts, max_query_words=6)
    excluded_concepts = sanitize_queries(expanded.excluded_concepts, max_query_words=6)
    if logger:
        logger(
            "LLM query planner expanded retrieval queries: "
            f"queries={queries[:16]}, concepts={concepts[:8]}, "
            f"required={required_concepts[:8]}, foundational={foundational_concepts[:8]}, "
            f"excluded={excluded_concepts[:8]}, focus={expanded.claim_focus!r}, "
            f"reason={expanded.reason}"
        )
    plan = RetrievalQueryPlan(
        queries=queries,
        concepts=concepts,
        required_concepts=required_concepts,
        foundational_concepts=foundational_concepts,
        excluded_concepts=excluded_concepts,
        claim_focus=expanded.claim_focus,
        reason=expanded.reason,
        source="llm",
        context_report=expanded.context_report,
    )
    cache.put(cache_key, retrieval_plan_to_mapping(plan))
    return plan


def call_litellm_query_planner(
    claim: str,
    context: str,
    base_queries: list[str],
    config: ModelConfig,
    logger: Callable[[str], None] | None = None,
) -> RetrievalQueryPlan:
    system_prompt = (
        "You are a citation retrieval planner. Return only valid JSON with keys "
        "queries, concepts, required_concepts, foundational_concepts, excluded_concepts, "
        "claim_focus, reason. First interpret the selected claim in its paragraph context. "
        "Separate the claim's core mechanism from broad background/domain words. Generate "
        "concise Zotero/library search queries that maximize recall for papers that could "
        "support the claim. Include synonyms, acronyms, canonical method-family names, and "
        "foundational background terms when the claim implies them. For example, if a claim "
        "mentions neural or differentiable representations in visual mapping, infer related "
        "terms such as neural radiance fields, NeRF, neural implicit representation, and "
        "differentiable rendering when appropriate. required_concepts are the minimal "
        "gatekeeping mechanisms or method families a valid candidate must use, introduce, "
        "or establish; do not put generic outcome words there unless they are the mechanism. "
        "foundational_concepts are canonical foundations or method families implied by those "
        "mechanisms. excluded_concepts are context/domain terms that should not be sufficient "
        "by themselves. Do not add unrelated terms from surrounding context. "
        "Each query should be 1-5 words. Prefer conceptual queries over full sentences."
    )
    context_packet = ContextManager(
        policy_for_stage(
            "query_planning",
            query_budget=config.context_query_budget_tokens,
        ),
        logger=logger,
    ).build_json_messages(
        system_prompt=system_prompt,
        items=[
            ContextItem(
                "claim",
                claim,
                priority="required",
                source="editor_selection",
                purpose="define_the_citation_claim",
                max_tokens=800,
            ),
            ContextItem(
                "context",
                context,
                priority="compressible",
                source="tex_paragraph",
                purpose="disambiguate_claim_scope",
                min_tokens=200,
                max_tokens=1400,
            ),
            ContextItem(
                "local_queries",
                base_queries[:20],
                priority="compressible",
                source="local_query_planner",
                purpose="seed_retrieval_expansion",
                min_tokens=80,
                max_tokens=600,
                min_items=min(len(base_queries[:20]), 4),
            ),
            ContextItem(
                "max_queries", 10, priority="required", source="node_policy", purpose="bound_output"
            ),
            ContextItem(
                "max_concepts", 8, priority="required", source="node_policy", purpose="bound_output"
            ),
        ],
    )
    kwargs = {
        "model": config.model,
        "messages": context_packet.messages,
        "api_key": config.api_key,
        "timeout": min(max(config.timeout_seconds, 20.0), 60.0),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if config.api_base:
        kwargs["api_base"] = config.api_base

    response = execute_litellm_completion(kwargs)
    data = parse_json_object(response.choices[0].message.content)
    raw_queries = data.get("queries")
    raw_concepts = data.get("concepts")
    return RetrievalQueryPlan(
        queries=[str(item) for item in raw_queries if item] if isinstance(raw_queries, list) else [],
        concepts=[str(item) for item in raw_concepts if item] if isinstance(raw_concepts, list) else [],
        required_concepts=[
            str(item) for item in data.get("required_concepts", []) if item
        ]
        if isinstance(data.get("required_concepts"), list)
        else [],
        foundational_concepts=[
            str(item) for item in data.get("foundational_concepts", []) if item
        ]
        if isinstance(data.get("foundational_concepts"), list)
        else [],
        excluded_concepts=[
            str(item) for item in data.get("excluded_concepts", []) if item
        ]
        if isinstance(data.get("excluded_concepts"), list)
        else [],
        claim_focus=str(data.get("claim_focus") or ""),
        reason=str(data.get("reason") or "LLM generated retrieval expansion."),
        source="llm",
        context_report=context_packet.report.to_dict(),
    )


def sanitize_queries(
    queries: list[str],
    max_queries: int = 32,
    max_query_words: int = 5,
) -> list[str]:
    cleaned: list[str] = []
    for query in queries:
        normalized = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", query)
        normalized = re.sub(r"[^\w\s+\-:/]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip(" -:/")
        if not normalized:
            continue
        words = normalized.split()
        if len(words) > max_query_words:
            normalized = " ".join(words[:max_query_words])
        if len(normalized) < 3:
            continue
        cleaned.append(normalized)
    return dedupe_preserve_order(cleaned)[:max_queries]


def parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def retrieval_plan_cache_key(
    model: str,
    claim: str,
    context: str,
    base_queries: list[str],
) -> str:
    payload = {
        "version": 2,
        "model": model,
        "claim": re.sub(r"\s+", " ", claim).strip(),
        "context": re.sub(r"\s+", " ", context).strip(),
        "base_queries": base_queries,
    }
    return stable_text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def retrieval_plan_to_mapping(plan: RetrievalQueryPlan) -> dict:
    return {
        "queries": plan.queries,
        "concepts": plan.concepts,
        "required_concepts": plan.required_concepts,
        "foundational_concepts": plan.foundational_concepts,
        "excluded_concepts": plan.excluded_concepts,
        "claim_focus": plan.claim_focus,
        "reason": plan.reason,
        "source": plan.source,
    }


def retrieval_plan_from_mapping(raw: dict) -> RetrievalQueryPlan | None:
    try:
        return RetrievalQueryPlan(
            queries=list_of_strings(raw.get("queries")),
            concepts=list_of_strings(raw.get("concepts")),
            required_concepts=list_of_strings(raw.get("required_concepts")),
            foundational_concepts=list_of_strings(raw.get("foundational_concepts")),
            excluded_concepts=list_of_strings(raw.get("excluded_concepts")),
            claim_focus=str(raw.get("claim_focus") or ""),
            reason=str(raw.get("reason") or "Cached LLM retrieval plan."),
            source=str(raw.get("source") or "llm"),
            context_report={},
        )
    except Exception:  # pylint: disable=broad-except
        return None


def list_of_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
