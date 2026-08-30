"""Claim-aware query planning for citation retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bibliography import STOPWORDS, tokenize


FIELD_PHRASES: dict[str, tuple[str, ...]] = {}

PREDICATE_MARKERS = {
    "rely": {"rely", "relies", "relying", "depend", "depends"},
    "offline": {"offline"},
    "optimization": {"optimization", "optimisation", "optimize", "optimise"},
    "global": {"global", "globally"},
    "consistency": {"consistent", "consistency"},
}

GENERIC_SCOPE_TERMS = STOPWORDS | {
    "about",
    "achieve",
    "achieves",
    "across",
    "also",
    "approach",
    "approaches",
    "because",
    "become",
    "between",
    "case",
    "challenge",
    "challenges",
    "changed",
    "changes",
    "changing",
    "claim",
    "claims",
    "common",
    "contain",
    "contains",
    "current",
    "design",
    "discuss",
    "discusses",
    "effect",
    "effects",
    "especially",
    "existing",
    "field",
    "fields",
    "general",
    "generally",
    "had",
    "has",
    "have",
    "high",
    "impact",
    "important",
    "large",
    "latest",
    "maintain",
    "many",
    "most",
    "multiple",
    "quality",
    "recently",
    "requires",
    "research",
    "result",
    "results",
    "several",
    "shows",
    "specific",
    "state",
    "study",
    "support",
    "supports",
    "system",
    "systems",
    "joint",
    "space",
    "task",
    "tasks",
    "treated",
    "treating",
    "trend",
    "usually",
    "work",
    "works",
}

MAX_SCOPE_TERMS = 12


@dataclass(frozen=True)
class ClaimProfile:
    """Structured retrieval plan derived from a selected claim."""

    claim: str
    terms: set[str] = field(default_factory=set)
    phrases: list[str] = field(default_factory=list)
    support_terms: set[str] = field(default_factory=set)
    query_phrases: list[str] = field(default_factory=list)
    claim_kind: str = "general_claim"
    retrieval_intent: str = "find_direct_evidence"
    must_include_terms: set[str] = field(default_factory=set)
    exclude_terms: set[str] = field(default_factory=set)
    context_only_terms: set[str] = field(default_factory=set)
    retrieval_terms: set[str] = field(default_factory=set)
    required_concept_terms: set[str] = field(default_factory=set)
    foundational_concept_terms: set[str] = field(default_factory=set)
    excluded_concept_terms: set[str] = field(default_factory=set)
    claim_focus: str = ""


def build_claim_profile(claim: str, context: str = "") -> ClaimProfile:
    """Build a compact, deterministic retrieval profile for a claim.

    The LLM can later refine this, but V2.5 keeps query planning local so Zotero
    recall stays stable and debuggable.
    """

    claim_lowered = claim.lower()
    terms = tokenize(claim)
    phrases = [
        phrase
        for phrase, required_terms in FIELD_PHRASES.items()
        if all(term in terms or re.search(rf"\b{re.escape(term)}\b", claim_lowered) for term in required_terms)
    ]
    support_terms = {
        label
        for label, variants in PREDICATE_MARKERS.items()
        if terms & variants or any(variant in claim_lowered for variant in variants)
    }

    claim_kind = classify_claim_kind(claim_lowered, terms)
    retrieval_intent = retrieval_intent_for_claim(claim_kind, terms)
    must_include_terms = derive_must_include_terms(terms, phrases)
    context_only_terms = derive_context_only_terms(terms, context)
    exclude_terms = derive_exclude_terms(terms, context)
    query_phrases = build_query_phrases(claim=claim, terms=terms, phrases=phrases)
    return ClaimProfile(
        claim=claim,
        terms=terms,
        phrases=phrases,
        support_terms=support_terms,
        query_phrases=query_phrases,
        claim_kind=claim_kind,
        retrieval_intent=retrieval_intent,
        must_include_terms=must_include_terms,
        exclude_terms=exclude_terms,
        context_only_terms=context_only_terms,
        retrieval_terms=set(),
        required_concept_terms=set(),
        foundational_concept_terms=set(),
        excluded_concept_terms=set(),
        claim_focus="",
    )


def classify_claim_kind(claim_lowered: str, terms: set[str]) -> str:
    if re.search(r"\b(this paper|we|our|proposed|prototype|implementation)\b", claim_lowered):
        return "author_method"
    if re.search(r"\b(recent|emerging|modern|new|current)\b", claim_lowered) and re.search(
        r"\b(changed|reshaped|transformed|enabled|opened|advanced|reframed)\b", claim_lowered
    ):
        return "field_level_background"
    if re.search(r"\b(existing|prior|previous|recent|many|most|usually|generally|typically)\b", claim_lowered):
        return "related_work_summary"
    if re.search(r"\b(outperform|superior|better|worse|compared|comparison)\b", claim_lowered):
        return "comparison"
    if re.search(r"\b(psnr|ssim|lpips|ate|rpe|benchmark|dataset|metric)\b", claim_lowered):
        return "benchmark_or_metric"
    return "general_claim"


def retrieval_intent_for_claim(claim_kind: str, terms: set[str]) -> str:
    if claim_kind == "field_level_background":
        return "prefer_survey_or_foundational_background"
    if claim_kind == "related_work_summary":
        return "prefer_survey_or_multiple_systems"
    if claim_kind == "comparison":
        return "prefer_evaluation_or_benchmark_evidence"
    if claim_kind == "benchmark_or_metric":
        return "prefer_benchmark_or_metric_reference"
    if "propose" in terms or "present" in terms:
        return "prefer_original_method_paper"
    return "find_direct_evidence"


def derive_must_include_terms(terms: set[str], phrases: list[str]) -> set[str]:
    phrase_terms: set[str] = set()
    for phrase in phrases:
        phrase_terms.update(tokenize(phrase))

    scoped_phrase_terms = {
        term for term in phrase_terms & terms if is_high_signal_scope_term(term)
    }
    if scoped_phrase_terms:
        return set(sorted(scoped_phrase_terms, key=lambda term: (-len(term), term))[:MAX_SCOPE_TERMS])

    scoped_terms = {term for term in terms if is_high_signal_scope_term(term)}
    return set(sorted(scoped_terms, key=lambda term: (-len(term), term))[:MAX_SCOPE_TERMS])


def derive_context_only_terms(terms: set[str], context: str) -> set[str]:
    """Return high-signal terms that appear in surrounding context but not the selected claim.

    These terms are treated as possible topic contamination. For example, if a
    selected sentence discusses one method family while the paragraph context
    also introduces a different method family, those context-only anchors should
    not pull the citation search toward the wrong family.
    """

    claim_scope = {term for term in terms if is_high_signal_scope_term(term)}
    if not claim_scope or not context.strip():
        return set()

    context_only: set[str] = set()
    for sentence in split_context_sentences(context):
        sentence_scope = {term for term in tokenize(sentence) if is_high_signal_scope_term(term)}
        if not sentence_scope:
            continue
        overlap = claim_scope & sentence_scope
        overlap_ratio = len(overlap) / max(len(sentence_scope), 1)
        if len(overlap) >= 2 or overlap_ratio >= 0.3:
            continue
        context_only.update(sentence_scope - claim_scope)
    return set(sorted(context_only, key=lambda term: (-len(term), term))[:MAX_SCOPE_TERMS])


def derive_exclude_terms(terms: set[str], context: str) -> set[str]:
    """Derive generic scope-exclusion terms from context-only topical anchors."""

    if len({term for term in terms if is_high_signal_scope_term(term)}) < 2:
        return set()
    return derive_context_only_terms(terms, context)


def build_query_phrases(claim: str, terms: set[str], phrases: list[str]) -> list[str]:
    """Create short Zotero-friendly queries from field phrases and support terms."""

    queries: list[str] = []
    for phrase in phrases:
        queries.append(phrase)

    keyphrases = extract_claim_keyphrases(claim, terms, max_phrases=8)
    queries.extend(keyphrases)
    queries.extend(extract_scope_pair_queries(claim, terms, max_phrases=10))

    scoped_terms = ordered_scope_terms(claim, terms)
    if len(scoped_terms) >= 2:
        head = " ".join(scoped_terms[:2])
        queries.extend(
            [
                head,
                f"{head} survey",
                f"{head} review",
            ]
        )
    queries.extend(scoped_terms[:6])

    top_terms = sorted(terms, key=lambda term: (-len(term), term))[:5]
    if len(top_terms) >= 2:
        queries.append(" ".join(top_terms[:3]))

    cleaned_claim = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", claim)
    cleaned_claim = re.sub(r"\s+", " ", cleaned_claim).strip(" .;:")
    if cleaned_claim:
        queries.append(cleaned_claim)

    return dedupe_preserve_order(query for query in queries if query)


def extract_claim_keyphrases(claim: str, terms: set[str], max_phrases: int = 5) -> list[str]:
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", claim)
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", cleaned.lower())
    phrases: list[str] = []
    for size in (4, 3, 2):
        for index in range(0, max(len(words) - size + 1, 0)):
            chunk = words[index : index + size]
            scoped = [
                word.strip("-")
                for word in chunk
                if word.strip("-") in terms and is_high_signal_scope_term(word.strip("-"))
            ]
            if len(scoped) < 2:
                continue
            if not any(is_high_signal_scope_term(word) for word in scoped):
                continue
            phrases.append(" ".join(scoped))
            if len(phrases) >= max_phrases:
                return dedupe_preserve_order(phrases)
    return dedupe_preserve_order(phrases)


def extract_scope_pair_queries(claim: str, terms: set[str], max_phrases: int = 10) -> list[str]:
    scoped = ordered_scope_terms(claim, terms)
    if len(scoped) < 2:
        return []

    queries: list[str] = []
    for index, term in enumerate(scoped):
        for other in scoped[index + 1 : index + 4]:
            if term == other:
                continue
            queries.append(f"{term} {other}")
            if len(queries) >= max_phrases:
                return dedupe_preserve_order(queries)
    return dedupe_preserve_order(queries)


def ordered_scope_terms(text: str, terms: set[str]) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    ordered: list[str] = []
    seen: set[str] = set()
    for word in words:
        normalized = word.strip("-")
        candidates = [normalized]
        if "-" in normalized:
            candidates.extend(part for part in normalized.split("-") if part)
        for candidate in candidates:
            if candidate not in terms or candidate in seen:
                continue
            if not is_high_signal_scope_term(candidate):
                continue
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def dedupe_preserve_order(values: list[str] | tuple[str, ...] | object) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in values:
        value = str(raw).strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def split_context_sentences(context: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", context).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", normalized)
        if sentence.strip()
    ]


def score_text_against_claim(profile: ClaimProfile, text: str) -> float:
    """Return a lexical relevance score for metadata/evidence passages."""

    if not text.strip():
        return 0.0

    terms = tokenize(text)
    phrase_hits = sum(1 for phrase in profile.phrases if phrase in text.lower())
    term_hits = len(profile.terms & terms)
    support_hits = len(profile.support_terms & terms)
    must_hits = len(profile.must_include_terms & terms)
    excluded_hits = len(profile.exclude_terms & terms)
    retrieval_hits = len(profile.retrieval_terms & terms)
    required_hits = len(profile.required_concept_terms & terms)
    foundational_hits = len(profile.foundational_concept_terms & terms)
    excluded_concept_hits = len(profile.excluded_concept_terms & terms)

    denominator = max(len(profile.terms), 1)
    score = (term_hits / denominator) + phrase_hits * 0.35 + support_hits * 0.2
    if profile.must_include_terms:
        score += (must_hits / max(len(profile.must_include_terms), 1)) * 0.25
    if excluded_hits:
        score -= min(excluded_hits * 0.35, 0.9)
    if retrieval_hits:
        score += min(0.35, retrieval_hits / max(len(profile.retrieval_terms), 1) * 0.35)
    if required_hits:
        score += min(0.45, required_hits / max(len(profile.required_concept_terms), 1) * 0.45)
    if foundational_hits:
        score += min(0.25, foundational_hits / max(len(profile.foundational_concept_terms), 1) * 0.25)
    if excluded_concept_hits and not required_hits:
        score -= min(0.7, excluded_concept_hits * 0.25)
    return min(score, 1.5)


def is_high_signal_scope_term(term: str) -> bool:
    normalized = term.lower().strip("-")
    if len(normalized) < 4:
        return False
    if normalized in GENERIC_SCOPE_TERMS:
        return False
    if normalized.isdigit():
        return False
    return True


def identity_scope_terms(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text.lower()))
    expanded: set[str] = set()
    for token in tokens:
        cleaned = token.strip("-")
        if not cleaned:
            continue
        expanded.add(cleaned)
        if "-" in cleaned:
            expanded.update(part for part in cleaned.split("-") if part)
    return {term for term in expanded if is_high_signal_scope_term(term)}


def scope_conflict_reason(profile: ClaimProfile, identity_text: str) -> str | None:
    """Return a reason when a candidate identity is dominated by context-only terms."""

    identity_terms = identity_scope_terms(identity_text)
    if not profile.exclude_terms or not identity_terms:
        return None

    excluded_hits = profile.exclude_terms & identity_terms
    if not excluded_hits:
        return None

    must_hits = profile.must_include_terms & identity_terms
    claim_hits = profile.terms & identity_terms
    if must_hits:
        return None

    if len(excluded_hits) >= 2:
        return (
            "候选主题主要命中选中文段之外的上下文主题词："
            f"{', '.join(sorted(excluded_hits))}"
        )

    if len(claim_hits) <= 1 and profile.must_include_terms:
        return (
            "候选缺少选中文段的核心主题词，且命中了上下文主题词："
            f"{', '.join(sorted(excluded_hits))}"
        )

    return None
