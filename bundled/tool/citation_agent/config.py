"""Runtime configuration for EasyCite."""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelConfig:
    use_llm: bool
    provider: str
    model: str
    api_key: str | None
    api_base: str | None
    timeout_seconds: float = 20.0
    translation_timeout_seconds: float = 60.0
    evidence_timeout_seconds: float = 60.0
    evidence_max_llm_candidates: int = 8
    context_query_budget_tokens: int = 3200
    context_evidence_budget_tokens: int = 9000
    context_translation_budget_tokens: int = 5000

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            use_llm=_truthy(os.getenv("CITATION_AGENT_USE_LLM")),
            provider=os.getenv("CITATION_AGENT_PROVIDER", "openai"),
            model=os.getenv("CITATION_AGENT_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("CITATION_AGENT_API_KEY") or None,
            api_base=os.getenv("CITATION_AGENT_API_BASE") or None,
            timeout_seconds=float(os.getenv("CITATION_AGENT_TIMEOUT_SECONDS", "20")),
            translation_timeout_seconds=float(
                os.getenv("CITATION_AGENT_TRANSLATION_TIMEOUT_SECONDS", "60")
            ),
            evidence_timeout_seconds=float(
                os.getenv("CITATION_AGENT_EVIDENCE_TIMEOUT_SECONDS", "60")
            ),
            evidence_max_llm_candidates=int(
                os.getenv("CITATION_AGENT_EVIDENCE_MAX_LLM_CANDIDATES", "8")
            ),
            context_query_budget_tokens=int(
                os.getenv("CITATION_AGENT_CONTEXT_QUERY_BUDGET_TOKENS", "3200")
            ),
            context_evidence_budget_tokens=int(
                os.getenv("CITATION_AGENT_CONTEXT_EVIDENCE_BUDGET_TOKENS", "9000")
            ),
            context_translation_budget_tokens=int(
                os.getenv("CITATION_AGENT_CONTEXT_TRANSLATION_BUDGET_TOKENS", "5000")
            ),
        )


@dataclass(frozen=True)
class SourceConfig:
    """Runtime configuration for bibliography and Zotero candidate sources."""

    max_suggestions: int
    zotero_enabled: bool
    zotero_endpoint: str | None
    zotero_timeout_seconds: float = 4.0
    max_metadata_candidates: int = 50
    max_abstract_candidates: int = 8
    max_evidence_passages: int = 5
    enable_evidence_verification: bool = True
    zotero_collection_keys: list[str] = field(default_factory=list)
    debug_target_papers: list[str] = field(default_factory=list)
    agent_runtime: str = "langgraph"
    memory_enabled: bool = True
    memory_path: str | None = None
    memory_positive_boost: float = 0.08
    memory_negative_penalty: float = 0.18
    memory_min_similarity: float = 0.22

    @classmethod
    def from_env(cls) -> "SourceConfig":
        return cls(
            max_suggestions=int(os.getenv("CITATION_AGENT_MAX_SUGGESTIONS", "5")),
            zotero_enabled=_truthy(os.getenv("CITATION_AGENT_ZOTERO_ENABLED")),
            zotero_endpoint=os.getenv("CITATION_AGENT_ZOTERO_ENDPOINT") or None,
            zotero_timeout_seconds=float(
                os.getenv("CITATION_AGENT_ZOTERO_TIMEOUT_SECONDS", "4")
            ),
            max_metadata_candidates=int(
                os.getenv("CITATION_AGENT_MAX_METADATA_CANDIDATES", "50")
            ),
            max_abstract_candidates=int(
                os.getenv("CITATION_AGENT_MAX_ABSTRACT_CANDIDATES", "8")
            ),
            max_evidence_passages=int(
                os.getenv("CITATION_AGENT_MAX_EVIDENCE_PASSAGES", "5")
            ),
            enable_evidence_verification=_truthy(
                os.getenv("CITATION_AGENT_ENABLE_EVIDENCE_VERIFICATION", "true")
            ),
            zotero_collection_keys=parse_string_list(
                os.getenv("CITATION_AGENT_ZOTERO_COLLECTION_KEYS")
            ),
            debug_target_papers=parse_debug_targets(
                os.getenv("CITATION_AGENT_DEBUG_TARGET_PAPERS")
            ),
            agent_runtime=(
                os.getenv("CITATION_AGENT_AGENT_RUNTIME", "langgraph").strip().lower()
                or "langgraph"
            ),
            memory_enabled=_truthy(os.getenv("CITATION_AGENT_MEMORY_ENABLED", "true")),
            memory_path=os.getenv("CITATION_AGENT_MEMORY_PATH") or None,
            memory_positive_boost=float(
                os.getenv("CITATION_AGENT_MEMORY_POSITIVE_BOOST", "0.08")
            ),
            memory_negative_penalty=float(
                os.getenv("CITATION_AGENT_MEMORY_NEGATIVE_PENALTY", "0.18")
            ),
            memory_min_similarity=float(
                os.getenv("CITATION_AGENT_MEMORY_MIN_SIMILARITY", "0.22")
            ),
        )


def parse_string_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [
        item.strip()
        for item in str(parsed).split(",")
        if item.strip()
    ]


def parse_debug_targets(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [
        item.strip()
        for item in str(parsed).split(",")
        if item.strip()
    ]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Runtime configuration for local semantic evidence retrieval."""

    enabled: bool
    model_provider: str = "fastembed"
    model_id: str = "BAAI/bge-small-en-v1.5"
    cache_dir: str | None = None
    auto_download: str = "ask"
    top_k: int = 12
    lexical_weight: float = 0.35
    embedding_weight: float = 0.65

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            enabled=_truthy(os.getenv("CITATION_AGENT_EMBEDDING_ENABLED")),
            model_provider=os.getenv("CITATION_AGENT_EMBEDDING_MODEL_PROVIDER", "fastembed"),
            model_id=os.getenv("CITATION_AGENT_EMBEDDING_MODEL_ID", "BAAI/bge-small-en-v1.5"),
            cache_dir=os.getenv("CITATION_AGENT_EMBEDDING_CACHE_DIR") or None,
            auto_download=os.getenv("CITATION_AGENT_EMBEDDING_AUTO_DOWNLOAD", "ask"),
            top_k=int(os.getenv("CITATION_AGENT_EMBEDDING_TOP_K", "12")),
            lexical_weight=float(os.getenv("CITATION_AGENT_HYBRID_LEXICAL_WEIGHT", "0.35")),
            embedding_weight=float(os.getenv("CITATION_AGENT_HYBRID_EMBEDDING_WEIGHT", "0.65")),
        )
