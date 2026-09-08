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


@dataclass(frozen=True)
class RagConfig:
    """Runtime configuration for the persistent local academic RAG pipeline."""

    enabled: bool = True
    index_path: str | None = None
    max_documents: int = 500
    child_chunk_tokens: int = 280
    child_chunk_max_tokens: int = 420
    chunk_overlap_tokens: int = 48
    parent_chunk_tokens: int = 900
    semantic_chunking_enabled: bool = True
    semantic_boundary_threshold: float = 0.45
    semantic_hard_boundary_threshold: float = 0.30
    semantic_min_parent_tokens: int = 180
    semantic_adaptive_enabled: bool = True
    semantic_adaptive_percentile: float = 0.20
    semantic_adaptive_max_threshold: float = 0.72
    semantic_min_score_drop: float = 0.08
    semantic_context_window: int = 2
    semantic_embedding_min_tokens: int = 24
    embedding_batch_size: int = 64
    paper_lexical_top_k: int = 50
    paper_dense_top_k: int = 50
    paper_fused_top_k: int = 30
    passage_lexical_top_k: int = 60
    passage_dense_top_k: int = 60
    passage_fused_top_k: int = 50
    pdf_index_top_k: int = 12
    reranker_enabled: bool = True
    reranker_model_id: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_input_k: int = 50
    reranker_top_k: int = 20
    max_passages_per_paper: int = 3
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.2
    chunker_version: str = "v22.semantic-parent-child.2"

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            enabled=_truthy(os.getenv("CITATION_AGENT_RAG_ENABLED", "true")),
            index_path=os.getenv("CITATION_AGENT_RAG_INDEX_PATH") or None,
            max_documents=int(os.getenv("CITATION_AGENT_RAG_MAX_DOCUMENTS", "500")),
            child_chunk_tokens=int(os.getenv("CITATION_AGENT_RAG_CHILD_CHUNK_TOKENS", "280")),
            child_chunk_max_tokens=int(
                os.getenv("CITATION_AGENT_RAG_CHILD_CHUNK_MAX_TOKENS", "420")
            ),
            chunk_overlap_tokens=int(
                os.getenv("CITATION_AGENT_RAG_CHUNK_OVERLAP_TOKENS", "48")
            ),
            parent_chunk_tokens=int(os.getenv("CITATION_AGENT_RAG_PARENT_CHUNK_TOKENS", "900")),
            semantic_chunking_enabled=_truthy(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_CHUNKING_ENABLED", "true")
            ),
            semantic_boundary_threshold=float(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_BOUNDARY_THRESHOLD", "0.45")
            ),
            semantic_hard_boundary_threshold=float(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_HARD_BOUNDARY_THRESHOLD", "0.30")
            ),
            semantic_min_parent_tokens=int(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_MIN_PARENT_TOKENS", "180")
            ),
            semantic_adaptive_enabled=_truthy(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_ADAPTIVE_ENABLED", "true")
            ),
            semantic_adaptive_percentile=float(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_ADAPTIVE_PERCENTILE", "0.20")
            ),
            semantic_adaptive_max_threshold=float(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_ADAPTIVE_MAX_THRESHOLD", "0.72")
            ),
            semantic_min_score_drop=float(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_MIN_SCORE_DROP", "0.08")
            ),
            semantic_context_window=int(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_CONTEXT_WINDOW", "2")
            ),
            semantic_embedding_min_tokens=int(
                os.getenv("CITATION_AGENT_RAG_SEMANTIC_EMBEDDING_MIN_TOKENS", "24")
            ),
            embedding_batch_size=int(
                os.getenv("CITATION_AGENT_RAG_EMBEDDING_BATCH_SIZE", "64")
            ),
            paper_lexical_top_k=int(
                os.getenv("CITATION_AGENT_RAG_PAPER_LEXICAL_TOP_K", "50")
            ),
            paper_dense_top_k=int(
                os.getenv("CITATION_AGENT_RAG_PAPER_DENSE_TOP_K", "50")
            ),
            paper_fused_top_k=int(
                os.getenv("CITATION_AGENT_RAG_PAPER_FUSED_TOP_K", "30")
            ),
            passage_lexical_top_k=int(
                os.getenv("CITATION_AGENT_RAG_PASSAGE_LEXICAL_TOP_K", "60")
            ),
            passage_dense_top_k=int(
                os.getenv("CITATION_AGENT_RAG_PASSAGE_DENSE_TOP_K", "60")
            ),
            passage_fused_top_k=int(
                os.getenv("CITATION_AGENT_RAG_PASSAGE_FUSED_TOP_K", "50")
            ),
            pdf_index_top_k=int(os.getenv("CITATION_AGENT_RAG_PDF_INDEX_TOP_K", "12")),
            reranker_enabled=_truthy(
                os.getenv("CITATION_AGENT_RAG_RERANKER_ENABLED", "true")
            ),
            reranker_model_id=os.getenv(
                "CITATION_AGENT_RAG_RERANKER_MODEL_ID",
                "Xenova/ms-marco-MiniLM-L-6-v2",
            ),
            reranker_input_k=int(os.getenv("CITATION_AGENT_RAG_RERANKER_INPUT_K", "50")),
            reranker_top_k=int(os.getenv("CITATION_AGENT_RAG_RERANKER_TOP_K", "20")),
            max_passages_per_paper=int(
                os.getenv("CITATION_AGENT_RAG_MAX_PASSAGES_PER_PAPER", "3")
            ),
            rrf_k=int(os.getenv("CITATION_AGENT_RAG_RRF_K", "60")),
            bm25_weight=float(os.getenv("CITATION_AGENT_RAG_BM25_WEIGHT", "1.0")),
            dense_weight=float(os.getenv("CITATION_AGENT_RAG_DENSE_WEIGHT", "1.2")),
            chunker_version=os.getenv(
                "CITATION_AGENT_RAG_CHUNKER_VERSION", "v22.semantic-parent-child.2"
            ),
        )
