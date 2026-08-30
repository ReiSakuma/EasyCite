"""Hybrid lexical/embedding passage ranking for evidence retrieval."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass, replace
from typing import Any

from ..config import EmbeddingConfig
from ..tools.adapters import execute_embedding
from .embedder import BaseEmbedder, EmbeddingUnavailable, create_embedder
from .model_manager import EmbeddingModelManager


def rank_evidence_passages(
    query: str,
    passages: Sequence[Any],
    config: EmbeddingConfig,
    logger: Callable[[str], None] | None = None,
    embedder: BaseEmbedder | None = None,
) -> list[Any]:
    """Rank passages using a deterministic lexical fallback and optional embeddings."""
    lexical_ranked = sorted(passages, key=_passage_sort_key)
    if not config.enabled:
        return lexical_ranked

    status = EmbeddingModelManager(config).status()
    if logger:
        logger(
            "Embedding ranker status: "
            f"enabled={status.enabled}, available={status.available}, "
            f"provider={status.provider}, model={status.model_id}, reason={status.reason}"
        )
    if not status.available and embedder is None:
        return lexical_ranked

    try:
        active_embedder = embedder or create_embedder(config)
        texts = [query] + [_passage_text(passage) for passage in lexical_ranked]
        result = execute_embedding(texts, lambda: active_embedder.embed(texts))
        query_vector = result.vectors[0]
        passage_vectors = result.vectors[1:]
        embedding_scores = [
            cosine_similarity(query_vector, passage_vector)
            for passage_vector in passage_vectors
        ]
    except EmbeddingUnavailable as exc:
        if logger:
            logger(f"Embedding ranker skipped: {exc}")
        return lexical_ranked
    except Exception as exc:  # pylint: disable=broad-except
        if logger:
            logger(f"Embedding ranker failed; lexical fallback used: {type(exc).__name__}: {exc}")
        return lexical_ranked

    lexical_scores = [float(getattr(passage, "score", 0.0) or 0.0) for passage in lexical_ranked]
    normalized_lexical = min_max_normalize(lexical_scores)
    normalized_embedding = min_max_normalize(embedding_scores)
    lexical_weight, embedding_weight = normalized_weights(
        config.lexical_weight,
        config.embedding_weight,
    )

    has_content_passage = any(
        str(getattr(passage, "source", "")) in {"abstract", "pdf"}
        for passage in lexical_ranked
    )
    rescored: list[Any] = []
    for passage, lexical_score, embedding_score in zip(
        lexical_ranked,
        normalized_lexical,
        normalized_embedding,
        strict=False,
    ):
        source = str(getattr(passage, "source", ""))
        source_bonus = {
            "abstract": 0.03,
            "pdf": 0.02,
            "metadata": 0.0,
        }.get(source, 0.0)
        hybrid_score = lexical_weight * lexical_score + embedding_weight * embedding_score + source_bonus
        if has_content_passage and source == "metadata":
            hybrid_score = min(hybrid_score, -0.01)
        rescored.append(_replace_score(passage, round(hybrid_score, 6)))

    ranked = sorted(rescored, key=_passage_sort_key)
    if logger:
        logger(
            "Embedding ranker applied: "
            f"provider={getattr(active_embedder, 'provider', '<unknown>')}, "
            f"passages={len(ranked)}, top_k={config.top_k}."
        )
    return ranked[: max(config.top_k, 1)]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second, strict=False))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot / (first_norm * second_norm)


def min_max_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def normalized_weights(lexical_weight: float, embedding_weight: float) -> tuple[float, float]:
    lexical = max(float(lexical_weight), 0.0)
    embedding = max(float(embedding_weight), 0.0)
    total = lexical + embedding
    if total <= 0:
        return 0.35, 0.65
    return lexical / total, embedding / total


def _replace_score(passage: Any, score: float) -> Any:
    if is_dataclass(passage):
        return replace(passage, score=score)
    try:
        passage.score = score
    except Exception:  # pylint: disable=broad-except
        return passage
    return passage


def _passage_text(passage: Any) -> str:
    return re.sub(r"\s+", " ", str(getattr(passage, "text", "") or "")).strip()


def _passage_sort_key(passage: Any) -> tuple[float, int, int, str, str]:
    score = float(getattr(passage, "score", 0.0) or 0.0)
    source_priority = {
        "abstract": 0,
        "pdf": 1,
        "metadata": 2,
        "unknown": 3,
    }.get(str(getattr(passage, "source", "")), 4)
    page = getattr(passage, "page", None)
    section = str(getattr(passage, "section", "") or "")
    text = _passage_text(passage).lower()
    return (-round(score, 6), source_priority, page if page is not None else 0, section.lower(), text)
