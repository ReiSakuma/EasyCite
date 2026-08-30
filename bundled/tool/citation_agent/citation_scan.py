"""Semantic ranking for background citation-needed scan diagnostics."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .config import EmbeddingConfig
from .embedding.embedder import EmbeddingUnavailable, create_embedder
from .embedding.hybrid_ranker import cosine_similarity
from .embedding.model_manager import EmbeddingModelManager
from .tools.adapters import execute_embedding


CITATION_NEED_PROTOTYPES = [
    "Existing methods generally rely on offline optimization and should be cited.",
    "Prior work has shown that this limitation is common across recent approaches.",
    "Several recent methods address this problem using different assumptions.",
    "A common limitation of previous approaches is poor scalability.",
    "This sentence summarizes the state of the art and needs references.",
    "This claim compares existing methods and should be supported by citations.",
    "已有研究表明该问题在现有方法中普遍存在，需要引用支持。",
    "现有方法通常依赖某种假设或技术路线，需要添加参考文献。",
    "这句话总结相关工作或已有方法，应当由文献支撑。",
]


def rank_citation_scan_sentences(
    params: dict[str, Any],
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rank citation-scan candidate sentences against citation-needed prototypes."""
    config = EmbeddingConfig.from_env()
    sentences = normalize_scan_sentences(params.get("sentences"))
    if not sentences:
        return {
            "available": False,
            "reason": "no scan sentences were provided",
            "provider": None,
            "model_id": None,
            "scores": [],
        }

    if not config.enabled:
        return {
            "available": False,
            "reason": "embedding retrieval is disabled",
            "provider": None,
            "model_id": None,
            "scores": [],
        }

    status = EmbeddingModelManager(config).status()
    if logger:
        logger(
            "Citation scan embedding status: "
            f"enabled={status.enabled}, available={status.available}, "
            f"provider={status.provider}, model={status.model_id}, reason={status.reason}"
        )
    if not status.available:
        return {
            "available": False,
            "reason": status.reason,
            "provider": status.provider,
            "model_id": status.model_id,
            "scores": [],
        }

    try:
        embedder = create_embedder(config)
        texts = CITATION_NEED_PROTOTYPES + [item["text"] for item in sentences]
        result = execute_embedding(texts, lambda: embedder.embed(texts))
        prototype_vectors = result.vectors[: len(CITATION_NEED_PROTOTYPES)]
        sentence_vectors = result.vectors[len(CITATION_NEED_PROTOTYPES):]
    except EmbeddingUnavailable as exc:
        if logger:
            logger(f"Citation scan embedding skipped: {exc}")
        return {
            "available": False,
            "reason": str(exc),
            "provider": status.provider,
            "model_id": status.model_id,
            "scores": [],
        }
    except Exception as exc:  # pylint: disable=broad-except
        if logger:
            logger(f"Citation scan embedding failed: {type(exc).__name__}: {exc}")
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "provider": status.provider,
            "model_id": status.model_id,
            "scores": [],
        }

    scores = []
    for item, vector in zip(sentences, sentence_vectors, strict=False):
        similarities = [
            cosine_similarity(vector, prototype_vector)
            for prototype_vector in prototype_vectors
        ]
        best_index, best_score = max(enumerate(similarities), key=lambda pair: pair[1])
        scores.append(
            {
                "id": item["id"],
                "score": round(float(best_score), 6),
                "matched_prototype": CITATION_NEED_PROTOTYPES[best_index],
            }
        )

    return {
        "available": True,
        "reason": "embedding semantic scan applied",
        "provider": result.provider,
        "model_id": result.model_id,
        "scores": scores,
    }


def normalize_scan_sentences(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []

    sentences: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        text = clean_text(item.get("text"))
        if not text:
            continue
        sentence_id = clean_text(item.get("id")) or str(index)
        if sentence_id in seen_ids:
            continue
        seen_ids.add(sentence_id)
        sentences.append(
            {
                "id": sentence_id,
                "text": text,
            }
        )
    return sentences


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())
