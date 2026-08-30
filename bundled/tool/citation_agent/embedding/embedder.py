"""Embedding adapters used by the V3 hybrid evidence ranker."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..cache import EmbeddingVectorCache
from ..config import EmbeddingConfig


class EmbeddingUnavailable(RuntimeError):
    """Raised when semantic embeddings are enabled but no local model is ready."""


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    provider: str
    model_id: str


class BaseEmbedder:
    """Minimal embedding interface shared by real and test embedders."""

    provider = "base"

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError


class DisabledEmbedder(BaseEmbedder):
    provider = "disabled"

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise EmbeddingUnavailable("embedding retrieval is disabled")


class HashingEmbedder(BaseEmbedder):
    """Deterministic lightweight embedder for tests and offline development.

    This is not a semantic model. It gives the ranker a stable vector interface
    before the real ModelScope/ONNX backend is wired in.
    """

    provider = "hash"

    def __init__(self, dimensions: int = 128):
        self.dimensions = max(dimensions, 8)

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors = [self._embed_one(text) for text in texts]
        return EmbeddingResult(vectors=vectors, provider=self.provider, model_id="hashing-dev")

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class FastEmbedder(BaseEmbedder):
    """CPU ONNX embedding backend powered by fastembed."""

    provider = "fastembed"

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.cache_dir = config.cache_dir or default_embedding_cache_dir()
        self.model_id = config.model_id
        self._model = None
        self._vector_cache = EmbeddingVectorCache(self.cache_dir)

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        documents = [str(text or "") for text in texts]
        cached = self._vector_cache.get_many(self.model_id, documents)
        vectors: list[list[float] | None] = [cached.get(index) for index in range(len(documents))]
        missing_indexes = [
            index
            for index, vector in enumerate(vectors)
            if vector is None
        ]
        if not missing_indexes:
            return EmbeddingResult(
                vectors=[vector for vector in vectors if vector is not None],
                provider=self.provider,
                model_id=self.model_id,
            )

        model = self._ensure_model()
        missing_documents = [documents[index] for index in missing_indexes]
        missing_vectors = [
            [float(value) for value in vector]
            for vector in model.embed(missing_documents)
        ]
        self._vector_cache.put_many(self.model_id, missing_documents, missing_vectors)
        for index, vector in zip(missing_indexes, missing_vectors, strict=False):
            vectors[index] = vector

        return EmbeddingResult(
            vectors=[vector if vector is not None else [] for vector in vectors],
            provider=self.provider,
            model_id=self.model_id,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingUnavailable(
                "fastembed is not installed in the EasyCite virtual environment"
            ) from exc

        kwargs: dict[str, object] = {
            "model_name": self.model_id,
            "cache_dir": self.cache_dir,
        }
        if self.config.auto_download != "always":
            kwargs["local_files_only"] = True

        try:
            self._model = TextEmbedding(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            raise EmbeddingUnavailable(
                f"fastembed model is unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        return self._model


def default_embedding_cache_dir() -> str:
    return str(Path.home() / ".cache" / "citation-agent" / "embeddings")


def create_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    if not config.enabled:
        return DisabledEmbedder()

    provider = config.model_provider.strip().lower()
    if provider == "hash":
        return HashingEmbedder()
    if provider == "fastembed":
        return FastEmbedder(config)

    raise EmbeddingUnavailable(
        f"unsupported embedding provider {config.model_provider!r}; "
        "use 'fastembed' for the current V3 ONNX backend"
    )
