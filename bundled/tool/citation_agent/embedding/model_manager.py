"""Model availability checks for local semantic retrieval."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from ..config import EmbeddingConfig


@dataclass(frozen=True)
class EmbeddingModelStatus:
    enabled: bool
    available: bool
    provider: str
    model_id: str
    cache_dir: str | None
    reason: str


class EmbeddingModelManager:
    """Small status facade for the future ModelScope/ONNX implementation."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def status(self) -> EmbeddingModelStatus:
        provider = self.config.model_provider.strip().lower()
        if not self.config.enabled:
            return EmbeddingModelStatus(
                enabled=False,
                available=False,
                provider=provider,
                model_id=self.config.model_id,
                cache_dir=self.config.cache_dir,
                reason="embedding retrieval is disabled",
            )

        if provider == "hash":
            return EmbeddingModelStatus(
                enabled=True,
                available=True,
                provider=provider,
                model_id="hashing-dev",
                cache_dir=None,
                reason="deterministic development embedder is available",
            )

        cache_dir = self.config.cache_dir or str(
            Path.home() / ".cache" / "citation-agent" / "embeddings"
        )
        if provider == "fastembed":
            if importlib.util.find_spec("fastembed") is None:
                return EmbeddingModelStatus(
                    enabled=True,
                    available=False,
                    provider=provider,
                    model_id=self.config.model_id,
                    cache_dir=cache_dir,
                    reason=(
                        "fastembed is not installed; install requirements-embedding.txt "
                        "inside the v3 virtual environment"
                    ),
                )
            return EmbeddingModelStatus(
                enabled=True,
                available=True,
                provider=provider,
                model_id=self.config.model_id,
                cache_dir=cache_dir,
                reason="fastembed ONNX backend is available",
            )

        return EmbeddingModelStatus(
            enabled=True,
            available=False,
            provider=provider,
            model_id=self.config.model_id,
            cache_dir=cache_dir,
            reason=(
                f"unsupported embedding provider {provider!r}; "
                "current V3 build supports fastembed and hash"
            ),
        )
