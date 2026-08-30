"""Tests for V3 hybrid evidence passage ranking."""
from __future__ import annotations

import pathlib
import sys


def _setup_path():
    tool_dir = str(pathlib.Path(__file__).parents[3] / "bundled" / "tool")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


_setup_path()

from citation_agent.config import EmbeddingConfig  # noqa: E402
from citation_agent.embedding.embedder import BaseEmbedder, EmbeddingResult  # noqa: E402
from citation_agent.embedding.hybrid_ranker import rank_evidence_passages  # noqa: E402
from citation_agent.evidence import EvidenceRetriever, Passage  # noqa: E402
from citation_agent.claim_analysis import build_claim_profile  # noqa: E402
from citation_agent.schemas import CitationSuggestion, EvidenceSnippet  # noqa: E402


class FakeSemanticEmbedder(BaseEmbedder):
    provider = "fake-semantic"

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "offline global optimization" in lowered or "centralized pose graph" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return EmbeddingResult(vectors=vectors, provider=self.provider, model_id="fake")


def test_hybrid_ranker_can_promote_semantic_match_over_lexical_match():
    config = EmbeddingConfig(
        enabled=True,
        model_provider="hash",
        model_id="fake",
        lexical_weight=0.1,
        embedding_weight=0.9,
    )
    passages = [
        Passage(
            text="Gaussian SLAM multi-agent paper metadata with many shared keywords.",
            source="metadata",
            section="Metadata",
            score=2.0,
        ),
        Passage(
            text="The system relies on centralized pose graph refinement to optimize global consistency.",
            source="pdf",
            section="PDF",
            page=4,
            score=0.6,
        ),
    ]

    ranked = rank_evidence_passages(
        query="Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization.",
        passages=passages,
        config=config,
        embedder=FakeSemanticEmbedder(),
    )

    assert ranked[0].source == "pdf"
    assert "centralized pose graph" in ranked[0].text


def test_evidence_retriever_keeps_lexical_fallback_when_embedding_disabled():
    profile = build_claim_profile(
        "Existing multi-agent Gaussian SLAM methods generally rely on offline global optimization."
    )
    suggestion = CitationSuggestion(
        paper_id="bib:fallback",
        title="Fallback Gaussian SLAM",
        citation_key="fallback2026",
        confidence=0.5,
        source="current_bib",
        reason="fallback",
        evidence=EvidenceSnippet(
            text="Gaussian SLAM optimization appears in the abstract.",
            source="abstract",
            section="Abstract",
        ),
    )

    retriever = EvidenceRetriever(
        max_passages=2,
        embedding_config=EmbeddingConfig(enabled=False),
    )
    passages = retriever.retrieve(suggestion, profile, claim="Gaussian SLAM optimization.")

    assert [passage.source for passage in passages] == ["abstract", "metadata"]
