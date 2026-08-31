from __future__ import annotations

from citation_agent.claim_analysis import build_claim_profile
from citation_agent.config import EmbeddingConfig, RagConfig
from citation_agent.rag import RagDocument, RagRetriever
from citation_agent.rag.chunking import TextUnit, build_parent_child_chunks
from citation_agent.rag.retriever import reciprocal_rank_fusion
from citation_agent.recommender import CitationRecommender
from citation_agent.config import SourceConfig


def test_parent_child_chunking_preserves_section_and_overlap() -> None:
    document = RagDocument(document_id="paper:1", source="zotero", title="Example Paper")
    units = [
        TextUnit(
            text=(
                "The proposed optimizer builds a shared geometric map. "
                "It jointly refines camera poses and scene parameters. "
                "This repeated optimization improves consistency across agents. "
                "Experiments compare the method with independent local mapping."
            ),
            page=3,
            section="Method",
        )
    ]
    config = RagConfig(
        child_chunk_tokens=20,
        child_chunk_max_tokens=35,
        chunk_overlap_tokens=8,
        parent_chunk_tokens=80,
    )

    chunks = build_parent_child_chunks(document, units, config)

    assert len(chunks) >= 2
    assert all(chunk.section == "Method" for chunk in chunks)
    assert all(chunk.page_start == 3 for chunk in chunks)
    assert len({chunk.parent_id for chunk in chunks}) == 1


def test_rrf_combines_lexical_and_dense_rankings() -> None:
    fused = reciprocal_rank_fusion(
        [(["a", "b", "c"], 1.0), (["b", "d", "a"], 1.2)],
        rrf_k=60,
    )

    assert fused[0][0] == "b"
    assert {identity for identity, _score in fused} == {"a", "b", "c", "d"}


def test_rag_retrieves_evidence_and_reuses_incremental_index(tmp_path) -> None:
    claim = "The method jointly optimizes camera poses and a shared scene representation."
    profile = build_claim_profile(claim)
    documents = [
        RagDocument(
            document_id="paper:joint",
            source="zotero",
            title="Joint Pose and Scene Optimization",
            abstract=(
                "We jointly optimize camera poses and a shared scene representation. "
                "The coupled objective improves geometric consistency across multiple views."
            ),
            citation_key="JointOptimization2025",
        ),
        RagDocument(
            document_id="paper:unrelated",
            source="zotero",
            title="Marine Temperature Forecasting",
            abstract=(
                "This study forecasts ocean temperature from seasonal observations. "
                "The model evaluates climate trends in coastal regions."
            ),
            citation_key="MarineForecast2025",
        ),
    ]
    rag_config = RagConfig(
        index_path=str(tmp_path / "rag.sqlite3"),
        reranker_enabled=False,
        paper_fused_top_k=5,
        passage_fused_top_k=10,
    )
    embedding_config = EmbeddingConfig(enabled=True, model_provider="hash")
    retriever = RagRetriever(
        workspace_path=str(tmp_path),
        config=rag_config,
        embedding_config=embedding_config,
    )

    first = retriever.retrieve(
        claim=claim,
        queries=["joint pose scene optimization"],
        profile=profile,
        documents=documents,
    )
    second = retriever.retrieve(
        claim=claim,
        queries=["joint pose scene optimization"],
        profile=profile,
        documents=documents,
    )

    assert first.suggestions
    assert first.suggestions[0].paper_id == "paper:joint"
    assert first.suggestions[0].evidence_passages
    assert first.stats.embedding_used is True
    assert second.stats.documents_indexed == 0
    assert second.stats.documents_reused == 2


def test_rag_hydrates_only_bounded_paper_finalists(tmp_path) -> None:
    claim = "Joint optimization improves shared map consistency."
    documents = [
        RagDocument(
            document_id=f"paper:{index}",
            source="zotero",
            title=f"Joint Optimization Study {index}",
            abstract="Joint optimization improves shared map consistency across multiple views.",
        )
        for index in range(5)
    ]
    requested: list[str] = []

    def hydrate(selected: list[RagDocument]) -> list[RagDocument]:
        requested.extend(document.document_id for document in selected)
        return [
            RagDocument.from_dict({**document.to_dict(), "pdf_path": f"/missing/{document.document_id}.pdf"})
            for document in selected
        ]

    retriever = RagRetriever(
        workspace_path=str(tmp_path),
        config=RagConfig(
            index_path=str(tmp_path / "lazy.sqlite3"),
            pdf_index_top_k=2,
            reranker_enabled=False,
        ),
        embedding_config=EmbeddingConfig(enabled=True, model_provider="hash"),
    )
    result = retriever.retrieve(
        claim=claim,
        queries=["joint optimization map consistency"],
        profile=build_claim_profile(claim),
        documents=documents,
        document_hydrator=hydrate,
    )

    assert len(requested) == 2
    assert result.stats.pdf_documents_requested == 2
    assert result.stats.pdf_documents_hydrated == 2
    assert result.stats.total_ms >= result.stats.paper_retrieval_ms


def test_langgraph_runs_rag_stages_for_project_bibliography(tmp_path, monkeypatch) -> None:
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text("\\addbibresource{references.bib}\n", encoding="utf-8")
    (tmp_path / "references.bib").write_text(
        """
@article{joint2025,
  title={Joint Pose and Shared Scene Optimization},
  author={A. Researcher},
  year={2025},
  abstract={We jointly optimize camera poses and a shared scene representation to improve geometric consistency.}
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CITATION_AGENT_RAG_ENABLED", "true")
    monkeypatch.setenv("CITATION_AGENT_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("CITATION_AGENT_EMBEDDING_MODEL_PROVIDER", "hash")
    monkeypatch.setenv("CITATION_AGENT_RAG_RERANKER_ENABLED", "false")

    bundle = CitationRecommender(
        SourceConfig(
            max_suggestions=5,
            zotero_enabled=False,
            zotero_endpoint=None,
            enable_evidence_verification=False,
        )
    ).recommend_with_diagnostics(
        {
            "workspace_path": str(tmp_path),
            "file_path": str(tex_path),
            "claim": "The method jointly optimizes camera poses and a shared scene representation.",
        }
    )

    stages = {event.stage for event in bundle.trace_events}
    assert "rag_index_sync" in stages
    assert "rag_paper_hybrid_retrieval" in stages
    assert "rag_passage_hybrid_retrieval" in stages
    assert any(suggestion.citation_key == "joint2025" for suggestion in bundle.suggestions)
