from __future__ import annotations

import sqlite3

from citation_agent.claim_analysis import build_claim_profile
from citation_agent.config import EmbeddingConfig, RagConfig
from citation_agent.rag import RagDocument, RagRetriever
from citation_agent.rag.chunking import (
    TextUnit,
    build_parent_child_chunks,
    build_parent_child_chunks_with_stats,
    extract_page_paragraphs,
)
from citation_agent.rag.index import RagIndex
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


def test_semantic_chunking_splits_topic_change_between_paragraphs() -> None:
    document = RagDocument(document_id="paper:semantic", source="zotero", title="Mixed Topics")
    units = [
        TextUnit(
            "The camera optimizer jointly refines poses and scene geometry across multiple views.",
            2,
            "Method",
            paragraph_id=10,
        ),
        TextUnit(
            "Pose refinement improves geometric consistency for the reconstructed shared map.",
            2,
            "Method",
            paragraph_id=11,
        ),
        TextUnit(
            "Ocean temperature forecasting instead models seasonal climate observations near coasts.",
            3,
            "Method",
            paragraph_id=12,
        ),
        TextUnit(
            "Marine measurements reveal annual trends in water temperature and salinity.",
            3,
            "Method",
            paragraph_id=13,
        ),
    ]

    def embed_paragraphs(_texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [0.98, 0.05], [0.0, 1.0], [0.05, 0.98]]

    result = build_parent_child_chunks_with_stats(
        document,
        units,
        RagConfig(
            parent_chunk_tokens=900,
            child_chunk_tokens=280,
            child_chunk_max_tokens=420,
            semantic_min_parent_tokens=0,
            semantic_embedding_min_tokens=0,
        ),
        paragraph_embedder=embed_paragraphs,
    )

    assert result.stats.semantic_used is True
    assert result.stats.semantic_boundaries == 1
    assert result.stats.parents == 2
    assert len({chunk.parent_id for chunk in result.chunks}) == 2
    assert {chunk.boundary_reason for chunk in result.chunks} == {"section", "semantic_hard"}
    assert all(
        not ("camera optimizer" in chunk.parent_text.lower() and "ocean temperature" in chunk.parent_text.lower())
        for chunk in result.chunks
    )


def test_pdf_layout_reconstruction_merges_lines_but_preserves_paragraphs_and_columns() -> None:
    class Rect:
        width = 600
        height = 800

    class Page:
        rect = Rect()

        @staticmethod
        def get_text(mode: str, sort: bool = False):
            del sort
            if mode == "blocks":
                return [
                    (40, 4, 250, 12, "Running header", 0, 0),
                    (40, 100, 280, 110, "The first logical paragraph starts on this line", 1, 0),
                    (40, 112, 275, 122, "and continues with the same scientific argument.", 2, 0),
                    (54, 125, 278, 135, "A newly indented paragraph introduces another claim", 3, 0),
                    (40, 137, 276, 147, "and its wrapped line returns to the column margin.", 4, 0),
                    (330, 100, 565, 110, "The right column must remain a separate paragraph.", 5, 0),
                    (290, 785, 310, 794, "12", 6, 0),
                ]
            return "fallback"

    paragraphs = extract_page_paragraphs(Page())

    assert paragraphs == [
        "The first logical paragraph starts on this line and continues with the same scientific argument.",
        "A newly indented paragraph introduces another claim and its wrapped line returns to the column margin.",
        "The right column must remain a separate paragraph.",
    ]


def test_adaptive_semantic_boundary_detects_relative_topic_valley() -> None:
    document = RagDocument(document_id="paper:adaptive", source="zotero", title="Adaptive")
    units = [
        TextUnit(f"Paragraph {index} develops one coherent research topic with enough detail.", 1, "Method", index)
        for index in range(6)
    ]

    def embed_paragraphs(_texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0], [0.995, 0.05], [0.99, 0.08],
            [0.65, 0.76], [0.62, 0.78], [0.60, 0.80],
        ]

    result = build_parent_child_chunks_with_stats(
        document,
        units,
        RagConfig(
            semantic_min_parent_tokens=0,
            semantic_boundary_threshold=0.45,
            semantic_hard_boundary_threshold=0.30,
            semantic_adaptive_percentile=0.2,
            semantic_adaptive_max_threshold=0.72,
            semantic_min_score_drop=0.08,
            semantic_embedding_min_tokens=0,
        ),
        paragraph_embedder=embed_paragraphs,
    )

    assert result.stats.adaptive_boundaries == 1
    assert result.stats.semantic_boundaries == 1
    assert result.stats.parents == 2
    assert "semantic_adaptive" in {chunk.boundary_reason for chunk in result.chunks}


def test_adaptive_semantic_boundary_ignores_small_similarity_wobbles() -> None:
    document = RagDocument(document_id="paper:coherent", source="zotero", title="Coherent")
    units = [TextUnit(f"Coherent method paragraph {index}.", 1, "Method", index) for index in range(5)]

    def embed_paragraphs(_texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.01 * index] for index in range(5)]

    result = build_parent_child_chunks_with_stats(
        document,
        units,
        RagConfig(semantic_min_parent_tokens=0, semantic_embedding_min_tokens=0),
        paragraph_embedder=embed_paragraphs,
    )

    assert result.stats.adaptive_boundaries == 0
    assert result.stats.semantic_boundaries == 0
    assert result.stats.parents == 1


def test_semantic_chunking_falls_back_to_paragraph_aware_size_windows() -> None:
    document = RagDocument(document_id="paper:fallback", source="zotero", title="Fallback")
    units = [
        TextUnit(
            "Camera pose optimization improves the consistency of reconstructed geometric maps.",
            1,
            "Method",
            paragraph_id=1,
        ),
        TextUnit(
            "The shared representation is refined using observations collected from multiple views.",
            1,
            "Method",
            paragraph_id=2,
        ),
    ]

    result = build_parent_child_chunks_with_stats(
        document,
        units,
        RagConfig(parent_chunk_tokens=900),
        paragraph_embedder=None,
    )

    assert result.stats.semantic_used is False
    assert result.stats.parents == 1
    assert "embedding is disabled" in (result.stats.fallback_reason or "")
    assert result.chunks
    assert result.chunks[0].paragraph_start == 1
    assert result.chunks[0].paragraph_end == 2


def test_v21_chunk_table_is_migrated_for_semantic_boundary_metadata(tmp_path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE rag_chunks(
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                parent_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                section TEXT NOT NULL, page_start INTEGER, page_end INTEGER,
                text TEXT NOT NULL, parent_text TEXT NOT NULL,
                chunk_type TEXT NOT NULL, quality_score REAL NOT NULL,
                quality_flags_json TEXT NOT NULL, vector_json TEXT, model_id TEXT
            )"""
        )
        RagIndex._ensure_schema(connection)  # pylint: disable=protected-access
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(rag_chunks)").fetchall()
        }

    assert {
        "paragraph_start",
        "paragraph_end",
        "boundary_reason",
        "boundary_score",
    } <= columns


def test_chunker_upgrade_defers_existing_pdf_until_lazy_top_k(tmp_path) -> None:
    pdf_path = tmp_path / "cached.pdf"
    pdf_path.write_text("not a real PDF", encoding="utf-8")
    db_path = tmp_path / "upgrade.sqlite3"
    document = RagDocument(
        document_id="paper:upgrade",
        source="zotero",
        title="Semantic Chunking Upgrade",
        abstract="The method refines camera poses and a shared scene representation.",
        pdf_path=str(pdf_path),
    )
    embedding = EmbeddingConfig(enabled=False)
    old_index = RagIndex(
        workspace_path=str(tmp_path),
        config=RagConfig(index_path=str(db_path), chunker_version="v21.parent-child.1"),
        embedding_config=embedding,
    )
    old_index.sync([document])

    new_index = RagIndex(
        workspace_path=str(tmp_path),
        config=RagConfig(index_path=str(db_path)),
        embedding_config=embedding,
    )
    metadata_only = RagDocument.from_dict({**document.to_dict(), "pdf_path": None})
    stats = new_index.sync([metadata_only])
    stored = new_index.load_documents([document.document_id])[document.document_id]

    assert stats.documents_indexed == 1
    assert stored.pdf_path is None


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
