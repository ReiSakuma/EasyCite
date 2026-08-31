"""Schemas used by the local RAG index and retriever."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import CitationSuggestion, EvidenceSnippet


@dataclass(frozen=True)
class RagDocument:
    document_id: str
    source: str
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    year: str | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    citation_key: str | None = None
    pdf_path: str | None = None
    collection_keys: tuple[str, ...] = ()

    @classmethod
    def from_suggestion(
        cls,
        suggestion: CitationSuggestion,
        *,
        collection_keys: list[str] | tuple[str, ...] = (),
    ) -> "RagDocument":
        abstract = ""
        if suggestion.evidence.source == "abstract" and suggestion.evidence.text:
            abstract = suggestion.evidence.text
        return cls(
            document_id=suggestion.paper_id,
            source=suggestion.source,
            title=suggestion.title,
            abstract=abstract,
            authors=tuple(suggestion.authors),
            year=suggestion.year,
            venue=suggestion.venue,
            doi=suggestion.doi,
            url=suggestion.url,
            citation_key=suggestion.citation_key,
            pdf_path=suggestion.pdf_path,
            collection_keys=tuple(collection_keys),
        )

    def to_suggestion(
        self,
        *,
        evidence_passages: list[EvidenceSnippet] | None = None,
        confidence: float = 0.58,
        reason: str = "Retrieved from the local RAG index.",
    ) -> CitationSuggestion:
        passages = list(evidence_passages or [])
        primary = passages[0] if passages else EvidenceSnippet(
            source="abstract" if self.abstract else "metadata",
            section="Abstract" if self.abstract else "Metadata",
            text=self.abstract or self.title,
        )
        source = self.source if self.source in {"current_bib", "zotero", "online", "mock"} else "zotero"
        return CitationSuggestion(
            paper_id=self.document_id,
            title=self.title,
            citation_key=self.citation_key,
            authors=list(self.authors),
            year=self.year,
            venue=self.venue,
            doi=self.doi,
            url=self.url,
            pdf_path=self.pdf_path,
            verdict="metadata_match",
            confidence=max(0.0, min(confidence, 1.0)),
            source=source,  # type: ignore[arg-type]
            evidence=primary,
            evidence_passages=passages,
            reason=reason,
        )

    def metadata_text(self) -> str:
        return " ".join(
            part
            for part in [
                self.title,
                self.abstract,
                " ".join(self.authors),
                self.venue or "",
                self.year or "",
            ]
            if part
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "citation_key": self.citation_key,
            "pdf_path": self.pdf_path,
            "collection_keys": list(self.collection_keys),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RagDocument":
        return cls(
            document_id=str(value.get("document_id") or ""),
            source=str(value.get("source") or "zotero"),
            title=str(value.get("title") or ""),
            abstract=str(value.get("abstract") or ""),
            authors=tuple(str(item) for item in value.get("authors") or []),
            year=str(value["year"]) if value.get("year") else None,
            venue=str(value["venue"]) if value.get("venue") else None,
            doi=str(value["doi"]) if value.get("doi") else None,
            url=str(value["url"]) if value.get("url") else None,
            citation_key=str(value["citation_key"]) if value.get("citation_key") else None,
            pdf_path=str(value["pdf_path"]) if value.get("pdf_path") else None,
            collection_keys=tuple(str(item) for item in value.get("collection_keys") or []),
        )


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    document_id: str
    parent_id: str
    ordinal: int
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    parent_text: str
    chunk_type: str
    quality_score: float
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagPassageHit:
    chunk: RagChunk
    fused_score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    reranker_score: float | None = None


@dataclass(frozen=True)
class RagRetrievalStats:
    documents_seen: int = 0
    documents_indexed: int = 0
    documents_reused: int = 0
    documents_failed: int = 0
    chunks_indexed: int = 0
    paper_lexical_hits: int = 0
    paper_dense_hits: int = 0
    paper_fused_hits: int = 0
    passage_lexical_hits: int = 0
    passage_dense_hits: int = 0
    passage_fused_hits: int = 0
    reranked_passages: int = 0
    embedding_used: bool = False
    reranker_used: bool = False
    pdf_documents_requested: int = 0
    pdf_documents_hydrated: int = 0
    metadata_sync_ms: float = 0.0
    pdf_index_ms: float = 0.0
    paper_retrieval_ms: float = 0.0
    passage_retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    total_ms: float = 0.0
    fallback_reason: str | None = None
    index_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RagRetrievalResult:
    suggestions: list[CitationSuggestion] = field(default_factory=list)
    stats: RagRetrievalStats = field(default_factory=RagRetrievalStats)
