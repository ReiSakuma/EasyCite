"""Persistent SQLite/FTS5 index used by the local RAG retriever."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ..cache import default_cache_dir, stable_text_hash
from ..config import EmbeddingConfig, RagConfig
from ..embedding.embedder import BaseEmbedder, EmbeddingUnavailable, create_embedder
from ..tools.adapters import execute_embedding
from .chunking import chunk_document
from .models import RagChunk, RagDocument, RagRetrievalStats


SCHEMA_VERSION = "v21.rag.1"


class RagIndex:
    def __init__(
        self,
        *,
        workspace_path: str,
        config: RagConfig,
        embedding_config: EmbeddingConfig,
        logger=None,
    ):
        self.config = config
        self.embedding_config = embedding_config
        self.logger = logger
        self.db_path = resolve_index_path(workspace_path, config.index_path)
        self._embedder: BaseEmbedder | None = None
        self._embedding_error: str | None = None

    def sync(self, documents: list[RagDocument], *, rebuild_index: bool = False) -> RagRetrievalStats:
        unique = {document.document_id: document for document in documents if document.document_id and document.title}
        indexed = reused = failed = chunks_indexed = 0
        with self._connect() as conn:
            self._ensure_schema(conn)
            for incoming in list(unique.values())[: self.config.max_documents]:
                document = incoming
                existing = conn.execute(
                    "SELECT fingerprint, metadata_json FROM rag_documents WHERE document_id = ?",
                    (document.document_id,),
                ).fetchone()
                if existing and not document.pdf_path:
                    try:
                        indexed_document = RagDocument.from_dict(json.loads(str(existing[1])))
                        if indexed_document.pdf_path:
                            document = replace(document, pdf_path=indexed_document.pdf_path)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                fingerprint = document_fingerprint(document, self.config, self.embedding_config)
                if existing and str(existing[0]) == fingerprint and not rebuild_index:
                    reused += 1
                    continue
                try:
                    chunks = chunk_document(document, self.config)
                    paper_vector, chunk_vectors = self._embed_document(document, chunks)
                    self._replace_document(conn, document, fingerprint, chunks, paper_vector, chunk_vectors)
                    indexed += 1
                    chunks_indexed += len(chunks)
                except Exception as exc:  # pylint: disable=broad-except
                    failed += 1
                    self._log(
                        f"RAG indexing failed for {document.title}: {type(exc).__name__}: {exc}"
                    )
        return RagRetrievalStats(
            documents_seen=len(unique),
            documents_indexed=indexed,
            documents_reused=reused,
            documents_failed=failed,
            chunks_indexed=chunks_indexed,
            embedding_used=self._embedder is not None and self._embedding_error is None,
            fallback_reason=self._embedding_error,
            index_path=str(self.db_path),
        )

    def paper_lexical_rank(self, queries: list[str], allowed_ids: set[str], limit: int) -> list[str]:
        return self._fts_rank("rag_paper_fts", queries, allowed_ids, limit)

    def chunk_lexical_rank(self, queries: list[str], allowed_ids: set[str], limit: int) -> list[str]:
        return self._fts_rank("rag_chunk_fts", queries, allowed_ids, limit, id_column="chunk_id")

    def paper_dense_rank(self, queries: list[str], allowed_ids: set[str], limit: int) -> list[str]:
        query_vectors = self._embed_queries(queries)
        if not query_vectors:
            return []
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT document_id, vector_json FROM rag_documents WHERE vector_json IS NOT NULL"
            ).fetchall()
        scores: dict[str, float] = {}
        for document_id, vector_json in rows:
            identity = str(document_id)
            if identity not in allowed_ids:
                continue
            vector = parse_vector(vector_json)
            if vector:
                scores[identity] = max(cosine_similarity(query, vector) for query in query_vectors)
        return [identity for identity, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]

    def chunk_dense_rank(self, queries: list[str], allowed_ids: set[str], limit: int) -> list[str]:
        query_vectors = self._embed_queries(queries)
        if not query_vectors:
            return []
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT chunk_id, document_id, vector_json FROM rag_chunks WHERE vector_json IS NOT NULL"
            ).fetchall()
        scores: dict[str, float] = {}
        for chunk_id, document_id, vector_json in rows:
            if str(document_id) not in allowed_ids:
                continue
            vector = parse_vector(vector_json)
            if vector:
                scores[str(chunk_id)] = max(cosine_similarity(query, vector) for query in query_vectors)
        return [identity for identity, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]

    def load_documents(self, document_ids: Iterable[str]) -> dict[str, RagDocument]:
        ids = list(dict.fromkeys(document_ids))
        if not ids:
            return {}
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                f"SELECT document_id, metadata_json FROM rag_documents WHERE document_id IN ({','.join('?' for _ in ids)})",
                ids,
            ).fetchall()
        return {
            str(document_id): RagDocument.from_dict(json.loads(str(metadata_json)))
            for document_id, metadata_json in rows
        }

    def load_chunks(self, chunk_ids: Iterable[str]) -> dict[str, RagChunk]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                f"""SELECT chunk_id, document_id, parent_id, ordinal, section, page_start,
                    page_end, text, parent_text, chunk_type, quality_score, quality_flags_json
                    FROM rag_chunks WHERE chunk_id IN ({','.join('?' for _ in ids)})""",
                ids,
            ).fetchall()
        return {
            str(row[0]): RagChunk(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                parent_id=str(row[2]),
                ordinal=int(row[3]),
                section=str(row[4]),
                page_start=int(row[5]) if row[5] is not None else None,
                page_end=int(row[6]) if row[6] is not None else None,
                text=str(row[7]),
                parent_text=str(row[8]),
                chunk_type=str(row[9]),
                quality_score=float(row[10]),
                quality_flags=tuple(json.loads(str(row[11]) or "[]")),
            )
            for row in rows
        }

    def _fts_rank(
        self,
        table: str,
        queries: list[str],
        allowed_ids: set[str],
        limit: int,
        *,
        id_column: str = "document_id",
    ) -> list[str]:
        best: dict[str, tuple[int, float]] = {}
        with self._connect() as conn:
            self._ensure_schema(conn)
            for query_index, query in enumerate(queries):
                match = fts_query(query)
                if not match:
                    continue
                try:
                    rows = conn.execute(
                        f"SELECT {id_column}, document_id, bm25({table}) AS score FROM {table} "
                        f"WHERE {table} MATCH ? ORDER BY score LIMIT ?",
                        (match, max(limit * 3, limit)),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    self._log(f"RAG FTS query skipped: {exc}")
                    continue
                for rank, row in enumerate(rows, start=1):
                    identity = str(row[0])
                    document_id = str(row[1])
                    if document_id not in allowed_ids:
                        continue
                    score = float(row[2])
                    previous = best.get(identity)
                    candidate = (rank + query_index, score)
                    if previous is None or candidate < previous:
                        best[identity] = candidate
        return [identity for identity, _value in sorted(best.items(), key=lambda item: (item[1], item[0]))[:limit]]

    def _embed_document(
        self,
        document: RagDocument,
        chunks: list[RagChunk],
    ) -> tuple[list[float] | None, list[list[float] | None]]:
        embedder = self._get_embedder()
        if embedder is None:
            return None, [None] * len(chunks)
        texts = [document.metadata_text(), *[chunk.text for chunk in chunks]]
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.config.embedding_batch_size):
                batch = texts[start : start + self.config.embedding_batch_size]
                vectors.extend(
                    execute_embedding(batch, lambda batch=batch: embedder.embed(batch)).vectors
                )
        except Exception as exc:  # pylint: disable=broad-except
            self._embedding_error = f"{type(exc).__name__}: {exc}"
            self._log(f"RAG embedding unavailable; BM25-only index retained: {self._embedding_error}")
            return None, [None] * len(chunks)
        paper = vectors[0] if vectors else None
        chunk_vectors: list[list[float] | None] = list(vectors[1:])
        while len(chunk_vectors) < len(chunks):
            chunk_vectors.append(None)
        return paper, chunk_vectors

    def _embed_queries(self, queries: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        if embedder is None:
            return []
        try:
            return execute_embedding(queries, lambda: embedder.embed(queries)).vectors
        except Exception as exc:  # pylint: disable=broad-except
            self._embedding_error = f"{type(exc).__name__}: {exc}"
            self._log(f"RAG dense retrieval unavailable; BM25-only retrieval used: {self._embedding_error}")
            return []

    def _get_embedder(self) -> BaseEmbedder | None:
        if not self.embedding_config.enabled or self._embedding_error:
            return None
        if self._embedder is not None:
            return self._embedder
        try:
            self._embedder = create_embedder(self.embedding_config)
        except EmbeddingUnavailable as exc:
            self._embedding_error = str(exc)
            return None
        return self._embedder

    def _replace_document(
        self,
        conn: sqlite3.Connection,
        document: RagDocument,
        fingerprint: str,
        chunks: list[RagChunk],
        paper_vector: list[float] | None,
        chunk_vectors: list[list[float] | None],
    ) -> None:
        conn.execute("DELETE FROM rag_paper_fts WHERE document_id = ?", (document.document_id,))
        conn.execute("DELETE FROM rag_chunk_fts WHERE document_id = ?", (document.document_id,))
        conn.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document.document_id,))
        conn.execute("DELETE FROM rag_documents WHERE document_id = ?", (document.document_id,))
        conn.execute(
            """INSERT INTO rag_documents(
                document_id, source, title, abstract, metadata_json, fingerprint,
                vector_json, model_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document.document_id,
                document.source,
                document.title,
                document.abstract,
                json.dumps(document.to_dict(), ensure_ascii=False),
                fingerprint,
                dump_vector(paper_vector),
                self.embedding_config.model_id if paper_vector else None,
                time.time(),
            ),
        )
        conn.execute(
            "INSERT INTO rag_paper_fts(document_id, title, abstract, authors, venue) VALUES (?, ?, ?, ?, ?)",
            (
                document.document_id,
                document.title,
                document.abstract,
                " ".join(document.authors),
                document.venue or "",
            ),
        )
        for chunk, vector in zip(chunks, chunk_vectors, strict=False):
            conn.execute(
                """INSERT INTO rag_chunks(
                    chunk_id, document_id, parent_id, ordinal, section, page_start,
                    page_end, text, parent_text, chunk_type, quality_score,
                    quality_flags_json, vector_json, model_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.parent_id,
                    chunk.ordinal,
                    chunk.section,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.text,
                    chunk.parent_text,
                    chunk.chunk_type,
                    chunk.quality_score,
                    json.dumps(chunk.quality_flags),
                    dump_vector(vector),
                    self.embedding_config.model_id if vector else None,
                ),
            )
            conn.execute(
                "INSERT INTO rag_chunk_fts(chunk_id, document_id, title, section, text) VALUES (?, ?, ?, ?, ?)",
                (chunk.chunk_id, document.document_id, document.title, chunk.section, chunk.text),
            )

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS rag_documents(
                document_id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL,
                abstract TEXT NOT NULL, metadata_json TEXT NOT NULL, fingerprint TEXT NOT NULL,
                vector_json TEXT, model_id TEXT, updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS rag_chunks(
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, parent_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, section TEXT NOT NULL, page_start INTEGER, page_end INTEGER,
                text TEXT NOT NULL, parent_text TEXT NOT NULL, chunk_type TEXT NOT NULL,
                quality_score REAL NOT NULL, quality_flags_json TEXT NOT NULL,
                vector_json TEXT, model_id TEXT
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id, ordinal)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS rag_paper_fts USING fts5(document_id UNINDEXED, title, abstract, authors, venue)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunk_fts USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, title, section, text)"
        )
        conn.execute(f"PRAGMA user_version = {schema_user_version(SCHEMA_VERSION)}")

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


def resolve_index_path(workspace_path: str, configured_path: str | None) -> Path:
    if configured_path:
        return Path(configured_path).expanduser()
    if workspace_path:
        return Path(workspace_path).expanduser() / ".citation-agent" / "rag.sqlite3"
    return default_cache_dir() / "rag.sqlite3"


def document_fingerprint(
    document: RagDocument,
    config: RagConfig,
    embedding_config: EmbeddingConfig,
) -> str:
    pdf_identity = ""
    if document.pdf_path:
        try:
            stat = os.stat(document.pdf_path)
            pdf_identity = f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            pdf_identity = "missing"
    payload = {
        "document": document.to_dict(),
        "pdf": pdf_identity,
        "chunker": config.chunker_version,
        "child": config.child_chunk_tokens,
        "parent": config.parent_chunk_tokens,
        "overlap": config.chunk_overlap_tokens,
        "embedding": embedding_config.model_id if embedding_config.enabled else "disabled",
    }
    return stable_text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def fts_query(query: str) -> str:
    terms = []
    seen: set[str] = set()
    for term in __import__("re").findall(r"[A-Za-z0-9][A-Za-z0-9_+\-]{1,}", query.lower()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(f'"{term.replace(chr(34), chr(34) * 2)}"')
        if len(terms) >= 14:
            break
    return " OR ".join(terms)


def dump_vector(vector: list[float] | None) -> str | None:
    return json.dumps(vector, separators=(",", ":")) if vector else None


def parse_vector(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        return [float(item) for item in json.loads(str(value))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or not second:
        return 0.0
    dot = sum(a * b for a, b in zip(first, second, strict=False))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot / (first_norm * second_norm)


def schema_user_version(value: str) -> int:
    return int(stable_text_hash(value)[:6], 16) % 2_000_000_000
