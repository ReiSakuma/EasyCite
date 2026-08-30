"""Persistent local caches for expensive evidence retrieval steps."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "citation-agent"


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedPdfPassage:
    page: int
    text: str


class PdfPassageCache:
    """Cache quality-filtered PDF passages by file identity."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or default_cache_dir())
        self.db_path = self.cache_dir / "pdf_passages.sqlite3"
        self._available = True

    def get(self, pdf_path: str) -> list[CachedPdfPassage] | None:
        try:
            stat = os.stat(pdf_path)
            with self._connect() as conn:
                self._ensure_schema(conn)
                rows = conn.execute(
                    """
                    SELECT page, text
                    FROM pdf_passages
                    WHERE path = ? AND mtime_ns = ? AND size = ?
                    ORDER BY page ASC, ordinal ASC
                    """,
                    (pdf_path, stat.st_mtime_ns, stat.st_size),
                ).fetchall()
        except Exception:  # pylint: disable=broad-except
            self._available = False
            return None

        if not rows:
            return None
        return [CachedPdfPassage(page=int(page), text=str(text)) for page, text in rows]

    def put(self, pdf_path: str, passages: Iterable[CachedPdfPassage]) -> None:
        if not self._available:
            return
        try:
            stat = os.stat(pdf_path)
            passage_list = list(passages)
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.execute("DELETE FROM pdf_passages WHERE path = ?", (pdf_path,))
                conn.executemany(
                    """
                    INSERT INTO pdf_passages(path, mtime_ns, size, page, ordinal, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            pdf_path,
                            stat.st_mtime_ns,
                            stat.st_size,
                            passage.page,
                            index,
                            passage.text,
                        )
                        for index, passage in enumerate(passage_list)
                    ],
                )
        except Exception:  # pylint: disable=broad-except
            self._available = False

    def _connect(self) -> sqlite3.Connection:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_passages (
                path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                page INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY(path, mtime_ns, size, page, ordinal)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pdf_passages_path ON pdf_passages(path)"
        )


class EmbeddingVectorCache:
    """Cache text embeddings by model and normalized text hash."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or default_cache_dir() / "embeddings")
        self.db_path = self.cache_dir / "embedding_vectors.sqlite3"
        self._available = True

    def get_many(self, model_id: str, texts: list[str]) -> dict[int, list[float]]:
        if not self._available or not texts:
            return {}
        try:
            keys = [stable_text_hash(text) for text in texts]
            with self._connect() as conn:
                self._ensure_schema(conn)
                rows = conn.execute(
                    f"""
                    SELECT text_hash, vector_json
                    FROM embedding_vectors
                    WHERE model_id = ? AND text_hash IN ({",".join("?" for _ in keys)})
                    """,
                    [model_id, *keys],
                ).fetchall()
            vectors_by_key = {
                str(text_hash): [float(value) for value in json.loads(str(vector_json))]
                for text_hash, vector_json in rows
            }
            return {
                index: vectors_by_key[key]
                for index, key in enumerate(keys)
                if key in vectors_by_key
            }
        except Exception:  # pylint: disable=broad-except
            self._available = False
            return {}

    def put_many(self, model_id: str, texts: list[str], vectors: list[list[float]]) -> None:
        if not self._available or not texts:
            return
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_vectors(model_id, text_hash, vector_json)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            model_id,
                            stable_text_hash(text),
                            json.dumps(vector, separators=(",", ":")),
                        )
                        for text, vector in zip(texts, vectors, strict=False)
                    ],
                )
        except Exception:  # pylint: disable=broad-except
            self._available = False

    def _connect(self) -> sqlite3.Connection:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_vectors (
                model_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY(model_id, text_hash)
            )
            """
        )


class EvidenceJudgeCache:
    """Cache LLM evidence judgments for reproducible candidate ranking."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or default_cache_dir())
        self.db_path = self.cache_dir / "evidence_judgments.sqlite3"
        self._available = True

    def get(self, key: str) -> dict | None:
        if not self._available:
            return None
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                row = conn.execute(
                    "SELECT result_json FROM evidence_judgments WHERE cache_key = ?",
                    (key,),
                ).fetchone()
            if not row:
                return None
            result = json.loads(str(row[0]))
            return result if isinstance(result, dict) else None
        except Exception:  # pylint: disable=broad-except
            self._available = False
            return None

    def put(self, key: str, result: dict) -> None:
        if not self._available:
            return
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evidence_judgments(cache_key, result_json)
                    VALUES (?, ?)
                    """,
                    (key, json.dumps(result, ensure_ascii=False, sort_keys=True)),
                )
        except Exception:  # pylint: disable=broad-except
            self._available = False

    def _connect(self) -> sqlite3.Connection:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_judgments (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL
            )
            """
        )


class RetrievalPlanCache:
    """Cache LLM retrieval plans so identical claims produce stable recall."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or default_cache_dir())
        self.db_path = self.cache_dir / "retrieval_plans.sqlite3"
        self._available = True

    def get(self, key: str) -> dict | None:
        if not self._available:
            return None
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                row = conn.execute(
                    "SELECT plan_json FROM retrieval_plans WHERE cache_key = ?",
                    (key,),
                ).fetchone()
            if not row:
                return None
            plan = json.loads(str(row[0]))
            return plan if isinstance(plan, dict) else None
        except Exception:  # pylint: disable=broad-except
            self._available = False
            return None

    def put(self, key: str, plan: dict) -> None:
        if not self._available:
            return
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO retrieval_plans(cache_key, plan_json)
                    VALUES (?, ?)
                    """,
                    (key, json.dumps(plan, ensure_ascii=False, sort_keys=True)),
                )
        except Exception:  # pylint: disable=broad-except
            self._available = False

    def _connect(self) -> sqlite3.Connection:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_plans (
                cache_key TEXT PRIMARY KEY,
                plan_json TEXT NOT NULL
            )
            """
        )
