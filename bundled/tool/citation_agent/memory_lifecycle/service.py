"""SQLite-backed memory service with scope and lifecycle enforcement."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from citation_agent.cache import default_cache_dir, stable_text_hash

from .models import MemoryRecord, MemoryScope, MemoryStatus


class MemoryService:
    """Store and retrieve governed records without leaking workspace data."""

    schema_version = "v18.memory.1"

    def __init__(self, db_path: Path, *, workspace_path: str = ""):
        self.db_path = db_path
        self.workspace_hash = stable_text_hash(workspace_path or "<global>")

    @classmethod
    def for_workspace(cls, workspace_path: str, memory_path: str | None = None) -> "MemoryService":
        path = Path(memory_path) if memory_path else (
            Path(workspace_path) / ".citation-agent" / "memory.sqlite3"
            if workspace_path else default_cache_dir() / "memory.sqlite3"
        )
        return cls(path, workspace_path=workspace_path)

    def put(
        self,
        *,
        memory_type: str,
        scope: MemoryScope,
        content: str,
        payload: dict[str, Any] | None = None,
        source: str = "system",
        provenance: dict[str, Any] | None = None,
        confidence: float = 0.5,
        ttl_seconds: float | None = None,
        evidence_refs: list[str] | None = None,
        correction_of: str | None = None,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Memory confidence must be between 0 and 1")
        if scope not in {"session", "project", "user", "bad_case"}:
            raise ValueError(f"Unsupported memory scope: {scope}")
        now = time.time()
        record = MemoryRecord(
            record_id=uuid.uuid4().hex,
            memory_type=memory_type,
            scope=scope,
            content=content.strip(),
            payload=payload or {},
            source=source,
            provenance=provenance or {},
            confidence=confidence,
            created_at=now,
            updated_at=now,
            expires_at=now + ttl_seconds if ttl_seconds else None,
            evidence_refs=tuple(evidence_refs or []),
            correction_of=correction_of,
        )
        with self._connect() as conn:
            self._ensure_schema(conn)
            conn.execute(
                """INSERT INTO memory_records(
                    workspace_hash, record_id, memory_type, scope, content,
                    payload_json, source, provenance_json, confidence,
                    created_at, updated_at, expires_at, version, status,
                    evidence_refs_json, correction_of
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.workspace_hash, record.record_id, record.memory_type,
                    record.scope, record.content,
                    json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                    record.source,
                    json.dumps(record.provenance, ensure_ascii=False, sort_keys=True),
                    record.confidence, record.created_at, record.updated_at,
                    record.expires_at, record.version, record.status,
                    json.dumps(list(record.evidence_refs)), record.correction_of,
                ),
            )
        return record

    def recall(
        self,
        *,
        memory_type: str | None = None,
        scopes: tuple[MemoryScope, ...] = ("session", "project", "user"),
        limit: int = 100,
        now: float | None = None,
    ) -> list[MemoryRecord]:
        current = time.time() if now is None else now
        placeholders = ",".join("?" for _ in scopes) or "?"
        args: list[Any] = [self.workspace_hash, *scopes]
        type_clause = ""
        if memory_type:
            type_clause = " AND memory_type = ?"
            args.append(memory_type)
        args.extend([current, max(1, limit)])
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                f"""SELECT record_id, memory_type, scope, content, payload_json,
                    source, provenance_json, confidence, created_at, updated_at,
                    expires_at, version, status, evidence_refs_json, correction_of
                    FROM memory_records
                    WHERE workspace_hash = ? AND scope IN ({placeholders})
                    AND status = 'active' AND (expires_at IS NULL OR expires_at > ?)
                    {type_clause}
                    ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
                args,
            ).fetchall()
        return [self._record(row) for row in rows]

    def transition(self, record_id: str, status: MemoryStatus) -> bool:
        if status not in {"active", "superseded", "invalid"}:
            raise ValueError(f"Unsupported memory status: {status}")
        with self._connect() as conn:
            self._ensure_schema(conn)
            cursor = conn.execute(
                "UPDATE memory_records SET status = ?, updated_at = ? "
                "WHERE workspace_hash = ? AND record_id = ?",
                (status, time.time(), self.workspace_hash, record_id),
            )
            return cursor.rowcount == 1

    def correct(self, record_id: str, **kwargs: Any) -> MemoryRecord:
        old = next((item for item in self.recall(limit=1000) if item.record_id == record_id), None)
        if old is None:
            raise KeyError(f"Memory record not found: {record_id}")
        self.transition(record_id, "superseded")
        return self.put(
            memory_type=str(kwargs.get("memory_type", old.memory_type)),
            scope=kwargs.get("scope", old.scope),
            content=str(kwargs.get("content", old.content)),
            payload=kwargs.get("payload", old.payload),
            source=str(kwargs.get("source", old.source)),
            provenance=kwargs.get("provenance", old.provenance),
            confidence=float(kwargs.get("confidence", old.confidence)),
            ttl_seconds=(
                max(0.0, float(old.expires_at - time.time()))
                if old.expires_at else None
            ),
            evidence_refs=list(old.evidence_refs),
            correction_of=record_id,
        )

    def clear(self, *, scopes: tuple[MemoryScope, ...] | None = None) -> int:
        with self._connect() as conn:
            self._ensure_schema(conn)
            if scopes:
                placeholders = ",".join("?" for _ in scopes)
                cursor = conn.execute(
                    f"DELETE FROM memory_records WHERE workspace_hash = ? AND scope IN ({placeholders})",
                    [self.workspace_hash, *scopes],
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM memory_records WHERE workspace_hash = ?",
                    (self.workspace_hash,),
                )
            return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT scope, status, COUNT(*) FROM memory_records "
                "WHERE workspace_hash = ? GROUP BY scope, status",
                (self.workspace_hash,),
            ).fetchall()
        return {
            "schema_version": self.schema_version,
            "workspace_hash": self.workspace_hash,
            "total": sum(int(row[2]) for row in rows),
            "by_scope": {f"{row[0]}:{row[1]}": int(row[2]) for row in rows},
        }

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""CREATE TABLE IF NOT EXISTS memory_records (
            workspace_hash TEXT NOT NULL, record_id TEXT NOT NULL,
            memory_type TEXT NOT NULL, scope TEXT NOT NULL, content TEXT NOT NULL,
            payload_json TEXT NOT NULL, source TEXT NOT NULL,
            provenance_json TEXT NOT NULL, confidence REAL NOT NULL,
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            expires_at REAL, version INTEGER NOT NULL, status TEXT NOT NULL,
            evidence_refs_json TEXT NOT NULL, correction_of TEXT,
            PRIMARY KEY(workspace_hash, record_id)
        )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_scope "
            "ON memory_records(workspace_hash, scope, status, updated_at)"
        )

    @staticmethod
    def _record(row: tuple[Any, ...]) -> MemoryRecord:
        return MemoryRecord(
            record_id=str(row[0]), memory_type=str(row[1]), scope=row[2],
            content=str(row[3]), payload=json.loads(row[4]), source=str(row[5]),
            provenance=json.loads(row[6]), confidence=float(row[7]),
            created_at=float(row[8]), updated_at=float(row[9]),
            expires_at=float(row[10]) if row[10] is not None else None,
            version=int(row[11]), status=row[12],
            evidence_refs=tuple(json.loads(row[13])),
            correction_of=row[14],
        )
