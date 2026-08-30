"""Private SQLite run store with TTL and cancellation support."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from citation_agent.cache import default_cache_dir, stable_text_hash

from .models import CheckpointRecord, RunStatus


class RunStore:
    schema_version = "v20.checkpoint.1"
    graph_version = "citation-agent-langgraph-v20"

    def __init__(self, db_path: Path, workspace_path: str = ""):
        self.db_path = db_path
        self.workspace_hash = stable_text_hash(workspace_path or "<global>")

    @classmethod
    def for_workspace(cls, workspace_path: str, path: str | None = None) -> "RunStore":
        db_path = Path(path) if path else (
            Path(workspace_path) / ".citation-agent" / "runs.sqlite3"
            if workspace_path else default_cache_dir() / "runs.sqlite3"
        )
        return cls(db_path, workspace_path)

    def save(
        self,
        *,
        run_id: str,
        request_hash: str,
        current_node: str,
        completed_nodes: list[str],
        state_snapshot: dict[str, Any],
        status: RunStatus = "running",
        pending_actions: list[str] | None = None,
        last_error: str | None = None,
        ttl_seconds: float = 7 * 86400,
    ) -> CheckpointRecord:
        now = time.time()
        record = CheckpointRecord(
            run_id=run_id,
            request_hash=request_hash,
            graph_version=self.graph_version,
            schema_version=self.schema_version,
            current_node=current_node,
            completed_nodes=tuple(dict.fromkeys(completed_nodes)),
            state_snapshot=_sanitize_snapshot(state_snapshot),
            pending_actions=tuple(pending_actions or []),
            status=status,
            last_error=last_error,
            created_at=now,
            updated_at=now,
            expires_at=now + ttl_seconds,
        )
        with self._connect() as conn:
            self._ensure_schema(conn)
            previous = conn.execute(
                "SELECT created_at FROM run_checkpoints WHERE workspace_hash = ? AND run_id = ?",
                (self.workspace_hash, run_id),
            ).fetchone()
            created_at = float(previous[0]) if previous else record.created_at
            conn.execute(
                """INSERT OR REPLACE INTO run_checkpoints(
                    workspace_hash, run_id, request_hash, graph_version, schema_version,
                    current_node, completed_nodes_json, snapshot_json, pending_json,
                    status, cancellation_state, last_error, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT cancellation_state
                FROM run_checkpoints WHERE workspace_hash = ? AND run_id = ?), ?), ?, ?, ?, ?)""",
                (
                    self.workspace_hash, run_id, request_hash, record.graph_version,
                    record.schema_version, current_node, json.dumps(record.completed_nodes),
                    json.dumps(record.state_snapshot, ensure_ascii=False),
                    json.dumps(record.pending_actions), status, self.workspace_hash, run_id,
                    record.cancellation_state, last_error, created_at, now, record.expires_at,
                ),
            )
        return CheckpointRecord(**{**record.__dict__, "created_at": created_at})

    def load(self, run_id: str) -> CheckpointRecord | None:
        with self._connect() as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                """SELECT run_id, request_hash, graph_version, schema_version, current_node,
                 completed_nodes_json, snapshot_json, pending_json, status,
                 cancellation_state, last_error, created_at, updated_at, expires_at
                 FROM run_checkpoints WHERE workspace_hash = ? AND run_id = ?""",
                (self.workspace_hash, run_id),
            ).fetchone()
        if not row or (row[13] is not None and float(row[13]) <= time.time()):
            return None
        return CheckpointRecord(
            run_id=str(row[0]), request_hash=str(row[1]), graph_version=str(row[2]),
            schema_version=str(row[3]), current_node=str(row[4]),
            completed_nodes=tuple(json.loads(row[5])), state_snapshot=json.loads(row[6]),
            pending_actions=tuple(json.loads(row[7])), status=row[8],
            cancellation_state=str(row[9]), last_error=row[10], created_at=float(row[11]),
            updated_at=float(row[12]), expires_at=float(row[13]) if row[13] else None,
        )

    def latest(self, *, include_completed: bool = False) -> CheckpointRecord | None:
        with self._connect() as conn:
            self._ensure_schema(conn)
            status_clause = "" if include_completed else "AND status = 'running'"
            row = conn.execute(
                "SELECT run_id FROM run_checkpoints WHERE workspace_hash = ? "
                f"{status_clause} ORDER BY updated_at DESC LIMIT 1",
                (self.workspace_hash,),
            ).fetchone()
        return self.load(str(row[0])) if row else None

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as conn:
            self._ensure_schema(conn)
            cursor = conn.execute(
                "UPDATE run_checkpoints SET cancellation_state = 'requested', updated_at = ? "
                "WHERE workspace_hash = ? AND run_id = ? AND status = 'running'",
                (time.time(), self.workspace_hash, run_id),
            )
            return cursor.rowcount == 1

    def is_cancel_requested(self, run_id: str) -> bool:
        record = self.load(run_id)
        return bool(record and record.cancellation_state == "requested")

    def cleanup_expired(self) -> int:
        with self._connect() as conn:
            self._ensure_schema(conn)
            cursor = conn.execute(
                "DELETE FROM run_checkpoints WHERE workspace_hash = ? AND expires_at <= ?",
                (self.workspace_hash, time.time()),
            )
            return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""CREATE TABLE IF NOT EXISTS run_checkpoints (
            workspace_hash TEXT NOT NULL, run_id TEXT NOT NULL, request_hash TEXT NOT NULL,
            graph_version TEXT NOT NULL, schema_version TEXT NOT NULL, current_node TEXT NOT NULL,
            completed_nodes_json TEXT NOT NULL, snapshot_json TEXT NOT NULL, pending_json TEXT NOT NULL,
            status TEXT NOT NULL, cancellation_state TEXT NOT NULL, last_error TEXT,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, expires_at REAL,
            PRIMARY KEY(workspace_hash, run_id)
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_updated ON run_checkpoints(workspace_hash, updated_at)")


def _sanitize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep checkpoints compact and exclude secrets/large document payloads."""
    allowed = {"claim", "context", "candidate_counts", "current_node", "completed_nodes", "run_id"}
    result = {key: snapshot[key] for key in allowed if key in snapshot}
    result["snapshot_schema"] = "v20.checkpoint_snapshot.1"
    return result
