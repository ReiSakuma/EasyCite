from __future__ import annotations

import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL_DIR = os.fsdecode(_PROJECT_ROOT / "bundled" / "tool")
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from citation_agent.memory_lifecycle.service import MemoryService  # noqa: E402


def test_memory_is_scoped_and_expired_records_are_not_recalled(tmp_path):
    db = tmp_path / "memory.sqlite3"
    project_a = MemoryService.for_workspace("/project/a", str(db))
    project_b = MemoryService.for_workspace("/project/b", str(db))
    project_a.put(
        memory_type="preference",
        scope="project",
        content="Prefer evidence from the active collection",
        source="explicit_user_feedback",
        confidence=0.8,
    )
    project_a.put(
        memory_type="temporary",
        scope="session",
        content="temporary result",
        ttl_seconds=1,
    )

    assert len(project_a.recall()) == 2
    assert project_b.recall() == []
    assert [item.memory_type for item in project_a.recall(now=10**12)] == ["preference"]


def test_memory_correction_preserves_audit_chain(tmp_path):
    service = MemoryService.for_workspace("/project", str(tmp_path / "memory.sqlite3"))
    original = service.put(
        memory_type="candidate_feedback",
        scope="project",
        content="accepted weak paper",
        confidence=0.3,
    )
    corrected = service.correct(
        original.record_id,
        content="rejected after reading the full evidence",
        confidence=0.9,
    )

    assert service.recall()[0].record_id == corrected.record_id
    assert service.recall()[0].correction_of == original.record_id
