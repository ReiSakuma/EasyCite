from __future__ import annotations

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fsdecode(_ROOT / "bundled" / "tool"))

from citation_agent.checkpoint.recovery_policy import recovery_action  # noqa: E402
from citation_agent.checkpoint.run_store import RunStore  # noqa: E402


def test_checkpoint_round_trip_and_cancellation(tmp_path):
    store = RunStore.for_workspace("/project", str(tmp_path / "runs.sqlite3"))
    saved = store.save(
        run_id="run-1",
        request_hash="request-1",
        current_node="verify_evidence",
        completed_nodes=["initialize", "verify_evidence"],
        state_snapshot={"claim": "claim", "api_key": "must not persist"},
    )
    loaded = store.load(saved.run_id)
    assert loaded is not None
    assert loaded.current_node == "verify_evidence"
    assert "api_key" not in loaded.state_snapshot
    assert store.request_cancel(saved.run_id) is True
    assert store.is_cancel_requested(saved.run_id) is True


def test_recovery_policy_rejects_cancelled_and_replays_safe_nodes():
    assert recovery_action("plan_retrieval", graph_compatible=True) == "replay"
    assert recovery_action("recall_zotero", graph_compatible=True) == "reuse"
    assert recovery_action("verify_evidence", graph_compatible=True, cancelled=True) == "reject"
