from __future__ import annotations

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fsdecode(_ROOT / "bundled" / "tool"))

from citation_agent.evaluation.comparison import compare_reports, evaluate_gates  # noqa: E402
from citation_agent.evaluation.replay import run_replay_file  # noqa: E402


def test_replay_is_deterministic_and_external_call_free():
    fixture = _ROOT / "eval" / "fixtures" / "smoke_replay.jsonl"
    first = run_replay_file(fixture)
    second = run_replay_file(fixture)
    assert first["mode"] == "replay"
    assert first["replay_contract"]["external_calls"] == 0
    assert first["metrics"] == second["metrics"]


def test_comparison_and_quality_gates():
    current = {"metrics": {"target_recall_at_5": 0.8, "end_to_end_success_rate": 0.9}}
    baseline = {"metrics": {"target_recall_at_5": 0.7, "end_to_end_success_rate": 0.8}}
    comparison = compare_reports(baseline, current)
    assert comparison["metrics"]["target_recall_at_5"]["delta"] == 0.1
    gates = evaluate_gates(current, {"target_recall_at_5": ("min", 0.75)})
    assert gates["passed"] is True
