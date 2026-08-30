"""Version comparison and release quality gates."""
from __future__ import annotations

from typing import Any


DEFAULT_GATES = {
    "target_recall_at_5": ("min", 0.75),
    "diagnostic_recall_at_10": ("min", 0.90),
    "forbidden_recommendation_rate": ("max", 0.05),
    "unsupported_recommendation_rate": ("max", 0.10),
    "end_to_end_success_rate": ("min", 0.75),
}


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    before = baseline.get("metrics", {})
    after = current.get("metrics", {})
    keys = sorted(set(before) | set(after))
    return {
        "schema_version": "v19.eval_comparison.1",
        "baseline": baseline.get("schema_version", ""),
        "current": current.get("schema_version", ""),
        "metrics": {
            key: {
                "baseline": before.get(key),
                "current": after.get(key),
                "delta": _delta(before.get(key), after.get(key)),
            }
            for key in keys
        },
    }


def evaluate_gates(report: dict[str, Any], gates: dict[str, tuple[str, float]] | None = None) -> dict[str, Any]:
    selected = gates or DEFAULT_GATES
    metrics = report.get("metrics", {})
    checks = {}
    for name, (operator, threshold) in selected.items():
        value = float(metrics.get(name, 0.0))
        passed = value >= threshold if operator == "min" else value <= threshold
        checks[name] = {"value": value, "operator": operator, "threshold": threshold, "passed": passed}
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def _delta(before: Any, after: Any) -> float | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    return round(float(after) - float(before), 6)
