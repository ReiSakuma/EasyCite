#!/usr/bin/env python3
"""Run EasyCite regression eval cases and write a metrics report."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "bundled" / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from citation_agent.config import SourceConfig  # noqa: E402
from citation_agent.eval_harness import (  # noqa: E402
    load_case_files,
    render_markdown_report,
    run_eval_cases,
    write_report,
)
from citation_agent.evaluation.comparison import compare_reports, evaluate_gates  # noqa: E402
from citation_agent.evaluation.replay import run_replay_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=pathlib.Path,
        action="append",
        default=None,
        help="Path to a JSONL eval case file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--workspace",
        type=pathlib.Path,
        default=ROOT,
        help="Default workspace path used by cases that do not set workspace_path.",
    )
    parser.add_argument(
        "--mode", choices=("live", "replay"), default="live",
        help="live runs the backend; replay evaluates frozen outcomes without external calls.",
    )
    parser.add_argument("--baseline", type=pathlib.Path, default=None,
                        help="Optional previous report to compare with the current report.")
    parser.add_argument("--release-check", action="store_true",
                        help="Apply v19 default quality gates.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "eval" / "reports" / "latest.json",
        help="Path for the JSON eval report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=pathlib.Path,
        default=ROOT / "eval" / "reports" / "latest.md",
        help="Path for a compact Markdown eval report.",
    )
    parser.add_argument(
        "--no-target-trace",
        action="store_true",
        help="Do not inject gold/acceptable papers into debugTargetPapers for lifecycle tracing.",
    )
    parser.add_argument(
        "--min-target-recall-at-5",
        type=float,
        default=None,
        help="Fail with exit code 2 if target_recall_at_5 is below this threshold.",
    )
    parser.add_argument(
        "--max-unsupported-rate",
        type=float,
        default=None,
        help="Fail with exit code 2 if unsupported_recommendation_rate is above this threshold.",
    )
    parser.add_argument(
        "--max-forbidden-rate",
        type=float,
        default=None,
        help="Fail with exit code 2 if forbidden_recommendation_rate is above this threshold.",
    )
    args = parser.parse_args()

    if args.mode == "replay":
        fixture = (args.cases or [ROOT / "eval" / "fixtures" / "smoke_replay.jsonl"])[0]
        report = run_replay_file(fixture)
    else:
        case_paths = args.cases or [ROOT / "eval" / "cases" / "nerf_slam_claims.jsonl"]
        cases = load_case_files(case_paths)
        report = run_eval_cases(
            cases,
            default_workspace_path=str(args.workspace),
            source_config=SourceConfig.from_env(),
            trace_targets=not args.no_target_trace,
        )
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(baseline, report)
    if args.release_check:
        report["quality_gates"] = evaluate_gates(report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_report(report, args.output)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown_report(report), encoding="utf-8")

    metrics = report.get("metrics", {})
    threshold_failures: list[str] = []
    if args.min_target_recall_at_5 is not None and metrics.get("target_recall_at_5", 0.0) < args.min_target_recall_at_5:
        threshold_failures.append("target_recall_at_5")
    if args.max_unsupported_rate is not None and metrics.get("unsupported_recommendation_rate", 0.0) > args.max_unsupported_rate:
        threshold_failures.append("unsupported_recommendation_rate")
    if args.max_forbidden_rate is not None and metrics.get("forbidden_recommendation_rate", 0.0) > args.max_forbidden_rate:
        threshold_failures.append("forbidden_recommendation_rate")
    if threshold_failures:
        print(f"Eval thresholds failed: {', '.join(threshold_failures)}", file=sys.stderr)
        return 2
    if args.release_check and not report["quality_gates"]["passed"]:
        print("Eval release gates failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
