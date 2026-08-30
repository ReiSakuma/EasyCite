# EasyCite Eval Harness

v10 introduces a repeatable regression eval harness for EasyCite. It turns
manual bad-case analysis into JSONL cases, runnable metrics, and saved reports.

The harness answers questions that matter for Agent engineering:

- Did the expected target paper enter the top recommendations?
- If not, did it at least appear in diagnostic candidates?
- Did the system recommend forbidden or wrong-field papers?
- Did the main recommendations contain unsupported evidence verdicts?
- How long did the run take, and did trace report tool or LLM failures?

## Case Format

Each case is one JSON object per line:

```json
{
  "id": "nerf_slam_001",
  "claim": "Recent neural and differentiable representations...",
  "context": "For many years, dense SLAM systems...",
  "citation_needed": true,
  "gold_papers": ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"],
  "acceptable_papers": ["iMAP", "NICE-SLAM", "Co-SLAM", "CP-SLAM"],
  "forbidden_papers": ["pure geometric SLAM paper without neural/differentiable representation"],
  "expected_evidence_keywords": ["neural radiance field", "differentiable rendering"],
  "notes": "Checks whether claim understanding separates SLAM context from neural/differentiable mechanism."
}
```

Optional per-case fields:

- `workspace_path`
- `file_path`
- `existing_citations`

If `workspace_path` is omitted, `--workspace` is used.

## Run

From the `v10` directory:

```bash
.venv/bin/python eval/run_eval.py \
  --cases eval/cases/nerf_slam_claims.jsonl \
  --output eval/reports/latest.json \
  --markdown-output eval/reports/latest.md
```

The runner uses the same backend code as the extension. Environment settings
such as Zotero, LLM, embedding, and LangGraph runtime are read from the normal
EasyCite environment variables.

## Thresholds

Use thresholds when you want the eval to fail on regression:

```bash
.venv/bin/python eval/run_eval.py \
  --min-target-recall-at-5 0.7 \
  --max-unsupported-rate 0.2 \
  --max-forbidden-rate 0.05
```

Exit code `2` means the run completed, but at least one metric violated the
configured threshold.

## Key Metrics

- `citation_needed_accuracy`
- `citation_needed_precision`
- `citation_needed_recall`
- `citation_needed_f1`
- `target_recall_at_5`
- `diagnostic_recall_at_10`
- `forbidden_recommendation_rate`
- `unsupported_recommendation_rate`
- `end_to_end_success_rate`
- `average_evidence_keyword_hit_rate`
- `average_latency_ms`
- `llm_timeout_rate`
- `tool_failure_rate`

## Report Files

By default, reports are written to:

```text
eval/reports/latest.json
eval/reports/latest.md
```

These files are generated artifacts and can be used to compare versions.
