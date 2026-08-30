# v19 Eval Evidence Technical Design

v19 adds two explicit evaluation modes. `replay` reads frozen candidate,
evidence, verdict and trace outcomes, so regression tests do not depend on
Zotero availability, PDF extraction, embedding downloads or LLM latency.
`live` keeps the existing end-to-end runner for integration validation.

The comparison module reports baseline/current/delta for every metric. The
release gate uses initial thresholds for target recall, diagnostic recall,
forbidden recommendations, unsupported recommendations and end-to-end success.
The thresholds are policy inputs, not ranking rules, and must never be satisfied
by hard-coding a target paper.

Run a deterministic smoke evaluation:

```bash
.venv/bin/python eval/run_eval.py --mode replay --release-check
```

Compare a report with a previous one using `--baseline path/to/report.json`.
