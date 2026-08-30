# Change Log

## 0.20.0

- Added durable SQLite checkpoints after each LangGraph node.
- Added cancellation state, TTL cleanup, snapshot sanitization and recovery policy.
- Added checkpoint round-trip and cancellation tests.

## 0.19.0

- Added replay/live evaluation modes and frozen replay fixtures.
- Added baseline comparison, metric deltas and release quality gates.
- Added deterministic evaluation tests and methodology documentation.

## 0.18.0

- Added lifecycle-managed session/project/user/bad-case memory contracts.
- Added workspace isolation, expiry filtering, correction chains and memory stats.
- Mirrored explicit candidate feedback into governed project memory.

## 0.17.0

- Added a versioned Tool Registry and shared Tool Executor.
- Added pre-execution input validation, output-contract validation, per-tool deadlines, retries, fallback, circuit breaking, permissions, and idempotency.
- Routed LLM, Zotero, Better BibTeX, PDF, and embedding calls through the reliability layer.
- Added an idempotent BibTeX Writer adapter with atomic writes and post-write verification.
- Migrated Tool Health to registry-provided health checks.
- Added structured tool observations to Trace, Agent State, and persisted Run Record JSON.
- Added user-visible tool degradation messages in the candidate review interface.
- Added reliability tests for success, invalid input/output, timeout, connection retry, fallback, permission, and duplicate write prevention.

## 0.16.0

- Added a standalone Context Manager with node-specific estimated-token budgets.
- Migrated retrieval query planning, evidence verification, and evidence translation to budgeted context packets.
- Added deterministic compaction while retaining claims, output constraints, candidate IDs, and evidence structure.
- Added stable duplicate removal plus per-item source, purpose, action, and pruning-reason metadata.
- Added context reports to Run Trace, Agent State, and persisted Run Record JSON.
- Added advanced workspace settings for query, evidence, and translation context budgets.
- Added Python regression tests for deterministic pruning, required-context retention, nested candidate preservation, and planner integration.

## 0.15.0

- Product MVP release-candidate polish.
- Updated VSIX package names to `easycite-v20.vsix` and `easycite-v20-Windows.vsix`.
- Reset first-run setup prompt for v15 so new installs are guided through Python, Zotero, Better BibTeX, model API, embedding, and PDF checks.
- Hid Trace, Agent State, Eval Bad Case, and Restart Server commands behind `citation-agent.developerMode`.
- Updated README with user setup, packaging, Zotero, and developer-mode guidance.
- Tightened VSIX ignore rules for release artifacts and local eval/cache files.
