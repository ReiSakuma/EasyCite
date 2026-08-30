# EasyCite v17 Tool Registry and Reliability

v17 turns EasyCite integrations into versioned tools with explicit contracts. The goal is to keep network, dependency, model, file, and permission failures out of business-node exception handling and make every recovery decision observable.

## Architecture

```text
LangGraph / LSP node
  -> Tool Registry resolve
  -> input contract validation
  -> permission and idempotency checks
  -> deadline-bound execution
  -> output contract validation
  -> per-tool retry or fallback
  -> Tool Observation
  -> business result or structured ToolExecutionError
```

The implementation is under `bundled/tool/citation_agent/tools/`:

- `base.py`: definitions, policies, results, observations, and health contracts.
- `registry.py`: registration, version resolution, listing, and capability discovery.
- `executor.py`: validation, deadlines, retries, fallback, circuit breaking, idempotency, and observation recording.
- `errors.py`: stable failure taxonomy and exception classification.
- `adapters.py`: adapters and health checks for production integrations.
- `observability.py`: request-local observation collection using `contextvars`.

## Registered Tools

| Tool | Mode | Retry | Fallback / Guard |
|---|---|---|---|
| `zotero.search` | read-only | connection, timeout, rate limit | `current_bib.search` |
| `better_bibtex.export` | read-only | connection, timeout, rate limit | candidate metadata resolver |
| `llm.completion` | read-only | connection, timeout, rate limit | caller's deterministic local logic |
| `pdf.read` | read-only | no blind retry | abstract and metadata evidence |
| `embedding.rank` | read-only | no blind retry | lexical ranker |
| `bibtex.write` | mutating | no blind retry | permission, idempotency key, atomic write, verification |

## Failure Taxonomy

Every final failure uses one searchable category:

- `configuration_error`
- `dependency_missing`
- `authentication_error`
- `connection_error`
- `timeout`
- `rate_limited`
- `invalid_input`
- `invalid_output`
- `not_found`
- `permission_denied`
- `internal_error`

Only explicitly retryable read-only failures are retried. Invalid input/output and permission failures are not retried. Three consecutive connection-class failures temporarily open a per-tool circuit.

## Tool Observations

Each attempt records:

- `tool_call_id`
- tool name and version
- attempt and latency
- input/output validation status
- fallback and idempotency cache hit
- error category and redacted error message
- compact output summary

Observations are persisted under `tool_reliability` in the Run Record and rendered as `tool_call` rows in developer Trace. Normal users see concise degradation notices in the candidate review page when a fallback was required.

## BibTeX Safety

The backend writer adapter requires an explicit `workspace_bibtex_write` permission and an idempotency key. It validates that each requested key matches the BibTeX entry, skips keys already present, writes through an atomic temporary-file replacement, and verifies appended keys afterward. Repeating the same idempotency key returns the stored successful result without duplicating entries.

The VS Code insertion surface continues to use a reviewed `WorkspaceEdit`, preserving the existing human confirmation step. The backend writer contract is available for future remote or headless execution paths.

## Tool Health

Tool Health now obtains LLM, Zotero, Better BibTeX, PDF, embedding, and BibTeX Writer checks from the Registry. The response includes both user-facing health items and machine-readable tool definitions, so adding a future academic source does not require maintaining a second independent inventory.

## Verification

```bash
.venv/bin/python -m pytest src/test/python_tests
npm run compile
```

The v17 test suite covers success, preflight rejection, timeout, invalid output, connection retry, fallback, permission denial, idempotent writes, capability discovery, health inventory, and existing recommendation regressions.
