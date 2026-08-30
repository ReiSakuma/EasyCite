# v20 Durable Recovery Technical Design

v20 adds a project-private SQLite Run Store. The LangGraph runner writes a
compact checkpoint after every node, including the graph/schema version,
completed nodes, request hash, candidate counts and cancellation state. The
snapshot deliberately excludes API keys, full PDF text and unbounded prompts.

Recovery policy is explicit: pure computation nodes are replayed, read-only
retrieval and verification results may be reused, and write-oriented finalization
is restarted after an idempotency check. A graph or schema mismatch, or a user
cancellation, rejects automatic recovery instead of silently mixing versions.

Useful smoke checks:

```bash
.venv/bin/python -m pytest src/test/python_tests/test_checkpoint_v20.py
```

The existing run record remains the user-facing summary; the Run Store is the
durable execution layer. These are intentionally separate so a display record
cannot be mistaken for a resumable state machine.
