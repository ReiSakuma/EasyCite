# v18 Memory Lifecycle Technical Design

v18 turns feedback memory into a governed Memory Service. The service stores
records in the project-private SQLite database and separates `session`,
`project`, `user`, and `bad_case` scopes. Every record carries provenance,
confidence, schema version, lifecycle status, optional expiry, and a correction
pointer.

## Runtime rules

- Only active, non-expired records are returned by `recall`.
- The workspace hash is part of every query, preventing cross-project leakage.
- A correction supersedes the old record and keeps an audit chain.
- Existing feedback APIs remain compatible and mirror explicit feedback into a
  low-confidence project record.
- Bad cases are stored for evaluation and diagnostics; they do not directly
  alter ordinary ranking.

The agent state now exposes memory schema and counts, while secrets, PDF text,
and unbounded prompts are not persisted by this service.
