# EasyCite v16 Context Engineering

v16 introduces an explicit Context Manager between LangGraph nodes and model-provider calls. The goal is to make model input selection deterministic, budgeted, and observable without changing citation retrieval or evidence-verdict semantics.

## Covered LLM Calls

- Retrieval query planning
- Batched and single-candidate evidence verification
- On-demand bilingual evidence translation

## Context Priorities

- `required`: selected claim, structured-output constraints, claim interpretation, and candidate identity. These fields are never silently dropped.
- `compressible`: paragraph context, local query hints, passage text, author lists, and translation text. Long string leaves can be shortened while their JSON schema remains intact.
- `droppable`: optional low-value context. The current production calls do not depend on droppable data, but the policy is available for later tool observations and memory retrieval.

## Default Budgets

| Stage | Setting | Default |
|---|---|---:|
| Query planning | `citation-agent.contextQueryBudgetTokens` | 3200 |
| Evidence verification | `citation-agent.contextEvidenceBudgetTokens` | 9000 |
| Translation | `citation-agent.contextTranslationBudgetTokens` | 5000 |

The estimator is provider-independent and deterministic. It estimates ASCII prose at roughly four characters per token and counts CJK characters conservatively. It is a budgeting estimate rather than provider billing usage.

## Compaction Rules

1. Apply per-item maximums.
2. Drop only items explicitly marked `droppable` when the total remains over budget.
3. Compact `compressible` items from lowest priority position while respecting `min_tokens` and `min_items`.
4. Preserve protected dictionary keys such as candidate IDs, titles, passage indexes, pages, and source labels.
5. If required content alone exceeds the configured budget, retain it and report `budget_exceeded=true` instead of silently losing the claim.

## Observability

Each real LLM request emits a context report containing:

- stage and configured budget
- estimated tokens before and after compaction
- system prompt estimate
- each context item's source and purpose
- per-item original/final estimate
- kept, compacted, or dropped action plus its reason
- whether required content still exceeded the budget

The LangGraph runtime adds these summaries to Run Trace and persists full reports under `context_engineering` in the Run Record JSON. Trace summaries expose `input_tokens_estimated`, `budget`, `compacted_items`, and `pruned_items`. Cached query plans and cached evidence judgments do not emit a new report because they do not perform a new LLM request.

## Scope Boundary

v16 does not change metadata ranking, candidate confidence, verdict thresholds, Zotero search behavior, or memory ranking. Those boundaries make v15 and v16 recommendation-quality comparisons meaningful: any change should come from more disciplined model context rather than hidden retrieval rules.
