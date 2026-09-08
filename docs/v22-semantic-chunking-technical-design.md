# v22 Paragraph-aware Semantic Chunking

## Goal

v21 preserved section labels but flattened natural paragraphs into sentence windows before creating parents. A parent could therefore contain two unrelated topics that happened to occur in the same section. v22 preserves paragraph identity and detects topic changes before child chunks are produced.

## Pipeline

The PDF path is processed as `Section -> Paragraph -> Semantic Parent -> Sentence -> Child`.

1. PyMuPDF extracts text blocks with page coordinates.
2. Columns, vertical gaps and first-line indentation rebuild logical paragraphs without joining two-column text.
3. Headings create hard section boundaries, references are excluded, and paragraphs receive stable ordinals.
4. Short layout fragments are attached to nearby prose for semantic encoding while searchable source text remains unchanged.
5. The local embedding model encodes all semantic inputs once per document, then vectors are mapped back to sections.
6. Each possible boundary compares adjacent paragraphs and the mean context on both sides.
7. A semantic or size boundary closes the parent.
8. Sentence-aware child chunks are built only inside that parent, so overlap cannot leak across topics.

## Boundary policy

The boundary score is `0.4 * adjacent_similarity + 0.6 * bilateral_context_similarity`, using two paragraphs on each side by default.

- `score < 0.30`: create a hard semantic boundary even when the parent is short.
- `score < 0.45` after 180 tokens: create a soft semantic boundary.
- after 180 tokens, also accept a local similarity valley below the section's 20th percentile (capped at 0.72) when it drops at least 0.08 relative to its neighbors.
- adding the next paragraph would exceed 900 tokens: create a size boundary.
- a section change always creates a boundary.

The thresholds are workspace settings. Higher thresholds create smaller, more numerous parents.

## Fallback

Semantic splitting reuses the same BGE/FastEmbed model as Dense retrieval. If the model is disabled or unavailable, indexing continues with section, paragraph and size boundaries. BM25 retrieval remains usable, and Trace reports the fallback reason.

## Persistence and observability

Each stored child records its paragraph range, parent boundary reason and optional boundary score. The chunker version participates in the document fingerprint, so older rows are rebuilt incrementally when encountered. The `rag_semantic_chunking` Trace stage reports reconstructed paragraphs, compacted embedding inputs, parents, adaptive/total semantic boundaries, size boundaries, latency and fallback state.

## Acceptance checks

- No parent crosses a section boundary.
- A strong topic shift creates a semantic boundary.
- Related adjacent paragraphs remain in the same parent until the size ceiling.
- Child overlap never crosses a semantic parent.
- Missing embeddings degrade to paragraph-aware size splitting without aborting retrieval.
- Unchanged inputs produce stable parent and child identifiers.
