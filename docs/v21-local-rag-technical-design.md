# v21 Local RAG Technical Design

## Goal

v21 replaces query-only Zotero metadata search with a standard local RAG pipeline. The retrieval corpus is determined only by the current TeX bibliography and the user-selected Zotero Collection scope. Debug targets remain diagnostic-only and never alter the main candidate set or rank.

## Pipeline

1. Synchronize lightweight metadata and abstracts for citable Zotero items in the selected Collection and entries in the current project bibliography.
2. Deduplicate by DOI/title while merging the trusted citation key, abstract and PDF path.
3. Parse available PDFs by page and section. Stop before References/Bibliography.
4. Build 900-token parent chunks and approximately 280-token child chunks with 48-token sentence-aware overlap.
5. Persist metadata, abstract chunks, vectors and fingerprints in workspace-private SQLite/FTS5.
6. Run paper-level BM25 and Dense retrieval and fuse rankings with weighted RRF.
7. Lazily resolve and incrementally index PDFs for at most the top 12 paper-level finalists per foreground query.
8. Run passage-level BM25 and Dense retrieval within the retained papers and fuse again with RRF.
9. Rerank a bounded passage set with a local Cross-Encoder, then apply per-paper diversity.
10. Pass only the selected evidence passages, pages and sections to the existing LLM evidence verifier.

## Incremental Indexing

The document fingerprint includes normalized metadata, PDF size/mtime, chunker version and embedding model. Unchanged papers reuse their stored chunks and vectors. A normal force-refresh reruns query planning and retrieval but does not rebuild unchanged documents.

The default index is `.citation-agent/rag.sqlite3`, which is excluded from Git. It can be moved with `citation-agent.ragIndexPath`.

## Fallbacks

- Embedding unavailable: paper and passage retrieval continue with SQLite FTS5 BM25.
- Cross-Encoder unavailable: a deterministic lexical, concept-coverage and quality reranker is used.
- RAG index or corpus sync failure: the legacy metadata retriever runs and the fallback is visible in Trace.
- PDF unavailable: abstract and metadata chunks remain searchable.

## Observability

Trace contains separate `rag_index_sync`, `rag_paper_hybrid_retrieval`, `rag_pdf_index`, `rag_passage_hybrid_retrieval`, `rag_reranker` and `rag_total` stages. Each stage records corpus size, lexical/dense/fused counts, index reuse, real latency and fallback reasons.

## Capacity

The default `ragMaxDocuments=500` covers a 100-paper Collection with room for growth. Initial ingestion cost is proportional to changed PDFs; subsequent runs reuse the index. Exact cosine search is sufficient at this scale and avoids adding a vector database service. A future large-library backend can replace the dense-search implementation without changing the LangGraph node contract.
