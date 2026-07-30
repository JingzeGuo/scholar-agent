# ScholarAgent interview refactor plan

The current target is:

> A compact multi-agent GraphRAG system for evidence-grounded academic research.

Each phase is dependency ordered. A phase is complete only after its acceptance
checks have objective evidence.

## Phase 0 — Scope and architecture

- Reduce the design to four agents and one `AgentState`.
- Keep BM25, dense retrieval, one-hop entity GraphRAG, RRF, cross-encoder
  reranking, verification, and page-aware citations.
- Remove compatibility requirements for the previous production-style API.

Acceptance:

- The implementation plan names exactly four workflow nodes.
- No old API compatibility layer is planned.
- The working tree is clean before the refactor starts.

## Phase 1 — Page-aware corpus and indexes

- Extract PDF text page by page with PyMuPDF.
- Split within physical-page boundaries at roughly 1,200 characters with
  roughly 150 characters of overlap.
- Persist plain JSONL chunks.
- Build BM25, NumPy dense embeddings, and a NetworkX entity co-occurrence graph.

Acceptance:

- Every chunk has `chunk_id`, `paper`, `page`, and `text`.
- Tests prove page provenance and no cross-page chunks.
- BM25, dense, and graph retrieval each return relevant chunks.

## Phase 2 — Retrieval and four-agent workflow

- Fuse the three retriever rankings with reciprocal rank fusion.
- Rerank up to 30 candidates and select up to eight evidence chunks.
- Implement Planner, Researcher, Verifier, and Writer as small node functions.
- Compile a four-node LangGraph with at most one corrective retrieval.

Acceptance:

- The only LangGraph state is `AgentState`.
- The compiled graph completes on sufficient and insufficient evidence.
- A test proves that corrective retrieval cannot run more than once.

## Phase 3 — Citations and CLI

- Generate evidence-only drafts with `[E1]` references.
- Deterministically discard nonexistent evidence IDs and render real
  `[paper.pdf p.N]` citations.
- Expose only `ingest`, `index`, `ask`, and `demo`.
- Include small page-aware PDF fixtures for offline demonstration.

Acceptance:

- Fake citations are removed.
- Final citations use filenames and physical pages found in stored chunks.
- All four CLI commands run without paid APIs.

## Phase 4 — Focused deterministic tests

- Replace legacy architecture tests with 12–18 behavior-focused tests.
- Keep any provider-dependent test behind the `live` marker.

Acceptance:

- `uv run pytest -q` passes deterministically.
- Tests cover ingestion, all retrievers, RRF, reranking, retry bounds,
  workflow completion, and citation validation.

## Phase 5 — Repository simplification and documentation

- Remove legacy model, storage, evaluation, UI, cache, routing, factory, and
  observability modules.
- Keep a flat, interview-readable package with no source file over 200 lines.
- Rewrite README to at most about 250 lines.

Acceptance:

- `src/` is about 1,200–1,800 lines and every source file is at most 200 lines.
- README accurately says: “Lightweight GraphRAG using entity co-occurrence and
  one-hop neighborhood expansion.”
- No unused production architecture remains on the main path.

## Phase 6 — Final acceptance and commit

Run:

```bash
uv sync
uv run scholar-agent ingest tests/fixtures/papers
uv run scholar-agent index
uv run scholar-agent ask "Compare Self-RAG and CRAG"
uv run pytest -q
uv run ruff check .
make quality
UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv lock --check
```

Acceptance:

- The CLI trace shows all four agents, retrieval result counts, fused candidate
  count, selected evidence, a real fixture filename, and a physical page.
- The answer contains no fabricated citation.
- All checks pass and the finished refactor is committed to Git.
