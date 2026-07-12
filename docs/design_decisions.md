# Design Decisions

This document records intentional deviations from `CODEX_IMPLEMENTATION_PLAN.md`
and key architectural choices. Update it whenever implementation diverges from
the plan or a non-obvious trade-off is made.

## Phase 0

### ADR-001: Package layout and dependency staging

**Decision:** Ship a minimal dependency set in the core package for Phase 0
(`pydantic`, `langgraph`, `openai`, `typer`, …). Heavy stacks (Chroma,
sentence-transformers, PyMuPDF, NetworkX, Streamlit, RAGAS) are declared as
optional extras (`retrieval`, `ui`, `eval`) and installed when later phases need
them.

**Rationale:** Keeps the compatibility spike and CI fast; avoids downloading
embedding models and PDF stacks before ingestion exists.

**Deviation:** The plan lists all technologies in one table; it does not require
installing every runtime dependency on day zero.

### ADR-002: Direct OpenAI-compatible client as primary LLM surface

**Decision:** Implement `scholar_agent.llm.client.LLMClient` on the official
`openai` Python SDK pointed at DeepSeek’s OpenAI-compatible endpoint. LangChain
chat models may wrap this later where LangGraph tool binding benefits, but the
canonical low-level surface is the thin client.

**Rationale:** The plan requires a compatibility spike for structured JSON, tool
calling, streaming, and reasoning-field handling. Provider-specific fields are
easier to inspect and sanitize with a thin wrapper than through a full LangChain
abstraction stack.

### ADR-003: Prototype loop uses a deterministic fake model

**Decision:** The Phase 0 LangGraph conditional loop (`agents/prototype_loop.py`)
uses `FakeResearchModel` instead of live DeepSeek calls.

**Rationale:** Acceptance requires “one conditional loop runs with a
deterministic fake model.” Offline CI must not depend on paid APIs. Live
provider verification is isolated in `scripts/deepseek_compatibility.py` and
optional `pytest -m live` tests.

### ADR-004: Thinking mode default off for structured tasks

**Decision:** Default `llm.thinking_enabled: false`. The client sends a soft
`extra_body` hint to disable thinking when the provider supports it. Reasoning
fields are extracted if present but never exposed as user-facing chain-of-thought.

**Rationale:** Plan guidance: use flash / non-thinking for extraction and
classification unless testing shows a clear benefit.

### ADR-005: Config is YAML + env, validated by Pydantic

**Decision:** `configs/default.yaml` holds non-secret defaults; `EnvSettings`
overrides API keys and hot budgets. `load_config()` validates with Pydantic v2
and resolves paths against the repository root.

**Rationale:** Startup validation is a reliability requirement (plan §13). Secrets
never live in committed YAML.

### ADR-006: Execution events are structured, not free-form CoT

**Decision:** `ExecutionEvent` records component, event type, short summary, and
JSON-serializable payload. Private chain-of-thought is never stored.

**Rationale:** Plan requires auditable traces without exposing CoT.

### ADR-007: Prototype loop merges events explicitly

**Decision:** The Phase 0 LangGraph prototype appends `events` inside each node
via `_append_events` rather than relying on `Annotated[..., reducer]` in the
prototype state TypedDict. Reducer helpers still live in `agents/state.py` for
later workflow state.

**Rationale:** Keeps the spike easy to type-check under current LangGraph stubs
while still demonstrating append semantics and budget-aware termination.

### ADR-008: Explicit, testable provider retries

**Decision:** Disable the OpenAI SDK's implicit retries and route completion creation through a
bounded project retry helper. Retry HTTP 429, HTTP 5xx, connection failures, and timeouts with
exponential backoff. Retry malformed structured output by making a fresh bounded request.

**Rationale:** Phase 0 requires retry behavior to be verified. An explicit policy can be tested
deterministically without manufacturing paid provider failures.

### ADR-009: Strict and auditable compatibility acceptance

**Decision:** A missing or malformed tool call fails the compatibility spike. Thinking-mode
errors also fail instead of becoming soft passes. Every run writes a secret-free JSON report;
generated reports remain ignored under `outputs/`.

**Rationale:** Successful HTTP requests are not evidence that provider features work. Phase
acceptance needs a reproducible result for every required capability.

---

## Phase 1

### ADR-010: Content-addressed IDs with arXiv/DOI preference

**Decision:** `scholar_agent.ids` generates stable paper IDs preferring
normalized arXiv IDs, then DOIs, then title+year hashes. Chunk, entity,
relation, and within-run evidence IDs are SHA-256–based compositions of their
identity fields. Run IDs remain random UUIDs for audit uniqueness.

**Rationale:** Re-ingestion and evidence reconstruction must not churn IDs used
by indexes, graphs, and citation trails.

### ADR-011: Models package + JSONL as canonical persistence

**Decision:** Domain models live under `scholar_agent.models` (package). Papers,
chunks, entities, relations, and the corpus manifest persist as typed JSONL via
`JsonlRepository`. Invalid lines fail with path+line diagnostics.

**Rationale:** The plan requires Pydantic at module boundaries and a canonical
chunk store as source of truth for every index. JSONL is portable, diffable, and
easy to fixture in tests.

### ADR-012: Evidence ledger dedupes by chunk + normalized span

**Decision:** `EvidenceLedger.merge` and the `merge_evidence` LangGraph reducer
deduplicate on `(chunk_id, normalize_text(evidence_text))`, keeping the higher
score and preserving contradiction flags.

**Rationale:** Plan §7.3 — avoid duplicate retrieval pollution without dropping
stronger scores.

---

## Phase 2

### ADR-013: PyMuPDF page-preserving extraction; OCR out of scope

**Decision:** Use PyMuPDF (`fitz`) as the only PDF parser. Extract text per page
with 1-indexed page numbers. Flag empty and scan-like pages; never index an empty
paper. Complex table/formula structure is out of scope.

**Rationale:** Plan §8.1. OCR would expand scope and dependencies without
improving the interview narrative for text-native arXiv PDFs.

### ADR-014: Token-aware chunking via tiktoken

**Decision:** Chunk sizes are measured with `tiktoken` (`cl100k_base`), not
character counts. Defaults: target 600 tokens, overlap 80, min 80. Prefer
section boundaries; split long sections with overlapping token windows.

**Rationale:** Plan §8.2 explicitly forbids marketing character splits as token
splits.

### ADR-015: Idempotent ingest keyed by PDF content hash

**Decision:** Skip re-processing when `Paper.content_hash` matches the on-disk
PDF hash and chunks already exist for that `paper_id`. `--force` rebuilds.
Rebuilt chunks keep stable IDs when text is unchanged.

**Rationale:** Plan acceptance requires idempotent duplicate ingestion.

---

## Phase 3

### ADR-016: Canonical chunk store is the only index source

**Decision:** Dense (Chroma) and BM25 indexes are built only from
`data/processed/chunks.jsonl`. Both persist a `corpus_fingerprint` of sorted
`chunk_id:content_hash` pairs and refuse to load on mismatch.

**Rationale:** Plan §8.3–8.4 — every index must share stable chunk IDs with the
canonical store.

### ADR-017: Explicit RRF + injectable embedder/reranker

**Decision:** Hybrid fusion is hand-written Reciprocal Rank Fusion
(`retrieval/fusion.py`). Embeddings default to sentence-transformers BGE, but
tests and offline CI use `HashingEmbedder` + `LexicalReranker` via
`--embedding-backend hash`.

**Rationale:** Keeps fusion auditable; avoids paid/model downloads in unit tests
while still supporting production models.

### ADR-018: Naive RAG baseline is extractive by default

**Decision:** `ask-naive` produces an evidence-backed answer with `[paper_id p.N]`
citations without requiring an LLM. Optional `--llm` uses DeepSeek when configured.

**Rationale:** Acceptance requires page references in baseline answers; offline
reproducibility must not depend on live APIs.

### ADR-019: Project-local production model cache

**Decision:** BGE and CrossEncoder use a shared cache under the repository's ignored
`.cache/huggingface/hub` directory by default. `SCHOLAR_MODEL_CACHE` and `HF_HOME` can override
the location. The production path was validated on all 5,858 canonical chunks with
`BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Rationale:** A deterministic project-local cache makes the production retrieval path reusable
across CLI commands without manual `HF_HOME` configuration, while keeping model weights out of
Git. Offline CI continues to use the lightweight hash/lexical backends.

---

## Phase 4

### ADR-019: Offline heuristic extraction first; LLM optional later

**Decision:** Default graph extraction is schema-constrained and heuristic
(seed aliases + cue patterns + co-occurrence), fully offline. Relations are
discarded unless `evidence_span` is grounded in the source chunk. Optional LLM
extraction prompt is documented but not required for acceptance.

**Rationale:** Phase 4 acceptance must not depend on paid APIs; grounding is
non-negotiable.

### ADR-020: Staged entity resolver with bounded LLM arbitration

**Decision:** Resolver stages: normalize → acronym map → exact/seed alias →
string candidate retrieval → embedding similarity. Deterministic high-confidence
matches avoid model calls; ambiguous residual candidates may use DeepSeek through
`graph build --llm-resolution`, capped by `--max-llm-resolutions`.

**Rationale:** Plan §8.6 staged design; alias fixtures must resolve
deterministically for tests and demos, while difficult variants still have a
schema-constrained adjudication path.

### ADR-021: Node-link JSON MultiDiGraph; triples never standalone facts

**Decision:** Persist NetworkX `MultiDiGraph` as portable node-link JSON.
`graph_search` returns supporting chunks/pages along paths (≤2 hops), not bare
triples. Ranking combines query relevance, relation confidence, evidence quality,
query-entity path coverage, and a hop penalty. Longest-span entity linking avoids
matching `RAG` inside `Self-RAG`; low-relevance tails are filtered.

**Rationale:** Plan §8.7 and §9.4.

---

## Phase 5

### ADR-022: Rule-based router offline; agent may override within budget

**Decision:** Query classification and default retrieval policy use deterministic
rules (acronyms, comparison/relation/synthesis cues, seed entity lexicon). The
Research Agent logs both the recommendation and any complementary-tool override.

**Rationale:** Acceptance requires different tools for labeled query types
without live LLM dependency.

### ADR-023: Hard execution budgets in the Research Agent

**Decision:** Per sub-question tool-call, inspect/act iteration, evidence, and
latency caps are enforced in code paths, not prompts. Exceeding a cap emits a
`BUDGET_HIT` execution event and stops the loop. Retrieval itself makes no LLM
call, so token consumption remains governed by the global workflow budget.

**Rationale:** Plan §10.2 safety budgets; tests assert budgets cannot be
exceeded.

### ADR-024: Parallel sub-question research with deterministic merge

**Decision:** Pending sub-questions with unique IDs and normalized question text
may run via a bounded thread pool. Duplicate or non-pending work falls back to
serial execution. Evidence is merged with the existing chunk+span ledger reducer
in original sub-question order.

**Rationale:** Plan allows fan-out where safe; reducers must remain deterministic.

---

## Phase 6

### ADR-025: Offline structured Planner (no free-form plans)

**Decision:** Planner always returns a Pydantic `QueryPlan`. Decomposition is
type-driven: simple factual → 1 sub-question; comparison/synthesis → several.
LLM planning is optional later and not required for acceptance.

### ADR-026: Independent Verifier with concrete corrective queries

**Decision:** Verifier only sees query, plan, and evidence ledger. It returns
coverage, conflicts, missing aspects, and **actionable** corrective queries—not
only `is_sufficient=false`. Each corrective action carries its target
sub-question ID and missing aspect so retrieved evidence repairs the original
plan gap. Contradictions are retained and listed by evidence ID.

### ADR-027: Corrective loop termination is exhaustive

**Decision:** The LangGraph workflow stops on sufficient evidence, iteration
budget, global tool budget, latency budget, no-new-evidence, unanswerable corpus,
or an empty corrective-action list. Empty or irrelevant first-pass results cause
targeted retrieval; the corpus is marked unanswerable only after corrective
retrieval also fails. Every path emits `run_finished` with a reason, and budget
terminations emit `BUDGET_HIT`.

---

## Pending (later phases)

- Writer and citation validator (Phase 7)
