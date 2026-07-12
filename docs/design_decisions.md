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

## Phase 7

### ADR-028: Evidence-constrained Writer (no retrieval, no memory fill)

**Decision:** The Writer is a deterministic graph node that receives only the
question, answer format/plan, verified evidence ledger, contradiction notes, and
corpus-insufficiency flags. It never calls retrieval tools. It first emits
structured `ClaimWithCitations` (claim text + evidence IDs), then renders Markdown
with inline page citations derived from those IDs. The Verifier records accepted
evidence IDs per sub-question, and the Writer filters the shared ledger to that
allowlist before forming claims. Each extractive claim is tied to one evidence
item; gaps are stated as limitations rather than filled from model memory. An
optional LLM path may be added later without changing the claim→evidence contract.

**Rationale:** Plan §10.4; Writer is not marketed as a retrieval agent.

### ADR-029: Citation validator repairs before final output

**Decision:** After drafting, `CitationValidator` checks that every citation ID
exists in the ledger, maps exactly to the canonical chunk and paper, falls within
the chunk page range, and points to a readable physical PDF containing that page.
Claim support uses a conservative content-overlap check plus numeric and polarity
consistency. Invalid IDs and unsupported claims are stripped (or explicitly
qualified); references are deduplicated into `SourceCard`s containing titles,
PDF paths, chunks, evidence IDs, and pages. Both a machine-readable
`CitationReport` and user-facing Sources section are emitted, and the workflow
always records `citation_validated` in the execution trace (including on
unanswerable paths).

**Rationale:** Plan §10.5 acceptance — no dangling IDs; every source maps to a
PDF page; unsupported claims are removed or qualified.

---

## Phase 8

### ADR-030: Frozen 50-question split with fingerprint

**Decision:** Evaluation questions live under `data/evaluation/` with a
SHA-256 fingerprint in `frozen_split.json`. Loaders refuse silent drift between
JSONL files and the freeze record. Type mix is fixed (10/10/15/10/5). Gold labels
use stable `paper_id` plus resolved `chunk_id`/pages from the processed corpus at
freeze time (`scripts/build_eval_dataset.py`). Before a run, every gold paper,
chunk, and page is checked against the current canonical store.

**Rationale:** Plan §11.1 and Phase-8 acceptance — identical split for every system.

### ADR-031: Offline-first metrics; optional RAGAS

**Decision:** Default evaluation computes deterministic retrieval (Recall@K, MRR,
nDCG), citation (precision/recall/validity/page traceability), and answer
(token/claim overlap, refusal accuracy, faithfulness proxy) metrics without paid
APIs. RAGAS is an optional extra (`scholar-agent[eval]` + `--ragas`) that degrades
gracefully when unavailable. Reports distinguish requested, available, and
actually-scored RAGAS rows; unavailable values remain null.

**Rationale:** Core tests and CI must stay free of live provider calls.

### ADR-032: Shared SystemRunner for all ablations

**Decision:** `naive_dense`, `hybrid_rag`, `hybrid_rerank`, `hybrid_graph`,
`hybrid_corrective`, `full_agent`, and `static_all_tools` all implement the same
`SystemOutput` contract and are scored by one ablation harness. Latency, tool
counts, token estimates, and optional USD cost are recorded per question×system
and category. Explicit hash evaluation uses an isolated index. Saved run configs
record the actual embedder/reranker, frozen split, selected question IDs, Git
commit/dirty state, and config/code fingerprints. Multi-tool baselines use
deterministic round-robin fusion so later graph results cannot be starved by the
first full top-k list.

**Rationale:** Plan §11.2 systems compared + operational metrics in §11.3.

---

## Phase 9

### ADR-033: Streamlit demo with offline saved-run replay

**Decision:** The interview UI is a Streamlit app (`scholar_agent.app.streamlit_app`)
backed by a non-UI `DemoService`. Live mode runs the full workflow (or static
all-tools ablation) with sidebar toggles for graph/corrective/static routing and
Naive RAG comparison. Replay mode loads fingerprint-stable JSON sessions from
`data/demo/runs/` so demos work without live API access or healthy indexes.

**Rationale:** Plan §12 and Phase-9 acceptance — auditable traces, ablation
toggles, and interview resilience.

### ADR-034: Trace panel is event-derived, not chain-of-thought

**Decision:** The research trace summarizes plan sub-questions, tool events,
coverage, corrective queries/iterations, citation validation, latency, and
token estimates from structured `ExecutionEvent`s and workflow results. No
provider reasoning fields are displayed.

**Rationale:** Plan constraints on no user-facing CoT; still supports “corrective
loops are visibly understandable.”

### ADR-035: Replay provenance is canonical and visually verifiable

**Decision:** Saved source cards must resolve to the canonical chunk store and
physical PDFs, carry the corpus fingerprint, and use snippets present in the
referenced chunk. The source viewer renders the cited page inside Streamlit and
rejects repository-escaping paths. Final claims are grouped with their evidence
IDs and source cards.

**Rationale:** Merely displaying a PDF path does not prove the claim-to-page
trace required by Phase 9; committed replay fixtures must meet the same
provenance standard as live answers.

---


## Phase 10

### ADR-036: Disk cache only for pure offline computations

**Decision:** Introduce `scholar_agent.storage.cache.DiskCache` for deterministic
JSON values (initially graph chunk extraction). Keys include namespace, schema
version, and content hashes. Atomic writes; corrupt entries become misses.
Mutable workflow state and live LLM outputs are never cached.

**Context:** Phase 10 requires caching with tested invalidation. Graph extraction
over thousands of chunks is pure and expensive enough to benefit.

**Alternatives considered:** (a) no cache; (b) cache full workflow runs; (c) SQLite
cache. Selected simple sharded JSON files for inspectability and zero new deps.

**Advantages:** Offline speedups; clear invalidation; easy audit of cache files.

**Trade-offs:** No cross-process locking beyond atomic replace; not a multi-tenant
cache.

**Evidence:** `tests/unit/test_cache.py`; wiring in `graph/pipeline.py` +
`graph/extract.py`. Policy documented in `docs/caching.md`.

**Status:** implemented.

### ADR-037: Explicit untrusted-content delimiters

**Decision:** Retrieved paper text is wrapped in
`<untrusted_retrieved_content>` blocks via `llm/prompts.py`. Nested closers are
neutralized. System policy states paper text cannot change tools or schema.

**Context:** Plan §13 treats paper content as untrusted data.

**Alternatives considered:** Rely only on system prompts without delimiters;
sanitize by stripping all instruction-like sentences (too lossy for evidence).

**Advantages:** Clear data/instruction boundary for LLM call sites; regression
tests for breakout attempts.

**Trade-offs:** Delimiters alone are not a security boundary against a fully
compromised model; still require schema-constrained outputs.

**Evidence:** `tests/unit/test_hardening.py`.

**Status:** implemented.

### ADR-038: Structured errors and observable degradation

**Decision:** Add `StructuredError` / `ErrorCategory`. Toolkit graph search can
`allow_degraded` to return empty hits with `degraded=true` debug. Researcher tool
failures classify errors and always fall back to **empty hits** (never invented
evidence).

**Context:** Phase 10 graceful degradation and observability requirements.

**Advantages:** Callers can see *why* a tool produced nothing; loops continue
under budgets.

**Trade-offs:** Broad catch around toolkit calls remains necessary to keep the
research loop alive; classification + logging prevents “silent success.”

**Evidence:** `tests/unit/test_hardening.py`; researcher `_call_toolkit`.

**Status:** implemented.

### ADR-039: Retry jitter + permanent error classification

**Decision:** Provider retries use exponential backoff with configurable jitter.
HTTP 401/403/400/404/422 and validation errors are not retried.

**Context:** Plan requires transient vs permanent classification and bounded
retries.

**Evidence:** `tests/unit/test_retry.py`.

**Status:** implemented.

### ADR-040: Portfolio docs distinguish measured vs fixture vs unavailable

**Decision:** README and evaluation docs label offline hash results, require
artifact/run IDs for numbers, and never claim production-ready / SOTA. Interview
and demo materials use committed replay fixtures when APIs are unavailable.

**Deviation from plan marketing-friendly language:** explicit refusal of
unverified superlatives (plan §18).

**Status:** implemented.

### ADR-041: Local prototype vs production deployment

**Decision:** Keep ScholarAgent as a single-user local prototype: CLI + Streamlit,
filesystem indexes, env-based secrets. Production multi-user needs (auth, quotas,
managed vector DB, job queues) are documented as future work, not implemented.

**Rationale:** Phase 10 acceptance is portfolio completeness + reliability, not
SaaS readiness.

**Status:** documented.
