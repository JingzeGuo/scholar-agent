# Interview guide

Concise answers for a junior Agent/RAG engineering interview. Numbers below are
from the **offline hash-embedding** clean run `run_1f4dc371453d4a1f` unless noted.
They are **not** production BGE claims. Full table:
[`docs/results/offline_hash_eval_summary.md`](results/offline_hash_eval_summary.md).

---

## Sixty-second explanation

**Problem.** Literature Q&A needs answers grounded in PDF pages, not fluent but
untraceable chat.

**Baseline failure.** Naive dense retrieval alone reaches only **0.16** paper
Recall@8 on the frozen 50-question split with hashing embeddings; exact names
and acronyms are easy to miss.

**Architecture.** Offline ingestion builds a canonical chunk store, dense + BM25
indexes, and a provenance-backed knowledge graph. Online, a Planner decomposes
questions; a Researcher runs a budgeted tool loop; a Verifier requests corrective
retrieval; a Writer answers only from verified evidence; a citation validator
checks chunk/page provenance.

**Three technical decisions.** (1) Canonical chunk store as the single source of
truth for every index. (2) Explicit hybrid RRF + adaptive routing instead of a
fixed retrieve→generate chain. (3) Separate Verifier and corrective budgets so
loops terminate measurably.

**Strongest measured result (this config).** Hybrid retrieval lifts paper
Recall@8 from **0.16** (dense) to **0.61**; adding graph reaches **0.67**.

**Main limitation.** Graph multiplies latency (~11 ms → ~290 ms). Corrective
loops trigger safely but show **0.0** gold-recall improvement offline. The 50-Q
labels are AI-assisted reviewed, not human-signed.

---

## Five-minute architecture walkthrough

1. **Offline ingestion** — PDFs → page text → sections → token-aware chunks with
   stable IDs (`scholar-agent ingest`).
2. **Canonical chunk store** — `ChunkStore` is source of truth; indexes must align
   on `chunk_id`.
3. **Dense + sparse indexes** — Chroma (BGE or hash embedder) + persistent BM25;
   fusion via explicit RRF; optional cross-encoder / lexical rerank.
4. **Graph construction** — constrained ontology, evidence spans, staged entity
   resolution, NetworkX MultiDiGraph.
5. **Query planning** — structured `QueryPlan` / sub-questions (no free-form plan).
6. **Retrieval routing** — rule-based policy (dense / sparse / hybrid / graph).
7. **Research Agent tool loop** — budgeted tool calls, evidence ledger merge.
8. **Verification loop** — coverage, conflicts, corrective queries, hard stop
   conditions.
9. **Citation validation** — claims must cite ledger evidence with real pages.
10. **Evaluation** — frozen 50-Q split, shared ablations, deterministic metrics.

---

## Detailed Q&A

### 1. Why LangGraph instead of a simple chain?

A chain is retrieve→generate once. We need **conditional edges**: plan → research
→ verify → either correct or write → finish. LangGraph makes budgets, iteration
state, and termination reasons first-class. The Phase 0 prototype already proves
a decide/retrieve/verify loop with a fake model offline.

### 2. What makes this system agentic?

The Researcher **selects tools** based on routing (not a fixed sequence), the
Verifier **decides** whether to re-retrieve, and the workflow **terminates** on
sufficiency, budgets, or no-new-evidence. Actions are recorded as structured
events, not free-form chain-of-thought.

### 3. Why separate Researcher and Verifier?

Mixing gathering and judging in one prompt encourages confirmation bias and
endless tool use. Separation lets the Researcher maximize useful evidence under
budgets while the Verifier scores coverage/conflicts and emits **targeted**
corrective queries. Writing is further separated so generation cannot retrieve
new (unverified) text.

### 4. When does BM25 outperform dense retrieval?

Exact terminology, dataset names, acronyms, and rare identifiers. On this offline
run, hybrid (dense+BM25) paper Recall@8 is **0.61** vs dense **0.16**. Failure
case `q_k08` (RAGAs) shows dense 0.0 vs hybrid 1.0 paper recall for that item.

### 5. How does Reciprocal Rank Fusion work?

For each document, \( \mathrm{RRF}(d)=\sum_i 1/(k+\mathrm{rank}_i(d)) \) with
default \(k=60\). Rank lists (dense, BM25) are fused without score calibration.
Implementation: `scholar_agent.retrieval.fusion.reciprocal_rank_fusion`.

### 6. Why use a cross-encoder after retrieval?

Bi-encoders retrieve cheaply; a cross-encoder (or lexical stand-in offline)
re-scores a small top-k with full query–passage interaction. Production uses
`cross-encoder/ms-marco-MiniLM-L-6-v2`; hash eval uses lexical overlap so CI stays
offline.

### 7. When is GraphRAG useful, and when does it hurt?

**Useful:** multi-hop / relational questions where entities connect across papers
(aggregate paper R@8 **0.67** for hybrid_graph vs **0.61** hybrid_rerank).  
**Hurts:** simple factual queries where graph fan-out displaces good hybrid hits
and adds ~300 ms latency; graph-evidence recall was only **0.056** overall.

### 8. How are aliases and duplicate graph entities resolved?

Staged resolver: seed acronym/alias map → string/embedding candidates → optional
LLM judge. Surfaces register into canonical entities; relation IDs include
canonical entity IDs + evidence span.

### 9. How does graph evidence map back to a PDF page?

Every relation stores `chunk_id` + `evidence_span`. Chunks carry `page_start` /
`page_end` and `paper_id`. The store never treats a triple as a fact without that
join. Demo source cards show claim → evidence → chunk → PDF page.

### 10. How do reducers prevent duplicate state during parallel execution?

Evidence ledger merge uses deterministic dedupe keys (`dedupe_key` on chunk /
text). Parallel sub-question results are re-ordered by original plan order before
merge so concurrent completion cannot reorder state non-deterministically.

### 11. How does the corrective loop terminate?

Hard stops: evidence sufficient; max corrective iterations; no new evidence IDs;
global tool/token/latency budgets; unanswerable after targeted exhaustion. Tests
in `tests/unit/test_workflow.py` cover these paths.

### 12. How are unsupported citations detected?

The citation validator checks each claim’s evidence IDs against the ledger,
canonical chunk store, optional physical PDF page bounds, and claim–evidence
token support. Unsupported claims are removed or flagged; invented IDs fail.

### 13. How was the evaluation dataset constructed and frozen?

50 manually curated questions (10 factual / 10 keyword / 15 comparison /
10 relational / 5 unanswerable) with gold papers/chunks. Fingerprint in
`data/evaluation/frozen_split.json` rejects silent JSONL drift.

### 14. Which ablation helped the most, and for which category?

Largest jump: **dense → hybrid** on aggregate paper recall (**0.16 → 0.61**),
especially factual/keyword (**0.20/0.20 → 0.90/0.90**). Graph adds a further lift
on comparison (**0.57 → 0.67** paper R@8). Full agent citation precision is best
(**0.288**) but not best recall.

### 15. Which module failed to help or caused regressions?

- **Corrective improvement**: triggers with precision 1.0 but **0.0** gold-recall gain offline.
- **Graph** can reduce paper recall on some comparison items vs pure hybrid while
  increasing latency.
- **Lexical rerank** vs plain hybrid_rag is nearly tied under hash embeddings
  (both paper R@8 **0.61**).

### 16. What must change for multi-user production deployment?

AuthN/Z and per-user quotas; multi-tenant index isolation; durable job queue for
ingestion; production vector DB; secret management; request tracing (no CoT
leakage); horizontal Streamlit/API servers; stronger refusal/entailment models;
cache isolation; rate-limit budgets; and larger out-of-domain evaluation beyond
the measured hash and BGE/cross-encoder frozen-split runs.

---

## Three strong failure stories

Use the full write-ups in [`docs/failure_analysis.md`](failure_analysis.md):

1. **Dense misses exact product name** (`q_k08`) → hybrid recovers.
2. **Graph helps recall but costs latency** (`q_r09`) → selective routing.
3. **Corrective loop triggers without recall gain** → termination is correct,
   but gap-to-query quality remains open work.

---

## Major trade-offs

| Trade-off | Choice | Cost |
|---|---|---|
| Deterministic offline eval | Hash embedder + lexical rerank | Not production retrieval quality |
| Graph expansion | Higher paper recall | ~30× latency in measured run |
| Separate Verifier | Safer loops | Extra pass / complexity |
| Evidence-only Writer | Fewer hallucinations | May under-answer |
| Fake-model unit tests | Free CI | Live LLM behavior is opt-in |

---

## Project limitations

- Offline hash metrics ≠ BGE/cross-encoder production numbers.
- Static baselines have weak refusal behavior; the full agent refused all 5/5
  unanswerable items in the measured offline run.
- Graph ontology is constrained and heuristic extraction is imperfect.
- PDFs and indexes are local artifacts (not committed).
- Streamlit is a single-user demo, not a multi-tenant product.

## Likely follow-ups

- “Show me the LangGraph graph definition.”
- “How do you stop infinite tool use?”
- “Where is the fingerprint checked?”
- “What would you A/B next with a real embedder?”
- “How do you prevent prompt injection from PDFs?”
