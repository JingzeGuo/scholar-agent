# Failure analysis

Evidence source: complete offline clean run **`run_7ef8e4f006d7449d`**
(50 frozen questions × 7 systems).

| Field | Value |
|---|---|
| Dataset fingerprint | `9382ed8119ad4072c3ef45fd6cc5b64b20d0ea91c09271ace302e6105c175d8e` |
| Local artifact | `outputs/evaluation/phase8-final-clean/` (gitignored) |
| Summary snapshot | [`docs/results/offline_hash_eval_summary.md`](results/offline_hash_eval_summary.md) |
| Config | `hashing-embedder-v1`, lexical rerank, no live LLM/RAGAS, cost $0.00 |

These are **real measured cases** from that run, not hypotheticals. They describe
this configuration only.

---

## F1 — Dense retrieval misses exact terminology (`q_k08`)

| Field | Detail |
|---|---|
| **Scenario** | Keyword question: “What does the RAGAs framework evaluate?” |
| **Observable symptom** | `naive_dense` paper Recall@8 = **0.0**; `hybrid_rerank` = **1.0** |
| **Affected component** | Dense retrieval / hashing embedder |
| **Reproduction** | Offline eval row `q_k08` in `per_question_metrics.csv` of run `run_7ef8e4f006d7449d` |
| **Root cause** | Hash embedding is weak for exact product/framework names; BM25 supplies the exact-token signal. |
| **Fix** | Keep hybrid (dense+BM25 RRF) as the default keyword/hybrid route; do not present dense-only as production quality. |
| **Verification after fix** | Hybrid systems recover paper recall on keyword slice to **0.90** aggregate. |
| **Remaining limitation** | Production BGE quality is **not measured** in this offline run. |
| **Interview lesson** | Sparse retrieval is not obsolete; hybrid fusion is a measured necessity for exact terms. |

---

## F2 — Graph helps a relational case but is expensive (`q_r09`)

| Field | Detail |
|---|---|
| **Scenario** | Relational: hallucination surveys → automated RAG evaluation metrics (RAGAs). |
| **Observable symptom** | `hybrid_rerank` paper R@8 = **0.0**; `hybrid_graph` = **0.5**. Aggregate hybrid latency **~10.1 ms** vs hybrid_graph **~290.9 ms**. |
| **Affected component** | Graph retrieval / path ranking |
| **Reproduction** | `q_r09` in the same run; aggregate latencies in `aggregate_metrics.csv`. |
| **Root cause** | Graph paths recover a related gold paper outside hybrid top-k, but traversal dominates CPU time under this index. |
| **Fix** | Route graph only for relational/multi-hop policies; keep hybrid-only for simple factual. |
| **Verification after fix** | Router policies restrict graph; measured graph-evidence recall overall only **0.0556** — selective use is justified. |
| **Remaining limitation** | Score-calibrated fusion of graph vs hybrid is still future work (round-robin can still hurt; see F3). |
| **Interview lesson** | GraphRAG is a tool with a latency tax, not a free upgrade. |

---

## F3 — Graph fan-out displaces good hybrid evidence (`q_c15`)

| Field | Detail |
|---|---|
| **Scenario** | Comparison: RETRO vs Atlas. |
| **Observable symptom** | `hybrid_rerank` paper R@8 = **1.0**; round-robin `hybrid_graph` = **0.5**. |
| **Affected component** | Multi-tool merge / graph slot allocation |
| **Reproduction** | `q_c15` per-question metrics in run `run_7ef8e4f006d7449d`. |
| **Root cause** | Reserving top-k slots for graph hits improves tool diversity but can evict a relevant hybrid hit. |
| **Fix (partial)** | Documented as design risk; adaptive slot allocation deferred. Prefer selective graph routing. |
| **Verification after fix** | Aggregate hybrid_graph paper R@8 still **0.67** (best), but per-question regressions remain. |
| **Remaining limitation** | No calibrated score fusion yet. |
| **Interview lesson** | More tools ≠ better ranking without careful fusion. |

---

## F4 — Planning helps a comparison but can lower aggregate recall (`q_c05`)

| Field | Detail |
|---|---|
| **Scenario** | Comparison: Toolformer vs ReAct. |
| **Observable symptom** | `hybrid_rerank` paper R@8 = **0.5**; `full_agent` = **1.0**. Yet aggregate paper recall is **0.61** (hybrid_rerank) vs **0.52** (full_agent). |
| **Affected component** | Planner decomposition + evidence budgets |
| **Reproduction** | `q_c05` plus system aggregates in the same run. |
| **Root cause** | Decomposition can retrieve both comparison sides, while per-sub-question caps and ledger ordering omit gold chunks on simpler questions. |
| **Fix** | Report **per-category** and per-question gains; avoid claiming universal agent wins from one comparison. |
| **Verification after fix** | Evaluation report always includes by-type tables; documented in `docs/evaluation.md`. |
| **Remaining limitation** | Evidence-budget allocation heuristics still manual. |
| **Interview lesson** | Always pair micro case studies with macro metrics. |

---

## F5 — Unanswerable detection fails for the agent (`q_u01`)

| Field | Detail |
|---|---|
| **Scenario** | Unanswerable private GPU price schedule question. |
| **Observable symptom** | `full_agent` unanswerable-slice refusal = **0/5** overall; answers from weak neighboring evidence. |
| **Affected component** | Verifier / Writer refusal path |
| **Reproduction** | `by_type.unanswerable.refusal_correct` for `full_agent` = **0.0** in results JSON. |
| **Root cause** | Lexical overlap with topical terms (OpenAI/GPT) can satisfy the deterministic Verifier without true answerability. |
| **Fix (partial)** | Writer marks corpus insufficiency when verification says unanswerable; stronger entailment checks not yet added (must not tune on frozen eval). |
| **Verification after fix** | Still **0.0** refusal on unanswerable for full_agent in this run — **open defect**. |
| **Remaining limitation** | Needs a development set for refusal training/heuristics separate from the frozen split. |
| **Interview lesson** | Citation validity ≠ answerability; refuse is a first-class product requirement. |

---

## F6 — Scope evidence helps baselines refuse while the agent over-accepts (`q_u03`)

| Field | Detail |
|---|---|
| **Scenario** | Pediatric amoxicillin dosage “according to Self-RAG” (domain mismatch). |
| **Observable symptom** | `naive_dense` / `hybrid_rerank` refusal score **1.0**; `full_agent` **0.0**. |
| **Affected component** | Agent evidence acceptance / scope modeling |
| **Reproduction** | `q_u03` in the same run. |
| **Root cause** | Gold Self-RAG title page is scope evidence of mismatch; agent treats method-name match as positive answer evidence. |
| **Fix (planned)** | Model “scope evidence” separately from “answer evidence” in a future dataset schema. |
| **Verification after fix** | Not fixed in this run; documented limitation. |
| **Remaining limitation** | Schema + verifier change required. |
| **Interview lesson** | Matching a method name is not answering a medical dosing question. |

---

## Aggregate interpretation

- Best paper Recall@8: **`hybrid_graph` 0.67** (latency **290.9 ms**).
- Best citation precision among systems: **`full_agent` 0.274**.
- Citation validity / page traceability: **1.0** (structural), not semantic correctness.
- Cost: **$0.00** in this configuration (no paid model).

## Hypothetical risks (not observed as failures in this run)

Labelled separately so they are not confused with measured cases:

- Prompt injection via PDF text (mitigated by delimiters + regression test; no live exploit measured).
- Cache serving stale extraction after heuristic change without schema bump (guarded by `EXTRACTION_CACHE_SCHEMA`).
- Live provider auth misconfiguration (fails fast; live tests opt-in).
