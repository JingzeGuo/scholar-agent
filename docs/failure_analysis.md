# Failure analysis

Evidence source: complete offline clean run **`run_5bb1f439f19842cb`**
(50 frozen questions × 7 systems).

| Field | Value |
|---|---|
| Dataset fingerprint | `c15f16c1cf086b81d90f7c53869cc1943cbca26d616b254761b924969e6bb059` |
| Corpus fingerprint | `79d20fac73b0ced268b99b457766bd67` |
| Local artifact | `outputs/evaluation/phase8-final-clean-c741887/` (gitignored) |
| Summary snapshot | [`docs/results/offline_hash_eval_summary.md`](results/offline_hash_eval_summary.md) |
| Config | `hashing-embedder-v1`, lexical rerank, graph loaded, no live LLM/RAGAS, cost $0.00 |

These are **real measured cases** from that run, not hypotheticals. They describe
this configuration only.

Dataset review status: **AI-assisted independent manual review** of all 50
questions (`data/evaluation/manual_review_manifest.json`). This is **not** a
human-signed annotation audit.

---

## F1 — Dense retrieval misses exact terminology (`q_k08`)

| Field | Detail |
|---|---|
| **Scenario** | Keyword question about an exact framework/product name (RAGAs). |
| **Observable symptom** | Dense-only paper recall stays low on keyword items (aggregate keyword paper R@8 **0.20**); hybrid_rerank keyword paper R@8 **0.90**. |
| **Affected component** | Dense retrieval / hashing embedder |
| **Reproduction** | Offline eval `run_5bb1f439f19842cb` keyword slice in `aggregate_metrics.csv` / per-question CSV. |
| **Root cause** | Hash embedding is weak for exact product/framework names; BM25 supplies the exact-token signal. |
| **Fix** | Keep hybrid (dense+BM25 RRF) as the default keyword/hybrid route; do not present dense-only as production quality. |
| **Verification after fix** | Hybrid systems recover keyword paper recall to **0.90**. |
| **Remaining limitation** | Production BGE quality is **not measured** in this offline run. |
| **Interview lesson** | Sparse retrieval is not obsolete; hybrid fusion is a measured necessity for exact terms. |

---

## F2 — Graph helps relational/comparison cases but is expensive

| Field | Detail |
|---|---|
| **Scenario** | Multi-hop / relational and some comparison questions. |
| **Observable symptom** | Aggregate paper R@8: hybrid_rerank **0.61** → hybrid_graph **0.67**. Latency **~11.6 ms → ~301.5 ms**. |
| **Affected component** | Graph retrieval / path ranking |
| **Reproduction** | System aggregates in `run_5bb1f439f19842cb`. |
| **Root cause** | Graph paths recover gold papers outside hybrid top-k; traversal dominates CPU under this index. |
| **Fix** | Route graph only for relational/multi-hop policies; keep hybrid-only for simple factual. |
| **Verification after fix** | Selective router policies retained; graph still optional. |
| **Remaining limitation** | Score-calibrated fusion of graph vs hybrid is still future work. |
| **Interview lesson** | GraphRAG is a tool with a latency tax, not a free upgrade. |

---

## F3 — Graph / multi-tool merge can displace good hybrid evidence

| Field | Detail |
|---|---|
| **Scenario** | Comparison items where hybrid already ranks both papers highly. |
| **Observable symptom** | Aggregate static_all_tools paper R@8 **0.65** vs hybrid_graph **0.67**, but individual questions can lose a hybrid hit when graph/static slots occupy top-k. |
| **Affected component** | Multi-tool merge / slot allocation |
| **Reproduction** | Per-question comparisons in the same run’s CSV. |
| **Root cause** | Round-robin / reserved slots improve diversity but can evict a relevant hybrid hit. |
| **Fix (partial)** | Documented design risk; selective graph routing preferred over always-on graph. |
| **Verification after fix** | Hybrid remains competitive; graph still best aggregate paper recall. |
| **Remaining limitation** | No calibrated score fusion yet. |
| **Interview lesson** | More tools ≠ better ranking without careful fusion. |

---

## F4 — Planning/agent can win comparisons yet lose aggregate paper recall

| Field | Detail |
|---|---|
| **Scenario** | Cross-paper comparisons under the full agent. |
| **Observable symptom** | full_agent comparison paper R@8 **0.70** (above hybrid_rerank **0.57**), but aggregate paper recall only **0.54** vs hybrid **0.61**. |
| **Affected component** | Planner decomposition + evidence budgets |
| **Reproduction** | `by_type` tables for `full_agent` vs `hybrid_rerank` in results JSON. |
| **Root cause** | Decomposition helps some multi-side questions while per-sub-question caps and ledger ordering omit gold chunks on simpler questions. |
| **Fix** | Report **per-category** and per-question gains; avoid claiming universal agent wins. |
| **Verification after fix** | Evaluation report always includes by-type tables. |
| **Remaining limitation** | Evidence-budget allocation heuristics still manual. |
| **Interview lesson** | Always pair micro case studies with macro metrics. |

---

## F5 — Corrective loop triggers but does not improve gold recall

| Field | Detail |
|---|---|
| **Scenario** | Verifier-driven corrective re-retrieval on insufficient evidence. |
| **Observable symptom** | `corrective_trigger_precision = 1.0` but `improvement_after_correction = 0.0` for both `hybrid_corrective` and `full_agent`. |
| **Affected component** | Corrective research merge / query targeting |
| **Reproduction** | Aggregate agent metrics in `run_5bb1f439f19842cb`. |
| **Root cause** | Corrective passes fire and terminate safely, but additional evidence does not increase gold paper recall on the scored subset (hash retrieval ceiling + targeting limits). |
| **Fix (partial)** | Loop termination, budgets, and merge behavior are tested; quality of corrective queries remains limited offline. |
| **Verification after fix** | Termination tests pass; metric remains **0.0** improvement — **open quality gap**. |
| **Remaining limitation** | Needs better gap→query mapping and/or stronger embedder before claiming corrective “optimization.” |
| **Interview lesson** | “Triggered and terminated” ≠ “helped the metric.” Measure both. |

---

## F6 — Hash ANN non-determinism (fixed for offline eval)

| Field | Detail |
|---|---|
| **Scenario** | Re-running the same hash dense index across processes. |
| **Observable symptom** | Dense-only paper R@8 occasionally drifted (e.g. **0.13** vs **0.14**) under Chroma HNSW with many tied 64-d hash vectors. |
| **Affected component** | DenseIndex hash search path |
| **Reproduction** | Pre-fix clean re-runs of hash evaluation; unit test now forces exact path. |
| **Root cause** | HNSW does not guarantee stable order among equal similarities. |
| **Fix** | Hash embedder searches use exact cosine over persisted embeddings with `chunk_id` tie-break (`DenseIndex._search_hash_exact`). Production ST/BGE path keeps ANN. |
| **Verification after fix** | `tests/unit/test_retrieval.py::test_hash_dense_search_is_exact_and_stably_tie_broken`. |
| **Remaining limitation** | Exact search is for hash offline eval scale; not a production ANN redesign. |
| **Interview lesson** | Reproducibility is a first-class eval property, not a footnote. |

---

## Aggregate interpretation

- Best paper Recall@8: **`hybrid_graph` 0.67** (latency **301.5 ms**).
- Best citation precision (offline): **`full_agent` 0.288**.
- Full agent unanswerable refusals: **5/5** in this offline run.
- Corrective improvement metric: **0.0** (triggers without measured gold-recall gain).
- Cost: **$0.00** in this configuration (no paid model / zeroed rate).

## Hypothetical risks (not observed as failures in this run)

Labelled separately so they are not confused with measured cases:

- Prompt injection via PDF text (mitigated by delimiters + regression test; no live exploit measured).
- Cache serving stale extraction after heuristic change without schema bump.
- Live provider auth misconfiguration (fails fast; live tests opt-in).
