# Failure analysis (Phase 8 seed)

At least five concrete failure modes observed or expected when comparing
baselines on the frozen split. Update measured rows after full evaluation runs.

## F1 — Dense-only miss on exact method names

| Field | Detail |
|---|---|
| **Symptom** | `naive_dense` ranks wrong papers for keyword questions like “What is HyDE?” |
| **Category** | keyword |
| **Root cause** | Dense embeddings underweight rare acronyms vs BM25 exact tokens |
| **Fix / mitigation** | Hybrid RRF + sparse path; router prefers keyword policy for acronym queries |
| **Measurement** | Compare `naive_dense` vs `hybrid_rag` paper Recall@K on `keyword` slice |

## F2 — Graph noise on simple factual questions

| Field | Detail |
|---|---|
| **Symptom** | `hybrid_graph` / `static_all_tools` retrieve related but off-claim edges for single-paper facts |
| **Category** | factual |
| **Root cause** | Graph paths expand entity neighborhoods beyond the needed definition span |
| **Fix / mitigation** | Adaptive routing: enable graph only for relational/comparison types |
| **Measurement** | Per-type citation precision; factual slice should not prefer static_all_tools |

## F3 — Corrective loop cannot invent missing evidence

| Field | Detail |
|---|---|
| **Symptom** | Unanswerable items still get fluent extractive notes from weak neighbors |
| **Category** | unanswerable |
| **Root cause** | Retrieve-then-read baselines always surface top-k text; refusal needs explicit policy |
| **Fix / mitigation** | Full agent Verifier marks `corpus_cannot_answer`; Writer emits Limitation |
| **Measurement** | `refusal_correct` on the 5 unanswerable questions |

## F4 — Comparison questions need multi-paper coverage

| Field | Detail |
|---|---|
| **Symptom** | Single-pass hybrid retrieves only one side of Self-RAG vs CRAG |
| **Category** | comparison |
| **Root cause** | Top-k dominated by one high-scoring paper |
| **Fix / mitigation** | Planner splits sub-questions; corrective queries target missing side |
| **Measurement** | Paper recall for two-gold-paper comparison items; corrective vs hybrid_rerank |

## F5 — Citation attached to weakly supporting chunk

| Field | Detail |
|---|---|
| **Symptom** | Answer mentions a gold paper but cited chunk is a header/title fragment |
| **Category** | relational / factual |
| **Root cause** | Extractive baselines cite whatever was retrieved, not claim entailment |
| **Fix / mitigation** | Phase-7 citation validator token-overlap + page provenance repair |
| **Measurement** | `citation_validity_rate` and unsupported claim rate for `full_agent` |

## How to extend

After `scholar-agent evaluate`, open `outputs/evaluation/failures.json` and append
rows here with run_id, system, question_id, and metric deltas. Keep at least five
narrative failures for interview stories (`docs/interview_guide.md` in Phase 10).
