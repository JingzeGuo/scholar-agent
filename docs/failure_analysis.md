# Failure analysis (Phase 8 measured run)

The cases below were manually reviewed from the complete offline run
`run_f23303cfda91408c` (50 frozen questions × 7 systems). Dataset fingerprint:
`9382ed8119ad4072c3ef45fd6cc5b64b20d0ea91c09271ace302e6105c175d8e`.
The run used the isolated `hashing-embedder-v1` index, lexical reranking, no live
LLM/RAGAS calls, and zero configured API cost. These findings describe this
configuration only; they are not production-BGE claims.

## F1 — Dense-only misses exact terminology (`q_k08`)

| Field | Measured finding |
|---|---|
| **Question** | “What does the RAGAs framework evaluate?” |
| **Result** | `naive_dense` paper Recall@8 = **0.0**; `hybrid_rerank` = **1.0** |
| **Root cause** | The deterministic hash embedding is weak for exact product names; BM25 supplies the missing exact-token signal. |
| **Action** | Retain hybrid retrieval for keyword/acronym routes; do not cite this result as evidence about BGE quality. |

## F2 — Graph retrieval helps some relational cases but is expensive (`q_r09`)

| Field | Measured finding |
|---|---|
| **Question** | “How do hallucination surveys motivate automated RAG evaluation metrics like those in RAGAs?” |
| **Result** | `hybrid_rerank` paper Recall@8 = **0.0**; `hybrid_graph` = **0.5**. Aggregate latency rose from **10.6 ms** to **309.1 ms**. |
| **Root cause** | Graph paths recover a related gold paper absent from hybrid top-k, but path traversal adds substantial CPU latency. |
| **Action** | Keep graph routing selective. The graph-evidence recall over graph-expected questions was only **0.0556**, so graph expansion is not a universal win. |

## F3 — Graph fan-out can displace useful hybrid evidence (`q_c15`)

| Field | Measured finding |
|---|---|
| **Question** | “Compare RETRO and Atlas as retrieval-augmented language models.” |
| **Result** | `hybrid_rerank` paper Recall@8 = **1.0**; round-robin `hybrid_graph` = **0.5**. |
| **Root cause** | Reserving top-k slots for graph evidence improves tool diversity but can evict a relevant lexical/dense hit. |
| **Action** | Future work should fuse by calibrated scores or allocate graph slots dynamically instead of using a fixed 50/50 merge. |

## F4 — Planning improves some comparisons but lowers aggregate retrieval recall (`q_c05`)

| Field | Measured finding |
|---|---|
| **Question** | “Compare Toolformer and ReAct for tool use in language models.” |
| **Result** | `hybrid_rerank` paper Recall@8 = **0.5**; `full_agent` = **1.0**. Yet aggregate paper recall was **0.61** for hybrid-rerank versus **0.53** for the full agent. |
| **Root cause** | Decomposition can retrieve both sides of a comparison, while per-sub-question evidence caps and ledger ordering can omit gold chunks on simpler questions. |
| **Action** | Report gains by category/question rather than claiming a universal agent improvement; investigate evidence-budget allocation before changing defaults. |

## F5 — Unanswerable detection is the largest observed weakness (`q_u01`)

| Field | Measured finding |
|---|---|
| **Question** | “What is the exact GPU price schedule used inside OpenAI's private 2026 training cluster for GPT-6?” |
| **Result** | `full_agent` refusal accuracy was **0/5** on the unanswerable slice; `q_u01` was answered from weak neighboring evidence instead of refused. |
| **Root cause** | Lexical overlap with terms such as OpenAI/GPT can make the deterministic Verifier accept topical but non-answering passages. |
| **Action** | Add explicit required-aspect/entailment checks using a development set; do not tune the Verifier against these frozen evaluation questions. |

## F6 — Scope evidence can help a simple baseline refuse while the agent over-accepts (`q_u03`)

| Field | Measured finding |
|---|---|
| **Question** | Pediatric amoxicillin dosage “according to Self-RAG.” |
| **Result** | `naive_dense` and `hybrid_rerank` refusal score = **1.0**; `full_agent` = **0.0**. |
| **Root cause** | The gold Self-RAG title page establishes domain mismatch, but the agent treats the matching method name as positive answer evidence. |
| **Action** | Model “scope evidence” separately from “answer evidence” in a future dataset/schema revision.

## Aggregate interpretation

- `hybrid_graph` had the highest paper Recall@8 (**0.67**) but higher average
  latency (**309.1 ms**) and modest citation precision (**0.214**).
- `full_agent` had the best citation precision among measured systems (**0.274**)
  but lower paper recall (**0.53**) and no correct refusals in this offline run.
- Citation validity and page traceability were **1.0** because emitted citations
  mapped to canonical chunks/pages; this does not imply answer correctness.
- Cost was reported as **$0.00** because the saved configuration used no paid
  model and `usd_per_1k_tokens=0.0`.

Raw artifacts are generated under `outputs/evaluation/`; use
`notebooks/error_analysis.ipynb` to reproduce slices from a saved run.
