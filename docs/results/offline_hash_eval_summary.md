# Offline evaluation summary (hash embeddings)

**Status:** measured on a local full frozen-split run.  
**Not** a production BGE / cross-encoder claim.

| Field | Value |
|---|---|
| Run ID | `run_1f4dc371453d4a1f` |
| Artifact path (local, gitignored) | `outputs/evaluation/phase8-final-hash-exact-eff976a/` |
| Dataset fingerprint | `c15f16c1cf086b81d90f7c53869cc1943cbca26d616b254761b924969e6bb059` |
| Corpus fingerprint | `79d20fac73b0ced268b99b457766bd67` |
| Code provenance | commit `eff976a`, `git_dirty=false` |
| Questions | 50 (frozen split) |
| Systems | 7 ablations |
| Embedding | `hashing-embedder-v1` (isolated eval index) |
| Reranker | `lexical-overlap-v1` |
| Graph | loaded (`graph-v2-physical-page-ranges`) |
| Live LLM / RAGAS | disabled (`use_llm=false`, `use_ragas=false`) |
| Estimated API cost | **$0.00** (`usd_per_1k_tokens=0.0`) |
| Structural | 50×7 = 350 rows, **0** system errors |

Reproduce (requires local processed corpus + hash eval indexes):

```bash
uv run scholar-agent evaluate \
  --eval-config configs/evaluation.yaml \
  --embedding-backend hash \
  --output-dir outputs/evaluation/reproduce
```

Hash dense search uses exact cosine over persisted vectors with `chunk_id`
tie-breaks (not Chroma HNSW) so offline metrics do not drift across processes.

## Aggregate metrics

| system | paper R@8 | chunk R@8 | MRR | cite P | cite R | cite valid | page trace | claim corr. | completeness | refusal* | latency ms | tools | cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.16 | 0.04 | 0.043 | 0.038 | 0.26 | 1.00 | 1.00 | 0.066 | 0.296 | 0.94 | 3.6 | 1.00 | 0.00 |
| hybrid_rag | 0.61 | 0.24 | 0.146 | 0.166 | 0.71 | 1.00 | 1.00 | 0.061 | 0.465 | 0.92 | 10.4 | 1.00 | 0.00 |
| hybrid_rerank | 0.61 | 0.24 | 0.160 | 0.166 | 0.71 | 1.00 | 1.00 | 0.061 | 0.465 | 0.92 | 10.7 | 1.00 | 0.00 |
| hybrid_graph | **0.67** | 0.26 | 0.139 | 0.212 | 0.75 | 1.00 | 1.00 | 0.084 | 0.397 | 0.92 | 289.9 | 2.00 | 0.00 |
| hybrid_corrective | 0.51 | 0.11 | 0.085 | 0.271 | 0.60 | 1.00 | 0.90 | 0.166 | 0.524 | 1.00 | 538.2 | 3.28 | 0.00 |
| full_agent | 0.54 | 0.11 | 0.085 | **0.288** | 0.63 | 1.00 | 0.92 | 0.166 | 0.526 | 1.00 | 572.2 | 3.52 | 0.00 |
| static_all_tools | 0.65 | 0.25 | 0.118 | 0.175 | 0.73 | 1.00 | 1.00 | 0.046 | 0.306 | 0.90 | 318.7 | 4.00 | 0.00 |

\*Overall `refusal_correct` includes answerable items (correct non-refusal). On the
**unanswerable slice alone**, `full_agent` scored **1.00** (5/5).

## Per-category paper Recall@8

| type | n | naive_dense | hybrid_rerank | hybrid_graph | full_agent |
|---|---:|---:|---:|---:|---:|
| factual | 10 | 0.20 | 0.90 | 0.90 | 0.50 |
| keyword | 10 | 0.20 | 0.90 | 0.90 | 0.70 |
| comparison | 15 | 0.17 | 0.57 | 0.67 | 0.70 |
| relational | 10 | 0.15 | 0.40 | 0.45 | 0.45 |
| unanswerable (refusal) | 5 | 0.40 | 0.20 | 0.20 | **1.00** |

## Corrective-loop operational metrics (agent systems)

| system | corrective_trigger_precision | improvement_after_correction | plan_coverage |
|---|---:|---:|---:|
| hybrid_corrective | 1.00 | **0.00** | 0.80 |
| full_agent | 1.00 | **0.00** | 0.82 |

Corrective retrieval **triggers and terminates safely**, but in this offline hash
configuration it did not improve gold paper recall on the scored trigger subset
(`improvement_after_correction = 0.0`). That is a measured limitation, not a
termination defect.

## Interpretation (bounded by this configuration)

1. Hybrid retrieval is the largest jump vs dense-only paper recall (**0.16 → 0.61**).
2. Graph expansion raises aggregate paper recall to **0.67** but multiplies latency (~11 ms → ~290 ms).
3. The full agent has the best offline citation precision (**0.288**) and **5/5**
   unanswerable refusals, but does not win aggregate paper recall vs hybrid/graph.
4. Citation validity remains **1.0** where citations are emitted; this does **not**
   imply answer correctness.

## Related live shared-LLM run (optional)

A separate full shared DeepSeek generation + RAGAS regime
(`run_a23467bb0aa84115`, artifact
`outputs/evaluation/phase8-full-llm-ragas-50x7/`) reuses the same retrieval stack
with `generation_regime=shared_live_llm` / `deepseek-v4-flash`. Retrieval paper
R@8 matches the offline table (within hash-path stability); citation precision
and claim correctness rise under live generation (e.g. hybrid_rerank cite P
**0.588**, claim correctness **0.313**). Cost fields remain **$0.00** when
`usd_per_1k_tokens=0.0` is configured — they are not measured provider invoices.

See also: `docs/failure_analysis.md`, `docs/evaluation.md`.
