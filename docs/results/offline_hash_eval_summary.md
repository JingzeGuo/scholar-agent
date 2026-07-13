# Offline evaluation summary (hash embeddings)

**Status:** measured on a local full frozen-split run.  
**Not** a production BGE / cross-encoder claim.

| Field | Value |
|---|---|
| Run ID | `run_7ef8e4f006d7449d` |
| Artifact path (local, gitignored) | `outputs/evaluation/phase8-final-clean/` |
| Dataset fingerprint | `9382ed8119ad4072c3ef45fd6cc5b64b20d0ea91c09271ace302e6105c175d8e` |
| Config fingerprint | `6c775367254bd6fa8aa003dfea4509f46d5d1532e311c19da470ddd624f536ff` |
| Code provenance | commit `6d5a9fdcb9d954289a574d4c9c371372e8f1edb0`, `git_dirty=false` |
| Questions | 50 (frozen split) |
| Systems | 7 ablations |
| Embedding | `hashing-embedder-v1` (isolated eval index) |
| Reranker | `lexical-overlap-v1` |
| Live LLM / RAGAS | disabled (`use_llm=false`, `use_ragas=false`) |
| Estimated API cost | **$0.00** (`usd_per_1k_tokens=0.0`) |

Reproduce (requires local processed corpus + hash eval indexes):

```bash
uv run scholar-agent evaluate \
  --eval-config configs/evaluation.yaml \
  --embedding-backend hash \
  --output-dir outputs/evaluation/reproduce
```

## Aggregate metrics

| system | paper R@8 | chunk R@8 | MRR | cite P | cite R | cite valid | page trace | token F1 | refusal* | latency ms | tools | cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.13 | 0.01 | 0.005 | 0.033 | 0.23 | 1.00 | 1.00 | 0.077 | 0.94 | 4.6 | 1.00 | 0.00 |
| hybrid_rag | 0.61 | 0.14 | 0.079 | 0.168 | 0.71 | 1.00 | 1.00 | 0.067 | 0.92 | 10.5 | 1.00 | 0.00 |
| hybrid_rerank | 0.61 | 0.14 | 0.090 | 0.168 | 0.71 | 1.00 | 1.00 | 0.067 | 0.92 | 10.1 | 1.00 | 0.00 |
| hybrid_graph | **0.67** | 0.23 | 0.114 | 0.214 | 0.75 | 1.00 | 1.00 | 0.103 | 0.92 | 290.9 | 2.00 | 0.00 |
| hybrid_corrective | 0.49 | 0.05 | 0.025 | 0.260 | 0.54 | 1.00 | 1.00 | 0.076 | 0.90 | 337.4 | 2.18 | 0.00 |
| full_agent | 0.52 | 0.05 | 0.025 | **0.274** | 0.56 | 1.00 | 1.00 | 0.075 | 0.90 | 350.5 | 2.28 | 0.00 |
| static_all_tools | 0.66 | 0.23 | 0.073 | 0.178 | 0.74 | 1.00 | 1.00 | 0.071 | 0.90 | 304.2 | 4.00 | 0.00 |

\*Overall `refusal_correct` mixes answerable items (correct non-refusal) with the 5 unanswerable items. On the unanswerable slice alone, `full_agent` scored **0.00** (0/5).

## Per-category paper Recall@8

| type | n | naive_dense | hybrid_rerank | hybrid_graph | full_agent |
|---|---:|---:|---:|---:|---:|
| factual | 10 | 0.10 | 0.90 | 0.90 | 0.50 |
| keyword | 10 | 0.20 | 0.90 | 0.90 | 0.70 |
| comparison | 15 | 0.17 | 0.57 | 0.67 | 0.63 |
| relational | 10 | 0.10 | 0.40 | 0.45 | 0.45 |
| unanswerable (refusal) | 5 | 0.40 | 0.20 | 0.20 | **0.00** |

## Interpretation (bounded by this configuration)

1. Hybrid retrieval is the largest jump vs dense-only paper recall (**0.13 → 0.61**).
2. Graph expansion raises aggregate paper recall to **0.67** but multiplies latency (~10 ms → ~291 ms).
3. The full agent improves citation precision relative to hybrid-rerank (**0.168 → 0.274**) but does not win aggregate paper recall and fails unanswerable refusals in this offline deterministic setup.
4. Citation validity and page traceability are **1.0** because emitted citations map to canonical chunks/pages; this does **not** imply answer correctness.

See also: `docs/failure_analysis.md`, `docs/evaluation.md`.
