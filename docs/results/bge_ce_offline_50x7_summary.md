# BGE + cross-encoder offline evaluation (50×7)

**Status:** measured full frozen-split run with production local models.  
Extractive / deterministic answers (no live LLM, no RAGAS).

| Field | Value |
|---|---|
| Run ID | `run_a4770534afb84db2` |
| Artifact path (local, gitignored) | `outputs/evaluation/phase8-bge-ce-offline-50x7/` |
| Dataset fingerprint | `c15f16c1cf086b81d90f7c53869cc1943cbca26d616b254761b924969e6bb059` |
| Corpus fingerprint | `79d20fac73b0ced268b99b457766bd67` |
| Embedding | **`BAAI/bge-small-en-v1.5`** (384-d) |
| Reranker | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** |
| Graph | loaded |
| Live LLM / RAGAS | disabled |
| Questions × systems | 50 × 7, **0** system errors |
| Wall time | ~2 minutes on Apple MPS (indexes already built) |

## Reproduce

```bash
export HF_HOME="$PWD/.cache/huggingface"  # optional offline cache
uv run scholar-agent evaluate \
  --eval-config configs/evaluation.yaml \
  --embedding-backend st \
  --output-dir outputs/evaluation/phase8-bge-ce-offline-50x7
```

## Aggregate metrics

| system | paper R@8 | chunk R@8 | MRR | cite P | refusal* | latency ms | tools | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.70 | 0.15 | 0.070 | 0.380 | 0.90 | 20.6 | 1.00 | 0 |
| hybrid_rag | 0.69 | 0.36 | 0.149 | 0.386 | 0.90 | 20.6 | 1.00 | 0 |
| hybrid_rerank | 0.69 | 0.36 | **0.236** | 0.386 | 0.90 | 52.1 | 1.00 | 0 |
| hybrid_graph | 0.72 | 0.34 | 0.206 | 0.307 | 0.92 | 344.8 | 2.00 | 0 |
| hybrid_corrective | 0.64 | 0.20 | 0.076 | 0.363 | 1.00 | 661.8 | 3.34 | 0 |
| full_agent | 0.68 | 0.21 | 0.083 | **0.386** | 1.00 | 714.7 | 3.60 | 0 |
| static_all_tools | **0.73** | 0.29 | 0.129 | 0.330 | 0.90 | 368.7 | 4.00 | 0 |

\*Overall refusal; unanswerable-slice refusals: full_agent / hybrid_corrective **5/5**,
naive/hybrid baselines **0/5** in this extractive offline path.

## Contrast with hash offline clean run

| metric | hash `naive_dense` | BGE `naive_dense` | hash `hybrid_rerank` | BGE `hybrid_rerank` |
|---|---:|---:|---:|---:|
| paper R@8 | 0.13 | **0.70** | 0.61 | **0.69** |
| chunk R@8 | 0.04 | 0.15 | 0.24 | **0.36** |

Interpretation: under BGE, dense-only is no longer the weak link; hybrid/RRF gains
are smaller on **paper** recall than under hash, while **chunk** recall and MRR
still improve with hybrid and cross-encoder rerank (MRR 0.149 → 0.236).

## Notes

- Answers are extractive/deterministic; citation precision is not comparable to
  live DeepSeek generation runs.
- For live generation + RAGAS on the same BGE stack, see
  `docs/results/bge_ce_llm_ragas_50x7_summary.md` (when present).
