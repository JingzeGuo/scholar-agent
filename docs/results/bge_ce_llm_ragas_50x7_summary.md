# BGE + cross-encoder + live LLM + RAGAS (50×7)

**Status:** measured full frozen-split run with production local retrieval models
and shared DeepSeek generation/judging.

| Field | Value |
|---|---|
| Run ID | `run_6270c2cf8cd94186` |
| Artifact path (local, gitignored) | `outputs/evaluation/phase8-bge-ce-llm-ragas-50x7/` |
| Dataset fingerprint | `c15f16c1cf086b81d90f7c53869cc1943cbca26d616b254761b924969e6bb059` |
| Corpus fingerprint | `79d20fac73b0ced268b99b457766bd67` |
| Embedding | **`BAAI/bge-small-en-v1.5`** |
| Reranker | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** |
| Generation | `deepseek-v4-flash`, prompt `evaluation-grounded-answer-v1`, regime `shared_live_llm` |
| Generation used | **350 / 350** |
| RAGAS | available; success **314**, partial **1**, skipped **35** (unanswerable refusals) |
| RAGAS coverage rate | **0.90** (expected; 5 refusals × 7 systems) |
| System errors | **0** |
| Wall time | ~3 h 25 m (2026-07-14 09:19 → 12:44 local) |
| Code note | `git_commit=a9e430c`, working tree had uncommitted docs during run (`git_dirty=true`) |

Companion offline extractive run on the same stack:
[`bge_ce_offline_50x7_summary.md`](bge_ce_offline_50x7_summary.md)
(`run_a4770534afb84db2`).

## Reproduce

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
uv sync --extra eval
uv run scholar-agent evaluate \
  --eval-config configs/evaluation.yaml \
  --embedding-backend st \
  --llm --ragas \
  --output-dir outputs/evaluation/phase8-bge-ce-llm-ragas-50x7
```

## Aggregate metrics

| system | paper R@8 | chunk R@8 | MRR | cite P | claim corr. | completeness | RAGAS faith. | RAGAS relev. | RAGAS cov. | refusal* | latency ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.70 | 0.15 | 0.070 | 0.613 | 0.309 | 0.603 | 0.939 | 0.794 | 0.90 | 1.00 | 2380 | 0 |
| hybrid_rag | 0.69 | 0.36 | 0.149 | 0.607 | 0.306 | **0.618** | 0.964 | **0.864** | 0.90 | 0.98 | 2376 | 0 |
| hybrid_rerank | 0.69 | 0.36 | **0.236** | **0.660** | 0.303 | 0.603 | 0.940 | 0.827 | 0.90 | 1.00 | 2571 | 0 |
| hybrid_graph | 0.72 | 0.34 | 0.206 | 0.623 | **0.315** | 0.616 | 0.943 | 0.785 | 0.90 | 1.00 | 2535 | 0 |
| hybrid_corrective | 0.64 | 0.20 | 0.076 | 0.576 | 0.289 | 0.588 | 0.939 | 0.767 | 0.90 | 1.00 | 2919 | 0 |
| full_agent | 0.68 | 0.21 | 0.083 | 0.618 | 0.294 | 0.612 | 0.965 | 0.788 | 0.90 | 1.00 | 2896 | 0 |
| static_all_tools | **0.73** | 0.29 | 0.129 | 0.637 | 0.301 | 0.615 | **0.984** | 0.854 | 0.90 | 1.00 | 3065 | 0 |

\*Overall refusal; unanswerable-slice refusals are **5/5** for full_agent,
hybrid_corrective, hybrid_rerank, hybrid_graph, naive_dense, static_all_tools
in this live run (hybrid_rag unanswerable refusal **0.8**).

## Contrast with prior hash live+RAGAS (`run_a23467bb0aa84115`)

| system | hash paper R@8 | BGE paper R@8 | hash cite P | BGE cite P | hash RAGAS faith. | BGE RAGAS faith. |
|---|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.16 | **0.70** | 0.170 | **0.613** | 0.726 | **0.939** |
| hybrid_rerank | 0.61 | **0.69** | 0.603 | **0.660** | 0.922 | **0.940** |
| hybrid_graph | 0.67 | **0.72** | 0.620 | 0.623 | **0.959** | 0.943 |
| full_agent | 0.54 | **0.68** | 0.491 | **0.618** | 0.877 | **0.965** |

Interpretation: switching dense retrieval from hash to BGE is the dominant
upgrade for paper recall and for generation quality under shared LLM+RAGAS.
Cross-encoder primarily improves ranking quality (MRR) rather than paper R@8.

## Notes

- Retrieval metrics match the offline BGE run (same index + reranker).
- Answer/citation/RAGAS metrics use live DeepSeek generation; they are not
  extractive baselines.
- Configured `usd_per_1k_tokens=0.0` → reported cost remains $0.00 (not an invoice).
