# Full live LLM + RAGAS evaluation (50×7)

**Status:** measured full frozen-split run with shared DeepSeek generation and RAGAS.  
**Not** a production BGE / cross-encoder retrieval claim (retrieval still uses hash embeddings).

| Field | Value |
|---|---|
| Run ID | `run_a23467bb0aa84115` |
| Artifact path (local, gitignored) | `outputs/evaluation/phase8-full-llm-ragas-50x7/` |
| Dataset fingerprint | `c15f16c1cf086b81d90f7c53869cc1943cbca26d616b254761b924969e6bb059` |
| Corpus fingerprint | `79d20fac73b0ced268b99b457766bd67` |
| Code provenance | commit `6af11f0`, `git_dirty=false` |
| Questions × systems | **50 × 7 = 350** rows |
| System errors | **0** |
| Embedding | `hashing-embedder-v1` |
| Reranker | lexical (hash path) |
| Graph | loaded |
| Generation | `deepseek-v4-flash`, prompt `evaluation-grounded-answer-v1`, regime `shared_live_llm` |
| Generation used | **350 / 350** |
| RAGAS | configured + available; metrics `faithfulness`, `answer_relevancy` |
| RAGAS row outcomes | success **308**, partial **7**, skipped **35** |
| RAGAS coverage rate | **0.90** (aggregate; see note) |
| Configured token cost rate | `$0.00` / 1k (not a provider invoice) |
| Wall time | ~2 h 46 m (2026-07-13 23:22 → 2026-07-14 02:08 local) |
| HTTP | ~1700+ successful DeepSeek chat calls; no fatal provider abort |

## Reproduce

```bash
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
uv sync --extra eval
uv run scholar-agent evaluate \
  --eval-config configs/evaluation.yaml \
  --embedding-backend hash \
  --llm \
  --ragas \
  --output-dir outputs/evaluation/phase8-full-llm-ragas-50x7
```

## Aggregate metrics

| system | paper R@8 | chunk R@8 | cite P | claim corr. | completeness | RAGAS faith. | RAGAS relev. | RAGAS cov. | refusal* | latency ms | tools | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.16 | 0.04 | 0.170 | 0.242 | 0.406 | 0.726 | 0.105 | 0.90 | 1.00 | 1389 | 1.00 | 0 |
| hybrid_rag | 0.61 | 0.24 | 0.610 | 0.307 | 0.604 | 0.903 | 0.562 | 0.90 | 0.98 | 2277 | 1.00 | 0 |
| hybrid_rerank | 0.61 | 0.24 | 0.603 | 0.310 | 0.583 | 0.922 | 0.549 | 0.90 | 0.98 | 2102 | 1.00 | 0 |
| hybrid_graph | **0.67** | 0.26 | **0.620** | **0.318** | **0.609** | **0.959** | **0.583** | 0.90 | 0.98 | 2482 | 2.00 | 0 |
| hybrid_corrective | 0.51 | 0.11 | 0.468 | 0.285 | 0.533 | 0.869 | 0.462 | 0.90 | 1.00 | 2344 | 3.28 | 0 |
| full_agent | 0.54 | 0.11 | 0.491 | 0.283 | 0.522 | 0.877 | 0.492 | 0.90 | 1.00 | 2310 | 3.52 | 0 |
| static_all_tools | 0.65 | 0.25 | 0.590 | 0.314 | 0.578 | 0.945 | 0.569 | 0.90 | 1.00 | 2507 | 4.00 | 0 |

\*Overall refusal includes correct non-refusal on answerable items. Full agent
unanswerable slice: **5/5**.

## RAGAS coverage note

- Aggregate `ragas_coverage_rate = 0.90` is **expected**, not a silent failure:
  - **35 skipped rows** = 5 unanswerable questions × 7 systems, where the system
    correctly **refuses** and emits an empty answer → RAGAS has nothing to score.
  - **7 partial** rows returned `out_of_range_score` on one metric and were not
    counted as full success.
  - **308 / 315** answerable-or-answered rows fully succeeded for both metrics
    under the success counter; partials still contribute available metric values
    where valid.
- Do not interpret 0.90 as “10% of the suite never called RAGAS.”

## Interpretation (this configuration only)

1. Shared live generation raises citation precision vs offline extractive baselines
   (e.g. hybrid_rag cite P **0.610** vs offline hash extractive **0.169**).
2. Best paper recall remains **hybrid_graph 0.67**; best RAGAS faithfulness also
   **hybrid_graph 0.959**.
3. Full agent retains strong refusal (**5/5** unanswerable) and solid RAGAS
   faithfulness (**0.877**), but does not top aggregate paper recall or RAGAS
   relevancy.
4. Retrieval is still **hash-embedding-bounded**; do not quote these as BGE quality.
5. Corrective `improvement_after_correction` remains **0.0** (same operational
   gap as the offline clean run).

## Artifacts

| File | Role |
|---|---|
| `results.json` | Full per-question + aggregate payload |
| `aggregate_metrics.csv` | System aggregates |
| `per_question_metrics.csv` | 350 rows |
| `run_config.json` | Fingerprints + generation/RAGAS flags |
| `report.md` | Human-readable tables |
| `corrective_before_after.json` | Corrective-loop before/after evidence |
| `failures.json` | Thresholded failure log (86 rows) |
