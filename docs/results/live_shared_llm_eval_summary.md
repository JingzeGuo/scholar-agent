# Live shared-LLM evaluation summary

**Status:** measured optional run with DeepSeek shared generation.  
**Not** a substitute for the offline deterministic suite.

| Field | Value |
|---|---|
| Run ID | `run_20de25c647c1433f` |
| Artifact path (local, gitignored) | `outputs/evaluation/phase8-final-live-79d20/` |
| Dataset fingerprint | `c15f16c1cf086b81d90f7c53869cc1943cbca26d616b254761b924969e6bb059` |
| Corpus fingerprint | `79d20fac73b0ced268b99b457766bd67` |
| Embedding | `hashing-embedder-v1` |
| Generation | `deepseek-v4-flash`, prompt `evaluation-grounded-answer-v1` |
| Generation regime | `shared_live_llm` |
| Questions × systems | 50 × 7, **0** system errors |
| Configured token cost rate | `$0.00` / 1k tokens (not an invoice) |

## Aggregate (selected columns)

| system | paper R@8 | cite P | claim corr. | completeness | refusal* | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| naive_dense | 0.14 | 0.140 | 0.243 | 0.408 | 1.00 | 1373 |
| hybrid_rag | 0.61 | 0.613 | 0.301 | 0.583 | 0.98 | 2283 |
| hybrid_rerank | 0.61 | 0.588 | 0.313 | 0.581 | 0.98 | 2358 |
| hybrid_graph | **0.67** | 0.600 | 0.313 | 0.588 | 0.98 | 2603 |
| hybrid_corrective | 0.51 | 0.479 | 0.282 | 0.516 | 1.00 | 2414 |
| full_agent | 0.54 | 0.499 | 0.281 | 0.533 | 1.00 | 2449 |
| static_all_tools | 0.65 | 0.593 | 0.308 | **0.595** | 1.00 | 2726 |

\*Overall refusal including answerable non-refusal. Full agent unanswerable slice: **5/5**.

## Notes

- Proves `--llm` is a real shared generation path, not a no-op switch.
- Retrieval scores remain hash-embedder bounded; do not read these as BGE quality.
- Default CI never runs this path (`pytest -m "not live"`).
