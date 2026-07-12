# Evaluation (Phase 8)

ScholarAgent evaluates retrieval, citation, and answer quality on a **frozen
50-question split** shared by every baseline and ablation.

## Dataset

| Type | Count |
|---|---:|
| Single-paper factual | 10 |
| Exact terminology / keyword | 10 |
| Cross-paper comparison | 15 |
| Multi-hop relational | 10 |
| Unanswerable from corpus | 5 |

Artifacts (committed):

- `data/evaluation/questions.jsonl`
- `data/evaluation/reference_evidence.jsonl`
- `data/evaluation/frozen_split.json` (SHA-256 fingerprint)

Rebuild gold chunk IDs after re-ingestion:

```bash
uv run python scripts/build_eval_dataset.py
```

The freeze fingerprint must be updated deliberately; `load_eval_dataset` rejects
silent drift between the JSONL files and `frozen_split.json`.

## Systems

| Name | Description |
|---|---|
| `naive_dense` | Dense top-k + extractive answer |
| `hybrid_rag` | Dense + BM25 RRF |
| `hybrid_rerank` | Hybrid + reranker |
| `hybrid_graph` | Hybrid-rerank merged with graph hits |
| `hybrid_corrective` | Plan→research→verify corrective loop (tighter budgets) |
| `full_agent` | Full ScholarAgent workflow + Writer/citations |
| `static_all_tools` | Routing ablation: always dense+sparse+hybrid(+graph) |

## Metrics (offline-first)

**Retrieval:** Recall@K (chunk + paper), MRR, nDCG@K (graded when available).
Graph-expected questions additionally report graph-evidence recall and every
system reports unique useful gold evidence per tool call.

**Citations:** precision/recall vs gold papers, validity rate, page traceability,
unsupported claim rate.

**Answers:** claim/token overlap with reference claims, refusal accuracy on
unanswerable items, faithfulness proxy (answer tokens covered by contexts).

**Optional RAGAS:** `uv sync --extra eval` and `scholar-agent evaluate --ragas`
(requires a configured DeepSeek/OpenAI-compatible API key and makes paid live
LLM calls). The evaluator explicitly uses the project's configured provider and
the retrieval embedder; it never falls back to RAGAS's implicit OpenAI defaults.
Reports record whether RAGAS was requested, installed, configured, and actually
used; unavailable scores remain `null`, never silently become zero. Default CI
stays deterministic without paid calls.

## Run

```bash
# Smoke (fast): two systems × first 5 questions, hash embeddings
uv run scholar-agent evaluate \
  --system hybrid_rerank --system naive_dense \
  --max-questions 5 \
  --embedding-backend hash \
  --output-dir outputs/evaluation/smoke

# Full frozen split (all systems) — longer when full_agent is included
uv run scholar-agent evaluate --config configs/default.yaml \
  --eval-config configs/evaluation.yaml \
  --embedding-backend hash

make evaluate-smoke
```

Outputs under `outputs/evaluation/` (gitignored recommended):

- `results.json`, `aggregate_metrics.csv`, `per_question_metrics.csv`
- `run_config.json` with dataset/config/code fingerprints
- `report.md`, `failures.json`
- SVG charts: aggregate recall, per-category recall, latency, cost, citation precision

An explicit `--embedding-backend hash` uses an isolated deterministic index at
`data/indexes/evaluation/hash/`; it never silently reuses the production BGE
collection. The report stores both requested and actual embedding/reranker names.

## Measured offline audit

The complete 50×7 audit run `run_f23303cfda91408c` used hashing embeddings and
lexical reranking. Paper Recall@8 ranged from 0.13 (`naive_dense`) to 0.67
(`hybrid_graph`); the full agent reached 0.53 recall and 0.274 citation precision
but failed all five unanswerable refusals. These are configuration-specific
measurements, not production-model claims. See `docs/failure_analysis.md` for six
manually reviewed cases.

## Discipline

- Do not tune systems against individual frozen questions in prompts.
- Report **per-category** metrics, not only micro-averages.
- Latency and estimated token cost are recorded for every question×system.
- Per-category tables include latency, tool calls, token estimates, and cost.
- Manual failure analysis: `notebooks/error_analysis.ipynb` and
  `docs/failure_analysis.md`.
