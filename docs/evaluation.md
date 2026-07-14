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

**Answers:** claim/token overlap, deterministic claim-correctness and
completeness proxies, refusal accuracy, faithfulness proxy, and contradiction
handling when an annotation or Verifier event makes that metric applicable.
Claim correctness averages the closest token-F1 reference match for each
predicted claim unit; completeness averages reference-claim token recall. These
are lexical proxies, not expert semantic judgments.

**Agents/operations:** plan coverage and tool-selection accuracy are reported
only for adaptive workflow rows. Corrective-trigger precision is computed only
for observed triggers with answerable gold labels: a trigger is justified when
the pre-correction gold-evidence recall is incomplete. Chunk recall is used when
chunk labels exist, otherwise paper recall; the basis is stored per row.
Improvement is final minus pre-correction recall on that same basis.
Input/output tokens, iterations, latency, estimated
cost, and an explicit error rate are also reported. Every nullable metric has a
coverage rate; missing evidence is `null`, never converted to zero.
Tool-selection labels accept dense/hybrid policies for factual questions,
sparse/hybrid policies for keyword questions, hybrid policies for comparisons,
and graph policies for relational questions; unanswerable rows have no tool
gold label and remain `null`.

Offline evaluation intentionally compares heterogeneous answer renderers
(extractive baselines versus the structured deterministic Writer) and records
`generation_regime=offline_heterogeneous`. With `--llm`, every system that
retrieves evidence—including workflow systems—is passed through the same
`evaluation-grounded-answer-v1` prompt and configured fast model after
retrieval. Per-question rows record whether generation was actually called,
the requested and provider-returned model, prompt ID, and provider/estimated
token source. A requested live run fails configuration early when no API key is
available instead of silently falling back.

**Optional RAGAS:** `uv sync --extra eval` and `scholar-agent evaluate --ragas`
(requires a configured DeepSeek/OpenAI-compatible API key and makes paid live
LLM calls). The evaluator explicitly uses the project's configured provider and
the retrieval embedder; it never falls back to RAGAS's implicit OpenAI defaults.
Reports record whether RAGAS was requested, installed, configured, and actually
used. RAGAS 0.3.1 metrics run independently through explicit LangChain
adapters, so one parser/provider failure cannot discard a valid peer metric.
Unavailable or non-finite scores remain `null`, never silently become zero;
per-question rows and run config contain secret-free status, failure code, and
exception class fields. Default CI stays deterministic without paid calls.

Paid RAGAS calls use a versioned disk cache under
`data/evaluation/.cache/ragas_metrics/`. Cache records contain only validated
numeric metrics; API keys, raw provider responses, questions, answers, and
contexts are not persisted. Partial successes are cached under schema
`ragas-metrics-v2`; a second identical run reports `ragas_cached=true` without
another paid judge call.

The eval extra pins `langchain-community` to the compatible 0.3 series and
includes Pillow, which RAGAS 0.3.1 imports even for text-only metrics. A live
DeepSeek smoke run should report `ragas_configured=true`, non-null scores, and
`ragas_coverage_rate=1.0` in `results.json`.

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

# Paid apples-to-apples answer generation (explicit opt-in)
uv run scholar-agent evaluate --embedding-backend hash --llm \
  --output-dir outputs/evaluation/live-shared-generation

make evaluate-smoke
```

Outputs under `outputs/evaluation/` (gitignored recommended):

- `results.json`, `aggregate_metrics.csv`, `per_question_metrics.csv`
- `run_config.json` with dataset/config/code fingerprints
- `report.md`, `failures.json`
- `corrective_before_after.json` with structured initial/final evidence and recall
- SVG charts: aggregate recall, per-category recall, latency, cost, citation precision

An explicit `--embedding-backend hash` uses an isolated deterministic index at
`data/indexes/evaluation/hash/`; it never silently reuses the production BGE
collection. The report stores both requested and actual embedding/reranker names.

## Measured offline audit

The complete 50×7 clean audit run `run_1f4dc371453d4a1f` (commit `eff976a`,
corpus fingerprint `79d20fac…`) used hashing embeddings, lexical reranking, and
a loaded provenance-backed graph. Paper Recall@8 ranged from 0.16
(`naive_dense`) to 0.67 (`hybrid_graph`); the full agent reached 0.54 paper
recall, 0.288 citation precision, and **5/5** unanswerable refusals. Corrective
loops triggered with precision 1.0 but `improvement_after_correction = 0.0`.
These are configuration-specific measurements, not production-model claims.

Committed numeric snapshots (for README / interviews without regenerating
gitignored `outputs/`):

- [`docs/results/offline_hash_eval_summary.md`](results/offline_hash_eval_summary.md)
  (deterministic offline extractive, hash embedder)
- [`docs/results/bge_ce_offline_50x7_summary.md`](results/bge_ce_offline_50x7_summary.md)
  (offline extractive, **BGE + cross-encoder**, `run_a4770534afb84db2`)
- [`docs/results/bge_ce_llm_ragas_50x7_summary.md`](results/bge_ce_llm_ragas_50x7_summary.md)
  (**BGE + CE + live DeepSeek + RAGAS**, `run_6270c2cf8cd94186`)
- [`docs/results/live_llm_ragas_50x7_summary.md`](results/live_llm_ragas_50x7_summary.md)
  (hash retrieval + live DeepSeek + RAGAS, `run_a23467bb0aa84115`)
- [`docs/results/live_shared_llm_eval_summary.md`](results/live_shared_llm_eval_summary.md)
  (earlier generation-only live run)

Local full artifacts (when present):

- `outputs/evaluation/phase8-bge-ce-offline-50x7/`
- `outputs/evaluation/phase8-bge-ce-llm-ragas-50x7/`
- `outputs/evaluation/phase8-final-hash-exact-eff976a/`
- `outputs/evaluation/phase8-full-llm-ragas-50x7/`
- `outputs/evaluation/phase8-final-live-79d20/`

Dataset review: AI-assisted independent review of all 50 questions
(`data/evaluation/manual_review_manifest.json`); **not** a human-signed audit.

See `docs/failure_analysis.md` for measured failure cases.

## Discipline

- Do not tune systems against individual frozen questions in prompts.
- Report **per-category** metrics, not only micro-averages.
- Latency, input/output token counts, estimated cost, and error rate are
  recorded for every question×system.
- Per-category tables include retrieval, answer, citation, latency, tool,
  token, and cost metrics.
- Manual failure analysis: `notebooks/error_analysis.ipynb` and
  `docs/failure_analysis.md`.
