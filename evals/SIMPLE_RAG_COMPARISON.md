# Controller-enabled Scholar-Agent vs Simple RAG

This paired 50-question experiment compares the complete current Scholar-Agent
pipeline, with its assessment-first Controller enabled, against a single-query
Simple RAG pipeline.

```text
Simple RAG
Original question
→ BM25 top-8 + Dense top-8
→ reciprocal rank fusion
→ cross-encoder rerank
→ score threshold + fixed maximum of 8 evidence chunks
→ current Writer policy with flat evidence context
→ deterministic Citation Validator

Current Scholar-Agent
Original question
→ Adaptive Planner
→ per-requirement retrieval, rerank, and evidence selection
→ Requirement–Evidence Blackboard
→ assessment-first Controller
→ zero to two recovery actions in one round
→ Writer
→ deterministic Citation Validator
```

Both variants use `deepseek-v4-flash` at temperature zero, the same corpus,
indexes, reranker, score threshold, Writer answer policy, and citation
validator. Simple RAG has no Planner, requirement decomposition, target-aware
selection, Blackboard grouping, Controller, or recovery. Its evidence budget is
always at most eight; it is not dynamically matched to the current system.

Gold requirements, answer keys, and gold pages are used only after generation
for review and retrieval diagnostics. They are not written to frozen runtime
states or prompts.

## Run

Create fresh pre-Writer states for all 50 updated questions. This stage is
resumable and includes the Planner and Controller model calls for the current
pipeline:

```bash
uv run python -m evals.evaluate_simple_rag \
  --run-id controller_vs_simple_rag_v1 prepare
```

Generate both answers from the frozen states. Writer order alternates 25/25:

```bash
uv run python -m evals.evaluate_simple_rag \
  --run-id controller_vs_simple_rag_v1 run
```

Create the blinded review materials:

```bash
uv run python -m evals.evaluate_simple_rag \
  --run-id controller_vs_simple_rag_v1 prepare-review
uv run python -m evals.evaluate_simple_rag \
  --run-id controller_vs_simple_rag_v1 extract-pages
```

Fill `evals/runs/controller_vs_simple_rag_v1/review.csv` with the binary
requirement and citation rubric in `evals/README.md`, then score:

```bash
uv run python -m evals.evaluate_simple_rag \
  --run-id controller_vs_simple_rag_v1 score
```

## Recorded controls

`metadata.json` records the fixed configuration and the benchmark hash.
`prepared_inputs.jsonl` stores one freshly generated pair per question,
including both states, prompts, per-state and per-prompt SHA-256 values,
pre-Writer latencies, LLM calls, retrieval operations, and preparation order.
`results.jsonl` stores the alternating Writer order, full-pipeline stage sums,
answers, selected evidence, retrieval stages, Controller assessments/actions,
and citation traces.

The scorer reports Strict Success, Requirement Accuracy, Citation Support,
retrieval/rerank/selected-evidence recall, full staged latency, LLM calls,
retrieval operations, Controller actions, paired repairs/regressions, and exact
McNemar p-values. Gold-page recall remains a non-exhaustive diagnostic rather
than a complete evidence-sufficiency judgment.
