# Fixed Hybrid vs Adaptive Retrieval evaluation

This directory compares the two production retrieval modes on the same 50
hand-authored English questions:

```text
Fixed Hybrid: every requirement → BM25 + Dense → RRF → rerank
Adaptive:     every requirement → planned BM25, Dense, or Hybrid → rerank
```

The Planner runs exactly once per question. Its sanitized plan is reused by
both variants, including identical requirements, queries, targets, and
`top_k`. The fixed variant overrides only the strategy field; the adaptive
variant executes the planned strategy. Both variants use the same
cross-encoder, requirement-aware evidence allocation, Writer, and deterministic
citation validation.

## Run

The local corpus must contain exactly 10,726 chunks and its BM25 and dense
indexes must be current. Evaluation uses `deepseek-v4-flash`.

```bash
export DEEPSEEK_API_KEY=...
export SCHOLAR_AGENT_LLM_MODEL=deepseek-v4-flash
uv run python evals/evaluate.py --run-id adaptive_v2 run
uv run python evals/evaluate.py --run-id adaptive_v2 prepare-review
uv run python evals/evaluate.py --run-id adaptive_v2 extract-pages
```

`run` writes one resumable record per question and mode to
`evals/runs/<run-id>/results.jsonl`. It stops at the first provider or model
error; rerunning skips successful records and retries the failed sample.

Every trace includes:

```python
{
    "plan": {...},
    "shared_planner_latency_seconds": 1.23,
    "shared_planner_llm_calls": 1,
    "retrieval_mode": "fixed_hybrid" | "adaptive",
    "retrieval_decisions": [
        {
            "requirement_id": "R1",
            "query": "...",
            "retrieval_strategy": "bm25" | "dense" | "hybrid",
            "top_k": 8,
        }
    ],
    "cited_pages": [...],
}
```

For `fixed_hybrid`, the recorded executed strategy is always `hybrid`; the
Planner's original choice remains in `plan`. These fields support analysis of
strategy proportions, successes by requirement type, retrieval-cost savings,
and failed routing choices.

## Manual review

`prepare-review` creates a deterministically shuffled `review.csv` without a
variant column. Do not inspect `review_key.json` while grading. Fill these
columns for every row:

- `requirement_scores`: JSON object containing every displayed requirement ID,
  for example `{"G1": 1, "G2": 0}`. Give `1` only when an answerable
  requirement is correctly answered, or an unanswerable requirement is not
  fabricated and is explicitly identified as unsupported.
- `citation_scores`: JSON list in displayed citation order, for example
  `[1, 0]`. Give `1` only when that physical page supports the cited claim.
- `unsupported_claims`: number of factual claims not supported by the corpus.
- `uncited_claims`: number of factual claims that require but lack a citation.
- `notes`: optional review notes.

`extract-pages` creates `review_evidence.jsonl`, with one variant-blinded packet
per answer. Each packet contains the requirements, answer, ordered citation
occurrences, and extracted text of every cited or gold physical PDF page.

If a review sheet already contains work, `prepare-review` refuses to overwrite
it. `--force` is available only when replacement is intentional.

After all 100 answers are labeled, run:

```bash
uv run python evals/evaluate.py --run-id adaptive_v2 score
```

The scoring definitions are unchanged. A question is a Strict Success only
when every requirement receives `1`, every citation supports its claim, and
the answer has no unsupported or uncited factual claims. Complete and partial
answers must include at least one citation; insufficient answers must include
none.

The summary reports for both variants:

- Strict Success
- Requirement Accuracy
- Citation Support
- average latency
- average LLM calls

The benchmark was authored before running either system. Do not edit questions
in response to individual evaluation failures.
