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
uv run python evals/evaluate.py --run-id adaptive_v4 run
uv run python evals/evaluate.py --run-id adaptive_v4 prepare-review
uv run python evals/evaluate.py --run-id adaptive_v4 extract-pages
```

`run` writes one resumable record per question and mode to
`evals/runs/<run-id>/results.jsonl`. It stops at the first provider or model
error; rerunning skips successful records and retries the failed sample.

Every trace includes:

```python
{
    "plan": {...},
    "evidence_board": {
        "R1": {"requirement": "...", "evidence_ids": ["E1", "E3"]},
        "R2": {"requirement": "...", "evidence_ids": []},
    },
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
    "retrieval_stages": {
        "retrieval": [{"paper": "1908.10084.pdf", "page": 1}, ...],
        "rerank": [{"paper": "1908.10084.pdf", "page": 1}, ...],
    },
    "cited_pages": [...],
}
```

For `fixed_hybrid`, the recorded executed strategy is always `hybrid`; the
Planner's original choice remains in `plan`. These fields support analysis of
strategy proportions, successes by requirement type, retrieval-cost savings,
and failed routing choices.

Saved `evidence` items also retain their stable `id`, `paper_id`, optional
`title`/`section`, `supports`, and `requirement_scores`, so the Writer's grouped
context can be inspected alongside the board. Links use the per-requirement
score threshold without requiring literal target-name matches; they are relevance
hints, not semantic support labels. The Writer sees empty requirements explicitly
and also receives any selected passages that were not linked to a requirement.

## Requirement-level stage evaluation

`run` automatically writes `requirement_metrics` to each result using the existing
`gold_pages`. This needs no extra annotation or LLM calls. `score` combines these
metrics with the existing answer review scores in `summary.json` and `summary.md`:

```text
Requirement  Retrieval Recall → Rerank Recall → Selected Evidence Recall → Answer Requirement Accuracy
G2                  ✓                ✓                    ✓                           ✗
```

- **Retrieval Recall**: gold pages found in the union of all executed BM25/Dense
  results, before candidate selection.
- **Rerank Recall**: gold pages entering the reranker, after the 30-chunk candidate
  limit and before reranker score filtering.
- **Selected Evidence Recall**: gold pages in the actual `evidence` passed to the
  Writer, after score filtering and evidence allocation.
- **Answer Requirement Accuracy**: the existing 0/1 review score for that gold
  requirement; `null` until the review is scored. The aggregate retains the JSON
  key `requirement_accuracy` for compatibility.

Each recall is `distinct gold pages present / distinct gold pages`, matching both
PDF filename and physical page number. Repeated chunks on a page count once.
Summaries macro-average the per-requirement recalls and include each stage's
eligible requirement count (`retrieval_requirements`, `rerank_requirements`,
`selected_evidence_requirements`). Requirements without gold pages are `null`
(`N/A` in Markdown) and excluded from recall averages, but still count toward
answer accuracy. Partial coverage appears as a percentage instead of ✓ or ✗.

Gold IDs (`G1`, `G2`, ...) need not align with the Planner's requirements (`R1`,
`R2`, ...). Each gold requirement is checked against the shared pool at each
stage, including pages retrieved for another planned requirement.

The example above directs investigation toward the Writer and the selected
chunks. A page hit establishes page coverage; it does not guarantee that the
selected chunk contains the supporting passage. Loss between Rerank Recall and
Selected Evidence Recall includes both score filtering and evidence allocation.

Use a new run ID for the new `adaptive_v4_evidence_board` pipeline version;
resuming an older version is rejected to avoid mixing trace formats. Old results
can still be scored: missing stage snapshots are `null`/`N/A`, not zero, and
Selected Evidence Recall can be recovered from their saved evidence.

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
uv run python evals/evaluate.py --run-id adaptive_v4 score
```

The scoring definitions are unchanged. A question is a Strict Success only
when every requirement receives `1`, every citation supports its claim, and
the answer has no unsupported or uncited factual claims. Complete and partial
answers must include at least one citation; insufficient answers must include
none.

The summary reports for both variants:

- Strict Success
- Retrieval Recall
- Rerank Recall
- Selected Evidence Recall
- Answer Requirement Accuracy
- Citation Support
- average latency
- average LLM calls

The summary also lists all four metrics for every question, variant, and gold
requirement, so a failure can be traced through the stages.

The benchmark was authored before running either system. Do not edit questions
in response to individual evaluation failures.
