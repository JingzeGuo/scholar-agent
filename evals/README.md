# Resume-oriented evaluation

This directory compares the production Scholar-Agent workflow with a small,
single-query Hybrid RAG baseline on 50 hand-authored English questions. It is
intentionally limited to three manually checked metrics: requirement accuracy,
citation support, and strict success.

## Run

The local corpus must contain exactly 10,726 chunks and its BM25 and dense
indexes must be current. The evaluation uses `deepseek-v4-flash` only.

```bash
export DEEPSEEK_API_KEY=...
export SCHOLAR_AGENT_LLM_MODEL=deepseek-v4-flash
uv run python evals/evaluate.py run
uv run python evals/evaluate.py prepare-review
```

`run` writes one resumable record per question and variant to `results.jsonl`.
It stops at the first provider or model error; rerunning the command skips
successful records and retries the failed sample.

## Manual review

`prepare-review` creates a deterministically shuffled `review.csv` without a
variant column. Do not inspect `review_key.json` while grading. Fill these
columns for every row:

- `requirement_scores`: JSON object containing every displayed requirement ID,
  for example `{"G1": 1, "G2": 0}`. Give `1` only when an answerable
  requirement is correctly answered, or an unanswerable requirement is not
  fabricated and is explicitly identified as unsupported.
- `citation_scores`: JSON list in the same order as the displayed citations,
  for example `[1, 0]`. Give `1` only when that physical page supports the
  cited claim. Use `[]` when there are no citations.
- `unsupported_claims`: count of factual claims not supported by the corpus.
- `uncited_claims`: count of factual claims that require but lack a citation.
- `notes`: optional review notes.

If a review sheet already contains work, `prepare-review` refuses to overwrite
it. `--force` is available only when replacement is intentional.

After all 100 answers are labeled, run:

```bash
uv run python evals/evaluate.py score
```

This writes `summary.json` and `summary.md`. A question is a strict success only
when every requirement receives `1`, every citation supports its claim, and the
answer has no unsupported or uncited factual claims. Complete and partial
answers must include at least one citation; insufficient answers must include
none.

The benchmark was authored before running either system. Do not edit questions
in response to individual evaluation failures. During authoring, every
unanswerable requirement was searched against the full 10,726-chunk local
corpus; answerable gold pages were also checked against their physical pages.
