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
uv run python evals/evaluate.py extract-pages
```

Pass `--run-id` before the command to keep a new pipeline run separate from
the legacy V0 artifacts:

```bash
uv run python evals/evaluate.py --run-id v1_soft run
uv run python evals/evaluate.py --run-id v1_soft prepare-review
```

To measure the value of the pre-write Coverage Analyzer, run the full system
once without it and once with its advisory annotations and single retrieval
retry. Keep all other settings unchanged:

```bash
uv run python evals/evaluate.py --run-id v2_no_coverage --coverage-mode none run
uv run python evals/evaluate.py --run-id v2_soft_coverage --coverage-mode soft run
```

Prepare and score each run with its matching `--run-id`, then compare the
`full` rows in their generated summaries. The default is `soft`; `none` routes
the Researcher directly to the Writer. The final Answer Verifier and its single
repair remain enabled in both modes.

`run` writes one resumable record per question and variant to `results.jsonl`.
It stops at the first provider or model error; rerunning the command skips
successful records and retries the failed sample.
Versioned runs are stored under `evals/runs/<run-id>/`. Full-system records
also include the plan, coverage and answer verification results, retry and repair
counts, stop reason, and cited pages under `trace`.

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

`extract-pages` creates `review_evidence.jsonl`, with one variant-blinded packet
per answer. Each packet contains the requirements, answer, ordered citation
occurrences, and the extracted text of every cited or gold physical PDF page.
Repeated citations point to one deduplicated `page_ref` inside that packet.

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
