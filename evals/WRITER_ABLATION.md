# Writer ablation: flat evidence vs Requirement–Evidence Blackboard

This experiment measures the effect of the blackboard context while holding the
retrieved evidence and answer policy fixed. Run commands from the repository root.

| Control | `flat` | `blackboard` |
|---|---|---|
| Planner requirements | Same frozen plan | Same frozen plan |
| Selected chunks and global E IDs | Same frozen passages | Same frozen passages |
| Source filename, title, page, section | Included when available | Included when available |
| Citation and evidence-gap instructions | Identical | Identical |
| Evidence layout | Requirements listed separately; each passage once, in selection order | Passages grouped and ranked per requirement; shared IDs may repeat; empty entries shown |

The baseline is a controlled flat version of the current Writer. This isolates
the blackboard's grouping, links, ordering and empty-entry display. It does not
measure the combined effect of all changes since the older Writer commit.
Repetition can make blackboard prompts longer even though call counts are equal.

## 1. Freeze the inputs — no LLM calls

```bash
uv run python -m evals.evaluate_writer --run-id board_ab_v1 prepare --source-run adaptive_v2 --source-variant adaptive
```

`prepare` uses the saved plans from `evals/runs/adaptive_v2/results.jsonl`, replays
the current Researcher once per question, and requires the selected chunk IDs,
filenames and pages to match the original evidence in order. It freezes all 50
questions, plans, selected passage text, board links and both exact prompts in
`evals/runs/board_ab_v1/inputs.json`. Gold answers and old generated answers are
not passed to the Writer. No Planner or Writer calls occur during preparation.

The corpus, current indexes and local embedding/reranker models are needed for
this step. With cached models, prefix the command with `HF_HUB_OFFLINE=1` to avoid
Hugging Face network checks. The source variant can also be `fixed_hybrid`; it
selects the shared retrieval inputs, not an additional experimental variable.

Inspect `inputs.json` before generation if you want to review the paired prompts.
An existing input snapshot is not overwritten. Use a new run ID for a different
configuration or new prompts.

## 2. Generate both sets of answers

Use the existing `.env` containing `DEEPSEEK_API_KEY`. The experiment uses
`deepseek-v4-flash`, temperature zero, with the same provider settings for both
variants. If `SCHOLAR_AGENT_LLM_MODEL` is set, it must match this model.

```bash
uv run python -m evals.evaluate_writer --run-id board_ab_v1 run
```

This is the step that calls the provider. It produces 100 result records, with
at most 100 Writer calls; questions with no evidence abstain without a call in
both variants. Order alternates by question: flat first, then blackboard first.
No Planner or retrieval runs occur during generation.

Results are appended to `results.jsonl`. If a provider call fails, rerun the same
`run` command: successful samples are skipped and the failed sample is retried.
The frozen-input SHA-256 must match existing records. Editing code after
preparation does not update the saved prompts; prepare a new run to test changes.

## 3. Blind review

```bash
uv run python -m evals.evaluate_writer --run-id board_ab_v1 prepare-review
uv run python -m evals.evaluate_writer --run-id board_ab_v1 extract-pages
```

Grade `evals/runs/board_ab_v1/review.csv` using the packets in
`review_evidence.jsonl` and the [existing scoring rules](README.md#manual-review).
Fill `requirement_scores`, `citation_scores`, `unsupported_claims`, and
`uncited_claims` for all 100 rows. Do not inspect `review_key.json`, results or
input prompts while grading. Both new answer sets need fresh labels; old labels
do not grade newly generated answers. Existing review files are preserved unless
`prepare-review --force` is explicitly requested.

## 4. Compare

```bash
uv run python -m evals.evaluate_writer --run-id board_ab_v1 score
```

`summary.json` and `summary.md` compare `flat` (baseline) with `blackboard`
(treatment), with deltas computed as blackboard minus flat. Inspect:

- **Answer Requirement Accuracy** as the primary metric, including the paired
  requirement rows to see which answers improved or regressed.
- Strict Success and Citation Support for accompanying changes.
- The three retrieval recalls, which should be identical because inputs are fixed.
- Latency and call counts, which cover only Writer generation and citation
  validation, excluding frozen Planner/retrieval preparation. These timings are
  not directly comparable with end-to-end retrieval experiments.

These are the same 50 questions already used for diagnosis. A second independent
generation under a new run ID can check whether apparent gains persist.
