# E3 evidence-gap Controller ablation

This paired 50-question experiment tests whether an LLM Controller can use the first Researcher
observation to choose a useful follow-up action. Both variants reuse the same saved Planner plan and
the same freshly frozen initial retrieval, rerank candidates, and selected evidence. They use the
same Blackboard Writer policy with `deepseek-v4-flash` at temperature zero.

- `baseline`: initial Researcher observation → Writer
- `controller`: initial observation → one Controller call → zero to two actions in one round → Writer

The Controller may choose at most one action per requirement and two actions per question from
`search_within_paper`, `expand_neighbors`, and `increase_depth`. An empty action list goes directly
to the Writer. Gold pages, answer keys, previous answers, critical chunk IDs, and question-specific
rules are never included in its prompt.

Prepare the frozen observations without making LLM calls:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v1 prepare \
  --source-run adaptive_v2 --source-variant adaptive
```

Generate the paired answers. This step needs the local retrieval indexes because Controller actions
run against the corpus:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v1 run
```

Generation is resumable and alternates variant order by question. Then prepare the blinded review
materials:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v1 prepare-review
uv run python -m evals.evaluate_controller --run-id controller_e3_v1 extract-pages
```

Fill `evals/runs/controller_e3_v1/review.csv` using the same binary requirement and citation rubric,
then score it:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v1 score
```

The summary reports Strict Success, Requirement Accuracy, Citation Support, stage recall, latency,
and LLM calls, plus:

- follow-up trigger rate, rejected action rate, and action distribution;
- Retrieval Recovery Rate over gold pages missed by initial retrieval;
- useful action rate, where an action adds at least one new evidence chunk;
- average retrieval operations;
- answer requirement repairs and regressions.

Latency covers Controller, recovery, Writer, and citation validation after the frozen initial
Researcher observation. The paired latency delta therefore isolates the follow-up overhead.
Production keeps `SCHOLAR_AGENT_RECOVERY_MODE=none` until this experiment supports enabling it.
