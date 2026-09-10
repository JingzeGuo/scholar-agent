# E3 v5 requirement-assessment Controller ablation

This paired 50-question experiment tests whether an LLM Controller can use the first Researcher
observation to choose a useful follow-up action. Both variants reuse the same saved Planner plan and
the same freshly frozen initial retrieval, rerank candidates, and selected evidence. They use the
same Blackboard Writer policy with `deepseek-v4-flash` at temperature zero.

- `baseline`: initial Researcher observation → Writer
- `controller`: initial observation → one Controller assessment call → zero to two actions in
  one round → Writer

The Controller returns one `status / covered / missing / action` assessment per requirement. It may
choose at most one action per requirement and two actions per question from `search_within_paper`,
`expand_neighbors`, and `increase_depth`. `sufficient` distinguishes supported requirements from
`unresolved` requirements that have no useful bounded action. Gold pages, answer keys, previous
answers, critical chunk IDs, and question-specific rules are never included in its prompt.

E3 v5 keeps the stable `P1`/`P2` paper selector and makes no-action decisions observable. Action
bounds, tools, evidence limits, and Writer are unchanged.

Q014 and Q025 were clarified after E3 v4, so first generate a fresh source run rather than reusing
plans and retrieval observations produced for the older wording:

```bash
uv run python -m evals.evaluate --run-id adaptive_v3_benchmark_aligned run
```

Prepare the frozen observations without making LLM calls:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v5 prepare \
  --source-run adaptive_v3_benchmark_aligned --source-variant adaptive
```

Generate the paired answers. This step needs the local retrieval indexes because Controller actions
run against the corpus:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v5 run
```

Generation is resumable and alternates variant order by question. Then prepare the blinded review
materials:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v5 prepare-review
uv run python -m evals.evaluate_controller --run-id controller_e3_v5 extract-pages
```

Fill `evals/runs/controller_e3_v5/review.csv` using the same binary requirement and citation rubric,
then score it:

```bash
uv run python -m evals.evaluate_controller --run-id controller_e3_v5 score
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
