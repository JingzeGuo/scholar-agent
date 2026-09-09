# E2a targeted retrieval recovery

This paired experiment checks whether deterministic follow-up actions recover the exact passages
missing from three known answer failures. It does not change the production workflow or test an
autonomous Controller.

| Case | Requirement | Follow-up action |
|---|---|---|
| Q014 | CRAG mechanisms | paper-local search, then adjacent chunks |
| Q018 | CRAG correction | paper-local search, then adjacent chunks |
| Q025 | LightRAG dual-level retrieval | increase the requirement's depth from 8 to 12 |

Both variants replay the same saved Planner plan and initial retrieval. Both use the Blackboard
Writer prompt and `deepseek-v4-flash`. The treatment reranks follow-up candidates with the existing
cross-encoder and adds at most two passages per targeted requirement. `prepare` makes no LLM calls.

```bash
uv run python -m evals.evaluate_recovery --run-id recovery_e2a_v1 prepare
uv run python -m evals.evaluate_recovery --run-id recovery_e2a_v1 run
uv run python -m evals.evaluate_recovery --run-id recovery_e2a_v1 prepare-review
uv run python -m evals.evaluate_recovery --run-id recovery_e2a_v1 extract-pages
```

Fill the six blinded rows in `evals/runs/recovery_e2a_v1/review.csv`, then run:

```bash
uv run python -m evals.evaluate_recovery --run-id recovery_e2a_v1 score
```

The summary reports the critical-evidence chain separately from page recall:

```text
Initial critical chunk → Recovered → Reranked → Selected → Requirement score before/after
```

The critical chunk IDs and manual action parameters exist only in this failure-validation script.
They are never written into production state or used by the production workflow.
