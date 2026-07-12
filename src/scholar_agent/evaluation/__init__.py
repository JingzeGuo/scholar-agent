"""Evaluation framework (Phase 8)."""

from scholar_agent.evaluation.dataset import (
    EvalQuestion,
    FrozenSplit,
    load_eval_dataset,
    load_frozen_split,
    validate_frozen_dataset,
)
from scholar_agent.evaluation.runner import EvaluationRunResult, run_evaluation

__all__ = [
    "EvalQuestion",
    "EvaluationRunResult",
    "FrozenSplit",
    "load_eval_dataset",
    "load_frozen_split",
    "run_evaluation",
    "validate_frozen_dataset",
]
