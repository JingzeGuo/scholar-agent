"""High-level evaluation entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.config import AppConfig, load_config
from scholar_agent.evaluation.ablation import AblationConfig, run_ablation
from scholar_agent.evaluation.baselines import ALL_SYSTEMS, SystemRunner
from scholar_agent.evaluation.dataset import EvalDataset, load_eval_dataset
from scholar_agent.evaluation.report import EvaluationReport, write_report
from scholar_agent.logging import get_logger
from scholar_agent.retrieval.index_builder import load_toolkit

logger = get_logger(__name__)


class EvaluationPaths(BaseModel):
    questions_path: Path
    reference_evidence_path: Path
    frozen_split_path: Path
    output_dir: Path


class EvaluationRunResult(BaseModel):
    report: EvaluationReport
    output_paths: dict[str, str] = Field(default_factory=dict)
    dataset_fingerprint: str | None = None
    n_questions: int = 0
    systems: list[str] = Field(default_factory=list)


def resolve_eval_paths(
    config: AppConfig,
    *,
    eval_yaml: Path | None = None,
    output_dir: Path | str | None = None,
) -> EvaluationPaths:
    data = _load_eval_yaml(eval_yaml) if eval_yaml else {}
    ds = data.get("dataset") or {}
    # Prefer paths relative to repo root
    repo = Path(__file__).resolve().parents[3]

    def _p(key: str, default: str) -> Path:
        raw = Path(ds.get(key, default))
        return raw if raw.is_absolute() else (repo / raw).resolve()

    questions = _p("questions_path", "data/evaluation/questions.jsonl")
    reference = _p("reference_evidence_path", "data/evaluation/reference_evidence.jsonl")
    frozen = _p("frozen_split_path", "data/evaluation/frozen_split.json")
    out = Path(output_dir) if output_dir else repo / "outputs" / "evaluation"
    if not out.is_absolute():
        out = (repo / out).resolve()
    return EvaluationPaths(
        questions_path=questions,
        reference_evidence_path=reference,
        frozen_split_path=frozen,
        output_dir=out,
    )


def _load_eval_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"evaluation config must be a mapping: {path}")
    return data


def run_evaluation(
    *,
    config: AppConfig | None = None,
    eval_config_path: Path | str | None = None,
    systems: list[str] | None = None,
    max_questions: int | None = None,
    question_ids: list[str] | None = None,
    embedding_backend: str = "hash",
    top_k: int = 8,
    use_ragas: bool = False,
    use_llm: bool = False,
    output_dir: Path | str | None = None,
    write_charts: bool = True,
) -> EvaluationRunResult:
    """Run all selected systems on the frozen split and write reports."""
    cfg = config or load_config()
    eval_yaml = Path(eval_config_path) if eval_config_path else None
    if eval_yaml is None:
        candidate = Path("configs/evaluation.yaml")
        if candidate.is_file():
            eval_yaml = candidate
    paths = resolve_eval_paths(cfg, eval_yaml=eval_yaml, output_dir=output_dir)

    eval_data = {}
    if eval_yaml and eval_yaml.is_file():
        eval_data = _load_eval_yaml(eval_yaml)
    budgets = eval_data.get("budgets") or {}
    if max_questions is None:
        max_questions = budgets.get("max_questions")
    if not use_llm:
        use_llm = bool(budgets.get("use_live_llm", False))
    if not use_ragas:
        metrics = (eval_data.get("metrics") or {}).get("answer") or []
        # only enable ragas when explicitly requested via flag; config lists metrics
        use_ragas = False
        _ = metrics

    configured_systems = systems or eval_data.get("systems") or list(ALL_SYSTEMS)
    # Normalize config names like naive_rag → naive_dense
    alias = {
        "naive_rag": "naive_dense",
        "hybrid": "hybrid_rag",
        "full_scholaragent": "full_agent",
        "full": "full_agent",
    }
    configured_systems = [alias.get(s, s) for s in configured_systems]

    dataset: EvalDataset = load_eval_dataset(
        questions_path=paths.questions_path,
        reference_evidence_path=paths.reference_evidence_path,
        frozen_split_path=paths.frozen_split_path,
        validate=True,
    )

    toolkit = load_toolkit(
        config=cfg,
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        reranker_backend="lexical" if embedding_backend == "hash" else "auto",
        load_graph=True,
    )
    runner = SystemRunner(
        toolkit,
        top_k=top_k,
        max_corrective_iterations=cfg.budgets.max_corrective_iterations,
        research_max_tools=cfg.budgets.max_tool_calls_per_research_pass,
        use_llm=use_llm,
    )
    ablation_cfg = AblationConfig(
        systems=list(configured_systems),
        top_k=top_k,
        max_questions=max_questions,
        question_ids=question_ids,
        use_ragas=use_ragas,
        use_llm=use_llm,
        max_corrective_iterations=cfg.budgets.max_corrective_iterations,
        research_max_tools=cfg.budgets.max_tool_calls_per_research_pass,
    )
    report, _rows = run_ablation(dataset, runner, ablation_cfg)
    out_paths = write_report(report, paths.output_dir, write_charts=write_charts)
    logger.info(
        "evaluation complete systems=%s questions=%s out=%s",
        configured_systems,
        report.config.get("n_questions"),
        paths.output_dir,
    )
    return EvaluationRunResult(
        report=report,
        output_paths={k: str(v) for k, v in out_paths.items()},
        dataset_fingerprint=(
            dataset.split.fingerprint_sha256 if dataset.split else None
        ),
        n_questions=int(report.config.get("n_questions") or 0),
        systems=list(configured_systems),
    )
