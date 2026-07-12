"""High-level evaluation entrypoint."""

from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.config import AppConfig, load_config
from scholar_agent.evaluation.ablation import AblationConfig, run_ablation
from scholar_agent.evaluation.baselines import ALL_SYSTEMS, SystemRunner
from scholar_agent.evaluation.dataset import (
    EvalDataset,
    load_eval_dataset,
    validate_dataset_against_store,
)
from scholar_agent.evaluation.ragas_runtime import create_ragas_evaluator
from scholar_agent.evaluation.report import (
    EvaluationReport,
    compute_config_fingerprint,
    write_report,
)
from scholar_agent.logging import get_logger
from scholar_agent.retrieval.index_builder import load_toolkit

logger = get_logger(__name__)


def _code_provenance(repo: Path) -> dict[str, Any]:
    """Record the exact Git state needed to interpret reproducibility claims."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None, "code_state_sha256": None}
    state_hash = sha256((commit + "\n").encode("utf-8") + diff + status.encode("utf-8")).hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "code_state_sha256": state_hash,
    }


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
    cost_config = eval_data.get("cost") or {}
    usd_per_1k_tokens = float(cost_config.get("usd_per_1k_tokens", 0.0))
    max_latency_ms = int(budgets.get("max_latency_ms_per_question", 120_000))
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

    toolkit_config = cfg
    if embedding_backend == "hash":
        # Keep the deterministic evaluation index isolated from the production
        # BGE collection so the requested backend is actually honored.
        toolkit_config = cfg.model_copy(
            update={
                "paths": cfg.paths.model_copy(
                    update={"indexes_dir": cfg.paths.indexes_dir / "evaluation" / "hash"}
                )
            }
        )
    toolkit = load_toolkit(
        config=toolkit_config,
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        reranker_backend="lexical" if embedding_backend == "hash" else "auto",
        load_graph=True,
    )
    validate_dataset_against_store(dataset, toolkit.store)
    ragas_evaluator = None
    ragas_status: dict[str, Any] = {
        "available": False,
        "configured": False,
        "reason": "not requested",
    }
    if use_ragas:
        if toolkit.dense is None:
            ragas_status["reason"] = "no embedding model is loaded"
        else:
            ragas_evaluator, ragas_status = create_ragas_evaluator(
                cfg.llm,
                toolkit.dense.embedder,
            )
    runner = SystemRunner(
        toolkit,
        top_k=top_k,
        max_corrective_iterations=cfg.budgets.max_corrective_iterations,
        research_max_tools=cfg.budgets.max_tool_calls_per_research_pass,
        use_llm=use_llm,
        usd_per_1k_tokens=usd_per_1k_tokens,
        max_latency_ms=max_latency_ms,
    )
    ablation_cfg = AblationConfig(
        systems=list(configured_systems),
        top_k=top_k,
        max_questions=max_questions,
        question_ids=question_ids,
        use_ragas=use_ragas,
        ragas_evaluator=ragas_evaluator,
        use_llm=use_llm,
        max_corrective_iterations=cfg.budgets.max_corrective_iterations,
        research_max_tools=cfg.budgets.max_tool_calls_per_research_pass,
        usd_per_1k_tokens=usd_per_1k_tokens,
    )
    report, _rows = run_ablation(dataset, runner, ablation_cfg)
    report.config.update(
        {
            "embedding_backend_requested": embedding_backend,
            "embedding_model_actual": (
                toolkit.dense.embedder.model_name if toolkit.dense is not None else None
            ),
            "reranker_model_actual": (
                toolkit.reranker.model_name if toolkit.reranker is not None else None
            ),
            "dataset_fingerprint": (dataset.split.fingerprint_sha256 if dataset.split else None),
            "questions_path": str(paths.questions_path),
            "reference_evidence_path": str(paths.reference_evidence_path),
            "frozen_split_path": str(paths.frozen_split_path),
            "max_latency_ms_per_question": max_latency_ms,
            "ragas_requested": use_ragas,
            "ragas_available": bool(ragas_status["available"]),
            "ragas_configured": bool(ragas_status.get("configured")),
            "ragas_provider": ragas_status.get("provider"),
            "ragas_model": ragas_status.get("model"),
            "ragas_embedding_model": ragas_status.get("embedding_model"),
            "ragas_status_reason": ragas_status.get("reason"),
            "use_live_llm": use_llm,
            "random_seed": int(eval_data.get("random_seed", 0)),
            **_code_provenance(Path(__file__).resolve().parents[3]),
        }
    )
    report.config_fingerprint_sha256 = compute_config_fingerprint(report.config)
    if use_ragas and ragas_evaluator is None:
        report.notes.append(
            "RAGAS was requested but could not be configured: "
            f"{ragas_status.get('reason', 'unknown reason')}. "
            "RAGAS fields are null rather than zero."
        )
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
        dataset_fingerprint=(dataset.split.fingerprint_sha256 if dataset.split else None),
        n_questions=int(report.config.get("n_questions") or 0),
        systems=list(configured_systems),
    )
