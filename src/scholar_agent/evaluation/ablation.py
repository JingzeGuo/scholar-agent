"""Ablation orchestration over the frozen evaluation split."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from scholar_agent.evaluation.answer_metrics import (
    AnswerMetrics,
    compute_answer_metrics,
)
from scholar_agent.evaluation.baselines import ALL_SYSTEMS, SystemOutput, SystemRunner
from scholar_agent.evaluation.citation_metrics import (
    CitationMetrics,
    compute_citation_metrics_from_papers,
)
from scholar_agent.evaluation.dataset import EvalDataset, EvalQuestion
from scholar_agent.evaluation.report import EvaluationReport, SystemSummary
from scholar_agent.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
)
from scholar_agent.ids import new_run_id
from scholar_agent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuestionSystemResult:
    question: EvalQuestion
    output: SystemOutput
    retrieval: RetrievalMetrics
    citation: CitationMetrics
    answer: AnswerMetrics


@dataclass
class AblationConfig:
    systems: list[str] = field(default_factory=lambda: list(ALL_SYSTEMS))
    top_k: int = 8
    max_questions: int | None = None
    question_ids: list[str] | None = None
    use_ragas: bool = False
    use_llm: bool = False
    max_corrective_iterations: int = 2
    research_max_tools: int = 3
    usd_per_1k_tokens: float = 0.0
    failure_threshold_recall: float = 0.0


def select_questions(
    dataset: EvalDataset, config: AblationConfig
) -> list[EvalQuestion]:
    questions = dataset.ordered()
    if config.question_ids:
        wanted = set(config.question_ids)
        questions = [q for q in questions if q.question_id in wanted]
    if config.max_questions is not None:
        questions = questions[: config.max_questions]
    return questions


def evaluate_one(
    runner: SystemRunner,
    system: str,
    question: EvalQuestion,
    *,
    top_k: int,
    use_ragas: bool,
) -> QuestionSystemResult:
    output = runner.run(system, question)
    retrieval = compute_retrieval_metrics(question, output.hits, k=top_k)
    cited = set(output.cited_paper_ids)
    # Prefer papers from hits if citations empty
    if not cited:
        cited = {h.paper_id for h in output.hits}
    page_ok = sum(1 for h in output.hits if h.page_start >= 1)
    page_total = len(output.hits)
    citation = compute_citation_metrics_from_papers(
        question,
        cited,
        validity_rate=0.0 if output.error else 1.0,
        page_ok=page_ok,
        page_total=page_total,
        n_claims=1 if output.answer_text.strip() else 0,
        n_unsupported_claims=0,
    )
    contexts = [h.text for h in output.hits]
    answer = compute_answer_metrics(
        question,
        output.answer_text,
        contexts=contexts,
        use_ragas=use_ragas,
    )
    # Override refusal_correct using system prediction when unanswerable
    if question.unanswerable:
        answer = answer.model_copy(
            update={
                "refusal_correct": 1.0 if output.unanswerable_predicted else 0.0,
            }
        )
    return QuestionSystemResult(
        question=question,
        output=output,
        retrieval=retrieval,
        citation=citation,
        answer=answer,
    )


def run_ablation(
    dataset: EvalDataset,
    runner: SystemRunner,
    config: AblationConfig | None = None,
) -> tuple[EvaluationReport, list[QuestionSystemResult]]:
    cfg = config or AblationConfig()
    questions = select_questions(dataset, cfg)
    systems: Sequence[str] = cfg.systems or list(ALL_SYSTEMS)
    run_id = new_run_id()
    results: list[QuestionSystemResult] = []

    for system in systems:
        logger.info("ablation system=%s questions=%s", system, len(questions))
        for question in questions:
            result = evaluate_one(
                runner,
                system,
                question,
                top_k=cfg.top_k,
                use_ragas=cfg.use_ragas,
            )
            results.append(result)

    report = build_report(
        run_id=run_id,
        dataset=dataset,
        results=results,
        systems=list(systems),
        config={
            "systems": list(systems),
            "top_k": cfg.top_k,
            "max_questions": cfg.max_questions,
            "use_ragas": cfg.use_ragas,
            "use_llm": cfg.use_llm,
            "n_questions": len(questions),
        },
        failure_threshold_recall=cfg.failure_threshold_recall,
    )
    return report, results


def build_report(
    *,
    run_id: str,
    dataset: EvalDataset,
    results: list[QuestionSystemResult],
    systems: list[str],
    config: dict[str, Any],
    failure_threshold_recall: float = 0.0,
) -> EvaluationReport:
    per_question: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summaries: list[SystemSummary] = []

    for system in systems:
        rows = [r for r in results if r.output.system == system]
        if not rows:
            continue
        summary = _summarize_system(system, rows)
        summaries.append(summary)
        for r in rows:
            row = {
                "system": system,
                "question_id": r.question.question_id,
                "question_type": r.question.question_type,
                "unanswerable": r.question.unanswerable,
                "recall_at_k": r.retrieval.recall_at_k,
                "recall_at_k_paper": r.retrieval.recall_at_k_paper,
                "mrr": r.retrieval.mrr,
                "ndcg_at_k": r.retrieval.ndcg_at_k,
                "citation_precision": r.citation.citation_precision,
                "citation_recall": r.citation.citation_recall,
                "citation_validity_rate": r.citation.citation_validity_rate,
                "page_traceability_rate": r.citation.page_traceability_rate,
                "claim_overlap": r.answer.claim_overlap,
                "token_f1": r.answer.token_f1,
                "refusal_correct": r.answer.refusal_correct,
                "faithfulness_proxy": r.answer.faithfulness_proxy,
                "latency_ms": r.output.latency_ms,
                "tool_call_count": r.output.tool_call_count,
                "token_estimate": r.output.token_estimate,
                "estimated_cost_usd": r.output.estimated_cost_usd,
                "error": r.output.error or "",
            }
            per_question.append(row)
            is_fail = bool(r.output.error)
            if not r.question.unanswerable:
                is_fail = is_fail or (
                    r.retrieval.recall_at_k_paper <= failure_threshold_recall
                    and r.retrieval.recall_at_k <= failure_threshold_recall
                )
            else:
                is_fail = is_fail or r.answer.refusal_correct < 1.0
            if is_fail:
                failures.append(
                    {
                        "system": system,
                        "question_id": r.question.question_id,
                        "question_type": r.question.question_type,
                        "question": r.question.question,
                        "reason": r.output.error
                        or (
                            "refusal miss"
                            if r.question.unanswerable
                            else "zero retrieval recall"
                        ),
                        "recall_at_k_paper": r.retrieval.recall_at_k_paper,
                        "token_f1": r.answer.token_f1,
                        "answer_preview": (r.output.answer_text or "")[:240],
                    }
                )

    # Keep at least structure for manual analysis even if few failures
    notes = [
        "All systems evaluated on the identical frozen question order.",
        "Deterministic metrics do not require paid APIs; RAGAS is optional.",
        f"Failure rows: {len(failures)} (threshold_recall={failure_threshold_recall}).",
    ]
    fp = dataset.split.fingerprint_sha256 if dataset.split else None
    return EvaluationReport(
        run_id=run_id,
        config=config,
        frozen_split_fingerprint=fp,
        systems=summaries,
        per_question=per_question,
        failures=failures,
        notes=notes,
    )


def _summarize_system(
    system: str, rows: list[QuestionSystemResult]
) -> SystemSummary:
    n = len(rows)
    n_errors = sum(1 for r in rows if r.output.error)

    def avg(getter: Callable[[QuestionSystemResult], float]) -> float:
        return sum(getter(r) for r in rows) / n if n else 0.0

    by_type: dict[str, dict[str, float]] = {}
    type_groups: dict[str, list[QuestionSystemResult]] = {}
    for r in rows:
        type_groups.setdefault(r.question.question_type, []).append(r)
    for qtype, group in type_groups.items():
        m = len(group)
        by_type[qtype] = {
            "n": float(m),
            "recall_at_k": sum(g.retrieval.recall_at_k for g in group) / m,
            "recall_at_k_paper": sum(g.retrieval.recall_at_k_paper for g in group) / m,
            "mrr": sum(g.retrieval.mrr for g in group) / m,
            "citation_precision": sum(g.citation.citation_precision for g in group) / m,
            "claim_overlap": sum(g.answer.claim_overlap for g in group) / m,
            "refusal_correct": sum(g.answer.refusal_correct for g in group) / m,
            "avg_latency_ms": sum(g.output.latency_ms for g in group) / m,
        }

    return SystemSummary(
        system=system,
        n_questions=n,
        n_errors=n_errors,
        recall_at_k=avg(lambda r: r.retrieval.recall_at_k),
        recall_at_k_paper=avg(lambda r: r.retrieval.recall_at_k_paper),
        mrr=avg(lambda r: r.retrieval.mrr),
        ndcg_at_k=avg(lambda r: r.retrieval.ndcg_at_k),
        citation_precision=avg(lambda r: r.citation.citation_precision),
        citation_recall=avg(lambda r: r.citation.citation_recall),
        citation_validity_rate=avg(lambda r: r.citation.citation_validity_rate),
        page_traceability_rate=avg(lambda r: r.citation.page_traceability_rate),
        claim_overlap=avg(lambda r: r.answer.claim_overlap),
        token_f1=avg(lambda r: r.answer.token_f1),
        refusal_correct=avg(lambda r: r.answer.refusal_correct),
        faithfulness_proxy=avg(lambda r: r.answer.faithfulness_proxy),
        avg_latency_ms=avg(lambda r: float(r.output.latency_ms)),
        avg_tool_calls=avg(lambda r: float(r.output.tool_call_count)),
        avg_tokens=avg(lambda r: float(r.output.token_estimate)),
        total_estimated_cost_usd=sum(r.output.estimated_cost_usd for r in rows),
        by_type=by_type,
    )
