"""Ablation orchestration over the frozen evaluation split."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from scholar_agent.evaluation.answer_metrics import (
    AnswerMetrics,
    RagasEvaluator,
    compute_answer_metrics,
    contradiction_handling_score,
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
    operations: AgentOperationMetrics


class AgentOperationMetrics(BaseModel):
    """Agent metrics with explicit nulls for non-applicable systems/rows."""

    plan_coverage: float | None = None
    tool_selection_accuracy: float | None = None
    corrective_triggered: bool = False
    corrective_trigger_correct: float | None = None
    initial_recall_at_k: float | None = None
    initial_recall_at_k_paper: float | None = None
    correction_recall_basis: str | None = None
    improvement_after_correction: float | None = None


@dataclass
class AblationConfig:
    systems: list[str] = field(default_factory=lambda: list(ALL_SYSTEMS))
    top_k: int = 8
    max_questions: int | None = None
    question_ids: list[str] | None = None
    use_ragas: bool = False
    ragas_evaluator: RagasEvaluator | None = None
    use_llm: bool = False
    max_corrective_iterations: int = 2
    research_max_tools: int = 3
    usd_per_1k_tokens: float = 0.0
    failure_threshold_recall: float = 0.0


def select_questions(dataset: EvalDataset, config: AblationConfig) -> list[EvalQuestion]:
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
    ragas_evaluator: RagasEvaluator | None = None,
) -> QuestionSystemResult:
    output = runner.run(system, question)
    retrieval = compute_retrieval_metrics(question, output.hits, k=top_k)
    cited = set(output.cited_paper_ids)
    # Prefer papers from hits if citations empty
    if not cited:
        cited = {h.paper_id for h in output.hits}
    citation = compute_citation_metrics_from_papers(
        question,
        cited,
        validity_rate=(0.0 if output.error else output.citation_validity_rate),
        page_ok=output.citation_page_ok,
        page_total=output.citation_page_total,
        n_claims=output.n_claims,
        n_unsupported_claims=output.n_unsupported_claims,
    )
    contexts = [h.text for h in output.hits]
    answer = compute_answer_metrics(
        question,
        output.answer_text,
        contexts=contexts,
        use_ragas=use_ragas,
        ragas_evaluator=ragas_evaluator,
    )
    # Override refusal_correct using system prediction when unanswerable
    if question.unanswerable:
        answer = answer.model_copy(
            update={
                "refusal_correct": 1.0 if output.unanswerable_predicted else 0.0,
            }
        )
    contradiction = contradiction_handling_score(
        output.answer_text,
        contradiction_expected=_contradiction_expected(question),
        contradiction_detected=bool(output.metadata.get("conflicting_evidence_ids")),
    )
    answer = answer.model_copy(update={"contradiction_handling_accuracy": contradiction})
    operations = _operation_metrics(question, output, retrieval)
    return QuestionSystemResult(
        question=question,
        output=output,
        retrieval=retrieval,
        citation=citation,
        answer=answer,
        operations=operations,
    )


def run_ablation(
    dataset: EvalDataset,
    runner: SystemRunner,
    config: AblationConfig | None = None,
) -> tuple[EvaluationReport, list[QuestionSystemResult]]:
    cfg = config or AblationConfig()
    if cfg.use_llm != runner.use_llm:
        raise ValueError(
            "AblationConfig.use_llm must match SystemRunner.use_llm so saved run "
            "configuration describes the executed generation path"
        )
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
                ragas_evaluator=cfg.ragas_evaluator,
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
            "question_ids": [question.question_id for question in questions],
            "max_corrective_iterations": cfg.max_corrective_iterations,
            "research_max_tools": cfg.research_max_tools,
            "usd_per_1k_tokens": cfg.usd_per_1k_tokens,
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
                "graph_evidence_recall": r.retrieval.graph_evidence_recall,
                "citation_precision": r.citation.citation_precision,
                "citation_recall": r.citation.citation_recall,
                "citation_validity_rate": r.citation.citation_validity_rate,
                "page_traceability_rate": r.citation.page_traceability_rate,
                "claim_overlap": r.answer.claim_overlap,
                "claim_correctness": r.answer.claim_correctness,
                "completeness": r.answer.completeness,
                "token_f1": r.answer.token_f1,
                "refusal_correct": r.answer.refusal_correct,
                "faithfulness_proxy": r.answer.faithfulness_proxy,
                "ragas_faithfulness": r.answer.ragas_faithfulness,
                "ragas_answer_relevancy": r.answer.ragas_answer_relevancy,
                "ragas_used": r.answer.used_ragas,
                "ragas_status": r.answer.ragas_status,
                "ragas_cached": r.answer.ragas_cached,
                "ragas_failures": [
                    failure.model_dump(mode="json") for failure in r.answer.ragas_failures
                ],
                "contradiction_handling_accuracy": (r.answer.contradiction_handling_accuracy),
                "plan_coverage": r.operations.plan_coverage,
                "tool_selection_accuracy": r.operations.tool_selection_accuracy,
                "corrective_triggered": r.operations.corrective_triggered,
                "corrective_trigger_correct": r.operations.corrective_trigger_correct,
                "initial_recall_at_k": r.operations.initial_recall_at_k,
                "initial_recall_at_k_paper": r.operations.initial_recall_at_k_paper,
                "correction_recall_basis": r.operations.correction_recall_basis,
                "improvement_after_correction": (r.operations.improvement_after_correction),
                "unique_useful_evidence_per_tool_call": (
                    _unique_useful_count(r) / max(1, r.output.tool_call_count)
                ),
                "iteration_count": r.output.iteration_count,
                "latency_ms": r.output.latency_ms,
                "tool_call_count": r.output.tool_call_count,
                "input_tokens": r.output.input_tokens,
                "output_tokens": r.output.output_tokens,
                "token_estimate": r.output.token_estimate,
                "estimated_cost_usd": r.output.estimated_cost_usd,
                "generation_used": bool(r.output.metadata.get("generation_used", False)),
                "generation_model": r.output.metadata.get("generation_model"),
                "generation_prompt_id": r.output.metadata.get("generation_prompt_id"),
                "generation_regime": r.output.metadata.get("generation_regime"),
                "selected_tools": list(r.output.metadata.get("selected_tools") or []),
                "selected_policies": list(r.output.metadata.get("selected_policies") or []),
                "initial_chunk_ids": list(r.output.metadata.get("initial_chunk_ids") or []),
                "initial_paper_ids": list(r.output.metadata.get("initial_paper_ids") or []),
                "final_chunk_ids": [hit.chunk_id for hit in r.output.hits],
                "final_paper_ids": sorted({hit.paper_id for hit in r.output.hits}),
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
                        or ("refusal miss" if r.question.unanswerable else "zero retrieval recall"),
                        "recall_at_k_paper": r.retrieval.recall_at_k_paper,
                        "token_f1": r.answer.token_f1,
                        "answer_preview": (r.output.answer_text or "")[:240],
                    }
                )

    # Keep at least structure for manual analysis even if few failures
    notes = [
        "All systems evaluated on the identical frozen question order.",
        "Deterministic metrics do not require paid APIs; RAGAS is optional.",
        (
            "Claim correctness and completeness are deterministic lexical proxies; "
            "semantic RAGAS fields remain null unless explicitly enabled."
        ),
        (
            "Agent/contradiction metrics use null for non-applicable or unobserved rows; "
            "each summary reports metric coverage."
        ),
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


def _summarize_system(system: str, rows: list[QuestionSystemResult]) -> SystemSummary:
    n = len(rows)
    n_errors = sum(1 for r in rows if r.output.error)

    def avg(getter: Callable[[QuestionSystemResult], float]) -> float:
        return sum(getter(r) for r in rows) / n if n else 0.0

    by_type: dict[str, dict[str, float | None]] = {}
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
            "ndcg_at_k": sum(g.retrieval.ndcg_at_k for g in group) / m,
            "citation_precision": sum(g.citation.citation_precision for g in group) / m,
            "citation_recall": sum(g.citation.citation_recall for g in group) / m,
            "citation_validity_rate": sum(g.citation.citation_validity_rate for g in group) / m,
            "page_traceability_rate": sum(g.citation.page_traceability_rate for g in group) / m,
            "claim_overlap": sum(g.answer.claim_overlap for g in group) / m,
            "claim_correctness": sum(g.answer.claim_correctness for g in group) / m,
            "completeness": sum(g.answer.completeness for g in group) / m,
            "faithfulness_proxy": sum(g.answer.faithfulness_proxy for g in group) / m,
            "refusal_correct": sum(g.answer.refusal_correct for g in group) / m,
            "contradiction_handling_accuracy": _optional_average(
                [g.answer.contradiction_handling_accuracy for g in group]
            ),
            "contradiction_metric_coverage_rate": _coverage(
                [g.answer.contradiction_handling_accuracy for g in group]
            ),
            "plan_coverage": _optional_average([g.operations.plan_coverage for g in group]),
            "plan_coverage_metric_coverage_rate": _coverage(
                [g.operations.plan_coverage for g in group]
            ),
            "tool_selection_accuracy": _optional_average(
                [g.operations.tool_selection_accuracy for g in group]
            ),
            "tool_selection_metric_coverage_rate": _coverage(
                [g.operations.tool_selection_accuracy for g in group]
            ),
            "corrective_trigger_precision": _optional_average(
                [g.operations.corrective_trigger_correct for g in group]
            ),
            "corrective_trigger_metric_coverage_rate": _coverage(
                [g.operations.corrective_trigger_correct for g in group]
            ),
            "improvement_after_correction": _optional_average(
                [g.operations.improvement_after_correction for g in group]
            ),
            "correction_improvement_metric_coverage_rate": _coverage(
                [g.operations.improvement_after_correction for g in group]
            ),
            "error_rate": sum(1 for g in group if g.output.error) / m,
            "avg_latency_ms": sum(g.output.latency_ms for g in group) / m,
            "avg_tool_calls": sum(g.output.tool_call_count for g in group) / m,
            "avg_iterations": sum(g.output.iteration_count for g in group) / m,
            "avg_input_tokens": sum(g.output.input_tokens for g in group) / m,
            "avg_output_tokens": sum(g.output.output_tokens for g in group) / m,
            "avg_tokens": sum(g.output.token_estimate for g in group) / m,
            "estimated_cost_usd": sum(g.output.estimated_cost_usd for g in group),
            "unique_useful_evidence_per_tool_call": sum(
                _unique_useful_count(g) / max(1, g.output.tool_call_count) for g in group
            )
            / m,
        }

    ragas_f = [
        row.answer.ragas_faithfulness for row in rows if row.answer.ragas_faithfulness is not None
    ]
    ragas_r = [
        row.answer.ragas_answer_relevancy
        for row in rows
        if row.answer.ragas_answer_relevancy is not None
    ]
    ragas_used = sum(1 for row in rows if row.answer.used_ragas)
    graph_recall = [
        row.retrieval.graph_evidence_recall
        for row in rows
        if row.retrieval.graph_evidence_recall is not None
    ]
    contradictions = [r.answer.contradiction_handling_accuracy for r in rows]
    plan_coverage = [r.operations.plan_coverage for r in rows]
    tool_accuracy = [r.operations.tool_selection_accuracy for r in rows]
    corrective_precision = [r.operations.corrective_trigger_correct for r in rows]
    corrective_improvement = [r.operations.improvement_after_correction for r in rows]
    return SystemSummary(
        system=system,
        n_questions=n,
        n_errors=n_errors,
        error_rate=n_errors / n if n else 0.0,
        recall_at_k=avg(lambda r: r.retrieval.recall_at_k),
        recall_at_k_paper=avg(lambda r: r.retrieval.recall_at_k_paper),
        mrr=avg(lambda r: r.retrieval.mrr),
        ndcg_at_k=avg(lambda r: r.retrieval.ndcg_at_k),
        graph_evidence_recall=(sum(graph_recall) / len(graph_recall) if graph_recall else None),
        citation_precision=avg(lambda r: r.citation.citation_precision),
        citation_recall=avg(lambda r: r.citation.citation_recall),
        citation_validity_rate=avg(lambda r: r.citation.citation_validity_rate),
        page_traceability_rate=avg(lambda r: r.citation.page_traceability_rate),
        claim_overlap=avg(lambda r: r.answer.claim_overlap),
        claim_correctness=avg(lambda r: r.answer.claim_correctness),
        completeness=avg(lambda r: r.answer.completeness),
        token_f1=avg(lambda r: r.answer.token_f1),
        refusal_correct=avg(lambda r: r.answer.refusal_correct),
        faithfulness_proxy=avg(lambda r: r.answer.faithfulness_proxy),
        ragas_faithfulness=(sum(ragas_f) / len(ragas_f)) if ragas_f else None,
        ragas_answer_relevancy=(sum(ragas_r) / len(ragas_r)) if ragas_r else None,
        ragas_coverage_rate=ragas_used / n if n else 0.0,
        contradiction_handling_accuracy=_optional_average(contradictions),
        contradiction_metric_coverage_rate=_coverage(contradictions),
        plan_coverage=_optional_average(plan_coverage),
        plan_coverage_metric_coverage_rate=_coverage(plan_coverage),
        tool_selection_accuracy=_optional_average(tool_accuracy),
        tool_selection_metric_coverage_rate=_coverage(tool_accuracy),
        corrective_trigger_precision=_optional_average(corrective_precision),
        corrective_trigger_metric_coverage_rate=_coverage(corrective_precision),
        improvement_after_correction=_optional_average(corrective_improvement),
        correction_improvement_metric_coverage_rate=_coverage(corrective_improvement),
        unique_useful_evidence_per_tool_call=avg(
            lambda r: _unique_useful_count(r) / max(1, r.output.tool_call_count)
        ),
        avg_latency_ms=avg(lambda r: float(r.output.latency_ms)),
        avg_tool_calls=avg(lambda r: float(r.output.tool_call_count)),
        avg_iterations=avg(lambda r: float(r.output.iteration_count)),
        avg_input_tokens=avg(lambda r: float(r.output.input_tokens)),
        avg_output_tokens=avg(lambda r: float(r.output.output_tokens)),
        avg_tokens=avg(lambda r: float(r.output.token_estimate)),
        total_estimated_cost_usd=sum(r.output.estimated_cost_usd for r in rows),
        by_type=by_type,
    )


def _unique_useful_count(result: QuestionSystemResult) -> int:
    """Unique retrieved gold chunks, falling back to gold papers."""
    gold_chunks = result.question.gold_chunk_ids()
    if gold_chunks:
        return len({hit.chunk_id for hit in result.output.hits} & gold_chunks)
    gold_papers = result.question.gold_paper_ids()
    return len({hit.paper_id for hit in result.output.hits} & gold_papers)


def _optional_average(values: list[float | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def _coverage(values: list[float | None]) -> float:
    return sum(value is not None for value in values) / len(values) if values else 0.0


def _contradiction_expected(question: EvalQuestion) -> bool:
    note = question.annotation_notes.lower()
    return any(term in note for term in ("contradict", "conflict", "disagree", "inconsisten"))


def _operation_metrics(
    question: EvalQuestion,
    output: SystemOutput,
    final_retrieval: RetrievalMetrics,
) -> AgentOperationMetrics:
    metadata = output.metadata
    adaptive = metadata.get("adaptive_routing") is True
    plan_coverage = _bounded_optional_float(metadata.get("coverage")) if adaptive else None
    tool_accuracy = _tool_selection_accuracy(question, metadata) if adaptive else None
    triggered = metadata.get("corrective_triggered") is True

    initial_chunk_recall: float | None = None
    initial_paper_recall: float | None = None
    if triggered and metadata.get("initial_results_observed") is True and not question.unanswerable:
        initial_chunk_recall = _id_recall(
            set(metadata.get("initial_chunk_ids") or []),
            question.gold_chunk_ids(),
        )
        initial_paper_recall = _id_recall(
            set(metadata.get("initial_paper_ids") or []),
            question.gold_paper_ids(),
        )

    trigger_correct: float | None = None
    improvement: float | None = None
    correction_basis: str | None = None
    initial_basis_recall: float | None = None
    final_basis_recall: float | None = None
    if initial_chunk_recall is not None:
        correction_basis = "chunk"
        initial_basis_recall = initial_chunk_recall
        final_basis_recall = final_retrieval.recall_at_k
    elif initial_paper_recall is not None:
        correction_basis = "paper"
        initial_basis_recall = initial_paper_recall
        final_basis_recall = final_retrieval.recall_at_k_paper
    if triggered and initial_basis_recall is not None and final_basis_recall is not None:
        trigger_correct = 1.0 if initial_basis_recall < 1.0 else 0.0
        improvement = final_basis_recall - initial_basis_recall

    return AgentOperationMetrics(
        plan_coverage=plan_coverage,
        tool_selection_accuracy=tool_accuracy,
        corrective_triggered=triggered,
        corrective_trigger_correct=trigger_correct,
        initial_recall_at_k=initial_chunk_recall,
        initial_recall_at_k_paper=initial_paper_recall,
        correction_recall_basis=correction_basis,
        improvement_after_correction=improvement,
    )


def _tool_selection_accuracy(
    question: EvalQuestion,
    metadata: dict[str, Any],
) -> float | None:
    if question.unanswerable:
        return None
    selected = {
        str(value).lower()
        for value in (
            list(metadata.get("selected_policies") or [])
            + list(metadata.get("selected_tools") or [])
        )
    }
    if not selected:
        return None
    accepted: dict[str, set[str]] = {
        "factual": {"dense", "hybrid", "hybrid_rerank"},
        "keyword": {"sparse", "hybrid", "hybrid_rerank"},
        "comparison": {"hybrid", "hybrid_rerank", "hybrid_plus_graph"},
        "relational": {"graph", "hybrid_plus_graph"},
    }
    expected = accepted.get(question.question_type)
    if not expected:
        return None
    return 1.0 if selected & expected else 0.0


def _id_recall(observed: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(observed & gold) / len(gold)


def _bounded_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None
