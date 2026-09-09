"""Paired E3 ablation of a one-shot evidence-gap Controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path

from evals import evaluate as evaluation
from evals import evaluate_writer as writer_experiment
from scholar_agent.agents.controller import _controller_prompt, controller_node
from scholar_agent.agents.recovery import recovery_node
from scholar_agent.agents.researcher import researcher_node
from scholar_agent.agents.writer import SAFE_ABSTENTION, _writer_prompt, citation_validator_node
from scholar_agent.config import Settings
from scholar_agent.indexes import ModelUnavailableError
from scholar_agent.retrieval import RetrievalEngine
from scholar_agent.workflow import initial_state

VARIANTS = ("baseline", "controller")
PIPELINE_VERSION = "evidence_gap_controller_e3_v1"


def prepare_inputs(
    questions: Sequence[dict],
    engine: RetrievalEngine,
    settings: Settings,
    source_results_path: Path,
    inputs_path: Path,
    *,
    source_variant: str = "adaptive",
    researcher_runner: Callable = researcher_node,
) -> dict:
    """Replay each saved plan once and freeze the initial Researcher observation."""
    if inputs_path.exists():
        raise evaluation.EvaluationError("Frozen inputs already exist; use a new --run-id")
    source = evaluation._complete_result_set(questions, source_results_path, (source_variant,))
    samples = []
    for question in questions:
        saved = source[(question["id"], source_variant)]
        state = initial_state(question["question"], source_variant)
        state["plan"] = deepcopy(saved["trace"]["plan"])
        state.update(researcher_runner(state, engine, settings))
        samples.append(
            {
                "question_id": question["id"],
                "state": state,
                "baseline_prompt": _writer_prompt(state),
                "controller_prompt": _controller_prompt(state),
            },
        )
        print(f"prepared {question['id']}", flush=True)

    inputs = {
        "pipeline_version": PIPELINE_VERSION,
        "model": evaluation.MODEL_NAME,
        "source_results": str(source_results_path.resolve()),
        "source_sha256": hashlib.sha256(source_results_path.read_bytes()).hexdigest(),
        "retrieval_mode": source_variant,
        "reranker_model": settings.reranker_model,
        "min_rerank_score": settings.min_rerank_score,
        "empty_evidence_answer": SAFE_ABSTENTION,
        "questions": list(questions),
        "samples": samples,
    }
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inputs


def _initial_operations(state: dict) -> int:
    return sum(
        2 if request["retrieval_strategy"] == "hybrid" else 1
        for request in state["retrieval_trace"]
    )


def _recovery_operations(state: dict) -> int:
    return sum(
        2
        if trace["action"] == "search_within_paper"
        or (
            trace["action"] == "increase_depth"
            and trace["parameters"]["retrieval_strategy"] == "hybrid"
        )
        else 1
        for trace in state["recovery_trace"]
    )


def run_experiment(
    inputs_path: Path,
    results_path: Path,
    engine: RetrievalEngine,
    settings: Settings,
    llm: evaluation.CountingLLM,
    *,
    run_id: str,
) -> None:
    """Alternate baseline and Controller generation from each frozen observation."""
    inputs, input_hash = writer_experiment.load_inputs(inputs_path, PIPELINE_VERSION)
    if (
        settings.reranker_model != inputs["reranker_model"]
        or settings.min_rerank_score != inputs["min_rerank_score"]
    ):
        raise evaluation.EvaluationError("Recovery reranker settings differ from frozen inputs")
    latest = writer_experiment._existing_results(
        results_path,
        run_id,
        input_hash,
        pipeline_version=PIPELINE_VERSION,
        variants=VARIANTS,
    )
    questions = {item["id"]: item for item in inputs["questions"]}
    for index, sample in enumerate(inputs["samples"]):
        question_id = sample["question_id"]
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            if evaluation._successful_result(latest, question_id, variant) is not None:
                continue
            state = deepcopy(sample["state"])
            if _writer_prompt(state) != sample["baseline_prompt"]:
                raise evaluation.EvaluationError("Writer policy differs from frozen input")
            state["recovery_mode"] = variant if variant == "controller" else "none"
            calls_before = llm.calls
            started = time.perf_counter()
            controller_latency = recovery_latency = 0.0
            try:
                if variant == "controller":
                    if _controller_prompt(state) != sample["controller_prompt"]:
                        raise evaluation.EvaluationError("Controller prompt differs from frozen input")
                    phase = time.perf_counter()
                    state.update(controller_node(state, llm))
                    controller_latency = time.perf_counter() - phase
                    if state["controller_trace"]["actions"]:
                        phase = time.perf_counter()
                        state.update(recovery_node(state, engine, settings))
                        recovery_latency = time.perf_counter() - phase

                writer_prompt = (
                    sample["baseline_prompt"] if variant == "baseline" else _writer_prompt(state)
                )
                phase = time.perf_counter()
                state["answer"] = (
                    llm.complete(writer_prompt).strip()
                    if state["evidence"]
                    else inputs["empty_evidence_answer"]
                )
                state.update(citation_validator_node(state))
                writer_latency = time.perf_counter() - phase
                trace = evaluation._trace(state["answer"], state, 0.0, 0)
                trace.update(
                    experiment_variant=variant,
                    controller_prompt_sha256=(
                        hashlib.sha256(sample["controller_prompt"].encode()).hexdigest()
                        if variant == "controller"
                        else None
                    ),
                    writer_prompt_sha256=hashlib.sha256(writer_prompt.encode()).hexdigest(),
                    controller_latency_seconds=round(controller_latency, 4),
                    recovery_latency_seconds=round(recovery_latency, 4),
                    writer_latency_seconds=round(writer_latency, 4),
                    retrieval_operations=(
                        _initial_operations(state)
                        + (_recovery_operations(state) if variant == "controller" else 0)
                    ),
                    latency_scope="controller_recovery_writer_after_frozen_initial_research",
                )
                result = evaluation._result_record(
                    run_id,
                    PIPELINE_VERSION,
                    question_id,
                    variant,
                    state["answer"],
                    state["evidence"],
                    time.perf_counter() - started,
                    llm.calls - calls_before,
                    trace,
                )
                result["requirement_metrics"] = evaluation.requirement_stage_metrics(
                    questions[question_id], result,
                )
            except Exception as exc:
                failed = evaluation._error_record(
                    run_id,
                    PIPELINE_VERSION,
                    question_id,
                    variant,
                    exc,
                    time.perf_counter() - started,
                    llm.calls - calls_before,
                )
                failed["input_sha256"] = input_hash
                evaluation._append_result(results_path, failed)
                raise evaluation.EvaluationError(
                    f"Controller experiment failed for {question_id}/{variant}: {exc}",
                ) from exc
            result["input_sha256"] = input_hash
            evaluation._append_result(results_path, result)
            latest[(question_id, variant)] = result
            print(f"completed {question_id} {variant}", flush=True)


def _pages(items: Sequence[dict]) -> set[tuple[str, int]]:
    return {(item["paper"], item["page"]) for item in items}


def controller_summary(summary: dict, questions: Sequence[dict], results_path: Path) -> str:
    """Add action, recovery, and answer-repair diagnostics to the scored summary."""
    results = evaluation._complete_result_set(questions, results_path, VARIANTS)
    actions = []
    useful_actions = 0
    missed_pages = recovered_pages = 0
    baseline_operations = controller_operations = 0
    triggered_questions = set()
    for question in questions:
        question_id = question["id"]
        baseline = results[(question_id, "baseline")]
        treatment = results[(question_id, "controller")]
        controller_actions = treatment["trace"]["controller"].get("actions", [])
        recovery_traces = treatment["trace"].get("recovery_actions", [])
        actions.extend(item["action"] for item in controller_actions)
        useful_actions += sum(any(result.get("added") for result in item["results"]) for item in recovery_traces)
        if controller_actions:
            triggered_questions.add(question_id)
        baseline_operations += int(baseline["trace"]["retrieval_operations"])
        controller_operations += int(treatment["trace"]["retrieval_operations"])

        initial = _pages(baseline["trace"]["retrieval_stages"]["retrieval"])
        post_items = treatment["trace"]["retrieval_stages"].get("post_recovery")
        post_recovery = _pages(post_items) if post_items is not None else initial
        for requirement in question["requirements"]:
            gold = _pages(requirement["gold_pages"])
            missed = gold - initial
            missed_pages += len(missed)
            recovered_pages += len(missed & post_recovery)

    scores = {
        (item["question_id"], item["variant"], item["requirement_id"]):
        item["answer_requirement_accuracy"]
        for item in summary["requirement_metrics"]
    }
    repairs = []
    regressions = []
    for question in questions:
        for requirement in question["requirements"]:
            question_id = question["id"]
            requirement_id = requirement["id"]
            before = scores[(question_id, "baseline", requirement_id)]
            after = scores[(question_id, "controller", requirement_id)]
            if before == 0 and after == 1:
                repairs.append(f"{question_id}/{requirement_id}")
            elif before == 1 and after == 0:
                regressions.append(f"{question_id}/{requirement_id}")

    count = len(questions)
    action_distribution = dict(sorted(Counter(actions).items()))
    metrics = {
        "questions": count,
        "triggered_questions": len(triggered_questions),
        "follow_up_trigger_rate": len(triggered_questions) / count,
        "actions": len(actions),
        "action_distribution": action_distribution,
        "useful_action_rate": useful_actions / len(actions) if actions else None,
        "initially_missed_gold_pages": missed_pages,
        "recovered_gold_pages": recovered_pages,
        "retrieval_recovery_rate": recovered_pages / missed_pages if missed_pages else None,
        "average_baseline_retrieval_operations": baseline_operations / count,
        "average_controller_retrieval_operations": controller_operations / count,
        "answer_requirement_repairs": repairs,
        "answer_requirement_regressions": regressions,
    }
    summary["controller_diagnostics"] = metrics
    recovery_rate = metrics["retrieval_recovery_rate"]
    useful_rate = metrics["useful_action_rate"]
    distribution = ", ".join(f"{key}: {value}" for key, value in action_distribution.items()) or "none"
    return f"""\

## Controller diagnostics

| Metric | Result |
|---|---:|
| Follow-up trigger rate | {100 * metrics['follow_up_trigger_rate']:.1f}% ({metrics['triggered_questions']}/{count}) |
| Retrieval Recovery Rate | {f'{100 * recovery_rate:.1f}%' if recovery_rate is not None else 'N/A'} ({recovered_pages}/{missed_pages} initially missed gold pages) |
| Useful action rate | {f'{100 * useful_rate:.1f}%' if useful_rate is not None else 'N/A'} |
| Average retrieval operations, baseline | {metrics['average_baseline_retrieval_operations']:.2f} |
| Average retrieval operations, Controller | {metrics['average_controller_retrieval_operations']:.2f} |
| Requirement repairs / regressions | {len(repairs)} / {len(regressions)} |

Action distribution: {distribution}.  
Repairs: {', '.join(repairs) or 'none'}.  
Regressions: {', '.join(regressions) or 'none'}.

Retrieval Recovery Rate is the fraction of initially missed gold-page occurrences reached by the
bounded follow-up. Initial retrieval and rerank recall remain frozen; Selected Evidence Recall
measures whether recovered pages reached the final Writer context.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze all initial observations; no LLM calls")
    prepare.add_argument("--source-run", default="adaptive_v2")
    prepare.add_argument("--source-variant", choices=evaluation.VARIANTS, default="adaptive")
    commands.add_parser("run", help="Generate the paired baseline and Controller answers")
    review = commands.add_parser("prepare-review", help="Create the blinded review CSV")
    review.add_argument("--force", action="store_true")
    commands.add_parser("extract-pages", help="Extract cited and gold pages for review")
    commands.add_parser("score", help="Score final quality and Controller behavior")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        directory = evaluation.evaluation_artifact_path("inputs.json", args.run_id).parent
        inputs_path = directory / "inputs.json"
        results_path = directory / "results.jsonl"
        if args.command == "prepare":
            if inputs_path.exists() or results_path.exists():
                raise evaluation.EvaluationError("Experiment already exists; choose a new --run-id")
            questions = evaluation.load_questions()
            settings = Settings.from_env()
            engine = RetrievalEngine.load(settings)
            if len(engine.chunks) != evaluation.EXPECTED_CORPUS_SIZE:
                raise evaluation.EvaluationError("Corpus size differs from the benchmark")
            evaluation.validate_gold_pages(questions, engine.chunks)
            prepare_inputs(
                questions,
                engine,
                settings,
                evaluation.evaluation_artifact_path("results.jsonl", args.source_run),
                inputs_path,
                source_variant=args.source_variant,
            )
            return 0

        inputs, input_hash = writer_experiment.load_inputs(inputs_path, PIPELINE_VERSION)
        questions = inputs["questions"]
        writer_experiment._existing_results(
            results_path,
            args.run_id,
            input_hash,
            pipeline_version=PIPELINE_VERSION,
            variants=VARIANTS,
        )
        if args.command == "run":
            settings = Settings.from_env()
            engine = RetrievalEngine.load(settings)
            if len(engine.chunks) != evaluation.EXPECTED_CORPUS_SIZE:
                raise evaluation.EvaluationError("Corpus size differs from the benchmark")
            run_experiment(
                inputs_path,
                results_path,
                engine,
                settings,
                evaluation.evaluation_llm(),
                run_id=args.run_id,
            )
        elif args.command == "prepare-review":
            evaluation.prepare_review(
                questions,
                results_path,
                directory / "review.csv",
                directory / "review_key.json",
                variants=VARIANTS,
                force=args.force,
            )
        elif args.command == "extract-pages":
            evaluation.export_review_evidence(
                questions,
                directory / "review.csv",
                Settings.from_env().data_dir / "papers",
                directory / "review_evidence.jsonl",
                variants=VARIANTS,
            )
        else:
            summary = evaluation.score_review(
                questions,
                results_path,
                directory / "review.csv",
                directory / "review_key.json",
                directory / "summary.json",
                directory / "summary.md",
                variants=VARIANTS,
            )
            note = controller_summary(summary, questions, results_path)
            (directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with (directory / "summary.md").open("a", encoding="utf-8") as handle:
                handle.write(note)
            print(evaluation._summary_markdown(summary) + note)
    except (evaluation.EvaluationError, ModelUnavailableError, OSError, ValueError, KeyError) as exc:
        print(f"Controller experiment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
