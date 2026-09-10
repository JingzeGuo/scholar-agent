"""Isolated Writer and Controller prompt-principle ablations over E3 v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from evals import evaluate as evaluation
from evals import evaluate_controller as controller_experiment
from evals import evaluate_writer as writer_experiment
from scholar_agent.agents.controller import _controller_prompt, controller_node
from scholar_agent.agents.recovery import recovery_node
from scholar_agent.agents.writer import SAFE_ABSTENTION, _writer_prompt, citation_validator_node
from scholar_agent.config import Settings
from scholar_agent.indexes import ModelUnavailableError
from scholar_agent.retrieval import RetrievalEngine

VARIANTS = ("baseline", "principle")
PIPELINES = {
    "writer": "writer_coverage_principle_e3_v4a",
    "controller": "controller_semantic_principle_e3_v4b",
}


def _refs(items: Sequence[dict]) -> list[tuple[str, str, int]]:
    return [(item["chunk_id"], item["paper"], item["page"]) for item in items]


def prepare_inputs(
    experiment: str,
    source_directory: Path,
    inputs_path: Path,
    engine: RetrievalEngine,
    settings: Settings,
) -> dict:
    """Freeze either v3's recovered evidence or its initial Controller observation."""
    if inputs_path.exists():
        raise evaluation.EvaluationError("Frozen inputs already exist; use a new --run-id")
    source_inputs_path = source_directory / "inputs.json"
    source_results_path = source_directory / "results.jsonl"
    source_inputs, _ = writer_experiment.load_inputs(
        source_inputs_path,
        controller_experiment.PIPELINE_VERSION,
    )
    source_results = evaluation._complete_result_set(
        source_inputs["questions"],
        source_results_path,
        controller_experiment.VARIANTS,
    )
    samples = []
    for source_sample in source_inputs["samples"]:
        question_id = source_sample["question_id"]
        state = deepcopy(source_sample["state"])
        if _writer_prompt(state) != source_sample["baseline_prompt"]:
            raise evaluation.EvaluationError("Writer policy differs from the E3 v3 input")
        if _controller_prompt(state) != source_sample["controller_prompt"]:
            raise evaluation.EvaluationError("Controller policy differs from the E3 v3 input")

        if experiment == "writer":
            saved = source_results[(question_id, "controller")]
            state["recovery_mode"] = "controller"
            state["controller_trace"] = deepcopy(saved["trace"]["controller"])
            if state["controller_trace"]["actions"]:
                state.update(recovery_node(state, engine, settings))
            if _refs(state["evidence"]) != _refs(saved["evidence"]):
                raise evaluation.EvaluationError(
                    f"Recovered evidence differs from E3 v3 for {question_id}",
                )
            baseline_prompt = _writer_prompt(state)
            if hashlib.sha256(baseline_prompt.encode()).hexdigest() != saved["trace"][
                "writer_prompt_sha256"
            ]:
                raise evaluation.EvaluationError(
                    f"Writer prompt differs from E3 v3 for {question_id}",
                )
            prompts = {
                "baseline": baseline_prompt,
                "principle": _writer_prompt(state, coverage_principle=True),
            }
        else:
            prompts = {
                "baseline": _controller_prompt(state),
                "principle": _controller_prompt(state, preserve_semantics=True),
            }
        samples.append({"question_id": question_id, "state": state, "prompts": prompts})
        print(f"prepared {question_id}", flush=True)

    inputs = {
        "pipeline_version": PIPELINES[experiment],
        "experiment": experiment,
        "model": evaluation.MODEL_NAME,
        "source_run": source_directory.name,
        "source_inputs_sha256": hashlib.sha256(source_inputs_path.read_bytes()).hexdigest(),
        "source_results_sha256": hashlib.sha256(source_results_path.read_bytes()).hexdigest(),
        "reranker_model": settings.reranker_model,
        "min_rerank_score": settings.min_rerank_score,
        "empty_evidence_answer": SAFE_ABSTENTION,
        "questions": source_inputs["questions"],
        "samples": samples,
    }
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
    return inputs


def run_controller_experiment(
    inputs_path: Path,
    results_path: Path,
    engine: RetrievalEngine,
    settings: Settings,
    llm: evaluation.CountingLLM,
    *,
    run_id: str,
) -> None:
    """Run both Controller prompts and the unchanged v3 Writer from frozen observations."""
    pipeline = PIPELINES["controller"]
    inputs, input_hash = writer_experiment.load_inputs(inputs_path, pipeline)
    latest = writer_experiment._existing_results(
        results_path,
        run_id,
        input_hash,
        pipeline_version=pipeline,
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
            guided = variant == "principle"
            if _controller_prompt(state, preserve_semantics=guided) != sample["prompts"][variant]:
                raise evaluation.EvaluationError("Controller prompt differs from frozen input")
            state["recovery_mode"] = f"controller_{variant}"
            calls_before = llm.calls
            started = time.perf_counter()
            controller_latency = recovery_latency = 0.0
            try:
                phase = time.perf_counter()
                state.update(controller_node(state, llm, preserve_semantics=guided))
                controller_latency = time.perf_counter() - phase
                if state["controller_trace"]["actions"]:
                    phase = time.perf_counter()
                    state.update(recovery_node(state, engine, settings))
                    recovery_latency = time.perf_counter() - phase

                writer_prompt = _writer_prompt(state)
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
                    controller_prompt_sha256=hashlib.sha256(
                        sample["prompts"][variant].encode(),
                    ).hexdigest(),
                    writer_prompt_sha256=hashlib.sha256(writer_prompt.encode()).hexdigest(),
                    controller_latency_seconds=round(controller_latency, 4),
                    recovery_latency_seconds=round(recovery_latency, 4),
                    writer_latency_seconds=round(writer_latency, 4),
                    retrieval_operations=(
                        controller_experiment._initial_operations(state)
                        + controller_experiment._recovery_operations(state)
                    ),
                    latency_scope="controller_recovery_writer_after_frozen_initial_research",
                )
                result = evaluation._result_record(
                    run_id,
                    pipeline,
                    question_id,
                    variant,
                    state["answer"],
                    state["evidence"],
                    time.perf_counter() - started,
                    llm.calls - calls_before,
                    trace,
                )
                result["requirement_metrics"] = evaluation.requirement_stage_metrics(
                    questions[question_id],
                    result,
                )
            except Exception as exc:
                failed = evaluation._error_record(
                    run_id,
                    pipeline,
                    question_id,
                    variant,
                    exc,
                    time.perf_counter() - started,
                    llm.calls - calls_before,
                )
                failed["input_sha256"] = input_hash
                evaluation._append_result(results_path, failed)
                raise evaluation.EvaluationError(
                    f"Controller prompt experiment failed for {question_id}/{variant}: {exc}",
                ) from exc
            result["input_sha256"] = input_hash
            evaluation._append_result(results_path, result)
            latest[(question_id, variant)] = result
            print(f"completed {question_id} {variant}", flush=True)


def controller_diagnostics(questions: Sequence[dict], results_path: Path) -> tuple[dict, str]:
    """Compare the two Controller policies without mixing in answer judgments."""
    results = evaluation._complete_result_set(questions, results_path, VARIANTS)
    diagnostics = {}
    for variant in VARIANTS:
        action_types: Counter = Counter()
        triggered = rejected = useful = operations = 0
        for question in questions:
            result = results[(question["id"], variant)]
            controller = result["trace"]["controller"]
            actions = controller.get("actions", [])
            recovery = result["trace"].get("recovery_actions", [])
            triggered += bool(actions)
            rejected += int(controller.get("rejected_actions", 0))
            action_types.update(item["action"] for item in actions)
            useful += sum(any(item.get("added") for item in trace["results"]) for trace in recovery)
            operations += int(result["trace"]["retrieval_operations"])
        action_count = sum(action_types.values())
        diagnostics[variant] = {
            "triggered_questions": triggered,
            "actions": action_count,
            "rejected_actions": rejected,
            "useful_actions": useful,
            "useful_action_rate": useful / action_count if action_count else None,
            "average_retrieval_operations": operations / len(questions),
            "action_distribution": dict(sorted(action_types.items())),
        }

    def percent(value: float | None) -> str:
        return f"{100 * value:.1f}%" if value is not None else "N/A"

    before, after = (diagnostics[variant] for variant in VARIANTS)
    note = f"""

## Controller prompt diagnostics

| Metric | Current prompt | Semantic-preservation principle |
|---|---:|---:|
| Follow-up triggers | {before['triggered_questions']}/50 | {after['triggered_questions']}/50 |
| Accepted actions | {before['actions']} | {after['actions']} |
| Rejected actions | {before['rejected_actions']} | {after['rejected_actions']} |
| Useful action rate | {percent(before['useful_action_rate'])} | {percent(after['useful_action_rate'])} |
| Average retrieval operations | {before['average_retrieval_operations']:.2f} | {after['average_retrieval_operations']:.2f} |

Current action distribution: {before['action_distribution']}.  
Principle action distribution: {after['action_distribution']}.
"""
    return diagnostics, note


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", choices=PIPELINES, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-run", default="controller_e3_v3")
    commands.add_parser("run")
    review = commands.add_parser("prepare-review")
    review.add_argument("--force", action="store_true")
    commands.add_parser("extract-pages")
    commands.add_parser("score")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pipeline = PIPELINES[args.experiment]
    try:
        directory = evaluation.evaluation_artifact_path("inputs.json", args.run_id).parent
        inputs_path = directory / "inputs.json"
        results_path = directory / "results.jsonl"
        if args.command == "prepare":
            settings = Settings.from_env()
            engine = RetrievalEngine.load(settings)
            if len(engine.chunks) != evaluation.EXPECTED_CORPUS_SIZE:
                raise evaluation.EvaluationError("Corpus size differs from the benchmark")
            source = evaluation.evaluation_artifact_path("inputs.json", args.source_run).parent
            prepare_inputs(args.experiment, source, inputs_path, engine, settings)
            return 0

        inputs, input_hash = writer_experiment.load_inputs(inputs_path, pipeline)
        questions = inputs["questions"]
        writer_experiment._existing_results(
            results_path,
            args.run_id,
            input_hash,
            pipeline_version=pipeline,
            variants=VARIANTS,
        )
        if args.command == "run":
            llm = evaluation.evaluation_llm()
            if args.experiment == "writer":
                writer_experiment.run_experiment(
                    inputs_path,
                    results_path,
                    llm,
                    run_id=args.run_id,
                    pipeline_version=pipeline,
                    variants=VARIANTS,
                )
            else:
                settings = Settings.from_env()
                engine = RetrievalEngine.load(settings)
                run_controller_experiment(
                    inputs_path,
                    results_path,
                    engine,
                    settings,
                    llm,
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
            if args.experiment == "controller":
                diagnostics, note = controller_diagnostics(questions, results_path)
                summary["controller_prompt_diagnostics"] = diagnostics
            else:
                note = "\nThis is a Writer-only experiment over frozen E3 v3 evidence.\n"
            (directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            )
            with (directory / "summary.md").open("a") as handle:
                handle.write(note)
            print(evaluation._summary_markdown(summary) + note)
    except (evaluation.EvaluationError, ModelUnavailableError, OSError, ValueError, KeyError) as exc:
        print(f"Prompt-principle experiment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
