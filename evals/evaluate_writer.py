"""Paired flat-vs-blackboard Writer ablation using frozen retrieval inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path

from evals import evaluate as evaluation
from scholar_agent.agents.researcher import researcher_node
from scholar_agent.agents.writer import SAFE_ABSTENTION, _writer_prompt, citation_validator_node
from scholar_agent.config import Settings
from scholar_agent.indexes import ModelUnavailableError
from scholar_agent.retrieval import RetrievalEngine
from scholar_agent.workflow import initial_state

VARIANTS = ("flat", "blackboard")
PIPELINE_VERSION = "writer_board_ablation_v1"


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
    """Replay saved plans once and freeze both prompts without generating answers."""
    if inputs_path.exists():
        raise evaluation.EvaluationError("Frozen inputs already exist; use a new --run-id")
    source = evaluation._complete_result_set(questions, source_results_path, (source_variant,))
    samples = []
    for question in questions:
        saved = source[(question["id"], source_variant)]
        state = initial_state(question["question"], source_variant)
        state["plan"] = deepcopy(saved["trace"]["plan"])
        state.update(researcher_runner(state, engine, settings))
        current_refs = [(item["chunk_id"], item["paper"], item["page"]) for item in state["evidence"]]
        saved_refs = [(item["chunk_id"], item["paper"], item["page"]) for item in saved["evidence"]]
        if current_refs != saved_refs:
            raise evaluation.EvaluationError(
                f"Selected evidence differs from the source run for {question['id']}",
            )
        samples.append(
            {
                "question_id": question["id"],
                "state": state,
                "prompts": {
                    variant: _writer_prompt(state, use_evidence_board=variant == "blackboard")
                    for variant in VARIANTS
                },
            },
        )
        print(f"prepared {question['id']}", flush=True)
    inputs = {
        "pipeline_version": PIPELINE_VERSION,
        "model": evaluation.MODEL_NAME,
        "source_results": str(source_results_path.resolve()),
        "source_sha256": hashlib.sha256(source_results_path.read_bytes()).hexdigest(),
        "retrieval_mode": source_variant,
        "min_rerank_score": settings.min_rerank_score,
        "empty_evidence_answer": SAFE_ABSTENTION,
        "questions": list(questions),
        "samples": samples,
    }
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inputs


def load_inputs(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    inputs = json.loads(payload)
    if inputs["pipeline_version"] != PIPELINE_VERSION or inputs["model"] != evaluation.MODEL_NAME:
        raise evaluation.EvaluationError("Frozen inputs use a different experiment version or model")
    return inputs, hashlib.sha256(payload).hexdigest()


def _existing_results(results_path: Path, run_id: str, input_hash: str) -> dict:
    latest = evaluation._resumable_results(results_path, run_id, PIPELINE_VERSION)
    if any(result.get("input_sha256") != input_hash for result in latest.values()):
        raise evaluation.EvaluationError("Frozen inputs changed after this run started; use a new --run-id")
    if any(variant not in VARIANTS for _, variant in latest):
        raise evaluation.EvaluationError("Results contain an unexpected Writer variant")
    return latest


def run_experiment(
    inputs_path: Path,
    results_path: Path,
    llm: evaluation.CountingLLM,
    *,
    run_id: str,
) -> None:
    """Alternate Writer order and resume only against the exact same frozen input file."""
    inputs, input_hash = load_inputs(inputs_path)
    latest = _existing_results(results_path, run_id, input_hash)
    questions = {question["id"]: question for question in inputs["questions"]}
    for index, sample in enumerate(inputs["samples"]):
        question_id = sample["question_id"]
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            if evaluation._successful_result(latest, question_id, variant) is not None:
                continue
            state = deepcopy(sample["state"])
            prompt = sample["prompts"][variant]
            calls_before = llm.calls
            started = time.perf_counter()
            try:
                state["answer"] = (
                    llm.complete(prompt).strip() if state["evidence"] else inputs["empty_evidence_answer"]
                )
                state.update(citation_validator_node(state))
                trace = evaluation._trace(state["answer"], state, 0.0, 0)
                trace.update(
                    writer_mode=variant,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    latency_scope="writer_and_citation_validation",
                )
                result = evaluation._result_record(
                    run_id, PIPELINE_VERSION, question_id, variant, state["answer"],
                    state["evidence"], time.perf_counter() - started, llm.calls - calls_before, trace,
                )
                result["requirement_metrics"] = evaluation.requirement_stage_metrics(
                    questions[question_id], result,
                )
            except Exception as exc:
                failed = evaluation._error_record(
                    run_id, PIPELINE_VERSION, question_id, variant, exc,
                    time.perf_counter() - started, llm.calls - calls_before,
                )
                failed["input_sha256"] = input_hash
                evaluation._append_result(results_path, failed)
                raise evaluation.EvaluationError(f"Writer failed for {question_id}/{variant}: {exc}") from exc
            result["input_sha256"] = input_hash
            evaluation._append_result(results_path, result)
            latest[(question_id, variant)] = result
            print(f"completed {question_id} {variant}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New experiment directory under evals/runs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Freeze plans, selected chunks and both prompts; no LLM calls")
    prepare.add_argument("--source-run", default="adaptive_v2")
    prepare.add_argument("--source-variant", choices=evaluation.VARIANTS, default="adaptive")
    subparsers.add_parser("run", help="Generate paired answers from the frozen prompts")
    review = subparsers.add_parser("prepare-review", help="Create the blinded review CSV")
    review.add_argument("--force", action="store_true")
    subparsers.add_parser("extract-pages", help="Extract cited and gold pages for review")
    subparsers.add_parser("score", help="Score both variants with the existing metrics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        directory = evaluation.evaluation_artifact_path("inputs.json", args.run_id).parent
        inputs_path = directory / "inputs.json"
        results_path = directory / "results.jsonl"
        review_path = directory / "review.csv"
        key_path = directory / "review_key.json"
        if args.command == "prepare":
            if inputs_path.exists() or results_path.exists():
                raise evaluation.EvaluationError("Experiment already exists; use run to resume or choose a new --run-id")
            questions = evaluation.load_questions()
            settings = Settings.from_env()
            engine = RetrievalEngine.load(settings)
            if len(engine.chunks) != evaluation.EXPECTED_CORPUS_SIZE:
                raise evaluation.EvaluationError("Corpus size differs from the benchmark")
            evaluation.validate_gold_pages(questions, engine.chunks)
            prepare_inputs(
                questions, engine, settings,
                evaluation.evaluation_artifact_path("results.jsonl", args.source_run),
                inputs_path, source_variant=args.source_variant,
            )
            return 0

        inputs, input_hash = load_inputs(inputs_path)
        questions = inputs["questions"]
        _existing_results(results_path, args.run_id, input_hash)
        if args.command == "run":
            # Only generation needs a provider; frozen inputs require no indexes or PDFs.
            llm = evaluation.evaluation_llm()
            run_experiment(inputs_path, results_path, llm, run_id=args.run_id)
        elif args.command == "prepare-review":
            evaluation.prepare_review(
                questions, results_path, review_path, key_path,
                variants=VARIANTS, force=args.force,
            )
        elif args.command == "extract-pages":
            evaluation.export_review_evidence(
                questions, review_path, Settings.from_env().data_dir / "papers",
                directory / "review_evidence.jsonl", variants=VARIANTS,
            )
        else:
            summary = evaluation.score_review(
                questions, results_path, review_path, key_path,
                directory / "summary.json", directory / "summary.md", variants=VARIANTS,
            )
            note = "\nThis is a paired Writer-only experiment. Latency and LLM calls exclude frozen Planner/retrieval preparation.\n"
            with (directory / "summary.md").open("a", encoding="utf-8") as handle:
                handle.write(note)
            print(evaluation._summary_markdown(summary) + note)
    except (evaluation.EvaluationError, ModelUnavailableError, OSError, ValueError, KeyError) as exc:
        print(f"Writer experiment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
