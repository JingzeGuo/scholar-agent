"""Compare the Controller-enabled Scholar-Agent pipeline with Simple RAG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path

from evals import evaluate as evaluation
from evals.evaluate_controller import _initial_operations, _recovery_operations
from scholar_agent.agents.controller import controller_node
from scholar_agent.agents.planner import planner_node
from scholar_agent.agents.recovery import recovery_node
from scholar_agent.agents.researcher import (
    MAX_RERANK_CANDIDATES,
    _build_evidence_board,
    researcher_node,
)
from scholar_agent.agents.writer import SAFE_ABSTENTION, _writer_prompt, citation_validator_node
from scholar_agent.config import Settings
from scholar_agent.indexes import ModelUnavailableError
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion
from scholar_agent.workflow import initial_state

VARIANTS = ("simple_rag", "controller")
PIPELINE_VERSION = "controller_vs_simple_rag_v1"
SIMPLE_EVIDENCE_LIMIT = 8
RerankFunction = Callable[[list[str], list[dict], str], list[dict]]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _page_refs(items: Sequence[dict]) -> list[dict]:
    return [
        {"paper": paper, "page": page}
        for paper, page in sorted({(item["paper"], item["page"]) for item in items})
    ]


def simple_rag_state(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    *,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Run single-query hybrid retrieval with a fixed eight-chunk evidence budget."""
    requirement = {
        "id": "R1",
        "description": question,
        "targets": [],
        "query": question,
        "retrieval_strategy": "hybrid",
        "top_k": SIMPLE_EVIDENCE_LIMIT,
    }
    sparse = engine.sparse_search([question], top_k=SIMPLE_EVIDENCE_LIMIT)
    dense_rankings = engine.dense_search_many([question], top_k=SIMPLE_EVIDENCE_LIMIT)
    dense = dense_rankings[0] if dense_rankings else []
    candidates = reciprocal_rank_fusion(sparse, dense)[:MAX_RERANK_CANDIDATES]
    ranked = rerank_function([question], candidates, settings.reranker_model)
    retained = [
        {
            **{key: value for key, value in item.items() if key != "_query_scores"},
            "_requirement_scores": {"R1": float(item["score"])},
        }
        for item in ranked
        if float(item["score"]) >= settings.min_rerank_score
    ][:SIMPLE_EVIDENCE_LIMIT]
    evidence, board = _build_evidence_board(
        retained,
        [requirement],
        settings.min_rerank_score,
    )
    state = initial_state(question, "fixed_hybrid", "none")
    state.update(
        plan={"requirements": [requirement]},
        evidence=evidence,
        evidence_board=board,
        retrieval_trace=[{
            "requirement_id": "R1",
            "query": question,
            "retrieval_strategy": "hybrid",
            "top_k": SIMPLE_EVIDENCE_LIMIT,
        }],
        retrieval_stages={
            "retrieval": _page_refs([*sparse, *dense]),
            "rerank": _page_refs(candidates),
        },
    )
    return state


def _controller_state(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    llm: evaluation.CountingLLM,
) -> tuple[dict, dict[str, float | int]]:
    state = initial_state(question, "adaptive", "controller")
    calls_before = llm.calls
    phase = time.perf_counter()
    state.update(planner_node(state, llm))
    planner_latency = time.perf_counter() - phase
    planner_calls = llm.calls - calls_before

    phase = time.perf_counter()
    state.update(researcher_node(state, engine, settings))
    researcher_latency = time.perf_counter() - phase

    calls_before_controller = llm.calls
    phase = time.perf_counter()
    state.update(controller_node(state, llm))
    controller_latency = time.perf_counter() - phase
    controller_calls = llm.calls - calls_before_controller

    recovery_latency = 0.0
    if state["controller_trace"]["actions"]:
        phase = time.perf_counter()
        state.update(recovery_node(state, engine, settings))
        recovery_latency = time.perf_counter() - phase

    return state, {
        "planner_latency_seconds": round(planner_latency, 4),
        "researcher_latency_seconds": round(researcher_latency, 4),
        "controller_latency_seconds": round(controller_latency, 4),
        "recovery_latency_seconds": round(recovery_latency, 4),
        "pre_writer_latency_seconds": round(
            planner_latency + researcher_latency + controller_latency + recovery_latency,
            4,
        ),
        "planner_llm_calls": planner_calls,
        "controller_llm_calls": controller_calls,
        "pre_writer_llm_calls": llm.calls - calls_before,
        "retrieval_operations": _initial_operations(state) + _recovery_operations(state),
    }


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _prepared_records(path: Path) -> dict[str, dict]:
    latest = {}
    if path.is_file():
        for item in evaluation._read_jsonl(path):
            question_id = item.get("question_id")
            if isinstance(question_id, str):
                latest[question_id] = item
    return latest


def _metadata(
    run_id: str,
    settings: Settings,
    questions_path: Path,
) -> dict:
    return {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "model": evaluation.MODEL_NAME,
        "temperature": 0,
        "variants": list(VARIANTS),
        "simple_rag": {
            "query": "original question",
            "retrieval": "BM25 + Dense -> RRF",
            "per_retriever_top_k": SIMPLE_EVIDENCE_LIMIT,
            "max_rerank_candidates": MAX_RERANK_CANDIDATES,
            "evidence_limit": SIMPLE_EVIDENCE_LIMIT,
            "writer_context": "flat",
        },
        "controller": {
            "retrieval_mode": "adaptive",
            "recovery_mode": "controller",
            "writer_context": "requirement_evidence_blackboard",
        },
        "reranker_model": settings.reranker_model,
        "min_rerank_score": settings.min_rerank_score,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
    }


def prepare_inputs(
    questions: Sequence[dict],
    engine: RetrievalEngine,
    settings: Settings,
    llm: evaluation.CountingLLM,
    prepared_path: Path,
    metadata_path: Path,
    *,
    run_id: str,
) -> None:
    """Prepare both pre-Writer states from scratch, with resumable per-question records."""
    metadata = _metadata(run_id, settings, evaluation.QUESTIONS_PATH)
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "input_set_sha256"}
        if comparable != metadata:
            raise evaluation.EvaluationError("Experiment metadata changed; use a new --run-id")
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    latest = _prepared_records(prepared_path)
    for index, question in enumerate(questions):
        question_id = question["id"]
        saved = latest.get(question_id)
        if saved is not None and saved.get("error") is None:
            continue
        started = time.perf_counter()
        simple_started = time.perf_counter()
        try:
            if index % 2 == 0:
                simple = simple_rag_state(question["question"], engine, settings)
                simple_latency = time.perf_counter() - simple_started
                controller, controller_metrics = _controller_state(
                    question["question"], engine, settings, llm,
                )
                preparation_order = ["simple_rag", "controller"]
            else:
                controller, controller_metrics = _controller_state(
                    question["question"], engine, settings, llm,
                )
                simple_started = time.perf_counter()
                simple = simple_rag_state(question["question"], engine, settings)
                simple_latency = time.perf_counter() - simple_started
                preparation_order = ["controller", "simple_rag"]

            prompts = {
                "simple_rag": _writer_prompt(simple, use_evidence_board=False),
                "controller": _writer_prompt(controller),
            }
            payload = {
                "question_id": question_id,
                "question": question["question"],
                "states": {"simple_rag": simple, "controller": controller},
                "prompts": prompts,
                "prompt_sha256": {
                    variant: hashlib.sha256(prompt.encode()).hexdigest()
                    for variant, prompt in prompts.items()
                },
                "state_sha256": {
                    variant: _canonical_hash(state)
                    for variant, state in (("simple_rag", simple), ("controller", controller))
                },
                "preparation_order": preparation_order,
                "metrics": {
                    "simple_rag": {
                        "planner_latency_seconds": 0.0,
                        "researcher_latency_seconds": round(simple_latency, 4),
                        "controller_latency_seconds": 0.0,
                        "recovery_latency_seconds": 0.0,
                        "pre_writer_latency_seconds": round(simple_latency, 4),
                        "planner_llm_calls": 0,
                        "controller_llm_calls": 0,
                        "pre_writer_llm_calls": 0,
                        "retrieval_operations": 2,
                    },
                    "controller": controller_metrics,
                },
            }
            record = {
                "run_id": run_id,
                "pipeline_version": PIPELINE_VERSION,
                **payload,
                "input_sha256": _canonical_hash(payload),
                "preparation_latency_seconds": round(time.perf_counter() - started, 4),
                "error": None,
            }
        except Exception as exc:
            record = {
                "run_id": run_id,
                "pipeline_version": PIPELINE_VERSION,
                "question_id": question_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _append_jsonl(prepared_path, record)
            raise evaluation.EvaluationError(
                f"Input preparation failed for {question_id}: {exc}",
            ) from exc
        _append_jsonl(prepared_path, record)
        latest[question_id] = record
        print(f"prepared {question_id}", flush=True)

    complete = _load_prepared(questions, prepared_path, metadata_path)
    metadata["input_set_sha256"] = _canonical_hash(
        [complete[question["id"]]["input_sha256"] for question in questions],
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_prepared(
    questions: Sequence[dict],
    prepared_path: Path,
    metadata_path: Path,
) -> dict[str, dict]:
    if not metadata_path.is_file():
        raise evaluation.EvaluationError("Experiment metadata does not exist; run prepare")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("pipeline_version") != PIPELINE_VERSION:
        raise evaluation.EvaluationError("Prepared inputs use a different pipeline version")
    latest = _prepared_records(prepared_path)
    expected_ids = {item["id"] for item in questions}
    if set(latest) != expected_ids:
        raise evaluation.EvaluationError("Prepared input set is incomplete")
    for question in questions:
        record = latest[question["id"]]
        if record.get("error") is not None:
            raise evaluation.EvaluationError(f"Prepared input failed for {question['id']}")
        if record["question"] != question["question"]:
            raise evaluation.EvaluationError(f"Question changed for {question['id']}")
        payload = {
            key: record[key]
            for key in (
                "question_id",
                "question",
                "states",
                "prompts",
                "prompt_sha256",
                "state_sha256",
                "preparation_order",
                "metrics",
            )
        }
        if record["input_sha256"] != _canonical_hash(payload):
            raise evaluation.EvaluationError(f"Prepared input hash changed for {question['id']}")
    input_set_hash = _canonical_hash(
        [latest[question["id"]]["input_sha256"] for question in questions],
    )
    if metadata.get("input_set_sha256") not in {None, input_set_hash}:
        raise evaluation.EvaluationError("Prepared input set hash changed")
    return latest


def run_experiment(
    questions: Sequence[dict],
    prepared_path: Path,
    metadata_path: Path,
    results_path: Path,
    llm: evaluation.CountingLLM,
    *,
    run_id: str,
) -> None:
    """Generate paired answers from frozen states, alternating Writer order."""
    prepared = _load_prepared(questions, prepared_path, metadata_path)
    latest = evaluation._resumable_results(results_path, run_id, PIPELINE_VERSION)
    for result in latest.values():
        expected_hash = prepared[result["question_id"]]["input_sha256"]
        if result.get("input_sha256") != expected_hash:
            raise evaluation.EvaluationError("Result input hash differs from frozen input")
        if result["variant"] not in VARIANTS:
            raise evaluation.EvaluationError("Unexpected comparison variant")

    for index, question in enumerate(questions):
        question_id = question["id"]
        sample = prepared[question_id]
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            if evaluation._successful_result(latest, question_id, variant) is not None:
                continue
            state = deepcopy(sample["states"][variant])
            prompt = _writer_prompt(state, use_evidence_board=variant == "controller")
            if hashlib.sha256(prompt.encode()).hexdigest() != sample["prompt_sha256"][variant]:
                raise evaluation.EvaluationError(f"Writer prompt changed for {question_id}/{variant}")
            calls_before = llm.calls
            started = time.perf_counter()
            try:
                state["answer"] = (
                    llm.complete(prompt).strip()
                    if state["evidence"]
                    else SAFE_ABSTENTION
                )
                state.update(citation_validator_node(state))
                writer_latency = time.perf_counter() - started
                prep_metrics = sample["metrics"][variant]
                trace = evaluation._trace(
                    state["answer"],
                    state,
                    float(prep_metrics["planner_latency_seconds"]),
                    int(prep_metrics["planner_llm_calls"]),
                )
                trace.update(
                    experiment_variant=variant,
                    scheduled_writer_order=list(order),
                    input_sha256=sample["input_sha256"],
                    pre_writer_state_sha256=sample["state_sha256"][variant],
                    writer_prompt_sha256=sample["prompt_sha256"][variant],
                    researcher_latency_seconds=prep_metrics["researcher_latency_seconds"],
                    controller_latency_seconds=prep_metrics["controller_latency_seconds"],
                    recovery_latency_seconds=prep_metrics["recovery_latency_seconds"],
                    writer_latency_seconds=round(writer_latency, 4),
                    retrieval_operations=prep_metrics["retrieval_operations"],
                    latency_scope="sum_of_full_variant_pipeline_stages",
                )
                result = evaluation._result_record(
                    run_id,
                    PIPELINE_VERSION,
                    question_id,
                    variant,
                    state["answer"],
                    state["evidence"],
                    float(prep_metrics["pre_writer_latency_seconds"]) + writer_latency,
                    int(prep_metrics["pre_writer_llm_calls"]) + llm.calls - calls_before,
                    trace,
                )
                result["input_sha256"] = sample["input_sha256"]
                result["requirement_metrics"] = evaluation.requirement_stage_metrics(
                    question,
                    result,
                )
            except Exception as exc:
                failed = evaluation._error_record(
                    run_id,
                    PIPELINE_VERSION,
                    question_id,
                    variant,
                    exc,
                    float(sample["metrics"][variant]["pre_writer_latency_seconds"])
                    + time.perf_counter()
                    - started,
                    int(sample["metrics"][variant]["pre_writer_llm_calls"])
                    + llm.calls
                    - calls_before,
                )
                failed["input_sha256"] = sample["input_sha256"]
                evaluation._append_result(results_path, failed)
                raise evaluation.EvaluationError(
                    f"Answer generation failed for {question_id}/{variant}: {exc}",
                ) from exc
            evaluation._append_result(results_path, result)
            latest[(question_id, variant)] = result
            print(f"completed {question_id} {variant}", flush=True)


def _strict_outcomes(
    questions: Sequence[dict],
    review_path: Path,
    review_key_path: Path,
) -> tuple[dict[tuple[str, str], bool], dict[tuple[str, str, str], int]]:
    questions_by_id = {item["id"]: item for item in questions}
    keys = {
        item["review_id"]: item
        for item in json.loads(review_key_path.read_text(encoding="utf-8"))
    }
    strict = {}
    requirements = {}
    with review_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = keys[row["review_id"]]
            question_id, variant = key["question_id"], key["variant"]
            scores = json.loads(row["requirement_scores"])
            citations = json.loads(row["citation_scores"])
            needs_citations = any(
                item["answerable"] for item in questions_by_id[question_id]["requirements"]
            )
            strict[(question_id, variant)] = (
                all(score == 1 for score in scores.values())
                and all(score == 1 for score in citations)
                and int(row["unsupported_claims"]) == 0
                and int(row["uncited_claims"]) == 0
                and (bool(citations) if needs_citations else not citations)
            )
            for requirement_id, score in scores.items():
                requirements[(question_id, variant, requirement_id)] = int(score)
    return strict, requirements


def _mcnemar(repairs: int, regressions: int) -> float | None:
    discordant = repairs + regressions
    if not discordant:
        return None
    tail = min(repairs, regressions)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / 2**discordant
    return min(1.0, 2 * probability)


def comparison_summary(
    summary: dict,
    questions: Sequence[dict],
    results_path: Path,
    review_path: Path,
    review_key_path: Path,
) -> str:
    """Add paired, latency, cost, and Controller-action diagnostics."""
    results = evaluation._complete_result_set(questions, results_path, VARIANTS)
    strict, requirement_scores = _strict_outcomes(questions, review_path, review_key_path)
    strict_repairs = []
    strict_regressions = []
    requirement_repairs = []
    requirement_regressions = []
    for question in questions:
        question_id = question["id"]
        if not strict[(question_id, "simple_rag")] and strict[(question_id, "controller")]:
            strict_repairs.append(question_id)
        elif strict[(question_id, "simple_rag")] and not strict[(question_id, "controller")]:
            strict_regressions.append(question_id)
        for requirement in question["requirements"]:
            requirement_id = requirement["id"]
            before = requirement_scores[(question_id, "simple_rag", requirement_id)]
            after = requirement_scores[(question_id, "controller", requirement_id)]
            key = f"{question_id}/{requirement_id}"
            if before == 0 and after == 1:
                requirement_repairs.append(key)
            elif before == 1 and after == 0:
                requirement_regressions.append(key)

    stages = {}
    for variant in VARIANTS:
        records = [results[(question["id"], variant)] for question in questions]
        stages[variant] = {
            "average_planner_latency_seconds": sum(
                item["trace"]["shared_planner_latency_seconds"] for item in records
            ) / len(records),
            "average_researcher_latency_seconds": sum(
                item["trace"]["researcher_latency_seconds"] for item in records
            ) / len(records),
            "average_controller_latency_seconds": sum(
                item["trace"]["controller_latency_seconds"] for item in records
            ) / len(records),
            "average_recovery_latency_seconds": sum(
                item["trace"]["recovery_latency_seconds"] for item in records
            ) / len(records),
            "average_writer_latency_seconds": sum(
                item["trace"]["writer_latency_seconds"] for item in records
            ) / len(records),
            "average_retrieval_operations": sum(
                item["trace"]["retrieval_operations"] for item in records
            ) / len(records),
            "average_selected_evidence": sum(len(item["evidence"]) for item in records)
            / len(records),
        }

    controller_records = [
        results[(question["id"], "controller")] for question in questions
    ]
    actions = [
        action
        for item in controller_records
        for action in item["trace"]["controller"].get("actions", [])
    ]
    recovery = [
        action
        for item in controller_records
        for action in item["trace"].get("recovery_actions", [])
    ]
    diagnostics = {
        "strict_repairs": strict_repairs,
        "strict_regressions": strict_regressions,
        "strict_exact_mcnemar_p_value": _mcnemar(
            len(strict_repairs), len(strict_regressions),
        ),
        "requirement_repairs": requirement_repairs,
        "requirement_regressions": requirement_regressions,
        "requirement_exact_mcnemar_p_value": _mcnemar(
            len(requirement_repairs), len(requirement_regressions),
        ),
        "stage_metrics": stages,
        "controller_triggered_questions": sum(
            bool(item["trace"]["controller"].get("actions"))
            for item in controller_records
        ),
        "controller_actions": len(actions),
        "controller_action_distribution": dict(
            sorted(Counter(item["action"] for item in actions).items()),
        ),
        "useful_controller_actions": sum(
            any(result.get("added") for result in item["results"])
            for item in recovery
        ),
    }
    summary["comparison_diagnostics"] = diagnostics
    strict_p = diagnostics["strict_exact_mcnemar_p_value"]
    requirement_p = diagnostics["requirement_exact_mcnemar_p_value"]
    return f"""\

## Paired and pipeline diagnostics

| Metric | Result |
|---|---:|
| Strict repairs / regressions | {len(strict_repairs)} / {len(strict_regressions)} |
| Strict exact McNemar p-value | {strict_p if strict_p is not None else 'N/A'} |
| Requirement repairs / regressions | {len(requirement_repairs)} / {len(requirement_regressions)} |
| Requirement exact McNemar p-value | {requirement_p if requirement_p is not None else 'N/A'} |
| Controller-triggered questions | {diagnostics['controller_triggered_questions']}/{len(questions)} |
| Controller actions | {len(actions)} |
| Useful Controller actions | {diagnostics['useful_controller_actions']}/{len(actions) if actions else 0} |

Strict repairs: {', '.join(strict_repairs) or 'none'}.  
Strict regressions: {', '.join(strict_regressions) or 'none'}.  
Requirement repairs: {', '.join(requirement_repairs) or 'none'}.  
Requirement regressions: {', '.join(requirement_regressions) or 'none'}.

Simple RAG uses the original question for BM25 and dense retrieval, RRF, cross-encoder reranking,
a fixed maximum of {SIMPLE_EVIDENCE_LIMIT} selected chunks, the current flat Writer policy, and the
same deterministic citation validator. The Controller variant is the complete current adaptive,
Blackboard, assessment, bounded-recovery, Writer, and citation-validation pipeline. Latency sums
every recorded stage in each variant's full pipeline.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Generate fresh frozen pre-Writer states")
    commands.add_parser("run", help="Generate paired answers with alternating Writer order")
    review = commands.add_parser("prepare-review", help="Create blinded review materials")
    review.add_argument("--force", action="store_true")
    commands.add_parser("extract-pages", help="Extract cited and gold physical pages")
    commands.add_parser("score", help="Score review and add paired diagnostics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    directory = evaluation.evaluation_artifact_path("results.jsonl", args.run_id).parent
    prepared_path = directory / "prepared_inputs.jsonl"
    metadata_path = directory / "metadata.json"
    results_path = directory / "results.jsonl"
    try:
        questions = evaluation.load_questions()
        if args.command == "prepare":
            _, engine, settings, llm = evaluation._runtime()
            evaluation._warm_up(questions[0]["question"], engine, settings)
            prepare_inputs(
                questions,
                engine,
                settings,
                llm,
                prepared_path,
                metadata_path,
                run_id=args.run_id,
            )
        elif args.command == "run":
            run_experiment(
                questions,
                prepared_path,
                metadata_path,
                results_path,
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
            stats = evaluation.export_review_evidence(
                questions,
                directory / "review.csv",
                Settings.from_env().data_dir / "papers",
                directory / "review_evidence.jsonl",
                variants=VARIANTS,
            )
            print(
                f"exported {stats['samples']} blind samples with "
                f"{stats['unique_pages']} unique physical pages",
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
            note = comparison_summary(
                summary,
                questions,
                results_path,
                directory / "review.csv",
                directory / "review_key.json",
            )
            (directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with (directory / "summary.md").open("a", encoding="utf-8") as handle:
                handle.write(note)
            print(evaluation._summary_markdown(summary) + note)
    except (evaluation.EvaluationError, ModelUnavailableError, OSError, ValueError, KeyError) as exc:
        print(f"Simple RAG comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
