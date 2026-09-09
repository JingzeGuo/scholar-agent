"""Paired E2a validation of targeted retrieval recovery on three known failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path

from evals import evaluate as evaluation
from evals import evaluate_writer as writer_experiment
from scholar_agent.agents.planner import evidence_matches_target
from scholar_agent.agents.researcher import (
    _attach_requirement_scores,
    _build_evidence_board,
    _execute_retrieval,
    _retrieval_requests,
    _select_candidates_for_reranking,
    recovery_trace_entry,
    researcher_node,
)
from scholar_agent.agents.writer import SAFE_ABSTENTION, _writer_prompt
from scholar_agent.config import Settings
from scholar_agent.indexes import ModelUnavailableError
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine
from scholar_agent.workflow import initial_state

VARIANTS = ("baseline", "recovery")
PIPELINE_VERSION = "targeted_retrieval_recovery_e2a_v1"
CASE_SPECS = {
    "Q014": {
        "requirement_id": "R1",
        "gold_requirement_id": "G1",
        "critical_chunk_id": "c1fe0c3e050b938f",
        "action": "paper_navigation",
        "paper": "2401.15884.pdf",
        "query": "large scale web searches extension decompose then recompose algorithm",
    },
    "Q018": {
        "requirement_id": "R2",
        "gold_requirement_id": "G2",
        "critical_chunk_id": "c1fe0c3e050b938f",
        "action": "paper_navigation",
        "paper": "2401.15884.pdf",
        "query": "large scale web searches extension decompose then recompose algorithm",
    },
    "Q025": {
        "requirement_id": "R1",
        "gold_requirement_id": "G1",
        "critical_chunk_id": "7cbf8d2a830bdcf4",
        "action": "increase_depth",
        "top_k": 12,
    },
}
RECOVERY_EVIDENCE_LIMIT = 2


def _unique(items: Sequence[dict]) -> list[dict]:
    return list({item["chunk_id"]: item for item in items}.values())


def _raw_evidence(item: dict) -> dict:
    return {
        **{
            key: value
            for key, value in item.items()
            if key not in {"id", "paper_id", "supports", "requirement_scores"}
        },
        "_requirement_scores": dict(item["requirement_scores"]),
    }


def _page_refs(items: Sequence[dict]) -> list[dict]:
    return [
        {"paper": paper, "page": page}
        for paper, page in sorted({(item["paper"], item["page"]) for item in items})
    ]


def _recovery_candidates(
    engine: RetrievalEngine,
    requirement: dict,
    spec: dict,
) -> tuple[list[dict], str, list[dict]]:
    if spec["action"] == "paper_navigation":
        seeds = engine.search_within_paper(spec["paper"], spec["query"], top_k=4)
        candidates = _unique(
            [item for seed in seeds for item in engine.expand_neighbors(seed["chunk_id"])],
        )
        trace = [
            recovery_trace_entry(
                requirement["id"], "search_within_paper", "manual_failure_validation", 1,
                {"paper": spec["paper"], "query": spec["query"], "top_k": 4}, seeds,
            ),
            recovery_trace_entry(
                requirement["id"], "expand_neighbors", "manual_failure_validation", 1,
                {"seed_chunk_ids": [item["chunk_id"] for item in seeds], "radius": 1},
                candidates,
            ),
        ]
        return candidates, spec["query"], trace

    request = {
        "requirement_id": requirement["id"],
        "query": requirement["query"],
        "retrieval_strategy": requirement["retrieval_strategy"],
        "top_k": spec["top_k"],
    }
    rankings, sources = _execute_retrieval(engine, [request])
    candidates = _select_candidates_for_reranking(rankings, sources)
    trace = [
        recovery_trace_entry(
            requirement["id"], "increase_depth", "manual_failure_validation", 1,
            {**request, "from_top_k": requirement["top_k"]}, candidates,
        ),
    ]
    return candidates, requirement["query"], trace


def recover_state(
    state: dict,
    engine: RetrievalEngine,
    settings: Settings,
    spec: dict,
    *,
    rerank_function: Callable = rerank,
) -> tuple[dict, dict]:
    """Apply one manually specified recovery action and add at most two passages."""
    requirement = next(
        item for item in state["plan"]["requirements"] if item["id"] == spec["requirement_id"]
    )
    candidates, query, trace = _recovery_candidates(engine, requirement, spec)
    reranked = rerank_function([query], candidates, settings.reranker_model)
    reranked = _attach_requirement_scores(reranked, [[requirement["id"]]])
    trace[-1] = recovery_trace_entry(
        requirement["id"], trace[-1]["action"], trace[-1]["trigger"], 1,
        trace[-1]["parameters"], candidates, reranked,
    )

    existing_ids = {item["chunk_id"] for item in state["evidence"]}
    retained = [
        item for item in reranked
        if item["score"] >= settings.min_rerank_score and item["chunk_id"] not in existing_ids
    ]
    if spec["action"] == "increase_depth" and requirement["targets"]:
        retained = [
            item for item in retained
            if any(evidence_matches_target(target, item) for target in requirement["targets"])
        ]
    added = retained[:RECOVERY_EVIDENCE_LIMIT]
    combined = [*map(_raw_evidence, state["evidence"]), *added]
    evidence, board = _build_evidence_board(
        combined, state["plan"]["requirements"], settings.min_rerank_score,
    )
    final_ids = {item["chunk_id"] for item in evidence}
    for action in trace:
        for result in action["results"]:
            result["selected"] = result["chunk_id"] in final_ids

    critical_id = spec["critical_chunk_id"]
    chain = {
        "chunk_id": critical_id,
        "recovered": any(item["chunk_id"] == critical_id for item in candidates),
        "reranked": any(item["chunk_id"] == critical_id for item in reranked),
        "passed_threshold": any(
            item["chunk_id"] == critical_id and item["score"] >= settings.min_rerank_score
            for item in reranked
        ),
        "selected": critical_id in final_ids,
    }
    stages = deepcopy(state["retrieval_stages"])
    initial_pages = stages.get("retrieval", [])
    stages["post_recovery"] = _page_refs([*initial_pages, *candidates])
    return {
        "evidence": evidence,
        "evidence_board": board,
        "recovery_trace": trace,
        "retrieval_stages": stages,
    }, chain


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
    """Replay frozen plans and prepare matched baseline/recovery Writer inputs."""
    if inputs_path.exists():
        raise evaluation.EvaluationError("Frozen inputs already exist; use a new --run-id")
    source = evaluation._complete_result_set(questions, source_results_path, (source_variant,))
    samples = []
    for question in questions:
        spec = CASE_SPECS[question["id"]]
        saved = source[(question["id"], source_variant)]
        baseline = initial_state(question["question"], source_variant)
        baseline["plan"] = deepcopy(saved["trace"]["plan"])
        baseline.update(researcher_runner(baseline, engine, settings))
        current_refs = [item["chunk_id"] for item in baseline["evidence"]]
        if current_refs != [item["chunk_id"] for item in saved["evidence"]]:
            raise evaluation.EvaluationError(
                f"Selected evidence differs from the source run for {question['id']}",
            )

        requests = _retrieval_requests(baseline["plan"], source_variant)
        _, source_rankings = _execute_retrieval(engine, requests)
        initial_ids = {
            item["chunk_id"] for ranking in source_rankings for item in ranking
        }
        recovery = deepcopy(baseline)
        updates, chain = recover_state(recovery, engine, settings, spec)
        recovery.update(updates)
        initial = spec["critical_chunk_id"] in initial_ids
        critical_evidence = {
            "baseline": {
                "chunk_id": spec["critical_chunk_id"],
                "initial_retrieval": initial,
                "recovered": False,
                "reranked": False,
                "passed_threshold": False,
                "selected": spec["critical_chunk_id"] in current_refs,
            },
            "recovery": {"initial_retrieval": initial, **chain},
        }
        samples.append(
            {
                "question_id": question["id"],
                "case": spec,
                "states": {"baseline": baseline, "recovery": recovery},
                "prompts": {
                    "baseline": _writer_prompt(baseline),
                    "recovery": _writer_prompt(recovery),
                },
                "critical_evidence": critical_evidence,
                "trace_fields": {
                    variant: {
                        "recovery_mode": variant,
                        "critical_evidence": critical,
                    }
                    for variant, critical in critical_evidence.items()
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
        "recovery_evidence_limit": RECOVERY_EVIDENCE_LIMIT,
        "empty_evidence_answer": SAFE_ABSTENTION,
        "questions": list(questions),
        "samples": samples,
    }
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inputs


def _recovery_summary(summary: dict, inputs: dict) -> str:
    scores = {
        (item["question_id"], item["variant"], item["requirement_id"]):
        item["answer_requirement_accuracy"]
        for item in summary["requirement_metrics"]
    }
    questions = {item["id"]: item for item in inputs["questions"]}
    rows = []
    recovered = selected = fixed = 0
    action_count = 0
    initial_page_recalls = []
    recovered_page_recalls = []
    selected_page_recalls = []
    for sample in inputs["samples"]:
        question_id = sample["question_id"]
        case = sample["case"]
        chain = sample["critical_evidence"]["recovery"]
        before = scores[(question_id, "baseline", case["gold_requirement_id"])]
        after = scores[(question_id, "recovery", case["gold_requirement_id"])]
        recovered += int(chain["recovered"])
        selected += int(chain["selected"])
        fixed += int(before == 0 and after == 1)
        action_count += len(sample["states"]["recovery"]["recovery_trace"])
        gold_requirement = next(
            item for item in questions[question_id]["requirements"]
            if item["id"] == case["gold_requirement_id"]
        )
        gold = {(item["paper"], item["page"]) for item in gold_requirement["gold_pages"]}
        states = sample["states"]
        stage_pages = [
            states["baseline"]["retrieval_stages"]["retrieval"],
            states["recovery"]["retrieval_stages"]["post_recovery"],
            states["recovery"]["evidence"],
        ]
        recalls = [
            len(gold & {(item["paper"], item["page"]) for item in items}) / len(gold)
            for items in stage_pages
        ]
        initial_page_recalls.append(recalls[0])
        recovered_page_recalls.append(recalls[1])
        selected_page_recalls.append(recalls[2])
        rows.append(
            f"| {question_id}/{case['gold_requirement_id']} | {case['action']} | "
            f"{'✓' if chain['initial_retrieval'] else '✗'} | "
            f"{'✓' if chain['recovered'] else '✗'} | "
            f"{'✓' if chain['reranked'] else '✗'} | "
            f"{'✓' if chain['selected'] else '✗'} | {before} → {after} |"
        )
    count = len(inputs["samples"])
    summary["targeted_recovery"] = {
        "known_failures": count,
        "follow_up_trigger_rate": 1.0,
        "critical_evidence_recovery_rate": recovered / count,
        "critical_evidence_selected_rate": selected / count,
        "answer_repair_rate": fixed / count,
        "average_recovery_actions": action_count / count,
        "initial_gold_page_recall": sum(initial_page_recalls) / count,
        "post_recovery_gold_page_recall": sum(recovered_page_recalls) / count,
        "post_recovery_selected_gold_page_recall": sum(selected_page_recalls) / count,
    }
    return """\

## Targeted critical-evidence recovery

This is a mechanism check on three manually identified failures. Critical chunk IDs and action
parameters are experiment annotations and are not available to the production workflow.

| Failure | Action | Initial | Recovered | Reranked | Selected | Requirement score |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + (
        f"\n\nCritical Evidence Recovery: {recovered}/{count} ({100 * recovered / count:.1f}%)  "
        f"\nCritical Evidence Selected: {selected}/{count} ({100 * selected / count:.1f}%)  "
        f"\nAnswer Repair: {fixed}/{count} ({100 * fixed / count:.1f}%)  "
        f"\nAverage recovery actions: {action_count / count:.2f}  "
        f"\nInitial Gold-page Recall: {100 * sum(initial_page_recalls) / count:.1f}%  "
        f"\nPost-Recovery Gold-page Recall: {100 * sum(recovered_page_recalls) / count:.1f}%  "
        f"\nPost-Recovery Selected Gold-page Recall: "
        f"{100 * sum(selected_page_recalls) / count:.1f}%\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze the three paired inputs; no LLM calls")
    prepare.add_argument("--source-run", default="adaptive_v2")
    prepare.add_argument("--source-variant", choices=evaluation.VARIANTS, default="adaptive")
    commands.add_parser("run", help="Generate six answers from frozen Blackboard prompts")
    review = commands.add_parser("prepare-review", help="Create the blinded review CSV")
    review.add_argument("--force", action="store_true")
    commands.add_parser("extract-pages", help="Extract cited and gold pages for review")
    commands.add_parser("score", help="Score the paired answers and recovery chain")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        directory = evaluation.evaluation_artifact_path("inputs.json", args.run_id).parent
        inputs_path = directory / "inputs.json"
        results_path = directory / "results.jsonl"
        if args.command == "prepare":
            questions = [
                item for item in evaluation.load_questions() if item["id"] in CASE_SPECS
            ]
            settings = Settings.from_env()
            engine = RetrievalEngine.load(settings)
            prepare_inputs(
                questions, engine, settings,
                evaluation.evaluation_artifact_path("results.jsonl", args.source_run),
                inputs_path, source_variant=args.source_variant,
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
            writer_experiment.run_experiment(
                inputs_path,
                results_path,
                evaluation.evaluation_llm(),
                run_id=args.run_id,
                pipeline_version=PIPELINE_VERSION,
                variants=VARIANTS,
            )
        elif args.command == "prepare-review":
            evaluation.prepare_review(
                questions, results_path, directory / "review.csv",
                directory / "review_key.json", variants=VARIANTS, force=args.force,
            )
        elif args.command == "extract-pages":
            evaluation.export_review_evidence(
                questions, directory / "review.csv", Settings.from_env().data_dir / "papers",
                directory / "review_evidence.jsonl", variants=VARIANTS,
            )
        else:
            summary = evaluation.score_review(
                questions, results_path, directory / "review.csv",
                directory / "review_key.json", directory / "summary.json",
                directory / "summary.md", variants=VARIANTS,
            )
            note = _recovery_summary(summary, inputs)
            (directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            with (directory / "summary.md").open("a", encoding="utf-8") as handle:
                handle.write(note)
            print(evaluation._summary_markdown(summary) + note)
    except (evaluation.EvaluationError, ModelUnavailableError, OSError, ValueError, KeyError) as exc:
        print(f"Recovery experiment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
