"""Run and manually score the small resume-oriented Scholar-Agent benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import fitz

from scholar_agent.agents.planner import planner_node
from scholar_agent.citations import cited_pages
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion
from scholar_agent.workflow import initial_state, run_question

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "evals" / "questions.jsonl"
RESULTS_PATH = ROOT / "evals" / "results.jsonl"
REVIEW_PATH = ROOT / "evals" / "review.csv"
REVIEW_KEY_PATH = ROOT / "evals" / "review_key.json"
REVIEW_EVIDENCE_PATH = ROOT / "evals" / "review_evidence.jsonl"
SUMMARY_PATH = ROOT / "evals" / "summary.json"
SUMMARY_MARKDOWN_PATH = ROOT / "evals" / "summary.md"

MODEL_NAME = "deepseek-v4-flash"
PIPELINE_VERSION = "adaptive_v4_evidence_board"
EXPECTED_CORPUS_SIZE = 10_726
EXPECTED_QUESTION_COUNT = 50
DEFAULT_EVIDENCE_LIMIT = 8
MAX_RERANK_CANDIDATES = 30
REVIEW_SEED = 20260906
VARIANTS = ("fixed_hybrid", "adaptive")
VARIANT_LABELS = {
    "fixed_hybrid": "Fixed Hybrid",
    "adaptive": "Adaptive Retrieval",
    "flat": "Flat Evidence",
    "blackboard": "Evidence Blackboard",
    "baseline": "Blackboard Baseline",
    "controller": "Evidence-Gap Controller",
}
RECALL_STAGES = {
    "retrieval": "Retrieval Recall",
    "rerank": "Rerank Recall",
    "selected_evidence": "Selected Evidence Recall",
}
CATEGORY_COUNTS = {
    "single": 15,
    "comparison": 10,
    "paraphrase": 5,
    "partial": 10,
    "insufficient": 10,
}
QUESTION_ID_RE = re.compile(r"Q\d{3}")
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

class EvaluationError(RuntimeError):
    """Raised for actionable benchmark or run failures."""


def evaluation_artifact_path(filename: str, run_id: str | None) -> Path:
    """Resolve a generated artifact without allowing run IDs to escape evals/runs."""
    if run_id is None:
        return ROOT / "evals" / filename
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise EvaluationError(f"Invalid run id: {run_id}")
    return ROOT / "evals" / "runs" / run_id / filename


class CountingLLM:
    """Count LLM calls without changing the production LLM client."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.delegate.complete(prompt)

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.calls += 1
        return self.delegate.complete_json(prompt)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"File not found: {path}")
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvaluationError(f"Expected an object at {path}:{line_number}")
            values.append(value)
    return values


def _nonempty_strings(value: object, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EvaluationError(f"{location} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise EvaluationError(f"{location} must contain non-empty strings")
    return [str(item).strip() for item in value]


def validate_questions(
    questions: Sequence[dict[str, Any]],
    *,
    expected_count: int | None = EXPECTED_QUESTION_COUNT,
    expected_categories: dict[str, int] | None = CATEGORY_COUNTS,
) -> list[dict[str, Any]]:
    """Validate the hand-authored benchmark and return normalized objects."""
    if expected_count is not None and len(questions) != expected_count:
        raise EvaluationError(f"Expected {expected_count} questions, found {len(questions)}")

    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, raw_question in enumerate(questions, start=1):
        question_id = raw_question.get("id")
        category = raw_question.get("category")
        question = raw_question.get("question")
        requirements = raw_question.get("requirements")
        expected_status = raw_question.get("expected_status")
        location = f"question {position}"

        if not isinstance(question_id, str) or QUESTION_ID_RE.fullmatch(question_id) is None:
            raise EvaluationError(f"{location} has an invalid id")
        if question_id in seen_ids:
            raise EvaluationError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)
        if not isinstance(category, str) or not category:
            raise EvaluationError(f"{question_id}.category must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(f"{question_id}.question must be a non-empty string")
        if CJK_RE.search(question):
            raise EvaluationError(f"{question_id}.question must be English")
        if not isinstance(requirements, list) or not 1 <= len(requirements) <= 3:
            raise EvaluationError(f"{question_id} must have one to three requirements")

        normalized_requirements: list[dict[str, Any]] = []
        answerable_flags: list[bool] = []
        seen_requirement_ids: set[str] = set()
        for requirement_position, raw_requirement in enumerate(requirements, start=1):
            if not isinstance(raw_requirement, dict):
                raise EvaluationError(f"{question_id}.requirements must contain objects")
            requirement_id = raw_requirement.get("id")
            description = raw_requirement.get("description")
            answerable = raw_requirement.get("answerable")
            answer_key = raw_requirement.get("answer_key")
            gold_pages = raw_requirement.get("gold_pages")
            requirement_location = f"{question_id}.requirements[{requirement_position}]"

            if requirement_id != f"G{requirement_position}":
                raise EvaluationError(
                    f"{requirement_location}.id must be G{requirement_position}",
                )
            if requirement_id in seen_requirement_ids:
                raise EvaluationError(f"Duplicate requirement id in {question_id}")
            seen_requirement_ids.add(str(requirement_id))
            if not isinstance(description, str) or not description.strip():
                raise EvaluationError(f"{requirement_location}.description is required")
            if CJK_RE.search(description):
                raise EvaluationError(f"{requirement_location}.description must be English")
            if not isinstance(answerable, bool):
                raise EvaluationError(f"{requirement_location}.answerable must be boolean")
            if not isinstance(answer_key, list) or not isinstance(gold_pages, list):
                raise EvaluationError(
                    f"{requirement_location} needs answer_key and gold_pages lists",
                )

            if answerable:
                normalized_answer_key = _nonempty_strings(
                    answer_key,
                    f"{requirement_location}.answer_key",
                )
                if any(CJK_RE.search(item) for item in normalized_answer_key):
                    raise EvaluationError(
                        f"{requirement_location}.answer_key must be English",
                    )
                if not gold_pages:
                    raise EvaluationError(f"{requirement_location}.gold_pages cannot be empty")
            else:
                if answer_key or gold_pages:
                    raise EvaluationError(
                        f"{requirement_location} is unanswerable and must have empty gold fields",
                    )
                normalized_answer_key = []

            normalized_pages: list[dict[str, Any]] = []
            for page_position, raw_page in enumerate(gold_pages, start=1):
                if not isinstance(raw_page, dict) or set(raw_page) != {"paper", "page"}:
                    raise EvaluationError(
                        f"{requirement_location}.gold_pages[{page_position}] is invalid",
                    )
                paper = raw_page.get("paper")
                page = raw_page.get("page")
                if not isinstance(paper, str) or not paper.endswith(".pdf"):
                    raise EvaluationError(f"{requirement_location} has an invalid paper")
                if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                    raise EvaluationError(f"{requirement_location} has an invalid page")
                normalized_pages.append({"paper": paper, "page": page})

            answerable_flags.append(answerable)
            normalized_requirements.append(
                {
                    "id": requirement_id,
                    "description": description.strip(),
                    "answerable": answerable,
                    "answer_key": normalized_answer_key,
                    "gold_pages": normalized_pages,
                },
            )

        derived_status = (
            "complete"
            if all(answerable_flags)
            else "insufficient"
            if not any(answerable_flags)
            else "partial"
        )
        if expected_status != derived_status:
            raise EvaluationError(
                f"{question_id}.expected_status must be {derived_status}, got {expected_status}",
            )
        expected_category_status = (
            category if category in {"partial", "insufficient"} else "complete"
        )
        if derived_status != expected_category_status:
            raise EvaluationError(
                f"{question_id}.{category} questions must have status "
                f"{expected_category_status}",
            )
        normalized.append(
            {
                "id": question_id,
                "category": category,
                "question": question.strip(),
                "requirements": normalized_requirements,
                "expected_status": derived_status,
            },
        )

    if expected_categories is not None:
        actual_categories = Counter(item["category"] for item in normalized)
        if actual_categories != Counter(expected_categories):
            raise EvaluationError(
                f"Unexpected category counts: {dict(sorted(actual_categories.items()))}",
            )
    if expected_count is not None:
        expected_ids = {f"Q{index:03d}" for index in range(1, expected_count + 1)}
        if seen_ids != expected_ids:
            raise EvaluationError("Question IDs must be consecutive from Q001")
    return normalized


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    return validate_questions(_read_jsonl(path))


def validate_gold_pages(questions: Sequence[dict[str, Any]], chunks: Sequence[dict]) -> None:
    available_pages = {(item["paper"], item["page"]) for item in chunks}
    for question in questions:
        for requirement in question["requirements"]:
            for gold_page in requirement["gold_pages"]:
                page = (gold_page["paper"], gold_page["page"])
                if page not in available_pages:
                    raise EvaluationError(
                        f"Gold page is absent from the corpus: {question['id']} {page}",
                    )


def _public_evidence(evidence: Iterable[dict]) -> list[dict[str, Any]]:
    return [
        {
            "paper": item["paper"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "score": round(float(item["score"]), 6),
            **{
                key: item[key]
                for key in ("id", "paper_id", "title", "section", "supports", "requirement_scores")
                if key in item
            },
        }
        for item in evidence
    ]


def requirement_stage_metrics(
    question: dict[str, Any],
    result: dict[str, Any],
    requirement_scores: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Measure each gold requirement against the shared pools, without aligning G/R IDs."""
    stages = {
        **result.get("trace", {}).get("retrieval_stages", {}),
        "selected_evidence": result.get("evidence"),
    }
    pages_by_stage = {}
    for stage in RECALL_STAGES:
        pages = stages.get(stage)
        pages_by_stage[stage] = (
            {(item["paper"], item["page"]) for item in pages} if pages is not None else None
        )
    metrics = []
    for requirement in question["requirements"]:
        gold = {(item["paper"], item["page"]) for item in requirement["gold_pages"]}
        metrics.append(
            {
                "requirement_id": requirement["id"],
                "description": requirement["description"],
                "gold_pages": requirement["gold_pages"],
                **{
                    f"{stage}_recall": len(gold & pages) / len(gold)
                    if gold and pages is not None else None
                    for stage, pages in pages_by_stage.items()
                },
                "answer_requirement_accuracy": (requirement_scores or {}).get(requirement["id"]),
            },
        )
    return metrics


def _trace(
    answer: str,
    state: dict[str, Any],
    planner_latency: float,
    planner_llm_calls: int,
) -> dict[str, Any]:
    return {
        "plan": state.get("plan"),
        "evidence_board": state.get("evidence_board", {}),
        "shared_planner_latency_seconds": round(planner_latency, 4),
        "shared_planner_llm_calls": planner_llm_calls,
        "retrieval_mode": state.get("retrieval_mode"),
        "recovery_mode": state.get("recovery_mode", "none"),
        "retrieval_decisions": state.get("retrieval_trace", []),
        "controller": state.get("controller_trace", {}),
        "recovery_actions": state.get("recovery_trace", []),
        "retrieval_stages": state.get("retrieval_stages", {}),
        "cited_pages": [
            {"paper": paper, "page": page} for paper, page in cited_pages(answer)
        ],
    }


def _append_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def latest_results(values: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep the latest record per question/variant, allowing failed runs to be retried."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for value in values:
        question_id = value.get("question_id")
        variant = value.get("variant")
        if isinstance(question_id, str) and isinstance(variant, str):
            latest[(question_id, variant)] = value
    return latest


def _successful_result(
    latest: dict[tuple[str, str], dict[str, Any]],
    question_id: str,
    variant: str,
) -> dict[str, Any] | None:
    value = latest.get((question_id, variant))
    return value if value is not None and value.get("error") is None else None


def _result_record(
    run_id: str,
    pipeline_version: str,
    question_id: str,
    variant: str,
    answer: str,
    evidence: Sequence[dict],
    latency: float,
    llm_calls: int,
    trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "question_id": question_id,
        "variant": variant,
        "answer": answer,
        "evidence": _public_evidence(evidence),
        "latency_seconds": round(latency, 4),
        "llm_calls": llm_calls,
        "trace": trace,
        "error": None,
    }


def _error_record(
    run_id: str,
    pipeline_version: str,
    question_id: str,
    variant: str,
    exc: Exception,
    latency: float,
    llm_calls: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "question_id": question_id,
        "variant": variant,
        "answer": "",
        "evidence": [],
        "latency_seconds": round(latency, 4),
        "llm_calls": llm_calls,
        "trace": {},
        "error": f"{type(exc).__name__}: {exc}",
    }


def _resumable_results(
    results_path: Path,
    run_id: str,
    pipeline_version: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    existing = _read_jsonl(results_path) if results_path.is_file() else []
    for value in existing:
        if value.get("run_id", run_id) != run_id:
            raise EvaluationError(f"Existing results belong to run {value.get('run_id')}")
        if value.get("pipeline_version") != pipeline_version:
            raise EvaluationError(
                "Existing results use a different pipeline version: "
                f"{value.get('pipeline_version')}. Use a new --run-id.",
            )
    return latest_results(existing)


def run_evaluation(
    questions: Sequence[dict[str, Any]],
    engine: RetrievalEngine,
    settings: Settings,
    llm: CountingLLM,
    results_path: Path,
    *,
    run_id: str = "legacy",
    pipeline_version: str = PIPELINE_VERSION,
    workflow_runner: Callable[..., dict] = run_question,
    planner_runner: Callable[..., dict] = planner_node,
) -> None:
    """Run both retrieval modes from one shared, sanitized plan per question."""
    latest = _resumable_results(results_path, run_id, pipeline_version)

    for question in questions:
        question_id = question["id"]
        missing_modes = [
            mode
            for mode in VARIANTS
            if _successful_result(latest, question_id, mode) is None
        ]
        if not missing_modes:
            continue

        saved_result = next(
            (
                result
                for mode in VARIANTS
                if (result := _successful_result(latest, question_id, mode)) is not None
            ),
            None,
        )
        if saved_result is not None:
            saved_trace = saved_result.get("trace")
            if not isinstance(saved_trace, dict) or not isinstance(
                saved_trace.get("plan"),
                dict,
            ):
                raise EvaluationError(f"Saved result has no reusable plan: {question_id}")
            shared_plan = deepcopy(saved_trace["plan"])
            planner_latency = float(saved_trace["shared_planner_latency_seconds"])
            planner_llm_calls = int(saved_trace["shared_planner_llm_calls"])
        else:
            calls_before = llm.calls
            planner_started = time.perf_counter()
            try:
                shared_plan = planner_runner(
                    initial_state(question["question"], recovery_mode="none"),
                    llm,
                )["plan"]
            except Exception as exc:
                raise EvaluationError(f"Planner failed for {question_id}: {exc}") from exc
            planner_latency = time.perf_counter() - planner_started
            planner_llm_calls = llm.calls - calls_before

        for retrieval_mode in missing_modes:
            calls_before = llm.calls
            started = time.perf_counter()
            try:
                state = workflow_runner(
                    question["question"],
                    engine,
                    settings,
                    llm,
                    retrieval_mode=retrieval_mode,
                    shared_plan=deepcopy(shared_plan),
                )
                if state.get("plan") != shared_plan:
                    raise EvaluationError(
                        f"{retrieval_mode} changed the shared plan for {question_id}",
                    )
                result = _result_record(
                    run_id,
                    pipeline_version,
                    question_id,
                    retrieval_mode,
                    state["answer"],
                    state["evidence"],
                    planner_latency + time.perf_counter() - started,
                    planner_llm_calls + llm.calls - calls_before,
                    _trace(
                        state["answer"],
                        state,
                        planner_latency,
                        planner_llm_calls,
                    ),
                )
                result["requirement_metrics"] = requirement_stage_metrics(question, result)
            except Exception as exc:
                failed = _error_record(
                    run_id,
                    pipeline_version,
                    question_id,
                    retrieval_mode,
                    exc,
                    planner_latency + time.perf_counter() - started,
                    planner_llm_calls + llm.calls - calls_before,
                )
                _append_result(results_path, failed)
                raise EvaluationError(
                    f"{retrieval_mode} run failed for {question_id}: {exc}",
                ) from exc
            _append_result(results_path, result)
            latest[(question_id, retrieval_mode)] = result
            print(f"completed {question_id} {retrieval_mode}", flush=True)


def _complete_result_set(
    questions: Sequence[dict[str, Any]],
    results_path: Path,
    variants: Sequence[str] = VARIANTS,
) -> dict[tuple[str, str], dict[str, Any]]:
    latest = latest_results(_read_jsonl(results_path))
    expected = {(item["id"], variant) for item in questions for variant in variants}
    missing = [
        key
        for key in sorted(expected)
        if _successful_result(latest, key[0], key[1]) is None
    ]
    if missing:
        raise EvaluationError(f"Missing successful results: {missing[:5]}")
    return {key: latest[key] for key in expected}


def prepare_review(
    questions: Sequence[dict[str, Any]],
    results_path: Path,
    review_path: Path,
    review_key_path: Path,
    *,
    variants: Sequence[str] = VARIANTS,
    force: bool = False,
) -> None:
    """Create a deterministic, variant-blinded manual review sheet."""
    if (review_path.exists() or review_key_path.exists()) and not force:
        raise EvaluationError("Review files already exist; pass --force to replace them")
    results = _complete_result_set(questions, results_path, variants)
    questions_by_id = {item["id"]: item for item in questions}
    samples = [
        (question_id, variant, results[(question_id, variant)])
        for question_id, variant in sorted(results)
    ]
    random.Random(REVIEW_SEED).shuffle(samples)

    rows: list[dict[str, str]] = []
    keys: list[dict[str, str]] = []
    for index, (question_id, variant, result) in enumerate(samples, start=1):
        review_id = f"V{index:03d}"
        question = questions_by_id[question_id]
        citations = [
            {"paper": paper, "page": page}
            for paper, page in cited_pages(result["answer"])
        ]
        rows.append(
            {
                "review_id": review_id,
                "question_id": question_id,
                "question": question["question"],
                "expected_status": question["expected_status"],
                "requirements": json.dumps(question["requirements"], ensure_ascii=False),
                "answer": result["answer"],
                "citations": json.dumps(citations, ensure_ascii=False),
                "requirement_scores": "",
                "citation_scores": "",
                "unsupported_claims": "",
                "uncited_claims": "",
                "notes": "",
            },
        )
        keys.append(
            {"review_id": review_id, "question_id": question_id, "variant": variant},
        )

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    review_key_path.write_text(
        json.dumps(keys, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _physical_page_texts(
    page_refs: set[tuple[str, int]],
    papers_dir: Path,
) -> dict[tuple[str, int], str]:
    pages_by_paper: dict[str, set[int]] = {}
    for paper, page in page_refs:
        if Path(paper).name != paper:
            raise EvaluationError(f"Unsafe paper filename: {paper}")
        pages_by_paper.setdefault(paper, set()).add(page)

    extracted: dict[tuple[str, int], str] = {}
    for paper, page_numbers in sorted(pages_by_paper.items()):
        pdf_path = papers_dir / paper
        if not pdf_path.is_file():
            raise EvaluationError(f"Cited PDF not found: {pdf_path}")
        with fitz.open(pdf_path) as document:
            for page_number in sorted(page_numbers):
                if not 1 <= page_number <= document.page_count:
                    raise EvaluationError(
                        f"Physical page is outside {paper}: p.{page_number}",
                    )
                text = re.sub(
                    r"\s+",
                    " ",
                    document[page_number - 1].get_text("text"),
                ).strip()
                if not text:
                    raise EvaluationError(f"Physical page has no extractable text: {paper} p.{page_number}")
                extracted[(paper, page_number)] = text
    return extracted


def export_review_evidence(
    questions: Sequence[dict[str, Any]],
    review_path: Path,
    papers_dir: Path,
    output_path: Path,
    *,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, int]:
    """Export variant-blinded cited and gold physical-page text for LLM judging."""
    questions_by_id = {item["id"]: item for item in questions}
    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(questions) * len(variants)
    if len(rows) != expected_rows:
        raise EvaluationError(f"Expected {expected_rows} review rows")

    prepared: list[tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]] = []
    all_page_refs: set[tuple[str, int]] = set()
    seen_review_ids: set[str] = set()
    for row in rows:
        review_id = row.get("review_id", "")
        question_id = row.get("question_id", "")
        if not review_id or review_id in seen_review_ids:
            raise EvaluationError(f"Invalid or duplicate review id: {review_id}")
        seen_review_ids.add(review_id)
        question = questions_by_id.get(question_id)
        if question is None:
            raise EvaluationError(f"Unknown question id in {review_id}: {question_id}")
        if row.get("question") != question["question"]:
            raise EvaluationError(f"Question text was modified in {review_id}")

        try:
            citations = json.loads(row.get("citations", "[]"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Invalid citations JSON in {review_id}") from exc
        expected_citations = [
            {"paper": paper, "page": page}
            for paper, page in cited_pages(row.get("answer", ""))
        ]
        if citations != expected_citations:
            raise EvaluationError(f"Citations do not match the answer in {review_id}")

        for citation in citations:
            all_page_refs.add((citation["paper"], citation["page"]))
        for requirement in question["requirements"]:
            for gold_page in requirement["gold_pages"]:
                all_page_refs.add((gold_page["paper"], gold_page["page"]))
        prepared.append((row, question, citations))

    page_texts = _physical_page_texts(all_page_refs, papers_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row, question, citations in prepared:
            citation_refs = [(item["paper"], item["page"]) for item in citations]
            gold_requirements: dict[tuple[str, int], list[str]] = {}
            for requirement in question["requirements"]:
                for gold_page in requirement["gold_pages"]:
                    ref = (gold_page["paper"], gold_page["page"])
                    gold_requirements.setdefault(ref, []).append(requirement["id"])

            ordered_refs = list(dict.fromkeys([*citation_refs, *gold_requirements]))
            page_ids = {ref: f"P{index}" for index, ref in enumerate(ordered_refs, start=1)}
            packet = {
                "review_id": row["review_id"],
                "question_id": question["id"],
                "question": question["question"],
                "expected_status": question["expected_status"],
                "requirements": question["requirements"],
                "answer": row["answer"],
                "citations": [
                    {
                        "index": index,
                        "paper": paper,
                        "page": page,
                        "page_ref": page_ids[(paper, page)],
                    }
                    for index, (paper, page) in enumerate(citation_refs, start=1)
                ],
                "pages": [
                    {
                        "page_ref": page_ids[ref],
                        "paper": ref[0],
                        "page": ref[1],
                        "citation_indexes": [
                            index
                            for index, citation_ref in enumerate(citation_refs, start=1)
                            if citation_ref == ref
                        ],
                        "gold_requirement_ids": gold_requirements.get(ref, []),
                        "text": page_texts[ref],
                    }
                    for ref in ordered_refs
                ],
            }
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    return {"samples": len(prepared), "unique_pages": len(page_texts)}


def _binary_mapping(value: str, expected_ids: set[str], review_id: str) -> dict[str, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{review_id} has invalid requirement_scores JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != expected_ids:
        raise EvaluationError(f"{review_id} must score every requirement exactly once")
    if any(isinstance(score, bool) or score not in (0, 1) for score in parsed.values()):
        raise EvaluationError(f"{review_id} requirement scores must be 0 or 1")
    return {str(key): int(score) for key, score in parsed.items()}


def _binary_list(value: str, expected_length: int, review_id: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{review_id} has invalid citation_scores JSON") from exc
    if not isinstance(parsed, list) or len(parsed) != expected_length:
        raise EvaluationError(
            f"{review_id} needs {expected_length} citation scores in citation order",
        )
    if any(isinstance(score, bool) or score not in (0, 1) for score in parsed):
        raise EvaluationError(f"{review_id} citation scores must be 0 or 1")
    return [int(score) for score in parsed]


def _nonnegative_integer(value: str, field: str, review_id: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise EvaluationError(f"{review_id}.{field} must be a non-negative integer") from exc
    if parsed < 0 or str(parsed) != value.strip():
        raise EvaluationError(f"{review_id}.{field} must be a non-negative integer")
    return parsed


def score_review(
    questions: Sequence[dict[str, Any]],
    results_path: Path,
    review_path: Path,
    review_key_path: Path,
    summary_path: Path,
    summary_markdown_path: Path,
    *,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, Any]:
    """Combine gold-page stage recall with the existing human answer labels."""
    if len(variants) != 2 or len(set(variants)) != 2:
        raise EvaluationError("Scoring requires exactly two distinct variants")
    results = _complete_result_set(questions, results_path, variants)
    questions_by_id = {item["id"]: item for item in questions}
    raw_keys = json.loads(review_key_path.read_text(encoding="utf-8"))
    if not isinstance(raw_keys, list):
        raise EvaluationError("Review key must be a list")
    key_by_review_id = {item["review_id"]: item for item in raw_keys}
    if len(key_by_review_id) != len(raw_keys):
        raise EvaluationError("Review key contains duplicate review IDs")

    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(questions) * len(variants):
        raise EvaluationError(f"Expected {len(questions) * len(variants)} review rows")

    totals = {
        variant: {
            "questions": 0,
            "requirements": 0,
            "correct_requirements": 0,
            "citations": 0,
            "supported_citations": 0,
            "strict_successes": 0,
            "latency": 0.0,
            "llm_calls": 0,
        }
        for variant in variants
    }
    seen_samples: set[tuple[str, str]] = set()
    requirement_metrics: list[dict[str, Any]] = []

    for row in rows:
        review_id = row.get("review_id", "")
        key = key_by_review_id.get(review_id)
        if key is None:
            raise EvaluationError(f"Unknown review id: {review_id}")
        question_id = key["question_id"]
        variant = key["variant"]
        sample_key = (question_id, variant)
        if sample_key in seen_samples:
            raise EvaluationError(f"Duplicate review sample: {sample_key}")
        seen_samples.add(sample_key)

        question = questions_by_id[question_id]
        expected_requirement_ids = {item["id"] for item in question["requirements"]}
        requirement_scores = _binary_mapping(
            row.get("requirement_scores", ""),
            expected_requirement_ids,
            review_id,
        )
        citations = json.loads(row.get("citations", "[]"))
        if not isinstance(citations, list):
            raise EvaluationError(f"{review_id}.citations must be a list")
        result = results[sample_key]
        expected_citations = [
            {"paper": paper, "page": page}
            for paper, page in cited_pages(result["answer"])
        ]
        if citations != expected_citations:
            raise EvaluationError(f"{review_id}.citations was modified after preparation")
        citation_scores = _binary_list(
            row.get("citation_scores", ""),
            len(citations),
            review_id,
        )
        unsupported_claims = _nonnegative_integer(
            row.get("unsupported_claims", ""),
            "unsupported_claims",
            review_id,
        )
        uncited_claims = _nonnegative_integer(
            row.get("uncited_claims", ""),
            "uncited_claims",
            review_id,
        )
        requires_citations = any(item["answerable"] for item in question["requirements"])
        citation_policy_satisfied = (
            bool(citation_scores) if requires_citations else not citation_scores
        )
        strict_success = (
            all(score == 1 for score in requirement_scores.values())
            and all(score == 1 for score in citation_scores)
            and unsupported_claims == 0
            and uncited_claims == 0
            and citation_policy_satisfied
        )

        total = totals[variant]
        total["questions"] += 1
        total["requirements"] += len(requirement_scores)
        total["correct_requirements"] += sum(requirement_scores.values())
        total["citations"] += len(citation_scores)
        total["supported_citations"] += sum(citation_scores)
        total["strict_successes"] += int(strict_success)
        total["latency"] += float(result["latency_seconds"])
        total["llm_calls"] += int(result["llm_calls"])
        requirement_metrics.extend(
            {"question_id": question_id, "variant": variant, **metric}
            for metric in requirement_stage_metrics(question, result, requirement_scores)
        )

    expected_samples = {(item["id"], variant) for item in questions for variant in variants}
    if seen_samples != expected_samples:
        raise EvaluationError("Review sheet does not cover every question and variant")

    metrics: dict[str, dict[str, float | int | None]] = {}
    for variant, total in totals.items():
        if total["citations"] == 0:
            raise EvaluationError(f"Cannot compute citation support for {variant}: no citations")
        metrics[variant] = {
            "questions": total["questions"],
            "strict_success_rate": total["strict_successes"] / total["questions"],
            "requirement_accuracy": total["correct_requirements"] / total["requirements"],
            "citation_support_rate": total["supported_citations"] / total["citations"],
            "average_latency_seconds": total["latency"] / total["questions"],
            "average_llm_calls": total["llm_calls"] / total["questions"],
        }
        for stage in RECALL_STAGES:
            recalls = [
                item[f"{stage}_recall"]
                for item in requirement_metrics
                if item["variant"] == variant and item[f"{stage}_recall"] is not None
            ]
            metrics[variant][f"{stage}_recall"] = sum(recalls) / len(recalls) if recalls else None
            metrics[variant][f"{stage}_requirements"] = len(recalls)

    baseline_name, treatment_name = variants
    baseline = metrics[baseline_name]
    treatment = metrics[treatment_name]
    summary = {
        "benchmark_questions": len(questions),
        "comparison": {
            "baseline": baseline_name,
            "treatment": treatment_name,
        },
        "variants": metrics,
        "requirement_metrics": sorted(
            requirement_metrics,
            key=lambda item: (item["question_id"], item["variant"], item["requirement_id"]),
        ),
        "delta": {
            "strict_success_percentage_points": 100
            * (
                float(treatment["strict_success_rate"])
                - float(baseline["strict_success_rate"])
            ),
            "requirement_accuracy_percentage_points": 100
            * (
                float(treatment["requirement_accuracy"])
                - float(baseline["requirement_accuracy"])
            ),
            "citation_support_percentage_points": 100
            * (
                float(treatment["citation_support_rate"])
                - float(baseline["citation_support_rate"])
            ),
            "average_latency_seconds": float(treatment["average_latency_seconds"])
            - float(baseline["average_latency_seconds"]),
            "average_llm_calls": float(treatment["average_llm_calls"])
            - float(baseline["average_llm_calls"]),
        },
    }
    for stage in RECALL_STAGES:
        before, after = baseline[f"{stage}_recall"], treatment[f"{stage}_recall"]
        summary["delta"][f"{stage}_recall_percentage_points"] = (
            100 * (after - before) if before is not None and after is not None else None
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    comparison = summary.get(
        "comparison",
        {"baseline": "fixed_hybrid", "treatment": "adaptive"},
    )
    baseline_name = comparison["baseline"]
    treatment_name = comparison["treatment"]
    baseline = summary["variants"][baseline_name]
    treatment = summary["variants"][treatment_name]
    delta = summary["delta"]
    baseline_label = VARIANT_LABELS.get(baseline_name, baseline_name.replace("_", " ").title())
    treatment_label = VARIANT_LABELS.get(
        treatment_name,
        treatment_name.replace("_", " ").title(),
    )

    def percent(value: float | None) -> str:
        return f"{100 * value:.1f}%" if value is not None else "N/A"

    recall_rows = []
    for stage, label in RECALL_STAGES.items():
        change = delta[f"{stage}_recall_percentage_points"]
        change_text = f"{change:+.1f} pp" if change is not None else "N/A"
        recall_rows.append(
            f"| {label} | {percent(baseline[f'{stage}_recall'])} | "
            f"{percent(treatment[f'{stage}_recall'])} | {change_text} |",
        )

    def outcome(value: float | None) -> str:
        if value is None:
            return "N/A"
        return "✓" if value == 1 else "✗" if value == 0 else percent(value)

    requirement_rows = []
    for item in summary["requirement_metrics"]:
        values = [item[f"{stage}_recall"] for stage in RECALL_STAGES]
        values.append(item["answer_requirement_accuracy"])
        requirement_rows.append(
            f"| {item['question_id']} | {item['variant']} | {item['requirement_id']} | "
            + " | ".join(outcome(value) for value in values)
            + " |",
        )
    recall_table = "\n".join(recall_rows)
    requirement_table = "\n".join(requirement_rows)

    return f"""# Scholar-Agent evaluation summary

| Metric | {baseline_label} | {treatment_label} | Delta |
|---|---:|---:|---:|
| Strict Success | {percent(baseline['strict_success_rate'])} | {percent(treatment['strict_success_rate'])} | {delta['strict_success_percentage_points']:+.1f} pp |
{recall_table}
| Answer Requirement Accuracy | {percent(baseline['requirement_accuracy'])} | {percent(treatment['requirement_accuracy'])} | {delta['requirement_accuracy_percentage_points']:+.1f} pp |
| Citation Support | {percent(baseline['citation_support_rate'])} | {percent(treatment['citation_support_rate'])} | {delta['citation_support_percentage_points']:+.1f} pp |
| Average latency | {baseline['average_latency_seconds']:.2f}s | {treatment['average_latency_seconds']:.2f}s | {delta['average_latency_seconds']:+.2f}s |
| Average LLM calls | {baseline['average_llm_calls']:.2f} | {treatment['average_llm_calls']:.2f} | {delta['average_llm_calls']:+.2f} |

Recall is the macro-average of per-requirement gold-page coverage. Requirements without
gold pages and stages missing from older traces are N/A and excluded from recall averages.
Answer accuracy includes all requirements and uses the existing review labels.

## Requirement-level stages

✓ = all gold pages reached the stage (or the answer passed); ✗ = zero recall (or the
answer failed). Partial gold-page coverage is shown as a percentage.
Rerank Recall measures entry into the candidate pool, before score filtering and selection.

| Question | Variant | Requirement | Retrieval Recall | Rerank Recall | Selected Evidence Recall | Answer Requirement Accuracy |
|---|---|---|---:|---:|---:|---:|
{requirement_table}
"""


def evaluation_llm() -> CountingLLM:
    """Use the same provider and model for retrieval and Writer experiments."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EvaluationError("DEEPSEEK_API_KEY is required")
    configured_model = os.getenv("SCHOLAR_AGENT_LLM_MODEL")
    if configured_model and configured_model != MODEL_NAME:
        raise EvaluationError(f"SCHOLAR_AGENT_LLM_MODEL must be {MODEL_NAME}")

    settings = replace(Settings.from_env(), llm_model=MODEL_NAME)
    llm = LLMClient.from_env(settings)
    if llm is None or llm.model != MODEL_NAME:
        raise EvaluationError(f"Evaluation requires the {MODEL_NAME} model")
    return CountingLLM(llm)


def _runtime() -> tuple[list[dict[str, Any]], RetrievalEngine, Settings, CountingLLM]:
    questions = load_questions()
    llm = evaluation_llm()
    settings = replace(Settings.from_env(), llm_model=MODEL_NAME)
    engine = RetrievalEngine.load(settings)
    if len(engine.chunks) != EXPECTED_CORPUS_SIZE:
        raise EvaluationError(
            f"Expected {EXPECTED_CORPUS_SIZE} chunks, found {len(engine.chunks)}",
        )
    if len(engine.bm25.tokens) != EXPECTED_CORPUS_SIZE:
        raise EvaluationError("BM25 index size does not match the expected corpus")
    if engine.dense.embeddings.shape[0] != EXPECTED_CORPUS_SIZE:
        raise EvaluationError("Dense index size does not match the expected corpus")
    validate_gold_pages(questions, engine.chunks)

    return questions, engine, settings, llm


def _warm_up(question: str, engine: RetrievalEngine, settings: Settings) -> None:
    sparse = engine.sparse_search([question], top_k=DEFAULT_EVIDENCE_LIMIT)
    dense_rankings = engine.dense_search_many(
        [question],
        top_k=DEFAULT_EVIDENCE_LIMIT,
    )
    dense = dense_rankings[0] if dense_rankings else []
    candidates = reciprocal_rank_fusion(sparse, dense)[:MAX_RERANK_CANDIDATES]
    rerank([question], candidates, settings.reranker_model)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        help="Store generated artifacts under evals/runs/<run-id>",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run Fixed Hybrid and Adaptive Retrieval")
    review = subparsers.add_parser("prepare-review", help="Create the blinded review CSV")
    review.add_argument("--force", action="store_true", help="Replace existing review files")
    subparsers.add_parser(
        "extract-pages",
        help="Extract cited and gold PDF pages for blinded judging",
    )
    subparsers.add_parser("score", help="Aggregate a completed review CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results_path = evaluation_artifact_path("results.jsonl", args.run_id)
        review_path = evaluation_artifact_path("review.csv", args.run_id)
        review_key_path = evaluation_artifact_path("review_key.json", args.run_id)
        review_evidence_path = evaluation_artifact_path(
            "review_evidence.jsonl",
            args.run_id,
        )
        summary_path = evaluation_artifact_path("summary.json", args.run_id)
        summary_markdown_path = evaluation_artifact_path("summary.md", args.run_id)
        run_id = args.run_id or "legacy"
        if args.command == "run":
            questions, engine, settings, llm = _runtime()
            _warm_up(questions[0]["question"], engine, settings)
            run_evaluation(
                questions,
                engine,
                settings,
                llm,
                results_path,
                run_id=run_id,
            )
        elif args.command == "prepare-review":
            questions = load_questions()
            prepare_review(
                questions,
                results_path,
                review_path,
                review_key_path,
                force=args.force,
            )
        elif args.command == "extract-pages":
            questions = load_questions()
            stats = export_review_evidence(
                questions,
                review_path,
                ROOT / "data" / "papers",
                review_evidence_path,
            )
            print(
                f"exported {stats['samples']} blind samples with "
                f"{stats['unique_pages']} unique physical pages",
            )
        else:
            questions = load_questions()
            summary = score_review(
                questions,
                results_path,
                review_path,
                review_key_path,
                summary_path,
                summary_markdown_path,
            )
            print(_summary_markdown(summary))
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
