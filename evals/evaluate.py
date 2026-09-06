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
from dataclasses import replace
from pathlib import Path
from typing import Any

import fitz

from scholar_agent.citations import PAGE_CITATION_RE, cited_pages, validate_citations
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion
from scholar_agent.workflow import run_question

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "evals" / "questions.jsonl"
RESULTS_PATH = ROOT / "evals" / "results.jsonl"
REVIEW_PATH = ROOT / "evals" / "review.csv"
REVIEW_KEY_PATH = ROOT / "evals" / "review_key.json"
REVIEW_EVIDENCE_PATH = ROOT / "evals" / "review_evidence.jsonl"
SUMMARY_PATH = ROOT / "evals" / "summary.json"
SUMMARY_MARKDOWN_PATH = ROOT / "evals" / "summary.md"

MODEL_NAME = "deepseek-v4-flash"
PIPELINE_VERSION = "v2_answer_verifier"
EXPECTED_CORPUS_SIZE = 10_726
EXPECTED_QUESTION_COUNT = 50
DEFAULT_EVIDENCE_LIMIT = 8
MAX_RERANK_CANDIDATES = 30
REVIEW_SEED = 20260906
VARIANTS = ("simple_rag", "full")
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

RerankFunction = Callable[[list[str], list[dict], str], list[dict]]


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


def baseline_evidence_limit(full_evidence_count: int) -> int:
    """Give the baseline at least as many evidence slots as the full system used."""
    return max(DEFAULT_EVIDENCE_LIMIT, full_evidence_count)


def _baseline_evidence(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    evidence_limit: int,
    rerank_function: RerankFunction,
) -> list[dict]:
    sparse = engine.sparse_search([question])
    dense_rankings = engine.dense_search_many([question])
    dense = dense_rankings[0] if dense_rankings else []
    candidates = reciprocal_rank_fusion(sparse, dense)[:MAX_RERANK_CANDIDATES]
    ranked = rerank_function([question], candidates, settings.reranker_model)
    return [
        item
        for item in ranked
        if float(item["score"]) >= settings.min_rerank_score
    ][:evidence_limit]


def _baseline_prompt(question: str, evidence: Sequence[dict]) -> str:
    evidence_text = "\n".join(
        f"[E{index}] {item['text']}" for index, item in enumerate(evidence, start=1)
    )
    return f"""Answer the academic question in English using only the supplied evidence.

Every factual statement must have an inline [E1], [E2], ... citation. Use only supplied
evidence IDs. If the evidence supports only part of the question, answer that part and state
what is missing. If it supports none of the question, give a concise abstention without
factual claims or citations. Do not fill gaps from memory.

Question: {question}

Evidence:
{evidence_text}
"""


def run_simple_rag(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    llm: Any,
    evidence_limit: int,
    *,
    rerank_function: RerankFunction = rerank,
) -> dict[str, Any]:
    """Run the single-query Hybrid RAG baseline without any agent nodes."""
    evidence = _baseline_evidence(
        question,
        engine,
        settings,
        evidence_limit,
        rerank_function,
    )
    draft = llm.complete(_baseline_prompt(question, evidence))
    draft = PAGE_CITATION_RE.sub("", draft)
    return {
        "answer": validate_citations(draft, list(evidence)),
        "evidence": evidence,
    }


def _public_evidence(evidence: Iterable[dict]) -> list[dict[str, Any]]:
    return [
        {
            "paper": item["paper"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "score": round(float(item["score"]), 6),
        }
        for item in evidence
    ]


def _trace(answer: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": state.get("plan"),
        "verification": state.get("verification"),
        "retry_count": int(state.get("retry_count", 0)),
        "stop_reason": state.get("stop_reason", ""),
        "answer_verification": state.get("answer_verification"),
        "repair_count": int(state.get("repair_count", 0)),
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
    verification_status: str | None,
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
        "verification_status": verification_status,
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
        "verification_status": None,
        "evidence": [],
        "latency_seconds": round(latency, 4),
        "llm_calls": llm_calls,
        "trace": {},
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_evaluation(
    questions: Sequence[dict[str, Any]],
    engine: RetrievalEngine,
    settings: Settings,
    llm: CountingLLM,
    results_path: Path,
    *,
    run_id: str = "legacy",
    pipeline_version: str = PIPELINE_VERSION,
    full_runner: Callable[[str, RetrievalEngine, Settings, Any], dict] = run_question,
    rerank_function: RerankFunction = rerank,
) -> None:
    """Run Full first, then a budget-matched baseline, with resumable JSONL output."""
    existing = _read_jsonl(results_path) if results_path.is_file() else []
    for value in existing:
        if value.get("run_id", run_id) != run_id:
            raise EvaluationError(f"Existing results belong to run {value.get('run_id')}")
        if value.get("pipeline_version", pipeline_version) != pipeline_version:
            raise EvaluationError(
                "Existing results use a different pipeline version: "
                f"{value.get('pipeline_version')}",
            )
    latest = latest_results(existing)

    for question in questions:
        question_id = question["id"]
        full = _successful_result(latest, question_id, "full")
        if full is None:
            calls_before = llm.calls
            started = time.perf_counter()
            try:
                state = full_runner(question["question"], engine, settings, llm)
                full = _result_record(
                    run_id,
                    pipeline_version,
                    question_id,
                    "full",
                    state["answer"],
                    state["evidence"],
                    state["verification"]["status"],
                    time.perf_counter() - started,
                    llm.calls - calls_before,
                    _trace(state["answer"], state),
                )
            except Exception as exc:
                failed = _error_record(
                    run_id,
                    pipeline_version,
                    question_id,
                    "full",
                    exc,
                    time.perf_counter() - started,
                    llm.calls - calls_before,
                )
                _append_result(results_path, failed)
                raise EvaluationError(f"Full run failed for {question_id}: {exc}") from exc
            _append_result(results_path, full)
            latest[(question_id, "full")] = full
            print(f"completed {question_id} full", flush=True)

        simple = _successful_result(latest, question_id, "simple_rag")
        if simple is not None:
            continue
        evidence_limit = baseline_evidence_limit(len(full["evidence"]))
        calls_before = llm.calls
        started = time.perf_counter()
        try:
            state = run_simple_rag(
                question["question"],
                engine,
                settings,
                llm,
                evidence_limit,
                rerank_function=rerank_function,
            )
            simple = _result_record(
                run_id,
                pipeline_version,
                question_id,
                "simple_rag",
                state["answer"],
                state["evidence"],
                None,
                time.perf_counter() - started,
                llm.calls - calls_before,
                _trace(state["answer"], state),
            )
        except Exception as exc:
            failed = _error_record(
                run_id,
                pipeline_version,
                question_id,
                "simple_rag",
                exc,
                time.perf_counter() - started,
                llm.calls - calls_before,
            )
            _append_result(results_path, failed)
            raise EvaluationError(f"Simple RAG failed for {question_id}: {exc}") from exc
        _append_result(results_path, simple)
        latest[(question_id, "simple_rag")] = simple
        print(f"completed {question_id} simple_rag", flush=True)


def _complete_result_set(
    questions: Sequence[dict[str, Any]],
    results_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    latest = latest_results(_read_jsonl(results_path))
    expected = {(item["id"], variant) for item in questions for variant in VARIANTS}
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
    force: bool = False,
) -> None:
    """Create a deterministic, variant-blinded manual review sheet."""
    if (review_path.exists() or review_key_path.exists()) and not force:
        raise EvaluationError("Review files already exist; pass --force to replace them")
    results = _complete_result_set(questions, results_path)
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
) -> dict[str, int]:
    """Export variant-blinded cited and gold physical-page text for LLM judging."""
    questions_by_id = {item["id"]: item for item in questions}
    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(questions) * len(VARIANTS)
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
) -> dict[str, Any]:
    """Validate human labels and aggregate resume-oriented metrics."""
    results = _complete_result_set(questions, results_path)
    questions_by_id = {item["id"]: item for item in questions}
    raw_keys = json.loads(review_key_path.read_text(encoding="utf-8"))
    if not isinstance(raw_keys, list):
        raise EvaluationError("Review key must be a list")
    key_by_review_id = {item["review_id"]: item for item in raw_keys}
    if len(key_by_review_id) != len(raw_keys):
        raise EvaluationError("Review key contains duplicate review IDs")

    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(questions) * len(VARIANTS):
        raise EvaluationError(f"Expected {len(questions) * len(VARIANTS)} review rows")

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
        for variant in VARIANTS
    }
    seen_samples: set[tuple[str, str]] = set()

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

    expected_samples = {(item["id"], variant) for item in questions for variant in VARIANTS}
    if seen_samples != expected_samples:
        raise EvaluationError("Review sheet does not cover every question and variant")

    metrics: dict[str, dict[str, float | int]] = {}
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

    simple = metrics["simple_rag"]
    full = metrics["full"]
    summary = {
        "benchmark_questions": len(questions),
        "variants": metrics,
        "delta": {
            "strict_success_percentage_points": 100
            * (float(full["strict_success_rate"]) - float(simple["strict_success_rate"])),
            "requirement_accuracy_percentage_points": 100
            * (float(full["requirement_accuracy"]) - float(simple["requirement_accuracy"])),
            "citation_support_percentage_points": 100
            * (float(full["citation_support_rate"]) - float(simple["citation_support_rate"])),
            "average_latency_seconds": float(full["average_latency_seconds"])
            - float(simple["average_latency_seconds"]),
            "average_llm_calls": float(full["average_llm_calls"])
            - float(simple["average_llm_calls"]),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    simple = summary["variants"]["simple_rag"]
    full = summary["variants"]["full"]
    delta = summary["delta"]

    def percent(value: float) -> str:
        return f"{100 * value:.1f}%"

    return f"""# Scholar-Agent evaluation summary

| Metric | Simple RAG | Scholar-Agent | Delta |
|---|---:|---:|---:|
| Strict Success | {percent(simple['strict_success_rate'])} | {percent(full['strict_success_rate'])} | {delta['strict_success_percentage_points']:+.1f} pp |
| Requirement Accuracy | {percent(simple['requirement_accuracy'])} | {percent(full['requirement_accuracy'])} | {delta['requirement_accuracy_percentage_points']:+.1f} pp |
| Citation Support | {percent(simple['citation_support_rate'])} | {percent(full['citation_support_rate'])} | {delta['citation_support_percentage_points']:+.1f} pp |
| Average latency | {simple['average_latency_seconds']:.2f}s | {full['average_latency_seconds']:.2f}s | {delta['average_latency_seconds']:+.2f}s |
| Average LLM calls | {simple['average_llm_calls']:.2f} | {full['average_llm_calls']:.2f} | {delta['average_llm_calls']:+.2f} |
"""


def _runtime() -> tuple[list[dict[str, Any]], RetrievalEngine, Settings, CountingLLM]:
    questions = load_questions()
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EvaluationError("DEEPSEEK_API_KEY is required")
    configured_model = os.getenv("SCHOLAR_AGENT_LLM_MODEL")
    if configured_model and configured_model != MODEL_NAME:
        raise EvaluationError(f"SCHOLAR_AGENT_LLM_MODEL must be {MODEL_NAME}")

    settings = replace(Settings.from_env(), llm_model=MODEL_NAME, max_retries=1)
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

    llm = LLMClient.from_env(settings)
    if llm is None or llm.model != MODEL_NAME:
        raise EvaluationError(f"Evaluation requires the {MODEL_NAME} model")
    return questions, engine, settings, CountingLLM(llm)


def _warm_up(question: str, engine: RetrievalEngine, settings: Settings) -> None:
    _baseline_evidence(question, engine, settings, DEFAULT_EVIDENCE_LIMIT, rerank)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        help="Store generated artifacts under evals/runs/<run-id>",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run Full Scholar-Agent and Simple RAG")
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
