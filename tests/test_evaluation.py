from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import evals.evaluate as evaluation
import fitz
from evals.evaluate import (
    CountingLLM,
    baseline_evidence_limit,
    export_review_evidence,
    prepare_review,
    run_evaluation,
    run_simple_rag,
    score_review,
    validate_questions,
)

from scholar_agent.config import Settings


class StubLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text

    def complete_json(self, prompt: str) -> dict[str, Any]:
        raise AssertionError("The Simple RAG baseline must not request JSON planning")


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.sparse_calls: list[list[str]] = []
        self.dense_calls: list[list[str]] = []

    def sparse_search(self, queries: list[str]) -> list[dict]:
        self.sparse_calls.append(queries)
        return self.chunks

    def dense_search_many(self, queries: list[str]) -> list[list[dict]]:
        self.dense_calls.append(queries)
        return [list(reversed(self.chunks))]


def small_questions() -> list[dict[str, Any]]:
    return validate_questions(
        [
            {
                "id": "Q001",
                "category": "single",
                "question": "Explain Method A.",
                "requirements": [
                    {
                        "id": "G1",
                        "description": "Explain Method A.",
                        "answerable": True,
                        "answer_key": ["Method A retrieves evidence."],
                        "gold_pages": [{"paper": "A.pdf", "page": 1}],
                    },
                ],
                "expected_status": "complete",
            },
            {
                "id": "Q002",
                "category": "insufficient",
                "question": "Report Method B's result on an unavailable task.",
                "requirements": [
                    {
                        "id": "G1",
                        "description": "Report the unavailable result.",
                        "answerable": False,
                        "answer_key": [],
                        "gold_pages": [],
                    },
                ],
                "expected_status": "insufficient",
            },
        ],
        expected_count=None,
        expected_categories=None,
    )


def test_committed_benchmark_has_fifty_english_questions() -> None:
    questions = evaluation.load_questions()

    assert len(questions) == 50
    assert sum(len(item["requirements"]) for item in questions) == 71
    assert {item["expected_status"] for item in questions} == {
        "complete",
        "partial",
        "insufficient",
    }


def test_simple_rag_runs_hybrid_reranking_and_validates_citations(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    llm = StubLLM("Supported [E1]. Fabricated [E99] [Fake.pdf p.999].")
    fusion_calls: list[tuple[list[str], list[str]]] = []

    def fuse(sparse: list[dict], dense: list[dict]) -> list[dict]:
        fusion_calls.append(
            ([item["chunk_id"] for item in sparse], [item["chunk_id"] for item in dense]),
        )
        return sparse

    rerank_calls: list[tuple[list[str], list[str]]] = []

    def score(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        rerank_calls.append((queries, [item["chunk_id"] for item in candidates]))
        return [{**item, "score": 1.0} for item in candidates]

    monkeypatch.setattr(evaluation, "reciprocal_rank_fusion", fuse)
    result = run_simple_rag(
        "Explain Self-RAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,
        evidence_limit=1,
        rerank_function=score,
    )

    assert engine.sparse_calls == [["Explain Self-RAG"]]
    assert engine.dense_calls == [["Explain Self-RAG"]]
    assert fusion_calls == [(["self-1", "crag-1"], ["crag-1", "self-1"])]
    assert rerank_calls == [(["Explain Self-RAG"], ["self-1", "crag-1"])]
    assert [item["chunk_id"] for item in result["evidence"]] == ["self-1"]
    assert result["answer"] == "Supported [Self-RAG.pdf p.1]. Fabricated."
    assert len(llm.prompts) == 1


def test_baseline_receives_at_least_the_full_evidence_budget() -> None:
    assert baseline_evidence_limit(0) == 8
    assert baseline_evidence_limit(8) == 8
    assert baseline_evidence_limit(11) == 11


def test_run_wires_the_full_evidence_count_into_the_baseline_budget(
    sample_chunks: list[dict],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    questions = small_questions()[:1]
    engine = FakeEngine(sample_chunks[:1])
    llm = CountingLLM(StubLLM("unused"))
    full_evidence = [
        {**sample_chunks[0], "chunk_id": f"full-{index}"}
        for index in range(11)
    ]
    observed_limits: list[int] = []

    def full_runner(question: str, engine: object, settings: Settings, llm: object) -> dict:
        return {
            "answer": "Full [Self-RAG.pdf p.1].",
            "evidence": full_evidence,
            "verification": {"status": "complete"},
        }

    def simple_runner(
        question: str,
        engine: object,
        settings: Settings,
        llm: object,
        evidence_limit: int,
        *,
        rerank_function: object,
    ) -> dict:
        observed_limits.append(evidence_limit)
        return {"answer": "Simple [Self-RAG.pdf p.1].", "evidence": sample_chunks[:1]}

    monkeypatch.setattr(evaluation, "run_simple_rag", simple_runner)
    run_evaluation(
        questions,
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,
        tmp_path / "results.jsonl",
        full_runner=full_runner,  # type: ignore[arg-type]
    )

    assert observed_limits == [11]


def test_run_is_resumable_and_does_not_repeat_successful_pairs(
    sample_chunks: list[dict],
    tmp_path: Path,
) -> None:
    questions = small_questions()[:1]
    engine = FakeEngine(sample_chunks[:1])
    llm = CountingLLM(StubLLM("Answer [E1]."))
    full_calls: list[str] = []

    def full_runner(question: str, engine: object, settings: Settings, llm: object) -> dict:
        full_calls.append(question)
        return {
            "answer": "Full [Self-RAG.pdf p.1].",
            "evidence": sample_chunks[:1],
            "verification": {"status": "complete"},
        }

    def score(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [{**item, "score": 1.0} for item in candidates]

    results_path = tmp_path / "results.jsonl"
    run_evaluation(
        questions,
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,
        results_path,
        full_runner=full_runner,  # type: ignore[arg-type]
        rerank_function=score,
    )
    first_contents = results_path.read_text(encoding="utf-8")
    run_evaluation(
        questions,
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,
        results_path,
        full_runner=full_runner,  # type: ignore[arg-type]
        rerank_function=score,
    )

    assert results_path.read_text(encoding="utf-8") == first_contents
    assert full_calls == ["Explain Method A."]
    assert len(first_contents.splitlines()) == 2


def test_blind_review_round_trip_computes_resume_metrics(tmp_path: Path) -> None:
    questions = small_questions()
    results_path = tmp_path / "results.jsonl"
    records = []
    for question in questions:
        for variant in evaluation.VARIANTS:
            is_full_insufficient = question["id"] == "Q002" and variant == "full"
            records.append(
                {
                    "question_id": question["id"],
                    "variant": variant,
                    "answer": (
                        "The corpus lacks enough evidence."
                        if is_full_insufficient
                        else "Claim [A.pdf p.1]."
                    ),
                    "verification_status": (
                        question["expected_status"] if variant == "full" else None
                    ),
                    "evidence": [],
                    "latency_seconds": 2.0 if variant == "full" else 1.0,
                    "llm_calls": 3 if variant == "full" else 1,
                    "error": None,
                },
            )
    results_path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    review_path = tmp_path / "review.csv"
    key_path = tmp_path / "review_key.json"
    prepare_review(questions, results_path, review_path, key_path)

    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "variant" not in rows[0]
    keys = {
        item["review_id"]: item
        for item in json.loads(key_path.read_text(encoding="utf-8"))
    }
    for row in rows:
        key = keys[row["review_id"]]
        is_full = key["variant"] == "full"
        is_first_simple = key["variant"] == "simple_rag" and key["question_id"] == "Q001"
        row["requirement_scores"] = json.dumps({"G1": int(is_full or is_first_simple)})
        citation_count = len(json.loads(row["citations"]))
        row["citation_scores"] = json.dumps(
            [int(is_full or is_first_simple)] * citation_count,
        )
        row["unsupported_claims"] = "0" if is_full or is_first_simple else "1"
        row["uncited_claims"] = "0"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    summary = score_review(
        questions,
        results_path,
        review_path,
        key_path,
        summary_path,
        markdown_path,
    )

    assert summary["variants"]["simple_rag"]["strict_success_rate"] == 0.5
    assert summary["variants"]["simple_rag"]["requirement_accuracy"] == 0.5
    assert summary["variants"]["simple_rag"]["citation_support_rate"] == 0.5
    assert summary["variants"]["full"]["strict_success_rate"] == 1.0
    assert summary["variants"]["full"]["requirement_accuracy"] == 1.0
    assert summary["variants"]["full"]["citation_support_rate"] == 1.0
    assert summary["delta"]["strict_success_percentage_points"] == 50.0
    assert "| Strict Success | 50.0% | 100.0% | +50.0 pp |" in markdown_path.read_text(
        encoding="utf-8",
    )


def test_export_review_evidence_extracts_and_deduplicates_physical_pages(
    tmp_path: Path,
) -> None:
    questions = small_questions()[:1]
    review_path = tmp_path / "review.csv"
    citations = json.dumps(
        [
            {"paper": "A.pdf", "page": 1},
            {"paper": "A.pdf", "page": 1},
        ],
    )
    rows = [
        {
            "review_id": f"V{index:03d}",
            "question_id": "Q001",
            "question": questions[0]["question"],
            "answer": "First [A.pdf p.1]. Second [A.pdf p.1].",
            "citations": citations,
        }
        for index in range(1, 3)
    ]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Method A retrieves supporting evidence.")
    document.save(papers_dir / "A.pdf")
    document.close()

    output_path = tmp_path / "review_evidence.jsonl"
    stats = export_review_evidence(questions, review_path, papers_dir, output_path)
    packets = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert stats == {"samples": 2, "unique_pages": 1}
    assert len(packets) == 2
    assert "variant" not in packets[0]
    assert [item["page_ref"] for item in packets[0]["citations"]] == ["P1", "P1"]
    assert packets[0]["pages"] == [
        {
            "page_ref": "P1",
            "paper": "A.pdf",
            "page": 1,
            "citation_indexes": [1, 2],
            "gold_requirement_ids": ["G1"],
            "text": "Method A retrieves supporting evidence.",
        },
    ]
