from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import evals.evaluate as evaluation
import fitz
import pytest
from evals.evaluate import (
    CountingLLM,
    evaluation_artifact_path,
    export_review_evidence,
    prepare_review,
    requirement_stage_metrics,
    run_evaluation,
    score_review,
    validate_questions,
)

from scholar_agent.config import Settings


class StubLLM:
    def complete(self, prompt: str) -> str:
        return "answer"

    def complete_json(self, prompt: str) -> dict[str, Any]:
        return {"requirements": []}


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks


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


def test_evaluation_runs_and_resumes_fixed_hybrid_and_adaptive(
    sample_chunks: list[dict],
    tmp_path: Path,
) -> None:
    questions = small_questions()[:1]
    llm = CountingLLM(StubLLM())
    calls: list[str] = []
    planner_calls: list[str] = []
    plan = {
        "requirements": [
            {
                "id": "R1",
                "description": "Explain Method A",
                "targets": [],
                "query": "Method A",
                "retrieval_strategy": "bm25",
                "top_k": 6,
            },
        ],
    }

    def planner_runner(state: dict, counting_llm: CountingLLM) -> dict[str, Any]:
        planner_calls.append(state["question"])
        counting_llm.complete_json("plan")
        return {"plan": plan}

    def workflow_runner(
        question: str,
        engine: object,
        settings: Settings,
        counting_llm: CountingLLM,
        *,
        retrieval_mode: str,
        shared_plan: dict,
    ) -> dict[str, Any]:
        calls.append(retrieval_mode)
        counting_llm.complete("write")
        executed_strategy = "hybrid" if retrieval_mode == "fixed_hybrid" else "bm25"
        return {
            "answer": "Answer [Self-RAG.pdf p.1].",
            "retrieval_mode": retrieval_mode,
            "plan": shared_plan,
            "retrieval_trace": [
                {
                    "requirement_id": "R1",
                    "query": "Method A",
                    "retrieval_strategy": executed_strategy,
                    "top_k": 6,
                },
            ],
            "evidence": sample_chunks[:1],
            "retrieval_stages": {
                "retrieval": [{"paper": "A.pdf", "page": 1}],
                "rerank": [{"paper": "A.pdf", "page": 1}],
            },
        }

    results_path = tmp_path / "results.jsonl"
    for _ in range(2):
        run_evaluation(
            questions,
            FakeEngine(sample_chunks),  # type: ignore[arg-type]
            Settings(),
            llm,
            results_path,
            workflow_runner=workflow_runner,
            planner_runner=planner_runner,
        )

    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert planner_calls == ["Explain Method A."]
    assert calls == ["fixed_hybrid", "adaptive"]
    assert [record["variant"] for record in records] == ["fixed_hybrid", "adaptive"]
    assert {record["pipeline_version"] for record in records} == {
        evaluation.PIPELINE_VERSION,
    }
    assert [record["llm_calls"] for record in records] == [2, 2]
    assert records[0]["trace"]["plan"] == records[1]["trace"]["plan"] == plan
    assert records[0]["trace"]["shared_planner_llm_calls"] == 1
    assert records[0]["trace"]["retrieval_decisions"] == [
        {
            "requirement_id": "R1",
            "query": "Method A",
            "retrieval_strategy": "hybrid",
            "top_k": 6,
        },
    ]
    assert records[1]["trace"]["retrieval_decisions"][0]["retrieval_strategy"] == "bm25"
    assert records[0]["trace"]["retrieval_stages"] == {
        "retrieval": [{"paper": "A.pdf", "page": 1}],
        "rerank": [{"paper": "A.pdf", "page": 1}],
    }
    assert records[0]["requirement_metrics"][0] == {
        "requirement_id": "G1",
        "description": "Explain Method A.",
        "gold_pages": [{"paper": "A.pdf", "page": 1}],
        "retrieval_recall": 1.0,
        "rerank_recall": 1.0,
        "selected_evidence_recall": 0.0,
        "answer_requirement_accuracy": None,
    }
    for removed in (
        "verification",
        "answer_verification",
        "retry_count",
        "repair_count",
    ):
        assert removed not in records[0]["trace"]


def test_evaluation_resume_reuses_the_saved_plan(
    sample_chunks: list[dict],
    tmp_path: Path,
) -> None:
    questions = small_questions()[:1]
    plan = {
        "requirements": [
            {
                "id": "R1",
                "description": "Explain Method A",
                "targets": [],
                "query": "shared query",
                "retrieval_strategy": "dense",
                "top_k": 8,
            },
        ],
    }
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "run_id": "legacy",
                "pipeline_version": evaluation.PIPELINE_VERSION,
                "question_id": "Q001",
                "variant": "fixed_hybrid",
                "answer": "Saved answer.",
                "evidence": [],
                "latency_seconds": 1.0,
                "llm_calls": 2,
                "trace": {
                    "plan": plan,
                    "shared_planner_latency_seconds": 0.25,
                    "shared_planner_llm_calls": 1,
                    "retrieval_mode": "fixed_hybrid",
                    "retrieval_decisions": [],
                    "cited_pages": [],
                },
                "error": None,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    observed: list[tuple[str, dict]] = []

    def workflow_runner(
        question: str,
        engine: object,
        settings: Settings,
        llm: CountingLLM,
        *,
        retrieval_mode: str,
        shared_plan: dict,
    ) -> dict[str, Any]:
        observed.append((retrieval_mode, shared_plan))
        llm.complete("write")
        return {
            "answer": "Resumed answer.",
            "retrieval_mode": retrieval_mode,
            "plan": shared_plan,
            "retrieval_trace": [],
            "evidence": sample_chunks[:1],
        }

    run_evaluation(
        questions,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(),
        CountingLLM(StubLLM()),
        results_path,
        workflow_runner=workflow_runner,
        planner_runner=lambda *args: pytest.fail("saved plan must be reused"),
    )

    assert observed == [("adaptive", plan)]
    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[1]["llm_calls"] == 2
    assert records[1]["trace"]["plan"] == plan


def test_evaluation_parser_is_the_retrieval_ablation_only() -> None:
    args = evaluation._parser().parse_args(["--run-id", "adaptive_v2", "run"])

    assert args.run_id == "adaptive_v2"
    assert args.command == "run"
    with pytest.raises(SystemExit):
        evaluation._parser().parse_args(["--coverage-mode", "soft", "run"])


def test_versioned_artifacts_stay_inside_the_run_directory() -> None:
    path = evaluation_artifact_path("results.jsonl", "adaptive_v2")

    assert path == evaluation.ROOT / "evals" / "runs" / "adaptive_v2" / "results.jsonl"
    with pytest.raises(evaluation.EvaluationError, match="Invalid run id"):
        evaluation_artifact_path("results.jsonl", "../outside")


@pytest.mark.parametrize("with_stage_trace", [True, False])
def test_blind_review_preserves_scoring_definitions_and_cost_metrics(
    tmp_path: Path,
    with_stage_trace: bool,
) -> None:
    questions = small_questions()
    results_path = tmp_path / "results.jsonl"
    records = [
        {
            "question_id": question["id"],
            "variant": variant,
            "answer": (
                "Claim [A.pdf p.1]."
                if question["id"] == "Q001"
                else "The corpus lacks enough evidence."
            ),
            "evidence": [{"paper": "A.pdf", "page": 1}],
            "latency_seconds": 2.0 if variant == "adaptive" else 1.0,
            "llm_calls": 2,
            "error": None,
        }
        for question in questions
        for variant in evaluation.VARIANTS
    ]
    if with_stage_trace:
        for record in records:
            record["trace"] = {"retrieval_stages": {
                "retrieval": [{"paper": "A.pdf", "page": 1}],
                "rerank": [{"paper": "A.pdf", "page": 1}],
            }}
    results_path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    review_path = tmp_path / "review.csv"
    key_path = tmp_path / "review_key.json"
    prepare_review(questions, results_path, review_path, key_path)

    keys = {
        item["review_id"]: item
        for item in json.loads(key_path.read_text(encoding="utf-8"))
    }
    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "variant" not in rows[0]
    for row in rows:
        key = keys[row["review_id"]]
        passed = key["variant"] == "adaptive" or key["question_id"] == "Q002"
        row["requirement_scores"] = json.dumps({"G1": int(passed)})
        citation_count = len(json.loads(row["citations"]))
        row["citation_scores"] = json.dumps([int(passed)] * citation_count)
        row["unsupported_claims"] = "0" if passed else "1"
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

    assert summary["comparison"] == {
        "baseline": "fixed_hybrid",
        "treatment": "adaptive",
    }
    assert summary["variants"]["fixed_hybrid"]["strict_success_rate"] == 0.5
    assert summary["variants"]["fixed_hybrid"]["requirement_accuracy"] == 0.5
    assert summary["variants"]["fixed_hybrid"]["citation_support_rate"] == 0.0
    assert summary["variants"]["fixed_hybrid"]["average_latency_seconds"] == 1.0
    assert summary["variants"]["fixed_hybrid"]["average_llm_calls"] == 2.0
    assert summary["variants"]["adaptive"]["strict_success_rate"] == 1.0
    assert summary["variants"]["adaptive"]["requirement_accuracy"] == 1.0
    assert summary["variants"]["adaptive"]["citation_support_rate"] == 1.0
    assert summary["variants"]["adaptive"]["average_latency_seconds"] == 2.0
    assert summary["variants"]["adaptive"]["average_llm_calls"] == 2.0
    assert "| Metric | Fixed Hybrid | Adaptive Retrieval | Delta |" in (
        markdown_path.read_text(encoding="utf-8")
    )
    for variant in evaluation.VARIANTS:
        metrics = summary["variants"][variant]
        assert metrics["retrieval_recall"] == (1.0 if with_stage_trace else None)
        assert metrics["rerank_recall"] == (1.0 if with_stage_trace else None)
        assert metrics["retrieval_requirements"] == int(with_stage_trace)
        assert metrics["selected_evidence_recall"] == 1.0
        assert metrics["selected_evidence_requirements"] == 1
    assert summary["delta"]["retrieval_recall_percentage_points"] == (
        0.0 if with_stage_trace else None
    )
    detail = next(
        item for item in summary["requirement_metrics"]
        if item["question_id"] == "Q001" and item["variant"] == "fixed_hybrid"
    )
    assert detail["selected_evidence_recall"] == 1.0
    assert detail["answer_requirement_accuracy"] == 0
    expected_stages = "✓ | ✓" if with_stage_trace else "N/A | N/A"
    assert f"| Q001 | fixed_hybrid | G1 | {expected_stages} | ✓ | ✗ |" in (
        markdown_path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("retrieved", "candidates", "selected", "answer", "expected"),
    [
        (False, False, False, 0, [0.0, 0.0, 0.0, 0]),
        (True, False, False, 0, [1.0, 0.0, 0.0, 0]),
        (True, True, False, 0, [1.0, 1.0, 0.0, 0]),
        (True, True, True, 0, [1.0, 1.0, 1.0, 0]),
        (True, True, True, 1, [1.0, 1.0, 1.0, 1]),
    ],
)
def test_requirement_metrics_expose_the_stage_where_gold_pages_are_lost(
    retrieved: bool,
    candidates: bool,
    selected: bool,
    answer: int,
    expected: list[float],
) -> None:
    question = small_questions()[0]
    gold = question["requirements"][0]["gold_pages"]
    result = {
        "trace": {"retrieval_stages": {
            "retrieval": gold if retrieved else [],
            "rerank": gold if candidates else [],
        }},
        "evidence": gold if selected else [],
    }

    metric = requirement_stage_metrics(question, result, {"G1": answer})[0]

    assert [metric[f"{stage}_recall"] for stage in evaluation.RECALL_STAGES] + [
        metric["answer_requirement_accuracy"],
    ] == expected


def test_requirement_recall_deduplicates_pages_and_requires_matching_papers() -> None:
    question = small_questions()[0]
    gold = [{"paper": "A.pdf", "page": 1}, {"paper": "B.pdf", "page": 2}]
    question["requirements"][0]["gold_pages"] = [*gold, gold[0]]
    question["requirements"].append(small_questions()[1]["requirements"][0] | {"id": "G2"})
    result = {
        "trace": {"retrieval_stages": {
            "retrieval": [*gold, gold[0]],
            "rerank": [gold[0], gold[0], {"paper": "Wrong.pdf", "page": 2}],
        }},
        "evidence": [gold[0], gold[0]],
    }

    answerable, unanswerable = requirement_stage_metrics(question, result, {"G1": 0, "G2": 1})

    assert answerable["retrieval_recall"] == 1.0
    assert answerable["rerank_recall"] == 0.5
    assert answerable["selected_evidence_recall"] == 0.5
    assert all(unanswerable[f"{stage}_recall"] is None for stage in evaluation.RECALL_STAGES)
    assert unanswerable["answer_requirement_accuracy"] == 1


@pytest.mark.parametrize("version", ["adaptive_v2_shared_plan", None])
def test_resume_rejects_results_without_the_new_pipeline_version(
    tmp_path: Path,
    version: str | None,
) -> None:
    path = tmp_path / "results.jsonl"
    record = {"pipeline_version": version} if version else {}
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(evaluation.EvaluationError, match="different pipeline version"):
        evaluation._resumable_results(path, "legacy", evaluation.PIPELINE_VERSION)


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
