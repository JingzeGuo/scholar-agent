from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy

import pytest
from evals import evaluate as evaluation
from evals import evaluate_writer as experiment

from scholar_agent.agents.researcher import _build_evidence_board
from scholar_agent.agents.writer import SAFE_ABSTENTION
from scholar_agent.config import Settings


class FakeLLM:
    def __init__(self, fail_on_call: int | None = None):
        self.prompts = []
        self.fail_on_call = fail_on_call

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) == self.fail_on_call:
            raise RuntimeError("provider unavailable")
        return "Supported claim [E1]."


@pytest.fixture
def frozen_experiment(tmp_path, sample_chunks, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    directory = tmp_path / "evals" / "runs" / "writer_test"
    questions, records = [], []
    for index in range(3):
        answerable = index < 2
        page = {key: sample_chunks[index][key] for key in ("paper", "page")}
        question = {
            "id": f"Q{index + 1:03d}",
            "category": "single" if answerable else "insufficient",
            "question": f"Explain aspect {index + 1}.",
            "expected_status": "complete" if answerable else "insufficient",
            "requirements": [{
                "id": "G1", "description": f"Explain aspect {index + 1}.",
                "answerable": answerable,
                "gold_pages": [page] if answerable else [],
                "answer_key": ["GOLD_ANSWER_MUST_NOT_ENTER_WRITER"] if answerable else [],
            }],
        }
        questions.append(question)
        records.append({
            "question_id": question["id"], "variant": "adaptive", "error": None,
            "answer": "OLD_ANSWER_MUST_NOT_ENTER_WRITER",
            "evidence": [sample_chunks[index]] if answerable else [],
            "trace": {"plan": {"requirements": [{
                "id": "R1", "description": question["question"], "targets": [],
                "query": f"query {index + 1}", "retrieval_strategy": "hybrid", "top_k": 8,
            }]}},
        })
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    researcher_calls = []

    def researcher(state, engine, settings):
        researcher_calls.append(state["question"])
        source = next(r for q, r in zip(questions, records, strict=True) if q["question"] == state["question"])
        items = [
            {**item, "title": "A supplied source title", "section": "Method", "_requirement_scores": {"R1": 2.0}}
            for item in source["evidence"]
        ]
        evidence, board = _build_evidence_board(items, state["plan"]["requirements"], -1.0)
        pages = [{"paper": item["paper"], "page": item["page"]} for item in evidence]
        return {"evidence": evidence, "evidence_board": board,
                "retrieval_stages": {"retrieval": pages, "rerank": pages}}

    inputs = experiment.prepare_inputs(
        questions, None, Settings(), source_path, directory / "inputs.json",  # type: ignore[arg-type]
        researcher_runner=researcher,
    )
    return directory, inputs, source_path, researcher_calls


def test_prepare_freezes_identical_policy_sources_and_requirement_text(frozen_experiment):
    directory, inputs, source_path, researcher_calls = frozen_experiment
    assert len(researcher_calls) == 3
    assert not (directory / "results.jsonl").exists()
    for sample in inputs["samples"]:
        flat, board = (sample["prompts"][variant] for variant in experiment.VARIANTS)
        assert flat.split("Question:")[0] == board.split("Question:")[0]
        assert "explicitly state which remaining requirements lack" in flat
        assert "Supporting evidence:" not in flat
        assert "Requirement–Evidence Blackboard:" not in flat
        assert "Requirement–Evidence Blackboard:" in board
        for prompt in (flat, board):
            assert "GOLD_ANSWER_MUST_NOT_ENTER_WRITER" not in prompt
            assert "OLD_ANSWER_MUST_NOT_ENTER_WRITER" not in prompt
            assert "requirement_scores" not in prompt
            for requirement in sample["state"]["plan"]["requirements"]:
                assert requirement["description"] in prompt
            for item in sample["state"]["evidence"]:
                assert item["text"] in prompt
                assert item["title"] in prompt
                assert item["section"] in prompt
                assert f"[{item['id']}]" in prompt
    with pytest.raises(evaluation.EvaluationError, match="already exist"):
        experiment.prepare_inputs([], None, Settings(), source_path, directory / "inputs.json")


def test_prepare_rejects_different_source_evidence(frozen_experiment):
    directory, inputs, source_path, _ = frozen_experiment
    with pytest.raises(evaluation.EvaluationError, match="Selected evidence differs"):
        experiment.prepare_inputs(
            inputs["questions"], None, Settings(), source_path, directory / "different.json",  # type: ignore[arg-type]
            researcher_runner=lambda *args: {"evidence": []},
        )
    assert not (directory / "different.json").exists()


def test_run_alternates_and_resumes_frozen_prompts_without_retrieval(frozen_experiment, monkeypatch):
    directory, inputs, source_path, researcher_calls = frozen_experiment
    source_path.unlink()
    monkeypatch.setattr(experiment, "_writer_prompt", lambda *a, **kw: pytest.fail("must use frozen prompts"))
    delegate = FakeLLM()
    llm = evaluation.CountingLLM(delegate)
    for _ in range(2):
        experiment.run_experiment(directory / "inputs.json", directory / "results.jsonl", llm, run_id="writer_test")
    records = evaluation._read_jsonl(directory / "results.jsonl")
    assert [r["variant"] for r in records] == ["flat", "blackboard", "blackboard", "flat", "flat", "blackboard"]
    assert llm.calls == 4
    assert len(researcher_calls) == 3
    assert [r["llm_calls"] for r in records] == [1, 1, 1, 1, 0, 0]
    assert records[4]["answer"] == records[5]["answer"] == SAFE_ABSTENTION
    assert "[Self-RAG.pdf p.1]" in records[0]["answer"]
    assert records[0]["evidence"] == records[1]["evidence"]
    assert records[0]["trace"]["plan"] == records[1]["trace"]["plan"]
    assert records[0]["requirement_metrics"] == records[1]["requirement_metrics"]
    assert delegate.prompts == [
        inputs["samples"][0]["prompts"]["flat"], inputs["samples"][0]["prompts"]["blackboard"],
        inputs["samples"][1]["prompts"]["blackboard"], inputs["samples"][1]["prompts"]["flat"],
    ]
    assert len({r["input_sha256"] for r in records}) == 1
    assert records[0]["trace"]["latency_scope"] == "writer_and_citation_validation"


def test_resume_retries_failure_and_rejects_changed_inputs(frozen_experiment):
    directory, _, _, _ = frozen_experiment
    results_path = directory / "results.jsonl"
    inputs_path = directory / "inputs.json"
    with pytest.raises(evaluation.EvaluationError, match="Q001/blackboard"):
        experiment.run_experiment(inputs_path, results_path, evaluation.CountingLLM(FakeLLM(2)), run_id="writer_test")
    records = evaluation._read_jsonl(results_path)
    assert records[0]["error"] is None
    assert records[1]["error"] is not None
    llm = evaluation.CountingLLM(FakeLLM())
    experiment.run_experiment(inputs_path, results_path, llm, run_id="writer_test")
    assert llm.calls == 3
    assert len(evaluation._read_jsonl(results_path)) == 7
    with inputs_path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(evaluation.EvaluationError, match="Frozen inputs changed"):
        experiment.run_experiment(inputs_path, results_path, llm, run_id="writer_test")
    assert llm.calls == 3


def test_writer_experiment_blind_review_and_score_commands(frozen_experiment, monkeypatch, tmp_path, papers_dir):
    directory, _, _, _ = frozen_experiment
    monkeypatch.setattr(evaluation, "evaluation_llm", lambda: evaluation.CountingLLM(FakeLLM()))
    assert experiment.main(["--run-id", "writer_test", "run"]) == 0
    assert experiment.main(["--run-id", "writer_test", "prepare-review"]) == 0
    data_dir = tmp_path / "data"
    shutil.copytree(papers_dir, data_dir / "papers")
    monkeypatch.setenv("SCHOLAR_AGENT_DATA_DIR", str(data_dir))
    assert experiment.main(["--run-id", "writer_test", "extract-pages"]) == 0
    assert len(evaluation._read_jsonl(directory / "review_evidence.jsonl")) == 6
    keys = {item["review_id"]: item for item in json.loads((directory / "review_key.json").read_text())}
    with (directory / "review.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "variant" not in rows[0]
    for row in rows:
        key = keys[row["review_id"]]
        score = int(key["variant"] == "blackboard" or key["question_id"] != "Q002")
        row.update(requirement_scores=json.dumps({"G1": score}),
                   citation_scores=json.dumps([1] * len(json.loads(row["citations"]))),
                   unsupported_claims="0", uncited_claims="0")
    with (directory / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert experiment.main(["--run-id", "writer_test", "score"]) == 0
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["comparison"] == {"baseline": "flat", "treatment": "blackboard"}
    assert summary["variants"]["flat"]["requirement_accuracy"] == pytest.approx(2 / 3)
    assert summary["variants"]["blackboard"]["requirement_accuracy"] == 1
    assert summary["delta"]["retrieval_recall_percentage_points"] == 0
    assert "exclude frozen Planner/retrieval preparation" in (directory / "summary.md").read_text()
    changed = deepcopy(json.loads((directory / "inputs.json").read_text()))
    changed["samples"][0]["prompts"]["flat"] += " changed"
    (directory / "inputs.json").write_text(json.dumps(changed))
    assert experiment.main(["--run-id", "writer_test", "score"]) == 1


def test_writer_experiment_parser_keeps_source_and_treatment_separate():
    args = experiment._parser().parse_args(["--run-id", "board_ab", "prepare"])
    assert args.source_run == "adaptive_v2"
    assert args.source_variant == "adaptive"
    assert experiment.VARIANTS == ("flat", "blackboard")
