from __future__ import annotations

import json

from evals import evaluate as evaluation
from evals import evaluate_controller as experiment

import scholar_agent.reranker
from scholar_agent.agents.researcher import _build_evidence_board
from scholar_agent.config import Settings


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    def expand_neighbors(self, chunk_id: str, radius: int = 1) -> list[dict]:
        return self.chunks


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        return [5.0] * len(pairs)


class FakeLLM:
    def __init__(self) -> None:
        self.controller_prompts = []
        self.writer_prompts = []

    def complete_json(self, prompt: str) -> dict:
        self.controller_prompts.append(prompt)
        return {"actions": [{
            "requirement_id": "R1",
            "action": "expand_neighbors",
            "chunk_id": "self-1",
            "query": "missing adjacent mechanism",
            "reason": "The mechanism may continue in the adjacent passage.",
        }]}

    def complete(self, prompt: str) -> str:
        self.writer_prompts.append(prompt)
        return "Recovered detail [E2]." if "[E2]" in prompt else "Initial detail [E1]."


def _question() -> dict:
    return {
        "id": "Q001",
        "category": "single",
        "question": "Explain the missing mechanism.",
        "expected_status": "complete",
        "requirements": [{
            "id": "G1",
            "description": "Explain the mechanism.",
            "answerable": True,
            "answer_key": ["GOLD_ANSWER_MUST_NOT_ENTER_PROMPTS"],
            "gold_pages": [{"paper": "CRAG.pdf", "page": 2}],
        }],
    }


def test_controller_experiment_freezes_observation_and_runs_one_round(
    tmp_path,
    sample_chunks,
    monkeypatch,
) -> None:
    question = _question()
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(json.dumps({
        "question_id": "Q001",
        "variant": "adaptive",
        "error": None,
        "answer": "OLD_ANSWER_MUST_NOT_ENTER_PROMPTS",
        "evidence": [sample_chunks[0]],
        "trace": {"plan": {"requirements": [{
            "id": "R1",
            "description": "Explain the mechanism.",
            "targets": [],
            "query": "mechanism",
            "retrieval_strategy": "bm25",
            "top_k": 8,
        }]}},
    }) + "\n")

    def researcher(state, engine, settings):
        raw = [{**sample_chunks[0], "_requirement_scores": {"R1": 5.0}}]
        evidence, board = _build_evidence_board(raw, state["plan"]["requirements"], -1.0)
        board["R1"]["candidate_papers"] = [
            {"paper": "Self-RAG.pdf", "title": "Self-RAG", "best_score": 5.0, "selected": True},
            {"paper": "CRAG.pdf", "title": "CRAG", "best_score": 2.0, "selected": False},
        ]
        return {
            "evidence": evidence,
            "evidence_board": board,
            "retrieval_trace": [{
                "requirement_id": "R1",
                "query": "mechanism",
                "retrieval_strategy": "bm25",
                "top_k": 8,
            }],
            "retrieval_stages": {
                "retrieval": [{"paper": "Self-RAG.pdf", "page": 1}],
                "rerank": [{"paper": "Self-RAG.pdf", "page": 1}],
            },
        }

    inputs_path = tmp_path / "inputs.json"
    results_path = tmp_path / "results.jsonl"
    inputs = experiment.prepare_inputs(
        [question],
        None,  # type: ignore[arg-type]
        Settings(),
        source_path,
        inputs_path,
        researcher_runner=researcher,
    )
    sample = inputs["samples"][0]
    assert "GOLD_ANSWER_MUST_NOT_ENTER_PROMPTS" not in sample["controller_prompt"]
    assert "OLD_ANSWER_MUST_NOT_ENTER_PROMPTS" not in sample["baseline_prompt"]
    assert "Observed candidate papers" in sample["controller_prompt"]

    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())
    delegate = FakeLLM()
    llm = evaluation.CountingLLM(delegate)
    experiment.run_experiment(
        inputs_path,
        results_path,
        FakeEngine(sample_chunks[:2]),  # type: ignore[arg-type]
        Settings(),
        llm,
        run_id="controller_test",
    )
    results = evaluation._read_jsonl(results_path)
    assert [item["variant"] for item in results] == ["baseline", "controller"]
    assert llm.calls == 3
    assert [item["chunk_id"] for item in results[0]["evidence"]] == ["self-1"]
    assert [item["chunk_id"] for item in results[1]["evidence"]] == ["self-1", "crag-1"]
    assert results[1]["trace"]["recovery_actions"][0]["results"][1]["added"] is True
    assert results[0]["trace"]["retrieval_operations"] == 1
    assert results[1]["trace"]["retrieval_operations"] == 2

    summary = {"requirement_metrics": [
        {"question_id": "Q001", "variant": "baseline", "requirement_id": "G1",
         "answer_requirement_accuracy": 0},
        {"question_id": "Q001", "variant": "controller", "requirement_id": "G1",
         "answer_requirement_accuracy": 1},
    ]}
    note = experiment.controller_summary(summary, [question], results_path)
    diagnostics = summary["controller_diagnostics"]
    assert diagnostics["retrieval_recovery_rate"] == 1
    assert diagnostics["useful_action_rate"] == 1
    assert diagnostics["answer_requirement_repairs"] == ["Q001/G1"]
    assert "Retrieval Recovery Rate" in note


def test_controller_experiment_parser_uses_full_benchmark() -> None:
    args = experiment._parser().parse_args(["--run-id", "controller_e3_v1", "prepare"])
    assert args.source_run == "adaptive_v2"
    assert args.source_variant == "adaptive"
    assert experiment.VARIANTS == ("baseline", "controller")
