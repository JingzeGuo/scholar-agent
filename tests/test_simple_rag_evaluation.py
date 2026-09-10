from __future__ import annotations

import json

from evals import evaluate as evaluation
from evals import evaluate_simple_rag as experiment

from scholar_agent.agents.researcher import _build_evidence_board
from scholar_agent.config import Settings
from scholar_agent.workflow import initial_state


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.sparse_top_k = None
        self.dense_top_k = None

    def sparse_search(self, queries: list[str], top_k: int = 8) -> list[dict]:
        self.sparse_top_k = top_k
        return self.chunks[:top_k]

    def dense_search_many(self, queries: list[str], top_k: int = 8) -> list[list[dict]]:
        self.dense_top_k = top_k
        return [self.chunks[-top_k:]]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Supported answer [E1]."


def fake_rerank(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
    return [
        {**item, "score": float(len(candidates) - index), "_query_scores": [1.0]}
        for index, item in enumerate(candidates)
    ]


def _controller_state(question: str, chunk: dict) -> tuple[dict, dict]:
    requirement = {
        "id": "R1",
        "description": question,
        "targets": [],
        "query": question,
        "retrieval_strategy": "bm25",
        "top_k": 8,
    }
    raw = [{**chunk, "_requirement_scores": {"R1": 5.0}}]
    evidence, board = _build_evidence_board(raw, [requirement], -1.0)
    state = initial_state(question, "adaptive", "controller")
    state.update(
        plan={"requirements": [requirement]},
        evidence=evidence,
        evidence_board=board,
        retrieval_trace=[{
            "requirement_id": "R1",
            "query": question,
            "retrieval_strategy": "bm25",
            "top_k": 8,
        }],
        retrieval_stages={
            "retrieval": [{"paper": chunk["paper"], "page": chunk["page"]}],
            "rerank": [{"paper": chunk["paper"], "page": chunk["page"]}],
        },
        controller_trace={
            "assessments": [{
                "requirement_id": "R1",
                "status": "sufficient",
                "covered": ["answer"],
                "missing": [],
                "action": None,
            }],
            "actions": [],
            "rejected_actions": 0,
            "rejections": [],
        },
    )
    metrics = {
        "planner_latency_seconds": 1.0,
        "researcher_latency_seconds": 2.0,
        "controller_latency_seconds": 3.0,
        "recovery_latency_seconds": 0.0,
        "pre_writer_latency_seconds": 6.0,
        "planner_llm_calls": 1,
        "controller_llm_calls": 1,
        "pre_writer_llm_calls": 2,
        "retrieval_operations": 1,
    }
    return state, metrics


def test_simple_rag_uses_original_question_hybrid_route_and_fixed_budget(sample_chunks):
    chunks = [
        {
            **sample_chunks[index % len(sample_chunks)],
            "chunk_id": f"chunk-{index}",
            "page": index + 1,
        }
        for index in range(16)
    ]
    engine = FakeEngine(chunks)
    state = experiment.simple_rag_state(
        "Original question?",
        engine,  # type: ignore[arg-type]
        Settings(),
        rerank_function=fake_rerank,
    )

    assert engine.sparse_top_k == engine.dense_top_k == 8
    assert state["plan"]["requirements"][0]["query"] == "Original question?"
    assert state["retrieval_trace"][0]["retrieval_strategy"] == "hybrid"
    assert len(state["evidence"]) == experiment.SIMPLE_EVIDENCE_LIMIT
    assert [item["id"] for item in state["evidence"]] == [f"E{i}" for i in range(1, 9)]
    assert state["evidence_board"]["R1"]["evidence_ids"] == [f"E{i}" for i in range(1, 9)]


def test_frozen_comparison_alternates_writer_order_and_counts_full_pipeline(
    tmp_path,
    sample_chunks,
    monkeypatch,
):
    questions = [
        {
            "id": f"Q{index:03d}",
            "question": f"Question {index}?",
            "category": "single",
            "expected_status": "complete",
            "requirements": [{
                "id": "G1",
                "description": "Answer it.",
                "answerable": True,
                "answer_key": ["answer"],
                "gold_pages": [{"paper": sample_chunks[0]["paper"], "page": 1}],
            }],
        }
        for index in range(1, 3)
    ]
    engine = FakeEngine(sample_chunks)
    monkeypatch.setattr(
        experiment,
        "_controller_state",
        lambda question, engine, settings, llm: _controller_state(question, sample_chunks[0]),
    )
    monkeypatch.setattr(
        experiment,
        "simple_rag_state",
        lambda question, engine, settings: _controller_state(question, sample_chunks[0])[0],
    )
    prepared_path = tmp_path / "prepared.jsonl"
    metadata_path = tmp_path / "metadata.json"
    experiment.prepare_inputs(
        questions,
        engine,  # type: ignore[arg-type]
        Settings(),
        evaluation.CountingLLM(FakeLLM()),
        prepared_path,
        metadata_path,
        run_id="simple_test",
    )
    prepared = evaluation._read_jsonl(prepared_path)
    assert len(prepared) == 2
    assert [item["preparation_order"] for item in prepared] == [
        ["simple_rag", "controller"],
        ["controller", "simple_rag"],
    ]
    assert all("answer_key" not in json.dumps(item) for item in prepared)

    delegate = FakeLLM()
    results_path = tmp_path / "results.jsonl"
    experiment.run_experiment(
        questions,
        prepared_path,
        metadata_path,
        results_path,
        evaluation.CountingLLM(delegate),
        run_id="simple_test",
    )
    results = evaluation._read_jsonl(results_path)
    assert [item["variant"] for item in results] == [
        "simple_rag",
        "controller",
        "controller",
        "simple_rag",
    ]
    assert [item["llm_calls"] for item in results] == [1, 3, 3, 1]
    assert [item["trace"]["scheduled_writer_order"] for item in results] == [
        ["simple_rag", "controller"],
        ["simple_rag", "controller"],
        ["controller", "simple_rag"],
        ["controller", "simple_rag"],
    ]
    assert all(item["latency_seconds"] >= (6.0 if item["variant"] == "controller" else 0.0)
               for item in results)
    assert all(item["error"] is None for item in results)


def test_parser_and_labels_define_controller_as_treatment():
    args = experiment._parser().parse_args(["--run-id", "simple_v1", "prepare"])
    assert args.command == "prepare"
    assert experiment.VARIANTS == ("simple_rag", "controller")
    assert evaluation.VARIANT_LABELS["simple_rag"] == "Simple RAG"
