from __future__ import annotations

from typing import Any

import pytest

import scholar_agent.reranker
import scholar_agent.workflow as workflow_module
from scholar_agent.agents.writer import SAFE_ABSTENTION
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.workflow import build_workflow, initial_state, run_question


class FakeEngine:
    def __init__(self, results: list[dict], neighbors: list[dict] | None = None) -> None:
        self.results = results
        self.neighbors = results if neighbors is None else neighbors
        self.chunks = self.neighbors
        self.sparse_calls: list[tuple[list[str], int]] = []
        self.dense_calls: list[tuple[list[str], int]] = []

    def sparse_search(self, queries: list[str], top_k: int = 8) -> list[dict]:
        self.sparse_calls.append((queries, top_k))
        return self.results

    def dense_search_many(
        self,
        queries: list[str],
        top_k: int = 8,
    ) -> list[list[dict]]:
        self.dense_calls.append((queries, top_k))
        return [self.results for _ in queries]

    def search_within_paper(self, paper: str, query: str, top_k: int = 4) -> list[dict]:
        return [item for item in self.results if item["paper"] == paper][:top_k]

    def expand_neighbors(self, chunk_id: str, radius: int = 1) -> list[dict]:
        return self.neighbors


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        return [5.0] * len(pairs)


class FakeLLM:
    def __init__(self) -> None:
        self.json_calls = 0
        self.complete_calls = 0
        self.last_prompt = ""

    def complete_json(self, prompt: str) -> dict:
        self.json_calls += 1
        if "Evidence-Gap Controller" in prompt:
            return {
                "assessments": [{
                    "requirement_id": "R1",
                    "status": "sufficient",
                    "covered": ["The selected evidence covers the requirement"],
                    "missing": [],
                    "action": None,
                }],
            }
        return {
            "requirements": [
                {
                    "description": "Answer the requested evidence question",
                    "targets": [],
                    "query": "Self-RAG CRAG retrieval",
                    "retrieval_strategy": "bm25",
                    "top_k": 6,
                },
            ],
        }

    def complete(self, prompt: str) -> str:
        self.complete_calls += 1
        self.last_prompt = prompt
        return "Self-RAG uses adaptive retrieval [E1]. CRAG uses corrective retrieval [E2]."


class ControllerLLM(FakeLLM):
    def complete_json(self, prompt: str) -> dict:
        if "Evidence-Gap Controller" not in prompt:
            return super().complete_json(prompt)
        self.json_calls += 1
        return {
            "assessments": [{
                "requirement_id": "R1",
                "status": "missing",
                "covered": ["Self-RAG retrieval"],
                "missing": ["adjacent correction evidence"],
                "action": {
                    "tool": "expand_neighbors",
                    "chunk_id": "self-1",
                    "query": "Self-RAG and adjacent correction evidence",
                },
            }],
        }


class SimpleQuestionLLM(FakeLLM):
    def complete_json(self, prompt: str) -> dict:
        self.json_calls += 1
        if "Evidence-Gap Controller" in prompt:
            return {
                "assessments": [{
                    "requirement_id": "R1",
                    "status": "sufficient",
                    "covered": ["The selected evidence supports a direct answer"],
                    "missing": [],
                    "action": None,
                }],
            }
        return {
            "requirements": [
                {
                    "description": "Define agentic RAG briefly",
                    "targets": ["agentic rag"],
                    "query": "agentic rag",
                    "retrieval_strategy": "bm25",
                    "top_k": 4,
                },
            ],
        }


class ConversationLLM(FakeLLM):
    def complete_json(self, prompt: str) -> dict:
        self.json_calls += 1
        return {
            "route": "conversation",
            "direct_response": "Hello! What would you like to research?",
            "requirements": [],
        }


class StreamingQuestionLLM(SimpleQuestionLLM):
    def stream(self, prompt: str):
        self.last_prompt = prompt
        yield "Agentic RAG retrieves iteratively ["
        yield "E1]."


def test_adaptive_workflow_reaches_writer_and_validates_citations(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )

    llm = FakeLLM()
    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
    )

    assert engine.sparse_calls == [(["Self-RAG CRAG retrieval"], 6)]
    assert engine.dense_calls == []
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "[CRAG.pdf p.2]" in result["answer"]
    assert result["evidence_board"] == {
        "R1": {
            "requirement": "Answer the requested evidence question",
            "evidence_ids": ["E1", "E2"],
            "status": "sufficient",
            "covered": ["The selected evidence covers the requirement"],
            "missing": [],
            "action": None,
            "candidate_papers": [
                {
                    "paper": "CRAG.pdf",
                    "title": None,
                    "best_score": 5.0,
                    "selected": True,
                },
                {
                    "paper": "Self-RAG.pdf",
                    "title": None,
                    "best_score": 5.0,
                    "selected": True,
                },
            ],
        },
    }
    assert llm.json_calls == 2
    assert llm.complete_calls == 1
    assert result["evidence_board"]["R1"]["status"] == "sufficient"
    assert "Requirement R1:" in llm.last_prompt
    assert "[E1] Self-RAG.pdf — p.1" in llm.last_prompt
    assert [item["supports"] for item in result["evidence"]] == [["R1"], ["R1"]]
    assert result["retrieval_stages"] == {
        "retrieval": [
            {"paper": "CRAG.pdf", "page": 2},
            {"paper": "Self-RAG.pdf", "page": 1},
        ],
        "rerank": [
            {"paper": "CRAG.pdf", "page": 2},
            {"paper": "Self-RAG.pdf", "page": 1},
        ],
    }
    assert result["retrieval_trace"] == [
        {
            "requirement_id": "R1",
            "query": "Self-RAG CRAG retrieval",
            "retrieval_strategy": "bm25",
            "top_k": 6,
        },
    ]


def test_simple_definition_uses_the_planned_retrieval_and_controller_stops(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:1])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )
    llm = SimpleQuestionLLM()

    result = run_question(
        "do u know agentic rag",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
    )

    assert len(result["plan"]["requirements"]) == 1
    assert engine.sparse_calls == [(["agentic rag"], 4)]
    assert engine.dense_calls == []
    assert llm.json_calls == 2
    assert "keeping simple definitions, facts, and introductory questions brief" in llm.last_prompt
    assert result["controller_trace"]["actions"] == []
    assert result["evidence_board"]["R1"]["status"] == "sufficient"


def test_workflow_streams_the_writer_with_rendered_citations(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:1])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )
    llm = StreamingQuestionLLM()
    emitted: list[str] = []

    result = run_question(
        "do u know agentic rag",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
        writer_emit=emitted.append,
    )

    expected = "Agentic RAG retrieves iteratively [Self-RAG.pdf p.1]."
    assert "".join(emitted) == expected + "\n"
    assert result["answer"] == expected
    assert llm.complete_calls == 0


def test_conversation_route_skips_the_research_workflow() -> None:
    engine = FakeEngine([])
    llm = ConversationLLM()

    result = run_question(
        "hello",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
    )

    assert result["route"] == "conversation"
    assert result["answer"] == "Hello! What would you like to research?"
    assert result["plan"] == {"requirements": []}
    assert engine.sparse_calls == []
    assert engine.dense_calls == []
    assert llm.json_calls == 1
    assert llm.complete_calls == 0
    assert result["evidence"] == []
    assert result["controller_trace"]["actions"] == []


def test_controller_workflow_executes_one_recovery_round(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:1], sample_chunks[:2])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )
    llm = ControllerLLM()

    result = run_question(
        "Explain Self-RAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
    )

    assert llm.json_calls == 2
    assert llm.complete_calls == 1
    assert result["evidence_board"]["R1"]["status"] == "missing"
    assert result["evidence_board"]["R1"]["action"]["tool"] == "expand_neighbors"
    assert result["evidence_board"]["R1"]["action"]["state"] == "executed"
    assert result["controller_trace"]["actions"][0]["action"] == "expand_neighbors"
    assert len(result["recovery_trace"]) == 1
    assert result["recovery_trace"][0]["results"][1]["added"] is True
    assert [item["chunk_id"] for item in result["evidence"]] == ["self-1", "crag-1"]
    assert "Recovery: executed after this assessment" in llm.last_prompt
    assert "check only whether the recovered candidate passages resolve" in llm.last_prompt


def test_empty_evidence_produces_deterministic_abstention() -> None:
    engine = FakeEngine([])

    result = run_question(
        "Evidence that does not exist",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["evidence"] == []
    assert result["evidence_board"]["R1"]["evidence_ids"] == []
    assert result["retrieval_stages"] == {"retrieval": [], "rerank": []}
    assert result["answer"] == SAFE_ABSTENTION


def test_run_question_starts_with_the_selected_initial_state(monkeypatch: Any) -> None:
    captured: list[dict] = []

    class FakeWorkflow:
        def invoke(self, state: dict) -> dict:
            captured.append(state)
            return state

    monkeypatch.setattr(
        workflow_module,
        "build_workflow",
        lambda *args, **kwargs: FakeWorkflow(),
    )

    run_question(
        "question",
        FakeEngine([]),  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert captured == [initial_state("question")]


def test_initial_state_is_minimal_and_does_not_invent_requirements() -> None:
    state = initial_state("question")

    assert state == {
        "question": "question",
        "route": "research",
        "plan": {"requirements": []},
        "evidence": [],
        "evidence_board": {},
        "retrieval_trace": [],
        "controller_trace": {
            "actions": [],
            "rejected_actions": 0,
            "rejections": [],
        },
        "recovery_trace": [],
        "retrieval_stages": {},
        "answer": "",
    }


def test_workflow_requires_an_llm() -> None:
    with pytest.raises(ValueError, match="llm is required"):
        build_workflow(FakeEngine([]), Settings(), None)  # type: ignore[arg-type]


def test_agent_state_contains_only_live_workflow_fields() -> None:
    assert set(AgentState.__annotations__) == {
        "question",
        "route",
        "plan",
        "evidence",
        "evidence_board",
        "retrieval_trace",
        "controller_trace",
        "recovery_trace",
        "retrieval_stages",
        "answer",
    }
