from __future__ import annotations

from typing import Any

from scholar_agent.agents.planner import planner_node
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import writer_node
from scholar_agent.workflow import initial_state


class StubLLM:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if isinstance(self.payload, Exception):
            raise self.payload
        assert "Question" in prompt
        return self.payload  # type: ignore[return-value]


def test_planner_caps_queries_and_entities() -> None:
    state = initial_state("Compare systems")
    payload = {
        "queries": ["q1", "q2", "q3", "q4"],
        "entities": ["a", "b", "c", "d", "e", "f"],
    }

    result = planner_node(state, StubLLM(payload))  # type: ignore[arg-type]

    assert result == {"queries": ["q1", "q2", "q3"], "entities": ["a", "b", "c", "d", "e"]}


def test_planner_json_failure_uses_original_question() -> None:
    state = initial_state("Original question")

    result = planner_node(state, StubLLM(ValueError("bad JSON")))  # type: ignore[arg-type]

    assert result == {"queries": ["Original question"], "entities": []}


def test_verifier_requires_two_papers_for_comparison(sample_chunks: list[dict]) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["evidence"] = [sample_chunks[0]]
    insufficient = verifier_node(state)
    state["evidence"] = sample_chunks[:2]
    sufficient = verifier_node(state)

    assert insufficient["sufficient"] is False
    assert sufficient["sufficient"] is True


def test_writer_expresses_uncertainty_and_uses_only_evidence(sample_chunks: list[dict]) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["evidence"] = sample_chunks[:1]

    result = writer_node(state)

    assert "limited" in result["answer"]
    assert "reflection tokens" in result["answer"]
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "retrieval evaluator" not in result["answer"]
