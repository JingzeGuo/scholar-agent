from __future__ import annotations

from typing import Any

import pytest

import scholar_agent.reranker
import scholar_agent.workflow as workflow_module
from scholar_agent.agents.writer import SAFE_ABSTENTION
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.workflow import (
    initial_state,
    route_after_answer_verification,
    route_after_verification,
    run_question,
)


class FakeEngine:
    def __init__(self, results: list[dict], chunks: list[dict] | None = None) -> None:
        self.results = results
        self.chunks = chunks or results
        self.calls = 0

    def sparse_search(self, queries: list[str]) -> list[dict]:
        self.calls += 1
        return self.results

    def dense_search_many(self, queries: list[str]) -> list[list[dict]]:
        return [self.results for _ in queries]


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        return [5.0] * len(pairs)


class FakeLLM:
    def complete_json(self, prompt: str) -> dict:
        if "<user_question>" in prompt:
            return {
                "requirements": [
                    {
                        "description": "Answer the requested evidence question",
                        "targets": [],
                        "query": "Self-RAG CRAG retrieval",
                    },
                ],
            }
        if "You verify a final answer" in prompt:
            requirements = {
                "R1": {"passed": True, "issue": ""},
            }
            if "R2:" in prompt:
                requirements["R2"] = {"passed": True, "issue": ""}
            return {
                "requirements": requirements,
                "citation_issues": [],
                "uncited_claims": [],
                "unsupported_claims": [],
                "incorrect_missing_claims": [],
                "repair_instructions": [],
            }
        covered = {"R1": ["E1"], "R2": ["E2"]} if "E2 [" in prompt else {"R1": ["E1"]}
        return {
            "covered": covered,
            "corrective_queries": [
                {
                    "requirement_id": "R2" if "R2:" in prompt else "R1",
                    "query": "Find CRAG retrieval evidence",
                },
            ],
        }

    def complete(self, prompt: str) -> str:
        if "Coverage status: complete" in prompt:
            return "Self-RAG uses adaptive retrieval [E1]. CRAG uses corrective retrieval [E2]."
        if "Coverage status: partial" in prompt:
            return "Self-RAG uses adaptive retrieval [E1]. Missing evidence: CRAG retrieval."
        return "The corpus does not contain sufficiently relevant evidence."


def _retrieval_plan(state: AgentState, llm: object) -> dict:
    return {
        "plan": {
            "requirements": [
                {
                    "id": "R1",
                    "description": "Explain Self-RAG retrieval",
                    "targets": ["Self-RAG"],
                    "query": "Self-RAG retrieval",
                },
                {
                    "id": "R2",
                    "description": "Explain CRAG retrieval",
                    "targets": ["CRAG"],
                    "query": "CRAG retrieval",
                },
            ],
        },
    }


def test_complete_evidence_reaches_writer(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())
    monkeypatch.setattr(workflow_module, "planner_node", _retrieval_plan)

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["verification"]["status"] == "complete"
    assert result["retry_count"] == 0
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "[CRAG.pdf p.2]" in result["answer"]


def test_no_relevant_evidence_retries_with_corrective_queries() -> None:
    engine = FakeEngine([])
    result = run_question(
        "Evidence that does not exist",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["verification"]["status"] == "insufficient"
    assert result["stop_reason"] == "no_new_evidence"
    assert result["retry_count"] == 1
    assert engine.calls == 2
    assert result["answer"] == SAFE_ABSTENTION


def test_partial_workflow_retries_exactly_once(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:1], sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())
    monkeypatch.setattr(workflow_module, "planner_node", _retrieval_plan)
    researcher_calls = 0
    verifier_calls = 0
    original_researcher = workflow_module.researcher_node
    original_verifier = workflow_module.verifier_node

    def counting_researcher(*args: Any, **kwargs: Any) -> dict:
        nonlocal researcher_calls
        researcher_calls += 1
        return original_researcher(*args, **kwargs)

    def counting_verifier(*args: Any, **kwargs: Any) -> dict:
        nonlocal verifier_calls
        verifier_calls += 1
        return original_verifier(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "researcher_node", counting_researcher)
    monkeypatch.setattr(workflow_module, "verifier_node", counting_verifier)

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["verification"]["status"] == "partial"
    assert result["retry_count"] == 1
    assert result["stop_reason"] == "no_new_evidence"
    assert researcher_calls == 2
    assert verifier_calls == 1
    assert "Missing evidence" in result["answer"]


def test_run_question_starts_with_initial_state(monkeypatch: Any) -> None:
    class CapturingWorkflow:
        state: AgentState | None = None

        def invoke(self, state: AgentState) -> AgentState:
            self.state = state
            return state

    compiled = CapturingWorkflow()
    monkeypatch.setattr(workflow_module, "build_workflow", lambda *args: compiled)

    result = run_question(
        "question",
        FakeEngine([]),  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result == initial_state("question")
    assert compiled.state == initial_state("question")


def test_initial_state_does_not_invent_a_requirement() -> None:
    state = initial_state("Compare two methods")

    assert state["plan"] == {
        "requirements": [],
    }
    assert state["verification"]["uncertain"] == {}


def test_workflow_requires_an_llm() -> None:
    with pytest.raises(ValueError, match="llm is required"):
        workflow_module.build_workflow(FakeEngine([]), Settings(), None)  # type: ignore[arg-type]


def test_verification_retry_limit_is_configurable() -> None:
    state = initial_state("question")
    state["verification"]["corrective_queries"] = [
        {"requirement_id": "R1", "query": "Find missing evidence"},
    ]
    state["retry_count"] = 1

    assert route_after_verification(state, Settings(max_retries=2)) == "researcher"

    state["retry_count"] = 2
    assert route_after_verification(state, Settings(max_retries=2)) == "writer"

    state["retry_count"] = 0
    state["verification"]["corrective_queries"] = []
    assert route_after_verification(state, Settings(max_retries=2)) == "writer"


def test_answer_repair_is_bounded_to_one_attempt(sample_chunks: list[dict]) -> None:
    state = initial_state("question")
    state["evidence"] = sample_chunks[:1]
    state["answer_verification"]["repair_required"] = True

    assert route_after_answer_verification(state) == "repair"

    state["repair_count"] = 1
    assert route_after_answer_verification(state) == "end"


def test_workflow_repairs_and_rechecks_the_answer_once(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    class RepairingLLM(FakeLLM):
        answer_checks = 0

        def complete_json(self, prompt: str) -> dict:
            if "You verify a final answer" not in prompt:
                return super().complete_json(prompt)
            self.answer_checks += 1
            issue = self.answer_checks == 1
            return {
                "requirements": {
                    "R1": {"passed": True, "issue": ""},
                    "R2": {"passed": True, "issue": ""},
                },
                "citation_issues": [],
                "uncited_claims": ["The answer has no citation."] if issue else [],
                "unsupported_claims": [],
                "incorrect_missing_claims": [],
                "repair_instructions": ["Add supporting citations."] if issue else [],
            }

        def complete(self, prompt: str) -> str:
            if "Repair an evidence-grounded" in prompt:
                return "Self-RAG uses retrieval [E1]. CRAG uses correction [E2]."
            return "Self-RAG uses retrieval. CRAG uses correction."

    llm = RepairingLLM()
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())
    monkeypatch.setattr(workflow_module, "planner_node", _retrieval_plan)

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
    )

    assert result["repair_count"] == 1
    assert result["answer_verification"]["passed"] is True
    assert llm.answer_checks == 2
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "[CRAG.pdf p.2]" in result["answer"]

def test_agent_state_has_answer_verification_fields() -> None:
    assert set(AgentState.__annotations__) == {
        "question",
        "plan",
        "evidence",
        "verification",
        "retry_count",
        "stop_reason",
        "answer_verification",
        "repair_count",
        "answer",
    }
