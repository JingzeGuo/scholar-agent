"""The complete LangGraph workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from scholar_agent.agents.answer_verifier import answer_verifier_node
from scholar_agent.agents.planner import planner_node
from scholar_agent.agents.researcher import researcher_node
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import repair_writer_node, writer_node
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState
from scholar_agent.retrieval import RetrievalEngine


def route_after_research(state: AgentState) -> str:
    return "writer" if state["stop_reason"] == "no_new_evidence" else "verifier"


def route_after_verification(state: AgentState, settings: Settings) -> str:
    if (
        not state["verification"]["corrective_queries"]
        or state["retry_count"] >= settings.max_retries
    ):
        return "writer"
    return "researcher"


def route_after_answer_verification(state: AgentState) -> str:
    if (
        state["evidence"]
        and state["answer_verification"]["repair_required"]
        and state["repair_count"] == 0
    ):
        return "repair"
    return "end"


def build_workflow(
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None,
    *,
    coverage_mode: str = "none",
) -> Any:
    if llm is None:
        raise ValueError("llm is required")
    if coverage_mode not in {"none", "soft"}:
        raise ValueError(f"Unknown coverage mode: {coverage_mode}")
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", lambda state: planner_node(state, llm))
    workflow.add_node(
        "researcher",
        lambda state: researcher_node(state, engine, settings),
    )
    workflow.add_node("writer", lambda state: writer_node(state, llm))
    workflow.add_node(
        "answer_verifier",
        lambda state: answer_verifier_node(state, llm),
    )
    workflow.add_node("repair", lambda state: repair_writer_node(state, llm))
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    if coverage_mode == "soft":
        workflow.add_node("verifier", lambda state: verifier_node(state, llm))
        workflow.add_conditional_edges(
            "researcher",
            route_after_research,
            {"verifier": "verifier", "writer": "writer"},
        )
        workflow.add_conditional_edges(
            "verifier",
            lambda state: route_after_verification(state, settings),
            {"researcher": "researcher", "writer": "writer"},
        )
    else:
        workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "answer_verifier")
    workflow.add_conditional_edges(
        "answer_verifier",
        route_after_answer_verification,
        {"repair": "repair", "end": END},
    )
    workflow.add_edge("repair", "answer_verifier")
    return workflow.compile()


def initial_state(question: str, coverage_mode: str = "none") -> AgentState:
    return {
        "question": question,
        "coverage_mode": coverage_mode,
        "plan": {
            "requirements": [],
        },
        "evidence": [],
        "verification": {
            "status": "insufficient" if coverage_mode == "soft" else "not_run",
            "covered": {},
            "uncertain": {},
            "missing": [],
            "corrective_queries": [],
        },
        "retry_count": 0,
        "stop_reason": "",
        "answer_verification": {
            "passed": None,
            "repair_required": False,
            "requirements": {},
            "citation_issues": [],
            "uncited_claims": [],
            "unsupported_claims": [],
            "incorrect_missing_claims": [],
            "repair_instructions": [],
            "error": "",
        },
        "repair_count": 0,
        "answer": "",
    }


def run_question(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None,
    *,
    coverage_mode: str = "none",
) -> AgentState:
    result = build_workflow(
        engine,
        settings,
        llm,
        coverage_mode=coverage_mode,
    ).invoke(initial_state(question, coverage_mode))
    return AgentState(**result)
