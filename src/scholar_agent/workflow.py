"""The complete four-node LangGraph workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from scholar_agent.agents.planner import planner_node
from scholar_agent.agents.researcher import researcher_node
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import writer_node
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState
from scholar_agent.retrieval import RetrievalEngine

WORKFLOW_RECURSION_LIMIT = 10


def route_after_research(state: AgentState) -> str:
    return (
        "writer"
        if state["stop_reason"] in {"no_relevant_evidence", "no_new_evidence"}
        else "verifier"
    )


def route_after_verification(state: AgentState, settings: Settings) -> str:
    if state["verification"]["status"] == "complete":
        return "writer"
    if state["retry_count"] >= settings.max_retries:
        return "writer"
    return "researcher"


def build_workflow(
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None = None,
) -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", lambda state: planner_node(state, llm))
    workflow.add_node(
        "researcher",
        lambda state: researcher_node(state, engine, settings),
    )
    workflow.add_node("verifier", lambda state: verifier_node(state, llm))
    workflow.add_node("writer", lambda state: writer_node(state, llm))
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
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
    workflow.add_edge("writer", END)
    return workflow.compile()


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "plan": {
            "queries": [],
            "entities": [],
            "targets": [],
            "facets": [],
            "output_language": "English",
        },
        "evidence": [],
        "verification": {
            "status": "insufficient",
            "covered": {},
            "missing": [],
            "corrective_query": "",
        },
        "retry_count": 0,
        "stop_reason": "",
        "answer": "",
    }


def run_question(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None = None,
) -> AgentState:
    result = build_workflow(engine, settings, llm).invoke(
        initial_state(question),
        config={
            "recursion_limit": max(
                WORKFLOW_RECURSION_LIMIT,
                settings.max_retries * 2 + 4,
            ),
        },
    )
    return AgentState(**result)
