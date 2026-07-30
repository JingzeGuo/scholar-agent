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


def route_after_verification(state: AgentState) -> str:
    if state["sufficient"]:
        return "writer"
    if state["retry_count"] >= 1:
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
    workflow.add_edge("researcher", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {"researcher": "researcher", "writer": "writer"},
    )
    workflow.add_edge("writer", END)
    return workflow.compile()


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "queries": [],
        "entities": [],
        "evidence": [],
        "sufficient": False,
        "feedback": "",
        "retry_count": 0,
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
        config={"recursion_limit": WORKFLOW_RECURSION_LIMIT},
    )
    return AgentState(**result)
