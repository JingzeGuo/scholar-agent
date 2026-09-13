"""The complete LangGraph workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from scholar_agent.agents.controller import board_recovery_actions, controller_node
from scholar_agent.agents.planner import planner_node
from scholar_agent.agents.recovery import recovery_node
from scholar_agent.agents.researcher import researcher_node
from scholar_agent.agents.writer import citation_validator_node, writer_node
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState
from scholar_agent.retrieval import RetrievalEngine


def build_workflow(
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None,
) -> Any:
    if llm is None:
        raise ValueError("llm is required")
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", lambda state: planner_node(state, llm))
    workflow.add_node(
        "researcher",
        lambda state: researcher_node(state, engine, settings),
    )
    workflow.add_node("controller", lambda state: controller_node(state, llm))
    workflow.add_node("recovery", lambda state: recovery_node(state, engine, settings))
    workflow.add_node("writer", lambda state: writer_node(state, llm))
    workflow.add_node("citation_validator", citation_validator_node)
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        lambda state: state["route"],
        {"research": "researcher", "conversation": END},
    )
    workflow.add_edge("researcher", "controller")
    workflow.add_conditional_edges(
        "controller",
        lambda state: "recovery" if board_recovery_actions(state) else "writer",
        {"recovery": "recovery", "writer": "writer"},
    )
    workflow.add_edge("recovery", "writer")
    workflow.add_edge("writer", "citation_validator")
    workflow.add_edge("citation_validator", END)
    return workflow.compile()


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "route": "research",
        "plan": {
            "requirements": [],
        },
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


def run_question(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None,
) -> AgentState:
    state = initial_state(question)
    result = build_workflow(engine, settings, llm).invoke(state)
    return AgentState(**result)
