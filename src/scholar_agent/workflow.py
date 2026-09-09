"""The complete LangGraph workflow."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from langgraph.graph import END, StateGraph

from scholar_agent.agents.planner import planner_node
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
    *,
    retrieval_mode: str | None = None,
    shared_plan: dict | None = None,
) -> Any:
    if llm is None:
        raise ValueError("llm is required")
    mode = retrieval_mode or settings.retrieval_mode
    if mode not in {"fixed_hybrid", "adaptive"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    workflow = StateGraph(AgentState)
    workflow.add_node(
        "researcher",
        lambda state: researcher_node(state, engine, settings),
    )
    workflow.add_node("writer", lambda state: writer_node(state, llm))
    workflow.add_node("citation_validator", citation_validator_node)
    if shared_plan is None:
        workflow.add_node("planner", lambda state: planner_node(state, llm))
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")
    else:
        workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "citation_validator")
    workflow.add_edge("citation_validator", END)
    return workflow.compile()


def initial_state(question: str, retrieval_mode: str = "adaptive") -> AgentState:
    if retrieval_mode not in {"fixed_hybrid", "adaptive"}:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
    return {
        "question": question,
        "retrieval_mode": retrieval_mode,
        "plan": {
            "requirements": [],
        },
        "evidence": [],
        "evidence_board": {},
        "retrieval_trace": [],
        "controller_trace": {"actions": [], "rejected_actions": 0},
        "recovery_trace": [],
        "retrieval_stages": {},
        "answer": "",
    }


def run_question(
    question: str,
    engine: RetrievalEngine,
    settings: Settings,
    llm: LLMClient | None,
    *,
    retrieval_mode: str | None = None,
    shared_plan: dict | None = None,
) -> AgentState:
    mode = retrieval_mode or settings.retrieval_mode
    state = initial_state(question, mode)
    if shared_plan is not None:
        state["plan"] = deepcopy(shared_plan)
    result = build_workflow(
        engine,
        settings,
        llm,
        retrieval_mode=mode,
        shared_plan=shared_plan,
    ).invoke(state)
    return AgentState(**result)
