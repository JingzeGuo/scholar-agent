"""Phase 0 LangGraph conditional loop prototype.

Demonstrates:
- typed graph state;
- a decision node with a deterministic fake model;
- a tool / retrieve node;
- a verify node with sufficiency checks;
- budget-aware conditional edges and clean termination.

No live LLM calls. Safe to run offline in CI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from scholar_agent.models import (
    EventType,
    ExecutionEvent,
    PrototypeDecision,
    PrototypeObservation,
    PrototypeResult,
    new_run_id,
)

# ---------------------------------------------------------------------------
# Deterministic "fake model" — replaces LLM decision making for the spike
# ---------------------------------------------------------------------------


class FakeResearchModel:
    """Rule-based stand-in that makes inspectable decisions.

    Policy:
    1. If no useful evidence yet → retrieve
    2. Else if evidence count < required → retrieve
    3. Else → verify (then finish if sufficient)
    """

    def __init__(self, required_evidence: int = 2) -> None:
        self.required_evidence = required_evidence

    def decide(
        self,
        *,
        query: str,
        observations: Sequence[PrototypeObservation],
        tool_call_count: int,
        max_tool_calls: int,
        iteration: int,
        max_iterations: int,
    ) -> PrototypeDecision:
        useful = [o for o in observations if o.is_useful]
        if tool_call_count >= max_tool_calls:
            return PrototypeDecision(
                action="verify",
                reason="tool budget exhausted; verify with current evidence",
                need_more_evidence=len(useful) < self.required_evidence,
            )
        if iteration >= max_iterations:
            return PrototypeDecision(
                action="verify",
                reason="iteration budget exhausted; verify with current evidence",
                need_more_evidence=len(useful) < self.required_evidence,
            )
        if len(useful) < self.required_evidence:
            return PrototypeDecision(
                action="retrieve",
                reason=(
                    f"need {self.required_evidence} useful observations, "
                    f"have {len(useful)}; retrieving"
                ),
                need_more_evidence=True,
            )
        return PrototypeDecision(
            action="verify",
            reason=f"collected {len(useful)} useful observations; verifying",
            need_more_evidence=False,
        )

    def retrieve(self, query: str, call_index: int) -> PrototypeObservation:
        """Emit deterministic fake retrieval results."""
        snippets = [
            (
                "dense_search",
                f"Chunk A discusses '{query}' with method Self-RAG and cites page 3.",
                0.91,
            ),
            (
                "sparse_search",
                f"Chunk B mentions corrective retrieval related to '{query}' on page 7.",
                0.84,
            ),
            (
                "hybrid_search",
                f"Chunk C compares agentic RAG approaches for '{query}' on pages 2-4.",
                0.88,
            ),
        ]
        tool_name, content, score = snippets[call_index % len(snippets)]
        return PrototypeObservation(
            tool_name=tool_name,
            content=content,
            score=score,
            is_useful=True,
        )


DecisionFn = Callable[..., PrototypeDecision]
RetrieveFn = Callable[[str, int], PrototypeObservation]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PrototypeState(TypedDict, total=False):
    run_id: str
    query: str
    iteration: int
    tool_call_count: int
    observations: list[dict[str, Any]]
    last_decision: dict[str, Any] | None
    is_sufficient: bool
    answer: str
    terminated_reason: str | None
    # Events are merged explicitly in nodes (append semantics).
    # LangGraph Annotated reducers remain available in agents.state for later phases.
    events: list[ExecutionEvent]
    max_tool_calls: int
    max_iterations: int
    required_evidence: int


class PrototypeLoopConfig(BaseModel):
    max_tool_calls: int = Field(default=4, ge=1)
    max_iterations: int = Field(default=3, ge=1)
    required_evidence: int = Field(default=2, ge=1)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _event(
    state: PrototypeState,
    event_type: EventType,
    component: str,
    summary: str,
    **payload: Any,
) -> ExecutionEvent:
    return ExecutionEvent(
        run_id=state["run_id"],
        event_type=event_type,
        component=component,
        summary=summary,
        payload=payload,
    )


def _append_events(
    state: PrototypeState,
    *new_events: ExecutionEvent,
) -> list[ExecutionEvent]:
    return list(state.get("events", [])) + list(new_events)


def make_decide_node(model: FakeResearchModel) -> Callable[[PrototypeState], PrototypeState]:
    def decide_node(state: PrototypeState) -> PrototypeState:
        observations = [
            PrototypeObservation.model_validate(item) for item in state.get("observations", [])
        ]
        decision = model.decide(
            query=state["query"],
            observations=observations,
            tool_call_count=state.get("tool_call_count", 0),
            max_tool_calls=state.get("max_tool_calls", 4),
            iteration=state.get("iteration", 0),
            max_iterations=state.get("max_iterations", 3),
        )
        event = _event(
            state,
            EventType.DECISION,
            "prototype.decide",
            f"decision={decision.action}: {decision.reason}",
            action=decision.action,
            reason=decision.reason,
        )
        return {
            "last_decision": decision.model_dump(),
            "iteration": state.get("iteration", 0) + 1,
            "events": _append_events(state, event),
        }

    return decide_node


def make_retrieve_node(model: FakeResearchModel) -> Callable[[PrototypeState], PrototypeState]:
    def retrieve_node(state: PrototypeState) -> PrototypeState:
        call_index = state.get("tool_call_count", 0)
        observation = model.retrieve(state["query"], call_index)
        obs_list = list(state.get("observations", []))
        obs_list.append(observation.model_dump())
        event = _event(
            state,
            EventType.TOOL_RESULT,
            "prototype.retrieve",
            f"retrieved via {observation.tool_name}",
            tool_name=observation.tool_name,
            score=observation.score,
            call_index=call_index,
        )
        return {
            "observations": obs_list,
            "tool_call_count": call_index + 1,
            "events": _append_events(state, event),
        }

    return retrieve_node


def verify_node(state: PrototypeState) -> PrototypeState:
    observations = [
        PrototypeObservation.model_validate(item) for item in state.get("observations", [])
    ]
    useful = [o for o in observations if o.is_useful]
    required = state.get("required_evidence", 2)
    is_sufficient = len(useful) >= required

    if is_sufficient:
        snippets = " | ".join(o.content for o in useful[:required])
        answer = f"Based on {len(useful)} evidence items for '{state['query']}': {snippets}"
        reason = "evidence_sufficient"
    elif state.get("tool_call_count", 0) >= state.get("max_tool_calls", 4):
        answer = (
            f"Insufficient evidence for '{state['query']}' after tool budget exhausted "
            f"({len(useful)}/{required} useful items)."
        )
        reason = "tool_budget_exhausted"
    elif state.get("iteration", 0) >= state.get("max_iterations", 3):
        answer = (
            f"Insufficient evidence for '{state['query']}' after iteration budget exhausted "
            f"({len(useful)}/{required} useful items)."
        )
        reason = "iteration_budget_exhausted"
    else:
        answer = f"Still need more evidence for '{state['query']}' ({len(useful)}/{required})."
        reason = "need_more_evidence"

    event = _event(
        state,
        EventType.VERIFICATION,
        "prototype.verify",
        f"sufficient={is_sufficient}; reason={reason}",
        is_sufficient=is_sufficient,
        useful_count=len(useful),
        required=required,
        reason=reason,
    )
    updates: PrototypeState = {
        "is_sufficient": is_sufficient,
        "answer": answer,
        "events": _append_events(state, event),
    }
    if is_sufficient or reason != "need_more_evidence":
        term = _event(
            state,
            EventType.TERMINATED,
            "prototype.verify",
            f"loop terminated: {reason}",
            reason=reason,
        )
        updates["terminated_reason"] = reason
        updates["events"] = _append_events(state, event, term)
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_decide(state: PrototypeState) -> Literal["retrieve", "verify", "finish"]:
    decision = state.get("last_decision") or {}
    action = decision.get("action", "finish")
    if action == "retrieve":
        if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 4):
            return "verify"
        return "retrieve"
    if action == "verify":
        return "verify"
    return "finish"


def route_after_verify(state: PrototypeState) -> Literal["decide", "finish"]:
    if state.get("is_sufficient"):
        return "finish"
    if state.get("terminated_reason"):
        return "finish"
    if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 4):
        return "finish"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "finish"
    return "decide"


def finish_node(state: PrototypeState) -> PrototypeState:
    reason = state.get("terminated_reason") or "completed"
    if state.get("terminated_reason"):
        return {}
    event = _event(
        state,
        EventType.TERMINATED,
        "prototype.finish",
        f"loop terminated: {reason}",
        reason=reason,
    )
    return {
        "terminated_reason": reason,
        "events": _append_events(state, event),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_prototype_graph(
    model: FakeResearchModel | None = None,
) -> Any:
    """Compile the conditional research loop graph."""
    fake = model or FakeResearchModel()
    graph: StateGraph[PrototypeState, None, PrototypeState, PrototypeState] = StateGraph(
        PrototypeState
    )
    # cast: LangGraph stubs are stricter than partial TypedDict node updates.
    graph.add_node("decide", cast(Any, make_decide_node(fake)))
    graph.add_node("retrieve", cast(Any, make_retrieve_node(fake)))
    graph.add_node("verify", cast(Any, verify_node))
    graph.add_node("finish", cast(Any, finish_node))

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "retrieve": "retrieve",
            "verify": "verify",
            "finish": "finish",
        },
    )
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "decide": "decide",
            "finish": "finish",
        },
    )
    graph.add_edge("finish", END)
    return graph.compile()


def run_prototype_loop(
    query: str,
    *,
    config: PrototypeLoopConfig | None = None,
    model: FakeResearchModel | None = None,
    run_id: str | None = None,
) -> PrototypeResult:
    """Execute the Phase 0 prototype loop and return a structured result."""
    cfg = config or PrototypeLoopConfig()
    fake = model or FakeResearchModel(required_evidence=cfg.required_evidence)
    app = build_prototype_graph(fake)
    rid = run_id or new_run_id()

    initial: PrototypeState = {
        "run_id": rid,
        "query": query,
        "iteration": 0,
        "tool_call_count": 0,
        "observations": [],
        "last_decision": None,
        "is_sufficient": False,
        "answer": "",
        "terminated_reason": None,
        "events": [
            ExecutionEvent(
                run_id=rid,
                event_type=EventType.RUN_STARTED,
                component="prototype",
                summary=f"prototype loop started for query: {query}",
            )
        ],
        "max_tool_calls": cfg.max_tool_calls,
        "max_iterations": cfg.max_iterations,
        "required_evidence": cfg.required_evidence,
    }

    final_state: PrototypeState = app.invoke(initial)
    events = list(final_state.get("events", []))
    events.append(
        ExecutionEvent(
            run_id=rid,
            event_type=EventType.RUN_FINISHED,
            component="prototype",
            summary="prototype loop finished",
            payload={
                "terminated_reason": final_state.get("terminated_reason"),
                "tool_call_count": final_state.get("tool_call_count", 0),
                "iteration": final_state.get("iteration", 0),
            },
        )
    )

    terminated = final_state.get("terminated_reason") or "completed"
    success = bool(final_state.get("is_sufficient"))
    return PrototypeResult(
        run_id=rid,
        query=query,
        answer=final_state.get("answer") or "",
        iterations=int(final_state.get("iteration", 0)),
        tool_call_count=int(final_state.get("tool_call_count", 0)),
        events=events,
        terminated_reason=terminated,
        success=success,
    )


def main() -> None:
    """CLI entry for ``python -m scholar_agent.agents.prototype_loop``."""
    import json
    import sys

    query = " ".join(sys.argv[1:]).strip() or "What is corrective RAG?"
    result = run_prototype_loop(query)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "query": result.query,
                "success": result.success,
                "answer": result.answer,
                "iterations": result.iterations,
                "tool_call_count": result.tool_call_count,
                "terminated_reason": result.terminated_reason,
                "event_types": [e.event_type.value for e in result.events],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
