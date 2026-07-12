"""Full multi-agent research workflow (Phase 6).

LangGraph flow:
  START → plan → research → verify → (corrective research | finish) → END

Termination when:
  - evidence is sufficient
  - corrective iteration budget exhausted
  - no new unique evidence in the last iteration
  - verifier marks corpus unanswerable
  - global tool-call budget exhausted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from scholar_agent.agents.planner import Planner
from scholar_agent.agents.researcher import ResearchAgent, ResearchAgentConfig
from scholar_agent.agents.verifier import Verifier
from scholar_agent.ids import make_sub_question_id, new_run_id, normalize_text
from scholar_agent.logging import get_logger
from scholar_agent.models.base import EventType, ExecutionEvent, QueryType
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import ResearchRunState, VerificationResult
from scholar_agent.retrieval.tools import RetrievalToolkit

logger = get_logger(__name__)


class WorkflowConfig(BaseModel):
    max_corrective_iterations: int = Field(default=3, ge=0)
    max_total_tool_calls: int = Field(default=20, ge=1)
    max_latency_ms: int = Field(default=180_000, ge=1)
    research: ResearchAgentConfig = Field(default_factory=ResearchAgentConfig)
    parallel_research: bool = True


class WorkflowResult(BaseModel):
    """User-facing outcome of the full plan→research→verify loop."""

    run_id: str
    query: str
    plan: QueryPlan
    evidence_ledger: EvidenceLedger
    verification: VerificationResult
    iteration: int
    tool_call_count: int
    latency_ms: int
    terminated_reason: str
    events: list[ExecutionEvent] = Field(default_factory=list)
    unanswerable: bool = False
    state: ResearchRunState | None = None


class WorkflowState(TypedDict, total=False):
    run_id: str
    query: str
    plan: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    events: list[dict[str, Any]]
    verification: dict[str, Any] | None
    corrective_queries: list[str]
    iteration: int
    tool_call_count: int
    prev_evidence_ids: list[str]
    terminated_reason: str | None
    unanswerable: bool
    max_corrective_iterations: int
    max_total_tool_calls: int
    started_ms: float


@dataclass
class ResearchWorkflow:
    """Compose Planner + ResearchAgent + Verifier into a corrective loop."""

    toolkit: RetrievalToolkit
    config: WorkflowConfig = field(default_factory=WorkflowConfig)
    planner: Planner = field(default_factory=Planner)
    verifier: Verifier = field(default_factory=Verifier)

    def __post_init__(self) -> None:
        self.researcher = ResearchAgent(self.toolkit, config=self.config.research)

    def run(self, query: str, *, run_id: str | None = None) -> WorkflowResult:
        """Execute the full workflow (LangGraph)."""
        app = self.build_graph()
        rid = run_id or new_run_id()
        initial: WorkflowState = {
            "run_id": rid,
            "query": query,
            "plan": None,
            "evidence": [],
            "events": [
                ExecutionEvent(
                    run_id=rid,
                    event_type=EventType.RUN_STARTED,
                    component="workflow",
                    summary=f"workflow start: {query[:120]}",
                ).model_dump(mode="json")
            ],
            "verification": None,
            "corrective_queries": [],
            "iteration": 0,
            "tool_call_count": 0,
            "prev_evidence_ids": [],
            "terminated_reason": None,
            "unanswerable": False,
            "max_corrective_iterations": self.config.max_corrective_iterations,
            "max_total_tool_calls": self.config.max_total_tool_calls,
            "started_ms": perf_counter() * 1000,
        }
        final: WorkflowState = app.invoke(initial)
        return self._to_result(final)

    def build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", self._node_plan)
        graph.add_node("research", self._node_research)
        graph.add_node("verify", self._node_verify)
        graph.add_node("finish", self._node_finish)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "research")
        graph.add_edge("research", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"research": "research", "finish": "finish"},
        )
        graph.add_edge("finish", END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _node_plan(self, state: WorkflowState) -> dict[str, Any]:
        plan = self.planner.plan(state["query"])
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.PLAN_CREATED,
            component="planner",
            summary=(
                f"plan answer_type={plan.answer_type} "
                f"sub_questions={len(plan.sub_questions)}"
            ),
            payload={
                "answer_type": plan.answer_type,
                "sub_question_ids": [sq.id for sq in plan.sub_questions],
                "sub_questions": [sq.question for sq in plan.sub_questions],
            },
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "events": _append_event_dicts(state, [event]),
        }

    def _node_research(self, state: WorkflowState) -> dict[str, Any]:
        plan = QueryPlan.model_validate(state["plan"])
        iteration = int(state.get("iteration") or 0)
        corrective = list(state.get("corrective_queries") or [])
        existing = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        prev_ids = set(state.get("prev_evidence_ids") or [])

        # First pass: research all sub-questions. Corrective passes: only missing.
        if iteration == 0 or not corrective:
            targets = list(plan.sub_questions)
            research_result = self.researcher.research_many(
                targets,
                original_query=plan.original_query,
                parallel=self.config.parallel_research,
                run_id=state["run_id"],
            )
            new_ledger = existing.merge(research_result.evidence_ledger.items)
            tool_calls = research_result.tool_call_count
            events = [e.model_dump(mode="json") for e in research_result.events]
        else:
            # Targeted corrective retrieval from verifier queries
            tool_calls = 0
            events = []
            new_ledger = existing
            for i, cq in enumerate(corrective):
                if tool_calls >= self.config.max_total_tool_calls:
                    break
                remaining_budget = self.config.max_total_tool_calls - (
                    int(state.get("tool_call_count") or 0) + tool_calls
                )
                if remaining_budget <= 0:
                    break
                sq = SubQuestion(
                    id=make_sub_question_id(
                        normalize_text(plan.original_query)[:32], cq, 1000 + iteration * 10 + i
                    ),
                    question=cq,
                    query_type=QueryType.KEYWORD,
                    required_evidence=["corrective evidence"],
                    status=SubQuestionStatus.MISSING,
                )
                # Clamp researcher tool budget to remaining global budget
                pass_cfg = self.config.research.model_copy(
                    update={
                        "max_tool_calls_per_pass": min(
                            self.config.research.max_tool_calls_per_pass,
                            remaining_budget,
                        )
                    }
                )
                agent = ResearchAgent(self.toolkit, config=pass_cfg)
                result = agent.research_sub_question(
                    sq,
                    run_id=state["run_id"],
                    corrective=True,
                    missing_aspect=cq,
                )
                tool_calls += result.tool_call_count
                new_ledger = new_ledger.merge(result.evidence)
                events.extend(e.model_dump(mode="json") for e in result.events)

        new_ids = {e.evidence_id for e in new_ledger.items}
        unique_new = len(new_ids - prev_ids)

        iter_event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.ITERATION,
            component="workflow",
            summary=(
                f"research iteration={iteration} tools+={tool_calls} "
                f"unique_new_evidence={unique_new}"
            ),
            payload={
                "iteration": iteration,
                "tool_calls_delta": tool_calls,
                "unique_new_evidence": unique_new,
                "corrective": bool(corrective) and iteration > 0,
            },
        )
        all_new_events = events + [iter_event.model_dump(mode="json")]

        return {
            "evidence": [e.model_dump(mode="json") for e in new_ledger.items],
            "tool_call_count": int(state.get("tool_call_count") or 0) + tool_calls,
            "events": _append_event_dicts(state, all_new_events),
            "prev_evidence_ids": list(new_ids),
            "iteration": iteration,
        }

    def _node_verify(self, state: WorkflowState) -> dict[str, Any]:
        plan = QueryPlan.model_validate(state["plan"])
        ledger = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        verification = self.verifier.verify(
            query=state["query"], plan=plan, ledger=ledger
        )
        updated_plan = self.verifier.update_sub_question_status(plan, verification)

        # Detect no-new-evidence: compare ledger size growth via prev snapshot
        # After research, prev_evidence_ids is the post-research set; we store
        # pre-research count in iteration payload. Simpler: if iteration>0 and
        # corrective ran but unique new was 0 — research node already logged it.
        # We recompute vs the ids before this iteration by reading last ITERATION event.
        unique_new = self._last_unique_new(state)
        unanswerable = "corpus_cannot_answer" in verification.missing_aspects

        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.VERIFICATION,
            component="verifier",
            summary=verification.rationale_summary,
            payload={
                "is_sufficient": verification.is_sufficient,
                "coverage_score": verification.coverage_score,
                "missing_sub_questions": verification.missing_sub_questions,
                "corrective_queries": verification.corrective_queries,
                "conflicting_evidence_ids": verification.conflicting_evidence_ids,
                "unanswerable": unanswerable,
                "unique_new_evidence": unique_new,
            },
        )

        # Determine termination reason if we should stop
        iteration = int(state.get("iteration") or 0)
        tool_calls = int(state.get("tool_call_count") or 0)
        max_iter = int(state.get("max_corrective_iterations") or 3)
        max_tools = int(state.get("max_total_tool_calls") or 20)
        terminated: str | None = None

        if verification.is_sufficient:
            terminated = "evidence_sufficient"
        elif unanswerable:
            terminated = "corpus_cannot_answer"
        elif tool_calls >= max_tools:
            terminated = "tool_budget_exhausted"
        elif iteration >= max_iter:
            terminated = "iteration_budget_exhausted"
        elif iteration > 0 and unique_new == 0:
            terminated = "no_new_evidence"
        elif not verification.corrective_queries and not verification.is_sufficient:
            terminated = "no_corrective_queries"

        new_events = [event]
        updates: dict[str, Any] = {
            "plan": updated_plan.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "corrective_queries": list(verification.corrective_queries),
            "unanswerable": unanswerable,
        }
        if terminated:
            updates["terminated_reason"] = terminated
        else:
            # Will loop: bump iteration for next corrective pass
            updates["iteration"] = iteration + 1
            new_events.append(
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.CORRECTIVE,
                    component="workflow",
                    summary=(
                        f"corrective iteration {iteration + 1}: "
                        f"{len(verification.corrective_queries)} queries"
                    ),
                    payload={"queries": verification.corrective_queries},
                )
            )
        updates["events"] = _append_event_dicts(state, new_events)
        return updates

    def _node_finish(self, state: WorkflowState) -> dict[str, Any]:
        reason = state.get("terminated_reason") or "completed"
        started = float(state.get("started_ms") or perf_counter() * 1000)
        latency = int(perf_counter() * 1000 - started)
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.RUN_FINISHED,
            component="workflow",
            summary=f"workflow finished: {reason}",
            payload={
                "terminated_reason": reason,
                "iteration": state.get("iteration"),
                "tool_call_count": state.get("tool_call_count"),
                "latency_ms": latency,
                "unanswerable": state.get("unanswerable"),
            },
        )
        return {
            "terminated_reason": reason,
            "events": _append_event_dicts(state, [event]),
        }

    def _route_after_verify(
        self, state: WorkflowState
    ) -> Literal["research", "finish"]:
        if state.get("terminated_reason"):
            return "finish"
        if state.get("unanswerable"):
            return "finish"
        verification = state.get("verification") or {}
        if verification.get("is_sufficient"):
            return "finish"
        # Continue corrective research if we still have queries and budget
        iteration = int(state.get("iteration") or 0)
        max_iter = int(state.get("max_corrective_iterations") or 3)
        if iteration > max_iter:
            return "finish"
        if not state.get("corrective_queries"):
            return "finish"
        if int(state.get("tool_call_count") or 0) >= int(
            state.get("max_total_tool_calls") or 20
        ):
            return "finish"
        return "research"

    def _last_unique_new(self, state: WorkflowState) -> int:
        for event in reversed(state.get("events") or []):
            if event.get("event_type") == EventType.ITERATION.value:
                payload = event.get("payload") or {}
                return int(payload.get("unique_new_evidence") or 0)
        # First iteration: treat all evidence as new
        return len(state.get("evidence") or [])

    def _to_result(self, state: WorkflowState) -> WorkflowResult:
        plan = QueryPlan.model_validate(state["plan"])
        ledger = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        verification = VerificationResult.model_validate(
            state.get("verification")
            or {
                "is_sufficient": False,
                "coverage_score": 0.0,
                "rationale_summary": "verification missing",
            }
        )
        events = [ExecutionEvent.model_validate(e) for e in state.get("events") or []]
        started = float(state.get("started_ms") or perf_counter() * 1000)
        latency = int(perf_counter() * 1000 - started)
        run_id = state["run_id"]
        snapshot = ResearchRunState(
            run_id=run_id,
            query=state["query"],
            plan=plan,
            active_sub_questions=[
                sq.id
                for sq in plan.sub_questions
                if sq.status != SubQuestionStatus.COVERED
            ],
            evidence_ledger=ledger,
            verification=verification,
            corrective_queries=list(state.get("corrective_queries") or []),
            iteration=int(state.get("iteration") or 0),
            tool_call_count=int(state.get("tool_call_count") or 0),
            latency_ms=latency,
            execution_events=events,
            errors=[],
        )
        return WorkflowResult(
            run_id=run_id,
            query=state["query"],
            plan=plan,
            evidence_ledger=ledger,
            verification=verification,
            iteration=int(state.get("iteration") or 0),
            tool_call_count=int(state.get("tool_call_count") or 0),
            latency_ms=latency,
            terminated_reason=str(state.get("terminated_reason") or "completed"),
            events=events,
            unanswerable=bool(state.get("unanswerable")),
            state=snapshot,
        )


def run_research_workflow(
    query: str,
    toolkit: RetrievalToolkit,
    *,
    config: WorkflowConfig | None = None,
) -> WorkflowResult:
    """Convenience entrypoint."""
    wf = ResearchWorkflow(toolkit, config=config or WorkflowConfig())
    return wf.run(query)


def _append_event_dicts(
    state: WorkflowState,
    new_events: list[ExecutionEvent] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Accumulate events (WorkflowState has no LangGraph reducer)."""
    existing = list(state.get("events") or [])
    for event in new_events:
        if isinstance(event, ExecutionEvent):
            existing.append(event.model_dump(mode="json"))
        else:
            existing.append(event)
    return existing
