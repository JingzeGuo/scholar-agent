"""Full multi-agent research workflow (Phases 6–7).

LangGraph flow:
  START → plan → research → verify → (corrective research | write)
        → validate_citations → finish → END

Termination of the research loop when:
  - evidence is sufficient
  - corrective iteration budget exhausted
  - no new unique evidence in the last iteration
  - verifier marks corpus unanswerable
  - global tool-call budget exhausted

Writing always runs after the research loop stops (including unanswerable paths)
so limitations are stated from the ledger rather than model memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from scholar_agent.agents.citation_validator import CitationValidator
from scholar_agent.agents.planner import Planner
from scholar_agent.agents.researcher import ResearchAgent, ResearchAgentConfig
from scholar_agent.agents.verifier import Verifier
from scholar_agent.agents.writer import Writer
from scholar_agent.ids import new_run_id
from scholar_agent.logging import get_logger
from scholar_agent.models.answer import DraftAnswer, FinalAnswer
from scholar_agent.models.base import EventType, ExecutionEvent, QueryType
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import (
    CorrectiveQuery,
    ResearchRunState,
    VerificationResult,
)
from scholar_agent.retrieval.tools import RetrievalToolkit

logger = get_logger(__name__)


class WorkflowConfig(BaseModel):
    max_corrective_iterations: int = Field(default=3, ge=0)
    max_total_tool_calls: int = Field(default=20, ge=1)
    max_latency_ms: int = Field(default=180_000, ge=1)
    research: ResearchAgentConfig = Field(default_factory=ResearchAgentConfig)
    parallel_research: bool = True


class WorkflowResult(BaseModel):
    """User-facing outcome of the full plan→research→verify→write loop."""

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
    draft_answer: DraftAnswer | None = None
    final_answer: FinalAnswer | None = None
    state: ResearchRunState | None = None


class WorkflowState(TypedDict, total=False):
    run_id: str
    query: str
    plan: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    events: list[dict[str, Any]]
    verification: dict[str, Any] | None
    corrective_queries: list[str]
    corrective_actions: list[dict[str, Any]]
    iteration: int
    tool_call_count: int
    prev_evidence_ids: list[str]
    terminated_reason: str | None
    unanswerable: bool
    max_corrective_iterations: int
    max_total_tool_calls: int
    max_latency_ms: int
    started_ms: float
    draft_answer: dict[str, Any] | None
    final_answer: dict[str, Any] | None


@dataclass
class ResearchWorkflow:
    """Compose Planner + ResearchAgent + Verifier + Writer + CitationValidator."""

    toolkit: RetrievalToolkit
    config: WorkflowConfig = field(default_factory=WorkflowConfig)
    planner: Planner = field(default_factory=Planner)
    verifier: Verifier = field(default_factory=Verifier)
    writer: Writer = field(default_factory=Writer)
    citation_validator: CitationValidator = field(default_factory=CitationValidator)

    def __post_init__(self) -> None:
        self.researcher = ResearchAgent(self.toolkit, config=self.config.research)
        store = getattr(self.toolkit, "store", None)
        if self.citation_validator.provenance_store is None and store is not None:
            self.citation_validator.provenance_store = store
            self.citation_validator.require_pdf_provenance = True

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
            "corrective_actions": [],
            "iteration": 0,
            "tool_call_count": 0,
            "prev_evidence_ids": [],
            "terminated_reason": None,
            "unanswerable": False,
            "max_corrective_iterations": self.config.max_corrective_iterations,
            "max_total_tool_calls": self.config.max_total_tool_calls,
            "max_latency_ms": self.config.max_latency_ms,
            "started_ms": perf_counter() * 1000,
            "draft_answer": None,
            "final_answer": None,
        }
        final: WorkflowState = app.invoke(initial)
        return self._to_result(final)

    def build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", self._node_plan)
        graph.add_node("research", self._node_research)
        graph.add_node("verify", self._node_verify)
        graph.add_node("write", self._node_write)
        graph.add_node("validate_citations", self._node_validate_citations)
        graph.add_node("finish", self._node_finish)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "research")
        graph.add_edge("research", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"research": "research", "write": "write"},
        )
        graph.add_edge("write", "validate_citations")
        graph.add_edge("validate_citations", "finish")
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
                f"plan answer_type={plan.answer_type} sub_questions={len(plan.sub_questions)}"
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
        corrective = [
            CorrectiveQuery.model_validate(action)
            for action in state.get("corrective_actions") or []
        ]
        existing = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        prev_ids = set(state.get("prev_evidence_ids") or [])

        # First pass: research all sub-questions. Corrective passes: only missing.
        if iteration == 0 or not corrective:
            targets = list(plan.sub_questions)
            new_ledger, tool_calls, events = self._research_targets(
                targets=targets,
                existing=existing,
                original_query=plan.original_query,
                run_id=state["run_id"],
                remaining_global=max(
                    0,
                    self.config.max_total_tool_calls - int(state.get("tool_call_count") or 0),
                ),
                deadline_ms=(
                    float(state.get("started_ms") or perf_counter() * 1000)
                    + int(state.get("max_latency_ms") or self.config.max_latency_ms)
                ),
            )
        else:
            # Targeted corrective retrieval from verifier queries
            tool_calls = 0
            events = []
            new_ledger = existing
            for action in corrective:
                if self._elapsed_ms(state) >= int(
                    state.get("max_latency_ms") or self.config.max_latency_ms
                ):
                    break
                remaining_budget = self.config.max_total_tool_calls - (
                    int(state.get("tool_call_count") or 0) + tool_calls
                )
                if remaining_budget <= 0:
                    break
                sq = SubQuestion(
                    id=action.target_sub_question_id,
                    question=action.query,
                    query_type=QueryType.KEYWORD,
                    required_evidence=[action.missing_aspect],
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
                    missing_aspect=action.query,
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
            "prev_evidence_ids": sorted(new_ids),
            "iteration": iteration,
        }

    def _research_targets(
        self,
        *,
        targets: list[SubQuestion],
        existing: EvidenceLedger,
        original_query: str,
        run_id: str,
        remaining_global: int,
        deadline_ms: float,
    ) -> tuple[EvidenceLedger, int, list[dict[str, Any]]]:
        """Run initial targets without ever oversubscribing the global tool cap."""
        if not targets or remaining_global <= 0 or perf_counter() * 1000 >= deadline_ms:
            return existing, 0, []

        per_pass = self.config.research.max_tool_calls_per_pass
        worst_case_calls = len(targets) * per_pass
        if self.config.parallel_research and worst_case_calls <= remaining_global:
            result = self.researcher.research_many(
                targets,
                original_query=original_query,
                parallel=True,
                run_id=run_id,
            )
            return (
                existing.merge(result.evidence_ledger.items),
                result.tool_call_count,
                [event.model_dump(mode="json") for event in result.events],
            )

        ledger = existing
        total_calls = 0
        events: list[dict[str, Any]] = []
        for target in targets:
            if perf_counter() * 1000 >= deadline_ms:
                break
            remaining = remaining_global - total_calls
            if remaining <= 0:
                break
            pass_config = self.config.research.model_copy(
                update={"max_tool_calls_per_pass": min(per_pass, remaining)}
            )
            pass_result = ResearchAgent(self.toolkit, config=pass_config).research_sub_question(
                target,
                run_id=run_id,
            )
            total_calls += pass_result.tool_call_count
            ledger = ledger.merge(pass_result.evidence)
            events.extend(event.model_dump(mode="json") for event in pass_result.events)
        return ledger, total_calls, events

    def _node_verify(self, state: WorkflowState) -> dict[str, Any]:
        plan = QueryPlan.model_validate(state["plan"])
        ledger = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        verification = self.verifier.verify(query=state["query"], plan=plan, ledger=ledger)
        updated_plan = self.verifier.update_sub_question_status(plan, verification)

        # Detect no-new-evidence: compare ledger size growth via prev snapshot
        # After research, prev_evidence_ids is the post-research set; we store
        # pre-research count in iteration payload. Simpler: if iteration>0 and
        # corrective ran but unique new was 0 — research node already logged it.
        # We recompute vs the ids before this iteration by reading last ITERATION event.
        unique_new = self._last_unique_new(state)
        unanswerable = verification.unanswerable

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
                "corrective_actions": [
                    action.model_dump(mode="json") for action in verification.corrective_actions
                ],
                "conflicting_evidence_ids": verification.conflicting_evidence_ids,
                "unanswerable": unanswerable,
                "unique_new_evidence": unique_new,
            },
        )

        # Determine termination reason if we should stop
        iteration = int(state.get("iteration") or 0)
        tool_calls = int(state.get("tool_call_count") or 0)
        max_iter_value = state.get("max_corrective_iterations")
        max_iter = int(max_iter_value if max_iter_value is not None else 3)
        max_tools_value = state.get("max_total_tool_calls")
        max_tools = int(max_tools_value if max_tools_value is not None else 20)
        max_latency_value = state.get("max_latency_ms")
        max_latency = int(max_latency_value if max_latency_value is not None else 180_000)
        elapsed_ms = self._elapsed_ms(state)
        terminated: str | None = None

        if verification.is_sufficient:
            terminated = "evidence_sufficient"
        elif unanswerable:
            terminated = "corpus_cannot_answer"
        elif elapsed_ms >= max_latency:
            terminated = "latency_budget_exhausted"
        elif tool_calls >= max_tools:
            terminated = "tool_budget_exhausted"
        elif iteration > 0 and unique_new == 0:
            if not verification.covered_sub_questions:
                terminated = "corpus_cannot_answer"
                unanswerable = True
            else:
                terminated = "no_new_evidence"
        elif iteration >= max_iter:
            if not verification.covered_sub_questions and iteration > 0:
                terminated = "corpus_cannot_answer"
                unanswerable = True
            else:
                terminated = "iteration_budget_exhausted"
        elif not verification.corrective_actions and not verification.is_sufficient:
            terminated = "no_corrective_queries"

        if unanswerable and not verification.unanswerable:
            verification = verification.model_copy(
                update={
                    "unanswerable": True,
                    "missing_aspects": list(verification.missing_aspects)
                    + ["corpus_cannot_answer"],
                    "corrective_queries": [],
                    "corrective_actions": [],
                }
            )
            event = event.model_copy(
                update={
                    "payload": {
                        **event.payload,
                        "unanswerable": True,
                        "corrective_queries": [],
                        "corrective_actions": [],
                    }
                }
            )

        new_events = [event]
        if terminated in {
            "tool_budget_exhausted",
            "iteration_budget_exhausted",
            "latency_budget_exhausted",
        }:
            new_events.append(
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.BUDGET_HIT,
                    component="workflow",
                    summary=terminated,
                    payload={
                        "iteration": iteration,
                        "tool_call_count": tool_calls,
                        "elapsed_ms": elapsed_ms,
                    },
                )
            )
        updates: dict[str, Any] = {
            "plan": updated_plan.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "corrective_queries": list(verification.corrective_queries),
            "corrective_actions": [
                action.model_dump(mode="json") for action in verification.corrective_actions
            ],
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
                        f"{len(verification.corrective_actions)} queries"
                    ),
                    payload={
                        "actions": [
                            action.model_dump(mode="json")
                            for action in verification.corrective_actions
                        ]
                    },
                )
            )
        updates["events"] = _append_event_dicts(state, new_events)
        return updates

    def _node_write(self, state: WorkflowState) -> dict[str, Any]:
        plan = QueryPlan.model_validate(state["plan"])
        ledger = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        verification = None
        if state.get("verification"):
            verification = VerificationResult.model_validate(state["verification"])
        draft = self.writer.write(
            query=state["query"],
            plan=plan,
            ledger=ledger,
            verification=verification,
            corpus_insufficient=bool(state.get("unanswerable")),
        )
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.ANSWER_DRAFTED,
            component="writer",
            summary=(
                f"draft claims={len(draft.claims)} corpus_insufficient={draft.corpus_insufficient}"
            ),
            payload={
                "claim_ids": [c.claim_id for c in draft.claims],
                "claim_evidence_ids": {c.claim_id: list(c.evidence_ids) for c in draft.claims},
                "corpus_insufficient": draft.corpus_insufficient,
                "notes": list(draft.notes),
            },
        )
        return {
            "draft_answer": draft.model_dump(mode="json"),
            "events": _append_event_dicts(state, [event]),
        }

    def _node_validate_citations(self, state: WorkflowState) -> dict[str, Any]:
        ledger = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        draft_raw = state.get("draft_answer") or {}
        draft = DraftAnswer.model_validate(draft_raw) if draft_raw else DraftAnswer()
        final = self.citation_validator.validate(draft, ledger)
        report = final.citation_report
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.CITATION_VALIDATED,
            component="citation_validator",
            summary=(
                f"citation valid={report.is_valid if report else False} "
                f"claims={len(final.claims)} "
                f"sources={len(final.source_cards)}"
            ),
            payload={
                "is_valid": report.is_valid if report else False,
                "cited_evidence_ids": list(report.cited_evidence_ids) if report else [],
                "cited_paper_ids": list(report.cited_paper_ids) if report else [],
                "issue_count": len(report.issues) if report else 0,
                "issues": (
                    [i.model_dump(mode="json") for i in report.issues[:20]] if report else []
                ),
                "final_claim_ids": [c.claim_id for c in final.claims],
            },
        )
        return {
            "final_answer": final.model_dump(mode="json"),
            "events": _append_event_dicts(state, [event]),
        }

    def _node_finish(self, state: WorkflowState) -> dict[str, Any]:
        reason = state.get("terminated_reason") or "completed"
        started = float(state.get("started_ms") or perf_counter() * 1000)
        latency = int(perf_counter() * 1000 - started)
        final_raw = state.get("final_answer") or {}
        citation_valid = None
        if final_raw:
            report = final_raw.get("citation_report") or {}
            citation_valid = report.get("is_valid")
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
                "citation_valid": citation_valid,
                "has_final_answer": bool(final_raw),
            },
        )
        return {
            "terminated_reason": reason,
            "events": _append_event_dicts(state, [event]),
        }

    def _route_after_verify(self, state: WorkflowState) -> Literal["research", "write"]:
        if state.get("terminated_reason"):
            return "write"
        if state.get("unanswerable"):
            return "write"
        verification = state.get("verification") or {}
        if verification.get("is_sufficient"):
            return "write"
        # Continue corrective research if we still have queries and budget
        iteration = int(state.get("iteration") or 0)
        max_iter_value = state.get("max_corrective_iterations")
        max_iter = int(max_iter_value if max_iter_value is not None else 3)
        if iteration > max_iter:
            return "write"
        if not state.get("corrective_actions"):
            return "write"
        if int(state.get("tool_call_count") or 0) >= int(state.get("max_total_tool_calls") or 20):
            return "write"
        return "research"

    def _elapsed_ms(self, state: WorkflowState) -> int:
        started = float(state.get("started_ms") or perf_counter() * 1000)
        return max(0, int(perf_counter() * 1000 - started))

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
        draft: DraftAnswer | None = None
        if state.get("draft_answer"):
            draft = DraftAnswer.model_validate(state["draft_answer"])
        final: FinalAnswer | None = None
        if state.get("final_answer"):
            final = FinalAnswer.model_validate(state["final_answer"])
        snapshot = ResearchRunState(
            run_id=run_id,
            query=state["query"],
            plan=plan,
            active_sub_questions=[
                sq.id for sq in plan.sub_questions if sq.status != SubQuestionStatus.COVERED
            ],
            evidence_ledger=ledger,
            verification=verification,
            corrective_queries=list(state.get("corrective_queries") or []),
            iteration=int(state.get("iteration") or 0),
            tool_call_count=int(state.get("tool_call_count") or 0),
            latency_ms=latency,
            execution_events=events,
            draft_answer=draft,
            final_answer=final,
            citation_report=final.citation_report if final else None,
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
            draft_answer=draft,
            final_answer=final,
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
