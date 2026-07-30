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

import re
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
from scholar_agent.agents.writer import Writer, WriterLLMError
from scholar_agent.ids import new_run_id
from scholar_agent.logging import get_logger
from scholar_agent.models.answer import AnswerStatus, DraftAnswer, FinalAnswer
from scholar_agent.models.base import BudgetStatus, EventType, ExecutionEvent, QueryType, TokenUsage
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
    max_total_tokens: int = Field(default=100_000, ge=1)
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
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    terminated_reason: str
    events: list[ExecutionEvent] = Field(default_factory=list)
    unanswerable: bool = False
    answer_status: AnswerStatus = AnswerStatus.INSUFFICIENT
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
    estimated_tokens: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int
    token_budget_blocked: bool
    prev_evidence_ids: list[str]
    terminated_reason: str | None
    unanswerable: bool
    max_corrective_iterations: int
    max_total_tool_calls: int
    max_total_tokens: int
    max_latency_ms: int
    started_ms: float
    draft_answer: dict[str, Any] | None
    final_answer: dict[str, Any] | None
    answer_status: str


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
            "estimated_tokens": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "token_budget_blocked": False,
            "prev_evidence_ids": [],
            "terminated_reason": None,
            "unanswerable": False,
            "max_corrective_iterations": self.config.max_corrective_iterations,
            "max_total_tool_calls": self.config.max_total_tool_calls,
            "max_total_tokens": self.config.max_total_tokens,
            "max_latency_ms": self.config.max_latency_ms,
            "started_ms": perf_counter() * 1000,
            "draft_answer": None,
            "final_answer": None,
            "answer_status": AnswerStatus.INSUFFICIENT.value,
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
        runtime = _component_runtime(self.planner)
        usage = runtime["token_usage"]
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.PLAN_CREATED,
            component="planner",
            summary=(
                f"plan answer_type={plan.answer_type} sub_questions={len(plan.sub_questions)} "
                f"backend={runtime['backend']}"
            ),
            payload={
                "answer_type": plan.answer_type,
                "sub_question_ids": [sq.id for sq in plan.sub_questions],
                "sub_questions": [sq.question for sq in plan.sub_questions],
                "backend": runtime["backend"],
                "model": runtime["model"],
                "prompt_version": runtime["prompt_version"],
                "fallback_reason": runtime["fallback_reason"],
                "fallback_fields": runtime["fallback_fields"],
                "token_usage": usage.model_dump(mode="json"),
            },
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "events": _append_event_dicts(state, [event]),
            **_usage_state_updates(state, usage),
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
            new_ledger, tool_calls, estimated_tokens, events = self._research_targets(
                targets=targets,
                existing=existing,
                original_query=plan.original_query,
                run_id=state["run_id"],
                remaining_global=max(
                    0,
                    self.config.max_total_tool_calls - int(state.get("tool_call_count") or 0),
                ),
                remaining_tokens=max(
                    0,
                    self.config.max_total_tokens - int(state.get("estimated_tokens") or 0),
                ),
                deadline_ms=(
                    float(state.get("started_ms") or perf_counter() * 1000)
                    + int(state.get("max_latency_ms") or self.config.max_latency_ms)
                ),
            )
        else:
            # Targeted corrective retrieval from verifier queries
            tool_calls = 0
            estimated_tokens = 0
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
                remaining_token_budget = self.config.max_total_tokens - (
                    int(state.get("estimated_tokens") or 0) + estimated_tokens
                )
                if remaining_token_budget <= 0:
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
                        ),
                        "max_total_tokens_per_pass": min(
                            self.config.research.max_total_tokens_per_pass,
                            remaining_token_budget,
                        ),
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
                estimated_tokens += result.token_usage.total_tokens
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
                f"tokens+={estimated_tokens} unique_new_evidence={unique_new}"
            ),
            payload={
                "iteration": iteration,
                "tool_calls_delta": tool_calls,
                "estimated_tokens_delta": estimated_tokens,
                "unique_new_evidence": unique_new,
                "corrective": bool(corrective) and iteration > 0,
                # Ordered post-merge snapshot. Evaluation compares this first-pass
                # top-k against the final top-k; aggregating every raw tool hit
                # would give the initial side an unfairly larger candidate pool.
                "evidence_chunk_ids": [item.chunk_id for item in new_ledger.items],
                "evidence_paper_ids": [item.paper_id for item in new_ledger.items],
            },
        )
        all_new_events = events + [iter_event.model_dump(mode="json")]
        token_budget_blocked = any(
            event.get("event_type") == EventType.BUDGET_HIT.value
            and "token" in str(event.get("summary") or "").lower()
            for event in events
        )

        return {
            "evidence": [e.model_dump(mode="json") for e in new_ledger.items],
            "tool_call_count": int(state.get("tool_call_count") or 0) + tool_calls,
            "estimated_tokens": int(state.get("estimated_tokens") or 0) + estimated_tokens,
            "token_budget_blocked": bool(state.get("token_budget_blocked")) or token_budget_blocked,
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
        remaining_tokens: int,
        deadline_ms: float,
    ) -> tuple[EvidenceLedger, int, int, list[dict[str, Any]]]:
        """Run targets without oversubscribing global tool or token caps."""
        if (
            not targets
            or remaining_global <= 0
            or remaining_tokens <= 0
            or perf_counter() * 1000 >= deadline_ms
        ):
            return existing, 0, 0, []

        per_pass = self.config.research.max_tool_calls_per_pass
        worst_case_calls = len(targets) * per_pass
        if self.config.parallel_research and worst_case_calls <= remaining_global:
            per_target_tokens = min(
                self.config.research.max_total_tokens_per_pass,
                max(1, remaining_tokens // len(targets)),
            )
            parallel_agent = ResearchAgent(
                self.toolkit,
                config=self.config.research.model_copy(
                    update={"max_total_tokens_per_pass": per_target_tokens}
                ),
            )
            result = parallel_agent.research_many(
                targets,
                original_query=original_query,
                parallel=True,
                run_id=run_id,
            )
            return (
                existing.merge(result.evidence_ledger.items),
                result.tool_call_count,
                result.token_usage.total_tokens,
                [event.model_dump(mode="json") for event in result.events],
            )

        ledger = existing
        total_calls = 0
        total_tokens = 0
        events: list[dict[str, Any]] = []
        for target in targets:
            if perf_counter() * 1000 >= deadline_ms:
                break
            remaining = remaining_global - total_calls
            token_remaining = remaining_tokens - total_tokens
            if remaining <= 0 or token_remaining <= 0:
                break
            pass_config = self.config.research.model_copy(
                update={
                    "max_tool_calls_per_pass": min(per_pass, remaining),
                    "max_total_tokens_per_pass": min(
                        self.config.research.max_total_tokens_per_pass,
                        token_remaining,
                    ),
                }
            )
            pass_result = ResearchAgent(self.toolkit, config=pass_config).research_sub_question(
                target,
                run_id=run_id,
            )
            total_calls += pass_result.tool_call_count
            total_tokens += pass_result.token_usage.total_tokens
            ledger = ledger.merge(pass_result.evidence)
            events.extend(event.model_dump(mode="json") for event in pass_result.events)
        return ledger, total_calls, total_tokens, events

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
        estimated_tokens = int(state.get("estimated_tokens") or 0)
        max_iter_value = state.get("max_corrective_iterations")
        max_iter = int(max_iter_value if max_iter_value is not None else 3)
        max_tools_value = state.get("max_total_tool_calls")
        max_tools = int(max_tools_value if max_tools_value is not None else 20)
        max_tokens_value = state.get("max_total_tokens")
        max_tokens = int(max_tokens_value if max_tokens_value is not None else 100_000)
        max_latency_value = state.get("max_latency_ms")
        max_latency = int(max_latency_value if max_latency_value is not None else 180_000)
        elapsed_ms = self._elapsed_ms(state)
        termination_conditions = {
            "latency_budget_exhausted": elapsed_ms >= max_latency,
            "tool_budget_exhausted": tool_calls >= max_tools,
            "token_budget_exhausted": (
                estimated_tokens >= max_tokens
                or bool(state.get("token_budget_blocked"))
            ),
            "no_new_evidence": iteration > 0 and unique_new == 0,
            "iteration_budget_exhausted": iteration >= max_iter,
        }
        event = event.model_copy(
            update={
                "payload": {
                    **event.payload,
                    "termination_conditions": termination_conditions,
                }
            }
        )
        terminated: str | None = None

        if verification.is_sufficient:
            terminated = "evidence_sufficient"
        elif unanswerable:
            terminated = "corpus_cannot_answer"
        elif elapsed_ms >= max_latency:
            terminated = "latency_budget_exhausted"
        elif tool_calls >= max_tools:
            terminated = "tool_budget_exhausted"
        elif estimated_tokens >= max_tokens or bool(state.get("token_budget_blocked")):
            terminated = "token_budget_exhausted"
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
            "token_budget_exhausted",
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
                        "estimated_tokens": estimated_tokens,
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
        corpus_insufficient = bool(state.get("unanswerable")) or bool(
            verification and verification.unanswerable
        )
        max_tokens = int(
            state.get("max_total_tokens") or self.config.max_total_tokens
        )
        token_budget_exhausted = bool(state.get("token_budget_blocked")) or int(
            state.get("estimated_tokens") or 0
        ) >= max_tokens
        force_deterministic = (
            token_budget_exhausted and getattr(self.writer, "llm", None) is not None
        )
        if force_deterministic and bool(getattr(self.writer, "strict_llm", False)):
            # Strict mode may neither exceed the budget nor silently degrade.
            # This fixed message deliberately excludes provider response data.
            raise WriterLLMError(
                "LLM writer blocked because the global token budget is exhausted"
            )
        draft = self.writer.write(
            query=state["query"],
            plan=plan,
            ledger=ledger,
            verification=verification,
            # Partial coverage and confirmed corpus exhaustion are distinct.
            # Verification already controls complete/partial status.
            corpus_insufficient=corpus_insufficient,
            force_deterministic=force_deterministic,
            forced_fallback_reason=(
                "token_budget_exhausted" if force_deterministic else None
            ),
        )
        # Writer status captures answer completeness; this flag is reserved for
        # confirmed corpus exhaustion and is therefore owned by the workflow.
        draft = draft.model_copy(
            update={"corpus_insufficient": corpus_insufficient}
        )
        runtime = _component_runtime(self.writer)
        usage = runtime["token_usage"]
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.ANSWER_DRAFTED,
            component="writer",
            summary=(
                f"draft status={draft.status.value} claims={len(draft.claims)} "
                f"corpus_insufficient={draft.corpus_insufficient} backend={runtime['backend']}"
            ),
            payload={
                "claim_ids": [c.claim_id for c in draft.claims],
                "claim_evidence_ids": {c.claim_id: list(c.evidence_ids) for c in draft.claims},
                "answer_status": draft.status.value,
                "corpus_insufficient": draft.corpus_insufficient,
                "notes": list(draft.notes),
                "backend": runtime["backend"],
                "model": runtime["model"],
                "prompt_version": runtime["prompt_version"],
                "fallback_reason": runtime["fallback_reason"],
                "fallback_fields": runtime["fallback_fields"],
                "token_usage": usage.model_dump(mode="json"),
            },
        )
        return {
            "draft_answer": draft.model_dump(mode="json"),
            "answer_status": draft.status.value,
            "events": _append_event_dicts(state, [event]),
            **_usage_state_updates(state, usage),
        }

    def _node_validate_citations(self, state: WorkflowState) -> dict[str, Any]:
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
        draft_raw = state.get("draft_answer") or {}
        draft = DraftAnswer.model_validate(draft_raw) if draft_raw else DraftAnswer()
        final = self.citation_validator.validate(draft, ledger)
        original_terminated_reason = str(
            state.get("terminated_reason") or "completed"
        )
        (
            plan,
            verification,
            final,
            terminated_reason,
            reconciliation,
        ) = self._reconcile_citation_outcome(
            plan=plan,
            ledger=ledger,
            verification=verification,
            draft=draft,
            final=final,
            terminated_reason=original_terminated_reason,
            state=state,
        )
        report = final.citation_report
        event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.CITATION_VALIDATED,
            component="citation_validator",
            summary=(
                f"citation valid={report.is_valid if report else False} "
                f"status={final.status.value} claims={len(final.claims)} "
                f"sources={len(final.source_cards)}"
            ),
            payload={
                "is_valid": report.is_valid if report else False,
                "answer_status": final.status.value,
                "cited_evidence_ids": list(report.cited_evidence_ids) if report else [],
                "cited_paper_ids": list(report.cited_paper_ids) if report else [],
                "issue_count": len(report.issues) if report else 0,
                "issues": (
                    [i.model_dump(mode="json") for i in report.issues[:20]] if report else []
                ),
                "final_claim_ids": [c.claim_id for c in final.claims],
                "verification_is_sufficient": verification.is_sufficient,
                "terminated_reason": terminated_reason,
                "status_reconciled": reconciliation["status_reconciled"],
            },
        )
        new_events: list[ExecutionEvent] = [event]
        budget_reasons = {
            "tool_budget_exhausted",
            "iteration_budget_exhausted",
            "latency_budget_exhausted",
            "token_budget_exhausted",
        }
        if (
            reconciliation["termination_recovered"]
            and terminated_reason in budget_reasons
            and not any(
                existing.get("event_type") == EventType.BUDGET_HIT.value
                and existing.get("summary") == terminated_reason
                for existing in state.get("events") or []
            )
        ):
            new_events.append(
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.BUDGET_HIT,
                    component="workflow_finalizer",
                    summary=terminated_reason,
                    payload={
                        "iteration": int(state.get("iteration") or 0),
                        "tool_call_count": int(state.get("tool_call_count") or 0),
                        "estimated_tokens": int(state.get("estimated_tokens") or 0),
                        "reconciled_after_final_answer": True,
                    },
                )
            )
        if reconciliation["state_changed"]:
            new_events.append(
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.VERIFICATION,
                    component="workflow_finalizer",
                    summary=verification.rationale_summary,
                    payload={
                        "is_sufficient": verification.is_sufficient,
                        "coverage_score": verification.coverage_score,
                        "covered_sub_questions": verification.covered_sub_questions,
                        "missing_sub_questions": verification.missing_sub_questions,
                        "answer_status": final.status.value,
                        "terminated_reason": terminated_reason,
                        "citation_reconciled": True,
                    },
                )
            )
        return {
            "plan": plan.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "final_answer": final.model_dump(mode="json"),
            "answer_status": final.status.value,
            "terminated_reason": terminated_reason,
            "unanswerable": verification.unanswerable,
            "events": _append_event_dicts(state, new_events),
        }

    def _reconcile_citation_outcome(
        self,
        *,
        plan: QueryPlan,
        ledger: EvidenceLedger,
        verification: VerificationResult,
        draft: DraftAnswer,
        final: FinalAnswer,
        terminated_reason: str,
        state: WorkflowState,
    ) -> tuple[
        QueryPlan,
        VerificationResult,
        FinalAnswer,
        str,
        dict[str, bool],
    ]:
        """Make final citation-validated status authoritative everywhere."""
        evidence_by_id = {item.evidence_id: item for item in ledger.items}
        plan_sub_question_ids = {sub_question.id for sub_question in plan.sub_questions}

        def bound_sub_questions(claim: Any) -> set[str]:
            bound: set[str] = set()
            if claim.sub_question_id in plan_sub_question_ids:
                bound.add(claim.sub_question_id)
            for evidence_id in claim.evidence_ids:
                item = evidence_by_id.get(evidence_id)
                if item is not None and item.sub_question_id in plan_sub_question_ids:
                    bound.add(item.sub_question_id)
            return bound

        final_claims_by_sq: dict[str, list[Any]] = {
            sub_question_id: [] for sub_question_id in plan_sub_question_ids
        }
        final_evidence_by_sq: dict[str, list[str]] = {
            sub_question_id: [] for sub_question_id in plan_sub_question_ids
        }
        for claim in final.claims:
            explicitly_bound = (
                claim.sub_question_id
                if claim.sub_question_id in plan_sub_question_ids
                else None
            )
            for sub_question_id in bound_sub_questions(claim):
                final_claims_by_sq[sub_question_id].append(claim)
                for evidence_id in claim.evidence_ids:
                    item = evidence_by_id.get(evidence_id)
                    if (
                        item is not None
                        and (
                            sub_question_id == explicitly_bound
                            or item.sub_question_id == sub_question_id
                        )
                        and evidence_id not in final_evidence_by_sq[sub_question_id]
                    ):
                        final_evidence_by_sq[sub_question_id].append(evidence_id)

        final_claim_ids = {claim.claim_id for claim in final.claims}
        removed_claims = [
            claim for claim in draft.claims if claim.claim_id not in final_claim_ids
        ]
        invalidated_sub_questions = {
            sub_question_id
            for claim in removed_claims
            for sub_question_id in bound_sub_questions(claim)
        }

        covered: list[str] = []
        final_gap_aspects: list[str] = []
        structured_comparison = bool(
            plan.answer_type == "comparison"
            and plan.target_entities
            and plan.answer_requirements
        )
        if structured_comparison:
            all_entity_ids = {entity.id for entity in plan.target_entities}
            expected_matrix = {
                (requirement.key, entity_id)
                for requirement in plan.answer_requirements
                for entity_id in (
                    requirement.target_entity_ids
                    if requirement.target_entity_ids
                    else all_entity_ids
                )
            }
            claims_by_id = {claim.claim_id: claim for claim in final.claims}
            supported_matrix: set[tuple[str, str]] = set()
            for row in final.rows:
                for cell in row.cells:
                    cell_claim = claims_by_id.get(cell.claim_id or "")
                    pair = (row.requirement_key, cell.entity_id)
                    if (
                        pair in expected_matrix
                        and cell.supported
                        and cell_claim is not None
                        and cell_claim.entity_id == cell.entity_id
                        and cell_claim.requirement_key == row.requirement_key
                        and cell_claim.dimension == row.dimension
                        and cell_claim.evidence_ids
                    ):
                        supported_matrix.add(pair)

            missing_matrix = sorted(expected_matrix - supported_matrix)
            final_gap_aspects = [
                f"final_answer:{requirement_key}:{entity_id}"
                for requirement_key, entity_id in missing_matrix
            ]
            for sub_question in plan.sub_questions:
                expected_for_sub_question = {
                    (requirement_key, entity_id)
                    for requirement_key in sub_question.requirement_keys
                    for entity_id in sub_question.target_entity_ids
                    if (requirement_key, entity_id) in expected_matrix
                }
                if expected_for_sub_question:
                    sub_question_covered = (
                        expected_for_sub_question <= supported_matrix
                    )
                else:
                    # Legacy/unstructured sub-questions may coexist with the
                    # structured answer matrix. Preserve claim-based coverage
                    # for those questions without treating them as requirements.
                    sub_question_covered = bool(
                        final_claims_by_sq.get(sub_question.id)
                    )
                if (
                    sub_question_covered
                    and final.status != AnswerStatus.INSUFFICIENT
                ):
                    covered.append(sub_question.id)
            computed_coverage = len(supported_matrix) / max(1, len(expected_matrix))
        else:
            covered_units = 0
            total_units = 0
            for sub_question in plan.sub_questions:
                sub_question_covered = bool(
                    final_claims_by_sq.get(sub_question.id)
                )
                if sub_question.id in invalidated_sub_questions:
                    sub_question_covered = False
                covered_units += int(sub_question_covered)
                total_units += 1
                if (
                    sub_question_covered
                    and final.status != AnswerStatus.INSUFFICIENT
                ):
                    covered.append(sub_question.id)
            computed_coverage = covered_units / max(1, total_units)

        # Citation validation can only preserve or reduce earlier coverage.
        original_covered = set(verification.covered_sub_questions)
        if original_covered:
            covered = [
                sub_question_id
                for sub_question_id in covered
                if sub_question_id in original_covered
            ]
        original_missing = set(verification.missing_sub_questions)
        covered = [
            sub_question_id
            for sub_question_id in covered
            if sub_question_id not in original_missing
        ]
        missing = [
            sub_question.id
            for sub_question in plan.sub_questions
            if (
                sub_question.id not in covered
                or sub_question.id in original_missing
            )
        ]
        coverage_score = min(verification.coverage_score, computed_coverage)

        supported_evidence_ids: dict[str, list[str]] = {}
        for sub_question_id, evidence_ids in final_evidence_by_sq.items():
            originally_supported = verification.supported_evidence_ids.get(
                sub_question_id
            )
            retained = (
                [
                    evidence_id
                    for evidence_id in evidence_ids
                    if evidence_id in originally_supported
                ]
                if originally_supported is not None
                else list(evidence_ids)
            )
            if retained:
                supported_evidence_ids[sub_question_id] = retained

        is_sufficient = (
            verification.is_sufficient
            and final.status == AnswerStatus.COMPLETE
            and not missing
            and not verification.unanswerable
        )
        if final.status == AnswerStatus.COMPLETE and not is_sufficient:
            stricter_status = (
                AnswerStatus.PARTIAL
                if final.claims
                else AnswerStatus.INSUFFICIENT
            )
            final = self.citation_validator.restatus(
                final,
                status=stricter_status,
                ledger=ledger,
            )

        status_reconciled = final.status != draft.status
        missing_aspects: list[str] = []
        for aspect in [*final_gap_aspects, *verification.missing_aspects]:
            if aspect not in missing_aspects:
                missing_aspects.append(aspect)
        if final.status != AnswerStatus.COMPLETE and (
            verification.is_sufficient or status_reconciled
        ):
            marker = f"final_answer:{final.status.value}"
            if marker not in missing_aspects:
                missing_aspects.append(marker)
        unsupported_claims = list(verification.unsupported_claims)
        for claim in removed_claims:
            snippet = claim.text[:120]
            if snippet not in unsupported_claims:
                unsupported_claims.append(snippet)

        if not is_sufficient and (
            verification.is_sufficient or status_reconciled
        ):
            rationale = (
                "Final answer reconciliation downgraded answer sufficiency "
                f"(status={final.status.value}; coverage={coverage_score:.2f}; "
                f"missing={len(missing)})."
            )
        else:
            rationale = verification.rationale_summary

        reconciled_verification = verification.model_copy(
            update={
                "is_sufficient": is_sufficient,
                "coverage_score": round(max(0.0, min(1.0, coverage_score)), 3),
                "covered_sub_questions": covered,
                "supported_evidence_ids": supported_evidence_ids,
                "missing_sub_questions": missing,
                "unsupported_claims": unsupported_claims[:20],
                "missing_aspects": missing_aspects[:20],
                "rationale_summary": rationale,
            }
        )

        updated_sub_questions = [
            sub_question.model_copy(
                update={
                    "status": (
                        SubQuestionStatus.COVERED
                        if sub_question.id in covered
                        else SubQuestionStatus.MISSING
                    )
                }
            )
            for sub_question in plan.sub_questions
        ]
        reconciled_plan = plan.model_copy(
            update={"sub_questions": updated_sub_questions}
        )

        reconciled_reason = terminated_reason
        termination_recovered = False
        if terminated_reason == "evidence_sufficient" and not is_sufficient:
            reached_reason = self._reached_research_stop_reason(state)
            reconciled_reason = reached_reason or f"final_answer_{final.status.value}"
            termination_recovered = reached_reason is not None

        state_changed = (
            reconciled_verification != verification
            or reconciled_plan != plan
            or reconciled_reason != terminated_reason
            or status_reconciled
        )
        return (
            reconciled_plan,
            reconciled_verification,
            final,
            reconciled_reason,
            {
                "state_changed": state_changed,
                "status_reconciled": status_reconciled,
                "termination_recovered": termination_recovered,
            },
        )

    def _reached_research_stop_reason(
        self,
        state: WorkflowState,
    ) -> str | None:
        """Recover a stop condition masked by an optimistic verifier result."""
        precedence = (
            "latency_budget_exhausted",
            "tool_budget_exhausted",
            "token_budget_exhausted",
            "no_new_evidence",
            "iteration_budget_exhausted",
        )
        for raw_event in reversed(state.get("events") or []):
            if (
                raw_event.get("event_type") != EventType.VERIFICATION.value
                or raw_event.get("component") != "verifier"
            ):
                continue
            conditions = (raw_event.get("payload") or {}).get(
                "termination_conditions"
            )
            if isinstance(conditions, dict):
                for reason in precedence:
                    if conditions.get(reason) is True:
                        return reason
            break

        # Compatibility fallback for states/replays written before the verifier
        # began recording the exact condition snapshot. Counter-based limits are
        # stable after research; latency is intentionally omitted because time
        # spent writing must not be misattributed to the research stop reason.
        tool_calls = int(state.get("tool_call_count") or 0)
        max_tools = int(
            state.get("max_total_tool_calls") or self.config.max_total_tool_calls
        )
        if tool_calls >= max_tools:
            return "tool_budget_exhausted"
        estimated_tokens = int(state.get("estimated_tokens") or 0)
        max_tokens = int(
            state.get("max_total_tokens") or self.config.max_total_tokens
        )
        if (
            estimated_tokens >= max_tokens
            or bool(state.get("token_budget_blocked"))
        ):
            return "token_budget_exhausted"
        iteration = int(state.get("iteration") or 0)
        if iteration > 0 and self._last_unique_new(state) == 0:
            return "no_new_evidence"
        configured_iterations = state.get("max_corrective_iterations")
        max_iterations = int(
            configured_iterations
            if configured_iterations is not None
            else self.config.max_corrective_iterations
        )
        if iteration >= max_iterations:
            return "iteration_budget_exhausted"
        return None

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
                "estimated_tokens": state.get("estimated_tokens"),
                "latency_ms": latency,
                "unanswerable": state.get("unanswerable"),
                "answer_status": state.get("answer_status"),
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
        if bool(state.get("token_budget_blocked")) or int(
            state.get("estimated_tokens") or 0
        ) >= int(state.get("max_total_tokens") or 100_000):
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
        answer = final or draft
        answer_status = answer.status if answer is not None else AnswerStatus.INSUFFICIENT
        token_usage = _workflow_token_usage(state)
        corrective_limit = state.get("max_corrective_iterations")
        budget_status = BudgetStatus(
            tool_call_count=int(state.get("tool_call_count") or 0),
            max_tool_calls=int(
                state.get("max_total_tool_calls") or self.config.max_total_tool_calls
            ),
            iteration=int(state.get("iteration") or 0),
            max_iterations=int(
                corrective_limit
                if corrective_limit is not None
                else self.config.max_corrective_iterations
            ),
            token_usage=token_usage,
            max_total_tokens=int(state.get("max_total_tokens") or self.config.max_total_tokens),
            latency_ms=latency,
            max_latency_ms=int(state.get("max_latency_ms") or self.config.max_latency_ms),
            terminated_reason=str(state.get("terminated_reason") or "completed"),
        )
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
            token_usage=token_usage,
            latency_ms=latency,
            budgets=budget_status,
            execution_events=events,
            draft_answer=draft,
            final_answer=final,
            answer_status=answer_status,
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
            token_usage=token_usage,
            terminated_reason=str(state.get("terminated_reason") or "completed"),
            events=events,
            unanswerable=bool(state.get("unanswerable")),
            answer_status=answer_status,
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


_SAFE_TRACE_REASON = re.compile(r"^[A-Za-z0-9_. _-]{1,120}$")


def _component_runtime(component: Any) -> dict[str, Any]:
    """Return a secret-free trace view of one Planner/Writer invocation."""
    backend = str(getattr(component, "last_backend", "") or "").strip()
    if not backend:
        backend = "llm" if getattr(component, "llm", None) is not None else "deterministic"
    model = str(getattr(component, "last_model", "") or "").strip() or None
    prompt_version = (
        str(getattr(component, "last_prompt_version", "") or "").strip() or None
    )
    fallback_reason = _safe_trace_reason(
        getattr(component, "last_fallback_reason", None)
    )
    fallback_fields = _safe_trace_fields(
        getattr(component, "last_fallback_fields", ())
    )
    return {
        "backend": backend[:40],
        "model": model[:120] if model else None,
        "prompt_version": prompt_version[:120] if prompt_version else None,
        "fallback_reason": fallback_reason,
        "fallback_fields": fallback_fields,
        "token_usage": _component_token_usage(component),
    }


def _safe_trace_reason(reason: Any) -> str | None:
    """Keep reason codes, never raw provider responses or exception payloads."""
    if reason is None:
        return None
    cleaned = str(reason).strip()
    if not cleaned:
        return None
    if _SAFE_TRACE_REASON.fullmatch(cleaned):
        return cleaned
    return "component_fallback"


def _safe_trace_fields(fields: Any) -> list[str]:
    """Keep only bounded schema paths, never provider-authored values."""
    if not isinstance(fields, (list, tuple)):
        return []
    safe: list[str] = []
    for field_path in fields[:8]:
        cleaned = re.sub(r"[^A-Za-z0-9_.\[\]-]+", "_", str(field_path))[:160]
        if cleaned and cleaned not in safe:
            safe.append(cleaned)
    return safe


def _component_token_usage(component: Any) -> TokenUsage:
    raw = getattr(component, "last_token_usage", None)
    if raw is None:
        return TokenUsage()
    try:
        usage = raw if isinstance(raw, TokenUsage) else TokenUsage.model_validate(raw)
    except (TypeError, ValueError):
        return TokenUsage()
    total = usage.total_tokens or usage.prompt_tokens + usage.completion_tokens
    return usage.model_copy(update={"total_tokens": total})


def _usage_state_updates(state: WorkflowState, usage: TokenUsage) -> dict[str, int]:
    return {
        "estimated_tokens": int(state.get("estimated_tokens") or 0) + usage.total_tokens,
        "llm_prompt_tokens": int(state.get("llm_prompt_tokens") or 0) + usage.prompt_tokens,
        "llm_completion_tokens": (
            int(state.get("llm_completion_tokens") or 0) + usage.completion_tokens
        ),
        "llm_total_tokens": int(state.get("llm_total_tokens") or 0) + usage.total_tokens,
    }


def _workflow_token_usage(state: WorkflowState) -> TokenUsage:
    """Combine retrieval estimates with actual Planner/Writer usage."""
    total = int(state.get("estimated_tokens") or 0)
    llm_total = min(total, int(state.get("llm_total_tokens") or 0))
    retrieval_estimate = max(0, total - llm_total)
    completion = int(state.get("llm_completion_tokens") or 0)
    prompt = retrieval_estimate + int(state.get("llm_prompt_tokens") or 0)
    # Defensive normalization for provider usage objects that only report total.
    if prompt + completion < total:
        prompt += total - (prompt + completion)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
