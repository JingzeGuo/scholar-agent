"""Research Agent: adaptive tool loop with budgets and evidence ledger.

Phase 5 deliverable. Offline-deterministic by default (router is rule-based;
retrieval uses the existing toolkit). No private chain-of-thought is stored —
only structured ToolAction / ExecutionEvent records.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from scholar_agent.ids import make_evidence_id, new_run_id
from scholar_agent.logging import get_logger, sanitize_for_log
from scholar_agent.models.base import (
    ErrorCategory,
    EventType,
    ExecutionEvent,
    QueryType,
    StructuredError,
    TokenUsage,
)
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import SubQuestion, SubQuestionStatus
from scholar_agent.models.retrieval import RetrievalHit, RetrievalResult
from scholar_agent.models.routing import RetrievalPolicy, RoutingDecision, ToolAction
from scholar_agent.retrieval.router import (
    policy_to_tool_modes,
    recommend_policy,
)
from scholar_agent.retrieval.tools import RetrievalToolkit

logger = get_logger(__name__)


class ResearchAgentConfig(BaseModel):
    max_tool_calls_per_pass: int = Field(default=4, ge=1)
    max_iterations_per_pass: int = Field(default=4, ge=1)
    max_evidence_per_sub_question: int = Field(default=8, ge=1)
    max_parallel_sub_questions: int = Field(default=4, ge=1)
    max_latency_ms: int = Field(default=120_000, ge=1)
    max_total_tokens_per_pass: int = Field(default=100_000, ge=1)
    # When True, may run a secondary tool if first yields little evidence
    allow_policy_override: bool = True


class ResearchPassResult(BaseModel):
    """Outcome of researching one sub-question."""

    sub_question_id: str
    question: str
    routing: RoutingDecision
    actions: list[ToolAction] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_call_count: int = 0
    iteration_count: int = 0
    latency_ms: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    terminated_reason: str
    events: list[ExecutionEvent] = Field(default_factory=list)


class ResearchRunResult(BaseModel):
    """Aggregated multi-sub-question research result."""

    run_id: str
    original_query: str
    passes: list[ResearchPassResult] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger = Field(default_factory=EvidenceLedger)
    events: list[ExecutionEvent] = Field(default_factory=list)
    tool_call_count: int = 0
    iteration_count: int = 0
    latency_ms: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    parallel: bool = False


class _SubResearchState(TypedDict, total=False):
    run_id: str
    sub_question: dict[str, Any]
    routing: dict[str, Any]
    tool_call_count: int
    iteration: int
    max_tool_calls: int
    max_iterations: int
    max_evidence: int
    token_usage: int
    max_total_tokens: int
    evidence: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    events: list[dict[str, Any]]
    last_action: dict[str, Any] | None
    last_new_count: int
    terminated_reason: str | None
    modes_queue: list[str]
    allow_override: bool


@dataclass
class ResearchAgent:
    """Budgeted research tool loop over a RetrievalToolkit."""

    toolkit: RetrievalToolkit
    config: ResearchAgentConfig = field(default_factory=ResearchAgentConfig)

    def research_sub_question(
        self,
        sub_question: SubQuestion,
        *,
        run_id: str | None = None,
        corrective: bool = False,
        missing_aspect: str | None = None,
    ) -> ResearchPassResult:
        rid = run_id or new_run_id()
        started = perf_counter()
        has_graph = self.toolkit.graph is not None
        routing = recommend_policy(
            sub_question.question,
            query_type=sub_question.query_type,
            has_graph=has_graph,
            corrective=corrective,
            missing_aspect=missing_aspect,
        )
        modes = policy_to_tool_modes(routing.recommended_policy)
        # Reserve evidence slots across planned modes so multi-tool policies
        # (e.g. hybrid+graph) are not starved by the first tool filling the budget.
        per_tool_cap = max(
            1,
            self.config.max_evidence_per_sub_question // max(1, len(modes)),
        )

        events: list[ExecutionEvent] = [
            ExecutionEvent(
                run_id=rid,
                event_type=EventType.RUN_STARTED,
                component="researcher",
                summary=f"research start: {sub_question.id}",
                payload={
                    "question": sub_question.question,
                    "query_type": sub_question.query_type.value,
                    "recommended_policy": routing.recommended_policy.value,
                },
            ),
            ExecutionEvent(
                run_id=rid,
                event_type=EventType.DECISION,
                component="router",
                summary=routing.rationale,
                payload=routing.model_dump(mode="json"),
            ),
        ]

        ledger = EvidenceLedger()
        actions: list[ToolAction] = []
        tool_calls = 0
        iterations = 0
        estimated_tokens = 0
        terminated = "completed"
        modes_queue = list(modes)
        seen_modes: set[str] = set()

        # Primary recommended mode(s)
        while modes_queue:
            latency_ms = int((perf_counter() - started) * 1000)
            if latency_ms >= self.config.max_latency_ms:
                terminated = "latency_budget_exhausted"
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.BUDGET_HIT,
                        component="researcher",
                        summary="max research latency reached",
                        payload={
                            "max_latency_ms": self.config.max_latency_ms,
                            "latency_ms": latency_ms,
                        },
                    )
                )
                break
            query_tokens = estimate_text_tokens(sub_question.question)
            if estimated_tokens + query_tokens > self.config.max_total_tokens_per_pass:
                terminated = "token_budget_exhausted"
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.BUDGET_HIT,
                        component="researcher",
                        summary="max estimated research tokens reached",
                        payload={
                            "max_total_tokens": self.config.max_total_tokens_per_pass,
                            "estimated_tokens": estimated_tokens,
                        },
                    )
                )
                break
            if tool_calls >= self.config.max_tool_calls_per_pass:
                terminated = "tool_budget_exhausted"
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.BUDGET_HIT,
                        component="researcher",
                        summary="max tool calls per research pass reached",
                        payload={"max_tool_calls": self.config.max_tool_calls_per_pass},
                    )
                )
                break
            if iterations >= self.config.max_iterations_per_pass:
                terminated = "iteration_budget_exhausted"
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.BUDGET_HIT,
                        component="researcher",
                        summary="max research iterations reached",
                        payload={"max_iterations": self.config.max_iterations_per_pass},
                    )
                )
                break
            mode = modes_queue.pop(0)
            if mode in seen_modes:
                continue
            seen_modes.add(mode)
            iterations += 1
            if len(ledger.items) >= self.config.max_evidence_per_sub_question:
                terminated = "evidence_budget_reached"
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.BUDGET_HIT,
                        component="researcher",
                        summary="max evidence per sub-question reached",
                        payload={"max_evidence": self.config.max_evidence_per_sub_question},
                    )
                )
                break

            remaining = self.config.max_evidence_per_sub_question - len(ledger.items)
            action, result, new_items = self._execute_mode(
                run_id=rid,
                sub_question=sub_question,
                mode=mode,
                routing=routing,
                existing=ledger,
                max_new=min(per_tool_cap, remaining),
            )
            estimated_tokens += query_tokens
            candidate_count = len(new_items)
            new_items, evidence_tokens = fit_evidence_to_token_budget(
                new_items,
                used_tokens=estimated_tokens,
                max_total_tokens=self.config.max_total_tokens_per_pass,
            )
            estimated_tokens += evidence_tokens
            actions.append(action)
            tool_calls += 1
            events.append(
                ExecutionEvent(
                    run_id=rid,
                    event_type=EventType.TOOL_SELECTED,
                    component="researcher",
                    summary=f"tool={action.tool_name} policy={action.policy.value}",
                    payload=action.model_dump(mode="json"),
                )
            )
            events.append(
                ExecutionEvent(
                    run_id=rid,
                    event_type=EventType.TOOL_RESULT,
                    component="researcher",
                    summary=(
                        f"{action.tool_name} returned {len(result.hits)} hits; "
                        f"new_unique={len(new_items)}"
                    ),
                    payload={
                        "hits": [
                            {
                                "chunk_id": hit.chunk_id,
                                "paper_id": hit.paper_id,
                                "page_start": hit.page_start,
                                "page_end": hit.page_end,
                                "score": hit.score,
                                "rerank_score": hit.rerank_score,
                            }
                            for hit in result.hits
                        ],
                        "new_evidence_ids": [e.evidence_id for e in new_items],
                        "method": result.method,
                        "estimated_tokens": estimated_tokens,
                    },
                )
            )
            if result.debug.get("error"):
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.ERROR,
                        component="researcher",
                        summary=f"{action.tool_name} failed",
                        payload={"error_type": "retrieval_tool_error"},
                    )
                )
            if new_items:
                ledger = ledger.merge(new_items)
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.EVIDENCE_ADDED,
                        component="researcher",
                        summary=f"added {len(new_items)} evidence items",
                        payload={"count": len(new_items)},
                    )
                )
            if candidate_count > len(new_items):
                terminated = "token_budget_exhausted"
                events.append(
                    ExecutionEvent(
                        run_id=rid,
                        event_type=EventType.BUDGET_HIT,
                        component="researcher",
                        summary="evidence truncated at estimated token budget",
                        payload={
                            "max_total_tokens": self.config.max_total_tokens_per_pass,
                            "estimated_tokens": estimated_tokens,
                            "dropped_evidence": candidate_count - len(new_items),
                        },
                    )
                )
                break

            # Adaptive override: if first tool weak, try a complementary mode once
            if (
                self.config.allow_policy_override
                and len(ledger.items) < max(1, self.config.max_evidence_per_sub_question // 2)
                and tool_calls < self.config.max_tool_calls_per_pass
                and iterations < self.config.max_iterations_per_pass
                and not modes_queue
            ):
                alt = self._complementary_mode(routing.recommended_policy, seen_modes)
                if alt:
                    modes_queue.append(alt)
                    events.append(
                        ExecutionEvent(
                            run_id=rid,
                            event_type=EventType.DECISION,
                            component="researcher",
                            summary=f"override: weak coverage → try {alt}",
                            payload={
                                "recommended": routing.recommended_policy.value,
                                "override_mode": alt,
                            },
                        )
                    )

        # Cap evidence strictly
        if len(ledger.items) > self.config.max_evidence_per_sub_question:
            ledger = EvidenceLedger(items=ledger.items[: self.config.max_evidence_per_sub_question])

        latency_ms = int((perf_counter() - started) * 1000)
        events.append(
            ExecutionEvent(
                run_id=rid,
                event_type=EventType.TERMINATED,
                component="researcher",
                summary=f"research pass terminated: {terminated}",
                payload={
                    "tool_call_count": tool_calls,
                    "iteration_count": iterations,
                    "evidence_count": len(ledger.items),
                    "latency_ms": latency_ms,
                    "estimated_tokens": estimated_tokens,
                },
            )
        )

        return ResearchPassResult(
            sub_question_id=sub_question.id,
            question=sub_question.question,
            routing=routing,
            actions=actions,
            evidence=list(ledger.items),
            tool_call_count=tool_calls,
            iteration_count=iterations,
            latency_ms=latency_ms,
            token_usage=TokenUsage(
                prompt_tokens=estimated_tokens,
                total_tokens=estimated_tokens,
            ),
            terminated_reason=terminated,
            events=events,
        )

    def research_many(
        self,
        sub_questions: list[SubQuestion],
        *,
        original_query: str,
        parallel: bool = True,
        run_id: str | None = None,
    ) -> ResearchRunResult:
        """Research multiple sub-questions; merge evidence with deterministic reducer."""
        rid = run_id or new_run_id()
        actual_parallel = parallel and self._parallel_safe(sub_questions)
        events: list[ExecutionEvent] = [
            ExecutionEvent(
                run_id=rid,
                event_type=EventType.RUN_STARTED,
                component="researcher",
                summary=(
                    f"multi research start: n={len(sub_questions)} parallel={actual_parallel}"
                ),
                payload={
                    "original_query": original_query,
                    "parallel_requested": parallel,
                    "parallel_safe": actual_parallel,
                },
            )
        ]
        passes: list[ResearchPassResult] = []

        if actual_parallel:
            max_workers = min(self.config.max_parallel_sub_questions, len(sub_questions))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self.research_sub_question, sq, run_id=rid): sq
                    for sq in sub_questions
                }
                # Deterministic order by original sub-question sequence
                by_id: dict[str, ResearchPassResult] = {}
                for fut in as_completed(futures):
                    result = fut.result()
                    by_id[result.sub_question_id] = result
                passes = [by_id[sq.id] for sq in sub_questions if sq.id in by_id]
        else:
            for sq in sub_questions:
                passes.append(self.research_sub_question(sq, run_id=rid))

        ledger = EvidenceLedger()
        total_tools = 0
        total_iterations = 0
        total_latency_ms = 0
        total_token_usage = TokenUsage()
        for p in passes:
            ledger = ledger.merge(p.evidence)
            total_tools += p.tool_call_count
            total_iterations += p.iteration_count
            total_latency_ms += p.latency_ms
            total_token_usage = total_token_usage.add(p.token_usage)
            events.extend(p.events)

        events.append(
            ExecutionEvent(
                run_id=rid,
                event_type=EventType.RUN_FINISHED,
                component="researcher",
                summary=(
                    f"multi research finished: passes={len(passes)} "
                    f"evidence={len(ledger.items)} tools={total_tools}"
                ),
                payload={
                    "parallel": actual_parallel,
                    "iteration_count": total_iterations,
                    "latency_ms": total_latency_ms,
                    "estimated_tokens": total_token_usage.total_tokens,
                },
            )
        )
        return ResearchRunResult(
            run_id=rid,
            original_query=original_query,
            passes=passes,
            evidence_ledger=ledger,
            events=events,
            tool_call_count=total_tools,
            iteration_count=total_iterations,
            latency_ms=total_latency_ms,
            token_usage=total_token_usage,
            parallel=actual_parallel,
        )

    def research_query(
        self,
        query: str,
        *,
        query_type: QueryType | None = None,
        sub_question_id: str = "sq_0",
    ) -> ResearchRunResult:
        """Convenience: single sub-question research for a free-form query."""
        from scholar_agent.retrieval.router import classify_query_type

        qtype = query_type or classify_query_type(query)[0]
        sq = SubQuestion(
            id=sub_question_id,
            question=query,
            query_type=qtype,
            required_evidence=["supporting passages"],
            status=SubQuestionStatus.PENDING,
        )
        return self.research_many([sq], original_query=query, parallel=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_mode(
        self,
        *,
        run_id: str,
        sub_question: SubQuestion,
        mode: str,
        routing: RoutingDecision,
        existing: EvidenceLedger,
        max_new: int | None = None,
    ) -> tuple[ToolAction, RetrievalResult, list[EvidenceItem]]:
        policy = _mode_to_policy(mode)
        overridden = policy != routing.recommended_policy and mode not in policy_to_tool_modes(
            routing.recommended_policy
        )
        action = ToolAction(
            tool_name=f"{mode}_search",
            policy=policy,
            query=sub_question.question,
            recommended_policy=routing.recommended_policy,
            overridden=overridden,
            reason=(
                f"execute {mode} per router"
                if not overridden
                else f"override complementary tool {mode}"
            ),
        )
        result = self._call_toolkit(mode, sub_question.question)
        budget = (
            max_new
            if max_new is not None
            else self.config.max_evidence_per_sub_question - len(existing.items)
        )
        new_items = hits_to_evidence(
            result.hits,
            run_id=run_id,
            sub_question_id=sub_question.id,
            existing=existing,
            max_new=max(0, budget),
        )
        return action, result, new_items

    def _call_toolkit(self, mode: str, query: str) -> RetrievalResult:
        mode_lit: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "graph"]
        if mode in {"dense", "sparse", "hybrid", "hybrid_rerank", "graph"}:
            mode_lit = mode  # type: ignore[assignment]
        else:
            mode_lit = "hybrid_rerank"
        try:
            return self.toolkit.search(query, mode=mode_lit)
        except Exception as exc:
            # Keep the research loop alive, but never silently invent hits.
            # Surface category + fallback so degradation is observable.
            error = classify_tool_error(exc, component="researcher", operation=f"{mode}_search")
            logger.warning(
                "tool %s failed category=%s retryable=%s msg=%s",
                mode,
                error.category.value,
                error.retryable,
                sanitize_for_log(error.message),
            )
            method: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "graph"]
            method = (
                mode_lit
                if mode_lit
                in {
                    "dense",
                    "sparse",
                    "hybrid",
                    "hybrid_rerank",
                    "graph",
                }
                else "hybrid_rerank"
            )
            return RetrievalResult(
                query=query,
                method=method,
                hits=[],
                debug={
                    "error": error.message,
                    "error_category": error.category.value,
                    "retryable": error.retryable,
                    "fallback_used": "empty_hits",
                    "degraded": True,
                },
            )

    def _complementary_mode(
        self,
        recommended: RetrievalPolicy,
        seen: set[str],
    ) -> str | None:
        candidates: list[str]
        if recommended in {RetrievalPolicy.DENSE}:
            candidates = ["sparse", "hybrid_rerank"]
        elif recommended in {RetrievalPolicy.SPARSE, RetrievalPolicy.HYBRID}:
            candidates = ["dense", "hybrid_rerank"]
        elif recommended == RetrievalPolicy.GRAPH:
            candidates = ["hybrid_rerank", "dense"]
        elif recommended == RetrievalPolicy.HYBRID_PLUS_GRAPH:
            candidates = ["sparse"]
        else:
            candidates = ["graph", "sparse"] if self.toolkit.graph else ["sparse", "dense"]
        for c in candidates:
            if c not in seen:
                if c == "graph" and self.toolkit.graph is None:
                    continue
                return c
        return None

    def _parallel_safe(self, sub_questions: list[SubQuestion]) -> bool:
        """Parallelize only independent-looking, uniquely keyed pending work."""
        if len(sub_questions) < 2 or self.config.max_parallel_sub_questions < 2:
            return False
        ids = [question.id for question in sub_questions]
        normalized = [" ".join(question.question.lower().split()) for question in sub_questions]
        return (
            len(ids) == len(set(ids))
            and len(normalized) == len(set(normalized))
            and all(question.status == SubQuestionStatus.PENDING for question in sub_questions)
        )


def classify_tool_error(
    exc: Exception,
    *,
    component: str,
    operation: str,
    run_id: str | None = None,
) -> StructuredError:
    """Map a toolkit exception into a sanitized StructuredError."""
    message = sanitize_for_log(str(exc) or type(exc).__name__)
    text = message.lower()
    if isinstance(exc, (TimeoutError,)):
        category = ErrorCategory.TIMEOUT
        retryable = True
    elif "not loaded" in text or "unavailable" in text or "not found" in text:
        category = ErrorCategory.INDEX_UNAVAILABLE
        retryable = False
    elif isinstance(exc, (ValueError, TypeError)):
        category = ErrorCategory.VALIDATION
        retryable = False
    else:
        category = ErrorCategory.TOOL
        retryable = False
    return StructuredError(
        run_id=run_id,
        component=component,
        operation=operation,
        category=category,
        retryable=retryable,
        message=message,
        fallback_used="empty_hits",
        details={"exception_type": type(exc).__name__},
    )


def hits_to_evidence(
    hits: list[RetrievalHit],
    *,
    run_id: str,
    sub_question_id: str,
    existing: EvidenceLedger | None = None,
    max_new: int = 8,
) -> list[EvidenceItem]:
    """Convert retrieval hits into EvidenceItems; drop duplicates vs existing ledger."""
    if max_new <= 0:
        return []
    base = existing or EvidenceLedger()
    existing_keys = {item.dedupe_key() for item in base.items}
    existing_ids = {item.evidence_id for item in base.items}
    out: list[EvidenceItem] = []
    for hit in hits:
        if len(out) >= max_new:
            break
        # Use a short claim from the first sentence / snippet
        claim = hit.snippet(160)
        item = EvidenceItem(
            evidence_id=make_evidence_id(
                run_id=run_id,
                chunk_id=hit.chunk_id,
                evidence_text=hit.text,
                sub_question_id=sub_question_id,
            ),
            sub_question_id=sub_question_id,
            claim=claim,
            evidence_text=hit.text,
            paper_id=hit.paper_id,
            chunk_id=hit.chunk_id,
            page_start=hit.page_start,
            page_end=hit.page_end,
            retrieval_method=hit.retrieval_method,
            retrieval_score=hit.score,
            rerank_score=hit.rerank_score,
            support_score=hit.rerank_score if hit.rerank_score is not None else hit.score,
        )
        if item.dedupe_key() in existing_keys or item.evidence_id in existing_ids:
            continue
        existing_keys.add(item.dedupe_key())
        existing_ids.add(item.evidence_id)
        out.append(item)
    return out


def estimate_text_tokens(text: str) -> int:
    """Stable offline estimate used to enforce token budgets without a model call."""
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def fit_evidence_to_token_budget(
    items: list[EvidenceItem],
    *,
    used_tokens: int,
    max_total_tokens: int,
) -> tuple[list[EvidenceItem], int]:
    """Keep whole evidence items while staying within an estimated token cap."""
    selected: list[EvidenceItem] = []
    added = 0
    for item in items:
        item_tokens = estimate_text_tokens(item.evidence_text)
        if used_tokens + added + item_tokens > max_total_tokens:
            break
        selected.append(item)
        added += item_tokens
    return selected, added


def _mode_to_policy(mode: str) -> RetrievalPolicy:
    mapping = {
        "dense": RetrievalPolicy.DENSE,
        "sparse": RetrievalPolicy.SPARSE,
        "hybrid": RetrievalPolicy.HYBRID,
        "hybrid_rerank": RetrievalPolicy.HYBRID_RERANK,
        "graph": RetrievalPolicy.GRAPH,
    }
    return mapping.get(mode, RetrievalPolicy.HYBRID_RERANK)


# ---------------------------------------------------------------------------
# Optional LangGraph subgraph (single sub-question) for composition in Phase 6
# ---------------------------------------------------------------------------


def build_research_subgraph(agent: ResearchAgent) -> Any:
    """Compile a minimal LangGraph loop for one sub-question (budget-aware)."""

    def append_event_dicts(
        state: _SubResearchState,
        *events: ExecutionEvent,
    ) -> list[dict[str, Any]]:
        return list(state.get("events") or []) + [event.model_dump(mode="json") for event in events]

    def decide(state: _SubResearchState) -> dict[str, Any]:
        tool_calls = int(state.get("tool_call_count") or 0)
        max_tools = int(state.get("max_tool_calls") or 4)
        iterations = int(state.get("iteration") or 0)
        max_iterations = int(state.get("max_iterations") or 4)
        evidence = list(state.get("evidence") or [])
        max_ev = int(state.get("max_evidence") or 8)
        token_usage = int(state.get("token_usage") or 0)
        max_tokens = int(state.get("max_total_tokens") or agent.config.max_total_tokens_per_pass)
        queue = list(state.get("modes_queue") or [])
        if tool_calls >= max_tools:
            return {
                "terminated_reason": "tool_budget_exhausted",
                "events": append_event_dicts(
                    state,
                    ExecutionEvent(
                        run_id=state["run_id"],
                        event_type=EventType.BUDGET_HIT,
                        component="researcher.graph",
                        summary="tool budget exhausted",
                    ),
                ),
            }
        question = str((state.get("sub_question") or {}).get("question") or "")
        if token_usage + estimate_text_tokens(question) > max_tokens:
            return {
                "terminated_reason": "token_budget_exhausted",
                "events": append_event_dicts(
                    state,
                    ExecutionEvent(
                        run_id=state["run_id"],
                        event_type=EventType.BUDGET_HIT,
                        component="researcher.graph",
                        summary="token budget exhausted",
                        payload={
                            "estimated_tokens": token_usage,
                            "max_total_tokens": max_tokens,
                        },
                    ),
                ),
            }
        if iterations >= max_iterations:
            return {
                "terminated_reason": "iteration_budget_exhausted",
                "events": append_event_dicts(
                    state,
                    ExecutionEvent(
                        run_id=state["run_id"],
                        event_type=EventType.BUDGET_HIT,
                        component="researcher.graph",
                        summary="iteration budget exhausted",
                    ),
                ),
            }
        if len(evidence) >= max_ev:
            return {
                "terminated_reason": "evidence_budget_reached",
                "events": append_event_dicts(
                    state,
                    ExecutionEvent(
                        run_id=state["run_id"],
                        event_type=EventType.BUDGET_HIT,
                        component="researcher.graph",
                        summary="evidence budget reached",
                    ),
                ),
            }
        if not queue:
            return {"terminated_reason": "completed"}
        mode = queue[0]
        return {
            "last_action": {"mode": mode},
            "modes_queue": queue[1:],
            "iteration": iterations + 1,
            "events": append_event_dicts(
                state,
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.DECISION,
                    component="researcher.graph",
                    summary=f"next mode={mode}",
                ),
            ),
        }

    def execute(state: _SubResearchState) -> dict[str, Any]:
        mode = (state.get("last_action") or {}).get("mode") or "hybrid_rerank"
        sq = SubQuestion.model_validate(state["sub_question"])
        routing = RoutingDecision.model_validate(state["routing"])
        existing = EvidenceLedger(
            items=[EvidenceItem.model_validate(e) for e in state.get("evidence") or []]
        )
        action, result, new_items = agent._execute_mode(
            run_id=state["run_id"],
            sub_question=sq,
            mode=mode,
            routing=routing,
            existing=existing,
            max_new=max(
                0,
                int(state.get("max_evidence") or 8) - len(existing.items),
            ),
        )
        query_tokens = estimate_text_tokens(sq.question)
        used_tokens = int(state.get("token_usage") or 0) + query_tokens
        candidate_count = len(new_items)
        new_items, evidence_tokens = fit_evidence_to_token_budget(
            new_items,
            used_tokens=used_tokens,
            max_total_tokens=int(
                state.get("max_total_tokens") or agent.config.max_total_tokens_per_pass
            ),
        )
        used_tokens += evidence_tokens
        merged = existing.merge(new_items)
        selected_event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.TOOL_SELECTED,
            component="researcher.graph",
            summary=f"tool={action.tool_name}",
            payload=action.model_dump(mode="json"),
        )
        result_event = ExecutionEvent(
            run_id=state["run_id"],
            event_type=EventType.TOOL_RESULT,
            component="researcher.graph",
            summary=f"{mode} hits={len(result.hits)} new={len(new_items)}",
            payload={
                "hits": [
                    {
                        "chunk_id": hit.chunk_id,
                        "paper_id": hit.paper_id,
                        "page_start": hit.page_start,
                        "page_end": hit.page_end,
                        "score": hit.score,
                    }
                    for hit in result.hits
                ]
            },
        )
        new_events = [selected_event, result_event]
        if result.debug.get("error"):
            new_events.append(
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.ERROR,
                    component="researcher.graph",
                    summary=f"{action.tool_name} failed",
                    payload={"error_type": "retrieval_tool_error"},
                )
            )
        if new_items:
            new_events.append(
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.EVIDENCE_ADDED,
                    component="researcher.graph",
                    summary=f"added {len(new_items)} evidence items",
                )
            )
        updates: dict[str, Any] = {
            "tool_call_count": int(state.get("tool_call_count") or 0) + 1,
            "token_usage": used_tokens,
            "evidence": [e.model_dump(mode="json") for e in merged.items],
            "actions": list(state.get("actions") or []) + [action.model_dump(mode="json")],
            "last_new_count": len(new_items),
            "events": append_event_dicts(state, *new_events),
        }
        if candidate_count > len(new_items):
            updates["terminated_reason"] = "token_budget_exhausted"
            updates["events"] = append_event_dicts(
                state,
                *new_events,
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.BUDGET_HIT,
                    component="researcher.graph",
                    summary="evidence truncated at token budget",
                ),
            )
        return updates

    def route_after_decide(state: _SubResearchState) -> Literal["execute", "finish"]:
        if state.get("terminated_reason"):
            return "finish"
        if state.get("last_action"):
            return "execute"
        return "finish"

    def route_after_execute(state: _SubResearchState) -> Literal["decide", "finish"]:
        if state.get("terminated_reason"):
            return "finish"
        if int(state.get("tool_call_count") or 0) >= int(state.get("max_tool_calls") or 4):
            return "finish"
        if int(state.get("iteration") or 0) >= int(state.get("max_iterations") or 4):
            return "finish"
        if len(state.get("evidence") or []) >= int(state.get("max_evidence") or 8):
            return "finish"
        if int(state.get("token_usage") or 0) >= int(
            state.get("max_total_tokens") or agent.config.max_total_tokens_per_pass
        ):
            return "finish"
        if not state.get("modes_queue"):
            return "finish"
        return "decide"

    def finish(state: _SubResearchState) -> dict[str, Any]:
        reason = state.get("terminated_reason")
        if reason is None and state.get("modes_queue"):
            if int(state.get("tool_call_count") or 0) >= int(state.get("max_tool_calls") or 4):
                reason = "tool_budget_exhausted"
            elif int(state.get("iteration") or 0) >= int(state.get("max_iterations") or 4):
                reason = "iteration_budget_exhausted"
            elif len(state.get("evidence") or []) >= int(state.get("max_evidence") or 8):
                reason = "evidence_budget_reached"
            elif int(state.get("token_usage") or 0) >= int(
                state.get("max_total_tokens") or agent.config.max_total_tokens_per_pass
            ):
                reason = "token_budget_exhausted"
        reason = reason or "completed"
        return {
            "terminated_reason": reason,
            "events": append_event_dicts(
                state,
                ExecutionEvent(
                    run_id=state["run_id"],
                    event_type=EventType.TERMINATED,
                    component="researcher.graph",
                    summary=f"subgraph terminated: {reason}",
                ),
            ),
        }

    graph = StateGraph(_SubResearchState)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_node("finish", finish)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide", route_after_decide, {"execute": "execute", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "execute", route_after_execute, {"decide": "decide", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    return graph.compile()


__all__ = [
    "ResearchAgent",
    "ResearchAgentConfig",
    "ResearchPassResult",
    "ResearchRunResult",
    "build_research_subgraph",
    "estimate_text_tokens",
    "fit_evidence_to_token_budget",
    "hits_to_evidence",
]
