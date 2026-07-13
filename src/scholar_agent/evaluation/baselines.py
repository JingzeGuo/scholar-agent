"""Baseline and ablation system runners for the frozen eval set."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from scholar_agent.agents.researcher import ResearchAgent, ResearchAgentConfig
from scholar_agent.agents.workflow import ResearchWorkflow, WorkflowConfig, WorkflowResult
from scholar_agent.evaluation.dataset import EvalQuestion
from scholar_agent.evaluation.generation import (
    EVALUATION_ANSWER_PROMPT_ID,
    generate_evaluation_answer,
    requested_generation_model,
)
from scholar_agent.llm.client import LLMClient
from scholar_agent.models.base import EventType
from scholar_agent.models.planning import SubQuestion, SubQuestionStatus
from scholar_agent.models.retrieval import NaiveRAGAnswer, RetrievalHit, RetrievalResult
from scholar_agent.retrieval.naive_rag import NaiveRAG
from scholar_agent.retrieval.router import classify_query_type
from scholar_agent.retrieval.tools import RetrievalToolkit

SystemName = Literal[
    "naive_dense",
    "hybrid_rag",
    "hybrid_rerank",
    "hybrid_graph",
    "hybrid_corrective",
    "full_agent",
    "static_all_tools",
]

ALL_SYSTEMS: list[SystemName] = [
    "naive_dense",
    "hybrid_rag",
    "hybrid_rerank",
    "hybrid_graph",
    "hybrid_corrective",
    "full_agent",
    "static_all_tools",
]


class SystemOutput(BaseModel):
    system: str
    question_id: str
    answer_text: str = ""
    hits: list[RetrievalHit] = Field(default_factory=list)
    cited_paper_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    citation_validity_rate: float = 0.0
    citation_page_ok: int = 0
    citation_page_total: int = 0
    n_claims: int = 0
    n_unsupported_claims: int = 0
    latency_ms: int = 0
    tool_call_count: int = 0
    iteration_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    token_estimate: int = 0
    estimated_cost_usd: float = 0.0
    unanswerable_predicted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _estimate_tokens(text: str) -> int:
    # Rough offline estimate (~4 chars / token)
    return max(1, len(text) // 4) if text else 0


def _estimate_cost(tokens: int, *, usd_per_1k: float = 0.0) -> float:
    return (tokens / 1000.0) * usd_per_1k


def _merge_hits(*groups: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
    """Round-robin merge so later tools are not starved by the first top-k list."""
    seen: set[str] = set()
    out: list[RetrievalHit] = []
    max_length = max((len(group) for group in groups), default=0)
    for rank in range(max_length):
        for group in groups:
            if rank >= len(group):
                continue
            hit = group[rank]
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(hit)
            if len(out) >= limit:
                return out
    return out


@dataclass
class SystemRunner:
    """Run a named baseline/ablation against one question."""

    toolkit: RetrievalToolkit
    top_k: int = 8
    max_corrective_iterations: int = 2
    research_max_tools: int = 3
    use_llm: bool = False
    llm: LLMClient | None = None
    usd_per_1k_tokens: float = 0.0
    max_latency_ms: int = 120_000

    def __post_init__(self) -> None:
        if self.use_llm and self.llm is None:
            raise ValueError("use_llm=True requires an explicitly configured LLM client")

    def run(self, system: SystemName | str, question: EvalQuestion) -> SystemOutput:
        started = perf_counter()
        name = str(system)
        try:
            if system == "naive_dense":
                out = self._run_naive(question, mode="dense", system_name=name)
            elif system == "hybrid_rag":
                out = self._run_naive(question, mode="hybrid", system_name=name)
            elif system == "hybrid_rerank":
                out = self._run_naive(question, mode="hybrid_rerank", system_name=name)
            elif system == "hybrid_graph":
                out = self._run_hybrid_graph(question)
            elif system == "hybrid_corrective":
                out = self._run_workflow(question, full=False)
            elif system == "full_agent":
                out = self._run_workflow(question, full=True)
            elif system == "static_all_tools":
                out = self._run_static_all_tools(question)
            else:
                raise ValueError(f"unknown system: {system}")
        except Exception as exc:  # noqa: BLE001
            latency = int((perf_counter() - started) * 1000)
            return SystemOutput(
                system=name,
                question_id=question.question_id,
                latency_ms=latency,
                error=str(exc),
            )
        out.system = name
        if self.use_llm and self.llm is not None:
            try:
                generation = generate_evaluation_answer(
                    question=question.question,
                    hits=list(out.hits[: self.top_k]),
                    llm=self.llm,
                    fallback_answer=out.answer_text,
                )
                out.answer_text = generation.answer_text
                out.input_tokens = generation.input_tokens
                out.output_tokens = generation.output_tokens
                out.token_estimate = generation.total_tokens
                if generation.used_llm:
                    out.cited_paper_ids = list(generation.cited_paper_ids)
                    out.citation_page_ok = generation.valid_citation_count
                    out.citation_page_total = generation.citation_count
                    out.citation_validity_rate = (
                        generation.valid_citation_count / generation.citation_count
                        if generation.citation_count
                        else 0.0
                    )
                out.metadata.update(
                    {
                        "generation_used": generation.used_llm,
                        "generation_model": generation.generation_model,
                        "generation_model_requested": requested_generation_model(self.llm),
                        "generation_prompt_id": generation.prompt_id,
                        "generation_regime": "shared_live_llm",
                        "generation_token_count_source": generation.token_count_source,
                        "generation_fallback_used": generation.fallback_used,
                        "generation_skip_reason": generation.skip_reason,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                # Provider exceptions may contain response bodies; persist only
                # the exception class in evaluation artifacts.
                out.error = f"shared generation failed: {type(exc).__name__}"
                out.input_tokens = sum(_estimate_tokens(h.text) for h in out.hits[: self.top_k])
                out.output_tokens = _estimate_tokens(out.answer_text)
                out.token_estimate = out.input_tokens + out.output_tokens
                out.metadata.update(
                    {
                        "generation_used": False,
                        "generation_model": None,
                        "generation_model_requested": requested_generation_model(self.llm),
                        "generation_prompt_id": EVALUATION_ANSWER_PROMPT_ID,
                        "generation_regime": "shared_live_llm",
                        "generation_token_count_source": "estimated",
                        "generation_fallback_used": True,
                        "generation_skip_reason": "generation_error",
                    }
                )
        else:
            out.input_tokens = sum(_estimate_tokens(h.text) for h in out.hits[: self.top_k])
            out.output_tokens = _estimate_tokens(out.answer_text)
            out.token_estimate = out.input_tokens + out.output_tokens
            out.metadata.update(
                {
                    "generation_used": False,
                    "generation_model": None,
                    "generation_model_requested": None,
                    "generation_prompt_id": None,
                    "generation_regime": "offline_heterogeneous",
                    "generation_token_count_source": "estimated",
                    "generation_fallback_used": False,
                }
            )
        out.estimated_cost_usd = _estimate_cost(
            out.token_estimate, usd_per_1k=self.usd_per_1k_tokens
        )
        out.unanswerable_predicted = out.unanswerable_predicted or _looks_unanswerable(
            out.answer_text
        )
        out.latency_ms = int((perf_counter() - started) * 1000)
        return out

    def _run_naive(
        self,
        question: EvalQuestion,
        *,
        mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"],
        system_name: str,
    ) -> SystemOutput:
        rag = NaiveRAG(
            self.toolkit,
            mode=mode,
            top_k=self.top_k,
        )
        # Generation is deliberately applied once in ``run`` so every live
        # system receives the same model and answer prompt.
        answer: NaiveRAGAnswer = rag.answer(question.question, use_llm=False)
        return SystemOutput(
            system=system_name,
            question_id=question.question_id,
            answer_text=answer.answer,
            hits=list(answer.hits),
            cited_paper_ids=[c.paper_id for c in answer.citations],
            citation_validity_rate=self._hit_provenance_rate(list(answer.hits)),
            citation_page_ok=sum(
                1
                for citation in answer.citations
                if citation.page_start >= 1 and citation.page_end >= citation.page_start
            ),
            citation_page_total=len(answer.citations),
            n_claims=1 if answer.answer.strip() else 0,
            tool_call_count=1,
            iteration_count=1,
            unanswerable_predicted=_looks_unanswerable(answer.answer),
            metadata={
                "mode": mode,
                "selected_tools": [mode],
                "adaptive_routing": False,
                "naive_rag_internal_llm_used": answer.used_llm,
            },
        )

    def _run_hybrid_graph(self, question: EvalQuestion) -> SystemOutput:
        hybrid = self.toolkit.search(question.question, mode="hybrid_rerank", k=self.top_k)
        graph_hits: list[RetrievalHit] = []
        if self.toolkit.graph is not None:
            try:
                graph_res: RetrievalResult = self.toolkit.search(
                    question.question, mode="graph", k=self.top_k
                )
                graph_hits = list(graph_res.hits)
            except Exception:  # noqa: BLE001
                graph_hits = []
        hits = _merge_hits(list(hybrid.hits), graph_hits, limit=self.top_k)
        # Extractive answer from merged hits
        answer = _extractive_from_hits(question.question, hits)
        return SystemOutput(
            system="hybrid_graph",
            question_id=question.question_id,
            answer_text=answer,
            hits=hits,
            cited_paper_ids=[h.paper_id for h in hits],
            citation_validity_rate=self._hit_provenance_rate(hits),
            citation_page_ok=sum(
                1 for hit in hits if hit.page_start >= 1 and hit.page_end >= hit.page_start
            ),
            citation_page_total=len(hits),
            n_claims=len(hits),
            tool_call_count=2 if graph_hits or self.toolkit.graph is not None else 1,
            iteration_count=1,
            unanswerable_predicted=_looks_unanswerable(answer),
            metadata={
                "n_graph_hits": len(graph_hits),
                "selected_tools": ["hybrid_rerank"] + (["graph"] if graph_hits else []),
                "adaptive_routing": False,
            },
        )

    def _run_workflow(self, question: EvalQuestion, *, full: bool) -> SystemOutput:
        cfg = WorkflowConfig(
            max_corrective_iterations=self.max_corrective_iterations
            if full
            else max(1, self.max_corrective_iterations),
            max_total_tool_calls=12 if full else 8,
            max_latency_ms=self.max_latency_ms,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=self.research_max_tools,
                max_evidence_per_sub_question=self.top_k,
                allow_policy_override=full,
            ),
            parallel_research=False,
        )
        # hybrid_corrective: still runs plan/verify loop but with tighter budgets
        if not full:
            cfg.max_corrective_iterations = max(1, min(2, self.max_corrective_iterations))
            cfg.research = ResearchAgentConfig(
                max_tool_calls_per_pass=min(2, self.research_max_tools),
                max_evidence_per_sub_question=self.top_k,
                allow_policy_override=False,
            )
        result: WorkflowResult = ResearchWorkflow(self.toolkit, config=cfg).run(question.question)
        answer_text = ""
        cited: list[str] = []
        hits: list[RetrievalHit] = []
        if result.final_answer is not None:
            answer_text = result.final_answer.markdown
            cited = (
                list(result.final_answer.citation_report.cited_paper_ids)
                if result.final_answer.citation_report
                else []
            )
            cited = cited or [c.paper_id for c in result.final_answer.source_cards]
        elif result.draft_answer is not None:
            answer_text = result.draft_answer.markdown
        else:
            answer_text = result.verification.rationale_summary

        for item in result.evidence_ledger.items[: self.top_k]:
            hits.append(
                RetrievalHit(
                    chunk_id=item.chunk_id,
                    paper_id=item.paper_id,
                    text=item.evidence_text,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    score=item.retrieval_score,
                    retrieval_method=item.retrieval_method,
                )
            )

        final = result.final_answer
        report = final.citation_report if final is not None else None
        cards = final.source_cards if final is not None else []
        unsupported = 0
        if report is not None:
            unsupported = sum(
                1
                for issue in report.issues
                if issue.severity == "error"
                and (
                    "no valid citations" in issue.message.lower()
                    or "does not support" in issue.message.lower()
                )
            )

        workflow_diagnostics = _workflow_diagnostics(result, top_k=self.top_k)
        return SystemOutput(
            system="full_agent" if full else "hybrid_corrective",
            question_id=question.question_id,
            answer_text=answer_text,
            hits=hits,
            cited_paper_ids=cited,
            cited_evidence_ids=(list(report.cited_evidence_ids) if report is not None else []),
            citation_validity_rate=(1.0 if report is not None and report.is_valid else 0.0),
            citation_page_ok=sum(
                1 for card in cards if card.page_start >= 1 and card.page_end >= card.page_start
            ),
            citation_page_total=len(cards),
            n_claims=len(final.claims) if final is not None else 0,
            n_unsupported_claims=unsupported,
            tool_call_count=result.tool_call_count,
            iteration_count=result.iteration,
            unanswerable_predicted=result.unanswerable or _looks_unanswerable(answer_text),
            metadata={
                "terminated_reason": result.terminated_reason,
                "coverage": result.verification.coverage_score,
                "adaptive_routing": True,
                "conflicting_evidence_ids": list(result.verification.conflicting_evidence_ids),
                **workflow_diagnostics,
            },
        )

    def _run_static_all_tools(self, question: EvalQuestion) -> SystemOutput:
        """Ablation: always run dense + sparse + hybrid_rerank (+ graph if present)."""
        modes: list[Literal["dense", "sparse", "hybrid_rerank", "graph"]] = [
            "dense",
            "sparse",
            "hybrid_rerank",
        ]
        groups: list[list[RetrievalHit]] = []
        tools = 0
        executed_modes: list[str] = []
        for mode in modes:
            try:
                res = self.toolkit.search(question.question, mode=mode, k=self.top_k)
                groups.append(list(res.hits))
                tools += 1
                executed_modes.append(mode)
            except Exception:  # noqa: BLE001
                continue
        if self.toolkit.graph is not None:
            try:
                res = self.toolkit.search(question.question, mode="graph", k=self.top_k)
                groups.append(list(res.hits))
                tools += 1
                executed_modes.append("graph")
            except Exception:  # noqa: BLE001
                pass
        hits = _merge_hits(*groups, limit=self.top_k)
        answer = _extractive_from_hits(question.question, hits)
        return SystemOutput(
            system="static_all_tools",
            question_id=question.question_id,
            answer_text=answer,
            hits=hits,
            cited_paper_ids=[h.paper_id for h in hits],
            citation_validity_rate=self._hit_provenance_rate(hits),
            citation_page_ok=sum(
                1 for hit in hits if hit.page_start >= 1 and hit.page_end >= hit.page_start
            ),
            citation_page_total=len(hits),
            n_claims=len(hits),
            tool_call_count=tools,
            iteration_count=1,
            unanswerable_predicted=_looks_unanswerable(answer),
            metadata={
                "modes": list(modes),
                "selected_tools": executed_modes,
                "adaptive_routing": False,
            },
        )

    def _hit_provenance_rate(self, hits: list[RetrievalHit]) -> float:
        """Fraction of cited hits that still map to the canonical store."""
        if not hits:
            return 1.0
        store = getattr(self.toolkit, "store", None)
        if store is None:
            return 0.0
        valid = 0
        for hit in hits:
            chunk = store.get_chunk(hit.chunk_id)
            paper = store.get_paper(hit.paper_id)
            if (
                chunk is not None
                and paper is not None
                and chunk.paper_id == hit.paper_id
                and hit.page_start >= chunk.page_start
                and hit.page_end <= chunk.page_end
                and paper.page_count is not None
                and hit.page_end <= paper.page_count
            ):
                valid += 1
        return valid / len(hits)


def _extractive_from_hits(question: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return (
            f"No supporting passages were retrieved for: {question}\n"
            "Limitation: the corpus may not contain an answer."
        )
    lines = [f"Question: {question}", "", "Evidence notes:"]
    for h in hits[:5]:
        page = (
            f"p.{h.page_start}" if h.page_start == h.page_end else f"p.{h.page_start}-{h.page_end}"
        )
        snip = " ".join(h.text.split())
        if len(snip) > 240:
            snip = snip[:239] + "…"
        lines.append(f"- [{h.paper_id} {page}] {snip}")
    return "\n".join(lines)


def _looks_unanswerable(text: str) -> bool:
    low = text.lower()
    cues = (
        "cannot answer",
        "limitation",
        "no supporting",
        "corpus may not",
        "corpus does not",
        "insufficient",
        "unanswerable",
        "no verified evidence",
    )
    return any(c in low for c in cues)


def _workflow_diagnostics(result: WorkflowResult, *, top_k: int) -> dict[str, Any]:
    """Extract measurable routing/corrective facts from structured events."""
    selected_tools: list[str] = []
    selected_policies: list[str] = []
    fallback_chunk_ids: set[str] = set()
    fallback_paper_ids: set[str] = set()
    initial_chunk_ids: list[str] = []
    initial_paper_ids: list[str] = []
    corrective_seen = False
    corrective_triggered = False
    initial_results_observed = False

    for event in result.events:
        if event.event_type == EventType.CORRECTIVE:
            corrective_seen = True
            corrective_triggered = True
            continue
        if event.event_type == EventType.TOOL_SELECTED:
            tool = event.payload.get("tool_name")
            policy = event.payload.get("policy")
            if tool:
                selected_tools.append(str(tool))
            if policy:
                selected_policies.append(str(policy))
        elif event.event_type == EventType.TOOL_RESULT and not corrective_seen:
            initial_results_observed = True
            for hit in event.payload.get("hits") or []:
                chunk_id = hit.get("chunk_id")
                paper_id = hit.get("paper_id")
                if chunk_id:
                    fallback_chunk_ids.add(str(chunk_id))
                if paper_id:
                    fallback_paper_ids.add(str(paper_id))
        elif (
            event.event_type == EventType.ITERATION
            and event.payload.get("iteration") == 0
            and not initial_chunk_ids
        ):
            initial_chunk_ids = [
                str(value) for value in event.payload.get("evidence_chunk_ids") or []
            ][:top_k]
            initial_paper_ids = [
                str(value) for value in event.payload.get("evidence_paper_ids") or []
            ][:top_k]

    if not initial_chunk_ids:
        # Backward-compatible fallback for saved/third-party workflow results
        # created before ordered iteration snapshots were added.
        initial_chunk_ids = sorted(fallback_chunk_ids)[:top_k]
        initial_paper_ids = sorted(fallback_paper_ids)[:top_k]

    return {
        "selected_tools": selected_tools,
        "selected_policies": selected_policies,
        "corrective_triggered": corrective_triggered,
        "initial_results_observed": initial_results_observed,
        "initial_chunk_ids": initial_chunk_ids,
        "initial_paper_ids": initial_paper_ids,
    }


def make_research_agent_for_question(
    toolkit: RetrievalToolkit, question: EvalQuestion, *, config: ResearchAgentConfig
) -> SystemOutput:
    """Utility used in tests: single ResearchAgent pass."""
    qtype, _ = classify_query_type(question.question)
    sq = SubQuestion(
        id="sq_0",
        question=question.question,
        query_type=qtype,
        required_evidence=["supporting passages"],
        status=SubQuestionStatus.PENDING,
    )
    started = perf_counter()
    agent = ResearchAgent(toolkit, config=config)
    result = agent.research_sub_question(sq)
    hits = [
        RetrievalHit(
            chunk_id=e.chunk_id,
            paper_id=e.paper_id,
            text=e.evidence_text,
            page_start=e.page_start,
            page_end=e.page_end,
            score=e.retrieval_score,
            retrieval_method=e.retrieval_method,
        )
        for e in result.evidence
    ]
    answer = _extractive_from_hits(question.question, hits)
    return SystemOutput(
        system="research_agent",
        question_id=question.question_id,
        answer_text=answer,
        hits=hits,
        cited_paper_ids=[h.paper_id for h in hits],
        latency_ms=int((perf_counter() - started) * 1000),
        tool_call_count=result.tool_call_count,
        iteration_count=1,
        unanswerable_predicted=_looks_unanswerable(answer),
    )


# Typing helper for external plugins
SystemFn = Callable[[EvalQuestion], SystemOutput]
