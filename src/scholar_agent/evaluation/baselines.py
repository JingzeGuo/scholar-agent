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
    latency_ms: int = 0
    tool_call_count: int = 0
    iteration_count: int = 0
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
    seen: set[str] = set()
    out: list[RetrievalHit] = []
    for group in groups:
        for hit in group:
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
    usd_per_1k_tokens: float = 0.0

    def run(self, system: SystemName | str, question: EvalQuestion) -> SystemOutput:
        started = perf_counter()
        name = str(system)
        try:
            if system == "naive_dense":
                out = self._run_naive(question, mode="dense", system_name=name)
            elif system == "hybrid_rag":
                out = self._run_naive(question, mode="hybrid", system_name=name)
            elif system == "hybrid_rerank":
                out = self._run_naive(
                    question, mode="hybrid_rerank", system_name=name
                )
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
        out.latency_ms = int((perf_counter() - started) * 1000)
        out.token_estimate = _estimate_tokens(out.answer_text) + sum(
            _estimate_tokens(h.text) for h in out.hits[: self.top_k]
        )
        out.estimated_cost_usd = _estimate_cost(
            out.token_estimate, usd_per_1k=self.usd_per_1k_tokens
        )
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
        answer: NaiveRAGAnswer = rag.answer(question.question, use_llm=self.use_llm)
        return SystemOutput(
            system=system_name,
            question_id=question.question_id,
            answer_text=answer.answer,
            hits=list(answer.hits),
            cited_paper_ids=[c.paper_id for c in answer.citations],
            tool_call_count=1,
            iteration_count=1,
            unanswerable_predicted=_looks_unanswerable(answer.answer),
            metadata={"mode": mode, "used_llm": answer.used_llm},
        )

    def _run_hybrid_graph(self, question: EvalQuestion) -> SystemOutput:
        hybrid = self.toolkit.search(
            question.question, mode="hybrid_rerank", k=self.top_k
        )
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
            tool_call_count=2 if graph_hits or self.toolkit.graph is not None else 1,
            iteration_count=1,
            unanswerable_predicted=_looks_unanswerable(answer),
            metadata={"n_graph_hits": len(graph_hits)},
        )

    def _run_workflow(self, question: EvalQuestion, *, full: bool) -> SystemOutput:
        cfg = WorkflowConfig(
            max_corrective_iterations=self.max_corrective_iterations if full else max(
                1, self.max_corrective_iterations
            ),
            max_total_tool_calls=12 if full else 8,
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
        result: WorkflowResult = ResearchWorkflow(self.toolkit, config=cfg).run(
            question.question
        )
        answer_text = ""
        cited: list[str] = []
        hits: list[RetrievalHit] = []
        if result.final_answer is not None:
            answer_text = result.final_answer.markdown
            cited = list(result.final_answer.citation_report.cited_paper_ids) if result.final_answer.citation_report else []
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
            if item.paper_id not in cited:
                cited.append(item.paper_id)

        return SystemOutput(
            system="full_agent" if full else "hybrid_corrective",
            question_id=question.question_id,
            answer_text=answer_text,
            hits=hits,
            cited_paper_ids=cited,
            tool_call_count=result.tool_call_count,
            iteration_count=result.iteration,
            unanswerable_predicted=result.unanswerable
            or _looks_unanswerable(answer_text),
            metadata={
                "terminated_reason": result.terminated_reason,
                "coverage": result.verification.coverage_score,
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
        for mode in modes:
            try:
                res = self.toolkit.search(question.question, mode=mode, k=self.top_k)
                groups.append(list(res.hits))
                tools += 1
            except Exception:  # noqa: BLE001
                continue
        if self.toolkit.graph is not None:
            try:
                res = self.toolkit.search(question.question, mode="graph", k=self.top_k)
                groups.append(list(res.hits))
                tools += 1
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
            tool_call_count=tools,
            iteration_count=1,
            unanswerable_predicted=_looks_unanswerable(answer),
            metadata={"modes": list(modes)},
        )


def _extractive_from_hits(question: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return (
            f"No supporting passages were retrieved for: {question}\n"
            "Limitation: the corpus may not contain an answer."
        )
    lines = [f"Question: {question}", "", "Evidence notes:"]
    for h in hits[:5]:
        page = (
            f"p.{h.page_start}"
            if h.page_start == h.page_end
            else f"p.{h.page_start}-{h.page_end}"
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
