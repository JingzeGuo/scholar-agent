"""Demo orchestration: live workflow + naive comparison + trace building."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from scholar_agent.agents.researcher import ResearchAgentConfig
from scholar_agent.agents.workflow import ResearchWorkflow, WorkflowConfig, WorkflowResult
from scholar_agent.app.demo_models import (
    DemoSessionResult,
    DemoSettings,
    NaiveComparisonView,
    SavedDemoRun,
    TraceSummary,
)
from scholar_agent.app.demo_runs import find_saved_run, list_saved_runs, save_demo_run
from scholar_agent.app.status import SystemStatus, collect_system_status
from scholar_agent.config import AppConfig, load_config
from scholar_agent.ids import new_run_id
from scholar_agent.logging import get_logger
from scholar_agent.models.answer import SourceCard
from scholar_agent.models.base import EventType, ExecutionEvent, utc_now_iso
from scholar_agent.models.evidence import EvidenceItem
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.index_builder import load_toolkit
from scholar_agent.retrieval.naive_rag import NaiveRAG
from scholar_agent.retrieval.router import classify_query_type
from scholar_agent.retrieval.tools import RetrievalToolkit

logger = get_logger(__name__)


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def build_corrective_steps(events: list[ExecutionEvent]) -> list[dict[str, Any]]:
    """Build a compact verifier/corrective timeline without exposing reasoning."""
    steps: list[dict[str, Any]] = []
    for event in events:
        if event.event_type not in {
            EventType.VERIFICATION,
            EventType.CORRECTIVE,
            EventType.TOOL_RESULT,
            EventType.RUN_FINISHED,
            EventType.BUDGET_HIT,
        }:
            continue
        if event.event_type == EventType.TOOL_RESULT and not steps:
            continue
        steps.append(
            {
                "number": len(steps) + 1,
                "kind": event.event_type.value,
                "component": event.component,
                "summary": event.summary,
                "timestamp": event.timestamp,
                "is_sufficient": event.payload.get("is_sufficient"),
                "method": event.payload.get("method") or event.payload.get("tool_name"),
            }
        )
    return steps[:24]


def build_trace_summary(
    *,
    query: str,
    result: WorkflowResult | None,
    settings: DemoSettings,
    naive: NaiveComparisonView | None = None,
) -> TraceSummary:
    if result is None:
        return TraceSummary()

    qtype, _ = classify_query_type(query)
    tool_events: list[dict[str, Any]] = []
    methods: set[str] = set()
    for event in result.events:
        if event.event_type in {EventType.TOOL_SELECTED, EventType.TOOL_RESULT}:
            tool_events.append(
                {
                    "event_type": event.event_type.value,
                    "component": event.component,
                    "summary": event.summary,
                    "payload": event.payload,
                }
            )
            method = event.payload.get("method") or event.payload.get("tool_name")
            if isinstance(method, str):
                methods.add(method)
        if event.event_type == EventType.EVIDENCE_ADDED:
            method = event.payload.get("retrieval_method")
            if isinstance(method, str):
                methods.add(method)

    for item in result.evidence_ledger.items:
        methods.add(item.retrieval_method)

    accepted_ids: set[str] = set()
    if result.final_answer and result.final_answer.citation_report:
        accepted_ids = set(result.final_answer.citation_report.cited_evidence_ids)
    elif result.final_answer:
        for card in result.final_answer.source_cards:
            accepted_ids.add(card.evidence_id)

    report = result.final_answer.citation_report if result.final_answer else None
    answer_text = (
        result.final_answer.markdown
        if result.final_answer
        else (result.draft_answer.markdown if result.draft_answer else "")
    )
    tokens = _token_estimate(answer_text) + sum(
        _token_estimate(e.evidence_text) for e in result.evidence_ledger.items[:12]
    )
    if naive is not None:
        tokens += _token_estimate(naive.answer)

    return TraceSummary(
        query_type=qtype.value,
        answer_type=result.plan.answer_type if result.plan else None,
        sub_questions=[
            {
                "id": sq.id,
                "question": sq.question,
                "status": sq.status.value,
                "query_type": sq.query_type.value,
            }
            for sq in result.plan.sub_questions
        ],
        tool_events=tool_events[:40],
        retrieval_methods=sorted(methods),
        evidence_count=len(result.evidence_ledger.items),
        verified_evidence_count=len(accepted_ids)
        if accepted_ids
        else len(result.evidence_ledger.items),
        coverage_score=result.verification.coverage_score,
        is_sufficient=result.verification.is_sufficient,
        corrective_iterations=result.iteration,
        corrective_queries=list(result.verification.corrective_queries),
        corrective_steps=build_corrective_steps(result.events),
        conflicting_evidence_ids=list(result.verification.conflicting_evidence_ids),
        citation_valid=report.is_valid if report else None,
        citation_issue_count=len(report.issues) if report else 0,
        terminated_reason=result.terminated_reason,
        unanswerable=result.unanswerable,
        latency_ms=result.latency_ms,
        tool_call_count=result.tool_call_count,
        token_estimate=tokens,
    )


def filter_verified_evidence(
    items: list[EvidenceItem],
    *,
    final_answer_cards: list[SourceCard],
    verified_only: bool,
) -> list[EvidenceItem]:
    if not verified_only or not final_answer_cards:
        return list(items)
    allowed = {c.evidence_id for c in final_answer_cards}
    filtered = [i for i in items if i.evidence_id in allowed]
    return filtered or list(items)


@dataclass
class DemoService:
    """Run live demo sessions or replay saved interview runs."""

    config: AppConfig = field(default_factory=load_config)
    toolkit: RetrievalToolkit | None = None
    _toolkit_key: tuple[str, bool] | None = None

    def get_status(self) -> SystemStatus:
        return collect_system_status(self.config)

    def list_replays(self) -> list[SavedDemoRun]:
        return list_saved_runs()

    def replay(self, demo_id: str) -> DemoSessionResult:
        saved = find_saved_run(demo_id)
        if saved is None:
            return DemoSessionResult(
                run_id=new_run_id(),
                query="",
                settings=DemoSettings(),
                offline_replay=True,
                error=f"Saved demo not found: {demo_id}",
            )
        session = saved.session.model_copy(deep=True)
        session.offline_replay = True
        session.error = None
        chunks_path = self.config.paths.processed_dir / "chunks.jsonl"
        if chunks_path.is_file():
            from scholar_agent.app.source_viewer import validate_saved_run_provenance

            store = ChunkStore.from_processed_dir(self.config.paths.processed_dir)
            provenance_issues = validate_saved_run_provenance(saved, store)
            if provenance_issues:
                session.error = (
                    "Saved replay provenance is stale; regenerate demo runs. "
                    + "; ".join(provenance_issues[:3])
                )
        if not session.trace.corrective_steps and session.events:
            # Older committed replays remain understandable after schema upgrades.
            session.trace.corrective_steps = build_corrective_steps(session.events)
        return session

    def ensure_toolkit(
        self,
        *,
        embedding_backend: Literal["auto", "hash", "st"] = "hash",
        enable_graph: bool = True,
    ) -> RetrievalToolkit:
        key = (embedding_backend, enable_graph)
        if self.toolkit is not None and self._toolkit_key == key:
            return self.toolkit
        toolkit = load_toolkit(
            config=self.config,
            embedding_backend=embedding_backend,
            reranker_backend="lexical" if embedding_backend == "hash" else "auto",
            load_graph=enable_graph,
        )
        if not enable_graph:
            toolkit.graph = None
        self.toolkit = toolkit
        self._toolkit_key = key
        return toolkit

    def run_live(self, query: str, settings: DemoSettings | None = None) -> DemoSessionResult:
        settings = settings or DemoSettings()
        status = self.get_status()
        if not status.ok and not status.sparse_index_ready:
            # Allow replay-only environments to fail clearly
            return DemoSessionResult(
                run_id=new_run_id(),
                query=query,
                settings=settings,
                status=status.as_dict(),
                error=(
                    "Indexes not ready for live demo. Use saved-run replay, or run "
                    "`scholar-agent index build --embedding-backend hash`."
                ),
            )

        try:
            toolkit = self.ensure_toolkit(
                embedding_backend=settings.embedding_backend,
                enable_graph=settings.enable_graph,
            )
        except Exception as exc:  # noqa: BLE001
            return DemoSessionResult(
                run_id=new_run_id(),
                query=query,
                settings=settings,
                status=status.as_dict(),
                error=f"Failed to load retrieval toolkit: {exc}",
            )

        naive_view: NaiveComparisonView | None = None
        if settings.compare_naive_rag:
            naive_view = self._run_naive(toolkit, query, settings)

        if settings.static_routing:
            # Interview ablation: static multi-tool retrieve-then-extract (no agent loop)
            session = self._run_static(toolkit, query, settings, status, naive_view)
            return session

        max_corr = settings.max_corrective_iterations if settings.enable_corrective else 0
        wf_cfg = WorkflowConfig(
            max_corrective_iterations=max_corr,
            max_total_tool_calls=settings.max_total_tool_calls,
            max_total_tokens=self.config.budgets.max_total_tokens,
            max_latency_ms=self.config.budgets.max_latency_ms,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=self.config.budgets.max_tool_calls_per_research_pass,
                max_evidence_per_sub_question=settings.top_k,
                max_iterations_per_pass=self.config.budgets.max_research_iterations_per_pass,
                max_latency_ms=self.config.budgets.max_latency_ms,
                max_total_tokens_per_pass=self.config.budgets.max_total_tokens,
                allow_policy_override=not settings.static_routing,
            ),
            parallel_research=False,
        )
        try:
            result = ResearchWorkflow(toolkit, config=wf_cfg).run(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception("demo workflow failed")
            return DemoSessionResult(
                run_id=new_run_id(),
                query=query,
                settings=settings,
                status=status.as_dict(),
                naive=naive_view,
                error=str(exc),
            )
        return self._from_workflow(result, settings, status, naive_view)

    def _run_naive(
        self, toolkit: RetrievalToolkit, query: str, settings: DemoSettings
    ) -> NaiveComparisonView:
        started = perf_counter()
        rag = NaiveRAG(
            toolkit,
            mode="hybrid_rerank",
            top_k=settings.top_k,
        )
        answer = rag.answer(query, use_llm=settings.use_llm)
        latency = int((perf_counter() - started) * 1000)
        return NaiveComparisonView(
            answer=answer.answer,
            method=answer.method,
            citations=[c.model_dump(mode="json") for c in answer.citations],
            hit_count=len(answer.hits),
            latency_ms=latency,
            used_llm=answer.used_llm,
        )

    def _run_static(
        self,
        toolkit: RetrievalToolkit,
        query: str,
        settings: DemoSettings,
        status: SystemStatus,
        naive: NaiveComparisonView | None,
    ) -> DemoSessionResult:
        """Static ablation: dense+sparse+hybrid(+graph) merge, extractive answer."""
        from scholar_agent.evaluation.baselines import SystemRunner
        from scholar_agent.evaluation.dataset import EvalQuestion

        started = perf_counter()
        runner = SystemRunner(
            toolkit,
            top_k=settings.top_k,
            max_corrective_iterations=0,
            research_max_tools=3,
            use_llm=settings.use_llm,
            max_latency_ms=self.config.budgets.max_latency_ms,
        )
        fake_q = EvalQuestion(
            question_id="demo",
            question=query,
            question_type="factual",
        )
        out = runner.run("static_all_tools", fake_q)
        latency = int((perf_counter() - started) * 1000)
        cards: list[SourceCard] = []
        for i, h in enumerate(out.hits[: settings.top_k], start=1):
            paper = toolkit.get_paper(h.paper_id)
            cards.append(
                SourceCard(
                    evidence_id=f"hit_{i}",
                    paper_id=h.paper_id,
                    chunk_id=h.chunk_id,
                    page_start=h.page_start,
                    page_end=h.page_end,
                    snippet=" ".join(h.text.split())[:240],
                    retrieval_method=h.retrieval_method,
                    title=paper.title if paper else None,
                    pdf_path=paper.pdf_path if paper else None,
                )
            )
        trace = TraceSummary(
            query_type=classify_query_type(query)[0].value,
            answer_type="static_all_tools",
            sub_questions=[],
            tool_events=[
                {
                    "event_type": "tool_result",
                    "component": "static_all_tools",
                    "summary": f"merged {len(out.hits)} hits",
                    "payload": {"modes": out.metadata.get("modes")},
                }
            ],
            retrieval_methods=sorted({h.retrieval_method for h in out.hits}),
            evidence_count=len(out.hits),
            verified_evidence_count=len(cards),
            coverage_score=None,
            is_sufficient=bool(out.hits),
            corrective_iterations=0,
            citation_valid=True,
            terminated_reason="static_complete",
            unanswerable=out.unanswerable_predicted,
            latency_ms=latency or out.latency_ms,
            tool_call_count=out.tool_call_count,
            token_estimate=out.token_estimate,
        )
        return DemoSessionResult(
            run_id=new_run_id(),
            query=query,
            settings=settings,
            offline_replay=False,
            answer_markdown=out.answer_text,
            claims=[],
            source_cards=cards,
            evidence=[],
            events=[],
            trace=trace,
            naive=naive,
            status=status.as_dict(),
            error=out.error,
        )

    def _from_workflow(
        self,
        result: WorkflowResult,
        settings: DemoSettings,
        status: SystemStatus,
        naive: NaiveComparisonView | None,
    ) -> DemoSessionResult:
        final = result.final_answer
        cards = list(final.source_cards) if final else []
        evidence = filter_verified_evidence(
            list(result.evidence_ledger.items),
            final_answer_cards=cards,
            verified_only=settings.verified_evidence_only,
        )
        answer_md = (
            final.markdown
            if final is not None
            else (
                result.draft_answer.markdown
                if result.draft_answer is not None
                else result.verification.rationale_summary
            )
        )
        claims = [c.model_dump(mode="json") for c in final.claims] if final is not None else []
        trace = build_trace_summary(
            query=result.query, result=result, settings=settings, naive=naive
        )
        return DemoSessionResult(
            run_id=result.run_id,
            query=result.query,
            settings=settings,
            offline_replay=False,
            answer_markdown=answer_md,
            claims=claims,
            source_cards=cards,
            evidence=evidence,
            plan=result.plan,
            verification=result.verification,
            final_answer=final,
            events=list(result.events),
            trace=trace,
            naive=naive,
            status=status.as_dict(),
        )

    def to_saved_run(
        self,
        session: DemoSessionResult,
        *,
        demo_id: str,
        title: str,
        notes: str = "",
    ) -> SavedDemoRun:
        fingerprint = self.toolkit.store.fingerprint if self.toolkit is not None else None
        return SavedDemoRun(
            demo_id=demo_id,
            title=title,
            query=session.query,
            settings=session.settings,
            created_at=utc_now_iso(),
            offline=True,
            notes=notes,
            corpus_fingerprint=fingerprint,
            provenance_verified=bool(fingerprint),
            session=session.model_copy(update={"offline_replay": True, "error": None}),
        )

    def persist_session(
        self,
        session: DemoSessionResult,
        *,
        demo_id: str,
        title: str,
        notes: str = "",
    ) -> Any:
        saved = self.to_saved_run(session, demo_id=demo_id, title=title, notes=notes)
        return save_demo_run(saved)
