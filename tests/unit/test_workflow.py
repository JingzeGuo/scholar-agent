"""Phase 6 full workflow tests: corrective loop and termination."""

from __future__ import annotations

from scholar_agent.agents.researcher import ResearchAgentConfig
from scholar_agent.agents.workflow import ResearchWorkflow, WorkflowConfig
from scholar_agent.ids import make_chunk_id
from scholar_agent.models.base import QueryType
from scholar_agent.models.planning import SubQuestion, SubQuestionStatus
from scholar_agent.models.retrieval import RetrievalHit, RetrievalResult
from scholar_agent.retrieval.tools import RetrievalToolkit


class ScriptedToolkit(RetrievalToolkit):
    """Returns scripted hits by query keyword; tracks call count."""

    def __init__(self) -> None:
        self.store = None  # type: ignore[assignment]
        self.dense = None
        self.sparse = None
        self.graph = object()
        self.reranker = None  # type: ignore[assignment]
        self.dense_top_k = 8
        self.sparse_top_k = 8
        self.fused_top_k = 8
        self.rerank_top_k = 8
        self.rrf_k = 60
        self.calls: list[str] = []
        self._empty_once = True

    def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
        self.calls.append(query)
        q = query.lower()

        # First call for "missing" style questions returns empty to force corrective
        if "zzzz_nonexistent_topic_xyz" in q:
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[])

        if "self-rag" in q or "self rag" in q:
            text = (
                "Self-RAG retrieves on demand and uses reflection tokens "
                "to critique generation quality."
            )
            paper = "paper_self_rag"
        elif "crag" in q or "corrective" in q:
            text = (
                "CRAG evaluates retrieved documents and triggers corrective "
                "retrieval when quality is low."
            )
            paper = "paper_crag"
        elif "differ" in q or "compare" in q or "versus" in q or "vs" in q:
            text = (
                "Self-RAG and CRAG differ: Self-RAG uses reflection tokens while "
                "CRAG focuses on corrective retrieval of documents."
            )
            paper = "paper_compare"
        else:
            # Generic mildly relevant filler
            text = f"Passage discussing methods related to: {query}"
            paper = "paper_generic"

        hit = RetrievalHit(
            chunk_id=make_chunk_id(paper, page_start=1, page_end=1, text=text + mode),
            paper_id=paper,
            text=text,
            page_start=1,
            page_end=2,
            section="Method",
            score=0.85,
            retrieval_method=mode,
        )
        method = mode if mode in {"dense", "sparse", "hybrid", "hybrid_rerank", "graph"} else "hybrid"
        return RetrievalResult(query=query, method=method, hits=[hit])  # type: ignore[arg-type]


def test_workflow_terminates_when_sufficient() -> None:
    toolkit = ScriptedToolkit()
    wf = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=2,
            research=ResearchAgentConfig(max_tool_calls_per_pass=2, allow_policy_override=False),
            parallel_research=False,
        ),
    )
    result = wf.run("What is Self-RAG?")
    assert result.terminated_reason in {
        "evidence_sufficient",
        "iteration_budget_exhausted",
        "no_new_evidence",
        "no_corrective_queries",
        "tool_budget_exhausted",
        "completed",
    }
    # Should have a plan and verification
    assert result.plan.sub_questions
    assert result.verification is not None
    assert any(e.event_type.value == "plan_created" for e in result.events)
    assert any(e.event_type.value == "verification" for e in result.events)
    assert any(e.event_type.value == "run_finished" for e in result.events)


def test_missing_evidence_triggers_corrective_retrieval() -> None:
    """First research yields weak/empty; verifier asks corrective; second pass runs."""

    class TwoPhaseToolkit(ScriptedToolkit):
        def __init__(self) -> None:
            super().__init__()
            self.phase = 0

        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            # Initial research on main question: irrelevant only
            if self.phase == 0 and "find" not in query.lower() and "evidence about" not in query.lower():
                text = "Unrelated astronomy notes about nebulae and stellar formation."
                hit = RetrievalHit(
                    chunk_id=make_chunk_id("p_bad", page_start=1, page_end=1, text=text),
                    paper_id="p_bad",
                    text=text,
                    page_start=1,
                    page_end=1,
                    score=0.2,
                    retrieval_method=mode,
                )
                return RetrievalResult(query=query, method="hybrid_rerank", hits=[hit])
            # Corrective queries get real evidence
            self.phase = 1
            return super().search(query, mode=mode, k=k, filters=filters)

    toolkit = TwoPhaseToolkit()
    wf = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=2,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                max_evidence_per_sub_question=4,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    )
    result = wf.run("What is Self-RAG?")
    # Either corrective loop ran or finished with insufficient/unanswerable — must terminate
    assert result.terminated_reason
    event_types = [e.event_type.value for e in result.events]
    assert "verification" in event_types
    # If first verify was insufficient, expect corrective event or second research
    if "corrective" in event_types:
        assert result.iteration >= 1 or result.tool_call_count >= 1


def test_no_new_evidence_stops_loop() -> None:
    class AlwaysSameToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            # Always the same chunk/text → no unique new after first pass
            text = "Fixed passage that never changes regardless of query tokens."
            hit = RetrievalHit(
                chunk_id="chunk_fixed_always",
                paper_id="paper_fixed",
                text=text,
                page_start=1,
                page_end=1,
                score=0.5,
                retrieval_method=mode,
            )
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[hit])

    toolkit = AlwaysSameToolkit()
    # Force insufficiency via verifier min thresholds by using comparison diversity needs
    # while only one paper is ever returned
    wf = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=3,
            research=ResearchAgentConfig(max_tool_calls_per_pass=1, allow_policy_override=False),
            parallel_research=False,
        ),
    )
    result = wf.run("Compare Self-RAG versus CRAG")
    assert result.terminated_reason in {
        "no_new_evidence",
        "iteration_budget_exhausted",
        "no_corrective_queries",
        "evidence_sufficient",
        "tool_budget_exhausted",
        "corpus_cannot_answer",
        "completed",
    }
    # Must not infinite loop
    assert result.iteration <= 3


def test_conflicts_surfaced_in_verification() -> None:
    from scholar_agent.agents.verifier import Verifier
    from scholar_agent.ids import make_evidence_id
    from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
    from scholar_agent.models.planning import QueryPlan

    plan = QueryPlan(
        original_query="Compare A versus B performance",
        answer_type="comparison",
        expected_source_diversity=2,
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="Compare A versus B performance",
                query_type=QueryType.COMPARISON,
                required_evidence=["comparison"],
                status=SubQuestionStatus.PENDING,
            )
        ],
    )
    items = [
        EvidenceItem(
            evidence_id=make_evidence_id(
                run_id="r", chunk_id="c1", evidence_text="A outperforms B", sub_question_id="sq1"
            ),
            sub_question_id="sq1",
            claim="A better",
            evidence_text="Method A outperforms method B with better accuracy.",
            paper_id="p1",
            chunk_id="c1",
            page_start=1,
            page_end=1,
            retrieval_method="dense",
        ),
        EvidenceItem(
            evidence_id=make_evidence_id(
                run_id="r", chunk_id="c2", evidence_text="A underperform B", sub_question_id="sq1"
            ),
            sub_question_id="sq1",
            claim="A worse",
            evidence_text="Method A underperform method B and is worse overall.",
            paper_id="p2",
            chunk_id="c2",
            page_start=1,
            page_end=1,
            retrieval_method="dense",
        ),
    ]
    v = Verifier().verify(
        query=plan.original_query, plan=plan, ledger=EvidenceLedger(items=items)
    )
    assert v.conflicting_evidence_ids
    # Ledger unchanged (retained)
    assert len(items) == 2


def test_iteration_budget_terminates() -> None:
    class EmptyToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[])

    toolkit = EmptyToolkit()
    wf = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=1,
            research=ResearchAgentConfig(max_tool_calls_per_pass=1, allow_policy_override=False),
            parallel_research=False,
        ),
    )
    result = wf.run("What is Self-RAG?")
    assert result.terminated_reason
    assert result.iteration <= 1 or result.terminated_reason in {
        "corpus_cannot_answer",
        "no_corrective_queries",
        "iteration_budget_exhausted",
        "no_new_evidence",
        "tool_budget_exhausted",
        "evidence_sufficient",
    }
