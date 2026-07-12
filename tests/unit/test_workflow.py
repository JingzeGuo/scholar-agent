"""Phase 6 full workflow tests: corrective loop and termination."""

from __future__ import annotations

from time import sleep

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
    assert result.terminated_reason == "evidence_sufficient"
    # Should have a plan and verification
    assert result.plan.sub_questions
    assert result.verification is not None
    assert any(e.event_type.value == "plan_created" for e in result.events)
    assert any(e.event_type.value == "verification" for e in result.events)
    assert any(e.event_type.value == "answer_drafted" for e in result.events)
    assert any(e.event_type.value == "citation_validated" for e in result.events)
    assert any(e.event_type.value == "run_finished" for e in result.events)
    assert result.final_answer is not None
    assert result.final_answer.citation_report is not None
    # All final citations must exist in the ledger
    ledger_ids = {e.evidence_id for e in result.evidence_ledger.items}
    for claim in result.final_answer.claims:
        for eid in claim.evidence_ids:
            assert eid in ledger_ids


def test_missing_evidence_triggers_corrective_retrieval() -> None:
    """First research yields weak/empty; verifier asks corrective; second pass runs."""

    class TwoPhaseToolkit(ScriptedToolkit):
        def __init__(self) -> None:
            super().__init__()
            self.phase = 0

        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            # Initial research on main question: irrelevant only
            if self.phase == 0:
                self.phase = 1
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
    assert result.terminated_reason == "evidence_sufficient"
    event_types = [e.event_type.value for e in result.events]
    assert "verification" in event_types
    assert "corrective" in event_types
    assert result.iteration == 1
    original_id = result.plan.sub_questions[0].id
    assert any(item.sub_question_id == original_id for item in result.evidence_ledger.items)


def test_empty_first_pass_still_triggers_targeted_retrieval() -> None:
    class EmptyThenEvidenceToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            if len(self.calls) == 1:
                return RetrievalResult(query=query, method="hybrid_rerank", hits=[])
            return super().search(query, mode=mode, k=k, filters=filters)

    toolkit = EmptyThenEvidenceToolkit()
    result = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=2,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What is Self-RAG?")
    assert result.terminated_reason == "evidence_sufficient"
    assert result.iteration == 1
    assert any(event.event_type.value == "corrective" for event in result.events)


def test_no_new_evidence_stops_loop() -> None:
    class AlwaysSameToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            # Always the same chunk/text → no unique new after first pass
            text = "Self-RAG retrieves on demand using reflection tokens."
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
    assert result.terminated_reason == "no_new_evidence"
    assert result.iteration == 1


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
            max_corrective_iterations=0,
            research=ResearchAgentConfig(max_tool_calls_per_pass=1, allow_policy_override=False),
            parallel_research=False,
        ),
    )
    result = wf.run("What is Self-RAG?")
    assert result.terminated_reason == "iteration_budget_exhausted"
    assert result.iteration == 0


def test_unanswerable_after_targeted_retrieval_exhaustion() -> None:
    class ChangingIrrelevantToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            index = len(self.calls)
            self.calls.append(query)
            text = f"Astronomy observation {index} about nebulae and stellar formation."
            hit = RetrievalHit(
                chunk_id=f"chunk_irrelevant_{index}",
                paper_id=f"paper_irrelevant_{index}",
                text=text,
                page_start=1,
                page_end=1,
                score=0.1,
                retrieval_method=mode,
            )
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[hit])

    result = ResearchWorkflow(
        ChangingIrrelevantToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=1,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What is ZZZZ_NONEXISTENT_TOPIC_XYZ?")
    assert result.terminated_reason == "corpus_cannot_answer"
    assert result.unanswerable is True
    assert result.verification.unanswerable is True
    assert result.verification.corrective_queries == []
    assert result.final_answer is not None
    assert result.final_answer.claims == []
    assert result.final_answer.corpus_insufficient
    assert "Astronomy observation" not in result.final_answer.markdown


def test_global_tool_budget_is_never_exceeded() -> None:
    toolkit = ScriptedToolkit()
    result = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_total_tool_calls=2,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=2,
                allow_policy_override=False,
            ),
            parallel_research=True,
        ),
    ).run("Compare Self-RAG versus CRAG")
    assert result.tool_call_count == 2
    assert len(toolkit.calls) == 2
    assert result.terminated_reason == "tool_budget_exhausted"
    assert any(event.event_type.value == "budget_hit" for event in result.events)


def test_latency_budget_terminates_workflow() -> None:
    class SlowToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            sleep(0.005)
            self.calls.append(query)
            text = "Astronomy notes about nebulae and stellar formation."
            hit = RetrievalHit(
                chunk_id="chunk_slow_irrelevant",
                paper_id="paper_astronomy",
                text=text,
                page_start=1,
                page_end=1,
                score=0.1,
                retrieval_method=mode,
            )
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[hit])

    result = ResearchWorkflow(
        SlowToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_latency_ms=1,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What is a nonexistent retrieval method?")
    assert result.terminated_reason == "latency_budget_exhausted"
    assert any(event.event_type.value == "budget_hit" for event in result.events)


def test_workflow_writes_and_validates_citations() -> None:
    """Phase 7: after research terminates, Writer + citation validator run."""
    toolkit = ScriptedToolkit()
    result = ResearchWorkflow(
        toolkit,  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=1,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=2,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What is Self-RAG?")
    assert result.draft_answer is not None
    assert result.final_answer is not None
    assert result.final_answer.claims
    report = result.final_answer.citation_report
    assert report is not None
    assert report.is_valid
    # Source cards map to real paper + page
    for card in result.final_answer.source_cards:
        assert card.paper_id
        assert card.chunk_id
        assert card.page_start >= 1
        assert card.page_end >= card.page_start
        assert card.evidence_id in {e.evidence_id for e in result.evidence_ledger.items}
    # Inline citations appear in markdown
    assert "paper_" in result.final_answer.markdown
    finished = next(e for e in result.events if e.event_type.value == "run_finished")
    assert finished.payload.get("citation_valid") is True
    assert result.state is not None
    assert result.state.final_answer is not None
    assert result.state.citation_report is not None


def test_workflow_unanswerable_still_emits_answer_with_limitation() -> None:
    class EmptyToolkit(ScriptedToolkit):
        def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[])

    result = ResearchWorkflow(
        EmptyToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=0,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What is Self-RAG?")
    assert result.final_answer is not None
    assert result.final_answer.corpus_insufficient or "Limitation" in result.final_answer.markdown
    assert any(e.event_type.value == "answer_drafted" for e in result.events)
    assert any(e.event_type.value == "citation_validated" for e in result.events)
