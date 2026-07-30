"""Phase 6 full workflow tests: corrective loop and termination."""

from __future__ import annotations

from time import sleep

import pytest

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

    def search(
        self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
    ) -> RetrievalResult:  # type: ignore[override]
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
        method = (
            mode if mode in {"dense", "sparse", "hybrid", "hybrid_rerank", "graph"} else "hybrid"
        )
        return RetrievalResult(query=query, method=method, hits=[hit])  # type: ignore[arg-type]


def test_workflow_terminates_when_sufficient() -> None:
    from scholar_agent.models.answer import AnswerStatus

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
    assert result.answer_status == AnswerStatus.COMPLETE
    assert result.final_answer.status == AnswerStatus.COMPLETE
    assert result.final_answer.corpus_insufficient is False
    assert result.verification.is_sufficient is True
    assert all(
        sub_question.status == SubQuestionStatus.COVERED
        for sub_question in result.plan.sub_questions
    )
    assert result.state is not None
    assert result.state.answer_status == AnswerStatus.COMPLETE
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

        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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
    initial_iteration = next(
        event
        for event in result.events
        if event.event_type.value == "iteration" and event.payload.get("iteration") == 0
    )
    assert initial_iteration.payload["evidence_chunk_ids"]
    assert initial_iteration.payload["evidence_paper_ids"] == ["p_bad"]


def test_empty_first_pass_still_triggers_targeted_retrieval() -> None:
    class EmptyThenEvidenceToolkit(ScriptedToolkit):
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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
    v = Verifier().verify(query=plan.original_query, plan=plan, ledger=EvidenceLedger(items=items))
    assert v.conflicting_evidence_ids
    # Ledger unchanged (retained)
    assert len(items) == 2


def test_iteration_budget_terminates() -> None:
    class EmptyToolkit(ScriptedToolkit):
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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
    assert result.answer_status.value == "insufficient"
    assert result.verification.is_sufficient is False
    assert result.terminated_reason == "corpus_cannot_answer"
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
    citation_event = next(
        event
        for event in result.events
        if event.event_type.value == "citation_validated"
    )
    assert citation_event.payload["terminated_reason"] == "tool_budget_exhausted"
    assert result.state is not None
    assert result.state.budgets.terminated_reason == "tool_budget_exhausted"


def test_final_downgrade_restores_tool_budget_masked_by_sufficient_verifier() -> None:
    from scholar_agent.agents.verifier import Verifier
    from scholar_agent.models.answer import AnswerStatus, DraftAnswer
    from scholar_agent.models.evidence import EvidenceLedger
    from scholar_agent.models.planning import QueryPlan
    from scholar_agent.models.workflow import VerificationResult

    class BroadSufficientVerifier(Verifier):
        def verify(
            self,
            *,
            query: str,
            plan: QueryPlan,
            ledger: EvidenceLedger,
        ) -> VerificationResult:
            del query
            evidence_ids = [item.evidence_id for item in ledger.items]
            return VerificationResult(
                is_sufficient=True,
                coverage_score=1.0,
                covered_sub_questions=[
                    sub_question.id for sub_question in plan.sub_questions
                ],
                supported_evidence_ids={
                    sub_question.id: evidence_ids
                    for sub_question in plan.sub_questions
                },
                rationale_summary="Broad verifier considered retrieval sufficient.",
            )

    class InsufficientWriter:
        llm = None
        strict_llm = False
        last_backend = "deterministic"
        last_model = None
        last_fallback_reason = None

        def write(self, **_kwargs):  # type: ignore[no-untyped-def]
            return DraftAnswer(status=AnswerStatus.INSUFFICIENT)

    result = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_total_tool_calls=2,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=2,
                allow_policy_override=False,
            ),
            parallel_research=True,
        ),
        verifier=BroadSufficientVerifier(),
        writer=InsufficientWriter(),  # type: ignore[arg-type]
    ).run("Compare Self-RAG versus CRAG")

    assert result.answer_status == AnswerStatus.INSUFFICIENT
    assert result.verification.is_sufficient is False
    assert result.terminated_reason == "tool_budget_exhausted"
    recovered_budget_event = next(
        event
        for event in result.events
        if event.event_type.value == "budget_hit"
        and event.component == "workflow_finalizer"
    )
    assert recovered_budget_event.summary == "tool_budget_exhausted"
    assert recovered_budget_event.payload["reconciled_after_final_answer"] is True
    verifier_event = next(
        event
        for event in result.events
        if event.event_type.value == "verification"
        and event.component == "verifier"
    )
    assert verifier_event.payload["is_sufficient"] is True
    assert (
        verifier_event.payload["termination_conditions"]["tool_budget_exhausted"]
        is True
    )
    citation_event = next(
        event
        for event in result.events
        if event.event_type.value == "citation_validated"
    )
    assert citation_event.payload["terminated_reason"] == "tool_budget_exhausted"
    assert result.state is not None
    assert result.state.budgets.terminated_reason == "tool_budget_exhausted"


def test_latency_budget_terminates_workflow() -> None:
    class SlowToolkit(ScriptedToolkit):
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
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


def test_global_token_budget_is_enforced_and_persisted() -> None:
    result = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_total_tokens=10,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=2,
                max_total_tokens_per_pass=100,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What is Self-RAG?")
    assert result.terminated_reason == "token_budget_exhausted"
    assert result.token_usage.total_tokens <= 10
    assert result.state is not None
    assert result.state.token_usage == result.token_usage
    assert result.state.budgets.max_total_tokens == 10
    assert any(event.event_type.value == "budget_hit" for event in result.events)


def test_auto_writer_degrades_without_llm_call_when_token_budget_is_exhausted() -> None:
    from scholar_agent.agents.writer import Writer

    class NeverCalledLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("Writer LLM must not run after token budget exhaustion")

    llm = NeverCalledLLM()
    result = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_total_tokens=10,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=2,
                max_total_tokens_per_pass=100,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
        writer=Writer(llm=llm),  # type: ignore[arg-type]
    ).run("What is Self-RAG?")

    assert llm.calls == 0
    assert result.terminated_reason == "token_budget_exhausted"
    writer_event = next(
        event for event in result.events if event.event_type.value == "answer_drafted"
    )
    assert writer_event.payload["backend"] == "deterministic"
    assert writer_event.payload["fallback_reason"] == "token_budget_exhausted"
    assert writer_event.payload["token_usage"]["total_tokens"] == 0
    assert result.token_usage.total_tokens <= 10


def test_strict_writer_fails_without_llm_call_when_token_budget_is_exhausted() -> None:
    from scholar_agent.agents.writer import Writer, WriterLLMError

    class NeverCalledLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("Writer LLM must not run after token budget exhaustion")

    llm = NeverCalledLLM()
    workflow = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_total_tokens=10,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=2,
                max_total_tokens_per_pass=100,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
        writer=Writer(llm=llm, strict_llm=True),  # type: ignore[arg-type]
    )

    with pytest.raises(
        WriterLLMError,
        match="global token budget is exhausted",
    ):
        workflow.run("What is Self-RAG?")
    assert llm.calls == 0


def test_private_named_future_fact_is_refused_after_targeted_exhaustion() -> None:
    class GenericTrainingToolkit(ScriptedToolkit):
        def search(
            self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None
        ) -> RetrievalResult:  # type: ignore[override]
            self.calls.append(query)
            text = (
                "A private training cluster uses GPU accelerators and retrieval methods; "
                "the public study reports a 2025 experiment."
            )
            hit = RetrievalHit(
                chunk_id="chunk_generic_training",
                paper_id="paper_generic_training",
                text=text,
                page_start=1,
                page_end=1,
                score=0.8,
                retrieval_method=mode,
            )
            return RetrievalResult(query=query, method="hybrid_rerank", hits=[hit])

    result = ResearchWorkflow(
        GenericTrainingToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=2,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
    ).run("What was AcmeAI's private 2027 GPU schedule for Model-X2?")
    assert result.terminated_reason == "corpus_cannot_answer"
    assert result.unanswerable
    assert result.final_answer is not None
    assert result.final_answer.corpus_insufficient
    assert result.final_answer.claims == []


def test_incomplete_verification_with_supported_evidence_is_partial() -> None:
    from scholar_agent.agents.verifier import Verifier
    from scholar_agent.models.answer import AnswerStatus
    from scholar_agent.models.evidence import EvidenceLedger
    from scholar_agent.models.planning import QueryPlan
    from scholar_agent.models.workflow import VerificationResult

    class PartialVerifier(Verifier):
        def verify(
            self,
            *,
            query: str,
            plan: QueryPlan,
            ledger: EvidenceLedger,
        ) -> VerificationResult:
            del query
            sub_question_id = plan.sub_questions[0].id
            evidence_ids = [item.evidence_id for item in ledger.items]
            return VerificationResult(
                is_sufficient=False,
                coverage_score=0.5,
                covered_sub_questions=[sub_question_id],
                supported_evidence_ids={sub_question_id: evidence_ids},
                missing_sub_questions=[sub_question_id],
                missing_aspects=["correction mechanism"],
                rationale_summary="Some verified evidence exists but one requirement is missing.",
            )

    result = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            max_corrective_iterations=0,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
        verifier=PartialVerifier(),
    ).run("What is Self-RAG?")

    assert result.verification.is_sufficient is False
    assert result.unanswerable is False
    assert result.draft_answer is not None
    assert result.final_answer is not None
    assert result.final_answer.claims
    assert result.draft_answer.status == AnswerStatus.PARTIAL
    assert result.final_answer.status == AnswerStatus.PARTIAL
    assert result.answer_status == AnswerStatus.PARTIAL
    assert result.state is not None
    assert result.state.answer_status == AnswerStatus.PARTIAL
    assert result.final_answer.corpus_insufficient is False
    assert result.terminated_reason != "evidence_sufficient"
    assert result.verification.is_sufficient is False
    assert all(
        sub_question.status == SubQuestionStatus.MISSING
        for sub_question in result.plan.sub_questions
    )
    final_verification_event = next(
        event
        for event in reversed(result.events)
        if event.event_type.value == "verification"
    )
    assert final_verification_event.payload["is_sufficient"] is False
    assert "Complete Answer" not in result.final_answer.core_answer


def test_citation_repair_downgrade_reconciles_public_workflow_state() -> None:
    from scholar_agent.models.answer import (
        AnswerStatus,
        ClaimWithCitations,
        DraftAnswer,
    )

    class OneInvalidClaimWriter:
        llm = None
        strict_llm = False
        last_backend = "deterministic"
        last_model = None
        last_fallback_reason = None

        def write(self, **kwargs):  # type: ignore[no-untyped-def]
            ledger = kwargs["ledger"]
            item = ledger.items[0]
            sub_question_id = kwargs["plan"].sub_questions[0].id
            return DraftAnswer(
                status=AnswerStatus.COMPLETE,
                claims=[
                    ClaimWithCitations(
                        claim_id="claim_valid",
                        text=item.claim,
                        evidence_ids=[item.evidence_id],
                        sub_question_id=sub_question_id,
                    ),
                    ClaimWithCitations(
                        claim_id="claim_invalid",
                        text="Quantum music occurs on Mars.",
                        evidence_ids=[item.evidence_id],
                        sub_question_id=sub_question_id,
                    ),
                ],
            )

    result = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
        writer=OneInvalidClaimWriter(),  # type: ignore[arg-type]
    ).run("What is Self-RAG?")

    assert result.draft_answer is not None
    assert result.draft_answer.status == AnswerStatus.COMPLETE
    assert result.final_answer is not None
    assert result.final_answer.status == AnswerStatus.PARTIAL
    assert result.final_answer.corpus_insufficient is False
    assert [claim.claim_id for claim in result.final_answer.claims] == [
        "claim_valid"
    ]
    assert result.answer_status == AnswerStatus.PARTIAL
    assert result.verification.is_sufficient is False
    assert result.verification.coverage_score == 0.0
    assert result.terminated_reason == "final_answer_partial"
    assert all(
        sub_question.status == SubQuestionStatus.MISSING
        for sub_question in result.plan.sub_questions
    )
    assert result.state is not None
    assert result.state.verification is not None
    assert result.state.verification.is_sufficient is False
    assert result.state.budgets.terminated_reason == "final_answer_partial"
    citation_event = next(
        event
        for event in result.events
        if event.event_type.value == "citation_validated"
    )
    assert citation_event.payload["verification_is_sufficient"] is False
    assert citation_event.payload["terminated_reason"] == "final_answer_partial"
    final_verification_event = next(
        event
        for event in reversed(result.events)
        if event.event_type.value == "verification"
    )
    assert final_verification_event.component == "workflow_finalizer"
    assert final_verification_event.payload["is_sufficient"] is False
    finished_event = next(
        event for event in result.events if event.event_type.value == "run_finished"
    )
    assert finished_event.payload["terminated_reason"] == "final_answer_partial"
    assert "Answer status:** partial" in result.final_answer.markdown


def test_final_comparison_coverage_includes_derived_difference_matrix() -> None:
    from scholar_agent.models.answer import (
        AnswerStatus,
        ClaimWithCitations,
        ComparisonCell,
        ComparisonRow,
        DraftAnswer,
        FinalAnswer,
    )
    from scholar_agent.models.evidence import EvidenceItem
    from scholar_agent.models.planning import (
        AnswerRequirement,
        PlannedEntity,
        QueryPlan,
    )
    from scholar_agent.models.workflow import VerificationResult

    entity_ids = ["self_rag", "corrective_rag"]
    requirements = [
        AnswerRequirement(
            key="retrieval_trigger",
            description="Retrieval trigger",
            target_entity_ids=entity_ids,
        ),
        AnswerRequirement(
            key="correction_mechanism",
            description="Correction mechanism",
            target_entity_ids=entity_ids,
        ),
        AnswerRequirement(
            key="key_differences",
            description="Key differences",
            target_entity_ids=entity_ids,
        ),
    ]
    sub_questions = [
        SubQuestion(
            id="sq_differences",
            question="What are the key differences?",
            query_type=QueryType.COMPARISON,
            target_entity_ids=entity_ids,
            requirement_keys=["key_differences"],
            dimension="key_differences",
        ),
        SubQuestion(
            id="sq_self_trigger",
            question="What triggers Self-RAG retrieval?",
            query_type=QueryType.SEMANTIC,
            target_entity_ids=["self_rag"],
            requirement_keys=["retrieval_trigger"],
            dimension="retrieval_trigger",
        ),
        SubQuestion(
            id="sq_crag_trigger",
            question="What triggers Corrective RAG retrieval?",
            query_type=QueryType.SEMANTIC,
            target_entity_ids=["corrective_rag"],
            requirement_keys=["retrieval_trigger"],
            dimension="retrieval_trigger",
        ),
        SubQuestion(
            id="sq_self_correction",
            question="How does Self-RAG correct generation?",
            query_type=QueryType.SEMANTIC,
            target_entity_ids=["self_rag"],
            requirement_keys=["correction_mechanism"],
            dimension="correction_mechanism",
        ),
        SubQuestion(
            id="sq_crag_correction",
            question="How does Corrective RAG correct retrieval?",
            query_type=QueryType.SEMANTIC,
            target_entity_ids=["corrective_rag"],
            requirement_keys=["correction_mechanism"],
            dimension="correction_mechanism",
        ),
    ]
    plan = QueryPlan(
        original_query="Compare Self-RAG versus CRAG.",
        answer_type="comparison",
        target_entities=[
            PlannedEntity(
                id="self_rag",
                surface_name="Self-RAG",
                canonical_name="Self-RAG",
            ),
            PlannedEntity(
                id="corrective_rag",
                surface_name="CRAG",
                canonical_name="Corrective RAG",
            ),
        ],
        answer_requirements=requirements,
        sub_questions=sub_questions,
    )
    evidence_specs = [
        (
            "ev_self_trigger",
            "sq_self_trigger",
            "Self-RAG uses reflection tokens to retrieve on demand.",
        ),
        (
            "ev_crag_trigger",
            "sq_crag_trigger",
            "Corrective RAG uses a retrieval evaluator to classify retrieval quality.",
        ),
        (
            "ev_self_correction",
            "sq_self_correction",
            "Self-RAG critiques and revises generated passages with reflection tokens.",
        ),
        (
            "ev_crag_correction",
            "sq_crag_correction",
            "Corrective RAG refines documents and can trigger web-search correction.",
        ),
    ]
    evidence = [
        EvidenceItem(
            evidence_id=evidence_id,
            sub_question_id=sub_question_id,
            claim=text,
            evidence_text=text,
            paper_id=f"paper_{evidence_id}",
            chunk_id=f"chunk_{evidence_id}",
            page_start=1,
            page_end=1,
            retrieval_method="hybrid",
            retrieval_score=0.9,
        )
        for evidence_id, sub_question_id, text in evidence_specs
    ]
    claims = [
        ClaimWithCitations(
            claim_id=f"claim_{index}",
            text=item.claim,
            evidence_ids=[item.evidence_id],
            sub_question_id=item.sub_question_id,
            entity_id=entity_id,
            requirement_key=requirement_key,
            dimension=requirement_key,
        )
        for index, (item, entity_id, requirement_key) in enumerate(
            zip(
                evidence,
                [
                    "self_rag",
                    "corrective_rag",
                    "self_rag",
                    "corrective_rag",
                ],
                [
                    "retrieval_trigger",
                    "retrieval_trigger",
                    "correction_mechanism",
                    "correction_mechanism",
                ],
                strict=True,
            ),
            start=1,
        )
    ]
    claims_by_binding = {
        (claim.requirement_key, claim.entity_id): claim for claim in claims
    }

    def supported_row(requirement_key: str, label: str) -> ComparisonRow:
        cells = []
        for entity_id, entity_label in [
            ("self_rag", "Self-RAG"),
            ("corrective_rag", "Corrective RAG"),
        ]:
            claim = claims_by_binding[(requirement_key, entity_id)]
            cells.append(
                ComparisonCell(
                    entity_id=entity_id,
                    entity_label=entity_label,
                    text=claim.text,
                    evidence_ids=claim.evidence_ids,
                    claim_id=claim.claim_id,
                    supported=True,
                )
            )
        return ComparisonRow(
            requirement_key=requirement_key,
            dimension=requirement_key,
            label=label,
            cells=cells,
        )

    draft = DraftAnswer(
        status=AnswerStatus.COMPLETE,
        claims=claims,
        rows=[
            supported_row("retrieval_trigger", "Retrieval trigger"),
            supported_row("correction_mechanism", "Correction mechanism"),
            ComparisonRow(
                requirement_key="key_differences",
                dimension="key_differences",
                label="Key differences",
                cells=[
                    ComparisonCell(
                        entity_id="self_rag",
                        entity_label="Self-RAG",
                    ),
                    ComparisonCell(
                        entity_id="corrective_rag",
                        entity_label="Corrective RAG",
                    ),
                ],
            ),
        ],
    )
    verification = VerificationResult(
        is_sufficient=True,
        coverage_score=1.0,
        covered_sub_questions=[sub_question.id for sub_question in sub_questions],
        supported_evidence_ids={
            item.sub_question_id: [item.evidence_id] for item in evidence
        },
        rationale_summary="All retrieval sub-questions were judged sufficient.",
    )
    state = {
        "run_id": "run_matrix",
        "plan": plan.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "verification": verification.model_dump(mode="json"),
        "draft_answer": draft.model_dump(mode="json"),
        "terminated_reason": "evidence_sufficient",
        "events": [],
    }

    update = ResearchWorkflow(ScriptedToolkit())._node_validate_citations(state)  # type: ignore[arg-type]
    final = FinalAnswer.model_validate(update["final_answer"])
    reconciled = VerificationResult.model_validate(update["verification"])
    reconciled_plan = QueryPlan.model_validate(update["plan"])

    assert final.status == AnswerStatus.PARTIAL
    assert final.corpus_insufficient is False
    assert reconciled.is_sufficient is False
    assert reconciled.coverage_score == 0.667
    assert reconciled.missing_sub_questions == ["sq_differences"]
    assert any(
        "key_differences" in aspect for aspect in reconciled.missing_aspects
    )
    assert update["terminated_reason"] == "final_answer_partial"
    assert next(
        sub_question
        for sub_question in reconciled_plan.sub_questions
        if sub_question.id == "sq_differences"
    ).status == SubQuestionStatus.MISSING
    assert all(
        sub_question.status == SubQuestionStatus.COVERED
        for sub_question in reconciled_plan.sub_questions
        if sub_question.id != "sq_differences"
    )
    finalizer_event = update["events"][-1]
    assert finalizer_event["component"] == "workflow_finalizer"
    assert finalizer_event["payload"]["is_sufficient"] is False
    assert finalizer_event["payload"]["terminated_reason"] == "final_answer_partial"


def test_workflow_traces_component_runtime_and_merges_llm_usage() -> None:
    from scholar_agent.agents.planner import Planner
    from scholar_agent.agents.writer import Writer
    from scholar_agent.models.base import TokenUsage

    class TracedPlanner:
        last_backend = "llm"
        last_model = "fast-model"
        last_fallback_reason = None
        last_fallback_fields: tuple[str, ...] = ()
        last_token_usage = TokenUsage(
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
        )

        def plan(self, query: str):  # type: ignore[no-untyped-def]
            return Planner().plan(query)

    class TracedWriter:
        last_backend = "deterministic"
        last_model = None
        last_fallback_reason = "provider: raw response must not be traced"
        last_fallback_fields = ("claims[0].evidence_ids", "private value")
        last_token_usage = TokenUsage(
            prompt_tokens=5,
            completion_tokens=4,
            total_tokens=9,
        )

        def write(self, **kwargs):  # type: ignore[no-untyped-def]
            return Writer().write(**kwargs)

    result = ResearchWorkflow(
        ScriptedToolkit(),  # type: ignore[arg-type]
        config=WorkflowConfig(
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=1,
                allow_policy_override=False,
            ),
            parallel_research=False,
        ),
        planner=TracedPlanner(),  # type: ignore[arg-type]
        writer=TracedWriter(),  # type: ignore[arg-type]
    ).run("What is Self-RAG?")

    plan_event = next(event for event in result.events if event.component == "planner")
    writer_event = next(
        event for event in result.events if event.event_type.value == "answer_drafted"
    )
    assert plan_event.payload["backend"] == "llm"
    assert plan_event.payload["model"] == "fast-model"
    assert plan_event.payload["token_usage"]["total_tokens"] == 10
    assert writer_event.payload["backend"] == "deterministic"
    assert writer_event.payload["fallback_reason"] == "component_fallback"
    assert writer_event.payload["fallback_fields"] == [
        "claims[0].evidence_ids",
        "private_value",
    ]
    assert writer_event.payload["token_usage"]["total_tokens"] == 9
    assert "raw response" not in str(writer_event.payload)
    assert result.token_usage.total_tokens >= 19
    assert result.token_usage.completion_tokens == 7
    assert result.state is not None
    assert result.state.token_usage == result.token_usage
