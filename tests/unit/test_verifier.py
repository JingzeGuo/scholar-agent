"""Tests for independent Verifier."""

from __future__ import annotations

from scholar_agent.agents.planner import Planner
from scholar_agent.agents.verifier import Verifier
from scholar_agent.ids import make_evidence_id
from scholar_agent.models.base import QueryType
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus


def _item(
    *,
    run_id: str,
    sq: str,
    paper: str,
    chunk: str,
    text: str,
    claim: str | None = None,
    contradiction: bool = False,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=make_evidence_id(
            run_id=run_id, chunk_id=chunk, evidence_text=text, sub_question_id=sq
        ),
        sub_question_id=sq,
        claim=claim or text[:80],
        evidence_text=text,
        paper_id=paper,
        chunk_id=chunk,
        page_start=1,
        page_end=1,
        retrieval_method="dense",
        retrieval_score=0.8,
        contradiction=contradiction,
    )


def test_missing_evidence_produces_corrective_queries() -> None:
    plan = Planner().plan("What is Self-RAG?")
    sq_id = plan.sub_questions[0].id
    # Irrelevant evidence only
    ledger = EvidenceLedger(
        items=[
            _item(
                run_id="r1",
                sq=sq_id,
                paper="p_other",
                chunk="c1",
                text="The stock market rose sharply amid commodity trading news today.",
            )
        ]
    )
    result = Verifier().verify(query=plan.original_query, plan=plan, ledger=ledger)
    assert result.is_sufficient is False
    # Either targeted corrective queries, or explicit unanswerable (no relevant overlap)
    assert result.corrective_queries or "corpus_cannot_answer" in result.missing_aspects
    assert result.missing_sub_questions or result.missing_aspects


def test_sufficient_when_relevant_evidence_present() -> None:
    plan = Planner().plan("What is Self-RAG?")
    sq_id = plan.sub_questions[0].id
    ledger = EvidenceLedger(
        items=[
            _item(
                run_id="r1",
                sq=sq_id,
                paper="paper_self_rag",
                chunk="c_self",
                text=(
                    "Self-RAG is a framework that retrieves passages on demand "
                    "and critiques generation with reflection tokens."
                ),
            )
        ]
    )
    result = Verifier().verify(query=plan.original_query, plan=plan, ledger=ledger)
    assert result.is_sufficient is True
    assert sq_id in result.covered_sub_questions
    assert result.supported_evidence_ids[sq_id] == [ledger.items[0].evidence_id]
    assert result.coverage_score >= 0.85


def test_contradictory_evidence_retained_and_surfaced() -> None:
    plan = QueryPlan(
        original_query="Does method A outperform B?",
        answer_type="comparison",
        expected_source_diversity=2,
        sub_questions=[
            SubQuestion(
                id="sq_cmp",
                question="Does method A outperform B?",
                query_type=QueryType.COMPARISON,
                required_evidence=["comparison"],
                status=SubQuestionStatus.PENDING,
            )
        ],
    )
    ledger = EvidenceLedger(
        items=[
            _item(
                run_id="r1",
                sq="sq_cmp",
                paper="paper_a",
                chunk="c_a",
                text="Method A outperforms method B on the benchmark with better accuracy.",
            ),
            _item(
                run_id="r1",
                sq="sq_cmp",
                paper="paper_b",
                chunk="c_b",
                text="Method A underperform method B and is worse on the same benchmark.",
            ),
        ]
    )
    result = Verifier().verify(query=plan.original_query, plan=plan, ledger=ledger)
    # Both items remain in ledger (verifier does not drop them)
    assert len(ledger.items) == 2
    assert result.conflicting_evidence_ids
    assert len(result.conflicting_evidence_ids) >= 2


def test_opposite_words_on_unrelated_topics_are_not_conflicts() -> None:
    plan = Planner().plan("What improves retrieval accuracy?")
    sq_id = plan.sub_questions[0].id
    ledger = EvidenceLedger(
        items=[
            _item(
                run_id="r1",
                sq=sq_id,
                paper="paper_retrieval",
                chunk="c_retrieval",
                text="Method A is better for retrieval accuracy on the benchmark.",
            ),
            _item(
                run_id="r1",
                sq=sq_id,
                paper="paper_weather",
                chunk="c_weather",
                text="Winter weather becomes worse when temperatures decrease.",
            ),
        ]
    )
    result = Verifier().verify(query=plan.original_query, plan=plan, ledger=ledger)
    assert result.conflicting_evidence_ids == []


def test_empty_ledger_unanswerable_or_missing() -> None:
    plan = Planner().plan("What is Self-RAG?")
    result = Verifier().verify(query=plan.original_query, plan=plan, ledger=EvidenceLedger())
    assert result.is_sufficient is False
    assert result.missing_sub_questions
    assert result.unanswerable is False
    assert result.corrective_actions
    assert result.corrective_actions[0].target_sub_question_id == plan.sub_questions[0].id


def test_generic_definition_words_do_not_cover_unknown_topic() -> None:
    plan = Planner().plan("What is ZZZZ_NONEXISTENT_TOPIC_XYZ?")
    sq_id = plan.sub_questions[0].id
    ledger = EvidenceLedger(
        items=[
            _item(
                run_id="r1",
                sq=sq_id,
                paper="p_other",
                chunk="c_other",
                text=(
                    "This paper provides a factual definition of topic models "
                    "for retrieval evaluation."
                ),
            )
        ]
    )
    result = Verifier().verify(query=plan.original_query, plan=plan, ledger=ledger)
    assert result.is_sufficient is False
    assert sq_id in result.missing_sub_questions
    assert result.corrective_actions


def test_partial_coverage_emits_corrective_queries() -> None:
    """One of two sub-questions covered → concrete corrective query for the rest."""
    plan = QueryPlan(
        original_query="Compare Self-RAG versus CRAG",
        answer_type="comparison",
        expected_source_diversity=1,  # relax diversity for this unit test
        sub_questions=[
            SubQuestion(
                id="sq_a",
                question="What is Self-RAG?",
                query_type=QueryType.SEMANTIC,
                required_evidence=["definition"],
                status=SubQuestionStatus.PENDING,
            ),
            SubQuestion(
                id="sq_b",
                question="What is CRAG?",
                query_type=QueryType.SEMANTIC,
                required_evidence=["definition"],
                status=SubQuestionStatus.PENDING,
            ),
        ],
    )
    ledger = EvidenceLedger(
        items=[
            _item(
                run_id="r1",
                sq="sq_a",
                paper="paper_self",
                chunk="c_self",
                text="Self-RAG retrieves on demand using reflection tokens for critique.",
            )
        ]
    )
    result = Verifier(min_source_diversity=1).verify(
        query=plan.original_query, plan=plan, ledger=ledger
    )
    assert result.is_sufficient is False
    assert "sq_b" in result.missing_sub_questions
    assert result.corrective_queries
    assert all(action.target_sub_question_id == "sq_b" for action in result.corrective_actions)
    assert any("CRAG" in q or "crag" in q.lower() for q in result.corrective_queries)
