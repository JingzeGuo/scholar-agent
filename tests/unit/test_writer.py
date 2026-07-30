"""Phase 7 Writer: evidence-constrained claims and citation rendering."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scholar_agent.agents.writer import (
    Writer,
    WriterLLMError,
    format_inline_citation,
    render_claim_markdown,
)
from scholar_agent.llm.client import ChatResponse
from scholar_agent.llm.structured import StructuredOutputError
from scholar_agent.models.answer import (
    AnswerStatus,
    ClaimWithCitations,
    ComparisonCell,
    ComparisonRow,
    DraftAnswer,
)
from scholar_agent.models.base import QueryType, TokenUsage
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import VerificationResult


def _item(
    *,
    evidence_id: str,
    sub_question_id: str = "sq_0",
    claim: str = "Self-RAG retrieves on demand.",
    evidence_text: str = "Self-RAG retrieves on demand and uses reflection tokens.",
    paper_id: str = "paper_self_rag",
    chunk_id: str = "chunk_a",
    page_start: int = 3,
    page_end: int = 3,
    contradiction: bool = False,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        sub_question_id=sub_question_id,
        claim=claim,
        evidence_text=evidence_text,
        paper_id=paper_id,
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_end,
        retrieval_method="hybrid_rerank",
        retrieval_score=0.9,
        contradiction=contradiction,
    )


def test_writer_only_uses_ledger_evidence() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_1",
                claim="Self-RAG uses reflection tokens for critique.",
                evidence_text=(
                    "Self-RAG retrieves on demand and uses reflection tokens "
                    "to critique generation quality."
                ),
            )
        ]
    )
    plan = QueryPlan(
        original_query="What is Self-RAG?",
        answer_type="factual",
        sub_questions=[
            SubQuestion(
                id="sq_0",
                question="What is Self-RAG?",
                query_type=QueryType.SEMANTIC,
                required_evidence=["definition"],
                status=SubQuestionStatus.COVERED,
            )
        ],
    )
    draft = Writer().write(query="What is Self-RAG?", plan=plan, ledger=ledger)
    assert draft.claims
    assert all(c.evidence_ids for c in draft.claims)
    assert all(eid == "ev_1" for c in draft.claims for eid in c.evidence_ids)
    assert "paper_self_rag" in draft.markdown
    assert "p.3" in draft.markdown
    # Must not invent papers outside the ledger
    assert "paper_hallucinated" not in draft.markdown


def test_writer_empty_ledger_states_limitation() -> None:
    draft = Writer().write(
        query="What is Self-RAG?",
        ledger=EvidenceLedger(),
        corpus_insufficient=True,
    )
    assert draft.corpus_insufficient
    assert not draft.claims
    assert "Limitation" in draft.markdown or "cannot answer" in draft.markdown.lower()


def test_writer_excludes_evidence_verifier_did_not_accept() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_irrelevant",
                claim="Nebulae form from interstellar gas.",
                evidence_text="Nebulae form from interstellar gas and dust.",
                paper_id="paper_astronomy",
            )
        ]
    )
    verification = VerificationResult(
        is_sufficient=False,
        coverage_score=0.0,
        missing_sub_questions=["sq_0"],
        supported_evidence_ids={},
        unanswerable=True,
        rationale_summary="The corpus cannot answer the question.",
    )
    draft = Writer().write(
        query="What is Self-RAG?",
        ledger=ledger,
        verification=verification,
    )
    assert draft.claims == []
    assert draft.corpus_insufficient
    assert "Nebulae" not in draft.markdown


def test_writer_surfaces_contradictions() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_yes",
                claim="Method A outperforms baselines.",
                evidence_text="Method A outperforms baselines on HotpotQA.",
                paper_id="paper_a",
                chunk_id="chunk_yes",
            ),
            _item(
                evidence_id="ev_no",
                claim="Method A underperforms baselines.",
                evidence_text="Method A underperforms baselines on HotpotQA.",
                paper_id="paper_b",
                chunk_id="chunk_no",
                contradiction=True,
            ),
        ]
    )
    verification = VerificationResult(
        is_sufficient=True,
        coverage_score=1.0,
        covered_sub_questions=["sq_0"],
        conflicting_evidence_ids=["ev_yes", "ev_no"],
        rationale_summary="Evidence present with conflicts retained.",
    )
    draft = Writer().write(
        query="Does Method A work?",
        ledger=ledger,
        verification=verification,
    )
    assert any(
        "disagree" in c.text.lower() or "conflict" in c.text.lower() for c in draft.claims
    ) or any("conflict" in n.lower() for n in draft.notes)
    assert "ev_yes" in draft.markdown or "paper_a" in draft.markdown


def test_inline_citation_from_evidence_ids() -> None:
    item = _item(evidence_id="ev_x", page_start=2, page_end=4, paper_id="paper_x")
    assert format_inline_citation(item) == "[paper_x p.2-4]"
    claim = ClaimWithCitations(
        claim_id="claim_1",
        text="CRAG triggers corrective retrieval.",
        evidence_ids=["ev_x"],
    )
    rendered = render_claim_markdown(claim, {"ev_x": item})
    assert "[paper_x p.2-4]" in rendered
    assert "CRAG triggers corrective retrieval." in rendered


def test_writer_claim_ids_stable_and_ordered() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_a",
                sub_question_id="sq_0",
                paper_id="paper_a",
                chunk_id="c1",
                claim="Self-RAG uses reflection tokens.",
                evidence_text="Self-RAG uses reflection tokens for critique.",
            ),
            _item(
                evidence_id="ev_b",
                sub_question_id="sq_1",
                paper_id="paper_b",
                chunk_id="c2",
                claim="CRAG evaluates retrieved documents.",
                evidence_text="CRAG evaluates retrieved documents and triggers corrective retrieval.",
            ),
        ]
    )
    plan = QueryPlan(
        original_query="Compare Self-RAG versus CRAG",
        answer_type="comparison",
        sub_questions=[
            SubQuestion(
                id="sq_0",
                question="What is Self-RAG?",
                query_type=QueryType.COMPARISON,
            ),
            SubQuestion(
                id="sq_1",
                question="What is CRAG?",
                query_type=QueryType.COMPARISON,
            ),
        ],
    )
    draft = Writer().write(query="Compare Self-RAG versus CRAG", plan=plan, ledger=ledger)
    assert len(draft.claims) >= 2
    assert draft.claims[0].claim_id == "claim_1"
    assert draft.claims[1].claim_id == "claim_2"


def test_writer_prefers_relevant_sentence_over_pdf_front_matter() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_front",
                claim="SELF-RAG PAPER AUTHORS AND AFFILIATIONS",
                evidence_text=(
                    "SELF-RAG PAPER AUTHORS AND AFFILIATIONS University Example. "
                    "Abstract Self-RAG uses reflection tokens to decide when to "
                    "retrieve and to critique generated passages. "
                    "The appendix contains implementation details."
                ),
            )
        ]
    )
    draft = Writer().write(
        query="How does Self-RAG use reflection tokens for retrieval and critique?",
        ledger=ledger,
    )
    assert draft.claims
    assert "reflection tokens" in draft.claims[0].text.lower()
    assert "affiliations" not in draft.claims[0].text.lower()


def test_comparison_writer_builds_matrix_and_marks_partial_cells() -> None:
    plan = QueryPlan(
        original_query="Compare Self-RAG versus CRAG",
        answer_type="comparison",
        sub_questions=[
            SubQuestion(
                id="sq_self",
                question="What is Self-RAG?",
                query_type=QueryType.COMPARISON,
            ),
            SubQuestion(
                id="sq_crag",
                question="What is CRAG?",
                query_type=QueryType.COMPARISON,
            ),
        ],
    )
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_self",
                sub_question_id="sq_self",
                paper_id="paper_self",
                claim="Self-RAG uses reflection tokens to retrieve on demand.",
                evidence_text="Self-RAG uses reflection tokens to retrieve on demand.",
            )
        ]
    )
    verification = VerificationResult(
        is_sufficient=False,
        coverage_score=0.5,
        supported_evidence_ids={"sq_self": ["ev_self"]},
        missing_sub_questions=["sq_crag"],
        rationale_summary="Only one comparison side is supported.",
    )

    draft = Writer().write(
        query=plan.original_query,
        plan=plan,
        ledger=ledger,
        verification=verification,
    )

    assert draft.status == AnswerStatus.PARTIAL
    assert draft.corpus_insufficient
    assert len(draft.rows) == 1
    assert [cell.supported for cell in draft.rows[0].cells] == [True, False]
    assert "Insufficient verified evidence" in draft.core_answer
    assert "| Overview |" in draft.core_answer


def test_legacy_corpus_insufficient_payload_migrates_to_answer_status() -> None:
    partial = DraftAnswer.model_validate(
        {
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "A supported fragment.",
                    "evidence_ids": ["ev_1"],
                }
            ],
            "corpus_insufficient": True,
        }
    )
    empty = DraftAnswer.model_validate({"corpus_insufficient": False})
    assert partial.status == AnswerStatus.PARTIAL
    assert partial.corpus_insufficient
    assert empty.status == AnswerStatus.INSUFFICIENT
    assert empty.corpus_insufficient


class _FakeWriterLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat_json(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, **kwargs})
        return ChatResponse(
            content=self.content,
            model="fake-main-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _two_sided_comparison() -> tuple[QueryPlan, EvidenceLedger]:
    plan = QueryPlan(
        original_query="Compare Self-RAG versus CRAG",
        answer_type="comparison",
        sub_questions=[
            SubQuestion(
                id="sq_self",
                question="What is Self-RAG?",
                query_type=QueryType.COMPARISON,
            ),
            SubQuestion(
                id="sq_crag",
                question="What is CRAG?",
                query_type=QueryType.COMPARISON,
            ),
        ],
    )
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_self",
                sub_question_id="sq_self",
                paper_id="paper_self",
                chunk_id="chunk_self",
                claim="Self-RAG retrieves on demand.",
                evidence_text="Self-RAG retrieves on demand using reflection tokens.",
            ),
            _item(
                evidence_id="ev_crag",
                sub_question_id="sq_crag",
                paper_id="paper_crag",
                chunk_id="chunk_crag",
                claim="CRAG evaluates retrieved documents.",
                evidence_text="CRAG evaluates retrieved documents before correction.",
            ),
        ]
    )
    return plan, ledger


def _valid_llm_comparison_payload() -> dict[str, Any]:
    return {
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "Self-RAG retrieves on demand.",
                "evidence_ids": ["ev_self"],
                "sub_question_id": "sq_self",
                "requirement_key": "overview",
                "entity_id": "entity_1",
                "dimension": "overview",
            },
            {
                "claim_id": "claim_2",
                "text": "CRAG evaluates retrieved documents.",
                "evidence_ids": ["ev_crag"],
                "sub_question_id": "sq_crag",
                "requirement_key": "overview",
                "entity_id": "entity_2",
                "dimension": "overview",
            },
        ],
        "rows": [
            {
                "requirement_key": "overview",
                "dimension": "overview",
                "label": "Overview",
                "cells": [
                    {
                        "entity_id": "entity_1",
                        "entity_label": "Self-RAG",
                        "text": "Self-RAG retrieves on demand.",
                        "evidence_ids": ["ev_self"],
                        "claim_id": "claim_1",
                        "supported": True,
                    },
                    {
                        "entity_id": "entity_2",
                        "entity_label": "CRAG",
                        "text": "CRAG evaluates retrieved documents.",
                        "evidence_ids": ["ev_crag"],
                        "claim_id": "claim_2",
                        "supported": True,
                    },
                ],
            }
        ],
    }


def test_structured_llm_writer_uses_main_model_and_records_safe_metadata() -> None:
    plan, ledger = _two_sided_comparison()
    payload = _valid_llm_comparison_payload()
    llm = _FakeWriterLLM(json.dumps(payload))
    writer = Writer(llm=llm)  # type: ignore[arg-type]

    draft = writer.write(query=plan.original_query, plan=plan, ledger=ledger)

    assert draft.status == AnswerStatus.COMPLETE
    assert writer.last_backend == "llm"
    assert writer.last_model == "fake-main-model"
    assert writer.last_fallback_reason is None
    assert writer.last_token_usage.total_tokens == 15
    assert llm.calls[0]["fast"] is False
    assert "raw" not in writer.__dict__


def test_llm_writer_rejects_unknown_evidence_and_falls_back() -> None:
    plan, ledger = _two_sided_comparison()
    bad = {
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "Invented.",
                "evidence_ids": ["ev_unknown"],
            }
        ],
        "rows": [],
    }
    writer = Writer(llm=_FakeWriterLLM(json.dumps(bad)))  # type: ignore[arg-type]

    draft = writer.write(query=plan.original_query, plan=plan, ledger=ledger)

    assert writer.last_backend == "deterministic"
    assert writer.last_fallback_reason == "structured_output_invalid"
    assert {eid for claim in draft.claims for eid in claim.evidence_ids} == {
        "ev_self",
        "ev_crag",
    }
    assert "ev_unknown" not in draft.markdown


def test_strict_llm_writer_propagates_malformed_structured_output() -> None:
    plan, ledger = _two_sided_comparison()
    writer = Writer(
        llm=_FakeWriterLLM("not-json"),  # type: ignore[arg-type]
        strict_llm=True,
    )
    with pytest.raises(WriterLLMError) as exc_info:
        writer.write(query=plan.original_query, plan=plan, ledger=ledger)
    assert isinstance(exc_info.value.__cause__, StructuredOutputError)
    assert writer.last_backend == "llm"
    assert writer.last_fallback_reason == "structured_output_invalid"


def test_llm_writer_rejects_evidence_bound_to_wrong_entity_and_subquestion() -> None:
    plan, ledger = _two_sided_comparison()
    payload = _valid_llm_comparison_payload()
    claims = payload["claims"]
    rows = payload["rows"]
    assert isinstance(claims, list)
    assert isinstance(rows, list)
    claims[0]["evidence_ids"] = ["ev_crag"]
    rows[0]["cells"][0]["evidence_ids"] = ["ev_crag"]
    writer = Writer(
        llm=_FakeWriterLLM(json.dumps(payload)),  # type: ignore[arg-type]
        strict_llm=True,
    )

    with pytest.raises(WriterLLMError) as exc_info:
        writer.write(query=plan.original_query, plan=plan, ledger=ledger)

    assert isinstance(exc_info.value.__cause__, StructuredOutputError)
    assert "bound" in str(exc_info.value.__cause__)


def test_llm_writer_rejects_claim_count_over_limit_instead_of_truncating() -> None:
    plan, ledger = _two_sided_comparison()
    writer = Writer(
        max_claims=1,
        llm=_FakeWriterLLM(  # type: ignore[arg-type]
            json.dumps(_valid_llm_comparison_payload())
        ),
        strict_llm=True,
    )

    with pytest.raises(WriterLLMError) as exc_info:
        writer.write(query=plan.original_query, plan=plan, ledger=ledger)

    assert isinstance(exc_info.value.__cause__, StructuredOutputError)
    assert "max_claims" in str(exc_info.value.__cause__)


def test_force_deterministic_skips_llm_and_records_budget_reason() -> None:
    plan, ledger = _two_sided_comparison()
    llm = _FakeWriterLLM("must not be called")
    writer = Writer(llm=llm, strict_llm=True)  # type: ignore[arg-type]

    draft = writer.write(
        query=plan.original_query,
        plan=plan,
        ledger=ledger,
        force_deterministic=True,
        forced_fallback_reason="token_budget_exhausted",
    )

    assert draft.claims
    assert llm.calls == []
    assert writer.last_backend == "deterministic"
    assert writer.last_fallback_reason == "token_budget_exhausted"
    assert writer.last_token_usage == TokenUsage()


def test_answer_status_ignores_supported_cell_when_bound_claim_is_missing() -> None:
    claim = ClaimWithCitations(
        claim_id="claim_real",
        text="Self-RAG retrieves on demand.",
        evidence_ids=["ev_self"],
    )
    row = ComparisonRow(
        requirement_key="overview",
        dimension="overview",
        label="Overview",
        cells=[
            ComparisonCell(
                entity_id="entity_1",
                entity_label="Self-RAG",
                text=claim.text,
                evidence_ids=["ev_self"],
                claim_id="claim_missing",
                supported=True,
            )
        ],
    )

    status = Writer()._answer_status(
        claims=[claim],
        rows=[row],
        verification=None,
        explicitly_insufficient=False,
    )

    assert status == AnswerStatus.INSUFFICIENT
