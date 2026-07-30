"""Phase 7 Writer: evidence-constrained claims and citation rendering."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scholar_agent.agents.planner import Planner
from scholar_agent.agents.writer import (
    Writer,
    WriterLLMError,
    format_inline_citation,
    render_claim_markdown,
)
from scholar_agent.llm.client import ChatResponse
from scholar_agent.llm.structured import (
    StructuredOutputError,
    StructuredOutputErrorCode,
)
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
    assert draft.corpus_insufficient is False
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
    assert empty.corpus_insufficient is False


def _phase11_comparison_fixture(
    *,
    include_crag_correction: bool = True,
    truncated_self_trigger: bool = False,
) -> tuple[QueryPlan, EvidenceLedger, dict[str, str]]:
    query = (
        "Compare Self-RAG versus CRAG. Explain their retrieval triggers, "
        "correction mechanisms, and key differences."
    )
    plan = Planner().plan(query)
    entity_ids = {entity.canonical_name: entity.id for entity in plan.target_entities}
    sub_question_ids = {
        (sub_question.target_entity_ids[0], sub_question.dimension): sub_question.id
        for sub_question in plan.sub_questions
        if len(sub_question.target_entity_ids) == 1
    }
    self_id = entity_ids["Self-RAG"]
    crag_id = entity_ids["Corrective RAG"]
    expected = {
        "self_trigger": "ev_self_trigger",
        "self_correction": "ev_self_correction",
        "crag_trigger": "ev_crag_trigger",
        "crag_correction": "ev_crag_correction",
        "direct_difference": "ev_direct_difference",
    }
    self_trigger = (
        "Self-RAG retrieves on demand using reflection tokens but the passage "
        "ends before completing the statement"
        if truncated_self_trigger
        else "Self-RAG uses reflection tokens to decide when to retrieve on demand."
    )
    items = [
        _item(
            evidence_id=expected["self_trigger"],
            sub_question_id=sub_question_ids[(self_id, "retrieval_trigger")],
            claim=self_trigger,
            evidence_text=self_trigger,
            paper_id="paper_arxiv_2310_11511",
            chunk_id="chunk_self_trigger",
        ),
        _item(
            evidence_id=expected["self_correction"],
            sub_question_id=sub_question_ids[(self_id, "correction_mechanism")],
            claim=("Self-RAG uses reflection tokens to critique and correct generated responses."),
            evidence_text=(
                "Self-RAG uses reflection tokens to critique and correct generated responses."
            ),
            paper_id="paper_arxiv_2310_11511",
            chunk_id="chunk_self_correction",
        ),
        _item(
            evidence_id=expected["crag_trigger"],
            sub_question_id=sub_question_ids[(crag_id, "retrieval_trigger")],
            claim=(
                "Corrective RAG (CRAG) uses a retrieval evaluator to classify "
                "retrieved documents as Correct, Ambiguous, or Incorrect."
            ),
            evidence_text=(
                "Corrective RAG (CRAG) uses a retrieval evaluator to classify "
                "retrieved documents as Correct, Ambiguous, or Incorrect."
            ),
            paper_id="paper_arxiv_2401_15884",
            chunk_id="chunk_crag_trigger",
        ),
        # A survey and the other method's primary paper may mention the target,
        # but neither may fill this target's primary-method cell.
        _item(
            evidence_id="ev_survey_junk",
            sub_question_id=sub_question_ids[(self_id, "correction_mechanism")],
            claim=("Self-RAG uses reflection tokens to critique and correct generated responses."),
            evidence_text=(
                "Self-RAG uses reflection tokens to critique and correct generated responses."
            ),
            paper_id="paper_rag_survey",
            chunk_id="chunk_survey",
        ),
        _item(
            evidence_id="ev_crag_mentions_self",
            sub_question_id=sub_question_ids[(self_id, "retrieval_trigger")],
            claim=("The CRAG paper says Self-RAG uses reflection tokens to retrieve on demand."),
            evidence_text=(
                "The CRAG paper says Self-RAG uses reflection tokens to retrieve on demand."
            ),
            paper_id="paper_arxiv_2401_15884",
            chunk_id="chunk_crag_mentions_self",
        ),
        _item(
            evidence_id="ev_acknowledgement",
            sub_question_id=sub_question_ids[(self_id, "retrieval_trigger")],
            claim=(
                "The Self-RAG authors would like to thank reviewers for retrieval "
                "trigger suggestions."
            ),
            evidence_text=(
                "The Self-RAG authors would like to thank reviewers for retrieval "
                "trigger suggestions."
            ),
            paper_id="paper_arxiv_2310_11511",
            chunk_id="chunk_acknowledgement",
        ),
        _item(
            evidence_id=expected["direct_difference"],
            sub_question_id="sq_direct_key_differences",
            claim=("Self-RAG and CRAG differ in their retrieval and correction strategies."),
            evidence_text=(
                "Self-RAG and CRAG differ in their retrieval and correction strategies."
            ),
            paper_id="paper_comparison_survey",
            chunk_id="chunk_direct_difference",
        ),
    ]
    if include_crag_correction:
        items.append(
            _item(
                evidence_id=expected["crag_correction"],
                sub_question_id=sub_question_ids[(crag_id, "correction_mechanism")],
                claim=(
                    "Corrective RAG (CRAG) refines retrieved documents and uses "
                    "web search to correct low-quality retrieval."
                ),
                evidence_text=(
                    "Corrective RAG (CRAG) refines retrieved documents and uses "
                    "web search to correct low-quality retrieval."
                ),
                paper_id="paper_arxiv_2401_15884",
                chunk_id="chunk_crag_correction",
            )
        )
    return plan, EvidenceLedger(items=items), expected


def test_phase11_writer_derives_explicit_differences_from_four_primary_cells() -> None:
    plan, ledger, expected = _phase11_comparison_fixture()

    draft = Writer().write(query=plan.original_query, plan=plan, ledger=ledger)

    rows = {row.requirement_key: row for row in draft.rows}
    assert list(rows) == [
        "retrieval_trigger",
        "correction_mechanism",
        "key_differences",
    ]
    assert all(cell.supported for row in rows.values() for cell in row.cells)
    assert draft.status == AnswerStatus.COMPLETE

    base_claims = {
        (claim.entity_id, claim.requirement_key): claim
        for claim in draft.claims
        if claim.requirement_key != "key_differences"
    }
    assert {claim.evidence_ids[0] for claim in base_claims.values()} == {
        expected["self_trigger"],
        expected["self_correction"],
        expected["crag_trigger"],
        expected["crag_correction"],
    }
    assert not {
        "ev_survey_junk",
        "ev_crag_mentions_self",
        "ev_acknowledgement",
        expected["direct_difference"],
    } & {evidence_id for claim in draft.claims for evidence_id in claim.evidence_ids}

    difference_claims = [
        claim for claim in draft.claims if claim.requirement_key == "key_differences"
    ]
    assert len(difference_claims) == 2
    expected_joint_ids = {
        expected["self_trigger"],
        expected["self_correction"],
        expected["crag_trigger"],
        expected["crag_correction"],
    }
    for claim in difference_claims:
        assert "differs from" in claim.text
        assert "Self-RAG" in claim.text
        assert "CRAG" in claim.text
        assert set(claim.evidence_ids) == expected_joint_ids


def test_phase11_writer_withholds_entire_difference_row_when_one_cell_missing() -> None:
    plan, ledger, _ = _phase11_comparison_fixture(include_crag_correction=False)

    draft = Writer().write(query=plan.original_query, plan=plan, ledger=ledger)

    rows = {row.requirement_key: row for row in draft.rows}
    assert [cell.supported for cell in rows["correction_mechanism"].cells] == [
        True,
        False,
    ]
    assert not any(cell.supported for cell in rows["key_differences"].cells)
    assert not any(claim.requirement_key == "key_differences" for claim in draft.claims)
    assert draft.status == AnswerStatus.PARTIAL


def test_phase11_writer_rejects_truncated_evidence_instead_of_excerpting_it() -> None:
    plan, ledger, _ = _phase11_comparison_fixture(truncated_self_trigger=True)

    draft = Writer().write(query=plan.original_query, plan=plan, ledger=ledger)

    rows = {row.requirement_key: row for row in draft.rows}
    assert rows["retrieval_trigger"].cells[0].supported is False
    assert not any(cell.supported for cell in rows["key_differences"].cells)
    assert "ends before completing" not in draft.core_answer


def test_phase11_writer_conservatively_rebinds_legacy_deduplicated_spans() -> None:
    plan, ledger, _ = _phase11_comparison_fixture()
    first_sub_question_id = plan.sub_questions[0].id
    legacy_ledger = EvidenceLedger(
        items=[
            item.model_copy(update={"sub_question_id": first_sub_question_id})
            for item in ledger.items
        ]
    )

    draft = Writer().write(
        query=plan.original_query,
        plan=plan,
        ledger=legacy_ledger,
    )

    planned_pairs = {
        (sub_question.target_entity_ids[0], sub_question.requirement_keys[0]): sub_question.id
        for sub_question in plan.sub_questions
        if (len(sub_question.target_entity_ids) == 1 and len(sub_question.requirement_keys) == 1)
    }
    base_claims = [claim for claim in draft.claims if claim.requirement_key != "key_differences"]
    assert len(base_claims) == 4
    assert all(
        claim.sub_question_id == planned_pairs[(claim.entity_id, claim.requirement_key)]
        for claim in base_claims
    )
    assert draft.status == AnswerStatus.COMPLETE


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
    messages = llm.calls[0]["messages"]
    assert isinstance(messages, list)
    system_prompt = messages[0].content
    assert '"claims"' in system_prompt
    assert '"rows"' in system_prompt
    assert "entity_1" in system_prompt
    assert "entity_2" in system_prompt
    assert "ev_self" in system_prompt
    assert "ev_crag" in system_prompt
    assert "entity_without_evidence" not in system_prompt
    assert "ev_allowed" not in system_prompt
    user_payload = json.loads(messages[1].content or "{}")
    assert {
        (
            binding["entity_id"],
            binding["requirement_key"],
            tuple(binding["allowed_evidence_ids"]),
        )
        for binding in user_payload["allowed_bindings"]
    } == {
        ("entity_1", "overview", ("ev_self",)),
        ("entity_2", "overview", ("ev_crag",)),
    }
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
    assert writer.last_fallback_reason == "unknown_evidence_id"
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
    assert writer.last_fallback_reason == "json_decode_failed"


def test_strict_llm_writer_reports_missing_top_level_field() -> None:
    plan, ledger = _two_sided_comparison()
    writer = Writer(
        llm=_FakeWriterLLM(json.dumps({"claims": []})),  # type: ignore[arg-type]
        strict_llm=True,
    )

    with pytest.raises(WriterLLMError) as exc_info:
        writer.write(query=plan.original_query, plan=plan, ledger=ledger)

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == StructuredOutputErrorCode.MISSING_REQUIRED_FIELD
    assert cause.field_paths == ("rows",)
    assert writer.last_fallback_reason == "missing_required_field"
    assert writer.last_fallback_fields == ("rows",)


@pytest.mark.parametrize(
    ("field_name", "private_value", "expected_code"),
    [
        (
            "entity_id",
            "private_unknown_entity",
            StructuredOutputErrorCode.UNKNOWN_ENTITY_ID,
        ),
        (
            "requirement_key",
            "private_unknown_requirement",
            StructuredOutputErrorCode.UNKNOWN_REQUIREMENT_KEY,
        ),
    ],
)
def test_strict_llm_writer_classifies_unknown_binding_ids(
    field_name: str,
    private_value: str,
    expected_code: StructuredOutputErrorCode,
) -> None:
    plan, ledger = _two_sided_comparison()
    payload = _valid_llm_comparison_payload()
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims[0][field_name] = private_value
    writer = Writer(
        llm=_FakeWriterLLM(json.dumps(payload)),  # type: ignore[arg-type]
        strict_llm=True,
    )

    with pytest.raises(WriterLLMError) as exc_info:
        writer.write(query=plan.original_query, plan=plan, ledger=ledger)

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == expected_code
    assert cause.field_paths == (f"claims[0].{field_name}",)
    assert private_value not in str(cause)
    assert writer.last_fallback_reason == expected_code.value


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

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED
    assert cause.field_paths == ("claims[0].evidence_ids",)


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

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED
    assert cause.field_paths == ("claims",)


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
