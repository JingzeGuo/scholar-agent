"""Schema round-trip and validation tests for Phase 1 domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_agent.ids import make_chunk_id, make_entity_id, make_paper_id, make_relation_id
from scholar_agent.models import (
    Chunk,
    CorpusManifestEntry,
    Entity,
    EntityType,
    EvidenceItem,
    EvidenceLedger,
    Paper,
    QueryPlan,
    Relation,
    RelationType,
    ResearchRunState,
    SubQuestion,
    VerificationResult,
)
from scholar_agent.models.base import QueryType
from scholar_agent.models.planning import SubQuestionStatus


def test_paper_and_chunk_round_trip() -> None:
    paper = Paper(
        paper_id=make_paper_id(arxiv_id="2310.11511"),
        title="Self-RAG",
        authors=["Asai et al."],
        year=2024,
        arxiv_id="2310.11511",
        pdf_path="data/papers/self_rag.pdf",
        content_hash="abc123def4567890",
    )
    data = paper.model_dump()
    assert Paper.model_validate(data) == paper

    text = "Self-RAG retrieves on demand."
    chunk = Chunk(
        chunk_id=make_chunk_id(paper.paper_id, page_start=1, page_end=1, text=text),
        paper_id=paper.paper_id,
        text=text,
        page_start=1,
        page_end=1,
        section="Abstract",
        token_count=6,
        content_hash="feedface01234567",
    )
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_chunk_invalid_page_range() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk_x",
            paper_id="paper_x",
            text="hello",
            page_start=5,
            page_end=2,
            token_count=1,
            content_hash="0123456789abcdef",
        )


def test_manifest_entry_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        CorpusManifestEntry(
            paper_id="paper_x",
            title="   ",
            pdf_filename="x.pdf",
            content_hash="0123456789abcdef",
        )


def test_manifest_entry_rejects_bad_year() -> None:
    with pytest.raises(ValidationError):
        CorpusManifestEntry(
            paper_id="paper_x",
            title="Title",
            year=1200,
            pdf_filename="x.pdf",
            content_hash="0123456789abcdef",
        )


def test_query_plan_unique_subquestion_ids() -> None:
    sq = SubQuestion(
        id="sq_1",
        question="What is Self-RAG?",
        query_type=QueryType.SEMANTIC,
        required_evidence=["definition"],
    )
    plan = QueryPlan(
        original_query="Explain Self-RAG",
        answer_type="definition",
        sub_questions=[sq],
        expected_source_diversity=1,
    )
    assert plan.sub_questions[0].status == SubQuestionStatus.PENDING

    with pytest.raises(ValidationError):
        QueryPlan(
            original_query="q",
            answer_type="a",
            sub_questions=[sq, sq.model_copy()],
        )


def test_evidence_ledger_dedupes_by_chunk_and_span() -> None:
    a = EvidenceItem.build(
        run_id="run_1",
        sub_question_id="sq_1",
        claim="Self-RAG uses reflection",
        evidence_text="uses reflection tokens",
        paper_id="paper_a",
        chunk_id="chunk_a",
        page_start=3,
        page_end=3,
        retrieval_method="dense",
        retrieval_score=0.5,
    )
    b = EvidenceItem.build(
        run_id="run_1",
        sub_question_id="sq_1",
        claim="Self-RAG uses reflection",
        evidence_text="  Uses Reflection Tokens  ",
        paper_id="paper_a",
        chunk_id="chunk_a",
        page_start=3,
        page_end=3,
        retrieval_method="hybrid",
        retrieval_score=0.9,
    )
    ledger = EvidenceLedger().merge(a).merge(b)
    assert len(ledger.items) == 1
    assert ledger.items[0].retrieval_score == 0.9
    assert ledger.items[0].retrieval_method == "hybrid"


def test_relation_requires_evidence_span() -> None:
    with pytest.raises(ValidationError):
        Relation(
            relation_id="rel_x",
            subject_surface="Self-RAG",
            object_surface="open-domain QA",
            relation_type=RelationType.PROPOSES,
            evidence_span="   ",
            paper_id="paper_a",
            chunk_id="chunk_a",
            page_number=1,
            confidence=0.8,
        )


def test_entity_relation_round_trip() -> None:
    ent = Entity(
        entity_id=make_entity_id("Method", "Self-RAG"),
        entity_type=EntityType.METHOD,
        canonical_name="Self-RAG",
        aliases=["Self RAG", "self-rag"],
    )
    rel = Relation(
        relation_id=make_relation_id(
            subject_entity_id=ent.entity_id,
            relation_type="PROPOSES",
            object_entity_id=make_entity_id("Task", "ODQA"),
            chunk_id="chunk_a",
            evidence_span="Self-RAG proposes a retrieve-on-demand framework",
        ),
        subject_surface="Self-RAG",
        object_surface="open-domain QA",
        subject_entity_id=ent.entity_id,
        object_entity_id=make_entity_id("Task", "ODQA"),
        subject_type=EntityType.METHOD,
        object_type=EntityType.TASK,
        relation_type=RelationType.PROPOSES,
        evidence_span="Self-RAG proposes a retrieve-on-demand framework",
        paper_id="paper_a",
        chunk_id="chunk_a",
        page_number=3,
        confidence=0.91,
    )
    assert Entity.model_validate(ent.model_dump()) == ent
    assert Relation.model_validate_json(rel.model_dump_json()) == rel


def test_verification_and_run_state_round_trip() -> None:
    verification = VerificationResult(
        is_sufficient=False,
        coverage_score=0.4,
        covered_sub_questions=["sq_1"],
        missing_sub_questions=["sq_2"],
        missing_aspects=["comparison metrics"],
        corrective_queries=["Self-RAG vs CRAG metrics"],
        rationale_summary="Missing comparative metrics evidence",
    )
    state = ResearchRunState(
        run_id="run_fixture_1",
        query="Compare Self-RAG and CRAG",
        verification=verification,
    )
    restored = ResearchRunState.model_validate_json(state.model_dump_json())
    assert restored.verification is not None
    assert restored.verification.coverage_score == 0.4


def test_paper_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        Paper(
            paper_id="",
            title="x",
            pdf_path="a.pdf",
            content_hash="0123456789abcdef",
        )
