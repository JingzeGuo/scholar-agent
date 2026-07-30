"""Phase 7 citation validator: ID existence, page provenance, support checks."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from scholar_agent.agents.citation_validator import CitationValidator
from scholar_agent.models.answer import (
    AnswerStatus,
    ClaimWithCitations,
    ComparisonCell,
    ComparisonRow,
    DraftAnswer,
)
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.retrieval.chunk_store import ChunkStore


def _item(
    *,
    evidence_id: str,
    claim: str = "Self-RAG retrieves on demand.",
    evidence_text: str = "Self-RAG retrieves on demand and uses reflection tokens.",
    paper_id: str = "paper_self_rag",
    chunk_id: str = "chunk_a",
    page_start: int = 3,
    page_end: int = 3,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        sub_question_id="sq_0",
        claim=claim,
        evidence_text=evidence_text,
        paper_id=paper_id,
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_end,
        retrieval_method="hybrid",
        retrieval_score=0.8,
    )


def _strict_validator(
    tmp_path: Path,
    item: EvidenceItem,
    *,
    actual_pages: int = 1,
    declared_pages: int | None = None,
    chunk_text: str | None = None,
) -> tuple[CitationValidator, Path]:
    pdf_path = tmp_path / f"{item.paper_id}.pdf"
    document = pymupdf.open()
    for _ in range(actual_pages):
        document.new_page()
    document.save(pdf_path)
    document.close()
    chunk = Chunk(
        chunk_id=item.chunk_id,
        paper_id=item.paper_id,
        text=chunk_text or item.evidence_text,
        page_start=item.page_start,
        page_end=item.page_end,
        token_count=10,
        content_hash="abc12345",
    )
    paper = Paper(
        paper_id=item.paper_id,
        title="Verified source paper",
        pdf_path=str(pdf_path),
        content_hash="def67890",
        page_count=declared_pages or actual_pages,
    )
    store = ChunkStore([chunk], [paper])
    return (
        CitationValidator(
            provenance_store=store,
            require_pdf_provenance=True,
        ),
        pdf_path,
    )


def test_nonexistent_evidence_id_is_rejected() -> None:
    ledger = EvidenceLedger(items=[_item(evidence_id="ev_real")])
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand and uses reflection tokens.",
                evidence_ids=["ev_real", "ev_ghost"],
            )
        ]
    )
    final = CitationValidator().validate(draft, ledger)
    assert final.citation_report is not None
    assert any("nonexistent" in i.message.lower() for i in final.citation_report.issues)
    # Ghost ID stripped; real ID kept
    assert final.claims
    assert final.claims[0].evidence_ids == ["ev_real"]
    assert "ev_ghost" not in final.citation_report.cited_evidence_ids


def test_every_source_maps_to_real_pdf_and_page(tmp_path: Path) -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_1",
                paper_id="paper_a",
                chunk_id="chunk_1",
                page_start=5,
                page_end=6,
            )
        ]
    )
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand and uses reflection tokens.",
                evidence_ids=["ev_1"],
            )
        ]
    )
    validator, pdf_path = _strict_validator(
        tmp_path,
        ledger.items[0],
        actual_pages=6,
    )
    final = validator.validate(draft, ledger)
    assert final.citation_report is not None
    assert final.citation_report.is_valid
    assert final.source_cards
    card = final.source_cards[0]
    assert card.paper_id == "paper_a"
    assert card.chunk_id == "chunk_1"
    assert card.page_start == 5
    assert card.page_end == 6
    assert card.title == "Verified source paper"
    assert card.pdf_path == str(pdf_path)
    assert "paper_a" in final.sources[0]
    assert "p.5-6" in final.sources[0] or "p.5" in final.sources[0]


def test_page_outside_physical_pdf_is_rejected(tmp_path: Path) -> None:
    item = _item(evidence_id="ev_1", page_start=2, page_end=2)
    ledger = EvidenceLedger(items=[item])
    validator, _ = _strict_validator(
        tmp_path,
        item,
        actual_pages=1,
        declared_pages=2,
    )
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand.",
                evidence_ids=["ev_1"],
            )
        ]
    )
    final = validator.validate(draft, ledger)
    assert not final.claims
    assert any(
        "pdf page count" in issue.message.lower()
        for issue in final.citation_report.issues  # type: ignore[union-attr]
    )


def test_evidence_text_must_map_to_canonical_chunk(tmp_path: Path) -> None:
    item = _item(evidence_id="ev_1")
    ledger = EvidenceLedger(items=[item])
    validator, _ = _strict_validator(
        tmp_path,
        item,
        actual_pages=3,
        chunk_text="A different canonical passage.",
    )
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand.",
                evidence_ids=["ev_1"],
            )
        ]
    )
    final = validator.validate(draft, ledger)
    assert not final.claims
    assert any(
        "canonical chunk" in issue.message.lower()
        for issue in final.citation_report.issues  # type: ignore[union-attr]
    )


def test_unsupported_claim_removed() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_1",
                claim="Dense retrieval misses exact keywords.",
                evidence_text="Dense retrieval can miss rare exact dataset names.",
            )
        ]
    )
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_bad",
                text=(
                    "Quantum entanglement proves that black holes emit musical notes "
                    "during evaporation cycles on Mars."
                ),
                evidence_ids=["ev_1"],
            )
        ]
    )
    final = CitationValidator(min_support_overlap=0.2).validate(draft, ledger)
    assert final.citation_report is not None
    assert any("does not support" in i.message.lower() for i in final.citation_report.issues)
    # Unsupported claim must not remain as a primary supported claim
    assert not final.claims
    assert "Limitation" in final.markdown or "No citation-validated" in final.markdown
    # Explicit qualification (not silent drop)
    assert "claim_bad" in final.markdown
    assert "unsupported" in final.markdown.lower()


def test_claim_without_citations_removed() -> None:
    ledger = EvidenceLedger(items=[_item(evidence_id="ev_1")])
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_bare",
                text="Something without any citation.",
                evidence_ids=[],
            ),
            ClaimWithCitations(
                claim_id="claim_ok",
                text="Self-RAG retrieves on demand and uses reflection tokens.",
                evidence_ids=["ev_1"],
            ),
        ]
    )
    final = CitationValidator().validate(draft, ledger)
    assert len(final.claims) == 1
    assert final.claims[0].claim_id == "claim_ok"
    assert any("no evidence_ids" in i.message.lower() for i in final.citation_report.issues)  # type: ignore[union-attr]


def test_references_deduplicated() -> None:
    ledger = EvidenceLedger(items=[_item(evidence_id="ev_1")])
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand.",
                evidence_ids=["ev_1"],
            ),
            ClaimWithCitations(
                claim_id="claim_2",
                text="Self-RAG uses reflection tokens.",
                evidence_ids=["ev_1", "ev_1"],
            ),
        ]
    )
    final = CitationValidator().validate(draft, ledger)
    assert final.citation_report is not None
    assert final.citation_report.cited_evidence_ids == ["ev_1"]
    assert len(final.source_cards) == 1
    assert len(final.sources) == 1


def test_supported_claim_passes() -> None:
    text = "Self-RAG retrieves on demand and uses reflection tokens to critique generation quality."
    ledger = EvidenceLedger(items=[_item(evidence_id="ev_1", claim=text, evidence_text=text)])
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand and uses reflection tokens.",
                evidence_ids=["ev_1"],
            )
        ],
        markdown="## Answer\n\n**Question:** What is Self-RAG?\n",
    )
    final = CitationValidator().validate(draft, ledger)
    assert final.citation_report is not None
    assert final.citation_report.is_valid
    assert len(final.claims) == 1
    assert "paper_self_rag" in final.markdown


def test_overlap_does_not_hide_wrong_number_or_polarity() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_1",
                claim="Method A underperforms B by 3 percent.",
                evidence_text="Method A underperforms B by 3 percent on the benchmark.",
            )
        ]
    )
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_wrong",
                text="Method A outperforms B by 30 percent on the benchmark.",
                evidence_ids=["ev_1"],
            )
        ]
    )
    final = CitationValidator().validate(draft, ledger)
    assert not final.claims
    assert any(
        "does not support" in issue.message.lower()
        for issue in final.citation_report.issues  # type: ignore[union-attr]
    )


def test_matching_negative_polarity_is_supported() -> None:
    text = "The retrieval strategy is ineffective on the adversarial benchmark."
    ledger = EvidenceLedger(items=[_item(evidence_id="ev_1", claim=text, evidence_text=text)])
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_negative",
                text=text,
                evidence_ids=["ev_1"],
            )
        ]
    )
    final = CitationValidator().validate(draft, ledger)
    assert len(final.claims) == 1


def test_citation_repair_preserves_comparison_matrix_and_downgrades_one_cell() -> None:
    self_text = "Self-RAG retrieves on demand using reflection tokens."
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_self",
                claim=self_text,
                evidence_text=self_text,
                paper_id="paper_self",
                chunk_id="chunk_self",
            ),
            _item(
                evidence_id="ev_crag",
                claim="CRAG evaluates retrieved documents.",
                evidence_text="CRAG evaluates retrieved documents before correction.",
                paper_id="paper_crag",
                chunk_id="chunk_crag",
            ),
        ]
    )
    claims = [
        ClaimWithCitations(
            claim_id="claim_self",
            text=self_text,
            evidence_ids=["ev_self"],
            requirement_key="retrieval_trigger",
            entity_id="self_rag",
            dimension="retrieval_trigger",
        ),
        ClaimWithCitations(
            claim_id="claim_crag",
            text="Quantum music occurs on Mars.",
            evidence_ids=["ev_crag"],
            requirement_key="retrieval_trigger",
            entity_id="crag",
            dimension="retrieval_trigger",
        ),
    ]
    draft = DraftAnswer(
        claims=claims,
        status=AnswerStatus.COMPLETE,
        rows=[
            ComparisonRow(
                requirement_key="retrieval_trigger",
                dimension="retrieval_trigger",
                label="Retrieval trigger",
                cells=[
                    ComparisonCell(
                        entity_id="self_rag",
                        entity_label="Self-RAG",
                        text=claims[0].text,
                        evidence_ids=["ev_self"],
                        claim_id="claim_self",
                        supported=True,
                    ),
                    ComparisonCell(
                        entity_id="crag",
                        entity_label="CRAG",
                        text=claims[1].text,
                        evidence_ids=["ev_crag"],
                        claim_id="claim_crag",
                        supported=True,
                    ),
                ],
            )
        ],
    )

    final = CitationValidator(min_support_overlap=0.5).validate(draft, ledger)

    assert final.status == AnswerStatus.PARTIAL
    assert final.corpus_insufficient is False
    assert len(final.rows) == 1
    assert [cell.supported for cell in final.rows[0].cells] == [True, False]
    assert final.rows[0].cells[1].text == "Insufficient verified evidence"
    assert "| Retrieval trigger |" in final.core_answer
    assert "Self-RAG retrieves on demand" in final.markdown
    assert "Quantum music" not in final.core_answer


def test_all_removed_partial_claims_make_legacy_citation_validity_false() -> None:
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_1",
                claim="Dense retrieval uses vector similarity.",
                evidence_text="Dense retrieval uses vector similarity for ranking.",
            )
        ]
    )
    draft = DraftAnswer(
        claims=[
            ClaimWithCitations(
                claim_id="claim_bad",
                text="Quantum music occurs on Mars.",
                evidence_ids=["ev_1"],
            )
        ],
        status=AnswerStatus.PARTIAL,
    )

    final = CitationValidator().validate(draft, ledger)

    assert final.claims == []
    assert final.status == AnswerStatus.INSUFFICIENT
    assert final.corpus_insufficient is False
    assert final.citation_report is not None
    assert final.citation_report.is_valid is False


def test_citation_repair_rejects_wrong_entity_requirement_cell_binding() -> None:
    text = "Self-RAG retrieves on demand using reflection tokens."
    ledger = EvidenceLedger(items=[_item(evidence_id="ev_self", claim=text, evidence_text=text)])
    claim = ClaimWithCitations(
        claim_id="claim_self",
        text=text,
        evidence_ids=["ev_self"],
        entity_id="self_rag",
        requirement_key="retrieval_trigger",
        dimension="retrieval_trigger",
    )
    draft = DraftAnswer(
        claims=[claim],
        status=AnswerStatus.COMPLETE,
        rows=[
            ComparisonRow(
                requirement_key="correction_mechanism",
                dimension="correction_mechanism",
                label="Correction mechanism",
                cells=[
                    ComparisonCell(
                        entity_id="crag",
                        entity_label="CRAG",
                        text=claim.text,
                        evidence_ids=["ev_self"],
                        claim_id=claim.claim_id,
                        supported=True,
                    )
                ],
            )
        ],
    )

    final = CitationValidator().validate(draft, ledger)

    assert final.claims == []
    assert final.status == AnswerStatus.INSUFFICIENT
    assert final.rows[0].cells[0].supported is False
    assert final.citation_report is not None
    assert final.citation_report.is_valid is False
    assert any("binding" in issue.message.lower() for issue in final.citation_report.issues)


def test_noncomparison_claim_repair_downgrades_complete_draft_to_partial() -> None:
    supported = "Dense retrieval uses vector similarity for ranking."
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id="ev_1",
                claim=supported,
                evidence_text=supported,
            )
        ]
    )
    draft = DraftAnswer(
        status=AnswerStatus.COMPLETE,
        claims=[
            ClaimWithCitations(
                claim_id="claim_valid",
                text=supported,
                evidence_ids=["ev_1"],
                sub_question_id="sq_0",
            ),
            ClaimWithCitations(
                claim_id="claim_invalid",
                text="Quantum music occurs on Mars.",
                evidence_ids=["ev_1"],
                sub_question_id="sq_0",
            ),
        ],
    )

    final = CitationValidator().validate(draft, ledger)

    assert [claim.claim_id for claim in final.claims] == ["claim_valid"]
    assert final.status == AnswerStatus.PARTIAL
    assert final.corpus_insufficient is False
    assert "**Answer status:** partial" in final.markdown


def test_citation_validation_preserves_confirmed_corpus_insufficiency() -> None:
    draft = DraftAnswer(
        status=AnswerStatus.INSUFFICIENT,
        corpus_insufficient=True,
    )

    final = CitationValidator().validate(draft, EvidenceLedger())

    assert final.status == AnswerStatus.INSUFFICIENT
    assert final.corpus_insufficient is True


def _joint_difference_draft() -> tuple[DraftAnswer, EvidenceLedger]:
    evidence_texts = {
        "ev_self_trigger": (
            "Self-RAG uses reflection tokens to decide when to retrieve on demand."
        ),
        "ev_crag_trigger": (
            "Corrective RAG (CRAG) uses a retrieval evaluator to classify retrieved "
            "documents as Correct, Ambiguous, or Incorrect."
        ),
        "ev_self_correction": (
            "Self-RAG uses reflection tokens to critique and correct generated responses."
        ),
        "ev_crag_correction": (
            "Corrective RAG (CRAG) refines retrieved documents and uses web search "
            "to correct low-quality retrieval."
        ),
    }
    ledger = EvidenceLedger(
        items=[
            _item(
                evidence_id=evidence_id,
                claim=text,
                evidence_text=text,
                paper_id=(
                    "paper_arxiv_2310_11511" if "self" in evidence_id else "paper_arxiv_2401_15884"
                ),
                chunk_id=f"chunk_{evidence_id}",
            )
            for evidence_id, text in evidence_texts.items()
        ]
    )
    base_specs = [
        ("claim_self_trigger", "self_rag", "retrieval_trigger", "ev_self_trigger"),
        ("claim_crag_trigger", "crag", "retrieval_trigger", "ev_crag_trigger"),
        (
            "claim_self_correction",
            "self_rag",
            "correction_mechanism",
            "ev_self_correction",
        ),
        (
            "claim_crag_correction",
            "crag",
            "correction_mechanism",
            "ev_crag_correction",
        ),
    ]
    base_claims = [
        ClaimWithCitations(
            claim_id=claim_id,
            text=evidence_texts[evidence_id],
            evidence_ids=[evidence_id],
            entity_id=entity_id,
            requirement_key=requirement_key,
            dimension=requirement_key,
        )
        for claim_id, entity_id, requirement_key, evidence_id in base_specs
    ]
    joint_ids = list(evidence_texts)
    difference_text = (
        "Self-RAG differs from CRAG across retrieval and correction: "
        f"{evidence_texts['ev_self_trigger']} "
        f"{evidence_texts['ev_self_correction']} By contrast, "
        f"{evidence_texts['ev_crag_trigger']} "
        f"{evidence_texts['ev_crag_correction']}"
    )
    difference_claims = [
        ClaimWithCitations(
            claim_id=f"claim_difference_{entity_id}",
            text=difference_text,
            evidence_ids=joint_ids,
            entity_id=entity_id,
            requirement_key="key_differences",
            dimension="key_differences",
        )
        for entity_id in ("self_rag", "crag")
    ]
    claims = [*base_claims, *difference_claims]
    claims_by_id = {claim.claim_id: claim for claim in claims}
    labels = {"self_rag": "Self-RAG", "crag": "CRAG"}

    def _row(requirement_key: str, claim_ids: list[str]) -> ComparisonRow:
        return ComparisonRow(
            requirement_key=requirement_key,
            dimension=requirement_key,
            label=requirement_key.replace("_", " ").title(),
            cells=[
                ComparisonCell(
                    entity_id=claims_by_id[claim_id].entity_id or "",
                    entity_label=labels[claims_by_id[claim_id].entity_id or ""],
                    text=claims_by_id[claim_id].text,
                    evidence_ids=list(claims_by_id[claim_id].evidence_ids),
                    claim_id=claim_id,
                    supported=True,
                )
                for claim_id in claim_ids
            ],
        )

    return (
        DraftAnswer(
            claims=claims,
            status=AnswerStatus.COMPLETE,
            rows=[
                _row(
                    "retrieval_trigger",
                    ["claim_self_trigger", "claim_crag_trigger"],
                ),
                _row(
                    "correction_mechanism",
                    ["claim_self_correction", "claim_crag_correction"],
                ),
                _row(
                    "key_differences",
                    ["claim_difference_self_rag", "claim_difference_crag"],
                ),
            ],
        ),
        ledger,
    )


def test_key_differences_use_joint_support_after_per_evidence_provenance() -> None:
    draft, ledger = _joint_difference_draft()
    validator = CitationValidator()
    difference = next(claim for claim in draft.claims if claim.requirement_key == "key_differences")
    by_id = {item.evidence_id: item for item in ledger.items}
    assert any(
        not validator._supports(difference, by_id[evidence_id])
        for evidence_id in difference.evidence_ids
    )

    final = validator.validate(draft, ledger)

    assert final.status == AnswerStatus.COMPLETE
    assert final.citation_report is not None
    assert final.citation_report.is_valid
    difference_claims = [
        claim for claim in final.claims if claim.requirement_key == "key_differences"
    ]
    assert len(difference_claims) == 2
    assert all("Self-RAG differs from CRAG" in claim.text for claim in difference_claims)


def test_key_differences_reject_any_missing_or_bad_provenance() -> None:
    draft, ledger = _joint_difference_draft()
    claims_with_ghost = [
        claim.model_copy(update={"evidence_ids": [*claim.evidence_ids, "ev_ghost"]})
        if claim.requirement_key == "key_differences"
        else claim
        for claim in draft.claims
    ]
    ghost_claims_by_id = {claim.claim_id: claim for claim in claims_with_ghost}
    rows_with_ghost = [
        row.model_copy(
            update={
                "cells": [
                    cell.model_copy(
                        update={
                            "evidence_ids": list(ghost_claims_by_id[cell.claim_id].evidence_ids)
                        }
                    )
                    if (row.requirement_key == "key_differences" and cell.claim_id is not None)
                    else cell
                    for cell in row.cells
                ]
            }
        )
        for row in draft.rows
    ]
    ghost_draft = draft.model_copy(update={"claims": claims_with_ghost, "rows": rows_with_ghost})

    missing_final = CitationValidator().validate(ghost_draft, ledger)

    assert not any(claim.requirement_key == "key_differences" for claim in missing_final.claims)
    assert any(
        "nonexistent" in issue.message.lower() for issue in missing_final.citation_report.issues
    )

    bad_evidence_id = "ev_crag_correction"
    bad_ledger = EvidenceLedger(
        items=[
            item.model_copy(update={"page_start": 0})
            if item.evidence_id == bad_evidence_id
            else item
            for item in ledger.items
        ]
    )

    bad_final = CitationValidator().validate(draft, bad_ledger)

    assert not any(claim.requirement_key == "key_differences" for claim in bad_final.claims)
    assert any(
        "invalid page range" in issue.message.lower() for issue in bad_final.citation_report.issues
    )
