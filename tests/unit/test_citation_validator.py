"""Phase 7 citation validator: ID existence, page provenance, support checks."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from scholar_agent.agents.citation_validator import CitationValidator
from scholar_agent.models.answer import ClaimWithCitations, DraftAnswer
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
    assert any(
        "does not support" in i.message.lower() for i in final.citation_report.issues
    )
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
    text = (
        "Self-RAG retrieves on demand and uses reflection tokens "
        "to critique generation quality."
    )
    ledger = EvidenceLedger(
        items=[_item(evidence_id="ev_1", claim=text, evidence_text=text)]
    )
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
    ledger = EvidenceLedger(
        items=[_item(evidence_id="ev_1", claim=text, evidence_text=text)]
    )
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
