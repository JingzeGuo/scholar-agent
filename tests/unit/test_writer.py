"""Phase 7 Writer: evidence-constrained claims and citation rendering."""

from __future__ import annotations

from scholar_agent.agents.writer import Writer, format_inline_citation, render_claim_markdown
from scholar_agent.models.answer import ClaimWithCitations
from scholar_agent.models.base import QueryType
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
    assert any("disagree" in c.text.lower() or "conflict" in c.text.lower() for c in draft.claims) or any(
        "conflict" in n.lower() for n in draft.notes
    )
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
    draft = Writer().write(
        query="Compare Self-RAG versus CRAG", plan=plan, ledger=ledger
    )
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
