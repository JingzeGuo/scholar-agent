from __future__ import annotations

from scholar_agent.citations import citation_summary, cited_pages, validate_citations


def test_fake_citation_is_removed(sample_chunks: list[dict]) -> None:
    answer = validate_citations(
        "Supported [E1], fabricated [E99] and [Fake.pdf p.999].",
        sample_chunks[:1],
    )

    assert "[E99]" not in answer
    assert "[Fake.pdf p.999]" not in answer
    assert "[Self-RAG.pdf p.1]" in answer


def test_citation_uses_real_filename_and_page(sample_chunks: list[dict]) -> None:
    answer = validate_citations("First [E1]; second [E2].", sample_chunks[:2])

    assert answer == "First [Self-RAG.pdf p.1]; second [CRAG.pdf p.2]."


def test_cited_pages_extracts_validated_provenance() -> None:
    assert cited_pages("See [Self-RAG.pdf p.3] and [CRAG.pdf p.11].") == [
        ("Self-RAG.pdf", 3),
        ("CRAG.pdf", 11),
    ]


def test_answer_without_citations_is_not_grounded(sample_chunks: list[dict]) -> None:
    summary = citation_summary("An uncited answer.", sample_chunks)

    assert summary["citations"] == 0
    assert summary["all_grounded"] is False
