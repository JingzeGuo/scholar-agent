"""Stable ID helper tests."""

from __future__ import annotations

import pytest

from scholar_agent.ids import (
    content_hash,
    make_chunk_id,
    make_entity_id,
    make_evidence_id,
    make_paper_id,
    make_relation_id,
    make_sub_question_id,
    normalize_text,
)


def test_normalize_text_unicode_and_case() -> None:
    assert normalize_text("  Self-RAG  ") == "self-rag"
    assert normalize_text("Foo   Bar") == "foo bar"
    # en-dash collapses via non-alnum path for IDs, plain normalize keeps punctuation
    assert "rag" in normalize_text("Self–RAG")


def test_content_hash_stable() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_paper_id_prefers_arxiv() -> None:
    pid = make_paper_id(arxiv_id="2310.11511", title="Other Title", year=2024)
    assert pid.startswith("paper_arxiv_")
    assert make_paper_id(arxiv_id="2310.11511") == pid


def test_paper_id_doi_and_title_paths() -> None:
    doi_id = make_paper_id(doi="10.1234/abcd.ef")
    assert doi_id.startswith("paper_doi_")
    title_id_a = make_paper_id(title="GraphRAG Survey", year=2024)
    title_id_b = make_paper_id(title="GraphRAG Survey", year=2024)
    assert title_id_a == title_id_b
    assert title_id_a.startswith("paper_")


def test_paper_id_requires_signal() -> None:
    with pytest.raises(ValueError):
        make_paper_id()


def test_chunk_id_stable_and_page_sensitive() -> None:
    a = make_chunk_id("paper_x", page_start=1, page_end=1, text="hello world")
    b = make_chunk_id("paper_x", page_start=1, page_end=1, text="hello world")
    c = make_chunk_id("paper_x", page_start=2, page_end=2, text="hello world")
    assert a == b
    assert a != c
    assert a.startswith("chunk_")


def test_chunk_id_invalid_pages() -> None:
    with pytest.raises(ValueError):
        make_chunk_id("paper_x", page_start=3, page_end=1, text="x")


def test_entity_and_relation_and_evidence_ids() -> None:
    ent = make_entity_id("Method", "Self-RAG")
    assert ent == make_entity_id("method", "self-rag")
    rel = make_relation_id(
        subject_entity_id=ent,
        relation_type="PROPOSES",
        object_entity_id=make_entity_id("Task", "open-domain QA"),
        chunk_id="chunk_1",
        evidence_span="Self-RAG proposes ...",
    )
    assert rel == make_relation_id(
        subject_entity_id=ent,
        relation_type="PROPOSES",
        object_entity_id=make_entity_id("Task", "open-domain QA"),
        chunk_id="chunk_1",
        evidence_span="Self-RAG proposes ...",
    )
    ev = make_evidence_id(
        run_id="run_abc",
        chunk_id="chunk_1",
        evidence_text="  Same Span  ",
        sub_question_id="sq_1",
    )
    ev2 = make_evidence_id(
        run_id="run_abc",
        chunk_id="chunk_1",
        evidence_text="same span",
        sub_question_id="sq_1",
    )
    assert ev == ev2


def test_sub_question_id_stable() -> None:
    a = make_sub_question_id("plan_seed", "What is CRAG?", 0)
    b = make_sub_question_id("plan_seed", "What is CRAG?", 0)
    c = make_sub_question_id("plan_seed", "What is CRAG?", 1)
    assert a == b
    assert a != c
