"""JSONL repository and corpus manifest tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.models import Chunk, CorpusManifestEntry, Paper
from scholar_agent.storage.jsonl import JsonlRepository, JsonlRepositoryError
from scholar_agent.storage.manifest import (
    ManifestError,
    load_corpus_manifest,
    save_corpus_manifest,
    validate_corpus_manifest,
)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    repo: JsonlRepository[Chunk] = JsonlRepository(path, Chunk)
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            paper_id="paper_1",
            text="alpha",
            page_start=1,
            page_end=1,
            token_count=1,
            content_hash="aaaaaaaaaaaaaaaa",
        ),
        Chunk(
            chunk_id="chunk_2",
            paper_id="paper_1",
            text="beta",
            page_start=2,
            page_end=2,
            token_count=1,
            content_hash="bbbbbbbbbbbbbbbb",
        ),
    ]
    repo.write_all(chunks)
    loaded = repo.read_all()
    assert loaded == chunks
    assert repo.count() == 2
    by_id = repo.index_by("chunk_id")
    assert set(by_id) == {"chunk_1", "chunk_2"}


def test_jsonl_invalid_line_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"chunk_id": "x"}\n', encoding="utf-8")
    repo: JsonlRepository[Chunk] = JsonlRepository(path, Chunk)
    with pytest.raises(JsonlRepositoryError) as exc:
        repo.read_all()
    assert "schema validation failed" in str(exc.value)


def test_jsonl_invalid_json_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    repo: JsonlRepository[Paper] = JsonlRepository(path, Paper)
    with pytest.raises(JsonlRepositoryError) as exc:
        repo.read_all()
    assert "invalid JSON" in str(exc.value)


def test_load_fixture_manifest(repo_root: Path) -> None:
    path = repo_root / "tests" / "fixtures" / "corpus_manifest.jsonl"
    manifest = load_corpus_manifest(path)
    assert len(manifest) == 3
    assert "paper_arxiv_2310_11511" in manifest.by_id()
    assert all(e.ingestion_status.value == "pending" for e in manifest.entries)


def test_load_fixture_papers_and_chunks(repo_root: Path) -> None:
    papers = JsonlRepository(
        repo_root / "tests" / "fixtures" / "papers.jsonl", Paper
    ).read_all()
    chunks = JsonlRepository(
        repo_root / "tests" / "fixtures" / "chunks.jsonl", Chunk
    ).read_all()
    assert len(papers) == 2
    assert len(chunks) == 2
    assert {c.paper_id for c in chunks}.issubset({p.paper_id for p in papers})


def test_duplicate_paper_id_rejected(tmp_path: Path) -> None:
    entry = CorpusManifestEntry(
        paper_id="paper_dup",
        title="Dup",
        pdf_filename="dup.pdf",
        content_hash="0123456789abcdef",
        year=2024,
    )
    path = tmp_path / "manifest.jsonl"
    with pytest.raises(ManifestError):
        save_corpus_manifest(path, [entry, entry.model_copy()])


def test_validate_manifest_missing_pdfs(tmp_path: Path, repo_root: Path) -> None:
    src = repo_root / "tests" / "fixtures" / "corpus_manifest.jsonl"
    path = tmp_path / "manifest.jsonl"
    path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    issues = validate_corpus_manifest(path, papers_dir=tmp_path / "empty_pdfs")
    assert issues
    assert any("missing PDF" in issue for issue in issues)


def test_validate_manifest_ok_without_pdf_check(repo_root: Path) -> None:
    path = repo_root / "tests" / "fixtures" / "corpus_manifest.jsonl"
    assert validate_corpus_manifest(path) == []


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        load_corpus_manifest(tmp_path / "nope.jsonl")
