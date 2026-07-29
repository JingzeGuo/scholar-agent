"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.retrieval.chunk_store import ChunkStore

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def default_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "default.yaml"


@pytest.fixture
def full_corpus_store(repo_root: Path) -> ChunkStore:
    """Load the optional canonical store or skip when full-corpus assets are absent."""
    processed_dir = repo_root / "data" / "processed"
    if not (processed_dir / "chunks.jsonl").is_file():
        pytest.skip("optional full-corpus chunk store not present (run ingest)")
    return ChunkStore.from_processed_dir(processed_dir)
