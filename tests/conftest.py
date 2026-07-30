"""Small deterministic corpus fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "self-1",
            "paper": "Self-RAG.pdf",
            "page": 1,
            "text": "Self-RAG uses adaptive retrieval and reflection tokens.",
            "score": 0.0,
        },
        {
            "chunk_id": "crag-1",
            "paper": "CRAG.pdf",
            "page": 2,
            "text": "CRAG uses a retrieval evaluator and corrective retrieval.",
            "score": 0.0,
        },
        {
            "chunk_id": "other-1",
            "paper": "Other.pdf",
            "page": 3,
            "text": "A transformer baseline processes ordinary language.",
            "score": 0.0,
        },
    ]


@pytest.fixture
def papers_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "papers"
