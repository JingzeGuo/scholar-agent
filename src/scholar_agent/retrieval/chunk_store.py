"""Canonical chunk store loader — source of truth for every index."""

from __future__ import annotations

from pathlib import Path

from scholar_agent.ids import content_hash
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.storage.jsonl import JsonlRepository


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Stable hash over chunk IDs and content hashes (order-independent)."""
    material = "\n".join(
        sorted(f"{c.chunk_id}:{c.content_hash}" for c in chunks)
    )
    return content_hash(material, length=32)


class ChunkStore:
    """In-memory view of the canonical chunk (+ paper) JSONL stores."""

    def __init__(
        self,
        chunks: list[Chunk],
        papers: list[Paper] | None = None,
        *,
        chunks_path: Path | None = None,
        papers_path: Path | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.papers = list(papers or [])
        self.chunks_path = chunks_path
        self.papers_path = papers_path
        self.by_chunk_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.by_paper_id: dict[str, Paper] = {p.paper_id: p for p in self.papers}
        if len(self.by_chunk_id) != len(self.chunks):
            raise ValueError("duplicate chunk_id in canonical chunk store")
        self.fingerprint = corpus_fingerprint(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.by_chunk_id.get(chunk_id)

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.by_paper_id.get(paper_id)

    def ordered_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @classmethod
    def from_processed_dir(cls, processed_dir: Path | str) -> ChunkStore:
        root = Path(processed_dir)
        chunks_path = root / "chunks.jsonl"
        papers_path = root / "papers.jsonl"
        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"canonical chunk store missing: {chunks_path}. Run ingest first."
            )
        chunks = JsonlRepository(chunks_path, Chunk).read_all()
        papers = (
            JsonlRepository(papers_path, Paper).read_all() if papers_path.is_file() else []
        )
        if not chunks:
            raise ValueError(f"chunk store is empty: {chunks_path}")
        return cls(chunks, papers, chunks_path=chunks_path, papers_path=papers_path)
