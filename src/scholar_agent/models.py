"""The small data boundary used by ScholarAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field


class AgentState(TypedDict):
    """The only state passed through the LangGraph workflow."""

    question: str
    queries: list[str]
    entities: list[str]
    candidates: list[dict]
    evidence: list[dict]
    sufficient: bool
    feedback: str
    retry_count: int
    answer: str


class ChunkRecord(BaseModel):
    """Validated only at the ingestion/storage boundary."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    paper: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str = Field(min_length=1)

    def evidence(self, score: float = 0.0) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper": self.paper,
            "page": self.page,
            "text": self.text,
            "score": float(score),
        }


def load_chunks(path: Path) -> list[dict]:
    """Load JSONL chunks and validate their page provenance."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunk store not found: {path}. Run `scholar-agent ingest`.")

    chunks: list[dict] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = ChunkRecord.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid chunk at {path}:{line_number}: {exc}") from exc
            if record.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id at {path}:{line_number}: {record.chunk_id}")
            seen_ids.add(record.chunk_id)
            chunks.append(record.evidence())
    if not chunks:
        raise ValueError(f"Chunk store is empty: {path}")
    return chunks


def save_chunks(chunks: list[dict], path: Path) -> None:
    """Validate and persist plain JSONL without a manifest or ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = ChunkRecord.model_validate(chunk)
            if record.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {record.chunk_id}")
            seen_ids.add(record.chunk_id)
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
