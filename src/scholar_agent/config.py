"""Minimal environment-based configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Only settings needed by the compact research path."""

    llm_model: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_rerank_score: float = -1.0
    retrieval_mode: str = "adaptive"
    data_dir: Path = Path("data")

    def __post_init__(self) -> None:
        if self.llm_model is not None and not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if not self.reranker_model.strip():
            raise ValueError("reranker_model cannot be empty")
        if not math.isfinite(self.min_rerank_score):
            raise ValueError("min_rerank_score must be finite")
        if self.retrieval_mode not in {"fixed_hybrid", "adaptive"}:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_model=os.getenv("SCHOLAR_AGENT_LLM_MODEL"),
            embedding_model=os.getenv(
                "SCHOLAR_AGENT_EMBEDDING_MODEL",
                cls.embedding_model,
            ),
            reranker_model=os.getenv(
                "SCHOLAR_AGENT_RERANKER_MODEL",
                cls.reranker_model,
            ),
            min_rerank_score=float(
                os.getenv("SCHOLAR_AGENT_MIN_RERANK_SCORE", str(cls.min_rerank_score)),
            ),
            retrieval_mode=os.getenv(
                "SCHOLAR_AGENT_RETRIEVAL_MODE",
                cls.retrieval_mode,
            ),
            data_dir=Path(os.getenv("SCHOLAR_AGENT_DATA_DIR", str(cls.data_dir))),
        )

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "processed" / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"
