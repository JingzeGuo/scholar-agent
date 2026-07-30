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

    llm_model: str = "deepseek-chat"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_rerank_score: float = -1.0
    top_k: int = 20
    data_dir: Path = Path("data")

    def __post_init__(self) -> None:
        if not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if not self.reranker_model.strip():
            raise ValueError("reranker_model cannot be empty")
        if not math.isfinite(self.min_rerank_score):
            raise ValueError("min_rerank_score must be finite")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_model=os.getenv("SCHOLAR_AGENT_LLM_MODEL", cls.llm_model),
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
            top_k=int(os.getenv("SCHOLAR_AGENT_TOP_K", str(cls.top_k))),
            data_dir=Path(os.getenv("SCHOLAR_AGENT_DATA_DIR", str(cls.data_dir))),
        )

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "processed" / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    def describe(self) -> dict[str, str | int | float]:
        """Return a secret-free configuration summary suitable for diagnostics."""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "min_rerank_score": self.min_rerank_score,
            "top_k": self.top_k,
            "data_dir": str(self.data_dir),
        }
