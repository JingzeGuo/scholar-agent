"""Small persisted BM25 and NumPy dense indexes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


class ModelUnavailableError(RuntimeError):
    """Raised when a required local model cannot be resolved or downloaded."""


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def resolve_model_path(model_name: str) -> str:
    """Resolve a local model, downloading its Hugging Face snapshot when online."""
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_name)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Model unavailable and download failed: {model_name}",
        ) from exc


class BM25Index:
    def __init__(self, chunks: list[dict], tokens: list[list[str]] | None = None) -> None:
        self.chunks = chunks
        self.tokens = tokens or [tokenize(item["text"]) for item in chunks]
        self.index = BM25Okapi(self.tokens)

    def search(self, queries: list[str], top_k: int = 20) -> list[dict]:
        if not queries or not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        for query in queries:
            scores = np.maximum(scores, np.asarray(self.index.get_scores(tokenize(query))))
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {**self.chunks[int(i)], "score": float(scores[int(i)])}
            for i in ranked
            if scores[int(i)] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.tokens}), encoding="utf-8")

    @classmethod
    def load(cls, chunks: list[dict], path: Path) -> BM25Index:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = payload["tokens"]
        if len(chunks) != len(tokens):
            raise ValueError("BM25 tokens do not align with chunks; rebuild the index")
        return cls(chunks, tokens=tokens)


def _sentence_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(resolve_model_path(model_name), local_files_only=True)
        encoded: Any = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except ModelUnavailableError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(f"Embedding model failed: {model_name}") from exc
    return np.asarray(encoded, dtype=np.float32)


class DenseIndex:
    def __init__(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        model_name: str,
        backend: str,
    ) -> None:
        if backend != "sentence-transformers":
            raise ModelUnavailableError(
                "Dense index uses an unsupported fallback backend; run `scholar-agent index`.",
            )
        self.chunks = chunks
        self.embeddings = embeddings
        self.model_name = model_name
        self.backend = backend

    @classmethod
    def build(cls, chunks: list[dict], model_name: str) -> DenseIndex:
        texts = [item["text"] for item in chunks]
        embeddings = _sentence_embeddings(texts, model_name)
        return cls(chunks, embeddings, model_name, "sentence-transformers")

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        return _sentence_embeddings(queries, self.model_name)

    def search(self, queries: list[str], top_k: int = 20) -> list[dict]:
        if not queries or not self.chunks:
            return []
        query_vectors = self._encode_queries(queries)
        scores = np.max(query_vectors @ self.embeddings.T, axis=0)
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {**self.chunks[int(i)], "score": float(scores[int(i)])}
            for i in ranked
            if scores[int(i)] > 0
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "dense.npy", self.embeddings)
        metadata = {"model": self.model_name, "backend": self.backend}
        (directory / "dense.json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def load(cls, chunks: list[dict], directory: Path) -> DenseIndex:
        metadata = json.loads((directory / "dense.json").read_text(encoding="utf-8"))
        embeddings = np.load(directory / "dense.npy")
        if len(chunks) != len(embeddings):
            raise ValueError("Dense embeddings do not align with chunks; rebuild the index")
        if embeddings.ndim != 2:
            raise ValueError("Dense index must be a two-dimensional NumPy matrix")
        return cls(chunks, embeddings, metadata["model"], metadata["backend"])
