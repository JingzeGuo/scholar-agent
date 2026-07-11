"""Embedding backends for dense retrieval.

Unit tests use ``HashingEmbedder`` (no model download). Production uses
sentence-transformers (default BAAI/bge-small-en-v1.5).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from scholar_agent.config import REPO_ROOT


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def model_cache_folder() -> str:
    """Return the shared Hugging Face cache used by production retrieval models.

    Keep model weights inside the repository's ignored ``.cache`` directory by
    default so CLI commands reuse downloads consistently. Operators can override
    it with ``SCHOLAR_MODEL_CACHE`` or the standard ``HF_HOME`` variable.
    """
    explicit = os.getenv("SCHOLAR_MODEL_CACHE")
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return str((Path(hf_home).expanduser() / "hub").resolve())
    return str((REPO_ROOT / ".cache" / "huggingface" / "hub").resolve())


class HashingEmbedder:
    """Deterministic bag-of-tokens hashing embedder for offline tests."""

    def __init__(self, dimension: int = 64, *, model_name: str = "hashing-embedder-v1") -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec: np.ndarray = np.zeros(self._dimension, dtype=np.float64)
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return [float(x) for x in vec.tolist()]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec.tolist()]


class SentenceTransformerEmbedder:
    """sentence-transformers wrapper (downloads model on first use)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name,
            cache_folder=model_cache_folder(),
        )
        # probe dimension
        probe = self._model.encode(["probe"], normalize_embeddings=True)
        self._dimension = int(probe.shape[1])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # BGE models recommend a query instruction prefix for retrieval
        query = text
        if "bge" in self._model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {text}"
        vector = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(x) for x in vector.tolist()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def create_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    *,
    backend: str = "auto",
) -> Embedder:
    """Create an embedder.

    backend:
      - ``hash``: HashingEmbedder (tests / no download)
      - ``st`` / ``sentence-transformers``: real model
      - ``auto``: hash if model_name starts with ``hash``, else ST
    """
    if backend == "hash" or model_name.startswith("hash"):
        return HashingEmbedder()
    if backend in {"st", "sentence-transformers", "auto"}:
        if backend == "auto" and model_name.startswith("hash"):
            return HashingEmbedder()
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"unknown embedding backend: {backend}")
