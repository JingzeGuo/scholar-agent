"""Rerankers: cross-encoder (production) and lexical (offline tests)."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from scholar_agent.models.retrieval import RetrievalHit
from scholar_agent.retrieval.embeddings import huggingface_offline_enabled, model_cache_folder

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[RetrievalHit], *, top_k: int) -> list[RetrievalHit]: ...

    @property
    def model_name(self) -> str: ...


class LexicalReranker:
    """Token-overlap reranker for deterministic offline tests."""

    def __init__(self, model_name: str = "lexical-overlap-v1") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, hits: list[RetrievalHit], *, top_k: int) -> list[RetrievalHit]:
        q = set(_TOKEN_RE.findall(query.lower()))
        scored: list[tuple[float, RetrievalHit]] = []
        for hit in hits:
            d = set(_TOKEN_RE.findall(hit.text.lower()))
            score = 0.0 if not q or not d else len(q & d) / len(q)
            scored.append((score, hit))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        out: list[RetrievalHit] = []
        for score, hit in scored[:top_k]:
            out.append(
                hit.model_copy(
                    update={
                        "rerank_score": score,
                        "retrieval_method": (
                            hit.retrieval_method
                            if "rerank" in hit.retrieval_method
                            else f"{hit.retrieval_method}+rerank"
                        ),
                    }
                )
            )
        return out


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        cache_folder = model_cache_folder()
        try:
            self._model = CrossEncoder(
                model_name,
                cache_folder=cache_folder,
                local_files_only=True,
            )
        except Exception as local_exc:  # noqa: BLE001 - library error types vary
            if huggingface_offline_enabled():
                raise RuntimeError(
                    f"Reranker model {model_name!r} is not available in the local cache"
                ) from local_exc
            self._model = CrossEncoder(
                model_name,
                cache_folder=cache_folder,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, hits: list[RetrievalHit], *, top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(scores, hits, strict=True),
            key=lambda item: (-float(item[0]), item[1].chunk_id),
        )
        out: list[RetrievalHit] = []
        for score, hit in ranked[:top_k]:
            out.append(
                hit.model_copy(
                    update={
                        "rerank_score": float(score),
                        "retrieval_method": (
                            hit.retrieval_method
                            if "rerank" in hit.retrieval_method
                            else f"{hit.retrieval_method}+rerank"
                        ),
                    }
                )
            )
        return out


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    if model_name.startswith("lexical") or model_name.startswith("hash"):
        return LexicalReranker(model_name=model_name)
    return CrossEncoderReranker(model_name=model_name)
