"""Persistent BM25 index aligned to stable chunk IDs."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from scholar_agent.models.corpus import Chunk
from scholar_agent.models.retrieval import RetrievalFilters, RetrievalHit
from scholar_agent.retrieval.chunk_store import ChunkStore

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25IndexMeta:
    corpus_fingerprint: str
    chunk_ids: list[str]
    token_counts: list[int]
    n_docs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "chunk_ids": self.chunk_ids,
            "token_counts": self.token_counts,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25IndexMeta:
        return cls(
            corpus_fingerprint=str(data["corpus_fingerprint"]),
            chunk_ids=list(data["chunk_ids"]),
            token_counts=list(data["token_counts"]),
            n_docs=int(data["n_docs"]),
        )


class BM25Index:
    """BM25 over canonical chunks with on-disk persistence."""

    def __init__(
        self,
        bm25: BM25Okapi,
        meta: BM25IndexMeta,
        chunks_by_id: dict[str, Chunk],
        *,
        index_dir: Path | None = None,
    ) -> None:
        self._bm25 = bm25
        self.meta = meta
        self._chunks_by_id = chunks_by_id
        self.index_dir = index_dir

    @classmethod
    def build(cls, store: ChunkStore) -> BM25Index:
        tokenized = [tokenize(c.text) for c in store.chunks]
        bm25 = BM25Okapi(tokenized)
        meta = BM25IndexMeta(
            corpus_fingerprint=store.fingerprint,
            chunk_ids=store.ordered_ids(),
            token_counts=[len(toks) for toks in tokenized],
            n_docs=len(store.chunks),
        )
        return cls(bm25, meta, store.by_chunk_id)

    def save(self, index_dir: Path | str) -> None:
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "bm25.pkl").open("wb") as handle:
            pickle.dump(self._bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (root / "meta.json").write_text(
            json.dumps(self.meta.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_dir: Path | str,
        store: ChunkStore,
        *,
        verify: bool = True,
    ) -> BM25Index:
        root = Path(index_dir)
        meta_path = root / "meta.json"
        pkl_path = root / "bm25.pkl"
        if not meta_path.is_file() or not pkl_path.is_file():
            raise FileNotFoundError(f"BM25 index incomplete under {root}")
        meta = BM25IndexMeta.from_dict(
            json.loads(meta_path.read_text(encoding="utf-8"))
        )
        if verify and meta.corpus_fingerprint != store.fingerprint:
            raise ValueError(
                "BM25 index fingerprint mismatch with chunk store; rebuild required "
                f"(index={meta.corpus_fingerprint[:12]}… store={store.fingerprint[:12]}…)"
            )
        if verify and meta.chunk_ids != store.ordered_ids():
            # order must match BM25 internal corpus rows
            raise ValueError("BM25 chunk_id order diverges from canonical chunk store")
        with pkl_path.open("rb") as handle:
            bm25 = pickle.load(handle)  # noqa: S301 — local trusted artifact
        return cls(bm25, meta, store.by_chunk_id, index_dir=root)

    def search(
        self,
        query: str,
        *,
        k: int = 12,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        tokens = tokenize(query)
        if not tokens or self.meta.n_docs == 0:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort descending
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.meta.chunk_ids[i]))
        hits: list[RetrievalHit] = []
        for rank_idx, doc_i in enumerate(order, start=1):
            chunk_id = self.meta.chunk_ids[doc_i]
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            if filters and not _passes_filters(chunk, filters):
                continue
            score = float(scores[doc_i])
            # Skip negative scores once we already have positive hits
            if score < 0 and hits:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    score=score,
                    sparse_rank=rank_idx,
                    retrieval_method="sparse",
                )
            )
            if len(hits) >= k:
                break
        return hits


def _passes_filters(chunk: Chunk, filters: RetrievalFilters) -> bool:
    if filters.paper_ids is not None and chunk.paper_id not in filters.paper_ids:
        return False
    if filters.page_min is not None and chunk.page_end < filters.page_min:
        return False
    if filters.page_max is not None and chunk.page_start > filters.page_max:
        return False
    if filters.section_contains:
        section = chunk.section or ""
        if filters.section_contains.lower() not in section.lower():
            return False
    return True
