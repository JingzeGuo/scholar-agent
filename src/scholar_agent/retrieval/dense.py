"""Dense retrieval over Chroma with stable chunk IDs."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from scholar_agent.models.corpus import Chunk
from scholar_agent.models.retrieval import RetrievalFilters, RetrievalHit
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.embeddings import Embedder, HashingEmbedder, create_embedder
from scholar_agent.retrieval.sparse import _passes_filters


class DenseIndex:
    """Chroma-backed dense index; IDs == canonical chunk_ids."""

    def __init__(
        self,
        *,
        collection: Any,
        embedder: Embedder,
        store: ChunkStore,
        meta: dict[str, Any],
        persist_dir: Path | None = None,
    ) -> None:
        self._collection = collection
        self.embedder = embedder
        self.store = store
        self.meta = meta
        self.persist_dir = persist_dir

    @classmethod
    def build(
        cls,
        store: ChunkStore,
        *,
        embedder: Embedder | None = None,
        persist_dir: Path | str | None = None,
        collection_name: str = "scholar_chunks",
        batch_size: int = 64,
    ) -> DenseIndex:
        embedder = embedder or HashingEmbedder()
        import chromadb
        from chromadb.config import Settings

        if persist_dir is not None:
            root = Path(persist_dir)
            root.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(root),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Recreate collection for a clean rebuild
        with contextlib.suppress(Exception):
            client.delete_collection(collection_name)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = store.ordered_ids()
        documents = [store.by_chunk_id[i].text for i in ids]
        metadatas = [
            {
                "paper_id": store.by_chunk_id[i].paper_id,
                "page_start": store.by_chunk_id[i].page_start,
                "page_end": store.by_chunk_id[i].page_end,
                "section": store.by_chunk_id[i].section or "",
                "content_hash": store.by_chunk_id[i].content_hash,
            }
            for i in ids
        ]

        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_docs = documents[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]
            embeddings = embedder.embed_documents(batch_docs)
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=embeddings,
            )

        meta = {
            "corpus_fingerprint": store.fingerprint,
            "chunk_ids": ids,
            "n_docs": len(ids),
            "embedder": embedder.model_name,
            "dimension": embedder.dimension,
            "collection_name": collection_name,
        }
        if persist_dir is not None:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            (Path(persist_dir) / "meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        return cls(
            collection=collection,
            embedder=embedder,
            store=store,
            meta=meta,
            persist_dir=Path(persist_dir) if persist_dir else None,
        )

    @classmethod
    def load(
        cls,
        persist_dir: Path | str,
        store: ChunkStore,
        *,
        embedder: Embedder | None = None,
        verify: bool = True,
        collection_name: str = "scholar_chunks",
    ) -> DenseIndex:
        root = Path(persist_dir)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"dense index meta missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if verify and meta.get("corpus_fingerprint") != store.fingerprint:
            raise ValueError("dense index fingerprint mismatch with chunk store; rebuild required")
        coll_name = str(meta.get("collection_name") or collection_name)
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(root),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(coll_name)
        if embedder is None:
            model_name = str(meta.get("embedder") or "hashing-embedder-v1")
            backend = "hash" if model_name.startswith("hash") else "st"
            embedder = create_embedder(model_name, backend=backend)
        return cls(
            collection=collection,
            embedder=embedder,
            store=store,
            meta=meta,
            persist_dir=root,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 12,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip() or len(self.store) == 0:
            return []
        # Over-fetch then filter
        fetch_k = min(len(self.store), max(k * 4, k))
        query_emb = self.embedder.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_emb],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        hits: list[RetrievalHit] = []
        for rank, chunk_id in enumerate(ids, start=1):
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                # reconstruct from chroma payload if needed
                meta = metadatas[rank - 1] if rank - 1 < len(metadatas) else {}
                text = documents[rank - 1] if rank - 1 < len(documents) else ""
                chunk = Chunk(
                    chunk_id=chunk_id,
                    paper_id=str(meta.get("paper_id") or "unknown"),
                    text=text or "",
                    page_start=int(meta.get("page_start") or 1),
                    page_end=int(meta.get("page_end") or 1),
                    section=(meta.get("section") or None) or None,
                    token_count=0,
                    content_hash=str(meta.get("content_hash") or "unknown"),
                )
            if filters and not _passes_filters(chunk, filters):
                continue
            dist = float(distances[rank - 1]) if rank - 1 < len(distances) else 0.0
            # cosine distance → similarity-ish score
            score = 1.0 - dist
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    score=score,
                    dense_rank=rank,
                    retrieval_method="dense",
                )
            )
            if len(hits) >= k:
                break
        return hits
