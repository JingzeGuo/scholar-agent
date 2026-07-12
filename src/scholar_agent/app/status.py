"""Corpus / index health checks for the demo sidebar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.config import AppConfig, load_config


class SystemStatus(BaseModel):
    ok: bool = False
    papers_dir_exists: bool = False
    n_pdfs: int = 0
    manifest_exists: bool = False
    n_manifest_entries: int = 0
    processed_papers: int = 0
    processed_chunks: int = 0
    dense_index_ready: bool = False
    sparse_index_ready: bool = False
    graph_ready: bool = False
    graph_nodes: int | None = None
    graph_edges: int | None = None
    demo_runs: int = 0
    messages: list[str] = Field(default_factory=list)
    paths: dict[str, str] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _count_pdfs(papers_dir: Path) -> int:
    if not papers_dir.is_dir():
        return 0
    return sum(1 for p in papers_dir.glob("*.pdf") if p.is_file())


def collect_system_status(
    config: AppConfig | None = None,
    *,
    demo_runs_dir: Path | str | None = None,
) -> SystemStatus:
    cfg = config or load_config()
    papers_dir = Path(cfg.paths.papers_dir)
    processed = Path(cfg.paths.processed_dir)
    indexes = Path(cfg.paths.indexes_dir)
    manifest = Path(cfg.paths.corpus_manifest)
    dense_meta = indexes / "chroma" / "meta.json"
    sparse_meta = indexes / "bm25" / "meta.json"
    graph_path = processed / "knowledge_graph.json"
    graph_stats_path = processed / "graph_stats.json"

    repo = Path(__file__).resolve().parents[3]
    runs_dir = Path(demo_runs_dir) if demo_runs_dir else repo / "data" / "demo" / "runs"
    demo_runs = len(list(runs_dir.glob("*.json"))) if runs_dir.is_dir() else 0

    messages: list[str] = []
    n_papers = _count_jsonl(processed / "papers.jsonl")
    n_chunks = _count_jsonl(processed / "chunks.jsonl")
    n_pdfs = _count_pdfs(papers_dir)
    n_manifest = _count_jsonl(manifest) if manifest.is_file() else 0

    dense_ready = dense_meta.is_file()
    sparse_ready = sparse_meta.is_file() and (indexes / "bm25" / "bm25.pkl").is_file()
    graph_ready = graph_path.is_file()
    graph_nodes = None
    graph_edges = None
    if graph_stats_path.is_file():
        try:
            stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
            graph_nodes = stats.get("n_nodes")
            graph_edges = stats.get("n_edges")
        except json.JSONDecodeError:
            messages.append("graph_stats.json is not valid JSON")

    if n_papers == 0:
        messages.append("No processed papers — run ingest.")
    if n_chunks == 0:
        messages.append("No processed chunks — run ingest.")
    if not dense_ready or not sparse_ready:
        messages.append("Indexes missing — run index build (hash or st).")
    if not graph_ready:
        messages.append("Knowledge graph missing — graph retrieval will be disabled.")
    if demo_runs == 0:
        messages.append("No saved demo runs — replay mode limited.")

    ok = n_papers > 0 and n_chunks > 0 and dense_ready and sparse_ready
    return SystemStatus(
        ok=ok,
        papers_dir_exists=papers_dir.is_dir(),
        n_pdfs=n_pdfs,
        manifest_exists=manifest.is_file(),
        n_manifest_entries=n_manifest,
        processed_papers=n_papers,
        processed_chunks=n_chunks,
        dense_index_ready=dense_ready,
        sparse_index_ready=sparse_ready,
        graph_ready=graph_ready,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        demo_runs=demo_runs,
        messages=messages,
        paths={
            "papers_dir": str(papers_dir),
            "processed_dir": str(processed),
            "indexes_dir": str(indexes),
            "manifest": str(manifest),
            "demo_runs_dir": str(runs_dir),
        },
    )
