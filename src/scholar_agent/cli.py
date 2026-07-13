"""Typer CLI entrypoint for ScholarAgent.

Phase 0: version, config, prototype loop.
Phase 1: corpus validate / summary against the manifest.
Phase 2: PDF ingestion into canonical paper/chunk stores.
Phase 3: index build, retrieve, Naive RAG baseline.
Phase 4: knowledge graph build / inspect / graph retrieval.
Phase 5: adaptive routing + Research Agent tool loop.
Phase 6: Planner, Verifier, and corrective LangGraph workflow.
Phase 7: Evidence-constrained Writer + citation validator.
Phase 8: Evaluation framework (frozen split, ablations, metrics).
Phase 9: Streamlit demo + saved-run replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from scholar_agent import __version__
from scholar_agent.agents.prototype_loop import PrototypeLoopConfig, run_prototype_loop
from scholar_agent.config import load_config
from scholar_agent.ingestion.pipeline import IngestOptions, ingest_corpus
from scholar_agent.ingestion.quality import summarize_report
from scholar_agent.ingestion.tokens import TokenizerUnavailableError
from scholar_agent.logging import setup_logging
from scholar_agent.storage.manifest import (
    ManifestError,
    load_corpus_manifest,
    validate_corpus_manifest,
)

app = typer.Typer(
    name="scholar-agent",
    help="Evidence-driven multi-agent GraphRAG for literature research.",
    no_args_is_help=True,
    add_completion=False,
)
corpus_app = typer.Typer(help="Corpus manifest utilities.")
app.add_typer(corpus_app, name="corpus")
index_app = typer.Typer(help="Build and inspect retrieval indexes.")
app.add_typer(index_app, name="index")
graph_app = typer.Typer(help="Knowledge graph build and inspection.")
app.add_typer(graph_app, name="graph")
console = Console()

# Typer Option defaults as module-level callables avoid Ruff B008 on arg defaults.
_CONFIG_PATH_OPT = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to YAML config (default: configs/default.yaml)",
    exists=False,
)
_SHOW_SECRETS_OPT = typer.Option(
    False,
    "--show-secrets",
    help="Include API key presence only (never prints the key).",
)
_MAX_TOOL_CALLS_OPT = typer.Option(4, help="Max tool calls")
_MAX_ITERATIONS_OPT = typer.Option(3, help="Max decide iterations")
_REQUIRED_EVIDENCE_OPT = typer.Option(2, help="Useful observations required")
_JSON_OUTPUT_OPT = typer.Option(False, "--json", help="Emit machine-readable JSON")


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    console.print(f"scholar-agent {__version__}")


@app.command("config")
def config_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    show_secrets: bool = _SHOW_SECRETS_OPT,
) -> None:
    """Load and display validated configuration."""
    cfg = load_config(config_path)
    setup_logging(cfg)

    table = Table(title="ScholarAgent configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("project", f"{cfg.project.name} v{cfg.project.version}")
    table.add_row("config_path", str(cfg.config_path))
    table.add_row("llm.provider", cfg.llm.provider)
    table.add_row("llm.base_url", cfg.llm.base_url)
    table.add_row("llm.main_model", cfg.llm.main_model)
    table.add_row("llm.fast_model", cfg.llm.fast_model)
    table.add_row("llm.thinking_enabled", str(cfg.llm.thinking_enabled))
    if show_secrets:
        table.add_row("llm.api_key_set", str(bool(cfg.llm.api_key)))
    table.add_row(
        "budgets.max_tool_calls",
        str(cfg.budgets.max_tool_calls_per_research_pass),
    )
    table.add_row(
        "budgets.max_corrective_iterations",
        str(cfg.budgets.max_corrective_iterations),
    )
    table.add_row(
        "budgets.max_research_iterations",
        str(cfg.budgets.max_research_iterations_per_pass),
    )
    table.add_row("chunking.target_tokens", str(cfg.chunking.target_tokens))
    table.add_row("chunking.encoding_name", cfg.chunking.encoding_name)
    table.add_row(
        "chunking.allow_tokenizer_fallback",
        str(cfg.chunking.allow_tokenizer_fallback),
    )
    table.add_row("paths.processed_dir", str(cfg.paths.processed_dir))
    console.print(table)


@app.command("prototype")
def prototype_cmd(
    query: str = typer.Argument(..., help="Research question for the fake-model loop"),
    max_tool_calls: int = _MAX_TOOL_CALLS_OPT,
    max_iterations: int = _MAX_ITERATIONS_OPT,
    required_evidence: int = _REQUIRED_EVIDENCE_OPT,
    json_output: bool = _JSON_OUTPUT_OPT,
) -> None:
    """Run the Phase 0 LangGraph prototype loop (deterministic fake model)."""
    result = run_prototype_loop(
        query,
        config=PrototypeLoopConfig(
            max_tool_calls=max_tool_calls,
            max_iterations=max_iterations,
            required_evidence=required_evidence,
        ),
    )
    if json_output:
        console.print_json(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "query": result.query,
                    "success": result.success,
                    "answer": result.answer,
                    "iterations": result.iterations,
                    "tool_call_count": result.tool_call_count,
                    "terminated_reason": result.terminated_reason,
                    "events": [e.model_dump(mode="json") for e in result.events],
                }
            )
        )
        return

    console.print(f"[bold]run_id[/bold]: {result.run_id}")
    console.print(f"[bold]success[/bold]: {result.success}")
    console.print(f"[bold]terminated[/bold]: {result.terminated_reason}")
    console.print(f"[bold]iterations[/bold]: {result.iterations}")
    console.print(f"[bold]tool_calls[/bold]: {result.tool_call_count}")
    console.print(f"[bold]answer[/bold]: {result.answer}")
    console.print("[bold]events[/bold]:")
    for event in result.events:
        console.print(f"  - {event.event_type.value}: {event.summary}")


_MANIFEST_OPT = typer.Option(
    None,
    "--manifest",
    "-m",
    help="Path to corpus_manifest.jsonl (default: from config)",
    exists=False,
)
_PAPERS_DIR_OPT = typer.Option(
    None,
    "--papers-dir",
    help="Directory containing PDFs (default: from config)",
    exists=False,
)
_CHECK_PDFS_OPT = typer.Option(
    False,
    "--check-pdfs",
    help="Verify that listed PDF files exist on disk",
)


@corpus_app.command("validate")
def corpus_validate_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    manifest: Path | None = _MANIFEST_OPT,
    papers_dir: Path | None = _PAPERS_DIR_OPT,
    check_pdfs: bool = _CHECK_PDFS_OPT,
) -> None:
    """Validate corpus manifest schema and optional PDF presence."""
    cfg = load_config(config_path)
    setup_logging(cfg)
    manifest_path = manifest or cfg.paths.corpus_manifest
    pdf_dir = papers_dir or (cfg.paths.papers_dir if check_pdfs else None)

    issues = validate_corpus_manifest(
        manifest_path,
        papers_dir=pdf_dir if check_pdfs else None,
    )
    if issues:
        console.print(f"[red]Manifest invalid[/red]: {manifest_path}")
        for issue in issues:
            console.print(f"  - {issue}")
        raise typer.Exit(code=1)

    try:
        loaded = load_corpus_manifest(manifest_path)
    except ManifestError as exc:
        console.print(f"[red]Manifest invalid[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Manifest OK[/green]: {manifest_path}")
    console.print(f"entries: {len(loaded)}")
    by_status: dict[str, int] = {}
    for entry in loaded.entries:
        key = entry.ingestion_status.value
        by_status[key] = by_status.get(key, 0) + 1
    for status, count in sorted(by_status.items()):
        console.print(f"  {status}: {count}")


@corpus_app.command("summary")
def corpus_summary_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    manifest: Path | None = _MANIFEST_OPT,
) -> None:
    """Print a short corpus manifest summary table."""
    cfg = load_config(config_path)
    manifest_path = manifest or cfg.paths.corpus_manifest
    try:
        loaded = load_corpus_manifest(manifest_path)
    except ManifestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Corpus manifest ({len(loaded)} papers)")
    table.add_column("paper_id", style="cyan")
    table.add_column("year")
    table.add_column("status")
    table.add_column("title")
    for entry in loaded.entries:
        table.add_row(
            entry.paper_id,
            str(entry.year or ""),
            entry.ingestion_status.value,
            entry.title[:60] + ("…" if len(entry.title) > 60 else ""),
        )
    console.print(table)


_FORCE_OPT = typer.Option(False, "--force", help="Re-ingest even if content_hash unchanged")
_LIMIT_OPT = typer.Option(None, "--limit", help="Max papers to ingest this run")
_PAPER_ID_OPT = typer.Option(
    None,
    "--paper-id",
    help="Ingest only this paper_id (repeatable)",
)
_NO_MANIFEST_UPDATE_OPT = typer.Option(
    False,
    "--no-manifest-update",
    help="Do not write ingestion_status back to the manifest",
)
_ALLOW_TOKENIZER_FALLBACK_OPT = typer.Option(
    False,
    "--allow-tokenizer-fallback",
    help=(
        "Explicitly allow the deterministic non-canonical tokenizer. "
        "This changes chunk IDs/fingerprints."
    ),
)


@app.command("ingest")
def ingest_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    manifest: Path | None = _MANIFEST_OPT,
    force: bool = _FORCE_OPT,
    limit: int | None = _LIMIT_OPT,
    paper_id: list[str] | None = _PAPER_ID_OPT,
    no_manifest_update: bool = _NO_MANIFEST_UPDATE_OPT,
    allow_tokenizer_fallback: bool = _ALLOW_TOKENIZER_FALLBACK_OPT,
) -> None:
    """Ingest PDFs from the corpus manifest into data/processed/."""
    cfg = load_config(config_path)
    if allow_tokenizer_fallback:
        cfg = cfg.model_copy(
            update={"chunking": cfg.chunking.model_copy(update={"allow_tokenizer_fallback": True})}
        )
    setup_logging(cfg)
    # Allow CLI papers override via config path only; papers_dir comes from config
    try:
        report = ingest_corpus(
            config=cfg,
            manifest_path=manifest,
            options=IngestOptions(
                force=force,
                limit=limit,
                paper_ids=list(paper_id) if paper_id else None,
                update_manifest=not no_manifest_update,
            ),
        )
    except TokenizerUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(summarize_report(report))
    console.print(f"report: {cfg.paths.processed_dir / 'ingestion_report.json'}")
    if report.papers_failed and report.papers_ingested == 0 and report.papers_skipped == 0:
        raise typer.Exit(code=1)


_EMBED_BACKEND_OPT = typer.Option(
    "auto",
    "--embedding-backend",
    help="auto | hash (offline) | st (sentence-transformers)",
)
_FORCE_INDEX_OPT = typer.Option(False, "--force", help="Rebuild indexes even if fingerprints match")
_MODE_OPT = typer.Option(
    "hybrid_rerank",
    "--mode",
    help="dense | sparse | hybrid | hybrid_rerank | graph",
)
_GRAPH_FORCE_OPT = typer.Option(False, "--force", help="Rebuild graph even if artifacts exist")
_GRAPH_LIMIT_OPT = typer.Option(
    None,
    "--limit-chunks",
    help="Only process first N chunks (debug)",
)
_GRAPH_SAMPLE_OPT = typer.Option(8, help="Number of sample edges to show")
_GRAPH_LLM_RESOLUTION_OPT = typer.Option(
    False,
    "--llm-resolution",
    help="Use DeepSeek only for ambiguous entity candidate pairs",
)
_GRAPH_MAX_LLM_OPT = typer.Option(
    50,
    "--max-llm-resolutions",
    min=0,
    help="Maximum ambiguous entity pairs sent to the LLM",
)
_TOP_K_OPT = typer.Option(None, "--k", help="Result count override")
_JSON_OPT = typer.Option(False, "--json", help="Machine-readable JSON output")
_DEBUG_OPT = typer.Option(False, "--debug", help="Include retrieval debug ranks")
_USE_LLM_OPT = typer.Option(
    False,
    "--llm",
    help="Call the configured LLM for Naive RAG (otherwise extractive baseline)",
)


@index_app.command("build")
def index_build_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    force: bool = _FORCE_INDEX_OPT,
) -> None:
    """Build dense (Chroma) + BM25 indexes from data/processed/chunks.jsonl."""
    from scholar_agent.retrieval.index_builder import build_indexes

    cfg = load_config(config_path)
    setup_logging(cfg)
    if embedding_backend not in {"auto", "hash", "st"}:
        console.print("[red]embedding-backend must be auto|hash|st[/red]")
        raise typer.Exit(code=1)
    backend = embedding_backend  # narrowed below
    built = build_indexes(
        config=cfg,
        embedding_backend=backend,  # type: ignore[arg-type]
        force=force,
    )
    console.print(
        f"[green]Indexes ready[/green]: chunks={len(built.store)} "
        f"fingerprint={built.store.fingerprint[:16]}…"
    )
    console.print(f"  dense:  {built.dense_dir}")
    console.print(f"  sparse: {built.sparse_dir}")
    console.print(f"  embedder: {built.dense.embedder.model_name}")


@app.command("retrieve")
def retrieve_cmd(
    query: str = typer.Argument(..., help="Search query"),
    config_path: Path | None = _CONFIG_PATH_OPT,
    mode: str = _MODE_OPT,
    k: int | None = _TOP_K_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    json_output: bool = _JSON_OPT,
    debug: bool = _DEBUG_OPT,
) -> None:
    """Run dense / sparse / hybrid retrieval against built indexes."""
    from scholar_agent.retrieval.index_builder import load_toolkit

    cfg = load_config(config_path)
    setup_logging(cfg)
    if mode not in {"dense", "sparse", "hybrid", "hybrid_rerank", "graph"}:
        console.print("[red]mode must be dense|sparse|hybrid|hybrid_rerank|graph[/red]")
        raise typer.Exit(code=1)
    if embedding_backend not in {"auto", "hash", "st"}:
        embedding_backend = "auto"
    toolkit = load_toolkit(
        config=cfg,
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        reranker_backend="lexical" if embedding_backend == "hash" else "auto",
    )
    result = toolkit.search(
        query,
        mode=mode,  # type: ignore[arg-type]
        k=k,
    )
    if json_output:
        payload = result.model_dump(mode="json")
        if debug:
            payload["debug_view"] = toolkit.debug_dict(result)
        console.print_json(json.dumps(payload))
        return

    console.print(f"[bold]method[/bold]: {result.method}  hits={len(result.hits)}")
    for i, hit in enumerate(result.hits, start=1):
        console.print(
            f"[cyan]{i}.[/cyan] {hit.paper_id} {hit.page_label()} "
            f"score={hit.score!s} dense={hit.dense_rank} sparse={hit.sparse_rank} "
            f"rerank={hit.rerank_score}"
        )
        console.print(f"   chunk={hit.chunk_id}", markup=False)
        console.print(f"   {hit.snippet(200)}", markup=False)
    if debug:
        console.print_json(json.dumps(toolkit.debug_dict(result)))


@app.command("ask-naive")
def ask_naive_cmd(
    query: str = typer.Argument(..., help="Question for Naive RAG baseline"),
    config_path: Path | None = _CONFIG_PATH_OPT,
    mode: str = _MODE_OPT,
    k: int | None = _TOP_K_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    use_llm: bool = _USE_LLM_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Naive RAG baseline: retrieve top-k and answer with page citations."""
    from scholar_agent.llm.client import create_llm_client
    from scholar_agent.retrieval.index_builder import load_toolkit
    from scholar_agent.retrieval.naive_rag import NaiveRAG

    cfg = load_config(config_path)
    setup_logging(cfg)
    if embedding_backend not in {"auto", "hash", "st"}:
        embedding_backend = "auto"
    toolkit = load_toolkit(
        config=cfg,
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        reranker_backend="lexical" if embedding_backend == "hash" else "auto",
    )
    llm = None
    if use_llm:
        try:
            llm = create_llm_client(cfg)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]LLM unavailable ({exc}); using extractive baseline[/yellow]")
    search_mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = (
        mode  # type: ignore[assignment]
        if mode in {"dense", "sparse", "hybrid", "hybrid_rerank"}
        else "hybrid_rerank"
    )
    rag = NaiveRAG(
        toolkit,
        llm=llm,
        mode=search_mode,
        top_k=k or cfg.retrieval.reranker.top_k,
    )
    answer = rag.answer(query, use_llm=use_llm and llm is not None)
    if json_output:
        console.print_json(json.dumps(answer.model_dump(mode="json")))
        return
    console.print(f"[bold]method[/bold]: {answer.method}  llm={answer.used_llm}")
    # markup=False: citation brackets like [paper_id p.3] must not be parsed as Rich tags
    console.print(answer.answer, markup=False)
    if answer.citations:
        console.print("[bold]citations[/bold]:")
        for c in answer.citations:
            console.print(
                f"  - {c.marker} {c.format_inline()} chunk={c.chunk_id}",
                markup=False,
            )


@graph_app.command("build")
def graph_build_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    force: bool = _GRAPH_FORCE_OPT,
    limit_chunks: int | None = _GRAPH_LIMIT_OPT,
    llm_resolution: bool = _GRAPH_LLM_RESOLUTION_OPT,
    max_llm_resolutions: int = _GRAPH_MAX_LLM_OPT,
) -> None:
    """Extract entities/relations and persist the evidence-linked knowledge graph."""
    from scholar_agent.graph.pipeline import build_knowledge_graph

    cfg = load_config(config_path)
    setup_logging(cfg)
    result = build_knowledge_graph(
        config=cfg,
        force=force,
        limit_chunks=limit_chunks,
        use_llm_resolution=llm_resolution,
        max_llm_resolutions=max_llm_resolutions,
    )
    stats = result.stats
    console.print(
        f"[green]Graph ready[/green]: nodes={stats.n_nodes} edges={stats.n_edges} "
        f"isolated={stats.n_isolated_nodes} "
        f"isolated_rate={stats.isolated_node_rate:.3f}"
    )
    console.print(f"  entities:  {result.entities_path}")
    console.print(f"  relations: {result.relations_path}")
    console.print(f"  graph:     {result.graph_path}")
    console.print(f"  stats:     {result.stats_path}")
    console.print(f"  metadata:  {result.meta_path}")
    console.print("  node types:", stats.node_type_counts)
    console.print("  relation types:", stats.relation_type_counts)
    if stats.n_relations_missing_evidence:
        console.print(
            f"[yellow]warning[/yellow]: {stats.n_relations_missing_evidence} edges missing evidence"
        )


@graph_app.command("inspect")
def graph_inspect_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    sample: int = _GRAPH_SAMPLE_OPT,
) -> None:
    """Print graph statistics and sample evidence-linked edges."""
    from scholar_agent.graph.pipeline import validate_graph_build_meta
    from scholar_agent.graph.stats import compute_graph_stats
    from scholar_agent.graph.store import KnowledgeGraphStore
    from scholar_agent.retrieval.chunk_store import ChunkStore

    cfg = load_config(config_path)
    graph_path = cfg.paths.processed_dir / "knowledge_graph.json"
    if not graph_path.is_file():
        console.print(f"[red]Graph not found:[/red] {graph_path}. Run graph build first.")
        raise typer.Exit(code=1)
    chunk_store = ChunkStore.from_processed_dir(cfg.paths.processed_dir)
    current, reason = validate_graph_build_meta(
        cfg.paths.processed_dir / "graph_meta.json",
        corpus_fingerprint=chunk_store.fingerprint,
    )
    if not current:
        console.print(f"[red]Graph artifacts are stale:[/red] {reason}. Run graph build.")
        raise typer.Exit(code=1)
    store = KnowledgeGraphStore.load_node_link_json(graph_path)
    stats = compute_graph_stats(store)
    console.print(
        f"nodes={stats.n_nodes} edges={stats.n_edges} "
        f"isolated={stats.n_isolated_nodes} rate={stats.isolated_node_rate:.3f}"
    )
    console.print("node_types:", stats.node_type_counts)
    console.print("relation_types:", stats.relation_type_counts)
    console.print(
        f"evidence coverage: {stats.n_relations_with_evidence}/"
        f"{stats.n_relations_with_evidence + stats.n_relations_missing_evidence}"
    )
    rels = store.relations()[:sample]
    console.print(f"[bold]sample edges[/bold] ({len(rels)}):")
    for rel in rels:
        page_label = (
            f"p.{rel.page_number}"
            if rel.page_end == rel.page_number
            else f"p.{rel.page_number}-{rel.page_end}"
        )
        console.print(
            f"  - {rel.relation_type.value}: {rel.subject_surface!r} → {rel.object_surface!r} "
            f"[{rel.paper_id} {page_label}] chunk={rel.chunk_id}",
            markup=False,
        )
        console.print(f"    evidence: {rel.evidence_span[:160]}", markup=False)


@graph_app.command("stats")
def graph_stats_cmd(config_path: Path | None = _CONFIG_PATH_OPT) -> None:
    """Emit graph statistics as JSON."""
    from scholar_agent.graph.pipeline import validate_graph_build_meta
    from scholar_agent.graph.stats import compute_graph_stats
    from scholar_agent.graph.store import KnowledgeGraphStore
    from scholar_agent.retrieval.chunk_store import ChunkStore

    cfg = load_config(config_path)
    graph_path = cfg.paths.processed_dir / "knowledge_graph.json"
    if not graph_path.is_file():
        console.print(f"[red]Graph not found:[/red] {graph_path}")
        raise typer.Exit(code=1)
    chunk_store = ChunkStore.from_processed_dir(cfg.paths.processed_dir)
    current, reason = validate_graph_build_meta(
        cfg.paths.processed_dir / "graph_meta.json",
        corpus_fingerprint=chunk_store.fingerprint,
    )
    if not current:
        console.print(f"[red]Graph artifacts are stale:[/red] {reason}. Run graph build.")
        raise typer.Exit(code=1)
    store = KnowledgeGraphStore.load_node_link_json(graph_path)
    stats = compute_graph_stats(store)
    console.print_json(json.dumps(stats.model_dump(mode="json")))


_RESEARCH_MAX_TOOLS_OPT = typer.Option(
    None,
    "--max-tools",
    help="Max tool calls per sub-question (default from config)",
)
_RESEARCH_MAX_EVIDENCE_OPT = typer.Option(
    None,
    "--max-evidence",
    help="Max evidence items per sub-question",
)
_RESEARCH_MAX_ITERATIONS_OPT = typer.Option(
    None,
    "--max-iterations",
    help="Max inspect/act iterations per sub-question",
)
_NO_PARALLEL_OPT = typer.Option(
    False,
    "--no-parallel",
    help="Disable parallel sub-question research",
)


@app.command("research")
def research_cmd(
    query: str = typer.Argument(..., help="Research question for the Research Agent"),
    config_path: Path | None = _CONFIG_PATH_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    max_tools: int | None = _RESEARCH_MAX_TOOLS_OPT,
    max_evidence: int | None = _RESEARCH_MAX_EVIDENCE_OPT,
    max_iterations: int | None = _RESEARCH_MAX_ITERATIONS_OPT,
    no_parallel: bool = _NO_PARALLEL_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Run Phase 5 Research Agent: route → tool loop → evidence ledger."""
    from scholar_agent.agents.researcher import ResearchAgent, ResearchAgentConfig
    from scholar_agent.models.planning import SubQuestion, SubQuestionStatus
    from scholar_agent.retrieval.index_builder import load_toolkit
    from scholar_agent.retrieval.router import classify_query_type, recommend_policy

    cfg = load_config(config_path)
    setup_logging(cfg)
    if embedding_backend not in {"auto", "hash", "st"}:
        embedding_backend = "auto"
    toolkit = load_toolkit(
        config=cfg,
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        reranker_backend="lexical" if embedding_backend == "hash" else "auto",
        load_graph=True,
    )
    agent_cfg = ResearchAgentConfig(
        max_tool_calls_per_pass=max_tools or cfg.budgets.max_tool_calls_per_research_pass,
        max_evidence_per_sub_question=max_evidence or cfg.budgets.max_evidence_per_sub_question,
        max_iterations_per_pass=max_iterations or cfg.budgets.max_research_iterations_per_pass,
        max_latency_ms=cfg.budgets.max_latency_ms,
        max_total_tokens_per_pass=cfg.budgets.max_total_tokens,
    )
    agent = ResearchAgent(toolkit, config=agent_cfg)

    # Light multi-subquestion split for comparison queries (full Planner is Phase 6)
    qtype, _ = classify_query_type(query)
    routing_preview = recommend_policy(query, query_type=qtype, has_graph=toolkit.graph is not None)
    sub_questions = [
        SubQuestion(
            id="sq_0",
            question=query,
            query_type=qtype,
            required_evidence=["supporting passages"],
            status=SubQuestionStatus.PENDING,
        )
    ]
    lowered = query.lower()
    if qtype.value == "comparison" and (" vs " in lowered or " versus " in lowered):
        # Optional second focus sub-question for diversity (full Planner is Phase 6)
        sub_questions.append(
            SubQuestion(
                id="sq_1",
                question=f"Key differences for: {query}",
                query_type=qtype,
                required_evidence=["comparative evidence"],
                status=SubQuestionStatus.PENDING,
            )
        )

    result = agent.research_many(
        sub_questions,
        original_query=query,
        parallel=not no_parallel,
    )

    if json_output:
        console.print_json(json.dumps(result.model_dump(mode="json")))
        return

    console.print(f"[bold]run_id[/bold]: {result.run_id}")
    console.print(
        f"[bold]router[/bold]: {routing_preview.query_type.value} → "
        f"{routing_preview.recommended_policy.value}"
    )
    console.print(f"  {routing_preview.rationale}", markup=False)
    console.print(
        f"[bold]tools[/bold]: {result.tool_call_count}  "
        f"iterations={result.iteration_count}  "
        f"estimated_tokens={result.token_usage.total_tokens}  "
        f"evidence={len(result.evidence_ledger.items)}  "
        f"parallel={result.parallel}"
    )
    for p in result.passes:
        tools = ", ".join(a.tool_name for a in p.actions) or "(none)"
        console.print(
            f"  - {p.sub_question_id}: policy={p.routing.recommended_policy.value} "
            f"tools=[{tools}] evidence={len(p.evidence)} end={p.terminated_reason}",
            markup=False,
        )
    if result.evidence_ledger.items:
        console.print("[bold]evidence sample[/bold]:")
        for item in result.evidence_ledger.items[:5]:
            console.print(
                f"  - [{item.paper_id} p.{item.page_start}-{item.page_end}] "
                f"{item.retrieval_method}: {item.claim[:120]}",
                markup=False,
            )


_ASK_MAX_ITER_OPT = typer.Option(
    None,
    "--max-iterations",
    help="Max corrective iterations (default from config)",
)


@app.command("ask")
def ask_cmd(
    query: str = typer.Argument(..., help="Complex research question"),
    config_path: Path | None = _CONFIG_PATH_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    max_iterations: int | None = _ASK_MAX_ITER_OPT,
    max_tools: int | None = _RESEARCH_MAX_TOOLS_OPT,
    json_output: bool = _JSON_OPT,
    no_parallel: bool = _NO_PARALLEL_OPT,
) -> None:
    """Full plan → research → verify → write → citation validate (Phases 6–7)."""
    from scholar_agent.agents.researcher import ResearchAgentConfig
    from scholar_agent.agents.workflow import ResearchWorkflow, WorkflowConfig
    from scholar_agent.retrieval.index_builder import load_toolkit

    cfg = load_config(config_path)
    setup_logging(cfg)
    if embedding_backend not in {"auto", "hash", "st"}:
        embedding_backend = "auto"
    toolkit = load_toolkit(
        config=cfg,
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        reranker_backend="lexical" if embedding_backend == "hash" else "auto",
        load_graph=True,
    )
    wf_cfg = WorkflowConfig(
        max_corrective_iterations=(
            max_iterations if max_iterations is not None else cfg.budgets.max_corrective_iterations
        ),
        max_latency_ms=cfg.budgets.max_latency_ms,
        max_total_tokens=cfg.budgets.max_total_tokens,
        research=ResearchAgentConfig(
            max_tool_calls_per_pass=max_tools or cfg.budgets.max_tool_calls_per_research_pass,
            max_iterations_per_pass=cfg.budgets.max_research_iterations_per_pass,
            max_evidence_per_sub_question=cfg.budgets.max_evidence_per_sub_question,
            max_latency_ms=cfg.budgets.max_latency_ms,
            max_total_tokens_per_pass=cfg.budgets.max_total_tokens,
        ),
        parallel_research=not no_parallel,
    )
    result = ResearchWorkflow(toolkit, config=wf_cfg).run(query)

    if json_output:
        console.print_json(json.dumps(result.model_dump(mode="json")))
        return

    console.print(f"[bold]run_id[/bold]: {result.run_id}")
    console.print(f"[bold]terminated[/bold]: {result.terminated_reason}")
    console.print(
        f"[bold]plan[/bold]: {result.plan.answer_type} "
        f"({len(result.plan.sub_questions)} sub-questions)"
    )
    for sq in result.plan.sub_questions:
        console.print(
            f"  - [{sq.status.value}] {sq.id}: {sq.question}",
            markup=False,
        )
    v = result.verification
    console.print(
        f"[bold]verification[/bold]: sufficient={v.is_sufficient} coverage={v.coverage_score:.2f}"
    )
    console.print(f"  {v.rationale_summary}", markup=False)
    if v.conflicting_evidence_ids:
        console.print(
            f"  conflicts: {', '.join(v.conflicting_evidence_ids[:8])}",
            markup=False,
        )
    if v.corrective_queries and not v.is_sufficient:
        console.print("[bold]last corrective queries[/bold]:")
        for cq in v.corrective_queries[:5]:
            console.print(f"  - {cq}", markup=False)
    console.print(
        f"[bold]stats[/bold]: iteration={result.iteration} "
        f"tools={result.tool_call_count} evidence={len(result.evidence_ledger.items)} "
        f"estimated_tokens={result.token_usage.total_tokens} "
        f"latency_ms={result.latency_ms} unanswerable={result.unanswerable}"
    )
    if result.final_answer is not None:
        fa = result.final_answer
        report = fa.citation_report
        console.print(
            f"[bold]answer[/bold]: claims={len(fa.claims)} "
            f"sources={len(fa.source_cards)} "
            f"citation_valid={report.is_valid if report else None} "
            f"corpus_insufficient={fa.corpus_insufficient}"
        )
        console.print(fa.markdown, markup=False)
        if fa.sources:
            console.print("[bold]source list[/bold]:")
            for src in fa.sources[:12]:
                console.print(f"  - {src}", markup=False)
        if report and report.issues:
            console.print("[bold]citation issues[/bold]:")
            for issue in report.issues[:8]:
                console.print(
                    f"  - [{issue.severity}] {issue.claim_id or issue.evidence_id or '—'}: "
                    f"{issue.message}",
                    markup=False,
                )
    elif result.evidence_ledger.items:
        console.print("[bold]evidence sample[/bold]:")
        for item in result.evidence_ledger.items[:5]:
            console.print(
                f"  - [{item.paper_id} p.{item.page_start}-{item.page_end}] "
                f"{item.retrieval_method}: {item.claim[:120]}",
                markup=False,
            )


_EVAL_CONFIG_OPT = typer.Option(
    None,
    "--eval-config",
    help="Path to configs/evaluation.yaml",
    exists=False,
)
_EVAL_SYSTEMS_OPT = typer.Option(
    None,
    "--system",
    help="System to run (repeatable). Default: all systems in eval config.",
)
_EVAL_MAX_Q_OPT = typer.Option(
    None,
    "--max-questions",
    help="Limit number of frozen questions (debug smoke runs)",
)
_EVAL_OUT_OPT = typer.Option(
    None,
    "--output-dir",
    help="Directory for results JSON/CSV/charts (default: outputs/evaluation)",
)
_EVAL_RAGAS_OPT = typer.Option(
    False,
    "--ragas",
    help="Enable optional RAGAS metrics (requires scholar-agent[eval] + LLM)",
)
_EVAL_LLM_OPT = typer.Option(
    False,
    "--llm",
    help="Use one shared live LLM prompt for every applicable ablation system",
)
_EVAL_TOP_K_OPT = typer.Option(8, "--k", help="Top-k for retrieval metrics")
_EVAL_NO_CHARTS_OPT = typer.Option(False, "--no-charts", help="Skip SVG chart generation")
_EVAL_ALL_OPT = typer.Option(
    False,
    "--all",
    help="Run every configured ablation system (the default when --system is omitted)",
)


@app.command("evaluate")
def evaluate_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    eval_config: Path | None = _EVAL_CONFIG_OPT,
    system: list[str] | None = _EVAL_SYSTEMS_OPT,
    max_questions: int | None = _EVAL_MAX_Q_OPT,
    output_dir: Path | None = _EVAL_OUT_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    top_k: int = _EVAL_TOP_K_OPT,
    use_ragas: bool = _EVAL_RAGAS_OPT,
    use_llm: bool = _EVAL_LLM_OPT,
    no_charts: bool = _EVAL_NO_CHARTS_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Run frozen-split ablations (Phase 8). Offline deterministic by default."""
    from scholar_agent.evaluation.runner import run_evaluation

    cfg = load_config(config_path)
    setup_logging(cfg)
    if embedding_backend not in {"auto", "hash", "st"}:
        embedding_backend = "hash"
    result = run_evaluation(
        config=cfg,
        eval_config_path=eval_config,
        systems=list(system) if system else None,
        max_questions=max_questions,
        embedding_backend=embedding_backend,
        top_k=top_k,
        use_ragas=use_ragas,
        use_llm=use_llm,
        output_dir=output_dir,
        write_charts=not no_charts,
    )
    if json_output:
        console.print_json(
            json.dumps(
                {
                    "run_id": result.report.run_id,
                    "fingerprint": result.dataset_fingerprint,
                    "n_questions": result.n_questions,
                    "systems": result.systems,
                    "output_paths": result.output_paths,
                    "aggregate": [s.model_dump(mode="json") for s in result.report.systems],
                }
            )
        )
        return

    console.print(f"[bold]run_id[/bold]: {result.report.run_id}")
    console.print(f"[bold]fingerprint[/bold]: {result.dataset_fingerprint}")
    console.print(
        f"[bold]questions[/bold]: {result.n_questions}  systems={', '.join(result.systems)}"
    )
    table = Table(title="Aggregate metrics")
    table.add_column("system", style="cyan")
    table.add_column("recall@paper")
    table.add_column("mrr")
    table.add_column("cite_p")
    table.add_column("token_f1")
    table.add_column("latency_ms")
    table.add_column("tools")
    for s in result.report.systems:
        table.add_row(
            s.system,
            f"{s.recall_at_k_paper:.3f}",
            f"{s.mrr:.3f}",
            f"{s.citation_precision:.3f}",
            f"{s.token_f1:.3f}",
            f"{s.avg_latency_ms:.0f}",
            f"{s.avg_tool_calls:.2f}",
        )
    console.print(table)
    console.print(f"[bold]outputs[/bold]: {result.output_paths.get('results_json')}")
    if result.report.failures:
        console.print(
            f"[yellow]failures logged[/yellow]: {len(result.report.failures)} "
            f"→ {result.output_paths.get('failures_json')}"
        )


@app.command("ablate")
def ablate_cmd(
    config_path: Path | None = _CONFIG_PATH_OPT,
    eval_config: Path | None = _EVAL_CONFIG_OPT,
    system: list[str] | None = _EVAL_SYSTEMS_OPT,
    max_questions: int | None = _EVAL_MAX_Q_OPT,
    output_dir: Path | None = _EVAL_OUT_OPT,
    embedding_backend: str = _EMBED_BACKEND_OPT,
    top_k: int = _EVAL_TOP_K_OPT,
    use_ragas: bool = _EVAL_RAGAS_OPT,
    use_llm: bool = _EVAL_LLM_OPT,
    no_charts: bool = _EVAL_NO_CHARTS_OPT,
    all_systems: bool = _EVAL_ALL_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Run all ablation systems on the frozen split (alias of ``evaluate``)."""
    if all_systems:
        system = None
    evaluate_cmd(
        config_path=config_path,
        eval_config=eval_config,
        system=system,
        max_questions=max_questions,
        output_dir=output_dir,
        embedding_backend=embedding_backend,
        top_k=top_k,
        use_ragas=use_ragas,
        use_llm=use_llm,
        no_charts=no_charts,
        json_output=json_output,
    )


_DEMO_PORT_OPT = typer.Option(8501, "--port", help="Streamlit server port")
_DEMO_REPLAY_OPT = typer.Option(
    None,
    "--replay",
    help="Print a saved demo run as JSON instead of launching Streamlit",
)


@app.command("demo")
def demo_cmd(
    port: int = _DEMO_PORT_OPT,
    replay: str | None = _DEMO_REPLAY_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """Launch Streamlit demo, or dump a saved offline replay."""
    if replay:
        from scholar_agent.app.demo_runs import find_saved_run

        saved = find_saved_run(replay)
        if saved is None:
            console.print(f"[red]Saved demo not found:[/red] {replay}")
            console.print("Available:")
            from scholar_agent.app.demo_runs import list_saved_runs

            for run in list_saved_runs():
                console.print(f"  - {run.demo_id}: {run.title}")
            raise typer.Exit(code=1)
        if json_output:
            console.print_json(json.dumps(saved.model_dump(mode="json")))
        else:
            console.print(f"[bold]{saved.title}[/bold] (`{saved.demo_id}`)")
            console.print(saved.query, markup=False)
            console.print(saved.session.answer_markdown, markup=False)
        return

    import shutil
    import subprocess
    import sys

    if shutil.which("streamlit") is None and not _streamlit_importable():
        console.print(
            "[red]Streamlit not installed.[/red] Install with: [cyan]uv sync --extra ui[/cyan]"
        )
        raise typer.Exit(code=1)

    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    console.print(f"Starting demo: {' '.join(cmd)}")
    raise typer.Exit(code=subprocess.call(cmd))


def _streamlit_importable() -> bool:
    try:
        import importlib

        importlib.import_module("streamlit")
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    app()


if __name__ == "__main__":
    main()
