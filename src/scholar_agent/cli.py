"""Typer CLI entrypoint for ScholarAgent.

Phase 0: version, config, prototype loop.
Phase 1: corpus validate / summary against the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from scholar_agent import __version__
from scholar_agent.agents.prototype_loop import PrototypeLoopConfig, run_prototype_loop
from scholar_agent.config import load_config
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
    table.add_row("chunking.target_tokens", str(cfg.chunking.target_tokens))
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
