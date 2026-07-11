"""Typer CLI entrypoint for ScholarAgent.

Phase 0 exposes scaffolding commands: version, config show, and prototype loop.
Full corpus / ask / evaluate commands arrive in later phases.
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

app = typer.Typer(
    name="scholar-agent",
    help="Evidence-driven multi-agent GraphRAG for literature research.",
    no_args_is_help=True,
    add_completion=False,
)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
