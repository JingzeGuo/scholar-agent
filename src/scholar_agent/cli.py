"""The three-command ScholarAgent CLI."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import typer
from openai import OpenAIError

from scholar_agent.config import Settings
from scholar_agent.indexes import ModelUnavailableError
from scholar_agent.ingest import ingest_directory
from scholar_agent.llm import LLMClient
from scholar_agent.retrieval import RetrievalEngine, build_all_indexes
from scholar_agent.workflow import run_question

app = typer.Typer(
    help="Compact agentic RAG for evidence-grounded academic research.",  # 项目说明
    add_completion=False,  # 不生成 shell 自动补全命令
    no_args_is_help=True,  # 不带参数时自动显示帮助
)
LOGGER = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    pass


# 统一初始化
def _settings() -> Settings:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    for library in ("huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(library).setLevel(logging.ERROR)
    return Settings.from_env()


@app.command()
def ingest(pdf_directory: Path) -> None:
    """Read PDFs and write page-aware chunks."""
    settings = _settings()
    chunks = ingest_directory(pdf_directory, settings.chunks_path)
    typer.echo(f"Ingested {len(chunks)} chunks from {pdf_directory}")


@app.command("index")
def build_indexes() -> None:
    """Build BM25 and dense indexes."""
    settings = _settings()
    try:
        summary = build_all_indexes(settings)
    except ModelUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    LOGGER.info(
        "[index] chunks=%d dense=%s",
        summary["chunks"],
        summary["dense_backend"],
    )
    typer.echo("Built BM25 and dense indexes")


def _ask(question: str, writer_emit: Callable[[str], None] | None = None) -> str:
    settings = _settings()
    llm = LLMClient.from_env(settings)
    if llm is None:
        raise MissingAPIKeyError
    engine = RetrievalEngine.load(settings)
    state = run_question(question, engine, settings, llm, writer_emit=writer_emit)
    return state["answer"]


@app.command()
def ask(question: str) -> None:
    """Run the evidence-grounded research workflow."""
    streamed = False
    ends_with_newline = False

    def emit(text: str) -> None:
        nonlocal streamed, ends_with_newline
        if not text:
            return
        typer.echo(text, nl=False)
        streamed = True
        ends_with_newline = text.endswith("\n")

    def finish_stream() -> None:
        if streamed and not ends_with_newline:
            typer.echo()

    try:
        answer = _ask(question, emit)
        if streamed:
            finish_stream()
        else:
            typer.echo(answer)
    except MissingAPIKeyError:
        finish_stream()
        typer.echo("LLM API key missing. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.", err=True)
        raise typer.Exit(code=2) from None
    except OpenAIError:
        finish_stream()
        typer.echo("LLM API unavailable.", err=True)
        raise typer.Exit(code=1) from None
    except ModelUnavailableError as exc:
        finish_stream()
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
