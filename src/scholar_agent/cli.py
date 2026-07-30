"""The four-command ScholarAgent CLI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from scholar_agent.config import Settings
from scholar_agent.ingest import ingest_directory
from scholar_agent.llm import LLMClient
from scholar_agent.retrieval import RetrievalEngine, build_all_indexes
from scholar_agent.workflow import run_question

app = typer.Typer(
    help="Compact multi-agent GraphRAG for evidence-grounded academic research.",
    add_completion=False,
    no_args_is_help=True,
)
LOGGER = logging.getLogger(__name__)


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
    """Build BM25, dense embeddings, and the entity graph."""
    settings = _settings()
    summary = build_all_indexes(settings)
    LOGGER.info(
        "[index] chunks=%d entities=%d edges=%d dense=%s",
        summary["chunks"],
        summary["entities"],
        summary["edges"],
        summary["dense_backend"],
    )
    typer.echo("Built BM25, dense, and graph indexes")


def _ask(question: str) -> str:
    settings = _settings()
    engine = RetrievalEngine.load(settings)
    llm = LLMClient.from_env(settings)
    state = run_question(question, engine, settings, llm)
    return state["answer"]


@app.command()
def ask(question: str) -> None:
    """Run the complete four-agent workflow."""
    typer.echo(_ask(question))


@app.command()
def demo() -> None:
    """Run the fixed interview demonstration question."""
    typer.echo(_ask("Compare Self-RAG and CRAG"))


if __name__ == "__main__":
    app()
