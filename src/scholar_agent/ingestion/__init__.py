"""PDF ingestion: load, parse pages/sections, chunk, persist canonical stores."""

from scholar_agent.ingestion.pipeline import IngestionPipeline, IngestOptions, ingest_corpus
from scholar_agent.ingestion.quality import summarize_report

__all__ = [
    "IngestOptions",
    "IngestionPipeline",
    "ingest_corpus",
    "summarize_report",
]
