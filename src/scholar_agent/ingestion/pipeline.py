"""End-to-end PDF ingestion pipeline with idempotent canonical storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from scholar_agent.config import AppConfig, ChunkingConfig, load_config
from scholar_agent.ids import new_run_id
from scholar_agent.ingestion.chunker import chunk_sections
from scholar_agent.ingestion.headers import strip_headers_footers
from scholar_agent.ingestion.loader import PDFLoadError, file_content_hash, load_pages
from scholar_agent.ingestion.metadata import build_paper, resolve_pdf_path
from scholar_agent.ingestion.quality import assess_pages
from scholar_agent.ingestion.sections import pages_to_sections
from scholar_agent.ingestion.tokens import require_encoding
from scholar_agent.logging import get_logger
from scholar_agent.models.corpus import (
    Chunk,
    CorpusManifestEntry,
    IngestionStatus,
    Paper,
    PaperPage,
)
from scholar_agent.models.ingestion import (
    CorpusIngestionReport,
    ExtractionIssue,
    ExtractionSeverity,
    PaperExtractionReport,
    SectionPageText,
)
from scholar_agent.storage.jsonl import JsonlRepository
from scholar_agent.storage.manifest import load_corpus_manifest, save_corpus_manifest

logger = get_logger(__name__)

INGESTION_SCHEMA = "ingestion-v3-page-exact"


@dataclass
class IngestOptions:
    force: bool = False
    limit: int | None = None
    paper_ids: list[str] | None = None
    update_manifest: bool = True


@dataclass
class PaperIngestResult:
    entry: CorpusManifestEntry
    paper: Paper | None
    pages: list[PaperPage]
    chunks: list[Chunk]
    report: PaperExtractionReport
    status: IngestionStatus


class IngestionPipeline:
    """Parse PDFs into paper/chunk JSONL stores (canonical chunk store)."""

    def __init__(
        self,
        *,
        papers_dir: Path,
        processed_dir: Path,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.papers_dir = Path(papers_dir)
        self.processed_dir = Path(processed_dir)
        self.chunking = chunking or ChunkingConfig()
        self.tokenizer_backend = require_encoding(
            self.chunking.encoding_name,
            allow_fallback=self.chunking.allow_tokenizer_fallback,
        )
        fingerprint_payload = {
            "schema": INGESTION_SCHEMA,
            "target_tokens": self.chunking.target_tokens,
            "overlap_tokens": self.chunking.overlap_tokens,
            "min_tokens": self.chunking.min_tokens,
            "encoding_name": self.chunking.encoding_name,
            "tokenizer_backend": self.tokenizer_backend,
        }
        self.ingestion_config_fingerprint = sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.papers_repo: JsonlRepository[Paper] = JsonlRepository(
            self.processed_dir / "papers.jsonl", Paper
        )
        self.chunks_repo: JsonlRepository[Chunk] = JsonlRepository(
            self.processed_dir / "chunks.jsonl", Chunk
        )
        self.quality_repo: JsonlRepository[PaperExtractionReport] = JsonlRepository(
            self.processed_dir / "extraction_reports.jsonl", PaperExtractionReport
        )

    def _existing_papers(self) -> dict[str, Paper]:
        if not self.papers_repo.exists():
            return {}
        return self.papers_repo.index_by("paper_id")

    def ingest_entry(self, entry: CorpusManifestEntry, *, force: bool = False) -> PaperIngestResult:
        existing = self._existing_papers().get(entry.paper_id)
        try:
            pdf_path = resolve_pdf_path(entry, self.papers_dir)
        except FileNotFoundError as exc:
            report = PaperExtractionReport(
                paper_id=entry.paper_id,
                pdf_path=str(self.papers_dir / entry.pdf_filename),
                page_count=0,
                empty_page_count=0,
                scanned_suspect_page_count=0,
                total_chars=0,
                total_tokens_est=0,
                chunk_count=0,
                tokenizer_encoding=self.chunking.encoding_name,
                tokenizer_backend=self.tokenizer_backend,
                ingestion_config_fingerprint=self.ingestion_config_fingerprint,
                is_empty_paper=True,
                issues=[
                    ExtractionIssue(
                        code="missing_pdf",
                        severity=ExtractionSeverity.ERROR,
                        message=str(exc),
                    )
                ],
            )
            updated = entry.model_copy(update={"ingestion_status": IngestionStatus.FAILED})
            return PaperIngestResult(
                entry=updated,
                paper=None,
                pages=[],
                chunks=[],
                report=report,
                status=IngestionStatus.FAILED,
            )

        file_hash = file_content_hash(pdf_path)
        if (
            not force
            and existing is not None
            and existing.content_hash == file_hash
            and existing.ingestion_config_fingerprint == self.ingestion_config_fingerprint
            and self._chunks_exist_for(entry.paper_id)
        ):
            report = self._quality_report_for_skip(
                entry.paper_id,
                pdf_path=pdf_path,
                page_count=existing.page_count or 0,
            )
            updated = entry.model_copy(
                update={
                    "ingestion_status": IngestionStatus.INGESTED,
                    "content_hash": file_hash,
                }
            )
            return PaperIngestResult(
                entry=updated,
                paper=existing,
                pages=[],
                chunks=[],
                report=report,
                status=IngestionStatus.SKIPPED,
            )

        try:
            pages, _images = load_pages(entry.paper_id, pdf_path)
        except PDFLoadError as exc:
            report = PaperExtractionReport(
                paper_id=entry.paper_id,
                pdf_path=str(pdf_path),
                page_count=0,
                empty_page_count=0,
                scanned_suspect_page_count=0,
                total_chars=0,
                total_tokens_est=0,
                chunk_count=0,
                tokenizer_encoding=self.chunking.encoding_name,
                tokenizer_backend=self.tokenizer_backend,
                ingestion_config_fingerprint=self.ingestion_config_fingerprint,
                is_empty_paper=True,
                issues=[
                    ExtractionIssue(
                        code="pdf_load_error",
                        severity=ExtractionSeverity.ERROR,
                        message=str(exc),
                    )
                ],
            )
            updated = entry.model_copy(update={"ingestion_status": IngestionStatus.FAILED})
            return PaperIngestResult(
                entry=updated,
                paper=None,
                pages=[],
                chunks=[],
                report=report,
                status=IngestionStatus.FAILED,
            )

        cleaned = strip_headers_footers(pages)
        quality = assess_pages(
            entry.paper_id,
            str(pdf_path),
            cleaned,
            encoding_name=self.chunking.encoding_name,
        ).model_copy(
            update={
                "tokenizer_encoding": self.chunking.encoding_name,
                "tokenizer_backend": self.tokenizer_backend,
                "ingestion_config_fingerprint": self.ingestion_config_fingerprint,
            }
        )
        if quality.is_empty_paper:
            # Never silently index empty papers
            quality = quality.model_copy(update={"chunk_count": 0})
            updated = entry.model_copy(
                update={
                    "ingestion_status": IngestionStatus.FAILED,
                    "content_hash": file_hash,
                }
            )
            paper = build_paper(
                entry, pdf_path, page_count=len(cleaned), content_hash=file_hash
            ).model_copy(update={"ingestion_config_fingerprint": self.ingestion_config_fingerprint})
            return PaperIngestResult(
                entry=updated,
                paper=paper,
                pages=cleaned,
                chunks=[],
                report=quality,
                status=IngestionStatus.FAILED,
            )

        sections = pages_to_sections(cleaned, encoding_name=self.chunking.encoding_name)
        if not sections:
            # Fallback: single document-wide block preserving full page range
            from scholar_agent.models.ingestion import SectionBlock

            non_empty = [p for p in cleaned if p.text.strip()]
            if non_empty:
                sections = [
                    SectionBlock(
                        title=None,
                        page_start=non_empty[0].page_number,
                        page_end=non_empty[-1].page_number,
                        text="\n\n".join(p.text for p in non_empty),
                        page_texts=[
                            SectionPageText(page_number=p.page_number, text=p.text)
                            for p in non_empty
                        ],
                    )
                ]

        chunks = chunk_sections(
            entry.paper_id,
            sections,
            target_tokens=self.chunking.target_tokens,
            overlap_tokens=self.chunking.overlap_tokens,
            min_tokens=self.chunking.min_tokens,
            encoding_name=self.chunking.encoding_name,
        )
        quality = quality.model_copy(update={"chunk_count": len(chunks)})
        paper = build_paper(
            entry, pdf_path, page_count=len(cleaned), content_hash=file_hash
        ).model_copy(update={"ingestion_config_fingerprint": self.ingestion_config_fingerprint})

        self._persist_paper(paper, chunks)
        self._persist_quality_report(quality)

        updated = entry.model_copy(
            update={
                "ingestion_status": IngestionStatus.INGESTED,
                "content_hash": file_hash,
            }
        )
        return PaperIngestResult(
            entry=updated,
            paper=paper,
            pages=cleaned,
            chunks=chunks,
            report=quality,
            status=IngestionStatus.INGESTED,
        )

    def _chunks_exist_for(self, paper_id: str) -> bool:
        return self._count_chunks(paper_id) > 0

    def _count_chunks(self, paper_id: str) -> int:
        if not self.chunks_repo.exists():
            return 0
        return sum(1 for c in self.chunks_repo.iter_rows() if c.paper_id == paper_id)

    def _quality_report_for_skip(
        self,
        paper_id: str,
        *,
        pdf_path: Path,
        page_count: int,
    ) -> PaperExtractionReport:
        stored: PaperExtractionReport | None = None
        if self.quality_repo.exists():
            stored = self.quality_repo.index_by("paper_id").get(paper_id)
        if stored is None:
            stored = self._quality_from_previous_aggregate(paper_id)
        chunk_rows = (
            [c for c in self.chunks_repo.iter_rows() if c.paper_id == paper_id]
            if self.chunks_repo.exists()
            else []
        )
        if stored is None:
            # Compatibility for stores created before extraction_reports.jsonl.
            # Chunk-derived totals are preferable to silently zeroing quality
            # metadata, while the issue clearly records what cannot be recovered.
            stored = PaperExtractionReport(
                paper_id=paper_id,
                pdf_path=str(pdf_path),
                page_count=page_count,
                empty_page_count=0,
                scanned_suspect_page_count=0,
                total_chars=sum(len(chunk.text) for chunk in chunk_rows),
                total_tokens_est=sum(chunk.token_count for chunk in chunk_rows),
                chunk_count=len(chunk_rows),
                tokenizer_encoding=self.chunking.encoding_name,
                tokenizer_backend=self.tokenizer_backend,
                ingestion_config_fingerprint=self.ingestion_config_fingerprint,
                issues=[
                    ExtractionIssue(
                        code="legacy_quality_metadata",
                        severity=ExtractionSeverity.INFO,
                        message=(
                            "Exact extraction quality metadata predates durable reports; "
                            "character and token totals were reconstructed from chunks"
                        ),
                    )
                ],
            )
        report = stored.model_copy(
            update={
                "pdf_path": str(pdf_path),
                "page_count": page_count,
                "chunk_count": len(chunk_rows),
                "tokenizer_encoding": self.chunking.encoding_name,
                "tokenizer_backend": self.tokenizer_backend,
                "ingestion_config_fingerprint": self.ingestion_config_fingerprint,
                "skipped": True,
                "skip_reason": "unchanged content_hash; idempotent skip",
            }
        )
        self._persist_quality_report(
            report.model_copy(update={"skipped": False, "skip_reason": None})
        )
        return report

    def _quality_from_previous_aggregate(self, paper_id: str) -> PaperExtractionReport | None:
        path = self.processed_dir / "ingestion_report.json"
        if not path.is_file():
            return None
        try:
            aggregate = CorpusIngestionReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        candidates = [report for report in aggregate.paper_reports if report.paper_id == paper_id]
        if not candidates:
            return None
        report = candidates[-1]
        # A legacy idempotent-skip row contains the very zeros we are avoiding.
        if report.skipped and report.total_chars == 0 and report.total_tokens_est == 0:
            return None
        return report.model_copy(update={"skipped": False, "skip_reason": None})

    def _persist_quality_report(self, report: PaperExtractionReport) -> None:
        rows = (
            [row for row in self.quality_repo.read_all() if row.paper_id != report.paper_id]
            if self.quality_repo.exists()
            else []
        )
        rows.append(report)
        rows.sort(key=lambda row: row.paper_id)
        self.quality_repo.write_all(rows)

    def _persist_paper(self, paper: Paper, chunks: list[Chunk]) -> None:
        """Upsert paper and replace that paper's chunks (stable IDs when content stable)."""
        papers = [p for p in self.papers_repo.read_all() if p.paper_id != paper.paper_id]
        papers.append(paper)
        papers.sort(key=lambda p: p.paper_id)
        self.papers_repo.write_all(papers)

        other_chunks = [c for c in self.chunks_repo.read_all() if c.paper_id != paper.paper_id]
        # Deterministic order: by page then chunk_id
        new_chunks = sorted(chunks, key=lambda c: (c.page_start, c.page_end, c.chunk_id))
        all_chunks = other_chunks + new_chunks
        all_chunks.sort(key=lambda c: (c.paper_id, c.page_start, c.chunk_id))
        self.chunks_repo.write_all(all_chunks)


def ingest_corpus(
    *,
    config: AppConfig | None = None,
    manifest_path: Path | str | None = None,
    options: IngestOptions | None = None,
) -> CorpusIngestionReport:
    """Ingest all (or selected) papers from the corpus manifest."""
    cfg = config or load_config()
    opts = options or IngestOptions()
    path = Path(manifest_path) if manifest_path else cfg.paths.corpus_manifest
    manifest = load_corpus_manifest(path)

    entries = list(manifest.entries)
    if opts.paper_ids:
        wanted = set(opts.paper_ids)
        entries = [e for e in entries if e.paper_id in wanted]
    if opts.limit is not None:
        entries = entries[: opts.limit]

    pipeline = IngestionPipeline(
        papers_dir=cfg.paths.papers_dir,
        processed_dir=cfg.paths.processed_dir,
        chunking=cfg.chunking,
    )

    run_id = new_run_id()
    paper_reports: list[PaperExtractionReport] = []
    updated_by_id: dict[str, CorpusManifestEntry] = {e.paper_id: e for e in manifest.entries}

    ingested = skipped = failed = 0
    total_pages = total_chunks = 0
    empty_papers: list[str] = []
    scanned: list[str] = []

    for entry in entries:
        logger.info("ingesting %s (%s)", entry.paper_id, entry.pdf_filename)
        result = pipeline.ingest_entry(entry, force=opts.force)
        paper_reports.append(result.report)
        updated_by_id[entry.paper_id] = result.entry

        if result.status == IngestionStatus.INGESTED:
            ingested += 1
            total_pages += result.report.page_count
            total_chunks += len(result.chunks)
        elif result.status == IngestionStatus.SKIPPED:
            skipped += 1
            total_chunks += result.report.chunk_count
            total_pages += result.report.page_count
        else:
            failed += 1
            if result.report.is_empty_paper:
                empty_papers.append(entry.paper_id)

        if result.report.is_scanned_suspect:
            scanned.append(entry.paper_id)

    if opts.update_manifest:
        ordered = [updated_by_id[e.paper_id] for e in manifest.entries]
        save_corpus_manifest(path, ordered)

    report = CorpusIngestionReport(
        run_id=run_id,
        manifest_path=str(path),
        processed_dir=str(pipeline.processed_dir),
        tokenizer_encoding=pipeline.chunking.encoding_name,
        tokenizer_backend=pipeline.tokenizer_backend,
        ingestion_config_fingerprint=pipeline.ingestion_config_fingerprint,
        papers_attempted=len(entries),
        papers_ingested=ingested,
        papers_skipped=skipped,
        papers_failed=failed,
        total_pages=total_pages,
        total_chunks=total_chunks,
        empty_papers=empty_papers,
        scanned_suspect_papers=scanned,
        paper_reports=paper_reports,
        notes=[
            "Table/formula structure understanding is out of scope for Phase 2.",
            "Canonical chunk store: data/processed/chunks.jsonl",
        ],
    )

    report_path = pipeline.processed_dir / "ingestion_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", report_path)
    return report
