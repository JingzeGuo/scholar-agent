"""Corpus manifest load / validate / save."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from scholar_agent.models.corpus import CorpusManifestEntry, IngestionStatus
from scholar_agent.storage.jsonl import JsonlRepository, JsonlRepositoryError


class ManifestError(ValueError):
    """Raised when the corpus manifest is missing or invalid."""


class CorpusManifest:
    """In-memory view of ``corpus_manifest.jsonl``."""

    def __init__(self, entries: Sequence[CorpusManifestEntry], *, path: Path | None = None) -> None:
        self.entries = list(entries)
        self.path = path
        self._validate_unique_ids()

    def _validate_unique_ids(self) -> None:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.paper_id in seen:
                raise ManifestError(f"duplicate paper_id in manifest: {entry.paper_id}")
            seen.add(entry.paper_id)

    def __len__(self) -> int:
        return len(self.entries)

    def by_id(self) -> dict[str, CorpusManifestEntry]:
        return {e.paper_id: e for e in self.entries}

    def pending(self) -> list[CorpusManifestEntry]:
        return [e for e in self.entries if e.ingestion_status == IngestionStatus.PENDING]

    def filter_status(self, status: IngestionStatus) -> list[CorpusManifestEntry]:
        return [e for e in self.entries if e.ingestion_status == status]


def load_corpus_manifest(path: Path | str) -> CorpusManifest:
    """Load and validate a corpus manifest JSONL file."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"corpus manifest not found: {manifest_path}")
    repo: JsonlRepository[CorpusManifestEntry] = JsonlRepository(
        manifest_path, CorpusManifestEntry
    )
    try:
        entries = repo.read_all()
    except JsonlRepositoryError as exc:
        raise ManifestError(str(exc)) from exc
    if not entries:
        raise ManifestError(f"corpus manifest is empty: {manifest_path}")
    try:
        return CorpusManifest(entries, path=manifest_path)
    except ManifestError:
        raise
    except ValidationError as exc:
        raise ManifestError(f"invalid manifest entry: {exc}") from exc


def save_corpus_manifest(path: Path | str, entries: Sequence[CorpusManifestEntry]) -> None:
    """Persist manifest entries with uniqueness validation."""
    # Validate uniqueness before write
    CorpusManifest(entries, path=Path(path))
    repo: JsonlRepository[CorpusManifestEntry] = JsonlRepository(
        path, CorpusManifestEntry
    )
    repo.write_all(list(entries))


def validate_corpus_manifest(
    path: Path | str,
    *,
    papers_dir: Path | str | None = None,
) -> list[str]:
    """Validate manifest structure and optional PDF presence.

    Returns a list of human-readable issues. Empty list means valid.
    """
    issues: list[str] = []
    try:
        manifest = load_corpus_manifest(path)
    except ManifestError as exc:
        return [str(exc)]

    pdf_root = Path(papers_dir) if papers_dir is not None else None
    for entry in manifest.entries:
        if pdf_root is not None:
            pdf_path = pdf_root / entry.pdf_filename
            if not pdf_path.is_file():
                issues.append(
                    f"{entry.paper_id}: missing PDF {pdf_path}"
                )
        if not entry.content_hash.strip():
            issues.append(f"{entry.paper_id}: empty content_hash")
        if entry.year is None and not entry.arxiv_id and not entry.doi:
            issues.append(
                f"{entry.paper_id}: missing year and external identifier (doi/arxiv)"
            )
    return issues
