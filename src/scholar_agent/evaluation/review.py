"""Independent semantic-review records for the frozen evaluation split."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from scholar_agent.evaluation.dataset import EvalDataset, validate_dataset_against_store
from scholar_agent.retrieval.chunk_store import ChunkStore

ReviewerType = Literal["ai_assisted_independent_manual"]
ReviewStatus = Literal["verified", "failed"]


class ReviewChecks(BaseModel):
    """Semantic and provenance checks performed while reading a question."""

    full_gold_text_read: bool
    question_type_valid: bool
    reference_claims_supported: bool
    paper_ids_valid: bool
    chunk_ids_valid: bool
    page_ranges_valid: bool
    answerability_valid: bool

    def all_passed(self) -> bool:
        return all(self.model_dump().values())


class CorpusSearchProbe(BaseModel):
    """Literal, case-insensitive conjunction searched across every corpus chunk."""

    terms_all: list[str] = Field(min_length=1)
    matched_chunk_ids: list[str] = Field(default_factory=list)
    interpretation: str

    @field_validator("terms_all")
    @classmethod
    def _non_empty_terms(cls, value: list[str]) -> list[str]:
        cleaned = [term.strip() for term in value]
        if any(not term for term in cleaned):
            raise ValueError("search terms must be non-empty")
        return cleaned

    @field_validator("interpretation")
    @classmethod
    def _non_empty_interpretation(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("interpretation must be non-empty")
        return cleaned


class ManualReviewRow(BaseModel):
    question_id: str
    status: ReviewStatus
    reviewer_type: ReviewerType
    reviewed_at: datetime
    checks: ReviewChecks
    reviewed_paper_ids: list[str] = Field(default_factory=list)
    reviewed_chunk_ids: list[str] = Field(default_factory=list)
    support_notes: str
    search_probes: list[CorpusSearchProbe] = Field(default_factory=list)

    @field_validator("question_id", "support_notes")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("review text must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _verified_requires_every_check(self) -> ManualReviewRow:
        if self.status == "verified" and not self.checks.all_passed():
            raise ValueError("verified review rows require every check to pass")
        return self


class ManualReviewManifest(BaseModel):
    schema_version: Literal[1] = 1
    dataset_name: str
    dataset_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{32}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_type: ReviewerType
    reviewed_at: datetime
    method: list[str] = Field(min_length=1)
    n_questions: int = Field(ge=1)
    question_ids: list[str] = Field(min_length=1)
    status_counts: dict[ReviewStatus, int]
    all_verified: bool
    limitations: str


class ManualReviewBundle(BaseModel):
    rows: list[ManualReviewRow]
    manifest: ManualReviewManifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid review JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def review_file_sha256(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_manual_review(
    review_path: Path | str,
    manifest_path: Path | str,
) -> ManualReviewBundle:
    review_file = Path(review_path)
    rows = [ManualReviewRow.model_validate(row) for row in _read_jsonl(review_file)]
    manifest = ManualReviewManifest.model_validate_json(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    actual_hash = review_file_sha256(review_file)
    if manifest.review_sha256 != actual_hash:
        raise ValueError("manual-review manifest review_sha256 mismatch")
    return ManualReviewBundle(rows=rows, manifest=manifest)


def _matching_chunk_ids(store: ChunkStore, terms: list[str]) -> list[str]:
    normalized = [term.casefold() for term in terms]
    return sorted(
        chunk.chunk_id
        for chunk in store.chunks
        if all(term in chunk.text.casefold() for term in normalized)
    )


def validate_manual_review(
    bundle: ManualReviewBundle,
    dataset: EvalDataset,
    *,
    store: ChunkStore | None = None,
) -> list[str]:
    """Validate full review coverage, freeze binding, and corpus provenance."""
    issues: list[str] = []
    rows = bundle.rows
    manifest = bundle.manifest
    dataset_ids = [question.question_id for question in dataset.ordered()]
    review_ids = [row.question_id for row in rows]

    if len(review_ids) != len(set(review_ids)):
        issues.append("duplicate manual-review question_id values")
    if review_ids != dataset_ids:
        issues.append("manual-review question IDs/order do not match frozen split")
    if manifest.question_ids != dataset_ids:
        issues.append("manual-review manifest question IDs/order do not match frozen split")
    if manifest.n_questions != len(dataset_ids) or len(rows) != len(dataset_ids):
        issues.append("manual-review question count mismatch")

    if dataset.split is None:
        issues.append("manual review requires a frozen split")
    else:
        if manifest.dataset_name != dataset.split.name:
            issues.append("manual-review dataset name mismatch")
        if manifest.dataset_fingerprint_sha256 != dataset.split.fingerprint_sha256:
            issues.append("manual-review dataset fingerprint mismatch")

    actual_counts: dict[ReviewStatus, int] = {"verified": 0, "failed": 0}
    questions = dataset.by_id()
    for row in rows:
        actual_counts[row.status] += 1
        question = questions.get(row.question_id)
        if question is None:
            continue
        if row.reviewed_paper_ids != question.required_paper_ids:
            issues.append(f"{row.question_id}: reviewed paper IDs do not match gold")
        if row.reviewed_chunk_ids != question.required_chunk_ids:
            issues.append(f"{row.question_id}: reviewed chunk IDs do not match gold")
        if question.unanswerable and not row.search_probes:
            issues.append(f"{row.question_id}: unanswerable review lacks corpus searches")
        if not question.unanswerable and row.search_probes:
            issues.append(f"{row.question_id}: answerable review has corpus-search probes")

    if manifest.status_counts != actual_counts:
        issues.append("manual-review status_counts mismatch")
    if manifest.all_verified != (actual_counts["verified"] == len(rows)):
        issues.append("manual-review all_verified mismatch")

    if store is not None:
        if manifest.corpus_fingerprint != store.fingerprint:
            issues.append("manual-review corpus fingerprint mismatch")
        try:
            validate_dataset_against_store(dataset, store)
        except ValueError as exc:
            issues.append(str(exc))
        for row in rows:
            for probe in row.search_probes:
                actual_matches = _matching_chunk_ids(store, probe.terms_all)
                if probe.matched_chunk_ids != actual_matches:
                    issues.append(
                        f"{row.question_id}: corpus-search result mismatch for {probe.terms_all!r}"
                    )

    if issues:
        raise ValueError("manual evaluation review invalid:\n- " + "\n- ".join(issues))
    return issues
