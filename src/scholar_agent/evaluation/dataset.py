"""Frozen evaluation dataset loading and validation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from scholar_agent.retrieval.chunk_store import ChunkStore

QuestionType = Literal["factual", "keyword", "comparison", "relational", "unanswerable"]

EXPECTED_TYPE_COUNTS: dict[str, int] = {
    "factual": 10,
    "keyword": 10,
    "comparison": 15,
    "relational": 10,
    "unanswerable": 5,
}


class GoldEvidence(BaseModel):
    paper_id: str
    chunk_id: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    snippet: str = ""

    @field_validator("paper_id")
    @classmethod
    def _non_empty_paper(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("paper_id must be non-empty")
        return cleaned


class EvalQuestion(BaseModel):
    question_id: str
    question: str
    question_type: QuestionType
    reference_answer: str = ""
    reference_claims: list[str] = Field(default_factory=list)
    required_paper_ids: list[str] = Field(default_factory=list)
    required_chunk_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    acceptable_alternate_paper_ids: list[str] = Field(default_factory=list)
    graph_reasoning_expected: bool = False
    unanswerable: bool = False
    annotation_notes: str = ""

    @field_validator("question_id", "question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _unanswerable_consistency(self) -> EvalQuestion:
        if self.unanswerable and self.question_type != "unanswerable":
            raise ValueError("unanswerable flag requires question_type=unanswerable")
        if self.question_type == "unanswerable" and not self.unanswerable:
            object.__setattr__(self, "unanswerable", True)
        return self

    def gold_paper_ids(self) -> set[str]:
        papers = set(self.required_paper_ids)
        for g in self.gold_evidence:
            papers.add(g.paper_id)
        papers |= set(self.acceptable_alternate_paper_ids)
        return papers

    def gold_chunk_ids(self) -> set[str]:
        chunks = set(self.required_chunk_ids)
        for g in self.gold_evidence:
            if g.chunk_id:
                chunks.add(g.chunk_id)
        return chunks


class ReferenceEvidenceRow(BaseModel):
    question_id: str
    paper_id: str
    chunk_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    relevance: float = 1.0
    keywords: list[str] = Field(default_factory=list)


class FrozenSplit(BaseModel):
    name: str
    created_at: str = ""
    n_questions: int
    type_counts: dict[str, int] = Field(default_factory=dict)
    question_ids: list[str] = Field(default_factory=list)
    fingerprint_sha256: str
    frozen: bool = True
    notes: str = ""


class EvalDataset(BaseModel):
    questions: list[EvalQuestion]
    reference_evidence: list[ReferenceEvidenceRow] = Field(default_factory=list)
    split: FrozenSplit | None = None

    def by_id(self) -> dict[str, EvalQuestion]:
        return {q.question_id: q for q in self.questions}

    def ordered(self) -> list[EvalQuestion]:
        if self.split and self.split.question_ids:
            index = {qid: i for i, qid in enumerate(self.split.question_ids)}
            return sorted(
                self.questions,
                key=lambda q: index.get(q.question_id, 10_000),
            )
        return list(self.questions)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_frozen_split(path: Path | str) -> FrozenSplit:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenSplit.model_validate(data)


def load_eval_dataset(
    *,
    questions_path: Path | str,
    reference_evidence_path: Path | str | None = None,
    frozen_split_path: Path | str | None = None,
    validate: bool = True,
) -> EvalDataset:
    q_path = Path(questions_path)
    questions = [EvalQuestion.model_validate(row) for row in _read_jsonl(q_path)]
    refs: list[ReferenceEvidenceRow] = []
    r_path: Path | None = None
    if reference_evidence_path is not None:
        r_path = Path(reference_evidence_path)
        if r_path.is_file():
            refs = [ReferenceEvidenceRow.model_validate(row) for row in _read_jsonl(r_path)]
    split: FrozenSplit | None = None
    if frozen_split_path is not None:
        s_path = Path(frozen_split_path)
        if s_path.is_file():
            split = load_frozen_split(s_path)
    dataset = EvalDataset(questions=questions, reference_evidence=refs, split=split)
    if validate:
        validate_frozen_dataset(dataset, questions_path=q_path, reference_path=r_path)
    return dataset


def compute_dataset_fingerprint(questions_path: Path, reference_path: Path | None) -> str:
    material = questions_path.read_bytes()
    if reference_path is not None and reference_path.is_file():
        material = material + b"\n" + reference_path.read_bytes()
    return sha256(material).hexdigest()


def validate_frozen_dataset(
    dataset: EvalDataset,
    *,
    questions_path: Path | None = None,
    reference_path: Path | None = None,
) -> list[str]:
    """Return list of validation issues (empty if OK). Raises on hard freeze mismatch."""
    issues: list[str] = []
    if len(dataset.questions) != 50:
        issues.append(f"expected 50 questions, got {len(dataset.questions)}")
    ids = [q.question_id for q in dataset.questions]
    if len(ids) != len(set(ids)):
        issues.append("duplicate question_id values")
    counts: dict[str, int] = {}
    for q in dataset.questions:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    for key, expected in EXPECTED_TYPE_COUNTS.items():
        if counts.get(key, 0) != expected:
            issues.append(f"type {key}: expected {expected}, got {counts.get(key, 0)}")
    for q in dataset.questions:
        if q.unanswerable:
            continue
        if not q.required_paper_ids and not q.gold_evidence:
            issues.append(f"{q.question_id}: answerable question lacks gold papers")

    if dataset.split is not None:
        if dataset.split.n_questions != len(dataset.questions):
            issues.append("frozen_split.n_questions mismatch")
        split_ids = list(dataset.split.question_ids)
        if set(split_ids) != set(ids):
            issues.append("frozen_split question_ids do not match dataset")
        if questions_path is not None:
            fp = compute_dataset_fingerprint(questions_path, reference_path)
            if dataset.split.fingerprint_sha256 != fp:
                issues.append(
                    "frozen_split fingerprint mismatch — dataset was modified after freeze"
                )
            if not dataset.split.frozen:
                issues.append("frozen_split.frozen is false")
    if issues:
        raise ValueError("evaluation dataset invalid:\n- " + "\n- ".join(issues))
    return issues


def validate_dataset_against_store(
    dataset: EvalDataset,
    store: ChunkStore,
) -> list[str]:
    """Ensure every frozen gold source still maps to canonical paper/chunk/pages."""
    issues: list[str] = []
    questions = dataset.by_id()
    refs_by_question: dict[str, list[ReferenceEvidenceRow]] = {}
    for reference in dataset.reference_evidence:
        if reference.question_id not in questions:
            issues.append(f"orphan reference_evidence question_id={reference.question_id}")
            continue
        refs_by_question.setdefault(reference.question_id, []).append(reference)
        if store.get_paper(reference.paper_id) is None:
            issues.append(
                f"{reference.question_id}: reference uses unknown paper {reference.paper_id}"
            )
        if reference.chunk_id:
            chunk = store.get_chunk(reference.chunk_id)
            if chunk is None:
                issues.append(
                    f"{reference.question_id}: reference uses unknown chunk {reference.chunk_id}"
                )
            elif chunk.paper_id != reference.paper_id:
                issues.append(f"{reference.question_id}: reference chunk/paper mismatch")

    for question in dataset.questions:
        # Some refusal questions include scope evidence showing that a named
        # paper is about a different domain; validate those sources below.
        if (
            question.unanswerable
            and not question.gold_paper_ids()
            and not question.gold_chunk_ids()
        ):
            continue
        if not question.reference_claims or not question.annotation_notes.strip():
            issues.append(f"{question.question_id}: missing reference claims or review notes")
        if question.question_id not in refs_by_question:
            issues.append(f"{question.question_id}: missing reference_evidence rows")
        for paper_id in question.required_paper_ids:
            if store.get_paper(paper_id) is None:
                issues.append(f"{question.question_id}: unknown paper {paper_id}")
        for chunk_id in question.required_chunk_ids:
            if store.get_chunk(chunk_id) is None:
                issues.append(f"{question.question_id}: unknown chunk {chunk_id}")
        for gold in question.gold_evidence:
            paper = store.get_paper(gold.paper_id)
            if paper is None:
                issues.append(f"{question.question_id}: unknown gold paper {gold.paper_id}")
            chunk = store.get_chunk(gold.chunk_id) if gold.chunk_id else None
            if gold.chunk_id and chunk is None:
                issues.append(f"{question.question_id}: unknown gold chunk {gold.chunk_id}")
                continue
            if chunk is not None:
                if chunk.paper_id != gold.paper_id:
                    issues.append(f"{question.question_id}: chunk/paper mismatch {gold.chunk_id}")
                if gold.page_start is not None and gold.page_start < chunk.page_start:
                    issues.append(f"{question.question_id}: gold page starts before chunk")
                if gold.page_end is not None and gold.page_end > chunk.page_end:
                    issues.append(f"{question.question_id}: gold page ends after chunk")
                if (
                    paper is not None
                    and paper.page_count is not None
                    and gold.page_end is not None
                    and gold.page_end > paper.page_count
                ):
                    issues.append(f"{question.question_id}: gold page exceeds paper")

    if issues:
        raise ValueError("evaluation gold provenance invalid:\n- " + "\n- ".join(issues))
    return issues
