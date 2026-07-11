"""Evidence ledger items and container helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from scholar_agent.ids import make_evidence_id, normalize_text


class EvidenceItem(BaseModel):
    evidence_id: str
    sub_question_id: str
    claim: str
    evidence_text: str
    paper_id: str
    chunk_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    retrieval_method: str
    retrieval_score: float | None = None
    rerank_score: float | None = None
    support_score: float | None = None
    contradiction: bool = False

    @field_validator(
        "evidence_id",
        "sub_question_id",
        "claim",
        "evidence_text",
        "paper_id",
        "chunk_id",
        "retrieval_method",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def _validate_pages(self) -> EvidenceItem:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self

    def dedupe_key(self) -> tuple[str, str]:
        """Key used to merge duplicate evidence (chunk + normalized span)."""
        return (self.chunk_id, normalize_text(self.evidence_text))

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        sub_question_id: str,
        claim: str,
        evidence_text: str,
        paper_id: str,
        chunk_id: str,
        page_start: int,
        page_end: int,
        retrieval_method: str,
        retrieval_score: float | None = None,
        rerank_score: float | None = None,
        support_score: float | None = None,
        contradiction: bool = False,
    ) -> EvidenceItem:
        """Construct an item with a deterministic within-run evidence ID."""
        evidence_id = make_evidence_id(
            run_id=run_id,
            chunk_id=chunk_id,
            evidence_text=evidence_text,
            sub_question_id=sub_question_id,
        )
        return cls(
            evidence_id=evidence_id,
            sub_question_id=sub_question_id,
            claim=claim,
            evidence_text=evidence_text,
            paper_id=paper_id,
            chunk_id=chunk_id,
            page_start=page_start,
            page_end=page_end,
            retrieval_method=retrieval_method,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
            support_score=support_score,
            contradiction=contradiction,
        )


class EvidenceLedger(BaseModel):
    """Ordered evidence collection with chunk/span deduplication."""

    items: list[EvidenceItem] = Field(default_factory=list)

    def merge(self, incoming: list[EvidenceItem] | EvidenceItem) -> EvidenceLedger:
        """Return a new ledger with duplicates merged (prefer higher scores)."""
        additions = [incoming] if isinstance(incoming, EvidenceItem) else list(incoming)
        by_key: dict[tuple[str, str], EvidenceItem] = {
            item.dedupe_key(): item for item in self.items
        }
        for item in additions:
            key = item.dedupe_key()
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = item
                continue
            by_key[key] = _prefer_evidence(existing, item)
        # Preserve first-seen order: existing keys first, then new ones by input order
        ordered: list[EvidenceItem] = []
        seen: set[tuple[str, str]] = set()
        for item in list(self.items) + additions:
            key = item.dedupe_key()
            if key in seen:
                continue
            ordered.append(by_key[key])
            seen.add(key)
        return EvidenceLedger(items=ordered)

    def for_sub_question(self, sub_question_id: str) -> list[EvidenceItem]:
        return [i for i in self.items if i.sub_question_id == sub_question_id]


def _score_tuple(item: EvidenceItem) -> tuple[float, float, float]:
    return (
        item.support_score if item.support_score is not None else float("-inf"),
        item.rerank_score if item.rerank_score is not None else float("-inf"),
        item.retrieval_score if item.retrieval_score is not None else float("-inf"),
    )


def _prefer_evidence(a: EvidenceItem, b: EvidenceItem) -> EvidenceItem:
    """Keep the higher-scored item; prefer non-contradiction on ties."""
    if _score_tuple(b) > _score_tuple(a):
        preferred, other = b, a
    elif _score_tuple(a) > _score_tuple(b):
        preferred, other = a, b
    else:
        preferred, other = (a, b) if not a.contradiction or b.contradiction else (b, a)
    # Surface contradiction if either source flagged it
    if other.contradiction and not preferred.contradiction:
        return preferred.model_copy(update={"contradiction": True})
    return preferred
