"""Independent frozen-dataset review validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.build_eval_dataset import pick_chunk

from scholar_agent.evaluation.dataset import load_eval_dataset
from scholar_agent.evaluation.review import (
    ManualReviewRow,
    load_manual_review,
    validate_manual_review,
)
from scholar_agent.retrieval.chunk_store import ChunkStore

REPO = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO / "data" / "evaluation"


def _dataset():
    return load_eval_dataset(
        questions_path=EVALUATION_DIR / "questions.jsonl",
        reference_evidence_path=EVALUATION_DIR / "reference_evidence.jsonl",
        frozen_split_path=EVALUATION_DIR / "frozen_split.json",
        validate=True,
    )


def _review():
    return load_manual_review(
        EVALUATION_DIR / "manual_review.jsonl",
        EVALUATION_DIR / "manual_review_manifest.json",
    )


@pytest.mark.full_corpus
def test_real_manual_review_covers_frozen_split_and_canonical_store(
    full_corpus_store: ChunkStore,
) -> None:
    review = _review()

    assert validate_manual_review(review, _dataset(), store=full_corpus_store) == []
    assert len(review.rows) == 50
    assert review.manifest.reviewer_type == "ai_assisted_independent_manual"
    assert review.manifest.all_verified is True


def test_review_file_hash_detects_tampering(tmp_path: Path) -> None:
    review_path = tmp_path / "manual_review.jsonl"
    review_path.write_bytes((EVALUATION_DIR / "manual_review.jsonl").read_bytes() + b"\n")
    manifest_path = tmp_path / "manual_review_manifest.json"
    manifest_path.write_bytes((EVALUATION_DIR / "manual_review_manifest.json").read_bytes())

    with pytest.raises(ValueError, match="review_sha256 mismatch"):
        load_manual_review(review_path, manifest_path)


def test_verified_review_cannot_contain_failed_check() -> None:
    payload = json.loads((EVALUATION_DIR / "manual_review.jsonl").read_text().splitlines()[0])
    payload["checks"]["reference_claims_supported"] = False

    with pytest.raises(ValidationError, match="every check"):
        ManualReviewRow.model_validate(payload)


def test_missing_question_review_is_rejected() -> None:
    review = _review()
    incomplete = review.model_copy(update={"rows": review.rows[:-1]})

    with pytest.raises(ValueError, match="question IDs|question count|status_counts"):
        validate_manual_review(incomplete, _dataset())


@pytest.mark.full_corpus
def test_recorded_unanswerable_search_is_recomputed_from_corpus(
    full_corpus_store: ChunkStore,
) -> None:
    review = _review()
    row = review.rows[-1]
    bad_probe = row.search_probes[0].model_copy(update={"matched_chunk_ids": ["chunk_not_real"]})
    bad_row = row.model_copy(update={"search_probes": [bad_probe, *row.search_probes[1:]]})
    bad_review = review.model_copy(update={"rows": [*review.rows[:-1], bad_row]})
    with pytest.raises(ValueError, match="corpus-search result mismatch"):
        validate_manual_review(bad_review, _dataset(), store=full_corpus_store)


def test_gold_picker_prefers_claim_support_over_short_title() -> None:
    title = {
        "chunk_id": "chunk_title",
        "page_start": 1,
        "section": None,
        "text": "MTEB: Massive Text Embedding Benchmark\nAuthors",
    }
    abstract = {
        "chunk_id": "chunk_abstract",
        "page_start": 1,
        "section": "Abstract",
        "text": (
            "We introduce the Massive Text Embedding Benchmark (MTEB). "
            "MTEB spans eight diverse embedding tasks, 58 datasets, and 112 languages. "
            "It evaluates text embedding models across clustering, retrieval, reranking, "
            "classification, and semantic similarity tasks."
        ),
    }

    selected = pick_chunk(
        [title, abstract],
        ["mteb", "embedding", "benchmark"],
        "MTEB is a massive text embedding benchmark covering diverse embedding tasks.",
    )

    assert selected is abstract
