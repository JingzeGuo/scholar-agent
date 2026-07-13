#!/usr/bin/env python3
"""Validate the independent review of the frozen evaluation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scholar_agent.evaluation.dataset import load_eval_dataset
from scholar_agent.evaluation.review import load_manual_review, validate_manual_review
from scholar_agent.retrieval.chunk_store import ChunkStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_dir: Path = args.evaluation_dir
    dataset = load_eval_dataset(
        questions_path=evaluation_dir / "questions.jsonl",
        reference_evidence_path=evaluation_dir / "reference_evidence.jsonl",
        frozen_split_path=evaluation_dir / "frozen_split.json",
        validate=True,
    )
    review = load_manual_review(
        evaluation_dir / "manual_review.jsonl",
        evaluation_dir / "manual_review_manifest.json",
    )
    store = ChunkStore.from_processed_dir(args.processed_dir)
    validate_manual_review(review, dataset, store=store)
    print(
        json.dumps(
            {
                "status": "ok",
                "reviewed_questions": len(review.rows),
                "dataset_fingerprint_sha256": review.manifest.dataset_fingerprint_sha256,
                "corpus_fingerprint": review.manifest.corpus_fingerprint,
                "reviewer_type": review.manifest.reviewer_type,
                "all_verified": review.manifest.all_verified,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
