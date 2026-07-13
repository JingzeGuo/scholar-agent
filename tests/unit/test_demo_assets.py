"""Truthfulness and reproducibility checks for committed demo assets."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from scripts import precompute_demo_runs
from scripts.build_demo_gif import build_demo_gif, extract_story

from scholar_agent.app.demo_runs import load_saved_run
from scholar_agent.ids import content_hash
from scholar_agent.models.base import EventType
from scholar_agent.models.corpus import Chunk
from scholar_agent.retrieval.chunk_store import ChunkStore

REPO = Path(__file__).resolve().parents[2]
REPLAY = REPO / "data" / "demo" / "runs" / "selfrag_vs_crag.json"
GIF = REPO / "docs" / "assets" / "scholaragent_replay.gif"


def _chunk(chunk_id: str, paper_id: str, text: str, *, tokens: int = 20) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        page_start=1,
        page_end=1,
        token_count=tokens,
        content_hash=content_hash(text),
    )


def test_support_chunk_selection_is_paper_scoped_and_claim_aware() -> None:
    store = ChunkStore(
        [
            _chunk(
                "wrong-paper",
                "paper_other",
                "Alpha method uses signal tokens to retrieve evidence on demand with critique.",
            ),
            _chunk(
                "weak",
                "paper_target",
                "Alpha method mentions signal tokens and on demand operation.",
                tokens=12,
            ),
            _chunk(
                "supported",
                "paper_target",
                (
                    "Alpha method uses signal tokens to retrieve evidence on demand "
                    "and critique its output."
                ),
                tokens=30,
            ),
        ]
    )
    selected = precompute_demo_runs._select_supporting_chunk(
        store,
        paper_id="paper_target",
        keywords=("Alpha method", "signal tokens", "on-demand"),
        claim="Alpha method uses signal tokens to retrieve evidence and critique output.",
    )
    assert selected.chunk_id == "supported"


def test_fixture_fallback_is_explicitly_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(precompute_demo_runs, "_load_canonical_store", lambda: None)
    run = precompute_demo_runs._fixture_compare()
    assert run.provenance_verified is False
    assert run.corpus_fingerprint is None
    assert run.session.status["fixture_provenance"] == "unverified_fallback"
    assert all(item.chunk_id.startswith("chunk_") for item in run.session.evidence)


def test_replay_story_is_projected_from_structured_run() -> None:
    run = load_saved_run(REPLAY)
    story = extract_story(run)
    assert story.query == run.query
    assert story.sub_questions == tuple(
        item.question
        for item in run.session.plan.sub_questions  # type: ignore[union-attr]
    )
    assert story.evidence_lines == tuple(item.claim for item in run.session.evidence)
    assert any(event.event_type == EventType.CORRECTIVE for event in run.session.events)
    assert story.corrective_queries == tuple(run.session.trace.corrective_queries)


def test_replay_story_rejects_missing_corrective_event() -> None:
    run = load_saved_run(REPLAY)
    session = run.session.model_copy(
        update={
            "events": [
                event for event in run.session.events if event.event_type != EventType.CORRECTIVE
            ]
        }
    )
    invalid = run.model_copy(update={"session": session})
    with pytest.raises(ValueError, match="corrective"):
        extract_story(invalid)


def test_committed_gif_and_rebuild_have_expected_frames(tmp_path: Path) -> None:
    rebuilt = tmp_path / "replay.gif"
    assert build_demo_gif(REPLAY, rebuilt, width=800, height=450) == 6

    for path, size in ((GIF, (960, 540)), (rebuilt, (800, 450))):
        assert path.read_bytes().startswith(b"GIF89a")
        with Image.open(path) as image:
            assert image.format == "GIF"
            assert image.size == size
            assert image.n_frames == 6
            assert image.info["loop"] == 0
