"""Phase 9 demo service, status, and saved-run replay tests."""

from __future__ import annotations

from pathlib import Path

from scholar_agent.app.demo_models import DemoSettings
from scholar_agent.app.demo_runs import find_saved_run, list_saved_runs, load_saved_run
from scholar_agent.app.demo_service import DemoService, filter_verified_evidence
from scholar_agent.app.status import collect_system_status
from scholar_agent.models.answer import SourceCard
from scholar_agent.models.evidence import EvidenceItem

REPO = Path(__file__).resolve().parents[2]
DEMO_RUNS = REPO / "data" / "demo" / "runs"


def test_status_collects_without_crash() -> None:
    status = collect_system_status()
    assert status.paths
    assert "processed_dir" in status.paths
    # Counts are non-negative
    assert status.processed_papers >= 0
    assert status.processed_chunks >= 0


def test_saved_demo_runs_exist_and_load() -> None:
    runs = list_saved_runs(DEMO_RUNS)
    assert len(runs) >= 3
    ids = {r.demo_id for r in runs}
    assert "selfrag_vs_crag" in ids
    assert "what_is_selfrag" in ids
    assert "unanswerable_market" in ids

    path = DEMO_RUNS / "selfrag_vs_crag.json"
    saved = load_saved_run(path)
    assert saved.offline is True
    assert saved.session.source_cards
    # Claim → page provenance
    card = saved.session.source_cards[0]
    assert card.page_start >= 1
    assert card.paper_id
    assert card.chunk_id


def test_replay_works_without_indexes() -> None:
    service = DemoService()
    session = service.replay("selfrag_vs_crag")
    assert session.error is None
    assert session.offline_replay is True
    assert "Self-RAG" in session.answer_markdown or "self-rag" in session.answer_markdown.lower()
    assert session.trace.corrective_iterations >= 1
    assert session.trace.corrective_queries
    # Source cards support PDF page trace
    assert any(c.page_start >= 1 for c in session.source_cards)


def test_replay_unanswerable_shows_limitation() -> None:
    session = DemoService().replay("unanswerable_market")
    assert session.error is None
    assert session.trace.unanswerable is True
    assert "Limitation" in session.answer_markdown or session.final_answer is not None


def test_filter_verified_evidence() -> None:
    items = [
        EvidenceItem(
            evidence_id="a",
            sub_question_id="sq",
            claim="c1",
            evidence_text="text a about self-rag reflection",
            paper_id="p1",
            chunk_id="c1",
            page_start=1,
            page_end=1,
            retrieval_method="hybrid",
        ),
        EvidenceItem(
            evidence_id="b",
            sub_question_id="sq",
            claim="c2",
            evidence_text="text b noise",
            paper_id="p2",
            chunk_id="c2",
            page_start=2,
            page_end=2,
            retrieval_method="dense",
        ),
    ]
    cards = [
        SourceCard(
            evidence_id="a",
            paper_id="p1",
            chunk_id="c1",
            page_start=1,
            page_end=1,
            snippet="text a",
        )
    ]
    filtered = filter_verified_evidence(items, final_answer_cards=cards, verified_only=True)
    assert [i.evidence_id for i in filtered] == ["a"]
    all_items = filter_verified_evidence(items, final_answer_cards=cards, verified_only=False)
    assert len(all_items) == 2


def test_find_saved_run() -> None:
    assert find_saved_run("missing_xyz", DEMO_RUNS) is None
    found = find_saved_run("what_is_selfrag", DEMO_RUNS)
    assert found is not None
    assert found.query.startswith("What is Self-RAG")


def test_demo_settings_label() -> None:
    s = DemoSettings(enable_graph=False, enable_corrective=True, static_routing=True)
    label = s.label()
    assert "no-graph" in label
    assert "static" in label
