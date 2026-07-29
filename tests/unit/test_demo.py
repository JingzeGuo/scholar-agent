"""Phase 9 demo service, status, and saved-run replay tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.app.demo_models import DemoSettings
from scholar_agent.app.demo_runs import find_saved_run, list_saved_runs, load_saved_run
from scholar_agent.app.demo_service import DemoService, filter_verified_evidence
from scholar_agent.app.source_viewer import (
    render_pdf_page_png,
    resolve_pdf_path,
    validate_saved_run_provenance,
)
from scholar_agent.app.status import collect_system_status
from scholar_agent.models.answer import SourceCard
from scholar_agent.models.evidence import EvidenceItem
from scholar_agent.retrieval.chunk_store import ChunkStore

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


@pytest.mark.full_corpus
def test_saved_runs_have_canonical_pdf_page_provenance(
    full_corpus_store: ChunkStore,
) -> None:
    for saved in list_saved_runs(DEMO_RUNS):
        assert saved.provenance_verified is True
        assert saved.corpus_fingerprint == full_corpus_store.fingerprint
        assert validate_saved_run_provenance(saved, full_corpus_store) == []
        for item in saved.session.evidence:
            chunk = full_corpus_store.get_chunk(item.chunk_id)
            assert chunk is not None
            assert item.paper_id == chunk.paper_id
            assert item.page_start == chunk.page_start
            assert item.page_end == chunk.page_end
            assert item.evidence_text == chunk.text


@pytest.mark.full_corpus
def test_source_viewer_renders_real_cited_page() -> None:
    saved = load_saved_run(DEMO_RUNS / "what_is_selfrag.json")
    card = saved.session.source_cards[0]
    pdf_path = card.pdf_path or ""
    resolved = REPO / pdf_path if pdf_path and not Path(pdf_path).is_absolute() else Path(pdf_path)
    if not resolved.is_file():
        pytest.skip(f"local PDF not present for page render: {pdf_path}")
    image = render_pdf_page_png(pdf_path, card.page_start)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 10_000


def test_source_viewer_rejects_path_outside_repo() -> None:
    with pytest.raises(ValueError, match="escapes repository"):
        resolve_pdf_path("../../etc/passwd.pdf")


def test_replay_works_without_indexes() -> None:
    service = DemoService()
    session = service.replay("selfrag_vs_crag")
    assert session.error is None
    assert session.offline_replay is True
    assert "Self-RAG" in session.answer_markdown or "self-rag" in session.answer_markdown.lower()
    assert session.trace.corrective_iterations >= 1
    assert session.trace.corrective_queries
    assert any(step["kind"] == "corrective" for step in session.trace.corrective_steps)
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


def test_streamlit_replay_renders_end_to_end_without_network() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = REPO / "src" / "scholar_agent" / "app" / "streamlit_app.py"
    app = testing.AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception

    app.radio[0].set_value("replay").run(timeout=30)
    assert app.radio[0].value == "replay"
    assert app.selectbox[0].value == "selfrag_vs_crag"
    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Answer", "Trace", "Sources", "Naive RAG"]
    assert any("Self-RAG uses reflection" in element.value for element in app.markdown)
    assert {metric.label for metric in app.metric} >= {
        "Latency (ms)",
        "Tool calls",
        "Evidence",
        "Corrective iters",
    }
