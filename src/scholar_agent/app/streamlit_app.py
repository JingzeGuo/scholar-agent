"""ScholarAgent Streamlit demo (Phase 9).

Launch:
  uv sync --extra ui
  uv run streamlit run src/scholar_agent/app/streamlit_app.py

Or:
  make demo
"""

from __future__ import annotations

from typing import Literal

from scholar_agent.app.demo_models import DemoSessionResult, DemoSettings, SavedDemoRun
from scholar_agent.app.demo_service import DemoService
from scholar_agent.app.source_viewer import render_pdf_page_png, resolve_pdf_path
from scholar_agent.app.status import SystemStatus
from scholar_agent.config import load_config
from scholar_agent.logging import setup_logging

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional UI extra
    raise SystemExit("Streamlit is not installed. Install with: uv sync --extra ui") from exc


def _init_state() -> None:
    if "service" not in st.session_state:
        cfg = load_config()
        setup_logging(cfg)
        st.session_state.service = DemoService(config=cfg)
    if "history" not in st.session_state:
        st.session_state.history = []  # list[DemoSessionResult]
    if "last_session" not in st.session_state:
        st.session_state.last_session = None


def _sidebar_settings(status: SystemStatus) -> tuple[DemoSettings, str, str | None]:
    st.sidebar.header("Corpus & indexes")
    st.sidebar.metric("Papers (processed)", status.processed_papers)
    st.sidebar.metric("Chunks", status.processed_chunks)
    st.sidebar.metric("PDFs on disk", status.n_pdfs)
    cols = st.sidebar.columns(3)
    cols[0].write("Dense ✅" if status.dense_index_ready else "Dense ❌")
    cols[1].write("BM25 ✅" if status.sparse_index_ready else "BM25 ❌")
    cols[2].write("Graph ✅" if status.graph_ready else "Graph ❌")
    if status.graph_nodes is not None:
        st.sidebar.caption(f"Graph nodes={status.graph_nodes} edges={status.graph_edges}")
    for msg in status.messages:
        st.sidebar.warning(msg)

    st.sidebar.header("Mode")
    mode = st.sidebar.radio(
        "Execution",
        options=["live", "replay"],
        format_func=lambda x: "Live run" if x == "live" else "Replay saved run",
        help="Replay works without live API / even if indexes are incomplete.",
    )

    demo_id: str | None = None
    service: DemoService = st.session_state.service
    if mode == "replay":
        runs = service.list_replays()
        if not runs:
            st.sidebar.error("No saved demo runs in data/demo/runs/")
        else:
            labels = {r.demo_id: f"{r.title} — {r.query[:48]}" for r in runs}
            demo_id = st.sidebar.selectbox(
                "Saved run",
                options=list(labels.keys()),
                format_func=lambda k: labels[k],
            )

    st.sidebar.header("Ablation toggles")
    compare_naive = st.sidebar.checkbox("Compare with Naive RAG", value=True)
    enable_graph = st.sidebar.checkbox("Enable graph retrieval", value=status.graph_ready)
    enable_corrective = st.sidebar.checkbox("Enable corrective loop", value=True)
    static_routing = st.sidebar.checkbox("Static all-tools (disable adaptive routing)", value=False)
    verified_only = st.sidebar.checkbox("Show only verified evidence", value=True)
    embedding_backend_raw = st.sidebar.selectbox(
        "Embedding backend",
        options=["hash", "auto", "st"],
        index=0,
        help="hash = offline; st/auto may load BGE weights",
    )
    embedding_backend: Literal["hash", "auto", "st"] = (
        embedding_backend_raw if embedding_backend_raw in {"hash", "auto", "st"} else "hash"
    )
    max_corr = st.sidebar.slider("Max corrective iterations", 0, 3, 2)
    top_k = st.sidebar.slider("Top-k", 3, 16, 8)

    settings = DemoSettings(
        compare_naive_rag=compare_naive,
        enable_graph=enable_graph,
        enable_corrective=enable_corrective and not static_routing,
        static_routing=static_routing,
        verified_evidence_only=verified_only,
        embedding_backend=embedding_backend,
        max_corrective_iterations=max_corr if enable_corrective else 0,
        top_k=top_k,
        use_llm=False,
    )

    st.sidebar.header("Session")
    if st.sidebar.button("Reset session"):
        st.session_state.history = []
        st.session_state.last_session = None
        st.rerun()

    return settings, mode, demo_id


def _render_sources(session: DemoSessionResult) -> None:
    st.subheader("Sources")
    if not session.source_cards and not session.evidence:
        st.info("No sources for this answer.")
        return
    cards = session.source_cards
    if not cards and session.evidence:
        # Fall back to evidence ledger items
        for item in session.evidence[:12]:
            with st.expander(
                f"{item.paper_id} p.{item.page_start}-{item.page_end} · {item.retrieval_method}"
            ):
                st.markdown(f"**Claim:** {item.claim}")
                st.write(item.evidence_text)
                st.caption(f"chunk=`{item.chunk_id}` · evidence=`{item.evidence_id}`")
        return

    claims_by_evidence: dict[str, list[str]] = {}
    for claim in session.claims:
        for evidence_id in claim.get("evidence_ids") or []:
            claims_by_evidence.setdefault(str(evidence_id), []).append(
                str(claim.get("text") or claim.get("claim_id") or "")
            )

    for index, card in enumerate(cards):
        title = card.title or card.paper_id
        with st.expander(f"{title} · {card.page_label()} · {card.paper_id}"):
            supported_claims = claims_by_evidence.get(card.evidence_id) or []
            for claim_text in supported_claims:
                st.markdown(f"**Supports final claim:** {claim_text}")
            st.markdown(f"**Snippet:** {card.snippet or '—'}")
            st.caption(
                f"chunk=`{card.chunk_id}` · evidence=`{card.evidence_id}`"
                + (f" · method={card.retrieval_method}" if card.retrieval_method else "")
            )
            if card.pdf_path:
                st.caption(f"PDF: `{card.pdf_path}`")
                preview = st.checkbox(
                    f"Preview cited PDF page {card.page_start}",
                    key=f"pdf-preview-{session.run_id}-{index}-{card.evidence_id}",
                )
                if preview:
                    try:
                        path = resolve_pdf_path(card.pdf_path)
                        image = render_pdf_page_png(path, card.page_start)
                    except (FileNotFoundError, ValueError, RuntimeError) as exc:
                        st.warning(f"PDF page preview unavailable: {exc}")
                    else:
                        st.image(
                            image,
                            caption=(
                                f"{title} — physical PDF page {card.page_start} "
                                f"(chunk {card.chunk_id})"
                            ),
                            width="stretch",
                        )
            st.success(
                f"Claim → evidence `{card.evidence_id}` → chunk `{card.chunk_id}` "
                f"→ {card.paper_id} {card.page_label()}."
            )


def _render_trace(session: DemoSessionResult) -> None:
    st.subheader("Research trace")
    tr = session.trace
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency (ms)", tr.latency_ms)
    c2.metric("Tool calls", tr.tool_call_count)
    c3.metric("Evidence", tr.evidence_count)
    c4.metric("Corrective iters", tr.corrective_iterations)

    st.markdown(
        f"**Query type:** `{tr.query_type or '—'}` · "
        f"**Answer type:** `{tr.answer_type or '—'}` · "
        f"**Terminated:** `{tr.terminated_reason or '—'}` · "
        f"**Unanswerable:** `{tr.unanswerable}`"
    )
    if tr.coverage_score is not None:
        st.progress(min(1.0, max(0.0, tr.coverage_score)), text=f"Coverage {tr.coverage_score:.2f}")

    if tr.sub_questions:
        st.markdown("#### Sub-questions")
        for sq in tr.sub_questions:
            st.markdown(f"- `{sq.get('status', '?')}` **{sq.get('id')}**: {sq.get('question')}")

    if tr.corrective_queries:
        st.markdown("#### Corrective queries")
        for q in tr.corrective_queries:
            st.code(q, language=None)

    if tr.corrective_steps:
        st.markdown("#### Corrective loop timeline")
        icons = {
            "verification": "🔎",
            "corrective": "🔁",
            "tool_result": "🧰",
            "run_finished": "✅",
            "budget_hit": "⛔",
        }
        for step in tr.corrective_steps:
            kind = str(step.get("kind") or "event")
            detail = []
            if step.get("is_sufficient") is not None:
                detail.append(f"sufficient={step['is_sufficient']}")
            if step.get("method"):
                detail.append(f"method={step['method']}")
            suffix = f" · {' · '.join(detail)}" if detail else ""
            st.markdown(
                f"{icons.get(kind, '•')} **{step.get('number')}. {kind}** — "
                f"{step.get('summary')}{suffix}"
            )

    if tr.conflicting_evidence_ids:
        st.warning("Conflicting evidence retained: " + ", ".join(tr.conflicting_evidence_ids[:12]))

    if tr.retrieval_methods:
        st.caption("Retrieval methods: " + ", ".join(tr.retrieval_methods))

    st.markdown(
        f"**Citation valid:** `{tr.citation_valid}` · issues={tr.citation_issue_count} · "
        f"tokens≈{tr.token_estimate}"
    )

    with st.expander("Tool / event log", expanded=False):
        if session.events:
            for evt in session.events:
                st.markdown(f"- `{evt.event_type.value}` · **{evt.component}**: {evt.summary}")
        elif tr.tool_events:
            for tool_evt in tr.tool_events:
                st.markdown(
                    f"- `{tool_evt.get('event_type')}` · **{tool_evt.get('component')}**: "
                    f"{tool_evt.get('summary')}"
                )
        else:
            st.caption("No tool events recorded for this session.")


def _render_naive(session: DemoSessionResult) -> None:
    if session.naive is None:
        return
    st.subheader("Naive RAG comparison")
    st.caption(
        f"method={session.naive.method} · hits={session.naive.hit_count} · "
        f"latency_ms={session.naive.latency_ms} · llm={session.naive.used_llm}"
    )
    st.markdown(session.naive.answer)


def _render_session(session: DemoSessionResult) -> None:
    if session.error:
        st.error(session.error)
    badge = "🔁 Replay" if session.offline_replay else "🔴 Live"
    st.caption(f"{badge} · run_id=`{session.run_id}` · settings: {session.settings.label()}")

    tab_answer, tab_trace, tab_sources, tab_naive = st.tabs(
        ["Answer", "Trace", "Sources", "Naive RAG"]
    )
    with tab_answer:
        st.subheader("Final answer")
        if session.naive is not None:
            full_col, naive_col = st.columns(2)
            with full_col:
                st.markdown("#### ScholarAgent")
                st.markdown(session.answer_markdown or "_No answer text._")
            with naive_col:
                st.markdown("#### Naive RAG baseline")
                st.caption(
                    f"hits={session.naive.hit_count} · latency={session.naive.latency_ms} ms"
                )
                st.markdown(session.naive.answer or "_No baseline answer._")
        elif session.answer_markdown:
            st.markdown(session.answer_markdown)
        else:
            st.info("No answer text.")
        if session.claims:
            st.markdown("#### Structured claims")
            for claim in session.claims:
                eids = ", ".join(claim.get("evidence_ids") or [])
                st.markdown(f"- **{claim.get('claim_id')}**: {claim.get('text')}  ")
                st.caption(f"evidence: {eids or '—'}")
    with tab_trace:
        _render_trace(session)
    with tab_sources:
        _render_sources(session)
    with tab_naive:
        if session.naive is None:
            st.info("Enable **Compare with Naive RAG** in the sidebar for a side-by-side baseline.")
        else:
            _render_naive(session)


def main() -> None:
    st.set_page_config(
        page_title="ScholarAgent Demo",
        page_icon="📚",
        layout="wide",
    )
    _init_state()
    service: DemoService = st.session_state.service
    status = service.get_status()

    st.title("ScholarAgent")
    st.markdown(
        "Evidence-driven multi-agent GraphRAG demo — plan → research → verify → "
        "write → cite. Use **Replay** for interview-safe offline demos."
    )

    settings, mode, demo_id = _sidebar_settings(status)

    # Chat-style input
    with st.form("query_form", clear_on_submit=False):
        default_q = "Compare Self-RAG versus CRAG"
        if mode == "replay" and demo_id:
            replay_index = {r.demo_id: r for r in service.list_replays()}
            if demo_id in replay_index:
                default_q = replay_index[demo_id].query
        query = st.text_area("Research question", value=default_q, height=80)
        submitted = st.form_submit_button(
            "Run" if mode == "live" else "Load replay",
            type="primary",
        )

    if submitted and query.strip():
        with st.spinner("Running ScholarAgent…" if mode == "live" else "Loading replay…"):
            if mode == "replay":
                if not demo_id:
                    st.error("Select a saved demo run.")
                    session = None
                else:
                    session = service.replay(demo_id)
            else:
                session = service.run_live(query.strip(), settings)
        if session is not None:
            st.session_state.last_session = session
            st.session_state.history.append(session)

    session = st.session_state.last_session
    if session is None:
        st.info(
            "Enter a question and click **Run**, or switch to **Replay saved run** "
            "for offline interview mode."
        )
        # Show available replays as cards
        saved_runs: list[SavedDemoRun] = service.list_replays()
        if saved_runs:
            st.markdown("### Available saved demos")
            for saved in saved_runs:
                st.markdown(f"- **{saved.title}** (`{saved.demo_id}`): {saved.query}")
        return

    _render_session(session)

    if st.session_state.history:
        with st.expander("Session history"):
            for i, past in enumerate(reversed(st.session_state.history[-8:]), start=1):
                st.markdown(
                    f"{i}. `{past.run_id[:12]}` · {past.query[:80]} · "
                    f"{'replay' if past.offline_replay else 'live'}"
                )


if __name__ == "__main__":
    main()
