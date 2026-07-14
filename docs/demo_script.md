# Demo script (interview + offline backup)

This document is a reproducible walkthrough. The committed
[`scholaragent_replay.gif`](assets/scholaragent_replay.gif) is generated from the
provenance-checked offline replay; it is explicitly labelled as replay-derived,
not a live browser recording.

Related: [`docs/demo.md`](demo.md) (UI modes), committed runs under
`data/demo/runs/`.

Rebuild the GIF deterministically:

```bash
uv run python scripts/build_demo_gif.py
```

---

## A. Stable 5–7 minute interview demo

**Preferred mode:** offline Replay (no API key). Live mode is optional when
indexes and credentials exist.

### Prep (2 minutes before)

```bash
uv sync --extra ui
uv run scholar-agent demo --replay selfrag_vs_crag   # sanity print
make demo   # or: uv run streamlit run src/scholar_agent/app/streamlit_app.py
```

In the UI, open **Replay** and select `selfrag_vs_crag`.

### Minute-by-minute

| Time | What to show | Talking point |
|---:|---|---|
| 0:00–0:30 | Project problem | Literature answers need **page-level evidence**, not chatty summaries. |
| 0:30–1:00 | Naive RAG baseline failure | Dense-only often misses exact names/acronyms; hybrid + agent tools recover (measured paper R@8 0.16 → 0.61 on the offline hash eval). |
| 1:00–1:30 | Complex query | Replay query: **“Compare Self-RAG versus CRAG”**. |
| 1:30–2:00 | Planner decomposition | Trace: two comparison sub-questions (Self-RAG side / CRAG side). |
| 2:00–2:40 | Adaptive tool selection | Trace: hybrid_rerank (and graph if enabled) chosen by the rule-based router, not a fixed pipeline. |
| 2:40–3:20 | Evidence ledger | Ledger items with paper_id, chunk_id, pages, scores. |
| 3:20–3:50 | Graph path (if live/graph built) | Graph inspect: edge → evidence span → chunk → PDF page. In pure replay, point to source cards with chunk IDs. |
| 3:50–4:30 | Verifier gap + corrective | Replay notes: first pass may miss one comparison side; corrective iteration pulls the missing paper. |
| 4:30–5:20 | Final page-level citations | Answer claims with `[paper … p.N]`; Sources tab maps claim → chunk → PDF page. |
| 5:20–6:00 | Ablation comparison | Mention offline results: hybrid_graph best paper recall (0.67); full_agent best cite precision (0.288); unanswerable refusals **5/5**. |
| 6:00–6:30 | One limitation | Offline hash run: corrective improvement metric **0.0**; graph adds latency (~11→290 ms). |

### Stable sample questions

1. `Compare Self-RAG versus CRAG` — replay `selfrag_vs_crag`
2. `What is Self-RAG?` — replay `what_is_selfrag`
3. Unanswerable market / out-of-corpus — replay `unanswerable_market`

### Expected UI states

- **Answer:** Markdown claims with page citations.
- **Trace:** plan → tool results → verification → corrective (when present) → finish.
- **Sources:** source cards with paper title, chunk id, pages, PDF path / page preview when PDF is local.
- **Status:** corpus/index health indicator (live mode).

---

## B. Backup offline demo (no provider, no Streamlit)

```bash
uv sync
uv run scholar-agent demo --replay selfrag_vs_crag
uv run scholar-agent demo --replay what_is_selfrag
uv run scholar-agent demo --replay unanswerable_market
uv run scholar-agent prototype "What is corrective RAG?"
uv run pytest tests/unit/test_e2e_fixture.py tests/unit/test_demo.py -q
```

Optional fixture corpus checks:

```bash
uv run scholar-agent corpus validate -m tests/fixtures/corpus_manifest.jsonl
uv run scholar-agent graph inspect   # needs local processed graph
```

---

## C. Recording checklist

### Environment

- [ ] `uv sync` (add `--extra ui` for Streamlit)
- [ ] Working tree clean of secrets (no `.env` on screen)
- [ ] Replay fixtures present under `data/demo/runs/`
- [ ] Optional: local PDFs for page preview (`data/papers/`)

### Assets

- [ ] Replay-derived GIF renders correctly in README
- [ ] Architecture Mermaid from README
- [ ] Aggregate metrics table from `docs/results/offline_hash_eval_summary.md`
- [ ] One failure story from `docs/failure_analysis.md`

### Sensitive-information check

- [ ] No API keys in terminal history on camera
- [ ] No private PDFs outside the curated arXiv corpus if screen-sharing files
- [ ] No provider reasoning / chain-of-thought panels (none are shown by design)

### Fallback if API fails

1. Switch to **Replay** mode immediately.
2. Or run `uv run scholar-agent demo --replay selfrag_vs_crag` in the terminal.
3. Walk through unit tests: `uv run pytest tests/unit/test_workflow.py -q`.

### Commands to prepare live mode (optional)

```bash
uv run python scripts/download_corpus.py --target 120 --skip-existing
uv run scholar-agent ingest --manifest data/corpus_manifest.jsonl
uv run scholar-agent index build --embedding-backend hash
uv run scholar-agent graph build
make demo
```
