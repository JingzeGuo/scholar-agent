# Streamlit demo (Phase 9–10)

Interview timing, offline backup, and recording checklist:
[`docs/demo_script.md`](demo_script.md).

## Install UI extra

```bash
uv sync --extra ui
```

## Launch

```bash
make demo
# or
uv run streamlit run src/scholar_agent/app/streamlit_app.py
# or
uv run scholar-agent demo --port 8501
```

## Modes

### Live

Requires processed corpus + indexes (`index build`). Runs:

- Full ScholarAgent workflow (plan → research → verify → write → cite), or
- **Static all-tools** ablation when the toggle is enabled.

Sidebar toggles:

| Toggle | Effect |
|---|---|
| Compare with Naive RAG | Side-by-side hybrid-rerank extractive baseline |
| Enable graph retrieval | Load / use knowledge graph |
| Enable corrective loop | Allow Verifier-driven re-retrieval |
| Static all-tools | Disable adaptive agent loop; merge dense+sparse+hybrid(+graph) |
| Show only verified evidence | Filter ledger to citation-validated IDs |
| Embedding backend | `hash` (offline) / `auto` / `st` |

### Replay (offline / interview-safe)

Load JSON sessions from `data/demo/runs/`:

| ID | Story |
|---|---|
| `selfrag_vs_crag` | Comparison + corrective iteration + dual source cards |
| `what_is_selfrag` | Single-paper factual + page provenance |
| `unanswerable_market` | Corpus insufficiency / refusal |

```bash
uv run scholar-agent demo --replay selfrag_vs_crag
```

Committed replay sources are checked against the canonical chunk store when
fixtures are built and in tests. Each saved run records the corpus fingerprint
and `provenance_verified`; invented demo-only chunk IDs are rejected.

Regenerate fixtures (or live captures when indexes exist):

```bash
make demo-precompute
# live preferred:
uv run python scripts/precompute_demo_runs.py --live
```

## Acceptance mapping

- **Claim → PDF page:** Sources tab maps each final claim to its evidence ID,
  canonical chunk, paper and page, and can render the physical cited PDF page.
- **Corrective loop visible:** Trace tab presents an ordered
  verify → corrective retrieval → finish timeline, plus queries and event log.
- **No live API required:** Replay mode uses only committed JSON under `data/demo/runs/`.
