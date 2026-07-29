# Implementation-plan acceptance audit

Audit date: **2026-07-29**. Scope: core Phases 0–10 and the Definition of Done
in `CODEX_IMPLEMENTATION_PLAN.md`. Section 21 is explicitly optional and is not
part of core completion.

## Phase evidence

| Phase | Status | Objective acceptance evidence |
|---|---|---|
| 0 — compatibility spike | Complete | Live `outputs/deepseek_compatibility.json`: `deepseek-v4-flash`, 8/8 checks pass (chat, streaming, structured JSON, tool calling, multi-turn tool use, reasoning/thinking fields, retry behavior). Offline structured parsing, retry, and report behavior are covered by `test_structured.py`, `test_retry.py`, and `test_compatibility_script.py`; `uv.lock` is present and checked. |
| 1 — models/storage | Complete | Pydantic round trips and invalid-schema rejection in `test_domain_models.py`; stable IDs in `test_ids.py`; JSONL/manifest validation in `test_storage.py`. |
| 2 — ingestion | Complete | Page-preserving, token-aware, empty/scanned, and idempotency tests in `test_ingestion.py`. Full-corpus provenance audit passes: **120/120 PDFs**, **5858/5858 chunks**, zero missing papers/chunks. |
| 3 — retrieval | Complete | Dense/BM25 stable-ID, RRF, exact deterministic hash search, debug, filtering, and page-reference tests in `test_retrieval.py`. Production BGE + cross-encoder was also exercised on all 50×7 rows (`run_a4770534afb84db2`). |
| 4 — graph | Complete | Span grounding, alias resolution, serialization, path support, ranking, and isolated-rate tests in `test_graph.py`. Full graph audit passes: **4263/4263** relations map to PDF evidence (4181 single-page, 82 cross-page), zero missing ranges. |
| 5 — routing/research | Complete | Different labeled routes, ledger dedupe, parallel deterministic merge, structured traces, and hard tool/iteration/token/latency/evidence budgets in `test_router.py`, `test_researcher.py`, and `test_state_reducers.py`. |
| 6 — corrective workflow | Complete | Targeted retrieval, no-new-evidence stop, conflict surfacing, all budget stops, and unanswerable exhaustion in `test_verifier.py` and `test_workflow.py`. Measured limitation: corrective recall improvement is **0.0**, despite correct trigger/termination behavior. |
| 7 — writer/citations | Complete | Evidence-only writing, accepted-evidence filtering, contradiction surfacing, invalid/nonexistent citation rejection, physical-PDF page validation, and unsupported-claim removal in `test_writer.py` and `test_citation_validator.py`. |
| 8 — evaluation | Complete | Frozen fingerprinted **50-question** split; all **7 systems × 50 = 350** rows in clean exact-hash run `run_1f4dc371453d4a1f` with zero system errors. Separate full DeepSeek+RAGAS and BGE+cross-encoder 50×7 runs are committed as numeric summaries under `docs/results/`. Six concrete failures are documented in `failure_analysis.md`. |
| 9 — demo | Complete | Streamlit AppTest covers provider-free end-to-end replay; tests cover canonical source→PDF-page rendering, corrective trace, ablation settings, status, and replay without indexes. The committed GIF is deterministically rebuilt and frame-checked. |
| 10 — hardening/docs | Complete | Startup validation, retries/jitter, caches, degradation, untrusted-text delimiters, secret-safe logs, and offline-by-default test selection are covered by hardening tests. README, ADRs, evaluation, failure analysis, demo script, and interview guide are present. |

## Final gates

- `make quality`: Ruff and Mypy pass.
- Fresh clone/default dependencies: **252 passed, 6 optional tests skipped,
  2 live tests deselected**.
- Local full corpus + UI extra: **258 passed, 2 live tests deselected**.
- `UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv lock --check`: passes.
- Default tests make no paid provider calls; provider tests use the `live` marker.
- Tests that need gitignored corpus artifacts use the `full_corpus` marker and
  skip when `data/processed/chunks.jsonl` is unavailable.
- Tracked-artifact audit contains no `.env`, API key, generated index, model
  cache, PDF corpus, or raw evaluation output. Only `.gitkeep` files retain the
  local artifact directories.
- The 50 evaluation items have a validated AI-assisted independent review
  manifest. This satisfies the repository's reproducible review check, but it
  is explicitly **not a human-signed annotation audit**.

## Reproducible full-corpus audits

```bash
uv run python scripts/audit_page_provenance.py
uv run python scripts/audit_graph_provenance.py
uv run scholar-agent evaluate \
  --eval-config configs/evaluation.yaml \
  --embedding-backend hash \
  --output-dir outputs/evaluation/reproduce
```

The full corpus, indexes, model cache, and evaluation outputs remain local and
gitignored by design; committed summaries record their fingerprints and clean
code provenance.
