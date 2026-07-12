# ScholarAgent

Evidence-driven multi-agent GraphRAG for literature research.

> A Planner decomposes complex questions, a Researcher chooses hybrid or graph
> retrieval tools, a Verifier checks evidence coverage, and a Writer answers
> only from verified evidence—with ablations to measure what actually helps.

**Status:** Phases 0–5 implemented (through adaptive Research Agent).
Full design: [`CODEX_IMPLEMENTATION_PLAN.md`](CODEX_IMPLEMENTATION_PLAN.md).

## Implemented phases

### Phase 0

- Repository scaffold (`uv` + `pyproject.toml` + lockfile)
- Validated YAML/env configuration
- DeepSeek OpenAI-compatible client + live compatibility script
- LangGraph conditional loop with a deterministic fake model
- Architecture notes and design decision records

### Phase 1

- Core Pydantic domain models (paper, chunk, plan, evidence, graph, workflow)
- Deterministic ID helpers
- Typed JSONL repositories + corpus manifest loader/validator
- Test fixtures under `tests/fixtures/`

### Phase 2

- PyMuPDF page-preserving PDF loader
- Header/footer cleanup + section heuristics
- Token-aware chunker (`tiktoken`)
- `scholar-agent ingest` → `data/processed/{papers,chunks}.jsonl` + quality report

### Phase 3

- Dense index (Chroma + BGE or offline hashing embedder)
- Persistent BM25 aligned to stable `chunk_id`s
- Explicit Reciprocal Rank Fusion + optional cross-encoder / lexical rerank
- Typed tools + `retrieve` / `ask-naive` CLI

### Phase 4

- Schema-constrained relation extraction with evidence-span validation
- Staged entity resolver: acronym/alias + string/embedding candidates + optional DeepSeek judge
- NetworkX `MultiDiGraph` node-link JSON persistence
- Query-aware graph path ranking with confidence and evidence-quality scoring
- `graph build|inspect|stats` CLI and `retrieve --mode graph`
- Production BGE/CrossEncoder weights are cached under ignored `.cache/huggingface/`

### Phase 5

- Adaptive query classifier / retrieval router (rule-based, offline)
- Research Agent tool loop with hard tool & evidence budgets
- Evidence ledger merge + structured execution events
- Optional parallel multi-sub-question research

## Quick start

```bash
# Requires Python 3.11+ and uv (https://github.com/astral-sh/uv)
uv sync
cp .env.example .env   # optional: set DEEPSEEK_API_KEY for live checks

# Offline quality gates
make quality           # ruff + mypy + pytest

# Deterministic prototype loop (no API key)
uv run scholar-agent prototype "What is corrective RAG?"
# or
make prototype

# Live DeepSeek compatibility spike (requires API key)
uv run python scripts/deepseek_compatibility.py
# Writes a secret-free report to outputs/deepseek_compatibility.json
```

## Commands (current)

| Command | Description |
|---|---|
| `uv run scholar-agent version` | Package version |
| `uv run scholar-agent config` | Show validated config |
| `uv run scholar-agent prototype "…"` | Run fake-model LangGraph loop |
| `uv run scholar-agent corpus validate -m tests/fixtures/corpus_manifest.jsonl` | Validate manifest |
| `uv run scholar-agent corpus summary -m tests/fixtures/corpus_manifest.jsonl` | Manifest table |
| `uv run scholar-agent ingest --manifest data/corpus_manifest.jsonl` | Ingest PDFs → processed JSONL |
| `uv run scholar-agent ingest --limit 5` | Smoke-ingest first N papers |
| `uv run scholar-agent index build --embedding-backend hash` | Build BM25 + dense indexes (offline) |
| `uv run scholar-agent index build --embedding-backend st` | Build with BGE embeddings |
| `uv run scholar-agent retrieve "What is Self-RAG?" --mode hybrid_rerank` | Search |
| `uv run scholar-agent ask-naive "What is Self-RAG?"` | Naive RAG + page citations |
| `uv run scholar-agent graph build` | Build evidence-linked knowledge graph |
| `uv run scholar-agent graph inspect` | Stats + sample edges |
| `uv run scholar-agent retrieve "Self-RAG HotpotQA" --mode graph` | Graph retrieval |
| `uv run scholar-agent research "Compare Self-RAG and CRAG"` | Research Agent loop |
| `make test` / `make lint` / `make typecheck` | Quality gates |
| `make compatibility` | Live provider checks |

### Corpus (≈120 arXiv PDFs)

PDFs are **not** committed (see `.gitignore`). Download them locally:

```bash
uv run python scripts/download_corpus.py --target 120 --skip-existing
uv run scholar-agent corpus validate -m data/corpus_manifest.jsonl --check-pdfs
uv run scholar-agent corpus summary -m data/corpus_manifest.jsonl
```

- Seed list: `data/seed_arxiv_ids.yaml` (curated by plan categories)
- Manifest: `data/corpus_manifest.jsonl` (metadata + content hashes)
- PDFs: `data/papers/{arxiv_id}.pdf`

Later phases add ingestion, retrieval, graph, full agent workflow, evaluation, and Streamlit demo.

## Project layout

```text
configs/           default + evaluation YAML
docs/              architecture + design decisions
scripts/           deepseek_compatibility.py
src/scholar_agent/ package source
tests/             unit + optional live tests
```

## Design notes

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/design_decisions.md`](docs/design_decisions.md).

## License

MIT (portfolio project).
