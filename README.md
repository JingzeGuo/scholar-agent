# ScholarAgent

Evidence-driven multi-agent GraphRAG for literature research.

> A Planner decomposes complex questions, a Researcher chooses hybrid or graph
> retrieval tools, a Verifier checks evidence coverage, and a Writer answers
> only from verified evidence—with ablations to measure what actually helps.

**Status:** Phase 0 implementation complete; live provider acceptance requires an API key.
Full design: [`CODEX_IMPLEMENTATION_PLAN.md`](CODEX_IMPLEMENTATION_PLAN.md).

## Phase 0

- Repository scaffold (`uv` + `pyproject.toml` + lockfile)
- Validated YAML/env configuration
- DeepSeek OpenAI-compatible client + live compatibility script
- LangGraph conditional loop with a deterministic fake model
- Architecture notes and design decision records

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
| `make test` / `make lint` / `make typecheck` | Quality gates |
| `make compatibility` | Live provider checks |

Later phases add corpus ingest, retrieval, graph, full agent workflow, evaluation, and Streamlit demo.

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
