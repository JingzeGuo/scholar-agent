# Architecture (implemented through Phase 9)

ScholarAgent is an evidence-driven multi-agent literature research system.
This document tracks the architecture as implemented; see
`CODEX_IMPLEMENTATION_PLAN.md` for the full target design.

## High-level paths

### Offline ingestion (Phases 2–4)

```text
PDF corpus
  → metadata + page parsing
  → token-aware section chunking
  → canonical chunk store (source of truth)
  → dense index / BM25 / evidence-linked graph
```

### Online research (Phases 5–7)

```text
User query
  → Planner (structured sub-questions)
  → Research Agent tool loop
  → Evidence Ledger
  → Verifier (corrective retrieval if needed)
  → Writer + citation validator
  → answer + sources + execution trace
```

## Phase 0–1 components

| Component | Location | Role |
|---|---|---|
| Config | `scholar_agent.config` | YAML + env, Pydantic validation |
| Logging | `scholar_agent.logging` | Secret-safe structured logs |
| Stable IDs | `scholar_agent.ids` | Content-addressed paper/chunk/entity/evidence IDs |
| Core models | `scholar_agent.models` | Corpus, plan, evidence, graph, workflow types |
| Storage | `scholar_agent.storage` | Typed JSONL + corpus manifest |
| LLM client | `scholar_agent.llm` | DeepSeek OpenAI-compatible wrapper |
| Prototype loop | `scholar_agent.agents.prototype_loop` | LangGraph decide→retrieve→verify loop |
| Compatibility script | `scripts/deepseek_compatibility.py` | Live provider spike |
| Ingestion | `scholar_agent.ingestion` | PDF → pages → sections → chunks |
| Retrieval | `scholar_agent.retrieval` | Dense + BM25 + RRF + rerank + Naive RAG |
| Knowledge graph | `scholar_agent.graph` | Extract, resolve, MultiDiGraph, graph retrieve |
| Router | `scholar_agent.retrieval.router` | Query type → retrieval policy |
| Research Agent | `scholar_agent.agents.researcher` | Adaptive tool loop + bounded safe fan-out + evidence ledger |
| Planner / Verifier / Workflow | `agents/planner.py`, `verifier.py`, `workflow.py` | Target-bound corrective loop + exhaustive termination |
| Writer / Citation validator | `agents/writer.py`, `citation_validator.py` | Verified-evidence claims + canonical PDF/page validation |
| Evaluation | `scholar_agent.evaluation` | Frozen 50-Q split, baselines/ablations, metrics, reports |
| Demo UI | `scholar_agent.app` | Streamlit chat, trace, sources, ablation toggles, saved-run replay |
| CLI | `scholar_agent.cli` | `ask`, `research`, `evaluate`, `demo`, `retrieve`, `graph`, … |

### Demo observability (Phase 9)

```text
Sidebar: corpus/index health + ablation toggles
Main: chat → Answer | Trace | Sources | Naive RAG tabs
Replay: data/demo/runs/*.json (no live API required)
```

## Prototype loop

```text
START → decide ──retrieve──► retrieve ──► decide
              └──verify───► verify ──decide or finish──► END
```

- **decide:** deterministic fake model chooses retrieve / verify
- **retrieve:** emits fake tool observations with scores
- **verify:** sufficiency check against `required_evidence` and budgets
- **finish:** records termination reason (no chain-of-thought)

Budgets: max tool calls, max iterations. Exhaustion forces verify then terminate.

## Constraints carried forward

1. Pydantic models at module boundaries
2. Stable paper / chunk / entity / evidence / run IDs
3. PDF page provenance end-to-end
4. Canonical chunk store is source of truth for every index
5. Graph triples are not facts without source evidence
6. No user-facing chain-of-thought
7. Tool, iteration, token, and latency budgets
8. Live paid-provider tests are optional
