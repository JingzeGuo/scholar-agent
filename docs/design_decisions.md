# Design Decisions

This document records intentional deviations from `CODEX_IMPLEMENTATION_PLAN.md`
and key architectural choices. Update it whenever implementation diverges from
the plan or a non-obvious trade-off is made.

## Phase 0

### ADR-001: Package layout and dependency staging

**Decision:** Ship a minimal dependency set in the core package for Phase 0
(`pydantic`, `langgraph`, `openai`, `typer`, …). Heavy stacks (Chroma,
sentence-transformers, PyMuPDF, NetworkX, Streamlit, RAGAS) are declared as
optional extras (`retrieval`, `ui`, `eval`) and installed when later phases need
them.

**Rationale:** Keeps the compatibility spike and CI fast; avoids downloading
embedding models and PDF stacks before ingestion exists.

**Deviation:** The plan lists all technologies in one table; it does not require
installing every runtime dependency on day zero.

### ADR-002: Direct OpenAI-compatible client as primary LLM surface

**Decision:** Implement `scholar_agent.llm.client.LLMClient` on the official
`openai` Python SDK pointed at DeepSeek’s OpenAI-compatible endpoint. LangChain
chat models may wrap this later where LangGraph tool binding benefits, but the
canonical low-level surface is the thin client.

**Rationale:** The plan requires a compatibility spike for structured JSON, tool
calling, streaming, and reasoning-field handling. Provider-specific fields are
easier to inspect and sanitize with a thin wrapper than through a full LangChain
abstraction stack.

### ADR-003: Prototype loop uses a deterministic fake model

**Decision:** The Phase 0 LangGraph conditional loop (`agents/prototype_loop.py`)
uses `FakeResearchModel` instead of live DeepSeek calls.

**Rationale:** Acceptance requires “one conditional loop runs with a
deterministic fake model.” Offline CI must not depend on paid APIs. Live
provider verification is isolated in `scripts/deepseek_compatibility.py` and
optional `pytest -m live` tests.

### ADR-004: Thinking mode default off for structured tasks

**Decision:** Default `llm.thinking_enabled: false`. The client sends a soft
`extra_body` hint to disable thinking when the provider supports it. Reasoning
fields are extracted if present but never exposed as user-facing chain-of-thought.

**Rationale:** Plan guidance: use flash / non-thinking for extraction and
classification unless testing shows a clear benefit.

### ADR-005: Config is YAML + env, validated by Pydantic

**Decision:** `configs/default.yaml` holds non-secret defaults; `EnvSettings`
overrides API keys and hot budgets. `load_config()` validates with Pydantic v2
and resolves paths against the repository root.

**Rationale:** Startup validation is a reliability requirement (plan §13). Secrets
never live in committed YAML.

### ADR-006: Execution events are structured, not free-form CoT

**Decision:** `ExecutionEvent` records component, event type, short summary, and
JSON-serializable payload. Private chain-of-thought is never stored.

**Rationale:** Plan requires auditable traces without exposing CoT.

### ADR-007: Prototype loop merges events explicitly

**Decision:** The Phase 0 LangGraph prototype appends `events` inside each node
via `_append_events` rather than relying on `Annotated[..., reducer]` in the
prototype state TypedDict. Reducer helpers still live in `agents/state.py` for
later workflow state.

**Rationale:** Keeps the spike easy to type-check under current LangGraph stubs
while still demonstrating append semantics and budget-aware termination.

### ADR-008: Explicit, testable provider retries

**Decision:** Disable the OpenAI SDK's implicit retries and route completion creation through a
bounded project retry helper. Retry HTTP 429, HTTP 5xx, connection failures, and timeouts with
exponential backoff. Retry malformed structured output by making a fresh bounded request.

**Rationale:** Phase 0 requires retry behavior to be verified. An explicit policy can be tested
deterministically without manufacturing paid provider failures.

### ADR-009: Strict and auditable compatibility acceptance

**Decision:** A missing or malformed tool call fails the compatibility spike. Thinking-mode
errors also fail instead of becoming soft passes. Every run writes a secret-free JSON report;
generated reports remain ignored under `outputs/`.

**Rationale:** Successful HTTP requests are not evidence that provider features work. Phase
acceptance needs a reproducible result for every required capability.

---

## Pending (later phases)

- Canonical content-addressed IDs for papers/chunks (Phase 1)
- Chunk store as sole source of truth for all indexes (Phase 2–3)
- Graph triples require evidence spans (Phase 4)
- Research agent tool budgets and evidence reducers (Phase 5–6)
