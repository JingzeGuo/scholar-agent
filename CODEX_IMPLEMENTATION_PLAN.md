# ScholarAgent: Evidence-Driven Multi-Agent GraphRAG for Literature Research

## Final Implementation Plan for Codex

**Version:** 2.0
**Date:** 2026-07-11
**Purpose:** Portfolio project for junior Agent/RAG engineering interviews
**Primary LLM provider:** DeepSeek API, OpenAI-compatible endpoint
**Positioning:** A portfolio-grade local prototype with production-oriented engineering practices

---

## 1. Project Goal

Build **ScholarAgent**, a multi-agent literature research system that answers complex questions over a curated corpus of approximately 120 papers about RAG, LLM agents, evaluation, and related methods.

ScholarAgent must do more than retrieve chunks and place them in a prompt. It must:

1. decompose a complex question into verifiable sub-questions;
2. select retrieval strategies adaptively;
3. collect structured, page-level evidence;
4. detect missing, unsupported, or conflicting evidence;
5. perform targeted corrective retrieval;
6. generate an answer only from verified evidence;
7. attach citations that can be traced back to the original PDF;
8. expose an auditable execution trace without exposing private chain-of-thought;
9. evaluate every major component against simpler baselines.

The central engineering hypothesis is:

> Complex literature questions fail not only because of weak generation, but because of incomplete task decomposition, one-size-fits-all retrieval, and unverified evidence. Structured planning, adaptive retrieval, an evidence ledger, and independent verification should address these failure modes.

---

## 2. Interview Narrative

Use the following as the project's one-sentence pitch:

> I built a multi-agent literature research system in which a Planner decomposes complex questions, a Researcher autonomously chooses hybrid or graph retrieval tools, a Verifier checks evidence coverage and citation support, and a Writer generates answers only from verified evidence; I then used ablation experiments to measure where each component actually helps.

The project must support five defensible interview stories:

1. **Adaptive retrieval:** why dense, sparse, reranking, and graph retrieval solve different query types.
2. **Evidence management:** how an evidence ledger prevents context dumping, duplicate retrieval, and unsupported citations.
3. **Multi-agent separation:** why research and verification have different objectives and context views.
4. **Graph construction:** how entity resolution and provenance determine whether a knowledge graph is useful.
5. **Evaluation:** how component-level ablations reveal when a sophisticated module helps or hurts.

Do not describe the project merely as “LangGraph + Chroma + NetworkX + RAGAS.” Every dependency must correspond to a measured design decision.

---

## 3. Scope and Corpus

### 3.1 Target corpus

Build a curated corpus of **100–150 papers**, with **120 papers** as the target.

Recommended composition:

| Category | Target count |
|---|---:|
| Foundational and representative papers | 20 |
| Retrieval, reranking, and query transformation | 30 |
| Agent planning, tool use, and memory | 25 |
| Agentic RAG, corrective RAG, and GraphRAG | 20 |
| Benchmarks, evaluation, and surveys | 25 |
| **Total** | **120** |

Do not collect papers randomly. Start from several high-quality surveys and representative papers, follow their references, and add recent relevant work. The corpus should contain meaningful relationships among papers, methods, datasets, metrics, and baselines.

### 3.2 Corpus manifest

Maintain `data/corpus_manifest.jsonl`. Every paper must have:

- stable `paper_id`;
- title;
- authors;
- publication year;
- venue or source;
- DOI or arXiv ID when available;
- local PDF filename;
- URL;
- topic labels;
- ingestion status;
- content hash.

The content hash makes ingestion idempotent and allows changed papers to be reprocessed selectively.

### 3.3 Target scale reported in README

Report actual values after ingestion, for example:

```text
Papers:                  120
Pages:                 2,100
Chunks:                6,000–8,000
Canonical graph nodes: 1,000–2,000
Graph relations:       3,000–6,000
Resolved aliases:        200–500
Evaluation questions:         50
```

These are targets, not acceptance thresholds. Report real results honestly.

---

## 4. System Architecture

### 4.1 Offline ingestion path

```text
PDF corpus
  → metadata extraction
  → page and section parsing
  → token-aware section chunking
  → canonical chunk store
  → dense index
  → BM25 index
  → schema-constrained relation extraction
  → entity resolution
  → evidence-linked knowledge graph
```

### 4.2 Online research path

```text
User query
  → Supervisor / Planner
  → structured sub-questions
  → Research Agent tool loop
  → Evidence Ledger
  → independent Verification Agent
       ├── sufficient → Writer
       └── insufficient → targeted corrective retrieval
  → citation validator
  → final answer + sources + execution trace
```

### 4.3 Agent boundaries

This project uses separate agents only where separation has a concrete purpose.

#### Supervisor / Planner Agent

- Sees the user query and compact corpus capabilities.
- Does not receive raw retrieval results.
- Classifies query type.
- Produces structured sub-questions and evidence requirements.
- Tracks completion of the research plan.

#### Research Agent

- Receives one or more sub-questions.
- Has retrieval tools.
- Autonomously chooses tools and can make multiple tool calls.
- Returns structured evidence, not a prose answer.
- Has a fixed tool-call and token budget.

#### Verification Agent

- Sees the original query, plan, and evidence ledger.
- Does not see the Research Agent's hidden reasoning.
- Checks coverage, relevance, claim support, contradiction, and source diversity.
- Returns concrete missing aspects and corrective queries.

#### Writer

- Reads only verified evidence and the writing instructions.
- Does not perform retrieval.
- Generates claims with evidence IDs.
- May be implemented as a deterministic graph node rather than being marketed as another agent.

#### Citation Validator

- Runs after drafting.
- Ensures every citation ID exists.
- Checks that cited evidence supports the attached claim.
- Removes, repairs, or flags unsupported claims before final output.

### 4.4 What makes this genuinely agentic

The Research Agent must contain an internal action-observation loop:

```text
inspect sub-question
  → choose tool
  → inspect observations
  → update evidence coverage
  → choose another tool or stop
```

A node that mechanically calls all retrievers once is not sufficient. Agent decisions and tool outcomes must be logged in the execution trace.

---

## 5. Technology Choices

| Category | Choice |
|---|---|
| Language | Python 3.11+ |
| Package management | `uv` + `pyproject.toml` + lockfile |
| Orchestration | LangGraph |
| LLM integration | `langchain-openai` or direct OpenAI-compatible client after compatibility spike |
| Main model | `deepseek-v4-pro` |
| Fast structured tasks | `deepseek-v4-flash` |
| Embeddings | `BAAI/bge-small-en-v1.5` initially |
| Dense store | Chroma persistent local client |
| Sparse retrieval | BM25 with a persisted, chunk-ID-aligned index |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Graph | NetworkX `MultiDiGraph` |
| PDF parsing | PyMuPDF as the default parser |
| Schemas | Pydantic v2 models |
| UI | Streamlit |
| Evaluation | deterministic retrieval metrics + citation metrics + RAGAS + human review subset |
| CLI | Typer |
| Testing | pytest |
| Quality | Ruff and mypy |

Do not use `pip freeze` as the dependency design. Declare direct dependencies in `pyproject.toml` and commit the generated lockfile.

### 5.1 DeepSeek compatibility spike

Before building the full system, create a small compatibility script that verifies:

- normal chat completion;
- streaming;
- structured JSON output;
- tool calling;
- thinking and non-thinking modes;
- multi-turn tool use;
- correct handling of reasoning-related response fields;
- retry behavior for malformed JSON and rate limits.

Use `deepseek-v4-flash` in non-thinking mode for extraction and classification unless testing shows a clear reason to use thinking mode. Do not assume every LangChain wrapper version handles provider-specific fields correctly.

---

## 6. Repository Structure

```text
scholar-agent/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── default.yaml
│   └── evaluation.yaml
├── data/
│   ├── papers/
│   ├── corpus_manifest.jsonl
│   ├── processed/
│   │   ├── papers.jsonl
│   │   ├── chunks.jsonl
│   │   ├── entities.jsonl
│   │   ├── relations.jsonl
│   │   └── knowledge_graph.json
│   ├── indexes/
│   │   ├── chroma/
│   │   └── bm25/
│   └── evaluation/
│       ├── questions.jsonl
│       ├── reference_evidence.jsonl
│       └── frozen_split.json
├── src/
│   └── scholar_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── models.py
│       ├── llm/
│       │   ├── client.py
│       │   ├── structured.py
│       │   └── prompts.py
│       ├── ingestion/
│       │   ├── loader.py
│       │   ├── metadata.py
│       │   ├── chunker.py
│       │   ├── graph_extractor.py
│       │   ├── entity_resolver.py
│       │   ├── graph_store.py
│       │   └── pipeline.py
│       ├── retrieval/
│       │   ├── dense.py
│       │   ├── sparse.py
│       │   ├── fusion.py
│       │   ├── reranker.py
│       │   ├── graph.py
│       │   ├── router.py
│       │   └── tools.py
│       ├── agents/
│       │   ├── state.py
│       │   ├── planner.py
│       │   ├── researcher.py
│       │   ├── verifier.py
│       │   ├── writer.py
│       │   ├── citation_validator.py
│       │   └── workflow.py
│       ├── evaluation/
│       │   ├── dataset.py
│       │   ├── baselines.py
│       │   ├── retrieval_metrics.py
│       │   ├── citation_metrics.py
│       │   ├── answer_metrics.py
│       │   ├── ablation.py
│       │   └── report.py
│       ├── app/
│       │   └── streamlit_app.py
│       └── cli.py
├── scripts/
│   ├── download_corpus.py
│   ├── inspect_graph.py
│   └── deepseek_compatibility.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── e2e/
├── notebooks/
│   └── error_analysis.ipynb
└── docs/
    ├── architecture.md
    ├── evaluation.md
    ├── design_decisions.md
    ├── failure_analysis.md
    ├── interview_guide.md
    └── demo_script.md
```

Generated indexes, processed corpora, model caches, secrets, and large PDFs should be gitignored. Keep a small legal test fixture corpus in the repository.

---

## 7. Core Data Models

Define explicit Pydantic models. Avoid untyped `dict` values across module boundaries.

### 7.1 Paper and chunk

```python
class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    arxiv_id: str | None
    doi: str | None
    source_url: str | None
    pdf_path: str
    content_hash: str


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int
    content_hash: str
```

### 7.2 Planning

```python
class SubQuestion(BaseModel):
    id: str
    question: str
    query_type: Literal[
        "semantic", "keyword", "comparison", "relational", "synthesis"
    ]
    required_evidence: list[str]
    status: Literal["pending", "covered", "missing"] = "pending"


class QueryPlan(BaseModel):
    original_query: str
    answer_type: str
    sub_questions: list[SubQuestion]
    expected_source_diversity: int
```

### 7.3 Evidence ledger

```python
class EvidenceItem(BaseModel):
    evidence_id: str
    sub_question_id: str
    claim: str
    evidence_text: str
    paper_id: str
    chunk_id: str
    page_start: int
    page_end: int
    retrieval_method: str
    retrieval_score: float | None
    rerank_score: float | None
    support_score: float | None
    contradiction: bool = False
```

Evidence IDs must be stable within a run. Deduplicate evidence by chunk ID and normalized evidence span.

### 7.4 Verification

```python
class VerificationResult(BaseModel):
    is_sufficient: bool
    coverage_score: float
    covered_sub_questions: list[str]
    missing_sub_questions: list[str]
    unsupported_claims: list[str]
    conflicting_evidence_ids: list[str]
    missing_aspects: list[str]
    corrective_queries: list[str]
    rationale_summary: str
```

`rationale_summary` is a concise decision explanation, not private chain-of-thought.

### 7.5 Workflow state

State should include:

- `run_id`;
- `query`;
- `plan`;
- `active_sub_questions`;
- `evidence_ledger`;
- `verification`;
- `corrective_queries`;
- `iteration`;
- `tool_call_count`;
- `token_usage`;
- `latency_ms`;
- `execution_events`;
- `draft_answer`;
- `final_answer`;
- `citation_report`;
- `errors`.

Define LangGraph reducers for evidence and execution events. Reducers must deduplicate rather than blindly append repeated retrieval results.

---

## 8. Ingestion and Indexing

### 8.1 PDF parsing

Use PyMuPDF to preserve page boundaries. Store extracted text page by page before chunking.

Required behavior:

- detect empty or scanned pages;
- normalize repeated headers and footers;
- preserve page numbers;
- detect section headings heuristically;
- flag extraction quality problems;
- never silently index an empty paper.

Complex table and formula understanding is explicitly out of scope for the first complete version. Record this limitation.

### 8.2 Chunking

Use section-aware, token-aware chunking:

- target: approximately 500–700 tokens;
- overlap: approximately 60–100 tokens;
- do not cross paper boundaries;
- prefer not to cross section boundaries;
- attach paper, page, section, and chunk metadata;
- keep a content hash for reproducibility.

Do not describe an 800-character recursive split as an 800-token split.

### 8.3 Canonical chunk store

`chunks.jsonl` is the source of truth for all retrieval indexes. Dense, sparse, graph, evaluation, and citation components must use the same stable chunk IDs.

### 8.4 Dense and sparse indexes

- Store dense embeddings in Chroma with stable chunk IDs.
- Build BM25 over the same canonical chunks.
- Persist BM25 vocabulary, corpus statistics, chunk order, and index metadata.
- Verify index and chunk-store hashes when loading.
- Rebuild only when the corpus or retrieval configuration changes.

### 8.5 Graph schema

Use a constrained ontology:

#### Node types

- `Paper`
- `Method`
- `Dataset`
- `Task`
- `Metric`
- `Author`
- `Organization`

#### Relation types

- `PROPOSES`
- `EXTENDS`
- `USES`
- `EVALUATES_ON`
- `REPORTS`
- `COMPARES_WITH`
- `OUTPERFORMS`
- `CITES`
- `AUTHORED_BY`

Every extracted relation must contain:

- subject and object surface forms;
- proposed canonical types;
- relation type;
- evidence span;
- paper ID;
- chunk ID;
- page number;
- extraction confidence.

Discard relations that cannot be connected to an evidence span in the source chunk.

### 8.6 Entity resolution

Implement a staged resolver:

1. normalize case, punctuation, whitespace, and Unicode;
2. expand or detect acronyms;
3. apply exact alias matches;
4. retrieve candidate canonical entities using string and embedding similarity;
5. use an LLM only for ambiguous candidate pairs;
6. assign a canonical entity ID;
7. persist aliases and resolution decisions.

Create evaluation fixtures for difficult examples such as acronyms, versioned model names, and similar dataset names.

### 8.7 Graph persistence

Use NetworkX `MultiDiGraph`. Persist a portable node-link JSON representation rather than relying on removed NetworkX gpickle helpers.

Graph facts are never treated as independent truth. Graph retrieval must return the original supporting chunks and pages.

---

## 9. Retrieval System

### 9.1 Retrieval tools

Expose typed tools:

```text
dense_search(query, k, filters)
sparse_search(query, k, filters)
hybrid_search(query, k, filters)
graph_search(query, max_hops, relation_filters)
get_chunk(chunk_id)
get_paper(paper_id)
```

Every tool returns structured results with scores, chunk IDs, paper IDs, pages, sections, and text snippets.

### 9.2 Hybrid retrieval

Implement explicit Reciprocal Rank Fusion instead of hiding fusion inside an opaque wrapper.

Initial defaults:

```text
dense candidates:   12
sparse candidates:  12
fused candidates:   20
reranked results:    8
```

Make all values configurable. Log component ranks and final fused scores for debugging.

### 9.3 Adaptive routing

The router selects a retrieval policy based on query type:

| Query pattern | Default policy |
|---|---|
| Conceptual paraphrase | Dense |
| Exact model, dataset, metric, or acronym | Sparse or hybrid |
| General evidence question | Hybrid + reranker |
| Method–dataset–metric relationship | Graph + supporting chunks |
| Cross-paper comparison | Hybrid plus graph when entities are available |
| Corrective search | Policy selected from the verifier's missing aspect |

The Research Agent may override the default policy within its budget. Log both the recommendation and final action.

### 9.4 Graph retrieval

Graph search must:

1. extract candidate entities from the query;
2. resolve them to canonical graph nodes;
3. retrieve schema-valid paths of at most two hops initially;
4. score paths using query relevance, relation confidence, and evidence quality;
5. return supporting chunks, not only textualized triples;
6. deduplicate evidence already present in the ledger.

Unfiltered neighborhood expansion is not acceptable.

---

## 10. Multi-Agent Workflow

### 10.1 Planner output

The Planner must return `QueryPlan` through structured output. It must not return a free-form plan string.

The Planner should avoid unnecessary decomposition. Simple factual questions may produce one sub-question, while synthesis questions may produce several.

### 10.2 Research loop

For each active sub-question, the Research Agent:

1. inspects evidence requirements;
2. selects a tool;
3. executes the tool;
4. converts useful results to `EvidenceItem` objects;
5. updates local coverage;
6. either chooses another tool or returns.

Initial safety budgets:

- maximum 4 tool calls per research pass;
- maximum 3 corrective iterations overall;
- maximum configurable evidence count per sub-question;
- explicit timeout and token budget.

Research for independent sub-questions may run in parallel through LangGraph fan-out, but evidence merging must use deterministic reducers.

### 10.3 Verification loop

The Verifier evaluates:

- coverage of every required sub-question;
- direct relevance of evidence;
- whether proposed claims are entailed by evidence;
- contradiction across sources;
- source diversity for comparison questions;
- whether the question is answerable from the corpus.

When evidence is insufficient, return targeted corrective queries tied to missing sub-questions. Do not merely return `is_sufficient = false`.

The workflow terminates corrective retrieval when any of the following is true:

- evidence is sufficient;
- the iteration budget is exhausted;
- no new unique evidence was found in the last iteration;
- the Verifier determines the corpus cannot answer the question;
- the cost or tool-call budget is exhausted.

### 10.4 Writing and citation generation

The Writer receives only:

- original question;
- answer format;
- verified evidence ledger;
- contradiction notes;
- corpus insufficiency notes.

The Writer must emit structured claims with evidence IDs before rendering Markdown. Final inline citations are generated from those IDs.

If the corpus cannot support a claim, the answer must state the limitation rather than fill the gap from model memory.

### 10.5 Citation validation

Validation must check:

- every citation refers to a real evidence ID;
- every evidence ID maps to a real paper, chunk, and page;
- the cited evidence supports the nearby claim;
- citations are not attached only at paragraph level when several independent claims exist;
- references are deduplicated and consistently formatted.

Store both a machine-readable citation report and a user-facing source list.

### 10.6 Execution trace

Expose:

- generated sub-questions;
- routing decisions;
- tool names and sanitized arguments;
- retrieved source IDs and scores;
- evidence coverage summaries;
- verifier decisions;
- corrective queries;
- iteration count;
- latency and token usage;
- final citation validation result.

Do not expose hidden chain-of-thought or provider reasoning fields.

---

## 11. Evaluation Design

### 11.1 Evaluation dataset

Create **50 manually verified questions**:

| Question type | Count |
|---|---:|
| Single-paper factual | 10 |
| Exact terminology / keyword | 10 |
| Cross-paper comparison | 15 |
| Multi-hop relational | 10 |
| Unanswerable from corpus | 5 |

For each question store:

- question ID;
- question type;
- reference answer or reference claims;
- required paper IDs;
- required evidence chunks/pages;
- acceptable alternate evidence;
- whether graph reasoning is expected;
- annotation notes.

Freeze the dataset before final tuning. Keep evaluation questions out of prompts and development fixtures.

### 11.2 Systems compared

Run the same frozen dataset against:

1. **Naive Dense RAG** — dense top-k + single generation call;
2. **Hybrid RAG** — dense + BM25 with RRF;
3. **Hybrid + Reranker**;
4. **Hybrid + Graph**;
5. **Hybrid + Corrective Retrieval**;
6. **Full ScholarAgent**;
7. optional routing ablation: static all-tools versus adaptive routing.

Use the same generation model and answer prompt where applicable so retrieval comparisons are meaningful.

### 11.3 Metrics

#### Retrieval

- Recall@K;
- MRR;
- nDCG@K when graded relevance is available;
- graph path evidence recall;
- unique useful evidence per tool call.

#### Citations

- citation precision;
- citation recall;
- citation validity rate;
- page-level traceability rate.

#### Answers

- claim-level correctness;
- faithfulness;
- response relevancy;
- completeness;
- unanswerable-question refusal accuracy;
- contradiction handling accuracy.

#### Agents and operations

- plan coverage;
- tool selection accuracy on labeled questions;
- corrective-loop trigger precision;
- improvement after correction;
- average number of tool calls;
- average iterations;
- latency;
- input/output tokens;
- estimated cost;
- error rate.

### 11.4 Evaluation discipline

- Do not require an arbitrary improvement such as “+20 points” in advance.
- Report confidence intervals or per-question paired differences where practical.
- Report results by question category, not only overall average.
- Manually review a representative subset.
- Record failures and regressions.
- If GraphRAG helps only relational questions, say so and route only those questions to GraphRAG.
- If a module does not help, retain the result as an engineering finding or remove the module from the default route.

### 11.5 Required experiment outputs

Generate:

- CSV/JSON results per run;
- aggregate metric table;
- per-category comparison chart;
- latency/cost chart;
- corrective-loop before/after examples;
- graph retrieval case study;
- failure analysis with at least five concrete cases.

---

## 12. Streamlit Demo

### 12.1 Main interface

- chat input;
- final answer with clickable source cards;
- paper title, page, section, and evidence snippet;
- corpus statistics;
- session reset;
- index health indicator.

### 12.2 Research trace panel

Show:

- query classification;
- sub-questions;
- tool calls;
- retrieval methods and scores;
- graph paths;
- evidence ledger summary;
- verifier coverage;
- corrective iterations;
- citation validation;
- latency and cost.

### 12.3 Interview controls

Include toggles or modes for:

- compare with Naive RAG;
- disable graph retrieval;
- disable corrective loop;
- static versus adaptive retrieval;
- show only verified evidence;
- replay saved demo runs.

Precompute several stable demo runs so an API outage does not ruin a live interview demo.

---

## 13. Reliability and Safety

Implement:

- configuration validation at startup;
- structured-output validation and repair;
- retries with exponential backoff;
- API timeouts;
- rate-limit handling;
- deterministic IDs;
- idempotent ingestion;
- per-run budgets;
- caching for extraction and evaluation calls;
- graceful degradation when graph or reranker is unavailable;
- explicit “corpus cannot answer” behavior;
- structured error events;
- sanitized logs with no API keys.

Treat paper content as untrusted data. Retrieval content must not be allowed to override system instructions or request tool execution. Clearly delimit source text in prompts.

For local development, use an in-memory checkpointer where appropriate. If conversation persistence is demonstrated, use a durable checkpointer rather than presenting in-memory state as persistent memory.

---

## 14. Testing Strategy

### 14.1 Unit tests

Test:

- stable ID generation;
- metadata normalization;
- page-preserving chunking;
- RRF calculations;
- BM25/chunk-ID alignment;
- entity normalization and alias resolution;
- graph serialization;
- evidence deduplication reducers;
- loop termination logic;
- citation formatting and validation;
- configuration parsing.

### 14.2 Integration tests

Use deterministic fake LLM responses to test:

- structured Planner output;
- Research Agent tool loops;
- insufficient-evidence correction;
- no-new-evidence termination;
- contradiction handling;
- unanswerable response;
- citation repair.

### 14.3 End-to-end tests

Maintain a small fixture corpus of approximately five papers and several fixed questions. Verify the complete path from ingestion to final cited answer.

Do not make the regular test suite depend on paid API calls. Put live-provider tests behind an explicit marker.

---

## 15. Implementation Phases

Phases are dependency-ordered, not time estimates. Complete acceptance checks before moving forward.

### Phase 0 — Compatibility and architecture spike

Deliver:

- initialized repository;
- `pyproject.toml`, linting, typing, and tests;
- configuration system;
- DeepSeek compatibility script;
- small LangGraph loop prototype;
- architectural decision records.

Acceptance:

- structured output, streaming, and tool calling verified;
- one conditional loop runs with a deterministic fake model;
- dependency versions are locked.

### Phase 1 — Domain models and canonical storage

Deliver:

- all core Pydantic models;
- deterministic ID helpers;
- JSONL repositories;
- corpus manifest loader;
- test fixtures.

Acceptance:

- schema round trips pass;
- IDs remain stable across runs;
- invalid metadata fails clearly.

### Phase 2 — PDF ingestion

Deliver:

- PDF loader;
- metadata extraction;
- page/section parser;
- token-aware chunker;
- ingestion CLI;
- extraction quality report.

Acceptance:

- fixture PDFs preserve correct pages;
- duplicate ingestion is idempotent;
- empty and scanned PDFs are flagged.

### Phase 3 — Baseline and hybrid retrieval

Deliver:

- dense index;
- persistent BM25 index;
- explicit RRF;
- reranker;
- typed retrieval tools;
- retrieval debug output;
- Naive RAG baseline.

Acceptance:

- all indexes share stable chunk IDs;
- retrieval unit tests pass;
- baseline answers include valid page references.

### Phase 4 — Knowledge graph

Deliver:

- graph extraction schema and prompts;
- evidence-span validation;
- staged entity resolver;
- `MultiDiGraph` builder;
- node-link JSON persistence;
- graph inspection CLI;
- graph retrieval tool.

Acceptance:

- every graph relation maps to source evidence;
- alias fixtures resolve correctly;
- graph paths return supporting chunks;
- graph statistics and isolated-node rates are reported.

### Phase 5 — Adaptive routing and Research Agent

Deliver:

- query classifier/router;
- Research Agent subgraph;
- tool loop and budgets;
- evidence ledger reducers;
- parallel sub-question research where safe;
- structured execution events.

Acceptance:

- Research Agent chooses different tools for labeled query types;
- duplicate evidence is merged;
- tool and iteration budgets cannot be exceeded.

### Phase 6 — Planner, Verifier, and corrective loop

Deliver:

- structured Planner;
- independent Verifier;
- concrete corrective query generation;
- loop termination conditions;
- unanswerable detection;
- full LangGraph workflow.

Acceptance:

- missing evidence triggers targeted retrieval;
- no-new-evidence stops the loop;
- contradictory evidence is retained and surfaced;
- workflow terminates under every test scenario.

### Phase 7 — Writer and citation validator

Deliver:

- evidence-constrained Writer;
- claim-to-evidence intermediate format;
- inline citation renderer;
- citation validator;
- source cards and reference list.

Acceptance:

- no citation refers to a nonexistent evidence ID;
- every source maps to a real PDF and page;
- unsupported claims are removed or explicitly qualified.

### Phase 8 — Evaluation framework

Deliver:

- frozen 50-question dataset;
- all baseline configurations;
- deterministic retrieval and citation metrics;
- RAGAS integration;
- ablation runner;
- result reports and charts;
- failure analysis notebook.

Acceptance:

- every system runs on the identical frozen split;
- results are reproducible from saved configs;
- per-category results, latency, and cost are reported;
- at least five failures are analyzed manually.

### Phase 9 — Streamlit demo and observability

Deliver:

- chat interface;
- trace panel;
- source viewer;
- baseline comparison;
- ablation toggles;
- saved-run replay;
- corpus and index status.

Acceptance:

- a user can trace a final claim to a PDF page;
- corrective loops are visibly understandable;
- the demo still works using saved runs without live API access.

### Phase 10 — Hardening and portfolio documentation

Deliver:

- error handling;
- caching;
- complete tests;
- README architecture diagram;
- quantitative results;
- design decisions;
- failure analysis;
- demo video/script;
- interview guide.

Acceptance:

- fresh setup works from documented commands;
- no secrets or generated indexes are committed;
- core tests run without paid API calls;
- README distinguishes measured results from claims and future work.

---

## 16. Definition of Done

- [ ] Approximately 120 curated papers are ingested with a reproducible manifest.
- [ ] PDF pages, sections, chunks, and citations remain traceable end to end.
- [ ] Dense, BM25, RRF, reranking, and graph retrieval are independently testable.
- [ ] Knowledge graph entities are canonicalized and every relation has provenance.
- [ ] Research Agent performs real tool-selection loops rather than fixed sequential retrieval.
- [ ] Planner produces structured sub-questions and evidence requirements.
- [ ] Verifier identifies concrete evidence gaps and contradictions.
- [ ] Corrective retrieval terminates safely and is measurable.
- [ ] Writer uses only verified evidence.
- [ ] Citation validator catches missing and unsupported citations.
- [ ] Fifty frozen, manually verified evaluation questions exist.
- [ ] All ablation systems are evaluated on the same split.
- [ ] Per-category retrieval, answer, citation, latency, and cost results are reported.
- [ ] Streamlit shows an auditable execution trace and baseline comparisons.
- [ ] Tests cover reducers, routing, graph resolution, termination, and citations.
- [ ] README, failure analysis, demo script, and interview guide are complete.

---

## 17. Interview Preparation Deliverables

Create `docs/interview_guide.md` containing:

### 17.1 Sixty-second explanation

- problem;
- baseline failure;
- architecture;
- three main technical decisions;
- measured result;
- most important limitation.

### 17.2 Five-minute architecture walkthrough

Explain:

- offline ingestion;
- query planning;
- tool routing;
- evidence ledger;
- verification loop;
- citation validation;
- evaluation.

### 17.3 Questions the project owner must answer

1. Why LangGraph instead of a simple chain?
2. What exactly makes each component an agent?
3. Why separate Researcher and Verifier?
4. When does BM25 beat dense retrieval?
5. How does RRF work?
6. Why use a cross-encoder after retrieval?
7. Why is GraphRAG needed for this corpus?
8. How are aliases and duplicate graph entities resolved?
9. How does graph evidence map back to a PDF page?
10. How do reducers prevent duplicate state during parallel execution?
11. How does the corrective loop terminate?
12. How are unsupported citations detected?
13. How was the evaluation dataset constructed and frozen?
14. Which ablation produced the largest improvement, and for which question type?
15. Which module failed to help, and why?
16. What would need to change for multi-user production deployment?

### 17.4 Required failure stories

Document at least three real failures discovered during implementation, such as:

- dense retrieval misses an exact dataset name;
- graph extraction creates duplicate entities before normalization;
- the Verifier repeatedly asks for unavailable evidence;
- a Writer attaches a valid citation to the wrong claim;
- GraphRAG increases noise on simple factual questions.

For each, explain symptom, root cause, fix, and measurement after the fix.

---

## 18. README Structure

The final README should contain:

1. concise project pitch;
2. demo GIF or video;
3. architecture diagram;
4. why naive RAG fails;
5. corpus statistics;
6. agent responsibilities;
7. evidence ledger example;
8. retrieval and graph design;
9. evaluation dataset;
10. ablation results;
11. latency and cost;
12. failure analysis;
13. setup and commands;
14. limitations;
15. future work.

Do not use unverified marketing claims such as “production-ready,” “industry standard,” or “state of the art.” Use measured statements.

---

## 19. Commands to Support

Exact CLI syntax may evolve, but the final project should provide equivalents of:

```bash
# Setup
uv sync
cp .env.example .env

# Compatibility check
uv run python scripts/deepseek_compatibility.py

# Corpus
uv run scholar-agent corpus validate
uv run scholar-agent ingest --manifest data/corpus_manifest.jsonl
uv run scholar-agent graph inspect

# Ask a question
uv run scholar-agent ask "Compare corrective RAG and self-RAG."

# Evaluation
uv run scholar-agent evaluate --config configs/evaluation.yaml
uv run scholar-agent ablate --all

# Quality
uv run pytest
uv run ruff check .
uv run mypy src

# Demo
uv run streamlit run src/scholar_agent/app/streamlit_app.py
```

---

## 20. Instructions for Codex

Use the following prompt to begin implementation:

```text
You are implementing ScholarAgent from CODEX_IMPLEMENTATION_PLAN.md.

Work phase by phase in dependency order. Start with Phase 0 only.

For each phase:
1. inspect the existing repository and preserve unrelated user changes;
2. state the phase acceptance criteria;
3. implement complete, typed code without placeholder pass statements;
4. add or update tests;
5. run the relevant tests, linting, and type checks;
6. report what passed, what remains, and any deviation from the plan;
7. do not begin the next phase until the current acceptance criteria pass.

Architectural constraints:
- use Pydantic models at module boundaries;
- keep stable paper, chunk, entity, evidence, and run IDs;
- preserve PDF page provenance end to end;
- use the canonical chunk store as the source of truth for every index;
- make retrieval components independently testable;
- do not treat graph triples as facts without source evidence;
- do not expose chain-of-thought;
- enforce tool, iteration, token, and latency budgets;
- keep live paid-provider tests optional;
- implement baseline systems before sophisticated agents;
- record design deviations in docs/design_decisions.md.

Begin with repository inspection and Phase 0: compatibility and architecture spike.
```

---

## 21. Optional Extensions After the Core Project

Only begin these after the complete evaluation and interview materials exist:

- incremental ingestion and index updates;
- multimodal figure/table retrieval;
- citation graph imported from Semantic Scholar or OpenAlex;
- human-in-the-loop correction of entity resolution;
- persistent multi-session research workspaces;
- export to Markdown, DOCX, or PDF;
- web search for papers outside the corpus;
- RAPTOR or parent-child hierarchical retrieval;
- server-backed Chroma or another production vector database;
- API service with authentication, quotas, and concurrent users;
- tracing integration such as LangSmith or OpenTelemetry.

These are extensions, not substitutes for rigorous evidence handling and evaluation.
