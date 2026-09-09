# Scholar-Agent

A compact agentic RAG workflow for evidence-grounded academic research.

Scholar-Agent answers questions over a local collection of academic PDFs. The
Planner makes bounded retrieval decisions before generation; the remaining
retrieval, reranking, evidence allocation, and citation-validation steps are
deterministic.

## Architecture

```text
Question
   ↓
Planner (atomic requirements + bounded retrieval decisions)
   ↓
Researcher
   ├── BM25, dense, or hybrid per requirement
   ├── RRF for hybrid requirements only
   ├── shared cross-encoder reranker
   └── requirement- and target-aware evidence selection
   ↓
Writer
   ↓
Deterministic physical-page citation validation
   ↓
Answer
```

LangGraph connects four production nodes: Planner, Researcher, Writer, and
Citation Validator. A paired 50-question ablation found that adding an LLM
answer verifier and repair step reduced Strict Success from 88% to 78% and
increased average latency from 17.99s to 50.32s, with no improvement in
Requirement Accuracy. The production workflow therefore omits these steps.

## Adaptive retrieval planning

The Planner produces one to five atomic requirements:

```python
{
    "requirements": [
        {
            "id": "R1",             # assigned by Python
            "description": "...",
            "targets": ["..."],     # 0–3 names copied from the question
            "query": "...",
            "retrieval_strategy": "bm25",  # bm25 | dense | hybrid
            "top_k": 8,
        }
    ]
}
```

The LLM chooses each strategy from the requirement's evidence need. Exact
titles, acronyms, or method names may favor BM25; conceptual mechanisms may
favor dense retrieval; mixed or ambiguous needs may favor hybrid retrieval.
Broad exploratory requirements may choose a larger depth. These are prompt
examples, not Python routing rules.

Planner output is sanitized before use:

- unknown or missing strategies fall back to `hybrid`;
- integer `top_k` values are clamped to 4–12, while malformed values fall back
  to 8;
- empty queries fall back to the requirement description;
- malformed or duplicate requirements are removed;
- if no requirement survives, the original question becomes one target-free
  hybrid requirement with `top_k=8`.

## Retrieval modes

One Researcher implementation supports both experiment modes:

- `adaptive` (production default): execute each requirement's planned
  `retrieval_strategy` and `top_k`.
- `fixed_hybrid`: ignore the planned strategy and run BM25 plus dense retrieval
  with RRF for every requirement. Planned `top_k` is retained so the ablation
  isolates strategy routing.

Configure production with:

```bash
export SCHOLAR_AGENT_RETRIEVAL_MODE=adaptive
# or
export SCHOLAR_AGENT_RETRIEVAL_MODE=fixed_hybrid
```

The Python API also accepts an explicit override:

```python
state = run_question(
    question,
    engine,
    settings,
    llm,
    retrieval_mode="fixed_hybrid",
)
```

## Researcher

Each atomic requirement follows exactly one route:

```text
bm25:   BM25(query) ───────────────────────→ rerank
dense:  Dense(query) ──────────────────────→ rerank
hybrid: BM25(query) ─┐
                     ├→ RRF ───────────────→ rerank
        Dense(query) ┘
```

Every route uses the same cross-encoder. The reranker retains a separate score
for every requirement, so adaptive candidates enter the same requirement-aware
selection pipeline regardless of their initial retrieval route. Selection
preserves:

- one relevant slot per atomic requirement when available;
- target-aware allocation for comparisons;
- per-paper and physical-page diversity while filling remaining slots;
- a normal global limit of eight evidence chunks, expanding only when distinct
  requirement/target coverage requires it;
- at most 30 cross-encoder candidates, with local candidates reserved for each
  requirement before global fusion fills the pool.

Dense queries with the same `top_k` are encoded together. A BM25-only
requirement never invokes dense retrieval, and a dense-only requirement never
invokes BM25.

## Writer and citation validation

The Writer sees all selected evidence and may cite only temporary IDs such as
`[E1]`. An empty evidence set produces a deterministic abstention without an
LLM call. The prompt forbids introductory summaries and concluding
restatements: the answer starts with a directly supported claim, every factual
sentence must carry its own adjacent evidence ID, and the answer ends after the
last cited detail.

The final deterministic node converts known IDs to physical-page citations
copied from stored metadata:

```text
[E1] → [Self-RAG.pdf p.1]
```

Unknown evidence IDs and fabricated page citations are removed. This proves
provenance to a retrieved page; it is not a semantic entailment verifier.

## Page-aware ingestion and indexes

PyMuPDF extracts each physical page independently. Character chunks are about
1,200 characters with about 150 characters of overlap and never cross a page
boundary. Every stored chunk has `chunk_id`, `paper`, `page`, and `text`.

Indexing writes a BM25 token file plus a NumPy dense-embedding matrix and
metadata. Both indexes store an ordered corpus fingerprint and refuse to load
after the chunks change. Local embedding and reranker models download on first
use and fail explicitly if unavailable.

## Installation and CLI

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export DEEPSEEK_API_KEY=...
uv run scholar-agent ingest tests/fixtures/papers
uv run scholar-agent index
uv run scholar-agent ask "Compare Self-RAG and CRAG"
```

The CLI intentionally contains only `ingest`, `index`, and `ask`.

| Variable | Default |
|---|---|
| `SCHOLAR_AGENT_LLM_MODEL` | provider default |
| `SCHOLAR_AGENT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `SCHOLAR_AGENT_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `SCHOLAR_AGENT_MIN_RERANK_SCORE` | `-1.0` |
| `SCHOLAR_AGENT_RETRIEVAL_MODE` | `adaptive` |
| `SCHOLAR_AGENT_DATA_DIR` | `data` |

## Tests

The deterministic suite covers Planner sanitization, all adaptive routes,
fixed-hybrid behavior, mixed per-requirement strategies, reranking and evidence
allocation, page provenance, and citation validation.

```bash
uv run ruff check .
uv run pytest -q
make quality
```

## Evaluation

The blinded 50-question evaluation pipeline supports two main findings from
separate experiments:

- **Adaptive retrieval reduced dense retrieval operations by 22.4%**, from 76
  to 59.
- **Scholar-Agent improved Strict Success by 16.0 percentage points over a
  conventional hybrid RAG baseline**, from 46% to 62%. 

### Adaptive retrieval (`adaptive_v2`)

The benchmark calls the Planner once per question and reuses that exact
sanitized plan for both `fixed_hybrid` and `adaptive`. Only retrieval execution
differs: fixed mode overrides every strategy to hybrid, while adaptive mode
uses the planned strategy. Blinded scoring reports Strict Success, Requirement
Accuracy, Citation Support, average latency, and average LLM calls. Each trace
records the shared plan and the executed query, strategy, and depth per
requirement. See [evals/README.md](evals/README.md).

The benchmark uses 50 hand-authored English questions, 10,726 page-aware corpus
chunks, and `deepseek-v4-flash` with temperature zero. Both variants use the same
Planner, cross-encoder, requirement-aware evidence selection, Writer, and
citation validator. This comparison measures retrieval strategy routing within
Scholar-Agent.

Each variant's reported latency and LLM call count include the shared Planner
cost. A normal answer with evidence uses one Planner call and one Writer call;
an empty evidence set skips the Writer call.

The evaluation tools prepare a blinded review sheet and extract cited and gold
physical PDF pages for review. Completed review labels are aggregated into
`evals/runs/<run-id>/summary.json` and `summary.md`. Follow
[evals/README.md](evals/README.md) to run, review, and score the comparison.


## Limitations

- Planner strategy choice is LLM-based and can still be wrong.
- Citation validation establishes provenance, not semantic truth.
- The corpus and NumPy indexes are intended for laptop-scale use.
- Indexes are rebuilt as a unit rather than updated incrementally.
- Local embedding and reranker models require a download on first use.
