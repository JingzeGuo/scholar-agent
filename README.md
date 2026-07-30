# ScholarAgent

> A compact multi-agent GraphRAG system for evidence-grounded academic research.

ScholarAgent is an interview-sized research system that keeps the interesting
parts of agentic retrieval visible. Four LangGraph agents plan a search, run
three complementary retrievers, verify the evidence, and write a page-cited
answer. The implementation deliberately avoids production wrappers, provider
factories, vector databases, registries, event ledgers, and speculative APIs.

## Architecture

```text
Planner
  │  up to 3 queries + 5 entities
  ▼
Researcher
  ├── BM25 sparse retrieval
  ├── Sentence Transformer dense retrieval
  ├── Lightweight GraphRAG
  ├── Reciprocal Rank Fusion
  └── Cross-encoder reranking → up to 8 evidence chunks
  ▼
Verifier
  ├── sufficient ──────────────────────────────┐
  └── insufficient → Researcher once at most  │
                                               ▼
Writer → deterministic citation validation → answer
```

The compiled graph has exactly four nodes:

```text
planner → researcher → verifier ─┬→ writer → END
                    ▲            │
                    └────────────┘  (one retry maximum)
```

## The four agents

### Planner

The Planner receives the original question and returns:

```python
{
    "queries": list[str],   # maximum 3
    "entities": list[str],  # maximum 5
}
```

It does not create a sub-question DAG or allocate budgets. With an API key it
requests a small JSON plan from the configured LLM. Invalid JSON falls back to
the original question as the only query. Without an API key, a transparent
method-name heuristic keeps the demo offline.

### Researcher

The Researcher always executes the core retrieval pipeline directly:

1. BM25 retrieval using `rank_bm25.BM25Okapi`.
2. Dense cosine retrieval over a NumPy embedding matrix.
3. Entity-graph retrieval.
4. Reciprocal Rank Fusion over the three rankings.
5. Cross-encoder reranking of at most 30 candidates.

There is no tool registry, retrieval toolkit, async task queue, vector-store
interface, provider factory, or dynamic fusion weighting.

### Verifier

The Verifier answers two questions: is the evidence sufficient, and if not,
what is missing? Insufficient evidence can return to the Researcher only once.
After that, the graph must continue to the Writer.

### Writer

The Writer sees only the current evidence. It drafts with `[E1]`, `[E2]`
references and states uncertainty when verification failed. A deterministic
validator removes nonexistent evidence IDs and any pre-rendered page citation
that is not backed by a real stored chunk. Valid IDs become
`[paper.pdf p.N]`.

## Retrieval

### Page-aware ingestion

PyMuPDF extracts each physical page independently. Character chunks are about
1,200 characters with about 150 characters of overlap. A chunk never crosses
a page boundary and stores only:

```python
{
    "chunk_id": str,
    "paper": str,
    "page": int,
    "text": str,
}
```

No corpus manifest, tokenizer fingerprint, header-frequency model, section
hierarchy, cross-page chunk, or cache-invalidation framework is involved.

### BM25 and dense indexes

BM25 tokens are persisted as a small JSON file. Dense embeddings are saved as
`dense.npy` and searched with cosine similarity. The configured Sentence
Transformer is downloaded and cached on first use. In an offline environment
without the model cache, a deterministic hashing encoder keeps tests and the
demo runnable; the log makes that fallback explicit.

### Lightweight GraphRAG

Lightweight GraphRAG using entity co-occurrence and one-hop neighborhood
expansion.

Each extracted entity is a NetworkX node whose `chunks` attribute contains the
supporting chunk IDs. Entities in the same chunk are connected. Retrieval
matches the Planner's entities, expands one hop, collects supporting chunks,
and ranks them by entity-hit score. It does not perform community detection,
global summaries, entity resolution, graph embeddings, or multi-hop search.

### Fusion and reranking

RRF adds `1 / (60 + rank)` for every appearance of a chunk in a retriever
ranking. A single Sentence Transformers `CrossEncoder` then reranks the first
30 fused candidates. When that model is unavailable offline, a logged lexical
scorer is used so the full architecture can still be demonstrated.

## Install and run

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run scholar-agent ingest tests/fixtures/papers
uv run scholar-agent index
uv run scholar-agent ask "Compare Self-RAG and CRAG"
```

The repository includes two tiny, synthetic two-page PDF excerpts for
deterministic tests. They are not redistributed full papers. To use your own
corpus, point `ingest` at a directory containing PDFs.

The interview shortcut runs the same fixed question:

```bash
uv run scholar-agent demo
```

Supported commands are intentionally limited to:

```text
scholar-agent ingest <pdf-directory>
scholar-agent index
scholar-agent ask "<question>"
scholar-agent demo
```

## Configuration

Configuration is a single environment-backed dataclass:

| Variable | Default |
|---|---|
| `SCHOLAR_AGENT_LLM_MODEL` | `deepseek-chat` |
| `SCHOLAR_AGENT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `SCHOLAR_AGENT_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `SCHOLAR_AGENT_TOP_K` | `20` |
| `SCHOLAR_AGENT_DATA_DIR` | `data` |

Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` to enable the optional LLM path.
Keys are read from the environment and are never logged. No key is required
for ingestion, indexing, tests, or the offline demonstration.

## Example

```text
[planner] generated 3 queries and 3 entities
[researcher] sparse=4 dense=4 graph=4
[fusion] 4 unique candidates
[reranker] selected 4 evidence chunks
[reranker] E1 CRAG.pdf p.2 score=5.354
[verifier] evidence sufficient
[writer] answer generated with 4 citations

The retrieved evidence supports this comparison:
- CRAG applies correction after initial retrieval ... [CRAG.pdf p.2]
- Self-RAG ... generate reflection tokens. [Self-RAG.pdf p.1]
```

Every displayed filename and page is copied from the evidence chunk. A draft
reference such as `[E99]` or `[Fake.pdf p.999]` is removed.

## Project layout

```text
src/scholar_agent/
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   ├── verifier.py
│   └── writer.py
├── citations.py
├── cli.py
├── config.py
├── graph_store.py
├── indexes.py
├── ingest.py
├── llm.py
├── models.py
├── reranker.py
├── retrieval.py
└── workflow.py
```

## Tests and quality

The 18 deterministic tests cover physical page provenance, BM25, dense and
graph retrieval, RRF, reranking, Planner bounds and fallback, verification,
the one-retry limit, complete LangGraph execution, evidence-only writing,
false-citation removal, real filename/page rendering, and the CLI surface.

```bash
uv run pytest -q
uv run ruff check .
make quality
```

Provider calls are optional. Any future provider-dependent test belongs behind
the `live` pytest marker so the default suite stays deterministic and free.

## Limitations

- Entity extraction is regex-based and intentionally has no resolution stage.
- One-hop co-occurrence graphs are useful for local connections, not corpus-wide
  synthesis or deep relationship reasoning.
- Hash embeddings and lexical reranking are offline fallbacks, not substitutes
  for the configured neural models in a quality evaluation.
- The deterministic Writer summarizes retrieved sentences; nuanced synthesis
  benefits from a configured LLM.
- Citation validation proves provenance, not semantic entailment of every word.
- Indexes are rebuilt as a unit and assume a laptop-scale interview corpus.
