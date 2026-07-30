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
  │  queries + explicit targets + shared facets
  ▼
Researcher
  ├── BM25 sparse retrieval
  ├── Sentence Transformer dense retrieval
  ├── Lightweight GraphRAG
  ├── Reciprocal Rank Fusion
  └── Per-query candidates + reranking + target balance
  ▼
Verifier
  ├── complete ────────────────────────────────┐
  └── partial/insufficient → Researcher once  │
                                               ▼
Writer → deterministic citation validation → answer
```

The compiled graph has exactly four nodes:

```text
planner → researcher ─┬→ verifier ─┬→ writer → END
                      │      ▲      │
                      └──────┴──────┘  (abstain or retry once)
```

## The four agents

### Planner

The Planner receives the original question and returns:

```python
{
    "queries": list[str],   # maximum 3
    "entities": list[str],  # maximum 5
    "targets": list[str],   # maximum 3
    "facets": list[str],    # target-level coverage
    "output_language": str,
}
```

Only method names written in the question become targets; open-ended discovery
keeps `targets=[]` and uses question-level facets. It does not create a
sub-question DAG or allocate budgets. Invalid JSON falls back to a method-name
heuristic. Deterministic behavior is explicit through `--offline`.

### Researcher

The Researcher always executes the core retrieval pipeline directly:

1. Per-query BM25, dense cosine, and entity-graph retrieval.
2. Eight candidates retained from each query/retriever route.
3. Reciprocal Rank Fusion across the independent rankings.
4. Up to 30 fused candidates retained for neural scoring.
5. Multi-query cross-encoder reranking of at most 30 candidates.
6. Relevance filtering, page deduplication, and named-target balance.

There is no tool registry, retrieval toolkit, async task queue, vector-store
interface, provider factory, or dynamic fusion weighting.

### Verifier

The Verifier maps evidence IDs to target-level facets, or global facets when
there are no explicit targets. Incomplete evidence can trigger one retry;
an unchanged evidence set skips the redundant second verification.

### Writer

The Writer sees only verifier-approved evidence IDs. It answers covered facets,
lists missing evidence for partial results, and abstains without citations when
evidence is insufficient. Valid IDs become `[paper.pdf p.N]`.

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
The first neural run may download and load model snapshots; report that cold
start separately from subsequent warm-query latency in benchmarks and demos.

```bash
uv sync
uv run scholar-agent ingest tests/fixtures/papers
uv run scholar-agent index
uv run scholar-agent ask "Compare Self-RAG and CRAG"
uv run scholar-agent ask "Compare Self-RAG and CRAG" --offline
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
scholar-agent ask "<question>" [--offline]
scholar-agent demo [--offline]
```

## Configuration

Configuration is a single environment-backed dataclass:

| Variable | Default |
|---|---|
| `SCHOLAR_AGENT_LLM_MODEL` | `deepseek-chat` |
| `SCHOLAR_AGENT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `SCHOLAR_AGENT_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `SCHOLAR_AGENT_MIN_RERANK_SCORE` | `-1.0` |
| `SCHOLAR_AGENT_TOP_K` | `20` |
| `SCHOLAR_AGENT_DATA_DIR` | `data` |

Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` for online questions. Keys are never
logged. Without a key, use `--offline`; online mode never silently downgrades.

## Example

```text
[planner] queries=3 targets=2 facets=3 language=English
[researcher] sparse=4 dense=4 graph=4
[fusion] 4 unique candidates
[reranker] retained=4 rejected=0 threshold=-1.000
[reranker] selected 4 evidence chunks
[reranker] E1 CRAG.pdf p.2 score=5.354
[verifier] status=complete covered=6/6 missing=0
[writer] status=complete citations=4 sources=2

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

The 30 deterministic tests cover physical page provenance, all retrievers,
multi-query reranking, target identity, thresholds, coverage, retry bounds,
strict abstention, page citations, and the four-command CLI.

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
