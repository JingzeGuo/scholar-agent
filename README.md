# ScholarAgent

A compact agentic RAG workflow for evidence-grounded academic research.

ScholarAgent answers questions over a small collection of academic PDFs while
keeping the full retrieval and grounding path easy to inspect. It combines
lexical and semantic search, reranks the fused candidates, checks whether the
evidence covers the question, and renders only validated physical-page
citations.

## Problem

Academic question answering needs more than a plausible response. The system
must retrieve evidence for each requested method and aspect, detect incomplete
support, restrict generation to approved passages, and preserve page-level
provenance. ScholarAgent implements that path without a vector database,
dynamic routing, or an open-ended tool loop.

## Architecture

```text
Question
   ↓
Planner
   ↓
Researcher
   ├── BM25
   ├── Dense retrieval
   ├── Reciprocal Rank Fusion
   └── Cross-encoder reranking
   ↓
Verifier
   ├── complete ───────────────────────────────┐
   ├── partial + corrective queries → Researcher once → Verifier
   └── insufficient ──────────────────────────┤
                                               ↓
                                             Writer
   ↓
Deterministic physical-page citation validation
   ↓
Answer
```

LangGraph connects four workflow nodes:

- Planner: LLM-based planning node.
- Researcher: deterministic retrieval, fusion, reranking, and
  evidence-selection node.
- Verifier: LLM-based evidence-coverage node.
- Writer: LLM-based grounded-answer node.

The Researcher is a deterministic workflow node, not an autonomous LLM agent.
Each pass through the bounded retry loop runs one batch of corrective retrievals
requested by the Verifier.

## Planner

The Planner decomposes the question into this compact plan:

```python
{
    "requirements": [             # 1–5 independent coverage checks
        {
            "id": str,            # assigned by code: R1, R2, ...
            "description": str,
            "targets": list[str], # 0–3 methods or papers named in the question
            "query": str,         # evidence query dedicated to this requirement
        },
    ],
}
```

Each requirement owns its retrieval query and is verified independently, so
asymmetric questions do not create unrequested target/aspect combinations and
different aspects of the same target retain separate retrieval signals. Targets
must be explicitly present in the question; open-ended requirements use
`targets=[]`. Queries preserve names and constraints but retrieve evidence
instead of proposing an answer. Retrieval plans and final answers are always in
English.

## Hybrid retrieval

Every query always follows the same readable path:

```text
BM25(query) ───┐
               ├── RRF ──→ at most 30 candidates ──→ cross-encoder
Dense(query) ──┘
```

For multiple queries, each BM25 and dense result remains an independent ranking
of at most eight candidates before fusion. Up to four reranking candidates are
reserved per requirement query before the 30-candidate pool is filled by global
RRF rank. Dense queries are encoded together in one batch.

BM25 supplies exact lexical matching for titles, acronyms, and technical terms.
Dense retrieval uses normalized Sentence Transformer embeddings and cosine
similarity for semantic matches.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines rankings without learned or dynamic
weights. Each appearance contributes:

```text
1 / (60 + rank)
```

A chunk found by both BM25 and dense retrieval therefore receives more support
than a chunk found in only one ranking. The fused list is capped at 30 candidates.

## Cross-encoder reranking and evidence selection

The cross-encoder scores each query/chunk pair, and each chunk keeps its best
query score for global ranking while retaining its per-requirement scores for
coverage selection. Candidates below the configured relevance threshold are
removed. The remaining evidence is selected with explicit, deterministic
bounds:

- normally at most eight evidence chunks, expanding up to fifteen when distinct
  requirement or comparison-target coverage needs more slots;
- after requirement coverage, at most four chunks per paper and no duplicate
  physical page when filling diversity slots;
- one relevant slot per requirement when matching evidence exists;
- comparison requirements receive evidence for each named target when possible;
- up to two early slots per explicitly named target when matching evidence exists.

During corrective retrieval, useful new evidence is merged with the existing
selection. If the retry produces the same evidence IDs, the workflow terminates
without repeating verification.

## Verifier

The Verifier checks every atomic requirement against supplied evidence IDs. It
rejects unknown IDs, evidence for the wrong named target, and unsupported
coverage. Its result is one of:

- `complete`: every requirement has direct support;
- `partial`: some requested coverage is supported;
- `insufficient`: none of the required coverage is supported.

For missing coverage it may return a batch containing one concise corrective
query per missing requirement. The default retry budget is one, so the workflow
cannot become an unrestricted loop. If no useful query exists or the budget is
exhausted, processing continues to the Writer.

## Writer and citation validation

The Writer sees only evidence approved by the Verifier. It cites temporary IDs
such as `[E1]`; citing an unknown or unapproved ID is an error. Partial answers
must name the missing coverage. Insufficient evidence produces a concise
abstention with no citations.

After writing, deterministic validation converts known IDs to citations copied
from stored metadata:

```text
[E1] → [Self-RAG.pdf p.1]
```

Invented IDs and fabricated page citations are removed. This establishes
provenance to a retrieved physical page; it does not prove that every generated
claim is semantically true.

## Page-aware ingestion and indexes

PyMuPDF extracts each physical page independently. Character chunks are about
1,200 characters with about 150 characters of overlap and never cross a page
boundary. Every stored chunk has `chunk_id`, `paper`, `page`, and `text`.

Indexing writes a small BM25 token file plus a NumPy dense-embedding matrix and
metadata. Both indexes store an ordered corpus fingerprint and refuse to load
after the chunks change. The configured local embedding and reranker models
download on first use and fail explicitly if unavailable.

## Installation

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` before asking a question. Ingestion
and indexing do not require a paid API.

## CLI

```bash
uv run scholar-agent ingest tests/fixtures/papers
uv run scholar-agent index
uv run scholar-agent ask "Compare Self-RAG and CRAG"
```

The public CLI intentionally contains only `ingest`, `index`, and `ask`.

Configuration uses environment variables:

| Variable | Default |
|---|---|
| `SCHOLAR_AGENT_LLM_MODEL` | provider default |
| `SCHOLAR_AGENT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `SCHOLAR_AGENT_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `SCHOLAR_AGENT_MIN_RERANK_SCORE` | `-1.0` |
| `SCHOLAR_AGENT_MAX_RETRIES` | `1` |
| `SCHOLAR_AGENT_DATA_DIR` | `data` |

## Tests

The default suite is deterministic and makes no paid provider calls. It covers
page provenance, BM25 and dense retrieval, batched query encoding, RRF,
reranking, evidence selection, atomic-requirement verification, retry bounds,
strict abstention, and physical-page citation validation.

```bash
uv run ruff check .
uv run pytest -q
make quality
```

Provider-dependent tests belong behind the `live` pytest marker.

## Limitations

- The corpus and NumPy indexes are intended for laptop-scale use.
- Retrieval is always BM25 plus dense search rather than adaptive routing.
- The Verifier relies on an LLM and is not a formal entailment checker.
- Citation validation establishes provenance, not semantic truth.
- Indexes are rebuilt as a unit rather than updated incrementally.
- Local embedding and reranker models require a download on first use.
