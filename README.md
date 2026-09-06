# ScholarAgent

A compact agentic RAG workflow for evidence-grounded academic research.

ScholarAgent answers questions over a small collection of academic PDFs while
keeping the full retrieval and grounding path easy to inspect. It combines
lexical and semantic search, reranks the fused candidates, optionally checks
whether the evidence covers the question, and renders only validated
physical-page citations.

On a 50-question English benchmark, the default workflow improved strict
answer success from 46% to 62% (+16 percentage points) and requirement
accuracy from 73.2% to 90.1% (+16.9 points) over a hybrid RAG baseline, while
achieving 87.8% citation support.

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
Writer (all evidence)
   ↓
Answer Verifier
   ├── pass ────────────────────────┐
   └── fail → Writer repair once → recheck
                                    ↓
Deterministic physical-page citation validation
   ↓
Answer

Optional soft-coverage path:

Researcher → Coverage Analyzer → corrective retrieval at most once → Writer
```

LangGraph connects the workflow nodes:

- Planner: LLM-based planning node.
- Researcher: deterministic retrieval, fusion, reranking, and
  evidence-selection node.
- Coverage Analyzer: optional LLM-based evidence annotation and
  corrective-query node; disabled by default.
- Writer: LLM-based grounded-answer node.
- Answer Verifier: LLM-based final requirement and grounding check.

The Researcher is a deterministic workflow node, not an autonomous LLM agent.
When soft coverage is explicitly enabled, the bounded retry loop runs one batch
of corrective retrievals requested by the Coverage Analyzer.

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
English. Comparisons are synthesized from separately supported target facts;
the Planner does not require a source that already states the comparison. If
the model returns no usable plan, the original question becomes one conservative
target-free requirement instead of terminating the workflow.

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

When a target name is not repeated verbatim in any candidate, the top semantic
candidates still reach the Coverage Analyzer instead of being discarded as a group.

During corrective retrieval, useful new evidence is merged with the existing
selection. If the retry produces the same evidence IDs, the workflow terminates
without repeating verification.

## Coverage Analyzer

The Coverage Analyzer checks every atomic requirement against supplied evidence IDs. It
rejects unknown IDs, evidence explicitly belonging to a different named target,
and unsupported coverage annotations. It can combine separately supported facts across
papers and does not require every evidence passage to repeat the target name.
Each requirement is annotated as supported, uncertain, or missing; the aggregate
status remains:

- `complete`: every requirement has direct support;
- `partial`: some requested coverage is supported;
- `insufficient`: none of the required coverage is supported.

For uncertain or missing coverage it may return one concise corrective query per
requirement. The default retry budget is one, so the workflow cannot become an
unrestricted loop. Its annotations never remove evidence or force an abstention.
If no useful query exists or the budget is exhausted, processing continues to
the Writer.

## Writer and citation validation

The Writer sees every selected evidence chunk and treats coverage annotations as
advice. It cites temporary IDs such as `[E1]`. Only an actually empty evidence
set causes an immediate abstention.

The Answer Verifier then checks planned requirement coverage, uncited and
unsupported claims, citation support, and incorrect claims that evidence is
missing. It sees no benchmark answer keys or gold pages. A failed answer receives
one constrained repair and one final check; it cannot enter an unrestricted
loop. Malformed verifier output is recorded without discarding the answer.

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

## Evaluation

The small, resume-oriented benchmark compares Simple RAG with the full
Scholar-Agent workflow on 50 English questions. See
[`evals/README.md`](evals/README.md) for the run, blinded review, and scoring
workflow, including the `none` versus `soft` Coverage Analyzer ablation.
Evaluation stays outside the production CLI and the default test suite never
calls DeepSeek.

The benchmark uses 10,726 page-aware corpus chunks, `deepseek-v4-flash` with
temperature zero, and variant-blinded external LLM review against extracted
cited and gold PDF pages. The default No Coverage workflow produced:

| Metric | Simple RAG | Scholar-Agent | Delta |
|---|---:|---:|---:|
| Strict Success | 46.0% | 62.0% | +16.0 pp |
| Requirement Accuracy | 73.2% | 90.1% | +16.9 pp |
| Citation Support | 89.6% | 87.8% | -1.8 pp |
| Average latency | 8.72s | 51.81s | +43.09s |
| Average LLM calls | 1.00 | 3.36 | +2.36 |

The agent workflow substantially improved end-to-end success and requirement
handling, but did not improve citation support. Citation grounding is therefore
the clearest remaining quality bottleneck.

The optional Soft Coverage path raised Full-system requirement accuracy from
90.1% to 94.4%, but Strict Success remained 62% and citation support remained
87.8%. It also increased average latency from 51.81s to 59.76s and LLM calls
from 3.36 to 4.48. Based on this ablation, No Coverage is the default and Soft
Coverage remains available for experiments that prioritize requirement
completeness.

These are descriptive results from one fixed-model run scored by one external
LLM judge. The benchmark does not include repeated runs, confidence intervals,
or a claim of statistical significance.

## Limitations

- The corpus and NumPy indexes are intended for laptop-scale use.
- Retrieval is always BM25 plus dense search rather than adaptive routing.
- Coverage annotations rely on an LLM and are not formal entailment checks.
- Citation validation establishes provenance, not semantic truth.
- Indexes are rebuilt as a unit rather than updated incrementally.
- Local embedding and reranker models require a download on first use.
