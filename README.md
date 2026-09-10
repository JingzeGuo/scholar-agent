# Scholar-Agent

Scholar-Agent is an agentic retrieval-augmented generation system for answering
questions over a local collection of academic PDFs. It decomposes a question
into evidence requirements, retrieves and reranks page-grounded passages,
checks whether each requirement has enough support, performs bounded recovery
when evidence is missing, and writes an answer with validated physical-page
citations.

The project is designed around a simple principle: retrieval is not treated as
complete when relevant text is merely found; the system explicitly assesses
whether each requirement has sufficient evidence before generation.

## Architecture

```text
Question
   ↓
Planner
   ↓
Researcher
   ↓
Assessment-first Controller
   ├── sufficient ─────────────────────→ Writer
   ├── missing ─→ bounded Recovery ───→ Writer
   └── unresolved ─────────────────────→ Writer with an explicit evidence gap
   ↓
Citation Validator
   ↓
Answer
```

LangGraph connects the Planner, Researcher, Controller, Recovery, Writer, and
Citation Validator through one shared state. The state contains the original
question, atomic requirements, selected evidence, the Requirement–Evidence
Blackboard, Controller assessments and actions, recovery traces, and the final
answer.

## Main components

### Page-aware ingestion and indexes

PDFs are extracted one physical page at a time with PyMuPDF. Chunks never cross
page boundaries, so every passage retains stable source metadata:

```python
{
    "chunk_id": "...",
    "paper": "paper.pdf",
    "page": 4,
    "chunk_index": 12,
    "page_chunk_index": 1,
    "text": "...",
}
```

Scholar-Agent builds both a BM25 index and a dense-vector index. Each index
stores a corpus fingerprint and refuses to load when it no longer matches the
processed chunks.

### Planner

The Planner turns a question into one to five atomic evidence requirements.
Each requirement contains a focused query, relevant target names, a retrieval
strategy, and a bounded retrieval depth:

```python
{
    "id": "R1",
    "description": "Explain how Self-RAG controls retrieval.",
    "targets": ["Self-RAG"],
    "query": "Self-RAG reflection tokens retrieval control",
    "retrieval_strategy": "hybrid",  # bm25 | dense | hybrid
    "top_k": 8,
}
```

Planner output is sanitized before execution. Unknown strategies fall back to
hybrid retrieval, depths are clamped to the supported range, empty queries are
replaced, duplicates are removed, and malformed plans receive a safe fallback
requirement.

### Researcher and Requirement–Evidence Blackboard

The Researcher executes the route selected for each requirement:

```text
bm25:   BM25(query) ───────────────────────→ rerank
dense:  Dense(query) ──────────────────────→ rerank
hybrid: BM25(query) ─┐
                     ├→ reciprocal rank fusion → rerank
        Dense(query) ┘
```

All routes use the same cross-encoder reranker. Candidate selection reserves
local coverage for individual requirements before filling the shared candidate
pool. Evidence allocation balances requirement coverage, named targets, source
diversity, and physical-page diversity.

Selected passages receive stable `E1`, `E2`, ... identifiers and are organized
on a Blackboard:

```python
evidence_board = {
    "R1": {
        "requirement": "Explain Self-RAG retrieval control.",
        "evidence_ids": ["E1", "E3"],
        "candidate_papers": [...],
    },
    "R2": {
        "requirement": "Explain Corrective RAG document repair.",
        "evidence_ids": ["E2"],
        "candidate_papers": [...],
    },
}
```

One passage may support multiple requirements. Requirements with no selected
support remain visible with an empty evidence list instead of disappearing.

### Assessment-first Controller and bounded Recovery

The Controller inspects the first complete Researcher observation once per
question. It returns one assessment for every planned requirement:

```python
{
    "requirement_id": "R2",
    "status": "missing",  # sufficient | missing | unresolved
    "covered": ["document refinement is identified"],
    "missing": ["the decompose-filter-recompose procedure"],
    "action": {
        "tool": "search_within_paper",
        "candidate_id": "P1",
        "query": "decompose filter recompose document refinement",
    },
}
```

- `sufficient` means all requested aspects have direct support.
- `missing` means a displayed bounded action can seek the absent evidence.
- `unresolved` means evidence is missing and no valid bounded action is
  available.

Every assessment is stored in the trace, including assessments that do not
produce an action. Recovery is limited to one round and at most two actions:

- `search_within_paper` searches a paper already identified by the Researcher;
- `expand_neighbors` inspects adjacent chunks around selected evidence;
- `increase_depth` combines the original requirement with a new
  Controller-generated query, raises `top_k` to `MAX_TOP_K`, and reruns the
  requirement's full retrieval route.

Actions use stable candidate-paper selectors and chunk IDs. Retrieved
candidates still pass through reranking and evidence selection before they can
enter the Writer context.

### Writer and Citation Validator

The Writer receives each requirement together with its linked evidence. It may
use only supplied passages, must cite factual claims with evidence IDs such as
`[E1]`, and explicitly identifies requirements that remain unsupported. An
empty evidence set produces a deterministic abstention without a Writer call.

The Citation Validator replaces evidence IDs with physical-page citations:

```text
[E1] → [Self-RAG.pdf p.1]
```

Unknown evidence IDs and fabricated page references are removed. This gives
every retained citation a deterministic provenance path from answer to chunk to
physical PDF page.

## Evaluation

The two most recent evaluations use the updated 50-question academic-RAG
benchmark, `deepseek-v4-flash` at temperature zero, alternating paired Writer
order, frozen runtime hashes, and variant-blind review with cross-review and
adjudication before unblinding.

The experiments answer different questions and use independently generated
paired runs, so their absolute percentages should not be compared across rows.

### Complete Scholar-Agent vs Simple RAG

This experiment compares the complete architecture with a Simple RAG pipeline:

```text
Original question
→ BM25 top-8 + Dense top-8
→ reciprocal rank fusion
→ cross-encoder rerank
→ fixed evidence budget
→ Writer
→ Citation Validator
```

| Quality metric | Simple RAG | Scholar-Agent | Improvement |
|---|---:|---:|---:|
| Strict Success | 74.0% (37/50) | **94.0% (47/50)** | **+20.0 pp** |
| Requirement Accuracy | 80.3% (57/71) | **98.6% (70/71)** | **+18.3 pp** |
| Initial Retrieval Recall | 49.0% | **76.5%** | **+27.5 pp** |
| Selected Evidence Recall | 35.3% | **62.7%** | **+27.5 pp** |

The paired run produced 12 Strict Success repairs and 2 regressions. The exact
two-sided McNemar p-value is `0.0129`. Requirement outcomes produced 13 repairs
and no regressions, with `p=0.000244`. Both improvements are statistically
significant within this benchmark.

Trace inspection linked four repairs to the complete Controller recovery chain:

```text
missing assessment
→ recovery action
→ new evidence
→ changed Writer context
→ repaired answer
```

The remaining gains came from atomic requirement decomposition, targeted
initial retrieval, requirement-aware evidence selection, and Blackboard-guided
writing. The result supports the architecture as a whole rather than assigning
all improvement to a single component.

### Controller contribution: full system vs no-Controller ablation

The second experiment freezes the same initial Planner output, retrieval
results, rerank candidates, selected evidence, and Blackboard for each pair.
One variant writes directly from that state; the other runs the assessment-first
Controller and bounded Recovery before using the same Writer policy.

| Quality metric | Without Controller | With Controller | Observed improvement |
|---|---:|---:|---:|
| Strict Success | 88.0% (44/50) | **96.0% (48/50)** | **+8.0 pp** |
| Requirement Accuracy | 94.4% (67/71) | **97.2% (69/71)** | **+2.8 pp** |
| Citation Support | 99.7% (396/397) | **100.0% (467/467)** | **+0.3 pp** |

The Controller produced four Strict Success repairs and no regressions. Its
exact McNemar p-value was `0.125`, so this isolated 50-question result is a
positive paired signal rather than a statistically proven standalone gain.
Manual attribution found two requirement repairs that followed the full
assessment-to-recovery-to-answer chain.

Taken together, the experiments support the current design: the complete
Scholar-Agent pipeline significantly improves over Simple RAG, and removing
the Controller weakens the same system in the paired ablation. The architecture
is therefore justified by both end-to-end quality and component-level evidence.

## Installation and usage

Requirements:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- a DeepSeek or OpenAI-compatible API key

Install the project:

```bash
git clone <repository-url>
cd scholar-agent
uv sync
```

Configure the current architecture:

```bash
export DEEPSEEK_API_KEY=...
export SCHOLAR_AGENT_LLM_MODEL=deepseek-chat
export SCHOLAR_AGENT_RECOVERY_MODE=controller
```

An OpenAI key can be used instead:

```bash
export OPENAI_API_KEY=...
export SCHOLAR_AGENT_LLM_MODEL=gpt-4.1-mini
```

Ingest a directory of PDFs, build the indexes, and ask a question:

```bash
uv run scholar-agent ingest path/to/papers
uv run scholar-agent index
uv run scholar-agent ask "Compare how Self-RAG and Corrective RAG handle retrieval quality."
```

The same workflow is available from Python:

```python
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient
from scholar_agent.retrieval import RetrievalEngine
from scholar_agent.workflow import run_question

settings = Settings.from_env()
engine = RetrievalEngine.load(settings)
llm = LLMClient.from_env(settings)

state = run_question(
    "How does Sentence-BERT make semantic search efficient?",
    engine,
    settings,
    llm,
)
print(state["answer"])
```

Important environment variables:

| Variable | Purpose | Example |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API authentication | `sk-...` |
| `OPENAI_API_KEY` | OpenAI API authentication | `sk-...` |
| `SCHOLAR_AGENT_LLM_MODEL` | Planner, Controller, and Writer model | `deepseek-chat` |
| `SCHOLAR_AGENT_EMBEDDING_MODEL` | Dense retrieval model | `sentence-transformers/all-MiniLM-L6-v2` |
| `SCHOLAR_AGENT_RERANKER_MODEL` | Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `SCHOLAR_AGENT_MIN_RERANK_SCORE` | Evidence retention threshold | `-1.0` |
| `SCHOLAR_AGENT_RETRIEVAL_MODE` | Per-requirement retrieval policy | `adaptive` |
| `SCHOLAR_AGENT_RECOVERY_MODE` | Assessment and bounded recovery path | `controller` |
| `SCHOLAR_AGENT_DATA_DIR` | Processed corpus and index directory | `data` |

Run the quality suite with:

```bash
make quality
```

## Project limitations

- Scholar-Agent answers from the indexed local PDF collection; it does not
  automatically search the open web for missing sources.
- Retrieval quality is limited by corpus coverage, PDF extraction quality,
  embedding quality, and cross-encoder ranking.
- The Controller can misjudge evidence sufficiency or choose an unproductive
  recovery action. Its decisions remain bounded and fully traceable, but they
  are not guaranteed to repair every evidence gap.
- Citation validation proves that a citation maps to a retrieved physical page;
  it does not perform semantic entailment checking for every claim.
- Page-level gold annotations in the evaluation benchmark are non-exhaustive
  diagnostic signals, not complete definitions of evidence sufficiency.
- The reported experiments contain 50 questions from one academic corpus.
  Broader generalization requires larger and independently sampled benchmarks.
- Indexes are rebuilt as a unit rather than updated incrementally, and the
  current NumPy-backed dense index is intended for laptop-scale collections.
- Embedding and reranker models may need to be downloaded on first use.

## License

MIT
