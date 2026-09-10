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
   ├── requirement- and target-aware evidence selection
   └── Requirement–Evidence Blackboard
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

An optional bounded evidence-gap Controller can inspect the first Researcher
observation and choose up to two recovery actions in one round before writing.
Its E3 v5 paired evaluation found a positive but not statistically significant
quality signal at substantial latency cost, so it remains disabled by default
while selective invocation and latency optimization are evaluated.

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

# Optional single-round evidence recovery
export SCHOLAR_AGENT_RECOVERY_MODE=controller
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

The retrieval engine also exposes deterministic paper-local hybrid search and
an inclusive neighboring-chunk window within one paper. These primitives return
candidates only; follow-up policy remains outside them.

After selection, the Researcher assigns stable `E1`, `E2`, ... IDs and builds a
Requirement–Evidence Blackboard:

```python
evidence_board = {
    "R1": {"requirement": "Explain Self-RAG retrieval", "evidence_ids": ["E1", "E3"]},
    "R2": {"requirement": "Report unavailable results", "evidence_ids": []},
}
```

Each selected evidence item retains `chunk_id`, `paper`, `page`, `text`, and
`score`, and adds `id`, `paper_id` (the source filename), optional `title` and
`section`, `supports`, and public `requirement_scores`. A link requires the
requirement's rerank score to meet `SCHOLAR_AGENT_MIN_RERANK_SCORE`. Board links
do not require a literal target-name match, so aliases such as CRAG and Corrective
RAG do not hide relevant selected passages. Scores are raw cross-encoder relevance
scores, not probabilities or proof of support. One passage may link to several
requirements; requirements with no matches remain on the board with an empty
list. Each board entry orders its evidence by that requirement's score. This
step preserves selected chunks and their global citation IDs.

## Writer and citation validation

The Writer sees each requirement followed by its linked passages, with source
filename, physical page, and title/section when available. Shared passages keep
the same evidence ID across requirements. Selected passages without a requirement
match appear as additional evidence, so none of the selected context is lost.
The Writer checks the text for actual support and may cite any supplied passage
that supports the claim. It explicitly identifies unsupported requirements when
answering a partially supported question.

The Writer may cite only temporary IDs such as `[E1]`. An empty evidence set
produces a deterministic abstention without an LLM call. Building the board adds
no LLM calls. The prompt forbids introductory summaries and concluding
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
boundary. Every stored chunk has 0-based `chunk_index` and `page_chunk_index`
positions in addition to `chunk_id`, `paper`, `page`, and `text`. PDF metadata
titles are retained; when absent, the largest heading near the top of the first
page is used. Chunk records accept an optional `section` when supplied, but
section headings are not inferred.

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
| `SCHOLAR_AGENT_RECOVERY_MODE` | `none` |
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

The [Writer ablation guide](evals/WRITER_ABLATION.md) provides commands to compare
flat evidence against the Requirement–Evidence Blackboard using identical frozen
plans, passages and answer instructions, with resumable generation and blind review.
The [Controller ablation guide](evals/CONTROLLER_ABLATION.md) documents the paired
experiment between the Blackboard baseline and one bounded, observation-driven
recovery round.
The [Simple RAG comparison guide](evals/SIMPLE_RAG_COMPARISON.md) compares the
complete Controller-enabled system against a one-query BM25+Dense, RRF,
cross-encoder, fixed-budget pipeline.

The blinded 50-question evaluation pipeline supports two main findings from
separate experiments:

- **Adaptive retrieval reduced dense retrieval operations by 22.4%**, from 76
  to 59.
- **The complete Controller-enabled Scholar-Agent improved Strict Success by
  20.0 percentage points over Simple RAG**, from 74% to 94%.

### Current Controller pipeline vs Simple RAG (`controller_vs_simple_rag_v1`)

This paired experiment compares the quality-oriented current configuration—an
adaptive Planner, per-requirement retrieval and Blackboard, the assessment-first
Controller with one bounded recovery round, Writer, and Citation Validator—with
a Simple RAG pipeline using the original question, BM25 top-8 plus Dense top-8,
RRF, the same cross-encoder threshold, a fixed maximum of eight flat evidence
chunks, the same Writer policy, and the same Citation Validator.

All 50 questions used freshly generated runtime states on the updated benchmark.
Generation used `deepseek-v4-flash` at temperature zero, Writer order alternated
25/25, and frozen input, state, and prompt hashes were verified. Blind primary
review and independent cross-review of every paired disagreement were completed
before variant identities were revealed.

| Metric | Simple RAG | Current + Controller | Delta |
|---|---:|---:|---:|
| Strict Success | 74.0% (37/50) | **94.0% (47/50)** | **+20.0 pp** |
| Requirement Accuracy | 80.3% (57/71) | **98.6% (70/71)** | **+18.3 pp** |
| Citation Support | **100.0% (322/322)** | 99.8% (458/459) | -0.2 pp |
| Initial Retrieval Recall | 49.0% | **76.5%** | **+27.5 pp** |
| Selected Evidence Recall | 35.3% | **62.7%** | **+27.5 pp** |
| Average staged latency | 12.06s | 34.39s | +22.33s |
| Average LLM calls | 0.80 | 2.88 | +2.08 |
| Average retrieval operations | 2.00 | 5.06 | +3.06 |

There were 12 Strict repairs and 2 regressions (exact two-sided McNemar
p=0.0129), plus 13 requirement repairs and no requirement regressions
(p=0.000244). The observed improvements are statistically significant at the
0.05 level within this benchmark, while the 50-question sample still limits
claims about broader generalization.

Four repairs—Q021, Q026, Q027, and Q034—completed the full causal chain from a
missing assessment to an executed action, added and cited evidence, and a
repaired answer. The remaining gains came mainly from requirement decomposition,
targeted initial retrieval, evidence selection, and Blackboard/Writer use; this
is therefore a full-system comparison, not a Controller-only ablation. The
Controller triggered on 34/50 questions, executed 41 actions, and added evidence
on 21. Its 55 `sufficient`, 41 `missing`, and 4 `unresolved` assessments included
one False-Sufficiency Case, Q005/R1.

The result supports the Controller-enabled pipeline as the quality-oriented
main path, with Simple RAG as a lower-latency fallback. Because measured latency
was 2.85× higher and only 4/41 actions caused an observed requirement repair,
the next work should be selective Controller invocation and latency reduction.
Detailed local artifacts are in `evals/runs/controller_vs_simple_rag_v1/`,
including the summary, full analysis, blind adjudication, raw traces, hashes,
and generation provenance.

### Assessment-first Controller (`controller_e3_v5`)

E3 v5 compares the Blackboard pipeline with and without one assessment-first
Controller call. All 50 questions used newly generated Planner and initial
retrieval observations after the Q014 and Q025 benchmark clarification. Within
each pair, both variants started from the same frozen plan, retrieval results,
rerank candidates, selected evidence, and Writer policy. Writer order alternated
25/25, and the review, two independent crosschecks, and final adjudication were
completed before variant identities were revealed.

| Metric | Blackboard baseline | Controller | Delta |
|---|---:|---:|---:|
| Strict Success | 88.0% (44/50) | 96.0% (48/50) | +8.0 pp |
| Requirement Accuracy | 94.4% (67/71) | 97.2% (69/71) | +2.8 pp |
| Citation Support | 99.7% (396/397) | 100.0% (467/467) | +0.3 pp |
| Initial Retrieval Recall | 74.5% (38/51) | 74.5% (38/51) | 0.0 pp |
| Selected Evidence Recall | 62.7% (32/51) | 62.7% (32/51) | 0.0 pp |
| Measured post-research latency | 14.40s | 28.52s | +14.12s |
| Average LLM calls | 0.86 | 1.90 | +1.04 |
| Average retrieval operations | 3.64 | 5.18 | +1.54 |

The paired run had four Strict Success repairs and no regressions. The exact
two-sided McNemar p-value is 0.125, so the observed +8-point result is not a
statistically significant improvement at the 0.05 level. Requirement outcomes
had two repairs and no regressions (p=0.500). Manual trace attribution found
that Q025 and Q032 completed the full causal chain from missing assessment to
new evidence, changed Writer context, and repaired answer. Q002 and Q018 were
Strict-only changes caused by Writer or citation variation rather than the
recovery target.

The Controller stored 100 valid assessments for 103 Planner requirements:
55 `sufficient`, 39 `missing`, and 6 `unresolved`. All 55 sufficient and all 6
unresolved assessments correctly stored no action. Two sufficient assessments
ended with an incorrect requirement: Q005 was a coverage-assessment failure,
while Q019 was a Writer utilization failure. The Controller triggered on 36/50
questions and executed 39 actions; 21/39 added evidence, but only two repaired a
requirement. Gold-page Retrieval Recovery Rate was 0/13 because both genuine
repairs used useful passages on non-gold pages, illustrating that gold pages are
non-exhaustive diagnostic labels rather than complete evidence-sufficiency
labels.

The result supports retaining the Controller implementation but not enabling it
universally from this 50-question sample. The next step is selective invocation
and latency reduction, followed by a larger confirmatory paired run—not more
retrieval tools or question-specific prompt patches. Detailed local artifacts
are in `evals/runs/controller_e3_v5/`, including `summary.md`, `analysis.md`,
`adjudication.json`, raw results, traces, labels, and input/prompt hashes.

### Adaptive retrieval (`adaptive_v2`)

The benchmark calls the Planner once per question and reuses that exact
sanitized plan for both `fixed_hybrid` and `adaptive`. Only retrieval execution
differs: fixed mode overrides every strategy to hybrid, while adaptive mode
uses the planned strategy. Blinded scoring reports Strict Success, Requirement
Accuracy, Citation Support, average latency, and average LLM calls. Each trace
records the shared plan and the executed query, strategy, and depth per
requirement. See [evals/README.md](evals/README.md).

New runs also evaluate each gold requirement through **Retrieval Recall → Rerank
Recall → Selected Evidence Recall → Answer Requirement Accuracy**. The first
three metrics use the existing `gold_pages` automatically; the last reuses the
answer review score. Summaries include both aggregate recall and a per-requirement
stage table. Use a new run ID such as `adaptive_v4` to collect stage traces and
the Writer's evidence board without mixing results from earlier Writer prompts.

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
