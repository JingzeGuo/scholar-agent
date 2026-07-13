# Cache design and invalidation

ScholarAgent uses caching only for pure, deterministic offline work. Mutable
workflow state (evidence ledgers, live run traces, provider responses that may
contain secrets) is **not** cached across runs.

## Layers

| Layer | Location | What is cached | Key inputs | Invalidation |
|---|---|---|---|---|
| Config process cache | `get_config` (`lru_cache`) | Validated `AppConfig` | Config path string | Process restart / path change |
| Tokenizer | `ingestion.tokens` (`lru_cache`) | Tiktoken encodings | Encoding name | Process restart |
| Model weights | `.cache/huggingface/` (gitignored) | Sentence-transformers / HF hub | Env `SCHOLAR_MODEL_CACHE` / `HF_HOME` | Delete directory |
| Dense / BM25 indexes | `data/indexes/` (gitignored) | Vectors + sparse postings | Corpus fingerprint | Rebuild when processed chunks change |
| Knowledge graph artifacts | `data/processed/{entities,relations,knowledge_graph,graph_meta}.*` | Extracted graph | Corpus fingerprint, graph/extraction schema, full-vs-partial build, resolver options | Identity mismatch or `graph build --force` |
| Extraction disk cache | `data/processed/.cache/extraction/` (gitignored) | Per-chunk relation JSON | chunk_id, content_hash, pages, schema `extract-v1` | Schema bump, content hash change, corrupt entry, `clear()` |
| RAGAS metric cache | `data/evaluation/.cache/ragas_metrics/` (gitignored) | Allowlisted numeric faithfulness/relevancy scores only | Hash of question, answer, contexts, reference, provider/model/embedder identity, schema `ragas-metrics-v2` | Schema/model/input change, corrupt or invalid value |
| PDF page counts | in-memory on `CitationValidator` | Page count per PDF path | Path string | Process lifetime |
| Demo saved runs | `data/demo/runs/*.json` (committed) | Offline interview sessions | Manual regeneration | Rebuild via `scripts/precompute_demo_runs.py` |

## Disk cache module

Implementation: `scholar_agent.storage.cache.DiskCache`.

Properties:

- Deterministic SHA-256 keys over namespace + schema version + canonical JSON payload.
- Atomic writes (`tempfile` + `os.replace`).
- Full 64-character SHA-256 digests (not truncated application IDs).
- Corrupt JSON or an embedded-key mismatch → miss + delete file; stats track `corruptions`.
- Schema mismatch → miss + delete; stats track `invalidations`.
- Observability: `hits` / `misses` / `stores` / `corruptions` / `invalidations`.
- No secrets: only JSON-serializable pure data.

## Invalidation policy

1. **Content change:** include content hashes in the key (chunk `content_hash`).
2. **Logic change:** bump `EXTRACTION_CACHE_SCHEMA` / `CACHE_SCHEMA_VERSION`.
3. **Index rebuild:** fingerprint mismatch refuses silent reuse of BM25/dense loads.
4. **Graph rebuild:** omitting `--force` reuses artifacts only when
   `graph_meta.json` matches the canonical corpus, graph/extraction schemas, and
   build options. Missing, stale, or partial metadata forces rebuild; `--force`
   always recomputes (while extraction-cache entries may still be reused when
   their content/schema identity matches).
5. **Corruption:** never return partial/invalid values; treat as miss.
6. **Paid evaluation:** model/provider/embedder identity and metric names are key
   inputs; changing any of them forces a RAGAS miss.
7. **Partial RAGAS success:** faithfulness and answer relevancy are validated
   independently. A valid metric is cached even when its peer fails; failure
   codes and exception class names are reported but are not persisted in the
   numeric cache record.

## What must not be cached

- Live LLM completions and raw RAGAS/provider responses that may include
  provider reasoning fields. The RAGAS layer stores only validated numbers;
  cache-key inputs are hashed and are not written into cache records.
- API keys or environment dumps.
- Cross-run evidence ledgers (would mix provenance across questions).
- Mutable corrective-loop state.

## Tests

`tests/unit/test_cache.py` covers hit, miss, full-length deterministic keys,
schema invalidation, corruption/key mismatch, non-finite payload rejection,
`get_or_set`, and clear.
