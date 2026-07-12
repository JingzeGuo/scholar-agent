#!/usr/bin/env python3
"""Download a curated arXiv corpus for ScholarAgent.

Reads ``data/seed_arxiv_ids.yaml``, fetches metadata via the arXiv API,
downloads PDFs into ``data/papers/``, and writes ``data/corpus_manifest.jsonl``.

Usage:
    uv run python scripts/download_corpus.py
    uv run python scripts/download_corpus.py --limit 20          # smoke test
    uv run python scripts/download_corpus.py --skip-existing
    uv run python scripts/download_corpus.py --metadata-only     # no PDFs

Respect arXiv rate limits: ~1 request / 3s for API, polite PDF download pacing.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scholar_agent.ids import make_paper_id  # noqa: E402
from scholar_agent.models.corpus import CorpusManifestEntry, IngestionStatus  # noqa: E402
from scholar_agent.storage.manifest import save_corpus_manifest  # noqa: E402

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

DEFAULT_SEED = REPO_ROOT / "data" / "seed_arxiv_ids.yaml"
DEFAULT_PAPERS_DIR = REPO_ROOT / "data" / "papers"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "corpus_manifest.jsonl"

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"

# Known off-topic / wrong-ID collisions to never include
DENYLIST: set[str] = {
    "2305.06934",
    "2406.00045",
    "2404.13534",
    "2305.12223",
    "2401.08500",
    "2309.15140",
    "2401.01234",
    "2406.18175",
    "2402.12330",
    "2406.01163",
    "2406.20094",
    "2405.04434",
    "2402.08317",
    "2406.11409",
    "2401.06823",
    "2406.00515",
    "2312.04724",
    "2405.06681",
    "2402.08377",
    "2406.09479",
}

RELEVANCE_RE = re.compile(
    r"\b(rag|retrieval|retriev|passage|embedd|rerank|re-rank|rewrite|"
    r"information retrieval|dense text|dense passage|"
    r"knowledge.?graph|graphrag|agent|tool-?use|toolformer|react|"
    r"multi-?agent|planning|reflexion|hallucin|benchmark|survey|"
    r"colbert|splade|contriever|realm|hyde|raptor|hipporag|memorag|"
    r"lightrag|rankrag|hybridrag|self-rag|corrective|g-?retriever|"
    r"factscore|beir|mteb|long.?context|query expansion|open-?domain|"
    r"question answering|attribution|dspy|autogen|metagpt|gorilla|"
    r"voyager|generative agents|agentbench|mixture-of-agents|"
    r"critic|language model.?agent|function.?call)\b",
    re.I,
)

# Supplemental search queries to top up toward target size if seed IDs fail
FILL_QUERIES: list[tuple[str, str]] = [
    (
        "foundational",
        'ti:"retrieval-augmented generation" OR ti:"dense passage retrieval" OR ti:ColBERT',
    ),
    ("retrieval_rerank_query", 'all:"query rewriting" AND all:retrieval AND all:"language model"'),
    (
        "agent_planning_tools",
        'ti:Toolformer OR ti:ReAct OR ti:Reflexion OR ti:AutoGen OR all:"language agents"',
    ),
    (
        "agentic_rag_graphrag",
        'ti:GraphRAG OR all:"Self-RAG" OR all:"corrective retrieval" OR ti:HippoRAG OR ti:LightRAG',
    ),
    (
        "benchmarks_eval_surveys",
        'ti:"retrieval-augmented" AND (ti:survey OR ti:benchmark OR ti:evaluation OR ti:Ragas)',
    ),
    ("agentic_rag_graphrag", 'all:HybridRAG OR all:RankRAG OR all:"G-Retriever" OR all:ARES'),
    ("agent_planning_tools", "all:ToolLLM OR all:AgentBench OR all:FireAct"),
]


@dataclass
class ArxivRecord:
    arxiv_id: str
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    categories: list[str]
    topic_labels: list[str]


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return re.sub(r"\s+", " ", el.text).strip()


def _normalize_arxiv_id(raw: str) -> str:
    value = raw.strip()
    value = value.removeprefix("arXiv:")
    value = value.removeprefix("http://arxiv.org/abs/")
    value = value.removeprefix("https://arxiv.org/abs/")
    # Drop version suffix: 2005.11401v4 -> 2005.11401
    value = re.sub(r"v\d+$", "", value)
    return value


def load_seed(path: Path) -> list[tuple[str, str]]:
    """Return ordered (arxiv_id, topic_label) pairs, de-duplicated by id."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Seed file must be a mapping of category -> id list: {path}")

    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for category, ids in data.items():
        if category.startswith("#") or not isinstance(ids, list):
            continue
        for raw in ids:
            arxiv_id = _normalize_arxiv_id(str(raw))
            if not arxiv_id or arxiv_id in seen or arxiv_id in DENYLIST:
                continue
            seen.add(arxiv_id)
            ordered.append((arxiv_id, str(category)))
    return ordered


def is_relevant(title: str, abstract: str = "") -> bool:
    return bool(RELEVANCE_RE.search(f"{title} {abstract}"))


def _parse_entries(
    xml_text: str,
    default_label: str | None = None,
    *,
    require_relevant: bool = True,
) -> list[ArxivRecord]:
    root = ET.fromstring(xml_text)
    records: list[ArxivRecord] = []
    for entry in root.findall(f"{ATOM}entry"):
        id_text = _text(entry.find(f"{ATOM}id"))
        # http://arxiv.org/abs/2005.11401v4
        match = re.search(r"arxiv\.org/abs/([^/\s]+)", id_text)
        if not match:
            continue
        arxiv_id = _normalize_arxiv_id(match.group(1))
        if arxiv_id in DENYLIST:
            continue
        title = _text(entry.find(f"{ATOM}title"))
        if not title or title.lower().startswith("error"):
            continue
        authors = [
            _text(a.find(f"{ATOM}name"))
            for a in entry.findall(f"{ATOM}author")
            if _text(a.find(f"{ATOM}name"))
        ]
        published = _text(entry.find(f"{ATOM}published"))
        year = int(published[:4]) if published[:4].isdigit() else None
        abstract = _text(entry.find(f"{ATOM}summary"))
        if require_relevant and not is_relevant(title, abstract):
            continue
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall(f"{ATOM}category")
            if c.attrib.get("term")
        ]
        primary = entry.find(f"{ARXIV_NS}primary_category")
        term = primary.attrib.get("term") if primary is not None else None
        if term and term not in categories:
            categories.insert(0, term)
        labels = [default_label] if default_label else []
        records.append(
            ArxivRecord(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                categories=categories,
                topic_labels=[lab for lab in labels if lab],
            )
        )
    return records


_SEED_BROAD_RE = re.compile(
    r"retriev|embed|agent|passage|rank|search|question answering|"
    r"language model|memory|tool|graph|rag|augment",
    re.I,
)


def fetch_by_ids(
    client: httpx.Client,
    arxiv_ids: Sequence[str],
    *,
    topic_by_id: dict[str, str],
    batch_size: int = 20,
    pause_s: float = 3.0,
) -> dict[str, ArxivRecord]:
    """Fetch metadata for known IDs. Missing/invalid IDs are omitted."""
    found: dict[str, ArxivRecord] = {}
    for i in range(0, len(arxiv_ids), batch_size):
        batch = list(arxiv_ids[i : i + batch_size])
        params = {
            "id_list": ",".join(batch),
            "max_results": len(batch),
        }
        response = client.get(API_URL, params=params, timeout=60.0)
        response.raise_for_status()
        # Seed IDs: allow broader match so classic IR papers are not dropped
        for rec in _parse_entries(response.text, require_relevant=False):
            if rec.arxiv_id in DENYLIST:
                continue
            if not is_relevant(rec.title, rec.abstract) and not _SEED_BROAD_RE.search(
                f"{rec.title} {rec.abstract}"
            ):
                print(f"  [skip off-topic seed] {rec.arxiv_id}: {rec.title[:70]}")
                continue
            label = topic_by_id.get(rec.arxiv_id)
            if label and label not in rec.topic_labels:
                rec.topic_labels.append(label)
            found[rec.arxiv_id] = rec
        missing = [aid for aid in batch if aid not in found]
        if missing:
            print(f"  [warn] no metadata / filtered for: {', '.join(missing)}")
        if i + batch_size < len(arxiv_ids):
            time.sleep(pause_s)
    return found


def search_arxiv(
    client: httpx.Client,
    query: str,
    *,
    label: str,
    max_results: int = 30,
    pause_s: float = 3.0,
) -> list[ArxivRecord]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = client.get(API_URL, params=params, timeout=60.0)
    response.raise_for_status()
    time.sleep(pause_s)
    return _parse_entries(response.text, default_label=label)


def sha256_file(path: Path, length: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:length]


def download_pdf(
    client: httpx.Client,
    arxiv_id: str,
    dest: Path,
    *,
    skip_existing: bool,
    pause_s: float = 1.0,
) -> tuple[bool, str | None]:
    """Download one PDF. Returns (ok, content_hash_or_none)."""
    if skip_existing and dest.is_file() and dest.stat().st_size > 1000:
        return True, sha256_file(dest)
    url = PDF_URL.format(arxiv_id=quote(arxiv_id))
    try:
        with client.stream("GET", url, timeout=120.0, follow_redirects=True) as response:
            if response.status_code != 200:
                print(f"  [fail] PDF {arxiv_id}: HTTP {response.status_code}")
                return False, None
            content_type = response.headers.get("content-type", "")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".pdf.part")
            size = 0
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    size += len(chunk)
            if size < 1000:
                tmp.unlink(missing_ok=True)
                print(f"  [fail] PDF {arxiv_id}: too small ({size} bytes)")
                return False, None
            # arXiv sometimes returns HTML error pages
            if "html" in content_type.lower():
                head = tmp.read_bytes()[:200].lower()
                if b"<html" in head or b"<!doctype" in head:
                    tmp.unlink(missing_ok=True)
                    print(f"  [fail] PDF {arxiv_id}: got HTML instead of PDF")
                    return False, None
            tmp.replace(dest)
        time.sleep(pause_s)
        return True, sha256_file(dest)
    except httpx.HTTPError as exc:
        print(f"  [fail] PDF {arxiv_id}: {exc}")
        return False, None


def pdf_filename_for(arxiv_id: str) -> str:
    safe = arxiv_id.replace("/", "_")
    return f"{safe}.pdf"


def record_to_entry(
    rec: ArxivRecord,
    *,
    content_hash: str,
    status: IngestionStatus = IngestionStatus.PENDING,
) -> CorpusManifestEntry:
    paper_id = make_paper_id(arxiv_id=rec.arxiv_id, title=rec.title, year=rec.year)
    labels = list(dict.fromkeys(rec.topic_labels + rec.categories[:2]))
    return CorpusManifestEntry(
        paper_id=paper_id,
        title=rec.title,
        authors=rec.authors,
        year=rec.year,
        venue="arXiv",
        doi=None,
        arxiv_id=rec.arxiv_id,
        pdf_filename=pdf_filename_for(rec.arxiv_id),
        source_url=f"https://arxiv.org/abs/{rec.arxiv_id}",
        topic_labels=labels,
        ingestion_status=status,
        content_hash=content_hash or "pending_download",
    )


def top_up_records(
    client: httpx.Client,
    records: dict[str, ArxivRecord],
    *,
    target: int,
) -> dict[str, ArxivRecord]:
    if len(records) >= target:
        return records
    print(f"Topping up corpus: have {len(records)}, target {target}")
    for label, query in FILL_QUERIES:
        if len(records) >= target:
            break
        print(f"  search [{label}]: {query[:70]}...")
        try:
            hits = search_arxiv(client, query, label=label, max_results=40)
        except httpx.HTTPError as exc:
            print(f"  [warn] search failed: {exc}")
            continue
        for rec in hits:
            if rec.arxiv_id in records:
                continue
            records[rec.arxiv_id] = rec
            if len(records) >= target:
                break
        print(f"  now {len(records)} unique papers")
    return records


def build_manifest_entries(
    records: Iterable[ArxivRecord],
    *,
    papers_dir: Path,
    download: bool,
    skip_existing: bool,
    client: httpx.Client,
    limit: int | None,
) -> list[CorpusManifestEntry]:
    entries: list[CorpusManifestEntry] = []
    for idx, rec in enumerate(records):
        if limit is not None and idx >= limit:
            break
        dest = papers_dir / pdf_filename_for(rec.arxiv_id)
        content_hash = "pending_download"
        if download:
            print(f"[{idx + 1}] download {rec.arxiv_id}: {rec.title[:70]}")
            ok, digest = download_pdf(client, rec.arxiv_id, dest, skip_existing=skip_existing)
            # Still record metadata so download gaps stay visible
            content_hash = "download_failed" if not ok or not digest else digest
        elif dest.is_file():
            content_hash = sha256_file(dest)
        entries.append(record_to_entry(rec, content_hash=content_hash))
    return entries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target", type=int, default=120, help="Target paper count")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Skip PDF download")
    parser.add_argument("--no-fill", action="store_true", help="Do not search to top up")
    parser.add_argument("--api-pause", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_pairs = load_seed(args.seed)
    print(f"Loaded {len(seed_pairs)} unique seed arXiv IDs from {args.seed}")

    topic_by_id = {aid: label for aid, label in seed_pairs}
    seed_ids = [aid for aid, _ in seed_pairs]

    headers = {"User-Agent": "scholar-agent-corpus-downloader/0.1 (research; local)"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        print("Fetching metadata for seed IDs...")
        records = fetch_by_ids(
            client,
            seed_ids,
            topic_by_id=topic_by_id,
            pause_s=args.api_pause,
        )
        print(f"Resolved {len(records)}/{len(seed_ids)} seed IDs")

        if not args.no_fill:
            records = top_up_records(client, records, target=args.target)

        # Preserve seed order, then append fill-ins
        ordered: list[ArxivRecord] = []
        seen: set[str] = set()
        for aid, _ in seed_pairs:
            if aid in records and aid not in seen:
                ordered.append(records[aid])
                seen.add(aid)
        for aid, rec in records.items():
            if aid not in seen:
                ordered.append(rec)
                seen.add(aid)

        ordered = ordered[: args.limit] if args.limit is not None else ordered[: args.target]

        print(f"Building corpus of {len(ordered)} papers → {args.papers_dir}")
        entries = build_manifest_entries(
            ordered,
            papers_dir=args.papers_dir,
            download=not args.metadata_only,
            skip_existing=args.skip_existing,
            client=client,
            limit=None,
        )

    save_corpus_manifest(args.manifest, entries)
    ok_pdfs = sum(
        1
        for e in entries
        if e.content_hash not in {"pending_download", "download_failed"}
        and (args.papers_dir / e.pdf_filename).is_file()
    )
    print(f"\nWrote manifest: {args.manifest} ({len(entries)} entries)")
    print(f"PDFs present:   {ok_pdfs}/{len(entries)} in {args.papers_dir}")
    by_label: dict[str, int] = {}
    for e in entries:
        key = e.topic_labels[0] if e.topic_labels else "unknown"
        by_label[key] = by_label.get(key, 0) + 1
    print("Topic distribution (first label):")
    for k, v in sorted(by_label.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    failed = [e.arxiv_id for e in entries if e.content_hash == "download_failed"]
    if failed:
        print(f"Failed downloads ({len(failed)}): {', '.join(str(x) for x in failed[:20])}")
        return 1 if ok_pdfs < max(1, len(entries) // 2) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
