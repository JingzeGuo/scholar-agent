"""Schema-constrained relation extraction (offline heuristic + optional LLM)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from scholar_agent.graph.aliases import SEED_ALIASES
from scholar_agent.graph.evidence import find_evidence_span
from scholar_agent.ids import make_relation_id
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.graph import EntityType, Relation, RelationType

# High-precision surface patterns for known literature terms (longest first)
_KNOWN_SURFACES: list[str] = sorted(
    {surface for surface in SEED_ALIASES},
    key=len,
    reverse=True,
)

# Relation cue patterns: (regex, relation_type, subject_group, object_group)
_CUE_PATTERNS: list[tuple[re.Pattern[str], RelationType]] = [
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:proposes?|introduces?|presents?)\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.PROPOSES,
    ),
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:extends?|builds\s+upon|is\s+based\s+on)\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.EXTENDS,
    ),
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:uses?|utilizes?|employs?|relies\s+on)\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.USES,
    ),
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:evaluates?\s+on|tested\s+on|benchmarked\s+on)\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.EVALUATES_ON,
    ),
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:reports?|achieves?)\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.REPORTS,
    ),
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:outperforms?|beats?|surpasses?)\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.OUTPERFORMS,
    ),
    (
        re.compile(
            r"(?P<sub>.{2,60}?)\s+(?:compared\s+(?:with|to)|compares?\s+(?:with|to))\s+(?P<obj>.{2,80}?)(?:\.|,|;|$)",
            re.I,
        ),
        RelationType.COMPARES_WITH,
    ),
]


def _find_known_mentions(text: str) -> list[tuple[str, str, EntityType]]:
    """Return (surface, canonical, type) mentions found in text."""
    lower = text.lower()
    found: list[tuple[str, str, EntityType]] = []
    occupied: list[tuple[int, int]] = []
    for surface in _KNOWN_SURFACES:
        start = 0
        while True:
            idx = lower.find(surface, start)
            if idx < 0:
                break
            end = idx + len(surface)
            # word-ish boundary check
            before_ok = idx == 0 or not lower[idx - 1].isalnum()
            after_ok = end >= len(lower) or not lower[end].isalnum()
            if before_ok and after_ok and not any(s <= idx < e or s < end <= e for s, e in occupied):
                canonical, etype = SEED_ALIASES[surface]
                # recover original casing slice
                original = text[idx:end]
                found.append((original, canonical, etype))
                occupied.append((idx, end))
            start = end
    return found


def extract_from_chunk(chunk: Chunk) -> list[Relation]:
    """Heuristic extraction grounded in the chunk text."""
    text = chunk.text
    if len(text.strip()) < 40:
        return []

    relations: list[Relation] = []
    mentions = _find_known_mentions(text)

    # Pair co-occurring known entities with inferred relation cues
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        if len(sent) < 20:
            continue
        sent_mentions = [
            (s, c, t) for s, c, t in mentions if s.lower() in sent.lower()
        ]
        # Cue-based patterns
        for pattern, rel_type in _CUE_PATTERNS:
            match = pattern.search(sent)
            if not match:
                continue
            sub = match.group("sub").strip(" ,;:")
            obj = match.group("obj").strip(" ,;:")
            if len(sub) < 2 or len(obj) < 2:
                continue
            span = find_evidence_span(text, match.group(0).strip()) or find_evidence_span(
                text, sent.strip()
            )
            if span is None:
                continue
            sub_type = _guess_type(sub, sent_mentions)
            obj_type = _guess_type(obj, sent_mentions)
            relations.append(
                _make_partial_relation(
                    subject_surface=sub[:80],
                    object_surface=obj[:80],
                    subject_type=sub_type,
                    object_type=obj_type,
                    relation_type=rel_type,
                    evidence_span=span[:400],
                    chunk=chunk,
                    confidence=0.55,
                )
            )

        # Co-occurrence of two known methods/datasets → USES / EVALUATES_ON / COMPARES_WITH
        if len(sent_mentions) >= 2:
            for i in range(len(sent_mentions)):
                for j in range(i + 1, len(sent_mentions)):
                    s_surf, s_can, s_type = sent_mentions[i]
                    o_surf, o_can, o_type = sent_mentions[j]
                    rel_type = _infer_pair_relation(s_type, o_type, sent)
                    span = find_evidence_span(text, sent.strip())
                    if span is None:
                        # use a short window containing both surfaces
                        span = find_evidence_span(text, s_surf) or s_surf
                        if find_evidence_span(text, str(span)) is None:
                            continue
                        # Prefer full sentence if both appear
                        if s_surf.lower() in sent.lower() and o_surf.lower() in sent.lower():
                            span = sent.strip()[:400]
                            if find_evidence_span(text, span) is None:
                                continue
                    relations.append(
                        _make_partial_relation(
                            subject_surface=s_surf,
                            object_surface=o_surf,
                            subject_type=s_type,
                            object_type=o_type,
                            relation_type=rel_type,
                            evidence_span=span[:400] if isinstance(span, str) else sent[:400],
                            chunk=chunk,
                            confidence=0.65,
                        )
                    )

    return _dedupe_relations(relations)


def extract_paper_structure(paper: Paper, chunks: Iterable[Chunk]) -> list[Relation]:
    """Add Paper–Author and Paper–topic structural edges with evidence from title/chunks."""
    relations: list[Relation] = []
    chunk_list = list(chunks)
    if not chunk_list:
        return relations
    # Use first chunk as provenance for paper-level facts when title appears
    anchor = chunk_list[0]
    title_span = find_evidence_span(anchor.text, paper.title)
    evidence = title_span or paper.title
    # If title not in first chunk, still attach with low confidence only if we find title later
    if title_span is None:
        for ch in chunk_list[:5]:
            title_span = find_evidence_span(ch.text, paper.title)
            if title_span:
                anchor = ch
                evidence = title_span
                break
        else:
            # No grounded span for title — skip structural edges that lack evidence
            # Authors still need evidence; skip if not grounded
            return relations

    for author in paper.authors[:12]:
        if not author.strip():
            continue
        # Author name must appear in some early chunk to be grounded
        author_chunk = None
        author_span = None
        for ch in chunk_list[:8]:
            author_span = find_evidence_span(ch.text, author)
            if author_span:
                author_chunk = ch
                break
        if author_chunk is None or author_span is None:
            continue
        relations.append(
            _make_partial_relation(
                subject_surface=paper.title[:80],
                object_surface=author,
                subject_type=EntityType.PAPER,
                object_type=EntityType.AUTHOR,
                relation_type=RelationType.AUTHORED_BY,
                evidence_span=author_span[:400],
                chunk=author_chunk,
                confidence=0.9,
            )
        )

    # Paper PROPOSES method when known method name is in title
    for surface, (canonical, etype) in SEED_ALIASES.items():
        if etype != EntityType.METHOD:
            continue
        if surface in paper.title.lower() or canonical.lower() in paper.title.lower():
            span = find_evidence_span(anchor.text, evidence) or evidence
            if find_evidence_span(anchor.text, str(span)) is None and span not in anchor.text:
                # try method surface in anchor
                span2 = find_evidence_span(anchor.text, canonical) or find_evidence_span(
                    anchor.text, surface
                )
                if span2 is None:
                    continue
                span = span2
            relations.append(
                _make_partial_relation(
                    subject_surface=paper.title[:80],
                    object_surface=canonical,
                    subject_type=EntityType.PAPER,
                    object_type=EntityType.METHOD,
                    relation_type=RelationType.PROPOSES,
                    evidence_span=str(span)[:400],
                    chunk=anchor,
                    confidence=0.8,
                )
            )
            break
    return relations


def _guess_type(
    surface: str,
    mentions: list[tuple[str, str, EntityType]],
) -> EntityType:
    low = surface.lower()
    for s, _c, t in mentions:
        if s.lower() in low or low in s.lower():
            return t
    key = low.strip()
    if key in SEED_ALIASES:
        return SEED_ALIASES[key][1]
    if any(x in low for x in ("dataset", "benchmark", "corpus")):
        return EntityType.DATASET
    if any(x in low for x in ("accuracy", "f1", "recall", "precision", "em", "ndcg", "mrr")):
        return EntityType.METRIC
    return EntityType.METHOD


def _infer_pair_relation(
    s_type: EntityType, o_type: EntityType, sentence: str
) -> RelationType:
    low = sentence.lower()
    if "outperform" in low or "better than" in low:
        return RelationType.OUTPERFORMS
    if "compar" in low:
        return RelationType.COMPARES_WITH
    if s_type == EntityType.METHOD and o_type == EntityType.DATASET:
        return RelationType.EVALUATES_ON
    if s_type == EntityType.METHOD and o_type == EntityType.METRIC:
        return RelationType.REPORTS
    if s_type == EntityType.METHOD and o_type == EntityType.METHOD:
        return RelationType.COMPARES_WITH
    if s_type == EntityType.PAPER and o_type == EntityType.METHOD:
        return RelationType.PROPOSES
    return RelationType.USES


def _make_partial_relation(
    *,
    subject_surface: str,
    object_surface: str,
    subject_type: EntityType,
    object_type: EntityType,
    relation_type: RelationType,
    evidence_span: str,
    chunk: Chunk,
    confidence: float,
) -> Relation:
    # Temporary IDs; pipeline rewrites after entity resolution
    return Relation(
        relation_id=make_relation_id(
            subject_entity_id=f"tmp_sub_{subject_surface}",
            relation_type=relation_type.value,
            object_entity_id=f"tmp_obj_{object_surface}",
            chunk_id=chunk.chunk_id,
            evidence_span=evidence_span,
        ),
        subject_surface=subject_surface.strip(),
        object_surface=object_surface.strip(),
        subject_type=subject_type,
        object_type=object_type,
        relation_type=relation_type,
        evidence_span=evidence_span,
        paper_id=chunk.paper_id,
        chunk_id=chunk.chunk_id,
        page_number=chunk.page_start,
        confidence=confidence,
    )


def _dedupe_relations(relations: list[Relation]) -> list[Relation]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Relation] = []
    for rel in relations:
        key = (
            rel.subject_surface.lower(),
            rel.object_surface.lower(),
            rel.relation_type.value,
            rel.chunk_id,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


# Prompt fragment for optional LLM extraction (used when live LLM is enabled)
GRAPH_EXTRACTION_SYSTEM_PROMPT = """You extract knowledge-graph relations from academic paper chunks.
Return JSON only: {"relations": [{"subject_surface": str, "object_surface": str,
"subject_type": "Paper|Method|Dataset|Task|Metric|Author|Organization",
"object_type": same, "relation_type": "PROPOSES|EXTENDS|USES|EVALUATES_ON|REPORTS|COMPARES_WITH|OUTPERFORMS|CITES|AUTHORED_BY",
"evidence_span": str, "confidence": float}]}.
Rules:
- evidence_span MUST be a verbatim substring of the chunk.
- Use only the allowed types and relations.
- If nothing is supported, return {"relations": []}.
- Do not invent citations or entities not present in the text.
"""
