"""Adaptive query classifier and retrieval policy router.

Rule-based offline classifier so unit tests and demos do not require an LLM.
The Research Agent may override the recommendation within its budget; both the
recommendation and the final action are logged.
"""

from __future__ import annotations

import re

from scholar_agent.graph.aliases import SEED_ALIASES
from scholar_agent.ids import normalize_text
from scholar_agent.models.base import QueryType
from scholar_agent.models.routing import RetrievalPolicy, RoutingDecision

# Exact-ish tokens that favor sparse/hybrid keyword retrieval
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")
_VERSIONED_RE = re.compile(r"\b[A-Za-z]{2,}\s*v?\d+(?:\.\d+)*\b")
_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ(?:ence|ent)?|better than|outperform)\b",
    re.I,
)
_RELATIONAL_RE = re.compile(
    r"\b(uses?|used on|evaluates? on|trained on|reports?|metric|dataset|"
    r"relation(?:ship)?|how does .+ relate|linked to)\b",
    re.I,
)
_SYNTHESIS_RE = re.compile(
    r"\b(survey|overview|summarize|summary|landscape|trends?|across papers|"
    r"what are the main|overall)\b",
    re.I,
)
_CONCEPTUAL_RE = re.compile(
    r"\b(what is|what are|explain|describe|definition of|how does .+ work|"
    r"intuitively|conceptually)\b",
    re.I,
)
_CORRECTIVE_RE = re.compile(
    r"\b(missing|additional evidence|find more|corrective|need evidence about)\b",
    re.I,
)
_GENERAL_EVIDENCE_RE = re.compile(
    r"\b(provide|find|show|give|gather)\s+(?:me\s+)?(?:the\s+)?evidence\b|"
    r"\bevidence\s+(?:for|supporting|about|on)\b",
    re.I,
)

# Known method/dataset tokens from seed aliases (favor keyword/hybrid)
_KNOWN_KEYWORDS = {normalize_text(k) for k in SEED_ALIASES}


def classify_query_type(query: str) -> tuple[QueryType, list[str]]:
    """Return (query_type, human-readable signals)."""
    q = query.strip()
    signals: list[str] = []
    if not q:
        return QueryType.SEMANTIC, ["empty_query"]

    if _COMPARE_RE.search(q):
        signals.append("comparison_cue")
        return QueryType.COMPARISON, signals
    if _SYNTHESIS_RE.search(q):
        signals.append("synthesis_cue")
        return QueryType.SYNTHESIS, signals
    if _RELATIONAL_RE.search(q):
        signals.append("relational_cue")
        return QueryType.RELATIONAL, signals
    if _GENERAL_EVIDENCE_RE.search(q):
        signals.append("default_general_evidence")
        return QueryType.SEMANTIC, signals

    acronyms = _ACRONYM_RE.findall(q)
    if acronyms:
        signals.append(f"acronyms={acronyms[:5]}")
    if _VERSIONED_RE.search(q):
        signals.append("versioned_name")

    # Multiword / known entity alias hits
    known_hits = [alias for alias in _KNOWN_KEYWORDS if alias in normalize_text(q)]
    if known_hits:
        signals.append(f"known_entities={known_hits[:5]}")

    # Conceptual questions prefer dense even if a known term is mentioned
    # (e.g. "What is RAG conceptually?") unless the query is acronym/entity-only.
    if _CONCEPTUAL_RE.search(q):
        signals.append("conceptual_cue")
        word_count = len(re.findall(r"\w+", q))
        entity_only = bool(acronyms or known_hits) and word_count <= 4
        if entity_only:
            signals.append("short_entity_lookup")
            return QueryType.KEYWORD, signals
        return QueryType.SEMANTIC, signals

    # Keyword-heavy if acronyms / known entities / versioned names
    if acronyms or known_hits or _VERSIONED_RE.search(q):
        signals.append("keyword_or_entity_focus")
        return QueryType.KEYWORD, signals

    # Default: general evidence question
    signals.append("default_general_evidence")
    return QueryType.SEMANTIC, signals


def recommend_policy(
    query: str,
    *,
    query_type: QueryType | None = None,
    has_graph: bool = True,
    corrective: bool = False,
    missing_aspect: str | None = None,
) -> RoutingDecision:
    """Map query type / pattern → default retrieval policy."""
    signals: list[str] = []
    if query_type is None:
        query_type, signals = classify_query_type(query)
    else:
        _, signals = classify_query_type(query)

    if corrective or _CORRECTIVE_RE.search(query):
        signals.append("corrective")
        # Policy from verifier missing aspect when provided
        aspect = missing_aspect or query
        aspect_type, aspect_signals = classify_query_type(aspect)
        signals.extend([f"aspect_{s}" for s in aspect_signals])
        policy = (
            RetrievalPolicy.HYBRID_RERANK
            if aspect_type == QueryType.SEMANTIC and "default_general_evidence" in aspect_signals
            else _policy_for_type(aspect_type, has_graph=has_graph)
        )
        return RoutingDecision(
            query=query,
            query_type=aspect_type,
            recommended_policy=policy,
            rationale=(
                f"Corrective retrieval for missing aspect ({aspect_type.value}) → {policy.value}"
            ),
            signals=signals,
        )

    # A conceptual paraphrase is dense-friendly, while a general evidence
    # request needs broader hybrid candidates plus reranking. Both use the
    # semantic query type, so preserve the classifier signal here.
    if query_type == QueryType.SEMANTIC and "default_general_evidence" in signals:
        policy = RetrievalPolicy.HYBRID_RERANK
    else:
        policy = _policy_for_type(query_type, has_graph=has_graph)
    rationale = _rationale(query_type, policy, has_graph=has_graph)
    return RoutingDecision(
        query=query,
        query_type=query_type,
        recommended_policy=policy,
        rationale=rationale,
        signals=signals,
    )


def _policy_for_type(query_type: QueryType, *, has_graph: bool) -> RetrievalPolicy:
    if query_type == QueryType.SEMANTIC:
        return RetrievalPolicy.DENSE
    if query_type == QueryType.KEYWORD:
        return RetrievalPolicy.HYBRID  # sparse+dense for exact names
    if query_type == QueryType.COMPARISON:
        return RetrievalPolicy.HYBRID_PLUS_GRAPH if has_graph else RetrievalPolicy.HYBRID_RERANK
    if query_type == QueryType.RELATIONAL:
        return RetrievalPolicy.GRAPH if has_graph else RetrievalPolicy.HYBRID_RERANK
    if query_type == QueryType.SYNTHESIS:
        return RetrievalPolicy.HYBRID_RERANK
    return RetrievalPolicy.HYBRID_RERANK


def _rationale(query_type: QueryType, policy: RetrievalPolicy, *, has_graph: bool) -> str:
    if query_type == QueryType.SEMANTIC and policy == RetrievalPolicy.HYBRID_RERANK:
        return "General evidence request → broad hybrid candidates + reranker"
    mapping = {
        QueryType.SEMANTIC: "Conceptual paraphrase → dense retrieval",
        QueryType.KEYWORD: "Exact model/dataset/metric/acronym → sparse or hybrid",
        QueryType.COMPARISON: "Cross-paper comparison → hybrid (+ graph when available)",
        QueryType.RELATIONAL: "Method–dataset–metric relation → graph + supporting chunks",
        QueryType.SYNTHESIS: "General / synthesis evidence → hybrid + reranker",
    }
    base = mapping.get(query_type, "Default hybrid+rerank")
    if not has_graph and "graph" in policy.value:
        return f"{base}; graph unavailable → {policy.value}"
    return f"{base} → {policy.value}"


def policy_to_tool_modes(policy: RetrievalPolicy) -> list[str]:
    """Expand a policy into ordered tool modes for one research step."""
    if policy == RetrievalPolicy.DENSE:
        return ["dense"]
    if policy == RetrievalPolicy.SPARSE:
        return ["sparse"]
    if policy == RetrievalPolicy.HYBRID:
        return ["hybrid"]
    if policy == RetrievalPolicy.HYBRID_RERANK:
        return ["hybrid_rerank"]
    if policy == RetrievalPolicy.GRAPH:
        return ["graph"]
    if policy == RetrievalPolicy.HYBRID_PLUS_GRAPH:
        return ["hybrid_rerank", "graph"]
    return ["hybrid_rerank"]
