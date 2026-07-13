"""Deterministic claim-to-evidence support checks shared by agents.

This is deliberately conservative.  It is not a replacement for a trained
NLI model, but it gives the independent Verifier and the CitationValidator the
same auditable floor: important numbers and polarity must agree and enough
content words from the claim must be present in the cited passage.
"""

from __future__ import annotations

import re

from scholar_agent.ids import normalize_text

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "also",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "supported",
    "the",
    "to",
    "with",
}
_POLARITY_ROOTS = (
    ("outperform", "underperform"),
    ("increase", "decrease"),
    ("improve", "degrade"),
    ("better", "worse"),
    ("effective", "ineffective"),
    ("succeed", "fail"),
)
_META_MARKERS = (
    "sources disagree",
    "conflicting passages",
    "retained for audit",
    "limitation",
)


def support_tokens(text: str) -> set[str]:
    """Return normalized content tokens used by the support heuristic."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if token not in _STOP_WORDS and len(token) > 1
    }


def claim_support_score(claim_text: str, evidence_text: str) -> float:
    """Return lexical support after hard number/polarity consistency gates."""
    claim_tokens = support_tokens(claim_text)
    if not claim_tokens:
        return 1.0
    evidence_tokens = support_tokens(evidence_text)
    if not evidence_tokens:
        return 0.0

    claim_numbers = {token for token in claim_tokens if token.isdigit()}
    if not claim_numbers.issubset(evidence_tokens):
        return 0.0

    for positive, negative in _POLARITY_ROOTS:
        claim_positive = any(token.startswith(positive) for token in claim_tokens)
        claim_negative = any(token.startswith(negative) for token in claim_tokens)
        evidence_positive = any(token.startswith(positive) for token in evidence_tokens)
        evidence_negative = any(token.startswith(negative) for token in evidence_tokens)
        if (claim_positive and evidence_negative) or (claim_negative and evidence_positive):
            return 0.0

    normalized_claim = normalize_text(claim_text)
    normalized_evidence = normalize_text(evidence_text)
    if normalized_claim and normalized_claim[:80] in normalized_evidence:
        return 1.0
    return len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))


def claim_is_supported(
    claim_text: str,
    evidence_text: str,
    *,
    min_overlap: float = 0.5,
    allow_meta_claims: bool = False,
) -> bool:
    """Decide whether ``evidence_text`` supports ``claim_text``."""
    if allow_meta_claims and any(marker in claim_text.lower() for marker in _META_MARKERS):
        return True
    return claim_support_score(claim_text, evidence_text) >= min_overlap
