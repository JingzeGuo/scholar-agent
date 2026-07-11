"""Core Pydantic models shared across module boundaries.

Phase 0 runtime/event models and Phase 1 domain models live here. Prefer these
typed objects over untyped dicts at package boundaries.
"""

from __future__ import annotations

from scholar_agent.ids import new_run_id
from scholar_agent.models.answer import (
    CitationIssue,
    CitationReport,
    ClaimWithCitations,
    DraftAnswer,
    FinalAnswer,
)
from scholar_agent.models.base import (
    BudgetStatus,
    EventType,
    ExecutionEvent,
    PrototypeDecision,
    PrototypeObservation,
    PrototypeResult,
    QueryType,
    TokenUsage,
    utc_now_iso,
)
from scholar_agent.models.corpus import (
    Chunk,
    CorpusManifestEntry,
    IngestionStatus,
    Paper,
    PaperPage,
)
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.graph import (
    Entity,
    EntityType,
    Relation,
    RelationType,
)
from scholar_agent.models.ingestion import (
    CorpusIngestionReport,
    ExtractionIssue,
    ExtractionSeverity,
    PaperExtractionReport,
    SectionBlock,
)
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import ResearchRunState, VerificationResult

__all__ = [
    "BudgetStatus",
    "Chunk",
    "CitationIssue",
    "CitationReport",
    "ClaimWithCitations",
    "CorpusIngestionReport",
    "CorpusManifestEntry",
    "DraftAnswer",
    "Entity",
    "EntityType",
    "EventType",
    "EvidenceItem",
    "EvidenceLedger",
    "ExecutionEvent",
    "ExtractionIssue",
    "ExtractionSeverity",
    "FinalAnswer",
    "IngestionStatus",
    "Paper",
    "PaperExtractionReport",
    "PaperPage",
    "SectionBlock",
    "PrototypeDecision",
    "PrototypeObservation",
    "PrototypeResult",
    "QueryPlan",
    "QueryType",
    "Relation",
    "RelationType",
    "ResearchRunState",
    "SubQuestion",
    "SubQuestionStatus",
    "TokenUsage",
    "VerificationResult",
    "new_run_id",
    "utc_now_iso",
]
