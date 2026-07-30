"""Core Pydantic models shared across module boundaries.

Shared runtime/event and domain models live here. Prefer these typed objects
over untyped dicts at package boundaries.
"""

from __future__ import annotations

from scholar_agent.ids import new_run_id
from scholar_agent.models.answer import (
    AnswerStatus,
    CitationIssue,
    CitationReport,
    ClaimWithCitations,
    ComparisonCell,
    ComparisonRow,
    DraftAnswer,
    FinalAnswer,
    SourceCard,
)
from scholar_agent.models.base import (
    BudgetStatus,
    CompatibilityDecision,
    ErrorCategory,
    EventType,
    ExecutionEvent,
    QueryType,
    StructuredError,
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
from scholar_agent.models.planning import (
    AnswerRequirement,
    PlannedEntity,
    QueryPlan,
    SubQuestion,
    SubQuestionStatus,
)
from scholar_agent.models.retrieval import (
    CitationRef,
    NaiveRAGAnswer,
    RankTrace,
    RetrievalFilters,
    RetrievalHit,
    RetrievalResult,
)
from scholar_agent.models.routing import RetrievalPolicy, RoutingDecision, ToolAction
from scholar_agent.models.workflow import (
    CorrectiveQuery,
    ResearchRunState,
    VerificationResult,
)

__all__ = [
    "AnswerRequirement",
    "AnswerStatus",
    "BudgetStatus",
    "Chunk",
    "CitationIssue",
    "CitationRef",
    "CitationReport",
    "ClaimWithCitations",
    "CompatibilityDecision",
    "ComparisonCell",
    "ComparisonRow",
    "CorrectiveQuery",
    "CorpusIngestionReport",
    "CorpusManifestEntry",
    "DraftAnswer",
    "Entity",
    "EntityType",
    "ErrorCategory",
    "EventType",
    "EvidenceItem",
    "EvidenceLedger",
    "ExecutionEvent",
    "ExtractionIssue",
    "ExtractionSeverity",
    "FinalAnswer",
    "IngestionStatus",
    "NaiveRAGAnswer",
    "Paper",
    "PaperExtractionReport",
    "PaperPage",
    "PlannedEntity",
    "SectionBlock",
    "QueryPlan",
    "QueryType",
    "RankTrace",
    "Relation",
    "RelationType",
    "ResearchRunState",
    "RetrievalFilters",
    "RetrievalHit",
    "RetrievalPolicy",
    "RetrievalResult",
    "RoutingDecision",
    "SourceCard",
    "StructuredError",
    "SubQuestion",
    "SubQuestionStatus",
    "TokenUsage",
    "ToolAction",
    "VerificationResult",
    "new_run_id",
    "utc_now_iso",
]
