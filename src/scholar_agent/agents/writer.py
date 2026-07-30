"""Evidence-constrained structured Writer.

The Writer never retrieves and never treats model output as provenance.  It
accepts only verified evidence IDs, builds an entity-by-requirement comparison
when the plan requests one, and derives an honest complete/partial/insufficient
status.  An optional LLM may synthesize the structured cells; deterministic
writing remains the default and the safe fallback.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.ids import normalize_text
from scholar_agent.llm.client import ChatMessage, LLMClient
from scholar_agent.llm.structured import StructuredOutputError, parse_structured_json
from scholar_agent.logging import get_logger
from scholar_agent.models.answer import (
    AnswerStatus,
    ClaimWithCitations,
    ComparisonCell,
    ComparisonRow,
    DraftAnswer,
)
from scholar_agent.models.base import TokenUsage
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan
from scholar_agent.models.workflow import VerificationResult

logger = get_logger(__name__)

WRITER_PROMPT_VERSION = "phase11-writer-v1"
INSUFFICIENT_CELL_TEXT = "Insufficient verified evidence"

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


class WriterLLMError(RuntimeError):
    """Sanitized strict-mode failure from the LLM-backed Writer."""


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if token not in _STOP_WORDS and len(token) > 1
    ]


def _snippet(text: str, max_chars: int = 320) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


def format_inline_citation(item: EvidenceItem) -> str:
    """User-facing page citation from evidence provenance."""
    if item.page_start == item.page_end:
        return f"[{item.paper_id} p.{item.page_start}]"
    return f"[{item.paper_id} p.{item.page_start}-{item.page_end}]"


def render_claim_markdown(
    claim: ClaimWithCitations,
    ledger_by_id: dict[str, EvidenceItem],
) -> str:
    """Render one claim with inline citations derived from evidence IDs."""
    markers: list[str] = []
    seen: set[str] = set()
    for evidence_id in claim.evidence_ids:
        item = ledger_by_id.get(evidence_id)
        if item is None:
            continue
        marker = format_inline_citation(item)
        if marker not in seen:
            markers.append(marker)
            seen.add(marker)
    if markers:
        return f"{claim.text} {' '.join(markers)}"
    return claim.text


def render_core_answer_markdown(
    *,
    status: AnswerStatus,
    claims: list[ClaimWithCitations],
    rows: list[ComparisonRow],
    ledger_by_id: dict[str, EvidenceItem],
) -> str:
    """Shared renderer used before and after citation validation."""
    lines = [f"**Answer status:** {status.value}", ""]
    if status == AnswerStatus.PARTIAL:
        lines.extend(
            [
                "> **Partial Answer:** Some requested cells lack verified evidence.",
                "",
            ]
        )
    elif status == AnswerStatus.INSUFFICIENT:
        lines.extend(
            [
                "> **Limitation — Insufficient Evidence:** No citation-validated answer can be "
                "formed from the available corpus.",
                "",
            ]
        )

    if rows:
        entity_order: list[tuple[str, str]] = []
        seen_entities: set[str] = set()
        for row in rows:
            for cell in row.cells:
                if cell.entity_id not in seen_entities:
                    entity_order.append((cell.entity_id, cell.entity_label))
                    seen_entities.add(cell.entity_id)
        lines.append("### Comparison")
        lines.append("")
        lines.append(
            "| Dimension | "
            + " | ".join(_escape_table(label) for _, label in entity_order)
            + " |"
        )
        lines.append("| --- | " + " | ".join("---" for _ in entity_order) + " |")
        claims_by_id = {claim.claim_id: claim for claim in claims}
        for row in rows:
            cells_by_entity = {cell.entity_id: cell for cell in row.cells}
            rendered_cells: list[str] = []
            for entity_id, _ in entity_order:
                matrix_cell = cells_by_entity.get(entity_id)
                rendered_claim = (
                    claims_by_id.get(matrix_cell.claim_id)
                    if matrix_cell is not None and matrix_cell.claim_id is not None
                    else None
                )
                if (
                    matrix_cell is None
                    or not matrix_cell.supported
                    or rendered_claim is None
                ):
                    rendered_cells.append(INSUFFICIENT_CELL_TEXT)
                else:
                    rendered_cells.append(
                        _escape_table(
                            render_claim_markdown(rendered_claim, ledger_by_id)
                        )
                    )
            lines.append(
                f"| {_escape_table(row.label)} | " + " | ".join(rendered_cells) + " |"
            )
        lines.append("")
    else:
        lines.append("### Claims")
        lines.append("")
        if claims:
            lines.extend(
                f"- {render_claim_markdown(claim, ledger_by_id)}" for claim in claims
            )
        else:
            lines.append("No evidence-backed claims could be formed.")
        lines.append("")
    return "\n".join(lines).rstrip()


def _escape_table(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")


@dataclass(frozen=True)
class _EntitySpec:
    entity_id: str
    label: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RequirementSpec:
    key: str
    dimension: str
    label: str


class _StructuredWriterOutput(BaseModel):
    claims: list[ClaimWithCitations] = Field(default_factory=list)
    rows: list[ComparisonRow] = Field(default_factory=list)


@dataclass
class Writer:
    """Write only from verified evidence, optionally using structured LLM synthesis."""

    max_claims: int = 12
    max_evidence_per_claim: int = 3
    llm: LLMClient | None = None
    strict_llm: bool = False
    last_backend: str = field(default="deterministic", init=False)
    last_model: str | None = field(default=None, init=False)
    last_fallback_reason: str | None = field(default=None, init=False)
    last_prompt_version: str = field(default=WRITER_PROMPT_VERSION, init=False)
    last_token_usage: TokenUsage = field(default_factory=TokenUsage, init=False)

    def write(
        self,
        *,
        query: str,
        plan: QueryPlan | None = None,
        ledger: EvidenceLedger,
        verification: VerificationResult | None = None,
        answer_format: str | None = None,
        contradiction_notes: list[str] | None = None,
        corpus_insufficient: bool | None = None,
        force_deterministic: bool = False,
        forced_fallback_reason: str | None = None,
    ) -> DraftAnswer:
        """Produce structured claims/cells and render their evidence IDs.

        ``force_deterministic`` is an explicit, zero-provider-call escape hatch
        for callers that have exhausted a workflow budget.  It intentionally
        overrides an attached LLM client (including strict mode).
        """
        self._reset_run_metadata()
        notes = self._notes(
            verification=verification,
            contradiction_notes=contradiction_notes,
        )
        if force_deterministic:
            self.last_backend = "deterministic"
            self.last_fallback_reason = self._forced_fallback_reason(
                forced_fallback_reason
            )
            notes.append(
                "Writer used deterministic synthesis "
                f"({self.last_fallback_reason})."
            )
        writing_ledger = self._verified_ledger(ledger, verification)
        fmt = answer_format or (plan.answer_type if plan else "factual")
        is_comparison = fmt == "comparison"
        entities, requirements = self._comparison_specs(
            query=query,
            plan=plan,
            enabled=is_comparison,
        )

        claims: list[ClaimWithCitations]
        rows: list[ComparisonRow]
        if not writing_ledger.items:
            claims, rows = [], self._empty_rows(entities, requirements)
        elif self.llm is not None and not force_deterministic:
            try:
                claims, rows = self._write_with_llm(
                    query=query,
                    plan=plan,
                    ledger=writing_ledger,
                    entities=entities,
                    requirements=requirements,
                    is_comparison=is_comparison,
                )
            except Exception as exc:
                self.last_fallback_reason = self._fallback_reason(exc)
                logger.warning(
                    "writer_llm_fallback reason=%s prompt_version=%s",
                    self.last_fallback_reason,
                    WRITER_PROMPT_VERSION,
                )
                if self.strict_llm:
                    self.last_backend = "llm"
                    raise WriterLLMError(
                        f"LLM writer failed: {self.last_fallback_reason}"
                    ) from exc
                self.last_backend = "deterministic"
                notes.append(
                    "Writer degraded to deterministic synthesis "
                    f"({self.last_fallback_reason})."
                )
                claims, rows = self._write_deterministically(
                    query=query,
                    plan=plan,
                    ledger=writing_ledger,
                    entities=entities,
                    requirements=requirements,
                    is_comparison=is_comparison,
                )
        else:
            claims, rows = self._write_deterministically(
                query=query,
                plan=plan,
                ledger=writing_ledger,
                entities=entities,
                requirements=requirements,
                is_comparison=is_comparison,
            )

        status = self._answer_status(
            claims=claims,
            rows=rows,
            verification=verification,
            explicitly_insufficient=bool(corpus_insufficient),
        )
        if status != AnswerStatus.COMPLETE:
            notes.append(
                "The answer is limited to verified evidence; unsupported requirements "
                "are explicitly marked."
            )
        if not claims:
            notes.append("No verified evidence was available for writing.")

        by_id = {item.evidence_id: item for item in writing_ledger.items}
        core_answer = render_core_answer_markdown(
            status=status,
            claims=claims,
            rows=rows,
            ledger_by_id=by_id,
        )
        markdown = self._render_document(
            query=query,
            answer_format=fmt,
            core_answer=core_answer,
            claims=claims,
            ledger_by_id=by_id,
            notes=notes,
            verification=verification,
        )
        draft = DraftAnswer(
            claims=claims,
            rows=rows,
            status=status,
            core_answer=core_answer,
            markdown=markdown,
            notes=notes,
        )
        logger.info(
            "writer_draft claims=%s rows=%s status=%s backend=%s",
            len(claims),
            len(rows),
            status.value,
            self.last_backend,
        )
        return draft

    def _reset_run_metadata(self) -> None:
        self.last_backend = "deterministic"
        self.last_model = None
        self.last_fallback_reason = None
        self.last_token_usage = TokenUsage()

    def _notes(
        self,
        *,
        verification: VerificationResult | None,
        contradiction_notes: list[str] | None,
    ) -> list[str]:
        notes = list(contradiction_notes or [])
        if (
            not contradiction_notes
            and verification
            and verification.conflicting_evidence_ids
        ):
            notes.append(
                "Conflicting evidence retained: "
                + ", ".join(verification.conflicting_evidence_ids[:12])
            )
        if verification and verification.missing_aspects:
            notes.append(
                "Missing aspects after verification: "
                + "; ".join(verification.missing_aspects[:8])
            )
        return notes

    def _verified_ledger(
        self,
        ledger: EvidenceLedger,
        verification: VerificationResult | None,
    ) -> EvidenceLedger:
        if verification is None:
            return ledger
        allowed = {
            evidence_id
            for ids in verification.supported_evidence_ids.values()
            for evidence_id in ids
        }
        return EvidenceLedger(
            items=[item for item in ledger.items if item.evidence_id in allowed]
        )

    def _write_deterministically(
        self,
        *,
        query: str,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
        is_comparison: bool,
    ) -> tuple[list[ClaimWithCitations], list[ComparisonRow]]:
        self.last_backend = "deterministic"
        if is_comparison and len(entities) >= 2 and requirements:
            return self._build_comparison(
                query=query,
                plan=plan,
                ledger=ledger,
                entities=entities,
                requirements=requirements,
            )
        return self._build_noncomparison_claims(query=query, plan=plan, ledger=ledger), []

    def _build_comparison(
        self,
        *,
        query: str,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
    ) -> tuple[list[ClaimWithCitations], list[ComparisonRow]]:
        candidates = self._comparison_candidates(
            plan=plan,
            ledger=ledger,
            entities=entities,
            requirements=requirements,
        )
        claims: list[ClaimWithCitations] = []
        rows: list[ComparisonRow] = []
        for requirement in requirements:
            cells: list[ComparisonCell] = []
            for entity in entities:
                items = candidates.get((entity.entity_id, requirement.key), [])
                selected = self._select_diverse(
                    self._sorted_evidence(items),
                    limit=self.max_evidence_per_claim,
                )
                if not selected or len(claims) >= self.max_claims:
                    cells.append(
                        ComparisonCell(
                            entity_id=entity.entity_id,
                            entity_label=entity.label,
                        )
                    )
                    continue
                primary = selected[0]
                claim_id = f"claim_{len(claims) + 1}"
                claim = ClaimWithCitations(
                    claim_id=claim_id,
                    text=self._grounded_claim_text(primary, query=query),
                    evidence_ids=[item.evidence_id for item in selected],
                    sub_question_id=primary.sub_question_id,
                    requirement_key=requirement.key,
                    entity_id=entity.entity_id,
                    dimension=requirement.dimension,
                )
                claims.append(claim)
                cells.append(
                    ComparisonCell(
                        entity_id=entity.entity_id,
                        entity_label=entity.label,
                        text=claim.text,
                        evidence_ids=list(claim.evidence_ids),
                        claim_id=claim.claim_id,
                        supported=True,
                    )
                )
            rows.append(
                ComparisonRow(
                    requirement_key=requirement.key,
                    dimension=requirement.dimension,
                    label=requirement.label,
                    cells=cells,
                )
            )
        return claims, rows

    def _comparison_candidates(
        self,
        *,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
    ) -> dict[tuple[str, str], list[EvidenceItem]]:
        result: dict[tuple[str, str], list[EvidenceItem]] = defaultdict(list)
        sub_questions = {
            sub_question.id: sub_question
            for sub_question in (plan.sub_questions if plan is not None else [])
        }
        for item in ledger.items:
            sub_question = sub_questions.get(item.sub_question_id)
            entity_ids = self._bound_entity_ids(item, sub_question, entities)
            requirement_keys = self._bound_requirement_keys(
                item,
                sub_question,
                requirements,
            )
            for entity_id in entity_ids:
                for requirement_key in requirement_keys:
                    result[(entity_id, requirement_key)].append(item)
        return result

    def _bound_entity_ids(
        self,
        item: EvidenceItem,
        sub_question: Any,
        entities: list[_EntitySpec],
    ) -> list[str]:
        assignment_ids: list[str] = []
        for assignment in getattr(item, "assignments", []) or []:
            entity_id = getattr(assignment, "entity_id", None)
            if entity_id:
                assignment_ids.append(str(entity_id))
        valid_ids = {entity.entity_id for entity in entities}
        assignment_ids = [value for value in assignment_ids if value in valid_ids]
        if assignment_ids:
            return list(dict.fromkeys(assignment_ids))

        planned_ids = [
            str(value)
            for value in (getattr(sub_question, "target_entity_ids", []) or [])
            if str(value) in valid_ids
        ]
        if len(planned_ids) == 1:
            return planned_ids

        blob = f"{item.claim} {item.evidence_text}"
        # A multi-entity comparison sub-question does not itself prove that a
        # passage supports both sides. Require each entity anchor in the
        # evidence span; otherwise Self-RAG-only text could fill a CRAG cell.
        if sub_question is not None and not planned_ids:
            blob += f" {sub_question.question}"
        matched = [
            entity.entity_id
            for entity in entities
            if any(
                _contains_anchor(blob, alias)
                for alias in (entity.label, *entity.aliases)
                if alias
            )
        ]
        if planned_ids:
            return [entity_id for entity_id in matched if entity_id in planned_ids]
        return matched

    def _bound_requirement_keys(
        self,
        item: EvidenceItem,
        sub_question: Any,
        requirements: list[_RequirementSpec],
    ) -> list[str]:
        explicit: list[str] = []
        for assignment in getattr(item, "assignments", []) or []:
            key = getattr(assignment, "requirement_key", None)
            if key:
                explicit.append(str(key))
        explicit.extend(
            str(value)
            for value in (getattr(sub_question, "requirement_keys", []) or [])
        )
        dimension = getattr(sub_question, "dimension", None)
        if dimension:
            explicit.extend(
                requirement.key
                for requirement in requirements
                if normalize_text(requirement.dimension) == normalize_text(str(dimension))
            )
        valid = {requirement.key for requirement in requirements}
        explicit = [value for value in explicit if value in valid]
        if explicit:
            return list(dict.fromkeys(explicit))
        if len(requirements) == 1:
            return [requirements[0].key]

        blob = f"{item.claim} {item.evidence_text}"
        if sub_question is not None:
            blob += f" {sub_question.question}"
        blob_tokens = set(_tokens(blob))
        scored = [
            (
                len(blob_tokens & set(_tokens(f"{requirement.dimension} {requirement.label}"))),
                requirement.key,
            )
            for requirement in requirements
        ]
        best = max((score for score, _ in scored), default=0)
        return [key for score, key in scored if score == best and score > 0]

    def _build_noncomparison_claims(
        self,
        *,
        query: str,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
    ) -> list[ClaimWithCitations]:
        if not ledger.items:
            return []
        by_sub_question: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in ledger.items:
            by_sub_question[item.sub_question_id].append(item)
        ordered_ids = [
            sub_question.id
            for sub_question in (plan.sub_questions if plan is not None else [])
            if sub_question.id in by_sub_question
        ]
        ordered_ids.extend(
            sub_question_id
            for sub_question_id in by_sub_question
            if sub_question_id not in ordered_ids
        )
        claims: list[ClaimWithCitations] = []
        for sub_question_id in ordered_ids:
            selected = self._select_diverse(
                self._sorted_evidence(by_sub_question[sub_question_id]),
                limit=self.max_evidence_per_claim,
            )
            if not selected or len(claims) >= self.max_claims:
                continue
            primary = selected[0]
            claims.append(
                ClaimWithCitations(
                    claim_id=f"claim_{len(claims) + 1}",
                    text=self._grounded_claim_text(primary, query=query),
                    evidence_ids=[item.evidence_id for item in selected],
                    sub_question_id=sub_question_id,
                )
            )
        return claims

    def _grounded_claim_text(self, item: EvidenceItem, *, query: str) -> str:
        claim = " ".join(item.claim.split())
        looks_like_front_matter = (
            claim.isupper()
            or "affiliation" in claim.lower()
            or "authors" in claim.lower()
            or normalize_text(claim) == normalize_text(query)
        )
        if claim and not looks_like_front_matter:
            return _snippet(claim)

        cleaned = " ".join(item.evidence_text.split())
        abstract_match = re.search(r"\babstract\b", cleaned[:1200], re.I)
        if abstract_match is not None:
            cleaned = cleaned[abstract_match.end() :].lstrip(" :-—")
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if 35 <= len(sentence.strip()) <= 600
            and "@" not in sentence
            and not re.search(r"\b(university|affiliation|authors?)\b", sentence, re.I)
        ]
        return _snippet(sentences[0] if sentences else cleaned)

    def _write_with_llm(
        self,
        *,
        query: str,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
        is_comparison: bool,
    ) -> tuple[list[ClaimWithCitations], list[ComparisonRow]]:
        if self.llm is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("LLM is not configured")
        prompt_payload = {
            "query": query,
            "answer_type": plan.answer_type if plan is not None else "factual",
            "entities": [
                {"entity_id": entity.entity_id, "label": entity.label}
                for entity in entities
            ],
            "requirements": [
                {
                    "requirement_key": requirement.key,
                    "dimension": requirement.dimension,
                    "label": requirement.label,
                }
                for requirement in requirements
            ],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "sub_question_id": item.sub_question_id,
                    "claim": item.claim,
                    "evidence_text": item.evidence_text,
                }
                for item in ledger.items
            ],
        }
        response = self.llm.chat_json(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Return one JSON object matching the requested claims/rows schema. "
                        "Use only supplied evidence IDs. Every supported comparison cell "
                        "must bind one atomic claim, entity_id, requirement_key, and "
                        "evidence_ids. Use exact requested IDs; do not add outside facts. "
                        f"Prompt version: {WRITER_PROMPT_VERSION}."
                    ),
                ),
                ChatMessage(role="user", content=json.dumps(prompt_payload, ensure_ascii=False)),
            ],
            fast=False,
            temperature=0.0,
        )
        self.last_model = response.model
        self.last_token_usage = response.usage.model_copy()
        if not response.content:
            raise StructuredOutputError("empty writer response")
        output = parse_structured_json(response.content, _StructuredWriterOutput)
        self._validate_llm_output(
            output=output,
            plan=plan,
            ledger=ledger,
            entities=entities,
            requirements=requirements,
            is_comparison=is_comparison,
        )
        self.last_backend = "llm"
        return output.claims, output.rows

    def _validate_llm_output(
        self,
        *,
        output: _StructuredWriterOutput,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
        is_comparison: bool,
    ) -> None:
        if len(output.claims) > self.max_claims:
            raise StructuredOutputError("writer output exceeds max_claims")
        evidence_ids = {item.evidence_id for item in ledger.items}
        evidence_by_id = {item.evidence_id: item for item in ledger.items}
        requirement_keys = {requirement.key for requirement in requirements}
        requirements_by_key = {
            requirement.key: requirement for requirement in requirements
        }
        entity_ids = {entity.entity_id for entity in entities}
        entities_by_id = {entity.entity_id: entity for entity in entities}
        sub_question_ids = {
            sub_question.id
            for sub_question in (plan.sub_questions if plan is not None else [])
        }
        sub_question_ids.update(item.sub_question_id for item in ledger.items)
        allowed_by_pair = (
            self._comparison_candidates(
                plan=plan,
                ledger=ledger,
                entities=entities,
                requirements=requirements,
            )
            if is_comparison
            else {}
        )
        claim_ids: set[str] = set()
        claims_by_id: dict[str, ClaimWithCitations] = {}
        for claim in output.claims:
            if claim.claim_id in claim_ids:
                raise StructuredOutputError("duplicate claim_id")
            claim_ids.add(claim.claim_id)
            claims_by_id[claim.claim_id] = claim
            if (
                not claim.evidence_ids
                or len(claim.evidence_ids) != len(set(claim.evidence_ids))
                or len(claim.evidence_ids) > self.max_evidence_per_claim
                or not set(claim.evidence_ids) <= evidence_ids
            ):
                raise StructuredOutputError("claim contains unknown evidence ID")
            if claim.sub_question_id and claim.sub_question_id not in sub_question_ids:
                raise StructuredOutputError("claim contains unknown sub-question ID")
            if claim.requirement_key and claim.requirement_key not in requirement_keys:
                raise StructuredOutputError("claim contains unknown requirement")
            if claim.entity_id and claim.entity_id not in entity_ids:
                raise StructuredOutputError("claim contains unknown entity")
            if is_comparison:
                if (
                    claim.sub_question_id is None
                    or claim.requirement_key is None
                    or claim.entity_id is None
                    or claim.dimension is None
                ):
                    raise StructuredOutputError(
                        "comparison claim is missing a structured binding"
                    )
                requirement = requirements_by_key[claim.requirement_key]
                if claim.dimension != requirement.dimension:
                    raise StructuredOutputError("claim dimension does not match requirement")
                allowed_ids = {
                    item.evidence_id
                    for item in allowed_by_pair.get(
                        (claim.entity_id, claim.requirement_key),
                        [],
                    )
                    if item.sub_question_id == claim.sub_question_id
                }
                if not set(claim.evidence_ids) <= allowed_ids:
                    raise StructuredOutputError(
                        "claim evidence is not bound to its entity, requirement, "
                        "and sub-question"
                    )
                if any(
                    evidence_by_id[evidence_id].sub_question_id
                    != claim.sub_question_id
                    for evidence_id in claim.evidence_ids
                ):
                    raise StructuredOutputError(
                        "claim evidence does not match its sub-question"
                    )

        if not is_comparison:
            if output.rows:
                raise StructuredOutputError("non-comparison output must not contain rows")
            return
        expected = {
            (requirement.key, entity.entity_id)
            for requirement in requirements
            for entity in entities
        }
        actual: set[tuple[str, str]] = set()
        bound_claim_ids: set[str] = set()
        for row in output.rows:
            if row.requirement_key not in requirement_keys:
                raise StructuredOutputError("row contains unknown requirement")
            requirement = requirements_by_key[row.requirement_key]
            if row.dimension != requirement.dimension:
                raise StructuredOutputError("row dimension does not match requirement")
            for cell in row.cells:
                key = (row.requirement_key, cell.entity_id)
                if cell.entity_id not in entity_ids or key in actual:
                    raise StructuredOutputError("row contains unknown or duplicate entity")
                if cell.entity_label != entities_by_id[cell.entity_id].label:
                    raise StructuredOutputError("cell label does not match entity")
                actual.add(key)
                if not cell.supported:
                    if cell.claim_id is not None or cell.evidence_ids:
                        raise StructuredOutputError(
                            "unsupported cell must not reference claim evidence"
                        )
                    continue
                bound_claim = claims_by_id.get(cell.claim_id or "")
                if bound_claim is None:
                    raise StructuredOutputError("cell references unknown claim")
                if (
                    bound_claim.entity_id != cell.entity_id
                    or bound_claim.requirement_key != row.requirement_key
                    or set(cell.evidence_ids) != set(bound_claim.evidence_ids)
                    or cell.text != bound_claim.text
                ):
                    raise StructuredOutputError("cell/claim bindings do not match")
                if bound_claim.claim_id in bound_claim_ids:
                    raise StructuredOutputError(
                        "comparison claim is bound to more than one cell"
                    )
                bound_claim_ids.add(bound_claim.claim_id)
        if actual != expected:
            raise StructuredOutputError("comparison matrix is incomplete")
        if bound_claim_ids != set(claims_by_id):
            raise StructuredOutputError("comparison output contains unbound claims")

    def _comparison_specs(
        self,
        *,
        query: str,
        plan: QueryPlan | None,
        enabled: bool,
    ) -> tuple[list[_EntitySpec], list[_RequirementSpec]]:
        if not enabled:
            return [], []
        entities = self._entities_from_plan(plan)
        if len(entities) < 2:
            entities = self._entities_from_query(query)
        requirements = self._requirements_from_plan(plan)
        if not requirements:
            requirements = self._requirements_from_query(query)
        return entities[:4], requirements[:8]

    def _entities_from_plan(self, plan: QueryPlan | None) -> list[_EntitySpec]:
        if plan is None:
            return []
        result: list[_EntitySpec] = []
        for index, raw in enumerate(getattr(plan, "target_entities", []) or []):
            entity_id = _model_value(raw, "entity_id", "id", "canonical_id")
            label = _model_value(
                raw,
                "surface_name",
                "display_name",
                "label",
                "canonical_name",
                "name",
            )
            aliases = _model_list(raw, "aliases")
            if label:
                result.append(
                    _EntitySpec(
                        entity_id=str(entity_id or f"entity_{index + 1}"),
                        label=str(label),
                        aliases=tuple(str(alias) for alias in aliases),
                    )
                )
        if result:
            return _unique_entities(result)

        # Legacy comparison plans had one "What is X?" sub-question per entity.
        for index, sub_question in enumerate(plan.sub_questions):
            match = re.match(r"\s*what\s+is\s+(.+?)\s*\?\s*$", sub_question.question, re.I)
            if not match:
                continue
            label = match.group(1).strip()
            result.append(
                _EntitySpec(
                    entity_id=f"entity_{index + 1}",
                    label=label,
                    aliases=(label,),
                )
            )
        return _unique_entities(result)

    def _entities_from_query(self, query: str) -> list[_EntitySpec]:
        patterns = [
            r"(?:compare\s+)?(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?=[.?!;\n]|$)",
            r"compare\s+(.+?)\s+and\s+(.+?)(?=[.?!;\n]|$)",
            r"differences?\s+between\s+(.+?)\s+and\s+(.+?)(?=[.?!;\n]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.I)
            if not match:
                continue
            labels = [
                re.sub(r"^(?:compare|comparison of)\s+", "", value, flags=re.I).strip()
                for value in match.groups()
            ]
            return [
                _EntitySpec(
                    entity_id=f"entity_{index + 1}",
                    label=label,
                    aliases=(label,),
                )
                for index, label in enumerate(labels)
                if label
            ]
        return []

    def _requirements_from_plan(
        self,
        plan: QueryPlan | None,
    ) -> list[_RequirementSpec]:
        if plan is None:
            return []
        result: list[_RequirementSpec] = []
        for index, raw in enumerate(getattr(plan, "answer_requirements", []) or []):
            key = _model_value(raw, "requirement_key", "key", "id")
            dimension = _model_value(raw, "dimension", "name", "label")
            label = _model_value(raw, "label", "description", "dimension", "name")
            if key or dimension:
                normalized_dimension = str(dimension or key)
                result.append(
                    _RequirementSpec(
                        key=str(key or f"requirement_{index + 1}"),
                        dimension=normalized_dimension,
                        label=str(label or normalized_dimension),
                    )
                )
        if result:
            return _unique_requirements(result)

        dimensions: dict[str, str] = {}
        for sub_question in plan.sub_questions:
            dimension = getattr(sub_question, "dimension", None)
            keys = list(getattr(sub_question, "requirement_keys", []) or [])
            if dimension and keys:
                for key in keys:
                    dimensions[str(key)] = str(dimension)
        if dimensions:
            return [
                _RequirementSpec(key=key, dimension=dimension, label=_humanize(dimension))
                for key, dimension in dimensions.items()
            ]
        if plan.answer_type == "comparison":
            return [_RequirementSpec(key="overview", dimension="overview", label="Overview")]
        return []

    def _requirements_from_query(self, query: str) -> list[_RequirementSpec]:
        normalized = normalize_text(query)
        result: list[_RequirementSpec] = []
        if "retrieval trigger" in normalized or "when to retrieve" in normalized:
            result.append(
                _RequirementSpec(
                    key="retrieval_trigger",
                    dimension="retrieval_trigger",
                    label="Retrieval trigger",
                )
            )
        if "correction" in normalized or "corrective" in normalized:
            result.append(
                _RequirementSpec(
                    key="correction_mechanism",
                    dimension="correction_mechanism",
                    label="Correction mechanism",
                )
            )
        if "difference" in normalized or "compare" in normalized:
            result.append(
                _RequirementSpec(
                    key="key_differences",
                    dimension="key_differences",
                    label="Key differences",
                )
            )
        return _unique_requirements(result) or [
            _RequirementSpec(key="overview", dimension="overview", label="Overview")
        ]

    def _empty_rows(
        self,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
    ) -> list[ComparisonRow]:
        return [
            ComparisonRow(
                requirement_key=requirement.key,
                dimension=requirement.dimension,
                label=requirement.label,
                cells=[
                    ComparisonCell(entity_id=entity.entity_id, entity_label=entity.label)
                    for entity in entities
                ],
            )
            for requirement in requirements
        ]

    def _answer_status(
        self,
        *,
        claims: list[ClaimWithCitations],
        rows: list[ComparisonRow],
        verification: VerificationResult | None,
        explicitly_insufficient: bool,
    ) -> AnswerStatus:
        claims_by_id = {claim.claim_id: claim for claim in claims}
        if rows:
            supported_claim_ids = {
                cell.claim_id
                for row in rows
                for cell in row.cells
                if cell.supported
                and cell.claim_id is not None
                and cell.claim_id in claims_by_id
            }
            if not supported_claim_ids:
                return AnswerStatus.INSUFFICIENT
            matrix_complete = all(
                cell.supported
                and cell.claim_id is not None
                and cell.claim_id in claims_by_id
                for row in rows
                for cell in row.cells
            )
        else:
            if not claims:
                return AnswerStatus.INSUFFICIENT
            matrix_complete = True
        verified_complete = verification is None or verification.is_sufficient
        if matrix_complete and verified_complete and not explicitly_insufficient:
            return AnswerStatus.COMPLETE
        return AnswerStatus.PARTIAL

    def _sorted_evidence(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        return sorted(
            items,
            key=lambda item: (
                item.support_score if item.support_score is not None else -1.0,
                item.rerank_score if item.rerank_score is not None else -1.0,
                item.retrieval_score if item.retrieval_score is not None else -1.0,
            ),
            reverse=True,
        )

    def _select_diverse(
        self,
        items: list[EvidenceItem],
        *,
        limit: int,
    ) -> list[EvidenceItem]:
        if limit <= 0 or not items:
            return []
        selected: list[EvidenceItem] = []
        seen_papers: set[str] = set()
        for item in items:
            if item.paper_id in seen_papers or item.contradiction:
                continue
            selected.append(item)
            seen_papers.add(item.paper_id)
            if len(selected) >= limit:
                return selected
        for item in items:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _render_document(
        self,
        *,
        query: str,
        answer_format: str,
        core_answer: str,
        claims: list[ClaimWithCitations],
        ledger_by_id: dict[str, EvidenceItem],
        notes: list[str],
        verification: VerificationResult | None,
    ) -> str:
        lines = [
            "## Answer",
            "",
            f"**Question:** {query}",
            "",
            f"**Format:** {answer_format}",
            "",
            core_answer,
            "",
        ]
        if notes:
            lines.extend(["### Notes", ""])
            lines.extend(f"- {note}" for note in notes)
            lines.append("")
        if verification and verification.conflicting_evidence_ids:
            lines.extend(
                [
                    "### Contradictions",
                    "",
                    "Potentially conflicting evidence retained for audit:",
                    "- " + ", ".join(verification.conflicting_evidence_ids[:12]),
                    "",
                ]
            )
        cited_ids: list[str] = []
        for claim in claims:
            for evidence_id in claim.evidence_ids:
                if evidence_id in ledger_by_id and evidence_id not in cited_ids:
                    cited_ids.append(evidence_id)
        if cited_ids:
            lines.extend(["### References", ""])
            for evidence_id in cited_ids:
                item = ledger_by_id[evidence_id]
                lines.append(
                    f"- {format_inline_citation(item)} `{item.chunk_id}` · "
                    f"`{item.evidence_id}`"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _fallback_reason(self, exc: Exception) -> str:
        if isinstance(exc, StructuredOutputError):
            return "structured_output_invalid"
        return f"{type(exc).__name__}_failure"

    def _forced_fallback_reason(self, reason: str | None) -> str:
        cleaned = (reason or "forced_deterministic").strip()
        if re.fullmatch(r"[A-Za-z0-9_. -]{1,120}", cleaned):
            return cleaned
        return "forced_deterministic"


def _contains_anchor(text: str, anchor: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_anchor = normalize_text(anchor)
    if not normalized_anchor:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_anchor)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _model_value(model: Any, *names: str) -> Any:
    for name in names:
        value = getattr(model, name, None)
        if value not in (None, ""):
            return value
        if isinstance(model, dict) and model.get(name) not in (None, ""):
            return model[name]
    return None


def _model_list(model: Any, name: str) -> list[Any]:
    value = getattr(model, name, None)
    if value is None and isinstance(model, dict):
        value = model.get(name)
    return list(value or [])


def _unique_entities(entities: list[_EntitySpec]) -> list[_EntitySpec]:
    result: list[_EntitySpec] = []
    seen: set[str] = set()
    for entity in entities:
        if entity.entity_id not in seen:
            result.append(entity)
            seen.add(entity.entity_id)
    return result


def _unique_requirements(
    requirements: list[_RequirementSpec],
) -> list[_RequirementSpec]:
    result: list[_RequirementSpec] = []
    seen: set[str] = set()
    for requirement in requirements:
        if requirement.key not in seen:
            result.append(requirement)
            seen.add(requirement.key)
    return result


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()
