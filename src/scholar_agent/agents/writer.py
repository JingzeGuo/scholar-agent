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

from pydantic import BaseModel

from scholar_agent.ids import normalize_text
from scholar_agent.llm.client import ChatMessage, LLMClient
from scholar_agent.llm.structured import (
    StructuredOutputError,
    StructuredOutputErrorCode,
    parse_structured_json,
)
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

WRITER_PROMPT_VERSION = "phase11-writer-v2"
INSUFFICIENT_CELL_TEXT = "Insufficient verified evidence"
_KEY_DIFFERENCES = "key_differences"
_DERIVED_DIFFERENCE_INPUTS = ("retrieval_trigger", "correction_mechanism")
_PRIMARY_PAPERS_BY_METHOD = {
    "self-rag": frozenset({"paper_arxiv_2310_11511"}),
    "self rag": frozenset({"paper_arxiv_2310_11511"}),
    "selfrag": frozenset({"paper_arxiv_2310_11511"}),
    "corrective rag": frozenset({"paper_arxiv_2401_15884"}),
    "crag": frozenset({"paper_arxiv_2401_15884"}),
}

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
            "| Dimension | " + " | ".join(_escape_table(label) for _, label in entity_order) + " |"
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
                if matrix_cell is None or not matrix_cell.supported or rendered_claim is None:
                    rendered_cells.append(INSUFFICIENT_CELL_TEXT)
                else:
                    rendered_cells.append(
                        _escape_table(render_claim_markdown(rendered_claim, ledger_by_id))
                    )
            lines.append(f"| {_escape_table(row.label)} | " + " | ".join(rendered_cells) + " |")
        lines.append("")
    else:
        lines.append("### Claims")
        lines.append("")
        if claims:
            lines.extend(f"- {render_claim_markdown(claim, ledger_by_id)}" for claim in claims)
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
    canonical_name: str | None = None


@dataclass(frozen=True)
class _RequirementSpec:
    key: str
    dimension: str
    label: str


class _StructuredWriterOutput(BaseModel):
    claims: list[ClaimWithCitations]
    rows: list[ComparisonRow]


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
    last_fallback_fields: tuple[str, ...] = field(default=(), init=False)
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
            self.last_fallback_reason = self._forced_fallback_reason(forced_fallback_reason)
            notes.append(f"Writer used deterministic synthesis ({self.last_fallback_reason}).")
        writing_ledger = self._verified_ledger(ledger, verification)
        resolved_corpus_insufficient = (
            bool(verification.unanswerable)
            if corpus_insufficient is None and verification is not None
            else bool(corpus_insufficient)
        )
        fmt = answer_format or (plan.answer_type if plan else "factual")
        is_comparison = fmt == "comparison"
        entities, requirements = self._comparison_specs(
            query=query,
            plan=plan,
            enabled=is_comparison,
        )
        base_requirements = [
            requirement for requirement in requirements if requirement.key != _KEY_DIFFERENCES
        ]

        claims: list[ClaimWithCitations]
        rows: list[ComparisonRow]
        if not writing_ledger.items:
            claims, rows = [], self._empty_rows(entities, base_requirements)
        elif is_comparison and not base_requirements:
            claims, rows = [], []
        elif self.llm is not None and not force_deterministic:
            try:
                claims, rows = self._write_with_llm(
                    query=query,
                    plan=plan,
                    ledger=writing_ledger,
                    entities=entities,
                    requirements=base_requirements,
                    is_comparison=is_comparison,
                )
            except Exception as exc:
                self.last_fallback_reason = self._fallback_reason(exc)
                self.last_fallback_fields = (
                    exc.field_paths if isinstance(exc, StructuredOutputError) else ()
                )
                logger.warning(
                    "writer_llm_fallback reason=%s fields=%s prompt_version=%s",
                    self.last_fallback_reason,
                    list(self.last_fallback_fields),
                    WRITER_PROMPT_VERSION,
                )
                if self.strict_llm:
                    self.last_backend = "llm"
                    raise WriterLLMError(f"LLM writer failed: {self.last_fallback_reason}") from exc
                self.last_backend = "deterministic"
                notes.append(
                    f"Writer degraded to deterministic synthesis ({self.last_fallback_reason})."
                )
                claims, rows = self._write_deterministically(
                    query=query,
                    plan=plan,
                    ledger=writing_ledger,
                    entities=entities,
                    requirements=base_requirements,
                    is_comparison=is_comparison,
                )
        else:
            claims, rows = self._write_deterministically(
                query=query,
                plan=plan,
                ledger=writing_ledger,
                entities=entities,
                requirements=base_requirements,
                is_comparison=is_comparison,
            )

        if is_comparison:
            claims, rows = self._with_derived_key_differences(
                plan=plan,
                claims=claims,
                rows=rows,
                entities=entities,
                requirements=requirements,
            )

        status = self._answer_status(
            claims=claims,
            rows=rows,
            verification=verification,
            explicitly_insufficient=resolved_corpus_insufficient,
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
            corpus_insufficient=resolved_corpus_insufficient,
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
        self.last_fallback_fields = ()
        self.last_token_usage = TokenUsage()

    def _notes(
        self,
        *,
        verification: VerificationResult | None,
        contradiction_notes: list[str] | None,
    ) -> list[str]:
        notes = list(contradiction_notes or [])
        if not contradiction_notes and verification and verification.conflicting_evidence_ids:
            notes.append(
                "Conflicting evidence retained: "
                + ", ".join(verification.conflicting_evidence_ids[:12])
            )
        if verification and verification.missing_aspects:
            notes.append(
                "Missing aspects after verification: " + "; ".join(verification.missing_aspects[:8])
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
        return EvidenceLedger(items=[item for item in ledger.items if item.evidence_id in allowed])

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
                if not items or len(claims) >= self.max_claims:
                    cells.append(
                        ComparisonCell(
                            entity_id=entity.entity_id,
                            entity_label=entity.label,
                        )
                    )
                    continue
                primary = self._best_comparison_item(
                    items,
                    entity=entity,
                    requirement=requirement,
                )
                strict_claim = (
                    self._deterministic_atomic_claim(
                        primary,
                        entity=entity,
                        requirement=requirement,
                        require_primary_source=True,
                    )
                    if self._has_structured_comparison_bindings(plan)
                    else None
                )
                claim_text = (
                    strict_claim
                    if strict_claim is not None
                    else (
                        None
                        if self._has_structured_comparison_bindings(plan)
                        else self._grounded_claim_text(primary, query=query)
                    )
                )
                if not claim_text:
                    cells.append(
                        ComparisonCell(
                            entity_id=entity.entity_id,
                            entity_label=entity.label,
                        )
                    )
                    continue
                claim_id = f"claim_{len(claims) + 1}"
                claim = ClaimWithCitations(
                    claim_id=claim_id,
                    text=claim_text,
                    # One exact passage is safer than attaching diverse passages
                    # that may not each support the same atomic statement.
                    evidence_ids=[primary.evidence_id],
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

    def _best_comparison_item(
        self,
        items: list[EvidenceItem],
        *,
        entity: _EntitySpec,
        requirement: _RequirementSpec,
    ) -> EvidenceItem:
        """Prefer the passage with the strongest dimension-specific cues."""

        def score(item: EvidenceItem) -> tuple[int, float, float, str]:
            atomic = self._deterministic_atomic_claim(
                item,
                entity=entity,
                requirement=requirement,
                require_primary_source=True,
            )
            return (
                _dimension_cue_score(atomic or "", requirement.dimension),
                float(item.support_score or 0.0),
                float(item.retrieval_score or 0.0),
                item.evidence_id,
            )

        return max(items, key=score)

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
        entities_by_id = {entity.entity_id: entity for entity in entities}
        requirements_by_key = {requirement.key: requirement for requirement in requirements}
        structured_plan = self._has_structured_comparison_bindings(plan)
        planned_pairs: dict[tuple[str, str], Any] = {}
        if structured_plan:
            for sub_question in sub_questions.values():
                if (
                    len(sub_question.target_entity_ids) == 1
                    and len(sub_question.requirement_keys) == 1
                    and sub_question.requirement_keys[0] != _KEY_DIFFERENCES
                ):
                    planned_pairs[
                        (
                            sub_question.target_entity_ids[0],
                            sub_question.requirement_keys[0],
                        )
                    ] = sub_question
        for item in ledger.items:
            if structured_plan:
                for (entity_id, requirement_key), sub_question in planned_pairs.items():
                    entity = entities_by_id.get(entity_id)
                    requirement = requirements_by_key.get(requirement_key)
                    if (
                        entity is None
                        or requirement is None
                        or sub_question.dimension != requirement.dimension
                        or self._deterministic_atomic_claim(
                            item,
                            entity=entity,
                            requirement=requirement,
                            require_primary_source=True,
                        )
                        is None
                    ):
                        continue
                    # Legacy ledgers stored only the first sub-question ID when
                    # the same exact span was reused. Rebind a local copy only
                    # after all primary-paper/entity/dimension/sentence gates
                    # pass; provenance and evidence identity remain unchanged.
                    rebound = (
                        item
                        if item.sub_question_id == sub_question.id
                        else item.model_copy(update={"sub_question_id": sub_question.id})
                    )
                    if not any(
                        existing.evidence_id == rebound.evidence_id
                        for existing in result[(entity_id, requirement_key)]
                    ):
                        result[(entity_id, requirement_key)].append(rebound)
                continue

            # Legacy comparison plans predate structured entity/requirement refs.
            # Never use this inference path for a modern plan.
            legacy_sub_question = sub_questions.get(item.sub_question_id)
            entity_ids = self._bound_entity_ids(item, legacy_sub_question, entities)
            requirement_keys = self._bound_requirement_keys(
                item,
                legacy_sub_question,
                requirements,
            )
            for entity_id in entity_ids:
                for requirement_key in requirement_keys:
                    result[(entity_id, requirement_key)].append(item)
        return result

    def _has_structured_comparison_bindings(self, plan: QueryPlan | None) -> bool:
        if plan is None or not plan.target_entities or not plan.answer_requirements:
            return False
        return any(
            sub_question.target_entity_ids and sub_question.requirement_keys
            for sub_question in plan.sub_questions
        )

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
                _contains_anchor(blob, alias) for alias in (entity.label, *entity.aliases) if alias
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
            str(value) for value in (getattr(sub_question, "requirement_keys", []) or [])
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

    def _deterministic_atomic_claim(
        self,
        item: EvidenceItem,
        *,
        entity: _EntitySpec,
        requirement: _RequirementSpec,
        require_primary_source: bool,
    ) -> str | None:
        """Select one complete, provenance-exact sentence after hard gates."""
        if require_primary_source:
            expected_papers = self._primary_papers(entity)
            if expected_papers is not None and item.paper_id not in expected_papers:
                return None

        candidates: list[str] = []
        for source in (item.claim, item.evidence_text):
            for sentence in _complete_sentences(source):
                if sentence not in candidates:
                    candidates.append(sentence)

        valid: list[str] = []
        normalized_evidence = normalize_text(item.evidence_text)
        for candidate in candidates:
            normalized_candidate = normalize_text(candidate)
            if (
                normalized_candidate not in normalized_evidence
                or _looks_like_nonclaim(candidate)
                or not any(
                    _contains_anchor(candidate, alias)
                    for alias in (entity.label, *entity.aliases)
                    if alias
                )
                or not _matches_dimension(candidate, requirement.dimension)
            ):
                continue
            valid.append(candidate)
        if not valid:
            return None
        return max(
            valid,
            key=lambda candidate: (
                _dimension_cue_score(candidate, requirement.dimension),
                -len(candidate),
            ),
        )

    def _primary_papers(self, entity: _EntitySpec) -> frozenset[str] | None:
        for name in [entity.canonical_name, entity.label, *entity.aliases]:
            if not name:
                continue
            expected = _PRIMARY_PAPERS_BY_METHOD.get(normalize_text(name))
            if expected is not None:
                return expected
        return None

    def _with_derived_key_differences(
        self,
        *,
        plan: QueryPlan | None,
        claims: list[ClaimWithCitations],
        rows: list[ComparisonRow],
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
    ) -> tuple[list[ClaimWithCitations], list[ComparisonRow]]:
        difference_requirement = next(
            (requirement for requirement in requirements if requirement.key == _KEY_DIFFERENCES),
            None,
        )
        if difference_requirement is None:
            return claims, rows

        rows_by_key = {
            row.requirement_key: row for row in rows if row.requirement_key != _KEY_DIFFERENCES
        }
        claims_by_id = {claim.claim_id: claim for claim in claims}
        prerequisite_claims: dict[tuple[str, str], ClaimWithCitations] = {}
        complete = True
        for requirement_key in _DERIVED_DIFFERENCE_INPUTS:
            row = rows_by_key.get(requirement_key)
            cells_by_entity = (
                {cell.entity_id: cell for cell in row.cells} if row is not None else {}
            )
            for entity in entities:
                cell = cells_by_entity.get(entity.entity_id)
                claim = (
                    claims_by_id.get(cell.claim_id or "")
                    if cell is not None and cell.supported
                    else None
                )
                if (
                    claim is None
                    or claim.entity_id != entity.entity_id
                    or claim.requirement_key != requirement_key
                ):
                    complete = False
                    continue
                prerequisite_claims[(requirement_key, entity.entity_id)] = claim

        derived_cells: list[ComparisonCell] = []
        if complete and len(entities) == 2 and len(claims) + len(entities) <= self.max_claims:
            difference_sub_question_id = _difference_sub_question_id(plan)
            used_claim_ids = set(claims_by_id)
            synthesis_inputs = [
                prerequisite_claims[(requirement_key, entity.entity_id)]
                for requirement_key in _DERIVED_DIFFERENCE_INPUTS
                for entity in entities
            ]
            # A derived contrast is jointly supported. Cite at least one
            # passage from each of the four prerequisite cells rather than
            # treating a direct comparison hit as independent evidence.
            joint_evidence_ids: list[str] = []
            for source_claim in synthesis_inputs:
                if not source_claim.evidence_ids:
                    complete = False
                    break
                evidence_id = source_claim.evidence_ids[0]
                if evidence_id not in joint_evidence_ids:
                    joint_evidence_ids.append(evidence_id)

            for entity in entities:
                if not complete:
                    break
                other = next(
                    candidate for candidate in entities if candidate.entity_id != entity.entity_id
                )
                trigger = prerequisite_claims[("retrieval_trigger", entity.entity_id)]
                correction = prerequisite_claims[("correction_mechanism", entity.entity_id)]
                other_trigger = prerequisite_claims[("retrieval_trigger", other.entity_id)]
                other_correction = prerequisite_claims[("correction_mechanism", other.entity_id)]
                claim_id = _next_claim_id(used_claim_ids)
                used_claim_ids.add(claim_id)
                text = (
                    f"{entity.label} differs from {other.label} across the requested "
                    f"retrieval and correction dimensions: {trigger.text} "
                    f"{correction.text} By contrast, {other_trigger.text} "
                    f"{other_correction.text}"
                )
                derived = ClaimWithCitations(
                    claim_id=claim_id,
                    text=text,
                    evidence_ids=list(joint_evidence_ids),
                    sub_question_id=difference_sub_question_id,
                    requirement_key=_KEY_DIFFERENCES,
                    entity_id=entity.entity_id,
                    dimension=_KEY_DIFFERENCES,
                )
                claims.append(derived)
                derived_cells.append(
                    ComparisonCell(
                        entity_id=entity.entity_id,
                        entity_label=entity.label,
                        text=derived.text,
                        evidence_ids=list(derived.evidence_ids),
                        claim_id=derived.claim_id,
                        supported=True,
                    )
                )

        if not complete or len(derived_cells) != len(entities):
            # The derived comparison is all-or-nothing.
            derived_ids = {cell.claim_id for cell in derived_cells if cell.claim_id}
            claims = [claim for claim in claims if claim.claim_id not in derived_ids]
            derived_cells = [
                ComparisonCell(
                    entity_id=entity.entity_id,
                    entity_label=entity.label,
                )
                for entity in entities
            ]

        derived_row = ComparisonRow(
            requirement_key=_KEY_DIFFERENCES,
            dimension=_KEY_DIFFERENCES,
            label=difference_requirement.label,
            cells=derived_cells,
        )
        ordered_rows: list[ComparisonRow] = []
        for requirement in requirements:
            if requirement.key == _KEY_DIFFERENCES:
                ordered_rows.append(derived_row)
                continue
            row = rows_by_key.get(requirement.key)
            if row is not None:
                ordered_rows.append(row)
            else:
                ordered_rows.extend(self._empty_rows(entities, [requirement]))
        return claims, ordered_rows

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

    def _llm_output_example(
        self,
        *,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
        entities: list[_EntitySpec],
        requirements: list[_RequirementSpec],
        is_comparison: bool,
    ) -> dict[str, Any]:
        """Build a schema example containing only IDs allowed for this request."""
        if not is_comparison:
            item = ledger.items[0]
            return {
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "text": _snippet(item.claim or item.evidence_text),
                        "evidence_ids": [item.evidence_id],
                        "sub_question_id": item.sub_question_id,
                        "requirement_key": None,
                        "entity_id": None,
                        "dimension": None,
                    }
                ],
                "rows": [],
            }

        candidates = self._comparison_candidates(
            plan=plan,
            ledger=ledger,
            entities=entities,
            requirements=requirements,
        )
        claims: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for requirement in requirements:
            cells: list[dict[str, Any]] = []
            for entity in entities:
                items = candidates.get((entity.entity_id, requirement.key), [])
                if not items or len(claims) >= self.max_claims:
                    cells.append(
                        {
                            "entity_id": entity.entity_id,
                            "entity_label": entity.label,
                            "text": INSUFFICIENT_CELL_TEXT,
                            "evidence_ids": [],
                            "claim_id": None,
                            "supported": False,
                        }
                    )
                    continue
                item = self._best_comparison_item(
                    items,
                    entity=entity,
                    requirement=requirement,
                )
                claim_id = f"claim_{len(claims) + 1}"
                claim_text = (
                    "REPLACE_WITH_COMPLETE_EVIDENCE_BACKED_CLAIM_FOR_"
                    f"{entity.label}_{requirement.key}"
                )
                claims.append(
                    {
                        "claim_id": claim_id,
                        "text": claim_text,
                        "evidence_ids": [item.evidence_id],
                        "sub_question_id": item.sub_question_id,
                        "requirement_key": requirement.key,
                        "entity_id": entity.entity_id,
                        "dimension": requirement.dimension,
                    }
                )
                cells.append(
                    {
                        "entity_id": entity.entity_id,
                        "entity_label": entity.label,
                        "text": claim_text,
                        "evidence_ids": [item.evidence_id],
                        "claim_id": claim_id,
                        "supported": True,
                    }
                )
            rows.append(
                {
                    "requirement_key": requirement.key,
                    "dimension": requirement.dimension,
                    "label": requirement.label,
                    "cells": cells,
                }
            )
        return {"claims": claims, "rows": rows}

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
        output_example = self._llm_output_example(
            plan=plan,
            ledger=ledger,
            entities=entities,
            requirements=requirements,
            is_comparison=is_comparison,
        )
        prompt_items = list(ledger.items)
        allowed_bindings: list[dict[str, Any]] = []
        if is_comparison:
            allowed_by_pair = self._comparison_candidates(
                plan=plan,
                ledger=ledger,
                entities=entities,
                requirements=requirements,
            )
            prompt_item_ids: set[str] = set()
            for requirement in requirements:
                for entity in entities:
                    candidates = allowed_by_pair.get(
                        (entity.entity_id, requirement.key),
                        [],
                    )
                    prompt_item_ids.update(item.evidence_id for item in candidates)
                    allowed_bindings.append(
                        {
                            "entity_id": entity.entity_id,
                            "requirement_key": requirement.key,
                            "dimension": requirement.dimension,
                            "sub_question_id": (
                                candidates[0].sub_question_id if candidates else None
                            ),
                            "allowed_evidence_ids": [
                                item.evidence_id for item in candidates
                            ],
                        }
                    )
            # Do not invite the model to select globally verified but
            # entity/dimension-ineligible passages. The deterministic binding
            # map above remains the authority.
            prompt_items = [
                item for item in ledger.items if item.evidence_id in prompt_item_ids
            ]
        prompt_payload = {
            "query": query,
            "answer_type": plan.answer_type if plan is not None else "factual",
            "entities": [
                {"entity_id": entity.entity_id, "label": entity.label} for entity in entities
            ],
            "requirements": [
                {
                    "requirement_key": requirement.key,
                    "dimension": requirement.dimension,
                    "label": requirement.label,
                }
                for requirement in requirements
            ],
            "allowed_bindings": allowed_bindings,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "sub_question_id": item.sub_question_id,
                    "claim": item.claim,
                    "evidence_text": item.evidence_text,
                }
                for item in prompt_items
            ],
        }
        response = self.llm.chat_json(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Return one JSON object. The top-level claims and rows arrays and "
                        "all keys shown in the output contract are required. For a "
                        "non-comparison answer, rows must be an empty array. "
                        "Use only supplied evidence IDs. Every supported comparison cell "
                        "must bind one atomic claim, entity_id, requirement_key, and "
                        "evidence_ids. For each supported comparison claim and cell, use "
                        "exactly one evidence ID from that pair's allowed_bindings entry "
                        "and copy its sub_question_id exactly. Never combine evidence IDs "
                        "across bindings. Use exact requested IDs; do not add outside facts. "
                        "The output example's REPLACE_WITH... text values are structural "
                        "placeholders. Replace every one with a complete, readable, atomic "
                        "sentence synthesized from the bound evidence. Each sentence must "
                        "explicitly name its entity and directly explain the requested "
                        "dimension. Never emit placeholders, ellipses, or truncated excerpts. "
                        "Do not emit a key_differences row or claim: the application "
                        "derives it only after both entities have verified retrieval-trigger "
                        "and correction-mechanism cells. "
                        "For an unsupported cell, use the exact insufficient-evidence "
                        "text, an empty evidence_ids array, a null claim_id, and "
                        "supported=false. Output contract example: "
                        + json.dumps(
                            output_example,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + f" Prompt version: {WRITER_PROMPT_VERSION}."
                    ),
                ),
                ChatMessage(role="user", content=json.dumps(prompt_payload, ensure_ascii=False)),
            ],
            fast=False,
            temperature=0.0,
        )
        self.last_model = response.model
        self.last_token_usage = response.usage.model_copy()
        failure: tuple[StructuredOutputErrorCode, tuple[str, ...]] | None = None
        output: _StructuredWriterOutput | None = None
        try:
            output = parse_structured_json(
                response.content or "",
                _StructuredWriterOutput,
            )
            self._validate_llm_output(
                output=output,
                plan=plan,
                ledger=ledger,
                entities=entities,
                requirements=requirements,
                is_comparison=is_comparison,
            )
        except StructuredOutputError as exc:
            failure = (exc.code, exc.field_paths)
        if failure is not None:
            del response
            output = None
            code, field_paths = failure
            raise StructuredOutputError(code, field_paths=field_paths)
        assert output is not None
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
            raise StructuredOutputError(
                StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                field_paths=("claims",),
            )
        evidence_ids = {item.evidence_id for item in ledger.items}
        requirement_keys = {requirement.key for requirement in requirements}
        requirements_by_key = {requirement.key: requirement for requirement in requirements}
        entity_ids = {entity.entity_id for entity in entities}
        entities_by_id = {entity.entity_id: entity for entity in entities}
        sub_question_ids = {
            sub_question.id for sub_question in (plan.sub_questions if plan is not None else [])
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
        for claim_index, claim in enumerate(output.claims):
            claim_path = f"claims[{claim_index}]"
            if claim.claim_id in claim_ids:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                    field_paths=(f"{claim_path}.claim_id",),
                )
            claim_ids.add(claim.claim_id)
            claims_by_id[claim.claim_id] = claim
            if not claim.evidence_ids:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.MISSING_REQUIRED_FIELD,
                    field_paths=(f"{claim_path}.evidence_ids",),
                )
            if (
                len(claim.evidence_ids) != len(set(claim.evidence_ids))
                or len(claim.evidence_ids) > self.max_evidence_per_claim
            ):
                raise StructuredOutputError(
                    StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                    field_paths=(f"{claim_path}.evidence_ids",),
                )
            if not set(claim.evidence_ids) <= evidence_ids:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.UNKNOWN_EVIDENCE_ID,
                    field_paths=(f"{claim_path}.evidence_ids",),
                )
            if claim.sub_question_id and claim.sub_question_id not in sub_question_ids:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                    field_paths=(f"{claim_path}.sub_question_id",),
                )
            if claim.requirement_key and claim.requirement_key not in requirement_keys:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.UNKNOWN_REQUIREMENT_KEY,
                    field_paths=(f"{claim_path}.requirement_key",),
                )
            if claim.entity_id and claim.entity_id not in entity_ids:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.UNKNOWN_ENTITY_ID,
                    field_paths=(f"{claim_path}.entity_id",),
                )
            if is_comparison:
                missing_paths = [
                    f"{claim_path}.{field_name}"
                    for field_name in (
                        "sub_question_id",
                        "requirement_key",
                        "entity_id",
                        "dimension",
                    )
                    if getattr(claim, field_name) is None
                ]
                if missing_paths:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.MISSING_REQUIRED_FIELD,
                        field_paths=missing_paths,
                    )
                requirement_key = claim.requirement_key
                entity_id = claim.entity_id
                sub_question_id = claim.sub_question_id
                assert requirement_key is not None
                assert entity_id is not None
                assert sub_question_id is not None
                requirement = requirements_by_key[requirement_key]
                if claim.dimension != requirement.dimension:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{claim_path}.dimension",),
                    )
                entity = entities_by_id[entity_id]
                if (
                    "REPLACE_WITH" in claim.text.upper()
                    or "…" in claim.text
                    or claim.text.rstrip().endswith("...")
                    or not any(
                        _contains_anchor(claim.text, alias)
                        for alias in (entity.label, *entity.aliases)
                        if alias
                    )
                    or (
                        requirement.key
                        in {"retrieval_trigger", "correction_mechanism"}
                        and not _matches_dimension(claim.text, requirement.dimension)
                    )
                ):
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{claim_path}.text",),
                    )
                allowed_ids = {
                    item.evidence_id
                    for item in allowed_by_pair.get(
                        (entity_id, requirement_key),
                        [],
                    )
                    if item.sub_question_id == sub_question_id
                }
                if not set(claim.evidence_ids) <= allowed_ids:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{claim_path}.evidence_ids",),
                    )
        if not is_comparison:
            if output.rows:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                    field_paths=("rows",),
                )
            return
        expected = {
            (requirement.key, entity.entity_id)
            for requirement in requirements
            for entity in entities
        }
        actual: set[tuple[str, str]] = set()
        bound_claim_ids: set[str] = set()
        for row_index, row in enumerate(output.rows):
            row_path = f"rows[{row_index}]"
            if row.requirement_key not in requirement_keys:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.UNKNOWN_REQUIREMENT_KEY,
                    field_paths=(f"{row_path}.requirement_key",),
                )
            requirement = requirements_by_key[row.requirement_key]
            if row.dimension != requirement.dimension:
                raise StructuredOutputError(
                    StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                    field_paths=(f"{row_path}.dimension",),
                )
            for cell_index, cell in enumerate(row.cells):
                cell_path = f"{row_path}.cells[{cell_index}]"
                key = (row.requirement_key, cell.entity_id)
                if cell.entity_id not in entity_ids:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.UNKNOWN_ENTITY_ID,
                        field_paths=(f"{cell_path}.entity_id",),
                    )
                if key in actual:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{cell_path}.entity_id",),
                    )
                if cell.entity_label != entities_by_id[cell.entity_id].label:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{cell_path}.entity_label",),
                    )
                actual.add(key)
                if not cell.supported:
                    if cell.claim_id is not None or cell.evidence_ids:
                        raise StructuredOutputError(
                            StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                            field_paths=(
                                f"{cell_path}.claim_id",
                                f"{cell_path}.evidence_ids",
                            ),
                        )
                    continue
                bound_claim = claims_by_id.get(cell.claim_id or "")
                if bound_claim is None:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{cell_path}.claim_id",),
                    )
                if (
                    bound_claim.entity_id != cell.entity_id
                    or bound_claim.requirement_key != row.requirement_key
                    or set(cell.evidence_ids) != set(bound_claim.evidence_ids)
                    or cell.text != bound_claim.text
                ):
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(cell_path,),
                    )
                if bound_claim.claim_id in bound_claim_ids:
                    raise StructuredOutputError(
                        StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                        field_paths=(f"{cell_path}.claim_id",),
                    )
                bound_claim_ids.add(bound_claim.claim_id)
        if actual != expected:
            raise StructuredOutputError(
                StructuredOutputErrorCode.MISSING_REQUIRED_FIELD,
                field_paths=("rows",),
            )
        if bound_claim_ids != set(claims_by_id):
            raise StructuredOutputError(
                StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED,
                field_paths=("claims",),
            )

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
            canonical_name = _model_value(raw, "canonical_name")
            if label:
                result.append(
                    _EntitySpec(
                        entity_id=str(entity_id or f"entity_{index + 1}"),
                        label=str(label),
                        aliases=tuple(str(alias) for alias in aliases),
                        canonical_name=(
                            str(canonical_name) if canonical_name is not None else None
                        ),
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
                if cell.supported and cell.claim_id is not None and cell.claim_id in claims_by_id
            }
            if not supported_claim_ids:
                return AnswerStatus.INSUFFICIENT
            matrix_complete = all(
                cell.supported and cell.claim_id is not None and cell.claim_id in claims_by_id
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
                    f"- {format_inline_citation(item)} `{item.chunk_id}` · `{item.evidence_id}`"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _fallback_reason(self, exc: Exception) -> str:
        if isinstance(exc, StructuredOutputError):
            return exc.code.value
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


def _complete_sentences(text: str) -> list[str]:
    """Return only bounded, visibly complete sentences from one evidence span."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return []
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", cleaned):
        candidate = part.strip()
        if (
            24 <= len(candidate) <= 420
            and candidate[-1:] in {".", "!", "?"}
            and "…" not in candidate
            and not candidate.endswith("...")
        ):
            sentences.append(candidate)
    return sentences


def _looks_like_nonclaim(text: str) -> bool:
    """Reject common PDF front matter and visibly clipped pseudo-claims."""
    normalized = normalize_text(text)
    if len(text) < 24 or len(text) > 420 or "…" in text or text.endswith("...") or text.isupper():
        return True
    nonclaim_markers = (
        "acknowledg",
        "would like to thank",
        "references",
        "bibliography",
        "authors and affiliations",
        "university",
        "copyright",
        "all rights reserved",
    )
    return any(marker in normalized for marker in nonclaim_markers)


def _matches_dimension(text: str, dimension: str) -> bool:
    normalized = normalize_text(dimension).replace("_", " ")
    lower = normalize_text(text)
    if normalized == "retrieval trigger":
        has_retrieval = bool(re.search(r"\bretriev[a-z]*\b", lower))
        has_trigger_cue = bool(
            re.search(
                r"\b(?:when|trigger[a-z]*|decid[a-z]*|on demand|"
                r"reflection token[a-z]*|retrieval evaluator|classif[a-z]*|"
                r"relevance)\b",
                lower,
            )
        ) or all(
            re.search(rf"\b{category}\b", lower)
            for category in ("correct", "ambiguous", "incorrect")
        )
        return has_retrieval and has_trigger_cue
    if normalized == "correction mechanism":
        return bool(
            re.search(
                r"\b(?:refin[a-z]*|web search|decompos[a-z]*|recompos[a-z]*|"
                r"critiqu[a-z]*|critic[a-z]*|correct(?:s|ed|ing)?\s+"
                r"(?:generated|generation|response|output|error|low-quality|"
                r"retrieval))\b",
                lower,
            )
        )
    return bool(set(_tokens(text)) & set(_tokens(dimension)))


def _dimension_cue_score(text: str, dimension: str) -> int:
    normalized = normalize_text(dimension).replace("_", " ")
    lower = normalize_text(text)
    cues: tuple[str, ...]
    if normalized == "retrieval trigger":
        cues = (
            r"\bretriev[a-z]*\b",
            r"\bon demand\b",
            r"\bwhen\b",
            r"\btrigger[a-z]*\b",
            r"\bdecid[a-z]*\b",
            r"\breflection token[a-z]*\b",
            r"\bretrieval evaluator\b",
            r"\b(?:correct|ambiguous|incorrect)\b",
        )
    elif normalized == "correction mechanism":
        cues = (
            r"\brefin[a-z]*\b",
            r"\bweb search\b",
            r"\bdecompos[a-z]*\b",
            r"\brecompos[a-z]*\b",
            r"\bcritiqu[a-z]*\b",
            r"\bcorrect(?:s|ed|ing)?\s+(?:generated|generation|response|output|"
            r"error|low-quality|retrieval)\b",
        )
    else:
        return len(set(_tokens(text)) & set(_tokens(dimension)))
    return sum(bool(re.search(cue, lower)) for cue in cues)


def _difference_sub_question_id(plan: QueryPlan | None) -> str | None:
    if plan is None:
        return None
    for sub_question in plan.sub_questions:
        if (
            sub_question.dimension == _KEY_DIFFERENCES
            or _KEY_DIFFERENCES in sub_question.requirement_keys
        ):
            return sub_question.id
    return None


def _next_claim_id(used: set[str]) -> str:
    index = 1
    while f"claim_{index}" in used:
        index += 1
    return f"claim_{index}"


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
