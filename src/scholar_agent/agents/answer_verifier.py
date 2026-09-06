"""LLM-based validation of the final grounded answer."""

from __future__ import annotations

from typing import Any

from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState


def _answer_verifier_prompt(state: AgentState) -> str:
    requirements = "\n".join(
        f'{item["id"]}: {item["description"]}'
        for item in state["plan"]["requirements"]
    )
    evidence = "\n".join(
        f'E{index} [{item["paper"]} p.{item["page"]}]: {item["text"]}'
        for index, item in enumerate(state["evidence"], start=1)
    )
    return f"""You verify a final answer in an evidence-grounded academic workflow.

Use only the question, requirements, and supplied evidence. Do not add outside knowledge.
The earlier coverage analysis is not authoritative and is intentionally omitted.

Return one JSON object with exactly these fields:
- "requirements": requirement ID -> {{"passed": boolean, "issue": string}}
- "citation_issues": list of concise strings
- "uncited_claims": list of factual claims that need a citation
- "unsupported_claims": list of factual claims unsupported by the supplied evidence
- "incorrect_missing_claims": list of claims that evidence is missing when supplied evidence
  actually supports the requested item
- "repair_instructions": list of concise evidence-grounded fixes

Rules:
- Check every requirement against the answer and evidence.
- A requirement passes when it is correctly answered, or when the answer clearly identifies
  that it cannot be answered and the supplied evidence truly lacks support.
- Every factual sentence, including summaries and restatements, needs a citation.
- Each citation must directly support its local claim.
- Do not fail a correct abstention merely because it has no citation.
- Keep every issue list empty when no such issue exists.

Question: {state["question"]}

Requirements:
{requirements}

Evidence:
{evidence}

Final answer:
{state["answer"]}
"""


def _strings(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _parse_verification(payload: dict[str, Any], state: AgentState) -> dict[str, Any]:
    requirement_ids = [item["id"] for item in state["plan"]["requirements"]]
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, dict) or not set(requirement_ids).issubset(
        raw_requirements,
    ):
        raise ValueError("requirements must contain every planned requirement")

    requirements: dict[str, dict[str, Any]] = {}
    for requirement_id in requirement_ids:
        value = raw_requirements[requirement_id]
        if not isinstance(value, dict) or not {"passed", "issue"}.issubset(value):
            raise ValueError(f"Invalid requirement check: {requirement_id}")
        passed = value["passed"]
        issue = value["issue"]
        if not isinstance(passed, bool) or not isinstance(issue, str):
            raise ValueError(f"Invalid requirement check: {requirement_id}")
        requirements[requirement_id] = {"passed": passed, "issue": issue.strip()}

    list_fields = (
        "citation_issues",
        "uncited_claims",
        "unsupported_claims",
        "incorrect_missing_claims",
        "repair_instructions",
    )
    result: dict[str, Any] = {
        "requirements": requirements,
        **{field: _strings(payload.get(field), field) for field in list_fields},
        "error": "",
    }
    result["repair_required"] = any(
        not item["passed"] for item in requirements.values()
    ) or any(result[field] for field in list_fields[:-1])
    result["passed"] = not result["repair_required"]
    return result


def answer_verifier_node(state: AgentState, llm: LLMClient) -> dict:
    """Check the final answer without turning malformed output into a hard gate."""
    try:
        payload = llm.complete_json(_answer_verifier_prompt(state))
        verification = _parse_verification(payload, state)
    except ValueError as exc:
        verification = {
            "passed": None,
            "repair_required": False,
            "requirements": {},
            "citation_issues": [],
            "uncited_claims": [],
            "unsupported_claims": [],
            "incorrect_missing_claims": [],
            "repair_instructions": [],
            "error": str(exc),
        }
    return {"answer_verification": verification}
