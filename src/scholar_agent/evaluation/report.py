"""Evaluation report writers: JSON/CSV aggregates + optional charts."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SystemSummary(BaseModel):
    system: str
    n_questions: int = 0
    n_errors: int = 0
    error_rate: float = 0.0
    recall_at_k: float = 0.0
    recall_at_k_paper: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    graph_evidence_recall: float | None = None
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_validity_rate: float = 0.0
    page_traceability_rate: float = 0.0
    claim_overlap: float = 0.0
    claim_correctness: float = 0.0
    completeness: float = 0.0
    token_f1: float = 0.0
    refusal_correct: float = 0.0
    faithfulness_proxy: float = 0.0
    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None
    ragas_coverage_rate: float = 0.0
    contradiction_handling_accuracy: float | None = None
    contradiction_metric_coverage_rate: float = 0.0
    plan_coverage: float | None = None
    plan_coverage_metric_coverage_rate: float = 0.0
    tool_selection_accuracy: float | None = None
    tool_selection_metric_coverage_rate: float = 0.0
    corrective_trigger_precision: float | None = None
    corrective_trigger_metric_coverage_rate: float = 0.0
    improvement_after_correction: float | None = None
    correction_improvement_metric_coverage_rate: float = 0.0
    unique_useful_evidence_per_tool_call: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tool_calls: float = 0.0
    avg_iterations: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_tokens: float = 0.0
    total_estimated_cost_usd: float = 0.0
    by_type: dict[str, dict[str, float | None]] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    run_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    frozen_split_fingerprint: str | None = None
    config_fingerprint_sha256: str | None = None
    systems: list[SystemSummary] = Field(default_factory=list)
    per_question: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def write_report(
    report: EvaluationReport,
    output_dir: Path | str,
    *,
    write_charts: bool = True,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    config_path = out / "run_config.json"
    config_path.write_text(
        json.dumps(report.config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["run_config_json"] = config_path

    json_path = out / "results.json"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["results_json"] = json_path

    # Aggregate CSV
    agg_path = out / "aggregate_metrics.csv"
    fieldnames = [
        "system",
        "n_questions",
        "n_errors",
        "error_rate",
        "recall_at_k",
        "recall_at_k_paper",
        "mrr",
        "ndcg_at_k",
        "graph_evidence_recall",
        "citation_precision",
        "citation_recall",
        "citation_validity_rate",
        "page_traceability_rate",
        "claim_overlap",
        "claim_correctness",
        "completeness",
        "token_f1",
        "refusal_correct",
        "faithfulness_proxy",
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_coverage_rate",
        "contradiction_handling_accuracy",
        "contradiction_metric_coverage_rate",
        "plan_coverage",
        "plan_coverage_metric_coverage_rate",
        "tool_selection_accuracy",
        "tool_selection_metric_coverage_rate",
        "corrective_trigger_precision",
        "corrective_trigger_metric_coverage_rate",
        "improvement_after_correction",
        "correction_improvement_metric_coverage_rate",
        "unique_useful_evidence_per_tool_call",
        "avg_latency_ms",
        "avg_tool_calls",
        "avg_iterations",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_tokens",
        "total_estimated_cost_usd",
    ]
    with agg_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in report.systems:
            writer.writerow({k: getattr(s, k) for k in fieldnames})
    paths["aggregate_csv"] = agg_path

    # Per-question CSV
    pq_path = out / "per_question_metrics.csv"
    if report.per_question:
        keys: list[str] = sorted({k for row in report.per_question for k in row})
        with pq_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in report.per_question:
                writer.writerow(row)
        paths["per_question_csv"] = pq_path

    # Failures
    fail_path = out / "failures.json"
    fail_path.write_text(
        json.dumps(report.failures, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["failures_json"] = fail_path

    corrective_path = out / "corrective_before_after.json"
    corrective_examples = [
        {
            key: row.get(key)
            for key in (
                "system",
                "question_id",
                "question_type",
                "initial_chunk_ids",
                "initial_paper_ids",
                "final_chunk_ids",
                "final_paper_ids",
                "initial_recall_at_k",
                "initial_recall_at_k_paper",
                "correction_recall_basis",
                "recall_at_k",
                "recall_at_k_paper",
                "improvement_after_correction",
                "corrective_trigger_correct",
            )
        }
        for row in report.per_question
        if row.get("corrective_triggered") is True
    ]
    corrective_path.write_text(
        json.dumps(
            {
                "definition": (
                    "Before is retrieval observed prior to the first corrective event; "
                    "after is the final evaluated evidence set. Null means the required "
                    "structured event or gold label was unavailable."
                ),
                "n_examples": len(corrective_examples),
                "examples": corrective_examples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["corrective_before_after_json"] = corrective_path

    # Category table markdown
    md_path = out / "report.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    paths["report_md"] = md_path

    if write_charts:
        chart_paths = write_simple_charts(report, out)
        paths.update(chart_paths)

    return paths


def compute_config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _markdown_report(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation report (`{report.run_id}`)",
        "",
        f"Frozen split fingerprint: `{report.frozen_split_fingerprint or 'n/a'}`",
        f"Run config fingerprint: `{report.config_fingerprint_sha256 or 'n/a'}`",
        "",
        "## Aggregate metrics",
        "",
        "| system | recall@k | mrr | cite_p | claim_correct | completeness | "
        "error_rate | latency_ms | tools |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report.systems:
        lines.append(
            f"| {s.system} | {s.recall_at_k:.3f} | {s.mrr:.3f} | "
            f"{s.citation_precision:.3f} | {s.claim_correctness:.3f} | "
            f"{s.completeness:.3f} | {s.error_rate:.3f} | "
            f"{s.avg_latency_ms:.0f} | {s.avg_tool_calls:.2f} |"
        )
    lines.append("")
    lines.append("## Per-category metrics")
    lines.append("")
    for s in report.systems:
        if not s.by_type:
            continue
        lines.append(f"### {s.system}")
        lines.append("")
        lines.append(
            "| type | n | recall_paper | mrr | nDCG | cite_p | cite_r | cite_valid | "
            "claim_correct | completeness | faithfulness | refusal | latency_ms | "
            "tools | tokens | cost_usd |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for qtype, vals in sorted(s.by_type.items()):
            n_value = vals.get("n") or 0.0
            lines.append(
                f"| {qtype} | {int(n_value)} | "
                f"{vals.get('recall_at_k_paper', 0):.3f} | "
                f"{vals.get('mrr', 0):.3f} | "
                f"{vals.get('ndcg_at_k', 0):.3f} | "
                f"{vals.get('citation_precision', 0):.3f} | "
                f"{vals.get('citation_recall', 0):.3f} | "
                f"{vals.get('citation_validity_rate', 0):.3f} | "
                f"{vals.get('claim_correctness', 0):.3f} | "
                f"{vals.get('completeness', 0):.3f} | "
                f"{vals.get('faithfulness_proxy', 0):.3f} | "
                f"{vals.get('refusal_correct', 0):.3f} | "
                f"{vals.get('avg_latency_ms', 0):.0f} | "
                f"{vals.get('avg_tool_calls', 0):.2f} | "
                f"{vals.get('avg_tokens', 0):.0f} | "
                f"{vals.get('estimated_cost_usd', 0):.6f} |"
            )
        lines.append("")
    if report.failures:
        lines.append("## Sample failures")
        lines.append("")
        for fail in report.failures[:10]:
            lines.append(
                f"- **{fail.get('system')} / {fail.get('question_id')}** "
                f"({fail.get('question_type')}): {fail.get('reason')}"
            )
        lines.append("")
    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def write_simple_charts(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    """Write SVG bar charts without hard dependency on matplotlib."""
    paths: dict[str, Path] = {}
    if not report.systems:
        return paths

    def bar_chart(
        filename: str,
        title: str,
        values: list[tuple[str, float]],
        *,
        ylabel: str,
    ) -> Path:
        width = 720
        height = 360
        margin_l, margin_r, margin_t, margin_b = 60, 20, 40, 80
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b
        max_v = max((v for _, v in values), default=1.0) or 1.0
        n = max(1, len(values))
        bar_w = plot_w / n * 0.7
        gap = plot_w / n

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2}" y="24" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{title}</text>',
            f'<text x="16" y="{margin_t + plot_h / 2}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" '
            f'transform="rotate(-90 16 {margin_t + plot_h / 2})">{ylabel}</text>',
        ]
        # axes
        parts.append(
            f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
            f'y2="{margin_t + plot_h}" stroke="#333"/>'
        )
        parts.append(
            f'<line x1="{margin_l}" y1="{margin_t + plot_h}" '
            f'x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333"/>'
        )
        for i, (label, val) in enumerate(values):
            h = (val / max_v) * plot_h if max_v else 0
            x = margin_l + i * gap + (gap - bar_w) / 2
            y = margin_t + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="#3b82f6"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_t + plot_h + 14}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'transform="rotate(25 {x + bar_w / 2:.1f} {margin_t + plot_h + 14})">'
                f"{label}</text>"
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10">{val:.2f}</text>'
            )
        parts.append("</svg>")
        path = output_dir / filename
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return path

    paths["chart_recall"] = bar_chart(
        "chart_recall_at_k.svg",
        "Paper/Chunk Recall@K by system",
        [(s.system, s.recall_at_k_paper or s.recall_at_k) for s in report.systems],
        ylabel="recall",
    )
    paths["chart_latency"] = bar_chart(
        "chart_latency.svg",
        "Average latency (ms) by system",
        [(s.system, s.avg_latency_ms) for s in report.systems],
        ylabel="ms",
    )
    paths["chart_cost"] = bar_chart(
        "chart_cost.svg",
        "Estimated token cost (USD) by system",
        [(s.system, s.total_estimated_cost_usd) for s in report.systems],
        ylabel="USD",
    )
    paths["chart_citation"] = bar_chart(
        "chart_citation_precision.svg",
        "Citation precision by system",
        [(s.system, s.citation_precision) for s in report.systems],
        ylabel="precision",
    )
    question_types = sorted(
        {question_type for summary in report.systems for question_type in summary.by_type}
    )
    for question_type in question_types:
        paths[f"chart_category_{question_type}"] = bar_chart(
            f"chart_recall_{question_type}.svg",
            f"Paper Recall@K — {question_type}",
            [
                (
                    summary.system,
                    float(
                        summary.by_type.get(question_type, {}).get("recall_at_k_paper", 0.0) or 0.0
                    ),
                )
                for summary in report.systems
            ],
            ylabel="recall",
        )
    return paths
