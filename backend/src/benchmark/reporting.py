"""
Benchmark Report Generation

Produces JSON and Markdown reports from BenchmarkSummary objects.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkSummary


class ReportGenerator:
    """Generates JSON and Markdown benchmark reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, summary: BenchmarkSummary) -> Path:
        """Write full results as JSON."""
        path = self.output_dir / "benchmark_results.json"
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        return path

    def generate_markdown(self, summary: BenchmarkSummary) -> Path:
        """Generate a Markdown report with tables and analysis."""
        path = self.output_dir / "benchmark_report.md"

        lines = [
            "# Software-as-a-Graph Benchmark Report",
            "",
            f"**Timestamp:** {summary.timestamp}",
            f"**Duration:** {summary.duration:.1f}s",
            f"**Total Runs:** {summary.total_runs}",
            f"**Passed Runs:** {summary.passed_runs} ({summary.overall_pass_rate:.1f}%)",
            "",
            "## Executive Summary",
            "",
            f"- **Overall Spearman ρ:** {summary.overall_spearman:.3f}",
            f"- **Overall F1 Score:** {summary.overall_f1:.3f}",
        ]

        if summary.best_config:
            lines.append(f"- **Best Config:** {summary.best_config}")
        if summary.worst_config:
            lines.append(f"- **Worst Config:** {summary.worst_config}")

        # ── Aggregated results ────────────────────────────────────
        lines.extend([
            "",
            "## Aggregated Results by Scale / Layer",
            "",
            "| Scale | Layer | Runs | Pass Rate | Spearman ρ (±σ) | F1 (±σ) | Sim Time (ms) | Speedup |",
            "|-------|-------|------|-----------|-----------------|---------|---------------|---------|",
        ])
        for a in summary.aggregates:
            sp = f"{a.avg_spearman:.3f}±{a.std_spearman:.3f}"
            f1 = f"{a.avg_f1:.3f}±{a.std_f1:.3f}"
            lines.append(
                f"| {a.scale} | {a.layer} | {a.num_runs} | {a.pass_rate:.1f}% | "
                f"{sp} | {f1} | {a.avg_time_simulation:.0f} | "
                f"{a.speedup_ratio:.1f}x |"
            )

        # ── Baseline Comparison ───────────────────────────────────
        lines.extend([
            "",
            "## Baseline Comparison (Spearman ρ)",
            "",
            "| Scale | Layer | Composite Q(v) | Betweenness | Degree | Random | Gain (%) |",
            "|-------|-------|----------------|-------------|--------|--------|----------|",
        ])
        for a in summary.aggregates:
            gain = ((a.avg_spearman - a.avg_spearman_bc) / a.avg_spearman_bc * 100) if a.avg_spearman_bc > 0 else 0.0
            lines.append(
                f"| {a.scale} | {a.layer} | **{a.avg_spearman:.3f}** | "
                f"{a.avg_spearman_bc:.3f} | {a.avg_spearman_degree:.3f} | "
                f"{a.avg_spearman_random:.3f} | +{gain:.1f}% |"
            )

        # ── Detailed validation ───────────────────────────────────
        lines.extend([
            "",
            "## Detailed Validation Metrics",
            "",
            "| Scale | Layer | Precision | Recall | Top-5 | AUC-PR | RMSE |",
            "|-------|-------|-----------|--------|-------|--------|------|",
        ])
        for a in summary.aggregates:
            lines.append(
                f"| {a.scale} | {a.layer} | {a.avg_precision:.3f} | "
                f"{a.avg_recall:.3f} | {a.avg_top5:.3f} | "
                f"{a.avg_auc_pr:.3f} | {a.avg_rmse:.3f} |"
            )

        # ── Performance ───────────────────────────────────────────
        lines.extend([
            "",
            "## Performance Analysis",
            "",
            "| Scale | Layer | Nodes | Edges | Density | Analysis (ms) | Total (ms) |",
            "|-------|-------|-------|-------|---------|---------------|------------|",
        ])
        for a in summary.aggregates:
            lines.append(
                f"| {a.scale} | {a.layer} | {a.avg_nodes:.0f} | "
                f"{a.avg_edges:.0f} | {a.avg_density:.4f} | "
                f"{a.avg_time_analysis:.0f} | {a.avg_time_total:.0f} |"
            )

        # ── Errors (if any) ──────────────────────────────────────
        errors = [r for r in summary.records if r.error]
        if errors:
            lines.extend([
                "",
                "## Errors",
                "",
                "| Run ID | Scale | Layer | Error |",
                "|--------|-------|-------|-------|",
            ])
            for r in errors:
                lines.append(f"| {r.run_id} | {r.scale} | {r.layer} | {r.error} |")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        return path