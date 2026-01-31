import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

from .models import BenchmarkSummary, AggregateResult, BenchmarkRecord

class ReportGenerator:
    """Generates reports from benchmark results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def save_json(self, summary: BenchmarkSummary) -> Path:
        """Save detailed results to JSON."""
        output_path = self.output_dir / "benchmark_results.json"
        
        with open(output_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
            
        return output_path
        
    def generate_markdown(self, summary: BenchmarkSummary) -> Path:
        """Generate a Markdown summary report."""
        output_path = self.output_dir / "benchmark_report.md"
        
        lines = [
            "# Software-as-a-Graph Benchmark Report",
            f"\n**Timestamp:** {summary.timestamp}",
            f"**Duration:** {summary.duration:.1f}s",
            f"**Total Runs:** {summary.total_runs}",
            f"**Passed Runs:** {summary.passed_runs} ({summary.overall_pass_rate:.1f}%)",
            "",
            "## Executive Summary",
            "",
            f"- **Overall Spearman Correlation:** {summary.overall_spearman:.3f}",
            f"- **Overall F1 Score:** {summary.overall_f1:.3f}",
            "",
            "## Aggregated Results by Scale/Layer",
            "",
            "| Scale | Layer | Runs | Pass Rate | Spearman ρ | F1 Score | Sim Time (ms) | Speedup |",
            "|-------|-------|------|-----------|------------|----------|---------------|---------|",
        ]
        
        for agg in summary.aggregates:
            lines.append(
                f"| {agg.scale} | {agg.layer} | {agg.num_runs} | {agg.pass_rate:.1f}% | "
                f"{agg.avg_spearman:.3f} | {agg.avg_f1:.3f} | {agg.avg_time_simulation:.0f} | "
                f"{agg.speedup_ratio:.1f}x |"
            )
            
        lines.extend([
            "",
            "## Detailed Validation Metrics",
            "",
            "| Scale | Layer | Precision | Recall | Top-5 Overlap | Top-10 Overlap | RMSE |",
            "|-------|-------|-----------|--------|---------------|----------------|------|",
        ])
        
        for agg in summary.aggregates:
            lines.append(
                f"| {agg.scale} | {agg.layer} | {agg.avg_precision:.3f} | {agg.avg_recall:.3f} | "
                f"{agg.avg_top5:.3f} | {agg.avg_top10:.3f} | {agg.avg_rmse:.3f} |"
            )
            
        lines.extend([
            "",
            "## Performance Analysis",
            "",
            "| Scale | Layer | Node Count | Edge Count | Density | Analysis (ms) | Total (ms) |",
            "|-------|-------|------------|------------|---------|---------------|------------|",
        ])
        
        for agg in summary.aggregates:
            lines.append(
                f"| {agg.scale} | {agg.layer} | {agg.avg_nodes:.1f} | {agg.avg_edges:.1f} | "
                f"{agg.avg_density:.4f} | {agg.avg_time_analysis:.0f} | {agg.avg_time_total:.0f} |"
            )

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
            
        return output_path
