#!/usr/bin/env python3
"""
Benchmark Script for Graph-Based Criticality Analysis

This script evaluates the methodology across different:
- System scales (tiny, small, medium, large, xlarge)
- Domain scenarios (iot, financial, healthcare, autonomous_vehicle, smart_city)
- Analysis methods (composite, betweenness, degree, pagerank)

Output:
- Detailed CSV results for statistical analysis
- Summary statistics with confidence intervals
- Publication-ready tables and charts
- Performance profiling data

Usage:
    python benchmark.py                     # Run default benchmark
    python benchmark.py --quick             # Quick validation run
    python benchmark.py --full              # Comprehensive benchmark
    python benchmark.py --scales small medium large
    python benchmark.py --scenarios iot financial
    python benchmark.py --output results/

Author: Ibrahim Onuralp Yigit
Research: Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems
"""

import argparse
import json
import sys
import time
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import generate_graph
from src.simulation import SimulationGraph, FailureSimulator, ComponentType
from src.validation import ValidationPipeline, GraphAnalyzer, Validator, ValidationTargets
from src.analysis import BoxPlotClassifier


# =============================================================================
# Configuration
# =============================================================================

SCALES = ["tiny", "small", "medium", "large", "xlarge"]
SCENARIOS = ["iot", "financial", "healthcare", "autonomous_vehicle", "smart_city"]
METHODS = ["composite", "betweenness", "degree", "pagerank", "message_path"]

# Research validation targets
TARGETS = ValidationTargets(
    spearman=0.70,
    f1=0.90,
    precision=0.80,
    recall=0.80,
    top_5=0.60,
)

# Scale configurations (for reference)
SCALE_CONFIG = {
    "tiny": {"apps": 5, "brokers": 1, "topics": 8, "nodes": 2},
    "small": {"apps": 10, "brokers": 2, "topics": 20, "nodes": 4},
    "medium": {"apps": 30, "brokers": 4, "topics": 60, "nodes": 8},
    "large": {"apps": 100, "brokers": 8, "topics": 200, "nodes": 20},
    "xlarge": {"apps": 300, "brokers": 16, "topics": 600, "nodes": 50},
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    run_id: str
    timestamp: str
    scale: str
    scenario: str
    method: str
    seed: int
    
    # Graph metrics
    n_components: int
    n_connections: int
    n_paths: int
    n_applications: int
    n_topics: int
    n_brokers: int
    n_nodes: int
    
    # Validation metrics
    spearman: float
    spearman_pvalue: float
    pearson: float
    f1_score: float
    precision: float
    recall: float
    top_3_overlap: float
    top_5_overlap: float
    top_10_overlap: float
    
    # Classification
    n_critical: int
    n_high: int
    n_medium: int
    n_low: int
    n_minimal: int
    n_articulation_points: int
    
    # Target achievement
    spearman_met: bool
    f1_met: bool
    precision_met: bool
    recall_met: bool
    top5_met: bool
    all_targets_met: bool
    targets_met_count: int
    
    # Timing (milliseconds)
    generation_time_ms: float
    analysis_time_ms: float
    simulation_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    
    # Errors
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark configuration."""
    scale: str
    scenario: str
    method: str
    n_runs: int
    
    # Aggregated metrics (mean ± std)
    spearman_mean: float
    spearman_std: float
    spearman_min: float
    spearman_max: float
    
    f1_mean: float
    f1_std: float
    f1_min: float
    f1_max: float
    
    precision_mean: float
    precision_std: float
    
    recall_mean: float
    recall_std: float
    
    top5_mean: float
    top5_std: float
    
    # Success rates
    spearman_success_rate: float
    f1_success_rate: float
    all_targets_success_rate: float
    
    # Timing
    total_time_mean_ms: float
    total_time_std_ms: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    duration_seconds: float
    total_runs: int
    successful_runs: int
    failed_runs: int
    
    scales_tested: List[str]
    scenarios_tested: List[str]
    methods_tested: List[str]
    seeds_used: List[int]
    
    results: List[BenchmarkResult] = field(default_factory=list)
    summaries: List[BenchmarkSummary] = field(default_factory=list)
    
    # Overall statistics
    overall_spearman_mean: float = 0.0
    overall_f1_mean: float = 0.0
    overall_success_rate: float = 0.0
    best_method: str = ""
    best_scale: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "scales_tested": self.scales_tested,
            "scenarios_tested": self.scenarios_tested,
            "methods_tested": self.methods_tested,
            "seeds_used": self.seeds_used,
            "overall_spearman_mean": self.overall_spearman_mean,
            "overall_f1_mean": self.overall_f1_mean,
            "overall_success_rate": self.overall_success_rate,
            "best_method": self.best_method,
            "best_scale": self.best_scale,
            "results": [r.to_dict() for r in self.results],
            "summaries": [s.to_dict() for s in self.summaries],
        }


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")


def print_section(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.DIM}{'-'*50}{Colors.END}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.END} {text}")


def print_metric(name: str, value: float, target: float, width: int = 20) -> None:
    met = value >= target
    status = f"{Colors.GREEN}✓{Colors.END}" if met else f"{Colors.RED}✗{Colors.END}"
    color = Colors.GREEN if met else Colors.YELLOW
    print(f"  {name:<{width}} {color}{value:>8.4f}{Colors.END}  (target: {target:.2f}) {status}")


def print_progress(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({pct:.0%})", end='', flush=True)


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Run benchmarks across scales, scenarios, and methods."""
    
    def __init__(
        self,
        scales: List[str] = None,
        scenarios: List[str] = None,
        methods: List[str] = None,
        n_runs: int = 3,
        seeds: List[int] = None,
        enable_cascade: bool = True,
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7,
        output_dir: str = "output/benchmark",
        verbose: bool = True,
    ):
        self.scales = scales or ["small", "medium", "large"]
        self.scenarios = scenarios or ["iot", "financial"]
        self.methods = methods or ["composite", "betweenness"]
        self.n_runs = n_runs
        self.seeds = seeds or list(range(42, 42 + n_runs))
        self.enable_cascade = enable_cascade
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.errors: List[Tuple[str, str]] = []
    
    def run(self) -> BenchmarkReport:
        """Run complete benchmark suite."""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        total_runs = len(self.scales) * len(self.scenarios) * len(self.methods) * len(self.seeds)
        
        if self.verbose:
            print_header("GRAPH-BASED CRITICALITY ANALYSIS BENCHMARK")
            print(f"  Scales:    {', '.join(self.scales)}")
            print(f"  Scenarios: {', '.join(self.scenarios)}")
            print(f"  Methods:   {', '.join(self.methods)}")
            print(f"  Seeds:     {self.seeds}")
            print(f"  Total runs: {total_runs}")
            print(f"  Cascade:   {'enabled' if self.enable_cascade else 'disabled'}")
        
        run_count = 0
        
        for scale in self.scales:
            for scenario in self.scenarios:
                for method in self.methods:
                    for seed in self.seeds:
                        run_count += 1
                        
                        if self.verbose:
                            print_progress(run_count, total_runs, 
                                         f"  Running {scale}/{scenario}/{method}")
                        
                        result = self._run_single(
                            scale=scale,
                            scenario=scenario,
                            method=method,
                            seed=seed,
                            run_id=f"{scale}_{scenario}_{method}_{seed}"
                        )
                        
                        self.results.append(result)
                        
                        if result.error:
                            self.errors.append((result.run_id, result.error))
        
        if self.verbose:
            print()  # New line after progress bar
        
        # Generate summaries
        summaries = self._generate_summaries()
        
        # Calculate overall statistics
        successful = [r for r in self.results if r.error is None]
        
        overall_spearman = statistics.mean([r.spearman for r in successful]) if successful else 0
        overall_f1 = statistics.mean([r.f1_score for r in successful]) if successful else 0
        overall_success = len([r for r in successful if r.all_targets_met]) / len(successful) if successful else 0
        
        # Find best performers
        best_method = self._find_best_method()
        best_scale = self._find_best_scale()
        
        duration = time.time() - start_time
        
        report = BenchmarkReport(
            timestamp=timestamp,
            duration_seconds=duration,
            total_runs=total_runs,
            successful_runs=len(successful),
            failed_runs=len(self.errors),
            scales_tested=self.scales,
            scenarios_tested=self.scenarios,
            methods_tested=self.methods,
            seeds_used=self.seeds,
            results=self.results,
            summaries=summaries,
            overall_spearman_mean=overall_spearman,
            overall_f1_mean=overall_f1,
            overall_success_rate=overall_success,
            best_method=best_method,
            best_scale=best_scale,
        )
        
        # Export results
        self._export_results(report)
        
        if self.verbose:
            self._print_report(report)
        
        return report
    
    def _run_single(
        self,
        scale: str,
        scenario: str,
        method: str,
        seed: int,
        run_id: str,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        timestamp = datetime.now().isoformat()
        
        try:
            # Step 1: Generate graph
            gen_start = time.time()
            graph_data = generate_graph(scale=scale, scenario=scenario, seed=seed)
            graph = SimulationGraph.from_dict(graph_data)
            gen_time = (time.time() - gen_start) * 1000
            
            # Count components by type
            apps = graph.get_components_by_type(ComponentType.APPLICATION)
            topics = graph.get_components_by_type(ComponentType.TOPIC)
            brokers = graph.get_components_by_type(ComponentType.BROKER)
            nodes = graph.get_components_by_type(ComponentType.NODE)
            
            # Step 2: Run validation pipeline
            pipeline = ValidationPipeline(
                targets=TARGETS,
                cascade_threshold=self.cascade_threshold,
                cascade_probability=self.cascade_probability,
                seed=seed,
            )
            
            result = pipeline.run(
                graph,
                analysis_method=method,
                enable_cascade=self.enable_cascade,
            )
            
            # Step 3: Run classification
            classifier = BoxPlotClassifier(k_factor=1.5)
            items = [
                {"id": k, "type": "component", "score": v}
                for k, v in result.predicted_scores.items()
            ]
            classification = classifier.classify(items, metric_name="composite")
            
            # Count by level
            level_counts = defaultdict(int)
            for item in classification.items:
                level_counts[item.level.value] += 1
            
            # Count articulation points (if available)
            n_aps = 0
            try:
                import networkx as nx
                # Build NetworkX graph for articulation point detection
                G = nx.Graph()
                for conn in graph.connections:
                    G.add_edge(conn.source_id, conn.target_id)
                aps = list(nx.articulation_points(G))
                n_aps = len(aps)
            except Exception:
                pass
            
            # Extract metrics
            v = result.validation
            
            # Check targets
            spearman_met = v.correlation.spearman >= TARGETS.spearman
            f1_met = v.classification.f1 >= TARGETS.f1
            precision_met = v.classification.precision >= TARGETS.precision
            recall_met = v.classification.recall >= TARGETS.recall
            top5_met = v.ranking.top_k_overlap.get(5, 0) >= TARGETS.top_5
            
            targets_met = sum([spearman_met, f1_met, precision_met, recall_met, top5_met])
            all_met = targets_met == 5
            
            return BenchmarkResult(
                run_id=run_id,
                timestamp=timestamp,
                scale=scale,
                scenario=scenario,
                method=method,
                seed=seed,
                
                n_components=len(graph.components),
                n_connections=len(graph.connections),
                n_paths=len(graph.get_all_message_paths()),
                n_applications=len(apps),
                n_topics=len(topics),
                n_brokers=len(brokers),
                n_nodes=len(nodes),
                
                spearman=v.correlation.spearman,
                spearman_pvalue=v.correlation.spearman_p,
                pearson=v.correlation.pearson,
                f1_score=v.classification.f1,
                precision=v.classification.precision,
                recall=v.classification.recall,
                top_3_overlap=v.ranking.top_k_overlap.get(3, 0),
                top_5_overlap=v.ranking.top_k_overlap.get(5, 0),
                top_10_overlap=v.ranking.top_k_overlap.get(10, 0),
                
                n_critical=level_counts.get("CRITICAL", 0),
                n_high=level_counts.get("HIGH", 0),
                n_medium=level_counts.get("MEDIUM", 0),
                n_low=level_counts.get("LOW", 0),
                n_minimal=level_counts.get("MINIMAL", 0),
                n_articulation_points=n_aps,
                
                spearman_met=spearman_met,
                f1_met=f1_met,
                precision_met=precision_met,
                recall_met=recall_met,
                top5_met=top5_met,
                all_targets_met=all_met,
                targets_met_count=targets_met,
                
                generation_time_ms=gen_time,
                analysis_time_ms=result.analysis_time_ms,
                simulation_time_ms=result.simulation_time_ms,
                validation_time_ms=result.validation_time_ms,
                total_time_ms=result.total_time_ms + gen_time,
            )
            
        except Exception as e:
            return BenchmarkResult(
                run_id=run_id,
                timestamp=timestamp,
                scale=scale,
                scenario=scenario,
                method=method,
                seed=seed,
                n_components=0,
                n_connections=0,
                n_paths=0,
                n_applications=0,
                n_topics=0,
                n_brokers=0,
                n_nodes=0,
                spearman=0,
                spearman_pvalue=1,
                pearson=0,
                f1_score=0,
                precision=0,
                recall=0,
                top_3_overlap=0,
                top_5_overlap=0,
                top_10_overlap=0,
                n_critical=0,
                n_high=0,
                n_medium=0,
                n_low=0,
                n_minimal=0,
                n_articulation_points=0,
                spearman_met=False,
                f1_met=False,
                precision_met=False,
                recall_met=False,
                top5_met=False,
                all_targets_met=False,
                targets_met_count=0,
                generation_time_ms=0,
                analysis_time_ms=0,
                simulation_time_ms=0,
                validation_time_ms=0,
                total_time_ms=0,
                error=str(e),
            )
    
    def _generate_summaries(self) -> List[BenchmarkSummary]:
        """Generate summary statistics for each configuration."""
        summaries = []
        
        # Group by scale, scenario, method
        groups = defaultdict(list)
        for r in self.results:
            if r.error is None:
                key = (r.scale, r.scenario, r.method)
                groups[key].append(r)
        
        for (scale, scenario, method), results in groups.items():
            if not results:
                continue
            
            n = len(results)
            
            # Extract metrics
            spearmans = [r.spearman for r in results]
            f1s = [r.f1_score for r in results]
            precisions = [r.precision for r in results]
            recalls = [r.recall for r in results]
            top5s = [r.top_5_overlap for r in results]
            times = [r.total_time_ms for r in results]
            
            def safe_std(values):
                return statistics.stdev(values) if len(values) > 1 else 0
            
            summaries.append(BenchmarkSummary(
                scale=scale,
                scenario=scenario,
                method=method,
                n_runs=n,
                
                spearman_mean=statistics.mean(spearmans),
                spearman_std=safe_std(spearmans),
                spearman_min=min(spearmans),
                spearman_max=max(spearmans),
                
                f1_mean=statistics.mean(f1s),
                f1_std=safe_std(f1s),
                f1_min=min(f1s),
                f1_max=max(f1s),
                
                precision_mean=statistics.mean(precisions),
                precision_std=safe_std(precisions),
                
                recall_mean=statistics.mean(recalls),
                recall_std=safe_std(recalls),
                
                top5_mean=statistics.mean(top5s),
                top5_std=safe_std(top5s),
                
                spearman_success_rate=len([r for r in results if r.spearman_met]) / n,
                f1_success_rate=len([r for r in results if r.f1_met]) / n,
                all_targets_success_rate=len([r for r in results if r.all_targets_met]) / n,
                
                total_time_mean_ms=statistics.mean(times),
                total_time_std_ms=safe_std(times),
            ))
        
        return summaries
    
    def _find_best_method(self) -> str:
        """Find the best performing analysis method."""
        successful = [r for r in self.results if r.error is None]
        if not successful:
            return ""
        
        method_scores = defaultdict(list)
        for r in successful:
            # Combined score: weighted average of Spearman and F1
            score = 0.5 * r.spearman + 0.5 * r.f1_score
            method_scores[r.method].append(score)
        
        best = max(method_scores.keys(), 
                   key=lambda m: statistics.mean(method_scores[m]))
        return best
    
    def _find_best_scale(self) -> str:
        """Find the best performing scale."""
        successful = [r for r in self.results if r.error is None]
        if not successful:
            return ""
        
        scale_scores = defaultdict(list)
        for r in successful:
            score = 0.5 * r.spearman + 0.5 * r.f1_score
            scale_scores[r.scale].append(score)
        
        best = max(scale_scores.keys(),
                   key=lambda s: statistics.mean(scale_scores[s]))
        return best
    
    def _export_results(self, report: BenchmarkReport) -> None:
        """Export results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export full JSON report
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Export CSV for detailed results
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        if report.results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=report.results[0].to_dict().keys())
                writer.writeheader()
                for r in report.results:
                    writer.writerow(r.to_dict())
        
        # Export summary CSV
        summary_path = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        if report.summaries:
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=report.summaries[0].to_dict().keys())
                writer.writeheader()
                for s in report.summaries:
                    writer.writerow(s.to_dict())
        
        # Generate markdown report
        md_path = self.output_dir / f"benchmark_report_{timestamp}.md"
        self._generate_markdown_report(report, md_path)
        
        if self.verbose:
            print_section("Exported Files")
            print(f"  JSON:    {json_path}")
            print(f"  CSV:     {csv_path}")
            print(f"  Summary: {summary_path}")
            print(f"  Report:  {md_path}")
    
    def _generate_markdown_report(self, report: BenchmarkReport, path: Path) -> None:
        """Generate publication-ready markdown report."""
        lines = [
            "# Benchmark Report: Graph-Based Criticality Analysis",
            "",
            f"**Generated**: {report.timestamp}",
            f"**Duration**: {report.duration_seconds:.2f} seconds",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"- **Total Runs**: {report.total_runs}",
            f"- **Successful**: {report.successful_runs}",
            f"- **Failed**: {report.failed_runs}",
            f"- **Overall Spearman ρ**: {report.overall_spearman_mean:.4f}",
            f"- **Overall F1-Score**: {report.overall_f1_mean:.4f}",
            f"- **Target Achievement Rate**: {report.overall_success_rate:.1%}",
            f"- **Best Method**: {report.best_method}",
            f"- **Best Scale**: {report.best_scale}",
            "",
            "---",
            "",
            "## Configuration",
            "",
            f"- **Scales**: {', '.join(report.scales_tested)}",
            f"- **Scenarios**: {', '.join(report.scenarios_tested)}",
            f"- **Methods**: {', '.join(report.methods_tested)}",
            f"- **Seeds**: {report.seeds_used}",
            "",
            "---",
            "",
            "## Results by Method",
            "",
            "| Method | Spearman ρ | F1-Score | Precision | Recall | Top-5 | Success Rate |",
            "|--------|------------|----------|-----------|--------|-------|--------------|",
        ]
        
        # Group summaries by method
        method_stats = defaultdict(list)
        for s in report.summaries:
            method_stats[s.method].append(s)
        
        for method in report.methods_tested:
            if method in method_stats:
                summaries = method_stats[method]
                avg_spearman = statistics.mean([s.spearman_mean for s in summaries])
                avg_f1 = statistics.mean([s.f1_mean for s in summaries])
                avg_precision = statistics.mean([s.precision_mean for s in summaries])
                avg_recall = statistics.mean([s.recall_mean for s in summaries])
                avg_top5 = statistics.mean([s.top5_mean for s in summaries])
                avg_success = statistics.mean([s.all_targets_success_rate for s in summaries])
                
                lines.append(
                    f"| {method} | {avg_spearman:.4f} | {avg_f1:.4f} | "
                    f"{avg_precision:.4f} | {avg_recall:.4f} | {avg_top5:.4f} | {avg_success:.1%} |"
                )
        
        lines.extend([
            "",
            "---",
            "",
            "## Results by Scale",
            "",
            "| Scale | Components | Spearman ρ | F1-Score | Time (ms) | Success Rate |",
            "|-------|------------|------------|----------|-----------|--------------|",
        ])
        
        # Group by scale
        scale_stats = defaultdict(list)
        for s in report.summaries:
            scale_stats[s.scale].append(s)
        
        for scale in report.scales_tested:
            if scale in scale_stats:
                summaries = scale_stats[scale]
                # Get component count from results
                scale_results = [r for r in report.results if r.scale == scale and r.error is None]
                avg_components = statistics.mean([r.n_components for r in scale_results]) if scale_results else 0
                
                avg_spearman = statistics.mean([s.spearman_mean for s in summaries])
                avg_f1 = statistics.mean([s.f1_mean for s in summaries])
                avg_time = statistics.mean([s.total_time_mean_ms for s in summaries])
                avg_success = statistics.mean([s.all_targets_success_rate for s in summaries])
                
                lines.append(
                    f"| {scale} | {avg_components:.0f} | {avg_spearman:.4f} | "
                    f"{avg_f1:.4f} | {avg_time:.0f} | {avg_success:.1%} |"
                )
        
        lines.extend([
            "",
            "---",
            "",
            "## Results by Scenario",
            "",
            "| Scenario | Spearman ρ | F1-Score | Precision | Recall | Success Rate |",
            "|----------|------------|----------|-----------|--------|--------------|",
        ])
        
        # Group by scenario
        scenario_stats = defaultdict(list)
        for s in report.summaries:
            scenario_stats[s.scenario].append(s)
        
        for scenario in report.scenarios_tested:
            if scenario in scenario_stats:
                summaries = scenario_stats[scenario]
                avg_spearman = statistics.mean([s.spearman_mean for s in summaries])
                avg_f1 = statistics.mean([s.f1_mean for s in summaries])
                avg_precision = statistics.mean([s.precision_mean for s in summaries])
                avg_recall = statistics.mean([s.recall_mean for s in summaries])
                avg_success = statistics.mean([s.all_targets_success_rate for s in summaries])
                
                lines.append(
                    f"| {scenario} | {avg_spearman:.4f} | {avg_f1:.4f} | "
                    f"{avg_precision:.4f} | {avg_recall:.4f} | {avg_success:.1%} |"
                )
        
        lines.extend([
            "",
            "---",
            "",
            "## Validation Targets",
            "",
            "| Metric | Target | Achieved (Mean) | Status |",
            "|--------|--------|-----------------|--------|",
            f"| Spearman ρ | ≥ {TARGETS.spearman} | {report.overall_spearman_mean:.4f} | {'✓' if report.overall_spearman_mean >= TARGETS.spearman else '✗'} |",
            f"| F1-Score | ≥ {TARGETS.f1} | {report.overall_f1_mean:.4f} | {'✓' if report.overall_f1_mean >= TARGETS.f1 else '✗'} |",
            "",
            "---",
            "",
            "## Detailed Results",
            "",
            "| Scale | Scenario | Method | Spearman | F1 | Precision | Recall | Time (ms) |",
            "|-------|----------|--------|----------|-----|-----------|--------|-----------|",
        ])
        
        for s in report.summaries:
            lines.append(
                f"| {s.scale} | {s.scenario} | {s.method} | "
                f"{s.spearman_mean:.4f}±{s.spearman_std:.4f} | "
                f"{s.f1_mean:.4f}±{s.f1_std:.4f} | "
                f"{s.precision_mean:.4f}±{s.precision_std:.4f} | "
                f"{s.recall_mean:.4f}±{s.recall_std:.4f} | "
                f"{s.total_time_mean_ms:.0f}±{s.total_time_std_ms:.0f} |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            "*Report generated by benchmark.py*",
        ])
        
        path.write_text("\n".join(lines))
    
    def _print_report(self, report: BenchmarkReport) -> None:
        """Print report to terminal."""
        print_header("BENCHMARK RESULTS")
        
        print_section("Summary")
        print(f"  Total runs:      {report.total_runs}")
        print(f"  Successful:      {report.successful_runs}")
        print(f"  Failed:          {report.failed_runs}")
        print(f"  Duration:        {report.duration_seconds:.2f}s")
        
        print_section("Overall Metrics")
        print_metric("Spearman ρ", report.overall_spearman_mean, TARGETS.spearman)
        print_metric("F1-Score", report.overall_f1_mean, TARGETS.f1)
        print(f"  {'Target Achievement':<20} {report.overall_success_rate:>8.1%}")
        
        print_section("Best Performers")
        print(f"  Best Method:     {report.best_method}")
        print(f"  Best Scale:      {report.best_scale}")
        
        if self.errors:
            print_section("Errors")
            for run_id, error in self.errors[:5]:
                print_error(f"{run_id}: {error[:50]}...")
        
        print_section("Results by Method")
        print(f"  {'Method':<15} {'Spearman':>10} {'F1':>10} {'Success':>10}")
        print(f"  {'-'*45}")
        
        method_stats = defaultdict(list)
        for s in report.summaries:
            method_stats[s.method].append(s)
        
        for method in sorted(method_stats.keys()):
            summaries = method_stats[method]
            avg_spearman = statistics.mean([s.spearman_mean for s in summaries])
            avg_f1 = statistics.mean([s.f1_mean for s in summaries])
            avg_success = statistics.mean([s.all_targets_success_rate for s in summaries])
            print(f"  {method:<15} {avg_spearman:>10.4f} {avg_f1:>10.4f} {avg_success:>10.1%}")
        
        print_section("Results by Scale")
        print(f"  {'Scale':<10} {'Components':>12} {'Spearman':>10} {'F1':>10} {'Time (ms)':>12}")
        print(f"  {'-'*55}")
        
        scale_stats = defaultdict(list)
        for s in report.summaries:
            scale_stats[s.scale].append(s)
        
        for scale in self.scales:
            if scale in scale_stats:
                summaries = scale_stats[scale]
                scale_results = [r for r in report.results if r.scale == scale and r.error is None]
                avg_components = statistics.mean([r.n_components for r in scale_results]) if scale_results else 0
                avg_spearman = statistics.mean([s.spearman_mean for s in summaries])
                avg_f1 = statistics.mean([s.f1_mean for s in summaries])
                avg_time = statistics.mean([s.total_time_mean_ms for s in summaries])
                print(f"  {scale:<10} {avg_components:>12.0f} {avg_spearman:>10.4f} {avg_f1:>10.4f} {avg_time:>12.0f}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark graph-based criticality analysis methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --quick              # Quick validation
  python benchmark.py --full               # Comprehensive benchmark
  python benchmark.py --scales small medium large
  python benchmark.py --scenarios iot financial healthcare
  python benchmark.py --methods composite betweenness
  python benchmark.py --runs 5 --output results/
        """
    )
    
    # Presets
    parser.add_argument("--quick", action="store_true",
                       help="Quick benchmark (small scale, 2 runs)")
    parser.add_argument("--full", action="store_true",
                       help="Full benchmark (all scales, 5 runs)")
    
    # Configuration
    parser.add_argument("--scales", nargs="+", choices=SCALES,
                       help="Scales to test")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                       help="Scenarios to test")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                       help="Analysis methods to test")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs per configuration (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Starting seed (default: 42)")
    
    # Simulation settings
    parser.add_argument("--no-cascade", action="store_true",
                       help="Disable cascade propagation")
    parser.add_argument("--cascade-threshold", type=float, default=0.5,
                       help="Cascade threshold (default: 0.5)")
    parser.add_argument("--cascade-prob", type=float, default=0.7,
                       help="Cascade probability (default: 0.7)")
    
    # Output
    parser.add_argument("--output", "-o", type=str, default="output/benchmark",
                       help="Output directory (default: output/benchmark)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Apply presets
    if args.quick:
        scales = ["small", "medium"]
        scenarios = ["iot"]
        methods = ["composite", "betweenness"]
        n_runs = 2
    elif args.full:
        scales = SCALES
        scenarios = SCENARIOS
        methods = METHODS
        n_runs = 5
    else:
        scales = args.scales or ["small", "medium", "large"]
        scenarios = args.scenarios or ["iot", "financial"]
        methods = args.methods or ["composite", "betweenness", "degree"]
        n_runs = args.runs
    
    # Generate seeds
    seeds = list(range(args.seed, args.seed + n_runs))
    
    # Run benchmark
    runner = BenchmarkRunner(
        scales=scales,
        scenarios=scenarios,
        methods=methods,
        n_runs=n_runs,
        seeds=seeds,
        enable_cascade=not args.no_cascade,
        cascade_threshold=args.cascade_threshold,
        cascade_probability=args.cascade_prob,
        output_dir=args.output,
        verbose=not args.quiet,
    )
    
    report = runner.run()
    
    # Exit code based on success rate
    if report.overall_success_rate >= 0.8:
        return 0
    elif report.overall_success_rate >= 0.5:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
