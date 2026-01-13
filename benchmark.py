#!/usr/bin/env python3
"""
Benchmark Suite for Graph-Based Criticality Prediction

Evaluates the Software-as-a-Graph methodology across varying system scales,
and layers to validate predictive accuracy and measure performance.

Evaluation Dimensions:
    1. Predictive Accuracy: Spearman ρ, F1, Precision, Recall
    2. Ranking Quality: Top-K overlap, NDCG
    3. Computational Performance: Analysis time vs Simulation time
    4. Scalability: Performance across system sizes

Workflow per Configuration:
    1. Generate synthetic system topology
    2. Import data into Neo4j
    3. Run graph analysis (prediction)
    4. Run failure simulation (ground truth)
    5. Statistical validation (comparison)
    6. Record and aggregate results

Output:
    - CSV data file with all metrics
    - JSON detailed results
    - Markdown summary report
    - HTML interactive report (optional)

Usage:
    python benchmark.py --scales small,medium,large --runs 3
    python benchmark.py --layers app,infra,system
    python benchmark.py --full-suite --output results/benchmark

Author: Software-as-a-Graph Research Project
"""

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Configuration
# =============================================================================

# Layer definitions
LAYER_DEFINITIONS = {
    "app": {"name": "Application", "full": "Application Layer"},
    "infra": {"name": "Infrastructure", "full": "Infrastructure Layer"},
    "mw-app": {"name": "MW-Application", "full": "Middleware-Application Layer"},
    "mw-infra": {"name": "MW-Infrastructure", "full": "Middleware-Infrastructure Layer"},
    "system": {"name": "System", "full": "Complete System"},
}

# Scale definitions (approximate component counts)
SCALE_DEFINITIONS = {
    "tiny": {"nodes": "5-10", "description": "Minimal test system"},
    "small": {"nodes": "10-25", "description": "Small deployment"},
    "medium": {"nodes": "30-50", "description": "Medium deployment"},
    "large": {"nodes": "60-100", "description": "Large deployment"},
    "xlarge": {"nodes": "150-300", "description": "Enterprise scale"},
}

# Validation targets
VALIDATION_TARGETS = {
    "spearman": 0.70,
    "f1": 0.80,
    "precision": 0.80,
    "recall": 0.80,
    "top5_overlap": 0.40,
    "top10_overlap": 0.60,
}


# =============================================================================
# Terminal Styling
# =============================================================================

class Colors:
    """ANSI color codes."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"


def print_header(title: str, char: str = "═", width: int = 70) -> None:
    """Print a formatted header."""
    line = char * width
    print(f"\n{colored(line, Colors.CYAN)}")
    print(f"{colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
    print(f"{colored(line, Colors.CYAN)}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{colored(f'▶ {title}', Colors.BLUE, bold=True)}")
    print(f"  {colored('─' * 50, Colors.GRAY)}")


def print_progress(current: int, total: int, message: str) -> None:
    """Print progress indicator."""
    pct = (current / total) * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}% {message:<40}", end="", flush=True)


def print_result(passed: bool, message: str) -> None:
    """Print a result line."""
    icon = colored("✓", Colors.GREEN) if passed else colored("✗", Colors.RED)
    print(f"  {icon} {message}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkRecord:
    """Single benchmark iteration result."""
    # Identification
    run_id: str
    timestamp: str
    scale: str
    layer: str
    seed: int
    
    # Graph statistics
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    components_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Performance timings (milliseconds)
    time_generation: float = 0.0
    time_import: float = 0.0
    time_analysis: float = 0.0
    time_simulation: float = 0.0
    time_validation: float = 0.0
    time_total: float = 0.0
    
    # Validation metrics
    spearman: float = 0.0
    spearman_p: float = 1.0
    pearson: float = 0.0
    kendall: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0
    
    # Classification counts
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Status
    passed: bool = False
    targets_met: int = 0
    targets_total: int = 6
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregateResult:
    """Aggregated results for a scale/layer combination."""
    scale: str
    layer: str
    num_runs: int
    
    # Graph stats (averages)
    avg_nodes: float = 0.0
    avg_edges: float = 0.0
    avg_density: float = 0.0
    
    # Timing stats (averages in ms)
    avg_time_analysis: float = 0.0
    avg_time_simulation: float = 0.0
    avg_time_total: float = 0.0
    speedup_ratio: float = 0.0  # simulation_time / analysis_time
    
    # Validation stats
    avg_spearman: float = 0.0
    std_spearman: float = 0.0
    avg_f1: float = 0.0
    std_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_top5: float = 0.0
    avg_top10: float = 0.0
    avg_rmse: float = 0.0
    
    # Success rate
    pass_rate: float = 0.0
    num_passed: int = 0


@dataclass
class BenchmarkSummary:
    """Complete benchmark summary."""
    timestamp: str
    duration: float
    total_runs: int
    passed_runs: int
    
    scales: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    
    records: List[BenchmarkRecord] = field(default_factory=list)
    aggregates: List[AggregateResult] = field(default_factory=list)
    
    # Overall metrics
    overall_spearman: float = 0.0
    overall_f1: float = 0.0
    overall_pass_rate: float = 0.0
    
    # Best/worst performers
    best_config: Optional[str] = None
    worst_config: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "total_runs": self.total_runs,
            "passed_runs": self.passed_runs,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_spearman": self.overall_spearman,
            "overall_f1": self.overall_f1,
            "scales": self.scales,
            "layers": self.layers,
            "best_config": self.best_config,
            "worst_config": self.worst_config,
            "aggregates": [asdict(a) for a in self.aggregates],
            "records": [r.to_dict() for r in self.records],
        }


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Executes comprehensive benchmark suite for the graph methodology.
    
    Workflow:
        1. For each scale: generate and import data
        2. For each layer: analyze, simulate, validate
        3. Aggregate results and generate reports
    """
    
    def __init__(
        self,
        scales: List[str],
        layers: List[str],
        runs: int,
        output_dir: Path,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        seed: int = 42,
        verbose: bool = False,
    ):
        self.scales = scales
        self.layers = layers
        self.runs = runs
        self.output_dir = output_dir
        self.uri = uri
        self.user = user
        self.password = password
        self.base_seed = seed
        self.verbose = verbose
        
        self.project_root = Path(__file__).parent.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("Benchmark")
        
        # Results storage
        self.records: List[BenchmarkRecord] = []
        
        # Lazy-loaded modules
        self._analyzer = None
        self._simulator = None
        self._validator = None
    
    # =========================================================================
    # Module Loading
    # =========================================================================
    
    def _load_modules(self) -> bool:
        """Load analysis, simulation, and validation modules."""
        try:
            from src.analysis import GraphAnalyzer
            from src.simulation import Simulator
            from src.validation import Validator, ValidationTargets
            
            self._analyzer_class = GraphAnalyzer
            self._simulator_class = Simulator
            self._validator_class = Validator
            self._targets_class = ValidationTargets
            
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import modules: {e}")
            return False
    
    def _get_analyzer(self):
        """Get or create analyzer instance."""
        return self._analyzer_class(
            uri=self.uri,
            user=self.user,
            password=self.password,
        )
    
    def _get_simulator(self):
        """Get or create simulator instance."""
        return self._simulator_class(
            uri=self.uri,
            user=self.user,
            password=self.password,
        )
    
    def _get_validator(self):
        """Get or create validator instance."""
        targets = self._targets_class(
            spearman=VALIDATION_TARGETS["spearman"],
            f1_score=VALIDATION_TARGETS["f1"],
            precision=VALIDATION_TARGETS["precision"],
            recall=VALIDATION_TARGETS["recall"],
        )
        return self._validator_class(targets)
    
    # =========================================================================
    # Data Generation & Import
    # =========================================================================
    
    def _generate_data(self, scale: str, seed: int, output_file: Path) -> Tuple[bool, float]:
        """Generate synthetic graph data."""
        start = time.time()
        
        script = self.project_root / "scripts" / "generate_graph.py"
        if not script.exists():
            script = self.project_root / "generate_graph.py"
        
        cmd = [
            sys.executable, str(script),
            "--scale", scale,
            "--seed", str(seed),
            "--output", str(output_file),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            
            if not success and self.verbose:
                self.logger.warning(f"Generation stderr: {result.stderr}")
            
            return success, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return False, 0.0
    
    def _import_data(self, input_file: Path) -> Tuple[bool, float]:
        """Import data into Neo4j."""
        start = time.time()
        
        script = self.project_root / "scripts" / "import_graph.py"
        if not script.exists():
            script = self.project_root / "import_graph.py"
        
        cmd = [
            sys.executable, str(script),
            "--input", str(input_file),
            "--clear",
            "--uri", self.uri,
            "--user", self.user,
            "--password", self.password,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            
            if not success and self.verbose:
                self.logger.warning(f"Import stderr: {result.stderr}")
            
            return success, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            return False, 0.0
    
    # =========================================================================
    # Analysis, Simulation, Validation
    # =========================================================================
    
    def _run_analysis(self, layer: str) -> Tuple[Optional[Any], float, Dict[str, Any]]:
        """Run graph analysis for a layer."""
        start = time.time()
        stats = {}
        
        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze_layer(layer)
            
            # Extract statistics
            stats["nodes"] = result.structural.graph_summary.nodes
            stats["edges"] = result.structural.graph_summary.edges
            stats["density"] = result.structural.graph_summary.density
            stats["type_breakdown"] = result.structural.graph_summary.node_types or {}
            
            # Extract predicted scores
            predicted = {
                c.id: c.scores.overall
                for c in result.quality.components
            }
            types = {
                c.id: c.type
                for c in result.quality.components
            }
            
            return (predicted, types), (time.time() - start) * 1000, stats
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return None, (time.time() - start) * 1000, stats
    
    def _run_simulation(self, layer: str) -> Tuple[Optional[Dict[str, float]], float]:
        """Run failure simulation for a layer."""
        start = time.time()
        
        try:
            simulator = self._get_simulator()
            results = simulator.run_failure_simulation_exhaustive(layer=layer)
            
            # Extract actual impact scores
            actual = {
                r.target_id: r.impact.composite_impact
                for r in results
            }
            
            return actual, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return None, (time.time() - start) * 1000
    
    def _run_validation(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        types: Dict[str, str],
        layer: str,
    ) -> Tuple[Optional[Any], float]:
        """Run validation comparing predicted vs actual."""
        start = time.time()
        
        try:
            validator = self._get_validator()
            result = validator.validate(
                predicted_scores=predicted,
                actual_scores=actual,
                component_types=types,
                layer=layer,
            )
            
            return result, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return None, (time.time() - start) * 1000
    
    # =========================================================================
    # Benchmark Execution
    # =========================================================================
    
    def _run_single_benchmark(
        self,
        scale: str,
        layer: str,
        seed: int,
        run_idx: int,
    ) -> BenchmarkRecord:
        """Execute a single benchmark iteration."""
        run_id = f"{scale}_{layer}_{seed}"
        
        record = BenchmarkRecord(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            scale=scale,
            layer=layer,
            seed=seed,
        )
        
        start_total = time.time()
        
        try:
            # Step 1: Analysis
            analysis_result, time_analysis, stats = self._run_analysis(layer)
            record.time_analysis = time_analysis
            
            if stats:
                record.nodes = stats.get("nodes", 0)
                record.edges = stats.get("edges", 0)
                record.density = stats.get("density", 0.0)
                record.components_by_type = stats.get("type_breakdown", {})
            
            if analysis_result is None:
                record.error = "Analysis failed"
                record.time_total = (time.time() - start_total) * 1000
                return record
            
            predicted, types = analysis_result
            
            # Step 2: Simulation
            actual, time_simulation = self._run_simulation(layer)
            record.time_simulation = time_simulation
            
            if actual is None:
                record.error = "Simulation failed"
                record.time_total = (time.time() - start_total) * 1000
                return record
            
            # Step 3: Validation
            val_result, time_validation = self._run_validation(
                predicted, actual, types, layer
            )
            record.time_validation = time_validation
            
            if val_result is None:
                record.error = "Validation failed"
                record.time_total = (time.time() - start_total) * 1000
                return record
            
            # Extract validation metrics
            overall = val_result.overall
            
            # Correlation metrics
            record.spearman = overall.correlation.spearman
            record.spearman_p = overall.correlation.spearman_p
            record.pearson = overall.correlation.pearson
            record.kendall = overall.correlation.kendall
            
            # Classification metrics
            record.f1_score = overall.classification.f1_score
            record.precision = overall.classification.precision
            record.recall = overall.classification.recall
            record.accuracy = overall.classification.accuracy
            
            # Confusion matrix
            cm = overall.classification.confusion_matrix
            record.true_positives = cm.get("tp", 0)
            record.false_positives = cm.get("fp", 0)
            record.true_negatives = cm.get("tn", 0)
            record.false_negatives = cm.get("fn", 0)
            
            # Error metrics
            record.rmse = overall.error.rmse
            record.mae = overall.error.mae
            
            # Ranking metrics
            record.top5_overlap = overall.ranking.top_5_overlap
            record.top10_overlap = overall.ranking.top_10_overlap
            
            # Check pass status
            record.passed = val_result.passed
            
            # Count targets met
            targets_met = 0
            if record.spearman >= VALIDATION_TARGETS["spearman"]:
                targets_met += 1
            if record.f1_score >= VALIDATION_TARGETS["f1"]:
                targets_met += 1
            if record.precision >= VALIDATION_TARGETS["precision"]:
                targets_met += 1
            if record.recall >= VALIDATION_TARGETS["recall"]:
                targets_met += 1
            if record.top5_overlap >= VALIDATION_TARGETS["top5_overlap"]:
                targets_met += 1
            if record.top10_overlap >= VALIDATION_TARGETS["top10_overlap"]:
                targets_met += 1
            
            record.targets_met = targets_met
            record.targets_total = 6
        
        except Exception as e:
            record.error = str(e)
            if self.verbose:
                traceback.print_exc()
        
        record.time_total = (time.time() - start_total) * 1000
        return record
    
    def run(self) -> BenchmarkSummary:
        """Execute the complete benchmark suite."""
        print_header("SOFTWARE-AS-A-GRAPH BENCHMARK SUITE", "═")
        
        # Configuration summary
        print(f"\n  {colored('Scales:', Colors.CYAN)} {', '.join(self.scales)}")
        print(f"  {colored('Layers:', Colors.CYAN)} {', '.join(self.layers)}")
        print(f"  {colored('Runs per config:', Colors.CYAN)} {self.runs}")
        print(f"  {colored('Output:', Colors.CYAN)} {self.output_dir}")
        
        # Calculate total iterations
        total_configs = len(self.scales)
        total_iterations = total_configs * len(self.layers) * self.runs
        
        print(f"  {colored('Total iterations:', Colors.CYAN)} {total_iterations}")
        
        # Load modules
        print_section("Loading Modules")
        if not self._load_modules():
            print_result(False, "Failed to load required modules")
            return BenchmarkSummary(
                timestamp=datetime.now().isoformat(),
                duration=0,
                total_runs=0,
                passed_runs=0,
            )
        print_result(True, "All modules loaded successfully")
        
        # Start benchmark
        start_time = time.time()
        iteration = 0
        
        temp_data_file = self.output_dir / "temp_benchmark_data.json"
        
        for scale in self.scales:
            print_section(f"Scale: {scale.upper()}")
            
            for run_idx in range(self.runs):
                seed = self.base_seed + run_idx
                    
                # Generate and import data
                print(f"\n    Run {run_idx + 1}/{self.runs} (seed={seed})")
                print(f"      Generating...", end=" ", flush=True)
                    
                gen_success, gen_time = self._generate_data(
                    scale, seed, temp_data_file
                )
                    
                if not gen_success:
                    print(colored("FAILED", Colors.RED))
                    continue
                    
                print(colored(f"OK ({gen_time:.0f}ms)", Colors.GREEN))
                print(f"      Importing...", end=" ", flush=True)
                    
                imp_success, imp_time = self._import_data(temp_data_file)
                    
                if not imp_success:
                    print(colored("FAILED", Colors.RED))
                    continue
                    
                print(colored(f"OK ({imp_time:.0f}ms)", Colors.GREEN))
                    
                # Run benchmarks for each layer
                for layer in self.layers:
                    iteration += 1
                    layer_name = LAYER_DEFINITIONS.get(layer, {}).get("name", layer)
                        
                    print(f"      {layer_name}...", end=" ", flush=True)
                        
                    record = self._run_single_benchmark(
                        scale, layer, seed, run_idx
                    )
                    record.time_generation = gen_time
                    record.time_import = imp_time
                        
                    self.records.append(record)
                        
                    if record.error:
                        print(colored(f"ERROR: {record.error}", Colors.RED))
                    elif record.passed:
                        print(colored(
                            f"PASS ρ={record.spearman:.3f} F1={record.f1_score:.3f} "
                            f"({record.time_analysis:.0f}+{record.time_simulation:.0f}ms)",
                            Colors.GREEN
                        ))
                    else:
                        print(colored(
                            f"FAIL ρ={record.spearman:.3f} F1={record.f1_score:.3f} "
                            f"({record.targets_met}/{record.targets_total} targets)",
                            Colors.YELLOW         
                        ))
        
        # Cleanup temp file
        if temp_data_file.exists():
            temp_data_file.unlink()
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(duration)
        
        # Save results
        self._save_results(summary)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    # =========================================================================
    # Results Processing
    # =========================================================================
    
    def _generate_summary(self, duration: float) -> BenchmarkSummary:
        """Generate benchmark summary from collected records."""
        summary = BenchmarkSummary(
            timestamp=datetime.now().isoformat(),
            duration=duration,
            total_runs=len(self.records),
            passed_runs=sum(1 for r in self.records if r.passed),
            scales=self.scales,
            layers=self.layers,
            records=self.records,
        )
        
        if not self.records:
            return summary
        
        # Calculate overall metrics
        valid_records = [r for r in self.records if not r.error]
        
        if valid_records:
            summary.overall_spearman = statistics.mean(r.spearman for r in valid_records)
            summary.overall_f1 = statistics.mean(r.f1_score for r in valid_records)
            summary.overall_pass_rate = summary.passed_runs / len(valid_records)
        
        # Generate aggregates
        grouped = defaultdict(list)
        for r in valid_records:
            key = (r.scale, r.layer)
            grouped[key].append(r)
        
        for (scale, layer), records in grouped.items():
            agg = self._aggregate_records(scale, layer, records)
            summary.aggregates.append(agg)
        
        # Find best/worst configurations
        if summary.aggregates:
            best = max(summary.aggregates, key=lambda a: a.avg_spearman)
            worst = min(summary.aggregates, key=lambda a: a.avg_spearman)
            
            summary.best_config = f"{best.scale}/{best.layer} (ρ={best.avg_spearman:.3f})"
            summary.worst_config = f"{worst.scale}/{worst.layer} (ρ={worst.avg_spearman:.3f})"
        
        return summary
    
    def _aggregate_records(
        self,
        scale: str,
        layer: str,
        records: List[BenchmarkRecord],
    ) -> AggregateResult:
        """Aggregate multiple records into summary statistics."""
        agg = AggregateResult(
            scale=scale,
            layer=layer,
            num_runs=len(records),
        )
        
        if not records:
            return agg
        
        # Graph stats
        agg.avg_nodes = statistics.mean(r.nodes for r in records)
        agg.avg_edges = statistics.mean(r.edges for r in records)
        agg.avg_density = statistics.mean(r.density for r in records)
        
        # Timing stats
        agg.avg_time_analysis = statistics.mean(r.time_analysis for r in records)
        agg.avg_time_simulation = statistics.mean(r.time_simulation for r in records)
        agg.avg_time_total = statistics.mean(r.time_total for r in records)
        
        if agg.avg_time_analysis > 0:
            agg.speedup_ratio = agg.avg_time_simulation / agg.avg_time_analysis
        
        # Validation stats
        agg.avg_spearman = statistics.mean(r.spearman for r in records)
        agg.avg_f1 = statistics.mean(r.f1_score for r in records)
        agg.avg_precision = statistics.mean(r.precision for r in records)
        agg.avg_recall = statistics.mean(r.recall for r in records)
        agg.avg_top5 = statistics.mean(r.top5_overlap for r in records)
        agg.avg_top10 = statistics.mean(r.top10_overlap for r in records)
        agg.avg_rmse = statistics.mean(r.rmse for r in records)
        
        if len(records) > 1:
            agg.std_spearman = statistics.stdev(r.spearman for r in records)
            agg.std_f1 = statistics.stdev(r.f1_score for r in records)
        
        # Pass rate
        agg.num_passed = sum(1 for r in records if r.passed)
        agg.pass_rate = agg.num_passed / len(records)
        
        return agg
    
    # =========================================================================
    # Report Generation
    # =========================================================================
    
    def _save_results(self, summary: BenchmarkSummary) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV file
        csv_file = self.output_dir / f"benchmark_{timestamp}.csv"
        self._write_csv(csv_file)
        
        # JSON file
        json_file = self.output_dir / f"benchmark_{timestamp}.json"
        self._write_json(json_file, summary)
        
        # Markdown report
        md_file = self.output_dir / f"report_{timestamp}.md"
        self._write_markdown(md_file, summary)
        
        print_section("Results Saved")
        print_result(True, f"CSV: {csv_file}")
        print_result(True, f"JSON: {json_file}")
        print_result(True, f"Report: {md_file}")
    
    def _write_csv(self, filepath: Path) -> None:
        """Write records to CSV file."""
        if not self.records:
            return
        
        # Get field names (excluding complex types)
        exclude_fields = {"components_by_type"}
        fieldnames = [
            f for f in self.records[0].to_dict().keys()
            if f not in exclude_fields
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.records:
                row = {k: v for k, v in record.to_dict().items() if k not in exclude_fields}
                writer.writerow(row)
    
    def _write_json(self, filepath: Path, summary: BenchmarkSummary) -> None:
        """Write summary to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
    
    def _write_markdown(self, filepath: Path, summary: BenchmarkSummary) -> None:
        """Write markdown report."""
        with open(filepath, 'w') as f:
            f.write("# Benchmark Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Duration:** {summary.duration:.2f}s\n")
            f.write(f"**Total Runs:** {summary.total_runs}\n")
            f.write(f"**Passed:** {summary.passed_runs} ({summary.overall_pass_rate*100:.1f}%)\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- **Scales:** {', '.join(summary.scales)}\n")
            f.write(f"- **Layers:** {', '.join(summary.layers)}\n")
            
            # Overall Results
            f.write("## Overall Results\n\n")
            f.write(f"- **Average Spearman ρ:** {summary.overall_spearman:.4f}")
            f.write(f" {'✅' if summary.overall_spearman >= VALIDATION_TARGETS['spearman'] else '❌'}\n")
            f.write(f"- **Average F1 Score:** {summary.overall_f1:.4f}")
            f.write(f" {'✅' if summary.overall_f1 >= VALIDATION_TARGETS['f1'] else '❌'}\n")
            f.write(f"- **Best Configuration:** {summary.best_config}\n")
            f.write(f"- **Worst Configuration:** {summary.worst_config}\n\n")
            
            # Detailed Results Table
            f.write("## Performance by Configuration\n\n")
            f.write("| Scale | Layer | Runs | Nodes | Spearman ρ | F1 Score | ")
            f.write("Analysis (ms) | Simulation (ms) | Speedup | Pass Rate |\n")
            f.write("|-------|-------|------|-------|------------|----------|")
            f.write("---------------|-----------------|---------|----------|\n")
            
            # Sort aggregates
            sorted_aggs = sorted(
                summary.aggregates,
                key=lambda a: (
                    self.scales.index(a.scale) if a.scale in self.scales else 99,
                    self.layers.index(a.layer) if a.layer in self.layers else 99,
                )
            )
            
            for agg in sorted_aggs:
                rho_status = "✅" if agg.avg_spearman >= VALIDATION_TARGETS["spearman"] else "❌"
                f1_status = "✅" if agg.avg_f1 >= VALIDATION_TARGETS["f1"] else "❌"
                
                f.write(f"| {agg.scale} | {agg.layer} | {agg.num_runs} | ")
                f.write(f"{agg.avg_nodes:.0f} | ")
                f.write(f"{agg.avg_spearman:.4f} {rho_status} | ")
                f.write(f"{agg.avg_f1:.4f} {f1_status} | ")
                f.write(f"{agg.avg_time_analysis:.0f} | ")
                f.write(f"{agg.avg_time_simulation:.0f} | ")
                f.write(f"{agg.speedup_ratio:.1f}x | ")
                f.write(f"{agg.pass_rate*100:.0f}% |\n")
            
            # Validation Targets
            f.write("\n## Validation Targets\n\n")
            f.write("| Metric | Target | Achieved | Status |\n")
            f.write("|--------|--------|----------|--------|\n")
            
            for metric, target in VALIDATION_TARGETS.items():
                if metric == "spearman":
                    achieved = summary.overall_spearman
                elif metric == "f1":
                    achieved = summary.overall_f1
                else:
                    # Calculate average for other metrics
                    valid = [r for r in self.records if not r.error]
                    if valid:
                        if metric == "precision":
                            achieved = statistics.mean(r.precision for r in valid)
                        elif metric == "recall":
                            achieved = statistics.mean(r.recall for r in valid)
                        elif metric == "top5_overlap":
                            achieved = statistics.mean(r.top5_overlap for r in valid)
                        elif metric == "top10_overlap":
                            achieved = statistics.mean(r.top10_overlap for r in valid)
                        else:
                            achieved = 0.0
                    else:
                        achieved = 0.0
                
                status = "✅ PASS" if achieved >= target else "❌ FAIL"
                f.write(f"| {metric} | ≥{target:.2f} | {achieved:.4f} | {status} |\n")
            
            # Key Findings
            f.write("\n## Key Findings\n\n")
            
            if summary.overall_spearman >= VALIDATION_TARGETS["spearman"]:
                f.write("1. **Topological metrics reliably predict criticality**: ")
                f.write(f"Strong correlation (ρ={summary.overall_spearman:.3f}) demonstrates ")
                f.write("that graph structure captures system vulnerabilities.\n\n")
            
            if summary.aggregates:
                # Check if performance improves at scale
                small_aggs = [a for a in summary.aggregates if a.scale == "small"]
                large_aggs = [a for a in summary.aggregates if a.scale == "large"]
                
                if small_aggs and large_aggs:
                    small_rho = statistics.mean(a.avg_spearman for a in small_aggs)
                    large_rho = statistics.mean(a.avg_spearman for a in large_aggs)
                    
                    if large_rho > small_rho:
                        f.write("2. **Performance improves at scale**: ")
                        f.write(f"Larger systems show better metrics ")
                        f.write(f"(small: ρ={small_rho:.3f}, large: ρ={large_rho:.3f}).\n\n")
                
                # Speedup analysis
                avg_speedup = statistics.mean(a.speedup_ratio for a in summary.aggregates if a.speedup_ratio > 0)
                if avg_speedup > 1:
                    f.write(f"3. **Analysis is {avg_speedup:.1f}x faster than simulation**: ")
                    f.write("Static analysis provides significant time savings over exhaustive testing.\n\n")
            
            f.write("---\n")
            f.write(f"*Generated by Software-as-a-Graph Benchmark Suite*\n")
    
    def _print_summary(self, summary: BenchmarkSummary) -> None:
        """Print summary to terminal."""
        print_header("BENCHMARK SUMMARY", "─")
        
        print(f"\n  {colored('Duration:', Colors.CYAN)} {summary.duration:.2f}s")
        print(f"  {colored('Total Runs:', Colors.CYAN)} {summary.total_runs}")
        print(f"  {colored('Passed:', Colors.CYAN)} {summary.passed_runs} ({summary.overall_pass_rate*100:.1f}%)")
        
        print(f"\n  {colored('Overall Metrics:', Colors.BOLD, bold=True)}")
        
        rho_status = "✓" if summary.overall_spearman >= VALIDATION_TARGETS["spearman"] else "✗"
        rho_color = Colors.GREEN if summary.overall_spearman >= VALIDATION_TARGETS["spearman"] else Colors.RED
        print(f"    Spearman ρ: {colored(f'{summary.overall_spearman:.4f} {rho_status}', rho_color)}")
        
        f1_status = "✓" if summary.overall_f1 >= VALIDATION_TARGETS["f1"] else "✗"
        f1_color = Colors.GREEN if summary.overall_f1 >= VALIDATION_TARGETS["f1"] else Colors.RED
        print(f"    F1 Score:   {colored(f'{summary.overall_f1:.4f} {f1_status}', f1_color)}")
        
        if summary.best_config:
            print(f"\n  {colored('Best:', Colors.GREEN)} {summary.best_config}")
        if summary.worst_config:
            print(f"  {colored('Worst:', Colors.YELLOW)} {summary.worst_config}")
        
        print()


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Suite for Graph-Based Criticality Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scales:
  tiny    5-10 nodes     Minimal test system
  small   10-25 nodes    Small deployment
  medium  30-50 nodes    Medium deployment
  large   60-100 nodes   Large deployment
  xlarge  150-300 nodes  Enterprise scale

Layers:
  app      Application layer
  infra    Infrastructure layer
  mw-app   Middleware-Application
  mw-infra Middleware-Infrastructure
  system   Complete system

Examples:
  %(prog)s --scales small,medium,large --runs 5
  %(prog)s --layers app,infra,system
  %(prog)s --full-suite --output results/benchmark
        """
    )
    
    # Benchmark configuration
    config = parser.add_argument_group("Benchmark Configuration")
    config.add_argument(
        "--scales",
        default="small,medium,large",
        help="Comma-separated scales (default: small,medium,large)"
    )
    config.add_argument(
        "--layers",
        default="app,infra,system",
        help="Comma-separated layers (default: app,infra,system)"
    )
    config.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Runs per configuration (default: 3)"
    )
    config.add_argument(
        "--full-suite",
        action="store_true",
        help="Run complete benchmark suite (all scales, layers)"
    )
    
    # Neo4j connection
    neo4j = parser.add_argument_group("Neo4j Connection")
    neo4j.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687)"
    )
    neo4j.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--output", "-o",
        default="benchmark_results",
        help="Output directory (default: benchmark_results)"
    )
    output.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Parse configuration
    if args.full_suite:
        scales = ["small", "medium", "large"]
        layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    else:
        scales = [s.strip() for s in args.scales.split(",")]
        layers = [l.strip() for l in args.layers.split(",")]
    
    # Validate scales
    valid_scales = [s for s in scales if s in SCALE_DEFINITIONS]
    if not valid_scales:
        print(f"No valid scales specified: {args.scales}")
        print(f"Valid scales: {', '.join(SCALE_DEFINITIONS.keys())}")
        return 1
    
    # Validate layers
    valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
    if not valid_layers:
        print(f"No valid layers specified: {args.layers}")
        print(f"Valid layers: {', '.join(LAYER_DEFINITIONS.keys())}")
        return 1
    
    # Create runner
    runner = BenchmarkRunner(
        scales=valid_scales,
        layers=valid_layers,
        runs=args.runs,
        output_dir=Path(args.output),
        uri=args.uri,
        user=args.user,
        password=args.password,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    try:
        summary = runner.run()
        return 0 if summary.overall_pass_rate >= 0.5 else 1
    
    except KeyboardInterrupt:
        print(f"\n{colored('Benchmark interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Benchmark failed")
        print(f"\n{colored(f'Benchmark failed: {e}', Colors.RED)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())