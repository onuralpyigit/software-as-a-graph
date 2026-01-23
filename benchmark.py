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
        
    def _neo4j_args(self) -> List[str]:
        """Get common Neo4j connection arguments."""
        return [
            "--uri", self.uri,
            "--user", self.user,
            "--password", self.password,
        ]

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
        ] + self._neo4j_args()
        
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
    
    def _run_analysis(self, layer: str, run_id: str) -> Tuple[Optional[Any], float]:
        """Run graph analysis for a layer via subprocess."""
        start = time.time()
        
        script = self.project_root / "scripts" / "analyze_graph.py"
        if not script.exists():
            script = self.project_root / "analyze_graph.py"
            
        output_file = self.output_dir / f"analysis_{run_id}.json"
        
        cmd = [
            sys.executable, str(script),
            "--layer", layer,
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.verbose:
            cmd.append("--verbose")
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            
            if not success and self.verbose:
                self.logger.warning(f"Analysis stderr: {result.stderr}")
                
            if success and output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                    # Extract layer data
                    if "layers" in data and layer in data["layers"]:
                        return data["layers"][layer], (time.time() - start) * 1000
            
            return None, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return None, (time.time() - start) * 1000
            
    def _run_simulation(self, layer: str, run_id: str) -> Tuple[Optional[Any], float]:
        """Run simulation for a layer via subprocess."""
        start = time.time()
        
        script = self.project_root / "scripts" / "simulate_graph.py"
        if not script.exists():
            script = self.project_root / "simulate_graph.py"
            
        output_file = self.output_dir / f"simulation_{run_id}.json"
        
        cmd = [
            sys.executable, str(script),
            "--layer", layer,
            "--exhaustive",
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.verbose:
            cmd.append("--verbose")
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            
            if not success and self.verbose:
                self.logger.warning(f"Simulation stderr: {result.stderr}")
                
            if success and output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                    return data, (time.time() - start) * 1000
            
            return None, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return None, (time.time() - start) * 1000
            
    def _run_validation(self, layer: str, run_id: str) -> Tuple[Optional[Any], float]:
        """Run validation via subprocess."""
        start = time.time()
        
        script = self.project_root / "scripts" / "validate_graph.py"
        if not script.exists():
            script = self.project_root / "validate_graph.py"
            
        output_file = self.output_dir / f"validation_{run_id}.json"
        
        cmd = [
            sys.executable, str(script),
            "--layer", layer,
            "--spearman", str(VALIDATION_TARGETS["spearman"]),
            "--f1", str(VALIDATION_TARGETS["f1"]),
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.verbose:
            print(f"DEBUG: Validation command: {' '.join(cmd)}")
            cmd.append("--verbose")
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            
            if not success and self.verbose:
                self.logger.warning(f"Validation stderr: {result.stderr}")
            
            # Note: validate_graph.py returns 1 if validation criteria are not met,
            # even if the script ran successfully and produced output.
            # So we rely on output_file presence as the primary success indicator.
            if output_file.exists():
                with open(output_file) as f:
                    try:
                        data = json.load(f)
                        if "layers" in data and layer in data["layers"]:
                            return data["layers"][layer], (time.time() - start) * 1000
                        else:
                            if self.verbose:
                                self.logger.warning(f"Validation JSON missing layer data. Keys: {data.keys()}")
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            self.logger.error(f"Validation JSON decode error: {e}")
            
            if not success and not output_file.exists():
                if self.verbose:
                     self.logger.warning(f"Validation failed (rc={result.returncode}) and no output file.")
                     self.logger.warning(f"Stderr: {result.stderr}")
            
            return None, (time.time() - start) * 1000
        
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
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
            analysis_data, time_analysis = self._run_analysis(layer, run_id)
            record.time_analysis = time_analysis
            
            if analysis_data:
                stats = analysis_data.get("structural", {}).get("graph_summary", {})
                record.nodes = stats.get("nodes", 0)
                record.edges = stats.get("edges", 0)
                record.density = stats.get("density", 0.0)
                record.components_by_type = stats.get("node_types", {})
            
            if analysis_data is None:
                record.error = "Analysis failed"
                record.time_total = (time.time() - start_total) * 1000
                return record
            
            # Step 2: Simulation
            sim_data, time_simulation = self._run_simulation(layer, run_id)
            record.time_simulation = time_simulation
            
            if sim_data is None:
                record.error = "Simulation failed"
                record.time_total = (time.time() - start_total) * 1000
                return record
            
            # Step 3: Validation
            val_data, time_validation = self._run_validation(layer, run_id)
            record.time_validation = time_validation
            
            if val_data is None:
                record.error = "Validation failed"
                record.time_total = (time.time() - start_total) * 1000
                # Still try to cleanup temp files
                # self._cleanup_temp_files(run_id)
                return record
            
            # Extract validation metrics
            # JSON structure: layer_data -> validation_result -> overall -> metrics
            validation_result = val_data.get("validation_result", {})
            overall = validation_result.get("overall", {})
            metrics = overall.get("metrics", {})
            
            correlation = metrics.get("correlation", {})
            classification = metrics.get("classification", {})
            ranking = metrics.get("ranking", {})
            error_metrics = metrics.get("error", {})
            
            # Correlation metrics
            record.spearman = correlation.get("spearman", 0.0)
            record.spearman_p = correlation.get("spearman_p_value", 1.0)
            record.pearson = correlation.get("pearson", 0.0)
            record.kendall = correlation.get("kendall", 0.0)
            
            # Classification metrics
            record.f1_score = classification.get("f1_score", 0.0)
            record.precision = classification.get("precision", 0.0)
            record.recall = classification.get("recall", 0.0)
            record.accuracy = classification.get("accuracy", 0.0)
            
            # Confusion matrix
            cm = classification.get("confusion_matrix", {})
            record.true_positives = cm.get("tp", 0)
            record.false_positives = cm.get("fp", 0)
            record.true_negatives = cm.get("tn", 0)
            record.false_negatives = cm.get("fn", 0)
            
            # Error metrics
            record.rmse = error_metrics.get("rmse", 0.0)
            record.mae = error_metrics.get("mae", 0.0)
            
            # Ranking metrics
            record.top5_overlap = ranking.get("top_5_overlap", 0.0)
            record.top10_overlap = ranking.get("top_10_overlap", 0.0)
            
            # Check pass status
            record.passed = val_data.get("summary", {}).get("passed", False)
            
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
            
            # Cleanup temp files
            self._cleanup_temp_files(run_id)
        
        except Exception as e:
            record.error = str(e)
            if self.verbose:
                traceback.print_exc()
        
        record.time_total = (time.time() - start_total) * 1000
        return record
        
    def _cleanup_temp_files(self, run_id: str):
        """Clean up temporary JSON files."""
        for prefix in ["analysis", "simulation", "validation"]:
            path = self.output_dir / f"{prefix}_{run_id}.json"
            if path.exists():
                try:
                    path.unlink()
                except:
                    pass
    
    def run(self) -> BenchmarkSummary:
        """Execute the complete benchmark suite."""
        print_header("SOFTWARE-AS-A-GRAPH BENCHMARK SUITE", "═")
        
        # Configuration summary
        print(f"\n  {colored('Scales:', Colors.CYAN)} {', '.join(self.scales)}")
        print(f"  {colored('Layers:', Colors.CYAN)} {', '.join(self.layers)}")
        print(f"  {colored('Runs per config:', Colors.CYAN)} {self.runs}")
        print(f"  {colored('Output:', Colors.CYAN)} {self.output_dir}")
        print(f"  {colored('Mode:', Colors.CYAN)} CLI Subprocess")
        
        # Calculate total iterations
        total_configs = len(self.scales)
        total_iterations = total_configs * len(self.layers) * self.runs
        
        print(f"  {colored('Total iterations:', Colors.CYAN)} {total_iterations}")
        
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
            try:
                temp_data_file.unlink()
            except:
                pass
        
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
            summary.best_config = self._find_best_config(summary.aggregates)
            summary.worst_config = self._find_worst_config(summary.aggregates)
        
        return summary
    
    def _aggregate_records(self, scale: str, layer: str, records: List[BenchmarkRecord]) -> AggregateResult:
        """Aggregate records for a specific configuration."""
        agg = AggregateResult(
            scale=scale,
            layer=layer,
            num_runs=len(records)
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
    
    def _find_best_config(self, aggregates: List[AggregateResult]) -> str:
        """Find best performing configuration."""
        best = max(aggregates, key=lambda a: (a.pass_rate, a.avg_spearman))
        return f"{best.scale}/{best.layer} (ρ={best.avg_spearman:.3f}, Rate={best.pass_rate:.0%})"
        
    def _find_worst_config(self, aggregates: List[AggregateResult]) -> str:
        """Find worst performing configuration."""
        worst = min(aggregates, key=lambda a: (a.pass_rate, a.avg_spearman))
        return f"{worst.scale}/{worst.layer} (ρ={worst.avg_spearman:.3f}, Rate={worst.pass_rate:.0%})"

    def _save_results(self, summary: BenchmarkSummary) -> None:
        """Save benchmark results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON output
        json_file = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
            
        # CSV output
        csv_file = self.output_dir / f"benchmark_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp", "scale", "layer", "seed",
                "nodes", "edges", "density",
                "time_total", "time_analysis", "time_simulation",
                "spearman", "f1_score", "top5_overlap", "passed"
            ])
            for r in summary.records:
                writer.writerow([
                    r.run_id, r.timestamp, r.scale, r.layer, r.seed,
                    r.nodes, r.edges, r.density,
                    f"{r.time_total:.2f}", f"{r.time_analysis:.2f}", f"{r.time_simulation:.2f}",
                    f"{r.spearman:.4f}", f"{r.f1_score:.4f}", f"{r.top5_overlap:.4f}", r.passed
                ])
                
        # Markdown report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(f"# Benchmark Report\n\n")
            f.write(f"**Date:** {summary.timestamp}\n")
            f.write(f"**Duration:** {summary.duration:.2f}s\n")
            f.write(f"**Total Runs:** {summary.total_runs}\n")
            f.write(f"**Pass Rate:** {summary.overall_pass_rate:.1%}\n\n")
            f.write(f"**Best Config:** {summary.best_config}\n")
            f.write(f"**Worst Config:** {summary.worst_config}\n\n")
            
            f.write("## Aggregated Results\n\n")
            f.write("| Scale | Layer | Runs | Pass Rate | Spearman ρ | F1 Score | Speedup |\n")
            f.write("|-------|-------|------|-----------|------------|----------|---------|\n")
            
            for agg in sorted(summary.aggregates, key=lambda x: (x.scale, x.layer)):
                f.write(
                    f"| {agg.scale} | {agg.layer} | {agg.num_runs} | {agg.pass_rate:.1%} | "
                    f"{agg.avg_spearman:.3f} | {agg.avg_f1:.3f} | {agg.speedup_ratio:.2f}x |\n"
                )

    def _print_summary(self, summary: BenchmarkSummary) -> None:
        """Print benchmark summary to console."""
        print_header("BENCHMARK COMPLETION", "─")
        
        print(f"\n  {colored('Duration:', Colors.CYAN)} {summary.duration:.2f}s")
        print(f"  {colored('Total Runs:', Colors.CYAN)} {summary.total_runs}")
        
        pass_color = Colors.GREEN if summary.overall_pass_rate == 1.0 else (
            Colors.YELLOW if summary.overall_pass_rate > 0.5 else Colors.RED
        )
        print(f"  {colored('Pass Rate:', Colors.CYAN)} {colored(f'{summary.overall_pass_rate:.1%}', pass_color, bold=True)}")
        
        print(f"  {colored('Avg Spearman:', Colors.CYAN)} {summary.overall_spearman:.3f}")
        print(f"  {colored('Avg F1 Score:', Colors.CYAN)} {summary.overall_f1:.3f}")
        
        if summary.best_config:
            print(f"\n  {colored('Best Config:', Colors.GREEN)} {summary.best_config}")
        if summary.worst_config:
            print(f"  {colored('Worst Config:', Colors.RED)} {summary.worst_config}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Graph Methodology Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--scales",
        default="medium",
        help="Comma-separated scales (tiny,small,medium,large,xlarge)"
    )
    parser.add_argument(
        "--layers",
        default="app,infra,system",
        help="Comma-separated layers (app,infra,system...)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per configuration"
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Run full suite (all scales, all layers, 3 runs)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="results/benchmark",
        help="Output directory"
    )
    
    # Neo4j connection
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Determine config
    if args.full_suite:
        scales = ["tiny", "small", "medium", "large"]
        layers = ["app", "infra", "system"]
        runs = 3
    else:
        scales = [s.strip() for s in args.scales.split(",")]
        layers = [l.strip() for l in args.layers.split(",")]
        runs = args.runs
        
    # Validate
    valid_scales = [s for s in scales if s in SCALE_DEFINITIONS]
    if not valid_scales:
        print(colored("Error: No valid scales", Colors.RED))
        return 1
        
    valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
    if not valid_layers:
        print(colored("Error: No valid layers", Colors.RED))
        return 1
        
    # Run
    runner = BenchmarkRunner(
        scales=valid_scales,
        layers=valid_layers,
        runs=runs,
        output_dir=Path(args.output),
        uri=args.uri,
        user=args.user,
        password=args.password,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    try:
        runner.run()
        return 0
    except KeyboardInterrupt:
        print(f"\n{colored('Benchmark interrupted.', Colors.YELLOW)}")
        return 130
    except Exception as e:
        print(colored(f"Benchmark failed: {e}", Colors.RED))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())