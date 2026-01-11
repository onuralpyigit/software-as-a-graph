#!/usr/bin/env python3
"""
Multi-Layer Benchmark Suite

Evaluates the Software-as-a-Graph methodology across varying system scales and layers.
Focuses on two key dimensions:
1. Predictive Accuracy (Correlation, F1 Score)
2. Computational Performance (Analysis Time vs. Simulation Time)

Workflow per Iteration:
1. Generate synthetic system topology (CLI)
2. Import data into Neo4j (CLI)
3. Run Structural & Quality Analysis (Python API)
4. Run Exhaustive Failure Simulation (Python API)
5. Statistical Validation (Python API)

Usage:
    python benchmark.py --scales small medium --runs 3
    python benchmark.py --layers application infrastructure --output results/v1

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import subprocess
import time
import statistics
import csv
import json
import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import GraphAnalyzer
from src.simulation import Simulator
from src.validation import Validator, ValidationTargets

# --- Configuration ---
DEFAULT_SCALES = ["small", "medium", "large"]
DEFAULT_LAYERS = ["application", "infrastructure", "complete"]

# ANSI Colors
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "RESET": "\033[0m"
}

def print_c(msg, color="RESET", bold=False):
    style = COLORS["BOLD"] if bold else ""
    code = COLORS.get(color, COLORS["RESET"])
    reset = COLORS["RESET"]
    print(f"{style}{code}{msg}{reset}")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Benchmark")

@dataclass
class BenchmarkRecord:
    """Represents a single benchmark iteration result."""
    run_id: str
    timestamp: str
    scale: str
    layer: str
    seed: int
    
    # Graph Stats
    nodes: int
    edges: int
    density: float
    
    # Performance Timings (ms)
    time_analysis: float
    time_simulation: float
    time_total: float
    
    # Validation Metrics
    spearman_rho: float
    f1_score: float
    rmse: float
    top5_overlap: float
    
    # Status
    passed: bool
    error: Optional[str] = None

class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.output_dir = Path(args.output).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.python_exe = sys.executable
        
        # Neo4j args for subprocesses
        self.neo4j_args = [
            "--uri", args.uri,
            "--user", args.user,
            "--password", args.password
        ]
        
        self.results: List[BenchmarkRecord] = []

    def run(self):
        """Main execution loop."""
        print_c("=== Multi-Layer Graph Methodology Benchmark ===", "HEADER", bold=True)
        print(f"Scales: {self.args.scales}")
        print(f"Layers: {self.args.layers}")
        print(f"Runs:   {self.args.runs} per config")
        print(f"Target: {self.args.uri}\n")

        start_global = time.time()
        
        # Iterate through scales (Outer loop to minimize DB churn/resizes)
        for scale in self.args.scales:
            print_c(f"\n>>> Benchmarking Scale: {scale.upper()}", "BLUE", bold=True)
            
            for i in range(self.args.runs):
                seed = self.args.seed + i
                run_id = f"{scale}_{seed}"
                
                # 1. Setup Environment (Generate & Import)
                if not self._setup_environment(scale, seed):
                    print_c(f"Skipping run {run_id} due to setup failure.", "RED")
                    continue
                
                # 2. Run Layers
                for layer in self.args.layers:
                    self._run_layer_benchmark(run_id, scale, layer, seed)

        duration = time.time() - start_global
        self._save_results(duration)

    def _setup_environment(self, scale: str, seed: int) -> bool:
        """Generates graph data and imports it into Neo4j."""
        temp_file = self.output_dir / "temp_benchmark_data.json"
        
        try:
            # A. Generate
            cmd_gen = [
                self.python_exe, str(self.project_root / "generate_graph.py"),
                "--scale", scale,
                "--seed", str(seed),
                "--output", str(temp_file)
            ]
            subprocess.run(cmd_gen, check=True, capture_output=True)

            # B. Import (Clears DB)
            cmd_import = [
                self.python_exe, str(self.project_root / "import_graph.py"),
                "--input", str(temp_file),
                "--clear"
            ] + self.neo4j_args
            subprocess.run(cmd_import, check=True, capture_output=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print_c(f"Setup failed: {e}", "RED")
            return False

    def _run_layer_benchmark(self, run_id: str, scale: str, layer: str, seed: int):
        """Executes the Analyze -> Simulate -> Validate pipeline for a specific layer."""
        print(f"   Running {layer:<15} ... ", end="", flush=True)
        
        # Initialize resources
        analyzer = GraphAnalyzer(self.args.uri, self.args.user, self.args.password)
        simulator = Simulator(self.args.uri, self.args.user, self.args.password)
        validator = Validator(ValidationTargets()) # Use default targets

        record = BenchmarkRecord(
            run_id=run_id, timestamp=datetime.now().isoformat(),
            scale=scale, layer=layer, seed=seed,
            nodes=0, edges=0, density=0.0,
            time_analysis=0.0, time_simulation=0.0, time_total=0.0,
            spearman_rho=0.0, f1_score=0.0, rmse=0.0, top5_overlap=0.0,
            passed=False
        )

        try:
            t_start = time.time()

            # 1. Analysis (Prediction)
            t0 = time.time()
            analysis_res = analyzer.analyze_layer(layer)
            t_analysis = (time.time() - t0) * 1000
            
            # Extract Graph Stats
            record.nodes = analysis_res.structural.graph_summary.nodes
            record.edges = analysis_res.structural.graph_summary.edges
            record.density = analysis_res.structural.graph_summary.density

            # 2. Simulation (Ground Truth)
            t0 = time.time()
            sim_results = simulator.run_failure_simulation_exhaustive(layer=layer)
            t_simulation = (time.time() - t0) * 1000
            
            # 3. Validation
            pred_scores = {c.id: c.scores.overall for c in analysis_res.quality.components}
            actual_scores = {r.target_id: r.impact.composite_impact for r in sim_results}
            comp_types = {c.id: c.type for c in analysis_res.quality.components}

            val_res = validator.validate(pred_scores, actual_scores, comp_types, layer=layer)
            
            # Fill Record
            record.time_analysis = t_analysis
            record.time_simulation = t_simulation
            record.time_total = (time.time() - t_start) * 1000
            
            metrics = val_res.overall
            record.spearman_rho = metrics.correlation.spearman
            record.f1_score = metrics.classification.f1_score
            record.rmse = metrics.error.rmse
            record.top5_overlap = metrics.ranking.top_5_overlap
            record.passed = val_res.passed
            
            self.results.append(record)
            
            # Console Feedback
            status_color = "GREEN" if record.passed else "RED"
            print_c(f"{'PASS' if record.passed else 'FAIL'}", status_color, bold=True)
            print(f"     ├─ Analysis: {t_analysis:.0f}ms | Sim: {t_simulation:.0f}ms")
            print(f"     └─ Rho: {record.spearman_rho:.3f} | F1: {record.f1_score:.3f}")

        except Exception as e:
            print_c("ERROR", "RED", bold=True)
            logger.error(f"Benchmark error: {traceback.format_exc()}")
            record.error = str(e)
            self.results.append(record)
        
        finally:
            # Cleanup
            if analyzer: analyzer.__exit__(None, None, None)
            if simulator: simulator.__exit__(None, None, None)

    def _save_results(self, total_duration: float):
        """Generates CSV, JSON, and Markdown reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV
        csv_file = self.output_dir / f"benchmark_{timestamp}.csv"
        if self.results:
            keys = [f.name for f in fields(BenchmarkRecord)]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))

        # 2. Markdown Summary
        md_file = self.output_dir / f"report_{timestamp}.md"
        self._write_markdown_report(md_file, total_duration)

        print_c("\n=== Benchmark Completed ===", "HEADER", bold=True)
        print(f"Total Time: {total_duration:.2f}s")
        print(f"Records:    {len(self.results)}")
        print(f"CSV Data:   {csv_file}")
        print(f"Report:     {md_file}")

    def _write_markdown_report(self, filepath: Path, duration: float):
        """Writes a formatted markdown summary."""
        # Aggregate data
        grouped = defaultdict(list)
        for r in self.results:
            if not r.error:
                grouped[(r.layer, r.scale)].append(r)
        
        with open(filepath, 'w') as f:
            f.write(f"# Benchmark Report\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Duration:** {duration:.2f}s\n\n")
            
            f.write("## Performance Summary\n")
            f.write("| Layer | Scale | Runs | Spearman (ρ) | F1 Score | Analysis (ms) | Simulation (ms) | Speedup |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            
            # Sort by Layer then Scale
            sorted_keys = sorted(grouped.keys(), key=lambda x: (x[0], DEFAULT_SCALES.index(x[1]) if x[1] in DEFAULT_SCALES else 99))

            for layer, scale in sorted_keys:
                recs = grouped[(layer, scale)]
                n = len(recs)
                avg_rho = statistics.mean(r.spearman_rho for r in recs)
                avg_f1 = statistics.mean(r.f1_score for r in recs)
                avg_an = statistics.mean(r.time_analysis for r in recs)
                avg_sim = statistics.mean(r.time_simulation for r in recs)
                
                # Speedup: How much faster is Analysis than exhaustive Simulation?
                speedup = f"{avg_sim / avg_an:.1f}x" if avg_an > 0 else "N/A"
                
                f.write(f"| **{layer.title()}** | {scale.title()} | {n} | "
                        f"{avg_rho:.3f} | {avg_f1:.3f} | {avg_an:.0f} | {avg_sim:.0f} | **{speedup}** |\n")

from dataclasses import fields

def main():
    parser = argparse.ArgumentParser(description="Software-as-a-Graph Benchmark Runner")
    parser.add_argument("--scales", nargs="+", default=DEFAULT_SCALES, choices=["tiny", "small", "medium", "large", "xlarge"])
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS, choices=DEFAULT_LAYERS)
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    parser.add_argument("--output", default="output/benchmarks", help="Output directory")
    
    # DB Args
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")

    args = parser.parse_args()
    runner = BenchmarkRunner(args)
    runner.run()

if __name__ == "__main__":
    main()