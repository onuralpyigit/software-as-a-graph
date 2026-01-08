#!/usr/bin/env python3
"""
Multi-Layer Benchmark Script

Evaluates the Graph-Based Criticality Analysis Methodology across:
- System Scales: (small, medium, large, xlarge)
- Graph Layers: (Application, Infrastructure, Complete System)
- Domain Scenarios: (iot, financial, healthcare)

Workflow per Iteration:
1. Generate synthetic system data (CLI)
2. Import data into Neo4j (CLI)
3. Run Validation Pipeline for EACH layer (Application, Infrastructure, Complete)
4. Record statistical accuracy metrics (Spearman, F1, RMSE)

Output:
- benchmark_results.csv: Raw data for every run/layer
- benchmark_report.md: Aggregated summary table
"""

import argparse
import sys
import subprocess
import json
import time
import statistics
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.pipeline import ValidationPipeline
from src.validation.metrics import ValidationTargets

# --- Configuration ---
SCALES = ["small", "medium", "large"]
LAYERS = ["application", "infrastructure", "complete"]

# ANSI Colors
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

@dataclass
class BenchmarkRecord:
    """Single benchmark data point."""
    run_id: str
    timestamp: str
    scale: str
    layer: str
    seed: int
    
    # Validation Metrics (Prediction vs Reality)
    spearman: float
    f1_score: float
    rmse: float
    top5_overlap: float
    precision: float
    recall: float
    
    # Metadata
    passed: bool
    duration_ms: float
    nodes_count: int

@dataclass
class LayerSummary:
    """Aggregated stats for a specific configuration."""
    layer: str
    scale: str
    n_runs: int
    avg_spearman: float
    avg_f1: float
    avg_rmse: float
    success_rate: float
    avg_duration: float

class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.python_exe = sys.executable
        
        # Neo4j connection args for subprocesses
        self.neo4j_args = [
            "--uri", args.uri,
            "--user", args.user,
            "--password", args.password
        ]
        
        self.results: List[BenchmarkRecord] = []

    def run(self):
        """Main execution loop."""
        print(f"{BOLD}{HEADER}=== Multi-Layer Graph Benchmark ==={RESET}")
        print(f"Scales:    {self.args.scales}")
        print(f"Layers:    {self.args.layers}")
        print(f"Runs:      {self.args.runs} per config")
        print(f"Output:    {self.output_dir}\n")

        start_time = time.time()
        
        # 1. Loop Scales
        for scale in self.args.scales:
            for i in range(self.args.runs):
                seed = self.args.seed + i
                run_id = f"{scale}_{seed}"
                self._run_iteration(scale, seed, run_id)

        duration = time.time() - start_time
        self._generate_reports(duration)

    def _run_iteration(self, scale: str, seed: int, run_id: str):
        """
        Orchestrates one full iteration:
        Generate -> Import -> Validate(Layer 1) -> Validate(Layer 2)...
        """
        print(f"{BLUE}>> Iteration: {run_id} {RESET}")
        
        input_file = self.output_dir / "temp_system.json"
        
        try:
            # A. Generate Data (Subprocess)
            cmd_gen = [
                self.python_exe, str(self.project_root / "generate_graph.py"),
                "--scale", scale,
                "--seed", str(seed),
                "--output", str(input_file)
            ]
            subprocess.run(cmd_gen, check=True, capture_output=True)

            # B. Import Data (Subprocess)
            cmd_import = [
                self.python_exe, str(self.project_root / "import_graph.py"),
                "--input", str(input_file),
                "--clear"
            ] + self.neo4j_args
            subprocess.run(cmd_import, check=True, capture_output=True)

            # C. Validate Each Layer
            # We use the Python API directly here for efficiency instead of CLI subprocess
            with ValidationPipeline(self.args.uri, self.args.user, self.args.password) as pipeline:
                
                for layer in self.args.layers:
                    t0 = time.time()
                    
                    # Run Validation (Analysis -> Simulation -> Comparison)
                    result = pipeline.run(
                        layer=layer,
                        targets=ValidationTargets(spearman=0.7, f1_score=0.8)
                    )
                    
                    dt = (time.time() - t0) * 1000
                    metrics = result.overall.to_dict()["metrics"]
                    
                    # Record Result
                    record = BenchmarkRecord(
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                        scale=scale,
                        layer=layer,
                        seed=seed,
                        
                        spearman=metrics["rho"],
                        f1_score=metrics["f1"],
                        rmse=metrics["rmse"],
                        top5_overlap=metrics["top5_overlap"],
                        precision=metrics["precision"],
                        recall=metrics["recall"],
                        
                        passed=result.overall.passed,
                        duration_ms=dt,
                        nodes_count=result.overall.sample_size
                    )
                    self.results.append(record)
                    
                    # Print quick status
                    status = f"{GREEN}PASS{RESET}" if record.passed else f"{RED}FAIL{RESET}"
                    print(f"   [{layer.ljust(15)}] {status} | Rho: {record.spearman:.3f} | F1: {record.f1_score:.3f}")

        except subprocess.CalledProcessError as e:
            print(f"{RED}Error in subprocess: {e}{RESET}")
        except Exception as e:
            print(f"{RED}Error in iteration: {e}{RESET}")

    def _generate_reports(self, total_duration: float):
        """Export CSV and Markdown summaries."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV Export
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        if self.results:
            keys = asdict(self.results[0]).keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))
        
        # 2. Aggregation
        summary_path = self.output_dir / f"benchmark_report_{timestamp}.md"
        
        # Group by (Layer, Scale)
        groups = defaultdict(list)
        for r in self.results:
            groups[(r.layer, r.scale)].append(r)
            
        lines = [
            "# Multi-Layer Graph Benchmark Report",
            f"**Date:** {datetime.now().isoformat()}",
            f"**Total Duration:** {total_duration:.2f}s",
            f"**Total Runs:** {len(self.results)}",
            "",
            "## Performance by Layer and Scale",
            "",
            "| Layer | Scale | Runs | Spearman ρ | F1 Score | RMSE | Success Rate | Avg Time (ms) |",
            "|---|---|---|---|---|---|---|---|",
        ]

        # Sort keys for consistent output
        sorted_keys = sorted(groups.keys(), key=lambda x: (x[0], SCALES.index(x[1]) if x[1] in SCALES else 99))
        
        for layer, scale in sorted_keys:
            recs = groups[(layer, scale)]
            n = len(recs)
            
            avg_rho = statistics.mean(r.spearman for r in recs)
            avg_f1 = statistics.mean(r.f1_score for r in recs)
            avg_rmse = statistics.mean(r.rmse for r in recs)
            avg_time = statistics.mean(r.duration_ms for r in recs)
            success = len([r for r in recs if r.passed]) / n
            
            lines.append(
                f"| **{layer.capitalize()}** | {scale.capitalize()} | {n} | "
                f"{avg_rho:.3f} | {avg_f1:.3f} | {avg_rmse:.3f} | "
                f"{success:.1%} | {avg_time:.0f} |"
            )
            
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
            
        print(f"\n{BOLD}{HEADER}=== Benchmark Complete ==={RESET}")
        print(f"Results CSV: {csv_path}")
        print(f"Summary Report: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Benchmark Runner")
    
    parser.add_argument("--scales", nargs="+", default=SCALES, choices=["tiny", "small", "medium", "large", "xlarge"])
    parser.add_argument("--layers", nargs="+", default=LAYERS, choices=LAYERS)
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/benchmarks")
    
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args)
    runner.run()

if __name__ == "__main__":
    main()