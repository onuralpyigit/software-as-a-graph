#!/usr/bin/env python3
"""
CLI-Based Benchmark Orchestrator

This script executes the full "Software-as-a-Graph" pipeline using the 
command-line interfaces of the individual tool components. 

Workflow per run:
1. Generate (generate_graph.py) -> JSON
2. Import (import_graph.py) -> Neo4j
3. Analyze (analyze_graph.py) -> Audit Log
4. Simulate (simulate_graph.py) -> Ground Truth Dataset
5. Validate (validate_graph.py) -> Metrics JSON
"""

import argparse
import subprocess
import json
import csv
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime

# Configuration Defaults
DEFAULT_SCALES = ["small", "medium", "large"]
DEFAULT_SEEDS = [42, 43, 44]
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def print_step(msg):
    print(f"{Colors.BLUE}[STEP]{Colors.END} {msg}")

def run_command(cmd, log_file=None):
    """Executes a shell command and optionally logs stdout."""
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"CMD: {' '.join(cmd)}\n")
                f.write(result.stdout + "\n")
                f.write("-" * 40 + "\n")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Command failed:{Colors.END} {' '.join(cmd)}")
        print(f"{Colors.RED}Error:{Colors.END} {e.stderr}")
        raise

class CLIBenchmarkRunner:
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def run_iteration(self, scale, seed, run_id):
        """Runs one full pipeline iteration."""
        timestamp = datetime.now().strftime("%H%M%S")
        file_prefix = f"{self.data_dir}/{run_id}_{timestamp}"
        log_file = f"{self.logs_dir}/{run_id}.log"
        
        graph_file = f"{file_prefix}_graph.json"
        sim_file = f"{file_prefix}_sim.json"
        val_file = f"{file_prefix}_val.json"

        start_time = time.time()

        # 1. GENERATE
        # Purpose: Create synthetic topology [cite: 563]
        print_step(f"Generating graph (Scale: {scale}, Seed: {seed})...")
        run_command([
            sys.executable, "generate_graph.py",
            "--scale", scale,
            "--seed", str(seed),
            "--output", graph_file
        ], log_file)

        # 2. IMPORT
        # Purpose: Load into Neo4j and compute static metrics (Formula 1-6) [cite: 586, 610]
        print_step("Importing to Neo4j...")
        run_command([
            sys.executable, "import_graph.py",
            "--input", graph_file,
            "--clear", # Clear DB for fresh start
            "--uri", NEO4J_URI,
            "--user", NEO4J_USER,
            "--password", NEO4J_PASS
        ], log_file)

        # 3. ANALYZE (Audit)
        # Purpose: Verify structural analysis and box-plot classification [cite: 630]
        print_step("Running static analysis audit...")
        run_command([
            sys.executable, "analyze_graph.py",
            "--layer", "application",
            "--uri", NEO4J_URI,
            "--user", NEO4J_USER,
            "--password", NEO4J_PASS
        ], log_file)

        # 4. SIMULATE
        # Purpose: Generate ground truth failure impact (Formula 7) [cite: 636]
        print_step("Running exhaustive failure simulation...")
        run_command([
            sys.executable, "simulate_graph.py",
            "--exhaustive",
            "--layer", "application",
            "--output", sim_file,
            "--uri", NEO4J_URI,
            "--user", NEO4J_USER,
            "--password", NEO4J_PASS
        ], log_file)

        # 5. VALIDATE
        # Purpose: Calculate Spearman Rho, F1, etc. [cite: 653]
        print_step("Validating predictions against simulation...")
        run_command([
            sys.executable, "validate_graph.py",
            "--layer", "application",
            "--output", val_file,
            "--uri", NEO4J_URI,
            "--user", NEO4J_USER,
            "--password", NEO4J_PASS
        ], log_file)

        # Load Results
        with open(val_file) as f:
            results = json.load(f)

        duration = time.time() - start_time
        
        # Flatten metrics for CSV
        metrics = results["overall"]["metrics"]
        return {
            "run_id": run_id,
            "scale": scale,
            "seed": seed,
            "spearman": metrics["rho"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "top5_overlap": metrics["top5_overlap"],
            "top10_overlap": metrics["top10_overlap"],
            "duration_sec": round(duration, 2),
            "status": "SUCCESS" if results["overall"]["passed"] else "FAIL"
        }

    def run_suite(self, scales=DEFAULT_SCALES, seeds=DEFAULT_SEEDS):
        print(f"{Colors.HEADER}=== Starting CLI Benchmark Suite ==={Colors.END}")
        print(f"Scales: {scales}")
        print(f"Seeds: {seeds}")
        print(f"Output: {self.output_dir}\n")

        all_results = []
        
        try:
            for scale in scales:
                for seed in seeds:
                    run_id = f"{scale}_{seed}"
                    print(f"\n{Colors.YELLOW}>>> Running Configuration: {scale} (Seed: {seed}){Colors.END}")
                    
                    try:
                        res = self.run_iteration(scale, seed, run_id)
                        all_results.append(res)
                        
                        # Print Quick Stat
                        print(f"    {Colors.GREEN}✓ Done{Colors.END} (Rho: {res['spearman']:.3f}, F1: {res['f1']:.3f})")
                        
                    except Exception as e:
                        print(f"    {Colors.RED}✗ Failed{Colors.END}: {e}")
                        all_results.append({
                            "run_id": run_id, "scale": scale, "seed": seed,
                            "status": "ERROR", "error": str(e)
                        })

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")

        self.save_report(all_results)

    def save_report(self, results):
        if not results:
            return

        # CSV Report
        csv_path = self.output_dir / "benchmark_summary.csv"
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        # Markdown Summary
        self.print_summary(results)
        print(f"\nResults saved to: {csv_path}")

    def print_summary(self, results):
        print(f"\n{Colors.HEADER}=== Benchmark Summary ==={Colors.END}")
        
        # Group by scale
        by_scale = {}
        for r in results:
            #if r.get("status") != "SUCCESS": continue
            s = r["scale"]
            if s not in by_scale: by_scale[s] = {"rho": [], "f1": []}
            by_scale[s]["rho"].append(r["spearman"])
            by_scale[s]["f1"].append(r["f1"])

        print(f"{'Scale':<10} | {'Runs':<5} | {'Avg Spearman':<12} | {'Avg F1':<10}")
        print("-" * 45)
        
        for scale, data in by_scale.items():
            n = len(data["rho"])
            avg_rho = statistics.mean(data["rho"])
            avg_f1 = statistics.mean(data["f1"])
            print(f"{scale:<10} | {n:<5} | {avg_rho:<12.3f} | {avg_f1:<10.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full CLI benchmark suite")
    parser.add_argument("--scales", nargs="+", default=DEFAULT_SCALES, choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--runs", type=int, default=3, help="Number of seeds/runs per scale")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.runs))
    
    runner = CLIBenchmarkRunner()
    runner.run_suite(scales=args.scales, seeds=seeds)