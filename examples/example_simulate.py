#!/usr/bin/env python3
"""
Simulation Examples
====================

Demonstrates simulate_graph.py CLI usage.

Examples:
    # Run failure simulation
    python simulate_graph.py --input graph.json --output output/

    # Event-driven simulation
    python simulate_graph.py --input graph.json --event --duration 5000

    # With cascade propagation
    python simulate_graph.py --input graph.json --cascade --threshold 0.5
"""

import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def ensure_graph_exists():
    """Generate a test graph if none exists"""
    graph_file = OUTPUT_DIR / "test_graph.json"
    
    if not graph_file.exists():
        print("Generating test graph...")
        subprocess.run([
            sys.executable, "generate_graph.py",
            "--scale", "small",
            "--scenario", "iot",
            "--output", str(graph_file)
        ], cwd=PROJECT_ROOT, capture_output=True)
    
    return graph_file


def run_cmd(cmd: list, description: str):
    """Run command and print output"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Show summary only
        lines = result.stdout.strip().split('\n')
        for line in lines[-15:]:  # Last 15 lines
            print(line)
        print("✓ Success")
    else:
        print(f"✗ Failed: {result.stderr[:200]}")
    
    return result.returncode == 0


def main():
    print("Simulation Examples")
    print("=" * 60)
    
    graph_file = ensure_graph_exists()
    
    examples = [
        # Basic failure simulation
        (
            [sys.executable, "simulate_graph.py",
             "--input", str(graph_file),
             "--output", str(OUTPUT_DIR),
             "--export-json"],
            "Basic failure simulation"
        ),
        
        # With cascade propagation
        (
            [sys.executable, "simulate_graph.py",
             "--input", str(graph_file),
             "--cascade",
             "--threshold", "0.5",
             "--output", str(OUTPUT_DIR)],
            "Failure simulation with cascade"
        ),
        
        # Event-driven simulation
        (
            [sys.executable, "simulate_graph.py",
             "--input", str(graph_file),
             "--event",
             "--duration", "2000",
             "--rate", "50",
             "--output", str(OUTPUT_DIR)],
            "Event-driven simulation (2 seconds)"
        ),
    ]
    
    success_count = 0
    for cmd, desc in examples:
        if run_cmd(cmd, desc):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(examples)} examples succeeded")
    print("=" * 60)


if __name__ == "__main__":
    main()
