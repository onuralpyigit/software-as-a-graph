#!/usr/bin/env python3
"""
Validation Examples
====================

Demonstrates validate_graph.py CLI usage.

Examples:
    # Basic validation
    python validate_graph.py --input graph.json

    # With custom targets
    python validate_graph.py --input graph.json --spearman 0.75 --f1 0.85

    # Compare analysis methods
    python validate_graph.py --input graph.json --compare-methods
"""

import subprocess
import sys
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
        print(result.stdout)
        print("✓ Success")
    else:
        print(f"✗ Failed: {result.stderr[:200]}")
    
    return result.returncode == 0


def main():
    print("Validation Examples")
    print("=" * 60)
    
    graph_file = ensure_graph_exists()
    
    examples = [
        # Basic validation
        (
            [sys.executable, "validate_graph.py",
             "--input", str(graph_file),
             "--no-color"],
            "Basic validation with default targets"
        ),
        
        # Custom targets
        (
            [sys.executable, "validate_graph.py",
             "--input", str(graph_file),
             "--spearman", "0.60",
             "--f1", "0.80",
             "--no-color"],
            "Validation with relaxed targets"
        ),
        
        # Export results
        (
            [sys.executable, "validate_graph.py",
             "--input", str(graph_file),
             "--output", str(OUTPUT_DIR / "validation_results.json"),
             "--no-color"],
            "Validation with JSON export"
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
