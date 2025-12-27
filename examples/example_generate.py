#!/usr/bin/env python3
"""
Graph Generation Examples
==========================

Demonstrates generate_graph.py CLI usage.

Examples:
    # Generate small IoT system
    python generate_graph.py --scale small --scenario iot --output output/iot.json

    # Generate large financial system
    python generate_graph.py --scale large --scenario financial --output output/fin.json

    # Generate with anti-patterns
    python generate_graph.py --scale medium --scenario iot --antipatterns --output output/ap.json
"""

import subprocess
import sys
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


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
        print(f"✗ Failed: {result.stderr}")
    
    return result.returncode == 0


def main():
    print("Graph Generation Examples")
    print("=" * 60)
    
    examples = [
        # Basic usage
        (
            [sys.executable, "generate_graph.py",
             "--scale", "small",
             "--scenario", "iot",
             "--output", str(OUTPUT_DIR / "example_iot_small.json")],
            "Small IoT system"
        ),
        
        # Different scenario
        (
            [sys.executable, "generate_graph.py",
             "--scale", "medium",
             "--scenario", "financial",
             "--seed", "123",
             "--output", str(OUTPUT_DIR / "example_financial.json")],
            "Medium financial system with fixed seed"
        ),
        
        # With anti-patterns
        (
            [sys.executable, "generate_graph.py",
             "--scale", "small",
             "--scenario", "healthcare",
             "--antipatterns", "god_topic", "spof",
             "--output", str(OUTPUT_DIR / "example_antipatterns.json")],
            "Healthcare system with injected anti-patterns"
        ),
    ]
    
    success_count = 0
    for cmd, desc in examples:
        if run_cmd(cmd, desc):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(examples)} examples succeeded")
    print(f"Output files in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
