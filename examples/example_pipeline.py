#!/usr/bin/env python3
"""
Pipeline Examples
==================

Demonstrates run.py end-to-end pipeline CLI usage.

Examples:
    # Quick demo
    python run.py --quick

    # Full IoT pipeline
    python run.py --scenario iot --scale medium

    # Custom configuration
    python run.py --scenario financial --scale large --spearman-target 0.75
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def run_cmd(cmd: list, description: str):
    """Run command and print output"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Show summary section
        lines = result.stdout.strip().split('\n')
        in_summary = False
        for line in lines:
            if 'PIPELINE COMPLETE' in line or 'Pipeline Complete' in line:
                in_summary = True
            if in_summary:
                print(line)
        print("✓ Success")
    else:
        print(f"✗ Failed: {result.stderr[:300]}")
    
    return result.returncode == 0


def main():
    print("Pipeline Examples (run.py)")
    print("=" * 60)
    
    examples = [
        # Quick demo
        (
            [sys.executable, "run.py",
             "--quick",
             "--no-color",
             "--output", str(OUTPUT_DIR / "quick_demo")],
            "Quick demo (small scale)"
        ),
        
        # Skip validation for faster run
        (
            [sys.executable, "run.py",
             "--scenario", "financial",
             "--scale", "small",
             "--skip-validate",
             "--no-color",
             "--output", str(OUTPUT_DIR / "financial_demo")],
            "Financial system (skip validation)"
        ),
    ]
    
    success_count = 0
    for cmd, desc in examples:
        if run_cmd(cmd, desc):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(examples)} examples succeeded")
    print(f"\nOutput directories:")
    print(f"  - {OUTPUT_DIR / 'quick_demo'}")
    print(f"  - {OUTPUT_DIR / 'financial_demo'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
