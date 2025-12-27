#!/usr/bin/env python3
"""
Run All Examples
=================

Runs all example scripts to verify they work correctly.

Usage:
    python examples/run_all.py           # Run all examples
    python examples/run_all.py --quick   # Quick mode (skip slow examples)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = Path(__file__).parent


def run_example(script: str, description: str, timeout: int = 120) -> bool:
    """Run an example script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script:  {script}")
    print("=" * 60)
    
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                print(line)
            print(f"\n✓ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"✗ Failed (exit code {result.returncode})")
            print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all examples")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode (skip slow examples)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOFTWARE-AS-A-GRAPH: Running All Examples")
    print("=" * 60)
    
    examples = [
        ("quick_start.py", "Quick Start (Python API)", False),
        ("example_generate.py", "Graph Generation CLI", False),
        ("example_simulate.py", "Simulation CLI", False),
        ("example_validate.py", "Validation CLI", True),  # Slow
        ("example_visualize.py", "Visualization CLI", False),
        ("example_pipeline.py", "Pipeline CLI (run.py)", True),  # Slow
    ]
    
    results = []
    
    for script, description, is_slow in examples:
        if args.quick and is_slow:
            print(f"\n○ Skipping (slow): {description}")
            continue
        
        success = run_example(script, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)
    
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {desc}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✓ All examples completed successfully!")
        return 0
    else:
        print(f"\n✗ {failed} example(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
