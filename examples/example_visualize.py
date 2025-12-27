#!/usr/bin/env python3
"""
Visualization Examples
=======================

Demonstrates visualize_graph.py CLI usage.

Examples:
    # Generate basic visualization
    python visualize_graph.py --input graph.json --output graph_viz.html

    # Dashboard with analysis
    python visualize_graph.py --input graph.json --dashboard --run-analysis

    # Multi-layer view
    python visualize_graph.py --input graph.json --multi-layer
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
        # Show key lines
        for line in result.stdout.strip().split('\n'):
            if '✓' in line or '.html' in line or 'Created' in line or 'Saved' in line:
                print(line)
        print("✓ Success")
    else:
        print(f"✗ Failed: {result.stderr[:200]}")
    
    return result.returncode == 0


def main():
    print("Visualization Examples")
    print("=" * 60)
    
    graph_file = ensure_graph_exists()
    vis_dir = OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    examples = [
        # Basic graph visualization
        (
            [sys.executable, "visualize_graph.py",
             "--input", str(graph_file),
             "--output", str(vis_dir / "basic_graph.html")],
            "Basic system graph"
        ),
        
        # Multi-layer view
        (
            [sys.executable, "visualize_graph.py",
             "--input", str(graph_file),
             "--multi-layer",
             "--output", str(vis_dir / "multi_layer.html")],
            "Multi-layer architecture view"
        ),
        
        # Dashboard with analysis
        (
            [sys.executable, "visualize_graph.py",
             "--input", str(graph_file),
             "--dashboard",
             "--run-analysis",
             "--output", str(vis_dir / "dashboard.html")],
            "Dashboard with criticality analysis"
        ),
    ]
    
    success_count = 0
    for cmd, desc in examples:
        if run_cmd(cmd, desc):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(examples)} examples succeeded")
    print(f"\nVisualization files in: {vis_dir}")
    print("Open HTML files in a browser to view.")
    print("=" * 60)


if __name__ == "__main__":
    main()
