#!/usr/bin/env python3
"""
CLI Reference
==============

Quick reference for all CLI tools with common usage patterns.

Usage:
    python examples/cli_reference.py              # Show all
    python examples/cli_reference.py --tool run   # Show run.py help
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

CLI_TOOLS = {
    "run": {
        "script": "run.py",
        "description": "End-to-end pipeline",
        "examples": [
            "python run.py --quick",
            "python run.py --scenario iot --scale medium",
            "python run.py --input graph.json --skip-generate",
            "python run.py --scenario financial --scale large --cascade",
        ]
    },
    "generate": {
        "script": "generate_graph.py",
        "description": "Graph generation",
        "examples": [
            "python generate_graph.py --scale small --scenario iot --output graph.json",
            "python generate_graph.py --scale medium --scenario financial --seed 42",
            "python generate_graph.py --scale large --antipatterns god_topic spof --output large.json",
        ]
    },
    "simulate": {
        "script": "simulate_graph.py",
        "description": "Failure and event simulation",
        "examples": [
            "python simulate_graph.py --input graph.json --output results/",
            "python simulate_graph.py --input graph.json --cascade --threshold 0.5",
            "python simulate_graph.py --input graph.json --event --duration 5000",
        ]
    },
    "validate": {
        "script": "validate_graph.py",
        "description": "Statistical validation",
        "examples": [
            "python validate_graph.py --input graph.json",
            "python validate_graph.py --input graph.json --spearman 0.75 --f1 0.85",
            "python validate_graph.py --input graph.json --output results.json",
        ]
    },
    "visualize": {
        "script": "visualize_graph.py",
        "description": "Visualization generation",
        "examples": [
            "python visualize_graph.py --input graph.json --output graph_viz.html",
            "python visualize_graph.py --input graph.json --dashboard --run-analysis",
            "python visualize_graph.py --input graph.json --multi-layer",
        ]
    },
    "import": {
        "script": "import_graph.py",
        "description": "Neo4j import (requires Neo4j)",
        "examples": [
            "python import_graph.py --uri bolt://localhost:7687 --user neo4j --password pass --input graph.json",
            "python import_graph.py --uri bolt://localhost:7687 --user neo4j --password pass --input graph.json --clear",
        ]
    },
}


def show_tool_help(tool_name: str):
    """Show help for a specific tool"""
    if tool_name not in CLI_TOOLS:
        print(f"Unknown tool: {tool_name}")
        print(f"Available: {', '.join(CLI_TOOLS.keys())}")
        return
    
    tool = CLI_TOOLS[tool_name]
    script = tool["script"]
    
    print(f"\n{'='*60}")
    print(f"{tool_name.upper()}: {tool['description']}")
    print(f"Script: {script}")
    print("=" * 60)
    
    # Show help
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    # Show examples
    print("\nCommon Examples:")
    print("-" * 40)
    for example in tool["examples"]:
        print(f"  {example}")


def show_all_tools():
    """Show summary of all tools"""
    print("\n" + "=" * 60)
    print("SOFTWARE-AS-A-GRAPH CLI REFERENCE")
    print("=" * 60)
    
    print("\nAvailable CLI Tools:\n")
    
    for name, tool in CLI_TOOLS.items():
        print(f"  {name:12} - {tool['description']}")
        print(f"               Script: {tool['script']}")
        print()
    
    print("\nQuick Examples:\n")
    print("  # Full pipeline (recommended)")
    print("  python run.py --quick")
    print()
    print("  # Step by step")
    print("  python generate_graph.py --scale small --scenario iot --output graph.json")
    print("  python validate_graph.py --input graph.json")
    print("  python visualize_graph.py --input graph.json --dashboard --output-dir output/")
    print()
    print("Use --tool <name> to see detailed help for a specific tool.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CLI Reference")
    parser.add_argument("--tool", "-t", choices=list(CLI_TOOLS.keys()),
                        help="Show help for specific tool")
    args = parser.parse_args()
    
    if args.tool:
        show_tool_help(args.tool)
    else:
        show_all_tools()


if __name__ == "__main__":
    main()
