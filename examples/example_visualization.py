#!/usr/bin/env python3
"""
Example script wrapper for Step 6: Visualization.
"""
import sys
from pathlib import Path

# Add examples and project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_visualization import run_visualization

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Worked Example Visualization Dashboard")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--open", "-b", action="store_true", help="Open dashboard in browser after generation")
    
    args = parser.parse_args()
    run_visualization(args)
