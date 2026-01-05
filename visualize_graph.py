#!/usr/bin/env python3
"""
Visualize Graph CLI

Generates a multi-layer analysis dashboard for the Software-as-a-Graph system.
"""

import argparse
import sys
import webbrowser
import os
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.visualizer import GraphVisualizer

# Colors
GREEN = "\033[92m"; BLUE = "\033[94m"; RED = "\033[91m"; RESET = "\033[0m"
BOLD = "\033[1m"

def main():
    parser = argparse.ArgumentParser(description="Generate Analysis Dashboard")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j User")
    parser.add_argument("--password", default="password", help="Neo4j Password")
    parser.add_argument("--output", default="dashboard.html", help="Output file path")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    try:
        print(f"{BLUE}{BOLD}=== Software-as-a-Graph Visualization Module ==={RESET}")
        print(f"Connecting to Neo4j at {args.uri}...")
        
        with GraphVisualizer(uri=args.uri, user=args.user, password=args.password) as viz:
            print(f"Running Multi-Layer Analysis & Validation Pipeline...")
            output_path = viz.generate_dashboard(args.output)
            
            print(f"\n{GREEN}Success! Dashboard generated at: {os.path.abspath(output_path)}{RESET}")
            
            if not args.no_browser:
                print("Opening dashboard in default browser...")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    main()