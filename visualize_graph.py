#!/usr/bin/env python3
"""
Visualize Graph CLI

Generates an HTML dashboard visualizing Graph Stats, Quality Analysis, and Simulation Impact.
Retrieves data directly from Neo4j.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.visualizer import GraphVisualizer

# Colors
GREEN = "\033[92m"; BLUE = "\033[94m"; RED = "\033[91m"; RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Graph Visualization Dashboard")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j User")
    parser.add_argument("--password", default="password", help="Neo4j Password")
    parser.add_argument("--output", default="dashboard.html", help="Output HTML file")
    parser.add_argument("--open", action="store_true", help="Open in browser automatically")
    
    args = parser.parse_args()
    
    try:
        print(f"{BLUE}Connecting to Neo4j and running analysis pipeline...{RESET}")
        
        with GraphVisualizer(uri=args.uri, user=args.user, password=args.password) as viz:
            output_path = viz.generate_dashboard(args.output)
            
            print(f"{GREEN}Dashboard generated successfully: {output_path}{RESET}")
            
            if args.open:
                import os
                print("Opening in browser...")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
    except Exception as e:
        print(f"{RED}Error generating dashboard: {e}{RESET}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())