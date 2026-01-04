#!/usr/bin/env python3
"""
Analysis CLI
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.analyzer import GraphAnalyzer

# Colors
RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; BLUE = "\033[94m"; RESET = "\033[0m"
BOLD = "\033[1m"

def print_results(results, title="Analysis Results"):
    print(f"\n{BOLD}=== {title} ==={RESET}")
    print(f"Nodes: {results['stats']['nodes']}, Edges: {results['stats']['edges']}")
    
    qual = results["quality"]
    
    print(f"\n{BOLD}Top Critical Components:{RESET}")
    critical = [c for c in qual.components if c.level.value in ["critical", "high"]]
    critical.sort(key=lambda x: x.scores.overall, reverse=True)
    
    if not critical:
        print("  None detected.")
    else:
        for c in critical[:5]:
            print(f"  {c.id:<15} ({c.type:<10}) | Q: {c.scores.overall:.2f} | {RED}{c.level.value.upper()}{RESET}")

    print(f"\n{BOLD}Top Critical Edges:{RESET}")
    crit_edges = [e for e in qual.edges if e.level.value in ["critical", "high"]]
    crit_edges.sort(key=lambda x: x.scores.overall, reverse=True)
    
    if not crit_edges:
        print("  None detected.")
    else:
        for e in crit_edges[:5]:
            print(f"  {e.source}->{e.target:<10} | Q: {e.scores.overall:.2f} | {RED}{e.level.value.upper()}{RESET}")

    print(f"\n{BOLD}Detected Problems:{RESET}")
    if not results["problems"]:
        print(f"  {GREEN}No specific problems detected.{RESET}")
    else:
        for p in results["problems"]:
            color = RED if p.severity == "CRITICAL" else YELLOW
            print(f"  [{color}{p.severity:<8}{RESET}] {p.entity_id} ({p.category}): {p.description}")

def main():
    parser = argparse.ArgumentParser(description="Graph Analysis CLI")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--type", help="Analyze specific component type (Application, Node, etc)")
    group.add_argument("--layer", help="Analyze specific layer (application, infrastructure)")
    
    args = parser.parse_args()
    
    try:
        with GraphAnalyzer(uri=args.uri, user=args.user, password=args.password) as analyzer:
            if args.type:
                results = analyzer.analyze_by_type(args.type)
                title = f"Type: {args.type}"
            elif args.layer:
                results = analyzer.analyze_layer(args.layer)
                title = f"Layer: {args.layer}"
            else:
                results = analyzer.analyze_full_system()
                title = "Full System"
                
            print_results(results, title)
            
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())