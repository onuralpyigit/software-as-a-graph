#!/usr/bin/env python3
"""
Analysis CLI - Software-as-a-Graph
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.analyzer import GraphAnalyzer

# ANSI Colors
RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; BLUE = "\033[94m"; RESET = "\033[0m"
BOLD = "\033[1m"

def print_results(results):
    print(f"\n{BOLD}=== Context: {results['context']} ==={RESET}")
    print(f"Graph Summary: {results['stats']['nodes']} Nodes, {results['stats']['edges']} Edges")
    
    qual = results["results"]
    
    print(f"\n{BOLD}Top Critical Components (Box-Plot Outliers):{RESET}")
    # Filter for High/Critical items
    critical = [c for c in qual.components if c.level.value in ["critical", "high"]]
    critical.sort(key=lambda x: x.scores.overall, reverse=True)
    
    if not critical:
        print(f"  {GREEN}No critical outliers detected.{RESET}")
    else:
        print(f"  {'ID':<20} {'Type':<12} {'Score':<8} {'Level':<10}")
        print(f"  {'-'*50}")
        for c in critical[:10]:
            color = RED if c.level.value == "critical" else YELLOW
            print(f"  {c.id:<20} {c.type:<12} {c.scores.overall:.2f}     {color}{c.level.value.upper()}{RESET}")

    print(f"\n{BOLD}Detected Problems & Risks:{RESET}")
    if not results["problems"]:
        print(f"  {GREEN}No structural problems detected.{RESET}")
    else:
        for p in results["problems"]:
            color = RED if p.severity == "CRITICAL" else YELLOW
            print(f"  [{color}{p.severity:<8}{RESET}] {BOLD}{p.entity_id}{RESET} ({p.category})")
            print(f"    Issue: {p.description}")
            print(f"    Fix:   {p.recommendation}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Graph Analysis Tool")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--type", help="Analyze specific component type (Application, Topic, Node, Broker)")
    group.add_argument("--layer", help="Analyze specific layer (application, infrastructure)")
    
    args = parser.parse_args()
    
    print(f"{BLUE}Connecting to {args.uri}...{RESET}")
    
    try:
        with GraphAnalyzer(uri=args.uri, user=args.user, password=args.password) as analyzer:
            if args.type:
                results = analyzer.analyze_by_type(args.type)
            elif args.layer:
                results = analyzer.analyze_layer(args.layer)
            else:
                results = analyzer.analyze()
                
            print_results(results)
            
    except Exception as e:
        print(f"{RED}Analysis Failed: {e}{RESET}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())