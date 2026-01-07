#!/usr/bin/env python3
"""
Analysis CLI - Software-as-a-Graph
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.analyzer import GraphAnalyzer
from src.analysis.quality_analyzer import CriticalityLevel

# ANSI Colors
RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; BLUE = "\033[94m"; RESET = "\033[0m"
BOLD = "\033[1m"

def print_results(results):
    print(f"\n{BOLD}=== Analysis Context: {results['context']} ==={RESET}")
    print(f"Graph Topology: {results['stats']['nodes']} Nodes, {results['stats']['edges']} Edges")
    
    qual = results["results"]
    
    print(f"\n{BOLD}Critical Components (Box-Plot Outliers):{RESET}")
    # Filter for High/Critical items in ANY dimension
    critical = [c for c in qual.components if c.levels.overall >= CriticalityLevel.HIGH]
    critical.sort(key=lambda x: x.scores.overall, reverse=True)
    
    if not critical:
        print(f"  {GREEN}No critical outliers detected in this scope.{RESET}")
    else:
        # Header
        print(f"  {'ID':<20} {'Overall':<8} {'Reliab.':<8} {'Maint.':<8} {'Avail.':<8}")
        print(f"  {'-'*60}")
        
        for c in critical[:15]:
            # Color based on overall level
            color = RED if c.levels.overall == CriticalityLevel.CRITICAL else YELLOW
            
            # Format sub-scores with indicators if they are high
            def fmt(lvl):
                return "*" if lvl >= CriticalityLevel.HIGH else "-"

            r_flag = fmt(c.levels.reliability)
            m_flag = fmt(c.levels.maintainability)
            a_flag = fmt(c.levels.availability)
            
            print(f"  {c.id:<20} {color}{c.scores.overall:.2f}{RESET}     {r_flag:<8} {m_flag:<8} {a_flag:<8}")
        print(f"  {BOLD}(* = Critical/High in specific dimension){RESET}")

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
    
    # Analysis Scope Selection
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument("--layer", choices=["application", "infrastructure"], 
                             help="Analyze a specific architectural layer.")
    scope_group.add_argument("--complete", action="store_true", 
                             help="Analyze the complete system (default).")
    scope_group.add_argument("--type", help="Analyze a specific component type (e.g., Application, Node).")
    
    args = parser.parse_args()
    
    print(f"{BLUE}Connecting to {args.uri}...{RESET}")
    
    try:
        with GraphAnalyzer(uri=args.uri, user=args.user, password=args.password) as analyzer:
            if args.layer:
                results = analyzer.analyze_layer(args.layer)
            elif args.type:
                results = analyzer.analyze_by_type(args.type)
            else:
                # Default to complete system
                results = analyzer.analyze()
                
            print_results(results)
            
    except Exception as e:
        print(f"{RED}Analysis Failed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())