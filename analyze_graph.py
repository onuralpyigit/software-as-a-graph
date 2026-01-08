#!/usr/bin/env python3
"""
Software-as-a-Graph Analysis CLI

Analyzes the multi-layer graph model to identify critical components and risks.
Supports JSON export and batch processing of layers.
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.analyzer import GraphAnalyzer
from src.analysis.quality_analyzer import CriticalityLevel

# ANSI Colors
RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; BLUE = "\033[94m"; CYAN = "\033[96m"; RESET = "\033[0m"
BOLD = "\033[1m"

def print_section_header(title: str):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD} {title} {RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

def display_results(results: dict):
    stats = results['stats']
    print(f"Topology: {CYAN}{stats.get('nodes',0)}{RESET} Nodes, {CYAN}{stats.get('edges',0)}{RESET} Edges")
    
    # 1. Critical Components
    qual_res = results["results"]
    critical = [c for c in qual_res.components if c.levels.overall >= CriticalityLevel.HIGH]
    critical.sort(key=lambda x: x.scores.overall, reverse=True)
    
    print(f"\n{BOLD}>> Top Critical Components (Box-Plot Outliers){RESET}")
    if not critical:
        print(f"  {GREEN}No outliers detected.{RESET}")
    else:
        print(f"  {'-'*75}")
        print(f"  {'ID':<25} {'Overall':<8} {'Rel.':<8} {'Maint.':<8} {'Avail.':<8}")
        print(f"  {'-'*75}")
        
        for c in critical[:15]:
            color = RED if c.levels.overall == CriticalityLevel.CRITICAL else YELLOW
            
            # Helper to mark high levels
            def mark(lvl):
                return f"{RED}*{RESET}" if lvl == CriticalityLevel.CRITICAL else (f"{YELLOW}*{RESET}" if lvl == CriticalityLevel.HIGH else "-")

            print(f"  {c.id:<25} {color}{c.scores.overall:.2f}{RESET}     "
                  f"{mark(c.levels.reliability):<8} {mark(c.levels.maintainability):<8} {mark(c.levels.availability):<8}")

    # 2. Problems
    problems = results["problems"] # List of dicts
    print(f"\n{BOLD}>> Detected Problems & Risks{RESET}")
    if not problems:
        print(f"  {GREEN}No architectural problems detected.{RESET}")
    else:
        for p in problems:
            # Handle both object and dict (if serialized)
            severity = p.get('severity') if isinstance(p, dict) else p.severity
            eid = p.get('entity_id') if isinstance(p, dict) else p.entity_id
            name = p.get('name') if isinstance(p, dict) else p.name
            desc = p.get('description') if isinstance(p, dict) else p.description
            
            badge = f"[{RED}{severity}{RESET}]" if severity == "CRITICAL" else f"[{YELLOW}{severity}{RESET}]"
            print(f"  {badge} {BOLD}{eid}{RESET}: {name}")
            print(f"     {desc}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Graph Analysis Tool")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    # Scope arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Analyze ALL layers sequentially")
    group.add_argument("--complete", action="store_true", help="Analyze complete system (default)")
    group.add_argument("--layer", choices=["application", "infrastructure"], help="Analyze specific layer")
    group.add_argument("--type", help="Analyze specific component type")
    
    parser.add_argument("--output", "-o", help="Path to export JSON results (e.g., results/analysis.json)")
    
    args = parser.parse_args()
    
    print(f"{BLUE}Connecting to Neo4j at {args.uri}...{RESET}")
    
    try:
        with GraphAnalyzer(uri=args.uri, user=args.user, password=args.password) as analyzer:
            tasks = []
            
            # Determine tasks
            if args.all:
                tasks = [
                    ("application", lambda: analyzer.analyze_layer("application")),
                    ("infrastructure", lambda: analyzer.analyze_layer("infrastructure")),
                    ("complete", analyzer.analyze)
                ]
            elif args.layer:
                tasks = [(args.layer, lambda: analyzer.analyze_layer(args.layer))]
            elif args.type:
                tasks = [(args.type, lambda: analyzer.analyze_by_type(args.type))]
            else:
                tasks = [("complete", analyzer.analyze)]
            
            # Execute tasks
            all_results = {}
            for name, func in tasks:
                print_section_header(f"Context: {name.upper()}")
                results = func()
                display_results(results)
                all_results[name] = results
            
            # Export if requested
            if args.output:
                path = Path(args.output)
                if len(all_results) == 1:
                    # Export single result object
                    key = list(all_results.keys())[0]
                    analyzer.export_results(all_results[key], str(path))
                else:
                    # Export combined results if --all was used
                    # Save separate files or a combined one? Combined is cleaner.
                    combined = {k: v for k, v in all_results.items()} 
                    # Note: We need to serialize the wrapper manually here since export_results expects one result
                    with open(path, 'w') as f:
                        # Create a serializable version of the map
                        serializable_map = {}
                        for k, res in combined.items():
                             # Reuse the logic inside export_results by temporary mocking or just relying on internal helper
                             # For simplicity, we just dump the 'problems' and 'summary' parts which are dicts
                             serializable_map[k] = {
                                 "summary": res["summary"],
                                 "stats": res["stats"],
                                 "problems": res["problems"]
                             }
                        json.dump(serializable_map, f, indent=2)
                    print(f"\n{GREEN}Combined summary exported to {path}{RESET}")

    except Exception as e:
        print(f"\n{RED}Analysis Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())