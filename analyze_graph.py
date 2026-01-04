#!/usr/bin/env python3
"""
Analysis CLI
"""
import argparse
from src.analysis.analyzer import GraphAnalyzer

# Colors
RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; BLUE = "\033[94m"; RESET = "\033[0m"

def print_problems(problems):
    if not problems:
        print(f"\n{GREEN}No specific problems detected.{RESET}")
        return

    print(f"\n{BOLD}Detected Problems:{RESET}")
    for p in problems:
        color = RED if p.severity.value == "CRITICAL" else YELLOW
        print(f"[{color}{p.severity.value:<8}{RESET}] {p.category}: {p.component_id}")
        print(f"  Issue: {p.description}")
        print(f"  Fix:   {p.recommendation}")
        print(f"  Signs: {', '.join(p.symptoms)}")
        print("-" * 50)

BOLD = "\033[1m"

def main():
    parser = argparse.ArgumentParser(description="Graph Analysis CLI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    args = parser.parse_args()
    
    with GraphAnalyzer(uri=args.uri, user=args.user, password=args.password) as analyzer:
        results = analyzer.analyze_full_pipeline()
        
        qual = results["quality"]
        
        print(f"\n{BOLD}Top Critical Components (Overall Quality Score):{RESET}")
        for c in qual.components[:10]:
            print(f"{c.id:<20} | Q-Score: {c.scores.overall:.4f} | Level: {c.level.value}")

        print_problems(results["problems"])

if __name__ == "__main__":
    main()