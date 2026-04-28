"""
examples/example_compare.py — Architectural Design Comparison

This script demonstrates the framework's capability to compare two different 
architectural designs side-by-side using RMAV quality dimensions.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent

from saag.infrastructure.memory_repo import MemoryRepository
from saag.analysis.service import AnalysisService

def load_json_with_comments(path: str):
    import re
    with open(path, 'r') as f:
        content = f.read()
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    return json.loads(content)

def analyze_design(topology_path: str) -> Dict[str, float]:
    """Load, analyze, and aggregate scores for a design."""
    topology = load_json_with_comments(topology_path)
    
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    
    analysis_service = AnalysisService(repo)
    
    # We analyze the 'system' layer for a holistic view
    # This automatically runs structural analysis + quality prediction
    result = analysis_service.analyze_layer("system")
    quality_res = result.quality
    
    # Calculate system-wide averages
    comps = quality_res.components
    if not comps:
        return {"Reliability": 0, "Maintainability": 0, "Availability": 0, "Vulnerability": 0, "Overall": 0}
        
    n = len(comps)
    return {
        "Reliability": sum(c.scores.reliability for c in comps) / n,
        "Maintainability": sum(c.scores.maintainability for c in comps) / n,
        "Availability": sum(c.scores.availability for c in comps) / n,
        "Vulnerability": sum(c.scores.vulnerability for c in comps) / n,
        "Overall": sum(c.scores.overall for c in comps) / n
    }

def print_comparison(scores_a, scores_b, name_a, name_b):
    print(f"\n{'Dimension':<20} | {name_a:<12} | {name_b:<12} | {'Delta':<8} | {'Safer Design'}")
    print("-" * 75)
    
    dimensions = ["Reliability", "Maintainability", "Availability", "Vulnerability", "Overall"]
    
    for dim in dimensions:
        sa = scores_a[dim]
        sb = scores_b[dim]
        delta = sb - sa
        
        # SCORE SEMANTICS:
        # Higher RMAV scores indicate higher predicted RISK/CRITICALITY.
        # Therefore, if Design B has a LOWER score than Design A, 
        # Design B is considered SAFER for that dimension.
        
        safer = name_b if sb < sa else name_a
        if abs(delta) < 0.001:
            safer = "Tie"
            
        indicator = "✓" if sb < sa else " "
        note = ""
        if dim == "Availability" and sb < sa - 0.05:
            note = "(Redundancy improved)"
            
        print(f"{dim:<20} | {sa:12.3f} | {sb:12.3f} | {delta:+8.3f} | {safer:<12} {note}")

def main():
    design_a_path = "examples/topologies/design_a_shared_broker.json"
    design_b_path = "examples/topologies/design_b_dedicated_brokers.json"

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("⚖️  Architectural Design Comparison")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  This script compares two architectural variants of the same system.")
    print(f"  Design A: {design_a_path} (Shared Infrastructure)")
    print(f"  Design B: {design_b_path} (Distributed Redundancy)")

    # Run full analysis for both
    print("\n  [1] Analyzing Design A...")
    scores_a = analyze_design(str(ROOT / design_a_path))
    
    print("  [2] Analyzing Design B...")
    scores_b = analyze_design(str(ROOT / design_b_path))

    # 3. Print Comparison table
    print("\n  [3] Decision Support Comparison (Lower Score = Safer):")
    print_comparison(scores_a, scores_b, "Design A", "Design B")
    print("\n  Insight:")
    print("    By isolating components onto dedicated brokers (Design B), we")
    print("    reduce the topological centrality of the messaging hub,")
    print("    lowering the ripple-effect risk across the system.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    main()
