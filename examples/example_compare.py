"""
examples/example_compare.py — Architectural Design Comparison

This script demonstrates the framework's capability to compare two different 
architectural designs side-by-side using RMAV quality dimensions.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "backend"))

from backend.src.infrastructure.memory_repo import MemoryRepository
from backend.src.analysis.service import AnalysisService
from backend.src.prediction.service import PredictionService

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
    prediction_service = PredictionService(repo)
    
    # We analyze the 'system' layer for a holistic view
    struct_res = analysis_service.analyze_layer("system")
    quality_res = prediction_service.predict_quality(struct_res.structural)
    
    # Calculate system-wide averages
    comps = quality_res.components
    if not comps:
        return {"Reliability": 0, "Maintainability": 0, "Availability": 0, "Vulnerability": 0}
        
    n = len(comps)
    return {
        "Reliability": sum(c.scores.reliability for c in comps) / n,
        "Maintainability": sum(c.scores.maintainability for c in comps) / n,
        "Availability": sum(c.scores.availability for c in comps) / n,
        "Vulnerability": sum(c.scores.vulnerability for c in comps) / n,
        "Overall": sum(c.scores.overall for c in comps) / n
    }

def print_comparison(scores_a, scores_b, name_a, name_b):
    print(f"\n{'Dimension':<20} | {name_a:<12} | {name_b:<12} | {'Delta':<8} | {'Better'}")
    print("-" * 75)
    
    dimensions = ["Reliability", "Maintainability", "Availability", "Vulnerability", "Overall"]
    
    for dim in dimensions:
        sa = scores_a[dim]
        sb = scores_b[dim]
        delta = sb - sa
        
        # In this framework, higher scores generally mean higher criticality/risk 
        # (inverse of "quality" in some contexts), but RMAV here represents 
        # estimated resilience/safety where HIGHER is BETTER for safety.
        # Wait, let's check the score semantics.
        # In SDD.md: "Higher RMAV scores indicate higher predicted resilience."
        
        better = name_b if sb > sa else name_a
        if abs(delta) < 0.001:
            better = "Tie"
            
        indicator = "✓" if sb > sa else " "
        note = ""
        if dim == "Availability" and sb > sa + 0.05:
            note = "(SPOF resolved)"
            
        print(f"{dim:<20} | {sa:12.3f} | {sb:12.3f} | {delta:+8.3f} | {better:<12} {note}")

def main():
    design_a_path = "examples/topologies/design_a_shared_broker.json"
    design_b_path = "examples/topologies/design_b_dedicated_brokers.json"

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("⚖️  Architectural Design Comparison")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Design A: {design_a_path}")
    print(f"  Design B: {design_b_path}")

    # Run full analysis for both
    print("\n  [1] Analyzing Design A...")
    scores_a = analyze_design(design_a_path)
    
    print("  [2] Analyzing Design B...")
    scores_b = analyze_design(design_b_path)

    # 3. Print Comparison table
    print("\n  [3] Decision Support Comparison:")
    print_comparison(scores_a, scores_b, "Design A", "Design B")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    main()
