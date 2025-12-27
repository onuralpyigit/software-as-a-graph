#!/usr/bin/env python3
"""
Quick Start Example
====================

Demonstrates the complete pipeline using Python API.
No Neo4j required - uses pure Python graph analysis.

Usage:
    python examples/quick_start.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import generate_graph
from src.simulation import SimulationGraph, FailureSimulator, EventSimulator
from src.validation import ValidationPipeline
from src.visualization import GraphRenderer, DashboardGenerator


def main():
    print("=" * 60)
    print("Software-as-a-Graph: Quick Start Demo")
    print("=" * 60)
    
    # Step 1: Generate a graph
    print("\n[1/5] Generating IoT system graph...")
    graph_data = generate_graph(scale="small", scenario="iot", seed=42)
    graph = SimulationGraph.from_dict(graph_data)
    print(f"  ✓ {len(graph.components)} components, {len(graph.connections)} connections")
    
    # Step 2: Analyze criticality
    print("\n[2/5] Analyzing criticality...")
    pipeline = ValidationPipeline(seed=42)
    result = pipeline.run(graph, analysis_method="composite")
    
    # Show top 5 critical components
    top_5 = sorted(result.predicted_scores.items(), key=lambda x: -x[1])[:5]
    print("  Top 5 critical components:")
    for comp_id, score in top_5:
        print(f"    {comp_id}: {score:.4f}")
    
    # Step 3: Run failure simulation
    print("\n[3/5] Running failure simulation...")
    simulator = FailureSimulator(seed=42)
    batch = simulator.simulate_all_failures(graph)
    print(f"  ✓ Tested {len(batch.results)} components")
    print(f"  ✓ Found {len(batch.critical_components)} critical failures")
    
    # Step 4: Validate predictions
    print("\n[4/5] Validating predictions...")
    validation = result.validation
    print(f"  Spearman ρ: {validation.correlation.spearman:.4f}")
    print(f"  F1-Score:   {validation.classification.f1:.4f}")
    print(f"  Status:     {validation.status.value}")
    
    # Step 5: Generate visualization
    print("\n[5/5] Generating dashboard...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    criticality = {
        k: {"score": v, "level": "critical" if v > 0.3 else "low"}
        for k, v in result.predicted_scores.items()
    }
    
    generator = DashboardGenerator()
    html = generator.generate(
        graph,
        criticality=criticality,
        validation=validation.to_dict(),
    )
    
    dashboard_path = output_dir / "quick_start_dashboard.html"
    dashboard_path.write_text(html)
    print(f"  ✓ Saved to {dashboard_path}")
    
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print(f"Open {dashboard_path} in a browser to view results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
