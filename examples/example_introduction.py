"""
example_introduction.py — Why Architectural Topology Predicts Risk

This example explains the core scientific intuition of Software-as-a-Graph
WITHOUT requiring a Neo4j database connection.

It demonstrates how simple graph properties (centrality, articulation points)
can predict which components are most likely to cause system-wide failures.
"""
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "backend"))

from src.infrastructure.memory_repo import MemoryRepository
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.core.layers import AnalysisLayer

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎓  Introduction: The Power of Architectural Topology")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  This script demonstrates how we can predict risk by looking")
    print("  ONLY at the structure of your system—no simulation required yet.")

    # 1. Define a tiny "Chain vs Star" system
    # A0 -> A1 -> A2 (Chain)
    # B1 (Hub) -> {C1, C2, C3} (Star)
    topology = {
        "nodes": [{ "id": "N0", "name": "Host" }],
        "applications": [
            { "id": "A0", "name": "Producer" },
            { "id": "A1", "name": "Middleware" },
            { "id": "A2", "name": "Consumer" },
            { "id": "B1", "name": "Central_Hub" },
            { "id": "C1", "name": "Worker_1" },
            { "id": "C2", "name": "Worker_2" },
            { "id": "C3", "name": "Worker_3" }
        ],
        "relationships": {
            "runs_on": [
                { "from": "A0", "to": "N0" }, { "from": "A1", "to": "N0" }, 
                { "from": "A2", "to": "N0" }, { "from": "B1", "to": "N0" },
                { "from": "C1", "to": "N0" }, { "from": "C2", "to": "N0" }, 
                { "from": "C3", "to": "N0" }
            ],
            "app_to_app": [
                # Chain
                { "from": "A0", "to": "A1" }, { "from": "A1", "to": "A2" },
                # Hub & Spoke
                { "from": "B1", "to": "C1" }, { "from": "B1", "to": "C2" }, 
                { "from": "B1", "to": "C3" }
            ]
        }
    }

    # 2. Analyze without Neo4j
    repo = MemoryRepository()
    repo.save_graph(topology)
    
    analyzer = StructuralAnalyzer()
    graph_data = repo.get_graph_data()
    result = analyzer.analyze(graph_data, layer=AnalysisLayer.SYSTEM)

    # 3. Explain findings
    print("\n  [Findings: Centrality & SPOF]")
    print(f"  {'Component':<15} | {'Betweenness':<12} | {'SPOF Status'}")
    print("  " + "-" * 45)
    
    sorted_comps = sorted(result.components.values(), key=lambda c: c.betweenness, reverse=True)
    for c in sorted_comps:
        spof_label = "⚠️  SPOF" if c.is_articulation_point else "✅ Safe"
        print(f"  {c.id:<15} | {c.betweenness:12.3f} | {spof_label}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("💡  Insight:")
    print("    • 'A1' and 'B1' have high betweenness because many paths flow")
    print("      through them. They are your architectural bottlenecks.")
    print("    • 'A1' is a SPOF (Single Point of Failure) because its removal")
    print("      physically disconnects the A0 -> A1 -> A2 chain.")
    print("\n    By identifying these BEFORE deployment, we can add redundancy")
    print("    exactly where it matters most, saving costs and preventing downtime.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if __name__ == "__main__":
    main()
