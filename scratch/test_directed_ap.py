import networkx as nx
import sys
import os

# Set up path to include backend
sys.path.append(os.path.abspath("backend"))

from src.analysis.structural_analyzer import StructuralAnalyzer

def test_directed_ap():
    # Scenario: A -> B -> C
    # A is the root. B is a directed AP for C.
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    
    analyzer = StructuralAnalyzer()
    directed_aps = analyzer._compute_directed_articulation_points(G)
    
    print(f"Graph A -> B -> C")
    print(f"Directed APs: {directed_aps}")
    assert "B" in directed_aps
    assert "A" in directed_aps # A is a root, but its removal isolates B and C
    
    # Scenario: A -> B -> C, A -> C
    # B is NOT a directed AP because C is still reachable via A -> C.
    G2 = nx.DiGraph()
    G2.add_edge("A", "B")
    G2.add_edge("B", "C")
    G2.add_edge("A", "C")
    
    directed_aps2 = analyzer._compute_directed_articulation_points(G2)
    print(f"Graph A -> B -> C, A -> C")
    print(f"Directed APs: {directed_aps2}")
    assert "B" not in directed_aps2

    # Scenario: Diamond with bottleneck
    # S -> A -> B -> T, S -> C -> B -> T
    # B is a directed AP for T.
    G3 = nx.DiGraph()
    G3.add_edge("S", "A")
    G3.add_edge("S", "C")
    G3.add_edge("A", "B")
    G3.add_edge("C", "B")
    G3.add_edge("B", "T")
    
    directed_aps3 = analyzer._compute_directed_articulation_points(G3)
    print(f"Diamond S->A,C->B->T")
    print(f"Directed APs: {directed_aps3}")
    assert "B" in directed_aps3
    assert "A" not in directed_aps3
    assert "C" not in directed_aps3

    print("All directed AP tests passed!")

if __name__ == "__main__":
    test_directed_ap()
