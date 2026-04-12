import pytest
import networkx as nx
import torch
import numpy as np
from src.prediction.data_preparation import networkx_to_hetero_data
from src.prediction.models import CriticalityLoss

def test_bridge_aware_edge_labels():
    """GNN-G3: Verify that edge labels are grounded in bridge property."""
    G = nx.DiGraph()
    # Simple line graph: 1 -> 2 -> 3
    # Both 1->2 and 2->3 are bridges in the underlying undirected graph
    G.add_edge("1", "2")
    G.add_edge("2", "3")
    G.nodes["1"]["type"] = "Application"
    G.nodes["2"]["type"] = "Application"
    G.nodes["3"]["type"] = "Topic"
    
    # Simulation results: node 1 is critical
    simulation = {
        "1": {"composite": 1.0, "reliability": 0.8, "maintainability": 0.0, "availability": 0.0, "vulnerability": 0.0},
        "2": {"composite": 0.5, "reliability": 0.4, "maintainability": 0.0, "availability": 0.0, "vulnerability": 0.0},
        "3": {"composite": 0.1, "reliability": 0.1, "maintainability": 0.0, "availability": 0.0, "vulnerability": 0.0},
    }
    
    # 1. Test Bridge (Line Graph)
    conv = networkx_to_hetero_data(G, simulation_results=simulation)
    data = conv.hetero_data
    
    # rel (Application, DEPENDS_ON, Application) [1->2]
    # In my code, for Apps it uses DEPENDS_ON ? 
    # Actually networkx_to_hetero_data uses the edge structure from G.
    # Let's check which relation it created.
    # By default edges in G are treated as DEPENDS_ON if not specified? 
    # No, it uses graph.edges(data=True) and checks 'type'.
    
    # Let's add explicit types
    G.edges["1", "2"]["type"] = "DEPENDS_ON"
    G.edges["2", "3"]["type"] = "PUBLISHES_TO"
    
    conv = networkx_to_hetero_data(G, simulation_results=simulation)
    data = conv.hetero_data
    
    # Edge 1->2 is a bridge. Multiplier should be 1.0.
    # Label should be sim["1"]["composite"] * 1.0 = 1.0
    rel_12 = ("Application", "DEPENDS_ON", "Application")
    assert data[rel_12].y_edge[0, 0] == pytest.approx(1.0)
    
    # 2. Test Non-Bridge (Cycle or Parallel)
    G.add_edge("1", "3", type="DEPENDS_ON")
    # Now 1->2 and 1->3 and 2->3 are not bridges individually for connectivity 1-3?
    # Actually undirected bridges: losing 1->2 still allows 1-3-2? Wait, 2->3 is directed.
    # Undirected: 1-2, 2-3, 1-3 forms a triangle. No bridges.
    
    conv = networkx_to_hetero_data(G, simulation_results=simulation)
    data = conv.hetero_data
    
    # Multiplier should be 0.1
    # Label should be sim["1"]["composite"] * 0.1 = 0.1
    assert data[rel_12].y_edge[0, 0] == pytest.approx(0.1)

def test_consistency_loss_logic():
    """GNN-G2: Verify that consistency loss applies only to unlabeled nodes."""
    loss_fn = CriticalityLoss(multitask_weight=0.5, rmav_consistency_weight=0.1, ranking_weight=0.3)
    
    # 2 nodes, 1 labeled (mask=True), 1 unlabeled (mask=False)
    pred = torch.tensor([[0.8, 0.7, 0.7, 0.7, 0.7], [0.2, 0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[1.0, 0.9, 0.9, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([True, False])
    rmav_target = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3, 0.3]])
    
    total, components = loss_fn(pred, target, mask, rmav_target)
    
    # Component loss: MSE(0.8, 1.0) = 0.04
    # Multitask loss: MSE([0.7]*4, [0.9]*4) = 0.04
    # Consistency loss: MSE([0.1]*4, [0.3]*4) = 0.04 (on index 1 only!)
    # Weight for consistency is 0.1
    
    assert components["composite"] == pytest.approx(0.04)
    assert components["multitask"] == pytest.approx(0.04)
    assert components["consistency"] == pytest.approx(0.04)
    
    # Total = 0.04 + 0.5*0.04 + 0.1*0.04 + 0.3*ranking
    # We just want to ensure it's calculated using the correct masks.
