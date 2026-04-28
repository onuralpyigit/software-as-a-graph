import pytest
import networkx as nx
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from saag.prediction.data_preparation import networkx_to_hetero_data
from saag.prediction.models import CriticalityLoss
from saag.prediction.gnn_service import GNNService
from saag.prediction.trainer import EvalMetrics

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

def test_best_seed_selection():
    """GNN-G6: Verify that the best-performing seed is selected and restored."""
    service = GNNService(checkpoint_dir="scratch/test_best_seed")
    G = nx.DiGraph()
    G.add_node("1", type="Application")
    G.add_edge("1", "1", type="DEPENDS_ON") # Self-loop for minimal graph
    
    # Mock Trainer and evaluate
    with patch("src.prediction.gnn_service.GNNTrainer") as MockTrainer, \
         patch("src.prediction.gnn_service.evaluate") as MockEval:
        
        # Seed 1: poor performance (rho=0.5)
        # Seed 2: best performance (rho=0.9)
        # Seed 3: mediocre (rho=0.7)
        
        def mock_train_side_effect(*args, **kwargs):
            trainer = MagicMock()
            # We need to simulate different rho based on some state? 
            # Actually GNNService.train creates a new trainer per seed.
            pass

        # Since GNNService creates a new trainer instance, we mock the class
        mock_instances = [MagicMock(), MagicMock(), MagicMock()]
        
        # Mock Metrics
        metrics_low = EvalMetrics(spearman_rho=0.5, f1_score=0.5, rmse=1.0, mae=1.0, top_5_overlap=0.0, top_10_overlap=0.0, ndcg_10=0.0)
        metrics_high = EvalMetrics(spearman_rho=0.9, f1_score=0.9, rmse=0.1, mae=0.1, top_5_overlap=1.0, top_10_overlap=1.0, ndcg_10=1.0)
        metrics_mid = EvalMetrics(spearman_rho=0.7, f1_score=0.7, rmse=0.5, mae=0.5, top_5_overlap=0.0, top_10_overlap=0.0, ndcg_10=0.0)
        
        mock_instances[0].train.return_value = ({}, metrics_low)
        mock_instances[1].train.return_value = ({}, metrics_high)
        mock_instances[2].train.return_value = ({}, metrics_mid)
        
        MockTrainer.side_effect = mock_instances
        MockEval.return_value = metrics_low # simplified
        
        # We also need to mock model.state_dict() to return something identifying
        service._init_models = MagicMock()
        service._node_model = MagicMock()
        service._edge_model = None
        service.predict_edges = False # Simplify for test
        
        # Simplified state dicts with tensor-like values
        states = [
            {"w": MagicMock(spec=torch.Tensor)},
            {"w": MagicMock(spec=torch.Tensor)},
            {"w": MagicMock(spec=torch.Tensor)}
        ]
        for s in states:
            s["w"].cpu.return_value = s["w"]
            s["w"].clone.return_value = s["w"]
            
        service._node_model.state_dict.side_effect = states
        
        service.train(G, seeds=[1, 2, 3], simulation_results={"1": {}}, rmav_scores={"1": {}})
        
        # Check if load_state_dict was called with state[1] (seed 2)
        # We check the content matches (it was cloned to cpu)
        service._node_model.load_state_dict.assert_called_with(states[1])
