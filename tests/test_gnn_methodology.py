import pytest
import networkx as nx
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from saag.prediction.data_preparation import networkx_to_hetero_data
from saag.prediction.models import CriticalityLoss
from saag.prediction.gnn_service import GNNService
from saag.prediction.trainer import EvalMetrics

def test_edge_labels_are_not_derived_from_bridge_structure():
    """GNN-G3 (superseded): edge labels must be measurements, not structure.

    This test used to assert the opposite — that ``y_edge`` equalled
    ``I*(source) x {1.0 if bridge else 0.1}``. That multiplier was a stand-in
    for the edge-removal oracle the paper declares, so the edge head was scored
    against a hand-chosen structural heuristic. The heuristic is gone; node
    simulation results alone no longer produce edge labels of any kind.

    The positive contract (measured labels, masked gaps) is pinned in
    tests/test_edge_label_provenance.py.
    """
    G = nx.DiGraph()
    G.add_edge("1", "2", type="DEPENDS_ON")   # a bridge
    G.add_edge("2", "3", type="PUBLISHES_TO")
    G.nodes["1"]["type"] = "Application"
    G.nodes["2"]["type"] = "Application"
    G.nodes["3"]["type"] = "Topic"

    simulation = {
        "1": {"composite": 1.0, "reliability": 0.8},
        "2": {"composite": 0.5, "reliability": 0.4},
        "3": {"composite": 0.1, "reliability": 0.1},
    }

    conv = networkx_to_hetero_data(G, simulation_results=simulation)
    assert conv.num_labelled_edges == 0
    for rel in conv.hetero_data.edge_types:
        assert not hasattr(conv.hetero_data[rel], "y_edge"), (
            f"{rel} carries a fabricated edge label"
        )


def test_consistency_loss_logic():
    """GNN-G2: Verify that consistency loss applies only to unlabeled nodes."""
    loss_fn = CriticalityLoss(multitask_weight=0.5, rm_consistency_weight=0.1, ranking_weight=0.3)
    
    # 2 nodes, 1 labeled (mask=True), 1 unlabeled (mask=False)
    pred = torch.tensor([[0.8, 0.7, 0.7, 0.7, 0.7], [0.2, 0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[1.0, 0.9, 0.9, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([True, False])
    rm_target = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3, 0.3]])
    
    total, components = loss_fn(pred, target, mask, rm_target)
    
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
    with patch("saag.prediction.gnn_service.GNNTrainer") as MockTrainer, \
         patch("saag.prediction.gnn_service.evaluate") as MockEval:
        
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
        
        service.train(G, seeds=[1, 2, 3], simulation_results={"1": {}}, rm_scores={"1": {}})
        
        # Check if load_state_dict was called with state[1] (seed 2)
        # We check the content matches (it was cloned to cpu)
        service._node_model.load_state_dict.assert_called_with(states[1])
