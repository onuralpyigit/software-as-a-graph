import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from saag.prediction.gnn_service import GNNService, GNNAnalysisResult
from saag.prediction.data_preparation import GraphConversionResult
from torch_geometric.data import HeteroData

@pytest.fixture
def mock_gnn_service():
    """Service with mocked models and internal metadata."""
    service = GNNService(checkpoint_dir="test_ckpt")
    service._node_model = MagicMock()
    service._node_model.predict_edges = False # Avoid unpacking error in trainer.evaluate
    service._node_model.to.return_value = service._node_model
    service._node_model.eval.return_value = None
    
    # Mocking conversion result for ID mapping
    conv = MagicMock(spec=GraphConversionResult)
    conv.node_id_map = {"Application": ["app1", "app2"]}
    conv.edge_name_map = {}
    service._conversion_result = conv
    
    return service

def test_mask_consistency_best_seed(mock_gnn_service):
    """G9: Verify that the best seed is persisted and used for splitting."""
    service = mock_gnn_service
    service._best_seed = 1234
    
    data = HeteroData()
    data["Application"].x = torch.randn(2, 23)
    data["Application"].y = torch.tensor([[0.5]*5, [0.8]*5])
    
    # Mock evaluate to return something
    from saag.prediction.trainer import EvalMetrics
    mock_metrics = EvalMetrics(0.9, 0.8, 0.1, 0.1, 1.0, 1.0, 1.0)
    
    with patch("src.prediction.gnn_service.create_node_splits") as mock_split:
        with patch("src.prediction.gnn_service.evaluate", return_value=mock_metrics):
            service._node_model.return_value = {"Application": torch.randn(2, 5)}
            service.predict_from_data(data, simulation_results={"app1": {}})
            
            # Verify create_node_splits was called with seed=1234
            from unittest.mock import ANY
            mock_split.assert_called_with(ANY, seed=1234)

def test_ensemble_metrics_computation(mock_gnn_service):
    """G10: Verify that ensemble metrics are computed during validation."""
    service = mock_gnn_service
    service._ensemble = MagicMock()
    service._ensemble.alpha = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    
    data = HeteroData()
    data["Application"].x = torch.randn(2, 23)
    data["Application"].y = torch.tensor([[0.5]*5, [0.8]*5])
    data["Application"].y_rmav = torch.tensor([[0.4]*5, [0.7]*5])
    data["Application"].test_mask = torch.tensor([True, True])
    
    # Mock GNN outputs
    gnn_out = torch.tensor([[0.6]*5, [0.9]*5])
    service._node_model.return_value = {"Application": gnn_out}
    
    # Mock evaluate_scores
    from saag.prediction.trainer import EvalMetrics
    
    with patch("src.prediction.trainer.evaluate_scores") as mock_eval_scores:
        mock_eval_scores.return_value = EvalMetrics(0.95, 0.9, 0.05, 0.05, 1.0, 1.0, 1.0)
        
        # We need ensemble_scores to be populated
        result = service.predict_from_data(data, simulation_results={"app1": {}}, mode="ensemble")
        
        assert result.ensemble_metrics is not None
        assert result.ensemble_metrics.spearman_rho == 0.95
        mock_eval_scores.assert_called()

def test_ablation_modes(mock_gnn_service):
    """G11: Verify that mode selection filters correct scores."""
    service = mock_gnn_service
    
    data = HeteroData()
    data["Application"].x = torch.randn(2, 23)
    data["Application"].y_rmav = torch.tensor([[0.1]*5, [0.2]*5])
    
    service._node_model.return_value = {"Application": torch.tensor([[0.8]*5, [0.9]*5])}
    
    # 1. Mode: RMAV
    result_rmav = service.predict_from_data(data, mode="rmav")
    assert result_rmav.node_scores["app1"].source == "RMAV"
    assert result_rmav.node_scores["app1"].composite_score == pytest.approx(0.1)
    
    # 2. Mode: GNN
    result_gnn = service.predict_from_data(data, mode="gnn")
    assert result_gnn.node_scores["app1"].source == "GNN"
    assert result_gnn.node_scores["app1"].composite_score == pytest.approx(0.8)
    
    # 3. Mode: Ensemble (mocked ensemble model)
    service._ensemble = lambda g, r: 0.5 * g + 0.5 * r
    service._ensemble.alpha = MagicMock()
    service._ensemble.alpha.detach.return_value.cpu.return_value.tolist.return_value = [0.5]*5
    
    result_ens = service.predict_from_data(data, mode="ensemble")
    assert result_ens.node_scores["app1"].source == "Ensemble"
    assert result_ens.node_scores["app1"].composite_score == pytest.approx(0.45) # (0.8+0.1)/2
