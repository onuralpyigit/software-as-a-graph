import pytest
import json
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.prediction.gnn_service import GNNService, GNNAnalysisResult
from torch_geometric.data import HeteroData

@pytest.fixture
def temp_checkpoint(tmp_path):
    """Create a temporary checkpoint with a service_config.json."""
    ckpt_dir = tmp_path / "gnn_ckpt"
    ckpt_dir.mkdir()
    config = {
        "hidden_channels": 64,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.2,
        "predict_edges": False,
        "layer": "app"
    }
    with open(ckpt_dir / "service_config.json", "w") as f:
        json.dump(config, f)
    return ckpt_dir

def test_layer_validation_error(temp_checkpoint):
    """G13: Verify that from_checkpoint raises ValueError on layer mismatch."""
    # Loading as 'infra' when ckpt is 'app'
    with pytest.raises(ValueError, match="GNN Layer Mismatch"):
        GNNService.from_checkpoint(str(temp_checkpoint), layer="infra")

def test_layer_validation_missing_ok(temp_checkpoint):
    """G13: Verify that loading without layer parameter or with matching layer is OK."""
    # Matching
    service = GNNService.from_checkpoint(str(temp_checkpoint), layer="app")
    assert service.layer == "app"
    
    # Missing (fallback)
    service2 = GNNService.from_checkpoint(str(temp_checkpoint))
    assert service2.layer == "app"

def test_ensemble_fallback_warning(caplog):
    """G12: Verify warning log and gnn_only mode when RMAV is missing."""
    service = GNNService()
    service._node_model = MagicMock()
    service._node_model.predict_edges = False
    service._node_model.to.return_value = service._node_model
    service._node_model.eval.return_value = None
    service._node_model.return_value = {"Application": torch.randn(2, 5)}
    
    # Mock conversion result
    conv = MagicMock()
    conv.node_id_map = {"Application": ["app1", "app2"]}
    service._conversion_result = conv
    
    data = HeteroData()
    data["Application"].x = torch.randn(2, 23)
    # y_rmav is missing
    
    import logging
    with caplog.at_level(logging.WARNING):
        result = service.predict_from_data(data, mode="ensemble")
        
        assert "Ensemble mode selected but RMAV or Ensemble model missing" in caplog.text
        assert result.prediction_mode == "gnn_only"

def test_ensemble_mode_reporting():
    """G12: Verify prediction_mode Literal reporting."""
    service = GNNService()
    service._node_model = MagicMock()
    service._node_model.predict_edges = False
    service._node_model.return_value = {"Application": torch.randn(2, 5)}
    service._conversion_result = MagicMock()
    service._conversion_result.node_id_map = {"Application": ["app1", "app2"]}
    
    data = HeteroData()
    data["Application"].x = torch.randn(2, 23)
    data["Application"].y_rmav = torch.randn(2, 5)
    
    # Mock ensemble
    service._ensemble = MagicMock()
    service._ensemble.alpha = torch.tensor([0.5]*5)
    service._ensemble.return_value = torch.randn(2, 5)
    
    result = service.predict_from_data(data, mode="ensemble")
    assert result.prediction_mode == "ensemble"
    
    result_gnn = service.predict_from_data(data, mode="gnn")
    assert result_gnn.prediction_mode == "gnn_only"
    
    result_rmav = service.predict_from_data(data, mode="rmav")
    assert result_rmav.prediction_mode == "rmav_only"
