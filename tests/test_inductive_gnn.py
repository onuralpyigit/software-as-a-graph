"""
Tests for the Inductive Split Protocol and optimized EdgeFeatureEncoder.
"""

import pytest
import torch
import networkx as nx
from unittest.mock import MagicMock, patch
from torch_geometric.data import HeteroData
from saag.prediction.trainer import get_inductive_subgraph, GNNTrainer, EvalMetrics
from saag.prediction.models import EdgeFeatureEncoder, build_node_gnn
from saag.prediction.gnn_service import GNNService

# Gracefully skip tests if PyTorch Geometric is not installed
pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")


def _build_complex_hetero_data():
    """Build a HeteroData graph with train, val, and test masks."""
    data = HeteroData()

    # Application nodes: 4 nodes
    # Node 0: train, Node 1: val, Node 2: test, Node 3: unlabelled
    data["Application"].x = torch.randn(4, 23)
    data["Application"].y = torch.tensor([
        [0.8, 0.7, 0.6, 0.8, 0.9],  # train node (labelled)
        [0.4, 0.3, 0.2, 0.4, 0.5],  # val node (labelled)
        [0.6, 0.5, 0.4, 0.6, 0.7],  # test node (labelled)
        [0.0, 0.0, 0.0, 0.0, 0.0]   # unlabelled node
    ], dtype=torch.float)
    data["Application"].y_rmav = torch.randn(4, 5)
    data["Application"].train_mask = torch.tensor([True, False, False, False], dtype=torch.bool)
    data["Application"].val_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
    data["Application"].test_mask = torch.tensor([False, False, True, False], dtype=torch.bool)
    data["Application"].num_nodes = 4

    # Topic nodes: 3 nodes (no masks, meaning all should be kept/queried)
    data["Topic"].x = torch.randn(3, 22)
    data["Topic"].num_nodes = 3

    # Edges: Application -> Topic (PUBLISHES_TO)
    # Edge 0: App 0 -> Topic 0 (train edge)
    # Edge 1: App 1 -> Topic 1 (val edge)
    # Edge 2: App 2 -> Topic 2 (test edge)
    # Edge 3: App 3 -> Topic 0 (unlabelled edge)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 0]
    ], dtype=torch.long)
    rel = ("Application", "PUBLISHES_TO", "Topic")
    data[rel].edge_index = edge_index
    data[rel].edge_attr = torch.randn(4, 16)

    return data


def test_get_inductive_subgraph_isolates_nodes():
    """Verify that get_inductive_subgraph isolates node partitions correctly."""
    data = _build_complex_hetero_data()

    # 1. Training partition isolation
    train_graph = get_inductive_subgraph(data, "train_mask")
    # Only Application node 0 should be present
    assert train_graph["Application"].num_nodes == 1
    # All 3 Topic nodes are kept because Topic has no mask
    assert train_graph["Topic"].num_nodes == 3

    # Check edges in train_graph: only edges involving Application node 0 should remain
    # Edge 0 (0->0) and Edge 3 (3->0) - but wait, App node 3 is dropped, so only 0->0 remains.
    # Original indices were 0, 1, 2, 3. The subgraph drops nodes 1, 2, 3.
    # So the only edge remaining should connect Application node 0 to Topic node 0.
    rel = ("Application", "PUBLISHES_TO", "Topic")
    assert train_graph[rel].edge_index.shape[1] == 1
    # Check updated edge index (should map to 0 -> 0)
    assert torch.equal(train_graph[rel].edge_index, torch.tensor([[0], [0]], dtype=torch.long))

    # 2. Validation partition isolation
    val_graph = get_inductive_subgraph(data, "val_mask")
    assert val_graph["Application"].num_nodes == 1
    assert val_graph["Topic"].num_nodes == 3
    # Only Edge 1 (1->1) remains
    assert val_graph[rel].edge_index.shape[1] == 1
    assert torch.equal(val_graph[rel].edge_index, torch.tensor([[0], [1]], dtype=torch.long))

    # 3. Testing partition isolation
    test_graph = get_inductive_subgraph(data, "test_mask")
    assert test_graph["Application"].num_nodes == 1
    assert test_graph["Topic"].num_nodes == 3
    # Only Edge 2 (2->2) remains
    assert test_graph[rel].edge_index.shape[1] == 1
    assert torch.equal(test_graph[rel].edge_index, torch.tensor([[0], [2]], dtype=torch.long))


def test_get_inductive_subgraph_no_mask_graceful_fallback():
    """Verify that get_inductive_subgraph falls back gracefully to original graph when no mask is found."""
    data = _build_complex_hetero_data()

    # Query with a non-existent mask
    fallback_graph = get_inductive_subgraph(data, "non_existent_mask")
    assert fallback_graph is data

    # Query when mask has zero sum
    data["Application"].empty_mask = torch.tensor([False, False, False, False], dtype=torch.bool)
    fallback_graph_2 = get_inductive_subgraph(data, "empty_mask")
    assert fallback_graph_2 is data


def test_edge_feature_encoder_fusion():
    """Verify that EdgeFeatureEncoder correctly concatenates node & edge features and runs without dimension error."""
    # hidden_channels = 8, edge_feat_dim = 6
    # expected proj input dimension = 8 * 2 + 6 = 22
    encoder = EdgeFeatureEncoder(edge_feat_dim=6, hidden_channels=8)

    h_dict = {
        "Application": torch.randn(3, 8),
        "Topic": torch.randn(2, 8)
    }
    edge_index = torch.tensor([
        [0, 1, 2],
        [0, 1, 0]
    ], dtype=torch.long)
    edge_attr = torch.randn(3, 6)

    edge_index_dict = {
        ("Application", "PUBLISHES_TO", "Topic"): edge_index
    }
    edge_attr_dict = {
        ("Application", "PUBLISHES_TO", "Topic"): edge_attr
    }

    # Execute forward pass
    augmented = encoder(h_dict, edge_index_dict, edge_attr_dict)

    # Destination nodes (Topic) should receive messages and change value
    assert augmented["Topic"].shape == (2, 8)
    assert not torch.allclose(augmented["Topic"], h_dict["Topic"])

    # Source nodes (Application) should not be updated by scatter-mean on destination index
    assert torch.allclose(augmented["Application"], h_dict["Application"])


def test_gnn_trainer_inductive_subgraphs():
    """Verify that GNNTrainer uses the inductive subgraphs during run epochs and validation."""
    data = _build_complex_hetero_data()
    # Build a small GNN
    model = build_node_gnn(data.metadata(), hidden_channels=8, num_heads=2, num_layers=1)
    trainer = GNNTrainer(model, checkpoint_dir="test_ckpt", num_epochs=1, patience=1)

    # We will spy on the model's forward method to ensure it only receives subgraphs matching the partition size
    called_node_counts = []

    original_forward = model.forward

    def spy_forward(x_dict, edge_index_dict, edge_attr_dict=None):
        # Record the number of Application nodes in the input
        called_node_counts.append(x_dict["Application"].shape[0])
        return original_forward(x_dict, edge_index_dict, edge_attr_dict)

    model.forward = spy_forward

    try:
        # Run one step of val loss computation
        _ = trainer._compute_val_loss(data)
        # The validation subgraph has only 1 Application node
        assert len(called_node_counts) == 1
        assert called_node_counts[0] == 1

        called_node_counts.clear()

        # Run one epoch
        # Create loader with 1 batch containing our data
        from torch_geometric.loader import DataLoader
        loader = DataLoader([data], batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        _ = trainer._run_epoch(loader, optimizer)

        # The epoch run uses train_mask subgraph which also has only 1 Application node
        assert len(called_node_counts) == 1
        assert called_node_counts[0] == 1

    finally:
        # Restore the original forward method
        model.forward = original_forward


def _mock_service_for_train(service):
    """Stub out the parts of GNNService.train() that aren't under test here
    (model construction/state_dict handling), mirroring the pattern used by
    test_gnn_methodology.py::test_best_seed_selection.
    """
    service._init_models = MagicMock()
    service._node_model = MagicMock()
    service._edge_model = None
    service.predict_edges = False
    w = MagicMock(spec=torch.Tensor)
    w.cpu.return_value = w
    w.clone.return_value = w
    service._node_model.state_dict.return_value = {"w": w}


def test_normalize_labels_robust_applied_to_all_inductive_graphs(tmp_path):
    """Regression test for the LOSO label-scale bug: normalize_labels_robust must
    run on every graph passed via inductive_graphs, not just the primary graph —
    otherwise the primary and auxiliary scenarios train against label distributions
    on different numeric scales within the same backward pass.
    """
    service = GNNService(checkpoint_dir=str(tmp_path / "ckpt"))
    G = nx.DiGraph()
    G.add_node("1", type="Application")
    G.add_edge("1", "1", type="DEPENDS_ON")

    ig1 = HeteroData()
    ig1["Application"].x = torch.randn(2, 23)
    ig1["Application"].y = torch.zeros(2, 5)
    ig1["Application"].num_nodes = 2
    ig2 = HeteroData()
    ig2["Application"].x = torch.randn(2, 23)
    ig2["Application"].y = torch.zeros(2, 5)
    ig2["Application"].num_nodes = 2

    seen_graphs = []

    def spy(hetero_data):
        seen_graphs.append(hetero_data)

    metrics = EvalMetrics(spearman_rho=0.5, f1_score=0.5, rmse=1.0, mae=1.0,
                           top_5_overlap=0.0, top_10_overlap=0.0, ndcg_10=0.0)

    with patch("saag.prediction.data_preparation.normalize_labels_robust", side_effect=spy), \
         patch("saag.prediction.gnn_service.GNNTrainer") as MockTrainer, \
         patch("saag.prediction.gnn_service.evaluate", return_value=metrics):
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = ({}, metrics)
        MockTrainer.return_value = mock_trainer_instance

        _mock_service_for_train(service)

        service.train(
            G, seeds=[1],
            simulation_results={"1": {"composite": 1.0, "reliability": 0.0,
                                       "maintainability": 0.0, "availability": 0.0,
                                       "security": 0.0}},
            rmav_scores={"1": {}},
            inductive_graphs=[ig1, ig2],
        )

    # The primary graph (built internally from G) plus both inductive graphs
    # must each have gone through normalize_labels_robust exactly once.
    assert len(seen_graphs) == 3
    assert ig1 in seen_graphs
    assert ig2 in seen_graphs


def test_gnn_trainer_validates_against_primary_graph_in_loso(tmp_path):
    """Regression test for the LOSO validation-target bug: when training with
    inductive_graphs, GNNTrainer.train() must validate/early-stop against the
    primary (held-in) scenario, not whichever graph happens to land first in
    the shuffled multi-graph DataLoader.
    """
    service = GNNService(checkpoint_dir=str(tmp_path / "ckpt"))
    G = nx.DiGraph()
    G.add_node("1", type="Application")
    G.add_edge("1", "1", type="DEPENDS_ON")

    ig1 = HeteroData()
    ig1["Application"].x = torch.randn(2, 23)
    ig1["Application"].y = torch.zeros(2, 5)
    ig1["Application"].num_nodes = 2

    metrics = EvalMetrics(spearman_rho=0.5, f1_score=0.5, rmse=1.0, mae=1.0,
                           top_5_overlap=0.0, top_10_overlap=0.0, ndcg_10=0.0)

    with patch("saag.prediction.gnn_service.GNNTrainer") as MockTrainer, \
         patch("saag.prediction.gnn_service.evaluate", return_value=metrics):
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = ({}, metrics)
        MockTrainer.return_value = mock_trainer_instance

        _mock_service_for_train(service)

        service.train(
            G, seeds=[1],
            simulation_results={"1": {"composite": 1.0, "reliability": 0.0,
                                       "maintainability": 0.0, "availability": 0.0,
                                       "security": 0.0}},
            rmav_scores={"1": {}},
            inductive_graphs=[ig1],
        )

    # trainer.train() must have been called with primary_data set to the
    # primary graph (not None, and not left to loader-shuffle chance).
    _, call_kwargs = mock_trainer_instance.train.call_args
    assert call_kwargs.get("primary_data") is not None
    assert call_kwargs["primary_data"] is not ig1


def test_gnn_trainer_train_uses_primary_data_when_provided():
    """Unit test for GNNTrainer.train()'s primary_data override: validation
    must run against the explicitly-provided primary graph, not the first item
    the DataLoader happens to yield.
    """
    data = _build_complex_hetero_data()
    other = _build_complex_hetero_data()

    model = build_node_gnn(data.metadata(), hidden_channels=8, num_heads=2, num_layers=1)
    trainer = GNNTrainer(model, checkpoint_dir="test_ckpt", num_epochs=1, patience=1)

    from torch_geometric.loader import DataLoader
    # Loader order deliberately puts `other` first, `data` (primary) second.
    loader = DataLoader([other, data], batch_size=1, shuffle=False)

    seen_val_targets = []
    original_compute_val_loss = trainer._compute_val_loss

    def spy_compute_val_loss(hetero_data):
        seen_val_targets.append(hetero_data)
        return original_compute_val_loss(hetero_data)

    trainer._compute_val_loss = spy_compute_val_loss

    trainer.train(loader, primary_data=data)

    assert len(seen_val_targets) >= 1
    assert all(t is data for t in seen_val_targets)
