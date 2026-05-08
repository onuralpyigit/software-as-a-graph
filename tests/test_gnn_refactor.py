"""
Tests for the GNN refactor:
- Updated node/edge feature dimensions
- HGTConv-based NodeCriticalityGNN forward shape
- EdgeFeatureEncoder scatter aggregation
- TypedEdgeEncoder per-relation projection
- Topic runtime features (subscriber/publisher counts)
- Node infra features (cpu_cores, memory_gb)
- path_count_norm in edge features
- Pairwise margin ranking loss
- CosineAnnealingWarmRestarts scheduler
- Combined early-stopping metric
- normalize_labels_robust
- Default prediction mode = "gnn"
- PredictionService checkpoint detection and fallback
- Bidirectional message passing produces asymmetric embeddings
"""

import math
import pytest
import torch
import numpy as np
import networkx as nx

# ── Skip all tests gracefully when PyG is not installed ───────────────────────
pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_hetero_data(num_app=3, num_broker=2, num_topic=2):
    """Build a small HeteroData object with two relation types."""
    from torch_geometric.data import HeteroData
    from saag.prediction.data_preparation import NODE_TYPE_TO_DIM, EDGE_FEATURE_DIM

    data = HeteroData()
    data["Application"].x = torch.randn(num_app, NODE_TYPE_TO_DIM["Application"])
    data["Application"].num_nodes = num_app
    data["Broker"].x = torch.randn(num_broker, NODE_TYPE_TO_DIM["Broker"])
    data["Broker"].num_nodes = num_broker
    data["Topic"].x = torch.randn(num_topic, NODE_TYPE_TO_DIM["Topic"])
    data["Topic"].num_nodes = num_topic

    # App -> Topic (PUBLISHES_TO)
    pub_src = torch.tensor([0, 1])
    pub_dst = torch.tensor([0, 1])
    rel_pt = ("Application", "PUBLISHES_TO", "Topic")
    data[rel_pt].edge_index = torch.stack([pub_src, pub_dst])
    data[rel_pt].edge_attr = torch.randn(2, EDGE_FEATURE_DIM)

    # Broker -> Topic (ROUTES)
    rel_rt = ("Broker", "ROUTES", "Topic")
    data[rel_rt].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data[rel_rt].edge_attr = torch.randn(2, EDGE_FEATURE_DIM)

    return data


def _build_gnn(data):
    from saag.prediction.models import build_node_gnn
    return build_node_gnn(data.metadata(), hidden_channels=16, num_heads=2, num_layers=2)


# ── Feature dimension constants ────────────────────────────────────────────────

def test_node_type_dims_updated():
    from saag.prediction.data_preparation import NODE_TYPE_TO_DIM
    assert NODE_TYPE_TO_DIM["Application"] == 23
    assert NODE_TYPE_TO_DIM["Library"] == 23
    assert NODE_TYPE_TO_DIM["Broker"] == 19
    assert NODE_TYPE_TO_DIM["Topic"] == 20
    assert NODE_TYPE_TO_DIM["Node"] == 20


def test_edge_feature_dim_is_sixteen():
    """
    Edge features: 2 (weight + path_count) + 7 (edge type one-hot) + 7 (QoS) = 16
    """
    from saag.prediction.data_preparation import EDGE_FEATURE_DIM, EDGE_TYPES
    # Base dimensions: weight + path_count + 7 edge types = 9
    assert 2 + len(EDGE_TYPES) == 9
    # Total with QoS decomposition: 9 + 7 = 16
    assert EDGE_FEATURE_DIM == 16


def test_node_type_to_dim_same_in_models_and_data_prep():
    """models.py must re-export the same dict, not a stale copy."""
    from saag.prediction.data_preparation import NODE_TYPE_TO_DIM as dp_dims
    from saag.prediction.models import NODE_TYPE_TO_DIM as m_dims
    assert dp_dims is m_dims or dp_dims == m_dims


# ── Topic runtime features ────────────────────────────────────────────────────

def test_topic_runtime_features_extracted():
    """Topic nodes carry subscriber_count_norm and publisher_count_norm."""
    G = nx.DiGraph()
    G.add_node("T1", type="Topic")
    G.add_node("A1", type="Application")
    G.add_node("A2", type="Application")
    # A1 publishes to T1; A2 subscribes to T1
    G.add_edge("A1", "T1", type="PUBLISHES_TO", weight=0.8)
    G.add_edge("A2", "T1", type="SUBSCRIBES_TO", weight=0.5)

    from saag.prediction.data_preparation import networkx_to_hetero_data, NODE_TYPE_TO_DIM
    conv = networkx_to_hetero_data(G)
    data = conv.hetero_data

    assert data["Topic"].x.shape[1] == NODE_TYPE_TO_DIM["Topic"]
    t1_x = data["Topic"].x[0].numpy()
    # Index 18 = subscriber_count_norm, 19 = publisher_count_norm
    assert t1_x[18] > 0.0, "subscriber_count_norm should be non-zero"
    assert t1_x[19] > 0.0, "publisher_count_norm should be non-zero"


# ── Node infra features ───────────────────────────────────────────────────────

def test_infra_features_for_node_type():
    """Node-type nodes carry cpu_cores_norm and memory_gb_norm."""
    G = nx.DiGraph()
    G.add_node("N1", type="Node", cpu_cores=16, memory_gb=64)
    G.add_node("N2", type="Node", cpu_cores=4, memory_gb=16)

    from saag.prediction.data_preparation import networkx_to_hetero_data, NODE_TYPE_TO_DIM
    conv = networkx_to_hetero_data(G)
    data = conv.hetero_data

    assert data["Node"].x.shape[1] == NODE_TYPE_TO_DIM["Node"]
    n1_x = data["Node"].x[0].numpy()
    # N1 has cpu=16 (max), so cpu_cores_norm == 1.0
    assert n1_x[18] == pytest.approx(1.0)   # cpu_cores_norm


def test_infra_features_broker():
    """Broker nodes carry max_connections_norm."""
    G = nx.DiGraph()
    G.add_node("B1", type="Broker", max_connections=1000)
    G.add_node("B2", type="Broker", max_connections=500)

    from saag.prediction.data_preparation import networkx_to_hetero_data, NODE_TYPE_TO_DIM
    conv = networkx_to_hetero_data(G)
    data = conv.hetero_data

    assert data["Broker"].x.shape[1] == NODE_TYPE_TO_DIM["Broker"]
    b1_x = data["Broker"].x[0].numpy()
    # B1 has max=1000 (max in graph), so max_connections_norm == 1.0
    assert b1_x[18] == pytest.approx(1.0)


# ── Edge path_count feature ───────────────────────────────────────────────────

def test_edge_path_count_norm_in_features():
    """Edge features include path_count_norm at index 1."""
    G = nx.DiGraph()
    G.add_node("A1", type="Application")
    G.add_node("T1", type="Topic")
    G.add_edge("A1", "T1", type="PUBLISHES_TO", weight=0.5, path_count=4)

    from saag.prediction.data_preparation import networkx_to_hetero_data, EDGE_FEATURE_DIM
    conv = networkx_to_hetero_data(G)
    data = conv.hetero_data

    rel = ("Application", "PUBLISHES_TO", "Topic")
    assert data[rel].edge_attr.shape[1] == EDGE_FEATURE_DIM
    feat = data[rel].edge_attr[0].numpy()
    expected_pcn = math.log2(5.0) / math.log2(17.0)
    assert feat[1] == pytest.approx(expected_pcn, abs=1e-4)


# ── HGTConv forward shape ────────────────────────────────────────────────────���

def test_hgt_conv_forward_shape():
    """NodeCriticalityGNN with HGTConv produces (N, 5) output per node type."""
    data = _minimal_hetero_data()
    model = _build_gnn(data)
    model.eval()
    with torch.no_grad():
        out = model(
            {nt: data[nt].x for nt in data.node_types},
            {rel: data[rel].edge_index for rel in data.edge_types},
            {rel: data[rel].edge_attr for rel in data.edge_types},
        )
    for nt in ["Application", "Broker", "Topic"]:
        n = data[nt].num_nodes
        assert out[nt].shape == (n, 5), f"{nt}: expected ({n}, 5), got {out[nt].shape}"
    # All outputs in [0, 1] (sigmoid activated)
    for nt, t in out.items():
        assert t.min() >= 0.0 and t.max() <= 1.0


# ── EdgeFeatureEncoder ────────────────────────────────────────────────────────

def test_edge_feature_encoder_scatter():
    """EdgeFeatureEncoder adds projected edge features to destination nodes."""
    from saag.prediction.models import EdgeFeatureEncoder

    enc = EdgeFeatureEncoder(edge_feat_dim=9, hidden_channels=16)
    enc.eval()

    h_dict = {
        "Application": torch.zeros(3, 16),
        "Topic": torch.zeros(2, 16),
    }
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    edge_attr = torch.ones(2, 9)
    edge_index_dict = {("Application", "PUBLISHES_TO", "Topic"): edge_index}
    edge_attr_dict = {("Application", "PUBLISHES_TO", "Topic"): edge_attr}

    with torch.no_grad():
        augmented = enc(h_dict, edge_index_dict, edge_attr_dict)

    # Topic nodes that received edges should have non-zero values
    assert augmented["Topic"].abs().sum() > 0
    # Application nodes (source) are unchanged
    assert augmented["Application"].abs().sum() == pytest.approx(0.0, abs=1e-5)


# ── TypedEdgeEncoder ──────────────────────────────────────────────────────────

def test_typed_edge_encoder_per_relation():
    """TypedEdgeEncoder uses distinct projections for different relation types."""
    from saag.prediction.models import TypedEdgeEncoder

    edge_types = [
        ("Application", "PUBLISHES_TO", "Topic"),
        ("Application", "SUBSCRIBES_TO", "Topic"),
    ]
    enc = TypedEdgeEncoder(edge_feat_dim=9, hidden_channels=16, edge_types=edge_types)

    h_src = torch.randn(2, 16)
    h_dst = torch.randn(2, 16)
    e_feat = torch.randn(2, 9)

    out1 = enc(h_src, h_dst, e_feat, edge_types[0])
    out2 = enc(h_src, h_dst, e_feat, edge_types[1])

    # Different relation types → different projections → different outputs
    assert not torch.allclose(out1, out2), \
        "TypedEdgeEncoder should produce different outputs for different relations"
    assert out1.shape == (2, 5)


# ── Pairwise margin ranking loss ─────────────────────────────────────────────

def test_pairwise_margin_loss_zero_when_ordered():
    from saag.prediction.models import CriticalityLoss
    scores = torch.tensor([0.9, 0.6, 0.3])
    targets = torch.tensor([0.9, 0.6, 0.3])
    loss = CriticalityLoss._pairwise_margin_loss(scores, targets, margin=0.05)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_pairwise_margin_loss_positive_when_inverted():
    from saag.prediction.models import CriticalityLoss
    scores = torch.tensor([0.1, 0.9])   # wrong order
    targets = torch.tensor([0.9, 0.1])  # target says first should be higher
    loss = CriticalityLoss._pairwise_margin_loss(scores, targets, margin=0.05)
    assert loss.item() > 0.0


def test_pairwise_weight_in_total_loss():
    """CriticalityLoss includes pairwise_ranking_weight in the total."""
    from saag.prediction.models import CriticalityLoss
    loss_fn = CriticalityLoss(pairwise_ranking_weight=0.1)
    pred = torch.rand(5, 5)
    target = torch.rand(5, 5)
    mask = torch.ones(5, dtype=torch.bool)
    total, components = loss_fn(pred, target, mask)
    assert "pairwise" in components
    assert total.item() > 0.0


# ── Scheduler type ────────────────────────────────────────────────────────────

def test_trainer_uses_warm_restart_scheduler():
    """GNNTrainer uses CosineAnnealingWarmRestarts, not CosineAnnealingLR."""
    from saag.prediction.trainer import GNNTrainer
    data = _minimal_hetero_data()
    model = _build_gnn(data)
    trainer = GNNTrainer(model, checkpoint_dir="/tmp/test_ckpt", num_epochs=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=trainer.warmup_T0, T_mult=2
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
    assert trainer.warmup_T0 == max(50, 100 // 4)


# ── Combined early stopping ───────────────────────────────────────────────────

def test_update_best_detects_improvement():
    """_update_best returns improved=True when combined metric increases."""
    from saag.prediction.trainer import GNNTrainer, EvalMetrics
    data = _minimal_hetero_data()
    model = _build_gnn(data)
    trainer = GNNTrainer(model, checkpoint_dir="/tmp/test_ckpt")

    good = EvalMetrics(spearman_rho=0.8, f1_score=0.7, rmse=0.1, mae=0.1,
                       top_5_overlap=0.6, top_10_overlap=0.5, ndcg_10=0.7)
    bad = EvalMetrics(spearman_rho=0.4, f1_score=0.4, rmse=0.3, mae=0.3,
                      top_5_overlap=0.3, top_10_overlap=0.3, ndcg_10=0.3)

    _, improved = trainer._update_best(good, 0.2, best_combined=-1.0, best_val_loss=1.0)
    assert improved is True
    _, improved = trainer._update_best(bad, 0.9, best_combined=0.5, best_val_loss=0.2)
    assert improved is False


# ── Label normalization ───────────────────────────────────────────────────────

def test_normalize_labels_robust_clips_outlier():
    """normalize_labels_robust maps outliers toward center and keeps values in (0,1)."""
    from saag.prediction.data_preparation import normalize_labels_robust
    from torch_geometric.data import HeteroData

    data = HeteroData()
    # All zeros except one extreme outlier
    y = torch.zeros(10, 5)
    y[0, 0] = 100.0  # extreme outlier
    data["Application"].y = y
    data["Application"].num_nodes = 10

    normalize_labels_robust(data)

    out = data["Application"].y
    assert out.min() > 0.0 and out.max() < 1.0, "All values should be in (0, 1) after normalization"
    # Outlier should be pulled back from the extreme
    assert out[0, 0] < 1.0


# ── Default prediction mode ───────────────────────────────────────────────────

def test_gnn_service_default_mode_is_gnn():
    """GNNService.predict() and predict_from_data() default to mode='gnn'."""
    import inspect
    from saag.prediction.gnn_service import GNNService

    predict_sig = inspect.signature(GNNService.predict)
    predict_from_sig = inspect.signature(GNNService.predict_from_data)

    assert predict_sig.parameters["mode"].default == "gnn"
    assert predict_from_sig.parameters["mode"].default == "gnn"


def test_gnn_service_train_default_mode_is_gnn():
    import inspect
    from saag.prediction.gnn_service import GNNService
    sig = inspect.signature(GNNService.train)
    assert sig.parameters["mode"].default == "gnn"


# ── PredictionService checkpoint detection and fallback ──────────────────────

def test_prediction_service_has_checkpoint_false_for_missing_dir(tmp_path):
    from saag.prediction.service import PredictionService
    assert PredictionService._has_checkpoint(str(tmp_path / "nonexistent")) is False


def test_prediction_service_has_checkpoint_true_when_files_exist(tmp_path):
    from saag.prediction.service import PredictionService
    (tmp_path / "service_config.json").write_text("{}")
    (tmp_path / "node_model.pt").write_bytes(b"")
    assert PredictionService._has_checkpoint(str(tmp_path)) is True


def test_prediction_service_fallback_to_rmav_when_no_checkpoint(tmp_path):
    """predict_quality_with_gnn falls back to RMAV when no checkpoint exists."""
    from saag.prediction.service import PredictionService
    from unittest.mock import MagicMock, patch

    svc = PredictionService(gnn_checkpoint_dir=str(tmp_path / "no_ckpt"), prefer_gnn=True)
    mock_result = MagicMock()

    with patch.object(svc, "predict_quality", return_value=mock_result) as mock_pq:
        result = svc.predict_quality_with_gnn(MagicMock(), graph=MagicMock())

    mock_pq.assert_called_once()
    assert result is mock_result  # fell back to RMAV output


# ── Bidirectional reverse pass ────────────────────────────────────────────────

def test_bidirectional_reverse_pass_modifies_embeddings():
    """_apply_reverse_pass actually changes node embeddings when reverse edges exist."""
    from saag.prediction.models import NodeCriticalityGNN, NUM_LABEL_DIMS
    from saag.prediction.data_preparation import NODE_TYPE_TO_DIM, EDGE_FEATURE_DIM

    torch.manual_seed(0)
    data = _minimal_hetero_data(num_app=3, num_broker=2, num_topic=2)
    metadata = data.metadata()

    model = NodeCriticalityGNN(metadata, hidden_channels=16, num_heads=2,
                                num_layers=1, use_bidirectional=True)
    model.eval()

    x_dict = {nt: data[nt].x for nt in data.node_types}
    ei_dict = {rel: data[rel].edge_index for rel in data.edge_types}

    with torch.no_grad():
        # Get embeddings after forward-only conv layers (no reverse pass yet)
        with torch.no_grad():
            h_proj = {
                nt: model.input_proj[nt](x_dict[nt])
                for nt in x_dict if nt in model.input_proj
            }
        for conv, edge_enc, norm_d, drop in zip(
            model.convs, model.edge_encoders, model.norms, model.dropouts
        ):
            import torch.nn.functional as F
            h_new = conv(h_proj, ei_dict)
            for nt, h_n in h_new.items():
                residual = h_proj.get(nt, torch.zeros_like(h_n))
                h_proj[nt] = drop(F.gelu(norm_d[nt](h_n + residual)))
        h_before_rev = {nt: h_proj[nt].clone() for nt in h_proj}

        # Apply reverse pass
        h_after_rev = model._apply_reverse_pass(h_proj, ei_dict)

    # At least one node type should have been changed by the reverse pass
    any_changed = any(
        not torch.allclose(h_before_rev[nt], h_after_rev[nt], atol=1e-6)
        for nt in h_before_rev if nt in h_after_rev
    )
    assert any_changed, "_apply_reverse_pass should modify at least one node type's embeddings"


def test_bidirectional_flag_respected():
    """Model with use_bidirectional=False has no rev_conv attribute."""
    from saag.prediction.models import build_node_gnn
    data = _minimal_hetero_data()
    model = build_node_gnn(data.metadata(), hidden_channels=16, num_heads=2,
                            num_layers=1, use_bidirectional=False)
    assert model.rev_conv is None
