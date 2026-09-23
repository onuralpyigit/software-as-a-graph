"""
test_hybrid_prior.py
────────────────────
Pins SaG-Hybrid (PREREGISTRATION.md Amendment 5): HGT-QoS reading the
rank-normalised Topo-QoS score as one extra input column and learning a
correction to its logit.

  * the column is appended only when asked for, after the base features;
  * the default model is untouched (the published 434,620-parameter count
    lives in tests/test_baselines.py);
  * with alpha = 0 the hybrid decodes exactly like the plain composite head;
  * the prior reproduces the published Topo-QoS ordering.
"""

from pathlib import Path

import networkx as nx
import pytest
import torch

pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from saag.prediction.data_preparation import NODE_TYPE_TO_DIM, networkx_to_hetero_data  # noqa: E402


def _tiny_graph():
    g = nx.DiGraph()
    for a in ("A1", "A2", "A3"):
        g.add_node(a, type="Application")
    g.add_node("T1", type="Topic")
    g.add_edge("A1", "T1", type="PUBLISHES_TO", weight=1.0)
    g.add_edge("A2", "T1", type="SUBSCRIBES_TO", weight=1.0)
    g.add_edge("A3", "T1", type="SUBSCRIBES_TO", weight=1.0)
    return g


def test_prior_column_appended_only_on_request():
    g = _tiny_graph()
    sm = {"A1": {"topo_prior": 1.0}, "A2": {"topo_prior": 0.25}}
    plain = networkx_to_hetero_data(g, sm).hetero_data
    hybrid = networkx_to_hetero_data(g, sm, append_prior=True)
    x = hybrid.hetero_data["Application"].x
    assert plain["Application"].x.shape[1] == NODE_TYPE_TO_DIM["Application"]
    assert x.shape[1] == NODE_TYPE_TO_DIM["Application"] + 1
    order = hybrid.node_id_map["Application"]
    got = {nid: float(x[i, -1]) for i, nid in enumerate(order)}
    assert got == {"A1": 1.0, "A2": 0.25, "A3": 0.0}
    # The base block is identical: the prior is appended, never mixed in.
    assert torch.equal(x[:, :-1], plain["Application"].x)


def _metadata():
    return (["Application", "Topic"], [("Application", "PUBLISHES_TO", "Topic")])


def test_default_model_has_no_prior_parameters():
    from saag.prediction.models import build_node_gnn

    base = build_node_gnn(_metadata(), hidden_channels=16, num_heads=2, num_layers=1)
    hyb = build_node_gnn(_metadata(), hidden_channels=16, num_heads=2, num_layers=1,
                         topo_prior=True)
    n_base = sum(p.numel() for p in base.parameters())
    n_hyb = sum(p.numel() for p in hyb.parameters())
    # One extra input weight per hidden unit per node type, plus alpha.
    assert n_hyb - n_base == 2 * 16 + 1
    assert not hasattr(base, "prior_alpha")


def test_zero_alpha_recovers_plain_decode():
    from saag.prediction.models import build_node_gnn

    torch.manual_seed(0)
    model = build_node_gnn(_metadata(), hidden_channels=16, num_heads=2, num_layers=1,
                           topo_prior=True).eval()
    x = {
        "Application": torch.rand(3, NODE_TYPE_TO_DIM["Application"] + 1),
        "Topic": torch.rand(1, NODE_TYPE_TO_DIM["Topic"] + 1),
    }
    ei = {("Application", "PUBLISHES_TO", "Topic"): torch.tensor([[0, 1, 2], [0, 0, 0]])}
    with torch.no_grad():
        model.prior_alpha.fill_(0.0)
        h = model.encode(x, ei)
        assert torch.allclose(model(x, ei)["Application"], model.decode(h)["Application"])
        # With alpha > 0 a higher prior raises the composite score.
        model.prior_alpha.fill_(5.0)
        x["Application"][:, -1] = torch.tensor([0.05, 0.5, 0.95])
        lo_hi = model(x, ei)["Application"][:, 0]
        x["Application"][:, -1] = torch.tensor([0.95, 0.5, 0.05])
        hi_lo = model(x, ei)["Application"][:, 0]
        assert lo_hi[2] > hi_lo[2] and lo_hi[0] < hi_lo[0]


@pytest.mark.skipif(not Path("output/loso_cache/atm_system").exists(),
                    reason="LOSO cache not present")
def test_prior_is_rank_normalised_topo_qos():
    from scipy.stats import spearmanr

    from reproduce.main_table import (
        _compute_topo_baseline_scores, _load_scenario_data, topo_qos_prior,
    )

    prior = topo_qos_prior("atm_system")
    graph, struct, *_ = _load_scenario_data("atm_system", substrate="projection")
    raw = _compute_topo_baseline_scores(graph, struct, use_qos=True)
    assert set(prior) == set(raw)
    # Average ranks: tied zeros share a rank above 0, as registered.
    assert all(0.0 <= v <= 1.0 for v in prior.values())
    ids = sorted(raw)
    rho = spearmanr([prior[i] for i in ids], [raw[i] for i in ids]).correlation
    assert rho == pytest.approx(1.0)
