"""
test_attribution_controls.py
────────────────────────────
Pins the attribution controls of PREREGISTRATION.md Amendment 7 and the
architectural fact the manuscript now discloses (Section 6.2):

  * ``node_qos_for`` leaves every pre-existing arm's QoS inputs exactly as they
    were: node-feature QoS follows the edge channel unless a variant decouples it;
  * GAT-QoS-nf's inputs are bit-identical to one parent arm each -- node
    features from GAT, edge attributes from GAT-QoS;
  * the untyped GATs score an Application from its own features: no relation
    targets an Application and they aggregate along edge direction only. If this
    test starts failing, the manuscript's RQ2/RQ3 wording no longer describes the
    model and must change with it.
"""

from pathlib import Path

import networkx as nx
import pytest
import torch

pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from saag.evaluation import variant_registry as registry  # noqa: E402
from saag.prediction.data_preparation import networkx_to_hetero_data  # noqa: E402
from saag.prediction.models.baselines import build_baseline  # noqa: E402


def test_node_qos_follows_edge_channel_except_where_decoupled():
    decoupled = {"tab_gbm_qos": True, "gl_full_qos16_nfmask": False}
    for vid in registry.VARIANTS:
        expected = decoupled.get(vid, registry.edge_dim(vid, "loso") is not None)
        assert registry.node_qos_for(vid, "loso") is expected, vid


def _tiny_graph():
    g = nx.DiGraph()
    for a in ("A1", "A2", "A3"):
        g.add_node(a, type="Application")
    g.add_node("T1", type="Topic")
    g.add_node("N1", type="Node")
    g.add_edge("A1", "T1", type="PUBLISHES_TO", weight=1.0)
    g.add_edge("A2", "T1", type="SUBSCRIBES_TO", weight=1.0)
    g.add_edge("A3", "T1", type="SUBSCRIBES_TO", weight=1.0)
    for a in ("A1", "A2", "A3"):
        g.add_edge(a, "N1", type="RUNS_ON", weight=1.0)
    return g


@pytest.mark.parametrize("variant", ["gl_full_cap", "gl_full_qos16_cap"])
def test_untyped_gat_scores_applications_per_node(variant):
    data = networkx_to_hetero_data(_tiny_graph(), {}).hetero_data
    edge_dim = registry.edge_dim(variant, "loso")
    torch.manual_seed(0)
    model = build_baseline(
        "homo_unweighted" if edge_dim is None else "homo_scalar",
        hidden_channels=32, num_heads=4, num_layers=3, dropout=0.0, edge_dim=edge_dim,
    ).eval()
    x = {nt: data[nt].x for nt in data.node_types}
    ei = {r: data[r].edge_index for r in data.edge_types}
    ea = {r: data[r].edge_attr for r in data.edge_types}
    with torch.no_grad():
        full = model(x, ei, ea)["Application"]
        bare = model(x, {r: e[:, :0] for r, e in ei.items()},
                     {r: e[:0] for r, e in ea.items()})["Application"]
    assert torch.equal(full, bare)


_CACHE = Path("output/loso_cache/atm_system")


@pytest.mark.skipif(not _CACHE.exists(), reason="output/loso_cache/atm_system not populated")
def test_nfmask_inputs_are_bit_identical_to_their_parents():
    from cli.loso_evaluate import _build_training_hetero, _graft_qos_edge_attr, load_scenario_bundle

    bundle = load_scenario_bundle(_CACHE)
    off = _build_training_hetero(bundle, False, False)
    on = _build_training_hetero(bundle, True, False)
    nf = _build_training_hetero(bundle, False, False)
    _graft_qos_edge_attr(nf, bundle, False)
    assert all(torch.equal(nf[t].x, off[t].x) for t in nf.node_types)
    assert all(torch.equal(nf[r].edge_attr, on[r].edge_attr) for r in nf.edge_types)
    # And the two parents really differ on both inputs, or the control is vacuous.
    assert not all(torch.equal(off[t].x, on[t].x) for t in off.node_types)
    assert not all(torch.equal(off[r].edge_attr, on[r].edge_attr) for r in off.edge_types)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_hgt_reaches_applications_only_through_its_reverse_pass(bidirectional):
    """HGT-QoS-U (no reverse pass) is per-node at Applications; HGT-QoS is not."""
    from saag.prediction.models.core import build_node_gnn

    data = networkx_to_hetero_data(_tiny_graph(), {}).hetero_data
    torch.manual_seed(0)
    model = build_node_gnn(data.metadata(), 32, 4, 3, 0.0, use_bidirectional=bidirectional).eval()
    x = {nt: data[nt].x for nt in data.node_types}
    ei = {r: data[r].edge_index for r in data.edge_types}
    ea = {r: data[r].edge_attr for r in data.edge_types}
    with torch.no_grad():
        full = model(x, ei, ea)["Application"]
        bare = model(x, {r: e[:, :0] for r, e in ei.items()},
                     {r: e[:0] for r, e in ea.items()})["Application"]
    assert torch.equal(full, bare) is (not bidirectional)
