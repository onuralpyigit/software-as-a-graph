"""
tests/test_amendment19.py — Amendment 19 arms (round-14 referee)
================================================================

Pins the four switches Amendment 19 adds and checks that none of them moves a
reported arm: sum aggregation with reverse edges on the raw multigraph, the
declared-rate inputs of the queue-flow arms, the tie-aware listwise loss, and
the training-scenario subsets of the learning curve.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from tests.test_baselines import (  # noqa: E402
    _make_simulation_results,
    _make_small_graph,
    _make_structural_metrics,
)
from tests.test_feature_ablation import A19_IDS  # noqa: E402

from saag.evaluation import variant_registry as R  # noqa: E402
from saag.prediction.data_preparation import networkx_to_hetero_data  # noqa: E402
from saag.prediction.models.baselines import build_baseline  # noqa: E402
from saag.prediction.models.core import CriticalityLoss  # noqa: E402


def _convert(**kw):
    g = _make_small_graph()
    sm = _make_structural_metrics(g)
    return networkx_to_hetero_data(g, sm, _make_simulation_results(g), **kw).hetero_data


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── registry ──────────────────────────────────────────────────────────────────

def test_every_amendment19_arm_is_registered_dispatched_and_swept():
    from cli.loso_evaluate import _HOMOGENEOUS_VARIANTS
    from reproduce.loso_all_variants import CONTROL_VARIANTS

    for vid in A19_IDS:
        assert vid in R.VARIANTS and vid in _HOMOGENEOUS_VARIANTS and vid in CONTROL_VARIANTS


def test_existing_arms_keep_their_edge_dim():
    for vid, v in R.VARIANTS.items():
        if v.rate_inputs != "node_edge":
            assert R.edge_dim(vid, "loso") == R._EDGE_DIM_BY_QOS[v.qos], vid
    assert R.edge_dim("gl_proj_qos16_cap_idyn_re", "loso") == 17
    assert R.edge_dim("gl_proj_qos16_cap_idyn_r", "loso") == 16


def test_rate_arms_have_no_prior_and_train_on_idyn():
    for vid in A19_IDS:
        v = R.VARIANTS[vid]
        if v.rate_inputs != "none":
            assert v.prior is None and v.label_source == "idyn_full", vid


def test_aggregator_controls_are_raw_graph_gins_with_reverse_edges():
    for vid in ("gin_full_qos16_rev", "gin_full_qos16_rev_min", "gin_full_qos16_rev_const"):
        v = R.VARIANTS[vid]
        assert (v.substrate, v.aggregator, v.reverse_edges, v.hidden_channels) == (
            "native", "gin", True, 228), vid
        assert R.baseline_name_for(vid) == "homo_gin"
    assert R.drop_features_for("gin_full_qos16_rev_min") == R.ORACLE_ALIGNED_FEATURES
    assert R.drop_features_for("gin_full_qos16_rev_const") == R.ALL_NODE_FEATURES


# ── models ────────────────────────────────────────────────────────────────────

def test_parameter_budgets():
    # Reverse edges share weights: GIN-QoS-R has GIN-P-QoS's budget.
    assert _n_params(build_baseline("homo_gin", hidden_channels=228, edge_dim=16,
                                    reverse_edges=True)) == 434_123
    # One extra input column per node type, one extra edge column.
    gat_r = build_baseline("homo_scalar", hidden_channels=288, edge_dim=16, extra_node_cols=1)
    gat_re = build_baseline("homo_scalar", hidden_channels=288, edge_dim=17, extra_node_cols=1)
    gin_re = build_baseline("homo_gin", hidden_channels=228, edge_dim=17, extra_node_cols=1)
    n_types = len(gat_r.input_proj)
    assert _n_params(gat_r) == 429_992 + n_types * 288
    assert _n_params(gat_re) > _n_params(gat_r)
    assert abs(_n_params(gin_re) / 434_123 - 1) < 0.01


def test_rate_columns_reach_the_model():
    g = _make_small_graph()
    sm = _make_structural_metrics(g)
    for n in sm:
        sm[n]["pub_rate"] = 0.5
    for _, _, a in g.edges(data=True):
        a["rate_share"] = 0.25
    data = networkx_to_hetero_data(g, sm, _make_simulation_results(g),
                                   extra_node_keys=("pub_rate",),
                                   extra_edge_keys=("rate_share",)).hetero_data
    for nt in data.node_types:
        assert torch.all(data[nt].x[:, -1] == 0.5)
    for rel in data.edge_types:
        assert data[rel].edge_attr.shape[1] == 17 and torch.all(data[rel].edge_attr[:, -1] == 0.25)
    model = build_baseline("homo_gin", hidden_channels=32, edge_dim=17, extra_node_cols=1)
    out = model({nt: data[nt].x for nt in data.node_types},
                {r: data[r].edge_index for r in data.edge_types},
                {r: data[r].edge_attr for r in data.edge_types})
    sum(o.sum() for o in out.values()).backward()
    assert all(torch.isfinite(o).all() for o in out.values())


def test_default_conversion_is_unchanged():
    a = _convert(rank_normalize_features=True)
    b = _convert(rank_normalize_features=True, extra_node_keys=(), extra_edge_keys=())
    for nt in a.node_types:
        assert torch.equal(a[nt].x, b[nt].x)
    for rel in a.edge_types:
        assert torch.equal(a[rel].edge_attr, b[rel].edge_attr)


# ── rate inputs ───────────────────────────────────────────────────────────────

def _toy_topology():
    return {
        "applications": [{"id": a} for a in ("p1", "p2", "s1", "s2")],
        "topics": [{"id": "t1", "frequency": 10.0}, {"id": "t2", "frequency": 2.0}],
        "relationships": {
            "publishes_to": [{"from": "p1", "to": "t1"}, {"from": "p2", "to": "t1"},
                             {"from": "p2", "to": "t2"}],
            "subscribes_to": [{"from": "s1", "to": "t1"}, {"from": "s2", "to": "t1"},
                              {"from": "s2", "to": "t2"}],
        },
    }


def test_rate_shares_sum_to_eq7():
    from reproduce.idyn_rate_expansion import closed_forms, rate_edge_shares

    topo = _toy_topology()
    shares = rate_edge_shares(topo)
    eq7 = closed_forms(topo)["Rate-I_dyn"]
    for v in ("p1", "p2", "s1", "s2"):
        assert sum(s for (_, t), s in shares.items() if t == v) == pytest.approx(eq7[v])
    assert closed_forms(topo)["PubRate"]["p2"] == 12.0


def test_with_rates_copies_and_adds_columns(monkeypatch):
    import cli.loso_evaluate as L

    b = _bundle()
    apps = L._applications(b)
    u, v = next(iter(b.graph.edges))
    monkeypatch.setitem(L._RATE_CACHE, f"{b.cache_dir}::{b.scenario_id}",
                        ({apps[0]: 0.9}, {(str(u), str(v)): 0.4}))
    out = L._with_rates(b, "node_edge")
    assert out.structural[apps[0]]["pub_rate"] == 0.9 and "pub_rate" not in b.structural.get(apps[0], {})
    assert out.graph.edges[u, v]["rate_share"] == 0.4 and "rate_share" not in b.graph.edges[u, v]
    assert L._extra_keys(out) == {"extra_node_keys": ("pub_rate",),
                                  "extra_edge_keys": ("rate_share",)}
    assert L._extra_keys(b) == {"extra_node_keys": (), "extra_edge_keys": ()}
    node_only = L._with_rates(b, "node")
    assert node_only.graph is b.graph


# ── tie-aware listwise loss ───────────────────────────────────────────────────

def test_tie_loss_is_invariant_to_input_order():
    torch.manual_seed(0)
    scores = torch.randn(40)
    targets = torch.tensor([0.0] * 15 + list(np.linspace(0.1, 1.0, 25)), dtype=torch.float32)
    base = CriticalityLoss._listmle_ties_loss(scores, targets)
    for seed in range(5):
        perm = torch.from_numpy(np.random.default_rng(seed).permutation(40))
        assert CriticalityLoss._listmle_ties_loss(scores[perm], targets[perm]) == pytest.approx(
            base.item(), abs=1e-5)
    # Plain ListMLE is not order-invariant under ties; that is what the arm tests.
    plain = {round(CriticalityLoss._listmle_loss(scores[p], targets[p]).item(), 5)
             for p in (torch.randperm(40) for _ in range(5))}
    assert len(plain) > 1


def test_tie_loss_equals_listmle_without_ties():
    torch.manual_seed(1)
    scores, targets = torch.randn(30), torch.rand(30)
    assert CriticalityLoss._listmle_ties_loss(scores, targets).item() == pytest.approx(
        CriticalityLoss._listmle_loss(scores, targets).item(), abs=1e-6)


def test_default_loss_is_plain_listmle():
    assert CriticalityLoss().ranking_loss == "listmle"
    assert R.ranking_loss_for("gl_proj_qos16_cap") == "listmle"
    assert R.ranking_loss_for("gl_proj_qos16_cap_tie") == "listmle_ties"
    with pytest.raises(ValueError):
        CriticalityLoss(ranking_loss="listnet")


# ── learning-curve subsets ────────────────────────────────────────────────────

def _bundle(sid: str = "atm_system", n: int = 0):
    from cli.loso_evaluate import ScenarioBundle

    g = _make_small_graph()
    return ScenarioBundle(
        scenario_id=sid, graph=g, structural={}, rm={},
        simulation=_make_simulation_results(g), hetero_data=None,
        n_nodes=n or g.number_of_nodes(), n_edges=g.number_of_edges(), n_labelled=0,
    )


def _corpus():
    return [_bundle(f"s{i:02d}", n=10 + i) for i in range(12)]


def test_train_subsets_are_nested_deterministic_and_draw_dependent():
    from cli.loso_evaluate import _train_subset

    train = _corpus()[1:]
    ids = lambda bs: {b.scenario_id for b in bs}  # noqa: E731
    subsets = [ids(_train_subset(train, "s00", k, 1)) for k in (1, 2, 4, 8, 11)]
    assert [len(s) for s in subsets] == [1, 2, 4, 8, 11]
    assert all(a <= b for a, b in zip(subsets, subsets[1:]))
    assert ids(_train_subset(train, "s00", 4, 1)) == subsets[2]
    assert any(ids(_train_subset(train, "s00", 4, d)) != subsets[2] for d in (2, 3))


def test_plan_fold_with_and_without_subset(tmp_path):
    from cli.loso_evaluate import _plan_fold

    corpus = _corpus()
    full = _plan_fold(corpus, 0, 3, False, "none", tmp_path)
    same = _plan_fold(corpus, 0, 3, False, "none", tmp_path, train_subset=(11, 1))
    assert full.train_ids == same.train_ids and full.primary.scenario_id == "s11"
    sub = _plan_fold(corpus, 0, 3, False, "none", tmp_path, train_subset=(4, 1))
    assert len(sub.train_ids) == 4 and "s00" not in sub.train_ids
    assert sub.primary.scenario_id == max(sub.train_set, key=lambda b: b.n_nodes).scenario_id
    assert {b.scenario_id for b in sub.inductives} == set(sub.train_ids) - {sub.primary.scenario_id}


def test_sweep_forwards_max_train_only_when_set():
    from reproduce.loso_all_variants import _extra_args

    base = Namespace(eval_population="application", auto_layers=True, inner_val_scenario="none",
                     skip="", rank_normalize_features=False, rank_normalize_labels=False,
                     device="cpu", jobs=10, torch_threads=1, preflight=True,
                     max_train=None, subset_seed=0)
    assert "--max-train" not in _extra_args(base)
    args = _extra_args(replace_ns(base, max_train=4, subset_seed=2))
    assert args[-4:] == ["--max-train", "4", "--subset-seed", "2"]


def replace_ns(ns: Namespace, **kw) -> Namespace:
    return Namespace(**{**vars(ns), **kw})


def test_variant_bundle_applies_rates_only_to_rate_arms(monkeypatch):
    import cli.loso_evaluate as L

    b = replace(_bundle(), cache_dir=None)
    assert L._variant_bundle(b, "gl_proj_qos16_cap", relabel=False) is not None
    assert L._variant_bundle(b, "gl_proj_qos16_cap", relabel=False).rate_inputs == "none"
    monkeypatch.setitem(L._RATE_CACHE, f"{b.cache_dir}::{b.scenario_id}", ({}, {}))
    out = L._variant_bundle(b, "gl_proj_qos16_cap_idyn_re", relabel=False)
    assert out.rate_inputs == "node_edge"
