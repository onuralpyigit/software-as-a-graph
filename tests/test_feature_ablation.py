"""
tests/test_feature_ablation.py — Amendment 14 feature switches, GIN arm and label sources
========================================================================================

Pins the three inputs Amendment 14 changes: zeroed node-feature columns (the
degree-free arms), QoS node columns exempt from QoS-off masking (the w_in-held
2x2), and the training-label source (the I_dyn surrogate). Every reported arm
must be untouched by all three.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch_geometric")

from tests.test_baselines import (  # noqa: E402
    _make_simulation_results,
    _make_small_graph,
    _make_structural_metrics,
)

from saag.evaluation import variant_registry as R  # noqa: E402
from saag.prediction.data_preparation import (  # noqa: E402
    BASE_METRIC_KEYS,
    networkx_to_hetero_data,
)

ROOT = Path(__file__).resolve().parent.parent
A14_IDS = [
    "gl_proj_qos16_cap_nodeg", "gl_proj_qos16_cap_nodeg_strict", "gl_full_qos16_cap_nodeg",
    "gin_proj_qos16", "gin_proj_qos16_nodeg", "gin_proj_qos16_nodeg_strict",
    "gl_full_cap_win", "hgl_win", "gl_proj_qos16_cap_idyn", "gl_proj_qos16_cap_istar_app",
]
#: Amendment 17 arms. They use the same switches, so they are held to the same invariants.
A17_IDS = [
    "gl_proj_qos16_cap_min", "gl_full_qos16_cap_rev_min", "gl_full_qos16_cap_min",
    "gin_proj_qos16_min", "gin_proj_qos16_const", "gl_proj_qos16_cap_perm",
    "gl_proj_qos16_cap_idyn_rate", "gl_proj_qos16_cap_perm18", "gl_proj_qos16_cap_perm19",
]


def _convert(**kw):
    g = _make_small_graph()
    return networkx_to_hetero_data(
        g, _make_structural_metrics(g), _make_simulation_results(g), **kw
    ).hetero_data


def _col(key: str) -> int:
    return list(BASE_METRIC_KEYS).index(key)


@pytest.mark.parametrize("rank_normalize", [False, True])
def test_drop_zeroes_only_the_named_columns(rank_normalize):
    drop = R.DEGREE_FEATURES
    base = _convert(rank_normalize_features=rank_normalize)
    dropped = _convert(rank_normalize_features=rank_normalize, drop_feature_keys=drop)
    idx = [_col(k) for k in drop]
    keep = [i for i in range(len(BASE_METRIC_KEYS)) if i not in idx]
    for nt in base.node_types:
        a, b = base[nt].x.numpy(), dropped[nt].x.numpy()
        assert a.shape == b.shape
        assert np.all(b[:, idx] == 0.0)
        np.testing.assert_array_equal(a[:, keep], b[:, keep])


def test_qos_exempt_keeps_w_in_and_masks_the_rest():
    on = _convert(qos_enabled=True)
    off = _convert(qos_enabled=False)
    exempt = _convert(qos_enabled=False, qos_exempt_keys=("qos_weight_in",))
    w, w_in, w_out = _col("qos_weight"), _col("qos_weight_in"), _col("qos_weight_out")
    for nt in on.node_types:
        np.testing.assert_array_equal(exempt[nt].x[:, w_in].numpy(), on[nt].x[:, w_in].numpy())
        assert np.all(exempt[nt].x[:, [w, w_out]].numpy() == 0.0)
        np.testing.assert_array_equal(off[nt].x[:, w_in].numpy(), 0.0)


def test_structural_mask_keep_and_default():
    from reproduce.main_table import _mask_qos_in_structural

    sm = {"a": {"qos_weight_in": 0.7, "w_in": 0.7, "qos_weight": 0.4, "pagerank": 0.2}}
    default = _mask_qos_in_structural(sm)
    assert default["a"] == {"qos_weight_in": 0.0, "w_in": 0.0, "qos_weight": 0.0, "pagerank": 0.2}
    kept = _mask_qos_in_structural(sm, keep=("qos_weight_in",))
    assert kept["a"] == {"qos_weight_in": 0.7, "w_in": 0.7, "qos_weight": 0.0, "pagerank": 0.2}
    assert sm["a"]["qos_weight_in"] == 0.7          # input not mutated


def test_registry_invariants():
    for vid in A14_IDS + A17_IDS:
        v = R.VARIANTS[vid]
        if v.drop_node_features != R.ALL_NODE_FEATURES:
            assert set(v.drop_node_features) <= set(BASE_METRIC_KEYS), vid
        assert set(v.qos_exempt_node_features) <= set(BASE_METRIC_KEYS), vid
        if v.qos_exempt_node_features:
            assert not R.node_qos_for(vid, "loso"), vid
        if v.aggregator == "gin":
            assert R.edge_dim(vid, "loso") == 16 and R.baseline_name_for(vid) == "homo_gin"
        if v.label_source != "i_star":
            assert v.substrate == "projection" and v.aggregator == "gat", vid
    # Every reported arm keeps the defaults.
    for vid, v in R.VARIANTS.items():
        if vid not in A14_IDS + A17_IDS:
            assert (v.drop_node_features, v.qos_exempt_node_features, v.aggregator,
                    v.label_source, v.permute_nodes, v.permutation_seed) == (
                        (), (), "gat", "i_star", False, None), vid


def test_new_arms_are_dispatched_and_swept():
    from cli.loso_evaluate import _HGT_VARIANTS, _HOMOGENEOUS_VARIANTS
    from reproduce.loso_all_variants import CONTROL_VARIANTS

    for vid in A14_IDS + A17_IDS:
        assert vid in CONTROL_VARIANTS
        assert vid in (_HGT_VARIANTS if vid == "hgl_win" else _HOMOGENEOUS_VARIANTS)


def test_oracle_aligned_set_is_base_and_contains_cdi_and_foc():
    assert set(R.ORACLE_ALIGNED_FEATURES) <= set(BASE_METRIC_KEYS)
    assert {"cdi", "fan_out_criticality", "in_degree_centrality"} <= set(R.ORACLE_ALIGNED_FEATURES)


@pytest.mark.parametrize("rank_normalize", [False, True])
def test_wildcard_drop_zeroes_every_column(rank_normalize):
    out = _convert(rank_normalize_features=rank_normalize, drop_feature_keys=R.ALL_NODE_FEATURES)
    for nt in out.node_types:
        assert np.all(out[nt].x.numpy() == 0.0), nt


def test_permuted_graph_keeps_content_and_changes_order():
    from cli.loso_evaluate import _permuted_graph

    g = _make_small_graph()
    p = _permuted_graph(g, R.PERMUTATION_SEED)
    assert set(p.nodes) == set(g.nodes) and list(p.nodes) != list(g.nodes)
    assert all(p.nodes[n] == g.nodes[n] for n in g.nodes)
    assert sorted(map(str, p.edges(data=True))) == sorted(map(str, g.edges(data=True)))
    assert p.graph == g.graph


def test_only_the_permutation_arm_permutes():
    from cli.loso_evaluate import _variant_bundle

    b = _bundle()
    assert _variant_bundle(b, "gl_proj_qos16_cap", relabel=False).graph is b.graph
    out = _variant_bundle(b, "gl_proj_qos16_cap_perm", relabel=False)
    assert list(out.graph.nodes) != list(b.graph.nodes)


def test_hgt_qos_flag_matches_the_retired_tuple():
    legacy = ("hgl_qos", "hgl_qos_uni", "hgl_qos_prior", "hgl_proj_qos")
    for vid in ("hgl", "hgl_qos", "hgl_qos_uni", "hgl_qos_prior", "topology_rm", "hgl_proj_qos"):
        assert R.node_qos_for(vid, "loso") == (vid in legacy), vid


def test_gin_parameter_budget_and_forward():
    import torch
    from saag.prediction.models.baselines import build_baseline

    gat = build_baseline("homo_scalar", hidden_channels=288, edge_dim=16)
    gin = build_baseline("homo_gin", hidden_channels=228, edge_dim=16)
    assert sum(p.numel() for p in gat.parameters()) == 429_992
    assert sum(p.numel() for p in gin.parameters()) == 434_123
    data = _convert()
    x = {nt: data[nt].x for nt in data.node_types}
    ei = {r: data[r].edge_index for r in data.edge_types}
    ea = {r: data[r].edge_attr for r in data.edge_types}
    out = gin(x, ei, ea)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    assert all(torch.isfinite(o).all() for o in out.values())


def test_gnn_service_round_trips_the_switches(tmp_path):
    from saag.prediction.gnn_service import GNNService

    svc = GNNService(checkpoint_dir=str(tmp_path), drop_feature_keys=R.DEGREE_FEATURES,
                     qos_exempt_keys=("qos_weight_in",))
    svc.layer = "app"
    svc._save_service_config()
    import json
    cfg = json.loads((tmp_path / "service_config.json").read_text())
    assert cfg["drop_feature_keys"] == list(R.DEGREE_FEATURES)
    assert cfg["qos_exempt_keys"] == ["qos_weight_in"]


def _bundle():
    from cli.loso_evaluate import ScenarioBundle

    g = _make_small_graph()
    return ScenarioBundle(
        scenario_id="atm_system", graph=g, structural={}, rm={},
        simulation=_make_simulation_results(g), hetero_data=None,
        n_nodes=g.number_of_nodes(), n_edges=g.number_of_edges(), n_labelled=0,
    )


def test_istar_app_label_source_keeps_applications_only():
    from cli.loso_evaluate import _applications, _variant_bundle

    b = _bundle()
    out = _variant_bundle(b, "gl_proj_qos16_cap_istar_app", relabel=True)
    assert set(out.simulation) == set(_applications(b))
    held = _variant_bundle(b, "gl_proj_qos16_cap_istar_app", relabel=False)
    assert held.simulation is b.simulation                      # holdout stays on I*


def test_idyn_label_source_rejects_non_applications(monkeypatch):
    import cli.loso_evaluate as L

    b = _bundle()
    apps = L._applications(b)
    monkeypatch.setitem(L._LABEL_FILE_CACHE, "idyn_full", {
        "labels": {"atm_system": {"mean": {apps[0]: 0.3, apps[1]: 0.1}}}})
    out = L._variant_bundle(b, "gl_proj_qos16_cap_idyn", relabel=True)
    assert out.simulation == {apps[0]: {"composite": 0.3}, apps[1]: {"composite": 0.1}}
    monkeypatch.setitem(L._LABEL_FILE_CACHE, "idyn_full", {
        "labels": {"atm_system": {"mean": {"Broker1": 0.3}}}})
    with pytest.raises(RuntimeError):
        L._variant_bundle(b, "gl_proj_qos16_cap_idyn", relabel=True)


def test_idyn_label_file_matches_the_corpus():
    import cli.loso_evaluate as L

    if not L._LABEL_FILES["idyn_full"].exists():
        pytest.skip("Amendment 11 labels not present")
    L._LABEL_FILE_CACHE.pop("idyn_full", None)
    assert "atm_system" in L._label_file("idyn_full")["labels"]


def test_harness_does_not_import_simulation():
    tree = ast.parse((ROOT / "cli" / "loso_evaluate.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("saag.simulation"), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("saag.simulation") for a in node.names)


def test_permutation_seeds_differ():
    seeds = {R.permutation_seed_for(v) for v in
             ("gl_proj_qos16_cap_perm", "gl_proj_qos16_cap_perm18", "gl_proj_qos16_cap_perm19")}
    assert seeds == {17, 18, 19}
