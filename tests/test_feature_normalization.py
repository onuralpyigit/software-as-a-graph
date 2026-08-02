"""
test_feature_normalization.py
──────────────────────────────
Pins the within-graph rank normalization added to close D1 (cross-scenario
feature scale drift, e.g. PageRank ~ 1/N spanning ~9x across a 74-to-520-node
corpus). Verifies: output range, exact rank preservation (including ties),
that type-specific "extra" columns are left untouched, that the flag is off
by default (bit-identical to pre-existing behaviour), and the n<=1 guard.
"""

import networkx as nx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.stats import spearmanr  # noqa: E402

from saag.prediction.data_preparation import (  # noqa: E402
    BASE_METRIC_KEYS,
    _rank_normalize_base_columns,
    networkx_to_hetero_data,
)


def _graph(n_apps: int = 5, n_brokers: int = 0) -> nx.DiGraph:
    g = nx.DiGraph()
    for i in range(n_apps):
        g.add_node(f"A{i}", component_type="Application", type="Application")
    for i in range(n_brokers):
        g.add_node(f"B{i}", component_type="Broker", type="Broker")
    g.add_node("T0", component_type="Topic", type="Topic")
    for i in range(n_apps):
        g.add_edge(f"A{i}", "T0", dependency_type="app_to_app",
                   type="PUBLISHES_TO", weight=1.0)
    return g


def _structural_metrics(n_apps: int = 5) -> dict:
    # pagerank has a tie (A0 and A2 both 0.5); betweenness spans two orders
    # of magnitude, matching the scale drift the diagnostic surfaced.
    pagerank = [0.5, 0.1, 0.5, 0.3, 0.9][:n_apps]
    betweenness = [10.0, 20.0, 30.0, 5.0, 100.0][:n_apps]
    sm = {}
    for i in range(n_apps):
        sm[f"A{i}"] = {
            "pagerank": pagerank[i],
            "betweenness_centrality": betweenness[i],
            "loc_norm": 0.1 * i,          # CQ extra — must stay untouched
            "complexity_norm": 0.2 * i,   # CQ extra — must stay untouched
        }
    return sm


def test_rank_normalize_range_and_order():
    g = _graph()
    sm = _structural_metrics()
    result = networkx_to_hetero_data(g, structural_metrics=sm, rank_normalize_features=True)
    x = result.hetero_data["Application"].x.numpy()

    n_base = len(BASE_METRIC_KEYS)
    assert np.all(x[:, :n_base] >= 0.0) and np.all(x[:, :n_base] <= 1.0)

    pr_col = BASE_METRIC_KEYS.index("pagerank")
    bt_col = BASE_METRIC_KEYS.index("betweenness_centrality")
    raw_pr = [0.5, 0.1, 0.5, 0.3, 0.9]
    raw_bt = [10.0, 20.0, 30.0, 5.0, 100.0]

    assert spearmanr(raw_pr, x[:, pr_col]).correlation == pytest.approx(1.0)
    assert spearmanr(raw_bt, x[:, bt_col]).correlation == pytest.approx(1.0)

    # Tied raw pagerank (A0, A2 both 0.5) must map to the same normalized value.
    assert x[0, pr_col] == pytest.approx(x[2, pr_col])


def test_rank_normalize_excludes_type_specific_extras():
    g = _graph()
    sm = _structural_metrics()
    off = networkx_to_hetero_data(g, structural_metrics=sm, rank_normalize_features=False)
    on = networkx_to_hetero_data(g, structural_metrics=sm, rank_normalize_features=True)

    x_off = off.hetero_data["Application"].x.numpy()
    x_on = on.hetero_data["Application"].x.numpy()

    n_base = len(BASE_METRIC_KEYS)
    # loc_norm / complexity_norm (already-normalized CQ extras) sit past the
    # base columns and must be bit-identical whether or not the flag is set.
    np.testing.assert_array_equal(x_off[:, n_base:], x_on[:, n_base:])


def test_rank_normalize_off_by_default_is_bit_identical():
    g = _graph()
    sm = _structural_metrics()
    default = networkx_to_hetero_data(g, structural_metrics=sm)
    explicit_off = networkx_to_hetero_data(g, structural_metrics=sm, rank_normalize_features=False)

    np.testing.assert_array_equal(
        default.hetero_data["Application"].x.numpy(),
        explicit_off.hetero_data["Application"].x.numpy(),
    )
    # And explicit-off must match the raw structural_metrics values exactly —
    # the ablation arm this flag exists to preserve.
    x = default.hetero_data["Application"].x.numpy()
    pr_col = BASE_METRIC_KEYS.index("pagerank")
    assert x[:, pr_col] == pytest.approx([0.5, 0.1, 0.5, 0.3, 0.9])


def test_rank_normalize_single_node_type_guard():
    # n=1 Broker: rank/(n-1) is undefined: must not raise or emit NaN.
    g = _graph(n_apps=5, n_brokers=1)
    sm = _structural_metrics()
    sm["B0"] = {"pagerank": 0.7, "betweenness_centrality": 15.0}
    result = networkx_to_hetero_data(g, structural_metrics=sm, rank_normalize_features=True)
    x = result.hetero_data["Broker"].x.numpy()

    n_base = len(BASE_METRIC_KEYS)
    assert not np.any(np.isnan(x[:, :n_base]))
    assert np.all(x[:, :n_base] == 0.5)


def test_rank_normalize_base_columns_helper_directly():
    # Direct unit test of the helper: a column of all-equal values (a common
    # case in this codebase — several BASE_METRIC_KEYS are mostly zero) must
    # collapse to the neutral midpoint via average-rank ties, not NaN.
    feat = np.array([
        [1.0, 5.0],
        [1.0, 5.0],
        [1.0, 5.0],
    ], dtype=np.float32)
    _rank_normalize_base_columns(feat)
    assert np.all(feat == 0.5)
