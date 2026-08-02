"""
test_label_normalization.py
────────────────────────────
Pins the rank-normalization variant of `normalize_labels_robust` (Stage 3a):
regressing toward a purely rank-based target instead of an IQR+sigmoid
magnitude-preserving one, matching the fact that the reported headline
metric (Spearman rho) only scores rank. Verifies: default (IQR) path is
unchanged, rank path lands in [0,1] and is rank-preserving, zeros stay
zero, and — the one real risk in this change — labels for multiple node
types are scattered back to the correct rows, not swapped between types.
"""

import networkx as nx
import pytest

torch = pytest.importorskip("torch")

from scipy.stats import spearmanr  # noqa: E402

from saag.prediction.data_preparation import (  # noqa: E402
    extract_simulation_dict,
    networkx_to_hetero_data,
    normalize_labels_robust,
)


def _graph_two_types(n_apps: int = 6, n_brokers: int = 4) -> nx.DiGraph:
    g = nx.DiGraph()
    for i in range(n_apps):
        g.add_node(f"A{i}", component_type="Application", type="Application")
    for i in range(n_brokers):
        g.add_node(f"B{i}", component_type="Broker", type="Broker")
    g.add_node("T0", component_type="Topic", type="Topic")
    for i in range(n_apps):
        g.add_edge(f"A{i}", "T0", dependency_type="app_to_app",
                   type="PUBLISHES_TO", weight=1.0)
    for i in range(n_brokers):
        g.add_edge(f"B{i}", "T0", dependency_type="broker_to_topic",
                   type="ROUTES", weight=1.0)
    return g


def _convert(scores: dict, n_apps: int = 6, n_brokers: int = 4):
    artifact = {"records": {nid: {"impact_score": v} for nid, v in scores.items()}}
    sim = extract_simulation_dict(artifact)
    return networkx_to_hetero_data(
        _graph_two_types(n_apps, n_brokers), structural_metrics={}, simulation_results=sim
    )


def test_default_is_iqr_and_unchanged():
    scores = {"A0": 0.9, "A1": 0.4, "A2": 0.1, "A3": 0.7, "B0": 0.5, "B1": 0.2}
    r1 = _convert(scores)
    normalize_labels_robust(r1.hetero_data)
    r2 = _convert(scores)
    normalize_labels_robust(r2.hetero_data, rank_normalize=False)

    torch.testing.assert_close(r1.hetero_data["Application"].y, r2.hetero_data["Application"].y)
    torch.testing.assert_close(r1.hetero_data["Broker"].y, r2.hetero_data["Broker"].y)


def test_rank_normalize_range_order_and_zeros_preserved():
    scores = {"A0": 0.9, "A1": 0.4, "A2": 0.1, "A3": 0.0, "B0": 0.5, "B1": 0.2}
    result = _convert(scores)
    data = result.hetero_data
    normalize_labels_robust(data, rank_normalize=True)

    app_y = data["Application"].y[:, 0]
    broker_y = data["Broker"].y[:, 0]

    # A3's genuine zero-impact observation must stay exactly 0.0, not be
    # ranked in among the non-zero population.
    assert app_y[3].item() == pytest.approx(0.0)

    # The 3 genuinely-nonzero labelled entries must be strictly inside (0, 1):
    # the lowest rank must never land on exactly 0.0, or it would silently
    # read back as "unlabelled" under this module's `.abs() > 1e-6` convention.
    nonzero_labelled = torch.cat([app_y[:3], broker_y[:2]])
    assert torch.all(nonzero_labelled > 0.0) and torch.all(nonzero_labelled < 1.0)

    # Rank of the pooled non-zero population must exactly reproduce the raw order.
    raw = [0.9, 0.4, 0.1, 0.5, 0.2]  # A0, A1, A2, B0, B1 (A3 excluded: zero)
    normed = [app_y[0].item(), app_y[1].item(), app_y[2].item(), broker_y[0].item(), broker_y[1].item()]
    assert spearmanr(raw, normed).correlation == pytest.approx(1.0)


def test_rank_normalize_does_not_swap_node_types():
    """The one real risk: rows from one node type landing on another's after
    scattering the pooled ranks back out via `_pooled_nonzero_offset`."""
    # Applications get the four LOWEST raw scores, Brokers the two HIGHEST —
    # if the scatter mixed up offsets, an Application would end up with a
    # rank that belongs to a Broker (or vice versa).
    scores = {"A0": 0.05, "A1": 0.10, "A2": 0.15, "A3": 0.20, "B0": 0.90, "B1": 0.95}
    result = _convert(scores)
    data = result.hetero_data
    normalize_labels_robust(data, rank_normalize=True)

    app_y = data["Application"].y[:4, 0]
    broker_y = data["Broker"].y[:2, 0]

    assert torch.all(app_y < broker_y.min()), (
        f"Application ranks {app_y.tolist()} overlap Broker ranks {broker_y.tolist()} — "
        "pooled-offset scatter mixed node types"
    )


def test_rank_normalize_single_labelled_node_guard():
    result = _convert({"A0": 0.5})
    data = result.hetero_data
    normalize_labels_robust(data, rank_normalize=True)
    assert data["Application"].y[0, 0].item() == pytest.approx(0.5)
