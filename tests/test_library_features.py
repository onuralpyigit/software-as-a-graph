"""
tests/test_library_features.py
────────────────────────────────
Pins Stage 2's Library causal features: library_uses_reach_norm and
library_downstream_subs_norm, derived from the transitive reverse-USES
closure (USES runs consumer -> library, transitively — matching both
simulators' library-blast rule). Verifies closure correctness (including
transitivity and dedup across shared subscribers), per-graph normalization,
that the columns land at the expected feature-matrix indices, that Stage 1's
rank normalization leaves them untouched, and — the one real risk in this
change — that Application's feature width did not silently widen alongside
Library's (KEYS_BY_TYPE["Application"] and ["Library"] used to be the same
list object).
"""

import networkx as nx
import pytest

torch = pytest.importorskip("torch")

from saag.prediction.data_preparation import (  # noqa: E402
    KEYS_BY_TYPE,
    LIBRARY_EXTRA_KEYS,
    NODE_TYPE_TO_DIM,
    _normalize_infra_features,
    networkx_to_hetero_data,
)


def _uses_chain_graph() -> nx.DiGraph:
    """A -> L1 -> L2 (USES), plus an isolated library L3 with no consumers."""
    g = nx.DiGraph()
    g.add_node("A0", type="Application")
    g.add_node("L1", type="Library")
    g.add_node("L2", type="Library")
    g.add_node("L3", type="Library")
    g.add_edge("A0", "L1", type="USES")
    g.add_edge("L1", "L2", type="USES")
    return g


def test_transitive_closure_is_actually_transitive():
    """L2's reach must include both its direct consumer (L1) and L1's own
    consumer (A0) — a non-transitive walk would only see L1."""
    infra = _normalize_infra_features(_uses_chain_graph())
    # L2 has the largest closure (2: L1, A0) -> normalized to 1.0.
    # L1 has the next largest (1: A0).
    # L3 has none -> 0.0.
    assert infra["L2"]["library_uses_reach_norm"] == pytest.approx(1.0)
    assert infra["L1"]["library_uses_reach_norm"] == pytest.approx(0.5)
    assert infra["L3"]["library_uses_reach_norm"] == pytest.approx(0.0)


def test_downstream_subscribers_dedup_across_shared_topic():
    """Two closure members publishing to topics with an overlapping
    subscriber must not double-count that subscriber."""
    g = nx.DiGraph()
    g.add_node("A0", type="Application")
    g.add_node("A1", type="Application")
    g.add_node("L0", type="Library")
    g.add_node("T0", type="Topic")
    g.add_node("T1", type="Topic")
    g.add_node("Sub", type="Application")
    g.add_edge("A0", "L0", type="USES")
    g.add_edge("A1", "L0", type="USES")
    g.add_edge("A0", "T0", type="PUBLISHES_TO")
    g.add_edge("A1", "T1", type="PUBLISHES_TO")
    g.add_edge("Sub", "T0", type="SUBSCRIBES_TO")
    g.add_edge("Sub", "T1", type="SUBSCRIBES_TO")

    infra = _normalize_infra_features(g)
    # "Sub" subscribes to both T0 and T1, both reachable from L0's closure
    # (A0, A1) -- must count once, not twice. Only one library, so its own
    # count is also the per-graph max -> normalized to 1.0 either way; the
    # real assertion is downstream_subs_norm == 1.0 corresponding to a raw
    # count of 1, not silently absorbing a double-count that would still
    # normalize to 1.0 by coincidence. Cross-check via a second library with
    # a strictly smaller (but nonzero) reach to confirm relative scale.
    g.add_node("L1", type="Library")
    g.add_node("A2", type="Application")
    g.add_node("T2", type="Topic")
    g.add_edge("A2", "L1", type="USES")
    g.add_edge("A2", "T2", type="PUBLISHES_TO")
    g.add_edge("Sub", "T2", type="SUBSCRIBES_TO")
    infra = _normalize_infra_features(g)
    # Both L0 and L1 reach exactly one distinct subscriber ("Sub") once
    # dedup is correct -- so both must normalize to the same (max) value.
    assert infra["L0"]["library_downstream_subs_norm"] == pytest.approx(1.0)
    assert infra["L1"]["library_downstream_subs_norm"] == pytest.approx(1.0)


def test_columns_land_at_expected_indices_and_survive_rank_normalization():
    g = _uses_chain_graph()
    result = networkx_to_hetero_data(g, structural_metrics={}, rank_normalize_features=True)
    x = result.hetero_data["Library"].x.numpy()

    reach_col = KEYS_BY_TYPE["Library"].index("library_uses_reach_norm")
    downstream_col = KEYS_BY_TYPE["Library"].index("library_downstream_subs_norm")
    assert reach_col == 23
    assert downstream_col == 24
    assert x.shape[1] == NODE_TYPE_TO_DIM["Library"] == 25

    # Stage 1's rank normalization only touches the first len(BASE_METRIC_KEYS)
    # columns; these already-normalized tail columns must be untouched by it,
    # i.e. identical to the flag-off run.
    result_off = networkx_to_hetero_data(g, structural_metrics={}, rank_normalize_features=False)
    x_off = result_off.hetero_data["Library"].x.numpy()
    assert x[:, reach_col:downstream_col + 1] == pytest.approx(x_off[:, reach_col:downstream_col + 1])


def test_application_feature_width_unchanged():
    """Regression guard: KEYS_BY_TYPE["Application"] and ["Library"] used to
    be the *same list object* (both TOPOLOGICAL_METRIC_KEYS); appending
    Library's extras in place would have silently widened Application too."""
    assert KEYS_BY_TYPE["Application"] is not KEYS_BY_TYPE["Library"]
    assert len(KEYS_BY_TYPE["Application"]) == NODE_TYPE_TO_DIM["Application"] == 23
    assert "library_uses_reach_norm" not in KEYS_BY_TYPE["Application"]

    g = _uses_chain_graph()
    result = networkx_to_hetero_data(g, structural_metrics={})
    assert result.hetero_data["Application"].x.shape[1] == 23


def test_keys_by_type_and_node_type_to_dim_agree_for_library():
    assert len(KEYS_BY_TYPE["Library"]) == NODE_TYPE_TO_DIM["Library"] == 25
    assert set(LIBRARY_EXTRA_KEYS) <= set(KEYS_BY_TYPE["Library"])
