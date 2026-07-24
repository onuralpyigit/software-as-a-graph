"""
test_label_mask.py
──────────────────
Pins the distinction between "simulated, impact was zero" and "never simulated".

Both used to collapse to a 0.0 label, and the training/split code identified
labelled nodes with the proxy ``|y_composite| > 1e-6``. That excluded genuine
zero-impact observations from the loss while still scoring the model on them —
7 to 115 nodes per scenario, 37% on enterprise_system. The model was evaluated
on a low end of the ranking it never saw in training.
"""

import networkx as nx
import pytest

torch = pytest.importorskip("torch")

from saag.prediction.data_preparation import (  # noqa: E402
    _labelled_index_mask,
    create_kfold_masks,
    create_node_splits,
    extract_simulation_dict,
    networkx_to_hetero_data,
)
from saag.prediction.trainer import _labelled_nodes  # noqa: E402


def _graph(n_apps: int = 8) -> nx.DiGraph:
    g = nx.DiGraph()
    for i in range(n_apps):
        g.add_node(f"A{i}", component_type="Application", type="Application")
    g.add_node("T0", component_type="Topic", type="Topic")
    for i in range(n_apps):
        g.add_edge(f"A{i}", "T0", dependency_type="app_to_app",
                   type="PUBLISHES_TO", weight=1.0)
    return g


def _convert(scores: dict, n_apps: int = 8):
    """Build hetero data where `scores` maps node -> impact_score."""
    artifact = {"records": {nid: {"impact_score": v} for nid, v in scores.items()}}
    sim = extract_simulation_dict(artifact)
    return networkx_to_hetero_data(_graph(n_apps), structural_metrics={}, simulation_results=sim)


def test_genuine_zero_is_labelled_but_absent_node_is_not():
    """A simulated 0.0 is an observation; an unsimulated node is missing data."""
    result = _convert({"A0": 0.9, "A1": 0.4, "A2": 0.0, "A3": 0.0})
    store = result.hetero_data["Application"]

    mask = store.label_mask.tolist()
    assert mask[:4] == [True, True, True, True], "simulated nodes must be labelled, zero or not"
    assert mask[4:] == [False] * 4, "nodes absent from the artifact must not be labelled"

    # The old proxy is exactly what this guards against.
    proxy = (store.y[:, 0].abs() > 1e-6).tolist()
    assert proxy[2:4] == [False, False], "sanity: the proxy does drop genuine zeros"
    assert _labelled_nodes(store).tolist() == mask


def test_split_helpers_agree_with_the_mask():
    """Train/val/test and k-fold splits must partition on the same definition."""
    result = _convert({"A0": 0.9, "A1": 0.4, "A2": 0.0, "A3": 0.0})
    store = result.hetero_data["Application"]

    assert _labelled_index_mask(store).tolist() == store.label_mask.tolist()


def test_genuine_zeros_reach_the_training_split():
    """The regression that motivated this file: zeros must land in a split."""
    result = _convert({f"A{i}": (0.0 if i >= 2 else 0.5) for i in range(8)})
    data = result.hetero_data
    create_node_splits(data, train_ratio=0.6, val_ratio=0.2, seed=0)

    store = data["Application"]
    assigned = store.train_mask | store.val_mask | store.test_mask
    labelled = store.label_mask

    assert bool((labelled & assigned).sum() == labelled.sum()), (
        "every labelled node — including the six genuine zeros — must be assigned to a split"
    )
    assert int(labelled.sum()) == 8


def test_kfold_masks_use_the_label_mask():
    result = _convert({f"A{i}": (0.0 if i % 2 else 0.7) for i in range(8)})
    data = result.hetero_data
    create_kfold_masks(data, k=4, fold_idx=0, val_ratio=0.25, seed=0)

    store = data["Application"]
    assert int(store.label_mask.sum()) == 8
    assert int((store.train_mask | store.val_mask | store.test_mask).sum()) >= 8


def test_dimension_mask_reflects_what_the_labeler_measured():
    """FaultInjector measures 3 of 5 dimensions; the other 2 must be marked unmeasured."""
    result = _convert({"A0": 0.9, "A1": 0.4})
    # order: composite, reliability, maintainability, availability, security
    assert result.dimension_mask == [True, True, False, True, False]


def test_dimension_mask_is_all_true_for_a_full_rmav_labeler():
    """A labeler that emits all five dimensions must not be down-weighted."""
    sim = {
        "A0": {"composite": 0.9, "reliability": 0.8, "maintainability": 0.7,
               "availability": 0.6, "security": 0.5},
    }
    result = networkx_to_hetero_data(_graph(), structural_metrics={}, simulation_results=sim)
    assert result.dimension_mask == [True] * 5


def test_loss_ignores_unmeasured_dimensions():
    """Unmeasured columns must not contribute gradient to the multitask term."""
    from saag.prediction.models import CriticalityLoss

    loss_fn = CriticalityLoss()
    pred = torch.rand(6, 5)
    target = torch.rand(6, 5)
    target[:, 2] = 0.0   # maintainability — never measured
    target[:, 4] = 0.0   # security — never measured
    mask = torch.ones(6, dtype=torch.bool)

    _, unmasked = loss_fn(pred, target, mask)
    _, masked = loss_fn(pred, target, mask, None, torch.tensor([1., 1., 0., 1., 0.]))

    assert masked["multitask"] != pytest.approx(unmasked["multitask"]), (
        "dim_weights must change the multitask loss; otherwise unmeasured "
        "dimensions are still being regressed toward a fabricated zero"
    )


def test_missing_label_mask_falls_back_to_the_old_proxy():
    """Stores built before label_mask existed must still train."""
    result = _convert({"A0": 0.9, "A1": 0.0})
    store = result.hetero_data["Application"]
    del store.label_mask

    assert _labelled_nodes(store).tolist() == (store.y[:, 0].abs() > 1e-6).tolist()
    assert _labelled_index_mask(store).tolist() == (store.y[:, 0].abs() > 1e-6).tolist()
