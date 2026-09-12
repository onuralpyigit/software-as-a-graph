"""
Edge criticality labels must come from a measurement, not from a proxy.

``networkx_to_hetero_data`` used to synthesise ``y_edge`` as
``I*(source) x {1.0 if bridge else 0.1}`` — a structural heuristic standing in
for the edge-removal oracle the paper declares. The edge head was therefore
scored against a hand-chosen multiplier. These tests pin the replacement:
labels come from ``FailureSimulator.simulate_edge_removal`` when supplied, and
no labels are written at all when they are not.
"""

import networkx as nx
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from saag.prediction.data_preparation import (  # noqa: E402
    extract_edge_simulation_dict,
    networkx_to_hetero_data,
)


def _graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for n in ("A1", "A2", "A3"):
        g.add_node(n, type="Application")
    g.add_node("T1", type="Topic")
    g.add_edge("A1", "T1", type="PUBLISHES_TO", weight=1.0)
    g.add_edge("A2", "T1", type="SUBSCRIBES_TO", weight=1.0)
    g.add_edge("A3", "T1", type="SUBSCRIBES_TO", weight=1.0)
    return g


_SIM = {n: {"composite": 0.5, "reliability": 0.5} for n in ("A1", "A2", "A3")}


def test_no_edge_labels_without_a_measurement():
    """Absent a sweep, L_edge goes inactive rather than training on a proxy."""
    conv = networkx_to_hetero_data(_graph(), None, _SIM, None)
    assert conv.num_labelled_edges == 0
    for rel in conv.hetero_data.edge_types:
        assert not hasattr(conv.hetero_data[rel], "y_edge")


def test_measured_edges_become_labels():
    edge_sim = {("A1", "T1"): 0.42}
    conv = networkx_to_hetero_data(
        _graph(), None, _SIM, None, edge_simulation_results=edge_sim
    )
    assert conv.num_labelled_edges == 1

    rel = ("Application", "PUBLISHES_TO", "Topic")
    store = conv.hetero_data[rel]
    assert store.y_edge[0, 0] == pytest.approx(0.42)
    assert bool(store.y_edge_mask[0])


def test_unmeasured_edges_are_masked_not_zeroed():
    """A1->T1 measured, the two SUBSCRIBES_TO edges not. The unmeasured
    relation must carry no labels at all, so the edge head is never taught that
    an unevaluated edge is a harmless one."""
    conv = networkx_to_hetero_data(
        _graph(), None, _SIM, None, edge_simulation_results={("A1", "T1"): 0.42}
    )
    sub_rel = ("Application", "SUBSCRIBES_TO", "Topic")
    assert not hasattr(conv.hetero_data[sub_rel], "y_edge")


def test_partially_measured_relation_masks_the_gap():
    conv = networkx_to_hetero_data(
        _graph(), None, _SIM, None,
        edge_simulation_results={("A2", "T1"): 0.7},
    )
    store = conv.hetero_data[("Application", "SUBSCRIBES_TO", "Topic")]
    assert store.y_edge_mask.sum().item() == 1
    assert conv.num_labelled_edges == 1


def test_extract_drops_unevaluated_candidates():
    """`evaluated=False` means the sweep never severed the edge. Reading it as
    0.0 would relabel 'unmeasured' as 'measured harmless'."""
    payload = {
        "edge_criticality": [
            {"source": "A1", "target": "T1", "combined_impact": 0.3, "evaluated": True},
            {"source": "A2", "target": "T1", "combined_impact": 0.0, "evaluated": False},
        ]
    }
    assert extract_edge_simulation_dict(payload) == {("A1", "T1"): 0.3}


def test_edge_loss_ignores_masked_rows():
    from saag.prediction.trainer import GNNTrainer

    conv = networkx_to_hetero_data(
        _graph(), None, _SIM, None,
        edge_simulation_results={("A2", "T1"): 1.0},
    )
    data = conv.hetero_data
    rel = ("Application", "SUBSCRIBES_TO", "Topic")
    n_edges = data[rel].edge_index.shape[1]
    mask = data[rel].y_edge_mask

    # Prediction is exact on the measured row and wrong on the masked one; the
    # loss must see only the former.
    preds = torch.zeros((n_edges, 1))
    preds[mask, 0] = 1.0

    trainer = GNNTrainer.__new__(GNNTrainer)
    trainer.device = torch.device("cpu")
    assert trainer._edge_loss({rel: preds}, data).item() == pytest.approx(0.0)
