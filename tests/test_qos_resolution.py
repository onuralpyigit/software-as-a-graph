"""
tests/test_qos_resolution.py — QoS reaches the impact path at all
=================================================================

These pin four defects that jointly made QoS inert in the ground-truth
simulation, each of which passed every pre-existing test:

1. Topic nodes reach consumers in two attribute shapes — flat
   (``qos_transport_priority``) and nested (``qos: {...}``). The simulation
   engines read only the flat keys, and the research loaders in ``cli/``
   produce only the nested one, so every topic silently scored at its default.
2. ``QoSPolicy.PRIORITY_SCORES`` omitted ``CRITICAL``/``HIGHEST``, so the top
   priority tier scored 0.0 — identical to ``LOW`` — while the repository
   scorers special-cased them to 1.0.
3. Topology JSON states QoS on Topic *nodes* only and never on edges, so every
   pub/sub edge carried ``qos_profile={}`` and ``weight=1.0``.
4. Consequently ``FaultInjector``'s I*(v) was numerically independent of QoS on
   the research path — the property test at the bottom of this file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from saag.core.models import (
    QoSPolicy,
    topic_weight_from_node_attrs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ATM = REPO_ROOT / "data" / "scenarios" / "atm_system.json"

MAX_QOS = {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}
MIN_QOS = {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}


# ── 1. Attribute-shape resolution ─────────────────────────────────────────────

def test_nested_and_flat_qos_shapes_resolve_identically():
    nested = {"qos": dict(MAX_QOS)}
    flat = {
        "qos_durability": "PERSISTENT",
        "qos_reliability": "RELIABLE",
        "qos_transport_priority": "CRITICAL",
    }
    assert QoSPolicy.from_node_attrs(nested) == QoSPolicy.from_node_attrs(flat)
    assert QoSPolicy.from_node_attrs(nested).calculate_weight() == pytest.approx(1.0)


def test_legacy_qos_priority_alias_is_honoured():
    """``qos_priority`` was the key the simulation graph read; keep it working."""
    assert QoSPolicy.from_node_attrs({"qos_priority": "HIGH"}).transport_priority == "HIGH"


def test_missing_qos_falls_back_to_documented_defaults():
    policy = QoSPolicy.from_node_attrs({})
    assert (policy.reliability, policy.durability, policy.transport_priority) == (
        "BEST_EFFORT",
        "VOLATILE",
        "MEDIUM",
    )


# ── 2. Priority table parity ──────────────────────────────────────────────────

@pytest.mark.parametrize("priority", ["LOW", "MEDIUM", "HIGH", "URGENT", "CRITICAL", "HIGHEST"])
def test_priority_scorers_agree_across_the_full_domain(priority):
    """QoSPolicy and MemoryRepository must score every priority value alike.

    The divergence between them is what allowed CRITICAL to rank as LOW.
    """
    from saag.infrastructure.memory_repo import MemoryRepository

    topic = {"id": "T0", "name": "t", "size": 256,
             "qos_reliability": "BEST_EFFORT",
             "qos_durability": "VOLATILE",
             "qos_transport_priority": priority}

    repo = MemoryRepository()
    repo.data = {
        "topics": [dict(topic)],
        "applications": [], "brokers": [], "nodes": [], "libraries": [],
        "relationships": {},
    }
    repo._calculate_intrinsic_weights()

    assert repo.data["topics"][0]["weight"] == pytest.approx(
        topic_weight_from_node_attrs(topic), abs=1e-9
    )


def test_critical_outranks_medium():
    """The regression itself: CRITICAL used to score below MEDIUM."""
    crit = QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="CRITICAL")
    med = QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="MEDIUM")
    assert crit.calculate_weight() > med.calculate_weight()


# ── 3. Edge-level QoS projection ──────────────────────────────────────────────

@pytest.mark.skipif(not ATM.exists(), reason="ATM scenario not present")
def test_pubsub_edges_inherit_topic_qos_and_weight():
    from cli.loso_evaluate import _build_graph_from_json

    graph = _build_graph_from_json(json.loads(ATM.read_text()))
    pubsub = [
        (u, v, d) for u, v, d in graph.edges(data=True)
        if (d.get("type") or "").upper() in ("PUBLISHES_TO", "SUBSCRIBES_TO", "ROUTES")
    ]
    assert pubsub, "scenario has no pub/sub edges"

    for _, topic, data in pubsub:
        expected = QoSPolicy.from_node_attrs(graph.nodes[topic])
        assert data["qos_profile"] == expected.to_dict()
        assert data["weight"] == pytest.approx(topic_weight_from_node_attrs(graph.nodes[topic]))

    # The defect was constancy, not absence: assert real spread across edges.
    assert len({d["weight"] for _, _, d in pubsub}) > 1
    assert len({d["qos_profile"]["reliability"] for _, _, d in pubsub}) > 1


@pytest.mark.skipif(not ATM.exists(), reason="ATM scenario not present")
def test_edge_qos_feature_dims_are_not_constant():
    """GNN edge dims 9-15 must vary; they were constant on every real scenario."""
    from cli.loso_evaluate import _build_graph_from_json
    from saag.prediction.data_preparation import networkx_to_hetero_data

    graph = _build_graph_from_json(json.loads(ATM.read_text()))
    data = networkx_to_hetero_data(graph, {}).hetero_data

    checked = 0
    for rel in data.edge_types:
        if rel[1] not in ("PUBLISHES_TO", "SUBSCRIBES_TO"):
            continue
        qos_block = data[rel].edge_attr[:, 9:16]
        assert float(qos_block.std(dim=0).sum()) > 0.0, f"{rel} QoS dims are constant"
        checked += 1
    assert checked, "no pub/sub relations found in HeteroData"


def test_explicit_edge_qos_is_not_overwritten():
    """A topology that does state edge QoS keeps it."""
    import networkx as nx
    from cli.loso_evaluate import _project_topic_qos_onto_edges

    stated = {"reliability": "BEST_EFFORT", "durability": "VOLATILE", "transport_priority": "LOW"}
    g = nx.DiGraph()
    g.add_node("A0", type="Application")
    g.add_node("T0", type="Topic", qos=dict(MAX_QOS), size=256)
    g.add_edge("A0", "T0", type="PUBLISHES_TO", qos_profile=dict(stated), weight=0.5)

    _project_topic_qos_onto_edges(g)

    assert g["A0"]["T0"]["qos_profile"] == stated
    assert g["A0"]["T0"]["weight"] == pytest.approx(0.5)


# ── 4. The property that motivates all of the above ───────────────────────────

@pytest.mark.skipif(not ATM.exists(), reason="ATM scenario not present")
@pytest.mark.parametrize("mode,should_respond", [("ladder", True), ("wt", True), ("none", False)])
def test_impact_score_responds_to_qos(mode, should_respond):
    """I*(v) must change when every topic's QoS changes — it previously did not.

    ``none`` is the topology-only ablation arm and must stay invariant, which is
    also what pins the pre-fix behaviour for comparison.
    """
    from cli.loso_evaluate import _build_graph_from_json
    from saag.simulation.fault_injector import FaultInjector

    topology = json.loads(ATM.read_text())

    def labels(topo):
        injector = FaultInjector(
            _build_graph_from_json(topo), seeds=[42], qos_factor_mode=mode
        )
        result = injector.run(node_types=["Application", "Broker", "Library"])
        return {k: v.impact_score for k, v in result.records.items()}

    maxed = copy.deepcopy(topology)
    for topic in maxed["topics"]:
        topic["qos"] = dict(MAX_QOS)

    base_scores, maxed_scores = labels(topology), labels(maxed)
    assert base_scores, "no nodes were labelled"

    shift = max(abs(base_scores[k] - maxed_scores[k]) for k in base_scores)
    if should_respond:
        assert shift > 1e-6, f"I*(v) is independent of QoS under mode={mode}"
    else:
        assert shift == pytest.approx(0.0, abs=1e-12)


def test_unknown_qos_factor_mode_is_rejected():
    import networkx as nx
    from saag.simulation.fault_injector import FaultInjector

    empty = nx.DiGraph()
    with pytest.raises(ValueError, match="qos_factor_mode"):
        FaultInjector(empty, qos_factor_mode="bogus")
