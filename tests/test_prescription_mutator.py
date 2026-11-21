"""
tests/test_prescription_mutator.py

Unit tests for the JSON graph rewriter. Pure — no repository, no simulation.
"""
import copy

import pytest

from saag.prescription.models import (
    NodeReallocation,
    PrescriptionPolicy,
    QosUpgrade,
    TopicSplit,
)
from saag.prescription.mutator import apply_policy


@pytest.fixture
def topology():
    """A topology exercising every relationship type the mutator rewrites."""
    return {
        "applications": [
            {"id": "AppA", "name": "App A"},
            {"id": "AppB", "name": "App B"},
        ],
        "brokers": [{"id": "BrokerMain", "name": "Broker Main"}],
        "nodes": [
            {"id": "NodeMain", "name": "Node Main", "cpu": 8},
            {"id": "NodeOther", "name": "Node Other", "cpu": 4},
        ],
        "topics": [
            {"id": "T1", "name": "Topic T1",
             "qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE"}},
            {"id": "T2", "name": "Topic T2",
             "qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE"}},
        ],
        "relationships": {
            "publishes_to": [
                {"from": "AppA", "to": "T1", "weight": 0.8},
                {"from": "AppB", "to": "T1", "weight": 0.5},
                {"from": "AppA", "to": "T2", "weight": 0.3},
            ],
            "subscribes_to": [
                {"from": "AppB", "to": "T1", "weight": 0.7},
                {"from": "AppB", "to": "T2", "weight": 0.2},
            ],
            "routes": [
                {"from": "BrokerMain", "to": "T1", "weight": 1.0},
                {"from": "BrokerMain", "to": "T2", "weight": 1.0},
            ],
            "runs_on": [
                {"from": "AppA", "to": "NodeMain", "weight": 1.0},
                {"from": "AppB", "to": "NodeMain", "weight": 1.0},
                {"from": "BrokerMain", "to": "NodeOther", "weight": 1.0},
            ],
            "connects_to": [
                {"from": "NodeMain", "to": "NodeOther", "weight": 1.0},
                {"from": "NodeOther", "to": "NodeMain", "weight": 1.0},
            ],
            "depends_on": [{"from": "AppB", "to": "AppA", "weight": 0.9}],
            "uses": [{"from": "AppA", "to": "LibX", "weight": 1.0}],
        },
    }


def _by_id(items):
    return {item["id"]: item for item in items}


def test_apply_policy_qos_upgrade_writes_nested_and_flat(topology):
    policy = PrescriptionPolicy(qos_upgrades=[
        QosUpgrade(topic="T1", original_reliability="BEST_EFFORT", original_durability="VOLATILE")
    ])

    topics = _by_id(apply_policy(topology, policy)["topics"])

    assert topics["T1"]["qos"] == {"reliability": "RELIABLE", "durability": "TRANSIENT"}
    # The layer projections read the flat properties, not the nested block.
    assert topics["T1"]["qos_reliability"] == "RELIABLE"
    assert topics["T1"]["qos_durability"] == "TRANSIENT"
    # Untargeted topics are left alone.
    assert topics["T2"]["qos"] == {"reliability": "BEST_EFFORT", "durability": "VOLATILE"}
    assert "qos_reliability" not in topics["T2"]


def test_apply_policy_reallocation_creates_node_and_rewires_runs_on(topology):
    policy = PrescriptionPolicy(node_reallocations=[
        NodeReallocation(component="AppB", from_node="NodeMain", to_node="NodeMain_AppB")
    ])

    mutated = apply_policy(topology, policy)
    nodes = _by_id(mutated["nodes"])
    runs_on = {r["from"]: r["to"] for r in mutated["relationships"]["runs_on"]}

    assert runs_on["AppB"] == "NodeMain_AppB"
    # Everything else stays where it was.
    assert runs_on["AppA"] == "NodeMain"
    assert runs_on["BrokerMain"] == "NodeOther"

    assert "NodeMain_AppB" in nodes
    assert nodes["NodeMain_AppB"]["name"] == "Node NodeMain_AppB"
    # The clone inherits the original host's properties.
    assert nodes["NodeMain_AppB"]["cpu"] == nodes["NodeMain"]["cpu"]


def test_apply_policy_reallocation_duplicates_connects_to_in_both_directions(topology):
    policy = PrescriptionPolicy(node_reallocations=[
        NodeReallocation(component="AppB", from_node="NodeMain", to_node="NodeMain_AppB")
    ])

    connects = {
        (c["from"], c["to"]) for c in apply_policy(topology, policy)["relationships"]["connects_to"]
    }

    # Originals survive...
    assert ("NodeMain", "NodeOther") in connects
    assert ("NodeOther", "NodeMain") in connects
    # ...and the clone inherits the host's links in both directions, so the
    # relocated process keeps the network reachability it had.
    assert ("NodeMain_AppB", "NodeOther") in connects
    assert ("NodeOther", "NodeMain_AppB") in connects


def test_reallocation_clone_keeps_link_capacity_but_not_interfaces(topology):
    """A new cable has the same capacity as the one it mirrors, on ports of its own."""
    import copy
    topology = copy.deepcopy(topology)
    for conn in topology["relationships"]["connects_to"]:
        conn.update({"capacity_mbps": 1000, "from_interface": "eth0", "to_interface": "port3"})
    policy = PrescriptionPolicy(node_reallocations=[
        NodeReallocation(component="AppB", from_node="NodeMain", to_node="NodeMain_AppB")
    ])
    clone_links = [c for c in apply_policy(topology, policy)["relationships"]["connects_to"]
                   if "NodeMain_AppB" in (c["from"], c["to"])]
    assert clone_links
    for link in clone_links:
        assert link["capacity_mbps"] == 1000
        assert "from_interface" not in link and "to_interface" not in link


def test_apply_policy_topic_split_rewrites_all_three_relationship_types(topology):
    policy = PrescriptionPolicy(topic_splits=[
        TopicSplit(topic="T1", publishers=["AppA", "AppB"], subscribers=["AppB"])
    ])

    mutated = apply_policy(topology, policy)
    rels = mutated["relationships"]
    topics = _by_id(mutated["topics"])

    # The original topic is replaced by one sub-topic per publisher.
    assert set(topics) == {"T1_AppA", "T1_AppB", "T2"}
    assert topics["T1_AppA"]["name"] == "Topic T1 for AppA"

    # A publisher goes to its own sub-topic only.
    assert {(r["from"], r["to"]) for r in rels["publishes_to"]} == {
        ("AppA", "T1_AppA"), ("AppB", "T1_AppB"), ("AppA", "T2"),
    }
    # Subscribers and brokers fan out to every sub-topic, so no channel is lost.
    assert {(r["from"], r["to"]) for r in rels["subscribes_to"]} == {
        ("AppB", "T1_AppA"), ("AppB", "T1_AppB"), ("AppB", "T2"),
    }
    assert {(r["from"], r["to"]) for r in rels["routes"]} == {
        ("BrokerMain", "T1_AppA"), ("BrokerMain", "T1_AppB"), ("BrokerMain", "T2"),
    }

    # Weights survive the fan-out.
    assert all(r["weight"] == 0.7 for r in rels["subscribes_to"] if r["to"].startswith("T1_"))


def test_apply_policy_split_publisher_edge_not_in_publishers_list(topology):
    """A publisher edge is retargeted by its own source, not by the split's list.

    The compiled publisher list comes from the same edges, so the two agree in
    practice; pinning it keeps the publish path independent of that coincidence.
    """
    policy = PrescriptionPolicy(topic_splits=[
        TopicSplit(topic="T1", publishers=["AppA"], subscribers=["AppB"])
    ])

    publishes = {
        (r["from"], r["to"]) for r in apply_policy(topology, policy)["relationships"]["publishes_to"]
    }

    assert ("AppB", "T1_AppB") in publishes


def test_apply_policy_does_not_mutate_input(topology):
    before = copy.deepcopy(topology)
    policy = PrescriptionPolicy(
        topic_splits=[TopicSplit(topic="T1", publishers=["AppA"], subscribers=["AppB"])],
        node_reallocations=[
            NodeReallocation(component="AppB", from_node="NodeMain", to_node="NodeMain_AppB")],
        qos_upgrades=[
            QosUpgrade(topic="T2", original_reliability="BEST_EFFORT", original_durability="VOLATILE")],
    )

    apply_policy(topology, policy)

    assert topology == before


def test_apply_policy_empty_policy_is_identity(topology):
    assert apply_policy(topology, PrescriptionPolicy()) == topology


def test_apply_policy_leaves_uses_and_depends_on_untouched(topology):
    policy = PrescriptionPolicy(topic_splits=[
        TopicSplit(topic="T1", publishers=["AppA", "AppB"], subscribers=["AppB"])
    ])

    rels = apply_policy(topology, policy)["relationships"]

    assert rels["uses"] == topology["relationships"]["uses"]
    # Derived edges are dropped and recomputed on re-import, so the mutator has
    # no business rewriting them.
    assert rels["depends_on"] == topology["relationships"]["depends_on"]


def test_apply_policy_hardens_before_splitting(topology):
    """Sub-topics inherit the hardened contract of the topic they replace."""
    policy = PrescriptionPolicy(
        topic_splits=[TopicSplit(topic="T1", publishers=["AppA"], subscribers=["AppB"])],
        qos_upgrades=[
            QosUpgrade(topic="T1", original_reliability="BEST_EFFORT", original_durability="VOLATILE")],
    )

    topics = _by_id(apply_policy(topology, policy)["topics"])

    assert topics["T1_AppA"]["qos_reliability"] == "RELIABLE"
    assert topics["T1_AppA"]["qos"]["durability"] == "TRANSIENT"


def test_reallocation_connects_source_host_to_clone_and_siblings():
    """Processes split off the same host must retain network reachability to each other."""
    graph = {
        "nodes": [{"id": "N1"}, {"id": "N2"}],
        "applications": [{"id": "AppA"}, {"id": "AppB"}, {"id": "AppC"}],
        "relationships": {
            "runs_on": [
                {"from": "AppA", "to": "N1"},
                {"from": "AppB", "to": "N1"},
                {"from": "AppC", "to": "N1"},
            ],
            "connects_to": [{"from": "N1", "to": "N2", "weight": 1.0}],
        },
    }
    policy = PrescriptionPolicy(node_reallocations=[
        NodeReallocation("AppB", "N1", "N1_AppB"),
        NodeReallocation("AppC", "N1", "N1_AppC"),
    ])

    connects = {
        (c["from"], c["to"]) for c in apply_policy(graph, policy)["relationships"]["connects_to"]
    }

    # Source <-> clone and clone <-> clone links are all present bidirectionally.
    assert ("N1", "N1_AppB") in connects
    assert ("N1_AppB", "N1") in connects
    assert ("N1", "N1_AppC") in connects
    assert ("N1_AppC", "N1") in connects
    assert ("N1_AppB", "N1_AppC") in connects
    assert ("N1_AppC", "N1_AppB") in connects


def test_reallocation_adds_cross_clone_links_when_both_endpoints_reallocated():
    """If N1 -> N2 exists and both N1 and N2 are cloned, N1_clone -> N2_clone must exist."""
    graph = {
        "nodes": [{"id": "N1"}, {"id": "N2"}],
        "applications": [{"id": "AppA"}, {"id": "AppB"}, {"id": "AppC"}, {"id": "AppD"}],
        "relationships": {
            "runs_on": [
                {"from": "AppA", "to": "N1"},
                {"from": "AppB", "to": "N1"},
                {"from": "AppC", "to": "N2"},
                {"from": "AppD", "to": "N2"},
            ],
            "connects_to": [{"from": "N1", "to": "N2", "weight": 0.9, "capacity_mbps": 500}],
        },
    }
    policy = PrescriptionPolicy(node_reallocations=[
        NodeReallocation("AppB", "N1", "N1_AppB"),
        NodeReallocation("AppD", "N2", "N2_AppD"),
    ])

    connects = apply_policy(graph, policy)["relationships"]["connects_to"]
    pairs = {(c["from"], c["to"]): c for c in connects}

    assert ("N1_AppB", "N2_AppD") in pairs
    assert pairs[("N1_AppB", "N2_AppD")]["capacity_mbps"] == 500
    assert pairs[("N1_AppB", "N2_AppD")]["weight"] == 0.9
