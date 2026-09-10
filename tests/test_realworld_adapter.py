"""
Unit tests for RealWorldAdapter and real-world system topologies.
"""

import json
import pytest
from collections import Counter
from pathlib import Path
from saag.adapters.realworld_adapter import RealWorldAdapter
from cli.validation.graph_io import load_graph


def test_autoware_ros2_topology_structure():
    data = RealWorldAdapter.create_autoware_ros2_topology()
    assert "metadata" in data
    assert data["metadata"]["domain"] == "autoware_ros2"
    assert len(data["nodes"]) == 6
    assert len(data["brokers"]) == 3
    assert len(data["topics"]) == 24
    assert len(data["applications"]) == 32
    assert len(data["libraries"]) == 10
    assert len(data["relationships"]) > 0


def test_cloud_microservices_topology_structure():
    data = RealWorldAdapter.create_cloud_microservices_topology()
    assert "metadata" in data
    assert data["metadata"]["domain"] == "cloud_microservices"
    assert len(data["nodes"]) == 6
    assert len(data["brokers"]) == 4
    assert len(data["topics"]) == 20
    assert len(data["applications"]) == 22
    assert len(data["libraries"]) == 8
    assert len(data["relationships"]) > 0


def test_trainticket_microservices_topology_structure():
    data = RealWorldAdapter.create_trainticket_microservices_topology()
    assert "metadata" in data
    assert data["metadata"]["domain"] == "trainticket_microservices"
    assert len(data["nodes"]) == 8
    assert len(data["brokers"]) == 3
    assert len(data["topics"]) == 30
    assert len(data["applications"]) == 41
    assert len(data["libraries"]) == 8
    assert len(data["relationships"]) > 0


def test_homeassistant_topology_structure():
    data = RealWorldAdapter.create_homeassistant_topology()
    assert "metadata" in data
    assert data["metadata"]["domain"] == "homeassistant_iot"
    assert len(data["nodes"]) == 6
    assert len(data["brokers"]) == 3
    assert len(data["topics"]) == 22
    assert len(data["applications"]) == 24
    assert len(data["libraries"]) == 8
    assert len(data["relationships"]) > 0


def test_edgex_foundry_topology_structure():
    data = RealWorldAdapter.create_edgex_foundry_topology()
    assert "metadata" in data
    assert data["metadata"]["domain"] == "edgex_foundry_iiot"
    assert len(data["nodes"]) == 6
    assert len(data["brokers"]) == 3
    assert len(data["topics"]) == 24
    assert len(data["applications"]) == 22
    assert len(data["libraries"]) == 8
    assert len(data["relationships"]) > 0


def test_realworld_scenarios_loadable_by_graph_io():
    repo_root = Path(__file__).resolve().parents[1]
    scenario_files = [
        ("realworld_autoware_ros2.json", "scenario_11_realworld_autoware_ros2.json"),
        ("realworld_cloud_microservices.json", "scenario_12_realworld_cloud_microservices.json"),
        ("realworld_trainticket.json", "scenario_13_realworld_trainticket.json"),
        ("realworld_homeassistant.json", "scenario_14_realworld_homeassistant.json"),
        ("realworld_edgex.json", "scenario_15_realworld_edgex.json"),
    ]

    for alias, numbered in scenario_files:
        path = repo_root / "data" / "scenarios" / alias
        if not path.exists():
            path = repo_root / "data" / "scenarios" / numbered
        assert path.exists(), f"Missing scenario dataset at {path}"
        G, raw = load_graph(path)
        assert G is not None
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        assert raw["metadata"]["generation_mode"] == "realworld_open_source"


def test_realworld_applications_have_criticality_and_hotstandby_and_no_priority():
    creators = [
        RealWorldAdapter.create_autoware_ros2_topology,
        RealWorldAdapter.create_cloud_microservices_topology,
        RealWorldAdapter.create_trainticket_microservices_topology,
        RealWorldAdapter.create_homeassistant_topology,
        RealWorldAdapter.create_edgex_foundry_topology,
    ]
    for creator in creators:
        data = creator()
        for app in data["applications"]:
            assert "criticality" in app, f"App {app['id']} missing criticality"
            assert app["criticality"] in {"HIGH", "MEDIUM", "LOW"}
            assert "hotstandby" in app, f"App {app['id']} missing hotstandby"
            assert isinstance(app["hotstandby"], bool)
            assert "priority" not in app, f"App {app['id']} should not have priority"
            if app["criticality"] == "HIGH":
                assert app["hotstandby"] is True
            else:
                assert app["hotstandby"] is False



_TOPOLOGY_CREATORS = [
    RealWorldAdapter.create_autoware_ros2_topology,
    RealWorldAdapter.create_cloud_microservices_topology,
    RealWorldAdapter.create_trainticket_microservices_topology,
    RealWorldAdapter.create_homeassistant_topology,
    RealWorldAdapter.create_edgex_foundry_topology,
]


@pytest.mark.parametrize("creator", _TOPOLOGY_CREATORS)
def test_realworld_topics_declare_the_temporal_contract(creator):
    """Every real-world topic must carry the same QoS fields as the synthetic corpus.

    The QoS edge encoder turns ``deadline_ms`` into the ``has_deadline`` and
    ``deadline_log`` dimensions.  When these fixtures carried neither field,
    those dims were live in training and constant-zero at test — a train/test
    shift on exactly the zero-shot transfer path.  ``history_depth`` is always
    declared; ``deadline_ms`` may legitimately be null where the real system
    declares no temporal contract, but must be present on some topics.
    """
    topics = creator()["topics"]
    for t in topics:
        assert "history_depth" in t, f"Topic {t['id']} missing history_depth"
        assert isinstance(t["history_depth"], int) and t["history_depth"] >= 1
        assert "deadline_ms" in t, f"Topic {t['id']} missing deadline_ms"
        assert t["deadline_ms"] is None or t["deadline_ms"] > 0

    with_deadline = [t for t in topics if t["deadline_ms"] is not None]
    assert with_deadline, "no topic in this fixture declares a deadline at all"


@pytest.mark.parametrize("creator", _TOPOLOGY_CREATORS)
def test_realworld_topic_criticality_is_not_near_constant(creator):
    """A fixture whose topics are ~all one tier carries no criticality signal.

    Folding the retired 5-level scale onto 3 tiers mechanically (critical+high
    -> HIGH) left Autoware at 22 HIGH out of 24, making the feature useless for
    the zero-shot evaluation.  Labels must reflect *relative* urgency within
    each system.
    """
    topics = creator()["topics"]
    counts = Counter(t["criticality"] for t in topics)
    assert set(counts) <= {"LOW", "MEDIUM", "HIGH"}
    assert len(counts) == 3, f"only {sorted(counts)} present — no spread"
    dominant = max(counts.values()) / len(topics)
    assert dominant <= 0.80, (
        f"{dominant:.0%} of topics are {counts.most_common(1)[0][0]}; "
        "the criticality feature is effectively constant for this fixture"
    )
