"""
Unit tests for RealWorldAdapter and real-world system topologies.
"""

import json
import pytest
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

