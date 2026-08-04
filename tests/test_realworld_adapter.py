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


def test_realworld_scenarios_loadable_by_graph_io():
    repo_root = Path(__file__).resolve().parents[1]
    json_path = repo_root / "data" / "scenarios" / "scenario_11_realworld_autoware_ros2.json"
    assert json_path.exists()
    G, raw = load_graph(json_path)
    assert G is not None
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0

    cloud_path = repo_root / "data" / "scenarios" / "scenario_12_realworld_cloud_microservices.json"
    assert cloud_path.exists()
    G_cloud, raw_cloud = load_graph(cloud_path)
    assert G_cloud is not None
    assert G_cloud.number_of_nodes() > 0
    assert G_cloud.number_of_edges() > 0
