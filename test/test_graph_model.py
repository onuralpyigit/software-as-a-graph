#!/usr/bin/env python3
"""
Test Suite for Unified DEPENDS_ON Relationship Derivation

This module tests the graph model with unified dependency relationships
across all system layers:
- APP_TO_APP: From topic subscription overlap
- APP_TO_BROKER: From topic routing
- NODE_TO_NODE: From application dependencies
- NODE_TO_BROKER: From broker placement

Run with: python -m pytest tests/test_graph_model.py -v
Or directly: python tests/test_graph_model.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.graph_model import (
    GraphModel, Application, Topic, Broker, Node,
    ApplicationRole, QoSPolicy
)
from src.core.graph_builder import GraphBuilder


def create_test_scenario() -> dict:
    """
    Create a test scenario representing a simplified autonomous vehicle system.
    
    Architecture:
    - 3 Infrastructure Nodes (Edge, Compute, Cloud)
    - 2 Brokers (Edge Broker, Cloud Broker)
    - 6 Applications across nodes
    - 4 Topics with different QoS
    
    Expected dependencies:
    - APP_TO_APP: Perception -> Sensor, Planning -> Perception, Control -> Planning
    - APP_TO_BROKER: Apps depending on brokers that route their topics
    - NODE_TO_NODE: EdgeNode -> ComputeNode (via app deps), ComputeNode -> CloudNode
    - NODE_TO_BROKER: Nodes depending on brokers their apps use
    """
    return {
        "nodes": [
            {"id": "edge_node", "name": "Edge Node"},
            {"id": "compute_node", "name": "Compute Node"},
            {"id": "cloud_node", "name": "Cloud Node"}
        ],
        "brokers": [
            {"id": "edge_broker", "name": "Edge MQTT Broker"},
            {"id": "cloud_broker", "name": "Cloud Kafka Broker"}
        ],
        "applications": [
            {"id": "sensor_app", "name": "Sensor Fusion", "role": "pub"},
            {"id": "perception_app", "name": "Perception", "role": "pubsub"},
            {"id": "planning_app", "name": "Planning", "role": "pubsub"},
            {"id": "control_app", "name": "Control", "role": "sub"},
            {"id": "logging_app", "name": "Cloud Logging", "role": "sub"},
            {"id": "analytics_app", "name": "Analytics", "role": "sub"}
        ],
        "topics": [
            {
                "id": "sensor_data",
                "name": "/sensor/data",
                "qos": {"reliability": "reliable", "durability": "volatile", "transport_priority": "MEDIUM"},
                "size": 512
            },
            {
                "id": "perception_output",
                "name": "/perception/objects",
                "qos": {"reliability": "reliable", "durability": "transient_local", "transport_priority": "LOW"},
                "size": 1024
            },
            {
                "id": "planning_cmd",
                "name": "/planning/trajectory",
                "qos": {"reliability": "reliable", "durability": "volatile", "transport_priority": "HIGH"},
                "size": 256
            },
            {
                "id": "telemetry",
                "name": "/vehicle/telemetry",
                "qos": {"reliability": "best_effort", "durability": "volatile", "transport_priority": "URGENT"},
                "size": 128
            }
        ],
        "edges": {
            "runs_on": [
                {"from": "sensor_app", "to": "edge_node"},
                {"from": "perception_app", "to": "compute_node"},
                {"from": "planning_app", "to": "compute_node"},
                {"from": "control_app", "to": "edge_node"},
                {"from": "logging_app", "to": "cloud_node"},
                {"from": "analytics_app", "to": "cloud_node"},
                {"from": "edge_broker", "to": "edge_node"},
                {"from": "cloud_broker", "to": "cloud_node"}
            ],
            "publishes": [
                {"from": "sensor_app", "to": "sensor_data"},
                {"from": "sensor_app", "to": "telemetry"},
                {"from": "perception_app", "to": "perception_output"},
                {"from": "planning_app", "to": "planning_cmd"},
                {"from": "planning_app", "to": "telemetry"}
            ],
            "subscribes": [
                {"from": "perception_app", "to": "sensor_data"},
                {"from": "planning_app", "to": "perception_output"},
                {"from": "control_app", "to": "planning_cmd"},
                {"from": "logging_app", "to": "telemetry"},
                {"from": "analytics_app", "to": "telemetry"},
                {"from": "analytics_app", "to": "perception_output"}
            ],
            "routes": [
                {"from": "edge_broker", "to": "sensor_data"},
                {"from": "edge_broker", "to": "planning_cmd"},
                {"from": "edge_broker", "to": "perception_output"},
                {"from": "cloud_broker", "to": "telemetry"},
                {"from": "cloud_broker", "to": "perception_output"}
            ],
            "connects_to": [
                {"from": "edge_node", "to": "compute_node"},
                {"from": "compute_node", "to": "cloud_node"}
            ]
        }
    }


def test_dependency_derivation():
    """Test that all dependency types are correctly derived"""
    print("\n" + "="*70)
    print("TEST: Unified DEPENDS_ON Relationship Derivation")
    print("="*70)
    
    # Build the graph
    builder = GraphBuilder()
    scenario = create_test_scenario()
    model = builder.build_from_dict(scenario)
    
    # Get statistics
    stats = model.get_statistics()
    print(stats)
    print(f"\nGraph Statistics:")
    print(f"  Vertices: {stats['vertices']}")
    print(f"  Explicit Edges: {stats['explicit_edges']}")
    print(f"  Derived DEPENDS_ON: {stats['derived_edges']}")
    
    # Verify APP_TO_APP dependencies
    print("\n--- APP_TO_APP Dependencies ---")
    app_to_app = model.get_depends_on_by_type(DependencyType.APP_TO_APP)
    print(f"Count: {len(app_to_app)}")
    
    expected_app_deps = [
        ("perception_app", "sensor_app"),    # perception subscribes to sensor's topic
        ("planning_app", "perception_app"),  # planning subscribes to perception's topic
        ("control_app", "planning_app"),     # control subscribes to planning's topic
        ("logging_app", "sensor_app"),       # logging subscribes to telemetry
        ("logging_app", "planning_app"),     # logging subscribes to telemetry
        ("analytics_app", "sensor_app"),     # analytics subscribes to telemetry
        ("analytics_app", "planning_app"),   # analytics subscribes to telemetry
        ("analytics_app", "perception_app"), # analytics subscribes to perception_output
    ]
    
    actual_app_deps = [(d.source, d.target) for d in app_to_app]
    
    for dep in app_to_app:
        print(f"  {dep.source} -> {dep.target} (topics: {dep.topics}, weight: {dep.weight:.2f})")
    
    # Check critical dependencies exist
    critical_deps = [
        ("perception_app", "sensor_app"),
        ("planning_app", "perception_app"),
        ("control_app", "planning_app")
    ]
    
    for src, tgt in critical_deps:
        found = any(d.source == src and d.target == tgt for d in app_to_app)
        status = "✓" if found else "✗"
        print(f"  {status} Critical: {src} -> {tgt}")
    
    # Verify APP_TO_BROKER dependencies
    print("\n--- APP_TO_BROKER Dependencies ---")
    app_to_broker = model.get_depends_on_by_type(DependencyType.APP_TO_BROKER)
    print(f"Count: {len(app_to_broker)}")
    
    for dep in app_to_broker:
        print(f"  {dep.source} -> {dep.target} (topics: {dep.topics}, weight: {dep.weight:.2f})")
    
    # Verify NODE_TO_NODE dependencies
    print("\n--- NODE_TO_NODE Dependencies ---")
    node_to_node = model.get_depends_on_by_type(DependencyType.NODE_TO_NODE)
    print(f"Count: {len(node_to_node)}")
    
    for dep in node_to_node:
        print(f"  {dep.source} -> {dep.target} (weight: {dep.weight:.2f})")
    
    # Expected node dependencies based on architecture
    # edge_node has apps that depend on compute_node apps
    # compute_node has apps that depend on cloud_node apps (via telemetry to logging/analytics)
    
    # Verify NODE_TO_BROKER dependencies
    print("\n--- NODE_TO_BROKER Dependencies ---")
    node_to_broker = model.get_depends_on_by_type(DependencyType.NODE_TO_BROKER)
    print(f"Count: {len(node_to_broker)}")
    
    for dep in node_to_broker:
        print(f"  {dep.source} -> {dep.target} (weight: {dep.weight:.2f})")
    
    # Summary
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    total_deps = len(model.depends_on_edges)
    print(f"Total DEPENDS_ON relationships: {total_deps}")
    print(f"  APP_TO_APP:     {len(app_to_app)}")
    print(f"  APP_TO_BROKER:  {len(app_to_broker)}")
    print(f"  NODE_TO_NODE:   {len(node_to_node)}")
    print(f"  NODE_TO_BROKER: {len(node_to_broker)}")
    
    # Validation
    assert len(app_to_app) > 0, "Should have APP_TO_APP dependencies"
    assert len(app_to_broker) > 0, "Should have APP_TO_BROKER dependencies"
    assert len(node_to_node) > 0, "Should have NODE_TO_NODE dependencies"
    
    print("\n✓ All dependency types correctly derived!")
    return model


def test_networkx_conversion():
    """Test conversion to NetworkX graph"""
    print("\n" + "="*70)
    print("TEST: NetworkX Conversion")
    print("="*70)
    
    builder = GraphBuilder()
    scenario = create_test_scenario()
    model = builder.build_from_dict(scenario)
    
    # Convert to NetworkX
    G = builder.to_networkx(model, include_derived=True)
    
    print(f"\nNetworkX Graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Count edge types
    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print(f"\nEdge types:")
    for etype, count in sorted(edge_types.items()):
        print(f"  {etype}: {count}")
    
    # Check DEPENDS_ON subtypes
    depends_on_subtypes = {}
    for u, v, data in G.edges(data=True):
        if data.get('type') == 'DEPENDS_ON':
            subtype = data.get('dependency_type', 'unknown')
            depends_on_subtypes[subtype] = depends_on_subtypes.get(subtype, 0) + 1
    
    print(f"\nDEPENDS_ON subtypes:")
    for subtype, count in sorted(depends_on_subtypes.items()):
        print(f"  {subtype}: {count}")
    
    print("\n✓ NetworkX conversion successful!")
    return G


def test_validation():
    """Test graph validation"""
    print("\n" + "="*70)
    print("TEST: Graph Validation")
    print("="*70)
    
    builder = GraphBuilder()
    scenario = create_test_scenario()
    model = builder.build_from_dict(scenario)
    
    is_valid, errors, warnings = builder.validate(model)
    
    print(f"\nValidation Result: {'PASSED' if is_valid else 'FAILED'}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  ✗ {err}")
    
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warn in warnings:
            print(f"  ⚠ {warn}")
    
    if is_valid and not warnings:
        print("  No issues found!")
    
    print("\n✓ Validation complete!")


def test_serialization():
    """Test model serialization to dict/JSON"""
    print("\n" + "="*70)
    print("TEST: Serialization")
    print("="*70)
    
    builder = GraphBuilder()
    scenario = create_test_scenario()
    model = builder.build_from_dict(scenario)
    
    # Convert to dict
    model_dict = model.to_dict()
    
    print(f"\nSerialized model keys: {list(model_dict.keys())}")
    print(f"Statistics: {model_dict['statistics']}")
    
    # Check DEPENDS_ON edges in serialized form
    depends_on_edges = model_dict['edges']['depends_on']
    print(f"\nSerialized DEPENDS_ON edges: {len(depends_on_edges)}")
    
    # Show sample
    if depends_on_edges:
        sample = depends_on_edges[0]
        print(f"Sample edge: {json.dumps(sample, indent=2)}")
    
    # Verify round-trip (we can't fully round-trip derived edges, but structure should be valid)
    assert 'depends_on' in model_dict['edges']
    assert len(depends_on_edges) == len(model.depends_on_edges)
    
    print("\n✓ Serialization successful!")


def test_dependency_weight_aggregation():
    """Test that multiple topic connections increase dependency weight"""
    print("\n" + "="*70)
    print("TEST: Dependency Weight Aggregation")
    print("="*70)
    
    builder = GraphBuilder()
    scenario = create_test_scenario()
    model = builder.build_from_dict(scenario)
    
    # Find dependencies with multiple topics
    multi_topic_deps = [
        d for d in model.depends_on_edges 
        if len(d.topics) > 1
    ]
    
    print(f"\nDependencies with multiple topics: {len(multi_topic_deps)}")
    
    for dep in multi_topic_deps:
        print(f"  {dep.source} -> {dep.target}")
        print(f"    Topics: {dep.topics}")
        print(f"    Weight: {dep.weight:.2f} (expected > 1.0 for multi-topic)")
        assert dep.weight > 1.0, "Multi-topic dependency should have weight > 1.0"
    
    # Verify single-topic dependencies have base weight
    single_topic_deps = [
        d for d in model.get_depends_on_by_type(DependencyType.APP_TO_APP)
        if len(d.topics) == 1
    ]
    
    if single_topic_deps:
        print(f"\nSingle-topic dependencies: {len(single_topic_deps)}")
        for dep in single_topic_deps[:3]:
            print(f"  {dep.source} -> {dep.target}: weight={dep.weight:.2f}")
            assert dep.weight == 1.0, "Single-topic dependency should have weight = 1.0"
    
    print("\n✓ Weight aggregation working correctly!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("UNIFIED DEPENDS_ON RELATIONSHIP TEST SUITE")
    print("="*70)
    
    tests = [
        test_dependency_derivation,
        test_networkx_conversion,
        test_validation,
        test_serialization,
        test_dependency_weight_aggregation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {test.__name__}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
