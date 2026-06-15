#!/usr/bin/env python3
"""
Example script to load, save, analyze, and verify the worked example
from Step 1 Modeling (docs/graph-model.md).
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag.infrastructure.memory_repo import MemoryRepository
from saag.infrastructure.neo4j_repo import Neo4jRepository


def print_table(title, headers, rows):
    """Utility to print a clean ASCII table."""
    print(f"\n=== {title} ===")
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    # Print header
    header_str = " | ".join(f"{str(h).ljust(col_widths[i])}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * (sum(col_widths) + len(headers) * 3 - 1))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{str(val).ljust(col_widths[i])}" for i, val in enumerate(row))
        print(row_str)


def run_worked_example(args):
    # 1. Resolve path to worked_example.json
    script_dir = Path(__file__).parent
    json_path = script_dir / "worked_example.json"
    
    if not json_path.exists():
        print(f"Error: worked_example.json not found at {json_path}")
        sys.exit(1)
        
    print(f"Loading topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Select and initialize the repository
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri}, User: {args.user})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository (in-memory mock)...")
        repo = MemoryRepository()

    try:
        # 3. Save graph (Phase 1, 2, 3, 5: entity creation, structural edges, weights)
        print("Saving graph (Phases 1, 2, 3, and 5 - Vertex Weights)...")
        repo.save_graph(topology_data, clear=True)

        # 4. Derive logical dependencies (Phase 4 and Phase 5 - Edge Weights)
        print("Deriving logical DEPENDS_ON dependencies (Phase 4 & 5 - Edge Weights)...")
        repo.derive_dependencies()

        # 5. Extract structural and derived data for verification
        print("Extracting graph data from repository...")
        graph_data = repo.get_graph_data(include_raw=True)

        # Build lookup tables for easy assertions
        comp_by_id = {c.id: c for c in graph_data.components}
        edges_by_key = {(e.source_id, e.target_id, e.dependency_type, e.relation_type): e for e in graph_data.edges}
        depends_on_edges = [e for e in graph_data.edges if e.relation_type == "DEPENDS_ON"]

        # 6. Display components and weights
        comp_rows = []
        for cid in ["T0", "A0", "A1", "B0", "L0", "N0", "N1"]:
            comp = comp_by_id.get(cid)
            if comp:
                comp_name = comp.properties.get("name", comp.id)
                comp_rows.append([comp.id, comp_name, comp.component_type, f"{comp.weight:.4f}"])
        print_table("Components and Computed Weights", ["ID", "Name", "Type", "Weight"], comp_rows)

        # 7. Display derived DEPENDS_ON edges
        edge_rows = []
        for e in depends_on_edges:
            edge_rows.append([e.source_id, e.target_id, e.dependency_type, f"{e.weight:.4f}", e.path_count])
        print_table("Derived DEPENDS_ON Edges", ["Source ID", "Target ID", "Dependency Type", "Weight", "Path Count"], edge_rows)

        # 8. Assertions verifying adherence to docs/graph-model.md
        print("\nVerifying computed weights against specifications in graph-model.md:")
        
        # Topic /temperature: Theoretical ~0.592, Code ~0.5936
        topic_w = comp_by_id["T0"].weight
        assert math.isclose(topic_w, 0.5936, abs_tol=1e-3), f"Topic weight mismatch: {topic_w} vs 0.5936"
        print(f"  [PASS] Topic '/temperature' weight = {topic_w:.4f} (matches theoretical ~0.592)")

        # Application SensorApp: Theoretical ~0.592, Code ~0.5936
        sensor_w = comp_by_id["A0"].weight
        assert math.isclose(sensor_w, 0.5936, abs_tol=1e-3), f"SensorApp weight mismatch: {sensor_w} vs 0.5936"
        print(f"  [PASS] SensorApp weight = {sensor_w:.4f} (matches theoretical ~0.592)")

        # Application MonitorApp: Theoretical ~0.592, Code ~0.5936
        monitor_w = comp_by_id["A1"].weight
        assert math.isclose(monitor_w, 0.5936, abs_tol=1e-3), f"MonitorApp weight mismatch: {monitor_w} vs 0.5936"
        print(f"  [PASS] MonitorApp weight = {monitor_w:.4f} (matches theoretical ~0.592)")

        # Broker MainBroker: Theoretical ~0.592, Code ~0.5936
        broker_w = comp_by_id["B0"].weight
        assert math.isclose(broker_w, 0.5936, abs_tol=1e-3), f"MainBroker weight mismatch: {broker_w} vs 0.5936"
        print(f"  [PASS] MainBroker weight = {broker_w:.4f} (matches theoretical ~0.592)")

        # Library NavLib: Theoretical ~0.733, Code ~0.7347 (due to DG_in=2 multiplier)
        lib_w = comp_by_id["L0"].weight
        assert math.isclose(lib_w, 0.7347, abs_tol=1e-3), f"NavLib weight mismatch: {lib_w} vs 0.7347"
        print(f"  [PASS] NavLib library weight = {lib_w:.4f} (matches theoretical ~0.733)")

        # Node N0 (ComputeNode1): hosts A0 and B0, should be max of hosted: ~0.5936
        node0_w = comp_by_id["N0"].weight
        assert math.isclose(node0_w, 0.5936, abs_tol=1e-3), f"ComputeNode1 weight mismatch: {node0_w} vs 0.5936"
        print(f"  [PASS] ComputeNode1 weight = {node0_w:.4f} (max of hosted A0 & B0)")

        print("\nVerifying derived DEPENDS_ON edges against specifications:")
        
        # Rule 1 app_to_app: MonitorApp depends on SensorApp via shared /temperature topic
        r1_key = ("A1", "A0", "app_to_app", "DEPENDS_ON")
        assert r1_key in edges_by_key, "Rule 1: app_to_app dependency missing"
        e_r1 = edges_by_key[r1_key]
        assert math.isclose(e_r1.weight, topic_w), f"Rule 1 weight mismatch: {e_r1.weight} vs {topic_w}"
        assert e_r1.path_count == 1, f"Rule 1 path count mismatch: {e_r1.path_count} vs 1"
        print("  [PASS] Rule 1: MonitorApp -> SensorApp (app_to_app) derived with path_count=1 and correct QoS weight")

        # Rule 2 app_to_broker: MonitorApp depends on MainBroker
        r2_sub_key = ("A1", "B0", "app_to_broker", "DEPENDS_ON")
        assert r2_sub_key in edges_by_key, "Rule 2: MonitorApp -> MainBroker dependency missing"
        e_r2_sub = edges_by_key[r2_sub_key]
        assert math.isclose(e_r2_sub.weight, topic_w), f"Rule 2 weight mismatch: {e_r2_sub.weight} vs {topic_w}"
        print("  [PASS] Rule 2: MonitorApp -> MainBroker (app_to_broker) derived with correct QoS weight")

        # Rule 2 app_to_broker: SensorApp depends on MainBroker
        r2_pub_key = ("A0", "B0", "app_to_broker", "DEPENDS_ON")
        assert r2_pub_key in edges_by_key, "Rule 2: SensorApp -> MainBroker dependency missing"
        e_r2_pub = edges_by_key[r2_pub_key]
        assert math.isclose(e_r2_pub.weight, topic_w), f"Rule 2 weight mismatch: {e_r2_pub.weight} vs {topic_w}"
        print("  [PASS] Rule 2: SensorApp -> MainBroker (app_to_broker) derived with correct QoS weight")

        # Rule 3 node_to_node: ComputeNode2 (hosting MonitorApp) depends on ComputeNode1 (hosting SensorApp)
        r3_key = ("N1", "N0", "node_to_node", "DEPENDS_ON")
        assert r3_key in edges_by_key, "Rule 3: ComputeNode2 -> ComputeNode1 dependency missing"
        e_r3 = edges_by_key[r3_key]
        assert math.isclose(e_r3.weight, topic_w), f"Rule 3 weight mismatch: {e_r3.weight} vs {topic_w}"
        print("  [PASS] Rule 3: ComputeNode2 -> ComputeNode1 (node_to_node) derived with correct weight")

        # Rule 4 node_to_broker: ComputeNode2 (hosting MonitorApp) depends on MainBroker
        r4_key = ("N1", "B0", "node_to_broker", "DEPENDS_ON")
        assert r4_key in edges_by_key, "Rule 4: ComputeNode2 -> MainBroker dependency missing"
        e_r4 = edges_by_key[r4_key]
        assert math.isclose(e_r4.weight, topic_w), f"Rule 4 weight mismatch: {e_r4.weight} vs {topic_w}"
        print("  [PASS] Rule 4: ComputeNode2 -> MainBroker (node_to_broker) derived with correct weight")

        # Rule 5 app_to_lib: SensorApp/MonitorApp depend on NavLib, inheriting app weights
        r5_sensor_key = ("A0", "L0", "app_to_lib", "DEPENDS_ON")
        assert r5_sensor_key in edges_by_key, "Rule 5: SensorApp -> NavLib dependency missing"
        e_r5_sensor = edges_by_key[r5_sensor_key]
        assert math.isclose(e_r5_sensor.weight, sensor_w), f"Rule 5 weight mismatch: {e_r5_sensor.weight} vs {sensor_w}"
        print("  [PASS] Rule 5: SensorApp -> NavLib (app_to_lib) derived and finalized with SensorApp weight")

        r5_monitor_key = ("A1", "L0", "app_to_lib", "DEPENDS_ON")
        assert r5_monitor_key in edges_by_key, "Rule 5: MonitorApp -> NavLib dependency missing"
        e_r5_monitor = edges_by_key[r5_monitor_key]
        assert math.isclose(e_r5_monitor.weight, monitor_w), f"Rule 5 weight mismatch: {e_r5_monitor.weight} vs {monitor_w}"
        print("  [PASS] Rule 5: MonitorApp -> NavLib (app_to_lib) derived and finalized with MonitorApp weight")

        print("\nWorked example verified successfully! All calculations and derived dependencies match spec.")

    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Worked Example Verification")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_worked_example(args)
