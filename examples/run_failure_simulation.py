#!/usr/bin/env python3
"""
Example script to run Step 4: Failure Simulation (Fault Injection & Message Flow)
on the worked example topology, adhering to docs/failure-simulation.md specifications.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
from saag.simulation.fault_injector import FaultInjector
from saag.simulation.message_flow_simulator import MessageFlowSimulator
from saag.infrastructure.memory_repo import MemoryRepository
from saag.infrastructure.neo4j_repo import Neo4jRepository


def print_table(title, headers, rows):
    """Utility to print a clean ASCII table."""
    print(f"\n=== {title} ===")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    header_str = " | ".join(f"{str(h).ljust(col_widths[i])}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * (sum(col_widths) + len(headers) * 3 - 1))
    
    for row in rows:
        row_str = " | ".join(f"{str(val).ljust(col_widths[i])}" for i, val in enumerate(row))
        print(row_str)


def graph_data_to_networkx(graph_data) -> nx.DiGraph:
    """Convert a flat GraphData object from the repository into a NetworkX DiGraph.
    
    Ensures that type-specific node/edge attributes needed by the simulation engines
    (such as component 'type', edge 'type', 'rate_hz', and 'qos_profile') are preserved
    and take precedence over any flattened dictionary keys.
    """
    g = nx.DiGraph()
    g.graph["id"] = "worked_example"
    
    # 1. Add all components as nodes
    for comp in graph_data.components:
        props = getattr(comp, "properties", {}) or {}
        # Add all properties, then explicitly assign standard 'type' and 'name'
        g.add_node(comp.id, **props)
        g.nodes[comp.id]["type"] = comp.component_type
        g.nodes[comp.id]["name"] = props.get("name", comp.id)
        g.nodes[comp.id]["weight"] = getattr(comp, "weight", 1.0)
        
    # 2. Add all relationships as edges
    for edge in graph_data.edges:
        props = getattr(edge, "properties", {}) or {}
        # Add all properties, then explicitly assign standard relationship type
        g.add_edge(edge.source_id, edge.target_id, **props)
        g.edges[edge.source_id, edge.target_id]["type"] = edge.relation_type
        g.edges[edge.source_id, edge.target_id]["weight"] = getattr(edge, "weight", 1.0)
        g.edges[edge.source_id, edge.target_id]["rate_hz"] = props.get("rate_hz", 10.0)
        g.edges[edge.source_id, edge.target_id]["qos_profile"] = props.get("qos_profile", {})
        
    return g


def run_failure_simulation(args):
    # 1. Resolve path to worked_example.json
    script_dir = Path(__file__).parent
    json_path = script_dir / "worked_example.json"
    
    if not json_path.exists():
        print(f"Error: worked_example.json not found at {json_path}")
        sys.exit(1)
        
    print(f"Loading topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Initialize the repository
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository...")
        repo = MemoryRepository()

    try:
        # 3. Save graph to initialize mathematical weights
        print("Saving graph to initialize component weights...")
        repo.save_graph(topology_data, clear=True)
        
        # 4. Derive logical dependencies (Step 1 derived rules)
        print("Deriving logical dependencies (G_analysis)...")
        repo.derive_dependencies()
        
        # 5. Extract structural and derived data
        print("Extracting graph data with raw structural edges...")
        graph_data = repo.get_graph_data(include_raw=True)
        
        # 6. Convert to NetworkX DiGraph representation
        g = graph_data_to_networkx(graph_data)
        print(f"Constructed NetworkX graph: {len(g.nodes)} nodes, {len(g.edges)} edges.")

        # ---------------------------------------------------------------------
        # MODE 1: BFS Cascade Fault Injection
        # ---------------------------------------------------------------------
        print("\n--- Running Mode 1: Fault Injection ---")
        seeds = [42, 123, 456, 789, 2024]
        injector = FaultInjector(
            graph=g,
            seeds=seeds,
            cascade_depth_limit=0,          # Unlimited cascade
            propagation_threshold=0.2,      # Starves sub when >= 20% feeds lost
        )
        
        # Run fault injection on candidate Application and Broker nodes
        fi_result = injector.run(node_types=["Application", "Broker"])
        
        # Print results table
        fi_rows = []
        for row in fi_result.top_k_by_impact:
            rec = fi_result.records[row["node_id"]]
            fi_rows.append([
                row["node_id"],
                row["node_type"],
                f"{row['impact_score']:.4f}",
                f"{row['impact_score_std']:.4f}",
                row["cascade_depth"],
                row["orphaned_topics"],
                row["impacted_subscribers"]
            ])
        print_table(
            "Fault Injection Impact Scores I(v)",
            ["Node ID", "Type", "Impact I(v)", "Std Dev", "Depth", "Orphaned", "Impacted"],
            fi_rows
        )
        
        # Verify Fault Injection outcomes
        print("\nVerifying Fault Injection results:")
        
        # SensorApp (A0) failure cascades and orphans topic T0 (/temperature)
        rec_a0 = fi_result.records["A0"]
        assert math.isclose(rec_a0.impact_score, 1.0), f"Expected A0 impact score 1.0, got {rec_a0.impact_score}"
        assert rec_a0.cascade_depth == 1, f"Expected A0 cascade depth 1, got {rec_a0.cascade_depth}"
        assert "T0" in rec_a0.all_orphaned_topics, "Expected T0 to be orphaned by A0 failure"
        assert "A1" in rec_a0.impacted_subscriber_ids, "Expected A1 to be impacted by A0 failure"
        print("  [PASS] SensorApp (A0) cascades to fail MonitorApp (A1), resulting in I(A0) = 1.0")

        # MonitorApp (A1) failure has no downstream publishers and does not cascade
        rec_a1 = fi_result.records["A1"]
        assert math.isclose(rec_a1.impact_score, 0.0), f"Expected A1 impact score 0.0, got {rec_a1.impact_score}"
        assert rec_a1.cascade_depth == 0, f"Expected A1 cascade depth 0, got {rec_a1.cascade_depth}"
        print("  [PASS] MonitorApp (A1) does not cascade, resulting in I(A1) = 0.0")

        # MainBroker (B0) failure in fault-injection does not cascade because SensorApp (A0) is still alive
        rec_b0 = fi_result.records["B0"]
        assert math.isclose(rec_b0.impact_score, 0.0), f"Expected B0 impact score 0.0, got {rec_b0.impact_score}"
        print("  [PASS] MainBroker (B0) does not cascade in fault injection, resulting in I(B0) = 0.0")

        # ---------------------------------------------------------------------
        # MODE 2: Message Flow Simulation (SimPy)
        # ---------------------------------------------------------------------
        print("\n--- Running Mode 2: Message Flow Simulation ---")
        
        # 1. Baseline Simulation (no fault)
        sim_baseline = MessageFlowSimulator(
            graph=g,
            duration=100.0,
            fault_node=None,
            seed=42,
        )
        res_baseline = sim_baseline.run()
        
        # 2. Injected Fault Simulation (A0 fails at t=50.0)
        sim_fault = MessageFlowSimulator(
            graph=g,
            duration=100.0,
            fault_node="A0",
            fault_time=50.0,
            seed=42,
        )
        res_fault = sim_fault.run()

        # Display Message Flow tables
        mf_rows = []
        for prefix, res in [("Baseline", res_baseline), ("Faulted (A0)", res_fault)]:
            for tid, ts in res.topic_stats.items():
                p50 = f"{ts.latency_p50:.2f} ms" if ts.latency_p50 is not None else "—"
                p95 = f"{ts.latency_p95:.2f} ms" if ts.latency_p95 is not None else "—"
                mf_rows.append([
                    prefix,
                    ts.topic_name,
                    f"{ts.delivery_rate:.4f}",
                    ts.total_published,
                    ts.total_delivered,
                    p50,
                    p95,
                    ts.total_dropped_deadline
                ])
        print_table(
            "Message Flow Delivery and Latency comparison",
            ["Scenario", "Topic", "Delivery Rate", "Published", "Delivered", "P50 Latency", "P95 Latency", "Deadline Viol"],
            mf_rows
        )

        # Verify Message Flow outcomes
        print("\nVerifying Message Flow Simulation results:")
        
        # Baseline check
        assert math.isclose(res_baseline.system_delivery_rate, 1.0), f"Expected baseline delivery rate 1.0, got {res_baseline.system_delivery_rate}"
        assert res_baseline.total_messages_published > 0, "Expected non-zero baseline published count"
        print(f"  [PASS] Baseline system delivery rate = {res_baseline.system_delivery_rate:.4f} (perfect delivery)")

        # Faulted check
        fe = res_fault.fault_event
        assert fe is not None, "Expected fault event record to be populated"
        assert fe.faulted_node_id == "A0", f"Expected faulted node A0, got {fe.faulted_node_id}"
        assert fe.cascade_orphaned_topics == ["T0"], f"Expected orphaned topics ['T0'], got {fe.cascade_orphaned_topics}"
        assert fe.cascade_impacted_subscribers == ["A1"], f"Expected impacted subscribers ['A1'], got {fe.cascade_impacted_subscribers}"
        assert math.isclose(fe.delivery_rate_before, 1.0, abs_tol=1e-2), f"Expected rate before fault near 1.0, got {fe.delivery_rate_before}"
        assert math.isclose(fe.delivery_rate_after, 0.0, abs_tol=1e-2), f"Expected rate after fault 0.0, got {fe.delivery_rate_after}"
        print(f"  [PASS] Injected fault on SensorApp (A0) at t={fe.fault_time:.1f} s:")
        print(f"         - Orphaned topic: {fe.cascade_orphaned_topics}")
        print(f"         - Impacted subscriber: {fe.cascade_impacted_subscribers}")
        print(f"         - Delivery rate before: {fe.delivery_rate_before:.4f}")
        print(f"         - Delivery rate after: {fe.delivery_rate_after:.4f}")

        # ---------------------------------------------------------------------
        # Save output JSON files
        # ---------------------------------------------------------------------
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        fi_path = output_dir / "worked_example_impact_scores.json"
        mf_path = output_dir / "worked_example_message_flow_results.json"
        
        fi_result.save(fi_path)
        res_fault.save(mf_path)
        print(f"\nSaved fault injection results to: {fi_path}")
        print(f"Saved message flow results to: {mf_path}")
        
        print("\nAll simulations executed and verified successfully! Worked example matches specification.")

    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Worked Example Failure Simulation")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_failure_simulation(args)
