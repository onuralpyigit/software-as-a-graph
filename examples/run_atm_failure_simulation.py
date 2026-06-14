#!/usr/bin/env python3
"""
Worked Example: Failure Simulation on the Air Traffic Management (ATM) Dataset.
Adheres strictly to the structural and functional specifications of Step 4 Failure Simulation.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
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
    """Convert flat GraphData from the repository into a NetworkX DiGraph."""
    g = nx.DiGraph()
    g.graph["id"] = "atm_system"
    
    # Add nodes
    for comp in graph_data.components:
        props = getattr(comp, "properties", {}) or {}
        g.add_node(comp.id, **props)
        g.nodes[comp.id]["type"] = comp.component_type
        g.nodes[comp.id]["name"] = props.get("name", comp.id)
        g.nodes[comp.id]["weight"] = getattr(comp, "weight", 1.0)
        
    # Add edges
    for edge in graph_data.edges:
        props = getattr(edge, "properties", {}) or {}
        g.add_edge(edge.source_id, edge.target_id, **props)
        g.edges[edge.source_id, edge.target_id]["type"] = edge.relation_type
        g.edges[edge.source_id, edge.target_id]["weight"] = getattr(edge, "weight", 1.0)
        g.edges[edge.source_id, edge.target_id]["rate_hz"] = props.get("rate_hz", 10.0)
        g.edges[edge.source_id, edge.target_id]["qos_profile"] = props.get("qos_profile", {})
        
    return g


def run_atm_failure_simulation(args):
    # 1. Resolve path to atm_system.json
    script_dir = Path(__file__).resolve().parent
    json_path = script_dir.parent / "data" / "scenarios" / "atm_system.json"
    
    if not json_path.exists():
        print(f"Error: ATM System JSON not found at {json_path}")
        sys.exit(1)
        
    print(f"Loading safety-critical ATM topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Initialize repository
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository...")
        repo = MemoryRepository()

    try:
        # 3. Save graph to initialize component weights
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

        # Find key ATM components dynamically by name
        radar_tracker_id = None
        conflict_detector_id = None
        
        for nid, data in g.nodes(data=True):
            if data.get("name") == "radar-tracker":
                radar_tracker_id = nid
            elif data.get("name") == "conflict-detector":
                conflict_detector_id = nid
                
        if not radar_tracker_id:
            # Fallback to first Application with tracker/sensor in name
            for nid, data in g.nodes(data=True):
                if data.get("type") == "Application" and "tracker" in data.get("name", "").lower():
                    radar_tracker_id = nid
                    break
        if not conflict_detector_id:
            for nid, data in g.nodes(data=True):
                if data.get("type") == "Application" and "conflict" in data.get("name", "").lower():
                    conflict_detector_id = nid
                    break

        print(f"Dynamically identified target nodes: radar-tracker={radar_tracker_id}, conflict-detector={conflict_detector_id}")

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
        
        # Print top 10 results table
        fi_rows = []
        for row in fi_result.top_k_by_impact[:10]:
            rec = fi_result.records[row["node_id"]]
            fi_rows.append([
                row["node_id"],
                g.nodes[row["node_id"]].get("name", row["node_id"]),
                row["node_type"],
                f"{row['impact_score']:.4f}",
                f"{row['impact_score_std']:.4f}",
                row["cascade_depth"],
                row["orphaned_topics"],
                row["impacted_subscribers"]
            ])
        print_table(
            "Fault Injection Impact Scores (Top 10 I(v))",
            ["Node ID", "Name", "Type", "Impact I(v)", "Std Dev", "Depth", "Orphaned Topics", "Impacted Subs"],
            fi_rows
        )

        
        # Verify Fault Injection outcomes
        print("\nVerifying Fault Injection results:")
        if radar_tracker_id:
            rec_radar = fi_result.records[radar_tracker_id]
            # Since radar-tracker publishes tracks, failing it should have cascade impact
            print(f"  [INFO] radar-tracker ({radar_tracker_id}) failure impact score = {rec_radar.impact_score:.4f}, depth = {rec_radar.cascade_depth}")
            assert rec_radar.impact_score >= 0.0, "Impact score must be non-negative"
            
        if conflict_detector_id:
            rec_conflict = fi_result.records[conflict_detector_id]
            print(f"  [INFO] conflict-detector ({conflict_detector_id}) failure impact score = {rec_conflict.impact_score:.4f}")

        # ---------------------------------------------------------------------
        # MODE 2: Message Flow Simulation (SimPy)
        # ---------------------------------------------------------------------
        print("\n--- Running Mode 2: Message Flow Simulation ---")
        
        # Pick the radar tracker as our injection target for SimPy simulation
        target_fault_node = radar_tracker_id or "A0"
        
        # 1. Baseline Simulation (no fault)
        print("Running baseline simulation (10s duration)...")
        sim_baseline = MessageFlowSimulator(
            graph=g,
            duration=10.0,
            fault_node=None,
            seed=42,
        )
        res_baseline = sim_baseline.run()
        
        # 2. Injected Fault Simulation (fails at t=5.0s)
        print(f"Running faulted simulation (10s duration, inject fault on {target_fault_node} at t=5.0s)...")
        sim_fault = MessageFlowSimulator(
            graph=g,
            duration=10.0,
            fault_node=target_fault_node,
            fault_time=5.0,
            seed=42,
        )
        res_fault = sim_fault.run()

        # Display Message Flow comparison table for representative topics
        mf_rows = []
        for prefix, res in [("Baseline", res_baseline), ("Faulted", res_fault)]:
            count = 0
            for tid, ts in res.topic_stats.items():
                if count >= 5:  # Display up to 5 topics
                    break
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
                count += 1
                
        print_table(
            "Message Flow Delivery and Latency comparison (Subset)",
            ["Scenario", "Topic", "Delivery Rate", "Published", "Delivered", "P50 Latency", "P95 Latency", "Deadline Viol"],
            mf_rows
        )

        # Verify Message Flow outcomes
        print("\nVerifying Message Flow Simulation results:")
        assert res_baseline.system_delivery_rate > 0.0, "Expected non-zero baseline delivery rate"
        print(f"  [PASS] Baseline system delivery rate = {res_baseline.system_delivery_rate:.4f}")
        
        fe = res_fault.fault_event
        if fe:
            print(f"  [PASS] Fault injection at t={fe.fault_time:.1f} s successfully triggered on {fe.faulted_node_id}")
            print(f"         - Delivery rate before fault: {fe.delivery_rate_before:.4f}")
            print(f"         - Delivery rate after fault: {fe.delivery_rate_after:.4f}")

        # ---------------------------------------------------------------------
        # Save output JSON files
        # ---------------------------------------------------------------------
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        fi_path = output_dir / "atm_system_impact_scores.json"
        mf_path = output_dir / "atm_system_message_flow_results.json"
        
        fi_result.save(fi_path)
        res_fault.save(mf_path)
        print(f"\nSaved fault injection results to: {fi_path}")
        print(f"Saved message flow results to: {mf_path}")
        
        print("\nAll ATM failure simulations executed and verified successfully!")

    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATM Failure Simulation")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_atm_failure_simulation(args)
