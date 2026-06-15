#!/usr/bin/env python3
"""
Worked Example: Structural Analysis and Quality Scoring on the Air Traffic Management (ATM) Dataset.
Adheres strictly to the structural and functional specifications of Step 2 Analyze.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
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


def run_atm_structural_analysis(args):
    # 1. Resolve path to atm_system.json
    script_dir = Path(__file__).resolve().parent
    json_path = script_dir.parent / "data" / "scenarios" / "atm_system.json"
    
    if not json_path.exists():
        print(f"Error: ATM System JSON not found at {json_path}")
        sys.exit(1)
        
    print(f"Loading safety-critical ATM topology JSON from: {json_path}")
    with open(json_path, "r") as f:
        topology_data = json.load(f)

    # 2. Select and initialize repository
    if args.neo4j:
        print(f"Initializing Neo4jRepository (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Initializing MemoryRepository...")
        repo = MemoryRepository()

    try:
        # 3. Save graph and derive dependencies
        print("Importing ATM topology and deriving dependencies...")
        repo.save_graph(topology_data, clear=True)
        
        client = Client(repo=repo)
        print("Executing structural analysis on the 'system' layer...")
        result = client.analyze(layer="system")

        # 4. Extract results
        raw_result = result.raw
        struct = raw_result.structural
        quality = raw_result.quality

        # Gather metrics and quality scores
        comp_metrics = struct.components
        comp_quality = {cq.id: cq for cq in quality.components}

        # Select a subset of representative ATM apps & topics to display
        target_ids = ["A0", "A1", "A3", "A7", "T0", "T1", "T5"]

        metrics_rows = []
        for cid in target_ids:
            m = comp_metrics.get(cid)
            if m:
                metrics_rows.append([
                    m.id,
                    m.name,
                    f"{m.reverse_pagerank:.3f}",
                    f"{m.in_degree:.3f}",
                    f"{m.mpci:.3f}",
                    f"{m.ap_c_directed:.3f}",
                    f"{m.betweenness:.3f}",
                    f"{m.dependency_weight_in:.3f}",
                    f"{m.fan_out_criticality:.3f}"
                ])
        print_table(
            "ATM Layer Normalized Structural Metrics",
            ["ID", "Name", "RPR", "DG_in", "MPCI", "AP_c_dir", "BT", "w_in", "FOC"],
            metrics_rows
        )

        quality_rows = []
        for cid in target_ids:
            cq = comp_quality.get(cid)
            if cq:
                cname = comp_metrics[cq.id].name if cq.id in comp_metrics else cq.id
                quality_rows.append([
                    cq.id,
                    cname,
                    f"{cq.scores.reliability:.3f} ({cq.levels.reliability.value})",
                    f"{cq.scores.maintainability:.3f} ({cq.levels.maintainability.value})",
                    f"{cq.scores.availability:.3f} ({cq.levels.availability.value})",
                    f"{cq.scores.overall:.3f} ({cq.levels.overall.value})"
                ])
        print_table(
            "ATM Component Criticality Scores and Levels (RMAV)",
            ["ID", "Name", "Reliability (R)", "Maintainability (M)", "Availability (A)", "Overall (Q)"],
            quality_rows
        )

        # Print graph summary stats
        summary = struct.graph_summary
        print("\n=== ATM Graph Summary S(G) ===")
        print(f"Nodes: {summary.nodes}")
        print(f"Edges: {summary.edges}")
        print(f"Density: {summary.density:.4f}")
        print(f"Average Degree: {summary.avg_degree:.2f}")
        print(f"Articulation Points: {summary.num_articulation_points}")
        print(f"Bridges: {summary.num_bridges}")
        print(f"Diameter: {summary.diameter}")
        print(f"Average Path Length: {summary.avg_path_length:.2f}")
        print(f"Assortativity: {summary.assortativity:.2f}")

        # Assertions confirming correct structural analysis behavior
        assert summary.nodes > 0, "No components in the system layer"
        assert summary.edges > 0, "No dependency edges derived"
        
        # Verify that safety-critical nodes like conflict-detector score high on overall Q
        for cq in quality.components:
            cname = comp_metrics[cq.id].name if cq.id in comp_metrics else ""
            if "conflict-detector" in cname.lower():
                assert cq.scores.overall >= 0.35, f"Expected conflict-detector to be high criticality, got {cq.scores.overall}"
                
        print("\n[PASS] ATM structural analysis completed and validated successfully!")


    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATM Structural Analysis Example")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_atm_structural_analysis(args)
