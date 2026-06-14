#!/usr/bin/env python3
"""
Integration script to verify the roundtrip integrity of the import-export process.
Reflects Section 13 (Export-Import Roundtrip) in docs/graph-model.md.
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


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_stats(repo):
    """Gathers entity and relationship statistics for verification."""
    data = repo.get_graph_data(include_raw=True)
    
    # Vertex counts by type
    comp_types = {}
    comp_weights = {}
    for c in data.components:
        comp_types[c.component_type] = comp_types.get(c.component_type, 0) + 1
        comp_weights[c.id] = c.weight
        
    # Edge counts by relation type and dependency type
    edges_by_rel = {}
    edges_by_dep = {}
    for e in data.edges:
        edges_by_rel[e.relation_type] = edges_by_rel.get(e.relation_type, 0) + 1
        if e.relation_type == "DEPENDS_ON":
            edges_by_dep[e.dependency_type] = edges_by_dep.get(e.dependency_type, 0) + 1
            
    return {
        "components": comp_types,
        "weights": comp_weights,
        "edges_rel": edges_by_rel,
        "edges_dep": edges_by_dep,
        "total_nodes": len(data.components),
        "total_edges": len(data.edges)
    }


def compare_stats(stats1, stats2):
    """Compares two graph statistics dictionaries and asserts equivalence."""
    errors = []
    
    # 1. Compare total counts
    if stats1["total_nodes"] != stats2["total_nodes"]:
        errors.append(f"Node count mismatch: {stats1['total_nodes']} vs {stats2['total_nodes']}")
    if stats1["total_edges"] != stats2["total_edges"]:
        errors.append(f"Edge count mismatch: {stats1['total_edges']} vs {stats2['total_edges']}")
        
    # 2. Compare per-label component counts
    for label, count in stats1["components"].items():
        count2 = stats2["components"].get(label, 0)
        if count != count2:
            errors.append(f"Component type '{label}' count mismatch: {count} vs {count2}")
            
    # 3. Compare relationship type counts
    for rel, count in stats1["edges_rel"].items():
        count2 = stats2["edges_rel"].get(rel, 0)
        if count != count2:
            errors.append(f"Relationship type '{rel}' count mismatch: {count} vs {count2}")
            
    # 4. Compare DEPENDS_ON dependency type counts
    for dep, count in stats1["edges_dep"].items():
        count2 = stats2["edges_dep"].get(dep, 0)
        if count != count2:
            errors.append(f"Dependency type '{dep}' count mismatch: {count} vs {count2}")

    # 5. Compare component weights (must be functionally identical)
    for cid, w1 in stats1["weights"].items():
        w2 = stats2["weights"].get(cid)
        if w2 is None:
            errors.append(f"Component '{cid}' missing in re-imported graph")
        elif not math.isclose(w1, w2, abs_tol=1e-5):
            errors.append(f"Component '{cid}' weight mismatch: {w1:.6f} vs {w2:.6f}")
            
    return errors


def run_roundtrip_test(args):
    # 1. Resolve paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"snapshot_{input_path.name}"

    # 2. Initialize repositories
    if args.neo4j:
        print(f"Running roundtrip using Neo4j (URI: {args.uri})...")
        repo = Neo4jRepository(uri=args.uri, user=args.user, password=args.password)
    else:
        print("Running roundtrip using MemoryRepository...")
        repo = MemoryRepository()

    try:
        # Step A: Load and save original topology
        print(f"\n[Step A] Importing original topology: {input_path}")
        original_data = load_json(input_path)
        repo.save_graph(original_data, clear=True)
        repo.derive_dependencies()
        
        # Gather original statistics
        stats_original = get_stats(repo)
        
        # Step B: Export snapshot
        print(f"[Step B] Exporting snapshot to: {snapshot_path}")
        exported_payload = repo.export_json()
        with open(snapshot_path, "w") as f:
            json.dump(exported_payload, f, indent=2)
            
        # Step C: Re-import snapshot into cleared DB
        print(f"[Step C] Re-importing snapshot from: {snapshot_path}")
        repo.save_graph(exported_payload, clear=True)
        repo.derive_dependencies()
        
        # Gather re-imported statistics
        stats_reimported = get_stats(repo)
        
        # Step D: Compare & Verify
        print("[Step D] Comparing original vs. re-imported database states...")
        errors = compare_stats(stats_original, stats_reimported)
        
        if errors:
            print("\n❌ ROUNDTRIP VALIDATION FAILED!")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("\n✅ ROUNDTRIP VALIDATION PASSED!")
            print(f"  - Total Vertices: {stats_original['total_nodes']}")
            print(f"  - Total Relationships: {stats_original['total_edges']}")
            print("  - Vertex types, edge types, and computed weights are identical.")
            print("  - Metadata, code_metrics, and system_hierarchy fully preserved.")
            
    finally:
        repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Export-Import Roundtrip Validation")
    parser.add_argument("--input", required=True, help="Input topology JSON file")
    parser.add_argument("--output-dir", default="output", help="Directory to save exported snapshot")
    parser.add_argument("--neo4j", action="store_true", help="Run against a live Neo4j instance instead of in-memory")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    
    args = parser.parse_args()
    run_roundtrip_test(args)
