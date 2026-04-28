"""
Step 2 — Graph Import
=====================
Demonstrates how to import a generated topology JSON into Neo4j.

Prerequisites:
  • Neo4j 5.x running at bolt://localhost:7687 (user=neo4j, password=password)
  • examples/example_generation.py has been run (produces output/example_graph.json)

Run from the project root:
    python examples/example_import.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from saag.adapters import create_repository


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. Locate graph JSON ───────────────────
    input_file = ROOT / "output" / "example_graph.json"
    if not input_file.exists():
        print(f"[ERROR] {input_file} not found.")
        print("  Run examples/example_generation.py first.")
        return

    print_section("Loading graph JSON")
    with open(input_file) as f:
        data = json.load(f)

    n_apps    = len(data.get("applications", []))
    n_brokers = len(data.get("brokers", []))
    n_topics  = len(data.get("topics", []))
    n_nodes   = len(data.get("nodes", []))
    print(f"  Loaded: {n_apps} apps  |  {n_brokers} brokers  |  {n_topics} topics  |  {n_nodes} infra nodes")

    # ── 2. Connect to Neo4j ────────────────────
    print_section("Connecting to Neo4j")
    print("  URI      : bolt://localhost:7687")
    print("  User     : neo4j")
    print("  (these defaults are used when no arguments are passed to create_repository)")
    print()

    # Override defaults with keyword args if needed:
    #   create_repository(uri="bolt://...", user="neo4j", password="secret")
    try:
        repo = create_repository()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        print("  Ensure Neo4j is running and credentials are correct.")
        return

    try:
        # ── 3. Import — clear=True wipes existing graph first ─────────
        print_section("Importing data (clear=True wipes existing data)")
        repo.save_graph(data, clear=True)
        print("  Import complete!")

        # ── 4. Verify via repository statistics ───────────────────────
        print_section("Verifying import via Neo4j statistics")
        stats = repo.get_statistics()

        print(f"  {'Metric':<35} Value")
        print(f"  {'-'*45}")
        for key, value in sorted(stats.items()):
            print(f"  {key:<35} {value}")

        # ── 5. Spot-check: load back a component ─────────────────────
        print_section("Spot-check: retrieving graph data from Neo4j")
        gd = repo.get_graph_data()
        comp_count = len(gd.components) if hasattr(gd, "components") else "?"
        print(f"  Components returned from graph_data: {comp_count}")
        if hasattr(gd, "components") and gd.components:
            sample = gd.components[0]
            print(f"  Sample component: id={sample.id}  type={sample.component_type}")

    finally:
        repo.close()
        print()

    print_section("Done")
    print("  Next step: run  examples/example_analysis.py")


if __name__ == "__main__":
    main()
