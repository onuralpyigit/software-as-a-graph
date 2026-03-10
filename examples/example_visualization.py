"""
Step 7 — Dashboard Visualization
==================================
Demonstrates how to generate the static HTML dashboard programmatically.

The dashboard embeds:
  • Interactive network graph (forced-directed layout)
  • RMAV score bar charts per layer
  • Top-K criticality tables
  • Per-dimension validation comparison tables
  • Cascade failure heatmap
  • Simulation flow metrics

The output is a single self-contained HTML file — suitable as a reproducible
research artefact (thesis appendix, peer review).

Prerequisites:
  • Neo4j running with imported data (run examples/example_import.py first)

Run from the project root:
    python examples/example_visualization.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.core import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.visualization import VisualizationService


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    output_dir  = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / "example_dashboard.html")

    try:
        repo = create_repository()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        print("  Ensure Neo4j is running and import has been done.")
        return

    try:
        print_section("Initialising services")
        analysis   = AnalysisService(repo)
        simulation = SimulationService(repo)
        validation = ValidationService(analysis, simulation, ndcg_k=10)

        viz = VisualizationService(
            analysis_service=analysis,
            simulation_service=simulation,
            validation_service=validation,
            repository=repo,
        )

        # ── 1. Generate full dashboard ─────────────────────────────────
        print_section("Generating dashboard")
        print("  Layers   : app, system")
        print("  Network  : enabled (interactive force-graph)")
        print(f"  Output   : {output_file}")
        print()

        path = viz.generate_dashboard(
            output_file=output_file,
            layers=["app", "system"],
            include_network=True,
        )

        size_kb = os.path.getsize(path) / 1024
        print(f"  ✓ Dashboard written: {os.path.abspath(path)}  ({size_kb:.1f} KB)")

        # ── 2. Quick sanity-check: ensure file is non-empty HTML ──────
        print_section("Sanity check")
        with open(path, encoding="utf-8") as f:
            content = f.read()

        checks = {
            "Contains <html>":    "<html" in content.lower(),
            "Contains <canvas>":  "<canvas" in content.lower() or "graph" in content.lower(),
            "Contains 'RMAV'":    "RMAV" in content or "rmav" in content,
            "Contains 'Critical'":"critical" in content.lower(),
            "File size > 10 KB":  size_kb > 10,
        }
        all_ok = True
        for desc, ok in checks.items():
            icon = "✓" if ok else "✗"
            print(f"    {icon} {desc}")
            if not ok:
                all_ok = False

        if all_ok:
            print("\n  All checks passed — open the file in a browser to explore.")
        else:
            print("\n  Some checks failed — the dashboard may be incomplete.")

    finally:
        repo.close()

    print()
    print_section("Done")
    print("  Open in browser:")
    print(f"    xdg-open {os.path.abspath(output_file)}")
    print()
    print("  For an end-to-end walkthrough see:")
    print("    examples/example_end_to_end.py")


if __name__ == "__main__":
    main()