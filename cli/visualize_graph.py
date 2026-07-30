#!/usr/bin/env python3
"""
Graph Visualization CLI (Step 7)
================================
Generates multi-layer analysis dashboards using the VisualizationService.

Usage
-----
  # Demo (no Neo4j required)
  python cli/visualize_graph.py --demo --open

  # Single layer
  python cli/visualize_graph.py --layer system -o output/dashboard.html

  # Multi-layer (comma-separated via --layer, or explicit --layers flag)
  python cli/visualize_graph.py --layers app,infra,system -o output/dashboard.html

  # With pre-computed cascade file and multi-seed paths
  python cli/visualize_graph.py --layer system \\
      --cascade-file results/cascade.json \\
      --multi-seed results/val_s42.json results/val_s123.json \\
      --open
"""

import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/visualize_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import webbrowser
import os
from saag import Client
from saag.visualization import ComponentDetail, LayerData, VisualizationService
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay


def _demo_layer_data() -> LayerData:
    """
    Hand-built LayerData covering every optional dashboard section.

    Feeding this through VisualizationService.build_html() renders exactly
    the production dashboard, so --demo doubles as a smoke test that needs
    no Neo4j.
    """
    # ── Core layer data ──────────────────────────────────────────────────────
    demo_data = LayerData(
        layer="system", name="Demo System", nodes=48, edges=127, density=0.056,
        critical_count=5, high_count=8, medium_count=15, low_count=12, minimal_count=8,
        spof_count=3, problems_count=2, spearman=0.876, f1_score=0.923,
        precision=0.912, recall=0.857, top5_overlap=0.80, top10_overlap=0.70,
        validation_passed=True, event_throughput=1000, event_delivery_rate=98.5,
        max_impact=0.734,
        # Per-dimension ρ
        reliability_spearman=0.841, maintainability_spearman=0.793,
        availability_spearman=0.882, security_spearman=0.714,
        composite_spearman=0.876,
        composite_ci=(0.831, 0.921),
        # Multi-seed stability (§6.4.6)
        multiseed_seeds=["s42", "s123", "s456", "s789", "s2024"],
        multiseed_rho=[0.871, 0.876, 0.868, 0.882, 0.879],
        multiseed_f1=[0.918, 0.923, 0.911, 0.928, 0.920],
    )

    # ── Component details ────────────────────────────────────────────────────
    demo_data.component_details = [
        ComponentDetail(
            "sensor_fusion", "Sensor Fusion", "Application",
            reliability=0.82, maintainability=0.88, availability=0.90,
            security=0.75, overall=0.84, level="CRITICAL",
            impact=0.79, cascade_risk=0.81, cascade_risk_topo=0.71,
        ),
        ComponentDetail(
            "planning_engine", "Planning Engine", "Application",
            reliability=0.75, maintainability=0.81, availability=0.85,
            security=0.60, overall=0.72, level="HIGH",
            impact=0.65, cascade_risk=0.68, cascade_risk_topo=0.59,
        ),
        ComponentDetail(
            "main_broker", "Main Broker", "Broker",
            reliability=0.88, maintainability=0.79, availability=0.93,
            security=0.65, overall=0.80, level="CRITICAL",
            impact=0.76, cascade_risk=0.75, cascade_risk_topo=0.66,
            spof=True,
        ),
        ComponentDetail(
            "nav_lib", "NavLib", "Library",
            reliability=0.62, maintainability=0.70, availability=0.78,
            security=0.55, overall=0.61, level="MEDIUM",
            impact=0.52, cascade_risk=0.49, cascade_risk_topo=0.44,
        ),
        ComponentDetail(
            "telemetry_topic", "Telemetry", "Topic",
            reliability=0.44, maintainability=0.55, availability=0.60,
            security=0.40, overall=0.40, level="LOW",
            impact=0.30, cascade_risk=0.28, cascade_risk_topo=0.25,
        ),
    ]
    demo_data.scatter_data = [
        (c.id, c.overall, c.impact, c.level)
        for c in demo_data.component_details
    ]
    # Hierarchy data (Section 10)
    demo_data.hierarchy_data = {
        "id": "ATM_System", "label": "ATM System (CSS)", "level": "CSS",
        "children": [
            {
                "id": "Surveillance", "label": "Surveillance CSCI", "level": "CSCI", "q": 0.76, "cbci": 0.42,
                "children": [
                    {"id": "sf", "label": "Sensor Fusion (CSU)", "level": "CSU", "q": 0.84, "spof": True},
                    {"id": "nl", "label": "NavLib (CSU)", "level": "CSU", "q": 0.61},
                ]
            },
            {
                "id": "Planning", "label": "Planning CSCI", "level": "CSCI", "q": 0.72, "cbci": 0.38,
                "children": [
                    {"id": "pe", "label": "Planning Engine (CSU)", "level": "CSU", "q": 0.72},
                ]
            }
        ]
    }
    # Cascade results (§6.4.5)
    demo_data.cascade_results = [
        {
            "id": c.id, "name": c.name, "type": c.type,
            "cascade_risk": c.cascade_risk, "cascade_risk_topo": c.cascade_risk_topo,
            "cascade_depth": 3, "level": c.level,
        }
        for c in demo_data.component_details
    ]
    demo_data.qos_gini = 0.347
    demo_data.cascade_wilcoxon_p = 0.031
    demo_data.cascade_delta_rho = 0.052

    # Topology: drives both the Cytoscape network and the dependency matrix.
    demo_data.network_nodes = [
        {
            "id": c.id,
            "label": c.name,
            "type": c.type,
            "level": c.level,
            "value": c.overall * 30 + 10,
            "title": f"<b>{c.id}</b><br>Type: {c.type}<br>Q(v): {c.overall:.3f}",
        }
        for c in demo_data.component_details
    ]
    demo_data.network_edges = [
        {"source": "sensor_fusion",   "target": "main_broker",     "weight": 2.5},
        {"source": "planning_engine", "target": "main_broker",     "weight": 1.2},
        {"source": "main_broker",     "target": "telemetry_topic", "weight": 1.0},
        {"source": "planning_engine", "target": "nav_lib",         "weight": 1.0},
    ]

    demo_data.anti_patterns = [
        {
            "pattern_id": "SPOF",
            "name": "Single Point of Failure",
            "severity": "critical",
            "description": "Main Broker is an articulation point — its removal partitions the graph.",
            "components": ["main_broker"],
        },
    ]

    demo_data.gates = {
        "G1_spearman": True, "G2_f1": True,
        "G3_precision": True, "G4_top5": True,
    }

    return demo_data


def run_demo(output_file: str, open_browser: bool) -> int:
    """Generate a demo dashboard with sample data (no Neo4j required)."""
    display = ConsoleDisplay()
    display.print_header("Software-as-a-Graph Demo Mode")
    display.print_step("Generating mock analysis data...")

    demo_data = _demo_layer_data()

    display.print_step("Assembling interactive charts...")
    # build_html() is pure — it reads only the LayerData it is handed, so the
    # service needs no wired-up backends here.
    service = VisualizationService(None, None, None, None, None)
    html = service.build_html(
        [demo_data], title="Software-as-a-Graph Demo Dashboard"
    )

    display.print_step("Finalizing dashboard export...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    display.display_visualization_summary(str(output_path))

    if open_browser:
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f"file://{abs_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-layer graph analysis dashboards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--demo", action="store_true",
                        help="Generate demo dashboard (no Neo4j required)")
    parser.add_argument("--no-network", action="store_true",
                        help="Exclude interactive network graphs")
    parser.add_argument("--no-matrix", action="store_true",
                        help="Exclude dependency matrices")
    parser.add_argument("--no-validation", action="store_true",
                        help="Exclude validation metrics")
    parser.add_argument("--antipatterns",
                        help="Path to pre-calculated anti-pattern JSON report")

    # --layers: explicit multi-layer comma-separated flag (documented usage).
    # --layer (from add_common_args) is the backwards-compat single/multi alias.
    # If --layers is given it takes precedence over --layer.
    parser.add_argument(
        "--layers",
        help="Comma-separated analysis layers, e.g. 'app,infra,system'. "
             "Alias for --layer when multiple layers are needed.",
    )

    # --multi-seed: accepts one or more JSON file paths (shell globs expand
    # before argv reaches this point, so nargs='*' collects them all).
    parser.add_argument(
        "--multi-seed", nargs="*", metavar="JSON_PATH", default=[],
        help="Paths to per-seed validation JSON files "
             "(e.g. results/val_s*.json). Renders the multi-seed stability "
             "panel on the Validation tab.",
    )

    # --cascade-file wires straight into generate_dashboard(cascade_file=).
    parser.add_argument(
        "--cascade-file",
        metavar="JSON_PATH",
        help="Path to a QoS cascade-risk JSON (schema in "
             "docs/visualization.md §3.4). Enables the Cascade risk tab.",
    )

    # Use -b (browser) for --open to avoid conflict with -o (output).
    parser.add_argument("--open", "-b", action="store_true",
                        help="Open dashboard in browser after generation")

    add_neo4j_arguments(parser)
    add_common_arguments(parser)

    args, unknown = parser.parse_known_args()

    # Configure logging (was previously never called)
    setup_logging(args)

    display = ConsoleDisplay()

    if args.demo:
        out = args.output if args.output else "dashboard.html"
        return run_demo(out, args.open)

    display.print_header("Analysis Dashboard Generation")

    # Resolve layer list: --layers flag takes precedence over --layer
    layer_str = args.layers if args.layers else (args.layer or "system")
    layers = [l.strip() for l in layer_str.split(",") if l.strip()]

    # Resolve multi-seed: empty list → pass 0 to skip; list of paths → pass as-is
    multi_seed_arg = args.multi_seed if args.multi_seed else 0

    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)

    display.print_step(f"Generating dashboard for layers: {', '.join(layers)}")
    out_path = client.visualize(
        output=args.output if args.output else "dashboard.html",
        layers=layers,
        include_network=not args.no_network,
        include_matrix=not args.no_matrix,
        include_validation=not args.no_validation,
        antipatterns_file=args.antipatterns,
        multi_seed=multi_seed_arg,
        cascade_file=args.cascade_file,
    )

    display.display_visualization_summary(out_path)

    if args.open:
        abs_path = os.path.abspath(out_path)
        webbrowser.open(f"file://{abs_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())