#!/usr/bin/env python3
"""
Graph Visualization CLI (Step 6)
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

import json
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/visualize_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import webbrowser
import os
from saag import Client
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay


def run_demo(output_file: str, open_browser: bool) -> int:
    """Generate a demo dashboard with sample data (no Neo4j required).

    Exercises all chart code paths — criticality, RMAV breakdown, scatter,
    cascade risk (§6.4.5), multi-seed stability (§6.4.6), and dim-ρ bars —
    so the demo mode is a useful smoke test for the visualization layer.
    """
    display = ConsoleDisplay()
    display.print_header("Software-as-a-Graph Demo Mode")
    display.print_step("Generating mock analysis data...")

    from saag.visualization import LayerData, ComponentDetail
    from saag.visualization.charts import ChartGenerator
    from saag.visualization.dashboard import DashboardGenerator

    dash = DashboardGenerator("Software-as-a-Graph Demo Dashboard")
    charts = ChartGenerator()

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
        availability_spearman=0.882, vulnerability_spearman=0.714,
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
            vulnerability=0.75, overall=0.84, level="CRITICAL",
            impact=0.79, cascade_risk=0.81, cascade_risk_topo=0.71,
        ),
        ComponentDetail(
            "planning_engine", "Planning Engine", "Application",
            reliability=0.75, maintainability=0.81, availability=0.85,
            vulnerability=0.60, overall=0.72, level="HIGH",
            impact=0.65, cascade_risk=0.68, cascade_risk_topo=0.59,
        ),
        ComponentDetail(
            "main_broker", "Main Broker", "Broker",
            reliability=0.88, maintainability=0.79, availability=0.93,
            vulnerability=0.65, overall=0.80, level="CRITICAL",
            impact=0.76, cascade_risk=0.75, cascade_risk_topo=0.66,
            spof=True,
        ),
        ComponentDetail(
            "nav_lib", "NavLib", "Library",
            reliability=0.62, maintainability=0.70, availability=0.78,
            vulnerability=0.55, overall=0.61, level="MEDIUM",
            impact=0.52, cascade_risk=0.49, cascade_risk_topo=0.44,
        ),
        ComponentDetail(
            "telemetry_topic", "Telemetry", "Topic",
            reliability=0.44, maintainability=0.55, availability=0.60,
            vulnerability=0.40, overall=0.40, level="LOW",
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

    # ── Tab 1: Executive Overview ──────────────────────────────────────────
    dash.add_tab("Overview", "overview")
    dash.start_section("📊 Demo Overview")
    dash.add_kpis(
        {"Total components": 14, "Critical / SPOFs": "2 / 2", "Validation \u03c1": 0.876, "F1 score": 0.893},
        {"Critical / SPOFs": "danger", "Validation \u03c1": "success"}
    )

    display.print_step("Assembling interactive charts...")
    charts_main = []
    if c := charts.criticality_distribution(demo_data.classification_distribution):
        charts_main.append(c)
    if c := charts.rmav_breakdown(demo_data.component_details, "RMAV dimension comparison — top 6", top_n=6):
        charts_main.append(c)
    dash.add_charts(charts_main)
    
    dash.add_top5_bars(demo_data.component_details)
    dash.end_section()
    dash.end_tab()

    # ── Tab 2: Component Table ─────────────────────────────────────────────
    dash.add_tab("Component table", "components")
    dash.start_section("Detailed component analysis")
    headers = ["Component", "Type", "R", "M", "A", "V", "Q(v)", "Impact", "Level", "SPOF"]
    rows = []
    for c in demo_data.component_details:
        rows.append([
            c.name, c.type, f"{c.reliability:.2f}", f"{c.maintainability:.2f}", 
            f"{c.availability:.2f}", f"{c.vulnerability:.2f}", f"{c.overall:.3f}", 
            f"{c.impact:.3f}", f'<span class="badge badge-{c.level.lower()}">{c.level}</span>',
            '<span class="badge badge-spof">SPOF</span>' if c.spof else ""
        ])
    dash.add_interactive_table(headers, rows, title="ATM System Components", type_col=1, level_col=8)
    dash.end_section()
    dash.end_tab()

    # ── Tab 3: Validation ──────────────────────────────────────────────────
    dash.add_tab("Validation", "validation")
    dash.start_section("Prediction Correlation & Stability")
    dash.add_kpis({
        "Spearman \u03c1": 0.876, "F1 (top-k)": 0.893, "Precision": 0.912, "Recall": 0.857
    }, {"Spearman \u03c1": "success", "F1 (top-k)": "success"})
    
    if sc := charts.correlation_scatter(demo_data.scatter_data, spearman=0.876):
        dash.add_charts([sc])
    
    dim_rho_html = charts.dim_rho_bars(demo_data.dim_rho)
    seed_chart = charts.multiseed_line_chart(demo_data.multiseed_seeds, demo_data.multiseed_rho, demo_data.multiseed_f1)
    dash.add_dim_rho_panel(dim_rho_html, seed_chart)
    dash.end_section()
    dash.end_tab()

    # ── Tab 4: Cascade Risk ───────────────────────────────────────────────
    dash.add_tab("Cascade risk", "cascade")
    dash.start_section("QoS-enriched failure propagation")
    class _CProxy:
        def __init__(self, d):
            self.name = d["name"]; self.cascade_risk = d["cascade_risk"]; self.cascade_risk_topo = d["cascade_risk_topo"]
    proxies = [_CProxy(r) for r in demo_data.cascade_results]
    proxies.sort(key=lambda x: x.cascade_risk, reverse=True)
    cascade_chart = charts.cascade_risk_chart(proxies)
    dash.add_cascade_risk_panel(
        cascade_chart_html=cascade_chart, qos_gini=demo_data.qos_gini,
        wilcoxon_p=demo_data.cascade_wilcoxon_p, delta_rho=demo_data.cascade_delta_rho,
        note="QoS enrichment adds statistically significant predictive signal."
    )
    dash.end_section()
    dash.end_tab()

    # ── Tab 5: Topology ────────────────────────────────────────────────────
    dash.add_tab("Topology", "topology")
    dash.start_section("Network Graph")
    mock_nodes = [
        {"id": "sf",   "label": "Sensor Fusion",  "type": "Application", "level": "CRITICAL", "value": 40},
        {"id": "pe",   "label": "Planning Engine", "type": "Application", "level": "HIGH",     "value": 30},
        {"id": "mb",   "label": "Main Broker",     "type": "Broker",      "level": "CRITICAL", "value": 35},
        {"id": "nl",   "label": "NavLib",          "type": "Library",     "level": "MEDIUM",   "value": 25},
        {"id": "tele", "label": "Telemetry",       "type": "Topic",       "level": "LOW",      "value": 20},
    ]
    mock_edges = [
        {"source": "sf",   "target": "mb",   "weight": 2.5, "dependency_type": "publishes_to"},
        {"source": "pe",   "target": "mb",   "weight": 1.2, "dependency_type": "subscribes_to"},
        {"source": "mb",   "target": "tele", "weight": 1.0, "dependency_type": "broadcasts"},
    ]
    dash.add_cytoscape_network("demo-net", mock_nodes, mock_edges, "System Connectivity (Demo)")
    dash.end_section()
    
    dash.start_section("Dependency Matrix")
    dash.add_dependency_matrix("demo-matrix", mock_nodes, mock_edges)
    dash.end_section()
    dash.end_tab()

    # ── Tab 6: MIL-STD-498 ─────────────────────────────────────────────────
    dash.add_tab("MIL-STD-498", "hierarchy")
    dash.start_section("Hierarchy tree — BPA_\u03b2 rollup scores")
    dash.add_hierarchy_tree(demo_data.hierarchy_data)
    dash.end_section()
    dash.end_tab()

    # ── Write output ─────────────────────────────────────────────────────────
    display.print_step("Finalizing dashboard export...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dash.generate())

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
             "(e.g. results/val_s*.json). Used to render §8 stability panel.",
    )

    # --cascade-file: path to the qos_ablation_experiment.py output JSON.
    # Wires directly into VisualizationService.generate_dashboard(cascade_file=).
    parser.add_argument(
        "--cascade-file",
        metavar="JSON_PATH",
        help="Path to QoS ablation experiment JSON (output of "
             "qos_ablation_experiment.py). Enables §9a Cascade Risk section.",
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


if __name__ == "__main__":
    main()