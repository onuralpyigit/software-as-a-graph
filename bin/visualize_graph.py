#!/usr/bin/env python3
"""
Graph Visualization CLI (Step 6)
================================
Generates multi-layer analysis dashboards using the VisualizationService.
"""

import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import webbrowser
import os
from saag import Client
from bin._shared import add_neo4j_args, add_common_args, setup_logging
from bin.common.console import ConsoleDisplay

def run_demo(output_file: str, open_browser: bool) -> int:
    """Generate a demo dashboard with sample data (no Neo4j required)."""
    display = ConsoleDisplay()
    display.print_header("Software-as-a-Graph Demo Mode")
    display.print_step("Generating mock analysis data...")

    from src.visualization import LayerData, ComponentDetail
    from src.visualization.charts import ChartGenerator
    from src.visualization.dashboard import DashboardGenerator
    
    dash = DashboardGenerator("Software-as-a-Graph Demo Dashboard")
    charts = ChartGenerator()
    
    demo_data = LayerData(
        layer="system", name="Demo System", nodes=48, edges=127, density=0.056,
        critical_count=5, high_count=8, medium_count=15, low_count=12, minimal_count=8,
        spof_count=3, problems_count=2, spearman=0.876, f1_score=0.923,
        precision=0.912, recall=0.857, top5_overlap=0.80, top10_overlap=0.70,
        validation_passed=True, event_throughput=1000, event_delivery_rate=98.5, max_impact=0.734,
    )
    
    demo_data.component_details = [
        ComponentDetail("sensor_fusion", "Sensor Fusion", "Application", 0.82, 0.88, 0.90, 0.75, 0.84, "CRITICAL", 0.79),
        ComponentDetail("planning_engine", "Planning Engine", "Application", 0.75, 0.81, 0.85, 0.60, 0.72, "HIGH", 0.65),
    ]
    demo_data.scatter_data = [(c.id, c.overall, c.impact, c.level) for c in demo_data.component_details]
    
    dash.start_section("📊 Demo Overview", "overview")
    dash.add_kpis({"Nodes": 48,"Edges": 127,"Critical": 5,"SPOFs": 3,"Anti-Patterns": 2}, {"Critical": "danger", "SPOFs": "warning"})
    
    display.print_step("Assembling interactive charts...")
    chart_list = []
    if c := charts.criticality_distribution(demo_data.classification_distribution): chart_list.append(c)
    if c := charts.rmav_breakdown(demo_data.component_details, "RMAV Breakdown"): chart_list.append(c)
    if c := charts.correlation_scatter(demo_data.scatter_data, spearman=0.876): chart_list.append(c)
    dash.add_charts(chart_list)
    
    display.print_step("Adding interactive network visualization...")
    # Add a simple mock network
    mock_nodes = [
        {"id": "sf", "label": "Sensor Fusion", "type": "Application", "level": "CRITICAL", "value": 0.84},
        {"id": "pe", "label": "Planning Engine", "type": "Application", "level": "HIGH", "value": 0.72},
        {"id": "mb", "label": "Main Broker", "type": "Broker", "level": "CRITICAL", "value": 0.80},
        {"id": "db", "label": "Database", "type": "Node", "level": "LOW", "value": 0.3},
    ]
    mock_edges = [
        {"source": "sf", "target": "mb", "weight": 2.5, "dependency_type": "publishes_to"},
        {"source": "pe", "target": "mb", "weight": 1.2, "dependency_type": "subscribes_to"},
        {"source": "sf", "target": "pe", "weight": 0.8, "dependency_type": "uses"},
        {"source": "mb", "target": "db", "weight": 1.0, "dependency_type": "connects_to"},
    ]
    dash.add_cytoscape_network("demo-net", mock_nodes, mock_edges, "System Connectivity (Mock)")
    
    dash.end_section()
    
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
    
    # custom layer flags are handled individually
    parser.add_argument("--demo", action="store_true", help="Generate demo dashboard")
    parser.add_argument("--no-network", action="store_true", help="Exclude interactive network graphs")
    parser.add_argument("--no-matrix", action="store_true", help="Exclude dependency matrices")
    parser.add_argument("--no-validation", action="store_true", help="Exclude validation metrics")
    parser.add_argument("--antipatterns", help="Path to pre-calculated anti-pattern JSON report")
    parser.add_argument("--multi-seed", type=int, default=0, help="Run validation across N random seeds")
    parser.add_argument("--open", "-O", action="store_true", help="Open in browser after generation")
    
    add_neo4j_args(parser)
    add_common_args(parser)
    
    args, unknown = parser.parse_known_args()
    display = ConsoleDisplay()
    
    if args.demo:
        return run_demo(args.output if args.output else "dashboard.html", args.open)

    display.print_header("Analysis Dashboard Generation")
    
    # Layer handling
    layers = [l.strip() for l in args.layer.split(",")] if args.layer else ["system"]
    
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    display.print_step(f"Generating dashboard for layers: {', '.join(layers)}")
    out_path = client.visualize(
        output=args.output if args.output else "dashboard.html",
        layers=layers,
        include_network=not args.no_network,
        include_matrix=not args.no_matrix,
        include_validation=not args.no_validation,
        antipatterns_file=args.antipatterns,
        multi_seed=args.multi_seed
    )
    
    display.display_visualization_summary(out_path)
    
    if args.open:
        abs_path = os.path.abspath(out_path)
        webbrowser.open(f"file://{abs_path}")

if __name__ == "__main__":
    main()