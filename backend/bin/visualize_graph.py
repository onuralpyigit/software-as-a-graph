#!/usr/bin/env python3
"""
Graph Visualization CLI (Step 6)

Generates multi-layer analysis dashboards using the VisualizationService.

Usage:
    # Single layer
    python bin/visualize_graph.py --layer system --output dashboard.html

    # Multiple layers
    python bin/visualize_graph.py --layers app,infra,system --output dashboard.html

    # All layers, open in browser
    python bin/visualize_graph.py --all --output dashboard.html --open

    # Demo mode (no Neo4j required)
    python bin/visualize_graph.py --demo --output demo_dashboard.html
"""
import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import create_repository
from src.visualization import VisualizationService, LAYER_DEFINITIONS
from src.cli.console import ConsoleDisplay


def display_available_layers(display) -> None:
    """Display available layers with descriptions."""
    display.print_subheader("Available Layers")
    for layer, definition in LAYER_DEFINITIONS.items():
        print(
            f"  {display.colored(f'{layer:<10}', display.Colors.CYAN)} "
            f"{definition['icon']} {definition['name']}"
        )
        print(
            f"  {' ' * 10} "
            f"{display.colored(definition['description'], display.Colors.GRAY)}"
        )
    print()


def run_demo(output_file: str, open_browser: bool) -> int:
    """
    Generate a demo dashboard with sample data (no Neo4j required).

    Creates a dashboard with synthetic data to demonstrate dashboard
    features and visual encodings without requiring a database connection.
    """
    try:
        from src.visualization import LayerData, ComponentDetail
        from src.visualization.charts import ChartGenerator
        from src.visualization.dashboard import DashboardGenerator

        print("Generating demo dashboard (no Neo4j required)...")

        dash = DashboardGenerator("Software-as-a-Graph Demo Dashboard")
        charts = ChartGenerator()

        # Create sample layer data
        demo_data = LayerData(
            layer="system",
            name="Demo System",
            nodes=48,
            edges=127,
            density=0.056,
            critical_count=5,
            high_count=8,
            medium_count=15,
            low_count=12,
            minimal_count=8,
            spof_count=3,
            problems_count=2,
            spearman=0.876,
            f1_score=0.923,
            precision=0.912,
            recall=0.857,
            top5_overlap=0.80,
            top10_overlap=0.70,
            validation_passed=True,
            event_throughput=1000,
            event_delivery_rate=98.5,
            max_impact=0.734,
        )

        # Add sample component details
        demo_data.component_details = [
            ComponentDetail("sensor_fusion", "Sensor Fusion", "Application",
                           0.82, 0.88, 0.90, 0.75, 0.84, "CRITICAL", 0.79),
            ComponentDetail("main_broker", "Main Broker", "Broker",
                           0.78, 0.65, 0.95, 0.80, 0.80, "CRITICAL", 0.73),
            ComponentDetail("planning_node", "Planning Node", "Application",
                           0.71, 0.73, 0.45, 0.68, 0.64, "HIGH", 0.58),
            ComponentDetail("gateway", "API Gateway", "Application",
                           0.65, 0.70, 0.42, 0.60, 0.59, "HIGH", 0.52),
            ComponentDetail("perception", "Perception Engine", "Application",
                           0.60, 0.55, 0.38, 0.62, 0.54, "MEDIUM", 0.45),
        ]

        demo_data.scatter_data = [
            (c.id, c.overall, c.impact, c.level)
            for c in demo_data.component_details
        ]

        # Build a simple dashboard
        dash.start_section("📊 Demo Overview", "overview")
        dash.add_kpis(
            {
                "Nodes": 48,
                "Edges": 127,
                "Critical": 5,
                "SPOFs": 3,
                "Anti-Patterns": 2,
            },
            {"Critical": "danger", "SPOFs": "warning"},
        )

        chart_list = []
        c = charts.criticality_distribution(demo_data.classification_distribution)
        if c:
            chart_list.append(c)
        c = charts.rmav_breakdown(demo_data.component_details, "RMAV Breakdown")
        if c:
            chart_list.append(c)
        c = charts.correlation_scatter(demo_data.scatter_data, spearman=0.876)
        if c:
            chart_list.append(c)
        dash.add_charts(chart_list)
        dash.end_section()

        # Write output
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = dash.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        abs_path = os.path.abspath(output_path)
        print(f"✓ Demo dashboard generated: {abs_path}")

        if open_browser:
            webbrowser.open(f"file://{abs_path}")

        return 0

    except Exception as e:
        print(f"✗ Demo generation failed: {e}")
        logging.exception("Demo generation failed")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-layer graph analysis dashboards (Step 6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --layer system                    Single layer dashboard
  %(prog)s --layers app,infra,system         Multi-layer dashboard
  %(prog)s --all --open                      All layers, open in browser
  %(prog)s --demo --output demo.html         Demo mode (no Neo4j)
  %(prog)s --all --no-network                Skip network graphs (faster)
  %(prog)s --all --no-matrix --no-network    Charts and tables only
        """,
    )

    layer_group = parser.add_argument_group("Layer Selection")
    layer_mutex = layer_group.add_mutually_exclusive_group(required=True)
    layer_mutex.add_argument(
        "--layers", "-l",
        help="Comma-separated layers (e.g., app,infra,system)",
    )
    layer_mutex.add_argument(
        "--layer",
        choices=list(LAYER_DEFINITIONS.keys()),
        help="Single layer",
    )
    layer_mutex.add_argument(
        "--all", "-a", action="store_true",
        help="All layers",
    )
    layer_mutex.add_argument(
        "--list-layers", action="store_true",
        help="List layers and exit",
    )
    layer_mutex.add_argument(
        "--demo", action="store_true",
        help="Generate demo dashboard with sample data (no Neo4j required)",
    )

    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri", default="bolt://localhost:7687", help="Neo4j URI"
    )
    neo4j_group.add_argument(
        "--user", "-u", default="neo4j", help="Neo4j username"
    )
    neo4j_group.add_argument(
        "--password", "-p", default="password", help="Neo4j password"
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o", default="dashboard.html", help="Output file"
    )
    output_group.add_argument(
        "--no-network", action="store_true",
        help="Exclude interactive network graphs",
    )
    output_group.add_argument(
        "--no-matrix", action="store_true",
        help="Exclude dependency matrices",
    )
    output_group.add_argument(
        "--no-validation", action="store_true",
        help="Exclude validation metrics",
    )
    output_group.add_argument(
        "--open", "-O", action="store_true",
        help="Open in browser after generation",
    )

    args = parser.parse_args()

    # ── Demo Mode ────────────────────────────────────────────────────────
    if args.demo:
        return run_demo(args.output, args.open)

    # ── Normal Mode ──────────────────────────────────────────────────────
    # Initialize repository and services
    repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    display = ConsoleDisplay()

    if args.list_layers:
        display_available_layers(display)
        repo.close()
        return 0

    # Determine layers
    if args.all:
        layers = list(LAYER_DEFINITIONS.keys())
    elif args.layer:
        layers = [args.layer]
    else:
        layers = [l.strip() for l in args.layers.split(",")]

    valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
    if not valid_layers:
        print(display.colored("✗ No valid layers specified", display.Colors.RED))
        repo.close()
        return 1

    display.print_header("SOFTWARE-AS-A-GRAPH VISUALIZATION", "═")
    print(
        f"\n  {display.colored('Neo4j:', display.Colors.CYAN)}  {args.uri}"
    )
    print(
        f"  {display.colored('Output:', display.Colors.CYAN)} {args.output}"
    )
    print(
        f"  {display.colored('Layers:', display.Colors.CYAN)} "
        f"{', '.join(valid_layers)}"
    )

    # Display options
    options = []
    if args.no_network:
        options.append("no-network")
    if args.no_matrix:
        options.append("no-matrix")
    if args.no_validation:
        options.append("no-validation")
    if options:
        print(
            f"  {display.colored('Flags:', display.Colors.CYAN)}  "
            f"{', '.join(options)}"
        )

    try:
        from src.analysis import AnalysisService
        from src.simulation import SimulationService
        from src.validation import ValidationService
        from src.visualization import VisualizationService
        
        analysis_service = AnalysisService(repo)
        simulation_service = SimulationService(repo)
        validation_service = ValidationService(analysis_service, simulation_service)
        
        viz_service = VisualizationService(
            analysis_service=analysis_service,
            simulation_service=simulation_service,
            validation_service=validation_service,
            repository=repo
        )
        
        output_path = viz_service.generate_dashboard(
            output_file=args.output,
            layers=valid_layers,
            include_network=not args.no_network,
            include_matrix=not args.no_matrix,
            include_validation=not args.no_validation,
        )

        display.print_header("DASHBOARD GENERATED", "-")
        abs_path = os.path.abspath(output_path)
        file_size = os.path.getsize(abs_path) / 1024
        print(
            f"\n  {display.colored('File:', display.Colors.GREEN)} {abs_path}"
        )
        print(
            f"  {display.colored('Size:', display.Colors.GREEN)} "
            f"{file_size:.0f} KB"
        )

        if args.open:
            webbrowser.open(f"file://{abs_path}")

        return 0

    except Exception as e:
        print(display.colored(f"\n✗ Error: {e}", display.Colors.RED))
        logging.exception("Visualization failed")
        return 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())