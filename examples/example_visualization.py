#!/usr/bin/env python3
"""
Visualization Module Demo - Version 5.0

Demonstrates the visualization capabilities:
1. Chart generation
2. Dashboard building
3. Graph visualization
4. Simulation visualization
5. Validation visualization
6. Combined dashboards
7. CLI usage

Run:
    python examples/example_visualization.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Terminal Colors
# =============================================================================

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{title:^60}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 40)


def print_success(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{BLUE}ℹ{RESET} {msg}")


def print_code(code: str) -> None:
    print(f"  {GRAY}{code}{RESET}")


# =============================================================================
# Test Data
# =============================================================================

def create_demo_graph():
    """Create a demo graph for visualization."""
    from src.simulation import (
        SimulationGraph,
        Component,
        Edge,
        ComponentType,
        EdgeType,
        QoSPolicy,
    )
    
    graph = SimulationGraph()
    
    # Add nodes
    for i in range(3):
        graph.add_component(Component(
            id=f"node_{i+1}",
            type=ComponentType.NODE,
            name=f"Node {i+1}",
        ))
    
    # Add brokers
    for i in range(2):
        graph.add_component(Component(
            id=f"broker_{i+1}",
            type=ComponentType.BROKER,
            name=f"Broker {i+1}",
        ))
        # Connect to nodes
        node_idx = i % 3 + 1
        graph.add_edge(Edge(
            source=f"broker_{i+1}",
            target=f"node_{node_idx}",
            edge_type=EdgeType.RUNS_ON,
        ))
        graph.add_edge(Edge(
            source=f"node_{node_idx}",
            target=f"broker_{i+1}",
            edge_type=EdgeType.CONNECTS_TO,
        ))
    
    # Add topics
    topics = ["sensor_data", "commands", "alerts", "telemetry", "status"]
    for i, topic_name in enumerate(topics):
        graph.add_component(Component(
            id=f"topic_{i+1}",
            type=ComponentType.TOPIC,
            name=topic_name,
        ))
        broker_idx = i % 2 + 1
        graph.add_edge(Edge(
            source=f"broker_{broker_idx}",
            target=f"topic_{i+1}",
            edge_type=EdgeType.ROUTES,
        ))
    
    # Add applications
    app_names = ["sensor_hub", "controller", "monitor", "logger", "dashboard", "analytics"]
    for i, app_name in enumerate(app_names):
        graph.add_component(Component(
            id=f"app_{i+1}",
            type=ComponentType.APPLICATION,
            name=app_name,
        ))
        node_idx = i % 3 + 1
        graph.add_edge(Edge(
            source=f"app_{i+1}",
            target=f"node_{node_idx}",
            edge_type=EdgeType.RUNS_ON,
        ))
        
        # Publish/subscribe relationships
        pub_topic = i % len(topics) + 1
        sub_topic = (i + 2) % len(topics) + 1
        
        graph.add_edge(Edge(
            source=f"app_{i+1}",
            target=f"topic_{pub_topic}",
            edge_type=EdgeType.PUBLISHES_TO,
            qos=QoSPolicy(reliability="RELIABLE"),
        ))
        graph.add_edge(Edge(
            source=f"app_{i+1}",
            target=f"topic_{sub_topic}",
            edge_type=EdgeType.SUBSCRIBES_TO,
        ))
    
    return graph


# =============================================================================
# Demos
# =============================================================================

def demo_chart_generation():
    """Demo 1: Chart Generation"""
    print_header("DEMO 1: CHART GENERATION")
    
    from src.visualization import check_matplotlib_available
    
    if not check_matplotlib_available():
        print(f"  {YELLOW}matplotlib not installed{RESET}")
        print(f"  Install with: pip install matplotlib")
        print_success("Charts require matplotlib")
        return
    
    from src.visualization import (
        chart_component_distribution,
        chart_impact_ranking,
        chart_correlation_comparison,
        ChartConfig,
    )
    
    print("Generating charts from data...")
    print()
    
    # Component distribution
    components = {
        "Application": 10,
        "Broker": 2,
        "Topic": 5,
        "Node": 3,
    }
    
    chart = chart_component_distribution(components)
    print(f"  {BOLD}Component Distribution Chart{RESET}")
    print(f"    Title: {chart.title}")
    print(f"    Size: {chart.width}x{chart.height}")
    print(f"    SVG length: {len(chart.svg)} chars")
    print(f"    PNG size: {len(chart.png_base64)} bytes (base64)")
    
    # Impact ranking
    impacts = [
        ("broker_1", 0.85),
        ("node_1", 0.65),
        ("app_critical", 0.45),
        ("broker_2", 0.35),
        ("app_1", 0.15),
    ]
    
    chart = chart_impact_ranking(impacts)
    print(f"\n  {BOLD}Impact Ranking Chart{RESET}")
    print(f"    Title: {chart.title}")
    print(f"    Components shown: {len(impacts)}")
    
    # Correlation comparison
    metrics = {"Spearman": 0.85, "Pearson": 0.80, "Kendall": 0.75}
    targets = {"Spearman": 0.70, "Pearson": 0.65, "Kendall": 0.60}
    
    chart = chart_correlation_comparison(metrics, targets)
    print(f"\n  {BOLD}Correlation Comparison Chart{RESET}")
    print(f"    Metrics: {list(metrics.keys())}")
    
    print_success("Charts generated successfully")


def demo_dashboard_builder():
    """Demo 2: Dashboard Builder"""
    print_header("DEMO 2: DASHBOARD BUILDER")
    
    from src.visualization import DashboardBuilder, DashboardConfig
    
    print("Building a custom dashboard...")
    print()
    
    config = DashboardConfig(
        title="Custom Dashboard Demo",
        primary_color="#9b59b6",
    )
    
    builder = DashboardBuilder(config)
    
    # Add components
    builder.add_header("System Analysis", "Multi-layer graph analysis results")
    
    builder.add_stats_row({
        "Components": 100,
        "Edges": 250,
        "Critical": 5,
        "Status": "PASSED",
    }, "Overview Statistics")
    
    builder.add_section_title("Details")
    
    builder.start_grid()
    builder.add_table(
        headers=["Component", "Type", "Impact", "Critical"],
        rows=[
            ["broker_1", "Broker", 0.85, True],
            ["node_1", "Node", 0.65, True],
            ["app_1", "Application", 0.25, False],
        ],
        title="Top Components",
    )
    builder.add_table(
        headers=["Layer", "Spearman", "F1"],
        rows=[
            ["application", 0.85, 0.92],
            ["infrastructure", 0.78, 0.88],
        ],
        title="Layer Results",
    )
    builder.end_grid()
    
    html = builder.build()
    
    print(f"  Dashboard generated:")
    print(f"    HTML length: {len(html)} chars")
    print(f"    Contains header: {'<header>' in html}")
    print(f"    Contains tables: {'<table>' in html}")
    
    print_success("Dashboard built successfully")


def demo_graph_dashboard():
    """Demo 3: Graph Model Dashboard"""
    print_header("DEMO 3: GRAPH MODEL DASHBOARD")
    
    from src.visualization import generate_graph_dashboard
    
    graph = create_demo_graph()
    summary = graph.summary()
    
    print(f"  Graph Summary:")
    print(f"    Components: {summary['total_components']}")
    print(f"    Edges: {summary['total_edges']}")
    print(f"    Message Paths: {summary['message_paths']}")
    print()
    
    html = generate_graph_dashboard(graph, title="Demo Graph Dashboard")
    
    print(f"  Dashboard generated:")
    print(f"    HTML length: {len(html)} chars")
    
    print_success("Graph dashboard generated")
    
    return html


def demo_simulation_dashboard():
    """Demo 4: Simulation Dashboard"""
    print_header("DEMO 4: SIMULATION DASHBOARD")
    
    from src.simulation import FailureSimulator
    from src.visualization import generate_simulation_dashboard
    
    graph = create_demo_graph()
    
    print("  Running failure simulation...")
    simulator = FailureSimulator(seed=42, cascade=True)
    campaign = simulator.simulate_all(graph)
    
    print(f"\n  Simulation Results:")
    print(f"    Simulations: {campaign.total_simulations}")
    print(f"    Critical: {len(campaign.get_critical())}")
    
    # Calculate max impact from results
    impacts = [r.impact_score for r in campaign.results]
    max_impact = max(impacts) if impacts else 0
    print(f"    Max Impact: {max_impact:.4f}")
    print()
    
    # Top impacts
    print(f"  {BOLD}Top 5 by Impact:{RESET}")
    for comp_id, impact in campaign.ranked_by_impact()[:5]:
        color = RED if impact >= 0.5 else YELLOW if impact >= 0.3 else BLUE
        print(f"    {color}{comp_id:15s}{RESET}: {impact:.4f}")
    print()
    
    html = generate_simulation_dashboard(graph, campaign, title="Demo Simulation Dashboard")
    
    print(f"  Dashboard generated:")
    print(f"    HTML length: {len(html)} chars")
    
    print_success("Simulation dashboard generated")
    
    return html


def demo_validation_dashboard():
    """Demo 5: Validation Dashboard"""
    print_header("DEMO 5: VALIDATION DASHBOARD")
    
    from src.validation import ValidationPipeline
    from src.visualization import generate_validation_dashboard
    
    graph = create_demo_graph()
    
    print("  Running validation pipeline...")
    pipeline = ValidationPipeline(seed=42)
    result = pipeline.run(graph, compare_methods=True)
    
    status_color = GREEN if result.validation.status.value == "passed" else YELLOW
    print(f"\n  Validation Results:")
    print(f"    Status: {status_color}{result.validation.status.value.upper()}{RESET}")
    print(f"    Spearman ρ: {result.spearman:.4f}")
    print(f"    F1-Score: {result.f1_score:.4f}")
    
    if result.method_comparison:
        print(f"\n  {BOLD}Method Comparison:{RESET}")
        for method, comp in result.method_comparison.items():
            print(f"    {method:12s}: ρ={comp.spearman:.4f}, F1={comp.f1_score:.4f}")
    print()
    
    html = generate_validation_dashboard(result, title="Demo Validation Dashboard")
    
    print(f"  Dashboard generated:")
    print(f"    HTML length: {len(html)} chars")
    
    print_success("Validation dashboard generated")
    
    return html


def demo_overview_dashboard(open_browser: bool = False):
    """Demo 6: Combined Overview Dashboard"""
    print_header("DEMO 6: COMBINED OVERVIEW DASHBOARD")
    
    from src.simulation import FailureSimulator
    from src.validation import ValidationPipeline
    from src.visualization import generate_overview_dashboard
    
    graph = create_demo_graph()
    
    print("  Running simulation and validation...")
    
    simulator = FailureSimulator(seed=42, cascade=True)
    campaign = simulator.simulate_all(graph)
    
    pipeline = ValidationPipeline(seed=42)
    validation = pipeline.run(graph, compare_methods=True)
    
    summary = graph.summary()
    impacts = [r.impact_score for r in campaign.results]
    max_impact = max(impacts) if impacts else 0
    
    print(f"\n  {BOLD}System Overview:{RESET}")
    print(f"    Components: {summary['total_components']}")
    print(f"    Critical: {len(campaign.get_critical())}")
    print(f"    Max Impact: {max_impact:.4f}")
    print(f"    Validation: {validation.validation.status.value.upper()}")
    print()
    
    html = generate_overview_dashboard(
        graph,
        campaign_result=campaign,
        validation_result=validation,
        title="System Overview Dashboard",
    )
    
    print(f"  Dashboard generated:")
    print(f"    HTML length: {len(html)} chars")
    
    # Optionally save and open
    if open_browser:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            temp_path = f.name
        
        print(f"\n  Opening dashboard in browser...")
        print(f"    File: {temp_path}")
        webbrowser.open(f"file://{temp_path}")
    
    print_success("Overview dashboard generated")
    
    return html


def demo_cli_usage():
    """Demo 7: CLI Usage"""
    print_header("DEMO 7: CLI USAGE")
    
    print("The visualize_graph.py CLI provides these commands:")
    print()
    
    examples = [
        ("Graph model dashboard",
         "python visualize_graph.py --input graph.json --type graph"),
        ("Simulation dashboard",
         "python visualize_graph.py --input graph.json --type simulation"),
        ("Validation dashboard",
         "python visualize_graph.py --input graph.json --type validation"),
        ("Combined overview",
         "python visualize_graph.py --input graph.json --type overview"),
        ("From Neo4j",
         "python visualize_graph.py --neo4j --type overview"),
        ("Custom output and open",
         "python visualize_graph.py --input graph.json --output dash.html --open"),
        ("Compare methods",
         "python visualize_graph.py --input graph.json --type validation --compare-methods"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print_code(cmd)
        print()
    
    print_success("See --help for all options")


def demo_python_api():
    """Demo 8: Python API"""
    print_header("DEMO 8: PYTHON API")
    
    print("Using visualization in Python code:")
    print()
    
    code = '''
from src.simulation import SimulationGraph, FailureSimulator
from src.validation import ValidationPipeline
from src.visualization import (
    generate_overview_dashboard,
    generate_simulation_dashboard,
    chart_impact_ranking,
    DashboardConfig,
    ChartConfig,
)

# Load graph
graph = SimulationGraph.from_json("system.json")

# Run analysis
simulator = FailureSimulator(cascade=True)
campaign = simulator.simulate_all(graph)

pipeline = ValidationPipeline(seed=42)
validation = pipeline.run(graph, compare_methods=True)

# Generate dashboard with custom config
config = DashboardConfig(
    title="My System Dashboard",
    primary_color="#2ecc71",
)

html = generate_overview_dashboard(
    graph,
    campaign_result=campaign,
    validation_result=validation,
    config=config,
)

# Save to file
with open("dashboard.html", "w") as f:
    f.write(html)

# Generate individual chart
chart = chart_impact_ranking(
    campaign.ranked_by_impact(),
    top_n=10,
    title="Critical Components"
)

# Get embeddable HTML
img_html = chart.to_html_img()
svg_html = chart.to_html_svg()
'''
    
    print_code(code)
    print_success("Python API demonstrated")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualization Module Demo"
    )
    parser.add_argument(
        "--open", "-o",
        action="store_true",
        help="Open overview dashboard in browser"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print_header("Visualization Module Demo")
    
    # Check matplotlib
    from src.visualization import check_matplotlib_available
    if check_matplotlib_available():
        print_info("matplotlib is available - charts will be generated")
    else:
        print(f"  {YELLOW}matplotlib not installed - charts will be skipped{RESET}")
        print(f"  Install with: pip install matplotlib")
    print()
    
    # Run demos
    demo_chart_generation()
    demo_dashboard_builder()
    demo_graph_dashboard()
    demo_simulation_dashboard()
    demo_validation_dashboard()
    demo_overview_dashboard(open_browser=args.open)
    demo_cli_usage()
    demo_python_api()
    
    print_header("Demo Complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
