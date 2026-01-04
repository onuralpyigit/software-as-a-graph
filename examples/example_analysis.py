#!/usr/bin/env python3
"""
Graph Analysis Example - Version 6.0

Demonstrates the analysis module capabilities:
1. Box-plot classification
2. NetworkX-based graph algorithms
3. Component type analysis
4. Layer analysis
5. Edge criticality analysis

This example can run in demo mode (no Neo4j required) or
live mode with an actual Neo4j connection.

Usage:
    # Demo mode (no Neo4j required)
    python examples/example_analysis.py --demo
    
    # Live mode with Neo4j
    python examples/example_analysis.py --password your_password

Author: Software-as-a-Graph Research Project
Version: 6.0
"""

import argparse
import json
import sys
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
    for line in code.strip().split("\n"):
        print(f"  {GRAY}{line}{RESET}")


def level_color(level: str) -> str:
    colors = {"critical": RED, "high": YELLOW, "medium": BLUE}
    return colors.get(level.lower(), GRAY)


# =============================================================================
# Demo Mode (No Neo4j Required)
# =============================================================================

def run_demo_mode():
    """Run demonstration without Neo4j."""
    print_header("Graph Analysis Demo")
    
    print("This demo shows the analysis capabilities without Neo4j.")
    print("For live analysis, run with: --password <neo4j_password>")
    
    # Import modules
    from src.analysis import (
        BoxPlotClassifier,
        CriticalityLevel,
        classify_items,
        LAYER_DEFINITIONS,
        COMPONENT_TYPES,
        DEPENDENCY_TYPES,
    )
    
    # =========================================================================
    # Demo 1: Box-Plot Classification
    # =========================================================================
    print_section("1. Box-Plot Classification")
    
    print("The box-plot method uses statistical quartiles for adaptive thresholds:")
    print()
    print("  CRITICAL: > Q3 + 1.5×IQR  (upper outliers)")
    print("  HIGH:     > Q3            (top quartile)")
    print("  MEDIUM:   > Median        (above average)")
    print("  LOW:      > Q1            (below average)")
    print("  MINIMAL:  ≤ Q1            (bottom quartile)")
    
    # Sample data
    sample_scores = [
        {"id": "app_order_svc", "type": "Application", "score": 0.95},
        {"id": "app_payment", "type": "Application", "score": 0.82},
        {"id": "broker_main", "type": "Broker", "score": 0.78},
        {"id": "app_inventory", "type": "Application", "score": 0.65},
        {"id": "node_server_1", "type": "Node", "score": 0.58},
        {"id": "app_notify", "type": "Application", "score": 0.45},
        {"id": "node_server_2", "type": "Node", "score": 0.35},
        {"id": "app_logging", "type": "Application", "score": 0.22},
        {"id": "app_health", "type": "Application", "score": 0.12},
    ]
    
    print(f"\n  Sample composite scores:")
    for item in sample_scores:
        print(f"    {item['id']:20}: {item['score']:.2f}")
    
    # Classify
    classifier = BoxPlotClassifier(k_factor=1.5)
    result = classifier.classify(sample_scores, metric_name="composite")
    
    print(f"\n  Box-plot statistics:")
    print(f"    Q1 (25%):     {result.stats.q1:.3f}")
    print(f"    Median (50%): {result.stats.median:.3f}")
    print(f"    Q3 (75%):     {result.stats.q3:.3f}")
    print(f"    IQR:          {result.stats.iqr:.3f}")
    print(f"    Upper fence:  {result.stats.upper_fence:.3f}")
    
    print(f"\n  Classification results:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = result.summary().get(level, 0)
        color = level_color(level)
        bar = "█" * count + "░" * (5 - count)
        print(f"    {color}{level.upper():10}{RESET}: [{bar}] {count}")
    
    print(f"\n  Critical components:")
    for item in result.get_critical():
        print(f"    {RED}→{RESET} {item.id}: {item.score:.3f} (outlier: {item.is_outlier})")
    
    if not result.get_critical():
        print(f"    {GRAY}(none - no statistical outliers){RESET}")
    
    print(f"\n  High and above:")
    for item in result.get_high_and_above():
        color = level_color(item.level.value)
        print(f"    {color}→{RESET} {item.id}: {item.score:.3f} [{item.level.value}]")
    
    # =========================================================================
    # Demo 2: NetworkX Analyzer
    # =========================================================================
    print_section("2. NetworkX Graph Algorithms")
    
    print("NetworkX algorithms compute centrality metrics:")
    print()
    print("  • PageRank: Transitive importance via random walk")
    print("  • Betweenness: How often node is on shortest paths")
    print("  • Degree: Number of direct connections")
    print("  • Articulation Points: Nodes whose removal disconnects graph")
    print("  • Bridges: Edges whose removal disconnects graph")
    
    # Create mock graph
    from src.analysis import (
        NetworkXAnalyzer,
        ComponentData,
        EdgeData,
        GraphData,
    )
    
    components = [
        ComponentData(id="app_1", component_type="Application", weight=1.0),
        ComponentData(id="app_2", component_type="Application", weight=1.5),
        ComponentData(id="app_3", component_type="Application", weight=0.8),
        ComponentData(id="broker_1", component_type="Broker", weight=2.0),
    ]
    
    edges = [
        EdgeData("app_1", "app_2", "Application", "Application", "app_to_app", 1.5),
        EdgeData("app_2", "app_3", "Application", "Application", "app_to_app", 1.0),
        EdgeData("app_1", "app_3", "Application", "Application", "app_to_app", 0.5),
        EdgeData("app_1", "broker_1", "Application", "Broker", "app_to_broker", 2.0),
        EdgeData("app_2", "broker_1", "Application", "Broker", "app_to_broker", 1.5),
    ]
    
    graph_data = GraphData(components=components, edges=edges)
    
    print(f"\n  Mock graph: {len(components)} components, {len(edges)} edges")
    
    # Analyze with NetworkX
    analyzer = NetworkXAnalyzer(k_factor=1.5)
    app_result = analyzer.analyze_component_type(graph_data, "Application")
    
    print(f"\n  Application analysis results:")
    for comp in app_result.components:
        ap_mark = " [AP]" if comp.is_articulation_point else ""
        color = level_color(comp.level.value)
        print(f"    {color}{comp.component_id}{RESET}: "
              f"PR={comp.pagerank:.4f}, BC={comp.betweenness:.4f}, "
              f"Composite={comp.composite_score:.4f}{ap_mark}")
    
    # =========================================================================
    # Demo 3: Layer Definitions
    # =========================================================================
    print_section("3. Multi-Layer Analysis")
    
    print("The system supports multiple dependency layers:")
    print()
    for layer_key, layer_def in LAYER_DEFINITIONS.items():
        print(f"  {BOLD}{layer_key}{RESET}:")
        print(f"    Name: {layer_def['name']}")
        print(f"    Component Types: {layer_def['component_types']}")
        print(f"    Dependency Types: {layer_def['dependency_types']}")
        print()
    
    # =========================================================================
    # Demo 4: Component Types
    # =========================================================================
    print_section("4. Component Types")
    
    print("Components are analyzed separately by type for fair comparison:")
    print()
    for comp_type in COMPONENT_TYPES:
        print(f"  • {comp_type}")
    
    print(f"\n  This ensures Applications are compared to Applications,")
    print(f"  Brokers to Brokers, etc., using type-specific statistics.")
    
    # =========================================================================
    # Demo 5: CLI Usage
    # =========================================================================
    print_section("5. CLI Usage")
    
    print("The analyze_graph.py CLI provides these commands:")
    print()
    
    examples = [
        ("Full analysis", "python analyze_graph.py"),
        ("Single type", "python analyze_graph.py --type Application"),
        ("All types", "python analyze_graph.py --all-types"),
        ("Single layer", "python analyze_graph.py --layer application"),
        ("All layers", "python analyze_graph.py --all-layers"),
        ("Edge analysis", "python analyze_graph.py --edges"),
        ("Export JSON", "python analyze_graph.py --output results.json"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print_code(cmd)
        print()
    
    # =========================================================================
    # Demo 6: Python API
    # =========================================================================
    print_section("6. Python API")
    
    print("Use the GraphAnalyzer class directly:")
    print()
    
    print_code("""
from src.analysis import GraphAnalyzer

with GraphAnalyzer(uri, user, password) as analyzer:
    # Analyze by component type
    apps = analyzer.analyze_component_type("Application")
    
    # Analyze by layer
    app_layer = analyzer.analyze_layer("application")
    
    # Full analysis
    result = analyzer.analyze_full()
    
    # Get critical components
    for comp in apps.get_critical():
        print(f"{comp.component_id}: {comp.composite_score:.4f}")
""")
    
    print_header("Demo Complete")
    
    print("To run with Neo4j:")
    print_code("""
# 1. Start Neo4j
docker run -d --name neo4j \\
    -p 7474:7474 -p 7687:7687 \\
    -e NEO4J_AUTH=neo4j/password \\
    neo4j:latest

# 2. Import graph data
python import_graph.py --input graph.json

# 3. Run analysis
python analyze_graph.py --password password
""")


# =============================================================================
# Live Mode (With Neo4j)
# =============================================================================

def run_live_mode(args):
    """Run with actual Neo4j connection."""
    print_header("Graph Analysis - Live Mode")
    
    from src.analysis import GraphAnalyzer, COMPONENT_TYPES
    
    print_info(f"Connecting to Neo4j at {args.uri}")
    
    try:
        with GraphAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            k_factor=args.k_factor,
        ) as analyzer:
            
            # Get stats
            stats = analyzer.get_graph_stats()
            print_success(f"Connected. Found {stats.get('total_nodes', 0)} nodes, "
                         f"{stats.get('total_edges', 0)} edges")
            
            # Run full analysis
            print_info("Running full analysis...")
            result = analyzer.analyze_full()
            
            # Print results
            print_section("Analysis Results")
            
            # Component types
            print(f"\n{BOLD}Component Type Analysis:{RESET}")
            for comp_type, type_result in result.component_types.items():
                critical = len(type_result.get_critical())
                total = len(type_result.components)
                aps = len(type_result.articulation_points)
                color = RED if critical > 0 else GREEN
                print(f"  {comp_type:15}: {total:4} total, {color}{critical:3} critical{RESET}, {aps} APs")
            
            # Layers
            print(f"\n{BOLD}Layer Analysis:{RESET}")
            for layer_key, layer_result in result.layers.items():
                critical = len(layer_result.get_critical())
                total = len(layer_result.components)
                bridges = len(layer_result.bridges)
                color = RED if critical > 0 else GREEN
                print(f"  {layer_key:15}: {total:4} total, {color}{critical:3} critical{RESET}, {bridges} bridges")
            
            # Edges
            if result.edges:
                print(f"\n{BOLD}Edge Analysis:{RESET}")
                critical_edges = len(result.edges.get_critical())
                total_edges = len(result.edges.edges)
                bridge_count = len(result.edges.get_bridges())
                color = RED if critical_edges > 0 else GREEN
                print(f"  Total: {total_edges}, {color}Critical: {critical_edges}{RESET}, Bridges: {bridge_count}")
            
            # Summary
            print_section("Summary")
            summary = result.summary
            print(f"  {RED}Total Critical Components: {summary.get('total_critical_components', 0)}{RESET}")
            print(f"  {RED}Total Critical Edges: {summary.get('total_critical_edges', 0)}{RESET}")
            
            # Export if requested
            if args.output:
                output_path = Path(args.output)
                with open(output_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print_success(f"Results exported to {output_path}")
            
            print_header("Analysis Complete")
            return 0
    
    except ConnectionError as e:
        print(f"{RED}Connection failed:{RESET} {e}")
        return 1
    except Exception as e:
        print(f"{RED}Analysis failed:{RESET} {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Graph Analysis Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Demo mode (no Neo4j required)
    python examples/example_analysis.py --demo
    
    # Live mode
    python examples/example_analysis.py --password your_password
    
    # Export results
    python examples/example_analysis.py --password your_password --output results.json
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode without Neo4j",
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password",
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=1.5,
        help="Box-plot k factor",
    )
    parser.add_argument(
        "--output", "-o",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_mode()
        return 0
    else:
        return run_live_mode(args)


if __name__ == "__main__":
    sys.exit(main())