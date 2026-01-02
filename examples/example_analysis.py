#!/usr/bin/env python3
"""
Graph Analysis Example - Version 5.0

Demonstrates the analysis module capabilities:
1. Box-plot classification
2. Multi-layer analysis
3. Component type analysis
4. Edge criticality analysis

This example can run in demo mode (no Neo4j required) or
live mode with an actual Neo4j connection.

Usage:
    # Demo mode (no Neo4j required)
    python examples/example_analysis.py --demo
    
    # Live mode with Neo4j
    python examples/example_analysis.py --password your_password

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Terminal Output
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
    print(f"{'-' * 40}")


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
# Demo Mode
# =============================================================================

def run_demo_mode():
    """Run demonstration without Neo4j."""
    print_header("Graph Analysis Demo")
    
    print("This demo shows the analysis capabilities without Neo4j.")
    print("For live analysis, run with: --password <neo4j_password>")
    
    # Import analysis module
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
    
    # Create sample data
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
    # Demo 2: Multi-Layer Analysis Concept
    # =========================================================================
    print_section("2. Multi-Layer Analysis")
    
    print("The system is analyzed by dependency layer:")
    print()
    
    for layer_key, layer_def in LAYER_DEFINITIONS.items():
        print(f"  {BOLD}{layer_key:15}{RESET}: {layer_def['name']}")
        print(f"    Components: {', '.join(layer_def['component_types'])}")
        print(f"    Dependencies: {', '.join(layer_def['dependency_types'])}")
        print()
    
    print("Each layer is analyzed independently with its own statistics.")
    
    # =========================================================================
    # Demo 3: Component Type Analysis Concept
    # =========================================================================
    print_section("3. Component Type Analysis")
    
    print("Components are compared within their type for fair evaluation:")
    print()
    
    for comp_type in COMPONENT_TYPES:
        print(f"  {BOLD}{comp_type}{RESET}")
        print(f"    - Compared with other {comp_type}s only")
        print(f"    - Gets own quartile statistics")
        print(f"    - Fair classification within type")
    
    print(f"\n  Why type-specific analysis?")
    print(f"    - Brokers naturally have higher betweenness")
    print(f"    - Topics have different connectivity patterns")
    print(f"    - Cross-type comparison can be misleading")
    
    # =========================================================================
    # Demo 4: Edge Criticality Concept
    # =========================================================================
    print_section("4. Edge Criticality Analysis")
    
    print("Edges are analyzed for criticality based on:")
    print()
    print("  {BOLD}Weight{RESET} (35%)")
    print("    - QoS-based dependency weight")
    print("    - Higher weight = more critical dependency")
    print()
    print("  {BOLD}Bridge Status{RESET} (40%)")
    print("    - Only path between components")
    print("    - Removal would disconnect the graph")
    print()
    print("  {BOLD}Endpoint Criticality{RESET} (25%)")
    print("    - Connects to critical components")
    print("    - Inherits importance from endpoints")
    
    print(f"\n  Criticality formula:")
    print_code("""score = 0.35 × normalized_weight
     + 0.40 × is_bridge
     + 0.25 × connects_critical""")
    
    # =========================================================================
    # Demo 5: CLI Usage
    # =========================================================================
    print_section("5. CLI Usage")
    
    print("The analyze_graph.py CLI provides these commands:")
    print()
    
    examples = [
        ("Full analysis", "python analyze_graph.py --full"),
        ("Single layer", "python analyze_graph.py --layer application"),
        ("All layers", "python analyze_graph.py --all-layers"),
        ("Component type", "python analyze_graph.py --type Broker"),
        ("All types", "python analyze_graph.py --all-types"),
        ("Edge analysis", "python analyze_graph.py --edges"),
        ("Export JSON", "python analyze_graph.py --full --output results.json"),
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
    
    print_code("""from src.analysis import GraphAnalyzer

with GraphAnalyzer(uri, user, password) as analyzer:
    # Full analysis
    result = analyzer.analyze_full()
    
    # Single layer
    app_layer = analyzer.analyze_layer("application")
    
    # Component type
    brokers = analyzer.analyze_component_type("Broker")
    
    # Get critical components
    for comp in brokers.get_critical():
        print(f"{comp.component_id}: {comp.composite_score:.4f}")""")
    
    print_header("Demo Complete")
    
    print("To run with Neo4j:")
    print_code("""# 1. Start Neo4j with GDS
docker run -d --name neo4j \\
    -p 7474:7474 -p 7687:7687 \\
    -e NEO4J_AUTH=neo4j/password \\
    -e NEO4J_PLUGINS='["graph-data-science"]' \\
    neo4j:latest

# 2. Import graph data
python import_graph.py --input graph.json

# 3. Run analysis
python analyze_graph.py --full""")


# =============================================================================
# Live Mode
# =============================================================================

def run_live_mode(args):
    """Run with actual Neo4j connection."""
    print_header("Graph Analysis - Live Mode")
    
    try:
        from src.analysis import GraphAnalyzer, CriticalityLevel
    except ImportError as e:
        print(f"{RED}✗{RESET} Import failed: {e}")
        print_info("Install neo4j driver: pip install neo4j")
        return 1
    
    try:
        with GraphAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            k_factor=args.k_factor,
        ) as analyzer:
            
            print_success(f"Connected to {args.uri}")
            
            # Get graph stats
            print_section("Graph Statistics")
            stats = analyzer.get_graph_stats()
            print(f"  Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"  Total DEPENDS_ON: {stats.get('total_relationships', 0)}")
            
            if stats.get("nodes"):
                print("\n  By Node Type:")
                for node_type, count in stats["nodes"].items():
                    print(f"    {node_type}: {count}")
            
            # Analyze all component types
            print_section("Component Type Analysis")
            
            all_types = analyzer.analyze_all_component_types()
            
            for comp_type, type_result in all_types.by_type.items():
                critical_count = len(type_result.get_critical())
                total = len(type_result.components)
                
                print(f"\n  {BOLD}{comp_type}{RESET}: {total} components")
                print(f"    Critical: {critical_count}")
                
                if type_result.get_critical():
                    print(f"    Top critical:")
                    for comp in type_result.get_critical()[:3]:
                        ap = f" {YELLOW}[AP]{RESET}" if comp.is_articulation_point else ""
                        print(f"      {comp.component_id}: {comp.composite_score:.4f}{ap}")
            
            # Analyze layers
            print_section("Layer Analysis")
            
            layers = analyzer.analyze_all_layers()
            
            for layer_key, layer_result in layers.layers.items():
                critical_count = len(layer_result.get_critical_components())
                total = len(layer_result.metrics)
                
                print(f"\n  {BOLD}{layer_key}{RESET}: {total} components")
                print(f"    Critical: {critical_count}")
            
            # Edge analysis
            print_section("Edge Analysis")
            
            edges = analyzer.analyze_edges()
            
            print(f"  Total Edges: {len(edges.edges)}")
            print(f"  Critical: {len(edges.get_critical())}")
            print(f"  Bridges: {len(edges.get_bridges())}")
            
            if edges.get_critical():
                print(f"\n  Critical Edges:")
                for edge in edges.get_critical()[:5]:
                    bridge = f" {YELLOW}[BRIDGE]{RESET}" if edge.is_bridge else ""
                    print(f"    {edge.source_id} → {edge.target_id}: {edge.criticality_score:.4f}{bridge}")
            
            # Export if requested
            if args.output:
                result = analyzer.analyze_full()
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                
                print_success(f"Results exported to {output_path}")
            
            print_header("Analysis Complete")
            return 0
    
    except Exception as e:
        print(f"{RED}✗{RESET} Analysis failed: {e}")
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
