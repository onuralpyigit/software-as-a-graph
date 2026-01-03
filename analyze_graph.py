#!/usr/bin/env python3
"""
Analyze Graph CLI - Version 5.0

Command-line interface for multi-layer graph analysis.

Features:
- Layer analysis (app_to_app, node_to_node, app_to_broker, node_to_broker)
- Component type analysis (Application, Broker, Node, Topic)
- Edge criticality analysis
- Box-plot statistical classification
- Weighted algorithm support

Usage:
    # Full analysis
    python analyze_graph.py
    
    # Analyze specific layer
    python analyze_graph.py --layer application
    python analyze_graph.py --layer infrastructure
    
    # Analyze component type
    python analyze_graph.py --type Application
    python analyze_graph.py --type Broker
    
    # Analyze all types
    python analyze_graph.py --all-types
    
    # Analyze edges
    python analyze_graph.py --edges
    
    # Export results
    python analyze_graph.py --output results.json

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    
    @classmethod
    def disable(cls):
        for attr in ["BOLD", "GREEN", "BLUE", "CYAN", "YELLOW", "RED", "GRAY", "RESET"]:
            setattr(cls, attr, "")


def use_colors() -> bool:
    """Check if terminal supports colors."""
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key}: {value}")


def level_color(level: str) -> str:
    """Get color for criticality level."""
    colors = {
        "critical": Colors.RED,
        "high": Colors.YELLOW,
        "medium": Colors.BLUE,
        "low": Colors.GRAY,
        "minimal": Colors.GRAY,
    }
    return colors.get(level.lower(), "")


# =============================================================================
# Result Formatters
# =============================================================================

def print_layer_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print layer analysis result."""
    print_kv("Layer", result.get("layer_name", "N/A"))
    print_kv("Components", result.get("component_count", 0))
    
    summary = result.get("summary", {})
    by_level = summary.get("by_level", {})
    
    print(f"\n  Distribution:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = by_level.get(level, 0)
        color = level_color(level)
        print(f"    {color}{level.upper():10}{Colors.RESET}: {count}")
    
    # Show critical components
    metrics = result.get("metrics", [])
    critical = [m for m in metrics if m.get("level") == "critical"]
    
    if critical:
        print(f"\n  {Colors.RED}Critical Components:{Colors.RESET}")
        for m in critical[:10]:
            ap_mark = f" {Colors.YELLOW}[AP]{Colors.RESET}" if m.get("is_articulation_point") else ""
            print(f"    {m['id']}: {m['composite_score']:.4f}{ap_mark}")
    
    if verbose:
        # Show top by each metric
        top_5 = sorted(metrics, key=lambda x: x.get("betweenness", 0), reverse=True)[:5]
        if top_5:
            print(f"\n  Top by Betweenness:")
            for m in top_5:
                print(f"    {m['id']}: {m.get('betweenness', 0):.4f}")


def print_component_type_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print component type analysis result."""
    print_kv("Type", result.get("component_type", "N/A"))
    print_kv("Count", result.get("count", 0))
    
    summary = result.get("summary", {})
    by_level = summary.get("by_level", {})
    
    print(f"\n  Distribution:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = by_level.get(level, 0)
        color = level_color(level)
        print(f"    {color}{level.upper():10}{Colors.RESET}: {count}")
    
    # Show critical
    components = result.get("components", [])
    critical = [c for c in components if c.get("level") == "critical"]
    
    if critical:
        print(f"\n  {Colors.RED}Critical:{Colors.RESET}")
        for c in critical[:10]:
            ap_mark = f" {Colors.YELLOW}[AP]{Colors.RESET}" if c.get("is_articulation_point") else ""
            print(f"    {c['id']}: {c['composite_score']:.4f}{ap_mark}")


def print_edge_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print edge analysis result."""
    summary = result.get("summary", {})
    
    print_kv("Total Edges", summary.get("total_edges", 0))
    print_kv("Bridges", summary.get("bridge_count", 0))
    print_kv("Connects Critical", summary.get("connects_critical_count", 0))
    
    by_level = summary.get("by_level", {})
    print(f"\n  By Criticality:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = by_level.get(level, 0)
        color = level_color(level)
        print(f"    {color}{level.upper():10}{Colors.RESET}: {count}")
    
    # Show critical edges
    edges = result.get("edges", [])
    critical = [e for e in edges if e.get("level") == "critical"]
    
    if critical:
        print(f"\n  {Colors.RED}Critical Edges:{Colors.RESET}")
        for e in critical[:10]:
            bridge_mark = f" {Colors.YELLOW}[BRIDGE]{Colors.RESET}" if e.get("is_bridge") else ""
            print(f"    {e['source_id']} → {e['target_id']}: {e['criticality_score']:.4f}{bridge_mark}")


def print_full_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print full analysis result."""
    summary = result.get("summary", {})
    graph = summary.get("graph", {})
    
    print_section("Graph Statistics")
    print_kv("Total Nodes", graph.get("total_nodes", 0))
    print_kv("Total DEPENDS_ON", graph.get("total_relationships", 0))
    
    # Layers summary
    if result.get("layers"):
        print_section("Layer Analysis")
        layers = result["layers"].get("layers", {})
        for layer_key, layer_data in layers.items():
            layer_summary = layer_data.get("summary", {})
            critical = layer_summary.get("critical_count", 0)
            total = layer_data.get("component_count", 0)
            print(f"  {layer_key:20}: {total:4} components, {critical:3} critical")
    
    # Component types summary
    if result.get("component_types"):
        print_section("Component Type Analysis")
        types = result["component_types"].get("types", {})
        for comp_type, type_data in types.items():
            type_summary = type_data.get("summary", {})
            critical = type_summary.get("critical_count", 0)
            total = type_data.get("count", 0)
            print(f"  {comp_type:15}: {total:4} components, {critical:3} critical")
    
    # Edge summary
    if result.get("edges"):
        print_section("Edge Analysis")
        edge_summary = summary.get("edges", {})
        print_kv("Total Edges", edge_summary.get("total", 0))
        print_kv("Critical Edges", edge_summary.get("critical", 0))
        print_kv("Bridges", edge_summary.get("bridges", 0))
    
    # Overall
    print_section("Summary")
    print_kv("Total Critical Components", summary.get("total_critical_components", 0))
    print_kv("Total Critical Edges", summary.get("total_critical_edges", 0))


# =============================================================================
# Main Functions
# =============================================================================

def run_analysis(args) -> int:
    """Run the analysis based on arguments."""
    try:
        from src.analysis import (
            GraphAnalyzer,
            QualityAnalyzer,
            LAYER_DEFINITIONS,
            COMPONENT_TYPES,
        )
    except ImportError as e:
        print_error(f"Import failed: {e}")
        print_info("Install neo4j driver: pip install neo4j")
        return 1
    
    print_header("Graph Analysis")
    
    try:
        with GraphAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            k_factor=args.k_factor,
        ) as analyzer:
            
            print_success(f"Connected to {args.uri}")
            
            result = None


            if args.quality:
                print_info("Running Quality Assessment (RMA)...")
                q_analyzer = QualityAnalyzer(analyzer.gds, k_factor=args.k_factor)
                result = q_analyzer.analyze_quality(weighted=not args.unweighted)
                
                print_section("Quality Assessment Results")
                print_kv("Average Reliability", f"{result.summary['avg_reliability']:.4f}")
                print_kv("Average Maintainability", f"{result.summary['avg_maintainability']:.4f}")
                print_kv("Average Availability", f"{result.summary['avg_availability']:.4f}")
                print_kv("Availability Risks (SPOFs)", result.summary['critical_availability_count'])
                
                # Show top critical components for Availability
                top_avail = sorted(result.components, key=lambda x: x.availability_score, reverse=True)[:5]
                print(f"\n  {Colors.BOLD}Top Availability Risks:{Colors.RESET}")
                for c in top_avail:
                    print(f"    {c.component_id}: {c.availability_score:.4f}")

            # Layer analysis
            elif args.layer:
                if args.layer not in LAYER_DEFINITIONS:
                    print_error(f"Unknown layer: {args.layer}")
                    print_info(f"Valid layers: {list(LAYER_DEFINITIONS.keys())}")
                    return 1
                
                print_info(f"Analyzing layer: {args.layer}")
                layer_result = analyzer.analyze_layer(args.layer, weighted=not args.unweighted)
                result = layer_result
                print_section(f"Layer: {args.layer}")
                print_layer_result(layer_result.to_dict(), verbose=args.verbose)
            
            # All layers
            elif args.all_layers:
                print_info("Analyzing all layers...")
                result = analyzer.analyze_all_layers(weighted=not args.unweighted)
                
                for layer_key, layer_data in result.layers.items():
                    print_section(f"Layer: {layer_key}")
                    print_layer_result(layer_data.to_dict(), verbose=args.verbose)
            
            # Component type analysis
            elif args.type:
                if args.type not in COMPONENT_TYPES:
                    print_error(f"Unknown type: {args.type}")
                    print_info(f"Valid types: {COMPONENT_TYPES}")
                    return 1
                
                print_info(f"Analyzing type: {args.type}")
                type_result = analyzer.analyze_component_type(args.type, weighted=not args.unweighted)
                result = type_result
                print_section(f"Type: {args.type}")
                print_component_type_result(type_result.to_dict(), verbose=args.verbose)
            
            # All types
            elif args.all_types:
                print_info("Analyzing all component types...")
                result = analyzer.analyze_all_component_types(weighted=not args.unweighted)
                
                for comp_type, type_data in result.by_type.items():
                    print_section(f"Type: {comp_type}")
                    print_component_type_result(type_data.to_dict(), verbose=args.verbose)
            
            # Edge analysis
            elif args.edges:
                print_info("Analyzing edges...")
                result = analyzer.analyze_edges()
                print_section("Edge Analysis")
                print_edge_result(result.to_dict(), verbose=args.verbose)
            
            # Default: show stats
            else:
                stats = analyzer.get_graph_stats()
                print_section("Graph Statistics")
                print_kv("Nodes", stats.get("total_nodes", 0))
                print_kv("Relationships", stats.get("total_relationships", 0))
                
                if stats.get("nodes"):
                    print("\n  By Node Type:")
                    for node_type, count in stats["nodes"].items():
                        print_kv(node_type, count, indent=4)
                
                if stats.get("relationships"):
                    print("\n  By Dependency Type:")
                    for dep_type, count in stats["relationships"].items():
                        print_kv(dep_type, count, indent=4)
                
                print_info("\n Running full analysis...")
                result = analyzer.analyze_full(weighted=not args.unweighted)
                print_full_result(result.to_dict(), verbose=args.verbose)

                return 0
            
            # Export results
            if args.output and result:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                
                print_success(f"Results exported to {output_path}")
            
            print_header("Analysis Complete")
            return 0
    
    except ConnectionError as e:
        print_error(f"Connection failed: {e}")
        print_connection_help()
        return 1
    
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_connection_help() -> None:
    """Print Neo4j connection help."""
    print(f"""
{Colors.YELLOW}Neo4j Connection Help:{Colors.RESET}

1. Start Neo4j with GDS plugin:
   docker run -d --name neo4j \\
       -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password \\
       -e NEO4J_PLUGINS='["graph-data-science"]' \\
       neo4j:latest

2. Import graph data:
   python import_graph.py --input graph.json

3. Run analysis:
   python analyze_graph.py
""")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-layer graph analysis for distributed pub-sub systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full analysis
    python analyze_graph.py
    
    # Analyze application layer
    python analyze_graph.py --layer application
    
    # Analyze all layers
    python analyze_graph.py --all-layers
    
    # Analyze brokers only
    python analyze_graph.py --type Broker
    
    # Analyze all component types
    python analyze_graph.py --all-types
    
    # Analyze edges
    python analyze_graph.py --edges
    
    # Export to JSON
    python analyze_graph.py --output results.json

Layers:
    application   - app_to_app dependencies
    infrastructure - node_to_node dependencies
    app_broker    - app_to_broker dependencies
    node_broker   - node_to_broker dependencies

Component Types:
    Application, Broker, Node, Topic
        """
    )
    
    # Analysis mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--layer", "-l",
        help="Analyze specific layer"
    )
    mode.add_argument(
        "--all-layers",
        action="store_true",
        help="Analyze all layers"
    )
    mode.add_argument(
        "--type", "-t",
        help="Analyze specific component type"
    )
    mode.add_argument(
        "--all-types",
        action="store_true",
        help="Analyze all component types"
    )
    mode.add_argument(
        "--edges", "-e",
        action="store_true",
        help="Analyze edge criticality"
    )
    
    # Neo4j connection
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password (default: password)"
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database (default: neo4j)"
    )
    
    # Analysis options
    parser.add_argument(
        "--k-factor", "-k",
        type=float,
        default=1.5,
        help="Box-plot k factor (default: 1.5)"
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted algorithms"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    # Quality analysis
    parser.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Analyze Reliability, Maintainability, and Availability"
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())