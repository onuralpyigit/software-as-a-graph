#!/usr/bin/env python3
"""
Analyze Graph CLI - Version 6.0

Command-line interface for multi-layer graph analysis.

Features:
- Component type analysis (Application, Broker, Node, Topic)
- Layer analysis (application, infrastructure, app_broker, node_broker)
- Edge criticality analysis
- Box-plot statistical classification
- NetworkX-based algorithms (PageRank, Betweenness, Articulation Points)

Usage:
    # Run all analysis
    python analyze_graph.py
    
    # Analyze by component type
    python analyze_graph.py --type Application
    python analyze_graph.py --all-types
    
    # Analyze by layer
    python analyze_graph.py --layer application
    python analyze_graph.py --all-layers
    
    # Analyze edges
    python analyze_graph.py --edges
    
    # Export results
    python analyze_graph.py --output results.json

Author: Software-as-a-Graph Research Project
Version: 6.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """Terminal color codes."""
    
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
        """Disable all colors."""
        for attr in ["BOLD", "GREEN", "BLUE", "CYAN", "YELLOW", "RED", "GRAY", "RESET"]:
            setattr(cls, attr, "")


def supports_color() -> bool:
    """Check if terminal supports colors."""
    import os
    return (
        hasattr(sys.stdout, "isatty") and 
        sys.stdout.isatty() and 
        os.name != "nt"
    )


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(title: str) -> None:
    """Print a header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print("-" * 40)


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


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

def print_component_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print component type analysis result."""
    print(f"  Type: {result.get('component_type', 'N/A')}")
    print(f"  Total: {result.get('count', 0)}")
    
    summary = result.get("summary", {})
    by_level = summary.get("by_level", {})
    
    print(f"\n  Distribution:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = by_level.get(level, 0)
        color = level_color(level)
        bar = "█" * min(count, 20)
        print(f"    {color}{level.upper():10}{Colors.RESET}: {bar} {count}")
    
    # Show critical components
    components = result.get("components", [])
    critical = [c for c in components if c.get("level") == "critical"]
    
    if critical:
        print(f"\n  {Colors.RED}Critical Components:{Colors.RESET}")
        for c in critical[:10]:
            ap = f" {Colors.YELLOW}[AP]{Colors.RESET}" if c.get("is_articulation_point") else ""
            print(f"    {c['id']}: {c['composite_score']:.4f}{ap}")
    
    # Show articulation points
    aps = result.get("articulation_points", [])
    if aps and verbose:
        print(f"\n  Articulation Points ({len(aps)}):")
        for ap in aps[:10]:
            print(f"    • {ap}")
    
    if verbose:
        # Show top by each metric
        top_pr = sorted(components, key=lambda x: x.get("pagerank", 0), reverse=True)[:5]
        if top_pr:
            print(f"\n  Top by PageRank:")
            for c in top_pr:
                print(f"    {c['id']}: {c.get('pagerank', 0):.6f}")
        
        top_bc = sorted(components, key=lambda x: x.get("betweenness", 0), reverse=True)[:5]
        if top_bc:
            print(f"\n  Top by Betweenness:")
            for c in top_bc:
                print(f"    {c['id']}: {c.get('betweenness', 0):.6f}")


def print_layer_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print layer analysis result."""
    print(f"  Layer: {result.get('layer_name', result.get('layer_key', 'N/A'))}")
    print(f"  Total: {result.get('count', 0)}")
    
    summary = result.get("summary", {})
    by_level = summary.get("by_level", {})
    
    print(f"\n  Distribution:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = by_level.get(level, 0)
        color = level_color(level)
        bar = "█" * min(count, 20)
        print(f"    {color}{level.upper():10}{Colors.RESET}: {bar} {count}")
    
    # Show critical
    components = result.get("components", [])
    critical = [c for c in components if c.get("level") == "critical"]
    
    if critical:
        print(f"\n  {Colors.RED}Critical Components:{Colors.RESET}")
        for c in critical[:10]:
            ap = f" {Colors.YELLOW}[AP]{Colors.RESET}" if c.get("is_articulation_point") else ""
            print(f"    {c['id']} ({c['type']}): {c['composite_score']:.4f}{ap}")
    
    # Show bridges
    bridges = result.get("bridges", [])
    if bridges and verbose:
        print(f"\n  Bridge Edges ({len(bridges)}):")
        for b in bridges[:10]:
            print(f"    {b['source']} → {b['target']}")


def print_edge_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print edge analysis result."""
    print(f"  Total Edges: {result.get('count', 0)}")
    
    summary = result.get("summary", {})
    by_level = summary.get("by_level", {})
    
    print(f"\n  Distribution:")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = by_level.get(level, 0)
        color = level_color(level)
        bar = "█" * min(count, 20)
        print(f"    {color}{level.upper():10}{Colors.RESET}: {bar} {count}")
    
    # Show critical edges
    edges = result.get("edges", [])
    critical = [e for e in edges if e.get("level") == "critical"]
    
    if critical:
        print(f"\n  {Colors.RED}Critical Edges:{Colors.RESET}")
        for e in critical[:10]:
            bridge = f" {Colors.YELLOW}[BRIDGE]{Colors.RESET}" if e.get("is_bridge") else ""
            print(f"    {e['source']} → {e['target']}: {e['criticality_score']:.4f}{bridge}")
    
    # Show bridges
    bridges = result.get("bridges", [])
    if bridges and verbose:
        print(f"\n  Bridge Edges ({len(bridges)}):")
        for b in bridges[:10]:
            print(f"    {b['source']} → {b['target']}")


def print_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print full analysis result."""
    summary = result.get("summary", {})
    graph = summary.get("graph", {})
    
    print_section("Graph Statistics")
    print(f"  Total Nodes: {graph.get('total_nodes', 0)}")
    print(f"  Total Edges: {graph.get('total_edges', 0)}")
    
    # Component types
    if result.get("component_types"):
        print_section("Component Type Analysis")
        types_summary = summary.get("component_types", {})
        for comp_type, info in types_summary.items():
            critical = info.get("critical", 0)
            total = info.get("total", 0)
            aps = info.get("articulation_points", 0)
            color = Colors.RED if critical > 0 else Colors.GREEN
            print(f"  {comp_type:15}: {total:4} total, {color}{critical:3} critical{Colors.RESET}, {aps} APs")
    
    # Layers
    if result.get("layers"):
        print_section("Layer Analysis")
        layer_summary = summary.get("layers", {})
        for layer_key, info in layer_summary.items():
            critical = info.get("critical", 0)
            total = info.get("total", 0)
            bridges = info.get("bridges", 0)
            color = Colors.RED if critical > 0 else Colors.GREEN
            print(f"  {layer_key:15}: {total:4} total, {color}{critical:3} critical{Colors.RESET}, {bridges} bridges")
    
    # Edges
    if result.get("edges"):
        print_section("Edge Analysis")
        edge_info = summary.get("edges", {})
        total = edge_info.get("total", 0)
        critical = edge_info.get("critical", 0)
        bridges = edge_info.get("bridges", 0)
        color = Colors.RED if critical > 0 else Colors.GREEN
        print(f"  Total: {total}, {color}Critical: {critical}{Colors.RESET}, Bridges: {bridges}")
    
    # Overall summary
    print_section("Overall Summary")
    print(f"  {Colors.RED}Total Critical Components: {summary.get('total_critical_components', 0)}{Colors.RESET}")
    print(f"  {Colors.RED}Total Critical Edges: {summary.get('total_critical_edges', 0)}{Colors.RESET}")


# =============================================================================
# Analysis Functions
# =============================================================================

def run_analysis(args) -> int:
    """Run the analysis based on arguments."""
    from src.analysis import (
        GraphAnalyzer
    )
    
    from src.core import (
        COMPONENT_TYPES,
        LAYER_DEFINITIONS,
    )
    
    print_header("Graph Analysis")
    print_info(f"Connecting to Neo4j at {args.uri}")
    
    result = None
    
    try:
        with GraphAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            k_factor=args.k_factor,
        ) as analyzer:
            
            # Get stats first
            stats = analyzer.get_graph_stats()
            print_success(f"Connected. Found {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
            
            # Determine analysis mode
            if args.type:
                # Single component type
                if args.type not in COMPONENT_TYPES:
                    print_error(f"Invalid type: {args.type}")
                    print_info(f"Valid types: {COMPONENT_TYPES}")
                    return 1
                
                print_info(f"Analyzing component type: {args.type}")
                result = analyzer.analyze_component_type(args.type, weighted=not args.unweighted)
                print_section(f"Component Type: {args.type}")
                print_component_result(result.to_dict(), verbose=args.verbose)
            
            elif args.all_types:
                # All component types
                print_info("Analyzing all component types...")
                results = analyzer.analyze_all_component_types(weighted=not args.unweighted)
                result = results
                
                for comp_type, type_result in results.items():
                    print_section(f"Component Type: {comp_type}")
                    print_component_result(type_result.to_dict(), verbose=args.verbose)
            
            elif args.layer:
                # Single layer
                if args.layer not in LAYER_DEFINITIONS:
                    print_error(f"Invalid layer: {args.layer}")
                    print_info(f"Valid layers: {list(LAYER_DEFINITIONS.keys())}")
                    return 1
                
                print_info(f"Analyzing layer: {args.layer}")
                result = analyzer.analyze_layer(args.layer, weighted=not args.unweighted)
                print_section(f"Layer: {args.layer}")
                print_layer_result(result.to_dict(), verbose=args.verbose)
            
            elif args.all_layers:
                # All layers
                print_info("Analyzing all layers...")
                results = analyzer.analyze_all_layers(weighted=not args.unweighted)
                result = results
                
                for layer_key, layer_result in results.items():
                    print_section(f"Layer: {layer_key}")
                    print_layer_result(layer_result.to_dict(), verbose=args.verbose)
            
            elif args.edges:
                # Edge analysis
                print_info("Analyzing edges...")
                result = analyzer.analyze_edges()
                print_section("Edge Analysis")
                print_edge_result(result.to_dict(), verbose=args.verbose)
            
            else:
                # Full analysis (default)
                print_info("Running analysis...")
                result = analyzer.analyze(weighted=not args.unweighted)
                print_result(result.to_dict(), verbose=args.verbose)
            
            # Export if requested
            if args.output and result is not None:
                export_data = result.to_dict() if hasattr(result, "to_dict") else {
                    k: v.to_dict() for k, v in result.items()
                }
                
                output_path = Path(args.output)
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)
                print_success(f"Results exported to {output_path}")
            
            print_header("Analysis Complete")
            return 0
    
    except ConnectionError as e:
        print_error(f"Connection failed: {e}")
        print_neo4j_help()
        return 1
    
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_neo4j_help():
    """Print Neo4j connection help."""
    print(f"""
{Colors.YELLOW}Neo4j Connection Help:{Colors.RESET}

1. Start Neo4j:
   docker run -d --name neo4j \\
       -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password \\
       neo4j:latest

2. Import graph data:
   python import_graph.py --input graph.json

3. Run analysis:
   python analyze_graph.py --password your_password
""")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-layer graph analysis for distributed pub-sub systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full analysis
    python analyze_graph.py
    
    # Analyze applications only
    python analyze_graph.py --type Application
    
    # Analyze all component types
    python analyze_graph.py --all-types
    
    # Analyze application layer
    python analyze_graph.py --layer application
    
    # Analyze all layers
    python analyze_graph.py --all-layers
    
    # Analyze edges
    python analyze_graph.py --edges
    
    # Export to JSON
    python analyze_graph.py --output results.json

Component Types:
    Application, Broker, Node, Topic

Layers:
    application   - app_to_app dependencies
    infrastructure - node_to_node dependencies
    app_broker    - app_to_broker dependencies
    node_broker   - node_to_broker dependencies
        """
    )
    
    # Analysis mode (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--type", "-t",
        help="Analyze specific component type (Application, Broker, Node, Topic)"
    )
    mode.add_argument(
        "--all-types",
        action="store_true",
        help="Analyze all component types separately"
    )
    mode.add_argument(
        "--layer", "-l",
        help="Analyze specific layer (application, infrastructure, app_broker, node_broker)"
    )
    mode.add_argument(
        "--all-layers",
        action="store_true",
        help="Analyze all layers"
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
        "--k-factor",
        type=float,
        default=1.5,
        help="Box-plot k factor for classification (default: 1.5)"
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted algorithms"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed metrics"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup colors
    if args.no_color or not supports_color():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())