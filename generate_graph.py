#!/usr/bin/env python3
"""
Graph Generation CLI - Version 4.0

Generates realistic pub-sub system graphs for testing and analysis.

Usage:
    # Basic generation
    python generate_graph.py --scale medium --output system.json
    
    # With specific scenario
    python generate_graph.py --scale large --scenario financial --output trading.json
    
    # With anti-patterns for testing
    python generate_graph.py --scale medium --antipatterns god_topic spof --output test.json
    
    # Preview without saving
    python generate_graph.py --scale small --preview
    
    # List available options
    python generate_graph.py --list-options

Scales:
    tiny     - 2 nodes, 5 apps, 3 topics, 1 broker
    small    - 4 nodes, 15 apps, 10 topics, 2 brokers
    medium   - 8 nodes, 40 apps, 25 topics, 4 brokers
    large    - 16 nodes, 100 apps, 60 topics, 8 brokers
    xlarge   - 32 nodes, 250 apps, 150 topics, 16 brokers
    extreme  - 64 nodes, 500 apps, 300 topics, 32 brokers

Scenarios:
    generic, iot, financial, healthcare, autonomous_vehicle, smart_city

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import generate_graph, GraphConfig


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'END', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def use_colors() -> bool:
    """Check if terminal supports colors"""
    import os
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and not os.getenv('NO_COLOR')


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")


def print_section(title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.DIM}{'-'*40}{Colors.END}")


def print_kv(key: str, value, indent: int = 2) -> None:
    print(f"{' '*indent}{Colors.BLUE}{key}:{Colors.END} {value}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


# =============================================================================
# List Options
# =============================================================================

def list_options() -> None:
    """Print available generation options"""
    print_header("Graph Generation Options")
    
    print_section("Scales")
    scales = [
        ("tiny", "2 nodes, 5 apps, 3 topics, 1 broker"),
        ("small", "4 nodes, 15 apps, 10 topics, 2 brokers"),
        ("medium", "8 nodes, 40 apps, 25 topics, 4 brokers"),
        ("large", "16 nodes, 100 apps, 60 topics, 8 brokers"),
        ("xlarge", "32 nodes, 250 apps, 150 topics, 16 brokers"),
        ("extreme", "64 nodes, 500 apps, 300 topics, 32 brokers"),
    ]
    for name, desc in scales:
        print(f"  {Colors.BOLD}{name:10}{Colors.END} {desc}")
    
    print_section("Scenarios")
    scenarios = [
        ("generic", "General-purpose pub-sub system"),
        ("iot", "IoT sensor network with telemetry"),
        ("financial", "Trading and market data system"),
        ("healthcare", "Medical device and patient monitoring"),
        ("autonomous_vehicle", "Self-driving vehicle perception/control"),
        ("smart_city", "Traffic, energy, and emergency systems"),
    ]
    for name, desc in scenarios:
        print(f"  {Colors.BOLD}{name:20}{Colors.END} {desc}")
    
    print_section("Anti-patterns")
    antipatterns = [
        ("god_topic", "Creates a topic with many subscribers"),
        ("spof", "Creates single point of failure broker"),
        ("chatty", "Creates an app that publishes to many topics"),
        ("bottleneck", "Creates a bottleneck in message flow"),
    ]
    for name, desc in antipatterns:
        print(f"  {Colors.BOLD}{name:15}{Colors.END} {desc}")
    
    print()


# =============================================================================
# Preview Graph
# =============================================================================

def preview_graph(graph: dict) -> None:
    """Print graph summary"""
    print_header("Graph Preview")
    
    metadata = graph.get("metadata", {})
    print_section("Metadata")
    print_kv("ID", metadata.get("id", "N/A"))
    print_kv("Scale", metadata.get("scale", "N/A"))
    print_kv("Scenario", metadata.get("scenario", "N/A"))
    print_kv("Seed", metadata.get("seed", "N/A"))
    print_kv("Generated", metadata.get("generated_at", "N/A"))
    
    metrics = graph.get("metrics", {})
    vertex_counts = metrics.get("vertex_counts", {})
    edge_counts = metrics.get("edge_counts", {})
    
    print_section("Vertices")
    print_kv("Applications", vertex_counts.get("applications", 0))
    print_kv("Topics", vertex_counts.get("topics", 0))
    print_kv("Brokers", vertex_counts.get("brokers", 0))
    print_kv("Nodes", vertex_counts.get("nodes", 0))
    print_kv("Total", vertex_counts.get("total", 0))
    
    print_section("Relationships")
    print_kv("PUBLISHES_TO", edge_counts.get("publishes_to", 0))
    print_kv("SUBSCRIBES_TO", edge_counts.get("subscribes_to", 0))
    print_kv("ROUTES", edge_counts.get("routes", 0))
    print_kv("RUNS_ON", edge_counts.get("runs_on", 0))
    print_kv("CONNECTS_TO", edge_counts.get("connects_to", 0))
    print_kv("Total", edge_counts.get("total", 0))
    
    pub_sub = metrics.get("pub_sub", {})
    print_section("Pub/Sub Statistics")
    print_kv("Avg Publishers/Topic", f"{pub_sub.get('avg_pubs_per_topic', 0):.2f}")
    print_kv("Avg Subscribers/Topic", f"{pub_sub.get('avg_subs_per_topic', 0):.2f}")
    print_kv("Max Fanout", pub_sub.get("max_fanout", 0))
    
    role_dist = metrics.get("role_distribution", {})
    if role_dist:
        print_section("Role Distribution")
        for role, count in role_dist.items():
            print_kv(role, count)
    
    antipatterns = metadata.get("antipatterns_applied")
    if antipatterns:
        print_section("Anti-patterns Applied")
        for ap, info in antipatterns.items():
            print_kv(ap, json.dumps(info))
    
    print()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pub-sub system graphs for testing and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_graph.py --scale medium --output system.json
    python generate_graph.py --scale large --scenario financial --output trading.json
    python generate_graph.py --scale small --preview
    python generate_graph.py --list-options
        """,
    )
    
    # Generation options
    parser.add_argument(
        "--scale", "-s",
        choices=["tiny", "small", "medium", "large", "xlarge", "extreme"],
        default="medium",
        help="Scale preset (default: medium)",
    )
    parser.add_argument(
        "--scenario", "-c",
        choices=["generic", "iot", "financial", "healthcare", "autonomous_vehicle", "smart_city"],
        default="generic",
        help="Domain scenario (default: generic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--antipatterns", "-a",
        nargs="+",
        choices=["god_topic", "spof", "chatty", "bottleneck"],
        help="Anti-patterns to inject",
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (JSON format)",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview graph without saving",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    
    # Info options
    parser.add_argument(
        "--list-options", "-l",
        action="store_true",
        help="List available options and exit",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # List options
    if args.list_options:
        list_options()
        return 0
    
    # Validate arguments
    if not args.preview and not args.output:
        print_error("Either --output or --preview is required")
        return 1
    
    try:
        # Generate graph
        if not args.quiet:
            print_header("Graph Generation")
            print_kv("Scale", args.scale)
            print_kv("Scenario", args.scenario)
            print_kv("Seed", args.seed)
            if args.antipatterns:
                print_kv("Anti-patterns", ", ".join(args.antipatterns))
            print()
        
        graph = generate_graph(
            scale=args.scale,
            scenario=args.scenario,
            seed=args.seed,
            antipatterns=args.antipatterns or [],
        )
        
        # Preview
        if args.preview:
            preview_graph(graph)
        
        # Save
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(graph, f, indent=2)
            
            if not args.quiet:
                print_success(f"Graph saved to {args.output}")
                file_size = args.output.stat().st_size / 1024
                print_kv("File size", f"{file_size:.1f} KB")
        
        return 0
        
    except Exception as e:
        print_error(f"Generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())