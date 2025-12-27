#!/usr/bin/env python3
"""
Graph Visualization CLI - Version 4.0

Comprehensive visualization for multi-layer pub-sub system graphs.
Generates interactive HTML visualizations and dashboards.

Features:
- Interactive graph with vis.js
- Multi-layer architecture view
- Dashboard with Chart.js
- Criticality-based coloring
- Analysis integration

Usage:
    # Basic graph visualization
    python visualize_graph.py --input system.json --output graph.html
    
    # Multi-layer view
    python visualize_graph.py --input system.json --output layers.html --multi-layer
    
    # Criticality coloring
    python visualize_graph.py --input system.json --output critical.html --color-by criticality
    
    # Dashboard with analysis
    python visualize_graph.py --input system.json --output dashboard.html --dashboard
    
    # Layer-specific view
    python visualize_graph.py --input system.json --output apps.html --layer application

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import SimulationGraph
from src.visualization import (
    GraphRenderer,
    DashboardGenerator,
    RenderConfig,
    DashboardConfig,
    Layer,
    ColorScheme,
)


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
    import os
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and not os.getenv('NO_COLOR')


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")


def print_section(title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.DIM}{'-'*50}{Colors.END}")


def print_kv(key: str, value, indent: int = 2) -> None:
    print(f"{' '*indent}{Colors.DIM}{key}:{Colors.END} {value}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_info(text: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


# =============================================================================
# Analysis Integration
# =============================================================================

def run_analysis(graph: SimulationGraph, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run analysis pipeline to get criticality and validation results"""
    from src.validation import ValidationPipeline, GraphAnalyzer
    
    # Compute centrality metrics
    analyzer = GraphAnalyzer(graph)
    analysis = analyzer.analyze_all()
    
    # Build criticality scores with levels
    composite = analysis["composite"]
    
    # Determine thresholds using percentiles
    scores = sorted(composite.values())
    n = len(scores)
    
    def get_percentile(p):
        idx = int(n * p / 100)
        return scores[min(idx, n - 1)]
    
    p90 = get_percentile(90)
    p75 = get_percentile(75)
    p50 = get_percentile(50)
    p25 = get_percentile(25)
    
    criticality = {}
    for comp_id, score in composite.items():
        if score >= p90:
            level = "critical"
        elif score >= p75:
            level = "high"
        elif score >= p50:
            level = "medium"
        elif score >= p25:
            level = "low"
        else:
            level = "minimal"
        
        criticality[comp_id] = {
            "score": score,
            "level": level,
        }
    
    # Run validation pipeline
    pipeline = ValidationPipeline(seed=seed)
    result = pipeline.run(graph, analysis_method="composite")
    
    validation = result.validation.to_dict()
    
    return {
        "criticality": criticality,
        "validation": validation,
        "analysis": analysis,
    }


def run_simulation(graph: SimulationGraph, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run event simulation to get message metrics"""
    from src.simulation import EventSimulator
    
    simulator = EventSimulator(seed=seed)
    result = simulator.simulate(
        graph,
        duration_ms=5000,
        message_rate=50,
    )
    
    return result.metrics.to_dict()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize pub-sub system graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic graph visualization
    python visualize_graph.py --input system.json --output graph.html
    
    # Multi-layer view
    python visualize_graph.py --input system.json --output layers.html --multi-layer
    
    # Criticality coloring with analysis
    python visualize_graph.py --input system.json --output critical.html \\
        --color-by criticality --run-analysis
    
    # Complete dashboard
    python visualize_graph.py --input system.json --output dashboard.html \\
        --dashboard --run-analysis --run-simulation
    
    # Layer-specific view
    python visualize_graph.py --input system.json --output apps.html \\
        --layer application
    
    # Custom title and theme
    python visualize_graph.py --input system.json --output viz.html \\
        --title "My System" --theme light
        """,
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input graph JSON file",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Output HTML file (default: <input>_viz.html)",
    )
    
    # Visualization mode
    mode_group = parser.add_argument_group("Visualization Mode")
    mode_group.add_argument(
        "--multi-layer", "-m", action="store_true",
        help="Generate multi-layer architecture view",
    )
    mode_group.add_argument(
        "--dashboard", "-d", action="store_true",
        help="Generate comprehensive dashboard",
    )
    mode_group.add_argument(
        "--layer", "-l", type=str,
        choices=["application", "topic", "broker", "infrastructure", "all"],
        default="all",
        help="Layer to display (default: all)",
    )
    
    # Styling
    style_group = parser.add_argument_group("Styling Options")
    style_group.add_argument(
        "--color-by", "-c", type=str,
        choices=["type", "criticality", "layer"],
        default="type",
        help="Color scheme (default: type)",
    )
    style_group.add_argument(
        "--title", "-t", type=str,
        help="Visualization title",
    )
    style_group.add_argument(
        "--theme", type=str,
        choices=["dark", "light"],
        default="dark",
        help="Dashboard theme (default: dark)",
    )
    style_group.add_argument(
        "--no-physics", action="store_true",
        help="Disable physics simulation",
    )
    style_group.add_argument(
        "--no-labels", action="store_true",
        help="Hide node labels",
    )
    
    # Analysis integration
    analysis_group = parser.add_argument_group("Analysis Integration")
    analysis_group.add_argument(
        "--run-analysis", "-a", action="store_true",
        help="Run analysis to compute criticality scores",
    )
    analysis_group.add_argument(
        "--run-simulation", "-s", action="store_true",
        help="Run event simulation for dashboard metrics",
    )
    analysis_group.add_argument(
        "--criticality-file", type=Path,
        help="Load criticality scores from JSON file",
    )
    analysis_group.add_argument(
        "--validation-file", type=Path,
        help="Load validation results from JSON file",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    output_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output",
    )
    output_group.add_argument(
        "--no-color", action="store_true",
        help="Disable colors",
    )
    output_group.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    try:
        if not args.quiet:
            print_header("Graph Visualization")
            print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Input: {args.input}")
        
        # Load graph
        if not args.quiet:
            print_section("Loading Graph")
        
        if not args.input.exists():
            print_error(f"File not found: {args.input}")
            return 1
        
        graph = SimulationGraph.from_json(args.input)
        
        if not args.quiet:
            print_success(f"Loaded: {len(graph.components)} components, {len(graph.connections)} connections")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.input.with_suffix('.html')
        
        # Initialize data containers
        criticality = None
        validation = None
        simulation = None
        analysis = None
        
        # Load or run analysis
        if args.criticality_file:
            if not args.quiet:
                print_section("Loading Criticality Data")
            with open(args.criticality_file) as f:
                criticality = json.load(f)
            print_success(f"Loaded criticality from {args.criticality_file}")
        
        if args.validation_file:
            if not args.quiet:
                print_section("Loading Validation Data")
            with open(args.validation_file) as f:
                validation = json.load(f)
            print_success(f"Loaded validation from {args.validation_file}")
        
        if args.run_analysis or args.color_by == "criticality":
            if not args.quiet:
                print_section("Running Analysis")
            
            results = run_analysis(graph, seed=args.seed)
            criticality = results["criticality"]
            validation = results["validation"]
            analysis = results["analysis"]
            
            if not args.quiet:
                print_success("Analysis complete")
                # Show top critical components
                top_crit = sorted(
                    criticality.items(),
                    key=lambda x: x[1]["score"],
                    reverse=True
                )[:5]
                print_info("Top critical components:")
                for comp, data in top_crit:
                    print(f"    {comp}: {data['score']:.4f} ({data['level']})")
        
        if args.run_simulation:
            if not args.quiet:
                print_section("Running Simulation")
            
            simulation = run_simulation(graph, seed=args.seed)
            
            if not args.quiet:
                print_success("Simulation complete")
                print_info(f"Messages: {simulation.get('messages', {}).get('published', 0)} published, "
                          f"{simulation.get('messages', {}).get('delivered', 0)} delivered")
        
        # Generate visualization
        if not args.quiet:
            print_section("Generating Visualization")
        
        # Map color scheme
        color_map = {
            "type": ColorScheme.TYPE,
            "criticality": ColorScheme.CRITICALITY,
            "layer": ColorScheme.LAYER,
        }
        color_scheme = color_map.get(args.color_by, ColorScheme.TYPE)
        
        # Map layer
        layer_map = {
            "application": Layer.APPLICATION,
            "topic": Layer.TOPIC,
            "broker": Layer.BROKER,
            "infrastructure": Layer.INFRASTRUCTURE,
            "all": Layer.ALL,
        }
        layer = layer_map.get(args.layer, Layer.ALL)
        
        if args.dashboard:
            # Generate dashboard
            config = DashboardConfig(
                title=args.title or "Pub-Sub System Analysis Dashboard",
                theme=args.theme,
            )
            generator = DashboardGenerator(config)
            
            html = generator.generate(
                graph=graph,
                criticality=criticality,
                validation=validation,
                simulation=simulation,
                analysis=analysis,
            )
            
            if not args.quiet:
                print_info("Mode: Dashboard")
        
        elif args.multi_layer:
            # Generate multi-layer view
            config = RenderConfig(
                title=args.title or "Multi-Layer System Architecture",
                color_scheme=color_scheme,
                physics_enabled=not args.no_physics,
                show_labels=not args.no_labels,
            )
            renderer = GraphRenderer(config)
            
            html = renderer.render_multi_layer(graph, criticality)
            
            if not args.quiet:
                print_info("Mode: Multi-layer view")
        
        else:
            # Generate standard graph
            config = RenderConfig(
                title=args.title or "Pub-Sub System Graph",
                color_scheme=color_scheme,
                physics_enabled=not args.no_physics,
                show_labels=not args.no_labels,
            )
            renderer = GraphRenderer(config)
            
            html = renderer.render(graph, criticality, layer)
            
            if not args.quiet:
                print_info(f"Mode: Graph view (layer: {args.layer})")
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        
        if not args.quiet:
            print_section("Output")
            print_success(f"Visualization saved: {output_path}")
            print_kv("File size", f"{output_path.stat().st_size / 1024:.1f} KB")
        else:
            print(output_path)
        
        return 0
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    
    except KeyboardInterrupt:
        print_warning("\nVisualization interrupted")
        return 130
    
    except Exception as e:
        print_error(f"Visualization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())