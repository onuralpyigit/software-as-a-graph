#!/usr/bin/env python3
"""
Graph Visualization CLI
=======================

Visualizes distributed pub-sub systems with analysis, simulation,
and validation results.

Usage:
    # Basic visualization
    python visualize_graph.py --output-dir visuals/
    
    # With full analysis, simulation, and validation
    python visualize_graph.py --output-dir visuals/ --full
    
    # Generate only HTML report
    python visualize_graph.py --output-dir visuals/ --html-only
    
    # Custom layout and styling
    python visualize_graph.py --output-dir visuals/ --layout circular
    
    # Export graph data for external tools
    python visualize_graph.py --export-format json --output-dir visuals/

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.analysis import GraphAnalyzer, NEO4J_AVAILABLE
from src.simulation import GraphSimulator
from src.validation import GraphValidator
from src.visualization import (
    GraphVisualizer,
    VisualizationConfig,
    visualize_system,
    MATPLOTLIB_AVAILABLE,
)


# ============================================================================
# Terminal Colors
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    _enabled = True
    
    @classmethod
    def disable(cls):
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls._enabled = False


def print_header(text: str):
    """Print a header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}── {text} ──{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


# ============================================================================
# Main Functions
# ============================================================================

def run_visualization(args) -> int:
    """Run the visualization"""
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('visualize_graph')
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("PUB-SUB SYSTEM VISUALIZATION")
    
    # Check matplotlib
    if not MATPLOTLIB_AVAILABLE and not args.html_only:
        print_warning("Matplotlib not available. Only HTML reports will be generated.")
        print_warning("Install with: pip install matplotlib")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if not args.quiet:
            print_section("Loading Data")
        
        analyzer = GraphAnalyzer(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma
        )
        
        if not NEO4J_AVAILABLE:
            print_error("Neo4j driver not installed. Install with: pip install neo4j")
            return 1
        
        analyzer.load_from_neo4j(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        if not args.quiet:
            print_success(f"Loaded from Neo4j: {args.neo4j_uri}")
        
        # Run analysis
        if not args.quiet:
            print_section("Running Analysis")
        
        analysis_result = analyzer.analyze()
        if not args.quiet:
            print_success(f"Analyzed {analyzer.G.number_of_nodes()} components")
        
        # Run simulation if requested
        simulation_result = None
        if args.full or args.simulate:
            if not args.quiet:
                print_section("Running Simulation")
            
            simulator = GraphSimulator(seed=args.seed)
            simulation_result = simulator.simulate_all_single_failures(
                analyzer.G,
                enable_cascade=args.cascade
            )
            if not args.quiet:
                print_success(f"Completed {simulation_result.total_simulations} simulations")
        
        # Run validation if requested
        validation_result = None
        if args.full or args.validate:
            if not args.quiet:
                print_section("Running Validation")
            
            if simulation_result is None:
                simulator = GraphSimulator(seed=args.seed)
            
            validator = GraphValidator(analyzer, seed=args.seed)
            validation_result = validator.validate()
            if not args.quiet:
                print_success(f"Validation status: {validation_result.status.value}")
        
        # Create visualizer
        config = VisualizationConfig(
            figsize=(args.width, args.height),
            dpi=args.dpi,
            node_size=args.node_size,
            font_size=args.font_size,
            show_labels=not args.no_labels,
            show_legend=not args.no_legend
        )
        
        visualizer = GraphVisualizer(analyzer, config)
        visualizer.set_analysis_result(analysis_result)
        
        if simulation_result:
            visualizer.set_simulation_result(simulation_result)
        
        if validation_result:
            visualizer.set_validation_result(validation_result)
        
        # Generate visualizations
        if not args.quiet:
            print_section("Generating Visualizations")
        
        generated_files = []
        
        # Export graph data if requested
        if args.export_format:
            export_path = output_dir / f'graph_data.{args.export_format}'
            visualizer.export_graph_data(str(export_path), args.export_format)
            generated_files.append(str(export_path))
            if not args.quiet:
                print_success(f"Exported graph data: {export_path}")
        
        # Generate images (if matplotlib available and not html-only)
        if MATPLOTLIB_AVAILABLE and not args.html_only:
            # Topology
            img_path = output_dir / 'topology.png'
            result = visualizer.plot_topology(
                str(img_path),
                layout=args.layout,
                color_by=args.color_by,
                highlight_critical=args.highlight_critical
            )
            if result:
                generated_files.append(result)
                if not args.quiet:
                    print_success(f"Generated: {img_path}")
            
            # Criticality heatmap
            img_path = output_dir / 'criticality.png'
            result = visualizer.plot_criticality_heatmap(str(img_path), top_n=args.top_n)
            if result:
                generated_files.append(result)
                if not args.quiet:
                    print_success(f"Generated: {img_path}")
            
            # Impact comparison (if validation available)
            if validation_result:
                img_path = output_dir / 'comparison.png'
                result = visualizer.plot_impact_comparison(str(img_path), top_n=args.top_n)
                if result:
                    generated_files.append(result)
                    if not args.quiet:
                        print_success(f"Generated: {img_path}")
                
                img_path = output_dir / 'scatter.png'
                result = visualizer.plot_validation_scatter(str(img_path))
                if result:
                    generated_files.append(result)
                    if not args.quiet:
                        print_success(f"Generated: {img_path}")
            
            # Impact distribution (if simulation available)
            if simulation_result:
                img_path = output_dir / 'distribution.png'
                result = visualizer.plot_impact_distribution(str(img_path))
                if result:
                    generated_files.append(result)
                    if not args.quiet:
                        print_success(f"Generated: {img_path}")
        
        # Generate HTML report
        if not args.no_html:
            html_path = output_dir / 'report.html'
            visualizer.generate_html_report(
                str(html_path),
                title=args.title or "Pub-Sub System Analysis Report"
            )
            generated_files.append(str(html_path))
            if not args.quiet:
                print_success(f"Generated: {html_path}")
        
        # Summary
        if not args.quiet:
            print_section("Summary")
            print(f"\n  Output directory: {output_dir}")
            print(f"  Files generated: {len(generated_files)}")
            for f in generated_files:
                print(f"    - {Path(f).name}")
        
        return 0
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        print_error(f"Visualization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Visualize pub-sub systems with analysis and simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic visualization
    python visualize_graph.py --output-dir visuals/
    
    # Full analysis with simulation and validation
    python visualize_graph.py --output-dir visuals/ --full
    
    # Custom layout
    python visualize_graph.py --output-dir visuals/ --layout circular
    
    # Color by criticality
    python visualize_graph.py --output-dir visuals/ --color-by criticality
    
    # HTML report only (no matplotlib required)
    python visualize_graph.py --output-dir visuals/ --html-only
    
    # Export for external tools (D3.js, Gephi)
    python visualize_graph.py --output-dir visuals/ --export-format json
    
    # Load from Neo4j
    python visualize_graph.py --output-dir visuals/

Layout Options:
    spring       - Force-directed layout (default)
    circular     - Circular layout
    shell        - Concentric shells by type
    kamada_kawai - Kamada-Kawai layout
        """
    )
    
    # Neo4j options
    neo4j_group = parser.add_argument_group('Neo4j Connection')
    neo4j_group.add_argument('--neo4j-uri', default='bolt://localhost:7687',
                             help='Neo4j URI (default: bolt://localhost:7687)')
    neo4j_group.add_argument('--neo4j-user', default='neo4j',
                             help='Neo4j username (default: neo4j)')
    neo4j_group.add_argument('--neo4j-password', default='password',
                             help='Neo4j password (default: password)')
    neo4j_group.add_argument('--neo4j-database', default='neo4j',
                             help='Neo4j database name (default: neo4j)')
    
    # Analysis weights
    weight_group = parser.add_argument_group('Analysis Weights')
    weight_group.add_argument('--alpha', type=float, default=0.4,
                              help='Betweenness centrality weight (default: 0.4)')
    weight_group.add_argument('--beta', type=float, default=0.3,
                              help='Articulation point weight (default: 0.3)')
    weight_group.add_argument('--gamma', type=float, default=0.3,
                              help='Impact score weight (default: 0.3)')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--full', action='store_true',
                            help='Run full analysis, simulation, and validation')
    proc_group.add_argument('--simulate', action='store_true',
                            help='Run exhaustive simulation')
    proc_group.add_argument('--validate', action='store_true',
                            help='Run validation (implies --simulate)')
    proc_group.add_argument('--cascade', action='store_true',
                            help='Enable cascade in simulation')
    proc_group.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    
    # Visualization options
    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument('--layout', choices=['spring', 'circular', 'shell', 'kamada_kawai'],
                           default='spring', help='Graph layout algorithm (default: spring)')
    vis_group.add_argument('--color-by', choices=['type', 'criticality'],
                           default='type', help='Node coloring scheme (default: type)')
    vis_group.add_argument('--highlight-critical', action='store_true', default=True,
                           help='Highlight critical nodes (default: True)')
    vis_group.add_argument('--no-highlight', dest='highlight_critical', action='store_false',
                           help='Do not highlight critical nodes')
    vis_group.add_argument('--top-n', type=int, default=15,
                           help='Number of top components to show (default: 15)')
    
    # Image options
    img_group = parser.add_argument_group('Image Options')
    img_group.add_argument('--width', type=int, default=12,
                           help='Figure width in inches (default: 12)')
    img_group.add_argument('--height', type=int, default=8,
                           help='Figure height in inches (default: 8)')
    img_group.add_argument('--dpi', type=int, default=150,
                           help='Image DPI (default: 150)')
    img_group.add_argument('--node-size', type=int, default=800,
                           help='Node size (default: 800)')
    img_group.add_argument('--font-size', type=int, default=10,
                           help='Font size (default: 10)')
    img_group.add_argument('--no-labels', action='store_true',
                           help='Hide node labels')
    img_group.add_argument('--no-legend', action='store_true',
                           help='Hide legend')
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', '-o', required=True,
                              help='Output directory for visualizations')
    output_group.add_argument('--title',
                              help='Report title')
    output_group.add_argument('--html-only', action='store_true',
                              help='Generate only HTML report (no images)')
    output_group.add_argument('--no-html', action='store_true',
                              help='Do not generate HTML report')
    output_group.add_argument('--export-format', choices=['json', 'graphml', 'gexf'],
                              help='Export graph data for external tools')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    
    args = parser.parse_args()
    
    # Validate implies simulate
    if args.validate:
        args.simulate = True
    
    return run_visualization(args)


if __name__ == '__main__':
    sys.exit(main())