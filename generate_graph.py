#!/usr/bin/env python3
"""
Graph Generation CLI

Command-line interface for generating realistic pub-sub system graphs.

Features:
- Multiple scale presets (tiny to extreme)
- 8 domain-specific scenarios
- Anti-pattern injection for testing
- Multiple export formats
- Validation and preview modes
- Integration with analysis pipeline

Usage Examples:
    # Basic generation
    python generate_graph.py --scale medium --output system.json
    
    # IoT system with anti-patterns
    python generate_graph.py --scale large --scenario iot \\
        --antipatterns spof god_topic --output iot_system.json
    
    # Preview configuration
    python generate_graph.py --scale xlarge --scenario financial --preview
    
    # Generate with validation
    python generate_graph.py --scale medium --validate --output validated.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.graph_generator import GraphGenerator, GraphConfig
from src.core.graph_builder import GraphBuilder


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''


def print_header(text: str):
    """Print formatted header"""
    width = 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^{width}}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ{Colors.ENDC} {text}")


def print_kv(key: str, value: Any, indent: int = 2):
    """Print key-value pair"""
    spaces = ' ' * indent
    print(f"{spaces}{Colors.DIM}{key}:{Colors.ENDC} {value}")


# =============================================================================
# Validation
# =============================================================================

def validate_graph(graph: Dict, verbose: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a generated graph for consistency.
    
    Args:
        graph: Generated graph dictionary
        verbose: Print detailed validation info
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required sections
    required_sections = ['nodes', 'brokers', 'applications', 'topics', 'relationships']
    for section in required_sections:
        if section not in graph:
            errors.append(f"Missing required section: {section}")
    
    if errors:
        return False, errors, warnings
    
    # Build ID sets
    node_ids = {n['id'] for n in graph['nodes']}
    broker_ids = {b['id'] for b in graph['brokers']}
    app_ids = {a['id'] for a in graph['applications']}
    topic_ids = {t['id'] for t in graph['topics']}
    all_ids = node_ids | broker_ids | app_ids | topic_ids
    
    # Check for duplicate IDs
    all_items = graph['nodes'] + graph['brokers'] + graph['applications'] + graph['topics']
    seen_ids = set()
    for item in all_items:
        if item['id'] in seen_ids:
            errors.append(f"Duplicate ID: {item['id']}")
        seen_ids.add(item['id'])
    
    # Validate relationships
    relationships = graph.get('relationships', {})
    
    # RUNS_ON validation
    for rel in relationships.get('runs_on', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source not in app_ids and source not in broker_ids:
            # Allow anti-pattern injected IDs
            if not any(prefix in source for prefix in ['spof_', 'coupling_', 'cycle_']):
                errors.append(f"RUNS_ON source not found: {source}")
        if target not in node_ids:
            errors.append(f"RUNS_ON target not found: {target}")
    
    # PUBLISHES_TO validation
    for rel in relationships.get('publishes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source not in app_ids:
            if not any(prefix in source for prefix in ['spof_', 'coupling_', 'cycle_']):
                warnings.append(f"PUBLISHES_TO source not in apps: {source}")
        if target not in topic_ids:
            if not any(prefix in target for prefix in ['spof_', 'coupling_', 'cycle_', 'god_', 'hidden_']):
                errors.append(f"PUBLISHES_TO target not found: {target}")
    
    # SUBSCRIBES_TO validation
    for rel in relationships.get('subscribes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source not in app_ids:
            if not any(prefix in source for prefix in ['spof_', 'coupling_', 'cycle_']):
                warnings.append(f"SUBSCRIBES_TO source not in apps: {source}")
        if target not in topic_ids:
            if not any(prefix in target for prefix in ['spof_', 'coupling_', 'cycle_', 'god_', 'hidden_']):
                errors.append(f"SUBSCRIBES_TO target not found: {target}")
    
    # ROUTES validation
    for rel in relationships.get('routes', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source not in broker_ids:
            errors.append(f"ROUTES source not found: {source}")
        if target not in topic_ids:
            if not any(prefix in target for prefix in ['spof_', 'coupling_', 'cycle_', 'god_', 'hidden_']):
                warnings.append(f"ROUTES target not in topics: {target}")
    
    # Check for orphan topics (no publishers or subscribers)
    published_topics = {r.get('to', r.get('target')) for r in relationships.get('publishes_to', [])}
    subscribed_topics = {r.get('to', r.get('target')) for r in relationships.get('subscribes_to', [])}
    
    for topic in graph['topics']:
        if topic['id'] not in published_topics:
            warnings.append(f"Topic has no publishers: {topic['id']}")
        if topic['id'] not in subscribed_topics:
            warnings.append(f"Topic has no subscribers: {topic['id']}")
    
    # Check for orphan applications
    publishing_apps = {r.get('from', r.get('source')) for r in relationships.get('publishes_to', [])}
    subscribing_apps = {r.get('from', r.get('source')) for r in relationships.get('subscribes_to', [])}
    connected_apps = publishing_apps | subscribing_apps
    
    for app in graph['applications']:
        if app['id'] not in connected_apps:
            warnings.append(f"Application has no pub/sub connections: {app['id']}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


# =============================================================================
# Preview Mode
# =============================================================================

def preview_generation(config: GraphConfig):
    """Preview what will be generated without actually generating"""
    print_header("GENERATION PREVIEW")
    
    scale_info = GraphGenerator.SCALES.get(config.scale, {})
    scenario_info = GraphGenerator.SCENARIOS.get(config.scenario, {})
    
    print_section("Configuration")
    print_kv("Scale", f"{config.scale} - {scale_info.get('description', 'Unknown')}")
    print_kv("Scenario", f"{config.scenario} - {scenario_info.get('description', 'Unknown')}")
    print_kv("Seed", config.seed)
    print_kv("Edge Density", config.edge_density)
    
    print_section("Expected Components")
    print_kv("Infrastructure Nodes", config.num_nodes or scale_info.get('nodes', 'default'))
    print_kv("Applications", config.num_applications or scale_info.get('apps', 'default'))
    print_kv("Topics", config.num_topics or scale_info.get('topics', 'default'))
    print_kv("Brokers", config.num_brokers or scale_info.get('brokers', 'default'))
    
    print_section("Features")
    print_kv("High Availability", "Enabled" if config.ha_enabled else "Disabled")
    print_kv("Multi-Zone", f"Enabled ({config.num_zones} zones)" if config.multi_zone else "Disabled")
    print_kv("Region-Aware", "Enabled" if config.region_aware else "Disabled")
    
    if config.antipatterns:
        print_section("Anti-Patterns to Inject")
        for ap in config.antipatterns:
            print(f"  • {ap}")
    
    print_section("Application Types")
    for app_type, role, weight in scenario_info.get('app_types', [])[:5]:
        print(f"  • {app_type} ({role}) - {weight*100:.0f}%")
    if len(scenario_info.get('app_types', [])) > 5:
        print(f"  ... and {len(scenario_info['app_types'])-5} more")
    
    print_section("Topic Patterns")
    for pattern, weight in scenario_info.get('topic_patterns', [])[:5]:
        print(f"  • {pattern} - {weight*100:.0f}%")
    if len(scenario_info.get('topic_patterns', [])) > 5:
        print(f"  ... and {len(scenario_info['topic_patterns'])-5} more")
    
    print()
    print_info("Use without --preview to generate the graph")


# =============================================================================
# Statistics Display
# =============================================================================

def print_statistics(graph: Dict):
    """Print graph statistics"""
    stats = graph.get('statistics', {})
    
    print_section("Graph Statistics")
    print_kv("Infrastructure Nodes", stats.get('total_nodes', 'N/A'))
    print_kv("Brokers", stats.get('total_brokers', 'N/A'))
    print_kv("Applications", stats.get('total_applications', 'N/A'))
    print_kv("Topics", stats.get('total_topics', 'N/A'))
    
    print_section("Relationships")
    print_kv("PUBLISHES_TO", stats.get('total_publishes', 'N/A'))
    print_kv("SUBSCRIBES_TO", stats.get('total_subscribes', 'N/A'))
    print_kv("ROUTES", stats.get('total_routes', 'N/A'))
    print_kv("RUNS_ON", len(graph.get('relationships', {}).get('runs_on', [])))
    
    print_section("Density Metrics")
    print_kv("Unique Publishers", stats.get('unique_publishers', 'N/A'))
    print_kv("Unique Subscribers", stats.get('unique_subscribers', 'N/A'))
    print_kv("Avg Publishers/Topic", stats.get('avg_publishers_per_topic', 'N/A'))
    print_kv("Avg Subscribers/Topic", stats.get('avg_subscribers_per_topic', 'N/A'))
    print_kv("Avg Apps/Node", stats.get('avg_apps_per_node', 'N/A'))
    
    # Anti-patterns
    antipatterns = graph.get('injected_antipatterns', [])
    if antipatterns:
        print_section("Injected Anti-Patterns")
        for ap in antipatterns:
            print(f"  • {Colors.WARNING}{ap['type']}{Colors.ENDC}: {ap.get('description', 'N/A')}")


# =============================================================================
# Export Functions
# =============================================================================

def export_graph(graph: Dict, output_path: str, format: str = 'json'):
    """
    Export graph to file.
    
    Args:
        graph: Graph dictionary
        output_path: Output file path
        format: Export format (json, graphml, etc.)
    """
    path = Path(output_path)
    
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(graph, f, indent=2)
    else:
        # For other formats, use GraphBuilder to convert
        builder = GraphBuilder()
        
        # Build model from generated data
        model_data = {
            'applications': graph['applications'],
            'topics': graph['topics'],
            'brokers': graph['brokers'],
            'nodes': graph['nodes'],
            'edges': graph['relationships']
        }
        model = builder.build_from_dict(model_data)
        
        # Convert to NetworkX
        G = builder.to_networkx(model)
        
        if format == 'graphml':
            import networkx as nx
            nx.write_graphml(G, str(path))
        elif format == 'gexf':
            import networkx as nx
            nx.write_gexf(G, str(path))
        else:
            raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Main CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Generate realistic pub-sub system graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scale medium --output system.json
  %(prog)s --scale large --scenario iot --antipatterns spof god_topic
  %(prog)s --scale xlarge --preview
  %(prog)s --validate-only --input existing.json
        """
    )
    
    # Scale and scenario
    parser.add_argument(
        '--scale', '-s',
        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme'],
        default='medium',
        help='Scale preset (default: medium)'
    )
    
    parser.add_argument(
        '--scenario', '-S',
        choices=['generic', 'iot', 'financial', 'ecommerce', 'autonomous_vehicle',
                 'smart_city', 'healthcare', 'gaming'],
        default='generic',
        help='Domain scenario (default: generic)'
    )
    
    # Component counts (overrides)
    parser.add_argument('--nodes', '-n', type=int, help='Number of infrastructure nodes')
    parser.add_argument('--apps', '-a', type=int, help='Number of applications')
    parser.add_argument('--topics', '-t', type=int, help='Number of topics')
    parser.add_argument('--brokers', '-b', type=int, help='Number of brokers')
    
    # Density and patterns
    parser.add_argument(
        '--density', '-d',
        type=float, default=0.3,
        help='Edge density 0.0-1.0 (default: 0.3)'
    )
    
    parser.add_argument(
        '--antipatterns', '-A',
        nargs='+',
        choices=['spof', 'god_topic', 'broker_overload', 'tight_coupling',
                 'chatty', 'circular', 'bottleneck', 'hidden_coupling'],
        help='Anti-patterns to inject'
    )
    
    # High availability options
    parser.add_argument(
        '--ha', '--high-availability',
        action='store_true',
        dest='ha_enabled',
        help='Enable high-availability patterns'
    )
    
    parser.add_argument(
        '--multi-zone',
        action='store_true',
        help='Enable multi-zone deployment'
    )
    
    parser.add_argument(
        '--num-zones',
        type=int, default=3,
        help='Number of availability zones (default: 3)'
    )
    
    parser.add_argument(
        '--region-aware',
        action='store_true',
        help='Enable region-aware topology'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'graphml', 'gexf'],
        default='json',
        help='Output format (default: json)'
    )
    
    # Seed
    parser.add_argument(
        '--seed',
        type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Modes
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview what will be generated without generating'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated graph'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate an existing graph'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input file for validation'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    # List options
    parser.add_argument(
        '--list-scales',
        action='store_true',
        help='List available scale presets'
    )
    
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available scenarios'
    )
    
    parser.add_argument(
        '--list-antipatterns',
        action='store_true',
        help='List available anti-patterns'
    )
    
    return parser


def list_scales():
    """List available scale presets"""
    print_header("SCALE PRESETS")
    
    for name, params in GraphGenerator.SCALES.items():
        print(f"\n{Colors.BOLD}{name}{Colors.ENDC}")
        print(f"  {Colors.DIM}{params.get('description', '')}{Colors.ENDC}")
        print(f"  Nodes: {params['nodes']}, Apps: {params['apps']}, "
              f"Topics: {params['topics']}, Brokers: {params['brokers']}")


def list_scenarios():
    """List available scenarios"""
    print_header("DOMAIN SCENARIOS")
    
    for name, config in GraphGenerator.SCENARIOS.items():
        print(f"\n{Colors.BOLD}{name}{Colors.ENDC}")
        print(f"  {Colors.DIM}{config.get('description', '')}{Colors.ENDC}")
        print(f"  App types: {len(config.get('app_types', []))}")
        print(f"  Topic patterns: {len(config.get('topic_patterns', []))}")
        print(f"  QoS profile: {config.get('qos_profile', 'balanced')}")


def list_antipatterns():
    """List available anti-patterns"""
    print_header("ANTI-PATTERNS")
    
    antipatterns = {
        'spof': 'Single Point of Failure - Creates a critical component many apps depend on',
        'god_topic': 'God Topic - Creates a topic with excessive publishers and subscribers',
        'broker_overload': 'Broker Overload - Routes most topics through a single broker',
        'tight_coupling': 'Tight Coupling - Creates a cluster of highly interdependent apps',
        'chatty': 'Chatty Communication - Apps sending many small messages very frequently',
        'circular': 'Circular Dependency - Creates cyclic dependency chains between apps',
        'bottleneck': 'Infrastructure Bottleneck - Concentrates many apps on one node',
        'hidden_coupling': 'Hidden Coupling - Creates implicit dependencies via shared topics'
    }
    
    for name, desc in antipatterns.items():
        print(f"\n{Colors.WARNING}{name}{Colors.ENDC}")
        print(f"  {desc}")


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # List modes
    if args.list_scales:
        list_scales()
        return 0
    
    if args.list_scenarios:
        list_scenarios()
        return 0
    
    if args.list_antipatterns:
        list_antipatterns()
        return 0
    
    # Validate-only mode
    if args.validate_only:
        if not args.input:
            print_error("--input required for --validate-only mode")
            return 1
        
        print_header("GRAPH VALIDATION")
        
        try:
            with open(args.input) as f:
                graph = json.load(f)
        except Exception as e:
            print_error(f"Failed to load graph: {e}")
            return 1
        
        is_valid, errors, warnings = validate_graph(graph, args.verbose)
        
        if errors:
            print_section("Errors")
            for err in errors[:20]:
                print_error(err)
            if len(errors) > 20:
                print_error(f"... and {len(errors)-20} more errors")
        
        if warnings and args.verbose:
            print_section("Warnings")
            for warn in warnings[:20]:
                print_warning(warn)
            if len(warnings) > 20:
                print_warning(f"... and {len(warnings)-20} more warnings")
        
        if is_valid:
            print_success(f"Graph is valid ({len(warnings)} warnings)")
            return 0
        else:
            print_error(f"Graph validation failed ({len(errors)} errors, {len(warnings)} warnings)")
            return 1
    
    # Create config
    scale_params = GraphGenerator.SCALES.get(args.scale, GraphGenerator.SCALES['medium'])
    
    config = GraphConfig(
        scale=args.scale,
        scenario=args.scenario,
        num_nodes=args.nodes or scale_params['nodes'],
        num_applications=args.apps or scale_params['apps'],
        num_topics=args.topics or scale_params['topics'],
        num_brokers=args.brokers or scale_params['brokers'],
        edge_density=args.density,
        antipatterns=args.antipatterns or [],
        seed=args.seed,
        ha_enabled=args.ha_enabled,
        multi_zone=args.multi_zone,
        num_zones=args.num_zones,
        region_aware=args.region_aware
    )
    
    # Preview mode
    if args.preview:
        preview_generation(config)
        return 0
    
    # Generate
    if not args.quiet:
        print_header("GRAPH GENERATION")
        print_info(f"Generating {config.scale} scale {config.scenario} system...")
    
    start_time = time.time()
    generator = GraphGenerator(config)
    graph = generator.generate()
    generation_time = time.time() - start_time
    
    if not args.quiet:
        print_success(f"Graph generated in {generation_time:.2f}s")
    
    # Validate if requested
    if args.validate:
        if not args.quiet:
            print_info("Validating graph...")
        
        is_valid, errors, warnings = validate_graph(graph, args.verbose)
        
        if errors:
            print_section("Validation Errors")
            for err in errors[:10]:
                print_error(err)
        
        if not is_valid:
            print_error("Validation failed - graph may have issues")
            response = input(f"{Colors.WARNING}Continue anyway? [y/N]: {Colors.ENDC}")
            if response.lower() != 'y':
                return 1
        else:
            print_success("Validation passed")
    
    # Print statistics
    if not args.quiet:
        print_statistics(graph)
    
    # Export
    if args.output:
        output_path = args.output
        if not output_path.endswith(f'.{args.format}'):
            output_path = f"{output_path}.{args.format}"
        
        if not args.quiet:
            print_info(f"Exporting to {output_path}...")
        
        try:
            export_graph(graph, output_path, args.format)
            print_success(f"Graph exported to {output_path}")
        except Exception as e:
            print_error(f"Export failed: {e}")
            return 1
    else:
        if not args.quiet:
            print_warning("No output file specified. Use --output to save the graph.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())