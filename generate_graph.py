#!/usr/bin/env python3
"""
Graph Generation Script - Version 2.0

Comprehensive graph generation tool for Software-as-a-Graph analysis with:
- Multiple scale presets (tiny to extreme)
- 7 domain-specific scenarios
- Sophisticated anti-pattern injection
- Zone/region-aware deployment
- Comprehensive validation
- Multiple export formats
- Detailed statistics and visualization

Key Improvements:
1. Better error handling and validation
2. Progress indicators for large graphs
3. Comprehensive statistics output
4. Validation with detailed reports
5. Multiple export formats in one command
6. Preview mode to see what will be generated
7. Better logging and debugging

Usage Examples:
    # Basic generation
    python generate_graph.py --scale small --output system.json
    
    # IoT system with high availability
    python generate_graph.py --scale large --scenario iot --ha --output iot_ha.json
    
    # Financial system with anti-patterns
    python generate_graph.py --scale medium --scenario financial \\
        --antipatterns spof broker_overload --output financial_bad.json
    
    # Preview without generating
    python generate_graph.py --scale xlarge --preview
    
    # Generate with all export formats
    python generate_graph.py --scale medium --formats json graphml gexf pickle \\
        --output system
    
    # Validate existing graph
    python generate_graph.py --validate-only --input existing_system.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from graph generator
from src.core.graph_generator import GraphGenerator, GraphConfig
from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ{Colors.ENDC} {text}")


def validate_graph(graph: Dict, verbose: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Comprehensive graph validation
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check basic structure
    required_keys = ['metadata', 'nodes', 'applications', 'topics', 'brokers', 'relationships']
    for key in required_keys:
        if key not in graph:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors, warnings
    
    # Collect IDs
    node_ids = {n['id'] for n in graph['nodes']}
    app_ids = {a['id'] for a in graph['applications']}
    topic_ids = {t['id'] for t in graph['topics']}
    broker_ids = {b['id'] for b in graph['brokers']}
    
    # Check for empty components
    if not graph['nodes']:
        errors.append("No nodes defined")
    if not graph['applications']:
        errors.append("No applications defined")
    if not graph['topics']:
        errors.append("No topics defined")
    if not graph['brokers']:
        errors.append("No brokers defined")
    
    # Validate relationships
    rels = graph['relationships']
    
    # 1. Check runs_on relationships
    apps_with_deployment = {r['from'] for r in rels['runs_on']}
    orphaned_apps = app_ids - apps_with_deployment
    if orphaned_apps:
        errors.append(f"Applications without deployment: {orphaned_apps}")
    
    # Validate runs_on references
    for r in rels['runs_on']:
        if r['from'] not in app_ids:
            errors.append(f"runs_on references unknown app: {r['from']}")
        if r['to'] not in node_ids:
            errors.append(f"runs_on references unknown node: {r['to']}")
    
    # 2. Check routes relationships
    topics_with_routes = {r['to'] for r in rels['routes']}
    orphaned_topics = topic_ids - topics_with_routes
    if orphaned_topics:
        errors.append(f"Topics without broker routes: {orphaned_topics}")
    
    # Validate route references
    for r in rels['routes']:
        if r['from'] not in broker_ids:
            errors.append(f"routes references unknown broker: {r['from']}")
        if r['to'] not in topic_ids:
            errors.append(f"routes references unknown topic: {r['to']}")
    
    # 3. Check publishes_to relationships
    for r in rels['publishes_to']:
        if r['from'] not in app_ids:
            errors.append(f"publishes_to references unknown app: {r['from']}")
        if r['to'] not in topic_ids:
            errors.append(f"publishes_to references unknown topic: {r['to']}")
    
    # 4. Check subscribes_to relationships
    for r in rels['subscribes_to']:
        if r['from'] not in app_ids:
            errors.append(f"subscribes_to references unknown app: {r['from']}")
        if r['to'] not in topic_ids:
            errors.append(f"subscribes_to references unknown topic: {r['to']}")
    
    # Check for disconnected components
    connected_apps = set()
    for pub in rels['publishes_to']:
        connected_apps.add(pub['from'])
    for sub in rels['subscribes_to']:
        connected_apps.add(sub['from'])
    
    disconnected_apps = app_ids - connected_apps
    if disconnected_apps:
        warnings.append(f"Disconnected applications (no pub/sub): {disconnected_apps}")
    
    # Check for topics with no publishers
    topics_with_publishers = {r['to'] for r in rels['publishes_to']}
    topics_without_publishers = topic_ids - topics_with_publishers
    if topics_without_publishers:
        warnings.append(f"Topics without publishers: {len(topics_without_publishers)} topics")
    
    # Check for topics with no subscribers
    topics_with_subscribers = {r['to'] for r in rels['subscribes_to']}
    topics_without_subscribers = topic_ids - topics_with_subscribers
    if topics_without_subscribers:
        warnings.append(f"Topics without subscribers: {len(topics_without_subscribers)} topics")
    
    # Check for node capacity issues
    apps_per_node = {}
    for r in rels['runs_on']:
        node = r['to']
        apps_per_node[node] = apps_per_node.get(node, 0) + 1
    
    for node_id, count in apps_per_node.items():
        if count > 50:  # Arbitrary threshold
            warnings.append(f"Node {node_id} has {count} applications (may be overloaded)")
    
    # Check for broker capacity issues
    topics_per_broker = {}
    for r in rels['routes']:
        broker = r['from']
        topics_per_broker[broker] = topics_per_broker.get(broker, 0) + 1
    
    for broker in graph['brokers']:
        broker_id = broker['id']
        topic_count = topics_per_broker.get(broker_id, 0)
    
    try:
        builder = GraphBuilder()
        model = builder.build_from_dict(graph)
        print_info("Refactored validation passed")
    except Exception as e:
        errors.append(f"Refactored validation error: {e}")
    
    return len(errors) == 0, errors, warnings


def export_graph(graph: Dict, output_path: str, formats: List[str], verbose: bool = False):
    """
    Export graph to multiple formats
    
    Args:
        graph: Graph dictionary
        output_path: Base output path
        formats: List of formats ['json', 'graphml', 'gexf', 'pickle', 'dot']
        verbose: Verbose output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    # Always save JSON
    if 'json' in formats:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(graph, f, indent=2)
        exported_files.append(str(json_path))
        print_success(f"Saved JSON: {json_path}")
    
    # Export to other formats
    if len(formats) > 1:
        try:
            # Build model
            builder = GraphBuilder()
            model = builder.build_from_dict(graph)
            
            # Export
            exporter = GraphExporter(model)
            
            for fmt in formats:
                if fmt == 'json':
                    continue
                
                try:
                    export_path = output_path.with_suffix(f'.{fmt}')
                    
                    if fmt == 'graphml':
                        exporter.export_to_graphml(str(export_path))
                    elif fmt == 'gexf':
                        exporter.export_to_gexf(str(export_path))
                    elif fmt == 'pickle':
                        exporter.export_to_pickle(str(export_path))
                    elif fmt == 'dot':
                        exporter.export_to_dot(str(export_path))
                    else:
                        print_warning(f"Unknown format: {fmt}")
                        continue
                    
                    exported_files.append(str(export_path))
                    print_success(f"Saved {fmt.upper()}: {export_path}")
                    
                except Exception as e:
                    print_error(f"Failed to export {fmt}: {e}")
        
        except Exception as e:
            print_error(f"Export failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    elif len(formats) > 1:
        print_warning("Additional formats require refactored modules (only JSON exported)")
    
    return exported_files


def print_statistics(graph: Dict, config: GraphConfig, generation_time: float):
    """Print comprehensive statistics about generated graph"""
    
    print_header("GRAPH GENERATION STATISTICS")
    
    # Configuration
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Scale: {config.scale}")
    print(f"  Scenario: {config.scenario}")
    if config.antipatterns:
        print(f"  Anti-patterns: {', '.join(config.antipatterns)}")
    print(f"  Generation Time: {generation_time:.2f}s")
    print()
    
    # Components
    print(f"{Colors.BOLD}Components:{Colors.ENDC}")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Brokers: {len(graph['brokers'])}")
    print(f"  Topics: {len(graph['topics'])}")
    print(f"  Applications: {len(graph['applications'])}")
    print()
    
    # Relationships
    rels = graph['relationships']
    print(f"{Colors.BOLD}Relationships:{Colors.ENDC}")
    print(f"  Publishes: {len(rels['publishes_to'])}")
    print(f"  Subscribes: {len(rels['subscribes_to'])}")
    print(f"  Routes: {len(rels['routes'])}")
    print(f"  Runs On: {len(rels['runs_on'])}")
    total_edges = sum(len(rels[k]) for k in rels.keys())
    print(f"  Total Edges: {total_edges}")
    print()
    
    # Application distribution
    app_types = {}
    
    for app in graph['applications']:
        app_type = app['app_type']
        app_types[app_type] = app_types.get(app_type, 0) + 1

    print(f"{Colors.BOLD}Application Distribution:{Colors.ENDC}")
    for app_type, count in sorted(app_types.items()):
        pct = (count / len(graph['applications'])) * 100
        print(f"  {app_type}: {count} ({pct:.1f}%)")
    print()
    
    # Topic distribution
    qos_durability = {}
    qos_reliability = {}
    
    for topic in graph['topics']:
        qos = topic.get('qos', {})
        durability = qos.get('durability', 'VOLATILE')
        qos_durability[durability] = qos_durability.get(durability, 0) + 1
        
        reliability = qos.get('reliability', 'BEST_EFFORT')
        qos_reliability[reliability] = qos_reliability.get(reliability, 0) + 1
    
    print(f"{Colors.BOLD}QoS Distribution:{Colors.ENDC}")
    print("  Durability:")
    for durability, count in sorted(qos_durability.items()):
        pct = (count / len(graph['topics'])) * 100
        print(f"    {durability}: {count} ({pct:.1f}%)")
    print("  Reliability:")
    for reliability, count in sorted(qos_reliability.items()):
        pct = (count / len(graph['topics'])) * 100
        print(f"    {reliability}: {count} ({pct:.1f}%)")
    print()
    
    # Connectivity metrics
    publishers = {r['from'] for r in rels['publishes_to']}
    subscribers = {r['from'] for r in rels['subscribes_to']}
    
    print(f"{Colors.BOLD}Connectivity Metrics:{Colors.ENDC}")
    print(f"  Publishing Apps: {len(publishers)} ({len(publishers)/len(graph['applications'])*100:.1f}%)")
    print(f"  Subscribing Apps: {len(subscribers)} ({len(subscribers)/len(graph['applications'])*100:.1f}%)")
    
    # Topic fanout
    topic_subs = {}
    for r in rels['subscribes_to']:
        topic_subs[r['to']] = topic_subs.get(r['to'], 0) + 1
    
    if topic_subs:
        avg_fanout = sum(topic_subs.values()) / len(topic_subs)
        max_fanout = max(topic_subs.values())
        print(f"  Average Topic Fanout: {avg_fanout:.1f}")
        print(f"  Maximum Topic Fanout: {max_fanout}")


def preview_generation(config: GraphConfig):
    """Preview what will be generated without actually generating"""
    
    print_header("GRAPH GENERATION PREVIEW")
    
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Scale: {config.scale}")
    print(f"  Scenario: {config.scenario}")
    print(f"  Nodes: {config.num_nodes}")
    print(f"  Applications: {config.num_applications}")
    print(f"  Topics: {config.num_topics}")
    print(f"  Brokers: {config.num_brokers}")
    print(f"  Edge Density: {config.edge_density}")
    if config.antipatterns:
        print(f"  Anti-patterns: {', '.join(config.antipatterns)}")
    print(f"  Random Seed: {config.seed}")
    print()
    
    # Estimate counts
    est_publishes = int(config.num_applications * 0.6 * 2)  # Rough estimate
    est_subscribes = int(config.num_applications * 0.7 * config.edge_density * config.num_topics)
    est_total_edges = est_publishes + est_subscribes + config.num_topics + config.num_applications
    
    print(f"{Colors.BOLD}Estimated Graph Size:{Colors.ENDC}")
    print(f"  Total Vertices: {config.num_nodes + config.num_applications + config.num_topics + config.num_brokers}")
    print(f"  Estimated Edges: ~{est_total_edges}")
    print()
    
    # Estimate memory
    avg_node_size = 500  # bytes
    avg_edge_size = 200  # bytes
    est_memory_mb = (
        (config.num_nodes + config.num_applications + config.num_topics + config.num_brokers) * avg_node_size +
        est_total_edges * avg_edge_size
    ) / (1024 * 1024)
    
    print(f"{Colors.BOLD}Resource Estimates:{Colors.ENDC}")
    print(f"  Estimated Memory: ~{est_memory_mb:.1f} MB")
    
    # Estimate generation time
    base_time = 0.5  # seconds for tiny
    scale_multiplier = {
        'tiny': 1,
        'small': 2,
        'medium': 5,
        'large': 20,
        'xlarge': 60,
        'extreme': 180
    }.get(config.scale, 10)
    
    est_time = base_time * scale_multiplier
    print(f"  Estimated Generation Time: ~{est_time:.1f}s")
    print()


def setup_argparse() -> argparse.ArgumentParser:
    """Setup argument parser with comprehensive options"""
    
    parser = argparse.ArgumentParser(
        description='Enhanced Graph Generation for Software-as-a-Graph Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic small system
  %(prog)s --scale small --output system.json
  
  # Large IoT system with HA
  %(prog)s --scale large --scenario iot --ha -output iot_system.json
  
  # Financial system with anti-patterns
  %(prog)s --scale medium --scenario financial --antipatterns spof broker_overload \\
      --output financial_bad.json
  
  # Preview large system
  %(prog)s --scale xlarge --preview
  
  # Export to multiple formats
  %(prog)s --scale medium --formats json graphml gexf --output system
  
  # Validate existing graph
  %(prog)s --validate-only --input existing.json
        """
    )
    
    # Scale and scenario
    parser.add_argument('--scale', '-s',
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme'],
                       default='small',
                       help='Graph scale preset (default: small)')
    
    parser.add_argument('--scenario', '-c',
                       choices=['generic', 'iot', 'financial', 'ecommerce', 'analytics', 
                               'smart_city', 'healthcare'],
                       default='generic',
                       help='Domain scenario (default: generic)')
    
    # Custom parameters (override scale)
    parser.add_argument('--nodes', type=int, help='Number of nodes (overrides scale)')
    parser.add_argument('--apps', type=int, help='Number of applications (overrides scale)')
    parser.add_argument('--topics', type=int, help='Number of topics (overrides scale)')
    parser.add_argument('--brokers', type=int, help='Number of brokers (overrides scale)')
    parser.add_argument('--density', type=float, default=0.3,
                       help='Edge density 0.0-1.0 (default: 0.3)')
    
    # High availability
    parser.add_argument('--ha', action='store_true',
                       help='Enable high availability patterns')
    
    # Anti-patterns
    parser.add_argument('--antipatterns', '-a', nargs='+',
                       choices=['spof', 'broker_overload', 'god_object', 'single_broker',
                               'tight_coupling', 'chatty_communication', 'bottleneck'],
                       help='Anti-patterns to inject')
    
    # Generation options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Output options
    parser.add_argument('--output', '-o', default='system.json',
                       help='Output file path (default: system.json)')
    parser.add_argument('--formats', '-f', nargs='+',
                       choices=['json', 'graphml', 'gexf', 'pickle', 'dot'],
                       default=['json'],
                       help='Export formats (default: json)')
    
    # Validation
    parser.add_argument('--validate', action='store_true',
                       help='Validate graph after generation')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing graph (requires --input)')
    parser.add_argument('--input', '-i',
                       help='Input file for validation')
    
    # Other options
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview configuration without generating')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-stats', action='store_true',
                       help='Skip statistics output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    return parser


def main():
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Validate-only mode
        if args.validate_only:
            if not args.input:
                print_error("--validate-only requires --input")
                return 1
            
            print_header("GRAPH VALIDATION")
            
            with open(args.input, 'r') as f:
                graph = json.load(f)
            
            print_info(f"Loaded graph from {args.input}")
            
            is_valid, errors, warnings = validate_graph(graph, verbose=args.verbose)
            
            if errors:
                print(f"\n{Colors.FAIL}Validation Errors:{Colors.ENDC}")
                for error in errors:
                    print_error(error)
            
            if warnings:
                print(f"\n{Colors.WARNING}Validation Warnings:{Colors.ENDC}")
                for warning in warnings:
                    print_warning(warning)
            
            if is_valid:
                print_success("\nGraph validation passed!")
                return 0
            else:
                print_error(f"\nGraph validation failed with {len(errors)} errors")
                return 1
        
        # Get scale parameters
        scale_params = GraphGenerator.SCALES[args.scale]
        
        # Create config
        config = GraphConfig(
            scale=args.scale,
            scenario=args.scenario,
            num_nodes=args.nodes or scale_params['nodes'],
            num_applications=args.apps or scale_params['apps'],
            num_topics=args.topics or scale_params['topics'],
            num_brokers=args.brokers or scale_params['brokers'],
            edge_density=args.density,
            antipatterns=args.antipatterns or [],
            seed=args.seed
        )
        
        # Preview mode
        if args.preview:
            preview_generation(config)
            return 0
        
        # Generate graph
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
                print()
                print_info("Validating graph...")
            
            is_valid, errors, warnings = validate_graph(graph, verbose=args.verbose)
            
            if errors:
                print(f"\n{Colors.FAIL}Validation Errors:{Colors.ENDC}")
                for error in errors[:10]:  # Show first 10
                    print_error(error)
                if len(errors) > 10:
                    print_error(f"... and {len(errors)-10} more errors")
            
            if warnings and args.verbose:
                print(f"\n{Colors.WARNING}Validation Warnings:{Colors.ENDC}")
                for warning in warnings[:10]:
                    print_warning(warning)
                if len(warnings) > 10:
                    print_warning(f"... and {len(warnings)-10} more warnings")
            
            if not is_valid:
                print()
                response = input(f"{Colors.WARNING}Continue with invalid graph? [y/N]: {Colors.ENDC}")
                if response.lower() != 'y':
                    return 1
            else:
                print_success("Validation passed!")
        
        # Export
        if not args.quiet:
            print()
            print_info("Exporting graph...")
        
        exported_files = export_graph(graph, args.output, args.formats, args.verbose)
        
        # Print statistics
        if not args.no_stats and not args.quiet:
            print_statistics(graph, config, generation_time)
        
        # Final summary
        if not args.quiet:
            print_header("GENERATION COMPLETE")
            print_success(f"Generated {args.scale} scale {args.scenario} system")
            print_success(f"Files created: {len(exported_files)}")
            for f in exported_files:
                print(f"  • {f}")
        
        return 0
        
    except KeyboardInterrupt:
        print_error("\nGeneration interrupted by user")
        return 130
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    except Exception as e:
        print_error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
