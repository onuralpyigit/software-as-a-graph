#!/usr/bin/env python3
"""
Graph Generation Script for Software-as-a-Graph Analysis

Generates realistic complex pub-sub system graphs at multiple scales with:
- Configurable topology parameters
- Realistic QoS policies
- Anti-pattern scenarios
- Domain-specific patterns (IoT, Financial, etc.)
- Output in multiple formats (JSON, NetworkX, GraphML)
- Uses graph_builder for validation
- Exports to multiple formats via graph_exporter
- Compatible with all analysis modules

Usage:
    # Generate small system
    python generate_graph.py --scale small --output system.json
    
    # Generate large IoT system
    python generate_graph.py --scale large --scenario iot --output iot_system.json
    
    # Generate with anti-patterns
    python generate_graph.py --scale medium --antipatterns spof broker_overload \\
        --output antipattern_system.json
    
    # Generate and validate
    python generate_graph.py --scale medium --validate --output validated_system.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from src.core.graph_generator import GraphGenerator, GraphConfig
from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter
from src.core.graph_builder import GraphBuilder

# Add directory to path
sys.path.insert(0, str(Path(__file__).parent / '.'))

def validate_graph(graph: Dict) -> Tuple[bool, List[str]]:
    """
    Validate generated graph using graph_builder
    
    Returns:
        Tuple of (is_valid, errors)
    """
    try:
        builder = GraphBuilder()
        model = builder.build_from_dict(graph)
        
        # Basic validation
        errors = []
        
        # Check for orphaned components
        app_ids = {a['id'] for a in graph['applications']}
        topic_ids = {t['id'] for t in graph['topics']}
        broker_ids = {b['id'] for b in graph['brokers']}
        
        # Check all apps have runs_on
        apps_with_deployment = {r['from'] for r in graph['relationships']['runs_on']}
        orphaned_apps = app_ids - apps_with_deployment
        if orphaned_apps:
            errors.append(f"Applications without deployment: {orphaned_apps}")
        
        # Check all topics have routes
        topics_with_routes = {r['to'] for r in graph['relationships']['routes']}
        orphaned_topics = topic_ids - topics_with_routes
        if orphaned_topics:
            errors.append(f"Topics without broker routes: {orphaned_topics}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Validation error: {e}"]


def export_graph(graph: Dict, output_path: str, formats: List[str] = None):
    """
    Export graph to multiple formats
    
    Args:
        graph: Graph dictionary
        output_path: Base output path
        formats: List of formats ['json', 'graphml', 'gexf', 'pickle']
    """
    if formats is None:
        formats = ['json']
    
    output_path = Path(output_path)
    
    # Always save JSON
    if 'json' in formats:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")
    
    # Export to other formats if requested
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
                
                export_path = output_path.with_suffix(f'.{fmt}')
                
                if fmt == 'graphml':
                    exporter.to_graphml(str(export_path))
                elif fmt == 'gexf':
                    exporter.to_gexf(str(export_path))
                elif fmt == 'pickle':
                    exporter.to_pickle(str(export_path))
                
                print(f"✓ Saved {fmt.upper()}: {export_path}")
        
        except Exception as e:
            print(f"Warning: Could not export to additional formats: {e}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Generate realistic DDS pub-sub system graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small generic system
  python generate_graph.py --scale small --output small_system.json
  
  # Large IoT system with high availability
  python generate_graph.py --scale large --scenario iot --ha --output iot_large.json
  
  # Medium system with anti-patterns
  python generate_graph.py --scale medium --antipatterns spof broker_overload \\
      --output antipattern_system.json
  
  # Generate and validate
  python generate_graph.py --scale medium --validate --output validated.json
  
  # Export to multiple formats
  python generate_graph.py --scale small --formats json graphml gexf \\
      --output multi_format
        """
    )
    
    # Scale and scenario
    parser.add_argument('--scale', '-s',
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme'],
                       default='medium',
                       help='System scale')
    
    parser.add_argument('--scenario', '-c',
                       choices=['generic', 'iot', 'financial', 'ecommerce', 'analytics'],
                       default='generic',
                       help='Domain scenario')
    
    # Custom parameters (override scale defaults)
    parser.add_argument('--nodes', type=int, help='Number of nodes')
    parser.add_argument('--apps', type=int, help='Number of applications')
    parser.add_argument('--topics', type=int, help='Number of topics')
    parser.add_argument('--brokers', type=int, help='Number of brokers')
    
    # Configuration
    parser.add_argument('--density', type=float, default=0.3,
                       help='Edge density (0.0-1.0)')
    parser.add_argument('--ha', '--high-availability',
                       action='store_true',
                       help='Enable high availability patterns')
    parser.add_argument('--antipatterns', nargs='+',
                       choices=['spof', 'broker_overload', 'god_object', 
                               'single_broker', 'tight_coupling'],
                       help='Apply anti-patterns')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output
    parser.add_argument('--output', '-o', required=True,
                       help='Output file path')
    parser.add_argument('--formats', '-f', nargs='+',
                       choices=['json', 'graphml', 'gexf', 'pickle'],
                       default=['json'],
                       help='Output formats')
    
    # Validation
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Validate generated graph')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
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
        high_availability=args.ha,
        antipatterns=args.antipatterns or [],
        seed=args.seed
    )
    
    # Generate graph
    print(f"\nGenerating {config.scale} scale {config.scenario} system...")
    print(f"  Nodes: {config.num_nodes}")
    print(f"  Applications: {config.num_applications}")
    print(f"  Topics: {config.num_topics}")
    print(f"  Brokers: {config.num_brokers}")
    if config.antipatterns:
        print(f"  Anti-patterns: {', '.join(config.antipatterns)}")
    print()
    
    generator = GraphGenerator(config)
    graph = generator.generate()
    
    # Validate if requested
    if args.validate:
        print("Validating graph...")
        is_valid, errors = validate_graph(graph)
        
        if is_valid:
            print("✓ Graph validation passed")
        else:
            print("✗ Graph validation failed:")
            for error in errors:
                print(f"  - {error}")
            
            response = input("\nContinue with invalid graph? [y/N]: ")
            if response.lower() != 'y':
                return 1
    
    # Export
    print("\nExporting graph...")
    export_graph(graph, args.output, args.formats)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    
    print(f"\nComponents:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Applications: {len(graph['applications'])}")
    print(f"  Topics: {len(graph['topics'])}")
    print(f"  Brokers: {len(graph['brokers'])}")
    
    print(f"\nRelationships:")
    print(f"  Publishes: {len(graph['relationships']['publishes_to'])}")
    print(f"  Subscribes: {len(graph['relationships']['subscribes_to'])}")
    print(f"  Routes: {len(graph['relationships']['routes'])}")
    print(f"  Runs On: {len(graph['relationships']['runs_on'])}")
    
    # Component distribution
    app_types = {}
    for app in graph['applications']:
        app_type = app['type']
        app_types[app_type] = app_types.get(app_type, 0) + 1
    
    print(f"\nApplication Types:")
    for app_type, count in app_types.items():
        print(f"  {app_type}: {count}")
    
    print(f"\nSuccess! Graph saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
