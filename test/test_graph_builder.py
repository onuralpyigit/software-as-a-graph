#!/usr/bin/env python3
"""
Test Script for Fixed GraphBuilder

Demonstrates how to use the fixed graph_builder.py to load dataset.json
and build a GraphModel with comprehensive error handling.

Usage:
    python test_graph_builder.py
    
    # Or with custom input path
    python test_graph_builder.py --input path/to/dataset.json
    
    # With verbose output
    python test_graph_builder.py --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.core.graph_builder import GraphBuilder


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_summary(model, builder):
    """Print a comprehensive summary of the built model"""
    summary = model.summary()
    
    print("\n" + "="*70)
    print(" " * 20 + "GRAPH MODEL SUMMARY")
    print("="*70)
    
    # Nodes
    print("\n📊 NODES:")
    print(f"  Total Nodes:        {summary['total_nodes']}")
    print(f"  ├─ Applications:    {summary['applications']}")
    print(f"  ├─ Topics:          {summary['topics']}")
    print(f"  ├─ Brokers:         {summary['brokers']}")
    print(f"  └─ Infrastructure:  {summary['infrastructure_nodes']}")
    
    # Edges
    print(f"\n🔗 EDGES:")
    print(f"  Total Edges:        {summary['total_edges']}")
    print(f"  ├─ Publishes:       {summary['publishes']}")
    print(f"  ├─ Subscribes:      {summary['subscribes']}")
    print(f"  ├─ Routes:          {summary['routes']}")
    print(f"  ├─ Runs On:         {summary['runs_on']}")
    print(f"  ├─ Connects To:     {summary['connects_to']}")
    print(f"  └─ Depends On:      {summary['depends_on']} (derived)")
    
    # Errors and warnings
    if builder.errors:
        print(f"\n⚠️  ERRORS: {len(builder.errors)}")
        for i, error in enumerate(builder.errors[:5], 1):
            print(f"  {i}. {error}")
        if len(builder.errors) > 5:
            print(f"  ... and {len(builder.errors) - 5} more")
    
    if builder.warnings:
        print(f"\n⚠️  WARNINGS: {len(builder.warnings)}")
        for i, warning in enumerate(builder.warnings[:5], 1):
            print(f"  {i}. {warning}")
        if len(builder.warnings) > 5:
            print(f"  ... and {len(builder.warnings) - 5} more")
    
    print("\n" + "="*70)


def print_sample_data(model):
    """Print samples of the loaded data"""
    print("\n" + "="*70)
    print(" " * 25 + "SAMPLE DATA")
    print("="*70)
    
    # Sample applications
    if model.applications:
        print("\n📱 Sample Applications:")
        for i, (name, app) in enumerate(list(model.applications.items())[:3], 1):
            print(f"  {i}. {name}")
            print(f"     Type: {app.app_type.value}")
            if app.qos_policy:
                print(f"     QoS: {app.qos_policy.durability.value} / "
                      f"{app.qos_policy.reliability.value}")
    
    # Sample topics
    if model.topics:
        print("\n💬 Sample Topics:")
        for i, (name, topic) in enumerate(list(model.topics.items())[:3], 1):
            print(f"  {i}. {name}")
            print(f"     Message Type: {topic.message_type}")
            if topic.qos_policy:
                print(f"     QoS: {topic.qos_policy.durability.value} / "
                      f"{topic.qos_policy.reliability.value}")
    
    # Sample brokers
    if model.brokers:
        print("\n🔄 Sample Brokers:")
        for i, (name, broker) in enumerate(list(model.brokers.items())[:3], 1):
            print(f"  {i}. {name}")
            print(f"     Type: {broker.broker_type}")
            print(f"     Max Topics: {broker.max_topics}")
    
    # Sample infrastructure nodes
    if model.nodes:
        print("\n🖥️  Sample Infrastructure Nodes:")
        for i, (name, node) in enumerate(list(model.nodes.items())[:3], 1):
            print(f"  {i}. {name}")
            print(f"     CPU: {node.cpu_cores} cores")
            print(f"     Memory: {node.memory_gb} GB")
    
    # Sample edges
    print("\n🔗 Sample Edges:")
    if model.publishes_edges:
        print(f"  Publishes (showing {min(3, len(model.publishes_edges))} of "
              f"{len(model.publishes_edges)}):")
        for i, edge in enumerate(model.publishes_edges[:3], 1):
            print(f"    {i}. {edge.source} → {edge.target}")
    
    if model.subscribes_edges:
        print(f"  Subscribes (showing {min(3, len(model.subscribes_edges))} of "
              f"{len(model.subscribes_edges)}):")
        for i, edge in enumerate(model.subscribes_edges[:3], 1):
            print(f"    {i}. {edge.source} → {edge.target}")
    
    print("\n" + "="*70)


def print_analysis_hints(model):
    """Print suggestions for next steps"""
    summary = model.summary()
    
    print("\n" + "="*70)
    print(" " * 20 + "NEXT STEPS & ANALYSIS")
    print("="*70)
    
    print("\n✅ Graph loaded successfully! Here's what you can do next:")
    
    print("\n1. Export to NetworkX for analysis:")
    print("   from src.core.graph_exporter import GraphExporter")
    print("   exporter = GraphExporter()")
    print("   nx_graph = exporter.export_to_networkx(model)")
    
    print("\n2. Run centrality analysis:")
    print("   from src.analysis.centrality_analyzer import CentralityAnalyzer")
    print("   analyzer = CentralityAnalyzer(model)")
    print("   results = analyzer.analyze_all()")
    
    print("\n3. Detect anti-patterns:")
    print("   from src.analysis.topological_criticality_analyzer import TopologicalCriticalityAnalyzer")
    print("   analyzer = TopologicalCriticalityAnalyzer(model)")
    print("   patterns = analyzer.detect_antipatterns()")
    
    print("\n4. Simulate failures:")
    print("   from src.simulation.failure_simulator import FailureSimulator")
    print("   simulator = FailureSimulator(model)")
    print("   impact = simulator.simulate_component_failure('A1')")
    
    print("\n5. Export to other formats:")
    print("   exporter.export_to_graphml(model, 'output/graph.graphml')")
    print("   exporter.export_to_dot(model, 'output/graph.dot')")
    
    print("\n" + "="*70)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Test the fixed GraphBuilder with dataset.json',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        default='input/dataset.json',
        help='Path to input JSON file (default: input/dataset.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--show-samples',
        action='store_true',
        help='Show sample data from the loaded graph'
    )
    parser.add_argument(
        '--show-hints',
        action='store_true',
        help='Show analysis hints and next steps'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Print header
    print("\n" + "="*70)
    print(" " * 15 + "GRAPHBUILDER TEST SCRIPT")
    print("="*70)
    print(f"\nInput File: {args.input}")
    
    # Create builder and load graph
    print("\n🔧 Initializing GraphBuilder...")
    builder = GraphBuilder()
    
    try:
        print(f"📂 Loading graph from {args.input}...")
        model = builder.build_from_dict(
            builder.build_from_json(args.input).__dict__ 
            if hasattr(builder.build_from_json(args.input), '__dict__')
            else {}
        ) if False else builder.build_from_json(args.input)
        
        # Print summary
        print_summary(model, builder)
        
        # Show sample data if requested
        if args.show_samples:
            print_sample_data(model)
        
        # Show hints if requested
        if args.show_hints:
            print_analysis_hints(model)
        
        # Final status
        print("\n✅ SUCCESS: Graph model built successfully!")
        
        if builder.errors:
            print(f"\n⚠️  Note: {len(builder.errors)} errors occurred during build")
            print("   These may indicate data quality issues but the graph was still built.")
        
        if builder.warnings:
            print(f"\n⚠️  Note: {len(builder.warnings)} warnings occurred during build")
            print("   These are minor issues that don't prevent graph construction.")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found")
        print(f"   {e}")
        print(f"\n💡 Tip: Make sure the input file exists at: {args.input}")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to build graph")
        print(f"   {type(e).__name__}: {e}")
        
        if args.verbose:
            import traceback
            print("\n" + "="*70)
            print("DETAILED TRACEBACK:")
            print("="*70)
            traceback.print_exc()
        
        print(f"\n💡 Tip: Run with --verbose flag for detailed error information")
        return 1


if __name__ == '__main__':
    sys.exit(main())
