#!/usr/bin/env python3
"""
Neo4j Graph Importer - Enhanced Version

Imports generated DDS pub-sub system graphs into Neo4j database with:
- Comprehensive error handling and validation
- Progress reporting with detailed statistics
- Transaction management and retry logic
- Schema validation and migration support
- Advanced query examples
- Export capabilities

Usage:
    # Basic import
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json
    
    # Import with validation and progress
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json \\
        --validate --progress
    
    # Clear and import with custom batch size
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json \\
        --clear --batch-size 500
    
    # Import and run analytics
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json \\
        --queries --analytics
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.graph_importer import GraphImporter
except ImportError:
    print("Error: Could not import GraphImporter from src.core.graph_importer")
    print("Make sure the src directory exists and contains the graph_importer module")
    sys.exit(1)


def validate_graph_data(data: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate graph data structure
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check required top-level keys
    if not isinstance(data, dict):
        errors.append("Graph data must be a dictionary")
        return False, errors
    
    # Check for expected sections
    expected_sections = ['nodes', 'applications', 'topics', 'brokers', 'relationships']
    for section in expected_sections:
        if section not in data:
            errors.append(f"Missing section: {section}")
    
    # Validate relationships structure
    if 'relationships' in data:
        rel_data = data['relationships']
        if not isinstance(rel_data, dict):
            errors.append("Relationships must be a dictionary")
        else:
            expected_rels = ['runs_on', 'publishes_to', 'subscribes_to', 'routes']
            for rel_type in expected_rels:
                if rel_type not in rel_data:
                    errors.append(f"Missing relationship type: {rel_type}")
    
    # Check for empty data
    if data.get('applications', []) == [] and data.get('nodes', []) == []:
        errors.append("Graph appears to be empty (no applications or nodes)")
    
    return len(errors) == 0, errors


def print_validation_report(data: Dict[str, Any]):
    """Print detailed validation report"""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    
    # Component counts
    node_count = len(data.get('nodes', []))
    app_count = len(data.get('applications', []))
    topic_count = len(data.get('topics', []))
    broker_count = len(data.get('brokers', []))
    
    print(f"\nComponents:")
    print(f"  Infrastructure Nodes: {node_count}")
    print(f"  Applications:         {app_count}")
    print(f"  Topics:              {topic_count}")
    print(f"  Brokers:             {broker_count}")
    print(f"  Total Components:    {node_count + app_count + topic_count + broker_count}")
    
    # Relationship counts
    relationships = data.get('relationships', {})
    runs_on = len(relationships.get('runs_on', []))
    publishes = len(relationships.get('publishes_to', []))
    subscribes = len(relationships.get('subscribes_to', []))
    routes = len(relationships.get('routes', []))
    
    print(f"\nRelationships:")
    print(f"  RUNS_ON:       {runs_on}")
    print(f"  PUBLISHES_TO:  {publishes}")
    print(f"  SUBSCRIBES_TO: {subscribes}")
    print(f"  ROUTES:        {routes}")
    print(f"  Total:         {runs_on + publishes + subscribes + routes}")
    
    # Check for potential issues
    print(f"\nPotential Issues:")
    issues = []
    
    if app_count > 0 and runs_on == 0:
        issues.append("Applications exist but none are assigned to nodes (RUNS_ON)")
    
    if topic_count > 0 and publishes == 0 and subscribes == 0:
        issues.append("Topics exist but no applications publish/subscribe to them")
    
    if broker_count > 0 and routes == 0:
        issues.append("Brokers exist but don't route any topics")
    
    # Check for orphaned components
    app_ids = {app['id'] for app in data.get('applications', [])}
    topic_ids = {topic['id'] for topic in data.get('topics', [])}
    node_ids = {node['id'] for node in data.get('nodes', [])}
    broker_ids = {broker['id'] for broker in data.get('brokers', [])}
    
    # Check runs_on references
    for rel in relationships.get('runs_on', []):
        if rel['from'] not in app_ids:
            issues.append(f"RUNS_ON references unknown application: {rel['from']}")
        if rel['to'] not in node_ids:
            issues.append(f"RUNS_ON references unknown node: {rel['to']}")
    
    if not issues:
        print("  ✓ No issues detected")
    else:
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  ⚠ {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")


def print_import_progress(phase: str, current: int, total: int, start_time: float):
    """Print progress bar for import operations"""
    if total == 0:
        return
    
    percent = 100 * current / total
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\r  {phase}: |{bar}| {percent:5.1f}% ({current}/{total}) "
          f"[{rate:.1f} items/s, ETA: {eta:.0f}s]", end='', flush=True)
    
    if current == total:
        print()  # New line when complete


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Import DDS pub-sub graphs into Neo4j with enhanced features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic import
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json
  
  # Import with validation and progress
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json \\
      --validate --progress
  
  # Clear database and import
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json --clear
  
  # Import with custom batch size for large graphs
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input large.json \\
      --batch-size 1000
  
  # Import and run all analytics
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json \\
      --queries --analytics --export-stats stats.json

Docker Quick Start:
  # Start Neo4j
  docker run -d -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/password neo4j:latest
  
  # Access Browser: http://localhost:7474
        """
    )
    
    # Connection parameters
    parser.add_argument('--uri', required=True,
                       help='Neo4j connection URI (e.g., bolt://localhost:7687)')
    parser.add_argument('--user', required=True,
                       help='Neo4j username')
    parser.add_argument('--password', required=True,
                       help='Neo4j password')
    parser.add_argument('--database', default='neo4j',
                       help='Database name (default: neo4j)')
    
    # Import options
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSON file containing graph data')
    parser.add_argument('--clear', action='store_true',
                       help='Clear database before import (WARNING: deletes all data)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for bulk imports (default: 100, use 500-1000 for large graphs)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate graph data before import')
    parser.add_argument('--progress', action='store_true',
                       help='Show detailed progress during import')
    
    # Query and analytics options
    parser.add_argument('--queries', '-q', action='store_true',
                       help='Run sample queries after import')
    parser.add_argument('--analytics', action='store_true',
                       help='Run advanced analytics (centrality, communities, etc.)')
    parser.add_argument('--export-stats', metavar='FILE',
                       help='Export statistics to JSON file')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (errors only)')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Print header
    if not args.quiet:
        print("\n" + "=" * 70)
        print("NEO4J GRAPH IMPORTER")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load graph data
    if not args.quiet:
        print(f"\n📂 Loading graph from: {args.input}")
    
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Error: Input file not found: {args.input}")
            return 1
        
        with open(input_path) as f:
            graph_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input file: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return 1
    
    # Validate graph data
    if args.validate:
        is_valid, errors = validate_graph_data(graph_data)
        if not is_valid:
            print("\n❌ Validation failed:")
            for error in errors:
                print(f"  • {error}")
            return 1
        print("✓ Validation passed")
    
    # Print validation report
    if not args.quiet:
        print_validation_report(graph_data)
    
    # Connect to Neo4j
    if not args.quiet:
        print(f"\n🔌 Connecting to Neo4j at {args.uri}...")
    
    try:
        importer = GraphImporter(
            args.uri, 
            args.user, 
            args.password, 
            args.database
        )
    except Exception as e:
        print(f"\n❌ Error connecting to Neo4j: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Neo4j is running")
        print("  2. Check connection URI (should be bolt://host:7687)")
        print("  3. Verify username and password")
        print("\nDocker Quick Start:")
        print("  docker run -d -p 7474:7474 -p 7687:7687 \\")
        print("      -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return 1
    
    try:
        # Clear database if requested
        if args.clear:
            if not args.quiet:
                response = input("\n⚠️  Clear database? This will delete ALL data! [y/N]: ")
                if response.lower() != 'y':
                    print("Skipping clear")
                else:
                    importer.clear_database()
            else:
                importer.clear_database()
        
        # Create schema
        if not args.quiet:
            print("\n🔧 Creating schema...")
        importer.create_schema()
        
        # Import graph
        if not args.quiet:
            print("\n📥 Importing graph...")
        
        import_start = time.time()
        importer.import_graph(
            graph_data, 
            args.batch_size,
            show_progress=args.progress
        )
        import_duration = time.time() - import_start
        
        # Get statistics
        if not args.quiet:
            print("\n" + "=" * 70)
            print("IMPORT STATISTICS")
            print("=" * 70)
        
        stats = importer.get_statistics()
        stats['import_duration_seconds'] = import_duration
        stats['import_timestamp'] = datetime.now().isoformat()
        stats['source_file'] = str(input_path)
        
        if not args.quiet:
            print(f"\nNodes:")
            for node_type, count in stats['nodes'].items():
                if node_type != 'total':
                    print(f"  {node_type:20s}: {count:6d}")
            print(f"  {'Total':20s}: {stats['nodes']['total']:6d}")
            
            print(f"\nRelationships:")
            for rel_type, count in stats['relationships'].items():
                if rel_type != 'total':
                    print(f"  {rel_type:20s}: {count:6d}")
            print(f"  {'Total':20s}: {stats['relationships']['total']:6d}")
            
            print(f"\nPerformance:")
            print(f"  Import Duration:     {import_duration:.2f}s")
            components_per_sec = stats['nodes']['total'] / import_duration if import_duration > 0 else 0
            print(f"  Components/sec:      {components_per_sec:.1f}")
        
        # Export statistics if requested
        if args.export_stats:
            with open(args.export_stats, 'w') as f:
                json.dump(stats, f, indent=2)
            if not args.quiet:
                print(f"\n✓ Statistics exported to: {args.export_stats}")
        
        # Run sample queries
        if args.queries:
            importer.run_sample_queries()
        
        # Run analytics
        if args.analytics:
            if not args.quiet:
                print("\n" + "=" * 70)
                print("ADVANCED ANALYTICS")
                print("=" * 70)
            importer.run_analytics()
        
        # Print success message
        if not args.quiet:
            print("\n" + "=" * 70)
            print("✅ SUCCESS!")
            print("=" * 70)
            
            print(f"\n✓ Graph successfully imported to Neo4j database '{args.database}'")
            print(f"✓ {stats['nodes']['total']} nodes and {stats['relationships']['total']} relationships created")
            
            print(f"\n🌐 Access Neo4j Browser: http://localhost:7474")
            print(f"   Username: {args.user}")
            print(f"   Password: {'*' * len(args.password)}")
            
            print("\n💡 Quick Cypher Queries:")
            print("   # View all applications")
            print("   MATCH (a:Application) RETURN a LIMIT 25")
            print()
            print("   # View pub-sub network")
            print("   MATCH (a:Application)-[r]->(t:Topic)")
            print("   RETURN a, r, t LIMIT 100")
            print()
            print("   # Find critical components")
            print("   MATCH (a:Application)")
            print("   WHERE a.criticality = 'CRITICAL'")
            print("   RETURN a")
            print()
            print("   # Analyze dependencies")
            print("   MATCH (a:Application)-[:DEPENDS_ON]->(b:Application)")
            print("   RETURN a.name, b.name")
            
            print("\n📚 For more queries, run with --queries or --analytics flags")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Import interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Import failed")
        print(f"\n❌ Error during import: {e}")
        return 1
    finally:
        importer.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
