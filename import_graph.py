"""
Neo4j Graph Importer

Imports generated DDS pub-sub system graphs into Neo4j database.

Features:
- Automatic schema creation
- Node and relationship import
- Property mapping
- Constraint and index creation
- Batch import for large graphs
- Query examples
- Visualization support

Usage:
    # Import basic graph
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json
    
    # Import with custom database
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --database mydb --input system.json
    
    # Clear before import
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json --clear
    
    # Import and run sample queries
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password \\
        --input system.json --queries
"""

import argparse
import json
import sys
import logging
from pathlib import Path

from src.core.graph_importer import GraphImporter

# Add directory to path
sys.path.insert(0, str(Path(__file__).parent / '.'))

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Import DDS pub-sub graphs into Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic import
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json
  
  # Clear and import
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json --clear
  
  # Import with queries
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --input system.json --queries
  
  # Custom database
  python import_graph.py --uri bolt://localhost:7687 \\
      --user neo4j --password password --database mydb --input system.json

Neo4j Installation:
  # Docker
  docker run -d -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/password \\
      neo4j:latest
  
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
                       help='Input JSON file')
    parser.add_argument('--clear', action='store_true',
                       help='Clear database before import')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for imports (default: 100)')
    
    # Query options
    parser.add_argument('--queries', '-q', action='store_true',
                       help='Run sample queries after import')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    # Load graph data
    print(f"\nLoading graph from {args.input}...")
    try:
        with open(args.input) as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return 1
    
    print(f"  Loaded: {len(graph_data.get('nodes', []))} nodes, "
          f"{len(graph_data.get('applications', []))} apps, "
          f"{len(graph_data.get('topics', []))} topics, "
          f"{len(graph_data.get('brokers', []))} brokers")
    
    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {args.uri}...")
    try:
        importer = GraphImporter(args.uri, args.user, args.password, args.database)
    except Exception as e:
        print(f"\nError connecting to Neo4j: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -d -p 7474:7474 -p 7687:7687 \\")
        print("      -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return 1
    
    try:
        # Clear if requested
        if args.clear:
            response = input("\nClear database? This will delete ALL data! [y/N]: ")
            if response.lower() == 'y':
                importer.clear_database()
            else:
                print("Skipping clear")
        
        # Create schema
        importer.create_schema()
        
        # Import graph
        print()
        importer.import_graph(graph_data, args.batch_size)
        
        # Get statistics
        print("\n" + "=" * 70)
        print("IMPORT STATISTICS")
        print("=" * 70)
        
        stats = importer.get_statistics()
        
        print("\nNodes:")
        for node_type, count in stats['nodes'].items():
            if node_type != 'total':
                print(f"  {node_type}: {count}")
        print(f"  Total: {stats['nodes']['total']}")
        
        print("\nRelationships:")
        for rel_type, count in stats['relationships'].items():
            if rel_type != 'total':
                print(f"  {rel_type}: {count}")
        print(f"  Total: {stats['relationships']['total']}")
        
        # Run queries if requested
        if args.queries:
            importer.run_sample_queries()
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        
        print(f"\n✓ Graph imported to Neo4j database '{args.database}'")
        print(f"\nAccess Neo4j Browser: http://localhost:7474")
        print(f"  Username: {args.user}")
        print(f"  Password: ********")
        
        print("\nSample Cypher Queries:")
        print("  # View all applications")
        print("  MATCH (a:Application) RETURN a LIMIT 25")
        print()
        print("  # View pub-sub relationships")
        print("  MATCH (a:Application)-[r]->(t:Topic) RETURN a, r, t LIMIT 100")
        print()
        print("  # Find critical applications")
        print("  MATCH (a:Application) WHERE a.criticality = 'CRITICAL' RETURN a")
        
    finally:
        importer.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
