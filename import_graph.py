#!/usr/bin/env python3
"""
Neo4j Graph Importer CLI

Imports generated pub-sub system graphs into Neo4j database with:
- Batch processing for large graphs
- Comprehensive validation
- Progress reporting
- Schema management
- Unified DEPENDS_ON relationship derivation
- Advanced analytics queries
- Export capabilities

Usage Examples:
    # Basic import
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password --input system.json
    
    # Import with validation and progress
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password --input system.json \\
        --validate --progress
    
    # Clear database and import with analytics
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password --input system.json \\
        --clear --analytics
    
    # Import large graph with custom batch size
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password --input large.json \\
        --batch-size 500 --progress
    
    # Export Cypher queries for reference
    python import_graph.py --uri bolt://localhost:7687 \\
        --user neo4j --password password --input system.json \\
        --export-queries queries.cypher

Docker Quick Start:
    docker run -d --name neo4j \\
        -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password \\
        neo4j:latest
    
    # Access Browser: http://localhost:7474
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.graph_importer import GraphImporter
    HAS_IMPORTER = True
except ImportError as e:
    HAS_IMPORTER = False
    IMPORT_ERROR = str(e)


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

def validate_graph_data(data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate graph data structure before import.
    
    Args:
        data: Graph dictionary to validate
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required sections
    required_sections = ['applications', 'topics']
    for section in required_sections:
        if section not in data:
            errors.append(f"Missing required section: {section}")
        elif not isinstance(data[section], list):
            errors.append(f"Section '{section}' must be a list")
        elif len(data[section]) == 0:
            warnings.append(f"Section '{section}' is empty")
    
    if errors:
        return False, errors, warnings
    
    # Build ID sets for reference checking
    node_ids = {n['id'] for n in data.get('nodes', [])}
    broker_ids = {b['id'] for b in data.get('brokers', [])}
    app_ids = {a['id'] for a in data.get('applications', [])}
    topic_ids = {t['id'] for t in data.get('topics', [])}
    
    # Check for duplicate IDs
    all_items = (
        data.get('nodes', []) + 
        data.get('brokers', []) + 
        data.get('applications', []) + 
        data.get('topics', [])
    )
    seen_ids = set()
    for item in all_items:
        item_id = item.get('id')
        if item_id in seen_ids:
            errors.append(f"Duplicate ID: {item_id}")
        seen_ids.add(item_id)
    
    # Validate required fields in each item
    for app in data.get('applications', []):
        if 'id' not in app:
            errors.append("Application missing 'id' field")
        if 'name' not in app:
            warnings.append(f"Application {app.get('id', 'unknown')} missing 'name' field")
    
    for topic in data.get('topics', []):
        if 'id' not in topic:
            errors.append("Topic missing 'id' field")
        if 'name' not in topic:
            warnings.append(f"Topic {topic.get('id', 'unknown')} missing 'name' field")
    
    # Validate relationships
    relationships = data.get('relationships', {})
    
    # RUNS_ON validation
    for rel in relationships.get('runs_on', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source and source not in app_ids and source not in broker_ids:
            # Allow anti-pattern IDs
            if not any(p in str(source) for p in ['spof_', 'coupling_', 'cycle_']):
                warnings.append(f"RUNS_ON source not found: {source}")
        if target and target not in node_ids:
            errors.append(f"RUNS_ON target not found: {target}")
    
    # PUBLISHES_TO validation
    for rel in relationships.get('publishes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source and source not in app_ids:
            if not any(p in str(source) for p in ['spof_', 'coupling_', 'cycle_']):
                warnings.append(f"PUBLISHES_TO source not in apps: {source}")
        if target and target not in topic_ids:
            if not any(p in str(target) for p in ['spof_', 'coupling_', 'god_', 'hidden_']):
                errors.append(f"PUBLISHES_TO target not found: {target}")
    
    # SUBSCRIBES_TO validation
    for rel in relationships.get('subscribes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source and source not in app_ids:
            if not any(p in str(source) for p in ['spof_', 'coupling_', 'cycle_']):
                warnings.append(f"SUBSCRIBES_TO source not in apps: {source}")
        if target and target not in topic_ids:
            if not any(p in str(target) for p in ['spof_', 'coupling_', 'god_', 'hidden_']):
                errors.append(f"SUBSCRIBES_TO target not found: {target}")
    
    # ROUTES validation
    for rel in relationships.get('routes', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        
        if source and source not in broker_ids:
            errors.append(f"ROUTES source not a broker: {source}")
        if target and target not in topic_ids:
            if not any(p in str(target) for p in ['spof_', 'coupling_', 'god_', 'hidden_']):
                warnings.append(f"ROUTES target not in topics: {target}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def print_validation_report(data: Dict[str, Any]):
    """Print a validation report for the graph data"""
    print_section("Graph Data Summary")
    
    # Count components
    nodes = data.get('nodes', [])
    brokers = data.get('brokers', [])
    apps = data.get('applications', [])
    topics = data.get('topics', [])
    
    print_kv("Infrastructure Nodes", len(nodes))
    print_kv("Brokers", len(brokers))
    print_kv("Applications", len(apps))
    print_kv("Topics", len(topics))
    
    # Count relationships
    relationships = data.get('relationships', {})
    print_kv("RUNS_ON", len(relationships.get('runs_on', [])))
    print_kv("PUBLISHES_TO", len(relationships.get('publishes_to', [])))
    print_kv("SUBSCRIBES_TO", len(relationships.get('subscribes_to', [])))
    print_kv("ROUTES", len(relationships.get('routes', [])))
    
    # Check for anti-patterns
    antipatterns = data.get('injected_antipatterns', [])
    if antipatterns:
        print_section("Injected Anti-Patterns")
        for ap in antipatterns:
            print(f"  {Colors.WARNING}•{Colors.ENDC} {ap.get('type', 'unknown')}: "
                  f"{ap.get('description', 'N/A')}")
    
    # Metadata
    metadata = data.get('metadata', {})
    if metadata:
        print_section("Metadata")
        config = metadata.get('config', {})
        print_kv("Scale", config.get('scale', 'N/A'))
        print_kv("Scenario", config.get('scenario', 'N/A'))
        print_kv("Generated", metadata.get('generated_at', 'N/A'))


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
          f"[{rate:.1f}/s, ETA: {eta:.0f}s]", end='', flush=True)
    
    if current == total:
        print()


# =============================================================================
# Quick Start Help
# =============================================================================

def print_quick_start():
    """Print quick start guide"""
    print_header("QUICK START GUIDE")
    
    print(f"""
{Colors.BOLD}1. Start Neo4j (using Docker):{Colors.ENDC}
   docker run -d --name neo4j \\
       -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password \\
       neo4j:latest

{Colors.BOLD}2. Generate a graph:{Colors.ENDC}
   python generate_graph.py --scale medium --scenario iot --output system.json

{Colors.BOLD}3. Import to Neo4j:{Colors.ENDC}
   python import_graph.py --uri bolt://localhost:7687 \\
       --user neo4j --password password \\
       --input system.json --validate

{Colors.BOLD}4. Access Neo4j Browser:{Colors.ENDC}
   Open http://localhost:7474 in your browser
   Username: neo4j
   Password: password

{Colors.BOLD}5. Try these Cypher queries:{Colors.ENDC}
   // View all applications
   MATCH (a:Application) RETURN a LIMIT 25
   
   // View pub-sub network
   MATCH (a:Application)-[r]->(t:Topic)
   RETURN a, r, t LIMIT 100
   
   // Find dependencies
   MATCH (a:Application)-[:DEPENDS_ON]->(b:Application)
   RETURN a.name, b.name LIMIT 50
""")


def print_connection_help():
    """Print connection troubleshooting help"""
    print(f"""
{Colors.WARNING}Connection Troubleshooting:{Colors.ENDC}

1. {Colors.BOLD}Check Neo4j is running:{Colors.ENDC}
   docker ps | grep neo4j
   # Or check: http://localhost:7474

2. {Colors.BOLD}Verify connection parameters:{Colors.ENDC}
   URI should be: bolt://localhost:7687 (not http)
   Default user: neo4j
   
3. {Colors.BOLD}Docker Quick Start:{Colors.ENDC}
   docker run -d --name neo4j \\
       -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password \\
       neo4j:latest

4. {Colors.BOLD}Check logs:{Colors.ENDC}
   docker logs neo4j

5. {Colors.BOLD}Reset password (if needed):{Colors.ENDC}
   docker exec neo4j neo4j-admin set-initial-password newpassword
""")


# =============================================================================
# Main CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Import pub-sub system graphs into Neo4j database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --uri bolt://localhost:7687 --user neo4j --password pass --input system.json
  %(prog)s --uri bolt://localhost:7687 --user neo4j --password pass --input system.json --clear --analytics
  %(prog)s --quick-start
        """
    )
    
    # Connection parameters
    conn_group = parser.add_argument_group('Connection')
    conn_group.add_argument(
        '--uri', '-U',
        default='bolt://localhost:7687',
        help='Neo4j connection URI (default: bolt://localhost:7687)'
    )
    conn_group.add_argument(
        '--user', '-u',
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )
    conn_group.add_argument(
        '--password', '-p',
        default='password',
        help='Neo4j password (default: password)'
    )
    conn_group.add_argument(
        '--database', '-d',
        default='neo4j',
        help='Database name (default: neo4j)'
    )
    
    # Input options
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--input', '-i',
        help='Input JSON file containing graph data'
    )
    input_group.add_argument(
        '--validate',
        action='store_true',
        help='Validate graph data before import'
    )
    
    # Import options
    import_group = parser.add_argument_group('Import Options')
    import_group.add_argument(
        '--clear',
        action='store_true',
        help='Clear database before import (WARNING: deletes all data)'
    )
    import_group.add_argument(
        '--batch-size',
        type=int, default=100,
        help='Batch size for bulk imports (default: 100)'
    )
    import_group.add_argument(
        '--progress',
        action='store_true',
        help='Show detailed progress during import'
    )
    import_group.add_argument(
        '--skip-dependencies',
        action='store_true',
        help='Skip DEPENDS_ON derivation'
    )
    
    # Query and analytics options
    analytics_group = parser.add_argument_group('Analytics')
    analytics_group.add_argument(
        '--queries', '-q',
        action='store_true',
        help='Run sample queries after import'
    )
    analytics_group.add_argument(
        '--analytics', '-a',
        action='store_true',
        help='Run advanced analytics after import'
    )
    
    # Export options
    export_group = parser.add_argument_group('Export')
    export_group.add_argument(
        '--export-stats',
        metavar='FILE',
        help='Export import statistics to JSON file'
    )
    export_group.add_argument(
        '--export-queries',
        metavar='FILE',
        help='Export useful Cypher queries to file'
    )
    export_group.add_argument(
        '--export-graph',
        metavar='FILE',
        help='Export graph from Neo4j to JSON file'
    )
    
    # Help options
    help_group = parser.add_argument_group('Help')
    help_group.add_argument(
        '--quick-start',
        action='store_true',
        help='Show quick start guide'
    )
    help_group.add_argument(
        '--connection-help',
        action='store_true',
        help='Show connection troubleshooting help'
    )
    
    # Verbosity
    verbosity_group = parser.add_argument_group('Verbosity')
    verbosity_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    verbosity_group.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (errors only)'
    )
    verbosity_group.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    return parser


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup colors
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
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
    
    # Help modes
    if args.quick_start:
        print_quick_start()
        return 0
    
    if args.connection_help:
        print_connection_help()
        return 0
    
    # Check for importer availability
    if not HAS_IMPORTER:
        print_error(f"Failed to import GraphImporter: {IMPORT_ERROR}")
        print_info("Make sure the src/core/graph_importer.py exists and neo4j is installed")
        print_info("Install neo4j driver: pip install neo4j")
        return 1
    
    # Require input file for most operations
    if not args.input and not args.export_graph:
        print_error("Input file required. Use --input to specify graph JSON file.")
        print_info("Use --quick-start for getting started guide")
        return 1
    
    # Print header
    if not args.quiet:
        print_header("NEO4J GRAPH IMPORTER")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Neo4j URI: {args.uri}")
        print(f"  Database:  {args.database}")
    
    # Load graph data
    graph_data = None
    if args.input:
        if not args.quiet:
            print_info(f"Loading graph from: {args.input}")
        
        try:
            input_path = Path(args.input)
            if not input_path.exists():
                print_error(f"Input file not found: {args.input}")
                return 1
            
            with open(input_path) as f:
                graph_data = json.load(f)
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON: {e}")
            return 1
        except Exception as e:
            print_error(f"Failed to load graph: {e}")
            return 1
        
        # Validate if requested
        if args.validate:
            if not args.quiet:
                print_info("Validating graph data...")
            
            is_valid, errors, warnings = validate_graph_data(graph_data)
            
            if errors:
                print_section("Validation Errors")
                for err in errors[:20]:
                    print_error(err)
                if len(errors) > 20:
                    print_error(f"... and {len(errors)-20} more errors")
            
            if warnings and args.verbose:
                print_section("Validation Warnings")
                for warn in warnings[:20]:
                    print_warning(warn)
                if len(warnings) > 20:
                    print_warning(f"... and {len(warnings)-20} more warnings")
            
            if not is_valid:
                print_error("Validation failed. Fix errors and retry.")
                return 1
            else:
                print_success(f"Validation passed ({len(warnings)} warnings)")
        
        # Print validation report
        if not args.quiet:
            print_validation_report(graph_data)
    
    # Connect to Neo4j
    if not args.quiet:
        print_info(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        importer = GraphImporter(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database
        )
        print_success("Connected to Neo4j")
    except Exception as e:
        print_error(f"Failed to connect to Neo4j: {e}")
        print_connection_help()
        return 1
    
    try:
        # Clear database if requested
        if args.clear:
            if not args.quiet:
                response = input(f"\n{Colors.WARNING}⚠ Clear ALL data from database? [y/N]: {Colors.ENDC}")
                if response.lower() != 'y':
                    print_info("Skipping clear")
                else:
                    print_info("Clearing database...")
                    importer.clear_database()
                    print_success("Database cleared")
            else:
                importer.clear_database()
        
        # Import graph data
        if graph_data:
            # Create schema
            if not args.quiet:
                print_info("Creating schema...")
            importer.create_schema()
            print_success("Schema created")
            
            # Import graph
            if not args.quiet:
                print_info("Importing graph...")
            
            import_start = time.time()
            importer.import_graph(
                graph_data,
                batch_size=args.batch_size,
                show_progress=args.progress
            )
            import_duration = time.time() - import_start
            
            print_success(f"Graph imported in {import_duration:.2f}s")
            
            # Get and display statistics
            if not args.quiet:
                stats = importer.get_statistics()
                
                print_section("Import Statistics")
                print(f"  {Colors.BOLD}Nodes:{Colors.ENDC}")
                for node_type, count in stats.get('nodes', {}).items():
                    if node_type != 'total':
                        print_kv(node_type, count, indent=4)
                print_kv("Total", stats.get('nodes', {}).get('total', 0), indent=4)
                
                print(f"\n  {Colors.BOLD}Relationships:{Colors.ENDC}")
                for rel_type, count in stats.get('relationships', {}).items():
                    if rel_type != 'total':
                        print_kv(rel_type, count, indent=4)
                print_kv("Total", stats.get('relationships', {}).get('total', 0), indent=4)
                
                print(f"\n  {Colors.BOLD}Performance:{Colors.ENDC}")
                total_items = stats.get('nodes', {}).get('total', 0) + stats.get('relationships', {}).get('total', 0)
                rate = total_items / import_duration if import_duration > 0 else 0
                print_kv("Duration", f"{import_duration:.2f}s", indent=4)
                print_kv("Items/sec", f"{rate:.1f}", indent=4)
                
                # Export stats if requested
                if args.export_stats:
                    stats['import_duration'] = import_duration
                    stats['timestamp'] = datetime.now().isoformat()
                    stats['source_file'] = str(args.input)
                    with open(args.export_stats, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print_success(f"Statistics exported to {args.export_stats}")
        
        # Run sample queries
        if args.queries:
            print_section("Sample Queries")
            importer.run_sample_queries()
        
        # Run analytics
        if args.analytics:
            print_section("Advanced Analytics")
            importer.run_analytics()
        
        # Export Cypher queries
        if args.export_queries:
            importer.export_cypher_queries(args.export_queries)
            print_success(f"Cypher queries exported to {args.export_queries}")
        
        # Export graph from Neo4j
        if args.export_graph:
            importer.export_to_json(args.export_graph)
            print_success(f"Graph exported to {args.export_graph}")
        
        # Print success message
        if not args.quiet and graph_data:
            print_section("Success!")
            print(f"""
{Colors.GREEN}✓{Colors.ENDC} Graph successfully imported to Neo4j database '{args.database}'

{Colors.BOLD}Access Neo4j Browser:{Colors.ENDC}
  URL:      http://localhost:7474
  Username: {args.user}
  Password: {'*' * len(args.password)}

{Colors.BOLD}Quick Cypher Queries:{Colors.ENDC}
  // View all applications
  MATCH (a:Application) RETURN a LIMIT 25
  
  // View pub-sub network
  MATCH (a:Application)-[r]->(t:Topic) RETURN a, r, t LIMIT 100
  
  // Find dependencies
  MATCH (a)-[:DEPENDS_ON]->(b) RETURN a.name, b.name LIMIT 50
  
  // Find critical components
  MATCH ()-[d:DEPENDS_ON]->(target)
  WITH target, count(*) AS dependents
  WHERE dependents >= 3
  RETURN target.name, dependents ORDER BY dependents DESC

{Colors.DIM}Use --queries or --analytics for more analysis{Colors.ENDC}
""")
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Import interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        logger.exception("Import failed")
        print_error(f"Import failed: {e}")
        return 1
    finally:
        importer.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())