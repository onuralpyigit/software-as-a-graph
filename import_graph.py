#!/usr/bin/env python3
"""
Neo4j Graph Importer CLI - Version 4.0

Imports generated pub-sub system graphs into Neo4j with:
- QoS-aware DEPENDS_ON relationship derivation
- Multi-layer dependency analysis (app, node, broker)
- Weight calculation based on topic QoS and size
- Progress reporting and statistics

Usage:
    # Basic import
    python import_graph.py --input system.json
    
    # With custom connection
    python import_graph.py --input system.json \\
        --uri bolt://localhost:7687 --password mypassword
    
    # Clear and reimport
    python import_graph.py --input system.json --clear
    
    # Show analytics after import
    python import_graph.py --input system.json --analytics
    
    # Export useful queries
    python import_graph.py --input system.json --export-queries queries.cypher

Docker Quick Start:
    docker run -d --name neo4j \\
        -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password \\
        -e NEO4J_PLUGINS='["graph-data-science"]' \\
        neo4j:latest
    
    # Access Browser: http://localhost:7474

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import GraphImporter


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


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_info(text: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


# =============================================================================
# Progress Callback
# =============================================================================

def make_progress_callback(quiet: bool):
    """Create a progress callback function"""
    if quiet:
        return None
    
    def callback(step: str, current: int, total: int):
        print(f"  {Colors.DIM}→{Colors.END} {step}: {current}/{total}")
    
    return callback


# =============================================================================
# Connection Help
# =============================================================================

def print_connection_help():
    print(f"""
{Colors.YELLOW}Connection Troubleshooting:{Colors.END}

1. Ensure Neo4j is running:
   docker ps | grep neo4j

2. Start Neo4j if needed:
   docker run -d --name neo4j \\
       -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password \\
       -e NEO4J_PLUGINS='["graph-data-science"]' \\
       neo4j:latest

3. Wait for startup (check logs):
   docker logs -f neo4j

4. Access browser:
   http://localhost:7474

5. Default credentials:
   Username: neo4j
   Password: password (or as configured)
""")


# =============================================================================
# Statistics Display
# =============================================================================

def print_statistics(stats: dict) -> None:
    """Print import statistics"""
    print_section("Import Statistics")
    
    nodes = stats.get("nodes", {})
    if nodes:
        print(f"  {Colors.BOLD}Nodes:{Colors.END}")
        for label, count in sorted(nodes.items()):
            print_kv(label, count, indent=4)
    
    rels = stats.get("relationships", {})
    if rels:
        print(f"  {Colors.BOLD}Relationships:{Colors.END}")
        for rel_type, count in sorted(rels.items()):
            print_kv(rel_type, count, indent=4)
    
    depends_on = stats.get("depends_on_by_type", {})
    if depends_on:
        print(f"  {Colors.BOLD}DEPENDS_ON by Type:{Colors.END}")
        for dep_type, count in sorted(depends_on.items()):
            print_kv(dep_type, count, indent=4)
    
    weight_stats = stats.get("weight_statistics", {})
    if weight_stats.get("average"):
        print(f"  {Colors.BOLD}Weight Statistics:{Colors.END}")
        print_kv("Average", f"{weight_stats.get('average', 0):.2f}", indent=4)
        print_kv("Min", f"{weight_stats.get('minimum', 0):.2f}", indent=4)
        print_kv("Max", f"{weight_stats.get('maximum', 0):.2f}", indent=4)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Import pub-sub graphs into Neo4j with DEPENDS_ON derivation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python import_graph.py --input system.json
    python import_graph.py --input system.json --uri bolt://localhost:7687 --password secret
    python import_graph.py --input system.json --clear --analytics
    python import_graph.py --input system.json --export-queries queries.cypher
        """,
    )
    
    # Input
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSON file with graph data",
    )
    
    # Neo4j connection
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password (default: password)",
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)",
    )
    
    # Import options
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear database before import",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for imports (default: 100)",
    )
    parser.add_argument(
        "--no-depends-on",
        action="store_true",
        help="Skip DEPENDS_ON relationship derivation",
    )
    
    # Output options
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Show sample analytics after import",
    )
    parser.add_argument(
        "--export-queries",
        type=Path,
        help="Export useful Cypher queries to file",
    )
    parser.add_argument(
        "--export-stats",
        type=Path,
        help="Export import statistics to JSON file",
    )
    parser.add_argument(
        "--export-graph",
        type=Path,
        help="Export graph from Neo4j to JSON file",
    )
    
    # General options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
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
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Validate input
    if not args.input.exists():
        print_error(f"Input file not found: {args.input}")
        return 1
    
    try:
        # Load graph data
        if not args.quiet:
            print_header("Neo4j Graph Import")
            print_section("Loading Graph")
        
        with open(args.input) as f:
            graph_data = json.load(f)
        
        if not args.quiet:
            metadata = graph_data.get("metadata", {})
            print_kv("File", args.input.name)
            print_kv("Scale", metadata.get("scale", "N/A"))
            print_kv("Scenario", metadata.get("scenario", "N/A"))
            
            metrics = graph_data.get("metrics", {}).get("vertex_counts", {})
            print_kv("Vertices", metrics.get("total", "N/A"))
        
        # Connect and import
        if not args.quiet:
            print_section("Connecting to Neo4j")
            print_kv("URI", args.uri)
            print_kv("Database", args.database)
        
        start_time = time.time()
        
        with GraphImporter(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as importer:
            
            # Import
            if not args.quiet:
                print_section("Importing Graph")
            
            counts = importer.import_graph(
                graph_data,
                batch_size=args.batch_size,
                clear_first=args.clear,
                derive_dependencies=not args.no_depends_on,
                progress_callback=make_progress_callback(args.quiet),
            )
            
            import_duration = time.time() - start_time
            
            # Get and display statistics
            stats = importer.get_statistics()
            
            if not args.quiet:
                print_statistics(stats)
                
                print_section("Performance")
                total_items = sum(stats.get("nodes", {}).values()) + sum(stats.get("relationships", {}).values())
                rate = total_items / import_duration if import_duration > 0 else 0
                print_kv("Duration", f"{import_duration:.2f}s")
                print_kv("Items/sec", f"{rate:.1f}")
            
            # Show analytics
            if args.analytics:
                print_section("Sample Analytics")
                importer.run_sample_queries()
            
            # Export queries
            if args.export_queries:
                importer.export_cypher_queries(str(args.export_queries))
                if not args.quiet:
                    print_success(f"Queries exported to {args.export_queries}")
            
            # Export stats
            if args.export_stats:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "source_file": str(args.input),
                    "import_duration": import_duration,
                    **stats,
                }
                with open(args.export_stats, "w") as f:
                    json.dump(export_data, f, indent=2)
                if not args.quiet:
                    print_success(f"Statistics exported to {args.export_stats}")
            
            # Export graph
            if args.export_graph:
                importer.export_to_json(str(args.export_graph))
                if not args.quiet:
                    print_success(f"Graph exported to {args.export_graph}")
        
        # Success message
        if not args.quiet:
            print_section("Success!")
            print(f"""
{Colors.GREEN}✓{Colors.END} Graph imported to Neo4j database '{args.database}'

{Colors.BOLD}Access Neo4j Browser:{Colors.END}
  URL:      http://localhost:7474
  Username: {args.user}

{Colors.BOLD}Quick Cypher Queries:{Colors.END}
  // View all DEPENDS_ON relationships
  MATCH (a)-[d:DEPENDS_ON]->(b)
  RETURN a.name, d.dependency_type, d.weight, b.name
  ORDER BY d.weight DESC LIMIT 20

  // Find critical dependencies (high weight)
  MATCH (a)-[d:DEPENDS_ON]->(b)
  WHERE d.weight > 5
  RETURN a.name, b.name, d.weight
  ORDER BY d.weight DESC

  // Dependency chain analysis
  MATCH path = (a:Application)-[:DEPENDS_ON*1..3]->(b)
  RETURN [n IN nodes(path) | n.name] AS chain
  LIMIT 20
""")
        
        return 0
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install neo4j driver: pip install neo4j")
        return 1
    
    except KeyboardInterrupt:
        print_warning("\nImport interrupted by user")
        return 130
    
    except Exception as e:
        print_error(f"Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print_connection_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())