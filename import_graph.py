#!/usr/bin/env python3
"""
Neo4j Graph Importer CLI - Version 5.0

Imports generated pub-sub system graphs into Neo4j with:
- QoS-aware DEPENDS_ON relationship derivation
- Multi-layer dependency analysis (app, node, broker)
- Weight calculation for edges and components
- Progress reporting and statistics

Usage:
    # Basic import
    python import_graph.py --input system.json
    
    # With custom connection
    python import_graph.py --input system.json \\
        --uri bolt://localhost:7687 --password mypassword
    
    # Clear and reimport with analytics
    python import_graph.py --input system.json --clear --analytics
    
    # Export queries for reference
    python import_graph.py --input system.json --export-queries queries.cypher

Docker Quick Start:
    docker run -d --name neo4j \\
        -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password \\
        -e NEO4J_PLUGINS='["graph-data-science"]' \\
        neo4j:latest
    
    # Access Browser: http://localhost:7474

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Terminal Output Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors."""
        for attr in ["HEADER", "BLUE", "CYAN", "GREEN", "YELLOW", "RED", "END", "BOLD", "DIM"]:
            setattr(cls, attr, "")


def should_use_colors() -> bool:
    """Check if terminal supports colors."""
    import os
    return (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and not os.getenv("NO_COLOR")
    )


def print_header(text: str) -> None:
    """Print a header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 40}{Colors.END}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ Error:{Colors.END} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}! Warning:{Colors.END} {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ{Colors.END} {message}")


def print_stat(label: str, value: str, indent: int = 2) -> None:
    """Print a statistic."""
    print(f"{' ' * indent}{Colors.DIM}{label}:{Colors.END} {value}")


def print_connection_help() -> None:
    """Print connection troubleshooting help."""
    print(f"""
{Colors.YELLOW}Troubleshooting:{Colors.END}
  1. Is Neo4j running?
     docker ps | grep neo4j
  
  2. Start Neo4j:
     docker run -d --name neo4j \\
         -p 7474:7474 -p 7687:7687 \\
         -e NEO4J_AUTH=neo4j/password \\
         -e NEO4J_PLUGINS='["graph-data-science"]' \\
         neo4j:latest
  
  3. Check connection settings:
     --uri bolt://localhost:7687
     --user neo4j
     --password <your-password>
""")


# =============================================================================
# Progress Callback
# =============================================================================

def create_progress_callback(quiet: bool = False):
    """Create a progress callback function."""
    if quiet:
        return None
    
    def callback(stage: str, current: int, total: int) -> None:
        pct = (current / total * 100) if total > 0 else 100
        print(f"  {stage}: {current:,}/{total:,} ({pct:.0f}%)")
    
    return callback


# =============================================================================
# Main Function
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import pub-sub system graphs into Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python import_graph.py --input system.json
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
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip component weight calculation",
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
    
    args = parser.parse_args()
    
    # Setup
    if not should_use_colors():
        Colors.disable()
    
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )
    elif not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )
    
    try:
        # Import the GraphImporter
        from src.core import GraphImporter
        
        # Validate input
        if not args.input.exists():
            print_error(f"Input file not found: {args.input}")
            return 1
        
        # Load graph data
        if not args.quiet:
            print_header("Neo4j Graph Import")
            print_info(f"Loading: {args.input}")
        
        with open(args.input) as f:
            graph_data = json.load(f)
        
        # Show graph summary
        if not args.quiet:
            print_section("Graph Summary")
            print_stat("Applications", str(len(graph_data.get("applications", []))))
            print_stat("Brokers", str(len(graph_data.get("brokers", []))))
            print_stat("Topics", str(len(graph_data.get("topics", []))))
            print_stat("Nodes", str(len(graph_data.get("nodes", []))))
            
            rels = graph_data.get("relationships", {})
            total_rels = sum(len(v) for v in rels.values())
            print_stat("Relationships", str(total_rels))
        
        # Connect and import
        if not args.quiet:
            print_section("Neo4j Connection")
            print_info(f"Connecting to {args.uri}...")
        
        start_time = time.time()
        
        with GraphImporter(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as importer:
            
            if not args.quiet:
                print_success("Connected to Neo4j")
                print_section("Importing Graph")
            
            # Import
            stats = importer.import_graph(
                graph_data,
                batch_size=args.batch_size,
                clear_first=args.clear,
                derive_dependencies=not args.no_depends_on,
                calculate_weights=not args.no_weights,
                progress_callback=create_progress_callback(args.quiet),
            )
            
            duration = time.time() - start_time
            
            # Show results
            if not args.quiet:
                print_section("Import Results")
                print_stat("Applications", str(stats.get("applications", 0)))
                print_stat("Brokers", str(stats.get("brokers", 0)))
                print_stat("Topics", str(stats.get("topics", 0)))
                print_stat("Nodes", str(stats.get("nodes", 0)))
                print_stat("DEPENDS_ON", str(stats.get("depends_on", 0)))
                
                if "depends_on_by_type" in stats:
                    for dep_type, count in stats["depends_on_by_type"].items():
                        print_stat(f"  {dep_type}", str(count), indent=4)
                
                print_stat("Duration", f"{duration:.2f}s")
                print_success("Import complete!")
            
            # Show analytics
            if args.analytics:
                print_section("Sample Analytics")
                importer.show_analytics()
            
            # Export queries
            if args.export_queries:
                importer.export_cypher_queries(str(args.export_queries))
                if not args.quiet:
                    print_success(f"Queries exported to {args.export_queries}")
            
            # Export statistics
            if args.export_stats:
                db_stats = importer.get_database_stats()
                all_stats = {**stats, "database": db_stats}
                with open(args.export_stats, "w") as f:
                    json.dump(all_stats, f, indent=2)
                if not args.quiet:
                    print_success(f"Statistics exported to {args.export_stats}")
            
            # Export graph
            if args.export_graph:
                importer.export_graph(str(args.export_graph))
                if not args.quiet:
                    print_success(f"Graph exported to {args.export_graph}")
        
        # Final instructions
        if not args.quiet:
            print(f"""
{Colors.GREEN}✓{Colors.END} Graph imported to Neo4j database '{args.database}'

{Colors.BOLD}Access Neo4j Browser:{Colors.END}
  URL:      http://localhost:7474
  Username: {args.user}

{Colors.BOLD}Quick Cypher Queries:{Colors.END}
  // View component criticality (by weight)
  MATCH (c)
  WHERE c:Application OR c:Broker OR c:Node
  RETURN labels(c)[0] AS type, c.name, round(c.weight * 100) / 100 AS weight
  ORDER BY c.weight DESC LIMIT 20

  // View DEPENDS_ON relationships
  MATCH (a)-[d:DEPENDS_ON]->(b)
  RETURN a.name, d.dependency_type, round(d.weight * 100) / 100 AS weight, b.name
  ORDER BY d.weight DESC LIMIT 20

  // Find critical dependencies
  MATCH (a)-[d:DEPENDS_ON]->(b)
  WHERE d.weight > 5
  RETURN a.name, b.name, d.weight
  ORDER BY d.weight DESC
""")
        
        return 0
    
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install neo4j driver: pip install neo4j")
        return 1
    
    except KeyboardInterrupt:
        print_warning("\nImport interrupted by user")
        return 130
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    
    except Exception as e:
        print_error(f"Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print_connection_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())