#!/usr/bin/env python3
"""
Import Graph Example - Version 5.0

Demonstrates the Neo4j import workflow using import_graph.py CLI.

This example shows:
1. Generating a graph for import
2. Understanding the import process
3. CLI options and their effects
4. Querying imported data in Neo4j
5. Working with DEPENDS_ON relationships and component weights

Requirements:
- Neo4j running (see Docker command below)
- neo4j Python driver: pip install neo4j

Usage:
    # Demo mode (no Neo4j required)
    python examples/example_import.py --demo
    
    # With Neo4j
    python examples/example_import.py --password your_password

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Terminal Output
# =============================================================================

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{title:^60}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{BLUE}ℹ{RESET} {msg}")


def print_code(code: str) -> None:
    """Print a code block."""
    for line in code.strip().split("\n"):
        print(f"  {GRAY}{line}{RESET}")


# =============================================================================
# Demo Mode (No Neo4j Required)
# =============================================================================

def run_demo_mode():
    """Run demo without Neo4j connection."""
    print_header("Import Graph Example - Demo Mode")
    
    print("This demo explains the import process without requiring Neo4j.")
    print("For live import, run with: --password <your_neo4j_password>")
    
    # Step 1: Generate graph
    print_section("Step 1: Generate Graph")
    
    from src.core import generate_graph
    
    graph = generate_graph(scale="small", scenario="iot", seed=42)
    
    print_success("Generated small IoT graph")
    print(f"  Applications: {len(graph['applications'])}")
    print(f"  Brokers: {len(graph['brokers'])}")
    print(f"  Topics: {len(graph['topics'])}")
    print(f"  Nodes: {len(graph['nodes'])}")
    
    # Save to file
    output_path = Path("output/example_import_graph.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)
    
    print_success(f"Saved to {output_path}")
    
    # Step 2: Explain import process
    print_section("Step 2: Import Process")
    
    print("The import_graph.py CLI performs these steps:")
    print()
    print(f"  1. {BOLD}Connect{RESET} to Neo4j database")
    print(f"  2. {BOLD}Create schema{RESET} (constraints and indexes)")
    print(f"  3. {BOLD}Import vertices{RESET} (Application, Broker, Topic, Node)")
    print(f"  4. {BOLD}Import relationships{RESET} (PUBLISHES_TO, SUBSCRIBES_TO, etc.)")
    print(f"  5. {BOLD}Derive DEPENDS_ON{RESET} relationships with weights")
    print(f"  6. {BOLD}Calculate component weights{RESET} based on dependencies")
    
    # Step 3: Show CLI usage
    print_section("Step 3: CLI Usage")
    
    print("Basic import:")
    print_code(f"python import_graph.py --input {output_path}")
    
    print("\nWith custom connection:")
    print_code(f"""python import_graph.py --input {output_path} \\
    --uri bolt://localhost:7687 \\
    --password your_password""")
    
    print("\nClear database and reimport:")
    print_code(f"python import_graph.py --input {output_path} --clear")
    
    print("\nShow analytics after import:")
    print_code(f"python import_graph.py --input {output_path} --analytics")
    
    print("\nExport useful Cypher queries:")
    print_code(f"python import_graph.py --input {output_path} --export-queries queries.cypher")
    
    # Step 4: Explain DEPENDS_ON derivation
    print_section("Step 4: DEPENDS_ON Derivation")
    
    print("Four dependency types are derived:")
    print()
    
    deps = [
        ("app_to_app", "Subscriber → Publisher", "Shared topic data flow"),
        ("app_to_broker", "Application → Broker", "Topic routing dependency"),
        ("node_to_node", "Node → Node", "Cross-node app dependencies"),
        ("node_to_broker", "Node → Broker", "Infrastructure to middleware"),
    ]
    
    for dep_type, direction, desc in deps:
        print(f"  {BOLD}{dep_type:15}{RESET} {direction:25} {GRAY}({desc}){RESET}")
    
    print()
    print("Weight calculation:")
    print_code("""weight = topic_count + qos_score + size_factor

where:
  topic_count = number of shared topics
  qos_score   = PERSISTENT(0.4) + RELIABLE(0.3) + URGENT(0.3)
  size_factor = min(message_size / 10000, 0.5)""")
    
    # Step 5: Show sample queries
    print_section("Step 5: Sample Cypher Queries")
    
    print("After import, query the data:")
    print()
    
    print("View component weights (criticality):")
    print_code("""MATCH (c)
WHERE c:Application OR c:Broker OR c:Node
RETURN labels(c)[0] AS type, c.name, c.weight
ORDER BY c.weight DESC LIMIT 10""")
    
    print("\nView DEPENDS_ON relationships:")
    print_code("""MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name, d.dependency_type, d.weight, b.name
ORDER BY d.weight DESC LIMIT 20""")
    
    print("\nFind critical paths (multi-hop dependencies):")
    print_code("""MATCH path = (a:Application)-[:DEPENDS_ON*1..3]->(b)
RETURN [n IN nodes(path) | n.name] AS chain
LIMIT 20""")
    
    # Step 6: Docker quick start
    print_section("Step 6: Start Neo4j with Docker")
    
    print_code("""docker run -d --name neo4j \\
    -p 7474:7474 -p 7687:7687 \\
    -e NEO4J_AUTH=neo4j/password \\
    -e NEO4J_PLUGINS='["graph-data-science"]' \\
    neo4j:latest""")
    
    print()
    print("Then access Neo4j Browser at: http://localhost:7474")
    
    print_header("Demo Complete")
    
    print(f"Generated graph saved to: {output_path}")
    print()
    print("To import into Neo4j, run:")
    print_code(f"python import_graph.py --input {output_path} --password password --analytics")


# =============================================================================
# Live Mode (With Neo4j)
# =============================================================================

def run_live_mode(args):
    """Run with actual Neo4j connection."""
    print_header("Import Graph Example - Live Mode")
    
    # Check for neo4j driver
    try:
        from src.core import GraphImporter, generate_graph
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_info("Install neo4j driver: pip install neo4j")
        return 1
    
    # Generate graph
    print_section("Generating Graph")
    
    graph = generate_graph(scale=args.scale, scenario=args.scenario, seed=42)
    
    print_success(f"Generated {args.scale} {args.scenario} graph")
    print(f"  Applications: {len(graph['applications'])}")
    print(f"  Topics: {len(graph['topics'])}")
    
    # Import
    print_section("Importing to Neo4j")
    
    try:
        with GraphImporter(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as importer:
            
            print_success(f"Connected to {args.uri}")
            
            # Import with options
            stats = importer.import_graph(
                graph,
                clear_first=args.clear,
                derive_dependencies=True,
                calculate_weights=True,
            )
            
            print_success("Import complete!")
            print()
            print("Statistics:")
            print(f"  Applications: {stats.get('applications', 0)}")
            print(f"  Brokers: {stats.get('brokers', 0)}")
            print(f"  Topics: {stats.get('topics', 0)}")
            print(f"  Nodes: {stats.get('nodes', 0)}")
            print(f"  DEPENDS_ON: {stats.get('depends_on', 0)}")
            
            if 'depends_on_by_type' in stats:
                print()
                print("  DEPENDS_ON by type:")
                for dep_type, count in stats['depends_on_by_type'].items():
                    print(f"    {dep_type}: {count}")
            
            # Show analytics
            if args.analytics:
                print_section("Sample Analytics")
                importer.show_analytics()
            
            # Get database stats
            print_section("Database Statistics")
            db_stats = importer.get_database_stats()
            
            if 'component_weights' in db_stats:
                print("Component weight statistics:")
                for comp_type, weight_stats in db_stats['component_weights'].items():
                    print(f"  {comp_type}: avg={weight_stats['avg']:.2f}, max={weight_stats['max']:.2f}")
    
    except Exception as e:
        print_error(f"Import failed: {e}")
        print()
        print_info("Make sure Neo4j is running:")
        print_code("""docker run -d --name neo4j \\
    -p 7474:7474 -p 7687:7687 \\
    -e NEO4J_AUTH=neo4j/password \\
    neo4j:latest""")
        return 1
    
    print_header("Import Complete!")
    
    print("Access Neo4j Browser: http://localhost:7474")
    print()
    print("Try these queries:")
    print_code("""// Top critical components
MATCH (c)
WHERE c:Application OR c:Broker OR c:Node
RETURN labels(c)[0] AS type, c.name, round(c.weight * 100) / 100 AS weight
ORDER BY c.weight DESC LIMIT 10""")
    
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Example: Neo4j Graph Import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no Neo4j required)
  python examples/example_import.py --demo
  
  # Live import to Neo4j
  python examples/example_import.py --password your_password
  
  # With custom settings
  python examples/example_import.py --password secret --scale medium --scenario financial
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode without Neo4j",
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password",
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name",
    )
    parser.add_argument(
        "--scale",
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Graph scale",
    )
    parser.add_argument(
        "--scenario",
        default="iot",
        choices=["generic", "iot", "financial", "healthcare"],
        help="Graph scenario",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear database before import",
    )
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Show analytics after import",
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_mode()
        return 0
    else:
        return run_live_mode(args)


if __name__ == "__main__":
    sys.exit(main())
