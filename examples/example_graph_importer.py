#!/usr/bin/env python3
"""
Quick Start Example: Neo4j Graph Import

This script demonstrates the complete workflow for importing
pub-sub system graphs into Neo4j database.

Features demonstrated:
1. Prerequisites checking
2. Sample data generation
3. Graph import
4. Basic queries
5. Analytics
6. Visualization tips

Usage:
    python quick_start_example.py
    
    # With custom Neo4j connection
    python quick_start_example.py --uri bolt://localhost:7687 --user neo4j --password mypassword
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}\n")


def print_step(step: int, text: str):
    """Print a step number and description"""
    print(f"{Colors.BOLD}{Colors.CYAN}Step {step}: {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.ENDC}")


def check_prerequisites() -> bool:
    """Check if all prerequisites are met"""
    print_step(1, "Checking Prerequisites")
    
    all_good = True
    
    # Check Neo4j driver
    try:
        from neo4j import GraphDatabase
        print_success("neo4j driver is installed")
    except ImportError:
        print_error("neo4j driver not installed")
        print_info("Install with: pip install neo4j")
        all_good = False
    
    # Check Neo4j connection
    try:
        from neo4j import GraphDatabase, basic_auth
        
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            if result.single()["test"] == 1:
                print_success(f"Neo4j is running at {uri}")
            else:
                print_error("Neo4j connection test failed")
                all_good = False
        driver.close()
    except Exception as e:
        print_error(f"Cannot connect to Neo4j: {e}")
        print_info("Start Neo4j with:")
        print_info("  docker run -d -p 7474:7474 -p 7687:7687 \\")
        print_info("      -e NEO4J_AUTH=neo4j/password neo4j:latest")
        all_good = False
    
    # Check GraphImporter module
    try:
        from src.core.graph_importer import GraphImporter
        print_success("GraphImporter module found")
    except ImportError:
        print_error("GraphImporter module not found")
        print_info("Make sure src/core/graph_importer.py exists")
        all_good = False
    
    if all_good:
        print(f"\n{Colors.GREEN}All prerequisites met!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}Some prerequisites are missing. Please fix them first.{Colors.ENDC}")
    
    return all_good


def create_sample_data() -> Dict[str, Any]:
    """Create sample pub-sub system data"""
    print_step(2, "Creating Sample Data")
    
    sample_data = {
        "metadata": {
            "name": "Smart Home IoT System",
            "description": "Example distributed pub-sub system for smart home",
            "version": "1.0"
        },
        "nodes": [
            {
                "id": "node_01",
                "name": "MainHub"
            },
            {
                "id": "node_02",
                "name": "SecondaryHub"
            },
            {
                "id": "node_03",
                "name": "CloudGateway"
            }
        ],
        "applications": [
            {
                "id": "app_sensor_temp",
                "name": "TemperatureSensor",
                "type": "PRODUCER"
            },
            {
                "id": "app_sensor_motion",
                "name": "MotionSensor",
                "type": "PRODUCER"
            },
            {
                "id": "app_hvac_controller",
                "name": "HVACController",
                "type": "CONSUMER"
            },
            {
                "id": "app_security_system",
                "name": "SecuritySystem",
                "type": "CONSUMER"
            },
            {
                "id": "app_dashboard",
                "name": "MonitoringDashboard",
                "type": "CONSUMER"
            },
            {
                "id": "app_analytics",
                "name": "AnalyticsEngine",
                "type": "PROSUMER"
            },
            {
                "id": "app_cloud_sync",
                "name": "CloudSync",
                "type": "PROSUMER"
            }
        ],
        "topics": [
            {
                "id": "topic_temp",
                "name": "home/sensors/temperature",
                "message_size_bytes": 128,
                "message_rate_hz": 1,
                "qos": {
                    "durability": "TRANSIENT_LOCAL",
                    "reliability": "RELIABLE",
                    "history_depth": 10,
                    "deadline_ms": 5000,
                    "transport_priority": "MEDIUM"
                }
            },
            {
                "id": "topic_motion",
                "name": "home/sensors/motion",
                "message_size_bytes": 64,
                "message_rate_hz": 10,
                "qos": {
                    "durability": "VOLATILE",
                    "reliability": "RELIABLE",
                    "history_depth": 5,
                    "deadline_ms": 500,
                    "transport_priority": "HIGH"
                }
            },
            {
                "id": "topic_hvac_cmd",
                "name": "home/hvac/commands",
                "message_size_bytes": 256,
                "message_rate_hz": 0,
                "qos": {
                    "durability": "TRANSIENT_LOCAL",
                    "reliability": "RELIABLE",
                    "history_depth": 1,
                    "deadline_ms": 1000,
                    "transport_priority": "HIGH"
                }
            },
            {
                "id": "topic_alerts",
                "name": "home/security/alerts",
                "message_size_bytes": 512,
                "message_rate_hz": 0,
                "qos": {
                    "durability": "PERSISTENT",
                    "reliability": "RELIABLE",
                    "history_depth": 50,
                    "deadline_ms": 100,
                    "transport_priority": "CRITICAL"
                }
            },
            {
                "id": "topic_analytics",
                "name": "home/analytics/metrics",
                "message_size_bytes": 2048,
                "message_rate_hz": 0,
                "qos": {
                    "durability": "TRANSIENT_LOCAL",
                    "reliability": "BEST_EFFORT",
                    "history_depth": 20,
                    "deadline_ms": 10000,
                    "transport_priority": "LOW"
                }
            }
        ],
        "brokers": [
            {
                "id": "broker_main",
                "name": "MainBroker"
            },
            {
                "id": "broker_backup",
                "name": "BackupBroker"
            }
        ],
        "relationships": {
            "runs_on": [
                {"from": "app_sensor_temp", "to": "node_01"},
                {"from": "app_sensor_motion", "to": "node_01"},
                {"from": "app_hvac_controller", "to": "node_02"},
                {"from": "app_security_system", "to": "node_02"},
                {"from": "app_dashboard", "to": "node_01"},
                {"from": "app_analytics", "to": "node_03"},
                {"from": "app_cloud_sync", "to": "node_03"}
            ],
            "publishes_to": [
                {"from": "app_sensor_temp", "to": "topic_temp", "period_ms": 1000, "msg_size": 128},
                {"from": "app_sensor_motion", "to": "topic_motion", "period_ms": 100, "msg_size": 64},
                {"from": "app_security_system", "to": "topic_alerts", "period_ms": 0, "msg_size": 512},
                {"from": "app_analytics", "to": "topic_analytics", "period_ms": 60000, "msg_size": 2048}
            ],
            "subscribes_to": [
                {"from": "app_hvac_controller", "to": "topic_temp"},
                {"from": "app_security_system", "to": "topic_motion"},
                {"from": "app_dashboard", "to": "topic_temp"},
                {"from": "app_dashboard", "to": "topic_motion"},
                {"from": "app_dashboard", "to": "topic_alerts"},
                {"from": "app_analytics", "to": "topic_temp"},
                {"from": "app_analytics", "to": "topic_motion"},
                {"from": "app_analytics", "to": "topic_alerts"},
                {"from": "app_cloud_sync", "to": "topic_analytics"}
            ],
            "routes": [
                {"from": "broker_main", "to": "topic_temp"},
                {"from": "broker_main", "to": "topic_motion"},
                {"from": "broker_main", "to": "topic_hvac_cmd"},
                {"from": "broker_backup", "to": "topic_alerts"},
                {"from": "broker_backup", "to": "topic_analytics"}
            ]
        }
    }
    
    # Print summary
    print(f"\nCreated sample data:")
    print(f"  • {len(sample_data['nodes'])} infrastructure nodes")
    print(f"  • {len(sample_data['applications'])} applications")
    print(f"  • {len(sample_data['topics'])} topics")
    print(f"  • {len(sample_data['brokers'])} brokers")
    print(f"  • {len(sample_data['relationships']['runs_on'])} RUNS_ON relationships")
    print(f"  • {len(sample_data['relationships']['publishes_to'])} PUBLISHES_TO relationships")
    print(f"  • {len(sample_data['relationships']['subscribes_to'])} SUBSCRIBES_TO relationships")
    print(f"  • {len(sample_data['relationships']['routes'])} ROUTES relationships")
    
    print_success("Sample data created: Smart Home IoT System")
    
    return sample_data


def save_sample_data(data: Dict[str, Any], filename: str = "sample_smart_home.json") -> str:
    """Save sample data to file"""
    filepath = Path(filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print_success(f"Saved to: {filepath.absolute()}")
    
    return str(filepath)


def import_graph_to_neo4j(data: Dict[str, Any]):
    """Import graph to Neo4j"""
    print_step(3, "Importing Graph to Neo4j")
    
    try:
        from src.core.graph_importer import GraphImporter
        
        # Connect to Neo4j
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        database = "neo4j"
        
        print(f"Connecting to Neo4j at {uri}...")
        importer = GraphImporter(uri, user, password, database)
        
        try:
            # Clear database (optional)
            print("\nClearing existing data...")
            importer.clear_database()
            print_success("Database cleared")
            
            # Create schema
            print("\nCreating schema (constraints and indexes)...")
            importer.create_schema()
            print_success("Schema created")
            
            # Import graph
            print("\nImporting graph data...")
            start_time = time.time()
            importer.import_graph(data, batch_size=50)
            duration = time.time() - start_time
            print_success(f"Graph imported in {duration:.2f}s")
            
            # Get statistics
            print("\nImport Statistics:")
            stats = importer.get_statistics()
            
            print(f"\n  Nodes:")
            for node_type, count in stats['nodes'].items():
                if node_type != 'total':
                    print(f"    {node_type:20s}: {count:3d}")
            print(f"    {'Total':20s}: {stats['nodes']['total']:3d}")
            
            print(f"\n  Relationships:")
            for rel_type, count in stats['relationships'].items():
                if rel_type != 'total':
                    print(f"    {rel_type:20s}: {count:3d}")
            print(f"    {'Total':20s}: {stats['relationships']['total']:3d}")
            
            print_success(f"Import complete!")
            
            return True
            
        finally:
            importer.close()
            
    except Exception as e:
        print_error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_sample_queries():
    """Run sample queries to demonstrate Neo4j capabilities"""
    print_step(4, "Running Sample Queries")
    
    try:
        from neo4j import GraphDatabase, basic_auth
        
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        
        with driver.session() as session:
            # Query 1: Count components
            print("\n1. Component Counts:")
            
            result = session.run("MATCH (a:Application) RETURN count(a) as count")
            app_count = result.single()["count"]
            print(f"   Applications: {app_count}")
            
            result = session.run("MATCH (t:Topic) RETURN count(t) as count")
            topic_count = result.single()["count"]
            print(f"   Topics:       {topic_count}")
            
            result = session.run("MATCH (b:Broker) RETURN count(b) as count")
            broker_count = result.single()["count"]
            print(f"   Brokers:      {broker_count}")
            
            # Query 2: Most connected application
            print("\n3. Most Connected Application:")
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
                WITH a, count(DISTINCT t) as topic_count
                RETURN a.name, topic_count
                ORDER BY topic_count DESC
                LIMIT 1
            """)
            record = result.single()
            if record:
                print(f"   {record['a.name']} connected to {record['topic_count']} topics")
            
            # Query 3: Dependencies
            print("\n4. Application Dependencies:")
            result = session.run("""
                MATCH (consumer:Application)-[:DEPENDS_ON]->(producer:Application)
                RETURN consumer.name, producer.name
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['consumer.name']} depends on {record['producer.name']}")
            
            # Query 4: High-frequency topics
            print("\n6. High-Frequency Topics:")
            result = session.run("""
                MATCH (t:Topic)
                WHERE t.message_rate_hz > 0
                RETURN t.name, t.message_rate_hz
                ORDER BY t.message_rate_hz DESC
            """)
            for record in result:
                print(f"   • {record['t.name']:40s} ({record['t.message_rate_hz']} Hz)")
        
        driver.close()
        print_success("Sample queries completed")
        
    except Exception as e:
        print_error(f"Query execution failed: {e}")


def show_visualization_tips():
    """Show tips for visualizing the graph"""
    print_step(5, "Visualization Tips")
    
    print("\n" + Colors.BOLD + "Access Neo4j Browser:" + Colors.ENDC)
    print(f"  URL: {Colors.CYAN}http://localhost:7474{Colors.ENDC}")
    print(f"  Username: neo4j")
    print(f"  Password: password")
    
    print("\n" + Colors.BOLD + "Try these Cypher queries in Neo4j Browser:" + Colors.ENDC)
    
    print(f"\n1. View entire system (small graphs only):")
    print(f"   {Colors.YELLOW}MATCH (n) RETURN n LIMIT 50{Colors.ENDC}")
    
    print(f"\n2. View pub-sub network:")
    print(f"   {Colors.YELLOW}MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic){Colors.ENDC}")
    print(f"   {Colors.YELLOW}RETURN a, r, t LIMIT 100{Colors.ENDC}")
    
    print(f"\n3. View dependencies:")
    print(f"   {Colors.YELLOW}MATCH (a:Application)-[r:DEPENDS_ON]->(b:Application){Colors.ENDC}")
    print(f"   {Colors.YELLOW}RETURN a, r, b{Colors.ENDC}")
    
    print(f"\n4. View critical components:")
    print(f"   {Colors.YELLOW}MATCH (a:Application){Colors.ENDC}")
    print(f"   {Colors.YELLOW}WHERE a.criticality = 'CRITICAL'{Colors.ENDC}")
    print(f"   {Colors.YELLOW}RETURN a{Colors.ENDC}")
    
    print("\n" + Colors.BOLD + "Styling tips:" + Colors.ENDC)
    print("  • Click on a node type in the legend")
    print("  • Adjust size, color, and captions")
    print("  • Use 'Style' panel for custom CSS")
    
    print_success("Ready to explore in Neo4j Browser!")


def show_next_steps():
    """Show next steps for users"""
    print_header("Next Steps")
    
    print(Colors.BOLD + "What to do next:" + Colors.ENDC)
    
    print(f"\n1. {Colors.CYAN}Explore in Neo4j Browser{Colors.ENDC}")
    print(f"   Open http://localhost:7474 and try the visualization queries")
    
    print(f"\n2. {Colors.CYAN}Import your own data{Colors.ENDC}")
    print(f"   python import_graph.py --uri bolt://localhost:7687 \\")
    print(f"       --user neo4j --password password \\")
    print(f"       --input your_system.json")
    
    print(f"\n3. {Colors.CYAN}Run advanced analytics{Colors.ENDC}")
    print(f"   python import_graph.py ... --analytics --export-stats stats.json")
    
    print(f"\n4. {Colors.CYAN}Learn more Cypher queries{Colors.ENDC}")
    print(f"   Check README.md for 15+ query examples")
    
    print(f"\n5. {Colors.CYAN}Run the test suite{Colors.ENDC}")
    print(f"   python test_import.py")
    
    print(f"\n6. {Colors.CYAN}Explore utilities{Colors.ENDC}")
    print(f"   python neo4j_utils.py --help")


def main():
    """Main function"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║          Neo4j Graph Import - Quick Start Example                ║")
    print("║                                                                   ║")
    print("║     Demonstrates importing pub-sub systems into Neo4j            ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(Colors.ENDC)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print(f"\n{Colors.RED}Cannot continue without prerequisites.{Colors.ENDC}")
        print("Please fix the issues above and try again.")
        return 1
    
    input("\nPress Enter to continue...")
    
    # Step 2: Create sample data
    print_header("Creating Sample Smart Home IoT System")
    sample_data = create_sample_data()
    
    # Save to file
    filepath = save_sample_data(sample_data)
    
    input("\nPress Enter to continue...")
    
    # Step 3: Import to Neo4j
    print_header("Importing to Neo4j")
    if not import_graph_to_neo4j(sample_data):
        print(f"\n{Colors.RED}Import failed. Check error messages above.{Colors.ENDC}")
        return 1
    
    input("\nPress Enter to continue...")
    
    # Step 4: Run sample queries
    print_header("Running Sample Queries")
    run_sample_queries()
    
    input("\nPress Enter to continue...")
    
    # Step 5: Visualization tips
    print_header("Visualization in Neo4j Browser")
    show_visualization_tips()
    
    # Show next steps
    show_next_steps()
    
    # Success
    print(f"\n{Colors.GREEN}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║                    ✓ Quick Start Complete!                       ║")
    print("║                                                                   ║")
    print("║     Your Smart Home IoT system is now in Neo4j and ready         ║")
    print("║              to explore and analyze!                              ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(Colors.ENDC)
    
    print(f"\nSample data file: {Colors.CYAN}{filepath}{Colors.ENDC}")
    print(f"Neo4j Browser: {Colors.CYAN}http://localhost:7474{Colors.ENDC}")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
