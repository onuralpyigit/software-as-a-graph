#!/usr/bin/env python3
"""
Quick Example - Graph Analysis Demo

Demonstrates the graph analysis capabilities with a simple IoT smart home system.
Creates a sample graph, analyzes it, and displays results.

Usage:
    python example_graph_analyzer.py
    python example_graph_analyzer.py --save-graph
    python example_graph_analyzer.py --verbose
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def create_smart_home_example():
    """
    Create a sample smart home IoT system graph
    
    Architecture:
    - 2 infrastructure nodes (main controller, backup controller)
    - 1 MQTT broker
    - 5 sensor applications (temperature, motion, door, window, camera)
    - 2 actuator applications (HVAC, lights)
    - 1 dashboard application
    - 1 cloud sync application
    - 6 topics for sensor data and commands
    """
    return {
        "nodes": [
            {
                "id": "controller_main",
                "name": "controller_main"
            },
            {
                "id": "controller_backup",
                "name": "controller_backup"
            }
        ],
        "brokers": [
            {
                "id": "mqtt_broker",
                "name": "mqtt_broker"
            }
        ],
        "applications": [
            # Sensors
            {
                "id": "temp_sensor",
                "name": "temp_sensor",
                "type": "PRODUCER"
            },
            {
                "id": "motion_sensor",
                "name": "motion_sensor",
                "type": "PRODUCER"
            },
            {
                "id": "door_sensor",
                "name": "door_sensor",
                "type": "PRODUCER"
            },
            {
                "id": "window_sensor",
                "name": "window_sensor",
                "type": "PRODUCER"
            },
            {
                "id": "camera",
                "name": "camera",
                "type": "PRODUCER"
            },
            # Actuators
            {
                "id": "hvac_controller",
                "name": "hvac_controller",
                "type": "PROSUMER"
            },
            {
                "id": "light_controller",
                "name": "light_controller",
                "type": "PROSUMER"
            },
            # Dashboard and Cloud
            {
                "id": "dashboard",
                "name": "dashboard",
                "type": "PROSUMER",
            },
            {
                "id": "cloud_sync",
                "name": "cloud_sync",
                "type": "CONSUMER",
            }
        ],
        "topics": [
            {"id": "sensor/temperature", "name": "sensor/temperature"},
            {"id": "sensor/motion", "name": "sensor/motion"},
            {"id": "sensor/door", "name": "sensor/door"},
            {"id": "sensor/window", "name": "sensor/window"},
            {"id": "sensor/camera", "name": "sensor/camera"},
            {"id": "command/hvac", "name": "command/hvac"},
            {"id": "command/lights", "name": "command/lights"},
            {"id": "alerts", "name": "alerts"}
        ],
        "relationships": {
            "runs_on": [
                # Sensors on main controller
                {"from": "temp_sensor", "to": "controller_main"},
                {"from": "motion_sensor", "to": "controller_main"},
                {"from": "door_sensor", "to": "controller_main"},
                {"from": "window_sensor", "to": "controller_main"},
                {"from": "camera", "to": "controller_main"},
                # Actuators on main controller
                {"from": "hvac_controller", "to": "controller_main"},
                {"from": "light_controller", "to": "controller_main"},
                # Dashboard on backup controller
                {"from": "dashboard", "to": "controller_backup"},
                {"from": "cloud_sync", "to": "controller_backup"},
                # Broker on main controller (critical!)
                {"from": "mqtt_broker", "to": "controller_main"}
            ],
            "publishes_to": [
                # Sensors publish data
                {"from": "temp_sensor", "to": "sensor/temperature"},
                {"from": "motion_sensor", "to": "sensor/motion"},
                {"from": "door_sensor", "to": "sensor/door"},
                {"from": "window_sensor", "to": "sensor/window"},
                {"from": "camera", "to": "sensor/camera"},
                # Actuators publish to alert topic
                {"from": "hvac_controller", "to": "alerts"},
                {"from": "light_controller", "to": "alerts"},
                # Dashboard publishes commands
                {"from": "dashboard", "to": "command/hvac"},
                {"from": "dashboard", "to": "command/lights"}
            ],
            "subscribes_to": [
                # HVAC subscribes to temp and commands
                {"from": "hvac_controller", "to": "sensor/temperature"},
                {"from": "hvac_controller", "to": "command/hvac"},
                # Lights subscribe to motion and commands
                {"from": "light_controller", "to": "sensor/motion"},
                {"from": "light_controller", "to": "command/lights"},
                # Dashboard subscribes to all sensors
                {"from": "dashboard", "to": "sensor/temperature"},
                {"from": "dashboard", "to": "sensor/motion"},
                {"from": "dashboard", "to": "sensor/door"},
                {"from": "dashboard", "to": "sensor/window"},
                {"from": "dashboard", "to": "sensor/camera"},
                {"from": "dashboard", "to": "alerts"},
                # Cloud sync subscribes to everything
                {"from": "cloud_sync", "to": "sensor/temperature"},
                {"from": "cloud_sync", "to": "sensor/motion"},
                {"from": "cloud_sync", "to": "sensor/door"},
                {"from": "cloud_sync", "to": "sensor/window"},
                {"from": "cloud_sync", "to": "sensor/camera"},
                {"from": "cloud_sync", "to": "alerts"}
            ],
            "routes": [
                # Broker routes all topics
                {"from": "mqtt_broker", "to": "sensor/temperature"},
                {"from": "mqtt_broker", "to": "sensor/motion"},
                {"from": "mqtt_broker", "to": "sensor/door"},
                {"from": "mqtt_broker", "to": "sensor/window"},
                {"from": "mqtt_broker", "to": "sensor/camera"},
                {"from": "mqtt_broker", "to": "command/hvac"},
                {"from": "mqtt_broker", "to": "command/lights"},
                {"from": "mqtt_broker", "to": "alerts"}
            ]
        }
    }


def print_banner():
    """Print welcome banner"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}")
    print("   QUICK EXAMPLE: Pub-Sub Graph Analysis")
    print("   Smart Home IoT System Demo")
    print(f"{'='*70}{Colors.ENDC}\n")


def print_section(title):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.ENDC}")


def describe_system(graph_data):
    """Describe the example system"""
    print_section("System Overview")
    
    print(f"\n📦 Components:")
    print(f"   • Nodes: {len(graph_data['nodes'])}")
    print(f"   • Brokers: {len(graph_data['brokers'])}")
    print(f"   • Applications: {len(graph_data['applications'])}")
    print(f"   • Topics: {len(graph_data['topics'])}")
    
    print(f"\n🔗 Relationships:")
    for rel_type, relationships in graph_data['relationships'].items():
        print(f"   • {rel_type}: {len(relationships)}")
    
    print(f"\n📝 Application Types:")
    app_types = {}
    for app in graph_data['applications']:
        app_type = app.get('app_type', 'unknown')
        app_types[app_type] = app_types.get(app_type, 0) + 1
    for app_type, count in app_types.items():
        print(f"   • {app_type}: {count}")


def run_analysis(graph_file):
    """Run the analysis script"""
    print_section("Running Analysis")
    
    try:
        result = subprocess.run(
            ['python3', '../analyze_graph.py', 
             '--input', graph_file,
             '--export-json', '--output', 'graph_analysis_results'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"{Colors.BOLD}Error running analysis:{Colors.ENDC}")
            print(result.stderr)
            return None
        
        print_success("Analysis completed successfully")
        
        # Load results
        with open('graph_analysis_results.json', 'r') as f:
            results = json.load(f)
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"{Colors.BOLD}Analysis timed out{Colors.ENDC}")
        return None
    except Exception as e:
        print(f"{Colors.BOLD}Error: {e}{Colors.ENDC}")
        return None


def display_key_findings(results):
    """Display key findings from the analysis"""
    print_section("Key Findings")
    
    # Structure
    structure = results['structure']
    print(f"\n📊 Graph Structure:")
    print(f"   • Total Components: {structure['nodes']}")
    print(f"   • Total Connections: {structure['edges']}")
    print(f"   • Network Density: {structure['density']:.3f}")
    print(f"   • Connected: {'Yes' if structure['is_weakly_connected'] else 'No'}")
    
    # Critical nodes
    node_analysis = results['node_analysis']
    print(f"\n🔴 Critical Components:")
    print(f"   • Articulation Points: {node_analysis['statistics']['articulation_points_count']}")
    print(f"\n   Top 3 Most Critical Nodes:")
    for idx, node in enumerate(node_analysis['top_critical_nodes'][:3], 1):
        ap = "⚠️  (Single Point of Failure)" if node['is_articulation_point'] else ""
        print(f"   {idx}. {node['node']} (score: {node['score']:.3f}) {ap}")
    
    # Critical edges
    if 'edge_analysis' in results and not results['edge_analysis'].get('skipped'):
        edge_analysis = results['edge_analysis']
        print(f"\n🔗 Critical Connections:")
        print(f"   • Bridge Edges: {edge_analysis['statistics']['bridge_count']}")
        if edge_analysis['top_edge_betweenness']:
            print(f"\n   Top 3 Most Critical Edges:")
            for idx, edge in enumerate(edge_analysis['top_edge_betweenness'][:3], 1):
                print(f"   {idx}. {edge['from']} → {edge['to']} (score: {edge['score']:.3f})")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        critical_recs = [r for r in recommendations if r['priority'] == 'CRITICAL']
        high_recs = [r for r in recommendations if r['priority'] == 'HIGH']
        
        print(f"   • Critical Issues: {len(critical_recs)}")
        print(f"   • High Priority Issues: {len(high_recs)}")
        
        if critical_recs:
            print(f"\n   Top Critical Recommendation:")
            rec = critical_recs[0]
            print(f"   ⚠️  {rec['type']}: {rec['component']}")
            print(f"      {rec['issue']}")
            print(f"      → {rec['recommendation']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Quick example demonstration of graph analysis'
    )
    parser.add_argument('--save-graph', action='store_true',
                       help='Save the example graph to smart_home_example.json')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if analyze_graph.py exists
    if not Path('../analyze_graph.py').exists():
        print(f"{Colors.BOLD}Error: analyze_graph.py not found{Colors.ENDC}")
        print("Make sure you're in the correct directory")
        return 1
    
    try:
        print_banner()
        
        # Create example system
        print_info("Creating smart home IoT system example...")
        graph_data = create_smart_home_example()
        print_success("Example system created")
        
        # Describe system
        describe_system(graph_data)
        
        # Save to temporary file
        if args.save_graph:
            output_file = 'smart_home_example.json'
        else:
            import tempfile
            temp_fd, output_file = tempfile.mkstemp(suffix='.json')
            import os
            os.close(temp_fd)
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        if args.save_graph:
            print_success(f"Graph saved to {output_file}")
        
        # Run analysis
        results = run_analysis(output_file)
        
        if results:
            # Display key findings
            display_key_findings(results)
            
            # Next steps
            print_section("Next Steps")
            print("\n📚 To explore further:\n")
            print("   1. View the detailed HTML report:")
            print(f"      python3 analyze_graph.py --input {output_file} --export-html")
            print(f"      # Opens: graph_analysis_results.html\n")
            
            print("   2. Export to CSV for spreadsheet analysis:")
            print(f"      python3 analyze_graph.py --input {output_file} --export-csv")
            print(f"      # Creates: graph_analysis_results_csv/ directory\n")
            
            print("   3. Try with your own graph:")
            print(f"      python3 analyze_graph.py --input your_system.json\n")
            
            print("   4. Connect to Neo4j database:")
            print(f"      python3 analyze_graph.py --neo4j --uri bolt://localhost:7687 \\")
            print(f"          --user neo4j --password your_password\n")
            
            print(f"{Colors.GREEN}✓ Example completed successfully!{Colors.ENDC}\n")
        else:
            print(f"\n{Colors.BOLD}Analysis failed. Check error messages above.{Colors.ENDC}\n")
            return 1
        
        # Cleanup
        if not args.save_graph:
            import os
            os.unlink(output_file)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
        return 130
    except Exception as e:
        print(f"\n{Colors.BOLD}Error: {e}{Colors.ENDC}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
