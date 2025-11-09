#!/usr/bin/env python3
"""
Example: Comprehensive Visualization Workflow

Demonstrates complete visualization capabilities for distributed pub-sub systems:
1. Complete system visualization
2. Multi-layer analysis
3. Hidden dependency detection
4. Failure impact visualization
5. Scenario-based analysis
6. Metrics dashboard creation

This example shows how to use visualize_graph.py for research, documentation,
and system analysis purposes.
"""

import sys
from pathlib import Path
import json
import subprocess
import time

def create_example_system():
    """
    Create example pub-sub system for demonstration
    
    Returns a realistic system with:
    - 3 brokers (2 main + 1 backup)
    - 15 applications
    - 10 topics
    - 5 nodes
    - Hidden dependencies
    - Single points of failure
    """
    
    system = {
        "nodes": [],
        "applications": [],
        "topics": [],
        "brokers": []
    }
    
    # Physical nodes
    nodes = [
        {
            "id": "N1",
            "name": "MainDataCenter",
            "ip": "192.168.1.10",
            "type": "DataCenter",
            "capacity": 1000
        },
        {
            "id": "N2",
            "name": "BackupDataCenter",
            "ip": "192.168.1.11",
            "type": "DataCenter",
            "capacity": 800
        },
        {
            "id": "N3",
            "name": "EdgeNode1",
            "ip": "192.168.2.10",
            "type": "Edge",
            "capacity": 500
        },
        {
            "id": "N4",
            "name": "EdgeNode2",
            "ip": "192.168.2.11",
            "type": "Edge",
            "capacity": 500
        },
        {
            "id": "N5",
            "name": "CloudNode",
            "ip": "10.0.0.100",
            "type": "Cloud",
            "capacity": 2000
        }
    ]
    
    # Brokers
    brokers = [
        {
            "id": "B1",
            "name": "MainBroker",
            "node_id": "N1",
            "max_topics": 50,
            "max_applications": 100
        },
        {
            "id": "B2",
            "name": "BackupBroker",
            "node_id": "N2",
            "max_topics": 50,
            "max_applications": 100
        },
        {
            "id": "B3",
            "name": "EdgeBroker",
            "node_id": "N3",
            "max_topics": 30,
            "max_applications": 50
        }
    ]
    
    # Topics with QoS policies
    topics = [
        {
            "id": "T1",
            "name": "CriticalAlerts",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "PERSISTENT",
                "deadline_ms": 100,
                "transport_priority": 10
            },
            "publishers": ["A1", "A2"],
            "subscribers": ["A3", "A4", "A5"]
        },
        {
            "id": "T2",
            "name": "SensorData",
            "qos": {
                "reliability": "BEST_EFFORT",
                "durability": "VOLATILE",
                "deadline_ms": 1000,
                "transport_priority": 5
            },
            "publishers": ["A6", "A7", "A8"],
            "subscribers": ["A9", "A10"]
        },
        {
            "id": "T3",
            "name": "Commands",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "TRANSIENT_LOCAL",
                "deadline_ms": 500,
                "transport_priority": 8
            },
            "publishers": ["A3"],
            "subscribers": ["A11", "A12"]
        },
        {
            "id": "T4",
            "name": "StatusUpdates",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "TRANSIENT",
                "deadline_ms": 2000,
                "transport_priority": 3
            },
            "publishers": ["A11", "A12", "A13"],
            "subscribers": ["A1", "A14"]
        },
        {
            "id": "T5",
            "name": "Telemetry",
            "qos": {
                "reliability": "BEST_EFFORT",
                "durability": "VOLATILE",
                "deadline_ms": 5000,
                "transport_priority": 2
            },
            "publishers": ["A6", "A7", "A8", "A9"],
            "subscribers": ["A15"]
        },
        {
            "id": "T6",
            "name": "Configuration",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "PERSISTENT",
                "deadline_ms": 10000,
                "transport_priority": 9
            },
            "publishers": ["A1"],
            "subscribers": ["A6", "A7", "A8", "A11", "A12"]
        },
        {
            "id": "T7",
            "name": "Analytics",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "PERSISTENT",
                "deadline_ms": 30000,
                "transport_priority": 4
            },
            "publishers": ["A15"],
            "subscribers": ["A1", "A14"]
        },
        {
            "id": "T8",
            "name": "Events",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "TRANSIENT",
                "deadline_ms": 1000,
                "transport_priority": 6
            },
            "publishers": ["A3", "A4", "A5"],
            "subscribers": ["A9", "A10", "A13"]
        },
        {
            "id": "T9",
            "name": "Logs",
            "qos": {
                "reliability": "BEST_EFFORT",
                "durability": "VOLATILE",
                "deadline_ms": 60000,
                "transport_priority": 1
            },
            "publishers": ["A6", "A7", "A8", "A9", "A10", "A11", "A12"],
            "subscribers": ["A15"]
        },
        {
            "id": "T10",
            "name": "Diagnostics",
            "qos": {
                "reliability": "RELIABLE",
                "durability": "TRANSIENT_LOCAL",
                "deadline_ms": 5000,
                "transport_priority": 5
            },
            "publishers": ["A11", "A12", "A13"],
            "subscribers": ["A14"]
        }
    ]
    
    # Applications
    applications = [
        {
            "id": "A1",
            "name": "MasterController",
            "node_id": "N1",
            "broker_id": "B1",
            "criticality": "CRITICAL",
            "replicas": 2
        },
        {
            "id": "A2",
            "name": "AlertGenerator",
            "node_id": "N1",
            "broker_id": "B1",
            "criticality": "HIGH",
            "replicas": 1
        },
        {
            "id": "A3",
            "name": "CommandProcessor",
            "node_id": "N1",
            "broker_id": "B1",
            "criticality": "CRITICAL",
            "replicas": 2
        },
        {
            "id": "A4",
            "name": "AlertSubscriber1",
            "node_id": "N2",
            "broker_id": "B2",
            "criticality": "HIGH",
            "replicas": 1
        },
        {
            "id": "A5",
            "name": "AlertSubscriber2",
            "node_id": "N2",
            "broker_id": "B2",
            "criticality": "HIGH",
            "replicas": 1
        },
        {
            "id": "A6",
            "name": "Sensor1",
            "node_id": "N3",
            "broker_id": "B3",
            "criticality": "MEDIUM",
            "replicas": 0
        },
        {
            "id": "A7",
            "name": "Sensor2",
            "node_id": "N3",
            "broker_id": "B3",
            "criticality": "MEDIUM",
            "replicas": 0
        },
        {
            "id": "A8",
            "name": "Sensor3",
            "node_id": "N4",
            "broker_id": "B3",
            "criticality": "MEDIUM",
            "replicas": 0
        },
        {
            "id": "A9",
            "name": "DataAggregator",
            "node_id": "N1",
            "broker_id": "B1",
            "criticality": "HIGH",
            "replicas": 1
        },
        {
            "id": "A10",
            "name": "DataProcessor",
            "node_id": "N2",
            "broker_id": "B2",
            "criticality": "MEDIUM",
            "replicas": 1
        },
        {
            "id": "A11",
            "name": "Actuator1",
            "node_id": "N3",
            "broker_id": "B3",
            "criticality": "HIGH",
            "replicas": 1
        },
        {
            "id": "A12",
            "name": "Actuator2",
            "node_id": "N4",
            "broker_id": "B3",
            "criticality": "HIGH",
            "replicas": 1
        },
        {
            "id": "A13",
            "name": "StatusMonitor",
            "node_id": "N5",
            "broker_id": "B1",
            "criticality": "MEDIUM",
            "replicas": 0
        },
        {
            "id": "A14",
            "name": "Dashboard",
            "node_id": "N5",
            "broker_id": "B1",
            "criticality": "LOW",
            "replicas": 0
        },
        {
            "id": "A15",
            "name": "AnalyticsEngine",
            "node_id": "N5",
            "broker_id": "B2",
            "criticality": "MEDIUM",
            "replicas": 1
        }
    ]
    
    # Broker routing
    broker_routes = {
        "B1": ["T1", "T3", "T4", "T6", "T7", "T8"],
        "B2": ["T2", "T5", "T7", "T9", "T10"],
        "B3": ["T2", "T3", "T8", "T9", "T10"]
    }
    
    system["nodes"] = nodes
    system["brokers"] = brokers
    system["topics"] = topics
    system["applications"] = applications
    system["broker_routes"] = broker_routes
    
    return system


def run_command(cmd, description):
    """Run shell command and display output"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main():
    """Run comprehensive visualization examples"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE VISUALIZATION WORKFLOW - EXAMPLE")
    print("="*70)
    print("\nThis example demonstrates all visualization capabilities:")
    print("  1. Complete system visualization")
    print("  2. Multi-layer analysis")
    print("  3. Hidden dependency detection")
    print("  4. Failure impact visualization")
    print("  5. Failure scenarios")
    print("  6. Metrics dashboard")
    print("  7. Comprehensive report generation")
    
    # Create output directories
    output_dir = Path("visualization_examples")
    output_dir.mkdir(exist_ok=True)
    
    system_file = output_dir / "example_system.json"
    
    # Create example system
    print("\n" + "="*70)
    print("STEP 1: Creating Example System")
    print("="*70)
    
    system = create_example_system()
    
    with open(system_file, 'w') as f:
        json.dump(system, f, indent=2)
    
    print(f"✓ Created example system: {system_file}")
    print(f"  - {len(system['nodes'])} nodes")
    print(f"  - {len(system['brokers'])} brokers")
    print(f"  - {len(system['topics'])} topics")
    print(f"  - {len(system['applications'])} applications")
    
    # Example 1: Complete visualization
    print("\n" + "="*70)
    print("STEP 2: Complete System Visualization")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--complete",
            "--layout", "spring",
            "--color-scheme", "criticality",
            "--output-dir", str(output_dir / "complete")
        ],
        "Creating complete system visualization with criticality coloring"
    )
    
    time.sleep(1)
    
    # Example 2: Multi-layer visualization
    print("\n" + "="*70)
    print("STEP 3: Multi-Layer Visualization")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--multi-layer",
            "--output-dir", str(output_dir / "layers")
        ],
        "Creating multi-layer visualizations with cross-layer interactions"
    )
    
    time.sleep(1)
    
    # Example 3: Hidden dependencies
    print("\n" + "="*70)
    print("STEP 4: Hidden Dependencies Analysis")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--hidden-deps",
            "--threshold", "0.6",
            "--output-dir", str(output_dir / "hidden_deps")
        ],
        "Detecting and visualizing hidden dependencies"
    )
    
    time.sleep(1)
    
    # Example 4: Failure impact
    print("\n" + "="*70)
    print("STEP 5: Failure Impact Visualization")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--simulate-failure", "B1",
            "--cascade",
            "--output-dir", str(output_dir / "failure_impact")
        ],
        "Simulating failure of MainBroker (B1) with cascade analysis"
    )
    
    time.sleep(1)
    
    # Example 5: Failure scenarios
    print("\n" + "="*70)
    print("STEP 6: Failure Scenarios")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--failure-scenarios", "spof", "cascade", "recovery",
            "--output-dir", str(output_dir / "scenarios")
        ],
        "Generating comprehensive failure scenario analysis"
    )
    
    time.sleep(1)
    
    # Example 6: Dashboard
    print("\n" + "="*70)
    print("STEP 7: Metrics Dashboard")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--dashboard",
            "--output-dir", str(output_dir / "dashboard")
        ],
        "Creating comprehensive metrics dashboard"
    )
    
    time.sleep(1)
    
    # Example 7: Complete workflow
    print("\n" + "="*70)
    print("STEP 8: Complete Workflow (All Visualizations)")
    print("="*70)
    
    run_command(
        [
            "python", "visualize_graph.py",
            "--input", str(system_file),
            "--all",
            "--layout", "hierarchical",
            "--output-dir", str(output_dir / "complete_workflow")
        ],
        "Generating all visualization types with comprehensive report"
    )
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION WORKFLOW COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Directory: {output_dir}")
    print("\n📊 Generated Visualizations:")
    
    # List all HTML files
    html_files = list(output_dir.rglob("*.html"))
    
    for category in ["complete", "layers", "hidden_deps", "failure_impact", 
                     "scenarios", "dashboard", "complete_workflow"]:
        category_path = output_dir / category
        if category_path.exists():
            files = list(category_path.glob("*.html"))
            if files:
                print(f"\n  {category.upper()}:")
                for f in files:
                    print(f"    ✓ {f.name}")
    
    print("\n💡 Next Steps:")
    print("  1. Open HTML files in your web browser")
    print("  2. Explore interactive features (zoom, pan, filter)")
    print("  3. Review the comprehensive report")
    print("  4. Use visualizations for presentations or papers")
    print("  5. Customize layouts and color schemes as needed")
    
    print("\n📖 For more information:")
    print("  - See VISUALIZE_GRAPH_README.md for complete documentation")
    print("  - Run 'python visualize_graph.py --help' for all options")
    print("  - Check examples/ directory for more use cases")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
