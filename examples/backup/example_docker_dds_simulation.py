"""
Example: Docker-based DDS Simulation

Demonstrates:
1. Building simulation from graph JSON
2. Generating FastDDS applications
3. Creating Docker infrastructure
4. Running timed simulations
5. Monitoring and metrics collection
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.simulation.docker_dds_simulator import (
    DockerDDSSimulator, SimulationConfig
)


def create_simple_graph():
    """Create a simple pub-sub graph for demonstration"""
    
    graph = {
        "metadata": {
            "scenario": "Simple Pub-Sub System",
            "generated_at": "2025-01-01T00:00:00"
        },
        "nodes": [
            {
                "id": "N1",
                "name": "Node1",
                "cpu_capacity": 8.0,
                "memory_gb": 16.0,
                "network_bandwidth_mbps": 10000.0
            },
            {
                "id": "N2",
                "name": "Node2",
                "cpu_capacity": 4.0,
                "memory_gb": 8.0,
                "network_bandwidth_mbps": 5000.0
            }
        ],
        "applications": [
            {
                "id": "A1",
                "name": "SensorPublisher",
                "type": "PRODUCER",
                "replicas": 1
            },
            {
                "id": "A2",
                "name": "DataProcessor",
                "type": "PROSUMER",
                "replicas": 1
            },
            {
                "id": "A3",
                "name": "Logger",
                "type": "CONSUMER",
                "replicas": 1
            }
        ],
        "topics": [
            {
                "id": "T1",
                "name": "sensor/temperature",
                "qos": {
                    "durability": "VOLATILE",
                    "reliability": "RELIABLE",
                    "transport_priority": "HIGH",
                    "deadline_ms": 100,
                    "history_depth": 10
                }
            },
            {
                "id": "T2",
                "name": "processed/data",
                "qos": {
                    "durability": "TRANSIENT_LOCAL",
                    "reliability": "RELIABLE",
                    "transport_priority": "MEDIUM",
                    "deadline_ms": 500,
                    "history_depth": 20
                }
            }
        ],
        "brokers": [
            {
                "id": "B1",
                "name": "MainBroker",
                "discovery_port": 11811
            }
        ],
        "relationships": {
            "publishes_to": [
                {
                    "from": "A1",
                    "to": "T1",
                    "period_ms": 100,
                    "msg_size": 1024
                },
                {
                    "from": "A2",
                    "to": "T2",
                    "period_ms": 500,
                    "msg_size": 2048
                }
            ],
            "subscribes_to": [
                {
                    "from": "A2",
                    "to": "T1"
                },
                {
                    "from": "A3",
                    "to": "T1"
                },
                {
                    "from": "A3",
                    "to": "T2"
                }
            ],
            "runs_on": [
                {
                    "from": "A1",
                    "to": "N1"
                },
                {
                    "from": "A2",
                    "to": "N2"
                },
                {
                    "from": "A3",
                    "to": "N2"
                }
            ],
            "routes": [
                {
                    "from": "B1",
                    "to": "T1"
                },
                {
                    "from": "B1",
                    "to": "T2"
                }
            ]
        }
    }
    
    return graph


def example_1_build_simulation():
    """Example 1: Build simulation from graph JSON"""
    
    print("\n" + "=" * 70)
    print("Example 1: Build Docker DDS Simulation")
    print("=" * 70)
    
    # Create simple graph
    print("\n[Step 1] Creating example graph...")
    graph = create_simple_graph()
    
    # Save to JSON
    graph_path = Path("output/simple_pubsub.json")
    graph_path.parent.mkdir(exist_ok=True)
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"  ✓ Saved graph to {graph_path}")
    
    # Configure simulation
    print("\n[Step 2] Configuring simulation...")
    config = SimulationConfig(
        duration_seconds=60,
        enable_discovery=True,
        enable_monitoring=True,
        output_dir="simulation_output"
    )
    print(f"  ✓ Duration: {config.duration_seconds}s")
    print(f"  ✓ Discovery: {config.enable_discovery}")
    print(f"  ✓ Monitoring: {config.enable_monitoring}")
    
    # Build simulation
    print("\n[Step 3] Building simulation...")
    simulator = DockerDDSSimulator(config)
    sim_dir = simulator.build_from_json(str(graph_path), "simple_pubsub")
    
    print(f"\n✓ Simulation built successfully!")
    print(f"  Directory: {sim_dir}")
    print(f"  Applications: {len(graph['applications'])}")
    print(f"  Topics: {len(graph['topics'])}")
    
    print("\nGenerated files:")
    sim_path = Path(sim_dir)
    for item in sorted(sim_path.rglob("*")):
        if item.is_file():
            print(f"  - {item.relative_to(sim_path)}")


def example_2_inspect_generated_code():
    """Example 2: Inspect generated FastDDS code"""
    
    print("\n" + "=" * 70)
    print("Example 2: Inspect Generated FastDDS Code")
    print("=" * 70)
    
    # Build simulation
    graph = create_simple_graph()
    graph_path = Path("output/inspect_pubsub.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    config = SimulationConfig(output_dir="simulation_output")
    simulator = DockerDDSSimulator(config)
    sim_dir = simulator.build_from_json(str(graph_path), "inspect_pubsub")
    
    print("\n[Inspecting generated C++ code...]")
    
    # Show generated code for first application
    sim_path = Path(sim_dir)
    app_dirs = list((sim_path / "apps").iterdir())
    
    if app_dirs:
        first_app = app_dirs[0]
        cpp_file = first_app / f"{first_app.name}.cpp"
        
        if cpp_file.exists():
            print(f"\nGenerated code for {first_app.name}:")
            print("=" * 70)
            
            content = cpp_file.read_text()
            lines = content.split('\n')
            
            # Show first 50 lines
            for i, line in enumerate(lines[:50], 1):
                print(f"{i:3d}: {line}")
            
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")


def example_3_docker_compose_config():
    """Example 3: Examine Docker Compose configuration"""
    
    print("\n" + "=" * 70)
    print("Example 3: Docker Compose Configuration")
    print("=" * 70)
    
    # Build simulation
    graph = create_simple_graph()
    graph_path = Path("output/compose_pubsub.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    config = SimulationConfig(
        enable_discovery=True,
        output_dir="simulation_output"
    )
    simulator = DockerDDSSimulator(config)
    sim_dir = simulator.build_from_json(str(graph_path), "compose_pubsub")
    
    # Read and display compose file
    compose_file = Path(sim_dir) / "docker-compose.yml"
    
    if compose_file.exists():
        print("\nDocker Compose Configuration:")
        print("=" * 70)
        print(compose_file.read_text())


def example_4_run_simulation():
    """Example 4: Run complete simulation (requires Docker)"""
    
    print("\n" + "=" * 70)
    print("Example 4: Run Complete Simulation")
    print("=" * 70)
    print("\n⚠️  This example requires Docker to be installed and running!")
    print("    If Docker is not available, this will show the commands needed.\n")
    
    # Build simulation
    print("[Step 1] Building simulation...")
    graph = create_simple_graph()
    graph_path = Path("output/run_pubsub.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    config = SimulationConfig(
        duration_seconds=30,  # 30 seconds for demo
        enable_discovery=True,
        enable_monitoring=True,
        output_dir="simulation_output"
    )
    simulator = DockerDDSSimulator(config)
    sim_dir = simulator.build_from_json(str(graph_path), "run_pubsub")
    
    print(f"  ✓ Simulation ready at {sim_dir}")
    
    # Show commands
    print("\n[Step 2] To run the simulation manually:")
    print(f"\n  cd {sim_dir}")
    print("  docker-compose build")
    print("  docker-compose up -d")
    print(f"  sleep {config.duration_seconds}")
    print("  docker-compose down")
    print("  docker-compose logs > logs/simulation.log")
    
    print("\nOr use the generated script:")
    print(f"  cd {sim_dir}")
    print("  ./run_simulation.sh")
    
    # Optionally try to run (commented out by default)
    """
    print("\n[Step 3] Running simulation...")
    try:
        results = simulator.run_simulation(sim_dir, duration=30)
        print(f"\n✓ Simulation completed!")
        print(f"  Duration: {results['duration_seconds']:.1f}s")
        print(f"  Containers: {results['containers'].get('container_count', 0)}")
        print(f"  Logs collected: {results['logs_collected']}")
    except Exception as e:
        print(f"\n⚠️  Could not run simulation: {e}")
        print("   Make sure Docker is installed and running")
    """


def example_5_complex_system():
    """Example 5: Build complex multi-node system"""
    
    print("\n" + "=" * 70)
    print("Example 5: Complex Multi-Node System")
    print("=" * 70)
    
    # Create complex graph
    print("\n[Creating complex pub-sub system...]")
    
    graph = {
        "metadata": {"scenario": "Complex IoT System"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}", 
             "cpu_capacity": 8.0, "memory_gb": 16.0, 
             "network_bandwidth_mbps": 10000.0}
            for i in range(1, 6)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"Service{i}", "type": "PROSUMER"}
            for i in range(1, 11)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {
                 "durability": "TRANSIENT_LOCAL",
                 "reliability": "RELIABLE",
                 "transport_priority": "HIGH",
                 "deadline_ms": 100,
                 "history_depth": 10
             }}
            for i in range(1, 16)
        ],
        "brokers": [
            {"id": "B1", "name": "DiscoveryServer", "discovery_port": 11811}
        ],
        "relationships": {
            "publishes_to": [],
            "subscribes_to": [],
            "runs_on": [],
            "routes": []
        }
    }
    
    # Create publish/subscribe relationships
    import random
    random.seed(42)
    
    for app_id in range(1, 11):
        # Each app publishes to 2 topics
        pub_topics = random.sample(range(1, 16), 2)
        for topic_id in pub_topics:
            graph["relationships"]["publishes_to"].append({
                "from": f"A{app_id}",
                "to": f"T{topic_id}",
                "period_ms": random.choice([100, 500, 1000]),
                "msg_size": random.choice([512, 1024, 2048])
            })
        
        # Each app subscribes to 3 topics
        sub_topics = random.sample(range(1, 16), 3)
        for topic_id in sub_topics:
            graph["relationships"]["subscribes_to"].append({
                "from": f"A{app_id}",
                "to": f"T{topic_id}"
            })
        
        # Assign to nodes
        node_id = ((app_id - 1) % 5) + 1
        graph["relationships"]["runs_on"].append({
            "from": f"A{app_id}",
            "to": f"N{node_id}"
        })
    
    # All topics through broker
    for topic_id in range(1, 16):
        graph["relationships"]["routes"].append({
            "from": "B1",
            "to": f"T{topic_id}"
        })
    
    # Save graph
    graph_path = Path("output/complex_pubsub.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"  ✓ Created system with:")
    print(f"    - {len(graph['nodes'])} nodes")
    print(f"    - {len(graph['applications'])} applications")
    print(f"    - {len(graph['topics'])} topics")
    print(f"    - {len(graph['relationships']['publishes_to'])} publish relationships")
    print(f"    - {len(graph['relationships']['subscribes_to'])} subscribe relationships")
    
    # Build simulation
    print("\n[Building simulation...]")
    config = SimulationConfig(
        duration_seconds=120,
        enable_discovery=True,
        output_dir="simulation_output"
    )
    simulator = DockerDDSSimulator(config)
    sim_dir = simulator.build_from_json(str(graph_path), "complex_system")
    
    print(f"\n✓ Complex simulation ready!")
    print(f"  Directory: {sim_dir}")
    print(f"  Estimated containers: {len(graph['applications']) + 1}")  # +1 for discovery


def example_6_qos_policies():
    """Example 6: Demonstrate QoS policy configuration"""
    
    print("\n" + "=" * 70)
    print("Example 6: QoS Policy Configuration")
    print("=" * 70)
    
    print("\n[Creating topics with different QoS policies...]")
    
    graph = {
        "metadata": {"scenario": "QoS Policy Demo"},
        "nodes": [{"id": "N1", "name": "Node1", 
                   "cpu_capacity": 8.0, "memory_gb": 16.0}],
        "applications": [
            {"id": "A1", "name": "HighPriorityPublisher", "type": "PRODUCER"},
            {"id": "A2", "name": "BestEffortPublisher", "type": "PRODUCER"},
            {"id": "A3", "name": "Subscriber", "type": "CONSUMER"}
        ],
        "topics": [
            {
                "id": "T1",
                "name": "critical/commands",
                "qos": {
                    "durability": "TRANSIENT_LOCAL",
                    "reliability": "RELIABLE",
                    "transport_priority": "URGENT",
                    "deadline_ms": 10,
                    "lifespan_ms": 50,
                    "history_depth": 1
                }
            },
            {
                "id": "T2",
                "name": "telemetry/stream",
                "qos": {
                    "durability": "VOLATILE",
                    "reliability": "BEST_EFFORT",
                    "transport_priority": "LOW",
                    "deadline_ms": 1000,
                    "lifespan_ms": 5000,
                    "history_depth": 10
                }
            }
        ],
        "brokers": [{"id": "B1", "name": "Broker"}],
        "relationships": {
            "publishes_to": [
                {"from": "A1", "to": "T1", "period_ms": 10, "msg_size": 256},
                {"from": "A2", "to": "T2", "period_ms": 100, "msg_size": 4096}
            ],
            "subscribes_to": [
                {"from": "A3", "to": "T1"},
                {"from": "A3", "to": "T2"}
            ],
            "runs_on": [
                {"from": "A1", "to": "N1"},
                {"from": "A2", "to": "N1"},
                {"from": "A3", "to": "N1"}
            ],
            "routes": []
        }
    }
    
    # Display QoS comparison
    print("\nTopic QoS Policies:")
    print("=" * 70)
    
    for topic in graph["topics"]:
        print(f"\n{topic['name']}:")
        qos = topic["qos"]
        print(f"  Durability: {qos['durability']}")
        print(f"  Reliability: {qos['reliability']}")
        print(f"  Priority: {qos['transport_priority']}")
        print(f"  Deadline: {qos['deadline_ms']}ms")
        print(f"  Lifespan: {qos['lifespan_ms']}ms")
        print(f"  History: {qos['history_depth']} samples")
    
    # Build simulation
    graph_path = Path("output/qos_demo.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    config = SimulationConfig(duration_seconds=60)
    simulator = DockerDDSSimulator(config)
    sim_dir = simulator.build_from_json(str(graph_path), "qos_demo")
    
    print(f"\n✓ QoS demo simulation ready at {sim_dir}")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("DOCKER DDS SIMULATION - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    try:
        example_1_build_simulation()
        example_2_inspect_generated_code()
        example_3_docker_compose_config()
        example_4_run_simulation()
        example_5_complex_system()
        example_6_qos_policies()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Build simulation from graph JSON")
        print("  ✓ Generate FastDDS C++ applications")
        print("  ✓ Create Docker Compose configurations")
        print("  ✓ Configure QoS policies")
        print("  ✓ Set resource limits per node")
        print("  ✓ Run timed simulations")
        print("  ✓ Enable discovery servers")
        print("  ✓ Support complex topologies")
        
        print("\n📖 How to Run Simulations:")
        print("""
# 1. Navigate to simulation directory
cd simulation_output/simple_pubsub

# 2. Build containers (first time only)
docker-compose build

# 3. Run simulation
./run_simulation.sh

# 4. Or manually
docker-compose up -d    # Start
sleep 60                # Wait
docker-compose down     # Stop

# 5. View logs
docker-compose logs

# 6. Monitor live
./monitor.py
        """)
        
        print("\n⚠️  Requirements:")
        print("  - Docker Engine (20.10+)")
        print("  - Docker Compose (2.0+)")
        print("  - Python 3.8+")
        print("  - FastDDS (auto-installed in containers)")
        
        print("\n📁 Generated Files:")
        print("  Check simulation_output/ for all generated simulations")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
