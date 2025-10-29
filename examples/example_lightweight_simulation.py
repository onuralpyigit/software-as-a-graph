"""
Example: Lightweight Event-Driven DDS Simulation

Demonstrates:
1. High-performance in-memory simulation
2. Scalability to 1000+ components
3. Real-time metrics collection
4. Performance benchmarking
5. Comparison with different scales
"""

import sys
import asyncio
import json
import time
from pathlib import Path

# Add refactored directory to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.simulation.lightweight_dds_simulator import (
    LightweightDDSSimulator,
    run_lightweight_simulation
)


def create_small_system():
    """Create small test system (10 apps, 5 topics)"""
    
    return {
        "metadata": {"scenario": "Small System - 10 apps"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}", 
             "cpu_capacity": 8.0, "memory_gb": 16.0}
            for i in range(1, 4)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"App{i}", "type": "PROSUMER"}
            for i in range(1, 11)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {"history_depth": 10, "deadline_ms": 100}}
            for i in range(1, 6)
        ],
        "brokers": [
            {"id": "B1", "name": "MainBroker"}
        ],
        "relationships": {
            "publishes_to": [
                {"from": f"A{i}", "to": f"T{(i % 5) + 1}", 
                 "period_ms": 100, "msg_size": 1024}
                for i in range(1, 11)
            ],
            "subscribes_to": [
                {"from": f"A{i}", "to": f"T{j}"}
                for i in range(1, 11)
                for j in range(1, 6)
                if i != j
            ],
            "runs_on": [
                {"from": f"A{i}", "to": f"N{((i-1) % 3) + 1}"}
                for i in range(1, 11)
            ],
            "routes": [
                {"from": "B1", "to": f"T{i}"}
                for i in range(1, 6)
            ]
        }
    }


def create_medium_system():
    """Create medium system (100 apps, 50 topics)"""
    
    import random
    random.seed(42)
    
    return {
        "metadata": {"scenario": "Medium System - 100 apps"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}",
             "cpu_capacity": 16.0, "memory_gb": 32.0}
            for i in range(1, 21)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"App{i}", "type": "PROSUMER"}
            for i in range(1, 101)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {"history_depth": 10, "deadline_ms": 100}}
            for i in range(1, 51)
        ],
        "brokers": [
            {"id": f"B{i}", "name": f"Broker{i}"}
            for i in range(1, 4)
        ],
        "relationships": {
            "publishes_to": [
                {"from": f"A{i}", "to": f"T{random.randint(1, 50)}",
                 "period_ms": random.choice([50, 100, 200]),
                 "msg_size": random.choice([512, 1024, 2048])}
                for i in range(1, 101)
            ],
            "subscribes_to": [
                {"from": f"A{i}", "to": f"T{j}"}
                for i in range(1, 101)
                for j in random.sample(range(1, 51), 5)
            ],
            "runs_on": [
                {"from": f"A{i}", "to": f"N{((i-1) % 20) + 1}"}
                for i in range(1, 101)
            ],
            "routes": [
                {"from": f"B{((i-1) % 3) + 1}", "to": f"T{i}"}
                for i in range(1, 51)
            ]
        }
    }


def create_large_system():
    """Create large system (1000 apps, 500 topics)"""
    
    import random
    random.seed(42)
    
    return {
        "metadata": {"scenario": "Large System - 1000 apps"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}",
             "cpu_capacity": 32.0, "memory_gb": 64.0}
            for i in range(1, 101)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"App{i}", "type": "PROSUMER"}
            for i in range(1, 1001)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {"history_depth": 5, "deadline_ms": 100}}
            for i in range(1, 501)
        ],
        "brokers": [
            {"id": f"B{i}", "name": f"Broker{i}"}
            for i in range(1, 11)
        ],
        "relationships": {
            "publishes_to": [
                {"from": f"A{i}", "to": f"T{random.randint(1, 500)}",
                 "period_ms": random.choice([100, 200, 500]),
                 "msg_size": random.choice([256, 512, 1024])}
                for i in range(1, 1001)
            ],
            "subscribes_to": [
                {"from": f"A{i}", "to": f"T{j}"}
                for i in range(1, 1001)
                for j in random.sample(range(1, 501), 3)
            ],
            "runs_on": [
                {"from": f"A{i}", "to": f"N{((i-1) % 100) + 1}"}
                for i in range(1, 1001)
            ],
            "routes": [
                {"from": f"B{((i-1) % 10) + 1}", "to": f"T{i}"}
                for i in range(1, 501)
            ]
        }
    }


async def example_1_small_system():
    """Example 1: Small system simulation"""
    
    print("\n" + "=" * 70)
    print("Example 1: Small System (10 apps, 5 topics)")
    print("=" * 70)
    
    # Create system
    graph = create_small_system()
    graph_path = Path("output/small_system.json")
    graph_path.parent.mkdir(exist_ok=True)
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"\nSystem: {len(graph['applications'])} applications, "
          f"{len(graph['topics'])} topics")
    
    # Run simulation
    print("\nRunning 30-second simulation...")
    results = await run_lightweight_simulation(
        str(graph_path),
        duration_seconds=30,
        output_path="output/small_results.json"
    )
    
    print(f"\n✓ Completed! See output/small_results.json for details")


async def example_2_medium_system():
    """Example 2: Medium system simulation"""
    
    print("\n" + "=" * 70)
    print("Example 2: Medium System (100 apps, 50 topics)")
    print("=" * 70)
    
    # Create system
    graph = create_medium_system()
    graph_path = Path("output/medium_system.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"\nSystem: {len(graph['applications'])} applications, "
          f"{len(graph['topics'])} topics, "
          f"{len(graph['brokers'])} brokers")
    
    # Run simulation
    print("\nRunning 60-second simulation...")
    results = await run_lightweight_simulation(
        str(graph_path),
        duration_seconds=60,
        output_path="output/medium_results.json"
    )
    
    print(f"\n✓ Completed! See output/medium_results.json for details")


async def example_3_large_system():
    """Example 3: Large system simulation"""
    
    print("\n" + "=" * 70)
    print("Example 3: Large System (1000 apps, 500 topics)")
    print("=" * 70)
    
    # Create system
    print("\nGenerating large system topology...")
    start = time.time()
    graph = create_large_system()
    print(f"  Generated in {time.time() - start:.2f}s")
    
    graph_path = Path("output/large_system.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"\nSystem: {len(graph['applications'])} applications, "
          f"{len(graph['topics'])} topics, "
          f"{len(graph['brokers'])} brokers")
    print(f"  Relationships: {len(graph['relationships']['publishes_to'])} publishers, "
          f"{len(graph['relationships']['subscribes_to'])} subscribers")
    
    # Run simulation
    print("\nRunning 120-second simulation...")
    results = await run_lightweight_simulation(
        str(graph_path),
        duration_seconds=120,
        output_path="output/large_results.json"
    )
    
    print(f"\n✓ Completed! See output/large_results.json for details")


async def example_4_performance_comparison():
    """Example 4: Performance comparison across scales"""
    
    print("\n" + "=" * 70)
    print("Example 4: Performance Comparison")
    print("=" * 70)
    
    scales = [
        ("Small", create_small_system(), 10),
        ("Medium", create_medium_system(), 30),
        ("Large", create_large_system(), 60)
    ]
    
    results_comparison = []
    
    for name, graph, duration in scales:
        print(f"\n[{name} System]")
        print(f"  Apps: {len(graph['applications'])}, "
              f"Topics: {len(graph['topics'])}, "
              f"Brokers: {len(graph['brokers'])}")
        
        # Save graph
        graph_path = Path(f"output/{name.lower()}_perf.json")
        with open(graph_path, 'w') as f:
            json.dump(graph, f)
        
        # Run simulation
        print(f"  Running {duration}s simulation...")
        start_time = time.time()
        
        results = await run_lightweight_simulation(
            str(graph_path),
            duration_seconds=duration
        )
        
        elapsed = time.time() - start_time
        
        # Collect metrics
        stats = results['global_stats']
        results_comparison.append({
            'name': name,
            'apps': len(graph['applications']),
            'topics': len(graph['topics']),
            'duration': duration,
            'elapsed': elapsed,
            'messages_sent': stats['messages_sent'],
            'messages_delivered': stats['messages_delivered'],
            'delivery_rate': stats['delivery_rate'],
            'avg_latency_ms': stats['avg_latency_ms'],
            'throughput_msg_s': stats['messages_delivered'] / elapsed,
            'throughput_mb_s': stats['bytes_transferred'] / elapsed / 1024 / 1024
        })
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Scale':<10} {'Apps':<6} {'Topics':<7} {'Duration':<9} "
          f"{'Msg/s':<10} {'MB/s':<8} {'Latency':<10}")
    print("-" * 70)
    
    for r in results_comparison:
        print(f"{r['name']:<10} {r['apps']:<6} {r['topics']:<7} "
              f"{r['duration']:<9} {r['throughput_msg_s']:<10.1f} "
              f"{r['throughput_mb_s']:<8.2f} {r['avg_latency_ms']:<10.2f}ms")
    
    # Save comparison
    with open("output/performance_comparison.json", 'w') as f:
        json.dump(results_comparison, f, indent=2)
    
    print("\n✓ Comparison saved to output/performance_comparison.json")


async def example_5_custom_simulation():
    """Example 5: Custom simulation with specific topology"""
    
    print("\n" + "=" * 70)
    print("Example 5: Custom Simulation")
    print("=" * 70)
    
    # Create custom IoT scenario
    graph = {
        "metadata": {"scenario": "IoT Sensor Network"},
        "nodes": [
            {"id": f"N{i}", "name": f"Gateway{i}",
             "cpu_capacity": 4.0, "memory_gb": 8.0}
            for i in range(1, 11)
        ],
        "applications": [
            # Sensor publishers (high frequency, small messages)
            *[{"id": f"S{i}", "name": f"Sensor{i}", "type": "PRODUCER"}
              for i in range(1, 51)],
            # Aggregators (prosumers)
            *[{"id": f"AGG{i}", "name": f"Aggregator{i}", "type": "PROSUMER"}
              for i in range(1, 6)],
            # Analytics (consumers)
            {"id": "ANALYTICS", "name": "AnalyticsEngine", "type": "CONSUMER"}
        ],
        "topics": [
            # Raw sensor data
            *[{"id": f"RAW{i}", "name": f"sensor/raw/{i}",
               "qos": {"history_depth": 5, "deadline_ms": 50}}
              for i in range(1, 51)],
            # Aggregated data
            *[{"id": f"AGG{i}", "name": f"sensor/aggregated/{i}",
               "qos": {"history_depth": 20, "deadline_ms": 200}}
              for i in range(1, 6)]
        ],
        "brokers": [
            {"id": "B1", "name": "EdgeBroker"},
            {"id": "B2", "name": "CloudBroker"}
        ],
        "relationships": {
            "publishes_to": [
                # Sensors publish raw data (high frequency)
                *[{"from": f"S{i}", "to": f"RAW{i}",
                   "period_ms": 50, "msg_size": 256}
                  for i in range(1, 51)],
                # Aggregators publish aggregated data
                *[{"from": f"AGG{i}", "to": f"AGG{i}",
                   "period_ms": 500, "msg_size": 4096}
                  for i in range(1, 6)]
            ],
            "subscribes_to": [
                # Aggregators subscribe to sensors
                *[{"from": f"AGG{(i-1)//10 + 1}", "to": f"RAW{i}"}
                  for i in range(1, 51)],
                # Analytics subscribes to aggregated data
                *[{"from": "ANALYTICS", "to": f"AGG{i}"}
                  for i in range(1, 6)]
            ],
            "runs_on": [
                # Sensors on gateways
                *[{"from": f"S{i}", "to": f"N{((i-1) % 10) + 1}"}
                  for i in range(1, 51)],
                # Aggregators on specific gateways
                *[{"from": f"AGG{i}", "to": f"N{i}"}
                  for i in range(1, 6)],
                # Analytics on gateway 10
                {"from": "ANALYTICS", "to": "N10"}
            ],
            "routes": [
                # Raw data through edge broker
                *[{"from": "B1", "to": f"RAW{i}"}
                  for i in range(1, 51)],
                # Aggregated data through cloud broker
                *[{"from": "B2", "to": f"AGG{i}"}
                  for i in range(1, 6)]
            ]
        }
    }
    
    # Save graph
    graph_path = Path("output/iot_scenario.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print("\nIoT Sensor Network:")
    print(f"  Sensors: 50 (high-frequency publishers)")
    print(f"  Aggregators: 5 (prosumers)")
    print(f"  Analytics: 1 (consumer)")
    print(f"  Gateways: 10")
    print(f"  Brokers: 2 (Edge + Cloud)")
    
    # Run simulation
    print("\nRunning 60-second simulation...")
    results = await run_lightweight_simulation(
        str(graph_path),
        duration_seconds=60,
        output_path="output/iot_results.json"
    )
    
    print(f"\n✓ Completed! See output/iot_results.json for details")


async def example_6_stress_test():
    """Example 6: Stress test with extreme scale"""
    
    print("\n" + "=" * 70)
    print("Example 6: Stress Test (2000 apps, 1000 topics)")
    print("=" * 70)
    print("\n⚠️  This is a stress test with 2000 applications!")
    
    import random
    random.seed(42)
    
    print("\nGenerating extreme-scale topology...")
    start = time.time()
    
    graph = {
        "metadata": {"scenario": "Stress Test - 2000 apps"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}",
             "cpu_capacity": 32.0, "memory_gb": 64.0}
            for i in range(1, 201)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"App{i}", "type": "PROSUMER"}
            for i in range(1, 2001)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {"history_depth": 1, "deadline_ms": 1000}}
            for i in range(1, 1001)
        ],
        "brokers": [
            {"id": f"B{i}", "name": f"Broker{i}"}
            for i in range(1, 21)
        ],
        "relationships": {
            "publishes_to": [
                {"from": f"A{i}", "to": f"T{random.randint(1, 1000)}",
                 "period_ms": 500, "msg_size": 256}
                for i in range(1, 2001)
            ],
            "subscribes_to": [
                {"from": f"A{i}", "to": f"T{j}"}
                for i in range(1, 2001)
                for j in random.sample(range(1, 1001), 2)
            ],
            "runs_on": [
                {"from": f"A{i}", "to": f"N{((i-1) % 200) + 1}"}
                for i in range(1, 2001)
            ],
            "routes": [
                {"from": f"B{((i-1) % 20) + 1}", "to": f"T{i}"}
                for i in range(1, 1001)
            ]
        }
    }
    
    gen_time = time.time() - start
    print(f"  Generated in {gen_time:.2f}s")
    
    # Save graph
    graph_path = Path("output/stress_test.json")
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    print(f"\nTopology:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Applications: {len(graph['applications'])}")
    print(f"  Topics: {len(graph['topics'])}")
    print(f"  Brokers: {len(graph['brokers'])}")
    print(f"  Publishers: {len(graph['relationships']['publishes_to'])}")
    print(f"  Subscribers: {len(graph['relationships']['subscribes_to'])}")
    
    # Run simulation
    print("\nRunning 60-second stress test...")
    print("(This will test the limits of the lightweight simulator)")
    
    results = await run_lightweight_simulation(
        str(graph_path),
        duration_seconds=60,
        output_path="output/stress_results.json"
    )
    
    print(f"\n✓ Stress test completed!")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("LIGHTWEIGHT DDS SIMULATION - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    try:
        # Run examples
        asyncio.run(example_1_small_system())
        asyncio.run(example_2_medium_system())
        asyncio.run(example_3_large_system())
        asyncio.run(example_4_performance_comparison())
        asyncio.run(example_5_custom_simulation())
        asyncio.run(example_6_stress_test())
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Small scale: 10 apps, 5 topics")
        print("  ✓ Medium scale: 100 apps, 50 topics")
        print("  ✓ Large scale: 1000 apps, 500 topics")
        print("  ✓ Stress test: 2000 apps, 1000 topics")
        print("  ✓ Performance comparison")
        print("  ✓ Custom IoT scenarios")
        print("  ✓ Real-time metrics")
        print("  ✓ No container overhead")
        
        print("\n⚡ Performance Characteristics:")
        print("  - In-memory execution")
        print("  - Event-driven (asyncio)")
        print("  - Millisecond precision")
        print("  - Low memory footprint")
        print("  - Scales to 10,000+ components")
        
        print("\n📖 Usage:")
        print("""
import asyncio
from src.simulation.lightweight_dds_simulator import run_lightweight_simulation

# Run simulation
results = asyncio.run(
    run_lightweight_simulation("system.json", duration_seconds=60)
)
        """)
        
        print("\n📁 Generated Files:")
        print("  Check output/ directory for all results")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
