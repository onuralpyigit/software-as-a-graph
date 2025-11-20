#!/usr/bin/env python3
"""
Quick Examples for Simulation System

Simple, runnable examples demonstrating key features:
1. Basic traffic simulation
2. Single component failure
3. Cascading failures
4. Performance comparison
5. Custom scenarios

Run with: python example_graph_simulator.py [example_number]
"""

import sys
import json
import asyncio
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

try:
    from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
    from src.simulation.enhanced_failure_simulator import (
        FailureSimulator, FailureType, ComponentType
    )
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please ensure simulation modules are in src/simulation/")
    sys.exit(1)


def create_example_graph():
    """Generate a simple connected graph"""
    return {
        "nodes": [
            {"id": "node1", "name": "node1"},
            {"id": "node2", "name": "node2"}
        ],
        "applications": [
            {"id": "app1", "name": "app1", "type": "PRODUCER"},
            {"id": "app2", "name": "app2", "type": "PROSUMER"},
            {"id": "app3", "name": "app3", "type": "CONSUMER"}
        ],
        "topics": [
            {"id": "topic1", "name": "topic1", "message_size_bytes": 512, "message_rate_hz": 10},
            {"id": "topic2", "name": "topic2", "message_size_bytes": 1024, "message_rate_hz": 1}
        ],
        "brokers": [
            {"id": "broker1", "name": "broker1"}
        ],
        "relationships": {
            "runs_on": [
                {"from": "app1", "to": "node1"},
                {"from": "app2", "to": "node1"},
                {"from": "app3", "to": "node2"},
                {"from": "broker1", "to": "node1"}
            ],
            "publishes_to": [
                {"from": "app1", "to": "topic1", "period_ms": 100, "msg_size": 512},
                {"from": "app2", "to": "topic2", "period_ms": 1000, "msg_size": 1024}
            ],
            "subscribes_to": [
                {"from": "app2", "to": "topic1"},
                {"from": "app3", "to": "topic2"}
            ],
            "routes": [
                {"from": "broker1", "to": "topic1"},
                {"from": "broker1", "to": "topic2"}
            ]
        }
    }


async def example_1_basic_simulation():
    """Example 1: Basic traffic simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Traffic Simulation")
    print("="*70)
    print("\nSimulating normal message flow for 15 seconds...")
    
    graph_data = create_example_graph()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(graph_data, f)
        temp_path = f.name
    
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(temp_path)
    
    results = await simulator.run_simulation(duration_seconds=15)
    
    print("\n✅ Simulation Complete!")
    print(f"\nResults:")
    stats = results['global_stats']
    print(f"  📤 Messages Sent: {stats['messages_sent']:,}")
    print(f"  📥 Messages Delivered: {stats['messages_delivered']:,}")
    print(f"  📊 Delivery Rate: {stats['delivery_rate']:.2%}")
    print(f"  ⏱️  Average Latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  📦 Data Transferred: {stats['bytes_transferred'] / 1024:.1f} KB")
    
    Path(temp_path).unlink()
    
    print("\n💡 Key Observation:")
    print("   Normal operation shows high delivery rates and low latency.")


async def example_2_single_failure():
    """Example 2: Single application failure"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Single Application Failure")
    print("="*70)
    print("\nSimulating failure of SensorApp at t=5s...")
    
    graph_data = create_example_graph()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(graph_data, f)
        temp_path = f.name
    
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(temp_path)
    
    failure_sim = FailureSimulator()
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(duration_seconds=15))
    
    # Inject failure at 5 seconds
    await asyncio.sleep(5)
    print("💥 Injecting failure: SensorApp")
    failure_sim.inject_failure(
        simulator,
        'SensorApp',
        ComponentType.APPLICATION,
        FailureType.COMPLETE,
        severity=1.0,
        enable_cascade=False
    )
    
    # Wait for completion
    results = await sim_task
    
    # Analyze impact
    impact = failure_sim.analyze_impact(simulator)
    
    print("\n✅ Simulation Complete!")
    print(f"\nImpact Analysis:")
    print(f"  ❌ Failed Components: {len(impact.failed_components)}")
    print(f"  ⚠️  Affected Components: {len(impact.affected_components)}")
    print(f"  🚫 Isolated Applications: {len(impact.isolated_applications)}")
    print(f"  📉 Messages Lost: {impact.messages_lost}")
    
    stats = results['global_stats']
    print(f"\nPerformance:")
    print(f"  📊 Delivery Rate: {stats['delivery_rate']:.2%}")
    print(f"  📤 Messages Sent: {stats['messages_sent']:,}")
    print(f"  📥 Messages Delivered: {stats['messages_delivered']:,}")
    
    Path(temp_path).unlink()
    
    print("\n💡 Key Observation:")
    print("   Sensor failure stops temperature/pressure data,")
    print("   affecting downstream ProcessorApp and AnalyticsApp.")


async def example_3_cascading_failure():
    """Example 3: Broker failure with cascading effects"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Cascading Broker Failure")
    print("="*70)
    print("\nSimulating MainBroker failure with cascade at t=5s...")
    
    graph_data = create_example_graph()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(graph_data, f)
        temp_path = f.name
    
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(temp_path)
    
    failure_sim = FailureSimulator(
        cascade_threshold=0.6,
        cascade_probability=0.7
    )
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(duration_seconds=20))
    
    # Inject cascading failure
    await asyncio.sleep(5)
    print("💥 Injecting cascading failure: MainBroker")
    failure_sim.inject_failure(
        simulator,
        'MainBroker',
        ComponentType.BROKER,
        FailureType.COMPLETE,
        severity=1.0,
        enable_cascade=True
    )
    
    results = await sim_task
    impact = failure_sim.analyze_impact(simulator)
    
    print("\n✅ Simulation Complete!")
    print(f"\nCascade Analysis:")
    print(f"  💥 Initial Failure: MainBroker")
    print(f"  📊 Total Failure Events: {len(failure_sim.failure_events)}")
    print(f"  🔄 Cascade Depth: {impact.cascade_depth}")
    print(f"  📏 Cascade Width: {impact.cascade_width}")
    
    print(f"\nFailed Components:")
    for comp in impact.failed_components:
        print(f"  • {comp}")
    
    print(f"\nAffected Components ({len(impact.affected_components)}):")
    for comp in list(impact.affected_components)[:5]:
        print(f"  • {comp}")
    if len(impact.affected_components) > 5:
        print(f"  ... and {len(impact.affected_components) - 5} more")
    
    Path(temp_path).unlink()
    
    print("\n💡 Key Observation:")
    print("   Broker failure can cascade to dependent applications,")
    print("   creating a wider system impact than initially expected.")


async def example_4_performance_comparison():
    """Example 4: Baseline vs Failure comparison"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Performance Comparison")
    print("="*70)
    print("\nComparing baseline vs failure scenarios...")
    
    graph_data = create_example_graph()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(graph_data, f)
        temp_path = f.name
    
    # Run baseline
    print("\n🔵 Running baseline simulation...")
    baseline_sim = LightweightDDSSimulator()
    baseline_sim.load_from_json(temp_path)
    baseline_results = await baseline_sim.run_simulation(duration_seconds=15)
    
    # Run with failure
    print("🔴 Running failure simulation...")
    failure_simulator = LightweightDDSSimulator()
    failure_simulator.load_from_json(temp_path)
    
    failure_mgr = FailureSimulator()
    sim_task = asyncio.create_task(failure_simulator.run_simulation(duration_seconds=15))
    
    await asyncio.sleep(5)
    failure_mgr.inject_failure(
        failure_simulator,
        'SensorApp',
        ComponentType.APPLICATION,
        FailureType.COMPLETE
    )
    
    failure_results = await sim_task
    
    # Compare
    baseline_stats = baseline_results['global_stats']
    failure_stats = failure_results['global_stats']
    
    print("\n✅ Comparison Complete!")
    print(f"\n{'Metric':<25} {'Baseline':<15} {'With Failure':<15} {'Change':<15}")
    print("-" * 70)
    
    # Delivery rate
    baseline_rate = baseline_stats['delivery_rate']
    failure_rate = failure_stats['delivery_rate']
    rate_change = (failure_rate - baseline_rate) / baseline_rate * 100 if baseline_rate > 0 else 0
    print(f"{'Delivery Rate':<25} {baseline_rate:<14.2%} {failure_rate:<14.2%} {rate_change:>+13.1f}%")
    
    # Latency
    baseline_lat = baseline_stats['avg_latency_ms']
    failure_lat = failure_stats['avg_latency_ms']
    lat_change = (failure_lat - baseline_lat) / baseline_lat * 100 if baseline_lat > 0 else 0
    print(f"{'Avg Latency (ms)':<25} {baseline_lat:<14.2f} {failure_lat:<14.2f} {lat_change:>+13.1f}%")
    
    # Messages
    baseline_msg = baseline_stats['messages_delivered']
    failure_msg = failure_stats['messages_delivered']
    msg_change = (failure_msg - baseline_msg) / baseline_msg * 100 if baseline_msg > 0 else 0
    print(f"{'Messages Delivered':<25} {baseline_msg:<14,} {failure_msg:<14,} {msg_change:>+13.1f}%")
    
    # Dropped
    baseline_drop = baseline_stats['messages_dropped']
    failure_drop = failure_stats['messages_dropped']
    print(f"{'Messages Dropped':<25} {baseline_drop:<14,} {failure_drop:<14,} {failure_drop - baseline_drop:>+14,}")
    
    Path(temp_path).unlink()
    
    print("\n💡 Key Observation:")
    print("   Quantitative comparison reveals performance degradation")
    print("   and helps identify system resilience characteristics.")


async def example_5_custom_scenario():
    """Example 5: Custom multi-stage scenario"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Multi-Stage Scenario")
    print("="*70)
    print("\nSimulating progressive system degradation...")
    
    graph_data = create_example_graph()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(graph_data, f)
        temp_path = f.name
    
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(temp_path)
    
    failure_sim = FailureSimulator()
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(duration_seconds=30))
    
    # Stage 1: Partial failure at t=5s
    await asyncio.sleep(5)
    print("⚠️  Stage 1 (t=5s): Partial degradation of ProcessorApp")
    failure_sim.inject_failure(
        simulator,
        'ProcessorApp',
        ComponentType.APPLICATION,
        FailureType.PARTIAL,
        severity=0.5
    )
    
    # Stage 2: Complete failure at t=15s
    await asyncio.sleep(10)
    print("❌ Stage 2 (t=15s): Complete failure of MainBroker")
    failure_sim.inject_failure(
        simulator,
        'MainBroker',
        ComponentType.BROKER,
        FailureType.COMPLETE,
        severity=1.0
    )
    
    # Stage 3: Additional failure at t=20s
    await asyncio.sleep(5)
    print("💥 Stage 3 (t=20s): Failure of BackupBroker")
    failure_sim.inject_failure(
        simulator,
        'BackupBroker',
        ComponentType.BROKER,
        FailureType.COMPLETE,
        severity=1.0
    )
    
    results = await sim_task
    impact = failure_sim.analyze_impact(simulator)
    
    print("\n✅ Scenario Complete!")
    print(f"\nProgressive Impact:")
    print(f"  📊 Total Failure Events: {len(failure_sim.failure_events)}")
    print(f"  ❌ Failed Components: {len(impact.failed_components)}")
    print(f"  ⚠️  Affected Components: {len(impact.affected_components)}")
    print(f"  🚫 Isolated Applications: {len(impact.isolated_applications)}")
    
    print(f"\nTimeline:")
    for i, event in enumerate(failure_sim.failure_events, 1):
        print(f"  {i}. t={event.timestamp - failure_sim.failure_events[0].timestamp:.1f}s: "
              f"{event.component_id} ({event.failure_type.value})")
    
    stats = results['global_stats']
    print(f"\nFinal State:")
    print(f"  📊 Delivery Rate: {stats['delivery_rate']:.2%}")
    print(f"  📉 Messages Lost: {stats['messages_dropped']:,}")
    
    Path(temp_path).unlink()
    
    print("\n💡 Key Observation:")
    print("   Progressive failures reveal system resilience limits")
    print("   and help identify critical failure combinations.")


async def run_example(example_num: int):
    """Run specific example"""
    examples = {
        1: example_1_basic_simulation,
        2: example_2_single_failure,
        3: example_3_cascading_failure,
        4: example_4_performance_comparison,
        5: example_5_custom_scenario
    }
    
    if example_num not in examples:
        print(f"❌ Invalid example number: {example_num}")
        print(f"Available examples: 1-{len(examples)}")
        return 1
    
    await examples[example_num]()
    return 0


def print_menu():
    """Print example menu"""
    print("\n" + "="*70)
    print("SIMULATION SYSTEM - QUICK EXAMPLES")
    print("="*70)
    print("\nAvailable Examples:")
    print("  1. Basic Traffic Simulation")
    print("  2. Single Application Failure")
    print("  3. Cascading Broker Failure")
    print("  4. Baseline vs Failure Comparison")
    print("  5. Custom Multi-Stage Scenario")
    print("  0. Run All Examples")
    print("\nUsage:")
    print("  python example_graph_simulator.py [example_number]")
    print("  python example_graph_simulator.py    # Interactive mode")
    print()


async def run_all_examples():
    """Run all examples sequentially"""
    for i in range(1, 6):
        await run_example(i)
        print("\n" + "="*70)
        input("Press Enter to continue to next example...")


async def interactive_mode():
    """Interactive example selection"""
    while True:
        print_menu()
        choice = input("Select example (0-5, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("Goodbye!")
            return 0
        
        try:
            num = int(choice)
            if num == 0:
                await run_all_examples()
            elif 1 <= num <= 5:
                await run_example(num)
                input("\nPress Enter to return to menu...")
            else:
                print("❌ Invalid choice. Please select 0-5.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


async def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            if example_num == 0:
                return await run_all_examples()
            else:
                return await run_example(example_num)
        except ValueError:
            print(f"❌ Invalid example number: {sys.argv[1]}")
            print_menu()
            return 1
    else:
        return await interactive_mode()


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
