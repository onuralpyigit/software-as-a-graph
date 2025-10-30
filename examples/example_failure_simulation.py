"""
Example: Failure Simulation and Cascading Failure Analysis

Demonstrates:
1. Single component failures
2. Multiple simultaneous failures
3. Cascading failure propagation
4. Network partition scenarios
5. Failure impact assessment
6. Resilience recommendations
"""

import sys
import asyncio
import json
import time
from pathlib import Path

# Add refactored directory to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
from src.simulation.enhanced_failure_simulator import (
    FailureSimulator,
    FailureType,
    ComponentType,
    run_failure_simulation
)


def create_test_system():
    """Create test system for failure simulation"""
    
    import random
    random.seed(42)
    
    return {
        "metadata": {"scenario": "Failure Test System"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}",
             "cpu_capacity": 16.0, "memory_gb": 32.0}
            for i in range(1, 11)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"App{i}", "type": "PROSUMER"}
            for i in range(1, 51)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {"history_depth": 10, "deadline_ms": 100}}
            for i in range(1, 26)
        ],
        "brokers": [
            {"id": f"B{i}", "name": f"Broker{i}"}
            for i in range(1, 4)
        ],
        "relationships": {
            "publishes_to": [
                {"from": f"A{i}", "to": f"T{random.randint(1, 25)}",
                 "period_ms": random.choice([100, 200]),
                 "msg_size": 1024}
                for i in range(1, 51)
            ],
            "subscribes_to": [
                {"from": f"A{i}", "to": f"T{j}"}
                for i in range(1, 51)
                for j in random.sample(range(1, 26), 3)
            ],
            "runs_on": [
                {"from": f"A{i}", "to": f"N{((i-1) % 10) + 1}"}
                for i in range(1, 51)
            ],
            "routes": [
                {"from": f"B{((i-1) % 3) + 1}", "to": f"T{i}"}
                for i in range(1, 26)
            ]
        }
    }


async def example_1_single_application_failure():
    """Example 1: Single application failure"""
    
    print("\n" + "=" * 70)
    print("Example 1: Single Application Failure")
    print("=" * 70)
    
    # Create system
    graph = create_test_system()
    graph_path = Path("output/failure_test.json")
    graph_path.parent.mkdir(exist_ok=True)
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    # Create simulator
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(str(graph_path))
    
    # Create failure simulator
    failure_sim = FailureSimulator(cascade_threshold=0.7, cascade_probability=0.6)
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(30))
    
    # Wait 10 seconds, then inject failure
    await asyncio.sleep(10)
    
    print("\n[10s] Injecting application failure: A1")
    failure_event = failure_sim.inject_failure(
        simulator,
        "A1",
        ComponentType.APPLICATION,
        FailureType.COMPLETE,
        severity=1.0,
        enable_cascade=True
    )
    
    # Wait for simulation to complete
    await sim_task
    
    # Analyze impact
    impact = failure_sim.analyze_impact(simulator)
    
    print("\n" + "-" * 70)
    print("IMPACT ANALYSIS")
    print("-" * 70)
    
    print(f"\nFailed Components: {len(impact.failed_components)}")
    print(f"  {', '.join(impact.failed_components)}")
    
    print(f"\nAffected Components: {len(impact.affected_components)}")
    if impact.affected_components:
        print(f"  {', '.join(list(impact.affected_components)[:5])}...")
    
    print(f"\nCascade Analysis:")
    print(f"  Depth: {impact.cascade_depth}")
    print(f"  Width: {impact.cascade_width}")
    
    # Generate recommendations
    recommendations = failure_sim.generate_recommendations(simulator, impact)
    print("\nRecommendations:")
    for category, recs in recommendations.items():
        if recs:
            print(f"  {category.title()}: {recs[0]}")


async def example_2_broker_failure_cascade():
    """Example 2: Broker failure with cascade"""
    
    print("\n" + "=" * 70)
    print("Example 2: Broker Failure with Cascading")
    print("=" * 70)
    
    # Create system
    graph = create_test_system()
    graph_path = Path("output/broker_failure.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    # Create simulator
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(str(graph_path))
    
    # Create failure simulator
    failure_sim = FailureSimulator(
        cascade_threshold=0.6,  # Lower threshold - more sensitive
        cascade_probability=0.8  # Higher probability
    )
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(40))
    
    # Wait 15 seconds, then inject broker failure
    await asyncio.sleep(15)
    
    print("\n[15s] Injecting broker failure: B1")
    print("  This broker routes multiple topics...")
    
    failure_event = failure_sim.inject_failure(
        simulator,
        "B1",
        ComponentType.BROKER,
        FailureType.COMPLETE,
        severity=1.0,
        enable_cascade=True
    )
    
    # Wait a bit to see cascade
    await asyncio.sleep(5)
    
    print(f"\n[20s] Active failures: {len(failure_sim.active_failures)}")
    for comp_id, event in failure_sim.active_failures.items():
        if event.failure_type == FailureType.CASCADE:
            print(f"  Cascaded to {comp_id} from {event.propagated_from}")
    
    # Wait for simulation to complete
    await sim_task
    
    # Analyze impact
    impact = failure_sim.analyze_impact(simulator)
    
    print("\n" + "-" * 70)
    print("CASCADE ANALYSIS")
    print("-" * 70)
    
    print(f"\nCascade Metrics:")
    print(f"  Initial Failure: B1 (Broker)")
    print(f"  Cascade Depth: {impact.cascade_depth}")
    print(f"  Cascade Width: {impact.cascade_width} components")
    print(f"  Total Failed: {len(impact.failed_components)}")
    
    print(f"\nUnavailable Topics: {len(impact.unavailable_topics)}")
    print(f"  {', '.join(list(impact.unavailable_topics)[:10])}...")
    
    print(f"\nIsolated Applications: {len(impact.isolated_applications)}")


async def example_3_scheduled_failures():
    """Example 3: Multiple scheduled failures"""
    
    print("\n" + "=" * 70)
    print("Example 3: Scheduled Failure Injection")
    print("=" * 70)
    
    # Create system
    graph = create_test_system()
    graph_path = Path("output/scheduled_failures.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    # Define failure schedule
    failure_schedule = [
        {
            'time': 10,
            'component': 'A5',
            'component_type': 'application',
            'failure_type': 'complete',
            'severity': 1.0,
            'enable_cascade': True
        },
        {
            'time': 20,
            'component': 'B2',
            'component_type': 'broker',
            'failure_type': 'complete',
            'severity': 1.0,
            'enable_cascade': True
        },
        {
            'time': 30,
            'component': 'A10',
            'component_type': 'application',
            'failure_type': 'partial',
            'severity': 0.5,
            'enable_cascade': False
        }
    ]
    
    print("\nFailure Schedule:")
    for i, failure in enumerate(failure_schedule, 1):
        print(f"  {i}. T+{failure['time']}s: {failure['component_type'].upper()} "
              f"{failure['component']} - {failure['failure_type']}")
    
    # Run simulation with failures
    results = await run_failure_simulation(
        str(graph_path),
        duration_seconds=45,
        failure_schedule=failure_schedule,
        output_path="output/scheduled_results.json"
    )
    
    print("\n✓ Completed! See output/scheduled_results.json")


async def example_4_partial_failure():
    """Example 4: Partial failure (degraded performance)"""
    
    print("\n" + "=" * 70)
    print("Example 4: Partial Failure (Performance Degradation)")
    print("=" * 70)
    
    # Create system
    graph = create_test_system()
    graph_path = Path("output/partial_failure.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    # Create simulator
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(str(graph_path))
    
    # Create failure simulator
    failure_sim = FailureSimulator()
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(30))
    
    # Wait 10 seconds, collect baseline
    await asyncio.sleep(10)
    baseline_delivered = simulator.global_stats.messages_delivered
    baseline_time = time.time()
    
    print(f"\n[10s] Baseline throughput: "
          f"{baseline_delivered / 10:.1f} msg/s")
    
    # Inject partial failure
    print("\n[10s] Injecting partial failure: A1 (50% degradation)")
    failure_sim.inject_failure(
        simulator,
        "A1",
        ComponentType.APPLICATION,
        FailureType.PARTIAL,
        severity=0.5,
        enable_cascade=False
    )
    
    # Wait and measure again
    await asyncio.sleep(10)
    current_delivered = simulator.global_stats.messages_delivered
    degraded_throughput = (current_delivered - baseline_delivered) / 10
    
    print(f"\n[20s] Degraded throughput: {degraded_throughput:.1f} msg/s")
    if baseline_delivered > 0:
        print(f"  Degradation: {((baseline_delivered/10 - degraded_throughput) / (baseline_delivered/10) * 100):.1f}%")
    elif baseline_delivered == 0:
        print("  No baseline messages delivered.")
    
    # Wait for completion
    await sim_task
    
    # Analyze
    impact = failure_sim.analyze_impact(simulator)
    print(f"\nFinal Impact:")
    print(f"  Messages Lost: {impact.messages_lost}")
    print(f"  Messages Delayed: {impact.messages_delayed}")


async def example_5_recovery_scenario():
    """Example 5: Failure with recovery"""
    
    print("\n" + "=" * 70)
    print("Example 5: Failure and Recovery")
    print("=" * 70)
    
    # Create system
    graph = create_test_system()
    graph_path = Path("output/recovery_test.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    # Create simulator
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(str(graph_path))
    
    # Create failure simulator
    failure_sim = FailureSimulator(recovery_enabled=True)
    
    # Start simulation
    sim_task = asyncio.create_task(simulator.run_simulation(45))
    
    # Inject failure at 10s
    await asyncio.sleep(10)
    print("\n[10s] Injecting failure: A1")
    
    failure_sim.inject_failure(
        simulator,
        "A1",
        ComponentType.APPLICATION,
        FailureType.COMPLETE,
        severity=1.0
    )
    
    print(f"  Active failures: {len(failure_sim.active_failures)}")
    
    # Recover at 25s
    await asyncio.sleep(15)
    print("\n[25s] Recovering component: A1")
    
    recovered = failure_sim.recover_component(simulator, "A1")
    print(f"  Recovery successful: {recovered}")
    print(f"  Active failures: {len(failure_sim.active_failures)}")
    
    # Wait for completion
    await sim_task
    
    # Analyze recovery
    recovery_event = [e for e in failure_sim.failure_events if e.component_id == "A1"][0]
    
    print(f"\nRecovery Analysis:")
    print(f"  Failure Duration: {recovery_event.recovery_time - recovery_event.timestamp:.1f}s")
    print(f"  Recovered: {recovery_event.recovered}")


async def example_6_resilience_testing():
    """Example 6: Systematic resilience testing"""
    
    print("\n" + "=" * 70)
    print("Example 6: Systematic Resilience Testing")
    print("=" * 70)
    
    # Create system
    graph = create_test_system()
    graph_path = Path("output/resilience_test.json")
    
    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    
    # Test each broker failure
    brokers = ["B1", "B2", "B3"]
    results = []
    
    for broker_id in brokers:
        print(f"\nTesting failure of {broker_id}...")
        
        # Create fresh simulator
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(str(graph_path))
        
        failure_sim = FailureSimulator(
            cascade_threshold=0.7,
            cascade_probability=0.6
        )
        
        # Run simulation
        sim_task = asyncio.create_task(simulator.run_simulation(20))
        
        # Inject failure at 5s
        await asyncio.sleep(5)
        failure_sim.inject_failure(
            simulator,
            broker_id,
            ComponentType.BROKER,
            FailureType.COMPLETE,
            enable_cascade=True
        )
        
        # Wait for completion
        await sim_task
        
        # Analyze impact
        impact = failure_sim.analyze_impact(simulator)
        
        results.append({
            'broker': broker_id,
            'failed_components': len(impact.failed_components),
            'affected_components': len(impact.affected_components),
            'unavailable_topics': len(impact.unavailable_topics),
            'cascade_depth': impact.cascade_depth,
            'cascade_width': impact.cascade_width
        })
        
        print(f"  Impact: {len(impact.affected_components)} affected, "
              f"{impact.cascade_width} cascaded")
    
    # Compare results
    print("\n" + "-" * 70)
    print("RESILIENCE COMPARISON")
    print("-" * 70)
    
    print(f"\n{'Broker':<10} {'Failed':<10} {'Affected':<12} {'Topics':<10} {'Cascade':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['broker']:<10} {r['failed_components']:<10} "
              f"{r['affected_components']:<12} {r['unavailable_topics']:<10} "
              f"{r['cascade_depth']}/{r['cascade_width']:<10}")
    
    # Identify most critical
    most_critical = max(results, key=lambda x: x['affected_components'])
    print(f"\nMost Critical: {most_critical['broker']} "
          f"({most_critical['affected_components']} components affected)")
    
    # Save comparison
    with open("output/resilience_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Comparison saved to output/resilience_comparison.json")


async def example_7_complex_cascade_scenario():
    """Example 7: Complex cascading failure scenario"""
    
    print("\n" + "=" * 70)
    print("Example 7: Complex Cascading Failure")
    print("=" * 70)
    
    # Create larger system for cascade
    import random
    random.seed(42)
    
    large_graph = {
        "metadata": {"scenario": "Large Cascade Test"},
        "nodes": [
            {"id": f"N{i}", "name": f"Node{i}"}
            for i in range(1, 21)
        ],
        "applications": [
            {"id": f"A{i}", "name": f"App{i}", "type": "PROSUMER"}
            for i in range(1, 101)
        ],
        "topics": [
            {"id": f"T{i}", "name": f"topic/channel{i}",
             "qos": {"history_depth": 5}}
            for i in range(1, 51)
        ],
        "brokers": [
            {"id": f"B{i}", "name": f"Broker{i}"}
            for i in range(1, 6)
        ],
        "relationships": {
            "publishes_to": [
                {"from": f"A{i}", "to": f"T{random.randint(1, 50)}",
                 "period_ms": 200, "msg_size": 512}
                for i in range(1, 101)
            ],
            "subscribes_to": [
                {"from": f"A{i}", "to": f"T{j}"}
                for i in range(1, 101)
                for j in random.sample(range(1, 51), 3)
            ],
            "runs_on": [
                {"from": f"A{i}", "to": f"N{((i-1) % 20) + 1}"}
                for i in range(1, 101)
            ],
            "routes": [
                {"from": f"B{((i-1) % 5) + 1}", "to": f"T{i}"}
                for i in range(1, 51)
            ]
        }
    }
    
    graph_path = Path("output/cascade_test.json")
    with open(graph_path, 'w') as f:
        json.dump(large_graph, f)
    
    print(f"\nSystem: 100 apps, 50 topics, 5 brokers")
    
    # Run with cascade
    failure_schedule = [
        {
            'time': 15,
            'component': 'B1',
            'component_type': 'broker',
            'failure_type': 'complete',
            'severity': 1.0,
            'enable_cascade': True
        }
    ]
    
    print("\nInitiating cascade from B1 failure...")
    
    results = await run_failure_simulation(
        str(graph_path),
        duration_seconds=60,
        failure_schedule=failure_schedule,
        output_path="output/cascade_results.json"
    )
    
    # Detailed cascade analysis
    cascade_events = [e for e in results['failures']['events'] 
                     if e['failure_type'] == 'cascade']
    
    if cascade_events:
        print(f"\nCascade Chain:")
        for i, event in enumerate(cascade_events[:10], 1):
            print(f"  {i}. {event['component_type']} {event['component_id']} "
                  f"← {event['propagated_from']}")
        
        if len(cascade_events) > 10:
            print(f"  ... and {len(cascade_events) - 10} more")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("FAILURE SIMULATION - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    try:
        asyncio.run(example_1_single_application_failure())
        asyncio.run(example_2_broker_failure_cascade())
        asyncio.run(example_3_scheduled_failures())
        asyncio.run(example_4_partial_failure())
        asyncio.run(example_5_recovery_scenario())
        asyncio.run(example_6_resilience_testing())
        asyncio.run(example_7_complex_cascade_scenario())
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Single component failures")
        print("  ✓ Broker failure cascades")
        print("  ✓ Scheduled failure injection")
        print("  ✓ Partial failures (degradation)")
        print("  ✓ Recovery scenarios")
        print("  ✓ Systematic resilience testing")
        print("  ✓ Complex cascade analysis")
        
        print("\n⚡ Failure Types Supported:")
        print("  - Complete failure (crash)")
        print("  - Partial failure (degradation)")
        print("  - Cascading failures")
        print("  - Network partitions")
        print("  - Intermittent failures")
        
        print("\n📊 Impact Metrics:")
        print("  - Failed components")
        print("  - Affected components")
        print("  - Isolated applications")
        print("  - Unavailable topics")
        print("  - Cascade depth & width")
        print("  - Performance degradation")
        print("  - Message loss rates")
        
        print("\n📖 Usage:")
        print("""
import asyncio
from src.simulation.enhanced_failure_simulator import run_failure_simulation

# Define failures
failures = [
    {'time': 10, 'component': 'B1', 'component_type': 'broker', 
     'failure_type': 'complete', 'enable_cascade': True}
]

# Run simulation
results = asyncio.run(
    run_failure_simulation("system.json", 60, failures)
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
