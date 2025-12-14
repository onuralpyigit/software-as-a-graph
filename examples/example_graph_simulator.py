#!/usr/bin/env python3
"""
Graph Simulator Examples
========================

Demonstrates how to use the GraphSimulator to simulate failures
in pub-sub systems and measure impact using DEPENDS_ON relationships.

Examples:
1. Basic single failure simulation
2. Multiple component failures
3. Cascade failure propagation
4. Exhaustive simulation for impact ranking
5. Using with GraphAnalyzer
6. Comparing component criticality
7. Generating reports

Usage:
    python example_graph_simulator.py

Author: Software-as-a-Graph Research Project
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.simulation import (
    GraphSimulator,
    SimulationResult,
    BatchSimulationResult,
    FailureMode,
    simulate_single_failure,
    simulate_and_rank,
)
from src.analysis import GraphAnalyzer


# ============================================================================
# Sample Data
# ============================================================================

SAMPLE_PUBSUB_DATA = {
    "nodes": [
        {"id": "N1", "name": "ComputeNode1", "type": "compute"},
        {"id": "N2", "name": "ComputeNode2", "type": "compute"},
        {"id": "N3", "name": "EdgeNode", "type": "edge"}
    ],
    "brokers": [
        {"id": "B1", "name": "MainBroker", "node": "N1"},
        {"id": "B2", "name": "BackupBroker", "node": "N2"}
    ],
    "applications": [
        {"id": "A1", "name": "SensorReader", "role": "pub", "node": "N3"},
        {"id": "A2", "name": "DataProcessor", "role": "both", "node": "N1"},
        {"id": "A3", "name": "Analytics", "role": "sub", "node": "N1"},
        {"id": "A4", "name": "Dashboard", "role": "sub", "node": "N2"},
        {"id": "A5", "name": "Alerting", "role": "sub", "node": "N2"},
        {"id": "A6", "name": "Logger", "role": "sub", "node": "N1"}
    ],
    "topics": [
        {"id": "T1", "name": "sensor/data", "broker": "B1"},
        {"id": "T2", "name": "processed/data", "broker": "B1"},
        {"id": "T3", "name": "alerts", "broker": "B2"}
    ],
    "relationships": {
        "publishes_to": [
            {"from": "A1", "to": "T1"},
            {"from": "A2", "to": "T2"},
            {"from": "A2", "to": "T3"}
        ],
        "subscribes_to": [
            {"from": "A2", "to": "T1"},
            {"from": "A3", "to": "T2"},
            {"from": "A4", "to": "T2"},
            {"from": "A5", "to": "T3"},
            {"from": "A6", "to": "T1"},
            {"from": "A6", "to": "T2"}
        ],
        "runs_on": [
            {"from": "A1", "to": "N3"},
            {"from": "A2", "to": "N1"},
            {"from": "A3", "to": "N1"},
            {"from": "A4", "to": "N2"},
            {"from": "A5", "to": "N2"},
            {"from": "A6", "to": "N1"},
            {"from": "B1", "to": "N1"},
            {"from": "B2", "to": "N2"}
        ],
        "routes": [
            {"from": "B1", "to": "T1"},
            {"from": "B1", "to": "T2"},
            {"from": "B2", "to": "T3"}
        ]
    }
}


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n── {title} ──")


# ============================================================================
# Example Functions
# ============================================================================

def example_1_basic_simulation():
    """Example 1: Basic single failure simulation"""
    print_header("Example 1: Basic Single Failure Simulation")
    
    # Build dependency graph using analyzer
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analyzer.derive_depends_on()
    graph = analyzer.build_dependency_graph()
    
    print(f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create simulator
    simulator = GraphSimulator(seed=42)
    
    # Simulate failure of DataProcessor (A2)
    print_subheader("Simulating failure of A2 (DataProcessor)")
    result = simulator.simulate_failure(graph, 'A2')
    
    # Print results
    print(f"\nImpact Score: {result.impact_score:.1%}")
    print(f"Resilience Score: {result.resilience_score:.1%}")
    print(f"Reachability Loss: {result.reachability_loss_pct:.1f}%")
    print(f"Affected Components: {len(result.affected_components)}")
    
    return result


def example_2_multiple_failures():
    """Example 2: Multiple simultaneous failures"""
    print_header("Example 2: Multiple Simultaneous Failures")
    
    # Build graph
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    graph = analyzer.build_dependency_graph()
    
    simulator = GraphSimulator(seed=42)
    
    # Compare single vs multiple failures
    print_subheader("Single failure (A1)")
    single_result = simulator.simulate_failure(graph, 'A1')
    print(f"Impact: {single_result.impact_score:.1%}")
    
    print_subheader("Multiple failures (A1, A2)")
    multi_result = simulator.simulate_multiple_failures(graph, ['A1', 'A2'])
    print(f"Impact: {multi_result.impact_score:.1%}")
    
    print_subheader("Comparison")
    print(f"Impact increase: {multi_result.impact_score - single_result.impact_score:.1%}")
    
    return single_result, multi_result


def example_3_cascade_failures():
    """Example 3: Cascade failure propagation"""
    print_header("Example 3: Cascade Failure Propagation")
    
    # Build graph
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    graph = analyzer.build_dependency_graph()
    
    # Use low threshold to encourage cascades
    simulator = GraphSimulator(
        cascade_threshold=0.3,
        cascade_probability=0.8,
        max_cascade_depth=5,
        seed=42
    )
    
    # Compare with and without cascade
    print_subheader("Without cascade propagation")
    no_cascade = simulator.simulate_failure(graph, 'A2', enable_cascade=False)
    print(f"Failed: {len(no_cascade.failed_components)}")
    print(f"Impact: {no_cascade.impact_score:.1%}")
    
    print_subheader("With cascade propagation")
    with_cascade = simulator.simulate_failure(graph, 'A2', enable_cascade=True)
    print(f"Primary failures: {len(with_cascade.failed_components) - len(with_cascade.cascade_failures)}")
    print(f"Cascade failures: {len(with_cascade.cascade_failures)}")
    print(f"Total failed: {len(with_cascade.failed_components)}")
    print(f"Impact: {with_cascade.impact_score:.1%}")
    
    if with_cascade.cascade_failures:
        print(f"\nCascade chain: {' → '.join(with_cascade.cascade_failures)}")
    
    return no_cascade, with_cascade


def example_4_exhaustive_simulation():
    """Example 4: Exhaustive simulation for impact ranking"""
    print_header("Example 4: Exhaustive Simulation")
    
    # Build graph
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    graph = analyzer.build_dependency_graph()
    
    simulator = GraphSimulator(seed=42)
    
    print_subheader("Testing all components individually")
    batch_result = simulator.simulate_all_single_failures(graph)
    
    print(f"\nTotal simulations: {batch_result.total_simulations}")
    
    # Summary statistics
    summary = batch_result.summary
    print(f"\nImpact Score Range:")
    print(f"  Min: {summary['impact_score']['min']:.1%}")
    print(f"  Max: {summary['impact_score']['max']:.1%}")
    print(f"  Mean: {summary['impact_score']['mean']:.1%}")
    
    print_subheader("Impact Ranking (Top 5)")
    ranking = batch_result.get_impact_ranking()
    for i, (comp, impact) in enumerate(ranking[:5], 1):
        # Classify severity
        if impact >= 0.7:
            sev = "CRITICAL"
        elif impact >= 0.5:
            sev = "HIGH"
        elif impact >= 0.3:
            sev = "MEDIUM"
        else:
            sev = "LOW"
        print(f"  {i}. {comp:20s} {impact:.1%} [{sev}]")
    
    return batch_result


def example_5_with_analyzer():
    """Example 5: Full workflow with GraphAnalyzer"""
    print_header("Example 5: Full Workflow with GraphAnalyzer")
    
    # Step 1: Analyze the system
    print_subheader("Step 1: Analyze System")
    
    analyzer = GraphAnalyzer(alpha=0.4, beta=0.3, gamma=0.3)
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis = analyzer.analyze()
    
    print(f"Total DEPENDS_ON edges: {len(analysis.depends_on_edges)}")
    print(f"By type: {analysis.to_dict()['depends_on']['by_type']}")
    
    # Step 2: Get predicted critical components
    print_subheader("Step 2: Predicted Critical Components")
    
    critical = [s for s in analysis.criticality_scores 
                if s.level.value in ('critical', 'high')]
    
    print(f"Critical/High components: {len(critical)}")
    for score in critical[:3]:
        print(f"  - {score.node_id}: {score.composite_score:.2f} ({score.level.value})")
    
    # Step 3: Simulate failures
    print_subheader("Step 3: Simulate Failures")
    
    simulator = GraphSimulator(seed=42)
    graph = analyzer.G  # Use the built graph
    
    # Test the predicted critical components
    for score in critical[:3]:
        result = simulator.simulate_failure(graph, score.node_id)
        print(f"  {score.node_id}: predicted={score.composite_score:.2f}, "
              f"actual_impact={result.impact_score:.2%}")
    
    return analysis, critical


def example_6_compare_criticality():
    """Example 6: Compare predicted vs actual criticality"""
    print_header("Example 6: Compare Predicted vs Actual Criticality")
    
    # Analyze
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis = analyzer.analyze()
    graph = analyzer.G
    
    # Simulate all
    simulator = GraphSimulator(seed=42)
    batch = simulator.simulate_all_single_failures(graph)
    
    # Build comparison
    print_subheader("Criticality Comparison")
    
    # Create lookup for predicted scores
    predicted = {s.node_id: s.composite_score for s in analysis.criticality_scores}
    
    # Create lookup for actual impact
    actual = {r.failed_components[0]: r.impact_score 
              for r in batch.results if len(r.failed_components) == 1}
    
    # Compare for applications
    print(f"\n{'Component':<20} {'Predicted':>10} {'Actual':>10} {'Match':>8}")
    print("-" * 50)
    
    for comp in sorted(predicted.keys()):
        if comp in actual:
            pred = predicted[comp]
            act = actual[comp]
            # Simple match check
            match = "✓" if abs(pred - act) < 0.3 else "○"
            print(f"{comp:<20} {pred:>10.2f} {act:>10.2%} {match:>8}")
    
    return predicted, actual


def example_7_generate_report():
    """Example 7: Generate detailed report"""
    print_header("Example 7: Generate Report")
    
    # Build graph and simulate
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    graph = analyzer.build_dependency_graph()
    
    simulator = GraphSimulator(seed=42)
    result = simulator.simulate_failure(graph, 'A2', enable_cascade=True)
    
    # Generate report
    report = simulator.generate_report(result)
    
    print_subheader("Summary")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    print_subheader("Failures")
    for key, value in report['failures'].items():
        if key != 'components':
            print(f"  {key}: {value}")
    
    print_subheader("Impact")
    for key, value in report['impact'].items():
        print(f"  {key}: {value}")
    
    print_subheader("Recommendations")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    return report


def example_convenience_functions():
    """Example: Using convenience functions"""
    print_header("Bonus: Convenience Functions")
    
    # Build graph
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    graph = analyzer.build_dependency_graph()
    
    # One-liner simulation
    print_subheader("simulate_single_failure()")
    result = simulate_single_failure(graph, 'B1')
    print(f"Broker failure impact: {result.impact_score:.1%}")
    
    # One-liner ranking
    print_subheader("simulate_and_rank()")
    ranking = simulate_and_rank(graph, component_types=['Application'])
    print("Application ranking by impact:")
    for comp, impact in ranking[:3]:
        print(f"  {comp}: {impact:.1%}")
    
    return result, ranking


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" GRAPH SIMULATOR EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_1_basic_simulation()
    example_2_multiple_failures()
    example_3_cascade_failures()
    example_4_exhaustive_simulation()
    example_5_with_analyzer()
    example_6_compare_criticality()
    example_7_generate_report()
    example_convenience_functions()
    
    print("\n" + "=" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()