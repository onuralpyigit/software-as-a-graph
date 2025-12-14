#!/usr/bin/env python3
"""
Example: Graph Analyzer Usage
=============================

This script demonstrates how to use the GraphAnalyzer to analyze
pub-sub systems by deriving DEPENDS_ON relationships.

Examples covered:
1. Basic analysis
2. Accessing derived dependencies
3. Working with criticality scores
4. Structural analysis
5. Custom weights
6. Export to different formats

Author: Software-as-a-Graph Research Project
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    GraphAnalyzer,
    DependencyType,
    CriticalityLevel,
    analyze_pubsub_system,
    derive_dependencies,
)


# ============================================================================
# Sample Data
# ============================================================================

def get_sample_iot_system() -> Dict[str, Any]:
    """
    Sample IoT pub-sub system with sensors, processors, and actuators.
    
    Architecture:
    - 3 infrastructure nodes (edge, processing, cloud)
    - 2 brokers (edge broker, cloud broker)
    - 6 applications (sensors, processors, dashboard)
    - 5 topics for data flow
    """
    return {
        "nodes": [
            {"id": "edge_node", "name": "Edge Gateway"},
            {"id": "proc_node", "name": "Processing Server"},
            {"id": "cloud_node", "name": "Cloud Instance"}
        ],
        "brokers": [
            {"id": "edge_broker", "name": "Edge MQTT Broker"},
            {"id": "cloud_broker", "name": "Cloud Message Broker"}
        ],
        "applications": [
            {"id": "temp_sensor", "name": "Temperature Sensor", "role": "pub"},
            {"id": "humidity_sensor", "name": "Humidity Sensor", "role": "pub"},
            {"id": "motion_sensor", "name": "Motion Sensor", "role": "pub"},
            {"id": "data_processor", "name": "Data Processor", "role": "pubsub"},
            {"id": "alert_service", "name": "Alert Service", "role": "sub"},
            {"id": "dashboard", "name": "Dashboard", "role": "sub"}
        ],
        "topics": [
            {"id": "raw_temp", "name": "sensor/temperature", "size": 256},
            {"id": "raw_humidity", "name": "sensor/humidity", "size": 256},
            {"id": "raw_motion", "name": "sensor/motion", "size": 128},
            {"id": "processed_data", "name": "processed/environment", "size": 512},
            {"id": "alerts", "name": "alerts/critical", "size": 128}
        ],
        "relationships": {
            "publishes_to": [
                {"from": "temp_sensor", "to": "raw_temp"},
                {"from": "humidity_sensor", "to": "raw_humidity"},
                {"from": "motion_sensor", "to": "raw_motion"},
                {"from": "data_processor", "to": "processed_data"},
                {"from": "data_processor", "to": "alerts"}
            ],
            "subscribes_to": [
                {"from": "data_processor", "to": "raw_temp"},
                {"from": "data_processor", "to": "raw_humidity"},
                {"from": "data_processor", "to": "raw_motion"},
                {"from": "alert_service", "to": "alerts"},
                {"from": "dashboard", "to": "processed_data"},
                {"from": "dashboard", "to": "alerts"}
            ],
            "runs_on": [
                {"from": "temp_sensor", "to": "edge_node"},
                {"from": "humidity_sensor", "to": "edge_node"},
                {"from": "motion_sensor", "to": "edge_node"},
                {"from": "data_processor", "to": "proc_node"},
                {"from": "alert_service", "to": "proc_node"},
                {"from": "dashboard", "to": "cloud_node"}
            ],
            "routes": [
                {"from": "edge_broker", "to": "raw_temp"},
                {"from": "edge_broker", "to": "raw_humidity"},
                {"from": "edge_broker", "to": "raw_motion"},
                {"from": "cloud_broker", "to": "processed_data"},
                {"from": "cloud_broker", "to": "alerts"}
            ],
            "connects_to": [
                {"from": "edge_node", "to": "proc_node"},
                {"from": "edge_node", "to": "cloud_node"},
                {"from": "proc_node", "to": "cloud_node"}
            ]
        }
    }


# ============================================================================
# Example Functions
# ============================================================================

def example_basic_analysis():
    """
    Example 1: Basic Analysis
    
    Shows how to run a complete analysis and access results.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Analysis")
    print("="*70)
    
    # Create analyzer
    analyzer = GraphAnalyzer()
    
    # Load data
    data = get_sample_iot_system()
    analyzer.load_from_dict(data)
    
    # Run analysis
    result = analyzer.analyze()
    
    # Print graph summary
    print("\nGraph Summary:")
    print(f"  Total Nodes: {result.graph_summary['total_nodes']}")
    print(f"  Total DEPENDS_ON Edges: {result.graph_summary['total_edges']}")
    print(f"  Graph Density: {result.graph_summary['density']:.4f}")
    print(f"  Connected: {result.graph_summary['is_connected']}")
    
    print("\n  Nodes by Type:")
    for ntype, count in result.graph_summary['nodes_by_type'].items():
        print(f"    - {ntype}: {count}")
    
    print("\n  Dependencies by Type:")
    for dtype, count in result.to_dict()['depends_on']['by_type'].items():
        print(f"    - {dtype}: {count}")
    
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")


def example_derived_dependencies():
    """
    Example 2: Working with Derived Dependencies
    
    Shows how to access and work with the derived DEPENDS_ON relationships.
    """
    print("\n" + "="*70)
    print("Example 2: Working with Derived Dependencies")
    print("="*70)
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(get_sample_iot_system())
    
    # Derive dependencies (without full analysis)
    edges = analyzer.derive_depends_on()
    
    print(f"\nTotal derived dependencies: {len(edges)}")
    
    # Group by type
    by_type: Dict[DependencyType, list] = {}
    for edge in edges:
        if edge.dep_type not in by_type:
            by_type[edge.dep_type] = []
        by_type[edge.dep_type].append(edge)
    
    # APP_TO_APP dependencies
    print("\n--- APP_TO_APP Dependencies ---")
    print("(Which applications depend on which other applications)")
    for edge in by_type.get(DependencyType.APP_TO_APP, []):
        topics = ', '.join(edge.via_topics)
        print(f"  {edge.source} → {edge.target}")
        print(f"    via topics: {topics}")
        print(f"    weight: {edge.weight:.2f}")
    
    # APP_TO_BROKER dependencies
    print("\n--- APP_TO_BROKER Dependencies ---")
    print("(Which applications depend on which brokers)")
    for edge in by_type.get(DependencyType.APP_TO_BROKER, []):
        topics = ', '.join(edge.via_topics)
        print(f"  {edge.source} → {edge.target}")
        print(f"    via topics: {topics}")
    
    # NODE_TO_NODE dependencies
    print("\n--- NODE_TO_NODE Dependencies ---")
    print("(Infrastructure dependencies between nodes)")
    for edge in by_type.get(DependencyType.NODE_TO_NODE, []):
        apps = ', '.join(edge.via_apps)
        print(f"  {edge.source} → {edge.target}")
        print(f"    via apps: {apps}")
    
    # NODE_TO_BROKER dependencies
    print("\n--- NODE_TO_BROKER Dependencies ---")
    print("(Which nodes depend on brokers on other nodes)")
    for edge in by_type.get(DependencyType.NODE_TO_BROKER, []):
        print(f"  {edge.source} → {edge.target}")


def example_criticality_scores():
    """
    Example 3: Working with Criticality Scores
    
    Shows how to access and interpret criticality scores.
    """
    print("\n" + "="*70)
    print("Example 3: Criticality Scores")
    print("="*70)
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(get_sample_iot_system())
    result = analyzer.analyze()
    
    print("\nCriticality Formula: C_score = α·BC + β·AP + γ·I")
    print(f"  α (betweenness weight): {analyzer.alpha}")
    print(f"  β (articulation point weight): {analyzer.beta}")
    print(f"  γ (impact weight): {analyzer.gamma}")
    
    # Group by level
    by_level: Dict[CriticalityLevel, list] = {}
    for score in result.criticality_scores:
        if score.level not in by_level:
            by_level[score.level] = []
        by_level[score.level].append(score)
    
    print("\n--- Components by Criticality Level ---")
    for level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH, 
                  CriticalityLevel.MEDIUM, CriticalityLevel.LOW]:
        scores = by_level.get(level, [])
        if scores:
            print(f"\n{level.value.upper()} ({len(scores)} components):")
            for score in scores[:3]:  # Show top 3 per level
                print(f"  • {score.node_id} ({score.node_type})")
                print(f"    Score: {score.composite_score:.4f}")
                print(f"    Betweenness: {score.betweenness:.4f}")
                print(f"    Articulation Point: {score.is_articulation_point}")
                print(f"    Impact: {score.impact_score:.4f}")
                print(f"    Reasons: {', '.join(score.reasons[:2])}")
    
    # Top 5 most critical
    print("\n--- Top 5 Most Critical Components ---")
    for i, score in enumerate(result.criticality_scores[:5], 1):
        print(f"\n  {i}. {score.node_id} ({score.node_type})")
        print(f"     Composite Score: {score.composite_score:.4f}")
        print(f"     Level: {score.level.value.upper()}")


def example_structural_analysis():
    """
    Example 4: Structural Analysis
    
    Shows how to interpret structural analysis results.
    """
    print("\n" + "="*70)
    print("Example 4: Structural Analysis")
    print("="*70)
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(get_sample_iot_system())
    result = analyzer.analyze()
    
    struct = result.structural_analysis
    
    print("\n--- Single Points of Failure ---")
    print(f"Articulation Points: {struct['articulation_point_count']}")
    if struct['articulation_points']:
        print("  These components are critical - their failure disconnects the system:")
        for ap in struct['articulation_points']:
            print(f"    • {ap}")
    
    print(f"\nBridge Edges: {struct['bridge_count']}")
    if struct['bridges']:
        print("  These connections are critical - their failure disconnects the system:")
        for bridge in struct['bridges'][:5]:
            print(f"    • {bridge[0]} ↔ {bridge[1]}")
    
    print("\n--- Connectivity ---")
    print(f"Weakly Connected Components: {struct['weakly_connected_components']}")
    print(f"Strongly Connected Components: {struct['strongly_connected_components']}")
    
    print("\n--- Circular Dependencies ---")
    print(f"Has Cycles: {struct['has_cycles']}")
    if struct['cycles']:
        print("  Detected cycles (potential issues):")
        for cycle in struct['cycles'][:3]:
            print(f"    • {' → '.join(cycle)} → {cycle[0]}")


def example_custom_weights():
    """
    Example 5: Custom Criticality Weights
    
    Shows how to customize the criticality scoring formula.
    """
    print("\n" + "="*70)
    print("Example 5: Custom Criticality Weights")
    print("="*70)
    
    data = get_sample_iot_system()
    
    # Default weights
    print("\n--- Default Weights (α=0.4, β=0.3, γ=0.3) ---")
    analyzer_default = GraphAnalyzer(alpha=0.4, beta=0.3, gamma=0.3)
    analyzer_default.load_from_dict(data)
    result_default = analyzer_default.analyze()
    
    print("Top 3 by default weights:")
    for i, score in enumerate(result_default.criticality_scores[:3], 1):
        print(f"  {i}. {score.node_id}: {score.composite_score:.4f}")
    
    # Emphasize structural importance (articulation points)
    print("\n--- Emphasize Structure (α=0.2, β=0.6, γ=0.2) ---")
    analyzer_struct = GraphAnalyzer(alpha=0.2, beta=0.6, gamma=0.2)
    analyzer_struct.load_from_dict(data)
    result_struct = analyzer_struct.analyze()
    
    print("Top 3 emphasizing articulation points:")
    for i, score in enumerate(result_struct.criticality_scores[:3], 1):
        print(f"  {i}. {score.node_id}: {score.composite_score:.4f}")
    
    # Emphasize impact
    print("\n--- Emphasize Impact (α=0.2, β=0.2, γ=0.6) ---")
    analyzer_impact = GraphAnalyzer(alpha=0.2, beta=0.2, gamma=0.6)
    analyzer_impact.load_from_dict(data)
    result_impact = analyzer_impact.analyze()
    
    print("Top 3 emphasizing impact:")
    for i, score in enumerate(result_impact.criticality_scores[:3], 1):
        print(f"  {i}. {score.node_id}: {score.composite_score:.4f}")


def example_export_formats():
    """
    Example 6: Export to Different Formats
    
    Shows how to export analysis results.
    """
    print("\n" + "="*70)
    print("Example 6: Export Formats")
    print("="*70)
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(get_sample_iot_system())
    result = analyzer.analyze()
    
    # Convert to dictionary (for JSON export)
    result_dict = result.to_dict()
    
    print("\n--- JSON Structure ---")
    print("The result.to_dict() method returns a dictionary with:")
    print("  • graph_summary: Basic graph metrics")
    print("  • depends_on: All derived dependencies")
    print("  • criticality: Component criticality scores")
    print("  • structural: Structural analysis results")
    print("  • recommendations: Generated recommendations")
    
    # Show sample JSON
    print("\n--- Sample JSON Output (truncated) ---")
    sample = {
        'graph_summary': result_dict['graph_summary'],
        'depends_on': {
            'total': result_dict['depends_on']['total'],
            'by_type': result_dict['depends_on']['by_type']
        },
        'criticality': {
            'by_level': result_dict['criticality']['by_level'],
            'top_component': result_dict['criticality']['scores'][0] if result_dict['criticality']['scores'] else None
        }
    }
    print(json.dumps(sample, indent=2))
    
    # Access NetworkX graph
    print("\n--- NetworkX Graph Access ---")
    G = analyzer.G
    print(f"  Graph type: {type(G).__name__}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print("  You can use any NetworkX algorithm on this graph!")


def example_convenience_functions():
    """
    Example 7: Convenience Functions
    
    Shows quick one-liner analysis options.
    """
    print("\n" + "="*70)
    print("Example 7: Convenience Functions")
    print("="*70)
    
    data = get_sample_iot_system()
    
    # Quick dependency derivation
    print("\n--- derive_dependencies() ---")
    print("Quick way to just get dependencies without full analysis:")
    deps = derive_dependencies(data)
    print(f"  Derived {len(deps)} dependencies")
    print(f"  Sample: {deps[0]['source']} → {deps[0]['target']} ({deps[0]['type']})")
    
    # Full analysis from file (would need actual file)
    print("\n--- analyze_pubsub_system() ---")
    print("One-liner for full analysis from file:")
    print("  result = analyze_pubsub_system('system.json')")
    print("  # Returns AnalysisResult with all data")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("  GRAPH ANALYZER EXAMPLES")
    print("  Demonstrating DEPENDS_ON Relationship Analysis")
    print("="*70)
    
    # Run examples
    example_basic_analysis()
    example_derived_dependencies()
    example_criticality_scores()
    example_structural_analysis()
    example_custom_weights()
    example_export_formats()
    example_convenience_functions()
    
    print("\n" + "="*70)
    print("  Examples Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. DEPENDS_ON relationships are derived on-the-fly")
    print("  2. No need to store derived relationships in input files")
    print("  3. Four dependency types: APP_TO_APP, APP_TO_BROKER, NODE_TO_NODE, NODE_TO_BROKER")
    print("  4. Criticality scoring uses: C_score = α·BC + β·AP + γ·Impact")
    print("  5. Results include recommendations for improving system resilience")
    print()


if __name__ == '__main__':
    main()