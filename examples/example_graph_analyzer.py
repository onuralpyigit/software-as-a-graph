#!/usr/bin/env python3
"""
Example Graph Analyzer - Demonstration of Advanced Graph Algorithms
====================================================================

This script demonstrates how to use the enhanced graph analysis features
for analyzing distributed publish-subscribe systems.

Examples Include:
  1. Basic criticality analysis
  2. Edge-centric analysis
  3. HITS role analysis
  4. Network motif detection
  5. Dependency chain analysis
  6. Multi-layer correlation
  7. Ensemble criticality scoring
  8. Full pipeline analysis

Usage:
    python example_graph_analyzer.py                    # Run all examples
    python example_graph_analyzer.py --example basic    # Run specific example
    python example_graph_analyzer.py --list             # List available examples
    python example_graph_analyzer.py --generate         # Generate test data only

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    sys.exit(1)


# ============================================================================
# Terminal Colors
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'WARNING', 'FAIL', 'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}▸ {text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"    {Colors.DIM}•{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"  {Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_critical(text: str):
    print(f"  {Colors.FAIL}✗{Colors.ENDC} {Colors.BOLD}{text}{Colors.ENDC}")


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_iot_smart_city_data() -> Dict[str, Any]:
    """
    Generate a realistic IoT Smart City pub-sub system.
    
    Architecture:
    - Sensors publish to topics
    - Processing apps subscribe and republish
    - Dashboard apps consume processed data
    - Brokers route messages
    - Multiple nodes for infrastructure
    """
    return {
        "metadata": {
            "name": "Smart City IoT System",
            "version": "1.0",
            "scenario": "iot_smart_city",
            "generated": datetime.now().isoformat()
        },
        "applications": [
            # Sensor publishers
            {"id": "sensor_traffic_1", "name": "Traffic Sensor Zone A", "role": "pub"},
            {"id": "sensor_traffic_2", "name": "Traffic Sensor Zone B", "role": "pub"},
            {"id": "sensor_air_1", "name": "Air Quality Sensor Downtown", "role": "pub"},
            {"id": "sensor_air_2", "name": "Air Quality Sensor Industrial", "role": "pub"},
            {"id": "sensor_weather", "name": "Weather Station", "role": "pub"},
            {"id": "sensor_noise", "name": "Noise Level Sensor", "role": "pub"},
            
            # Processing apps (pub-sub)
            {"id": "proc_traffic", "name": "Traffic Processor", "role": "pubsub"},
            {"id": "proc_environment", "name": "Environment Processor", "role": "pubsub"},
            {"id": "proc_alerts", "name": "Alert Generator", "role": "pubsub"},
            {"id": "proc_analytics", "name": "Analytics Engine", "role": "pubsub"},
            
            # Consumer apps
            {"id": "dashboard_ops", "name": "Operations Dashboard", "role": "sub"},
            {"id": "dashboard_public", "name": "Public Dashboard", "role": "sub"},
            {"id": "mobile_app", "name": "Citizen Mobile App", "role": "sub"},
            {"id": "archive_service", "name": "Data Archiver", "role": "sub"},
        ],
        "brokers": [
            {"id": "broker_edge_1", "name": "Edge Broker Zone A"},
            {"id": "broker_edge_2", "name": "Edge Broker Zone B"},
            {"id": "broker_central", "name": "Central Broker"},
        ],
        "topics": [
            # Raw sensor topics
            {"id": "topic_traffic_raw", "name": "Raw Traffic Data", 
             "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}},
            {"id": "topic_air_raw", "name": "Raw Air Quality", 
             "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}},
            {"id": "topic_weather_raw", "name": "Raw Weather Data", 
             "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}},
            {"id": "topic_noise_raw", "name": "Raw Noise Levels", 
             "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}},
            
            # Processed topics
            {"id": "topic_traffic_proc", "name": "Processed Traffic", 
             "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}},
            {"id": "topic_env_proc", "name": "Processed Environment", 
             "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}},
            {"id": "topic_alerts", "name": "System Alerts", 
             "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "URGENT"}},
            {"id": "topic_analytics", "name": "Analytics Results", 
             "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}},
        ],
        "nodes": [
            {"id": "node_edge_1", "name": "Edge Server Zone A", "type": "compute"},
            {"id": "node_edge_2", "name": "Edge Server Zone B", "type": "compute"},
            {"id": "node_cloud_1", "name": "Cloud Server 1", "type": "compute"},
            {"id": "node_cloud_2", "name": "Cloud Server 2", "type": "compute"},
        ],
        "relationships": {
            "publishes_to": [
                # Sensors -> Raw topics
                {"from": "sensor_traffic_1", "to": "topic_traffic_raw"},
                {"from": "sensor_traffic_2", "to": "topic_traffic_raw"},
                {"from": "sensor_air_1", "to": "topic_air_raw"},
                {"from": "sensor_air_2", "to": "topic_air_raw"},
                {"from": "sensor_weather", "to": "topic_weather_raw"},
                {"from": "sensor_noise", "to": "topic_noise_raw"},
                
                # Processors -> Processed topics
                {"from": "proc_traffic", "to": "topic_traffic_proc"},
                {"from": "proc_environment", "to": "topic_env_proc"},
                {"from": "proc_alerts", "to": "topic_alerts"},
                {"from": "proc_analytics", "to": "topic_analytics"},
            ],
            "subscribes_to": [
                # Processors <- Raw topics
                {"from": "proc_traffic", "to": "topic_traffic_raw"},
                {"from": "proc_environment", "to": "topic_air_raw"},
                {"from": "proc_environment", "to": "topic_weather_raw"},
                {"from": "proc_environment", "to": "topic_noise_raw"},
                {"from": "proc_alerts", "to": "topic_traffic_proc"},
                {"from": "proc_alerts", "to": "topic_env_proc"},
                {"from": "proc_analytics", "to": "topic_traffic_proc"},
                {"from": "proc_analytics", "to": "topic_env_proc"},
                
                # Consumers <- Processed topics
                {"from": "dashboard_ops", "to": "topic_traffic_proc"},
                {"from": "dashboard_ops", "to": "topic_env_proc"},
                {"from": "dashboard_ops", "to": "topic_alerts"},
                {"from": "dashboard_public", "to": "topic_env_proc"},
                {"from": "dashboard_public", "to": "topic_analytics"},
                {"from": "mobile_app", "to": "topic_alerts"},
                {"from": "mobile_app", "to": "topic_analytics"},
                {"from": "archive_service", "to": "topic_traffic_proc"},
                {"from": "archive_service", "to": "topic_env_proc"},
                {"from": "archive_service", "to": "topic_alerts"},
            ],
            "routes": [
                {"from": "broker_edge_1", "to": "topic_traffic_raw"},
                {"from": "broker_edge_1", "to": "topic_air_raw"},
                {"from": "broker_edge_2", "to": "topic_weather_raw"},
                {"from": "broker_edge_2", "to": "topic_noise_raw"},
                {"from": "broker_central", "to": "topic_traffic_proc"},
                {"from": "broker_central", "to": "topic_env_proc"},
                {"from": "broker_central", "to": "topic_alerts"},
                {"from": "broker_central", "to": "topic_analytics"},
            ],
            "runs_on": [
                # Edge deployment
                {"from": "sensor_traffic_1", "to": "node_edge_1"},
                {"from": "sensor_traffic_2", "to": "node_edge_2"},
                {"from": "sensor_air_1", "to": "node_edge_1"},
                {"from": "sensor_air_2", "to": "node_edge_2"},
                {"from": "sensor_weather", "to": "node_edge_1"},
                {"from": "sensor_noise", "to": "node_edge_2"},
                {"from": "broker_edge_1", "to": "node_edge_1"},
                {"from": "broker_edge_2", "to": "node_edge_2"},
                
                # Cloud deployment
                {"from": "proc_traffic", "to": "node_cloud_1"},
                {"from": "proc_environment", "to": "node_cloud_1"},
                {"from": "proc_alerts", "to": "node_cloud_2"},
                {"from": "proc_analytics", "to": "node_cloud_2"},
                {"from": "broker_central", "to": "node_cloud_1"},
                {"from": "dashboard_ops", "to": "node_cloud_2"},
                {"from": "dashboard_public", "to": "node_cloud_2"},
            ],
            "connects_to": [
                {"from": "node_edge_1", "to": "node_cloud_1"},
                {"from": "node_edge_2", "to": "node_cloud_1"},
                {"from": "node_cloud_1", "to": "node_cloud_2"},
            ]
        }
    }


def build_graph_from_dict(data: Dict[str, Any]) -> nx.DiGraph:
    """Build NetworkX graph from dictionary data"""
    G = nx.DiGraph()
    
    # Add nodes - filter out 'type' from kwargs if present to avoid conflicts
    for app in data.get('applications', []):
        attrs = {k: v for k, v in app.items() if k != 'type'}
        G.add_node(app['id'], type='Application', **attrs)
    
    for broker in data.get('brokers', []):
        attrs = {k: v for k, v in broker.items() if k != 'type'}
        G.add_node(broker['id'], type='Broker', **attrs)
    
    for topic in data.get('topics', []):
        attrs = {k: v for k, v in topic.items() if k != 'type'}
        G.add_node(topic['id'], type='Topic', **attrs)
    
    for node in data.get('nodes', []):
        attrs = {k: v for k, v in node.items() if k != 'type'}
        G.add_node(node['id'], type='Node', **attrs)
    
    # Add edges
    edges = data.get('edges', data.get('relationships', {}))
    
    for pub in edges.get('publishes_to', []):
        G.add_edge(pub['from'], pub['to'], type='PUBLISHES_TO')
    
    for sub in edges.get('subscribes_to', []):
        G.add_edge(sub['from'], sub['to'], type='SUBSCRIBES_TO')
    
    for route in edges.get('routes', []):
        G.add_edge(route['from'], route['to'], type='ROUTES')
    
    for runs in edges.get('runs_on', []):
        G.add_edge(runs['from'], runs['to'], type='RUNS_ON')
    
    for conn in edges.get('connects_to', []):
        G.add_edge(conn['from'], conn['to'], type='CONNECTS_TO')
    
    return G


# ============================================================================
# Example Functions
# ============================================================================

def example_basic_criticality():
    """
    Example 1: Basic Criticality Analysis
    
    Demonstrates:
    - Loading a pub-sub system graph
    - Computing composite criticality scores
    - Analyzing structural properties
    """
    print_header("Example 1: Basic Criticality Analysis")
    
    print_section("Loading Smart City IoT System")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    
    print_success(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Count by type
    types = {}
    for n in G.nodes():
        t = G.nodes[n].get('type', 'Unknown')
        types[t] = types.get(t, 0) + 1
    
    for t, count in types.items():
        print_info(f"{t}: {count}")
    
    print_section("Computing Criticality Scores")
    
    from analyze_graph import calculate_criticality_scores, analyze_structure
    
    # Structural analysis
    structural = analyze_structure(G)
    print_success(f"Found {len(structural.articulation_points)} articulation points")
    print_success(f"Found {len(structural.bridges)} bridges")
    print_success(f"Graph density: {structural.density:.4f}")
    
    # Criticality scoring
    scores = calculate_criticality_scores(G, alpha=0.4, beta=0.3, gamma=0.3)
    
    print_section("Top 10 Critical Components")
    
    for i, score in enumerate(scores[:10], 1):
        level_color = Colors.FAIL if score.criticality_level == 'CRITICAL' else \
                     Colors.WARNING if score.criticality_level == 'HIGH' else Colors.ENDC
        
        print(f"  {i:2d}. {level_color}{score.node_id:30s}{Colors.ENDC} "
              f"({score.node_type:12s}) Score: {score.composite_score:.4f} "
              f"[{score.criticality_level}]")
        
        if score.is_articulation_point:
            print_info("   → Articulation point (single point of failure)")
    
    print_section("Criticality Distribution")
    
    level_counts = {}
    for s in scores:
        level_counts[s.criticality_level] = level_counts.get(s.criticality_level, 0) + 1
    
    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = level_counts.get(level, 0)
        bar = '█' * (count * 2)
        print(f"  {level:10s}: {count:3d} {bar}")
    
    return True


def example_edge_analysis():
    """
    Example 2: Edge-Centric Analysis
    
    Demonstrates:
    - Edge betweenness centrality
    - Bridge edge detection
    - Simmelian strength calculation
    - Critical relationship identification
    """
    print_header("Example 2: Edge-Centric Analysis")
    
    try:
        from src.analysis.relationship_analyzer import EdgeCriticalityAnalyzer
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    print_success(f"Loaded {G.number_of_edges()} edges")
    
    print_section("Analyzing Edge Criticality")
    
    analyzer = EdgeCriticalityAnalyzer(G)
    results = analyzer.analyze_all_edges()
    
    # Count by level
    edge_levels = {}
    for r in results:
        edge_levels[r.criticality_level] = edge_levels.get(r.criticality_level, 0) + 1
    
    print_success(f"Analyzed {len(results)} edges")
    for level, count in edge_levels.items():
        print_info(f"{level}: {count} edges")
    
    print_section("Top 10 Critical Edges")
    
    for i, result in enumerate(results[:10], 1):
        u, v = result.edge
        level_color = Colors.FAIL if result.criticality_level == 'CRITICAL' else Colors.WARNING
        
        bridge_marker = " [BRIDGE]" if result.is_bridge else ""
        
        print(f"  {i:2d}. {level_color}{u} → {v}{Colors.ENDC}")
        print(f"      Type: {result.edge_type}, Score: {result.criticality_score:.4f}{bridge_marker}")
        print(f"      BC: {result.betweenness_centrality:.4f}, Simmelian: {result.simmelian_strength}")
    
    print_section("Bridge Edges (Single Points of Failure)")
    
    bridges = [r for r in results if r.is_bridge]
    if bridges:
        for r in bridges[:5]:
            print_critical(f"{r.edge[0]} → {r.edge[1]} ({r.edge_type})")
    else:
        print_success("No bridge edges found - good redundancy!")
    
    return True


def example_hits_analysis():
    """
    Example 3: HITS Role Analysis
    
    Demonstrates:
    - Hub and Authority score computation
    - Role classification (publisher/subscriber/relay)
    - Role alignment analysis
    """
    print_header("Example 3: HITS Role Analysis")
    
    try:
        from src.analysis.relationship_analyzer import HITSRoleAnalyzer, ComponentRole
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    print_success(f"Loaded {G.number_of_nodes()} nodes")
    
    print_section("Computing HITS Scores")
    
    analyzer = HITSRoleAnalyzer(G)
    results = analyzer.analyze_roles()
    
    print_success(f"Analyzed roles for {len(results)} components")
    
    # Count roles
    role_counts = {}
    for r in results.values():
        role_counts[r.role.value] = role_counts.get(r.role.value, 0) + 1
    
    print("\n  Role Distribution:")
    for role, count in sorted(role_counts.items()):
        bar = '█' * count
        print(f"    {role:20s}: {count:2d} {bar}")
    
    print_section("Top 5 Hubs (Data Sources)")
    
    top_hubs = analyzer.get_top_hubs(5)
    for i, hub in enumerate(top_hubs, 1):
        r = results[hub]
        print(f"  {i}. {Colors.BLUE}{hub:30s}{Colors.ENDC} "
              f"Hub: {r.hub_score:.4f}, Rank: #{r.hub_rank}")
    
    print_section("Top 5 Authorities (Data Sinks)")
    
    top_auth = analyzer.get_top_authorities(5)
    for i, auth in enumerate(top_auth, 1):
        r = results[auth]
        print(f"  {i}. {Colors.GREEN}{auth:30s}{Colors.ENDC} "
              f"Auth: {r.authority_score:.4f}, Rank: #{r.authority_rank}")
    
    print_section("Role Alignment Analysis")
    
    # Find misaligned components
    misaligned = []
    for node_id, r in results.items():
        if r.role_alignment < 0.5:
            misaligned.append((node_id, r))
    
    if misaligned:
        print_warning(f"Found {len(misaligned)} components with potential role misalignment:")
        for node_id, r in misaligned[:5]:
            print_info(f"{node_id}: Expected {r.node_type} behavior, got {r.role.value}")
    else:
        print_success("All components aligned with expected roles")
    
    return True


def example_motif_detection():
    """
    Example 4: Network Motif Detection
    
    Demonstrates:
    - Fan-out pattern detection (broadcast)
    - Fan-in pattern detection (aggregation)
    - Chain pattern detection (pipelines)
    - Diamond pattern detection (redundancy)
    """
    print_header("Example 4: Network Motif Detection")
    
    try:
        from src.analysis.relationship_analyzer import MotifDetector, MotifType
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    print_success(f"Loaded graph with {G.number_of_nodes()} nodes")
    
    print_section("Detecting Network Motifs")
    
    detector = MotifDetector(G)
    motifs = detector.detect_all_motifs()
    summary = detector.get_motif_summary(motifs)
    
    print_success(f"Found {len(motifs)} motif instances")
    
    print("\n  Motif Distribution:")
    for motif_type, count in summary.items():
        icon = {
            'fan_out': '📡',
            'fan_in': '🔄',
            'chain': '⛓️',
            'diamond': '💎',
            'star': '⭐',
            'triangle': '🔺'
        }.get(motif_type, '•')
        
        bar = '█' * min(count, 20)
        print(f"    {icon} {motif_type:15s}: {count:3d} {bar}")
    
    print_section("Sample Motifs")
    
    # Show examples of each type
    shown_types = set()
    for motif in motifs:
        if motif.motif_type.value in shown_types:
            continue
        
        shown_types.add(motif.motif_type.value)
        
        print(f"\n  {Colors.CYAN}{motif.motif_type.value.upper()}{Colors.ENDC}")
        print(f"    Central: {motif.central_node}")
        print(f"    Nodes: {', '.join(motif.nodes[:5])}{'...' if len(motif.nodes) > 5 else ''}")
        print(f"    Criticality Impact: {motif.criticality_impact:.2f}")
        print(f"    {Colors.DIM}{motif.description}{Colors.ENDC}")
        
        if len(shown_types) >= 4:
            break
    
    print_section("Architectural Implications")
    
    fan_out_count = summary.get('fan_out', 0)
    fan_in_count = summary.get('fan_in', 0)
    star_count = summary.get('star', 0)
    
    if fan_out_count > 3:
        print_warning(f"{fan_out_count} fan-out patterns: Consider topic partitioning for scalability")
    
    if fan_in_count > 2:
        print_info(f"{fan_in_count} fan-in patterns: Data aggregation points detected")
    
    if star_count > 5:
        print_warning(f"{star_count} star patterns: High-connectivity hubs may be bottlenecks")
    
    return True


def example_dependency_chains():
    """
    Example 5: Dependency Chain Analysis
    
    Demonstrates:
    - Transitive dependency depth calculation
    - Upstream/downstream dependency counting
    - Fan-out/fan-in ratio analysis
    - Cascade risk identification
    """
    print_header("Example 5: Dependency Chain Analysis")
    
    try:
        from src.analysis.relationship_analyzer import DependencyChainAnalyzer
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    print_success(f"Loaded {G.number_of_nodes()} nodes")
    
    print_section("Analyzing Dependency Chains")
    
    analyzer = DependencyChainAnalyzer(G)
    results = analyzer.analyze_all()
    
    print_success(f"Analyzed {len(results)} components")
    
    # Find interesting statistics
    max_depth = max(r.transitive_depth for r in results.values())
    max_downstream = max(r.downstream_count for r in results.values())
    max_upstream = max(r.upstream_count for r in results.values())
    
    print(f"\n  Maximum transitive depth: {max_depth}")
    print(f"  Maximum downstream dependents: {max_downstream}")
    print(f"  Maximum upstream dependencies: {max_upstream}")
    
    print_section("Components with Deepest Dependencies")
    
    deepest = analyzer.get_deepest_chains(results, 5)
    for i, node_id in enumerate(deepest, 1):
        r = results[node_id]
        print(f"  {i}. {Colors.BLUE}{node_id:30s}{Colors.ENDC}")
        print(f"     Depth: {r.transitive_depth}, Upstream: {r.upstream_count}, "
              f"Downstream: {r.downstream_count}")
    
    print_section("High Fan-Out Components (Foundational)")
    
    high_fan_out = analyzer.get_highest_fan_out(results, 5)
    for i, node_id in enumerate(high_fan_out, 1):
        r = results[node_id]
        print(f"  {i}. {Colors.WARNING}{node_id:30s}{Colors.ENDC}")
        print(f"     Downstream: {r.downstream_count}, Ratio: {r.dependency_ratio:.2f}")
    
    print_section("Cascade Risk Analysis")
    
    # High cascade risk = high downstream + central position
    high_risk = [
        (n, r) for n, r in results.items()
        if r.downstream_count > 3 and r.chain_criticality > 0.3
    ]
    
    if high_risk:
        print_warning(f"Found {len(high_risk)} components with high cascade failure risk:")
        for node_id, r in sorted(high_risk, key=lambda x: -x[1].chain_criticality)[:5]:
            print_critical(f"  {node_id}: {r.downstream_count} dependents, "
                          f"criticality {r.chain_criticality:.2f}")
    else:
        print_success("No high cascade risk components identified")
    
    return True


def example_layer_correlation():
    """
    Example 6: Multi-Layer Correlation Analysis
    
    Demonstrates:
    - Cross-layer centrality correlation
    - Layer coupling coefficient calculation
    - Misalignment detection between layers
    """
    print_header("Example 6: Multi-Layer Correlation Analysis")
    
    try:
        from src.analysis.relationship_analyzer import LayerCorrelationAnalyzer
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    
    # Count layers
    layers = {}
    for n in G.nodes():
        layer = G.nodes[n].get('type', 'Unknown')
        layers[layer] = layers.get(layer, 0) + 1
    
    print_success(f"Loaded graph with {len(layers)} layers:")
    for layer, count in layers.items():
        print_info(f"{layer}: {count} nodes")
    
    print_section("Analyzing Layer Correlations")
    
    analyzer = LayerCorrelationAnalyzer(G)
    results = analyzer.analyze_correlations()
    
    print_success(f"Analyzed {len(results)} layer pairs")
    
    print_section("Correlation Matrix")
    
    print(f"\n  {'Layers':<30s} {'Correlation':>12s} {'Coupling':>10s} {'Misaligned':>10s}")
    print(f"  {'-'*65}")
    
    for r in results:
        layers_str = f"{r.layer1} ↔ {r.layer2}"
        
        # Color code correlation
        if r.centrality_correlation > 0.7:
            corr_color = Colors.GREEN
        elif r.centrality_correlation > 0.3:
            corr_color = Colors.WARNING
        else:
            corr_color = Colors.FAIL
        
        print(f"  {layers_str:<30s} "
              f"{corr_color}{r.centrality_correlation:>12.3f}{Colors.ENDC} "
              f"{r.coupling_coefficient:>10.3f} "
              f"{len(r.misaligned_components):>10d}")
    
    print_section("Layer Pair Insights")
    
    for r in results:
        print(f"\n  {Colors.CYAN}{r.layer1} ↔ {r.layer2}{Colors.ENDC}")
        print(f"    {r.interpretation}")
        
        if r.misaligned_components:
            print_warning(f"    Misalignments:")
            for comp1, comp2, reason in r.misaligned_components[:3]:
                print_info(f"    {comp1} → {comp2}")
    
    return True


def example_ensemble_criticality():
    """
    Example 7: Ensemble Criticality Scoring
    
    Demonstrates:
    - Combining multiple algorithms for robust scoring
    - Algorithm agreement analysis (confidence)
    - Multi-perspective criticality assessment
    """
    print_header("Example 7: Ensemble Criticality Scoring")
    
    try:
        from src.analysis.relationship_analyzer import EnsembleCriticalityScorer, ComponentRole
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    print_success(f"Loaded {G.number_of_nodes()} nodes")
    
    print_section("Computing Ensemble Criticality")
    
    print("  Algorithms included:")
    print_info("Betweenness Centrality (25%) - routing importance")
    print_info("PageRank (20%) - influence importance")
    print_info("HITS Hub (10%) - publishing importance")
    print_info("HITS Authority (10%) - consuming importance")
    print_info("Articulation Points (20%) - structural importance")
    print_info("K-Core (10%) - connectivity importance")
    print_info("Closeness (5%) - accessibility importance")
    
    scorer = EnsembleCriticalityScorer(G)
    results = scorer.score_all()
    
    print_success(f"Scored {len(results)} components")
    
    # Distribution by level
    level_counts = {}
    for r in results.values():
        level_counts[r.ensemble_level] = level_counts.get(r.ensemble_level, 0) + 1
    
    print("\n  Ensemble Criticality Distribution:")
    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = level_counts.get(level, 0)
        bar = '█' * (count * 2)
        print(f"    {level:10s}: {count:3d} {bar}")
    
    print_section("Top 10 Critical Components (Ensemble)")
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1].ensemble_score)
    
    for i, (node_id, r) in enumerate(sorted_results[:10], 1):
        level_color = Colors.FAIL if r.ensemble_level == 'CRITICAL' else \
                     Colors.WARNING if r.ensemble_level == 'HIGH' else Colors.ENDC
        
        confidence_bar = '●' * int(r.confidence * 5) + '○' * (5 - int(r.confidence * 5))
        
        print(f"  {i:2d}. {level_color}{node_id:30s}{Colors.ENDC}")
        print(f"      Score: {r.ensemble_score:.4f} [{r.ensemble_level}] "
              f"Confidence: {confidence_bar}")
        
        # Show indicators
        indicators = []
        if r.is_structural_critical:
            indicators.append("🏗️ Structural")
        if r.is_flow_critical:
            indicators.append("🌊 Flow")
        if r.is_connectivity_critical:
            indicators.append("🔗 Connectivity")
        if r.is_articulation_point:
            indicators.append("⚠️ Articulation Point")
        
        if indicators:
            print(f"      {' | '.join(indicators)}")
        
        if r.primary_reasons:
            print(f"      Reason: {r.primary_reasons[0]}")
    
    print_section("Algorithm Agreement Analysis")
    
    high_confidence = [r for r in results.values() if r.confidence > 0.7]
    low_confidence = [r for r in results.values() if r.confidence < 0.3]
    
    print(f"  High confidence (>70%): {len(high_confidence)} components")
    print(f"  Low confidence (<30%): {len(low_confidence)} components")
    
    if low_confidence:
        print_warning("\n  Low-confidence assessments (algorithms disagree):")
        for r in sorted(low_confidence, key=lambda x: x.confidence)[:3]:
            print_info(f"  {r.node_id}: confidence {r.confidence:.2f}")
    
    return True


def example_full_pipeline():
    """
    Example 8: Full Analysis Pipeline
    
    Demonstrates:
    - Complete end-to-end analysis
    - All features combined
    - Report generation
    """
    print_header("Example 8: Full Analysis Pipeline")
    
    try:
        from src.analysis.relationship_analyzer import RelationshipAnalyzer
    except ImportError:
        print_warning("Relationship analyzer module not available")
        return False
    
    print_section("Loading Graph")
    data = generate_iot_smart_city_data()
    G = build_graph_from_dict(data)
    print_success(f"Loaded: {data['metadata']['name']}")
    print_success(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    print_section("Running Full Analysis")
    
    import time
    start = time.time()
    
    analyzer = RelationshipAnalyzer(G)
    result = analyzer.analyze(
        include_edge_analysis=True,
        include_hits=True,
        include_motifs=True,
        include_chains=True,
        include_layers=True,
        include_ensemble=True
    )
    
    elapsed = time.time() - start
    print_success(f"Analysis completed in {elapsed:.2f}s")
    
    print_section("Analysis Summary")
    
    print(f"\n  📊 {Colors.BOLD}Graph Summary{Colors.ENDC}")
    print(f"     Total Nodes: {result.total_nodes}")
    print(f"     Total Edges: {result.total_edges}")
    
    print(f"\n  🔗 {Colors.BOLD}Edge Analysis{Colors.ENDC}")
    print(f"     Analyzed: {len(result.edge_criticality)} edges")
    print(f"     Critical: {len(result.critical_edges)}")
    print(f"     Bridges: {len(result.bridge_edges)}")
    
    print(f"\n  🎯 {Colors.BOLD}HITS Analysis{Colors.ENDC}")
    print(f"     Top Hubs: {', '.join(result.top_hubs[:3])}")
    print(f"     Top Authorities: {', '.join(result.top_authorities[:3])}")
    
    print(f"\n  🔄 {Colors.BOLD}Motif Analysis{Colors.ENDC}")
    print(f"     Total Motifs: {len(result.motifs)}")
    for mtype, count in result.motif_summary.items():
        print(f"     {mtype}: {count}")
    
    print(f"\n  ⛓️ {Colors.BOLD}Dependency Analysis{Colors.ENDC}")
    print(f"     Deepest Chains: {', '.join(result.deepest_chains[:3])}")
    print(f"     Highest Fan-Out: {', '.join(result.highest_fan_out[:3])}")
    
    print(f"\n  📈 {Colors.BOLD}Ensemble Criticality{Colors.ENDC}")
    level_counts = {}
    for e in result.ensemble_criticality.values():
        level_counts[e.ensemble_level] = level_counts.get(e.ensemble_level, 0) + 1
    for level, count in level_counts.items():
        print(f"     {level}: {count}")
    
    print_section("Recommendations")
    
    if result.recommendations:
        for i, rec in enumerate(result.recommendations[:5], 1):
            priority_color = Colors.FAIL if rec['priority'] == 'CRITICAL' else Colors.WARNING
            print(f"\n  {i}. {priority_color}[{rec['priority']}]{Colors.ENDC} {rec['category']}")
            print(f"     Issue: {rec['issue']}")
            print(f"     Action: {rec['recommendation']}")
    else:
        print_success("No critical recommendations - system looks healthy!")
    
    print_section("Export Results")
    
    result_dict = analyzer.to_dict(result)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(result_dict, f, indent=2, default=str)
        print_success(f"Results saved to: {f.name}")
    
    return True


# ============================================================================
# Main Entry Point
# ============================================================================

def list_examples():
    """List all available examples"""
    examples = [
        ("basic", "Basic Criticality Analysis", example_basic_criticality),
        ("edges", "Edge-Centric Analysis", example_edge_analysis),
        ("hits", "HITS Role Analysis", example_hits_analysis),
        ("motifs", "Network Motif Detection", example_motif_detection),
        ("chains", "Dependency Chain Analysis", example_dependency_chains),
        ("layers", "Multi-Layer Correlation", example_layer_correlation),
        ("ensemble", "Ensemble Criticality Scoring", example_ensemble_criticality),
        ("full", "Full Analysis Pipeline", example_full_pipeline),
    ]
    
    print_header("Available Examples")
    print()
    
    for name, desc, _ in examples:
        print(f"  {Colors.CYAN}{name:12s}{Colors.ENDC} - {desc}")
    
    print(f"\n  Use: python example_graph_analyzer.py --example <name>")
    print(f"  Or:  python example_graph_analyzer.py  (runs all examples)")


def main():
    parser = argparse.ArgumentParser(
        description='Graph Analyzer Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python example_graph_analyzer.py                    # Run all examples
    python example_graph_analyzer.py --example basic    # Run specific example
    python example_graph_analyzer.py --list             # List available examples
    python example_graph_analyzer.py --generate         # Generate test data only
        """
    )
    
    parser.add_argument('--example', '-e', type=str,
                        choices=['basic', 'edges', 'hits', 'motifs', 'chains',
                                'layers', 'ensemble', 'full'],
                        help='Run specific example')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available examples')
    parser.add_argument('--generate', '-g', action='store_true',
                        help='Generate test data and save to file')
    parser.add_argument('--output', '-o', type=str, default='smart_city_system.json',
                        help='Output file for generated data')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    # List examples
    if args.list:
        list_examples()
        return 0
    
    # Generate test data
    if args.generate:
        print_header("Generating Test Data")
        data = generate_iot_smart_city_data()
        
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        
        print_success(f"Generated: {args.output}")
        print_info(f"Applications: {len(data['applications'])}")
        print_info(f"Brokers: {len(data['brokers'])}")
        print_info(f"Topics: {len(data['topics'])}")
        print_info(f"Nodes: {len(data['nodes'])}")
        return 0
    
    # Example mapping
    examples = {
        'basic': example_basic_criticality,
        'edges': example_edge_analysis,
        'hits': example_hits_analysis,
        'motifs': example_motif_detection,
        'chains': example_dependency_chains,
        'layers': example_layer_correlation,
        'ensemble': example_ensemble_criticality,
        'full': example_full_pipeline,
    }
    
    # Run specific example
    if args.example:
        example_func = examples.get(args.example)
        if example_func:
            try:
                success = example_func()
                return 0 if success else 1
            except Exception as e:
                print_critical(f"Example failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
    
    # Run all examples
    print_header("GRAPH ANALYZER EXAMPLES")
    print("\n  Running all examples to demonstrate analysis capabilities...\n")
    
    results = []
    for name, example_func in examples.items():
        try:
            success = example_func()
            results.append((name, success))
            print()  # Space between examples
        except Exception as e:
            print_critical(f"Example '{name}' failed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Example Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n  Passed: {passed}/{total}")
    
    for name, success in results:
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if success else f"{Colors.FAIL}✗{Colors.ENDC}"
        print(f"    {status} {name}")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())