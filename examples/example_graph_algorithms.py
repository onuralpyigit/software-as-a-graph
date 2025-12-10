#!/usr/bin/env python3
"""
Graph Algorithms Demo for Pub-Sub System Analysis
==================================================

This demo showcases various graph algorithms and their application
to analyzing distributed publish-subscribe systems.

Run: python demo_graph_algorithms.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from src.analysis.graph_algorithms import (
    CentralityAnalyzer,
    StructuralAnalyzer,
    CommunityDetector,
    PathFlowAnalyzer,
    SimilarityCouplingAnalyzer,
    RobustnessAnalyzer,
    LayerAwareAnalyzer,
    ComprehensiveGraphAnalyzer,
    get_algorithm_recommendations,
    print_recommendations
)


# ============================================================================
# Color Helpers
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*70}{Colors.END}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}▶ {text}{Colors.END}")


def print_result(label: str, value):
    print(f"  {Colors.GREEN}•{Colors.END} {label}: {Colors.YELLOW}{value}{Colors.END}")


# ============================================================================
# Generate Sample Graph
# ============================================================================

def create_sample_pubsub_graph() -> nx.DiGraph:
    """Create a realistic pub-sub graph for demonstration"""
    G = nx.DiGraph()
    
    # Infrastructure Nodes
    nodes = [
        ('N1', {'type': 'Node', 'name': 'EdgeServer1', 'location': 'edge'}),
        ('N2', {'type': 'Node', 'name': 'EdgeServer2', 'location': 'edge'}),
        ('N3', {'type': 'Node', 'name': 'CloudServer1', 'location': 'cloud'}),
        ('N4', {'type': 'Node', 'name': 'CloudServer2', 'location': 'cloud'}),
    ]
    
    # Brokers
    brokers = [
        ('B1', {'type': 'Broker', 'name': 'EdgeBroker', 'node': 'N1'}),
        ('B2', {'type': 'Broker', 'name': 'CloudBroker1', 'node': 'N3'}),
        ('B3', {'type': 'Broker', 'name': 'CloudBroker2', 'node': 'N4'}),
    ]
    
    # Topics
    topics = [
        ('T1', {'type': 'Topic', 'name': 'sensor/temperature', 'broker': 'B1'}),
        ('T2', {'type': 'Topic', 'name': 'sensor/humidity', 'broker': 'B1'}),
        ('T3', {'type': 'Topic', 'name': 'sensor/pressure', 'broker': 'B1'}),
        ('T4', {'type': 'Topic', 'name': 'aggregated/weather', 'broker': 'B2'}),
        ('T5', {'type': 'Topic', 'name': 'alerts/critical', 'broker': 'B2'}),
        ('T6', {'type': 'Topic', 'name': 'control/commands', 'broker': 'B2'}),
        ('T7', {'type': 'Topic', 'name': 'analytics/results', 'broker': 'B3'}),
        ('T8', {'type': 'Topic', 'name': 'logs/system', 'broker': 'B3'}),
    ]
    
    # Applications
    applications = [
        ('A1', {'type': 'Application', 'name': 'TempSensor', 'role': 'publisher'}),
        ('A2', {'type': 'Application', 'name': 'HumiditySensor', 'role': 'publisher'}),
        ('A3', {'type': 'Application', 'name': 'PressureSensor', 'role': 'publisher'}),
        ('A4', {'type': 'Application', 'name': 'DataAggregator', 'role': 'processor'}),
        ('A5', {'type': 'Application', 'name': 'AlertEngine', 'role': 'processor'}),
        ('A6', {'type': 'Application', 'name': 'Analytics', 'role': 'processor'}),
        ('A7', {'type': 'Application', 'name': 'Dashboard', 'role': 'subscriber'}),
        ('A8', {'type': 'Application', 'name': 'MobileApp', 'role': 'subscriber'}),
        ('A9', {'type': 'Application', 'name': 'ControlSystem', 'role': 'controller'}),
        ('A10', {'type': 'Application', 'name': 'Logger', 'role': 'subscriber'}),
    ]
    
    # Add all nodes
    for node_id, attrs in nodes + brokers + topics + applications:
        G.add_node(node_id, **attrs)
    
    # Add edges
    
    # Publishes (Application -> Topic)
    publishes = [
        ('A1', 'T1'), ('A2', 'T2'), ('A3', 'T3'),  # Sensors publish raw data
        ('A4', 'T4'), ('A4', 'T8'),  # Aggregator publishes aggregated + logs
        ('A5', 'T5'), ('A5', 'T8'),  # AlertEngine publishes alerts + logs
        ('A6', 'T7'), ('A6', 'T8'),  # Analytics publishes results + logs
        ('A9', 'T6'), ('A9', 'T8'),  # Control publishes commands + logs
    ]
    
    # Subscribes (Topic -> Application)
    subscribes = [
        ('T1', 'A4'), ('T2', 'A4'), ('T3', 'A4'),  # Aggregator subscribes to sensors
        ('T1', 'A5'), ('T4', 'A5'),  # AlertEngine monitors temp and aggregated
        ('T4', 'A6'),  # Analytics processes aggregated data
        ('T4', 'A7'), ('T5', 'A7'), ('T7', 'A7'),  # Dashboard shows everything
        ('T5', 'A8'),  # Mobile gets alerts
        ('T6', 'A1'), ('T6', 'A2'), ('T6', 'A3'),  # Sensors receive commands
        ('T8', 'A10'),  # Logger receives all logs
    ]
    
    for src, dst in publishes:
        G.add_edge(src, dst, type='PUBLISHES')
    
    for src, dst in subscribes:
        G.add_edge(src, dst, type='SUBSCRIBES')
    
    # Topics hosted on brokers
    for topic_id, attrs in topics:
        G.add_edge(topic_id, attrs['broker'], type='HOSTS_ON')
    
    # Apps/Brokers run on nodes
    G.add_edge('B1', 'N1', type='RUNS_ON')
    G.add_edge('B2', 'N3', type='RUNS_ON')
    G.add_edge('B3', 'N4', type='RUNS_ON')
    
    G.add_edge('A1', 'N1', type='RUNS_ON')
    G.add_edge('A2', 'N1', type='RUNS_ON')
    G.add_edge('A3', 'N2', type='RUNS_ON')
    G.add_edge('A4', 'N3', type='RUNS_ON')
    G.add_edge('A5', 'N3', type='RUNS_ON')
    G.add_edge('A6', 'N4', type='RUNS_ON')
    G.add_edge('A7', 'N4', type='RUNS_ON')
    G.add_edge('A8', 'N2', type='RUNS_ON')
    G.add_edge('A9', 'N3', type='RUNS_ON')
    G.add_edge('A10', 'N4', type='RUNS_ON')
    
    # Broker routes
    G.add_edge('B1', 'B2', type='ROUTES_TO')
    G.add_edge('B2', 'B3', type='ROUTES_TO')
    G.add_edge('B2', 'B1', type='ROUTES_TO')
    
    return G


# ============================================================================
# Demo Functions
# ============================================================================

def demo_centrality_algorithms(G: nx.DiGraph):
    """Demonstrate centrality algorithms"""
    print_header("CENTRALITY ALGORITHMS")
    
    analyzer = CentralityAnalyzer(G)
    
    # Betweenness Centrality
    print_section("Betweenness Centrality")
    print("  Purpose: Find nodes that lie on shortest paths between others")
    print("  Pub-Sub Use: Identify critical message routing points")
    
    bc = analyzer.betweenness_centrality()
    print(f"\n  Top 5 nodes by betweenness:")
    for node, score in bc.top_k[:5]:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"    {node} ({node_type}): {score:.4f}")
    
    # PageRank
    print_section("PageRank")
    print("  Purpose: Importance based on incoming links from important nodes")
    print("  Pub-Sub Use: Find influential topics receiving from important publishers")
    
    pr = analyzer.pagerank()
    print(f"\n  Top 5 nodes by PageRank:")
    for node, score in pr.top_k[:5]:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"    {node} ({node_type}): {score:.4f}")
    
    # HITS - Particularly relevant for pub-sub!
    print_section("HITS Algorithm (Hubs & Authorities)")
    print("  Purpose: Find hubs (many outlinks) and authorities (many valuable inlinks)")
    print("  Pub-Sub Use: HIGHLY RELEVANT - Publishers are hubs, Topics are authorities!")
    
    hubs, authorities = analyzer.hits()
    
    print(f"\n  Top 5 HUBS (key publishers/data sources):")
    for node, score in hubs.top_k[:5]:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"    {node} ({node_type}): {score:.4f}")
    
    print(f"\n  Top 5 AUTHORITIES (key topics/subscribers):")
    for node, score in authorities.top_k[:5]:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"    {node} ({node_type}): {score:.4f}")
    
    # Degree Analysis
    print_section("Degree Centrality Analysis")
    print("  In-Degree: How many sources feed this node (subscriber behavior)")
    print("  Out-Degree: How many targets this node feeds (publisher behavior)")
    
    degrees = analyzer.degree_centrality_analysis()
    
    print(f"\n  Top 3 by In-Degree (major consumers):")
    for node, score in degrees['in_degree'].top_k[:3]:
        print(f"    {node}: {score:.4f}")
    
    print(f"\n  Top 3 by Out-Degree (major producers):")
    for node, score in degrees['out_degree'].top_k[:3]:
        print(f"    {node}: {score:.4f}")


def demo_structural_algorithms(G: nx.DiGraph):
    """Demonstrate structural algorithms"""
    print_header("STRUCTURAL ANALYSIS ALGORITHMS")
    
    analyzer = StructuralAnalyzer(G)
    
    # Articulation Points
    print_section("Articulation Points (Cut Vertices)")
    print("  Purpose: Nodes whose removal disconnects the graph")
    print("  Pub-Sub Use: Single points of failure in message routing")
    
    aps = analyzer.find_articulation_points()
    print(f"\n  Found {aps['count']} articulation points:")
    for ap, info in list(aps['analysis'].items())[:5]:
        print(f"    {ap} ({info['type']}): removal creates {info['components_after_removal']} components")
    
    # Bridges
    print_section("Bridges (Cut Edges)")
    print("  Purpose: Edges whose removal disconnects the graph")
    print("  Pub-Sub Use: Critical connections in message flow")
    
    bridges = analyzer.find_bridges()
    print(f"\n  Found {bridges['count']} bridges:")
    for bridge in bridges['analysis'][:5]:
        print(f"    {bridge['from']} -> {bridge['to']} ({bridge['type']})")
    
    # K-Core
    print_section("K-Core Decomposition")
    print("  Purpose: Find hierarchical dense subgraphs")
    print("  Pub-Sub Use: Identify core vs periphery architecture")
    
    kcore = analyzer.k_core_decomposition()
    print(f"\n  Maximum k-core: {kcore['max_k']}")
    print(f"  Innermost core ({kcore['max_k']}-core) contains: {kcore['innermost_core']}")
    
    # Strongly Connected Components
    print_section("Strongly Connected Components")
    print("  Purpose: Find cyclic dependencies (A can reach B and B can reach A)")
    print("  Pub-Sub Use: Detect circular message flows - potential infinite loops!")
    
    sccs = analyzer.strongly_connected_components()
    print(f"\n  Total SCCs: {sccs['count']}")
    print(f"  Non-trivial SCCs (size > 1): {sccs['non_trivial_count']}")
    if sccs['non_trivial_count'] > 0:
        print(f"  ⚠️  Warning: Cyclic dependencies detected!")
    
    # Cycles
    print_section("Cycle Detection")
    print("  Purpose: Find explicit cycles in message flow")
    
    cycles = analyzer.cycle_detection()
    print(f"\n  Cycles found: {cycles['count']}")
    if cycles['cycles']:
        print(f"  Shortest cycle: {' -> '.join(cycles['shortest_cycle'])}")


def demo_community_detection(G: nx.DiGraph):
    """Demonstrate community detection"""
    print_header("COMMUNITY DETECTION ALGORITHMS")
    
    detector = CommunityDetector(G)
    
    # Louvain
    print_section("Louvain Algorithm")
    print("  Purpose: Fast modularity-based community detection")
    print("  Pub-Sub Use: Identify logical subsystems, microservice boundaries")
    
    louvain = detector.louvain_communities()
    print(f"\n  Communities found: {louvain.statistics['num_communities']}")
    print(f"  Modularity: {louvain.modularity:.4f}")
    
    # Analyze composition
    composition = detector.analyze_community_composition(louvain)
    print(f"\n  Community Composition:")
    for comm in composition['communities'][:3]:
        print(f"    Community {comm['community_id']}: {comm['size']} members")
        print(f"      Types: {comm['types']}")
    
    # Greedy Modularity
    print_section("Greedy Modularity Communities")
    greedy = detector.greedy_modularity()
    print(f"\n  Communities: {greedy.statistics['num_communities']}")
    print(f"  Modularity: {greedy.modularity:.4f}")


def demo_path_flow_analysis(G: nx.DiGraph):
    """Demonstrate path and flow analysis"""
    print_header("PATH & FLOW ANALYSIS")
    
    analyzer = PathFlowAnalyzer(G)
    
    # Critical Paths
    print_section("Critical Path Analysis")
    print("  Purpose: Find longest dependency chains")
    print("  Pub-Sub Use: Determine end-to-end latency paths")
    
    critical = analyzer.critical_path_analysis()
    print(f"\n  Maximum path length: {critical['max_length']}")
    if critical['critical_paths']:
        print(f"  Longest path: {' -> '.join(critical['critical_paths'][0])}")
    
    # Bottlenecks
    print_section("Bottleneck Detection")
    print("  Purpose: Find nodes handling disproportionate traffic")
    print("  Pub-Sub Use: Capacity planning, scaling decisions")
    
    bottlenecks = analyzer.find_bottlenecks()
    print(f"\n  Top 5 potential bottlenecks:")
    for node, score in bottlenecks['top_bottlenecks'][:5]:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"    {node} ({node_type}): {score:.4f}")
    
    # Message Flow
    print_section("Message Flow Analysis")
    print("  Purpose: Analyze publisher -> topic -> subscriber patterns")
    
    flow = analyzer.message_flow_analysis()
    print(f"\n  Topics analyzed: {flow['total_topics_analyzed']}")
    print(f"  Max fanout: {flow['max_fanout']} subscribers")
    print(f"  Max fanin: {flow['max_fanin']} publishers")
    
    print(f"\n  Top flow chains:")
    for chain in flow['flow_chains'][:3]:
        print(f"    Topic {chain['topic']}: {chain['fanin']} publishers -> {chain['fanout']} subscribers")
    
    # Dependency Depth
    print_section("Dependency Depth Analysis")
    print("  Purpose: How far can failures cascade?")
    
    depth = analyzer.dependency_depth_analysis()
    print(f"\n  Maximum dependency depth: {depth['max_depth']}")
    print(f"  Deepest chains:")
    for node, d in depth['deepest_chains'][:5]:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"    {node} ({node_type}): depth {d}")


def demo_robustness_analysis(G: nx.DiGraph):
    """Demonstrate robustness analysis"""
    print_header("ROBUSTNESS ANALYSIS")
    
    analyzer = RobustnessAnalyzer(G)
    
    print_section("Attack Simulation Comparison")
    print("  Comparing random failures vs targeted attacks")
    print("  Pub-Sub Use: Understand system resilience, plan redundancy")
    
    comparison = analyzer.compare_robustness()
    
    print(f"\n  Results:")
    for attack_type, metrics in comparison['comparison'].items():
        print(f"    {attack_type}:")
        print(f"      Critical threshold: {metrics['critical_threshold']*100:.1f}% nodes removed")
        print(f"      AUC Robustness: {metrics['auc_robustness']:.4f}")
    
    print(f"\n  Vulnerability to targeted attacks: {comparison['vulnerability_to_targeted_attacks']*100:.1f}%")
    
    if comparison['vulnerability_to_targeted_attacks'] > 0.3:
        print(f"  ⚠️  High vulnerability! System has critical single points of failure.")
    else:
        print(f"  ✓ Relatively robust to targeted attacks.")


def demo_layer_aware_analysis(G: nx.DiGraph):
    """Demonstrate layer-aware analysis"""
    print_header("LAYER-AWARE ANALYSIS")
    
    analyzer = LayerAwareAnalyzer(G)
    
    print_section("Cross-Layer Dependencies")
    print("  Purpose: Understand how system layers interact")
    
    cross = analyzer.cross_layer_dependencies()
    print(f"\n  Layer sizes:")
    for layer, size in cross['layer_sizes'].items():
        print(f"    {layer}: {size} nodes")
    
    print(f"\n  Cross-layer edges:")
    for (src_type, dst_type), count in sorted(cross['cross_layer_edges'].items()):
        print(f"    {src_type} -> {dst_type}: {count} edges")
    
    # Layer Impact
    print_section("Layer Impact Analysis")
    print("  Purpose: Which layer is most critical to connectivity?")
    
    impact = analyzer.layer_impact_analysis()
    print(f"\n  Layer impacts:")
    for layer, metrics in impact['layer_impacts'].items():
        print(f"    {layer}: impact score {metrics['impact_score']:.4f}")
    
    print(f"\n  Most critical layer: {impact['most_critical_layer']}")


def demo_coupling_analysis(G: nx.DiGraph):
    """Demonstrate coupling analysis"""
    print_header("COUPLING & SIMILARITY ANALYSIS")
    
    analyzer = SimilarityCouplingAnalyzer(G)
    
    print_section("Coupling Analysis (Martin Metrics)")
    print("  Afferent Coupling (Ca): incoming dependencies")
    print("  Efferent Coupling (Ce): outgoing dependencies")
    print("  Instability = Ce/(Ca+Ce): high = depends on others more than others depend on it")
    
    coupling = analyzer.coupling_analysis()
    
    print(f"\n  Highly coupled components:")
    for node, metrics in coupling['highly_coupled'][:5]:
        print(f"    {node}: Ca={metrics['afferent_coupling']}, Ce={metrics['efferent_coupling']}, "
              f"I={metrics['instability']:.2f}")
    
    print(f"\n  Average instability: {coupling['average_instability']:.4f}")
    
    # Similar components
    print_section("Similar Components")
    print("  Purpose: Find components with similar connectivity (potential redundancy)")
    
    similar = analyzer.find_similar_components(threshold=0.3)
    if similar:
        print(f"\n  Similar pairs (Jaccard similarity):")
        for n1, n2, sim in similar[:5]:
            print(f"    {n1} <-> {n2}: {sim:.4f}")
    else:
        print("\n  No highly similar components found (threshold=0.3)")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print(f"""
{Colors.HEADER}{'='*70}
  GRAPH ALGORITHMS FOR PUB-SUB SYSTEM ANALYSIS
  Comprehensive Demonstration
{'='*70}{Colors.END}

This demo showcases various graph algorithms and their application
to analyzing distributed publish-subscribe systems.

The algorithms help identify:
  • Critical components (centrality)
  • Single points of failure (structural)
  • Logical subsystems (community detection)
  • Message flow patterns (path analysis)
  • Component relationships (coupling)
  • System resilience (robustness)
""")
    
    # Create sample graph
    print(f"{Colors.CYAN}Creating sample pub-sub graph...{Colors.END}")
    G = create_sample_pubsub_graph()
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Edges: {len(G.edges())}")
    
    # Node type distribution
    types = defaultdict(int)
    for _, data in G.nodes(data=True):
        types[data.get('type', 'Unknown')] += 1
    print(f"  Types: {dict(types)}")
    
    # Run demos
    demo_centrality_algorithms(G)
    demo_structural_algorithms(G)
    demo_community_detection(G)
    demo_path_flow_analysis(G)
    demo_robustness_analysis(G)
    demo_layer_aware_analysis(G)
    demo_coupling_analysis(G)
    
    # Algorithm recommendations
    print_recommendations()
    
    # Comprehensive analysis
    print_header("COMPREHENSIVE ANALYSIS")
    
    comprehensive = ComprehensiveGraphAnalyzer(G)
    results = comprehensive.run_comprehensive_analysis()
    report = comprehensive.generate_summary_report(results)
    print(report)
    
    # Save results
    output_path = Path('demo_output/graph_algorithms_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-serializable format
    def make_serializable(obj):
        if hasattr(obj, '__dict__'):
            return {k: make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, float):
            if obj == float('inf'):
                return "inf"
            return obj
        return obj
    
    serializable = make_serializable(results)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\n{Colors.GREEN}✓ Results saved to: {output_path}{Colors.END}")
    
    print(f"""
{Colors.HEADER}{'='*70}
  DEMO COMPLETE
{'='*70}{Colors.END}

Key Takeaways for Pub-Sub Analysis:

1. {Colors.YELLOW}HITS Algorithm{Colors.END} is particularly valuable for pub-sub because:
   - Publishers are natural "hubs" (many outgoing edges)
   - Topics are natural "authorities" (receive from many sources)
   
2. {Colors.YELLOW}Articulation Points{Colors.END} identify single points of failure
   that MUST have redundancy for reliable messaging.
   
3. {Colors.YELLOW}Community Detection{Colors.END} reveals natural subsystems
   useful for microservice boundary identification.
   
4. {Colors.YELLOW}Path Analysis{Colors.END} helps understand end-to-end latency
   and cascade failure risks.
   
5. {Colors.YELLOW}Robustness Analysis{Colors.END} shows how the system degrades
   under different failure scenarios.
""")


if __name__ == '__main__':
    main()
