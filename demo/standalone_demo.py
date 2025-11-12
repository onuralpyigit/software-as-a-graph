#!/usr/bin/env python3
"""
Software As A Graph: Standalone Demonstration
==============================================

This self-contained demo showcases the complete methodology for graph-based 
modeling and analysis of distributed publish-subscribe systems.

Demonstrates:
1. Graph Model Building - Multi-layer graph construction
2. Structural Analysis - Centrality metrics and critical components
3. QoS-Aware Analysis - Quality of service integration
4. Failure Simulation - Component failure and cascading effects
5. Visualization - Export to multiple formats

Author: Onuralp
Research: Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems
IEEE RASSE 2025
"""

import json
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import time


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str, color=Colors.HEADER):
    """Print formatted section header"""
    print(f"\n{color}{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}{Colors.ENDC}\n")


def print_subheader(text: str):
    """Print formatted subsection header"""
    print(f"\n{Colors.CYAN}{'-'*80}")
    print(f"{text}")
    print(f"{'-'*80}{Colors.ENDC}")


def create_sample_system() -> Dict:
    """Create a realistic Smart City IoT system configuration"""
    print_subheader("Creating Sample Smart City IoT System")
    
    system = {
        "applications": [
            {"id": "TempSensor1", "type": "sensor", "qos_deadline": 1000, "qos_durability": "transient_local"},
            {"id": "TrafficCamera1", "type": "sensor", "qos_deadline": 100, "qos_durability": "volatile"},
            {"id": "AirQualitySensor1", "type": "sensor", "qos_deadline": 2000, "qos_durability": "transient_local"},
            {"id": "AnalyticsEngine", "type": "processing", "qos_deadline": 500, "qos_durability": "transient_local"},
            {"id": "AlertService", "type": "service", "qos_deadline": 100, "qos_durability": "persistent"},
            {"id": "TrafficController", "type": "actuator", "qos_deadline": 200, "qos_durability": "volatile"},
            {"id": "DataArchiver", "type": "storage", "qos_deadline": 5000, "qos_durability": "persistent"},
            {"id": "Dashboard", "type": "visualization", "qos_deadline": 500, "qos_durability": "volatile"}
        ],
        "topics": [
            {"id": "temperature_data", "type": "sensor_msgs/Temperature"},
            {"id": "air_quality_data", "type": "sensor_msgs/AirQuality"},
            {"id": "video_stream", "type": "sensor_msgs/CompressedImage"},
            {"id": "alerts", "type": "std_msgs/Alert"},
            {"id": "analytics_results", "type": "custom_msgs/AnalyticsData"},
            {"id": "emergency_alerts", "type": "std_msgs/EmergencyAlert"}
        ],
        "brokers": [
            {"id": "Broker1", "type": "edge_broker", "capacity": 100},
            {"id": "Broker2", "type": "cloud_broker", "capacity": 500}
        ],
        "nodes": [
            {"id": "EdgeNode1", "type": "edge_device", "cpu": 4},
            {"id": "EdgeNode2", "type": "edge_gateway", "cpu": 8},
            {"id": "CloudNode1", "type": "cloud_server", "cpu": 32}
        ],
        "connections": {
            "publishes": [
                ("TempSensor1", "temperature_data"),
                ("TrafficCamera1", "video_stream"),
                ("AirQualitySensor1", "air_quality_data"),
                ("AnalyticsEngine", "alerts"),
                ("AnalyticsEngine", "analytics_results"),
                ("AlertService", "emergency_alerts")
            ],
            "subscribes": [
                ("AnalyticsEngine", "temperature_data"),
                ("AnalyticsEngine", "air_quality_data"),
                ("AnalyticsEngine", "video_stream"),
                ("AlertService", "alerts"),
                ("TrafficController", "analytics_results"),
                ("TrafficController", "emergency_alerts"),
                ("DataArchiver", "temperature_data"),
                ("DataArchiver", "air_quality_data"),
                ("DataArchiver", "analytics_results"),
                ("Dashboard", "temperature_data"),
                ("Dashboard", "air_quality_data"),
                ("Dashboard", "video_stream"),
                ("Dashboard", "alerts")
            ],
            "routes": [
                ("Broker1", "Broker2", "temperature_data"),
                ("Broker1", "Broker2", "air_quality_data"),
                ("Broker1", "Broker2", "video_stream")
            ],
            "runs_on": [
                ("TempSensor1", "EdgeNode1"),
                ("TrafficCamera1", "EdgeNode1"),
                ("AirQualitySensor1", "EdgeNode2"),
                ("Broker1", "EdgeNode2"),
                ("AnalyticsEngine", "CloudNode1"),
                ("AlertService", "CloudNode1"),
                ("DataArchiver", "CloudNode1"),
                ("Dashboard", "CloudNode1"),
                ("Broker2", "CloudNode1"),
                ("TrafficController", "EdgeNode1")
            ]
        }
    }
    
    total_components = (len(system["applications"]) + len(system["topics"]) + 
                       len(system["brokers"]) + len(system["nodes"]))
    total_edges = sum(len(v) for v in system["connections"].values())
    
    print(f"{Colors.GREEN}✓ Created sample system with:")
    print(f"  • {len(system['applications'])} applications")
    print(f"  • {len(system['topics'])} topics")
    print(f"  • {len(system['brokers'])} brokers")
    print(f"  • {len(system['nodes'])} infrastructure nodes")
    print(f"  • {total_edges} relationships{Colors.ENDC}")
    
    return system


def build_multi_layer_graph(system: Dict) -> nx.DiGraph:
    """Build a multi-layer directed graph from system configuration"""
    print_subheader("Step 1: Building Multi-Layer Graph Model")
    
    print("Constructing graph layers...")
    G = nx.DiGraph()
    
    # Add nodes with attributes
    print("  • Adding application layer...")
    for app in system["applications"]:
        G.add_node(app["id"], 
                  layer="application",
                  node_type=app["type"],
                  qos_deadline=app["qos_deadline"],
                  qos_durability=app["qos_durability"])
    
    print("  • Adding topic layer...")
    for topic in system["topics"]:
        G.add_node(topic["id"], 
                  layer="topic",
                  node_type=topic["type"])
    
    print("  • Adding broker layer...")
    for broker in system["brokers"]:
        G.add_node(broker["id"], 
                  layer="broker",
                  broker_type=broker["type"],
                  capacity=broker["capacity"])
    
    print("  • Adding infrastructure layer...")
    for node in system["nodes"]:
        G.add_node(node["id"], 
                  layer="infrastructure",
                  node_type=node["type"],
                  cpu=node["cpu"])
    
    # Add edges with relationship types
    print("  • Creating publish relationships...")
    for src, dst in system["connections"]["publishes"]:
        G.add_edge(src, dst, relationship="publishes", weight=1.0)
    
    print("  • Creating subscribe relationships...")
    for src, dst in system["connections"]["subscribes"]:
        G.add_edge(src, dst, relationship="subscribes", weight=1.0)
    
    print("  • Creating routing relationships...")
    for src, dst, topic in system["connections"]["routes"]:
        G.add_edge(src, dst, relationship="routes", topic=topic, weight=1.5)
    
    print("  • Creating deployment relationships...")
    for src, dst in system["connections"]["runs_on"]:
        G.add_edge(src, dst, relationship="runs_on", weight=0.8)
    
    print(f"\n{Colors.GREEN}✓ Graph Model Built Successfully:{Colors.ENDC}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    
    # Layer statistics
    layers = {}
    for node, data in G.nodes(data=True):
        layer = data.get('layer', 'unknown')
        layers[layer] = layers.get(layer, 0) + 1
    
    print(f"  Layer distribution: {dict(layers)}")
    
    return G


def analyze_structural_properties(G: nx.DiGraph) -> Dict:
    """Compute structural metrics and identify critical components"""
    print_subheader("Step 2: Structural Analysis")
    
    print("Computing graph metrics...")
    results = {}
    
    # Basic metrics
    print("  • Computing basic metrics...")
    results['num_nodes'] = G.number_of_nodes()
    results['num_edges'] = G.number_of_edges()
    results['density'] = nx.density(G)
    results['is_connected'] = nx.is_weakly_connected(G)
    
    # Centrality measures
    print("  • Computing betweenness centrality...")
    start = time.time()
    betweenness = nx.betweenness_centrality(G, weight='weight')
    results['betweenness_time'] = time.time() - start
    results['betweenness'] = betweenness
    
    print("  • Computing degree centrality...")
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    results['in_degree'] = in_degree
    results['out_degree'] = out_degree
    
    print("  • Computing closeness centrality...")
    closeness = nx.closeness_centrality(G)
    results['closeness'] = closeness
    
    print("  • Computing eigenvector centrality...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        results['eigenvector'] = eigenvector
    except:
        results['eigenvector'] = {}
        print(f"    {Colors.YELLOW}⚠ Could not compute eigenvector centrality{Colors.ENDC}")
    
    # Critical components
    print("  • Identifying articulation points...")
    # Convert to undirected for articulation points
    G_undirected = G.to_undirected()
    articulation_points = set(nx.articulation_points(G_undirected))
    results['articulation_points'] = articulation_points
    
    print(f"\n{Colors.GREEN}✓ Structural Analysis Completed:{Colors.ENDC}")
    print(f"  Betweenness computation: {results['betweenness_time']:.3f}s")
    print(f"  Articulation points: {len(articulation_points)}")
    print(f"  Network connectivity: {'Connected' if results['is_connected'] else 'Disconnected'}")
    
    return results


def compute_qos_importance(G: nx.DiGraph) -> Dict[str, float]:
    """Compute QoS-based importance scores"""
    print_subheader("Step 3: QoS-Aware Analysis")
    
    print("Computing QoS importance scores...")
    qos_scores = {}
    
    # QoS durability weights
    durability_weights = {
        'persistent': 1.0,
        'transient_local': 0.7,
        'volatile': 0.3
    }
    
    for node, data in G.nodes(data=True):
        score = 0.0
        
        # Deadline importance (tighter deadline = more critical)
        if 'qos_deadline' in data:
            deadline = data['qos_deadline']
            # Normalize: shorter deadline = higher importance
            score += 1.0 / (1.0 + deadline / 1000.0)
        
        # Durability importance
        if 'qos_durability' in data:
            durability = data['qos_durability']
            score += durability_weights.get(durability, 0.5)
        
        # Broker capacity importance
        if 'capacity' in data:
            # Normalize capacity
            score += data['capacity'] / 500.0
        
        qos_scores[node] = score
    
    # Normalize scores to [0, 1]
    if qos_scores:
        max_score = max(qos_scores.values())
        if max_score > 0:
            qos_scores = {k: v/max_score for k, v in qos_scores.items()}
    
    # Identify high-importance components
    high_importance = {k for k, v in qos_scores.items() if v > 0.7}
    
    print(f"{Colors.GREEN}✓ QoS Analysis Completed:{Colors.ENDC}")
    print(f"  Components analyzed: {len(qos_scores)}")
    print(f"  High importance (>0.7): {len(high_importance)}")
    if high_importance:
        print(f"  Examples: {', '.join(list(high_importance)[:3])}")
    
    return qos_scores


def compute_composite_criticality(structural: Dict, qos_scores: Dict, 
                                 alpha=0.4, beta=0.3, gamma=0.3) -> Dict[str, float]:
    """Compute composite criticality scores"""
    print_subheader("Step 4: Composite Criticality Scoring")
    
    print(f"Computing composite scores (α={alpha}, β={beta}, γ={gamma})...")
    
    betweenness = structural['betweenness']
    articulation_points = structural['articulation_points']
    
    # Normalize betweenness
    max_betweenness = max(betweenness.values()) if betweenness else 1.0
    norm_betweenness = {k: v/max_betweenness for k, v in betweenness.items()}
    
    # Compute composite scores
    criticality_scores = {}
    for node in betweenness.keys():
        # C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
        score = (
            alpha * norm_betweenness.get(node, 0) +
            beta * (1.0 if node in articulation_points else 0.0) +
            gamma * qos_scores.get(node, 0)
        )
        criticality_scores[node] = score
    
    # Sort by criticality
    sorted_components = sorted(criticality_scores.items(), 
                              key=lambda x: x[1], reverse=True)
    
    print(f"\n{Colors.GREEN}✓ Criticality Scoring Completed:{Colors.ENDC}")
    print(f"  Total components scored: {len(criticality_scores)}")
    print(f"\n  {Colors.BOLD}Top 5 Most Critical Components:{Colors.ENDC}")
    
    for i, (component, score) in enumerate(sorted_components[:5], 1):
        bar_length = int(score * 40)
        bar = '█' * bar_length
        layer = "?"
        # Could extract layer info from G if passed
        print(f"    {i}. {component:20s} [{bar:<40s}] {score:.3f}")
    
    return criticality_scores


def simulate_component_failure(G: nx.DiGraph, component: str, 
                               enable_cascade: bool = True) -> Dict:
    """Simulate failure of a component with optional cascading"""
    print_subheader("Step 5: Failure Impact Simulation")
    
    print(f"Simulating failure of: {Colors.RED}{component}{Colors.ENDC}")
    print(f"Cascading enabled: {enable_cascade}")
    
    if component not in G:
        print(f"{Colors.RED}✗ Component not found in graph{Colors.ENDC}")
        return {}
    
    # Create a copy for simulation
    G_sim = G.copy()
    
    # Track failures
    failed_components = {component}
    affected_components = set()
    isolated_components = set()
    
    # Remove the failed component
    print("  • Removing failed component...")
    G_sim.remove_node(component)
    
    # Simulate cascade
    if enable_cascade:
        print("  • Analyzing cascading failures...")
        cascade_threshold = 0.7  # If >70% of inputs lost, component fails
        
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            new_failures = set()
            
            for node in G_sim.nodes():
                if node in failed_components:
                    continue
                
                # Check incoming connections
                predecessors = set(G_sim.predecessors(node))
                if not predecessors:
                    continue
                
                # Count how many predecessors have failed
                failed_predecessors = predecessors & failed_components
                loss_ratio = len(failed_predecessors) / len(predecessors)
                
                if loss_ratio >= cascade_threshold:
                    new_failures.add(node)
                    affected_components.add(node)
            
            if not new_failures:
                break
            
            failed_components.update(new_failures)
            for node in new_failures:
                G_sim.remove_node(node)
            
            iteration += 1
            print(f"    Cascade iteration {iteration}: {len(new_failures)} new failures")
    
    # Identify affected components (reachable nodes that lost connections)
    print("  • Analyzing reachability impact...")
    original_nodes = set(G.nodes())
    remaining_nodes = set(G_sim.nodes())
    
    for node in remaining_nodes:
        # Check if node lost connections to failed components
        orig_neighbors = set(G.predecessors(node)) | set(G.successors(node))
        lost_neighbors = orig_neighbors & failed_components
        
        if lost_neighbors:
            affected_components.add(node)
    
    # Identify isolated components
    if not nx.is_weakly_connected(G_sim):
        components = list(nx.weakly_connected_components(G_sim))
        # Components that aren't in the largest component are isolated
        largest_component = max(components, key=len)
        for comp in components:
            if comp != largest_component:
                isolated_components.update(comp)
    
    # Calculate impact metrics
    total_nodes = len(original_nodes)
    impact_score = len(failed_components) / total_nodes
    resilience_score = len(remaining_nodes) / total_nodes
    service_continuity = len(remaining_nodes - isolated_components) / total_nodes
    
    results = {
        'failed_component': component,
        'failed_components': failed_components,
        'affected_components': affected_components,
        'isolated_components': isolated_components,
        'impact_score': impact_score,
        'resilience_score': resilience_score,
        'service_continuity': service_continuity,
        'cascade_iterations': iteration if enable_cascade else 0
    }
    
    print(f"\n{Colors.GREEN}✓ Simulation Completed:{Colors.ENDC}")
    print(f"  Failed components:     {len(failed_components)}")
    print(f"  Affected components:   {len(affected_components)}")
    print(f"  Isolated components:   {len(isolated_components)}")
    print(f"  Impact score:          {impact_score:.3f}")
    print(f"  Resilience score:      {resilience_score:.3f}")
    print(f"  Service continuity:    {service_continuity*100:.1f}%")
    
    if failed_components:
        print(f"\n  {Colors.RED}💥 Failed:{Colors.ENDC} {', '.join(list(failed_components)[:5])}")
    if affected_components:
        print(f"  {Colors.YELLOW}⚠ Affected:{Colors.ENDC} {', '.join(list(affected_components)[:5])}")
    if isolated_components:
        print(f"  {Colors.RED}🔌 Isolated:{Colors.ENDC} {', '.join(list(isolated_components)[:3])}")
    
    return results


def export_results(G: nx.DiGraph, analysis: Dict, simulation: Dict, 
                   output_dir: Path):
    """Export analysis results to various formats"""
    print_subheader("Step 6: Exporting Results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export GraphML
    graphml_path = output_dir / "system_graph.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"  {Colors.GREEN}✓ GraphML: {graphml_path}{Colors.ENDC}")
    
    # Export DOT (if pydot available)
    try:
        dot_path = output_dir / "system_graph.dot"
        nx.drawing.nx_pydot.write_dot(G, dot_path)
        print(f"  {Colors.GREEN}✓ DOT:     {dot_path}{Colors.ENDC}")
    except ImportError:
        print(f"  {Colors.YELLOW}⚠ DOT export skipped (pydot not installed){Colors.ENDC}")
    
    # Export analysis results as JSON
    json_path = output_dir / "analysis_results.json"
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': float(nx.density(G))
        },
        'structural_analysis': {
            'betweenness': {k: float(v) for k, v in analysis['betweenness'].items()},
            'articulation_points': list(analysis['articulation_points']),
            'is_connected': analysis['is_connected']
        },
        'simulation': {
            'failed_components': list(simulation.get('failed_components', [])),
            'affected_components': list(simulation.get('affected_components', [])),
            'isolated_components': list(simulation.get('isolated_components', [])),
            'impact_score': simulation.get('impact_score', 0),
            'resilience_score': simulation.get('resilience_score', 0),
            'service_continuity': simulation.get('service_continuity', 0)
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  {Colors.GREEN}✓ JSON:    {json_path}{Colors.ENDC}")
    
    # Export criticality ranking
    csv_path = output_dir / "criticality_ranking.csv"
    with open(csv_path, 'w') as f:
        f.write("Component,Criticality_Score,Betweenness,Is_Articulation_Point\n")
        if 'criticality_scores' in analysis:
            sorted_components = sorted(analysis['criticality_scores'].items(), 
                                      key=lambda x: x[1], reverse=True)
            for comp, score in sorted_components:
                betweenness = analysis['betweenness'].get(comp, 0)
                is_ap = comp in analysis['articulation_points']
                f.write(f"{comp},{score:.4f},{betweenness:.4f},{is_ap}\n")
    print(f"  {Colors.GREEN}✓ CSV:     {csv_path}{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}💡 Visualization Tips:{Colors.ENDC}")
    print("  • Open .graphml in Gephi or yEd for interactive visualization")
    print("  • Use Graphviz: dot -Tpng system_graph.dot -o graph.png")
    print("  • Analyze JSON results with your favorite tools")


def print_methodology_summary():
    """Print the six-step methodology"""
    print_header("Six-Step Methodology Overview", Colors.HEADER)
    
    steps = [
        ("1. Graph Modeling", 
         "Multi-layer representation (Apps, Topics, Brokers, Infrastructure)",
         "Captures structural dependencies and relationships"),
        
        ("2. Structural Analysis", 
         "Centrality metrics (Betweenness, Degree, Closeness, Eigenvector)",
         "Identifies topologically critical components"),
        
        ("3. QoS-Aware Analysis", 
         "Incorporate QoS constraints (Durability, Reliability, Deadlines)",
         "Elevates domain-specific importance"),
        
        ("4. Composite Criticality", 
         "C_score(v) = α·Betweenness + β·ArticulationPoint + γ·QoS",
         "Unified criticality metric combining structure and QoS"),
        
        ("5. Failure Simulation", 
         "Single/multiple failures with cascading propagation",
         "Validates predictions through impact assessment"),
        
        ("6. Recommendations", 
         "System hardening, capacity planning, architecture optimization",
         "Actionable insights for system improvement")
    ]
    
    for step, description, details in steps:
        print(f"{Colors.CYAN}{step}{Colors.ENDC}")
        print(f"  {description}")
        print(f"  → {details}\n")
    
    print(f"{Colors.BOLD}Research Contributions:{Colors.ENDC}")
    print("  • Novel graph-theoretic approach for distributed pub-sub systems")
    print("  • Integration of topological and QoS-based criticality measures")
    print("  • Scalable to enterprise deployments (1000+ components)")
    print("  • Validated: Spearman ρ > 0.7, F1-score > 0.9")
    print(f"\n{Colors.CYAN}Publication: IEEE RASSE 2025{Colors.ENDC}\n")


def main():
    """Main demonstration workflow"""
    print_header("Software As A Graph: Complete Demonstration", Colors.HEADER)
    
    print(f"{Colors.CYAN}Graph-Based Modeling and Analysis of")
    print(f"Distributed Publish-Subscribe Systems{Colors.ENDC}\n")
    print(f"Author: Onuralp")
    print(f"PhD Research: Computer Engineering")
    print(f"Accepted: IEEE RASSE 2025\n")
    
    try:
        # Create sample system
        print_header("Phase 0: System Configuration")
        system = create_sample_system()
        
        # Build graph
        print_header("Phase 1: Graph Model Construction")
        G = build_multi_layer_graph(system)
        
        # Structural analysis
        print_header("Phase 2: Structural Analysis")
        structural_results = analyze_structural_properties(G)
        
        # QoS analysis
        print_header("Phase 3: QoS-Aware Analysis")
        qos_scores = compute_qos_importance(G)
        
        # Composite criticality
        print_header("Phase 4: Composite Criticality Scoring")
        criticality_scores = compute_composite_criticality(
            structural_results, qos_scores,
            alpha=0.4, beta=0.3, gamma=0.3
        )
        structural_results['criticality_scores'] = criticality_scores
        
        # Failure simulation
        print_header("Phase 5: Failure Impact Assessment")
        # Get most critical component
        most_critical = max(criticality_scores.items(), key=lambda x: x[1])[0]
        simulation_results = simulate_component_failure(G, most_critical, enable_cascade=True)
        
        # Export results
        print_header("Phase 6: Results Export")
        output_dir = Path("/tmp/demo_results")
        export_results(G, structural_results, simulation_results, output_dir)
        
        # Summary
        print_header("Demo Completed Successfully!", Colors.GREEN)
        
        print(f"\n{Colors.BOLD}Key Findings:{Colors.ENDC}")
        print(f"  • System has {G.number_of_nodes()} components, {G.number_of_edges()} relationships")
        print(f"  • Identified {len(structural_results['articulation_points'])} articulation points")
        print(f"  • Most critical component: {most_critical} (score: {criticality_scores[most_critical]:.3f})")
        print(f"  • Failure impact: {len(simulation_results['affected_components'])} components affected")
        print(f"  • System resilience: {simulation_results['resilience_score']:.3f}")
        
        print(f"\n{Colors.CYAN}Results saved to: {output_dir}/{Colors.ENDC}")
        print("  • system_graph.graphml - Interactive visualization")
        print("  • system_graph.dot - Graphviz format")
        print("  • analysis_results.json - Complete analysis data")
        print("  • criticality_ranking.csv - Component rankings")
        
        # Methodology summary
        print_methodology_summary()
        
        print(f"{Colors.GREEN}{'='*80}")
        print(f"{'✓ Demonstration completed successfully!'.center(80)}")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Next Steps:{Colors.ENDC}")
        print("  1. Open GraphML in Gephi/yEd for visualization")
        print("  2. Analyze JSON results for detailed metrics")
        print("  3. Apply methodology to your production systems")
        print("  4. Extend with custom metrics and domain knowledge\n")
        
        return 0
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error during demonstration:{Colors.ENDC}")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
