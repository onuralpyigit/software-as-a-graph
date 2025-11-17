#!/usr/bin/env python3
"""
Graph Analysis Script for Pub-Sub Systems

Comprehensive analysis including:
- Critical nodes (articulation points, high betweenness)
- Critical edges (bridges, bottleneck links)
- Multi-layer dependency analysis
- QoS-aware criticality scoring
- Failure simulation
- Support for both JSON and Neo4j input

Usage:
    # From JSON file
    python analyze_graph.py --input system.json
    
    # From Neo4j database
    python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
        --user neo4j --password password
    
    # With edge criticality analysis
    python analyze_graph.py --input system.json --analyze-edges
    
    # Export to multiple formats
    python analyze_graph.py --input system.json \\
        --export-json --export-csv --export-html
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# NetworkX for graph analysis
try:
    import networkx as nx
except ImportError:
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_graph_from_json(filepath: str) -> Tuple[Dict, nx.DiGraph]:
    """
    Load graph from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Tuple of (graph_data dict, NetworkX DiGraph)
    """
    logger.info(f"Loading graph from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Build NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            
        for app in graph_data.get('applications', []):
            G.add_node(app['id'], **{k: v for k, v in app.items() if k != 'id'})
            
        for topic in graph_data.get('topics', []):
            G.add_node(topic['id'], **{k: v for k, v in topic.items() if k != 'id'})
            
        for broker in graph_data.get('brokers', []):
            G.add_node(broker['id'], **{k: v for k, v in broker.items() if k != 'id'})
        
        # Add edges
        relationships = graph_data.get('relationships', {})
        
        for rel in relationships.get('runs_on', []):
            G.add_edge(rel['from'], rel['to'], type='RUNS_ON')
            
        for rel in relationships.get('publishes_to', []):
            G.add_edge(rel['from'], rel['to'], type='PUBLISHES_TO')
            
        for rel in relationships.get('subscribes_to', []):
            G.add_edge(rel['from'], rel['to'], type='SUBSCRIBES_TO')
            
        for rel in relationships.get('routes', []):
            G.add_edge(rel['from'], rel['to'], type='ROUTES')
        
        logger.info(f"✓ Loaded graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        return graph_data, G
        
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise


def load_graph_from_neo4j(uri: str, user: str, password: str, 
                          database: str = "neo4j") -> Tuple[Dict, nx.DiGraph]:
    """
    Load graph from Neo4j database
    
    Args:
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        Tuple of (graph_data dict, NetworkX DiGraph)
    """
    logger.info(f"Connecting to Neo4j: {uri}, database: {database}")
    
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("neo4j package required. Install with: pip install neo4j")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        G = nx.DiGraph()
        graph_data = {
            'nodes': [],
            'applications': [],
            'topics': [],
            'brokers': [],
            'relationships': {
                'runs_on': [],
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'depends_on': [],
                'connects_to': []
            }
        }
        
        with driver.session(database=database) as session:
            # Load nodes
            result = session.run("""
                MATCH (n)
                RETURN n, labels(n) as labels
            """)
            
            for record in result:
                node = record['n']
                labels = record['labels']
                node_props = dict(node.items())
                node_name = node_props.get('name', None)
                node_id = node_props.get('id', None)
                
                # Add to NetworkX
                G.add_node(node_name, type=labels[0] if labels else 'Unknown', id=node_id)
                
                # Add to graph_data
                if 'Node' in labels:
                    graph_data['nodes'].append({'name': node_name, 'id': node_id, **node_props})
                elif 'Application' in labels:
                    graph_data['applications'].append({'name': node_name, **node_props})
                elif 'Topic' in labels:
                    graph_data['topics'].append({'name': node_name, **node_props})
                elif 'Broker' in labels:
                    graph_data['brokers'].append({'name': node_name, **node_props})
            
            # Load relationships
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.name as source, b.name as target, type(r) as rel_type, properties(r) as props
            """)
            
            for record in result:
                source = record['source']
                target = record['target']
                rel_type = record['rel_type']
                props = record['props'] or {}
                
                # Add to NetworkX
                G.add_edge(source, target, type=rel_type, **props)
                
                # Add to graph_data
                edge_data = {'from': source, 'to': target, **props}
                if rel_type == 'RUNS_ON':
                    graph_data['relationships']['runs_on'].append(edge_data)
                elif rel_type == 'PUBLISHES_TO':
                    graph_data['relationships']['publishes_to'].append(edge_data)
                elif rel_type == 'SUBSCRIBES_TO':
                    graph_data['relationships']['subscribes_to'].append(edge_data)
                elif rel_type == 'ROUTES':
                    graph_data['relationships']['routes'].append(edge_data)
                elif rel_type == 'DEPENDS_ON':
                    graph_data['relationships']['depends_on'].append(edge_data)
                elif rel_type == 'CONNECTS_TO':
                    graph_data['relationships']['connects_to'].append(edge_data)
        
        driver.close()
        logger.info(f"✓ Loaded from Neo4j: {len(G.nodes())} nodes, {len(G.edges())} edges")
        return graph_data, G
        
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        raise


def analyze_node_criticality(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze node criticality using multiple metrics
    
    Returns:
        Dictionary with node criticality analysis
    """
    logger.info("Analyzing node criticality...")
    
    results = {}
    
    # 1. Articulation points (single points of failure)
    logger.info("  - Finding articulation points...")
    # Convert to undirected for articulation point analysis
    G_undirected = G.to_undirected()
    articulation_points = list(nx.articulation_points(G_undirected))
    results['articulation_points'] = articulation_points
    
    # 2. Betweenness centrality (bottleneck nodes)
    logger.info("  - Computing betweenness centrality...")
    betweenness = nx.betweenness_centrality(G, normalized=True)
    # Top 10 by betweenness
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    results['top_betweenness'] = [
        {'node': node, 'score': round(score, 4)} 
        for node, score in sorted_betweenness[:10]
    ]
    
    # 3. Degree centrality (hub nodes)
    logger.info("  - Computing degree centrality...")
    degree_cent = nx.degree_centrality(G)
    sorted_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
    results['top_degree'] = [
        {'node': node, 'score': round(score, 4)} 
        for node, score in sorted_degree[:10]
    ]
    
    # 4. PageRank (influential nodes)
    logger.info("  - Computing PageRank...")
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
        sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        results['top_pagerank'] = [
            {'node': node, 'score': round(score, 4)} 
            for node, score in sorted_pagerank[:10]
        ]
    except:
        results['top_pagerank'] = []
    
    # 5. Composite criticality score
    logger.info("  - Computing composite criticality scores...")
    composite_scores = {}
    for node in G.nodes():
        score = 0.0
        # Betweenness (40%)
        score += 0.4 * betweenness.get(node, 0.0)
        # Articulation point (30%)
        score += 0.3 * (1.0 if node in articulation_points else 0.0)
        # Degree centrality (30%)
        score += 0.3 * degree_cent.get(node, 0.0)
        composite_scores[node] = score
    
    sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    results['top_critical_nodes'] = [
        {
            'node': node, 
            'score': round(score, 4),
            'is_articulation_point': node in articulation_points,
            'betweenness': round(betweenness.get(node, 0.0), 4),
            'degree_centrality': round(degree_cent.get(node, 0.0), 4)
        }
        for node, score in sorted_composite[:15]
    ]
    
    # Statistics
    results['statistics'] = {
        'total_nodes': len(G.nodes()),
        'articulation_points_count': len(articulation_points),
        'articulation_points_percentage': round(100 * len(articulation_points) / len(G.nodes()), 2),
        'avg_betweenness': round(sum(betweenness.values()) / len(betweenness), 4),
        'max_betweenness': round(max(betweenness.values()), 4)
    }
    
    logger.info(f"✓ Node analysis complete: {len(articulation_points)} articulation points found")
    return results


def analyze_edge_criticality(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze edge criticality - identify critical connections
    
    Returns:
        Dictionary with edge criticality analysis
    """
    logger.info("Analyzing edge criticality...")
    
    results = {}
    
    # 1. Bridge edges (critical links)
    logger.info("  - Finding bridge edges...")
    G_undirected = G.to_undirected()
    bridges = list(nx.bridges(G_undirected))
    results['bridges'] = [{'from': u, 'to': v} for u, v in bridges]
    
    # 2. Edge betweenness centrality
    logger.info("  - Computing edge betweenness centrality...")
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    sorted_edge_betweenness = sorted(edge_betweenness.items(), 
                                     key=lambda x: x[1], reverse=True)
    results['top_edge_betweenness'] = [
        {'from': u, 'to': v, 'score': round(score, 4)} 
        for (u, v), score in sorted_edge_betweenness[:15]
    ]
    
    # 3. Critical edges by type
    logger.info("  - Analyzing edges by type...")
    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        if edge_type not in edge_types:
            edge_types[edge_type] = []
        edge_types[edge_type].append((u, v))
    
    results['edges_by_type'] = {
        edge_type: len(edges) 
        for edge_type, edges in edge_types.items()
    }
    
    # 4. Impact analysis - simulate edge removal
    logger.info("  - Simulating edge removal impact...")
    edge_impact = []
    
    # Analyze top 10 edges by betweenness
    for (u, v), betweenness_score in sorted_edge_betweenness[:10]:
        # Create test graph
        G_test = G.copy()
        G_test.remove_edge(u, v)
        
        # Measure impact
        original_components = nx.number_weakly_connected_components(G)
        new_components = nx.number_weakly_connected_components(G_test)
        
        # Check if it creates disconnection
        creates_disconnection = new_components > original_components
        
        edge_impact.append({
            'from': u,
            'to': v,
            'betweenness': round(betweenness_score, 4),
            'creates_disconnection': creates_disconnection,
            'components_after_removal': new_components
        })
    
    results['edge_impact_analysis'] = edge_impact
    
    # Statistics
    results['statistics'] = {
        'total_edges': len(G.edges()),
        'bridge_count': len(bridges),
        'bridge_percentage': round(100 * len(bridges) / len(G.edges()), 2) if len(G.edges()) > 0 else 0,
        'avg_edge_betweenness': round(sum(edge_betweenness.values()) / len(edge_betweenness), 4) if edge_betweenness else 0,
        'max_edge_betweenness': round(max(edge_betweenness.values()), 4) if edge_betweenness else 0
    }
    
    logger.info(f"✓ Edge analysis complete: {len(bridges)} bridges found")
    return results


def analyze_graph_structure(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze basic graph structure and connectivity
    
    Returns:
        Dictionary with structural analysis
    """
    logger.info("Analyzing graph structure...")
    
    results = {}
    
    # Basic metrics
    results['nodes'] = len(G.nodes())
    results['edges'] = len(G.edges())
    results['density'] = round(nx.density(G), 4)
    
    # Connectivity
    results['is_weakly_connected'] = nx.is_weakly_connected(G)
    results['is_strongly_connected'] = nx.is_strongly_connected(G)
    results['number_weakly_connected_components'] = nx.number_weakly_connected_components(G)
    results['number_strongly_connected_components'] = nx.number_strongly_connected_components(G)
    
    # Node types
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    results['node_types'] = node_types
    
    # Edge types
    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    results['edge_types'] = edge_types
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    if degrees:
        results['degree_stats'] = {
            'mean': round(sum(degrees) / len(degrees), 2),
            'min': min(degrees),
            'max': max(degrees)
        }
    
    logger.info("✓ Structure analysis complete")
    return results


def analyze_layer_dependencies(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze dependencies across different system layers
    
    Returns:
        Dictionary with layer-wise analysis
    """
    logger.info("Analyzing layer dependencies...")
    
    results = {}
    
    # Extract nodes by type
    applications = [n for n, d in G.nodes(data=True) if d.get('type') == 'Application']
    topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
    brokers = [n for n, d in G.nodes(data=True) if d.get('type') == 'Broker']
    nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Node']
    
    # Application layer analysis
    app_dependencies = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get('type') == 'DEPENDS_ON']
    
    results['application_layer'] = {
        'total_applications': len(applications),
        'direct_dependencies': len(app_dependencies),
        'avg_dependencies_per_app': round(len(app_dependencies) / len(applications), 2) if applications else 0
    }
    
    # Topic layer analysis
    topic_publishers = {}
    topic_subscribers = {}
    for u, v, d in G.edges(data=True):
        if d.get('type') == 'PUBLISHES_TO' and v in topics:
            topic_publishers[v] = topic_publishers.get(v, 0) + 1
        elif d.get('type') == 'SUBSCRIBES_TO' and v in topics:
            topic_subscribers[v] = topic_subscribers.get(v, 0) + 1
    
    results['topic_layer'] = {
        'total_topics': len(topics),
        'avg_publishers_per_topic': round(sum(topic_publishers.values()) / len(topic_publishers), 2) if topic_publishers else 0,
        'avg_subscribers_per_topic': round(sum(topic_subscribers.values()) / len(topic_subscribers), 2) if topic_subscribers else 0,
        'most_popular_topics': sorted(
            [{'topic': t, 'total_connections': topic_publishers.get(t, 0) + topic_subscribers.get(t, 0)} 
             for t in topics],
            key=lambda x: x['total_connections'],
            reverse=True
        )[:5]
    }
    
    # Infrastructure layer analysis
    nodes_dependencies = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get('type') == 'CONNECTS_TO']
    
    results['infrastructure_layer'] = {
        'total_nodes': len(nodes),
        'direct_dependencies': len(nodes_dependencies),
        'avg_dependencies_per_node': round(len(nodes_dependencies) / len(nodes), 2) if nodes else 0
    }
    
    logger.info("✓ Layer analysis complete")
    return results


def generate_recommendations(node_analysis: Dict, edge_analysis: Dict, 
                            structure: Dict) -> list:
    """
    Generate actionable recommendations based on analysis
    
    Returns:
        List of recommendations
    """
    logger.info("Generating recommendations...")
    
    recommendations = []
    
    # 1. Articulation point recommendations
    if node_analysis['articulation_points']:
        for ap in node_analysis['articulation_points'][:5]:
            recommendations.append({
                'priority': 'CRITICAL',
                'type': 'Single Point of Failure',
                'component': ap,
                'issue': f'Node {ap} is an articulation point - its failure will disconnect the system',
                'recommendation': 'Add redundant paths or backup components',
                'impact': 'High - system fragmentation'
            })
    
    # 2. Bridge edge recommendations
    if edge_analysis.get('bridges'):
        for bridge in edge_analysis['bridges'][:3]:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'Critical Link',
                'component': f"{bridge['from']} → {bridge['to']}",
                'issue': 'This connection is a bridge - its failure will partition the network',
                'recommendation': 'Establish alternative communication paths',
                'impact': 'High - network fragmentation'
            })
    
    # 3. High betweenness nodes
    high_betweenness_nodes = [n for n in node_analysis['top_betweenness'][:5] 
                              if n['score'] > 0.1]
    for node_info in high_betweenness_nodes:
        if node_info['node'] not in node_analysis['articulation_points']:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'Bottleneck',
                'component': node_info['node'],
                'issue': f"High betweenness centrality ({node_info['score']}) indicates bottleneck",
                'recommendation': 'Consider load balancing or adding parallel paths',
                'impact': 'Medium - performance degradation'
            })
    
    # 4. High betweenness edges
    high_betweenness_edges = [e for e in edge_analysis['top_edge_betweenness'][:5] 
                             if e['score'] > 0.1]
    for edge_info in high_betweenness_edges:
        edge_id = f"{edge_info['from']} → {edge_info['to']}"
        if not any(b['from'] == edge_info['from'] and b['to'] == edge_info['to'] 
                  for b in edge_analysis.get('bridges', [])):
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'Communication Bottleneck',
                'component': edge_id,
                'issue': f"High edge betweenness ({edge_info['score']}) indicates traffic bottleneck",
                'recommendation': 'Add redundant communication channels or increase capacity',
                'impact': 'Medium - latency and throughput issues'
            })
    
    # 5. Connectivity issues
    if not structure['is_weakly_connected']:
        recommendations.append({
            'priority': 'CRITICAL',
            'type': 'Disconnected System',
            'component': 'Overall System',
            'issue': f'System has {structure["number_weakly_connected_components"]} disconnected components',
            'recommendation': 'Establish connections between isolated components',
            'impact': 'Critical - some components cannot communicate'
        })
    
    logger.info(f"✓ Generated {len(recommendations)} recommendations")
    return recommendations


def export_results_json(results: Dict, output_path: str):
    """Export results to JSON file"""
    logger.info(f"Exporting results to JSON: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("✓ JSON export complete")


def export_results_csv(results: Dict, output_dir: str):
    """Export results to CSV files"""
    logger.info(f"Exporting results to CSV: {output_dir}")
    
    try:
        import csv
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export critical nodes
        with open(output_path / 'critical_nodes.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['node', 'score', 'is_articulation_point', 
                                                   'betweenness', 'degree_centrality'])
            writer.writeheader()
            for node_info in results['node_analysis']['top_critical_nodes']:
                writer.writerow(node_info)
        
        # Export critical edges
        with open(output_path / 'critical_edges.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['from', 'to', 'score'])
            writer.writeheader()
            for edge_info in results['edge_analysis']['top_edge_betweenness']:
                writer.writerow(edge_info)
        
        # Export recommendations
        with open(output_path / 'recommendations.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['priority', 'type', 'component', 
                                                   'issue', 'recommendation', 'impact'])
            writer.writeheader()
            for rec in results['recommendations']:
                writer.writerow(rec)
        
        logger.info("✓ CSV export complete")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")


def export_results_html(results: Dict, output_path: str):
    """Export results to HTML report"""
    logger.info(f"Exporting results to HTML: {output_path}")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pub-Sub System Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; 
                         box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; }}
            h3 {{ color: #7f8c8d; }}
            .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
            .critical {{ border-left-color: #e74c3c; }}
            .warning {{ border-left-color: #f39c12; }}
            .info {{ border-left-color: #3498db; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .badge {{ display: inline-block; padding: 4px 8px; border-radius: 3px; 
                     font-size: 0.85em; font-weight: bold; }}
            .badge-critical {{ background-color: #e74c3c; color: white; }}
            .badge-high {{ background-color: #f39c12; color: white; }}
            .badge-medium {{ background-color: #3498db; color: white; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pub-Sub System Analysis Report</h1>
            <p class="timestamp">Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Graph Structure</h2>
            <div class="metric info">
                <strong>Nodes:</strong> {results['structure']['nodes']}<br>
                <strong>Edges:</strong> {results['structure']['edges']}<br>
                <strong>Density:</strong> {results['structure']['density']}<br>
                <strong>Weakly Connected:</strong> {'Yes' if results['structure']['is_weakly_connected'] else 'No'}<br>
                <strong>Connected Components:</strong> {results['structure']['number_weakly_connected_components']}
            </div>
            
            <h2>Critical Nodes (Top 10)</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Node</th>
                    <th>Criticality Score</th>
                    <th>Articulation Point</th>
                    <th>Betweenness</th>
                </tr>
    """
    
    for idx, node in enumerate(results['node_analysis']['top_critical_nodes'][:10], 1):
        html += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>{node['node']}</strong></td>
                    <td>{node['score']}</td>
                    <td>{'✓ Yes' if node['is_articulation_point'] else 'No'}</td>
                    <td>{node['betweenness']}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>Critical Edges (Top 10)</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>From</th>
                    <th>To</th>
                    <th>Betweenness Score</th>
                </tr>
    """
    
    for idx, edge in enumerate(results['edge_analysis']['top_edge_betweenness'][:10], 1):
        html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{edge['from']}</td>
                    <td>{edge['to']}</td>
                    <td>{edge['score']}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>Recommendations</h2>
    """
    
    for rec in results['recommendations']:
        priority_class = rec['priority'].lower()
        badge_class = f"badge-{priority_class}" if priority_class in ['critical', 'high', 'medium'] else 'badge-medium'
        html += f"""
            <div class="metric {priority_class}">
                <span class="badge {badge_class}">{rec['priority']}</span>
                <strong>{rec['type']}</strong><br>
                <strong>Component:</strong> {rec['component']}<br>
                <strong>Issue:</strong> {rec['issue']}<br>
                <strong>Recommendation:</strong> {rec['recommendation']}<br>
                <strong>Impact:</strong> {rec['impact']}
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info("✓ HTML export complete")


def print_summary(results: Dict):
    """Print summary to console"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}  GRAPH ANALYSIS SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
    
    # Structure
    print(f"{Colors.OKBLUE}📊 Graph Structure{Colors.ENDC}")
    structure = results['structure']
    print(f"   Nodes: {structure['nodes']}")
    print(f"   Edges: {structure['edges']}")
    print(f"   Density: {structure['density']}")
    print(f"   Connected: {'Yes' if structure['is_weakly_connected'] else 'No'}")
    
    # Node Analysis
    print(f"\n{Colors.OKBLUE}🔴 Critical Nodes{Colors.ENDC}")
    node_stats = results['node_analysis']['statistics']
    print(f"   Articulation Points: {node_stats['articulation_points_count']} ({node_stats['articulation_points_percentage']}%)")
    print(f"   Avg Betweenness: {node_stats['avg_betweenness']}")
    print(f"\n   Top 5 Critical Nodes:")
    for idx, node in enumerate(results['node_analysis']['top_critical_nodes'][:5], 1):
        ap_marker = '⚠️ ' if node['is_articulation_point'] else '   '
        print(f"   {idx}. {ap_marker}{node['node']} (score: {node['score']})")
    
    # Edge Analysis
    print(f"\n{Colors.OKBLUE}🔗 Critical Edges{Colors.ENDC}")
    edge_stats = results['edge_analysis']['statistics']
    print(f"   Bridges: {edge_stats['bridge_count']} ({edge_stats['bridge_percentage']}%)")
    print(f"   Avg Edge Betweenness: {edge_stats['avg_edge_betweenness']}")
    print(f"\n   Top 5 Critical Edges:")
    for idx, edge in enumerate(results['edge_analysis']['top_edge_betweenness'][:5], 1):
        print(f"   {idx}. {edge['from']} → {edge['to']} (score: {edge['score']})")
    
    # Recommendations
    print(f"\n{Colors.WARNING}💡 Recommendations ({len(results['recommendations'])}){Colors.ENDC}")
    priority_counts = {}
    for rec in results['recommendations']:
        priority_counts[rec['priority']] = priority_counts.get(rec['priority'], 0) + 1
    
    for priority, count in sorted(priority_counts.items()):
        print(f"   {priority}: {count}")
    
    if results['recommendations']:
        print(f"\n   Top Recommendations:")
        for idx, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {idx}. [{rec['priority']}] {rec['type']}: {rec['component']}")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Pub-Sub System Graph Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from JSON
  python analyze_graph.py --input system.json
  
  # Analyze from Neo4j
  python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
      --user neo4j --password password
  
  # With edge analysis and exports
  python analyze_graph.py --input system.json --analyze-edges \\
      --export-json --export-csv --export-html
  
  # Verbose logging
  python analyze_graph.py --input system.json --verbose
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input JSON file path')
    input_group.add_argument('--neo4j', action='store_true',
                            help='Load from Neo4j database')
    
    # Neo4j connection
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--user', '-u', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--password', '-p', type=str,
                       help='Neo4j password')
    parser.add_argument('--database', '-d', type=str, default='neo4j',
                       help='Neo4j database name (default: neo4j)')
    
    # Analysis options
    parser.add_argument('--analyze-edges', action='store_true',
                       help='Perform detailed edge criticality analysis (default: on)')
    parser.add_argument('--no-edge-analysis', action='store_true',
                       help='Skip edge criticality analysis')
    
    # Export options
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV files')
    parser.add_argument('--export-html', action='store_true',
                       help='Export results to HTML report')
    parser.add_argument('--output', '-o', type=str, default='analysis_results',
                       help='Output directory/file prefix (default: analysis_results)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Load graph
        start_time = time.time()
        
        if args.input:
            graph_data, G = load_graph_from_json(args.input)
        else:
            if not args.password:
                print("Error: --password required for Neo4j connection")
                return 1
            graph_data, G = load_graph_from_neo4j(args.uri, args.user, 
                                                   args.password, args.database)
        
        # Perform analysis
        results = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': args.input if args.input else f'neo4j:{args.database}'
            }
        }
        
        # 1. Structure analysis
        results['structure'] = analyze_graph_structure(G)
        
        # 2. Node criticality
        results['node_analysis'] = analyze_node_criticality(G)
        
        # 3. Edge criticality (unless disabled)
        if not args.no_edge_analysis:
            results['edge_analysis'] = analyze_edge_criticality(G)
        else:
            results['edge_analysis'] = {'skipped': True}
        
        # 4. Layer dependencies
        results['layer_analysis'] = analyze_layer_dependencies(G)
        
        # 5. Generate recommendations
        results['recommendations'] = generate_recommendations(
            results['node_analysis'],
            results['edge_analysis'],
            results['structure']
        )
        
        # Analysis time
        analysis_time = time.time() - start_time
        results['metadata']['analysis_time_seconds'] = round(analysis_time, 2)
        
        # Print summary
        if not args.quiet:
            print_summary(results)
        
        # Export results
        if args.export_json:
            export_results_json(results, f"{args.output}.json")
        
        if args.export_csv:
            export_results_csv(results, f"{args.output}_csv")
        
        if args.export_html:
            export_results_html(results, f"{args.output}.html")
        
        logger.info(f"✓ Analysis complete in {analysis_time:.2f}s")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
