"""
Structural Analyzer

Performs structural analysis of pub-sub system graphs including:
- Graph topology metrics (density, diameter, clustering)
- Articulation points and bridges detection
- Community detection
- Path analysis
- Layer decomposition
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import logging


class StructuralAnalyzer:
    """
    Analyzes structural properties of pub-sub system graphs.
    
    Provides insights into:
    - Graph connectivity and topology
    - Critical structural elements (articulation points, bridges)
    - Community structure
    - Layer decomposition (application, broker, infrastructure)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Perform comprehensive structural analysis.
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary containing all structural analysis results
        """
        self.logger.info("Performing structural analysis...")
        
        results = {}
        
        # Basic metrics
        results['basic_metrics'] = self._analyze_basic_metrics(graph)
        
        # Node and edge type distribution
        results['type_distribution'] = self._analyze_type_distribution(graph)
        
        # Connectivity analysis
        results['connectivity'] = self._analyze_connectivity(graph)
        
        # Critical elements
        results['critical_elements'] = self._analyze_critical_elements(graph)
        
        # Degree analysis
        results['degree_analysis'] = self._analyze_degrees(graph)
        
        # Path analysis
        results['path_analysis'] = self._analyze_paths(graph)
        
        # Layer analysis
        results['layer_analysis'] = self._analyze_layers(graph)
        
        return results
    
    def _analyze_basic_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze basic graph metrics"""
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': round(nx.density(graph), 6),
            'is_directed': graph.is_directed(),
            'self_loops': nx.number_of_selfloops(graph)
        }
    
    def _analyze_type_distribution(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze distribution of node and edge types"""
        # Node types
        node_types = defaultdict(int)
        for _, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            node_types[node_type] += 1
        
        # Edge types
        edge_types = defaultdict(int)
        for _, _, data in graph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            edge_types[edge_type] += 1
        
        return {
            'node_types': dict(node_types),
            'edge_types': dict(edge_types)
        }
    
    def _analyze_connectivity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        undirected = graph.to_undirected()
        
        results = {
            'is_weakly_connected': nx.is_weakly_connected(graph),
            'weakly_connected_components': nx.number_weakly_connected_components(graph),
            'is_strongly_connected': nx.is_strongly_connected(graph),
            'strongly_connected_components': nx.number_strongly_connected_components(graph)
        }
        
        # Largest component size
        if results['weakly_connected_components'] > 0:
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            results['largest_component_size'] = len(largest_cc)
            results['largest_component_ratio'] = round(
                len(largest_cc) / graph.number_of_nodes(), 4
            )
        
        return results
    
    def _analyze_critical_elements(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Identify critical structural elements"""
        undirected = graph.to_undirected()
        
        # Articulation points
        aps = list(nx.articulation_points(undirected))
        ap_by_type = defaultdict(list)
        for ap in aps:
            node_type = graph.nodes[ap].get('type', 'Unknown')
            ap_by_type[node_type].append(ap)
        
        # Bridges
        bridges = list(nx.bridges(undirected))
        bridge_types = defaultdict(int)
        for u, v in bridges:
            edge_data = graph.get_edge_data(u, v) or graph.get_edge_data(v, u) or {}
            edge_type = edge_data.get('type', 'Unknown')
            bridge_types[edge_type] += 1
        
        return {
            'articulation_points': {
                'count': len(aps),
                'nodes': aps[:50],  # Limit for large graphs
                'by_type': dict(ap_by_type)
            },
            'bridges': {
                'count': len(bridges),
                'edges': bridges[:50],
                'by_type': dict(bridge_types)
            }
        }
    
    def _analyze_degrees(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze degree distribution"""
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        total_degrees = [d for _, d in graph.degree()]
        
        results = {}
        
        if total_degrees:
            results['total'] = {
                'avg': round(sum(total_degrees) / len(total_degrees), 2),
                'max': max(total_degrees),
                'min': min(total_degrees)
            }
        
        if in_degrees:
            results['in_degree'] = {
                'avg': round(sum(in_degrees) / len(in_degrees), 2),
                'max': max(in_degrees),
                'min': min(in_degrees)
            }
        
        if out_degrees:
            results['out_degree'] = {
                'avg': round(sum(out_degrees) / len(out_degrees), 2),
                'max': max(out_degrees),
                'min': min(out_degrees)
            }
        
        # Top nodes by degree
        degree_list = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        results['top_by_degree'] = [
            {'node': n, 'degree': d, 'type': graph.nodes[n].get('type', 'Unknown')}
            for n, d in degree_list[:10]
        ]
        
        return results
    
    def _analyze_paths(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze path characteristics"""
        results = {}
        
        # Only for connected graphs or largest component
        if nx.is_weakly_connected(graph):
            undirected = graph.to_undirected()
            
            try:
                results['diameter'] = nx.diameter(undirected)
                results['radius'] = nx.radius(undirected)
                results['avg_shortest_path'] = round(
                    nx.average_shortest_path_length(undirected), 4
                )
            except:
                results['diameter'] = None
                results['radius'] = None
                results['avg_shortest_path'] = None
        else:
            # Analyze largest component
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc).to_undirected()
            
            try:
                results['largest_component_diameter'] = nx.diameter(subgraph)
                results['largest_component_avg_path'] = round(
                    nx.average_shortest_path_length(subgraph), 4
                )
            except:
                pass
        
        return results
    
    def _analyze_layers(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze graph by layer (application, broker, infrastructure)"""
        layers = {
            'Application': [],
            'Topic': [],
            'Broker': [],
            'Node': []
        }
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            if node_type in layers:
                layers[node_type].append(node)
        
        results = {}
        
        for layer_name, nodes in layers.items():
            if nodes:
                subgraph = graph.subgraph(nodes)
                
                # Count internal edges
                internal_edges = subgraph.number_of_edges()
                
                # Count cross-layer edges
                cross_layer_edges = 0
                for node in nodes:
                    for _, target in graph.out_edges(node):
                        if target not in nodes:
                            cross_layer_edges += 1
                    for source, _ in graph.in_edges(node):
                        if source not in nodes:
                            cross_layer_edges += 1
                
                results[layer_name] = {
                    'node_count': len(nodes),
                    'internal_edges': internal_edges,
                    'cross_layer_edges': cross_layer_edges // 2  # Avoid double counting
                }
        
        return results
    
    # =========================================================================
    # Specialized Analysis Methods
    # =========================================================================
    
    def find_dependency_chains(self, 
                              graph: nx.DiGraph, 
                              max_length: int = 5) -> List[List[str]]:
        """Find long dependency chains"""
        chains = []
        
        # Look for DEPENDS_ON edges
        dep_graph = nx.DiGraph()
        for s, t, d in graph.edges(data=True):
            if d.get('type') == 'DEPENDS_ON':
                dep_graph.add_edge(s, t)
        
        # Find all simple paths up to max_length
        for source in dep_graph.nodes():
            for target in dep_graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(
                            dep_graph, source, target, cutoff=max_length
                        ))
                        for path in paths:
                            if len(path) >= 3:  # At least 3 nodes
                                chains.append(path)
                    except:
                        pass
        
        # Sort by length and return unique
        chains.sort(key=len, reverse=True)
        return chains[:100]  # Limit
    
    def analyze_topic_fanout(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze topic fan-out patterns"""
        topics = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Topic']
        
        fanout_data = []
        for topic in topics:
            publishers = [s for s, t, d in graph.in_edges(topic, data=True)
                         if d.get('type') == 'PUBLISHES_TO']
            subscribers = [s for s, t, d in graph.in_edges(topic, data=True)
                          if d.get('type') == 'SUBSCRIBES_TO']
            
            fanout_data.append({
                'topic': topic,
                'publishers': len(publishers),
                'subscribers': len(subscribers),
                'total': len(publishers) + len(subscribers),
                'ratio': len(subscribers) / max(1, len(publishers))
            })
        
        # Sort by total connections
        fanout_data.sort(key=lambda x: x['total'], reverse=True)
        
        return {
            'topics_analyzed': len(topics),
            'high_fanout_topics': [t for t in fanout_data if t['total'] >= 10],
            'avg_publishers': round(
                sum(t['publishers'] for t in fanout_data) / max(1, len(fanout_data)), 2
            ),
            'avg_subscribers': round(
                sum(t['subscribers'] for t in fanout_data) / max(1, len(fanout_data)), 2
            ),
            'top_topics': fanout_data[:10]
        }
    
    def analyze_infrastructure_distribution(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze how components are distributed across infrastructure"""
        nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Node']
        
        distribution = {}
        for node in nodes:
            # Find components on this node
            hosted = []
            for s, t, d in graph.in_edges(node, data=True):
                if d.get('type') == 'RUNS_ON':
                    hosted.append({
                        'component': s,
                        'type': graph.nodes[s].get('type', 'Unknown')
                    })
            
            distribution[node] = {
                'hosted_count': len(hosted),
                'hosted_components': hosted,
                'by_type': defaultdict(int)
            }
            
            for comp in hosted:
                distribution[node]['by_type'][comp['type']] += 1
            
            distribution[node]['by_type'] = dict(distribution[node]['by_type'])
        
        # Summary statistics
        hosted_counts = [d['hosted_count'] for d in distribution.values()]
        
        return {
            'nodes': distribution,
            'summary': {
                'total_infrastructure_nodes': len(nodes),
                'avg_components_per_node': round(
                    sum(hosted_counts) / max(1, len(hosted_counts)), 2
                ),
                'max_components_on_node': max(hosted_counts) if hosted_counts else 0,
                'min_components_on_node': min(hosted_counts) if hosted_counts else 0
            }
        }