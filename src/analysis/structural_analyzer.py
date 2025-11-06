"""
Structural Analyzer

Analyzes structural properties of the graph including:
- Articulation points
- Bridges
- Cycles
- Connected components
"""

import networkx as nx
from typing import Dict, List, Set, Tuple


class StructuralAnalyzer:
    """Analyzes structural vulnerabilities in the graph"""
    
    def analyze(self, graph: nx.DiGraph) -> Dict:
        """
        Perform comprehensive structural analysis
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary with structural analysis results
        """
        # Convert to undirected for some analyses
        undirected = graph.to_undirected()
        
        results = {
            'articulation_points': list(nx.articulation_points(undirected)),
            'bridges': list(nx.bridges(undirected)),
            'connected_components': {
                'count': nx.number_weakly_connected_components(graph),
                'sizes': [len(c) for c in nx.weakly_connected_components(graph)]
            },
            'strongly_connected_components': {
                'count': nx.number_strongly_connected_components(graph),
                'sizes': [len(c) for c in nx.strongly_connected_components(graph)]
            },
            #cycles': self._find_cycles(graph),
            'node_connectivity': nx.node_connectivity(undirected) if nx.is_connected(undirected) else 0,
            'edge_connectivity': nx.edge_connectivity(undirected) if nx.is_connected(undirected) else 0
        }
        
        return results
    
    def _find_cycles(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find all cycles in the directed graph"""
        try:
            cycles = list(nx.simple_cycles(graph))
            # Limit to reasonable number
            return cycles[:50]
        except:
            return []
    
    def find_single_points_of_failure(self, 
                                     graph: nx.DiGraph,
                                     component_types: List[str] = None) -> List[Dict]:
        """
        Identify single points of failure
        
        Args:
            graph: NetworkX directed graph
            component_types: Types of components to check (e.g., ['Broker', 'Node'])
        
        Returns:
            List of SPOFs with details
        """
        undirected = graph.to_undirected()
        articulation_points = set(nx.articulation_points(undirected))
        
        spofs = []
        for node in articulation_points:
            node_data = graph.nodes.get(node, {})
            node_type = node_data.get('type', 'Unknown')
            
            if component_types is None or node_type in component_types:
                # Count affected components
                affected = len(list(graph.neighbors(node)))
                
                spofs.append({
                    'component': node,
                    'type': node_type,
                    'is_articulation_point': True,
                    'components_affected': affected,
                    'risk_level': 'CRITICAL' if affected > 5 else 'HIGH'
                })
        
        return sorted(spofs, key=lambda x: x['components_affected'], reverse=True)
    
    def analyze_redundancy(self, graph: nx.DiGraph) -> Dict:
        """Analyze redundancy in the system"""
        results = {
            'single_brokers': [],
            'single_paths': [],
            'isolated_components': []
        }
        
        # Find single broker dependencies
        brokers = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Broker']
        
        for topic in [n for n, d in graph.nodes(data=True) if d.get('type') == 'Topic']:
            # Count brokers for this topic
            topic_brokers = [
                b for b in brokers
                if graph.has_edge(b, topic)
            ]
            
            if len(topic_brokers) == 1:
                results['single_brokers'].append({
                    'topic': topic,
                    'broker': topic_brokers[0]
                })
        
        # Find isolated components (degree = 0)
        for node in graph.nodes():
            if graph.degree(node) == 0:
                results['isolated_components'].append(node)
        
        return results
