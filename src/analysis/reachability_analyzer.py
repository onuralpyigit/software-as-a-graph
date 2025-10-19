"""
Reachability Analyzer

Analyzes reachability and impact of component failures on system connectivity
"""

import networkx as nx
from typing import Dict, Set, List


class ReachabilityAnalyzer:
    """Analyzes reachability and connectivity impact"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.baseline_reachability = self._calculate_reachability_matrix()
    
    def _calculate_reachability_matrix(self) -> Dict[str, Set[str]]:
        """Calculate which nodes can reach which other nodes"""
        reachability = {}
        
        for node in self.graph.nodes():
            reachable = set(nx.descendants(self.graph, node))
            reachability[node] = reachable
        
        return reachability
    
    def analyze_impact(self, component: str) -> Dict:
        """
        Analyze the impact of removing a component
        
        Args:
            component: Component to remove
        
        Returns:
            Dictionary with impact analysis
        """
        # Create graph without component
        test_graph = self.graph.copy()
        test_graph.remove_node(component)
        
        # Calculate new reachability
        new_reachability = {}
        for node in test_graph.nodes():
            reachable = set(nx.descendants(test_graph, node))
            new_reachability[node] = reachable
        
        # Calculate lost connections
        lost_connections = 0
        affected_components = set()
        
        for node in self.baseline_reachability:
            if node == component:
                continue
            
            if node in new_reachability:
                lost = self.baseline_reachability[node] - new_reachability[node]
                lost_connections += len(lost)
                if lost:
                    affected_components.add(node)
        
        # Calculate metrics
        total_pairs = sum(len(r) for r in self.baseline_reachability.values())
        impact_score = lost_connections / total_pairs if total_pairs > 0 else 0
        
        return {
            'component': component,
            'lost_connections': lost_connections,
            'affected_components': list(affected_components),
            'impact_score': impact_score,
            'connectivity_loss_percentage': impact_score * 100
        }
    
    def find_critical_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all paths between source and target"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=10))
            return paths[:10]  # Limit to 10 paths
        except:
            return []
    
    def calculate_resilience_score(self) -> float:
        """
        Calculate overall system resilience score
        
        Higher score = more resilient
        """
        undirected = self.graph.to_undirected()
        
        # Factors for resilience
        factors = []
        
        # 1. Connectivity (0-1)
        if nx.is_connected(undirected):
            factors.append(1.0)
        else:
            # Penalize disconnected graph
            largest_cc = max(nx.connected_components(undirected), key=len)
            factors.append(len(largest_cc) / len(self.graph))
        
        # 2. Node connectivity (normalized)
        node_conn = nx.node_connectivity(undirected) if nx.is_connected(undirected) else 0
        factors.append(min(1.0, node_conn / 3))  # 3+ is good
        
        # 3. Average degree (normalized)
        avg_degree = sum(dict(self.graph.degree()).values()) / len(self.graph)
        factors.append(min(1.0, avg_degree / 4))  # 4+ connections is good
        
        # 4. Inverse of articulation points (normalized)
        art_points = len(list(nx.articulation_points(undirected)))
        factors.append(1.0 - min(1.0, art_points / (len(self.graph) * 0.2)))
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        resilience = sum(f * w for f, w in zip(factors, weights))
        
        return round(resilience, 3)
