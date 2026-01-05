"""
Simulation Graph

Wraps GraphData into a NetworkX graph for simulation.
Handles raw structural relationships and graph topology analysis.
"""

import networkx as nx
import logging
from typing import Dict, List, Set, Tuple
from src.core.graph_exporter import GraphData

class SimulationGraph:
    def __init__(self, graph_data: GraphData):
        self.graph = self._build_graph(graph_data)
        self.logger = logging.getLogger(__name__)
        # Cache initial total paths for reachability calculations
        self.initial_paths = self.get_pub_sub_paths()
        self.initial_component_count = self.get_connected_components_count()

    def _build_graph(self, data: GraphData) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # Add Nodes
        for c in data.components:
            G.add_node(
                c.id, 
                type=c.component_type, 
                weight=c.weight if hasattr(c, 'weight') else 1.0, # Default weight if missing
                state="active",
                load=0.0
            )
            
        # Add Edges
        for e in data.edges:
            G.add_edge(
                e.source_id,
                e.target_id,
                relation_type=e.relation_type,
                weight=e.weight
            )
        return G

    def reset(self):
        """Reset simulation state to initial healthy state."""
        nx.set_node_attributes(self.graph, "active", "state")
        nx.set_node_attributes(self.graph, 0.0, "load")

    def get_nodes_by_type(self, c_type: str) -> List[str]:
        return [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == c_type]

    def get_successors_by_type(self, node: str, relation_type: str) -> List[str]:
        """Get neighbors connected by a specific outgoing relationship type."""
        if node not in self.graph: return []
        successors = []
        for succ in self.graph.successors(node):
            edge_data = self.graph.get_edge_data(node, succ)
            if edge_data.get("relation_type") == relation_type:
                successors.append(succ)
        return successors

    def get_predecessors_by_type(self, node: str, relation_type: str) -> List[str]:
        """
        Get neighbors connected by a specific incoming relationship type.
        Example: If relation is A->B (A depends on B), and we ask for preds of B, we get A.
        """
        if node not in self.graph: return []
        predecessors = []
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            if edge_data.get("relation_type") == relation_type:
                predecessors.append(pred)
        return predecessors
    
    def get_edge_weight(self, source: str, target: str) -> float:
        """Return the weight of the edge between source and target."""
        if self.graph.has_edge(source, target):
            return self.graph[source][target].get("weight", 0.0)
        return 0.0

    def get_pub_sub_paths(self, active_only: bool = False) -> Set[Tuple[str, str, str]]:
        """
        Identify all valid Pub-Sub paths: Publisher -> Topic -> Subscriber.
        Returns a set of (Publisher, Topic, Subscriber) tuples.
        """
        paths = set()
        # Find all Topics
        topics = self.get_nodes_by_type("Topic")
        
        for topic in topics:
            if active_only and self.graph.nodes[topic].get("state") != "active":
                continue

            # Publishers: Nodes that PUBLISH_TO this topic (Incoming or Outgoing depending on model)
            # Standard model: Publisher -> Topic (PUBLISHES_TO)
            publishers = self.get_predecessors_by_type(topic, "PUBLISHES_TO")
            
            # Subscribers: Nodes that SUBSCRIBE_TO this topic
            # Standard model: Subscriber -> Topic (SUBSCRIBES_TO)
            subscribers = self.get_predecessors_by_type(topic, "SUBSCRIBES_TO")
            
            for pub in publishers:
                if active_only and self.graph.nodes[pub].get("state") != "active":
                    continue
                for sub in subscribers:
                    if active_only and self.graph.nodes[sub].get("state") != "active":
                        continue
                    paths.add((pub, topic, sub))
        return paths

    def get_connected_components_count(self, active_only: bool = False) -> int:
        """
        Calculate fragmentation. Using Weakly Connected Components for directed graph.
        """
        if active_only:
            # Create subgraph of active nodes
            active_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("state") == "active"]
            subgraph = self.graph.subgraph(active_nodes)
            return nx.number_weakly_connected_components(subgraph)
        else:
            return nx.number_weakly_connected_components(self.graph)