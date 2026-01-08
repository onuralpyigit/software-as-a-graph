"""
Simulation Graph

Wraps GraphData into a NetworkX graph for simulation.
Strictly operates on RAW relationships to avoid using derived dependencies.
"""

import networkx as nx
import logging
from typing import Dict, List, Set, Tuple, Optional
from src.core.graph_exporter import GraphData

class SimulationGraph:
    # Explicitly define allowed RAW relationship types
    ALLOWED_RELATIONS = {
        "PUBLISHES_TO", 
        "SUBSCRIBES_TO", 
        "RUNS_ON", 
        "CONNECTS_TO", 
        "ROUTES"
    }

    def __init__(self, graph_data: GraphData):
        self.logger = logging.getLogger(__name__)
        self.graph = self._build_graph(graph_data)
        
        # Cache initial state for impact calculations
        self.initial_paths = self.get_pub_sub_paths()
        self.initial_node_count = self.graph.number_of_nodes()
        self.initial_component_count = self.get_connected_components_count()
        
        self.logger.debug(f"Simulation Graph initialized: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def _build_graph(self, data: GraphData) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # 1. Add Nodes
        for c in data.components:
            G.add_node(
                c.id, 
                type=c.component_type, 
                weight=c.weight if hasattr(c, 'weight') else 1.0,
                state="active",
                load=0.0
            )
            
        # 2. Add Edges (Filtering for Raw Types only)
        count = 0
        for e in data.edges:
            if e.relation_type in self.ALLOWED_RELATIONS:
                G.add_edge(
                    e.source_id,
                    e.target_id,
                    relation_type=e.relation_type,
                    weight=e.weight
                )
                count += 1
        
        return G

    def reset(self):
        """Reset simulation state to initial healthy state."""
        nx.set_node_attributes(self.graph, "active", "state")
        nx.set_node_attributes(self.graph, 0.0, "load")

    def get_successors_by_type(self, node: str, relation_type: str) -> List[str]:
        """Get outgoing neighbors connected by a specific relationship type."""
        if node not in self.graph: return []
        return [
            v for _, v, attr in self.graph.out_edges(node, data=True)
            if attr.get("relation_type") == relation_type
        ]

    def get_predecessors_by_type(self, node: str, relation_type: str) -> List[str]:
        """Get incoming neighbors connected by a specific relationship type."""
        if node not in self.graph: return []
        return [
            u for u, _, attr in self.graph.in_edges(node, data=True)
            if attr.get("relation_type") == relation_type
        ]
        
    def get_hosted_components(self, node_id: str) -> List[str]:
        """
        Find components that run on this node.
        Logic: (App)-[:RUNS_ON]->(Node). We want the Apps (Predecessors).
        """
        return self.get_predecessors_by_type(node_id, "RUNS_ON")

    def get_pub_sub_paths(self, active_only: bool = False) -> Set[Tuple[str, str, str]]:
        """
        Identify all valid Pub-Sub paths: Publisher -> Topic -> Subscriber.
        Returns a set of (Publisher, Topic, Subscriber) tuples.
        """
        paths = set()
        topics = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "Topic"]
        
        for topic in topics:
            if active_only and self.graph.nodes[topic].get("state") != "active":
                continue

            # (Publisher)-[:PUBLISHES_TO]->(Topic)
            publishers = self.get_predecessors_by_type(topic, "PUBLISHES_TO")
            
            # (Subscriber)-[:SUBSCRIBES_TO]->(Topic)
            subscribers = self.get_predecessors_by_type(topic, "SUBSCRIBES_TO")
            
            for pub in publishers:
                if active_only and self.graph.nodes[pub].get("state") != "active": continue
                for sub in subscribers:
                    if active_only and self.graph.nodes[sub].get("state") != "active": continue
                    paths.add((pub, topic, sub))
        return paths

    def get_connected_components_count(self, active_only: bool = False) -> int:
        """Calculate fragmentation using Weakly Connected Components."""
        if active_only:
            active_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("state") == "active"]
            if not active_nodes: return 0
            subgraph = self.graph.subgraph(active_nodes)
            return nx.number_weakly_connected_components(subgraph)
        return nx.number_weakly_connected_components(self.graph)