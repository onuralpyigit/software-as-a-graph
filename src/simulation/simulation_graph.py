"""
Simulation Graph

Wraps GraphData into a NetworkX graph for simulation.
Handles raw structural relationships.
"""

import networkx as nx
import logging
from typing import Dict, List, Set, Optional
from src.core.graph_exporter import GraphData

class SimulationGraph:
    def __init__(self, graph_data: GraphData):
        self.graph = self._build_graph(graph_data)
        self.logger = logging.getLogger(__name__)

    def _build_graph(self, data: GraphData) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # Add Nodes
        for c in data.components:
            G.add_node(
                c.id, 
                type=c.component_type, 
                weight=c.weight,
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
        """Reset simulation state."""
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
        """Get neighbors connected by a specific incoming relationship type."""
        if node not in self.graph: return []
        predecessors = []
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            if edge_data.get("relation_type") == relation_type:
                predecessors.append(pred)
        return predecessors