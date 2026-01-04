"""
Simulation Graph

Wraps the core GraphData into a NetworkX graph optimized for simulation.
Handles graph construction and state management for nodes during simulation.

Author: Software-as-a-Graph Research Project
"""

import networkx as nx
import logging
from typing import Dict, Any, Set, List
from src.core.graph_exporter import GraphData, ComponentData, EdgeData

class SimulationGraph:
    """
    NetworkX wrapper for simulation operations.
    Supports isolating subgraphs for specific component types or layers.
    """
    
    def __init__(self, graph_data: GraphData, directed: bool = True):
        self.graph = self._build_graph(graph_data, directed)
        self.initial_states = {n: "active" for n in self.graph.nodes}
        self.logger = logging.getLogger(__name__)

    def _build_graph(self, data: GraphData, directed: bool) -> nx.DiGraph:
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Add nodes with initial metadata
        for comp in data.components:
            G.add_node(
                comp.id, 
                type=comp.component_type, 
                weight=comp.weight,
                state="active",
                load=0.0
            )
            
        # Add edges
        for edge in data.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                weight=edge.weight,
                dependency_type=edge.dependency_type
            )
            
        return G

    def get_subgraph(self, component_type: str = None, nodes: Set[str] = None) -> 'SimulationGraph':
        """Create a new SimulationGraph restricted to specific nodes/types."""
        if nodes is None and component_type is None:
            return self
            
        if nodes is None:
            nodes = {n for n, attr in self.graph.nodes(data=True) if attr.get("type") == component_type}
            
        sub_G = self.graph.subgraph(nodes).copy()
        
        # Create a dummy GraphData to wrap the subgraph back into SimulationGraph
        # (Simplified for internal use)
        # Note: We strictly use the NetworkX graph for simulation logic, 
        # so we can return a lightweight wrapper or just modify self for the simulation context.
        # Here we return a new instance with the induced subgraph.
        
        # We need to reconstruct GraphData-like structure or modify logic. 
        # Easier approach: This class holds the nx.Graph. We return a new instance manually.
        new_instance = SimulationGraph.__new__(SimulationGraph)
        new_instance.graph = sub_G
        new_instance.initial_states = {n: "active" for n in sub_G.nodes}
        new_instance.logger = self.logger
        return new_instance

    def reset(self):
        """Reset node states."""
        nx.set_node_attributes(self.graph, "active", "state")
        nx.set_node_attributes(self.graph, 0.0, "load")
        
    @property
    def nodes(self):
        return self.graph.nodes(data=True)
        
    def neighbors(self, node_id: str):
        return self.graph.neighbors(node_id)
        
    def successors(self, node_id: str):
        if self.graph.is_directed():
            return self.graph.successors(node_id)
        return self.graph.neighbors(node_id)
    
    def get_edge_weight(self, u: str, v: str) -> float:
        return self.graph[u][v].get("weight", 1.0)