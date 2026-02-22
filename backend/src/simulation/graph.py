"""
Simulation Graph

Graph representation for simulation using RAW structural relationships.
Works directly on PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
relationships without deriving DEPENDS_ON.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Set, Tuple, Any, Optional, FrozenSet
from collections import defaultdict

import networkx as nx

from .models import ComponentState, RelationType, ComponentInfo, TopicInfo
from src.core.layers import SimulationLayer, SIMULATION_LAYERS

class SimulationGraph:
    """
    Graph representation for pub-sub system simulation.
    
    Works on RAW structural relationships:
        - Application -[PUBLISHES_TO]-> Topic
        - Application -[SUBSCRIBES_TO]-> Topic
        - Topic -[ROUTES]-> Broker (or Broker -[ROUTES]-> Topic)
        - Application -[RUNS_ON]-> Node
        - Broker -[RUNS_ON]-> Node
        - Node -[CONNECTS_TO]-> Node
    
    This domain model assumes data is loaded via an external mechanism (GraphData).
    """
    
    def __init__(self, graph_data: Any = None):
        """
        Initialize simulation graph.
        
        Args:
            graph_data: Pre-loaded GraphData object containing components and edges.
        """
        self.logger = logging.getLogger(__name__)
        
        # NetworkX graph for structural queries
        self.graph = nx.DiGraph()
        
        # Component registries
        self.components: Dict[str, ComponentInfo] = {}
        self.topics: Dict[str, TopicInfo] = {}
        
        # Relationship indices for fast lookups
        self._publishers: Dict[str, List[Tuple[str, float]]] = defaultdict(list)   # topic -> [(apps, weight)]
        self._subscribers: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # topic -> [(apps, weight)]
        self._routing: Dict[str, List[Tuple[str, float]]] = defaultdict(list)      # topic -> [(brokers, weight)]
        self._hosted_on: Dict[str, str] = {}                                       # comp -> node
        self._hosts: Dict[str, List[str]] = defaultdict(list)                      # node -> [comps]
        self._connections: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # node -> [(nodes, weight)]
        self._uses: Dict[str, List[str]] = defaultdict(list)                       # app/lib -> [libs]
        self._used_by: Dict[str, List[str]] = defaultdict(list)                    # lib -> [apps/libs]
        
        # Load graph
        if graph_data:
            self._load_from_data(graph_data)
    
    def _load_from_data(self, graph_data: Any) -> None:
        """Load from pre-loaded GraphData object."""
        self.logger.info("Loading simulation graph from GraphData")
        
        # Load components
        for comp in graph_data.components:
            comp_id = comp.id if hasattr(comp, 'id') else comp.get('id')
            comp_type = comp.component_type if hasattr(comp, 'component_type') else comp.get('type')
            props = comp.properties if hasattr(comp, 'properties') else {}
            
            if comp_type == "Topic":
                self.topics[comp_id] = TopicInfo(
                    id=comp_id,
                    name=props.get("name", comp_id),
                    message_size=props.get("message_size", 1024),
                    qos_reliability=props.get("qos_reliability", "BEST_EFFORT"),
                    qos_durability=props.get("qos_durability", "VOLATILE"),
                    qos_priority=props.get("qos_priority", "LOW"),
                    weight=props.get("weight", 1.0),
                )
            
            self.components[comp_id] = ComponentInfo(
                id=comp_id,
                type=comp_type,
                weight=props.get("weight", 1.0),
                properties=props
            )
            self.graph.add_node(comp_id, type=comp_type)
        
        # Load edges
        for edge in graph_data.edges:
            src = edge.source_id if hasattr(edge, 'source_id') else edge.get('source')
            tgt = edge.target_id if hasattr(edge, 'target_id') else edge.get('target')
            rel = edge.relation_type if hasattr(edge, 'relation_type') else edge.get('relation_type', 'UNKNOWN')
            weight = edge.weight if hasattr(edge, 'weight') else edge.get('weight', 1.0)
            
            self.graph.add_edge(src, tgt, relation=rel, weight=weight)
            
            # Index by relationship type
            if rel == "PUBLISHES_TO":
                self._publishers[tgt].append((src, weight))
            elif rel == "SUBSCRIBES_TO":
                self._subscribers[tgt].append((src, weight))
            elif rel == "ROUTES":
                self._routing[tgt].append((src, weight))
            elif rel == "RUNS_ON":
                self._hosted_on[src] = tgt
                self._hosts[tgt].append(src)  # Hosts don't strictly need weight for these cascades
            elif rel == "CONNECTS_TO":
                self._connections[src].append((tgt, weight))
            elif rel == "USES":
                self._uses[src].append(tgt)
                self._used_by[tgt].append(src)
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def reset(self) -> None:
        """Reset all component states and metrics for a new simulation."""
        for comp in self.components.values():
            comp.state = ComponentState.ACTIVE
            comp.reset_metrics()
    
    def fail_component(self, comp_id: str) -> None:
        """Mark a component as failed."""
        if comp_id in self.components:
            self.components[comp_id].state = ComponentState.FAILED
    
    def recover_component(self, comp_id: str) -> None:
        """Recover a failed component."""
        if comp_id in self.components:
            self.components[comp_id].state = ComponentState.ACTIVE
    
    def is_active(self, comp_id: str) -> bool:
        """Check if a component is active (including degraded)."""
        comp = self.components.get(comp_id)
        if not comp:
            return False
        return comp.state in (ComponentState.ACTIVE, ComponentState.DEGRADED)
    
    def set_degraded(self, comp_id: str) -> None:
        """Mark a component as degraded."""
        if comp_id in self.components:
            self.components[comp_id].state = ComponentState.DEGRADED
    
    # =========================================================================
    # Graph Queries
    # =========================================================================
    
    def get_publishers(self, topic_id: str) -> List[str]:
        """Get all publishers for a topic."""
        return [p for p in self._publishers.get(topic_id, []) if self.is_active(p)]
    
    def get_subscribers(self, topic_id: str) -> List[str]:
        """Get all subscribers for a topic."""
        return [s for s in self._subscribers.get(topic_id, []) if self.is_active(s)]
    
    def get_routing_brokers(self, topic_id: str) -> List[str]:
        """Get all brokers that route a topic."""
        return [b for b in self._routing.get(topic_id, []) if self.is_active(b)]
    
    def get_hosted_components(self, node_id: str) -> List[str]:
        """Get all components hosted on a node."""
        return self._hosts.get(node_id, [])
    
    def get_host_node(self, comp_id: str) -> Optional[str]:
        """Get the node that hosts a component."""
        return self._hosted_on.get(comp_id)
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get nodes connected to a given node."""
        return [n for n in self._connections.get(node_id, []) if self.is_active(n)]
    
    def get_app_topics(self, app_id: str) -> Tuple[List[str], List[str]]:
        """Get topics an application publishes to and subscribes from."""
        publishes = []
        subscribes = []
        
        for topic_id, publishers in self._publishers.items():
            if app_id in publishers:
                publishes.append(topic_id)
        
        for topic_id, subscribers in self._subscribers.items():
            if app_id in subscribers:
                subscribes.append(topic_id)
        
        return publishes, subscribes
    
    def get_pub_sub_paths(self, active_only: bool = True):  # -> List[Tuple[str, str, str]]
        """
        Get all publisher -> topic -> subscriber paths.
        
        Returns List of (publisher, topic, subscriber) tuples.
        """
        paths = []
        for topic_id in self.topics:
            publishers = [p[0] for p in self._publishers.get(topic_id, [])]
            subscribers = [s[0] for s in self._subscribers.get(topic_id, [])]
            
            if active_only:
                publishers = [p for p in publishers if self.is_active(p)]
                subscribers = [s for s in subscribers if self.is_active(s)]
                if not self.get_routing_brokers(topic_id):
                    continue
            
            for pub in publishers:
                for sub in subscribers:
                    paths.append((pub, topic_id, sub))
        return paths

    def get_weighted_pub_sub_paths(self, active_only: bool = True) -> List[Tuple[str, str, str, float]]:
        """
        Get all publisher -> topic -> subscriber paths with their remaining capacity.
        
        Capacity = min(
            perf(publisher),
            weight(pub->topic),
            max(perf(broker_i) * weight(broker_i->topic)),
            weight(sub->topic),
            perf(subscriber)
        )
        
        Returns:
            List of (publisher, topic, subscriber, capacity)
        """
        paths = []
        
        for topic_id in self.topics:
            topic_info = self.topics[topic_id]
            pubs_raw = self._publishers.get(topic_id, [])
            subs_raw = self._subscribers.get(topic_id, [])
            brokers_raw = self._routing.get(topic_id, [])
            
            # 1. Broker segment capacity (Max of any active broker path)
            broker_capacities = []
            for b_id, b_weight in brokers_raw:
                if not active_only or self.is_active(b_id):
                    b_perf = self.components[b_id].performance
                    broker_capacities.append(b_perf * b_weight)
            
            broker_segment_capacity = max(broker_capacities) if broker_capacities else 0.0
            
            if active_only and broker_segment_capacity <= 0:
                continue
                
            for p_id, p_weight in pubs_raw:
                p_perf = self.components[p_id].performance
                if active_only and p_perf <= 0:
                    continue
                    
                path_prefix_capacity = min(p_perf, p_weight, broker_segment_capacity)
                
                for s_id, s_weight in subs_raw:
                    s_perf = self.components[s_id].performance
                    if active_only and s_perf <= 0:
                        continue
                        
                    capacity = min(path_prefix_capacity, s_weight, s_perf)
                    
                    if not active_only or capacity > 0:
                        paths.append((p_id, topic_id, s_id, capacity))
                        
        return paths

    def count_active_connected_components(self):  # -> int
        """
        Count weakly-connected components in the active subgraph.
        
        Builds a temporary undirected graph from active components and their
        active relationships, then counts connected components. Used by
        FailureSimulator to compute true graph fragmentation rather than
        simple component loss ratio.
        
        Returns:
            Number of weakly-connected components among active components.
            Returns 0 if no active components exist.
        """
        import networkx as nx
        
        # Build undirected graph of active components
        active_graph = nx.Graph()
        
        # Add all active non-Topic components as nodes
        for comp_id, comp in self.components.items():
            if comp.state == ComponentState.ACTIVE and comp.type in ("Application", "Broker", "Node"):
                active_graph.add_node(comp_id)
        
        if len(active_graph) == 0:
            return 0
        
        # Add edges for active relationships
        # RUNS_ON: app/broker <-> node
        for comp_id, node_id in self._hosted_on.items():
            if comp_id in active_graph and node_id in active_graph:
                active_graph.add_edge(comp_id, node_id)
        
        # CONNECTS_TO: node <-> node
        for node_id, connected in self._connections.items():
            for neighbor_id in connected:
                if node_id in active_graph and neighbor_id in active_graph:
                    active_graph.add_edge(node_id, neighbor_id)
        
        # Pub/Sub paths through topics (app <-> app via shared topic)
        for topic_id in self.topics:
            active_pubs = [p for p in self._publishers.get(topic_id, []) if p in active_graph]
            active_subs = [s for s in self._subscribers.get(topic_id, []) if s in active_graph]
            active_brokers = [b for b in self._routing.get(topic_id, []) if b in active_graph]
            
            # Connect publishers and subscribers through brokers for this topic
            for pub in active_pubs:
                for broker in active_brokers:
                    active_graph.add_edge(pub, broker)
            for sub in active_subs:
                for broker in active_brokers:
                    active_graph.add_edge(sub, broker)
        
        return nx.number_connected_components(active_graph)
    
    def get_message_path(self, publisher: str, topic_id: str) -> List[Tuple[str, str]]:
        """
        Get the path a message takes from publisher to subscribers.
        
        Returns:
            List of (from, to) tuples representing the path
        """
        path = []
        
        # Publisher -> Topic
        path.append((publisher, topic_id))
        
        # Topic -> Broker(s)
        brokers = self.get_routing_brokers(topic_id)
        for broker in brokers:
            path.append((topic_id, broker))
        
        # Broker(s) -> Subscribers
        subscribers = self.get_subscribers(topic_id)
        for broker in brokers:
            for sub in subscribers:
                path.append((broker, sub))
        
        return path

    def get_library_usage(self) -> Dict[str, List[str]]:
        """
        Get library usage for all components.
        
        Returns:
            Dict mapping component ID to list of library IDs
        """
        return dict(self._uses)

    def get_node_allocations(self) -> Dict[str, List[str]]:
        """
        Get node allocations (Node -> [Apps]).
        
        Returns:
            Dict mapping node ID to list of allocated component IDs
        """
        return dict(self._hosts)

    def get_broker_routing(self) -> Dict[str, List[str]]:
        """
        Get broker routing (Broker -> [Topics]).
        
        Returns:
            Dict mapping broker ID to list of routed topic IDs
        """
        return dict(self._routing)
    
    # =========================================================================
    # Layer Filtering
    # =========================================================================
    
    def get_components_by_layer(self, layer: str) -> List[str]:
        """
        Get component IDs included in a specific layer's simulation graph.
        
        Layers:
            - app: Application, Topic, Library components
            - infra: Node, Application, Broker components
            - mw: Broker, Topic, Application components
            - system: All components
        
        Args:
            layer: Layer name (app, infra, mw, system) or string alias
            
        Returns:
            List of component IDs included in the layer's graph
        """
        try:
            sim_layer = SimulationLayer.from_string(layer)
        except ValueError:
            self.logger.warning(f"Unknown layer '{layer}', defaulting to 'system'")
            sim_layer = SimulationLayer.SYSTEM
        
        layer_def = SIMULATION_LAYERS[sim_layer]
        return [c.id for c in self.components.values() if c.type in layer_def.component_types]
    
    def get_analyze_components_by_layer(self, layer: str) -> List[str]:
        """
        Get component IDs to analyze/report for a specific layer.
        
        This returns only the components that should be analyzed,
        not all components in the simulation graph.
        
        Args:
            layer: Layer name (app, infra, mw, system)
            
        Returns:
            List of component IDs to analyze
        """
        try:
            sim_layer = SimulationLayer.from_string(layer)
        except ValueError:
            sim_layer = SimulationLayer.SYSTEM
        
        layer_def = SIMULATION_LAYERS[sim_layer]
        return [c.id for c in self.components.values() if c.type in layer_def.analyze_types]
    
    def get_layer_relationships(self, layer: str) -> FrozenSet[str]:
        """
        Get the relationship types to traverse for a specific layer.
        
        Args:
            layer: Layer name (app, infra, mw, system)
            
        Returns:
            FrozenSet of relationship type names
        """
        try:
            sim_layer = SimulationLayer.from_string(layer)
        except ValueError:
            sim_layer = SimulationLayer.SYSTEM
        
        return SIMULATION_LAYERS[sim_layer].relationships
    
    def get_layer_cascade_rules(self, layer: str) -> FrozenSet[str]:
        """
        Get the cascade rules for failure propagation in a specific layer.
        
        Args:
            layer: Layer name (app, infra, mw, system)
            
        Returns:
            FrozenSet of cascade rule names (physical, logical, network)
        """
        try:
            sim_layer = SimulationLayer.from_string(layer)
        except ValueError:
            sim_layer = SimulationLayer.SYSTEM
        
        return SIMULATION_LAYERS[sim_layer].cascade_rules
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the graph."""
        type_counts = defaultdict(int)
        for comp in self.components.values():
            type_counts[comp.type] += 1
        
        return {
            "total_nodes": len(self.components),
            "total_edges": self.graph.number_of_edges(),
            "component_types": dict(type_counts),
            "topics": len(self.topics),
            "pub_sub_paths": len(self.get_pub_sub_paths()),
            "active_components": sum(1 for c in self.components.values() if c.state == ComponentState.ACTIVE),
        }
