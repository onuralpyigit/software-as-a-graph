"""
Simulation Graph - Version 5.0

Graph model for simulation using ORIGINAL edge types.

Key Design Decision:
    Uses original pub-sub relationships (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
    NOT derived DEPENDS_ON relationships. This enables accurate simulation
    of actual message flow paths through the system.

Edge Types:
    - PUBLISHES_TO: Application -> Topic (message source)
    - SUBSCRIBES_TO: Application -> Topic (message sink)
    - ROUTES: Broker -> Topic (routing responsibility)
    - RUNS_ON: Component -> Node (deployment)
    - CONNECTS_TO: Node -> Broker (infrastructure)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import json
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Iterator, Tuple
from pathlib import Path


# =============================================================================
# Enums
# =============================================================================

class ComponentType(Enum):
    """Types of components in the pub-sub system."""
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"


class EdgeType(Enum):
    """Original edge types (NOT DEPENDS_ON)."""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"


class ComponentStatus(Enum):
    """Runtime status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QoSPolicy:
    """Quality of Service policy for message delivery."""
    reliability: str = "BEST_EFFORT"  # RELIABLE, BEST_EFFORT
    durability: str = "VOLATILE"       # PERSISTENT, TRANSIENT, VOLATILE
    priority: str = "MEDIUM"           # URGENT, HIGH, MEDIUM, LOW
    
    def criticality_score(self) -> float:
        """Calculate criticality score (0.0 - 1.0)."""
        reliability_scores = {"RELIABLE": 0.5, "BEST_EFFORT": 0.0}
        durability_scores = {"PERSISTENT": 0.3, "TRANSIENT": 0.15, "VOLATILE": 0.0}
        priority_scores = {"URGENT": 0.2, "HIGH": 0.15, "MEDIUM": 0.1, "LOW": 0.0}
        
        return (
            reliability_scores.get(self.reliability, 0.0) +
            durability_scores.get(self.durability, 0.0) +
            priority_scores.get(self.priority, 0.0)
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "reliability": self.reliability,
            "durability": self.durability,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QoSPolicy:
        return cls(
            reliability=data.get("reliability", "BEST_EFFORT"),
            durability=data.get("durability", "VOLATILE"),
            priority=data.get("priority", "MEDIUM"),
        )


@dataclass
class Component:
    """A component in the pub-sub system."""
    id: str
    type: ComponentType
    name: str = ""
    status: ComponentStatus = ComponentStatus.HEALTHY
    node_id: Optional[str] = None  # Which node it runs on
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name or self.id,
            "status": self.status.value,
            "node_id": self.node_id,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Component:
        return cls(
            id=data["id"],
            type=ComponentType(data.get("type", "Application")),
            name=data.get("name", data["id"]),
            status=ComponentStatus(data.get("status", "healthy")),
            node_id=data.get("node_id"),
            properties=data.get("properties", {}),
        )


@dataclass
class Edge:
    """An edge in the simulation graph."""
    source: str
    target: str
    edge_type: EdgeType
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "qos": self.qos.to_dict(),
            "weight": self.weight,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=EdgeType(data.get("type", "PUBLISHES_TO")),
            qos=QoSPolicy.from_dict(data.get("qos", {})),
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
        )


# =============================================================================
# Simulation Graph
# =============================================================================

class SimulationGraph:
    """
    Graph model for simulation using original edge types.
    
    Provides methods for:
    - Component and edge management
    - Message path queries
    - Layer-specific subgraph extraction
    - Component removal for failure simulation
    
    Example:
        graph = SimulationGraph.from_json("system.json")
        
        # Get publishers for a topic
        publishers = graph.get_publishers("topic1")
        
        # Get message paths
        paths = graph.get_message_paths()
        
        # Extract application layer
        app_layer = graph.get_layer_subgraph("application")
    """
    
    # Layer definitions matching DEPENDS_ON dependency types
    # These layers report on different aspects of system dependencies
    # without actually deriving DEPENDS_ON relationships
    LAYER_DEFINITIONS = {
        "application": {
            "name": "Application Layer (app_to_app)",
            "dependency_type": "app_to_app",
            "component_types": [ComponentType.APPLICATION],
            "edge_types": [EdgeType.PUBLISHES_TO, EdgeType.SUBSCRIBES_TO],
            "description": "Application-to-application dependencies via shared topics",
        },
        "infrastructure": {
            "name": "Infrastructure Layer (node_to_node)",
            "dependency_type": "node_to_node",
            "component_types": [ComponentType.NODE],
            "edge_types": [EdgeType.CONNECTS_TO],
            "description": "Node-to-node dependencies via shared brokers",
        },
        "app_broker": {
            "name": "Application-Broker Layer (app_to_broker)",
            "dependency_type": "app_to_broker",
            "component_types": [ComponentType.APPLICATION, ComponentType.BROKER],
            "edge_types": [EdgeType.PUBLISHES_TO, EdgeType.SUBSCRIBES_TO, EdgeType.ROUTES],
            "description": "Application dependencies on brokers for message routing",
        },
        "node_broker": {
            "name": "Node-Broker Layer (node_to_broker)",
            "dependency_type": "node_to_broker",
            "component_types": [ComponentType.NODE, ComponentType.BROKER],
            "edge_types": [EdgeType.CONNECTS_TO, EdgeType.RUNS_ON],
            "description": "Node dependencies on brokers for connectivity",
        },
    }
    
    def __init__(self):
        """Initialize empty simulation graph."""
        self.components: Dict[str, Component] = {}
        self.edges: List[Edge] = []
        
        # Index structures for fast lookups
        self._outgoing: Dict[str, List[Edge]] = {}
        self._incoming: Dict[str, List[Edge]] = {}
        self._by_type: Dict[ComponentType, Set[str]] = {t: set() for t in ComponentType}
        self._edges_by_type: Dict[EdgeType, List[Edge]] = {t: [] for t in EdgeType}
    
    # =========================================================================
    # Component Management
    # =========================================================================
    
    def add_component(self, component: Component) -> None:
        """Add a component to the graph."""
        self.components[component.id] = component
        self._by_type[component.type].add(component.id)
        
        if component.id not in self._outgoing:
            self._outgoing[component.id] = []
        if component.id not in self._incoming:
            self._incoming[component.id] = []
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return self.components.get(component_id)
    
    def get_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Get all components of a specific type."""
        return [
            self.components[cid] 
            for cid in self._by_type.get(comp_type, set())
        ]
    
    def remove_component(self, component_id: str) -> None:
        """Remove a component and all its edges."""
        if component_id not in self.components:
            return
        
        comp = self.components[component_id]
        
        # Remove from type index
        self._by_type[comp.type].discard(component_id)
        
        # Remove edges
        self.edges = [
            e for e in self.edges 
            if e.source != component_id and e.target != component_id
        ]
        
        # Rebuild edge indexes
        self._rebuild_edge_indexes()
        
        # Remove component
        del self.components[component_id]
    
    # =========================================================================
    # Edge Management
    # =========================================================================
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        
        # Update indexes
        if edge.source not in self._outgoing:
            self._outgoing[edge.source] = []
        self._outgoing[edge.source].append(edge)
        
        if edge.target not in self._incoming:
            self._incoming[edge.target] = []
        self._incoming[edge.target].append(edge)
        
        self._edges_by_type[edge.edge_type].append(edge)
    
    def get_outgoing_edges(self, component_id: str) -> List[Edge]:
        """Get all outgoing edges from a component."""
        return self._outgoing.get(component_id, [])
    
    def get_incoming_edges(self, component_id: str) -> List[Edge]:
        """Get all incoming edges to a component."""
        return self._incoming.get(component_id, [])
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type."""
        return self._edges_by_type.get(edge_type, [])
    
    def _rebuild_edge_indexes(self) -> None:
        """Rebuild all edge indexes."""
        self._outgoing = {}
        self._incoming = {}
        self._edges_by_type = {t: [] for t in EdgeType}
        
        for edge in self.edges:
            if edge.source not in self._outgoing:
                self._outgoing[edge.source] = []
            self._outgoing[edge.source].append(edge)
            
            if edge.target not in self._incoming:
                self._incoming[edge.target] = []
            self._incoming[edge.target].append(edge)
            
            self._edges_by_type[edge.edge_type].append(edge)
    
    # =========================================================================
    # Pub-Sub Queries
    # =========================================================================
    
    def get_publishers(self, topic_id: str) -> Set[str]:
        """Get all publishers for a topic."""
        publishers = set()
        for edge in self._incoming.get(topic_id, []):
            if edge.edge_type == EdgeType.PUBLISHES_TO:
                publishers.add(edge.source)
        return publishers
    
    def get_subscribers(self, topic_id: str) -> Set[str]:
        """Get all subscribers for a topic."""
        subscribers = set()
        for edge in self._incoming.get(topic_id, []):
            if edge.edge_type == EdgeType.SUBSCRIBES_TO:
                subscribers.add(edge.source)
        return subscribers
    
    def get_topics_published_by(self, app_id: str) -> Set[str]:
        """Get topics that an application publishes to."""
        topics = set()
        for edge in self._outgoing.get(app_id, []):
            if edge.edge_type == EdgeType.PUBLISHES_TO:
                topics.add(edge.target)
        return topics
    
    def get_topics_subscribed_by(self, app_id: str) -> Set[str]:
        """Get topics that an application subscribes to."""
        topics = set()
        for edge in self._outgoing.get(app_id, []):
            if edge.edge_type == EdgeType.SUBSCRIBES_TO:
                topics.add(edge.target)
        return topics
    
    def get_broker_for_topic(self, topic_id: str) -> Optional[str]:
        """Get the broker that routes a topic."""
        for edge in self._incoming.get(topic_id, []):
            if edge.edge_type == EdgeType.ROUTES:
                return edge.source
        return None
    
    def get_topics_routed_by(self, broker_id: str) -> Set[str]:
        """Get topics routed by a broker."""
        topics = set()
        for edge in self._outgoing.get(broker_id, []):
            if edge.edge_type == EdgeType.ROUTES:
                topics.add(edge.target)
        return topics
    
    def get_components_on_node(self, node_id: str) -> Set[str]:
        """Get components running on a node."""
        components = set()
        for edge in self._incoming.get(node_id, []):
            if edge.edge_type == EdgeType.RUNS_ON:
                components.add(edge.source)
        return components
    
    def get_node_for_component(self, component_id: str) -> Optional[str]:
        """Get the node a component runs on."""
        for edge in self._outgoing.get(component_id, []):
            if edge.edge_type == EdgeType.RUNS_ON:
                return edge.target
        return None
    
    # =========================================================================
    # Message Path Analysis
    # =========================================================================
    
    def get_message_paths(self) -> List[Tuple[str, str, str]]:
        """
        Get all message paths: (publisher, topic, subscriber).
        
        Returns:
            List of (publisher_id, topic_id, subscriber_id) tuples
        """
        paths = []
        
        for topic_id in self._by_type[ComponentType.TOPIC]:
            publishers = self.get_publishers(topic_id)
            subscribers = self.get_subscribers(topic_id)
            
            for pub in publishers:
                for sub in subscribers:
                    if pub != sub:  # Skip self-loops
                        paths.append((pub, topic_id, sub))
        
        return paths
    
    def get_paths_through_component(self, component_id: str) -> List[Tuple[str, str, str]]:
        """Get message paths that go through a specific component."""
        comp = self.get_component(component_id)
        if not comp:
            return []
        
        paths = []
        
        if comp.type == ComponentType.TOPIC:
            # Topic is directly on the path
            publishers = self.get_publishers(component_id)
            subscribers = self.get_subscribers(component_id)
            for pub in publishers:
                for sub in subscribers:
                    if pub != sub:
                        paths.append((pub, component_id, sub))
        
        elif comp.type == ComponentType.APPLICATION:
            # Application can be publisher or subscriber
            for topic_id in self.get_topics_published_by(component_id):
                for sub in self.get_subscribers(topic_id):
                    if sub != component_id:
                        paths.append((component_id, topic_id, sub))
            
            for topic_id in self.get_topics_subscribed_by(component_id):
                for pub in self.get_publishers(topic_id):
                    if pub != component_id:
                        paths.append((pub, topic_id, component_id))
        
        elif comp.type == ComponentType.BROKER:
            # Broker affects all topics it routes
            for topic_id in self.get_topics_routed_by(component_id):
                publishers = self.get_publishers(topic_id)
                subscribers = self.get_subscribers(topic_id)
                for pub in publishers:
                    for sub in subscribers:
                        if pub != sub:
                            paths.append((pub, topic_id, sub))
        
        elif comp.type == ComponentType.NODE:
            # Node affects components running on it
            for comp_id in self.get_components_on_node(component_id):
                paths.extend(self.get_paths_through_component(comp_id))
        
        return paths
    
    # =========================================================================
    # Layer Subgraphs
    # =========================================================================
    
    def get_layer_subgraph(self, layer: str) -> SimulationGraph:
        """
        Extract a subgraph for a specific layer.
        
        Args:
            layer: Layer name (application, infrastructure, app_broker, node_broker)
        
        Returns:
            New SimulationGraph containing only the layer components and edges
        """
        if layer not in self.LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}")
        
        layer_def = self.LAYER_DEFINITIONS[layer]
        subgraph = SimulationGraph()
        
        # Add components of matching types
        included_ids = set()
        for comp_type in layer_def["component_types"]:
            for comp_id in self._by_type[comp_type]:
                subgraph.add_component(copy.deepcopy(self.components[comp_id]))
                included_ids.add(comp_id)
        
        # For pub-sub layers, also include topics
        if layer in ["application", "app_broker"]:
            for comp_id in self._by_type[ComponentType.TOPIC]:
                subgraph.add_component(copy.deepcopy(self.components[comp_id]))
                included_ids.add(comp_id)
        
        # Add edges of matching types where both endpoints are included
        for edge_type in layer_def["edge_types"]:
            for edge in self._edges_by_type[edge_type]:
                if edge.source in included_ids and edge.target in included_ids:
                    subgraph.add_edge(copy.deepcopy(edge))
        
        return subgraph
    
    def get_layer_stats(self, layer: str) -> Dict[str, Any]:
        """Get statistics for a specific layer."""
        subgraph = self.get_layer_subgraph(layer)
        
        return {
            "layer": layer,
            "name": self.LAYER_DEFINITIONS[layer]["name"],
            "components": len(subgraph.components),
            "edges": len(subgraph.edges),
            "by_type": {
                t.value: len(subgraph._by_type[t])
                for t in ComponentType
                if subgraph._by_type[t]
            },
        }
    
    # =========================================================================
    # Graph Operations
    # =========================================================================
    
    def copy(self) -> SimulationGraph:
        """Create a deep copy of the graph."""
        new_graph = SimulationGraph()
        
        for comp in self.components.values():
            new_graph.add_component(copy.deepcopy(comp))
        
        for edge in self.edges:
            new_graph.add_edge(copy.deepcopy(edge))
        
        return new_graph
    
    def summary(self) -> Dict[str, Any]:
        """Get graph summary statistics."""
        return {
            "total_components": len(self.components),
            "total_edges": len(self.edges),
            "by_component_type": {
                t.value: len(ids) for t, ids in self._by_type.items()
            },
            "by_edge_type": {
                t.value: len(edges) for t, edges in self._edges_by_type.items()
            },
            "message_paths": len(self.get_message_paths()),
        }
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "components": [c.to_dict() for c in self.components.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationGraph:
        """Create from dictionary."""
        graph = cls()
        
        # Load components (handle different formats)
        components_data = data.get("components", [])
        if isinstance(components_data, dict):
            # Handle dict format (keyed by type)
            for type_key, comp_list in components_data.items():
                for comp_data in comp_list:
                    if "type" not in comp_data:
                        # Infer type from key
                        type_map = {
                            "applications": "Application",
                            "brokers": "Broker",
                            "topics": "Topic",
                            "nodes": "Node",
                        }
                        comp_data["type"] = type_map.get(type_key, "Application")
                    graph.add_component(Component.from_dict(comp_data))
        else:
            # Handle list format
            for comp_data in components_data:
                graph.add_component(Component.from_dict(comp_data))
        
        # Load edges (handle different formats)
        edges_data = data.get("edges", data.get("relationships", []))
        if isinstance(edges_data, dict):
            # Handle dict format (keyed by type)
            for type_key, edge_list in edges_data.items():
                for edge_data in edge_list:
                    if "type" not in edge_data:
                        edge_data["type"] = type_key.upper()
                    graph.add_edge(Edge.from_dict(edge_data))
        else:
            # Handle list format
            for edge_data in edges_data:
                graph.add_edge(Edge.from_dict(edge_data))
        
        return graph
    
    @classmethod
    def from_json(cls, path: str) -> SimulationGraph:
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, path: str, indent: int = 2) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    @classmethod
    def from_core_model(cls, graph_model) -> SimulationGraph:
        """
        Create SimulationGraph from src.core.GraphModel.
        
        Converts the core graph model to simulation format.
        """
        sim_graph = cls()
        
        # Convert vertices to components
        for vertex in graph_model.vertices.values():
            comp = Component(
                id=vertex.id,
                type=ComponentType(vertex.type.value),
                name=vertex.name,
            )
            
            # Extract node_id from Application vertices
            if hasattr(vertex, 'node_id'):
                comp.node_id = vertex.node_id
            
            sim_graph.add_component(comp)
        
        # Convert edges
        for edge in graph_model.edges:
            sim_edge = Edge(
                source=edge.source,
                target=edge.target,
                edge_type=EdgeType(edge.type.value),
                weight=edge.weight,
            )
            
            # Extract QoS if available
            if hasattr(edge, 'qos') and edge.qos:
                sim_edge.qos = QoSPolicy(
                    reliability=edge.qos.reliability,
                    durability=edge.qos.durability,
                    priority=edge.qos.priority,
                )
            
            sim_graph.add_edge(sim_edge)
        
        return sim_graph


# =============================================================================
# Factory Functions
# =============================================================================

def create_simulation_graph(
    applications: int = 5,
    brokers: int = 1,
    topics: int = 10,
    nodes: int = 2,
    seed: Optional[int] = None,
) -> SimulationGraph:
    """
    Create a simple simulation graph for testing.
    
    Args:
        applications: Number of applications
        brokers: Number of brokers
        topics: Number of topics
        nodes: Number of nodes
        seed: Random seed
    
    Returns:
        SimulationGraph with generated components and edges
    """
    import random
    rng = random.Random(seed)
    
    graph = SimulationGraph()
    
    # Create nodes
    node_ids = []
    for i in range(nodes):
        node_id = f"node_{i+1}"
        node_ids.append(node_id)
        graph.add_component(Component(
            id=node_id,
            type=ComponentType.NODE,
            name=f"Node {i+1}",
        ))
    
    # Create brokers
    broker_ids = []
    for i in range(brokers):
        broker_id = f"broker_{i+1}"
        broker_ids.append(broker_id)
        node = rng.choice(node_ids)
        graph.add_component(Component(
            id=broker_id,
            type=ComponentType.BROKER,
            name=f"Broker {i+1}",
            node_id=node,
        ))
        # Add RUNS_ON edge
        graph.add_edge(Edge(
            source=broker_id,
            target=node,
            edge_type=EdgeType.RUNS_ON,
        ))
    
    # Create topics
    topic_ids = []
    for i in range(topics):
        topic_id = f"topic_{i+1}"
        topic_ids.append(topic_id)
        graph.add_component(Component(
            id=topic_id,
            type=ComponentType.TOPIC,
            name=f"Topic {i+1}",
        ))
        # Route to random broker
        broker = rng.choice(broker_ids)
        graph.add_edge(Edge(
            source=broker,
            target=topic_id,
            edge_type=EdgeType.ROUTES,
        ))
    
    # Create applications
    app_ids = []
    for i in range(applications):
        app_id = f"app_{i+1}"
        app_ids.append(app_id)
        node = rng.choice(node_ids)
        graph.add_component(Component(
            id=app_id,
            type=ComponentType.APPLICATION,
            name=f"Application {i+1}",
            node_id=node,
        ))
        # Add RUNS_ON edge
        graph.add_edge(Edge(
            source=app_id,
            target=node,
            edge_type=EdgeType.RUNS_ON,
        ))
        
        # Add pub/sub relationships
        pub_topics = rng.sample(topic_ids, k=min(3, len(topic_ids)))
        sub_topics = rng.sample(topic_ids, k=min(3, len(topic_ids)))
        
        for topic in pub_topics:
            graph.add_edge(Edge(
                source=app_id,
                target=topic,
                edge_type=EdgeType.PUBLISHES_TO,
                qos=QoSPolicy(
                    reliability=rng.choice(["RELIABLE", "BEST_EFFORT"]),
                    durability=rng.choice(["PERSISTENT", "TRANSIENT", "VOLATILE"]),
                    priority=rng.choice(["URGENT", "HIGH", "MEDIUM", "LOW"]),
                ),
            ))
        
        for topic in sub_topics:
            graph.add_edge(Edge(
                source=app_id,
                target=topic,
                edge_type=EdgeType.SUBSCRIBES_TO,
            ))
    
    # Add CONNECTS_TO edges between nodes and brokers
    for node_id in node_ids:
        for broker_id in broker_ids:
            graph.add_edge(Edge(
                source=node_id,
                target=broker_id,
                edge_type=EdgeType.CONNECTS_TO,
            ))
    
    return graph
