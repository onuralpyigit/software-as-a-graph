"""
Simulation Graph Model - Version 5.0

Graph model optimized for failure and event-driven simulation.
Supports loading from JSON, dictionary, or Neo4j database.

Key Features:
- Multi-layer graph representation (Application, Broker, Topic, Node)
- Original edge types (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO)
- Component-type specific simulation support
- NO derived DEPENDS_ON relationships for simulation
- Neo4j integration for live data retrieval

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Tuple, Iterator
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


# =============================================================================
# Enums
# =============================================================================

class ComponentType(Enum):
    """Types of components in the pub-sub system"""
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"
    
    @classmethod
    def from_string(cls, s: str) -> 'ComponentType':
        """Create from string (case-insensitive)"""
        mapping = {t.value.lower(): t for t in cls}
        return mapping.get(s.lower(), cls.APPLICATION)


class EdgeType(Enum):
    """Original edge types (NOT derived DEPENDS_ON)"""
    PUBLISHES_TO = "PUBLISHES_TO"      # Application -> Topic
    SUBSCRIBES_TO = "SUBSCRIBES_TO"    # Application -> Topic
    ROUTES = "ROUTES"                  # Broker -> Topic
    RUNS_ON = "RUNS_ON"                # Application/Broker -> Node
    CONNECTS_TO = "CONNECTS_TO"        # Node -> Node, Broker -> Broker
    
    @classmethod
    def from_string(cls, s: str) -> 'EdgeType':
        """Create from string"""
        mapping = {t.value: t for t in cls}
        return mapping.get(s, cls.PUBLISHES_TO)


class ComponentStatus(Enum):
    """Component operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QoSPolicy:
    """Quality of Service policy for edges"""
    reliability: str = "best_effort"  # best_effort, reliable
    durability: str = "volatile"      # volatile, transient, persistent
    priority: int = 0                 # 0-10 (higher = more critical)
    bandwidth: float = 1.0            # Relative bandwidth capacity
    latency_ms: float = 10.0          # Expected latency
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reliability": self.reliability,
            "durability": self.durability,
            "priority": self.priority,
            "bandwidth": self.bandwidth,
            "latency_ms": self.latency_ms,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QoSPolicy':
        return cls(
            reliability=d.get("reliability", "best_effort"),
            durability=d.get("durability", "volatile"),
            priority=d.get("priority", 0),
            bandwidth=d.get("bandwidth", 1.0),
            latency_ms=d.get("latency_ms", 10.0),
        )


@dataclass
class Component:
    """A component in the pub-sub system"""
    id: str
    type: ComponentType
    name: str = ""
    status: ComponentStatus = ComponentStatus.HEALTHY
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = self.id
        if isinstance(self.type, str):
            self.type = ComponentType.from_string(self.type)
        if isinstance(self.status, str):
            self.status = ComponentStatus(self.status)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "status": self.status.value,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Component':
        return cls(
            id=d["id"],
            type=ComponentType.from_string(d.get("type", "Application")),
            name=d.get("name", d["id"]),
            status=ComponentStatus(d.get("status", "healthy")),
            properties=d.get("properties", {}),
        )


@dataclass
class Edge:
    """An edge (connection) between components"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType.from_string(self.edge_type)
    
    @property
    def key(self) -> Tuple[str, str, str]:
        """Unique key for this edge"""
        return (self.source, self.target, self.edge_type.value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "weight": self.weight,
            "qos": self.qos.to_dict(),
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Edge':
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=EdgeType.from_string(d.get("type", "PUBLISHES_TO")),
            weight=d.get("weight", 1.0),
            qos=QoSPolicy.from_dict(d.get("qos", {})),
            properties=d.get("properties", {}),
        )


# =============================================================================
# Simulation Graph
# =============================================================================

class SimulationGraph:
    """
    Graph model optimized for simulation.
    
    Uses ORIGINAL edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
    NOT derived DEPENDS_ON relationships.
    
    This allows simulation to follow actual message flow paths
    through the pub-sub system.
    """
    
    def __init__(self):
        self._components: Dict[str, Component] = {}
        self._edges: Dict[Tuple[str, str, str], Edge] = {}
        self._graph: Optional[nx.DiGraph] = None
        
        # Indices for fast lookup
        self._by_type: Dict[ComponentType, Set[str]] = defaultdict(set)
        self._outgoing: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self._incoming: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        self._by_edge_type: Dict[EdgeType, Set[Tuple[str, str, str]]] = defaultdict(set)
        
        self._logger = logging.getLogger(__name__)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def components(self) -> Dict[str, Component]:
        """All components by ID"""
        return self._components
    
    @property
    def edges(self) -> Dict[Tuple[str, str, str], Edge]:
        """All edges by key (source, target, type)"""
        return self._edges
    
    @property
    def connections(self) -> List[Edge]:
        """List of all edges (alias for compatibility)"""
        return list(self._edges.values())
    
    @property
    def nx_graph(self) -> 'nx.DiGraph':
        """NetworkX graph representation (lazy initialized)"""
        if self._graph is None:
            self._build_nx_graph()
        return self._graph
    
    # =========================================================================
    # Component Management
    # =========================================================================
    
    def add_component(self, component: Component) -> None:
        """Add a component to the graph"""
        self._components[component.id] = component
        self._by_type[component.type].add(component.id)
        self._graph = None  # Invalidate cache
    
    def remove_component(self, component_id: str) -> Optional[Component]:
        """Remove a component and its edges"""
        if component_id not in self._components:
            return None
        
        component = self._components.pop(component_id)
        self._by_type[component.type].discard(component_id)
        
        # Remove connected edges
        edges_to_remove = list(self._outgoing[component_id] | self._incoming[component_id])
        for edge_key in edges_to_remove:
            self._remove_edge_by_key(edge_key)
        
        self._graph = None
        return component
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID"""
        return self._components.get(component_id)
    
    def get_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Get all components of a specific type"""
        return [self._components[cid] for cid in self._by_type.get(comp_type, set())]
    
    def get_component_ids_by_type(self, comp_type: ComponentType) -> Set[str]:
        """Get component IDs of a specific type"""
        return self._by_type.get(comp_type, set()).copy()
    
    # =========================================================================
    # Edge Management
    # =========================================================================
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        key = edge.key
        self._edges[key] = edge
        self._outgoing[edge.source].add(key)
        self._incoming[edge.target].add(key)
        self._by_edge_type[edge.edge_type].add(key)
        self._graph = None
    
    def _remove_edge_by_key(self, key: Tuple[str, str, str]) -> Optional[Edge]:
        """Remove an edge by its key"""
        if key not in self._edges:
            return None
        
        edge = self._edges.pop(key)
        self._outgoing[edge.source].discard(key)
        self._incoming[edge.target].discard(key)
        self._by_edge_type[edge.edge_type].discard(key)
        self._graph = None
        return edge
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type"""
        return [self._edges[key] for key in self._by_edge_type.get(edge_type, set())]
    
    def get_outgoing_edges(self, component_id: str) -> List[Edge]:
        """Get edges originating from a component"""
        return [self._edges[key] for key in self._outgoing.get(component_id, set())]
    
    def get_incoming_edges(self, component_id: str) -> List[Edge]:
        """Get edges targeting a component"""
        return [self._edges[key] for key in self._incoming.get(component_id, set())]
    
    def get_neighbors(self, component_id: str, direction: str = "both") -> Set[str]:
        """Get neighboring component IDs"""
        neighbors = set()
        if direction in ("out", "both"):
            for key in self._outgoing.get(component_id, set()):
                neighbors.add(key[1])  # target
        if direction in ("in", "both"):
            for key in self._incoming.get(component_id, set()):
                neighbors.add(key[0])  # source
        return neighbors
    
    # =========================================================================
    # Pub-Sub Specific Queries
    # =========================================================================
    
    def get_publishers(self, topic_id: str) -> Set[str]:
        """Get applications that publish to a topic"""
        publishers = set()
        for key in self._incoming.get(topic_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.PUBLISHES_TO:
                publishers.add(edge.source)
        return publishers
    
    def get_subscribers(self, topic_id: str) -> Set[str]:
        """Get applications that subscribe to a topic"""
        subscribers = set()
        for key in self._incoming.get(topic_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.SUBSCRIBES_TO:
                subscribers.add(edge.source)
        return subscribers
    
    def get_topics_published_by(self, app_id: str) -> Set[str]:
        """Get topics an application publishes to"""
        topics = set()
        for key in self._outgoing.get(app_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.PUBLISHES_TO:
                topics.add(edge.target)
        return topics
    
    def get_topics_subscribed_by(self, app_id: str) -> Set[str]:
        """Get topics an application subscribes to"""
        topics = set()
        for key in self._outgoing.get(app_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.SUBSCRIBES_TO:
                topics.add(edge.target)
        return topics
    
    def get_broker_for_topic(self, topic_id: str) -> Optional[str]:
        """Get the broker that routes a topic"""
        for key in self._incoming.get(topic_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.ROUTES:
                return edge.source
        return None
    
    def get_node_for_component(self, component_id: str) -> Optional[str]:
        """Get the node a component runs on"""
        for key in self._outgoing.get(component_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.RUNS_ON:
                return edge.target
        return None
    
    def get_components_on_node(self, node_id: str) -> Set[str]:
        """Get all components running on a node"""
        components = set()
        for key in self._incoming.get(node_id, set()):
            edge = self._edges[key]
            if edge.edge_type == EdgeType.RUNS_ON:
                components.add(edge.source)
        return components
    
    # =========================================================================
    # Message Flow Analysis
    # =========================================================================
    
    def trace_message_path(self, publisher_id: str, topic_id: str) -> List[List[str]]:
        """
        Trace all paths a message takes from publisher through topic to subscribers.
        Returns list of paths: [[publisher, topic, subscriber1], [publisher, topic, subscriber2], ...]
        """
        paths = []
        
        # Verify publisher publishes to topic
        if topic_id not in self.get_topics_published_by(publisher_id):
            return paths
        
        # Get broker routing the topic
        broker = self.get_broker_for_topic(topic_id)
        
        # Get all subscribers
        subscribers = self.get_subscribers(topic_id)
        
        for subscriber in subscribers:
            if broker:
                paths.append([publisher_id, broker, topic_id, subscriber])
            else:
                paths.append([publisher_id, topic_id, subscriber])
        
        return paths
    
    def get_downstream_components(self, component_id: str, 
                                   visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Get all components that depend on this component (downstream).
        Follows message flow direction.
        """
        if visited is None:
            visited = set()
        
        if component_id in visited:
            return set()
        
        visited.add(component_id)
        downstream = set()
        
        component = self.get_component(component_id)
        if not component:
            return downstream
        
        # Different propagation rules based on component type
        if component.type == ComponentType.APPLICATION:
            # Publishers affect their topics
            for topic_id in self.get_topics_published_by(component_id):
                downstream.add(topic_id)
                downstream.update(self.get_downstream_components(topic_id, visited))
        
        elif component.type == ComponentType.TOPIC:
            # Topics affect their subscribers
            for subscriber_id in self.get_subscribers(component_id):
                downstream.add(subscriber_id)
                downstream.update(self.get_downstream_components(subscriber_id, visited))
        
        elif component.type == ComponentType.BROKER:
            # Brokers affect all topics they route
            for key in self._outgoing.get(component_id, set()):
                edge = self._edges[key]
                if edge.edge_type == EdgeType.ROUTES:
                    downstream.add(edge.target)
                    downstream.update(self.get_downstream_components(edge.target, visited))
        
        elif component.type == ComponentType.NODE:
            # Nodes affect all components running on them
            for comp_id in self.get_components_on_node(component_id):
                downstream.add(comp_id)
                downstream.update(self.get_downstream_components(comp_id, visited))
        
        return downstream
    
    # =========================================================================
    # NetworkX Integration
    # =========================================================================
    
    def _build_nx_graph(self) -> None:
        """Build NetworkX graph representation"""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for graph operations")
        
        self._graph = nx.DiGraph()
        
        # Add nodes
        for comp_id, comp in self._components.items():
            self._graph.add_node(
                comp_id,
                type=comp.type.value,
                name=comp.name,
                status=comp.status.value,
                **comp.properties
            )
        
        # Add edges
        for edge in self._edges.values():
            self._graph.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                **edge.properties
            )
    
    def get_subgraph_by_type(self, comp_type: ComponentType) -> 'SimulationGraph':
        """Extract subgraph containing only components of a specific type"""
        subgraph = SimulationGraph()
        
        component_ids = self._by_type.get(comp_type, set())
        
        for comp_id in component_ids:
            subgraph.add_component(self._components[comp_id])
        
        # Add edges between components of this type
        for edge in self._edges.values():
            if edge.source in component_ids and edge.target in component_ids:
                subgraph.add_edge(edge)
        
        return subgraph
    
    def get_subgraph_by_edge_type(self, edge_type: EdgeType) -> 'SimulationGraph':
        """Extract subgraph containing only edges of a specific type"""
        subgraph = SimulationGraph()
        
        # Collect components involved in these edges
        involved_components = set()
        for key in self._by_edge_type.get(edge_type, set()):
            involved_components.add(key[0])  # source
            involved_components.add(key[1])  # target
        
        # Add components
        for comp_id in involved_components:
            if comp_id in self._components:
                subgraph.add_component(self._components[comp_id])
        
        # Add edges
        for key in self._by_edge_type.get(edge_type, set()):
            subgraph.add_edge(self._edges[key])
        
        return subgraph
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            "total_components": len(self._components),
            "total_edges": len(self._edges),
            "by_component_type": {},
            "by_edge_type": {},
        }
        
        for comp_type in ComponentType:
            count = len(self._by_type.get(comp_type, set()))
            if count > 0:
                stats["by_component_type"][comp_type.value] = count
        
        for edge_type in EdgeType:
            count = len(self._by_edge_type.get(edge_type, set()))
            if count > 0:
                stats["by_edge_type"][edge_type.value] = count
        
        return stats
    
    def summary(self) -> str:
        """Get human-readable summary"""
        stats = self.get_statistics()
        lines = [
            f"SimulationGraph:",
            f"  Components: {stats['total_components']}",
            f"  Edges: {stats['total_edges']}",
            f"  By Type:",
        ]
        for comp_type, count in stats["by_component_type"].items():
            lines.append(f"    {comp_type}: {count}")
        lines.append("  By Edge Type:")
        for edge_type, count in stats["by_edge_type"].items():
            lines.append(f"    {edge_type}: {count}")
        return "\n".join(lines)
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "components": [c.to_dict() for c in self._components.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "statistics": self.get_statistics(),
        }
    
    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationGraph':
        """Create from dictionary"""
        graph = cls()
        
        # Handle different input formats
        components = data.get("components", data.get("vertices", []))
        edges = data.get("edges", data.get("connections", []))
        
        # Add components
        for comp_data in components:
            # Handle both flat and nested formats
            if "id" not in comp_data and "name" in comp_data:
                comp_data["id"] = comp_data["name"]
            graph.add_component(Component.from_dict(comp_data))
        
        # Add edges
        for edge_data in edges:
            # Handle different edge formats
            if "source" not in edge_data:
                if "from" in edge_data:
                    edge_data["source"] = edge_data["from"]
                elif "publisher" in edge_data:
                    edge_data["source"] = edge_data["publisher"]
            if "target" not in edge_data:
                if "to" in edge_data:
                    edge_data["target"] = edge_data["to"]
                elif "subscriber" in edge_data:
                    edge_data["target"] = edge_data["subscriber"]
            
            graph.add_edge(Edge.from_dict(edge_data))
        
        return graph
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SimulationGraph':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_generated(cls, generated_data: Dict[str, Any]) -> 'SimulationGraph':
        """
        Create from generated graph data (from src.core.generate_graph).
        Maps the generated format to simulation format.
        """
        graph = cls()
        
        # Map vertices
        vertices = generated_data.get("vertices", {})
        
        # Applications
        for app in vertices.get("applications", []):
            graph.add_component(Component(
                id=app.get("id", app.get("name")),
                type=ComponentType.APPLICATION,
                name=app.get("name", ""),
                properties=app.get("properties", {}),
            ))
        
        # Brokers
        for broker in vertices.get("brokers", []):
            graph.add_component(Component(
                id=broker.get("id", broker.get("name")),
                type=ComponentType.BROKER,
                name=broker.get("name", ""),
                properties=broker.get("properties", {}),
            ))
        
        # Topics
        for topic in vertices.get("topics", []):
            graph.add_component(Component(
                id=topic.get("id", topic.get("name")),
                type=ComponentType.TOPIC,
                name=topic.get("name", ""),
                properties=topic.get("properties", {}),
            ))
        
        # Nodes
        for node in vertices.get("nodes", []):
            graph.add_component(Component(
                id=node.get("id", node.get("name")),
                type=ComponentType.NODE,
                name=node.get("name", ""),
                properties=node.get("properties", {}),
            ))
        
        # Map edges
        edges = generated_data.get("edges", {})
        
        # PUBLISHES_TO
        for pub in edges.get("publishes_to", []):
            graph.add_edge(Edge(
                source=pub.get("source", pub.get("application")),
                target=pub.get("target", pub.get("topic")),
                edge_type=EdgeType.PUBLISHES_TO,
                weight=pub.get("weight", 1.0),
                qos=QoSPolicy.from_dict(pub.get("qos", {})),
            ))
        
        # SUBSCRIBES_TO
        for sub in edges.get("subscribes_to", []):
            graph.add_edge(Edge(
                source=sub.get("source", sub.get("application")),
                target=sub.get("target", sub.get("topic")),
                edge_type=EdgeType.SUBSCRIBES_TO,
                weight=sub.get("weight", 1.0),
                qos=QoSPolicy.from_dict(sub.get("qos", {})),
            ))
        
        # ROUTES
        for route in edges.get("routes", []):
            graph.add_edge(Edge(
                source=route.get("source", route.get("broker")),
                target=route.get("target", route.get("topic")),
                edge_type=EdgeType.ROUTES,
                weight=route.get("weight", 1.0),
            ))
        
        # RUNS_ON
        for runs in edges.get("runs_on", []):
            graph.add_edge(Edge(
                source=runs.get("source", runs.get("component")),
                target=runs.get("target", runs.get("node")),
                edge_type=EdgeType.RUNS_ON,
                weight=runs.get("weight", 1.0),
            ))
        
        # CONNECTS_TO
        for conn in edges.get("connects_to", []):
            graph.add_edge(Edge(
                source=conn.get("source"),
                target=conn.get("target"),
                edge_type=EdgeType.CONNECTS_TO,
                weight=conn.get("weight", 1.0),
            ))
        
        return graph


# =============================================================================
# Factory Functions
# =============================================================================

def create_simulation_graph(data: Any) -> SimulationGraph:
    """
    Factory function to create SimulationGraph from various sources.
    
    Args:
        data: Can be:
            - str: Path to JSON file
            - dict: Graph data dictionary
            - SimulationGraph: Returns as-is
    """
    if isinstance(data, SimulationGraph):
        return data
    
    if isinstance(data, str):
        return SimulationGraph.from_json(data)
    
    if isinstance(data, dict):
        # Detect format and use appropriate loader
        if "vertices" in data and isinstance(data["vertices"], dict):
            return SimulationGraph.from_generated(data)
        return SimulationGraph.from_dict(data)
    
    raise ValueError(f"Cannot create SimulationGraph from {type(data)}")
