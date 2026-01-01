"""
Graph Model - Version 5.0

Simplified, clean data models for pub-sub system graphs with:
- Clear vertex and edge types with weight properties
- QoS-aware weight calculation for DEPENDS_ON relationships
- Component criticality weight derived from relationship weights
- Immutable dataclasses for safety
- Comprehensive query methods

Vertices (with weight property):
    Application: {id, name, role, weight}
    Broker: {id, name, weight}
    Topic: {id, name, size, qos, weight}
    Node: {id, name, weight}

Edges:
    PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
    DEPENDS_ON (derived with weight based on QoS and message size)

Component Weight Calculation:
    weight(v) = sum(incoming_deps) + sum(outgoing_deps)
    
    This captures how critical a component is based on its relationships.
    Components with many high-weight dependencies are more critical.

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Iterator
from collections import defaultdict


# =============================================================================
# Enumerations
# =============================================================================

class VertexType(str, Enum):
    """Vertex types in the graph."""
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"


class EdgeType(str, Enum):
    """Edge types in the graph."""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"
    DEPENDS_ON = "DEPENDS_ON"


class DependencyType(str, Enum):
    """
    Types of derived DEPENDS_ON relationships.
    
    Multi-layer dependency types enable analysis across architectural layers:
    - app_to_app: Application-level data flow dependencies
    - node_to_node: Infrastructure-level dependencies
    - app_to_broker: Application-to-middleware dependencies
    - node_to_broker: Infrastructure-to-middleware dependencies
    """
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"


# =============================================================================
# QoS Enumerations with Weight Properties
# =============================================================================

class Durability(str, Enum):
    """
    DDS Durability QoS Policy.
    
    Higher durability means more critical data that must persist,
    contributing more weight to dependency relationships.
    """
    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"
    TRANSIENT = "TRANSIENT"
    PERSISTENT = "PERSISTENT"

    @property
    def weight(self) -> float:
        """Weight contribution for dependency scoring."""
        return {
            "VOLATILE": 0.0,
            "TRANSIENT_LOCAL": 0.20,
            "TRANSIENT": 0.25,
            "PERSISTENT": 0.40,
        }.get(self.value, 0.0)


class Reliability(str, Enum):
    """
    DDS Reliability QoS Policy.
    
    RELIABLE delivery indicates critical data that must not be lost,
    adding weight to the dependency relationship.
    """
    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE = "RELIABLE"

    @property
    def weight(self) -> float:
        """Weight contribution for dependency scoring."""
        return 0.30 if self.value == "RELIABLE" else 0.0


class Priority(str, Enum):
    """
    Transport Priority levels.
    
    Higher priority messages indicate more time-critical dependencies.
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

    @property
    def weight(self) -> float:
        """Weight contribution for dependency scoring."""
        return {"LOW": 0.0, "MEDIUM": 0.10, "HIGH": 0.20, "URGENT": 0.30}.get(
            self.value, 0.0
        )


# =============================================================================
# QoS Policy
# =============================================================================

@dataclass(frozen=True)
class QoSPolicy:
    """
    Quality of Service policy for topics.
    
    Used to calculate dependency weights - higher QoS requirements
    indicate more critical dependencies that should be weighted higher.
    
    Attributes:
        durability: Data persistence level
        reliability: Delivery guarantee level
        transport_priority: Message priority level
    """
    durability: str = "VOLATILE"
    reliability: str = "BEST_EFFORT"
    transport_priority: str = "MEDIUM"

    def criticality_score(self) -> float:
        """
        Calculate criticality score from QoS settings.
        
        Returns:
            Float between 0.0 and 1.0 representing QoS criticality.
            Maximum: 0.40 (durability) + 0.30 (reliability) + 0.30 (priority) = 1.0
        """
        d_weight = self._get_enum_weight(Durability, self.durability)
        r_weight = self._get_enum_weight(Reliability, self.reliability)
        p_weight = self._get_enum_weight(Priority, self.transport_priority)
        return d_weight + r_weight + p_weight

    @staticmethod
    def _get_enum_weight(enum_class, value: str) -> float:
        """Safely get weight from enum value."""
        try:
            return enum_class(value).weight
        except ValueError:
            return 0.0

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "durability": self.durability,
            "reliability": self.reliability,
            "transport_priority": self.transport_priority,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> QoSPolicy:
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            durability=data.get("durability", "VOLATILE"),
            reliability=data.get("reliability", "BEST_EFFORT"),
            transport_priority=data.get("transport_priority", "MEDIUM"),
        )


# =============================================================================
# Vertex Classes
# =============================================================================

@dataclass
class Application:
    """
    Application vertex - a software component that publishes/subscribes.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        role: pub, sub, or pubsub
        weight: Criticality weight based on relationship weights
    """
    id: str
    name: str
    role: str = "pubsub"
    weight: float = 0.0

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.APPLICATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Application:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            role=data.get("role", "pubsub"),
            weight=data.get("weight", 0.0),
        )


@dataclass
class Broker:
    """
    Broker vertex - message routing middleware.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        weight: Criticality weight based on relationship weights
    """
    id: str
    name: str
    weight: float = 0.0

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.BROKER

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Dict) -> Broker:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            weight=data.get("weight", 0.0),
        )


@dataclass
class Topic:
    """
    Topic vertex - message channel with QoS properties.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        size: Average message size in bytes
        qos: Quality of Service policy
        weight: Criticality weight (derived from QoS)
    """
    id: str
    name: str
    size: int = 256
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    weight: float = 0.0

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.TOPIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "qos": self.qos.to_dict(),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Topic:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            size=data.get("size", 256),
            qos=QoSPolicy.from_dict(data.get("qos")),
            weight=data.get("weight", 0.0),
        )


@dataclass
class Node:
    """
    Infrastructure node vertex.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        weight: Criticality weight based on relationship weights
    """
    id: str
    name: str
    weight: float = 0.0

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.NODE

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Dict) -> Node:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            weight=data.get("weight", 0.0),
        )


# =============================================================================
# Edge Classes
# =============================================================================

@dataclass
class Edge:
    """Basic edge in the graph."""
    source: str
    target: str
    edge_type: str

    def to_dict(self) -> Dict[str, str]:
        return {"from": self.source, "to": self.target}

    @classmethod
    def from_dict(cls, data: Dict, edge_type: str) -> Edge:
        return cls(
            source=data.get("from", data.get("source", "")),
            target=data.get("to", data.get("target", "")),
            edge_type=edge_type,
        )


@dataclass
class DependsOnEdge:
    """
    DEPENDS_ON relationship with weight and metadata.
    
    Weight calculation incorporates:
    - Number of shared topics (more topics = stronger coupling)
    - QoS criticality of topics (higher QoS = more critical dependency)
    - Message size (larger messages = more data dependency)
    
    Attributes:
        source: Source component ID
        target: Target component ID
        dependency_type: Type of dependency relationship
        weight: Calculated criticality weight
        via_topics: List of topic IDs this dependency passes through
    """
    source: str
    target: str
    dependency_type: DependencyType
    weight: float = 1.0
    via_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.source,
            "to": self.target,
            "dependency_type": self.dependency_type.value,
            "weight": self.weight,
            "via_topics": self.via_topics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> DependsOnEdge:
        dep_type_str = data.get("dependency_type", "app_to_app")
        try:
            dep_type = DependencyType(dep_type_str)
        except ValueError:
            dep_type = DependencyType.APP_TO_APP
        
        return cls(
            source=data.get("from", data.get("source", "")),
            target=data.get("to", data.get("target", "")),
            dependency_type=dep_type,
            weight=data.get("weight", 1.0),
            via_topics=data.get("via_topics", []),
        )


# =============================================================================
# Graph Model
# =============================================================================

class GraphModel:
    """
    Container for the complete pub-sub graph with query and traversal methods.
    
    Provides:
    - Type-safe access to vertices and edges
    - Query methods for graph traversal
    - Automatic component weight calculation
    - Dictionary serialization
    """

    def __init__(self) -> None:
        """Initialize empty graph model."""
        self._applications: Dict[str, Application] = {}
        self._brokers: Dict[str, Broker] = {}
        self._topics: Dict[str, Topic] = {}
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[Edge]] = defaultdict(list)
        self._depends_on: List[DependsOnEdge] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def applications(self) -> List[Application]:
        """Get all applications."""
        return list(self._applications.values())

    @property
    def brokers(self) -> List[Broker]:
        """Get all brokers."""
        return list(self._brokers.values())

    @property
    def topics(self) -> List[Topic]:
        """Get all topics."""
        return list(self._topics.values())

    @property
    def nodes(self) -> List[Node]:
        """Get all infrastructure nodes."""
        return list(self._nodes.values())

    @property
    def depends_on_edges(self) -> List[DependsOnEdge]:
        """Get all DEPENDS_ON relationships."""
        return self._depends_on

    # -------------------------------------------------------------------------
    # Add Methods
    # -------------------------------------------------------------------------

    def add_application(self, app: Application) -> None:
        """Add an application."""
        self._applications[app.id] = app

    def add_broker(self, broker: Broker) -> None:
        """Add a broker."""
        self._brokers[broker.id] = broker

    def add_topic(self, topic: Topic) -> None:
        """Add a topic."""
        self._topics[topic.id] = topic

    def add_node(self, node: Node) -> None:
        """Add a node."""
        self._nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge."""
        self._edges[edge.edge_type].append(edge)

    def add_depends_on(self, edge: DependsOnEdge) -> None:
        """Add a DEPENDS_ON relationship."""
        self._depends_on.append(edge)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_application(self, app_id: str) -> Optional[Application]:
        """Get application by ID."""
        return self._applications.get(app_id)

    def get_broker(self, broker_id: str) -> Optional[Broker]:
        """Get broker by ID."""
        return self._brokers.get(broker_id)

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Get topic by ID."""
        return self._topics.get(topic_id)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_edges(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type."""
        return self._edges.get(edge_type.value, [])

    def get_publishers(self, topic_id: str) -> List[Application]:
        """Get applications publishing to a topic."""
        result = []
        for edge in self._edges.get(EdgeType.PUBLISHES_TO.value, []):
            if edge.target == topic_id:
                app = self._applications.get(edge.source)
                if app:
                    result.append(app)
        return result

    def get_subscribers(self, topic_id: str) -> List[Application]:
        """Get applications subscribing to a topic."""
        result = []
        for edge in self._edges.get(EdgeType.SUBSCRIBES_TO.value, []):
            if edge.target == topic_id:
                app = self._applications.get(edge.source)
                if app:
                    result.append(app)
        return result

    def get_routed_topics(self, broker_id: str) -> List[Topic]:
        """Get topics routed by a broker."""
        result = []
        for edge in self._edges.get(EdgeType.ROUTES.value, []):
            if edge.source == broker_id:
                topic = self._topics.get(edge.target)
                if topic:
                    result.append(topic)
        return result

    def get_node_for_component(self, component_id: str) -> Optional[Node]:
        """Get the node that a component runs on."""
        for edge in self._edges.get(EdgeType.RUNS_ON.value, []):
            if edge.source == component_id:
                return self._nodes.get(edge.target)
        return None

    def get_dependencies(
        self, component_id: str, dep_type: Optional[DependencyType] = None
    ) -> List[DependsOnEdge]:
        """Get outgoing dependencies for a component."""
        result = []
        for dep in self._depends_on:
            if dep.source == component_id:
                if dep_type is None or dep.dependency_type == dep_type:
                    result.append(dep)
        return result

    def get_dependents(
        self, component_id: str, dep_type: Optional[DependencyType] = None
    ) -> List[DependsOnEdge]:
        """Get incoming dependencies for a component (who depends on this)."""
        result = []
        for dep in self._depends_on:
            if dep.target == component_id:
                if dep_type is None or dep.dependency_type == dep_type:
                    result.append(dep)
        return result

    # -------------------------------------------------------------------------
    # Component Weight Calculation
    # -------------------------------------------------------------------------

    def calculate_component_weights(self) -> None:
        """
        Calculate criticality weights for all components based on their relationships.
        
        Weight formula:
            weight(v) = sum(incoming_deps.weight) + sum(outgoing_deps.weight)
        
        This captures both:
        - How much other components depend on this component (incoming)
        - How much this component depends on others (outgoing)
        
        Components with many high-weight dependencies are more critical.
        """
        # Initialize weight accumulators
        weights: Dict[str, float] = defaultdict(float)
        
        # Accumulate weights from DEPENDS_ON relationships
        for dep in self._depends_on:
            weights[dep.source] += dep.weight
            weights[dep.target] += dep.weight
        
        # Update component weights
        for app_id, app in self._applications.items():
            app.weight = weights.get(app_id, 0.0)
        
        for broker_id, broker in self._brokers.items():
            broker.weight = weights.get(broker_id, 0.0)
        
        for node_id, node in self._nodes.items():
            node.weight = weights.get(node_id, 0.0)
        
        # Topic weight is based on QoS criticality
        for topic in self._topics.values():
            topic.weight = topic.qos.criticality_score()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Graph Model Summary",
            "=" * 40,
            f"Applications: {len(self._applications)}",
            f"Brokers: {len(self._brokers)}",
            f"Topics: {len(self._topics)}",
            f"Nodes: {len(self._nodes)}",
            "",
            "Relationships:",
        ]
        
        for edge_type in EdgeType:
            count = len(self._edges.get(edge_type.value, []))
            if count > 0:
                lines.append(f"  {edge_type.value}: {count}")
        
        if self._depends_on:
            lines.append(f"  DEPENDS_ON: {len(self._depends_on)}")
            
            # Count by type
            by_type: Dict[str, int] = defaultdict(int)
            for dep in self._depends_on:
                by_type[dep.dependency_type.value] += 1
            
            for dep_type, count in sorted(by_type.items()):
                lines.append(f"    - {dep_type}: {count}")
        
        return "\n".join(lines)

    def statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        dep_by_type: Dict[str, int] = defaultdict(int)
        total_weight = 0.0
        
        for dep in self._depends_on:
            dep_by_type[dep.dependency_type.value] += 1
            total_weight += dep.weight
        
        avg_weight = total_weight / len(self._depends_on) if self._depends_on else 0.0
        
        return {
            "vertices": {
                "applications": len(self._applications),
                "brokers": len(self._brokers),
                "topics": len(self._topics),
                "nodes": len(self._nodes),
                "total": (
                    len(self._applications)
                    + len(self._brokers)
                    + len(self._topics)
                    + len(self._nodes)
                ),
            },
            "edges": {
                edge_type.value: len(self._edges.get(edge_type.value, []))
                for edge_type in EdgeType
                if edge_type != EdgeType.DEPENDS_ON
            },
            "depends_on": {
                "total": len(self._depends_on),
                "by_type": dict(dep_by_type),
                "total_weight": round(total_weight, 2),
                "average_weight": round(avg_weight, 2),
            },
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "applications": [a.to_dict() for a in self._applications.values()],
            "brokers": [b.to_dict() for b in self._brokers.values()],
            "topics": [t.to_dict() for t in self._topics.values()],
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "relationships": {
                "publishes_to": [e.to_dict() for e in self._edges.get("PUBLISHES_TO", [])],
                "subscribes_to": [e.to_dict() for e in self._edges.get("SUBSCRIBES_TO", [])],
                "routes": [e.to_dict() for e in self._edges.get("ROUTES", [])],
                "runs_on": [e.to_dict() for e in self._edges.get("RUNS_ON", [])],
                "connects_to": [e.to_dict() for e in self._edges.get("CONNECTS_TO", [])],
            },
            "depends_on": [d.to_dict() for d in self._depends_on],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> GraphModel:
        """Create from dictionary format."""
        model = cls()
        
        # Load vertices
        for app_data in data.get("applications", []):
            model.add_application(Application.from_dict(app_data))
        
        for broker_data in data.get("brokers", []):
            model.add_broker(Broker.from_dict(broker_data))
        
        for topic_data in data.get("topics", []):
            model.add_topic(Topic.from_dict(topic_data))
        
        for node_data in data.get("nodes", []):
            model.add_node(Node.from_dict(node_data))
        
        # Load relationships
        rels = data.get("relationships", {})
        
        for edge_data in rels.get("publishes_to", []):
            model.add_edge(Edge.from_dict(edge_data, EdgeType.PUBLISHES_TO.value))
        
        for edge_data in rels.get("subscribes_to", []):
            model.add_edge(Edge.from_dict(edge_data, EdgeType.SUBSCRIBES_TO.value))
        
        for edge_data in rels.get("routes", []):
            model.add_edge(Edge.from_dict(edge_data, EdgeType.ROUTES.value))
        
        for edge_data in rels.get("runs_on", []):
            model.add_edge(Edge.from_dict(edge_data, EdgeType.RUNS_ON.value))
        
        for edge_data in rels.get("connects_to", []):
            model.add_edge(Edge.from_dict(edge_data, EdgeType.CONNECTS_TO.value))
        
        # Load DEPENDS_ON if present
        for dep_data in data.get("depends_on", []):
            model.add_depends_on(DependsOnEdge.from_dict(dep_data))
        
        return model