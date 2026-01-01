"""
Graph Model - Refactored Version 4.0

Simplified, clean data models for pub-sub system graphs with:
- Clear vertex and edge types
- QoS-aware weight calculation for DEPENDS_ON relationships
- Immutable dataclasses for safety
- Comprehensive query methods

Vertices:
    Application: {id, name, role}
    Broker: {id, name}
    Topic: {id, name, size, qos}
    Node: {id, name}

Edges:
    PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
    DEPENDS_ON (derived with weight based on QoS and message size)

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Iterator, Any
from collections import defaultdict


# =============================================================================
# Enumerations
# =============================================================================

class VertexType(str, Enum):
    """Vertex types in the graph"""
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"


class EdgeType(str, Enum):
    """Edge types in the graph"""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"
    DEPENDS_ON = "DEPENDS_ON"


class DependencyType(str, Enum):
    """Types of derived DEPENDS_ON relationships"""
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"


class Durability(str, Enum):
    """DDS Durability QoS Policy"""
    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"
    TRANSIENT = "TRANSIENT"
    PERSISTENT = "PERSISTENT"

    @property
    def weight(self) -> float:
        """Weight contribution for dependency scoring"""
        weights = {
            "VOLATILE": 0.0,
            "TRANSIENT_LOCAL": 0.20,
            "TRANSIENT": 0.25,
            "PERSISTENT": 0.40,
        }
        return weights.get(self.value, 0.0)


class Reliability(str, Enum):
    """DDS Reliability QoS Policy"""
    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE = "RELIABLE"

    @property
    def weight(self) -> float:
        """Weight contribution for dependency scoring"""
        return 0.30 if self.value == "RELIABLE" else 0.0


class Priority(str, Enum):
    """Transport Priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

    @property
    def weight(self) -> float:
        """Weight contribution for dependency scoring"""
        weights = {"LOW": 0.0, "MEDIUM": 0.10, "HIGH": 0.20, "URGENT": 0.30}
        return weights.get(self.value, 0.0)


# =============================================================================
# QoS Policy
# =============================================================================

@dataclass(frozen=True)
class QoSPolicy:
    """
    Quality of Service policy for topics.
    
    Used to calculate dependency weights - higher QoS requirements
    indicate more critical dependencies that should be weighted higher.
    """
    durability: str = "VOLATILE"
    reliability: str = "BEST_EFFORT"
    transport_priority: str = "MEDIUM"

    def criticality_score(self) -> float:
        """
        Calculate criticality score from QoS settings.
        
        Returns:
            Float between 0.0 and 1.0 representing QoS criticality
        """
        try:
            d_weight = Durability(self.durability).weight
        except ValueError:
            d_weight = 0.0
        
        try:
            r_weight = Reliability(self.reliability).weight
        except ValueError:
            r_weight = 0.0
            
        try:
            p_weight = Priority(self.transport_priority).weight
        except ValueError:
            p_weight = 0.0
            
        return d_weight + r_weight + p_weight

    def to_dict(self) -> Dict[str, str]:
        return {
            "durability": self.durability,
            "reliability": self.reliability,
            "transport_priority": self.transport_priority,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> QoSPolicy:
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
    """Application vertex - a software component that publishes/subscribes"""
    id: str
    name: str
    role: str = "pubsub"  # pub, sub, pubsub

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.APPLICATION

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name, "role": self.role}

    @classmethod
    def from_dict(cls, data: Dict) -> Application:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            role=data.get("role", "pubsub"),
        )


@dataclass
class Broker:
    """Broker vertex - message routing infrastructure"""
    id: str
    name: str

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.BROKER

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict) -> Broker:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
        )


@dataclass
class Topic:
    """Topic vertex - message channel with QoS settings"""
    id: str
    name: str
    size: int = 256  # Message size in bytes
    qos: QoSPolicy = field(default_factory=QoSPolicy)

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.TOPIC

    def criticality_score(self) -> float:
        """Combined criticality from QoS and message size"""
        qos_score = self.qos.criticality_score()
        # Normalize size contribution (larger messages = more critical)
        size_score = min(self.size / 10000, 0.5)  # Cap at 0.5
        return qos_score + size_score

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "qos": self.qos.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Topic:
        qos_data = data.get("qos", {})
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            size=data.get("size", 256),
            qos=QoSPolicy.from_dict(qos_data),
        )


@dataclass
class Node:
    """Infrastructure node vertex"""
    id: str
    name: str

    @property
    def vertex_type(self) -> VertexType:
        return VertexType.NODE

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict) -> Node:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
        )


# =============================================================================
# Edge Classes
# =============================================================================

@dataclass
class Edge:
    """Basic edge in the graph"""
    source: str
    target: str
    edge_type: str

    def to_dict(self) -> Dict:
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
    """
    source: str
    target: str
    dependency_type: DependencyType
    weight: float = 1.0
    via_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "from": self.source,
            "to": self.target,
            "dependency_type": self.dependency_type.value,
            "weight": self.weight,
            "via_topics": self.via_topics,
        }


# =============================================================================
# Graph Model
# =============================================================================

class GraphModel:
    """
    Container for the complete pub-sub graph with query and traversal methods.
    
    Provides efficient lookup and traversal for:
    - Vertices by type and ID
    - Edges by type
    - Publisher/subscriber relationships
    - Dependency chains
    """

    def __init__(self):
        # Vertex storage
        self.applications: Dict[str, Application] = {}
        self.brokers: Dict[str, Broker] = {}
        self.topics: Dict[str, Topic] = {}
        self.nodes: Dict[str, Node] = {}
        
        # Edge storage
        self._edges: List[Edge] = []
        self._depends_on: List[DependsOnEdge] = []
        
        # Indexes for fast lookup
        self._edges_by_type: Dict[str, List[Edge]] = defaultdict(list)
        self._edges_from: Dict[str, List[Edge]] = defaultdict(list)
        self._edges_to: Dict[str, List[Edge]] = defaultdict(list)
        
        # Metadata
        self.metadata: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Vertex Operations
    # -------------------------------------------------------------------------

    def add_application(self, app: Application) -> None:
        self.applications[app.id] = app

    def add_broker(self, broker: Broker) -> None:
        self.brokers[broker.id] = broker

    def add_topic(self, topic: Topic) -> None:
        self.topics[topic.id] = topic

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def get_vertex(self, vertex_id: str) -> Optional[Any]:
        """Get any vertex by ID"""
        return (
            self.applications.get(vertex_id)
            or self.brokers.get(vertex_id)
            or self.topics.get(vertex_id)
            or self.nodes.get(vertex_id)
        )

    def vertex_exists(self, vertex_id: str) -> bool:
        return self.get_vertex(vertex_id) is not None

    def all_vertex_ids(self) -> Set[str]:
        """Get all vertex IDs"""
        return (
            set(self.applications.keys())
            | set(self.brokers.keys())
            | set(self.topics.keys())
            | set(self.nodes.keys())
        )

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """Add an edge with index updates"""
        self._edges.append(edge)
        self._edges_by_type[edge.edge_type].append(edge)
        self._edges_from[edge.source].append(edge)
        self._edges_to[edge.target].append(edge)

    def add_depends_on(self, edge: DependsOnEdge) -> None:
        """Add a DEPENDS_ON relationship"""
        self._depends_on.append(edge)

    @property
    def edges(self) -> List[Edge]:
        return self._edges

    @property
    def depends_on_edges(self) -> List[DependsOnEdge]:
        return self._depends_on

    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        return self._edges_by_type.get(edge_type, [])

    def get_edges_from(self, source_id: str) -> List[Edge]:
        return self._edges_from.get(source_id, [])

    def get_edges_to(self, target_id: str) -> List[Edge]:
        return self._edges_to.get(target_id, [])

    # -------------------------------------------------------------------------
    # Pub/Sub Queries
    # -------------------------------------------------------------------------

    def get_publishers(self, topic_id: str) -> List[str]:
        """Get all applications that publish to a topic"""
        return [
            e.source for e in self._edges_by_type.get(EdgeType.PUBLISHES_TO.value, [])
            if e.target == topic_id
        ]

    def get_subscribers(self, topic_id: str) -> List[str]:
        """Get all applications that subscribe to a topic"""
        return [
            e.source for e in self._edges_by_type.get(EdgeType.SUBSCRIBES_TO.value, [])
            if e.target == topic_id
        ]

    def get_published_topics(self, app_id: str) -> List[str]:
        """Get topics an application publishes to"""
        return [
            e.target for e in self._edges_by_type.get(EdgeType.PUBLISHES_TO.value, [])
            if e.source == app_id
        ]

    def get_subscribed_topics(self, app_id: str) -> List[str]:
        """Get topics an application subscribes to"""
        return [
            e.target for e in self._edges_by_type.get(EdgeType.SUBSCRIBES_TO.value, [])
            if e.source == app_id
        ]

    def get_broker_topics(self, broker_id: str) -> List[str]:
        """Get topics routed by a broker"""
        return [
            e.target for e in self._edges_by_type.get(EdgeType.ROUTES.value, [])
            if e.source == broker_id
        ]

    def get_host_node(self, component_id: str) -> Optional[str]:
        """Get the node a component runs on"""
        for e in self._edges_by_type.get(EdgeType.RUNS_ON.value, []):
            if e.source == component_id:
                return e.target
        return None

    # -------------------------------------------------------------------------
    # Summary & Export
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Get graph summary statistics"""
        edge_counts = {et.value: len(self._edges_by_type.get(et.value, [])) for et in EdgeType if et != EdgeType.DEPENDS_ON}
        
        return {
            "vertices": {
                "applications": len(self.applications),
                "brokers": len(self.brokers),
                "topics": len(self.topics),
                "nodes": len(self.nodes),
                "total": len(self.all_vertex_ids()),
            },
            "edges": {
                **edge_counts,
                "depends_on": len(self._depends_on),
                "total": len(self._edges) + len(self._depends_on),
            },
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict:
        """Export model to dictionary format"""
        return {
            "metadata": self.metadata,
            "applications": [a.to_dict() for a in self.applications.values()],
            "brokers": [b.to_dict() for b in self.brokers.values()],
            "topics": [t.to_dict() for t in self.topics.values()],
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "relationships": {
                "publishes_to": [e.to_dict() for e in self.get_edges_by_type(EdgeType.PUBLISHES_TO.value)],
                "subscribes_to": [e.to_dict() for e in self.get_edges_by_type(EdgeType.SUBSCRIBES_TO.value)],
                "routes": [e.to_dict() for e in self.get_edges_by_type(EdgeType.ROUTES.value)],
                "runs_on": [e.to_dict() for e in self.get_edges_by_type(EdgeType.RUNS_ON.value)],
                "connects_to": [e.to_dict() for e in self.get_edges_by_type(EdgeType.CONNECTS_TO.value)],
            },
            "depends_on": [d.to_dict() for d in self._depends_on],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> GraphModel:
        """Build model from dictionary format"""
        model = cls()
        model.metadata = data.get("metadata", {})
        
        # Load vertices
        for app_data in data.get("applications", []):
            model.add_application(Application.from_dict(app_data))
        for broker_data in data.get("brokers", []):
            model.add_broker(Broker.from_dict(broker_data))
        for topic_data in data.get("topics", []):
            model.add_topic(Topic.from_dict(topic_data))
        for node_data in data.get("nodes", []):
            model.add_node(Node.from_dict(node_data))
        
        # Load edges
        relationships = data.get("relationships", {})
        edge_mappings = [
            ("publishes_to", EdgeType.PUBLISHES_TO.value),
            ("subscribes_to", EdgeType.SUBSCRIBES_TO.value),
            ("routes", EdgeType.ROUTES.value),
            ("runs_on", EdgeType.RUNS_ON.value),
            ("connects_to", EdgeType.CONNECTS_TO.value),
        ]
        for key, edge_type in edge_mappings:
            for edge_data in relationships.get(key, []):
                model.add_edge(Edge.from_dict(edge_data, edge_type))
        
        return model

    def __repr__(self) -> str:
        s = self.summary()
        return f"GraphModel(apps={s['vertices']['applications']}, topics={s['vertices']['topics']}, edges={s['edges']['total']})"