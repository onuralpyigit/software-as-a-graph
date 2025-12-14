"""
Graph Model - Simplified Version 3.0

Simplified data models for representing pub-sub system graphs:

Vertices:
- Application: {id, name, role (pub|sub|pubsub)}
- Broker: {id, name}
- Topic: {id, name, size, qos {durability, reliability, transport_priority}}
- Node: {id, name}

Edges:
- PUBLISHES_TO (App → Topic): {from, to}
- SUBSCRIBES_TO (App → Topic): {from, to}
- ROUTES (Broker → Topic): {from, to}
- RUNS_ON (App/Broker → Node): {from, to}
- CONNECTS_TO (Node → Node): {from, to}

Author: Research Team
Version: 3.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict


# =============================================================================
# Enumerations
# =============================================================================

class ApplicationRole(str, Enum):
    """Application role in pub-sub system"""
    PUB = "pub"
    SUB = "sub"
    PUBSUB = "pubsub"


class DurabilityPolicy(str, Enum):
    """DDS Durability QoS Policy"""
    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"
    TRANSIENT = "TRANSIENT"
    PERSISTENT = "PERSISTENT"


class ReliabilityPolicy(str, Enum):
    """DDS Reliability QoS Policy"""
    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE = "RELIABLE"


class TransportPriority(str, Enum):
    """Transport Priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class EdgeType(str, Enum):
    """Edge types in the graph"""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"


class VertexType(str, Enum):
    """Vertex types in the graph"""
    APPLICATION = "APPLICATION"
    BROKER = "BROKER"
    TOPIC = "TOPIC"
    NODE = "NODE"


# =============================================================================
# QoS Policy
# =============================================================================

@dataclass
class QoSPolicy:
    """Quality of Service policy for topics"""
    durability: str = "VOLATILE"
    reliability: str = "BEST_EFFORT"
    transport_priority: str = "MEDIUM"
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QoSPolicy':
        """Create QoSPolicy from dictionary"""
        if not data:
            return cls()
        return cls(
            durability=data.get('durability', 'VOLATILE'),
            reliability=data.get('reliability', 'BEST_EFFORT'),
            transport_priority=data.get('transport_priority', 'MEDIUM')
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'durability': self.durability,
            'reliability': self.reliability,
            'transport_priority': self.transport_priority
        }
    
    def get_criticality_score(self) -> float:
        """Calculate QoS-based criticality score (0-1)"""
        score = 0.0
        durability_scores = {'VOLATILE': 0.0, 'TRANSIENT_LOCAL': 0.33, 'TRANSIENT': 0.66, 'PERSISTENT': 1.0}
        score += 0.33 * durability_scores.get(self.durability, 0.0)
        reliability_scores = {'BEST_EFFORT': 0.0, 'RELIABLE': 1.0}
        score += 0.34 * reliability_scores.get(self.reliability, 0.0)
        priority_scores = {'LOW': 0.0, 'MEDIUM': 0.33, 'HIGH': 0.66, 'URGENT': 1.0}
        score += 0.33 * priority_scores.get(self.transport_priority, 0.33)
        return round(score, 3)
    
    def is_critical(self) -> bool:
        """Check if this QoS indicates a critical topic"""
        return (self.reliability == 'RELIABLE' or 
                self.durability in ('PERSISTENT', 'TRANSIENT') or
                self.transport_priority in ('HIGH', 'URGENT'))


# =============================================================================
# Vertex Classes
# =============================================================================

@dataclass
class Application:
    """Application vertex"""
    id: str
    name: str
    role: str = "pubsub"
    
    @property
    def vertex_type(self) -> str:
        return VertexType.APPLICATION.value
    
    def to_dict(self) -> Dict:
        return {'id': self.id, 'name': self.name, 'role': self.role}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Application':
        return cls(id=data.get('id', ''), name=data.get('name', data.get('id', '')), role=data.get('role', 'pubsub'))
    
    def is_publisher(self) -> bool:
        return self.role in ('pub', 'pubsub')
    
    def is_subscriber(self) -> bool:
        return self.role in ('sub', 'pubsub')


@dataclass
class Broker:
    """Broker vertex"""
    id: str
    name: str
    
    @property
    def vertex_type(self) -> str:
        return VertexType.BROKER.value
    
    def to_dict(self) -> Dict:
        return {'id': self.id, 'name': self.name}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Broker':
        return cls(id=data.get('id', ''), name=data.get('name', data.get('id', '')))


@dataclass
class Topic:
    """Topic vertex"""
    id: str
    name: str
    size: int = 256
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    
    @property
    def vertex_type(self) -> str:
        return VertexType.TOPIC.value
    
    def to_dict(self) -> Dict:
        return {'id': self.id, 'name': self.name, 'size': self.size,
                'qos': self.qos.to_dict() if isinstance(self.qos, QoSPolicy) else self.qos}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Topic':
        qos_data = data.get('qos', {})
        qos = QoSPolicy.from_dict(qos_data) if qos_data else QoSPolicy()
        return cls(id=data.get('id', ''), name=data.get('name', data.get('id', '')), size=data.get('size', 256), qos=qos)
    
    def get_criticality_score(self) -> float:
        if isinstance(self.qos, QoSPolicy):
            return self.qos.get_criticality_score()
        return 0.0


@dataclass
class Node:
    """Infrastructure node vertex"""
    id: str
    name: str
    
    @property
    def vertex_type(self) -> str:
        return VertexType.NODE.value
    
    def to_dict(self) -> Dict:
        return {'id': self.id, 'name': self.name}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        return cls(id=data.get('id', ''), name=data.get('name', data.get('id', '')))


# =============================================================================
# Edge Class
# =============================================================================

@dataclass
class Edge:
    """Edge in the graph"""
    source: str
    target: str
    edge_type: str
    
    def to_dict(self) -> Dict:
        return {'from': self.source, 'to': self.target}
    
    @classmethod
    def from_dict(cls, data: Dict, edge_type: str) -> 'Edge':
        return cls(source=data.get('from', ''), target=data.get('to', ''), edge_type=edge_type)
    
    def reversed(self) -> 'Edge':
        return Edge(source=self.target, target=self.source, edge_type=self.edge_type)


# =============================================================================
# Graph Model
# =============================================================================

class GraphModel:
    """Container for the complete graph model with query and traversal methods"""
    
    EDGE_TYPES = {
        'publishes_to': EdgeType.PUBLISHES_TO.value,
        'subscribes_to': EdgeType.SUBSCRIBES_TO.value,
        'routes': EdgeType.ROUTES.value,
        'runs_on': EdgeType.RUNS_ON.value,
        'connects_to': EdgeType.CONNECTS_TO.value
    }
    
    def __init__(self):
        self.applications: Dict[str, Application] = {}
        self.brokers: Dict[str, Broker] = {}
        self.topics: Dict[str, Topic] = {}
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._edges_by_type: Dict[str, List[Edge]] = {et.value: [] for et in EdgeType}
        self._outgoing: Dict[str, List[Edge]] = defaultdict(list)
        self._incoming: Dict[str, List[Edge]] = defaultdict(list)
        self.metadata: Dict = {}
    
    # Vertex operations
    def add_application(self, app: Application) -> None:
        self.applications[app.id] = app
    
    def add_broker(self, broker: Broker) -> None:
        self.brokers[broker.id] = broker
    
    def add_topic(self, topic: Topic) -> None:
        self.topics[topic.id] = topic
    
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
    
    def get_vertex(self, vertex_id: str) -> Optional[Any]:
        for store in (self.applications, self.brokers, self.topics, self.nodes):
            if vertex_id in store:
                return store[vertex_id]
        return None
    
    def get_vertex_type(self, vertex_id: str) -> Optional[str]:
        if vertex_id in self.applications: return VertexType.APPLICATION.value
        if vertex_id in self.brokers: return VertexType.BROKER.value
        if vertex_id in self.topics: return VertexType.TOPIC.value
        if vertex_id in self.nodes: return VertexType.NODE.value
        return None
    
    def get_all_vertex_ids(self) -> Set[str]:
        return set(self.applications) | set(self.brokers) | set(self.topics) | set(self.nodes)
    
    def vertex_exists(self, vertex_id: str) -> bool:
        return vertex_id in self.get_all_vertex_ids()
    
    # Edge operations
    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        if edge.edge_type in self._edges_by_type:
            self._edges_by_type[edge.edge_type].append(edge)
        self._outgoing[edge.source].append(edge)
        self._incoming[edge.target].append(edge)
    
    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        return self._edges_by_type.get(edge_type, [])
    
    def get_outgoing_edges(self, vertex_id: str, edge_type: Optional[str] = None) -> List[Edge]:
        edges = self._outgoing.get(vertex_id, [])
        return [e for e in edges if e.edge_type == edge_type] if edge_type else edges
    
    def get_incoming_edges(self, vertex_id: str, edge_type: Optional[str] = None) -> List[Edge]:
        edges = self._incoming.get(vertex_id, [])
        return [e for e in edges if e.edge_type == edge_type] if edge_type else edges
    
    def get_neighbors(self, vertex_id: str, direction: str = 'both') -> Set[str]:
        neighbors = set()
        if direction in ('out', 'both'):
            neighbors.update(e.target for e in self._outgoing.get(vertex_id, []))
        if direction in ('in', 'both'):
            neighbors.update(e.source for e in self._incoming.get(vertex_id, []))
        return neighbors
    
    # Query methods
    def get_publishers_of(self, topic_id: str) -> List[str]:
        return [e.source for e in self.get_edges_by_type(EdgeType.PUBLISHES_TO.value) if e.target == topic_id]
    
    def get_subscribers_of(self, topic_id: str) -> List[str]:
        return [e.source for e in self.get_edges_by_type(EdgeType.SUBSCRIBES_TO.value) if e.target == topic_id]
    
    def get_topics_published_by(self, app_id: str) -> List[str]:
        return [e.target for e in self.get_outgoing_edges(app_id, EdgeType.PUBLISHES_TO.value)]
    
    def get_topics_subscribed_by(self, app_id: str) -> List[str]:
        return [e.target for e in self.get_outgoing_edges(app_id, EdgeType.SUBSCRIBES_TO.value)]
    
    def get_broker_of(self, topic_id: str) -> Optional[str]:
        for e in self.get_edges_by_type(EdgeType.ROUTES.value):
            if e.target == topic_id:
                return e.source
        return None
    
    def get_topics_routed_by(self, broker_id: str) -> List[str]:
        return [e.target for e in self.get_outgoing_edges(broker_id, EdgeType.ROUTES.value)]
    
    def get_node_of(self, component_id: str) -> Optional[str]:
        edges = self.get_outgoing_edges(component_id, EdgeType.RUNS_ON.value)
        return edges[0].target if edges else None
    
    def get_components_on_node(self, node_id: str) -> List[str]:
        return [e.source for e in self.get_incoming_edges(node_id, EdgeType.RUNS_ON.value)]
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        outgoing = [e.target for e in self.get_outgoing_edges(node_id, EdgeType.CONNECTS_TO.value)]
        incoming = [e.source for e in self.get_incoming_edges(node_id, EdgeType.CONNECTS_TO.value)]
        return list(set(outgoing + incoming))
    
    # Analysis methods
    def get_orphan_topics(self) -> List[str]:
        return [t for t in self.topics if not self.get_publishers_of(t) and not self.get_subscribers_of(t)]
    
    def get_isolated_apps(self) -> List[str]:
        return [a for a in self.applications if not self.get_topics_published_by(a) and not self.get_topics_subscribed_by(a)]
    
    def get_critical_topics(self, threshold: float = 0.5) -> List[str]:
        return [t for t, topic in self.topics.items() if topic.get_criticality_score() >= threshold]
    
    def get_topic_fanout(self, topic_id: str) -> int:
        return len(self.get_subscribers_of(topic_id))
    
    def get_topic_fanin(self, topic_id: str) -> int:
        return len(self.get_publishers_of(topic_id))
    
    # Serialization
    def to_dict(self) -> Dict:
        return {
            'metadata': self.metadata,
            'applications': [a.to_dict() for a in self.applications.values()],
            'brokers': [b.to_dict() for b in self.brokers.values()],
            'topics': [t.to_dict() for t in self.topics.values()],
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'relationships': {
                'publishes_to': [e.to_dict() for e in self._edges_by_type[EdgeType.PUBLISHES_TO.value]],
                'subscribes_to': [e.to_dict() for e in self._edges_by_type[EdgeType.SUBSCRIBES_TO.value]],
                'routes': [e.to_dict() for e in self._edges_by_type[EdgeType.ROUTES.value]],
                'runs_on': [e.to_dict() for e in self._edges_by_type[EdgeType.RUNS_ON.value]],
                'connects_to': [e.to_dict() for e in self._edges_by_type[EdgeType.CONNECTS_TO.value]]
            }
        }
    
    def get_statistics(self) -> Dict:
        return {
            'num_applications': len(self.applications),
            'num_brokers': len(self.brokers),
            'num_topics': len(self.topics),
            'num_nodes': len(self.nodes),
            'num_relationships': len(self.edges),
            'edges_by_type': {k: len(v) for k, v in self._edges_by_type.items()}
        }
    
    def __repr__(self) -> str:
        return f"GraphModel(apps={len(self.applications)}, brokers={len(self.brokers)}, topics={len(self.topics)}, nodes={len(self.nodes)}, relationships={len(self.edges)})"