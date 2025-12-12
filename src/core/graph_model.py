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
from typing import Dict, List, Optional, Any
from enum import Enum


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


@dataclass
class QoSPolicy:
    """Quality of Service policy for topics (simplified)"""
    durability: str = "VOLATILE"
    reliability: str = "BEST_EFFORT"
    transport_priority: str = "MEDIUM"
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QoSPolicy':
        """Create QoSPolicy from dictionary"""
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
        """
        Calculate QoS-based criticality score (0-1)
        Higher score = more critical
        """
        score = 0.0
        
        # Durability score (0.33 weight)
        durability_scores = {
            'VOLATILE': 0.0,
            'TRANSIENT_LOCAL': 0.33,
            'TRANSIENT': 0.66,
            'PERSISTENT': 1.0
        }
        score += 0.33 * durability_scores.get(self.durability, 0.0)
        
        # Reliability score (0.34 weight)
        reliability_scores = {
            'BEST_EFFORT': 0.0,
            'RELIABLE': 1.0
        }
        score += 0.34 * reliability_scores.get(self.reliability, 0.0)
        
        # Priority score (0.33 weight)
        priority_scores = {
            'LOW': 0.0,
            'MEDIUM': 0.33,
            'HIGH': 0.66,
            'URGENT': 1.0
        }
        score += 0.33 * priority_scores.get(self.transport_priority, 0.33)
        
        return round(score, 3)


@dataclass
class Application:
    """Application vertex"""
    id: str
    name: str
    role: str = "pubsub"  # pub, sub, pubsub
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role
        }


@dataclass
class Broker:
    """Broker vertex"""
    id: str
    name: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name
        }


@dataclass
class Topic:
    """Topic vertex"""
    id: str
    name: str
    size: int = 256  # message size in bytes
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'size': self.size,
            'qos': self.qos.to_dict() if isinstance(self.qos, QoSPolicy) else self.qos
        }
    
    def get_qos_criticality(self) -> float:
        """Get QoS-based criticality score"""
        if isinstance(self.qos, QoSPolicy):
            return self.qos.get_criticality_score()
        return 0.0


@dataclass
class Node:
    """Infrastructure node vertex"""
    id: str
    name: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name
        }


@dataclass
class Edge:
    """Generic edge in the graph"""
    source: str  # from
    target: str  # to
    edge_type: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'from': self.source,
            'to': self.target
        }


class GraphModel:
    """
    Container for the complete graph model (simplified)
    
    Vertices:
    - Applications
    - Brokers
    - Topics
    - Nodes
    
    Edges:
    - PUBLISHES_TO
    - SUBSCRIBES_TO
    - ROUTES
    - RUNS_ON
    - CONNECTS_TO
    """
    
    def __init__(self):
        """Initialize empty graph model"""
        self.applications: Dict[str, Application] = {}
        self.brokers: Dict[str, Broker] = {}
        self.topics: Dict[str, Topic] = {}
        self.nodes: Dict[str, Node] = {}
        
        self.edges: List[Edge] = []
        self.metadata: Dict = {}
        
        # Index edges by type for fast lookup
        self._edges_by_type: Dict[str, List[Edge]] = {
            'PUBLISHES_TO': [],
            'SUBSCRIBES_TO': [],
            'ROUTES': [],
            'RUNS_ON': [],
            'CONNECTS_TO': []
        }
    
    def add_application(self, app: Application):
        """Add application vertex"""
        self.applications[app.id] = app
    
    def add_broker(self, broker: Broker):
        """Add broker vertex"""
        self.brokers[broker.id] = broker
    
    def add_topic(self, topic: Topic):
        """Add topic vertex"""
        self.topics[topic.id] = topic
    
    def add_node(self, node: Node):
        """Add infrastructure node vertex"""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge):
        """Add edge and update index"""
        self.edges.append(edge)
        
        if edge.edge_type in self._edges_by_type:
            self._edges_by_type[edge.edge_type].append(edge)
    
    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        """Get all edges of a specific type"""
        return self._edges_by_type.get(edge_type, [])
    
    def get_vertex(self, vertex_id: str) -> Optional[Any]:
        """Get any vertex by ID"""
        if vertex_id in self.applications:
            return self.applications[vertex_id]
        if vertex_id in self.brokers:
            return self.brokers[vertex_id]
        if vertex_id in self.topics:
            return self.topics[vertex_id]
        if vertex_id in self.nodes:
            return self.nodes[vertex_id]
        return None
    
    def to_dict(self) -> Dict:
        """Export complete model to dictionary"""
        return {
            'metadata': self.metadata,
            'applications': [a.to_dict() for a in self.applications.values()],
            'brokers': [b.to_dict() for b in self.brokers.values()],
            'topics': [t.to_dict() for t in self.topics.values()],
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': {
                'publishes_to': [e.to_dict() for e in self._edges_by_type['PUBLISHES_TO']],
                'subscribes_to': [e.to_dict() for e in self._edges_by_type['SUBSCRIBES_TO']],
                'routes': [e.to_dict() for e in self._edges_by_type['ROUTES']],
                'runs_on': [e.to_dict() for e in self._edges_by_type['RUNS_ON']],
                'connects_to': [e.to_dict() for e in self._edges_by_type['CONNECTS_TO']]
            }
        }
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'num_applications': len(self.applications),
            'num_brokers': len(self.brokers),
            'num_topics': len(self.topics),
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'edges_by_type': {k: len(v) for k, v in self._edges_by_type.items()}
        }