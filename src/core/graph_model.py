"""
Graph Data Model for Software-as-a-Graph Analysis

This module defines the data model for representing distributed pub-sub systems
as multi-layer graphs with comprehensive properties for analysis.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

class QoSDurability(Enum):
    """QoS Durability Policy"""
    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"
    TRANSIENT = "TRANSIENT"
    PERSISTENT = "PERSISTENT"


class QoSReliability(Enum):
    """QoS Reliability Policy"""
    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE = "RELIABLE"

class QosTransportPriority(Enum):
    """QoS Transport Priority Levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

class ComponentType(Enum):
    """Types of components in the system"""
    APPLICATION = "Application"
    TOPIC = "Topic"
    BROKER = "Broker"
    NODE = "Node"

class ApplicationType(Enum):
    """Classification of application behavior"""
    PRODUCER = "PRODUCER"  # Only publishes
    CONSUMER = "CONSUMER"  # Only subscribes
    PROSUMER = "PROSUMER"  # Both publishes and subscribes

class MessagePattern(Enum):
    """Message Pattern Types"""
    EVENT_DRIVEN = "EVENT_DRIVEN"
    REQUEST_RESPONSE = "REQUEST_RESPONSE"
    PERIODIC = "PERIODIC"
    STREAMING = "STREAMING"

@dataclass
class QoSPolicy:
    """Quality of Service policies for topics"""
    durability: QoSDurability = QoSDurability.VOLATILE
    reliability: QoSReliability = QoSReliability.BEST_EFFORT
    deadline_ms: Optional[float] = None  # None = infinite
    lifespan_ms: Optional[float] = None  # None = infinite
    transport_priority: QosTransportPriority = QosTransportPriority.MEDIUM
    history_depth: int = 1
    
    def get_criticality_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate QoS-based criticality score
        
        Args:
            weights: Optional dict with keys: durability, reliability, deadline, 
                    lifespan, transport_priority, history
        
        Returns:
            Composite QoS score [0, 1]
        """
        if weights is None:
            weights = {
                'durability': 0.20,
                'reliability': 0.25,
                'deadline': 0.20,
                'lifespan': 0.10,
                'transport_priority': 0.15,
                'history': 0.10
            }
        
        # Durability score
        durability_map = {
            QoSDurability.VOLATILE: 0.2,
            QoSDurability.TRANSIENT_LOCAL: 0.5,
            QoSDurability.TRANSIENT: 0.7,
            QoSDurability.PERSISTENT: 1.0
        }
        durability_score = durability_map[self.durability]
        
        # Reliability score
        reliability_score = 1.0 if self.reliability == QoSReliability.RELIABLE else 0.3
        
        # Deadline score (inverse exponential - shorter deadline = more critical)
        if self.deadline_ms is None or self.deadline_ms == float('inf'):
            deadline_score = 0.1
        else:
            import math
            deadline_score = 1.0 - math.exp(-1000 / self.deadline_ms)
        
        # Lifespan score (log transformation - longer lifespan = more critical)
        if self.lifespan_ms is None or self.lifespan_ms == float('inf'):
            lifespan_score = 0.1
        else:
            import math
            lifespan_score = math.log(self.lifespan_ms + 1) / math.log(86400000)  # Normalized to 24h
        
        # Transport priority score (normalized)
        transport_score_map = {
            QosTransportPriority.LOW: 0.25,
            QosTransportPriority.MEDIUM: 0.5,
            QosTransportPriority.HIGH: 0.75,
            QosTransportPriority.CRITICAL: 1.0
        }
        transport_score = transport_score_map[self.transport_priority]
        
        # History depth score (log scale)
        import math
        history_score = min(1.0, math.log(self.history_depth + 1) / math.log(100))
        
        # Composite score
        score = (
            weights['durability'] * durability_score +
            weights['reliability'] * reliability_score +
            weights['deadline'] * deadline_score +
            weights['lifespan'] * lifespan_score +
            weights['transport_priority'] * transport_score +
            weights['history'] * history_score
        )
        
        return round(score, 3)


@dataclass
class ApplicationNode:
    """Application component in the system"""
    id: str
    name: str
    type: str
    component_type: ComponentType = ComponentType.APPLICATION
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'component_type': self.component_type.value
        }


@dataclass
class TopicNode:
    """Topic (message channel) in the system"""
    id: str
    name: str
    
    # QoS Policies
    qos_policy: QoSPolicy = field(default_factory=QoSPolicy)
    
    # Traffic Characteristics
    message_size_bytes: float = 0.0
    message_rate_hz: float = 0.0

    component_type: ComponentType = ComponentType.TOPIC
    
    def get_qos_criticality(self) -> float:
        """Get QoS-based criticality score for this topic"""
        return self.qos_policy.get_criticality_score()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'name': self.name,
            'durability': self.qos_policy.durability.value,
            'reliability': self.qos_policy.reliability.value,
            'deadline_ms': self.qos_policy.deadline_ms,
            'lifespan_ms': self.qos_policy.lifespan_ms,
            'transport_priority': self.qos_policy.transport_priority,
            'history_depth': self.qos_policy.history_depth,
            'message_size_bytes': self.message_size_bytes,
            'message_rate_hz': self.message_rate_hz,
            'component_type': self.component_type.value,
        }


@dataclass
class BrokerNode:
    """Message broker in the system"""
    id: str
    name: str
    component_type: ComponentType = ComponentType.BROKER
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'name': self.name,
            'component_type': self.component_type.value
        }


@dataclass
class InfrastructureNode:
    """Physical or virtual machine hosting components"""
    id: str
    name: str
    component_type: ComponentType = ComponentType.NODE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'name': self.name,
            'component_type': self.component_type.value
        }
    
class PublishesEdge:
    """Application publishes to Topic"""
    def __init__(self, source: str, target: str, period_ms: Optional[float] = None, msg_size_bytes: Optional[float] = None):
        self.source = source
        self.target = target
        self.period_ms = period_ms
        self.msg_size_bytes = msg_size_bytes

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target,
            'period_ms': self.period_ms,
            'msg_size_bytes': self.msg_size_bytes
        }


class SubscribesEdge:
    """Application subscribes to Topic"""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


class RoutesEdge:
    """Broker routes Topic"""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


class RunsOnEdge:
    """Application runs on Infrastructure Node"""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }

class ConnectsToEdge:
    """Broker connects to Broker"""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


class DependsOnEdge:
    """Derived dependency relationship"""
    def __init__(self, source: str, target: str, topic: str):
        self.source = source
        self.target = target
        self.topic = topic

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target,
            'topic': self.topic
        }


class GraphModel:
    """
    Core graph data model for distributed pub-sub systems
    """
    def __init__(self):
        self.applications: Dict[str, ApplicationNode] = {}
        self.topics: Dict[str, TopicNode] = {}
        self.brokers: Dict[str, BrokerNode] = {}
        self.nodes: Dict[str, InfrastructureNode] = {}
        
        self.publishes_edges: List[PublishesEdge] = []
        self.subscribes_edges: List[SubscribesEdge] = []
        self.routes_edges: List[RoutesEdge] = []
        self.runs_on_edges: List[RunsOnEdge] = []
        self.connects_to_edges: List[ConnectsToEdge] = []
        self.depends_on_edges: List[DependsOnEdge] = []
    
    def add_application(self, app: ApplicationNode):
        """Add application node"""
        self.applications[app.id] = app
    
    def add_topic(self, topic: TopicNode):
        """Add topic node"""
        self.topics[topic.id] = topic
    
    def add_broker(self, broker: BrokerNode):
        """Add broker node"""
        self.brokers[broker.id] = broker
    
    def add_node(self, node: InfrastructureNode):
        """Add infrastructure node"""
        self.nodes[node.id] = node

    def get_all_nodes(self) -> Dict[str, Dict]:
        """Get all nodes as dictionaries"""
        all_nodes = {}
        
        for app in self.applications.values():
            all_nodes[app.id] = app.to_dict()
        
        for topic in self.topics.values():
            all_nodes[topic.id] = topic.to_dict()
        
        for broker in self.brokers.values():
            all_nodes[broker.id] = broker.to_dict()
        
        for node in self.nodes.values():
            all_nodes[node.id] = node.to_dict()
        
        return all_nodes
    
    def get_all_edges(self) -> List[Dict]:
        """Get all edges as dictionaries"""
        all_edges = []
        
        for edge_list in [
            self.publishes_edges,
            self.subscribes_edges,
            self.routes_edges,
            self.runs_on_edges,
            self.connects_to_edges,
            self.depends_on_edges
        ]:
            for edge in edge_list:
                edge_dict = edge.to_dict()
                edge_dict['source'] = edge.source
                edge_dict['target'] = edge.target
                all_edges.append(edge_dict)
        
        return all_edges
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary statistics"""
        return {
            'total_nodes': len(self.applications) + len(self.topics) + 
                          len(self.brokers) + len(self.nodes),
            'applications': len(self.applications),
            'topics': len(self.topics),
            'brokers': len(self.brokers),
            'infrastructure_nodes': len(self.nodes),
            'total_edges': len(self.publishes_edges) + len(self.subscribes_edges) +
                          len(self.routes_edges) + len(self.runs_on_edges) +
                          len(self.connects_to_edges) + len(self.depends_on_edges),
            'publishes': len(self.publishes_edges),
            'subscribes': len(self.subscribes_edges),
            'routes': len(self.routes_edges),
            'runs_on': len(self.runs_on_edges),
            'connects_to': len(self.connects_to_edges),
            'depends_on': len(self.depends_on_edges)
        }