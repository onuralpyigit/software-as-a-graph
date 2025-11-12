"""
Graph Data Model for Software-as-a-Graph Analysis

This module defines the data model for representing distributed pub-sub systems
as multi-layer graphs with comprehensive properties for analysis.
"""

from typing import Dict, List, Optional, Any
from enum import Enum

# ============================================================================
# Data Classes
# ============================================================================

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


class ApplicationType(Enum):
    """Application Type Classification"""
    PUBLISHER = "PUBLISHER"
    SUBSCRIBER = "SUBSCRIBER"
    BIDIRECTIONAL = "BIDIRECTIONAL"


class MessagePattern(Enum):
    """Message Pattern Types"""
    EVENT_DRIVEN = "EVENT_DRIVEN"
    REQUEST_RESPONSE = "REQUEST_RESPONSE"
    PERIODIC = "PERIODIC"
    STREAMING = "STREAMING"


# ============================================================================
# Core Classes
# ============================================================================

class QoSPolicy:
    """Quality of Service Policy"""
    def __init__(self,
                 durability: QoSDurability = QoSDurability.VOLATILE,
                 reliability: QoSReliability = QoSReliability.BEST_EFFORT,
                 deadline_ms: Optional[float] = None,
                 lifespan_ms: Optional[float] = None,
                 transport_priority: int = 0,
                 history_depth: int = 1):
        self.durability = durability
        self.reliability = reliability
        self.deadline_ms = deadline_ms
        self.lifespan_ms = lifespan_ms
        self.transport_priority = transport_priority
        self.history_depth = history_depth


class ApplicationNode:
    """Application/Process Node"""
    def __init__(self, name: str, app_type: ApplicationType,
                 qos_policy: Optional[QoSPolicy] = None, **kwargs):
        self.name = name
        self.app_type = app_type
        self.qos_policy = qos_policy or QoSPolicy()
        self.properties = kwargs


class TopicNode:
    """Topic/Channel Node"""
    def __init__(self, name: str, message_type: str = "unknown",
                 qos_policy: Optional[QoSPolicy] = None, **kwargs):
        self.name = name
        self.message_type = message_type
        self.qos_policy = qos_policy or QoSPolicy()
        self.properties = kwargs


class BrokerNode:
    """Broker/Router Node"""
    def __init__(self, name: str, broker_type: str = "DDS",
                 max_topics: int = 1000, **kwargs):
        self.name = name
        self.broker_type = broker_type
        self.max_topics = max_topics
        self.properties = kwargs


class InfrastructureNode:
    """Infrastructure/Physical Node"""
    def __init__(self, name: str, cpu_cores: int = 1,
                 memory_gb: float = 4.0, **kwargs):
        self.name = name
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.properties = kwargs


class PublishesEdge:
    """Application publishes to Topic"""
    def __init__(self, source: str, target: str, **kwargs):
        self.source = source
        self.target = target
        self.properties = kwargs


class SubscribesEdge:
    """Application subscribes to Topic"""
    def __init__(self, source: str, target: str, **kwargs):
        self.source = source
        self.target = target
        self.properties = kwargs


class RoutesEdge:
    """Broker routes Topic"""
    def __init__(self, source: str, target: str, **kwargs):
        self.source = source
        self.target = target
        self.properties = kwargs


class RunsOnEdge:
    """Application runs on Infrastructure Node"""
    def __init__(self, source: str, target: str, **kwargs):
        self.source = source
        self.target = target
        self.properties = kwargs


class ConnectsToEdge:
    """Broker connects to Broker"""
    def __init__(self, source: str, target: str, **kwargs):
        self.source = source
        self.target = target
        self.properties = kwargs


class DependsOnEdge:
    """Derived dependency relationship"""
    def __init__(self, source: str, target: str,
                 dependency_type: str = "FUNCTIONAL", strength: float = 1.0):
        self.source = source
        self.target = target
        self.dependency_type = dependency_type
        self.strength = strength


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
        self.applications[app.name] = app
    
    def add_topic(self, topic: TopicNode):
        """Add topic node"""
        self.topics[topic.name] = topic
    
    def add_broker(self, broker: BrokerNode):
        """Add broker node"""
        self.brokers[broker.name] = broker
    
    def add_node(self, node: InfrastructureNode):
        """Add infrastructure node"""
        self.nodes[node.name] = node
    
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