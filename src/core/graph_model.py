"""
Graph Model

Defines the core data structures for the Pub-Sub system graph.
Includes logic for QoS scoring and component weighting.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional
import math

class VertexType(str, Enum):
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"
    LIBRARY = "Library"

class EdgeType(str, Enum):
    RUNS_ON = "RUNS_ON"
    ROUTES = "ROUTES"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    CONNECTS_TO = "CONNECTS_TO"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"

class DependencyType(str, Enum):
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"

class ApplicationType(str, Enum):
    """Type/category of an application in the pub-sub system."""
    SERVICE = "service"           # General microservice
    DRIVER = "driver"             # Hardware/sensor driver
    CONTROLLER = "controller"     # Control logic component
    GATEWAY = "gateway"           # External interface/API gateway
    PROCESSOR = "processor"       # Data processing component
    MONITOR = "monitor"           # Monitoring/observability component
    AGGREGATOR = "aggregator"     # Data aggregation component
    SCHEDULER = "scheduler"       # Task scheduling component
    LOGGER = "logger"             # Logging/audit component
    UI = "ui"                     # User interface component

@dataclass
class QoSPolicy:
    """
    Defines Quality of Service attributes for a Topic.
    """
    durability: str = "VOLATILE"       # VOLATILE, TRANSIENT, PERSISTENT
    reliability: str = "BEST_EFFORT"   # BEST_EFFORT, RELIABLE
    transport_priority: str = "MEDIUM" # LOW, MEDIUM, HIGH, URGENT

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> QoSPolicy:
        return QoSPolicy(
            durability=data.get("durability", "VOLATILE"),
            reliability=data.get("reliability", "BEST_EFFORT"),
            transport_priority=data.get("transport_priority", "MEDIUM")
        )
    
    def calculate_weight(self, message_size: int = 256) -> float:
        """
        Calculate QoS-based weight for a topic.
        
        Formula: W = S_reliability + S_durability + S_priority + S_size
        """
        # Reliability score
        reliability_scores = {"BEST_EFFORT": 0.0, "RELIABLE": 0.3}
        s_reliability = reliability_scores.get(self.reliability, 0.0)
        
        # Durability score
        durability_scores = {
            "VOLATILE": 0.0,
            "TRANSIENT_LOCAL": 0.2,
            "TRANSIENT": 0.25,
            "PERSISTENT": 0.4
        }
        s_durability = durability_scores.get(self.durability, 0.0)
        
        # Priority score
        priority_scores = {"LOW": 0.0, "MEDIUM": 0.1, "HIGH": 0.2, "URGENT": 0.3}
        s_priority = priority_scores.get(self.transport_priority, 0.0)
        
        return s_reliability + s_durability + s_priority

@dataclass
class GraphEntity:
    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Application(GraphEntity):
    """
    A software service that publishes and/or subscribes to topics.
    
    Applications can use libraries to interact with topics indirectly.
    
    Attributes:
        role: pub (publisher), sub (subscriber), or pubsub (both)
        app_type: Type/category of the application (service, driver, controller, etc.)
        criticality: Whether the application is critical to system operation
        version: Optional version string for the application
    """
    role: str = "pubsub"  # pub, sub, pubsub
    app_type: str = "service"  # service, driver, controller, gateway, processor, etc.
    criticality: bool = False  # True if application is critical
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "app_type": self.app_type,
            "criticality": self.criticality,
        }
        if self.version:
            result["version"] = self.version
        return result

@dataclass
class Broker(GraphEntity):
    pass

@dataclass
class Node(GraphEntity):
    pass


@dataclass
class Topic(GraphEntity):
    """
    A named channel for message exchange with QoS policies.
    
    Attributes:
        size: Message payload size in bytes
        qos: Quality of Service policy
    """
    size: int = 256
    qos: QoSPolicy = field(default_factory=QoSPolicy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "qos": self.qos.to_dict()
        }
    
    def calculate_weight(self) -> float:
        """Calculate topic weight based on QoS and size."""
        # QoS scores
        # S_reliability + S_durability + S_priority
        qos_weight = self.qos.calculate_weight()
        
        # Size score
        # S_size = min( log₂(1 + size/1024) / 10, 1.0 )
        size_weight = min(math.log2(1 + self.size / 1024) / 10, 1.0)
        
        return qos_weight + size_weight


@dataclass
class Library(GraphEntity):
    """
    A reusable code component that can publish/subscribe to topics.
    
    Libraries provide an abstraction layer between applications and topics.
    An application can use [0..*] libraries to interact with topics indirectly.
    Libraries can also depend on other libraries (USES relationship).
    
    Attributes:
        role: Derived from PUBLISHES_TO/SUBSCRIBES_TO relationships
              - pub: Only publishes to topics
              - sub: Only subscribes to topics  
              - pubsub: Both publishes and subscribes
        version: Optional version string for the library
    
    Communication Patterns:
        Direct:   App -> Topic
        Indirect: App -> Lib -> Topic
        Chained:  App -> Lib -> Lib -> Topic
    """
    role: str = "pubsub"  # pub, sub, pubsub (derived from relationships)
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "role": self.role
        }
        if self.version:
            result["version"] = self.version
        return result