from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import math

from .value_objects import QoSPolicy

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
        id: Unique identifier for the library
        name: Name of the library
        version: Optional version string for the library
    
    Communication Patterns:
        Direct:   App -> Topic
        Indirect: App -> Lib -> Topic
        Chained:  App -> Lib -> Lib -> Topic
    """
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
        }
        if self.version:
            result["version"] = self.version
        return result
