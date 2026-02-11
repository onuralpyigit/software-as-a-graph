"""
Domain Entities

Business entities with identity (graph elements).

Corresponds to Definition 1 in docs/graph-model.md:
    G = (V, E, τ_V, τ_E, L, w, QoS)
    
    τ_V : V → { Node, Broker, Topic, Application, Library }
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import math

from .value_objects import QoSPolicy, MIN_TOPIC_WEIGHT


@dataclass
class GraphEntity:
    """Base entity with identity. All graph vertices extend this."""
    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Application(GraphEntity):
    """
    A software service that publishes and/or subscribes to topics.
    
    τ_V(v) = Application,  L(v) = app
    
    Applications can use libraries to interact with topics indirectly.
    Weight is computed during import as:
        w(a) = Σ w(t) for direct topics + Σ w(l) for directly used libraries
    
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
    """
    A message broker routing messages between publishers and subscribers.
    
    τ_V(v) = Broker,  L(v) = mw
    
    Weight is computed during import as:
        w(b) = Σ w(t) for all topics routed by this broker
    """
    pass


@dataclass
class Node(GraphEntity):
    """
    A compute node hosting applications and/or brokers.
    
    τ_V(v) = Node,  L(v) = infra
    
    Weight is computed during import as:
        w(n) = Σ w(c) for all components hosted on this node
    """
    pass


@dataclass
class Topic(GraphEntity):
    """
    A named channel for message exchange with QoS policies.
    
    τ_V(v) = Topic,  L(v) = — (no layer; intermediary for derivation)
    
    Topics are the weight source for the entire graph. Weight formula:
        W_topic = max(ε, S_reliability + S_durability + S_priority + S_size)
    where ε = MIN_TOPIC_WEIGHT = 0.01
    
    See docs/graph-model.md §1.5 for complete scoring table.
    
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
        """
        Calculate topic weight based on QoS policy and message size.
        
        Formula:
            W_topic = max(ε, W_qos + S_size)
            
            W_qos  = S_reliability + S_durability + S_priority   ∈ [0.0, 1.0]
            S_size = min(log₂(1 + size/1024) / 10, 1.0)         ∈ [0.0, 1.0]
            
            W_topic ∈ [ε, 2.0]
        
        The minimum weight floor (ε = 0.01) ensures no topic has zero
        importance, which would propagate zero weights to all dependent
        components.
        
        Returns:
            Topic weight in range [MIN_TOPIC_WEIGHT, 2.0]
        """
        # QoS scores: S_reliability + S_durability + S_priority
        qos_weight = self.qos.calculate_weight()
        
        # Size score: S_size = min( log₂(1 + size/1024) / 10, 1.0 )
        size_weight = min(math.log2(1 + self.size / 1024) / 10, 1.0)
        
        return max(MIN_TOPIC_WEIGHT, qos_weight + size_weight)


@dataclass
class Library(GraphEntity):
    """
    A reusable code component that can publish/subscribe to topics.
    
    τ_V(v) = Library,  L(v) = app
    
    Libraries provide an abstraction layer between applications and topics.
    An application can use [0..*] libraries to interact with topics indirectly.
    Libraries can also depend on other libraries (USES relationship), creating
    transitive dependency chains captured by uses*(a) in Definition 2.
    
    Weight is computed during import as:
        w(l) = Σ w(t) for all topics this library publishes/subscribes to
    
    Attributes:
        version: Optional version string for the library
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