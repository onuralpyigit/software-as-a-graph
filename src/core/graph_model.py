"""
Graph Model

Defines the core data structures for the Pub-Sub system graph.
Includes logic for QoS scoring and component weighting.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional

class VertexType(str, Enum):
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"

class EdgeType(str, Enum):
    RUNS_ON = "RUNS_ON"
    ROUTES = "ROUTES"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    CONNECTS_TO = "CONNECTS_TO"
    DEPENDS_ON = "DEPENDS_ON"

class DependencyType(str, Enum):
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"

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

@dataclass
class GraphEntity:
    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Application(GraphEntity):
    role: str = "pubsub"  # pub, sub, pubsub

@dataclass
class Broker(GraphEntity):
    pass

@dataclass
class Node(GraphEntity):
    pass

@dataclass
class Topic(GraphEntity):
    size: int = 256
    qos: QoSPolicy = field(default_factory=QoSPolicy)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['qos'] = self.qos.to_dict()
        return data