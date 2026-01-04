"""
Graph Model

Defines the core data structures for the Pub-Sub system graph.
Includes logic for QoS scoring and component weighting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional

# =============================================================================
# Enums & Constants
# =============================================================================

class VertexType(str, Enum):
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"

class DependencyType(str, Enum):
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"

# =============================================================================
# Quality of Service (QoS)
# =============================================================================

@dataclass(frozen=True)
class QoSPolicy:
    """
    Defines Quality of Service attributes for a Topic.
    Used to calculate the criticality/weight of dependencies.
    """
    durability: str = "VOLATILE"       # VOLATILE, TRANSIENT, PERSISTENT
    reliability: str = "BEST_EFFORT"   # BEST_EFFORT, RELIABLE
    priority: str = "MEDIUM"           # LOW, MEDIUM, HIGH, URGENT

    def criticality_score(self) -> float:
        """
        Calculate a scalar score (0.0 - 1.0) representing QoS criticality.
        """
        score = 0.0
        
        # Durability contribution (Max 0.4)
        if self.durability == "PERSISTENT": score += 0.40
        elif self.durability == "TRANSIENT": score += 0.25
        elif self.durability == "TRANSIENT_LOCAL": score += 0.20
        
        # Reliability contribution (Max 0.3)
        if self.reliability == "RELIABLE": score += 0.30
        
        # Priority contribution (Max 0.3)
        if self.priority == "URGENT": score += 0.30
        elif self.priority == "HIGH": score += 0.20
        elif self.priority == "MEDIUM": score += 0.10
        
        return round(score, 2)

    def to_dict(self) -> Dict[str, str]:
        return {
            "durability": self.durability,
            "reliability": self.reliability,
            "transport_priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> QoSPolicy:
        if not data: return cls()
        return cls(
            durability=data.get("durability", "VOLATILE"),
            reliability=data.get("reliability", "BEST_EFFORT"),
            priority=data.get("transport_priority", "MEDIUM")
        )

# =============================================================================
# Graph Entities
# =============================================================================

@dataclass
class Application:
    id: str
    name: str
    role: str = "pubsub"  # pub, sub, pubsub
    weight: float = 0.0   # Derived from relationships

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "role": self.role, "weight": self.weight}

@dataclass
class Broker:
    id: str
    name: str
    weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "weight": self.weight}

@dataclass
class Topic:
    id: str
    name: str
    size: int = 256
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, 
            "name": self.name, 
            "size": self.size, 
            "qos": self.qos.to_dict(),
            "weight": self.weight
        }

@dataclass
class Node:
    id: str
    name: str
    weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "weight": self.weight}