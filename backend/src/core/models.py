"""
Core Value Objects and Entities
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, ClassVar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum weight floor for any topic, preventing zero-importance components.
MIN_TOPIC_WEIGHT: float = 0.01

@dataclass
class ComponentData:
    """Domain entity representing a graph component (vertex)."""
    id: str
    component_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.component_type,
            "weight": self.weight,
            **self.properties,
        }


@dataclass
class EdgeData:
    """Domain entity representing a graph edge (dependency)."""
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    dependency_type: str
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "dependency_type": self.dependency_type,
            "relation_type": self.relation_type,
            "weight": self.weight,
            **self.properties,
        }


@dataclass
class GraphData:
    """Domain entity representing a complete graph with components and edges."""
    components: List[ComponentData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
        }
    
    def get_components_by_type(self, comp_type: str) -> List[ComponentData]:
        return [c for c in self.components if c.component_type == comp_type]
    
    def get_edges_by_type(self, dep_type: str) -> List[EdgeData]:
        return [e for e in self.edges if e.dependency_type == dep_type]


@dataclass
class QoSPolicy:
    """
    Defines Quality of Service attributes for a Topic.
    """
    # QoS scoring constants - centralized for use in both Python and Cypher
    RELIABILITY_SCORES: ClassVar[Dict[str, float]] = {
        "BEST_EFFORT": 0.0,
        "RELIABLE": 0.3,
    }
    DURABILITY_SCORES: ClassVar[Dict[str, float]] = {
        "VOLATILE": 0.0,
        "TRANSIENT_LOCAL": 0.2,
        "TRANSIENT": 0.25,
        "PERSISTENT": 0.4,
    }
    PRIORITY_SCORES: ClassVar[Dict[str, float]] = {
        "LOW": 0.0,
        "MEDIUM": 0.1,
        "HIGH": 0.2,
        "URGENT": 0.3,
    }
    
    durability: str = "VOLATILE"
    reliability: str = "BEST_EFFORT"
    transport_priority: str = "MEDIUM"

    def to_dict(self) -> Dict[str, str]:
        return {
            "durability": self.durability,
            "reliability": self.reliability,
            "transport_priority": self.transport_priority,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QoSPolicy":
        return QoSPolicy(
            durability=data.get("durability", "VOLATILE"),
            reliability=data.get("reliability", "BEST_EFFORT"),
            transport_priority=data.get("transport_priority", "MEDIUM")
        )
    
    def calculate_weight(self) -> float:
        s_reliability = self.RELIABILITY_SCORES.get(self.reliability, 0.0)
        s_durability = self.DURABILITY_SCORES.get(self.durability, 0.0)
        s_priority = self.PRIORITY_SCORES.get(self.transport_priority, 0.0)
        return s_reliability + s_durability + s_priority

@dataclass
class GraphEntity:
    """Base entity with identity. All graph vertices extend this."""
    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Application(GraphEntity):
    """A software service that publishes and/or subscribes to topics."""
    role: str = "pubsub"  # pub, sub, pubsub
    app_type: str = "service"
    criticality: bool = False
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
    """A message broker routing messages."""
    pass

@dataclass
class Node(GraphEntity):
    """A compute node hosting applications and/or brokers."""
    pass

@dataclass
class Topic(GraphEntity):
    """A named channel for message exchange with QoS policies."""
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
        qos_weight = self.qos.calculate_weight()
        size_weight = min(math.log2(1 + self.size / 1024) / 10, 1.0)
        return max(MIN_TOPIC_WEIGHT, qos_weight + size_weight)

@dataclass
class Library(GraphEntity):
    """A reusable code component."""
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
        }
        if self.version:
            result["version"] = self.version
        return result

@dataclass
class GraphSummary:
    """Summary of graph structural properties."""
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    num_components: int = 0
    num_articulation_points: int = 0
    node_types: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)