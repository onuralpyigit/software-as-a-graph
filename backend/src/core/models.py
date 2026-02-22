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
        "RELIABLE": 1.0,  # Full weight if reliable
    }
    DURABILITY_SCORES: ClassVar[Dict[str, float]] = {
        "VOLATILE": 0.0,
        "TRANSIENT_LOCAL": 0.5,
        "TRANSIENT": 0.6,
        "PERSISTENT": 1.0,
    }
    PRIORITY_SCORES: ClassVar[Dict[str, float]] = {
        "LOW": 0.0,
        "MEDIUM": 0.33,
        "HIGH": 0.66,
        "URGENT": 1.0,
    }
    
    # Justification (AHP): 
    # Durability (0.4) > Reliability (0.3) = Priority (0.3)
    # Rationale: In DDS systems, durability defines state survival which is 
    # fundamentally critical for resilience, while reliability/priority 
    # govern transient delivery quality.
    W_RELIABILITY: ClassVar[float] = 0.30
    W_DURABILITY: ClassVar[float] = 0.40
    W_PRIORITY: ClassVar[float] = 0.30

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
        """
        Calculates the weighted QoS score based on AHP-derived coefficients.
        
        QoS = 0.30*Rel + 0.40*Dur + 0.30*Pri
        """
        s_reliability = self.RELIABILITY_SCORES.get(self.reliability, 0.0)
        s_durability = self.DURABILITY_SCORES.get(self.durability, 0.0)
        s_priority = self.PRIORITY_SCORES.get(self.transport_priority, 0.0)
        
        return (
            self.W_RELIABILITY * s_reliability + 
            self.W_DURABILITY * s_durability + 
            self.W_PRIORITY * s_priority
        )

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
        """
        Topic importance = QoS_Score + Size_Penalty.
        
        Refined Size Penalty: 
        - Logarithmic scaling to avoid dominance by massive messages.
        - Capped at 0.20 (max 20% of total importance) to preserve 
          QoS-driven semantics.
        - Divisor of 50 ensures size only pushes a topic into a higher
          criticality bracket if it is significantly larger than typical
          DDS control packets (e.g. > 100KB).
        """
        qos_weight = self.qos.calculate_weight()
        # size / 1024 converts to KB
        size_kb = self.size / 1024
        size_weight = min(math.log2(1 + size_kb) / 50, 0.20)
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