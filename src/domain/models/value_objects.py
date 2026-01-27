from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class QoSPolicy:
    """
    Defines Quality of Service attributes for a Topic.
    
    The QoS scoring constants can be accessed at class level for reuse:
        - RELIABILITY_SCORES: Score mapping for reliability levels
        - DURABILITY_SCORES: Score mapping for durability levels
        - PRIORITY_SCORES: Score mapping for priority levels
    """
    # QoS scoring constants - centralized for use in both Python and Cypher
    RELIABILITY_SCORES: Dict[str, float] = None  # type: ignore (class-level dict)
    DURABILITY_SCORES: Dict[str, float] = None   # type: ignore
    PRIORITY_SCORES: Dict[str, float] = None     # type: ignore
    
    durability: str = "VOLATILE"       # VOLATILE, TRANSIENT_LOCAL, TRANSIENT, PERSISTENT
    reliability: str = "BEST_EFFORT"   # BEST_EFFORT, RELIABLE
    transport_priority: str = "MEDIUM" # LOW, MEDIUM, HIGH, URGENT

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
        Calculate QoS-based weight for a topic.
        
        Formula: W_qos = S_reliability + S_durability + S_priority
        
        Note: S_size is calculated separately in Topic.calculate_weight()
        """
        s_reliability = QoSPolicy.RELIABILITY_SCORES.get(self.reliability, 0.0)
        s_durability = QoSPolicy.DURABILITY_SCORES.get(self.durability, 0.0)
        s_priority = QoSPolicy.PRIORITY_SCORES.get(self.transport_priority, 0.0)
        
        return s_reliability + s_durability + s_priority


# Initialize QoS scoring constants after class definition
QoSPolicy.RELIABILITY_SCORES = {"BEST_EFFORT": 0.0, "RELIABLE": 0.3}
QoSPolicy.DURABILITY_SCORES = {
    "VOLATILE": 0.0,
    "TRANSIENT_LOCAL": 0.2,
    "TRANSIENT": 0.25,
    "PERSISTENT": 0.4,
}
QoSPolicy.PRIORITY_SCORES = {"LOW": 0.0, "MEDIUM": 0.1, "HIGH": 0.2, "URGENT": 0.3}
