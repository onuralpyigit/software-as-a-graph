from dataclasses import dataclass, field
from typing import Dict, Any
from .types import ComponentState

@dataclass
class ComponentInfo:
    """Information about a component in the simulation."""
    id: str
    type: str  # Application, Topic, Broker, Node
    state: ComponentState = ComponentState.ACTIVE
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime metrics (accumulated during simulation)
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    messages_routed: int = 0
    total_latency: float = 0.0
    
    def reset_metrics(self) -> None:
        """Reset runtime metrics for a new simulation run."""
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0
        self.messages_routed = 0
        self.total_latency = 0.0
    
    @property
    def avg_latency(self) -> float:
        """Average latency per message processed."""
        total = self.messages_received + self.messages_routed
        return self.total_latency / total if total > 0 else 0.0
    
    @property
    def throughput(self) -> int:
        """Total messages processed (sent + routed)."""
        return self.messages_sent + self.messages_routed


@dataclass
class TopicInfo:
    """Information about a topic including QoS settings."""
    id: str
    name: str
    message_size: int = 1024
    qos_reliability: str = "BEST_EFFORT"  # BEST_EFFORT, RELIABLE
    qos_durability: str = "VOLATILE"      # VOLATILE, TRANSIENT, PERSISTENT
    qos_priority: str = "LOW"             # LOW, MEDIUM, HIGH, URGENT
    weight: float = 1.0
    
    @property
    def requires_ack(self) -> bool:
        """Check if topic requires acknowledgment (reliable delivery)."""
        return self.qos_reliability == "RELIABLE"
    
    @property
    def priority_value(self) -> int:
        """Numeric priority for scheduling."""
        return {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.qos_priority, 1)
    
    @property
    def persistence_factor(self) -> float:
        """Factor for persistence overhead."""
        return {"PERSISTENT": 1.5, "TRANSIENT": 1.2, "VOLATILE": 1.0}.get(self.qos_durability, 1.0)
