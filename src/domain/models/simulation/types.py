from enum import Enum

class ComponentState(Enum):
    """State of a component during simulation."""
    ACTIVE = "active"
    FAILED = "failed"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"


class RelationType(Enum):
    """RAW structural relationship types in the pub-sub graph."""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"
    USES = "USES"


class EventType(Enum):
    """Types of discrete events in the simulation."""
    PUBLISH = "publish"       # Message published by app
    ROUTE = "route"           # Message routed by broker
    DELIVER = "deliver"       # Message delivered to subscriber
    ACK = "ack"               # Acknowledgment (for reliable delivery)
    TIMEOUT = "timeout"       # Delivery timeout
    DROP = "drop"             # Message dropped


class FailureMode(Enum):
    """
    Types of component failure modes.
    """
    CRASH = "crash"           # Complete failure - component stops
    DEGRADED = "degraded"     # Partial failure - reduced capacity
    PARTITION = "partition"   # Network partition - unreachable
    OVERLOAD = "overload"     # Resource exhaustion


class CascadeRule(Enum):
    """Rules governing failure cascade propagation."""
    PHYSICAL = "physical"     # Node failure cascades to hosted components
    LOGICAL = "logical"       # Broker failure affects topic routing
    NETWORK = "network"       # Network partition propagation
    ALL = "all"               # All cascade rules applied
