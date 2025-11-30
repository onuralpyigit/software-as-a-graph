"""
Graph Model

Core data model for representing distributed publish-subscribe systems as graphs.
Supports multi-layer dependency modeling with unified DEPENDS_ON relationships
across application, broker, and infrastructure layers.

Key Components:
- Vertices: Applications, Topics, Brokers, Infrastructure Nodes
- Explicit Edges: PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON
- Derived Edges: DEPENDS_ON (App-App, App-Broker, Node-Node)

The DEPENDS_ON relationship provides semantic consistency across all layers,
enabling uniform criticality analysis using the composite scoring formula:
C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# =============================================================================
# Enumerations
# =============================================================================

class ComponentType(Enum):
    """Types of components in a pub-sub system"""
    APPLICATION = "Application"
    TOPIC = "Topic"
    BROKER = "Broker"
    NODE = "Node"


class ApplicationType(Enum):
    """Classification of application behavior"""
    PRODUCER = "PRODUCER"  # Only publishes
    CONSUMER = "CONSUMER"  # Only subscribes
    PROSUMER = "PROSUMER"  # Both publishes and subscribes


class DependencyType(Enum):
    """
    Types of derived DEPENDS_ON relationships.
    
    Enables uniform criticality analysis across layers while preserving
    the semantic origin of each dependency.
    """
    # Application layer dependencies
    APP_TO_APP = "app_to_app"           # Derived from topic subscription overlap
    APP_TO_BROKER = "app_to_broker"     # Derived from topic routing
    
    # Infrastructure layer dependencies  
    NODE_TO_NODE = "node_to_node"       # Derived from application dependencies
    NODE_TO_BROKER = "node_to_broker"   # Derived from broker placement


class QoSReliability(Enum):
    """DDS/ROS2 Reliability QoS"""
    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"


class QoSDurability(Enum):
    """DDS/ROS2 Durability QoS"""
    VOLATILE = "volatile"
    TRANSIENT_LOCAL = "transient_local"
    TRANSIENT = "transient"
    PERSISTENT = "persistent"


class QosTransportPriority(Enum):
    """Transport priority levels"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    URGENT = 3


# =============================================================================
# QoS Policy
# =============================================================================

@dataclass
class QoSPolicy:
    """
    Quality of Service policy for topics.
    
    QoS policies influence criticality scoring - topics with strict
    reliability requirements or high priority are inherently more critical.
    """
    reliability: QoSReliability = QoSReliability.RELIABLE
    durability: QoSDurability = QoSDurability.VOLATILE
    deadline_ms: Optional[float] = None
    lifespan_ms: Optional[float] = None
    history_depth: int = 10
    transport_priority: QosTransportPriority = QosTransportPriority.MEDIUM
    
    def get_criticality_score(self) -> float:
        """
        Calculate QoS-based criticality contribution.
        
        Returns:
            Float between 0.0 and 1.0 indicating QoS criticality
        """
        score = 0.0
        
        # Reliability contributes up to 0.3
        if self.reliability == QoSReliability.RELIABLE:
            score += 0.3
        
        # Durability contributes up to 0.2
        durability_scores = {
            QoSDurability.VOLATILE: 0.0,
            QoSDurability.TRANSIENT_LOCAL: 0.05,
            QoSDurability.TRANSIENT: 0.1,
            QoSDurability.PERSISTENT: 0.2
        }
        score += durability_scores.get(self.durability, 0.0)
        
        # Deadline contributes up to 0.25 (tighter deadline = higher criticality)
        if self.deadline_ms is not None:
            if self.deadline_ms <= 10:
                score += 0.25
            elif self.deadline_ms <= 100:
                score += 0.15
            elif self.deadline_ms <= 1000:
                score += 0.05
        
        # Transport priority contributes up to 0.25
        priority_scores = {
            QosTransportPriority.LOW: 0.0,
            QosTransportPriority.MEDIUM: 0.05,
            QosTransportPriority.HIGH: 0.15,
            QosTransportPriority.URGENT: 0.25
        }
        score += priority_scores.get(self.transport_priority, 0.0)
        
        return min(score, 1.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'reliability': self.reliability.value,
            'durability': self.durability.value,
            'deadline_ms': self.deadline_ms,
            'lifespan_ms': self.lifespan_ms,
            'history_depth': self.history_depth,
            'transport_priority': self.transport_priority.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QoSPolicy':
        """Create QoSPolicy from dictionary"""
        return cls(
            reliability=QoSReliability(data.get('reliability', 'reliable')),
            durability=QoSDurability(data.get('durability', 'volatile')),
            deadline_ms=data.get('deadline_ms'),
            lifespan_ms=data.get('lifespan_ms'),
            history_depth=data.get('history_depth', 10),
            transport_priority=QosTransportPriority(data.get('transport_priority', 1))
        )


# =============================================================================
# Node Types (Vertices)
# =============================================================================

@dataclass
class ApplicationNode:
    """
    Application component in the pub-sub system.
    
    Applications are the primary actors - they publish and subscribe to topics,
    creating the data flow patterns that define system dependencies.
    """
    id: str
    name: str
    app_type: ApplicationType = ApplicationType.PROSUMER
    component_type: ComponentType = ComponentType.APPLICATION
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'name': self.name,
            'app_type': self.app_type.value,
            'component_type': self.component_type.value
        }


@dataclass
class TopicNode:
    """
    Topic/channel in the pub-sub system.
    
    Topics are the communication medium - they connect publishers to subscribers
    and carry QoS policies that influence message delivery guarantees.
    """
    id: str
    name: str
    qos: Optional[QoSPolicy] = None
    message_size_bytes: Optional[float] = None  # Average message size
    message_rate_hz: Optional[float] = None     # Average message rate
    component_type: ComponentType = ComponentType.TOPIC
    
    def get_qos_criticality(self) -> float:
        """Get QoS-based criticality score"""
        if self.qos:
            return self.qos.get_criticality_score()
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        result = {
            'id': self.id,
            'name': self.name,
            'durability': self.qos.durability.value,
            'reliability': self.qos.reliability.value,
            'deadline_ms': self.qos.deadline_ms,
            'lifespan_ms': self.qos.lifespan_ms,
            'transport_priority': self.qos.transport_priority,
            'history_depth': self.qos.history_depth,
            'message_size_bytes': self.message_size_bytes,
            'message_rate_hz': self.message_rate_hz,
            'component_type': self.component_type.value,
        }
        return result


@dataclass
class BrokerNode:
    """
    Message broker in the pub-sub system.
    
    Brokers route messages between publishers and subscribers. They are
    infrastructure components that can become critical single points of failure.
    """
    id: str
    name: str
    broker_type: str = "generic"  # e.g., "kafka", "rabbitmq", "dds", "ros2"
    component_type: ComponentType = ComponentType.BROKER
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'name': self.name,
            'broker_type': self.broker_type
        }


@dataclass
class InfrastructureNode:
    """
    Physical or virtual infrastructure node.
    
    Infrastructure nodes host applications and brokers. Dependencies between
    infrastructure nodes are derived from the application-level dependencies
    of their hosted components.
    """
    id: str
    name: str
    component_type: ComponentType = ComponentType.NODE
    node_type: str = "compute"  # e.g., "compute", "edge", "cloud", "gateway"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        result = {
            'id': self.id,
            'name': self.name,
            'node_type': self.node_type,
            'component_type': self.component_type.value
        }
        return result


# =============================================================================
# Edge Types (Relationships)
# =============================================================================

@dataclass
class PublishesEdge:
    """
    Application publishes to Topic.
    
    Explicit relationship capturing the data production pattern.
    """
    source: str  # Application ID
    target: str  # Topic ID
    period_ms: Optional[float] = None  # Publishing period
    message_size_bytes: Optional[float] = None  # Average message size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target,
            'period_ms': self.period_ms,
            'message_size_bytes': self.message_size_bytes
        }


@dataclass
class SubscribesEdge:
    """
    Application subscribes to Topic.
    
    Explicit relationship capturing the data consumption pattern.
    """
    source: str  # Application ID
    target: str  # Topic ID
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


@dataclass
class RoutesEdge:
    """
    Broker routes Topic.
    
    Explicit relationship indicating which broker handles which topic.
    """
    source: str  # Broker ID
    target: str  # Topic ID
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


@dataclass
class RunsOnEdge:
    """
    Application or Broker runs on Infrastructure Node.
    
    Explicit relationship mapping software to hardware.
    """
    source: str  # Application or Broker ID
    target: str  # Node ID
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


@dataclass 
class DependsOnEdge:
    """
    Unified dependency relationship across all layers.
    
    This edge type provides semantic consistency for criticality analysis,
    representing operational dependencies at multiple levels:
    
    - APP_TO_APP: Application A depends on Application B (via topic subscription)
    - APP_TO_BROKER: Application depends on Broker (for topic routing)
    - NODE_TO_NODE: Infrastructure Node X depends on Node Y (derived from app dependencies)
    - NODE_TO_BROKER: Infrastructure Node depends on Broker node
    
    The unified DEPENDS_ON relationship enables the composite criticality formula
    to work uniformly across all system layers.
    """
    source: str
    target: str
    dependency_type: DependencyType
    
    # Provenance: what this dependency was derived from
    derived_from: List[str] = field(default_factory=list)
    
    # For APP_TO_APP: the topic(s) creating the dependency
    topics: List[str] = field(default_factory=list)
    
    # Dependency strength (for weighted analysis)
    # Higher weight = stronger coupling
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target,
            'dependency_type': self.dependency_type.value,
            'derived_from': self.derived_from,
            'topics': self.topics,
            'weight': self.weight
        }
    
    @property
    def is_cross_layer(self) -> bool:
        """Check if this is a cross-layer dependency"""
        return self.dependency_type in [
            DependencyType.APP_TO_BROKER,
            DependencyType.NODE_TO_BROKER
        ]


@dataclass
class ConnectsToEdge:
    """
    Physical/network connectivity between infrastructure nodes.
    
    This is an OPTIONAL explicit relationship for modeling physical network
    topology separately from operational dependencies. Use this when you have
    explicit network topology information (e.g., from infrastructure config).
    
    If not provided, infrastructure connectivity is inferred from the
    NODE_TO_NODE DEPENDS_ON relationships.
    """
    source: str  # Node ID
    target: str  # Node ID
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'source': self.source,
            'target': self.target
        }


# =============================================================================
# Graph Model Container
# =============================================================================

class GraphModel:
    """
    Core graph data model for distributed pub-sub systems.
    
    This model captures:
    1. System components (applications, topics, brokers, infrastructure)
    2. Explicit relationships (publishes, subscribes, routes, runs_on)
    3. Derived dependencies (DEPENDS_ON across all layers)
    
    The unified DEPENDS_ON relationship model enables consistent criticality
    analysis across application, broker, and infrastructure layers.
    """
    
    def __init__(self):
        # Vertices (nodes)
        self.applications: Dict[str, ApplicationNode] = {}
        self.topics: Dict[str, TopicNode] = {}
        self.brokers: Dict[str, BrokerNode] = {}
        self.nodes: Dict[str, InfrastructureNode] = {}
        
        # Explicit edges
        self.publishes_edges: List[PublishesEdge] = []
        self.subscribes_edges: List[SubscribesEdge] = []
        self.routes_edges: List[RoutesEdge] = []
        self.runs_on_edges: List[RunsOnEdge] = []
        
        # Optional physical connectivity (explicit)
        self.connects_to_edges: List[ConnectsToEdge] = []
        
        # Derived dependency edges (unified DEPENDS_ON)
        self.depends_on_edges: List[DependsOnEdge] = []
    
    # =========================================================================
    # Node Management
    # =========================================================================
    
    def add_application(self, app: ApplicationNode):
        """Add application node"""
        self.applications[app.id] = app
    
    def add_topic(self, topic: TopicNode):
        """Add topic node"""
        self.topics[topic.id] = topic
    
    def add_broker(self, broker: BrokerNode):
        """Add broker node"""
        self.brokers[broker.id] = broker
    
    def add_node(self, node: InfrastructureNode):
        """Add infrastructure node"""
        self.nodes[node.id] = node
    
    # =========================================================================
    # Accessors
    # =========================================================================
    
    def get_all_nodes(self) -> Dict[str, Dict]:
        """Get all nodes as dictionaries"""
        all_nodes = {}
        
        for app_id, app in self.applications.items():
            all_nodes[app_id] = app.to_dict()
        
        for topic_id, topic in self.topics.items():
            all_nodes[topic_id] = topic.to_dict()
        
        for broker_id, broker in self.brokers.items():
            all_nodes[broker_id] = broker.to_dict()
        
        for node_id, node in self.nodes.items():
            all_nodes[node_id] = node.to_dict()
        
        return all_nodes
    
    def get_component_type(self, component_id: str) -> Optional[ComponentType]:
        """Get the type of a component by ID"""
        if component_id in self.applications:
            return ComponentType.APPLICATION
        elif component_id in self.topics:
            return ComponentType.TOPIC
        elif component_id in self.brokers:
            return ComponentType.BROKER
        elif component_id in self.nodes:
            return ComponentType.NODE
        return None
    
    def get_depends_on_by_type(self, dep_type: DependencyType) -> List[DependsOnEdge]:
        """Get all DEPENDS_ON edges of a specific type"""
        return [e for e in self.depends_on_edges if e.dependency_type == dep_type]
    
    def get_app_dependencies(self) -> List[DependsOnEdge]:
        """Get application-level dependencies (APP_TO_APP and APP_TO_BROKER)"""
        return [
            e for e in self.depends_on_edges 
            if e.dependency_type in [DependencyType.APP_TO_APP, DependencyType.APP_TO_BROKER]
        ]
    
    def get_infrastructure_dependencies(self) -> List[DependsOnEdge]:
        """Get infrastructure-level dependencies (NODE_TO_NODE and NODE_TO_BROKER)"""
        return [
            e for e in self.depends_on_edges
            if e.dependency_type in [DependencyType.NODE_TO_NODE, DependencyType.NODE_TO_BROKER]
        ]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""
        # Count dependencies by type
        dep_counts = {}
        for dep_type in DependencyType:
            dep_counts[dep_type.value] = len(self.get_depends_on_by_type(dep_type))
        
        return {
            'vertices': {
                'applications': len(self.applications),
                'topics': len(self.topics),
                'brokers': len(self.brokers),
                'nodes': len(self.nodes),
                'total': len(self.applications) + len(self.topics) + 
                        len(self.brokers) + len(self.nodes)
            },
            'explicit_edges': {
                'publishes_to': len(self.publishes_edges),
                'subscribes_to': len(self.subscribes_edges),
                'routes': len(self.routes_edges),
                'runs_on': len(self.runs_on_edges),
                'connects_to': len(self.connects_to_edges),
                'total': len(self.publishes_edges) + len(self.subscribes_edges) +
                        len(self.routes_edges) + len(self.runs_on_edges) +
                        len(self.connects_to_edges)
            },
            'derived_edges': {
                'depends_on': len(self.depends_on_edges),
                'by_type': dep_counts
            }
        }
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Convert entire model to dictionary"""
        return {
            'applications': [app.to_dict() for app in self.applications.values()],
            'topics': [topic.to_dict() for topic in self.topics.values()],
            'brokers': [broker.to_dict() for broker in self.brokers.values()],
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': {
                'publishes_to': [e.to_dict() for e in self.publishes_edges],
                'subscribes_to': [e.to_dict() for e in self.subscribes_edges],
                'routes': [e.to_dict() for e in self.routes_edges],
                'runs_on': [e.to_dict() for e in self.runs_on_edges],
                'connects_to': [e.to_dict() for e in self.connects_to_edges],
                'depends_on': [e.to_dict() for e in self.depends_on_edges]
            },
            'statistics': self.get_statistics()
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"GraphModel("
            f"apps={stats['vertices']['applications']}, "
            f"topics={stats['vertices']['topics']}, "
            f"brokers={stats['vertices']['brokers']}, "
            f"nodes={stats['vertices']['nodes']}, "
            f"depends_on={stats['derived_edges']['depends_on']})"
        )