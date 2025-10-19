"""
Graph Data Model for Software-as-a-Graph Analysis

This module defines the data model for representing distributed pub-sub systems
as multi-layer graphs with comprehensive properties for analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime


class ComponentType(Enum):
    """Types of components in the system"""
    APPLICATION = "Application"
    TOPIC = "Topic"
    BROKER = "Broker"
    NODE = "Node"


class ApplicationType(Enum):
    """Classification of application behavior"""
    PRODUCER = "Producer"  # Only publishes
    CONSUMER = "Consumer"  # Only subscribes
    PROSUMER = "Prosumer"  # Both publishes and subscribes


class MessagePattern(Enum):
    """Message communication patterns"""
    PERIODIC = "Periodic"
    BURST = "Burst"
    EVENT_DRIVEN = "EventDriven"
    REQUEST_RESPONSE = "RequestResponse"


class QoSDurability(Enum):
    """DDS Durability QoS Policy"""
    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"
    TRANSIENT = "TRANSIENT"
    PERSISTENT = "PERSISTENT"


class QoSReliability(Enum):
    """DDS Reliability QoS Policy"""
    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE = "RELIABLE"


@dataclass
class QoSPolicy:
    """Quality of Service policies for topics"""
    durability: QoSDurability = QoSDurability.VOLATILE
    reliability: QoSReliability = QoSReliability.BEST_EFFORT
    deadline_ms: Optional[float] = None  # None = infinite
    lifespan_ms: Optional[float] = None  # None = infinite
    transport_priority: int = 0  # 0-100 scale
    history_depth: int = 1
    
    def get_criticality_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate QoS-based criticality score
        
        Args:
            weights: Optional dict with keys: durability, reliability, deadline, 
                    lifespan, transport_priority, history
        
        Returns:
            Composite QoS score [0, 1]
        """
        if weights is None:
            weights = {
                'durability': 0.20,
                'reliability': 0.25,
                'deadline': 0.20,
                'lifespan': 0.10,
                'transport_priority': 0.15,
                'history': 0.10
            }
        
        # Durability score
        durability_map = {
            QoSDurability.VOLATILE: 0.2,
            QoSDurability.TRANSIENT_LOCAL: 0.5,
            QoSDurability.TRANSIENT: 0.7,
            QoSDurability.PERSISTENT: 1.0
        }
        durability_score = durability_map[self.durability]
        
        # Reliability score
        reliability_score = 1.0 if self.reliability == QoSReliability.RELIABLE else 0.3
        
        # Deadline score (inverse exponential - shorter deadline = more critical)
        if self.deadline_ms is None or self.deadline_ms == float('inf'):
            deadline_score = 0.1
        else:
            import math
            deadline_score = 1.0 - math.exp(-1000 / self.deadline_ms)
        
        # Lifespan score (log transformation - longer lifespan = more critical)
        if self.lifespan_ms is None or self.lifespan_ms == float('inf'):
            lifespan_score = 0.1
        else:
            import math
            lifespan_score = math.log(self.lifespan_ms + 1) / math.log(86400000)  # Normalized to 24h
        
        # Transport priority score (normalized)
        transport_score = self.transport_priority / 100
        
        # History depth score (log scale)
        import math
        history_score = min(1.0, math.log(self.history_depth + 1) / math.log(100))
        
        # Composite score
        score = (
            weights['durability'] * durability_score +
            weights['reliability'] * reliability_score +
            weights['deadline'] * deadline_score +
            weights['lifespan'] * lifespan_score +
            weights['transport_priority'] * transport_score +
            weights['history'] * history_score
        )
        
        return round(score, 3)


@dataclass
class ApplicationNode:
    """Application component in the system"""
    name: str
    component_type: ComponentType = ComponentType.APPLICATION
    app_type: ApplicationType = ApplicationType.PROSUMER
    node_host: Optional[str] = None
    
    # QoS Requirements
    required_latency_ms: Optional[float] = None
    required_throughput_msgs_per_sec: Optional[float] = None
    
    # Resource Requirements
    cpu_cores: float = 1.0
    memory_mb: float = 512.0
    
    # Runtime Metrics
    actual_latency_ms: Optional[float] = None
    actual_throughput: Optional[float] = None
    health_score: float = 1.0  # 0-1 scale
    
    # Metadata
    business_domain: Optional[str] = None
    team_owner: Optional[str] = None
    version: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'name': self.name,
            'type': self.component_type.value,
            'app_type': self.app_type.value,
            'node_host': self.node_host,
            'required_latency_ms': self.required_latency_ms,
            'required_throughput_msgs_per_sec': self.required_throughput_msgs_per_sec,
            'cpu_cores': self.cpu_cores,
            'memory_mb': self.memory_mb,
            'actual_latency_ms': self.actual_latency_ms,
            'actual_throughput': self.actual_throughput,
            'health_score': self.health_score,
            'business_domain': self.business_domain,
            'team_owner': self.team_owner,
            'version': self.version,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class TopicNode:
    """Topic (message channel) in the system"""
    name: str
    component_type: ComponentType = ComponentType.TOPIC
    broker: Optional[str] = None
    
    # QoS Policies
    qos_policy: QoSPolicy = field(default_factory=QoSPolicy)
    
    # Traffic Characteristics
    message_rate_per_sec: float = 0.0
    avg_message_size_bytes: float = 0.0
    peak_message_rate: float = 0.0
    
    # Schema Information
    message_type: Optional[str] = None
    schema_version: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def get_qos_criticality(self) -> float:
        """Get QoS-based criticality score for this topic"""
        return self.qos_policy.get_criticality_score()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'name': self.name,
            'type': self.component_type.value,
            'broker': self.broker,
            'durability': self.qos_policy.durability.value,
            'reliability': self.qos_policy.reliability.value,
            'deadline_ms': self.qos_policy.deadline_ms,
            'lifespan_ms': self.qos_policy.lifespan_ms,
            'transport_priority': self.qos_policy.transport_priority,
            'history_depth': self.qos_policy.history_depth,
            'qos_score': self.get_qos_criticality(),
            'message_rate_per_sec': self.message_rate_per_sec,
            'avg_message_size_bytes': self.avg_message_size_bytes,
            'peak_message_rate': self.peak_message_rate,
            'message_type': self.message_type,
            'schema_version': self.schema_version,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class BrokerNode:
    """Message broker in the system"""
    name: str
    component_type: ComponentType = ComponentType.BROKER
    node_host: Optional[str] = None
    
    # Configuration
    max_connections: int = 1000
    max_throughput_msgs_per_sec: float = 10000.0
    replication_factor: int = 1
    partition_count: int = 1
    
    # Performance Metrics
    current_connections: int = 0
    current_throughput: float = 0.0
    avg_latency_ms: float = 0.0
    cpu_utilization: float = 0.0  # 0-1 scale
    memory_utilization: float = 0.0  # 0-1 scale
    
    # Health
    health_score: float = 1.0
    uptime_seconds: float = 0.0
    
    # Metadata
    broker_type: str = "Generic"  # Kafka, RabbitMQ, MQTT, etc.
    version: Optional[str] = None
    
    def get_capacity_utilization(self) -> float:
        """Calculate overall capacity utilization"""
        connection_util = self.current_connections / self.max_connections
        throughput_util = self.current_throughput / self.max_throughput_msgs_per_sec
        return (connection_util + throughput_util + 
                self.cpu_utilization + self.memory_utilization) / 4
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'name': self.name,
            'type': self.component_type.value,
            'node_host': self.node_host,
            'max_connections': self.max_connections,
            'max_throughput_msgs_per_sec': self.max_throughput_msgs_per_sec,
            'replication_factor': self.replication_factor,
            'partition_count': self.partition_count,
            'current_connections': self.current_connections,
            'current_throughput': self.current_throughput,
            'avg_latency_ms': self.avg_latency_ms,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'capacity_utilization': self.get_capacity_utilization(),
            'health_score': self.health_score,
            'uptime_seconds': self.uptime_seconds,
            'broker_type': self.broker_type,
            'version': self.version
        }


@dataclass
class InfrastructureNode:
    """Physical or virtual machine hosting components"""
    name: str
    component_type: ComponentType = ComponentType.NODE
    
    # Location Hierarchy
    datacenter: Optional[str] = None
    rack: Optional[str] = None
    zone: Optional[str] = None
    region: Optional[str] = None
    
    # Capacity
    total_cpu_cores: float = 4.0
    total_memory_mb: float = 8192.0
    total_disk_gb: float = 100.0
    network_bandwidth_mbps: float = 1000.0
    
    # Utilization
    cpu_utilization: float = 0.0  # 0-1 scale
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_utilization: float = 0.0
    
    # Health
    health_score: float = 1.0
    uptime_seconds: float = 0.0
    
    # Metadata
    node_type: str = "VM"  # VM, Container, Physical, etc.
    os: Optional[str] = None
    ip_address: Optional[str] = None
    
    def get_resource_utilization(self) -> float:
        """Calculate overall resource utilization"""
        return (self.cpu_utilization + self.memory_utilization + 
                self.disk_utilization + self.network_utilization) / 4
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'name': self.name,
            'type': self.component_type.value,
            'datacenter': self.datacenter,
            'rack': self.rack,
            'zone': self.zone,
            'region': self.region,
            'total_cpu_cores': self.total_cpu_cores,
            'total_memory_mb': self.total_memory_mb,
            'total_disk_gb': self.total_disk_gb,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'disk_utilization': self.disk_utilization,
            'network_utilization': self.network_utilization,
            'resource_utilization': self.get_resource_utilization(),
            'health_score': self.health_score,
            'uptime_seconds': self.uptime_seconds,
            'node_type': self.node_type,
            'os': self.os,
            'ip_address': self.ip_address
        }


@dataclass
class Edge:
    """Base class for edges in the graph"""
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j"""
        return {
            'type': self.edge_type,
            'weight': self.weight
        }


@dataclass
class PublishesEdge(Edge):
    """Application publishes to Topic"""
    edge_type: str = "PUBLISHES_TO"
    message_pattern: MessagePattern = MessagePattern.EVENT_DRIVEN
    message_rate_per_sec: float = 0.0
    is_synchronous: bool = False
    timeout_ms: Optional[float] = None
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'message_pattern': self.message_pattern.value,
            'message_rate_per_sec': self.message_rate_per_sec,
            'is_synchronous': self.is_synchronous,
            'timeout_ms': self.timeout_ms
        })
        return base


@dataclass
class SubscribesEdge(Edge):
    """Application subscribes to Topic"""
    edge_type: str = "SUBSCRIBES_TO"
    filter_expression: Optional[str] = None
    qos_compatible: bool = True
    acknowledgment_mode: str = "AUTO"  # AUTO, MANUAL
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'filter_expression': self.filter_expression,
            'qos_compatible': self.qos_compatible,
            'acknowledgment_mode': self.acknowledgment_mode
        })
        return base


@dataclass
class RoutesEdge(Edge):
    """Broker routes Topic"""
    edge_type: str = "ROUTES"
    routing_weight: float = 1.0  # Based on topic criticality
    partition_count: int = 1
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'routing_weight': self.routing_weight,
            'partition_count': self.partition_count
        })
        return base


@dataclass
class RunsOnEdge(Edge):
    """Application/Broker runs on Infrastructure Node"""
    edge_type: str = "RUNS_ON"
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'resource_allocation': str(self.resource_allocation)
        })
        return base


@dataclass
class ConnectsToEdge(Edge):
    """Network connectivity between Nodes"""
    edge_type: str = "CONNECTS_TO"
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 1.0
    packet_loss_rate: float = 0.0
    is_redundant: bool = False
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'bandwidth_mbps': self.bandwidth_mbps,
            'latency_ms': self.latency_ms,
            'packet_loss_rate': self.packet_loss_rate,
            'is_redundant': self.is_redundant
        })
        return base


@dataclass
class DependsOnEdge(Edge):
    """Derived dependency between components"""
    edge_type: str = "DEPENDS_ON"
    dependency_type: str = "FUNCTIONAL"  # FUNCTIONAL, TEMPORAL, RESOURCE
    strength: float = 1.0  # 0-1 scale
    is_critical: bool = False
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'dependency_type': self.dependency_type,
            'strength': self.strength,
            'is_critical': self.is_critical
        })
        return base


class GraphModel:
    """Complete graph model for the system"""
    
    def __init__(self):
        self.applications: Dict[str, ApplicationNode] = {}
        self.topics: Dict[str, TopicNode] = {}
        self.brokers: Dict[str, BrokerNode] = {}
        self.nodes: Dict[str, InfrastructureNode] = {}
        
        self.publishes_edges: List[PublishesEdge] = []
        self.subscribes_edges: List[SubscribesEdge] = []
        self.routes_edges: List[RoutesEdge] = []
        self.runs_on_edges: List[RunsOnEdge] = []
        self.connects_to_edges: List[ConnectsToEdge] = []
        self.depends_on_edges: List[DependsOnEdge] = []
    
    def add_application(self, app: ApplicationNode):
        """Add application to the model"""
        self.applications[app.name] = app
    
    def add_topic(self, topic: TopicNode):
        """Add topic to the model"""
        self.topics[topic.name] = topic
    
    def add_broker(self, broker: BrokerNode):
        """Add broker to the model"""
        self.brokers[broker.name] = broker
    
    def add_node(self, node: InfrastructureNode):
        """Add infrastructure node to the model"""
        self.nodes[node.name] = node
    
    def get_all_nodes(self) -> Dict[str, Dict]:
        """Get all nodes as dictionaries"""
        all_nodes = {}
        
        for app in self.applications.values():
            all_nodes[app.name] = app.to_dict()
        
        for topic in self.topics.values():
            all_nodes[topic.name] = topic.to_dict()
        
        for broker in self.brokers.values():
            all_nodes[broker.name] = broker.to_dict()
        
        for node in self.nodes.values():
            all_nodes[node.name] = node.to_dict()
        
        return all_nodes
    
    def get_all_edges(self) -> List[Dict]:
        """Get all edges as dictionaries"""
        all_edges = []
        
        for edge_list in [
            self.publishes_edges,
            self.subscribes_edges,
            self.routes_edges,
            self.runs_on_edges,
            self.connects_to_edges,
            self.depends_on_edges
        ]:
            for edge in edge_list:
                edge_dict = edge.to_dict()
                edge_dict['source'] = edge.source
                edge_dict['target'] = edge.target
                all_edges.append(edge_dict)
        
        return all_edges
    
    def summary(self) -> Dict:
        """Get summary statistics of the model"""
        return {
            'applications': len(self.applications),
            'topics': len(self.topics),
            'brokers': len(self.brokers),
            'nodes': len(self.nodes),
            'total_nodes': (len(self.applications) + len(self.topics) + 
                           len(self.brokers) + len(self.nodes)),
            'publishes_edges': len(self.publishes_edges),
            'subscribes_edges': len(self.subscribes_edges),
            'routes_edges': len(self.routes_edges),
            'runs_on_edges': len(self.runs_on_edges),
            'connects_to_edges': len(self.connects_to_edges),
            'depends_on_edges': len(self.depends_on_edges),
            'total_edges': (len(self.publishes_edges) + len(self.subscribes_edges) +
                           len(self.routes_edges) + len(self.runs_on_edges) +
                           len(self.connects_to_edges) + len(self.depends_on_edges))
        }
