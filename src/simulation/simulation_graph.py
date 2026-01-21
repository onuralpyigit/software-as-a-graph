"""
Simulation Graph

Graph representation for simulation using RAW structural relationships.
Works directly on PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
relationships without deriving DEPENDS_ON.

Supports:
    - Event propagation simulation (pub-sub message flow)
    - Failure cascade simulation (component/infrastructure failures)
    - Runtime metrics extraction (throughput, latency, drops)

Retrieves graph data directly from Neo4j for simulation.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional, Iterator
from enum import Enum
from collections import defaultdict

import networkx as nx


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


class SimulationGraph:
    """
    Graph representation for pub-sub system simulation.
    
    Works on RAW structural relationships:
        - Application -[PUBLISHES_TO]-> Topic
        - Application -[SUBSCRIBES_TO]-> Topic
        - Topic -[ROUTES]-> Broker (or Broker -[ROUTES]-> Topic)
        - Application -[RUNS_ON]-> Node
        - Broker -[RUNS_ON]-> Node
        - Node -[CONNECTS_TO]-> Node
    
    Retrieves data directly from Neo4j for simulation.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: str = "neo4j",
        password: str = "password",
        graph_data: Optional[Any] = None
    ):
        """
        Initialize simulation graph.
        
        Args:
            uri: Neo4j connection URI (if loading from database)
            user: Neo4j username
            password: Neo4j password
            graph_data: Pre-loaded GraphData object (alternative to Neo4j)
        """
        self.logger = logging.getLogger(__name__)
        
        # NetworkX graph for structural queries
        self.graph = nx.DiGraph()
        
        # Component registries
        self.components: Dict[str, ComponentInfo] = {}
        self.topics: Dict[str, TopicInfo] = {}
        
        # Relationship indices for fast lookups
        self._publishers: Dict[str, List[str]] = defaultdict(list)   # topic -> [apps]
        self._subscribers: Dict[str, List[str]] = defaultdict(list)  # topic -> [apps]
        self._routing: Dict[str, List[str]] = defaultdict(list)      # topic -> [brokers]
        self._hosted_on: Dict[str, str] = {}                         # comp -> node
        self._hosts: Dict[str, List[str]] = defaultdict(list)        # node -> [comps]
        self._connections: Dict[str, List[str]] = defaultdict(list)  # node -> [nodes]
        
        # Load graph
        if graph_data:
            self._load_from_data(graph_data)
        elif uri:
            self._load_from_neo4j(uri, user, password)
    
    def _load_from_neo4j(self, uri: str, user: str, password: str) -> None:
        """Load graph data directly from Neo4j."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j driver required. Install with: pip install neo4j")
        
        self.logger.info(f"Loading simulation graph from Neo4j: {uri}")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        try:
            with driver.session() as session:
                # Load all components
                self._load_components_from_neo4j(session)
                
                # Load all relationships
                self._load_relationships_from_neo4j(session)
        finally:
            driver.close()
        
        self.logger.info(
            f"Loaded: {len(self.components)} components, "
            f"{len(self.topics)} topics, "
            f"{self.graph.number_of_edges()} edges"
        )
    
    def _load_components_from_neo4j(self, session) -> None:
        """Load components from Neo4j."""
        # Load Applications, Brokers, Nodes
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node
        RETURN n.id as id, labels(n)[0] as type, properties(n) as props
        """
        result = session.run(query)
        for record in result:
            comp_id = record["id"]
            comp_type = record["type"]
            props = dict(record["props"])
            props.pop("id", None)
            
            self.components[comp_id] = ComponentInfo(
                id=comp_id,
                type=comp_type,
                weight=props.pop("weight", 1.0),
                properties=props
            )
            self.graph.add_node(comp_id, type=comp_type, **props)
        
        # Load Topics separately (with QoS info)
        topic_query = """
        MATCH (t:Topic)
        RETURN t.id as id, t.name as name, properties(t) as props
        """
        result = session.run(topic_query)
        for record in result:
            topic_id = record["id"]
            props = dict(record["props"])
            
            self.topics[topic_id] = TopicInfo(
                id=topic_id,
                name=record["name"] or topic_id,
                message_size=props.get("message_size", 1024),
                qos_reliability=props.get("qos_reliability", "BEST_EFFORT"),
                qos_durability=props.get("qos_durability", "VOLATILE"),
                qos_priority=props.get("qos_priority", "LOW"),
                weight=props.get("weight", 1.0),
            )
            
            # Also add to components
            self.components[topic_id] = ComponentInfo(
                id=topic_id,
                type="Topic",
                weight=self.topics[topic_id].weight,
                properties=props
            )
            self.graph.add_node(topic_id, type="Topic", **props)
    
    def _load_relationships_from_neo4j(self, session) -> None:
        """Load relationships from Neo4j."""
        # PUBLISHES_TO: App -> Topic
        query = """
        MATCH (a:Application)-[r:PUBLISHES_TO]->(t:Topic)
        RETURN a.id as source, t.id as target, properties(r) as props
        """
        result = session.run(query)
        for record in result:
            src, tgt = record["source"], record["target"]
            self._publishers[tgt].append(src)
            self.graph.add_edge(src, tgt, relation=RelationType.PUBLISHES_TO.value)
        
        # SUBSCRIBES_TO: App -> Topic
        query = """
        MATCH (a:Application)-[r:SUBSCRIBES_TO]->(t:Topic)
        RETURN a.id as source, t.id as target, properties(r) as props
        """
        result = session.run(query)
        for record in result:
            src, tgt = record["source"], record["target"]
            self._subscribers[tgt].append(src)
            self.graph.add_edge(src, tgt, relation=RelationType.SUBSCRIBES_TO.value)
        
        # ROUTES: Broker <-> Topic (direction may vary)
        query = """
        MATCH (b:Broker)-[r:ROUTES]->(t:Topic)
        RETURN b.id as broker, t.id as topic
        UNION
        MATCH (t:Topic)-[r:ROUTES]->(b:Broker)
        RETURN b.id as broker, t.id as topic
        """
        result = session.run(query)
        for record in result:
            broker, topic = record["broker"], record["topic"]
            if broker not in self._routing[topic]:
                self._routing[topic].append(broker)
            self.graph.add_edge(topic, broker, relation=RelationType.ROUTES.value)
        
        # RUNS_ON: App/Broker -> Node
        query = """
        MATCH (c)-[r:RUNS_ON]->(n:Node)
        WHERE c:Application OR c:Broker
        RETURN c.id as comp, n.id as node
        """
        result = session.run(query)
        for record in result:
            comp, node = record["comp"], record["node"]
            self._hosted_on[comp] = node
            self._hosts[node].append(comp)
            self.graph.add_edge(comp, node, relation=RelationType.RUNS_ON.value)
        
        # CONNECTS_TO: Node -> Node
        query = """
        MATCH (n1:Node)-[r:CONNECTS_TO]->(n2:Node)
        RETURN n1.id as source, n2.id as target
        """
        result = session.run(query)
        for record in result:
            src, tgt = record["source"], record["target"]
            self._connections[src].append(tgt)
            self.graph.add_edge(src, tgt, relation=RelationType.CONNECTS_TO.value)
    
    def _load_from_data(self, graph_data: Any) -> None:
        """Load from pre-loaded GraphData object."""
        self.logger.info("Loading simulation graph from GraphData")
        
        # Load components
        for comp in graph_data.components:
            comp_id = comp.id if hasattr(comp, 'id') else comp.get('id')
            comp_type = comp.component_type if hasattr(comp, 'component_type') else comp.get('type')
            props = comp.properties if hasattr(comp, 'properties') else {}
            
            if comp_type == "Topic":
                self.topics[comp_id] = TopicInfo(
                    id=comp_id,
                    name=props.get("name", comp_id),
                    message_size=props.get("message_size", 1024),
                    qos_reliability=props.get("qos_reliability", "BEST_EFFORT"),
                    qos_durability=props.get("qos_durability", "VOLATILE"),
                    qos_priority=props.get("qos_priority", "LOW"),
                    weight=props.get("weight", 1.0),
                )
            
            self.components[comp_id] = ComponentInfo(
                id=comp_id,
                type=comp_type,
                weight=props.get("weight", 1.0),
                properties=props
            )
            self.graph.add_node(comp_id, type=comp_type)
        
        # Load edges
        for edge in graph_data.edges:
            src = edge.source_id if hasattr(edge, 'source_id') else edge.get('source')
            tgt = edge.target_id if hasattr(edge, 'target_id') else edge.get('target')
            rel = edge.relation_type if hasattr(edge, 'relation_type') else edge.get('relation_type', 'UNKNOWN')
            
            self.graph.add_edge(src, tgt, relation=rel)
            
            # Index by relationship type
            if rel == "PUBLISHES_TO":
                self._publishers[tgt].append(src)
            elif rel == "SUBSCRIBES_TO":
                self._subscribers[tgt].append(src)
            elif rel == "ROUTES":
                self._routing[tgt].append(src)
            elif rel == "RUNS_ON":
                self._hosted_on[src] = tgt
                self._hosts[tgt].append(src)
            elif rel == "CONNECTS_TO":
                self._connections[src].append(tgt)
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def reset(self) -> None:
        """Reset all component states and metrics for a new simulation."""
        for comp in self.components.values():
            comp.state = ComponentState.ACTIVE
            comp.reset_metrics()
    
    def fail_component(self, comp_id: str) -> None:
        """Mark a component as failed."""
        if comp_id in self.components:
            self.components[comp_id].state = ComponentState.FAILED
    
    def recover_component(self, comp_id: str) -> None:
        """Recover a failed component."""
        if comp_id in self.components:
            self.components[comp_id].state = ComponentState.ACTIVE
    
    def is_active(self, comp_id: str) -> bool:
        """Check if a component is active."""
        comp = self.components.get(comp_id)
        return comp is not None and comp.state == ComponentState.ACTIVE
    
    def set_degraded(self, comp_id: str) -> None:
        """Mark a component as degraded."""
        if comp_id in self.components:
            self.components[comp_id].state = ComponentState.DEGRADED
    
    # =========================================================================
    # Graph Queries
    # =========================================================================
    
    def get_publishers(self, topic_id: str) -> List[str]:
        """Get all publishers for a topic."""
        return [p for p in self._publishers.get(topic_id, []) if self.is_active(p)]
    
    def get_subscribers(self, topic_id: str) -> List[str]:
        """Get all subscribers for a topic."""
        return [s for s in self._subscribers.get(topic_id, []) if self.is_active(s)]
    
    def get_routing_brokers(self, topic_id: str) -> List[str]:
        """Get all brokers that route a topic."""
        return [b for b in self._routing.get(topic_id, []) if self.is_active(b)]
    
    def get_hosted_components(self, node_id: str) -> List[str]:
        """Get all components hosted on a node."""
        return self._hosts.get(node_id, [])
    
    def get_host_node(self, comp_id: str) -> Optional[str]:
        """Get the node that hosts a component."""
        return self._hosted_on.get(comp_id)
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get nodes connected to a given node."""
        return [n for n in self._connections.get(node_id, []) if self.is_active(n)]
    
    def get_app_topics(self, app_id: str) -> Tuple[List[str], List[str]]:
        """Get topics an application publishes to and subscribes from."""
        publishes = []
        subscribes = []
        
        for topic_id, publishers in self._publishers.items():
            if app_id in publishers:
                publishes.append(topic_id)
        
        for topic_id, subscribers in self._subscribers.items():
            if app_id in subscribers:
                subscribes.append(topic_id)
        
        return publishes, subscribes
    
    def get_pub_sub_paths(self, active_only: bool = True) -> List[Tuple[str, str, str]]:
        """
        Get all publisher -> topic -> subscriber paths.
        
        Returns:
            List of (publisher, topic, subscriber) tuples
        """
        paths = []
        
        for topic_id in self.topics:
            publishers = self._publishers.get(topic_id, [])
            subscribers = self._subscribers.get(topic_id, [])
            
            if active_only:
                publishers = [p for p in publishers if self.is_active(p)]
                subscribers = [s for s in subscribers if self.is_active(s)]
            
            for pub in publishers:
                for sub in subscribers:
                    paths.append((pub, topic_id, sub))
        
        return paths
    
    def get_message_path(self, publisher: str, topic_id: str) -> List[Tuple[str, str]]:
        """
        Get the path a message takes from publisher to subscribers.
        
        Returns:
            List of (from, to) tuples representing the path
        """
        path = []
        
        # Publisher -> Topic
        path.append((publisher, topic_id))
        
        # Topic -> Broker(s)
        brokers = self.get_routing_brokers(topic_id)
        for broker in brokers:
            path.append((topic_id, broker))
        
        # Broker(s) -> Subscribers
        subscribers = self.get_subscribers(topic_id)
        for broker in brokers:
            for sub in subscribers:
                path.append((broker, sub))
        
        return path
    
    # =========================================================================
    # Layer Filtering
    # =========================================================================
    
    def get_components_by_layer(self, layer: str) -> List[str]:
        """
        Get component IDs for a specific layer.
        
        Layers:
            - app: Application components
            - infra: Node components
            - mw-app: Application + Broker components
            - mw-infra: Node + Broker components
            - system: All components
        """
        layer_types = {
            "app": {"Application"},
            "infra": {"Node"},
            "mw-app": {"Application", "Broker"},
            "mw-infra": {"Node", "Broker"},
            "system": {"Application", "Broker", "Node", "Topic", "Library"},
        }
        
        types = layer_types.get(layer, layer_types["system"])
        return [c.id for c in self.components.values() if c.type in types]
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the graph."""
        type_counts = defaultdict(int)
        for comp in self.components.values():
            type_counts[comp.type] += 1
        
        return {
            "total_nodes": len(self.components),
            "total_edges": self.graph.number_of_edges(),
            "component_types": dict(type_counts),
            "topics": len(self.topics),
            "pub_sub_paths": len(self.get_pub_sub_paths()),
            "active_components": sum(1 for c in self.components.values() if c.state == ComponentState.ACTIVE),
        }