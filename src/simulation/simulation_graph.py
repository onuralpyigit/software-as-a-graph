"""
Simulation Graph

Graph representation for simulation using RAW structural relationships.
Works directly on PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON relationships
without deriving DEPENDS_ON.

Supports:
- Event propagation simulation (pub-sub message flow)
- Failure cascade simulation (component/infrastructure failures)
- Runtime metrics extraction (throughput, latency, drops)

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional, Iterator
from pathlib import Path
from enum import Enum

import networkx as nx


class ComponentState(Enum):
    """State of a component in simulation."""
    ACTIVE = "active"
    FAILED = "failed"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"


class RelationType(Enum):
    """Structural relationship types in the raw graph."""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"


@dataclass
class ComponentInfo:
    """Information about a component in the simulation."""
    id: str
    type: str
    state: ComponentState = ComponentState.ACTIVE
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime metrics
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    processing_time: float = 0.0
    
    def reset_metrics(self):
        """Reset runtime metrics for new simulation run."""
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0
        self.processing_time = 0.0


@dataclass
class TopicInfo:
    """Information about a topic including QoS settings."""
    id: str
    name: str
    size: int = 0
    qos_reliability: str = "BEST_EFFORT"
    qos_durability: str = "VOLATILE"
    qos_priority: str = "LOW"
    weight: float = 1.0
    
    @property
    def requires_ack(self) -> bool:
        """Check if topic requires acknowledgment (reliable delivery)."""
        return self.qos_reliability == "RELIABLE"
    
    @property
    def priority_value(self) -> int:
        """Numeric priority value for scheduling."""
        return {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.qos_priority, 1)


class SimulationGraph:
    """
    Graph representation for pub-sub system simulation.
    
    Uses RAW structural relationships without DEPENDS_ON derivation:
    - PUBLISHES_TO: Application -> Topic (publishing)
    - SUBSCRIBES_TO: Application -> Topic (subscribing)
    - ROUTES: Broker -> Topic (message routing)
    - RUNS_ON: Application/Broker -> Node (deployment)
    - CONNECTS_TO: Node -> Node (network connectivity)
    
    Attributes:
        graph: NetworkX DiGraph with raw structural edges
        components: Dict of component ID -> ComponentInfo
        topics: Dict of topic ID -> TopicInfo
    """
    
    def __init__(
        self, 
        data: Optional[Dict] = None,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        Initialize simulation graph from JSON file, dict, or Neo4j.
        
        Args:
            data: Dict containing system definition
            uri: Neo4j connection URI
            user: Neo4j username (when using uri)
            password: Neo4j password (when using uri)
        """
        self.graph = nx.DiGraph()
        self.components: Dict[str, ComponentInfo] = {}
        self.topics: Dict[str, TopicInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self._pub_sub_paths: Optional[List[Tuple[str, str, str]]] = None
        self._initial_state: Optional[Dict[str, ComponentState]] = None
        
        # Track data source
        self._load_from_neo4j(uri, user, password)
    
    @classmethod
    def from_neo4j(
        cls,
        uri: str,
        user: str = "neo4j",
        password: str = "password"
    ) -> "SimulationGraph":
        """
        Create SimulationGraph from Neo4j database.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            
        Returns:
            SimulationGraph instance loaded from Neo4j
        """
        return cls(uri=uri, user=user, password=password)
    
    def _load_from_neo4j(self, uri: str, user: str, password: str) -> None:
        """Load graph from Neo4j database."""
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from core.graph_exporter import GraphExporter
        except ImportError:
            from src.core.graph_exporter import GraphExporter
        
        self.logger.info(f"Connecting to Neo4j at {uri}...")
        
        with GraphExporter(uri, user, password) as client:
            data = client.get_raw_structural_graph()
        
        self._load_from_dict(data)
    
    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load graph from dictionary."""
        # Clear existing data
        self.graph.clear()
        self.components.clear()
        self.topics.clear()
        
        # 1. Load Nodes (Infrastructure)
        for node in data.get("nodes", []):
            node_id = node["id"]
            self.graph.add_node(
                node_id,
                type="Node",
                name=node.get("name", node_id),
                weight=node.get("weight", 1.0),
                state=ComponentState.ACTIVE.value,
            )
            self.components[node_id] = ComponentInfo(
                id=node_id,
                type="Node",
                weight=node.get("weight", 1.0),
                properties={"name": node.get("name", node_id)}
            )
        
        # 2. Load Brokers
        for broker in data.get("brokers", []):
            broker_id = broker["id"]
            self.graph.add_node(
                broker_id,
                type="Broker",
                name=broker.get("name", broker_id),
                weight=broker.get("weight", 1.0),
                state=ComponentState.ACTIVE.value,
            )
            self.components[broker_id] = ComponentInfo(
                id=broker_id,
                type="Broker",
                weight=broker.get("weight", 1.0),
                properties={"name": broker.get("name", broker_id)}
            )
        
        # 3. Load Applications
        for app in data.get("applications", []):
            app_id = app["id"]
            self.graph.add_node(
                app_id,
                type="Application",
                name=app.get("name", app_id),
                role=app.get("role", "pubsub"),
                weight=app.get("weight", 1.0),
                state=ComponentState.ACTIVE.value,
            )
            self.components[app_id] = ComponentInfo(
                id=app_id,
                type="Application",
                weight=app.get("weight", 1.0),
                properties={
                    "name": app.get("name", app_id),
                    "role": app.get("role", "pubsub"),
                }
            )
        
        # 4. Load Topics
        for topic in data.get("topics", []):
            topic_id = topic["id"]
            qos = topic.get("qos", {})
            weight = self._compute_topic_weight(topic)
            
            self.graph.add_node(
                topic_id,
                type="Topic",
                name=topic.get("name", topic_id),
                size=topic.get("size", 0),
                weight=weight,
                state=ComponentState.ACTIVE.value,
            )
            
            self.topics[topic_id] = TopicInfo(
                id=topic_id,
                name=topic.get("name", topic_id),
                size=topic.get("size", 0),
                qos_reliability=qos.get("reliability", "BEST_EFFORT"),
                qos_durability=qos.get("durability", "VOLATILE"),
                qos_priority=qos.get("transport_priority", "LOW"),
                weight=weight,
            )
            
            # Also add to components for unified access
            self.components[topic_id] = ComponentInfo(
                id=topic_id,
                type="Topic",
                weight=weight,
                properties={
                    "name": topic.get("name", topic_id),
                    "size": topic.get("size", 0),
                    "qos": qos,
                }
            )
        
        # 5. Load Relationships
        
        # PUBLISHES_TO: Application -> Topic
        for pub in data.get("publications", []):
            app_id = pub.get("application") or pub.get("app")
            topic_id = pub.get("topic")
            if app_id and topic_id:
                topic_weight = self.topics[topic_id].weight if topic_id in self.topics else 1.0
                self.graph.add_edge(
                    app_id, topic_id,
                    relation=RelationType.PUBLISHES_TO.value,
                    weight=topic_weight,
                )
        
        # SUBSCRIBES_TO: Application -> Topic
        for sub in data.get("subscriptions", []):
            app_id = sub.get("application") or sub.get("app")
            topic_id = sub.get("topic")
            if app_id and topic_id:
                topic_weight = self.topics[topic_id].weight if topic_id in self.topics else 1.0
                self.graph.add_edge(
                    app_id, topic_id,
                    relation=RelationType.SUBSCRIBES_TO.value,
                    weight=topic_weight,
                )
        
        # ROUTES: Broker -> Topic
        for route in data.get("routes", []):
            broker_id = route.get("broker")
            topic_id = route.get("topic")
            if broker_id and topic_id:
                topic_weight = self.topics[topic_id].weight if topic_id in self.topics else 1.0
                self.graph.add_edge(
                    broker_id, topic_id,
                    relation=RelationType.ROUTES.value,
                    weight=topic_weight,
                )
        
        # RUNS_ON: Application/Broker -> Node
        for runs in data.get("runs_on", []):
            comp_id = runs.get("component") or runs.get("application") or runs.get("broker")
            node_id = runs.get("node")
            if comp_id and node_id:
                self.graph.add_edge(
                    comp_id, node_id,
                    relation=RelationType.RUNS_ON.value,
                    weight=1.0,
                )
        
        # CONNECTS_TO: Node -> Node (if provided)
        for conn in data.get("connections", data.get("connects_to", [])):
            src_node = conn.get("source") or conn.get("from")
            tgt_node = conn.get("target") or conn.get("to")
            if src_node and tgt_node:
                self.graph.add_edge(
                    src_node, tgt_node,
                    relation=RelationType.CONNECTS_TO.value,
                    weight=conn.get("weight", 1.0),
                )
        
        # Save initial state for reset
        self._save_initial_state()
        
        self.logger.info(
            f"Loaded graph: {len(self.components)} components, "
            f"{len(self.topics)} topics, {self.graph.number_of_edges()} edges"
        )
    
    def _compute_topic_weight(self, topic: Dict[str, Any]) -> float:
        """Compute topic weight from QoS and size."""
        import math
        qos = topic.get("qos", {})
        
        # Reliability score
        rel_map = {"RELIABLE": 0.3, "BEST_EFFORT": 0.0}
        s_rel = rel_map.get(qos.get("reliability", "BEST_EFFORT"), 0.0)
        
        # Durability score
        dur_map = {"PERSISTENT": 0.4, "TRANSIENT": 0.25, "TRANSIENT_LOCAL": 0.2, "VOLATILE": 0.0}
        s_dur = dur_map.get(qos.get("durability", "VOLATILE"), 0.0)
        
        # Priority score
        pri_map = {"URGENT": 0.3, "HIGH": 0.2, "MEDIUM": 0.1, "LOW": 0.0}
        s_pri = pri_map.get(qos.get("transport_priority", "LOW"), 0.0)
        
        # Size score (logarithmic)
        size = topic.get("size", 0)
        s_size = min(math.log2(1 + size / 1024) / 10, 1.0) if size > 0 else 0.0
        
        return 1.0 + s_rel + s_dur + s_pri + s_size
    
    def _save_initial_state(self) -> None:
        """Save initial state for reset."""
        self._initial_state = {}
        for node_id in self.graph.nodes():
            self._initial_state[node_id] = ComponentState.ACTIVE
        self._pub_sub_paths = None  # Will be computed on demand
    
    def reset(self) -> None:
        """Reset all components to initial state."""
        if self._initial_state:
            for node_id, state in self._initial_state.items():
                self.graph.nodes[node_id]["state"] = state.value
                if node_id in self.components:
                    self.components[node_id].state = state
                    self.components[node_id].reset_metrics()
        self._pub_sub_paths = None
    
    # =========================================================================
    # Component State Management
    # =========================================================================
    
    def set_state(self, comp_id: str, state: ComponentState) -> None:
        """Set the state of a component."""
        if comp_id in self.graph:
            self.graph.nodes[comp_id]["state"] = state.value
        if comp_id in self.components:
            self.components[comp_id].state = state
    
    def get_state(self, comp_id: str) -> ComponentState:
        """Get the state of a component."""
        if comp_id in self.graph:
            state_str = self.graph.nodes[comp_id].get("state", "active")
            return ComponentState(state_str)
        return ComponentState.FAILED
    
    def is_active(self, comp_id: str) -> bool:
        """Check if a component is active."""
        return self.get_state(comp_id) == ComponentState.ACTIVE
    
    def fail_component(self, comp_id: str) -> None:
        """Mark a component as failed."""
        self.set_state(comp_id, ComponentState.FAILED)
    
    # =========================================================================
    # Graph Queries
    # =========================================================================
    
    def get_components_by_type(self, comp_type: str) -> List[str]:
        """Get all component IDs of a specific type."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == comp_type
        ]
    
    def get_active_components(self, comp_type: Optional[str] = None) -> List[str]:
        """Get all active components, optionally filtered by type."""
        result = []
        for n, d in self.graph.nodes(data=True):
            if d.get("state") != ComponentState.ACTIVE.value:
                continue
            if comp_type is None or d.get("type") == comp_type:
                result.append(n)
        return result
    
    def get_publishers(self, topic_id: str) -> List[str]:
        """Get all applications publishing to a topic."""
        publishers = []
        for src, tgt, data in self.graph.in_edges(topic_id, data=True):
            if data.get("relation") == RelationType.PUBLISHES_TO.value:
                publishers.append(src)
        return publishers
    
    def get_subscribers(self, topic_id: str) -> List[str]:
        """Get all applications subscribing to a topic."""
        subscribers = []
        for src, tgt, data in self.graph.in_edges(topic_id, data=True):
            if data.get("relation") == RelationType.SUBSCRIBES_TO.value:
                subscribers.append(src)
        return subscribers
    
    def get_routing_brokers(self, topic_id: str) -> List[str]:
        """Get all brokers routing a topic."""
        brokers = []
        for src, tgt, data in self.graph.in_edges(topic_id, data=True):
            if data.get("relation") == RelationType.ROUTES.value:
                brokers.append(src)
        return brokers
    
    def get_host_node(self, comp_id: str) -> Optional[str]:
        """Get the node that hosts a component."""
        for src, tgt, data in self.graph.out_edges(comp_id, data=True):
            if data.get("relation") == RelationType.RUNS_ON.value:
                return tgt
        return None
    
    def get_hosted_components(self, node_id: str) -> List[str]:
        """Get all components hosted on a node."""
        hosted = []
        for src, tgt, data in self.graph.in_edges(node_id, data=True):
            if data.get("relation") == RelationType.RUNS_ON.value:
                hosted.append(src)
        return hosted
    
    def get_app_topics(self, app_id: str) -> Tuple[List[str], List[str]]:
        """Get topics an application publishes to and subscribes from."""
        publishes = []
        subscribes = []
        for src, tgt, data in self.graph.out_edges(app_id, data=True):
            if data.get("relation") == RelationType.PUBLISHES_TO.value:
                publishes.append(tgt)
            elif data.get("relation") == RelationType.SUBSCRIBES_TO.value:
                subscribes.append(tgt)
        return publishes, subscribes
    
    # =========================================================================
    # Pub-Sub Path Analysis
    # =========================================================================
    
    def get_pub_sub_paths(self, active_only: bool = False) -> List[Tuple[str, str, str]]:
        """
        Get all publisher -> topic -> subscriber paths.
        
        Returns:
            List of (publisher_id, topic_id, subscriber_id) tuples
        """
        if self._pub_sub_paths is not None and not active_only:
            return self._pub_sub_paths
        
        paths = []
        for topic_id in self.topics:
            publishers = self.get_publishers(topic_id)
            subscribers = self.get_subscribers(topic_id)
            
            for pub in publishers:
                if active_only and not self.is_active(pub):
                    continue
                for sub in subscribers:
                    if active_only and not self.is_active(sub):
                        continue
                    if pub != sub:  # Exclude self-loops
                        paths.append((pub, topic_id, sub))
        
        if not active_only:
            self._pub_sub_paths = paths
        
        return paths
    
    def get_message_path(
        self, 
        publisher: str, 
        topic_id: str, 
        subscriber: str
    ) -> List[str]:
        """
        Get the full message path from publisher to subscriber.
        
        Path: Publisher -> Topic -> Broker(s) -> Subscriber
        """
        path = [publisher, topic_id]
        
        # Get routing brokers
        brokers = self.get_routing_brokers(topic_id)
        if brokers:
            path.extend(brokers)
        
        path.append(subscriber)
        return path
    
    def is_path_active(self, path: List[str]) -> bool:
        """Check if all components in a path are active."""
        for comp_id in path:
            if not self.is_active(comp_id):
                return False
        return True
    
    # =========================================================================
    # Infrastructure Analysis
    # =========================================================================
    
    def get_node_graph(self) -> nx.Graph:
        """Extract undirected node connectivity graph."""
        node_graph = nx.Graph()
        
        for node_id in self.get_components_by_type("Node"):
            node_graph.add_node(node_id)
        
        for src, tgt, data in self.graph.edges(data=True):
            if data.get("relation") == RelationType.CONNECTS_TO.value:
                node_graph.add_edge(src, tgt, weight=data.get("weight", 1.0))
        
        return node_graph
    
    def get_connected_components_count(self, active_only: bool = False) -> int:
        """Get number of connected components in node graph."""
        node_graph = self.get_node_graph()
        
        if active_only:
            inactive_nodes = [
                n for n in node_graph.nodes() 
                if not self.is_active(n)
            ]
            node_graph.remove_nodes_from(inactive_nodes)
        
        if len(node_graph) == 0:
            return 0
        
        return nx.number_connected_components(node_graph)
    
    # =========================================================================
    # Metrics
    # =========================================================================
    
    @property
    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        return self.graph.number_of_edges()
    
    @property
    def topic_count(self) -> int:
        """Total number of topics."""
        return len(self.topics)
    
    @property
    def initial_path_count(self) -> int:
        """Total number of pub-sub paths."""
        return len(self.get_pub_sub_paths())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the graph."""
        type_counts = {}
        for n, d in self.graph.nodes(data=True):
            t = d.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        
        rel_counts = {}
        for _, _, d in self.graph.edges(data=True):
            r = d.get("relation", "Unknown")
            rel_counts[r] = rel_counts.get(r, 0) + 1
        
        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "component_types": type_counts,
            "relationship_types": rel_counts,
            "pub_sub_paths": self.initial_path_count,
        }