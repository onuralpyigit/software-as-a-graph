"""
Simulation Graph Model - Version 4.0

In-memory graph representation for simulation, built directly from
GraphModel or JSON without requiring Neo4j or DEPENDS_ON relationships.

The simulation graph models:
- Applications, Brokers, Topics, Nodes as components
- PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, HOSTS as connections
- Component state (active, degraded, capacity)
- Message flow paths through the pub-sub topology

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import json
import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path


# =============================================================================
# Enums
# =============================================================================

class ComponentType(Enum):
    """Types of system components"""
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"


class ConnectionType(Enum):
    """Types of connections between components"""
    PUBLISHES_TO = "PUBLISHES_TO"      # App -> Topic
    SUBSCRIBES_TO = "SUBSCRIBES_TO"    # App -> Topic
    ROUTES = "ROUTES"                  # Broker -> Topic
    RUNS_ON = "RUNS_ON"                # App -> Node
    HOSTS = "HOSTS"                    # Node -> Broker


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Component:
    """A component in the simulation graph"""
    id: str
    type: ComponentType
    name: str = ""
    
    # State
    is_active: bool = True
    is_degraded: bool = False
    capacity: float = 1.0  # 0.0 to 1.0
    
    # Properties (from original graph)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = self.id

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "is_active": self.is_active,
            "is_degraded": self.is_degraded,
            "capacity": self.capacity,
            "properties": self.properties,
        }

    def copy(self) -> Component:
        """Create a deep copy"""
        return Component(
            id=self.id,
            type=self.type,
            name=self.name,
            is_active=self.is_active,
            is_degraded=self.is_degraded,
            capacity=self.capacity,
            properties=copy.deepcopy(self.properties),
        )


@dataclass
class Connection:
    """A connection between components"""
    source: str
    target: str
    type: ConnectionType
    
    # State
    is_active: bool = True
    
    # Properties
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "is_active": self.is_active,
            "weight": self.weight,
            "properties": self.properties,
        }

    def copy(self) -> Connection:
        """Create a deep copy"""
        return Connection(
            source=self.source,
            target=self.target,
            type=self.type,
            is_active=self.is_active,
            weight=self.weight,
            properties=copy.deepcopy(self.properties),
        )


# =============================================================================
# Simulation Graph
# =============================================================================

class SimulationGraph:
    """
    In-memory graph for simulation.
    
    Models the pub-sub topology with components and connections.
    Provides methods for:
    - Graph traversal and path finding
    - Component state management
    - Reachability analysis
    - Message flow path computation
    
    Does NOT require DEPENDS_ON relationships - works directly
    with PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, etc.
    """

    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.connections: List[Connection] = []
        
        # Indexes for fast lookup
        self._outgoing: Dict[str, List[Connection]] = defaultdict(list)
        self._incoming: Dict[str, List[Connection]] = defaultdict(list)
        self._by_type: Dict[ComponentType, List[str]] = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)

    # =========================================================================
    # Loading
    # =========================================================================

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationGraph:
        """
        Load from dictionary (supports multiple formats).
        
        Format 1 (generate_graph output):
        {
            "applications": [...],
            "brokers": [...],
            "topics": [...],
            "nodes": [...],
            "relationships": {
                "publishes_to": [...],
                "subscribes_to": [...],
                ...
            }
        }
        
        Format 2 (nested vertices/edges):
        {
            "vertices": {"applications": [...], ...},
            "edges": {"publishes_to": [...], ...}
        }
        """
        graph = cls()
        
        # Detect format
        if "vertices" in data:
            # Format 2: nested
            vertices = data.get("vertices", {})
            edges = data.get("edges", {})
        else:
            # Format 1: flat
            vertices = {
                "applications": data.get("applications", []),
                "brokers": data.get("brokers", []),
                "topics": data.get("topics", []),
                "nodes": data.get("nodes", []),
            }
            edges = data.get("relationships", {})
        
        # Load components
        type_mapping = {
            "applications": ComponentType.APPLICATION,
            "brokers": ComponentType.BROKER,
            "topics": ComponentType.TOPIC,
            "nodes": ComponentType.NODE,
        }
        
        for key, comp_type in type_mapping.items():
            for item in vertices.get(key, []):
                comp = Component(
                    id=item.get("id", ""),
                    type=comp_type,
                    name=item.get("name", item.get("id", "")),
                    properties={k: v for k, v in item.items() if k not in ("id", "name")},
                )
                graph.add_component(comp)
        
        # Load connections
        conn_mapping = {
            "publishes_to": ConnectionType.PUBLISHES_TO,
            "subscribes_to": ConnectionType.SUBSCRIBES_TO,
            "routes": ConnectionType.ROUTES,
            "runs_on": ConnectionType.RUNS_ON,
            "hosts": ConnectionType.HOSTS,
            "connects_to": ConnectionType.HOSTS,  # alias
        }
        
        for key, conn_type in conn_mapping.items():
            for item in edges.get(key, []):
                # Handle various naming conventions
                # Primary: from/to (used by generate_graph)
                source = item.get("from") or item.get("source") or item.get("from_id", "")
                target = item.get("to") or item.get("target") or item.get("to_id", "")
                
                if source and target:
                    conn = Connection(
                        source=source,
                        target=target,
                        type=conn_type,
                        weight=item.get("weight", 1.0),
                        properties={k: v for k, v in item.items() 
                                   if k not in ("from", "to", "source", "target", "from_id", "to_id")},
                    )
                    graph.add_connection(conn)
        
        graph.logger.info(f"Loaded graph: {len(graph.components)} components, {len(graph.connections)} connections")
        return graph

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationGraph:
        """Load from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    # =========================================================================
    # Building
    # =========================================================================

    def add_component(self, component: Component) -> None:
        """Add a component to the graph"""
        self.components[component.id] = component
        self._by_type[component.type].append(component.id)

    def add_connection(self, connection: Connection) -> None:
        """Add a connection to the graph"""
        self.connections.append(connection)
        self._outgoing[connection.source].append(connection)
        self._incoming[connection.target].append(connection)

    # =========================================================================
    # Querying
    # =========================================================================

    def get_component(self, component_id: str) -> Optional[Component]:
        """Get component by ID"""
        return self.components.get(component_id)

    def get_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Get all components of a type"""
        return [self.components[cid] for cid in self._by_type.get(comp_type, [])]

    def get_outgoing(self, component_id: str) -> List[Connection]:
        """Get outgoing connections from a component"""
        return self._outgoing.get(component_id, [])

    def get_incoming(self, component_id: str) -> List[Connection]:
        """Get incoming connections to a component"""
        return self._incoming.get(component_id, [])

    def get_active_outgoing(self, component_id: str) -> List[Connection]:
        """Get active outgoing connections"""
        return [c for c in self.get_outgoing(component_id) if c.is_active]

    def get_active_incoming(self, component_id: str) -> List[Connection]:
        """Get active incoming connections"""
        return [c for c in self.get_incoming(component_id) if c.is_active]

    def get_neighbors(self, component_id: str) -> Set[str]:
        """Get all neighbors (sources and targets)"""
        neighbors = set()
        for conn in self.get_outgoing(component_id):
            neighbors.add(conn.target)
        for conn in self.get_incoming(component_id):
            neighbors.add(conn.source)
        return neighbors

    def get_active_neighbors(self, component_id: str) -> Set[str]:
        """Get active neighbors"""
        neighbors = set()
        for conn in self.get_active_outgoing(component_id):
            if self.components.get(conn.target, Component("", ComponentType.APPLICATION)).is_active:
                neighbors.add(conn.target)
        for conn in self.get_active_incoming(component_id):
            if self.components.get(conn.source, Component("", ComponentType.APPLICATION)).is_active:
                neighbors.add(conn.source)
        return neighbors

    # =========================================================================
    # Pub-Sub Specific Queries
    # =========================================================================

    def get_publishers(self, topic_id: str) -> List[str]:
        """Get applications publishing to a topic"""
        return [
            conn.source for conn in self.get_incoming(topic_id)
            if conn.type == ConnectionType.PUBLISHES_TO
        ]

    def get_subscribers(self, topic_id: str) -> List[str]:
        """Get applications subscribed to a topic"""
        return [
            conn.source for conn in self.get_incoming(topic_id)
            if conn.type == ConnectionType.SUBSCRIBES_TO
        ]

    def get_published_topics(self, app_id: str) -> List[str]:
        """Get topics an application publishes to"""
        return [
            conn.target for conn in self.get_outgoing(app_id)
            if conn.type == ConnectionType.PUBLISHES_TO
        ]

    def get_subscribed_topics(self, app_id: str) -> List[str]:
        """Get topics an application subscribes to"""
        return [
            conn.target for conn in self.get_outgoing(app_id)
            if conn.type == ConnectionType.SUBSCRIBES_TO
        ]

    def get_broker_for_topic(self, topic_id: str) -> Optional[str]:
        """Get broker routing a topic"""
        for conn in self.get_incoming(topic_id):
            if conn.type == ConnectionType.ROUTES:
                return conn.source
        return None

    def get_node_for_app(self, app_id: str) -> Optional[str]:
        """Get node where an application runs"""
        for conn in self.get_outgoing(app_id):
            if conn.type == ConnectionType.RUNS_ON:
                return conn.target
        return None

    def get_apps_on_node(self, node_id: str) -> List[str]:
        """Get applications running on a node"""
        return [
            conn.source for conn in self.get_incoming(node_id)
            if conn.type == ConnectionType.RUNS_ON
        ]

    # =========================================================================
    # Message Flow Paths
    # =========================================================================

    def get_message_path(self, publisher_id: str, subscriber_id: str, topic_id: str) -> List[str]:
        """
        Get message flow path from publisher to subscriber through topic.
        
        Path: Publisher -> Topic -> [Broker] -> Subscriber
        """
        path = [publisher_id]
        
        # Check publisher publishes to topic
        if topic_id not in self.get_published_topics(publisher_id):
            return []
        
        path.append(topic_id)
        
        # Add broker if exists
        broker = self.get_broker_for_topic(topic_id)
        if broker:
            path.append(broker)
        
        # Check subscriber subscribes to topic
        if topic_id not in self.get_subscribed_topics(subscriber_id):
            return []
        
        path.append(subscriber_id)
        return path

    def get_all_message_paths(self) -> List[Tuple[str, str, str, List[str]]]:
        """
        Get all possible message paths in the system.
        
        Returns list of (publisher, subscriber, topic, path)
        """
        paths = []
        
        for topic_id in self._by_type.get(ComponentType.TOPIC, []):
            publishers = self.get_publishers(topic_id)
            subscribers = self.get_subscribers(topic_id)
            
            for pub in publishers:
                for sub in subscribers:
                    if pub != sub:
                        path = self.get_message_path(pub, sub, topic_id)
                        if path:
                            paths.append((pub, sub, topic_id, path))
        
        return paths

    def is_path_active(self, path: List[str]) -> bool:
        """Check if all components in a path are active"""
        for comp_id in path:
            comp = self.components.get(comp_id)
            if not comp or not comp.is_active:
                return False
        return True

    def count_active_paths(self) -> int:
        """Count number of currently active message paths"""
        count = 0
        for _, _, _, path in self.get_all_message_paths():
            if self.is_path_active(path):
                count += 1
        return count

    # =========================================================================
    # Reachability Analysis
    # =========================================================================

    def calculate_reachability(self, component_id: str) -> Set[str]:
        """Calculate all components reachable from a component"""
        reachable = set()
        visited = set()
        queue = [component_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            comp = self.components.get(current)
            if not comp or not comp.is_active:
                continue
            
            reachable.add(current)
            
            for neighbor in self.get_active_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return reachable

    def calculate_total_reachability(self) -> int:
        """Calculate total reachable pairs (sum of reachability from each active component)"""
        total = 0
        for comp_id, comp in self.components.items():
            if comp.is_active:
                total += len(self.calculate_reachability(comp_id))
        return total

    def count_connected_components(self) -> int:
        """Count number of connected components (considering only active nodes)"""
        visited = set()
        count = 0
        
        for comp_id, comp in self.components.items():
            if not comp.is_active or comp_id in visited:
                continue
            
            # BFS from this component
            queue = [comp_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor in self.get_active_neighbors(current):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            count += 1
        
        return count

    # =========================================================================
    # Copy and State
    # =========================================================================

    def copy(self) -> SimulationGraph:
        """Create a deep copy of the graph"""
        new_graph = SimulationGraph()
        
        for comp in self.components.values():
            new_graph.add_component(comp.copy())
        
        for conn in self.connections:
            new_graph.add_connection(conn.copy())
        
        return new_graph

    def reset(self) -> None:
        """Reset all components and connections to active state"""
        for comp in self.components.values():
            comp.is_active = True
            comp.is_degraded = False
            comp.capacity = 1.0
        
        for conn in self.connections:
            conn.is_active = True

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            "components": {
                "total": len(self.components),
                "active": sum(1 for c in self.components.values() if c.is_active),
                "by_type": {
                    t.value: len(ids) for t, ids in self._by_type.items()
                },
            },
            "connections": {
                "total": len(self.connections),
                "active": sum(1 for c in self.connections if c.is_active),
                "by_type": {},
            },
            "paths": {
                "total": len(self.get_all_message_paths()),
                "active": self.count_active_paths(),
            },
            "connectivity": {
                "connected_components": self.count_connected_components(),
            },
        }
        
        # Count connections by type
        conn_by_type = defaultdict(int)
        for conn in self.connections:
            conn_by_type[conn.type.value] += 1
        stats["connections"]["by_type"] = dict(conn_by_type)
        
        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            "components": [c.to_dict() for c in self.components.values()],
            "connections": [c.to_dict() for c in self.connections],
            "stats": self.get_stats(),
        }
