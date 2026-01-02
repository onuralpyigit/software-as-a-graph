"""
Neo4j Simulation Client - Version 5.0

Loads simulation graph data directly from Neo4j database.

Features:
- Load full graph with all components and edges
- Load by component type (Application, Broker, Topic, Node)
- Load by edge type (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
- Load layer-specific subgraphs
- Database statistics and verification

Usage:
    from src.simulation import Neo4jSimulationClient, load_graph_from_neo4j
    
    # Using context manager
    with Neo4jSimulationClient(uri, user, password) as client:
        graph = client.load_full_graph()
        stats = client.get_statistics()
    
    # Using factory function
    graph = load_graph_from_neo4j(uri, user, password)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set
from contextlib import contextmanager

from .simulation_graph import (
    SimulationGraph,
    Component,
    Edge,
    ComponentType,
    EdgeType,
    ComponentStatus,
    QoSPolicy,
)

# Check for Neo4j driver
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None
    ServiceUnavailable = Exception
    ClientError = Exception


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> Neo4jConfig:
        return cls(
            uri=data.get("uri", "bolt://localhost:7687"),
            user=data.get("user", "neo4j"),
            password=data.get("password", "password"),
            database=data.get("database", "neo4j"),
        )
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "uri": self.uri,
            "user": self.user,
            "database": self.database,
        }


# =============================================================================
# Neo4j Simulation Client
# =============================================================================

class Neo4jSimulationClient:
    """
    Client for loading simulation graphs from Neo4j.
    
    Loads original edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
    for accurate message flow simulation.
    
    Example:
        with Neo4jSimulationClient(uri, user, password) as client:
            # Load full graph
            graph = client.load_full_graph()
            
            # Load specific layer
            app_graph = client.load_layer("application")
            
            # Get statistics
            stats = client.get_statistics()
    """
    
    # Mapping from ComponentType to Neo4j label
    COMPONENT_LABELS = {
        ComponentType.APPLICATION: "Application",
        ComponentType.BROKER: "Broker",
        ComponentType.TOPIC: "Topic",
        ComponentType.NODE: "Node",
    }
    
    # Mapping from EdgeType to Neo4j relationship type
    EDGE_TYPES = {
        EdgeType.PUBLISHES_TO: "PUBLISHES_TO",
        EdgeType.SUBSCRIBES_TO: "SUBSCRIBES_TO",
        EdgeType.ROUTES: "ROUTES",
        EdgeType.RUNS_ON: "RUNS_ON",
        EdgeType.CONNECTS_TO: "CONNECTS_TO",
    }
    
    # Layer definitions
    LAYER_EDGE_TYPES = {
        "application": [EdgeType.PUBLISHES_TO, EdgeType.SUBSCRIBES_TO],
        "infrastructure": [EdgeType.CONNECTS_TO],
        "app_broker": [EdgeType.PUBLISHES_TO, EdgeType.SUBSCRIBES_TO, EdgeType.ROUTES],
        "node_broker": [EdgeType.CONNECTS_TO, EdgeType.RUNS_ON],
    }
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j simulation client.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
        
        Raises:
            ImportError: If neo4j driver not installed
        """
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )
        
        self.config = Neo4jConfig(uri, user, password, database)
        self._driver = None
        self._logger = logging.getLogger(__name__)
    
    def __enter__(self) -> Neo4jSimulationClient:
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
        )
        self._driver.verify_connectivity()
        self._logger.info(f"Connected to Neo4j at {self.config.uri}")
    
    def close(self) -> None:
        """Close connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._logger.info("Disconnected from Neo4j")
    
    @contextmanager
    def session(self):
        """Get a database session."""
        if not self._driver:
            self.connect()
        
        session = self._driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def verify_connection(self) -> bool:
        """Verify database connection."""
        try:
            with self.session() as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except Exception as e:
            self._logger.error(f"Connection verification failed: {e}")
            return False
    
    # =========================================================================
    # Graph Loading
    # =========================================================================
    
    def load_full_graph(self) -> SimulationGraph:
        """
        Load complete simulation graph from Neo4j.
        
        Returns:
            SimulationGraph with all components and edges
        """
        graph = SimulationGraph()
        
        # Load all components
        self._load_components(graph)
        
        # Load all edges
        self._load_edges(graph)
        
        self._logger.info(
            f"Loaded graph: {len(graph.components)} components, "
            f"{len(graph.edges)} edges"
        )
        
        return graph
    
    def load_layer(self, layer: str) -> SimulationGraph:
        """
        Load a specific layer subgraph.
        
        Args:
            layer: Layer name (application, infrastructure, app_broker, node_broker)
        
        Returns:
            SimulationGraph for the specified layer
        """
        if layer not in self.LAYER_EDGE_TYPES:
            raise ValueError(f"Unknown layer: {layer}. "
                           f"Valid: {list(self.LAYER_EDGE_TYPES.keys())}")
        
        edge_types = self.LAYER_EDGE_TYPES[layer]
        return self.load_by_edge_types(edge_types)
    
    def load_by_component_type(
        self,
        component_type: ComponentType,
    ) -> SimulationGraph:
        """
        Load graph containing only specified component type.
        
        Args:
            component_type: Type of components to load
        
        Returns:
            SimulationGraph with filtered components
        """
        graph = SimulationGraph()
        
        # Load only specified component type
        self._load_components(graph, component_types=[component_type])
        
        # Load edges where both endpoints exist
        self._load_edges(graph, filter_missing=True)
        
        return graph
    
    def load_by_edge_types(
        self,
        edge_types: List[EdgeType],
    ) -> SimulationGraph:
        """
        Load graph with only specified edge types.
        
        Args:
            edge_types: List of edge types to include
        
        Returns:
            SimulationGraph with filtered edges
        """
        graph = SimulationGraph()
        
        # Load all components
        self._load_components(graph)
        
        # Load only specified edge types
        self._load_edges(graph, edge_types=edge_types)
        
        return graph
    
    def _load_components(
        self,
        graph: SimulationGraph,
        component_types: Optional[List[ComponentType]] = None,
    ) -> None:
        """Load components from Neo4j."""
        types_to_load = component_types or list(ComponentType)
        
        with self.session() as session:
            for comp_type in types_to_load:
                label = self.COMPONENT_LABELS[comp_type]
                
                result = session.run(f"""
                    MATCH (n:{label})
                    RETURN n.id AS id, n.name AS name, 
                           n.weight AS weight, n.status AS status,
                           properties(n) AS props
                """)
                
                for record in result:
                    comp_id = record["id"]
                    props = record["props"] or {}
                    
                    # Extract QoS properties for topics
                    properties = {
                        k: v for k, v in props.items()
                        if k not in ["id", "name", "weight", "status"]
                    }
                    
                    component = Component(
                        id=comp_id,
                        type=comp_type,
                        name=record["name"] or comp_id,
                        status=ComponentStatus(record["status"] or "healthy"),
                        properties=properties,
                    )
                    
                    graph.add_component(component)
    
    def _load_edges(
        self,
        graph: SimulationGraph,
        edge_types: Optional[List[EdgeType]] = None,
        filter_missing: bool = False,
    ) -> None:
        """Load edges from Neo4j."""
        types_to_load = edge_types or list(EdgeType)
        
        with self.session() as session:
            for edge_type in types_to_load:
                rel_type = self.EDGE_TYPES[edge_type]
                
                result = session.run(f"""
                    MATCH (s)-[r:{rel_type}]->(t)
                    RETURN s.id AS source, t.id AS target,
                           r.weight AS weight,
                           r.qos_reliability AS reliability,
                           r.qos_durability AS durability,
                           r.qos_priority AS priority,
                           properties(r) AS props
                """)
                
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    
                    # Skip if filtering and component doesn't exist
                    if filter_missing:
                        if source not in graph.components or target not in graph.components:
                            continue
                    
                    # Build QoS policy
                    qos = QoSPolicy(
                        reliability=record["reliability"] or "BEST_EFFORT",
                        durability=record["durability"] or "VOLATILE",
                        priority=record["priority"] or "MEDIUM",
                    )
                    
                    # Extract additional properties
                    props = record["props"] or {}
                    properties = {
                        k: v for k, v in props.items()
                        if k not in ["weight", "qos_reliability", "qos_durability", "qos_priority"]
                    }
                    
                    edge = Edge(
                        source=source,
                        target=target,
                        edge_type=edge_type,
                        qos=qos,
                        weight=record["weight"] or 1.0,
                        properties=properties,
                    )
                    
                    graph.add_edge(edge)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics from Neo4j.
        
        Returns:
            Dictionary with component and edge counts
        """
        stats = {
            "components": {},
            "edges": {},
            "total_components": 0,
            "total_edges": 0,
        }
        
        with self.session() as session:
            # Count components by type
            for comp_type in ComponentType:
                label = self.COMPONENT_LABELS[comp_type]
                result = session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS count"
                )
                count = result.single()["count"]
                if count > 0:
                    stats["components"][comp_type.value] = count
                    stats["total_components"] += count
            
            # Count edges by type
            for edge_type in EdgeType:
                rel_type = self.EDGE_TYPES[edge_type]
                result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
                )
                count = result.single()["count"]
                if count > 0:
                    stats["edges"][edge_type.value] = count
                    stats["total_edges"] += count
        
        return stats
    
    def get_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each layer.
        
        Returns:
            Dictionary with per-layer statistics
        """
        layer_stats = {}
        
        for layer, edge_types in self.LAYER_EDGE_TYPES.items():
            layer_graph = self.load_layer(layer)
            summary = layer_graph.summary()
            
            layer_stats[layer] = {
                "components": summary["total_components"],
                "edges": summary["total_edges"],
                "message_paths": len(layer_graph.get_message_paths()),
            }
        
        return layer_stats


# =============================================================================
# Factory Functions
# =============================================================================

def load_graph_from_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    layer: Optional[str] = None,
) -> SimulationGraph:
    """
    Factory function to load simulation graph from Neo4j.
    
    Args:
        uri: Neo4j bolt URI
        user: Username
        password: Password
        database: Database name
        layer: Optional layer to load (application, infrastructure, etc.)
    
    Returns:
        SimulationGraph loaded from Neo4j
    
    Example:
        # Load full graph
        graph = load_graph_from_neo4j(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mypassword"
        )
        
        # Load specific layer
        app_graph = load_graph_from_neo4j(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mypassword",
            layer="application"
        )
    """
    with Neo4jSimulationClient(uri, user, password, database) as client:
        if layer:
            return client.load_layer(layer)
        return client.load_full_graph()


def check_neo4j_available() -> bool:
    """Check if Neo4j driver is available."""
    return HAS_NEO4J
