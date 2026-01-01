"""
Neo4j Client for Simulation - Version 5.0

Retrieves graph data directly from Neo4j for simulation.
Supports original edge types (NOT derived DEPENDS_ON).

Features:
- Load full graph or by component type
- Filter by edge types
- Support for multiple scenarios
- Connection pooling

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None

from .simulation_graph import (
    SimulationGraph,
    Component,
    Edge,
    ComponentType,
    EdgeType,
    ComponentStatus,
    QoSPolicy,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Neo4jConfig:
    """Neo4j connection configuration"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    
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
    
    Retrieves original edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
    NOT derived DEPENDS_ON relationships.
    """
    
    # Node labels in Neo4j
    NODE_LABELS = {
        ComponentType.APPLICATION: "Application",
        ComponentType.BROKER: "Broker",
        ComponentType.TOPIC: "Topic",
        ComponentType.NODE: "Node",
    }
    
    # Edge types in Neo4j
    EDGE_TYPES = {
        EdgeType.PUBLISHES_TO: "PUBLISHES_TO",
        EdgeType.SUBSCRIBES_TO: "SUBSCRIBES_TO",
        EdgeType.ROUTES: "ROUTES",
        EdgeType.RUNS_ON: "RUNS_ON",
        EdgeType.CONNECTS_TO: "CONNECTS_TO",
    }
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j package is required. Install with: pip install neo4j"
            )
        
        self.config = Neo4jConfig(uri, user, password, database)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._logger = logging.getLogger(__name__)
    
    def __enter__(self) -> 'Neo4jSimulationClient':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def close(self) -> None:
        """Close the database connection"""
        if self._driver:
            self._driver.close()
    
    # =========================================================================
    # Graph Loading
    # =========================================================================
    
    def load_full_graph(self) -> SimulationGraph:
        """
        Load the complete graph from Neo4j.
        Includes all component types and original edge types.
        """
        graph = SimulationGraph()
        
        # Load all components
        for comp_type in ComponentType:
            components = self._load_components_by_type(comp_type)
            for comp in components:
                graph.add_component(comp)
        
        # Load all original edges (NOT DEPENDS_ON)
        for edge_type in EdgeType:
            edges = self._load_edges_by_type(edge_type)
            for edge in edges:
                graph.add_edge(edge)
        
        self._logger.info(
            f"Loaded graph: {len(graph.components)} components, "
            f"{len(graph.edges)} edges"
        )
        
        return graph
    
    def load_graph_by_component_type(
        self,
        comp_type: ComponentType,
        include_related: bool = True,
    ) -> SimulationGraph:
        """
        Load graph containing only specific component type.
        
        Args:
            comp_type: Component type to load
            include_related: If True, include edges connecting to this type
        """
        graph = SimulationGraph()
        
        # Load components of this type
        components = self._load_components_by_type(comp_type)
        component_ids = set()
        
        for comp in components:
            graph.add_component(comp)
            component_ids.add(comp.id)
        
        if include_related:
            # Load edges involving these components
            for edge_type in EdgeType:
                edges = self._load_edges_by_type(edge_type)
                for edge in edges:
                    if edge.source in component_ids or edge.target in component_ids:
                        # Add the other component if needed
                        for comp_id in [edge.source, edge.target]:
                            if comp_id not in graph.components:
                                comp = self._load_component_by_id(comp_id)
                                if comp:
                                    graph.add_component(comp)
                        graph.add_edge(edge)
        
        return graph
    
    def load_graph_by_edge_types(
        self,
        edge_types: List[EdgeType],
    ) -> SimulationGraph:
        """
        Load graph with only specific edge types.
        
        Args:
            edge_types: List of edge types to include
        """
        graph = SimulationGraph()
        involved_ids = set()
        
        # Load edges of specified types
        for edge_type in edge_types:
            edges = self._load_edges_by_type(edge_type)
            for edge in edges:
                involved_ids.add(edge.source)
                involved_ids.add(edge.target)
                graph.add_edge(edge)
        
        # Load involved components
        for comp_id in involved_ids:
            comp = self._load_component_by_id(comp_id)
            if comp:
                graph.add_component(comp)
        
        return graph
    
    def load_messaging_graph(self) -> SimulationGraph:
        """
        Load only the messaging layer (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES).
        Useful for simulating message flow without infrastructure.
        """
        return self.load_graph_by_edge_types([
            EdgeType.PUBLISHES_TO,
            EdgeType.SUBSCRIBES_TO,
            EdgeType.ROUTES,
        ])
    
    def load_infrastructure_graph(self) -> SimulationGraph:
        """
        Load only the infrastructure layer (RUNS_ON, CONNECTS_TO).
        Useful for simulating node failures.
        """
        return self.load_graph_by_edge_types([
            EdgeType.RUNS_ON,
            EdgeType.CONNECTS_TO,
        ])
    
    # =========================================================================
    # Component Loading
    # =========================================================================
    
    def _load_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Load all components of a specific type"""
        label = self.NODE_LABELS[comp_type]
        
        query = f"""
        MATCH (n:{label})
        RETURN n.id AS id, n.name AS name, labels(n) AS labels, properties(n) AS props
        """
        
        components = []
        
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query)
            for record in result:
                props = record["props"] or {}
                comp = Component(
                    id=record["id"] or props.get("name", "unknown"),
                    type=comp_type,
                    name=record["name"] or record["id"] or "",
                    status=ComponentStatus.HEALTHY,
                    properties={k: v for k, v in props.items() 
                               if k not in ("id", "name")},
                )
                components.append(comp)
        
        return components
    
    def _load_component_by_id(self, comp_id: str) -> Optional[Component]:
        """Load a single component by ID"""
        query = """
        MATCH (n)
        WHERE n.id = $id OR n.name = $id
        RETURN n.id AS id, n.name AS name, labels(n) AS labels, properties(n) AS props
        LIMIT 1
        """
        
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query, id=comp_id)
            record = result.single()
            
            if not record:
                return None
            
            # Determine type from labels
            labels = record["labels"] or []
            comp_type = ComponentType.APPLICATION
            for label in labels:
                for ct, neo_label in self.NODE_LABELS.items():
                    if label == neo_label:
                        comp_type = ct
                        break
            
            props = record["props"] or {}
            return Component(
                id=record["id"] or comp_id,
                type=comp_type,
                name=record["name"] or comp_id,
                status=ComponentStatus.HEALTHY,
                properties={k: v for k, v in props.items() 
                           if k not in ("id", "name")},
            )
    
    # =========================================================================
    # Edge Loading
    # =========================================================================
    
    def _load_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Load all edges of a specific type"""
        rel_type = self.EDGE_TYPES[edge_type]
        
        query = f"""
        MATCH (a)-[r:{rel_type}]->(b)
        RETURN 
            COALESCE(a.id, a.name) AS source,
            COALESCE(b.id, b.name) AS target,
            type(r) AS rel_type,
            properties(r) AS props
        """
        
        edges = []
        
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query)
            for record in result:
                props = record["props"] or {}
                
                # Parse QoS if present
                qos = QoSPolicy(
                    reliability=props.get("reliability", "best_effort"),
                    durability=props.get("durability", "volatile"),
                    priority=props.get("priority", 0),
                    bandwidth=props.get("bandwidth", 1.0),
                    latency_ms=props.get("latency_ms", 10.0),
                )
                
                edge = Edge(
                    source=record["source"],
                    target=record["target"],
                    edge_type=edge_type,
                    weight=props.get("weight", 1.0),
                    qos=qos,
                    properties={k: v for k, v in props.items() 
                               if k not in ("weight", "reliability", "durability", 
                                           "priority", "bandwidth", "latency_ms")},
                )
                edges.append(edge)
        
        return edges
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics from Neo4j"""
        stats = {
            "components": {},
            "edges": {},
            "total_components": 0,
            "total_edges": 0,
        }
        
        with self._driver.session(database=self.config.database) as session:
            # Count components by type
            for comp_type in ComponentType:
                label = self.NODE_LABELS[comp_type]
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
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
    
    def verify_connection(self) -> bool:
        """Verify database connection is working"""
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except Exception as e:
            self._logger.error(f"Connection verification failed: {e}")
            return False


# =============================================================================
# Factory Function
# =============================================================================

def load_graph_from_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    component_type: Optional[ComponentType] = None,
    edge_types: Optional[List[EdgeType]] = None,
) -> SimulationGraph:
    """
    Factory function to load simulation graph from Neo4j.
    
    Args:
        uri: Neo4j bolt URI
        user: Username
        password: Password
        database: Database name
        component_type: If specified, load only this component type
        edge_types: If specified, load only these edge types
    
    Returns:
        SimulationGraph loaded from Neo4j
    """
    with Neo4jSimulationClient(uri, user, password, database) as client:
        if component_type is not None:
            return client.load_graph_by_component_type(component_type)
        elif edge_types is not None:
            return client.load_graph_by_edge_types(edge_types)
        else:
            return client.load_full_graph()
