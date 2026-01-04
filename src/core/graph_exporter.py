"""
Neo4j Client - Version 6.0

Neo4j client for graph data retrieval only.
All graph algorithms are performed using NetworkX.

This module handles:
- Connecting to Neo4j database
- Retrieving components (Application, Broker, Node, Topic)
- Retrieving DEPENDS_ON relationships
- Filtering by component type and dependency type

Author: Software-as-a-Graph Research Project
Version: 6.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


COMPONENT_TYPES = ["Application", "Broker", "Node", "Topic"]

DEPENDENCY_TYPES = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]

LAYER_DEFINITIONS = {
    "application": {
        "name": "Application Layer",
        "component_types": ["Application"],
        "dependency_types": ["app_to_app"],
    },
    "infrastructure": {
        "name": "Infrastructure Layer",
        "component_types": ["Node"],
        "dependency_types": ["node_to_node"],
    },
    "app_broker": {
        "name": "Application-Broker Layer",
        "component_types": ["Application", "Broker"],
        "dependency_types": ["app_to_broker"],
    },
    "node_broker": {
        "name": "Node-Broker Layer",
        "component_types": ["Node", "Broker"],
        "dependency_types": ["node_to_broker"],
    },
}


@dataclass
class ComponentData:
    """Data for a single component retrieved from Neo4j."""
    
    id: str
    component_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.component_type,
            "weight": self.weight,
            **self.properties,
        }


@dataclass
class EdgeData:
    """Data for a single DEPENDS_ON edge retrieved from Neo4j."""
    
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    dependency_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "dependency_type": self.dependency_type,
            "weight": self.weight,
            **self.properties,
        }


@dataclass
class GraphData:
    """Complete graph data retrieved from Neo4j."""
    
    components: List[ComponentData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }
    
    @property
    def component_count(self) -> int:
        return len(self.components)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    def get_components_by_type(self, comp_type: str) -> List[ComponentData]:
        """Get all components of a specific type."""
        return [c for c in self.components if c.component_type == comp_type]
    
    def get_edges_by_type(self, dep_type: str) -> List[EdgeData]:
        """Get all edges of a specific dependency type."""
        return [e for e in self.edges if e.dependency_type == dep_type]
    
    def get_component_ids(self) -> Set[str]:
        """Get set of all component IDs."""
        return {c.id for c in self.components}
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        by_type = {}
        for comp in self.components:
            by_type[comp.component_type] = by_type.get(comp.component_type, 0) + 1
        
        by_dep = {}
        for edge in self.edges:
            by_dep[edge.dependency_type] = by_dep.get(edge.dependency_type, 0) + 1
        
        return {
            "total_components": len(self.components),
            "total_edges": len(self.edges),
            "components_by_type": by_type,
            "edges_by_dependency_type": by_dep,
        }


class GraphExporter:
    """
    Neo4j client for graph data retrieval.
    
    This client retrieves graph data from Neo4j but does NOT perform
    any graph algorithms. All algorithms are handled by NetworkX.
    
    Example:
        with GraphExporter(uri, user, password) as client:
            # Get all data
            graph = client.get_full_graph()
            
            # Get specific component type
            apps = client.get_components_by_type("Application")
            
            # Get specific layer
            app_layer = client.get_layer("application")
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
        """
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j package not installed. "
                "Install with: pip install neo4j"
            )
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connection
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            raise ConnectionError(f"Cannot connect to Neo4j at {self.uri}: {e}")
        except AuthError as e:
            raise ConnectionError(f"Authentication failed for Neo4j: {e}")
    
    def __enter__(self) -> GraphExporter:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def get_graph_data(self) -> GraphData:
        """
        Retrieve complete graph data from Neo4j.
        
        Returns:
            GraphData with all components and DEPENDS_ON edges
        """
        components = self._get_all_components()
        edges = self._get_all_edges()
        
        return GraphData(
            components=components,
            edges=edges,
            metadata={"source": "neo4j", "uri": self.uri},
        )
    
    def get_components_by_type(self, component_type: str) -> List[ComponentData]:
        """
        Get all components of a specific type.
        
        Args:
            component_type: Type to retrieve (Application, Broker, Node, Topic)
        
        Returns:
            List of ComponentData
        """
        if component_type not in COMPONENT_TYPES:
            raise ValueError(f"Invalid component type: {component_type}")
        
        query = f"""
        MATCH (n:{component_type})
        RETURN n.id AS id, 
               labels(n)[0] AS type,
               COALESCE(n.weight, 1.0) AS weight,
               properties(n) AS props
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            components = []
            for record in result:
                props = dict(record["props"])
                # Remove duplicate keys
                props.pop("id", None)
                props.pop("weight", None)
                
                components.append(ComponentData(
                    id=record["id"],
                    component_type=record["type"],
                    weight=float(record["weight"]),
                    properties=props,
                ))
            return components
    
    def get_edges_by_type(self, dependency_type: str) -> List[EdgeData]:
        """
        Get all edges of a specific dependency type.
        
        Args:
            dependency_type: Type to retrieve (app_to_app, node_to_node, etc.)
        
        Returns:
            List of EdgeData
        """
        if dependency_type not in DEPENDENCY_TYPES:
            raise ValueError(f"Invalid dependency type: {dependency_type}")
        
        query = """
        MATCH (s)-[r:DEPENDS_ON {dependency_type: $dep_type}]->(t)
        RETURN s.id AS source_id,
               t.id AS target_id,
               labels(s)[0] AS source_type,
               labels(t)[0] AS target_type,
               r.dependency_type AS dependency_type,
               COALESCE(r.weight, 1.0) AS weight,
               properties(r) AS props
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, dep_type=dependency_type)
            edges = []
            for record in result:
                props = dict(record["props"])
                props.pop("dependency_type", None)
                props.pop("weight", None)
                
                edges.append(EdgeData(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    source_type=record["source_type"],
                    target_type=record["target_type"],
                    dependency_type=record["dependency_type"],
                    weight=float(record["weight"]),
                    properties=props,
                ))
            return edges
    
    def get_layer(self, layer_key: str) -> GraphData:
        """
        Get graph data for a specific layer.
        
        Args:
            layer_key: Layer to retrieve (application, infrastructure, etc.)
        
        Returns:
            GraphData with components and edges for that layer
        """
        if layer_key not in LAYER_DEFINITIONS:
            raise ValueError(
                f"Invalid layer: {layer_key}. "
                f"Valid: {list(LAYER_DEFINITIONS.keys())}"
            )
        
        layer_def = LAYER_DEFINITIONS[layer_key]
        
        # Get edges for this layer
        edges = []
        for dep_type in layer_def["dependency_types"]:
            edges.extend(self.get_edges_by_type(dep_type))
        
        # Get unique component IDs from edges
        component_ids = set()
        for edge in edges:
            component_ids.add(edge.source_id)
            component_ids.add(edge.target_id)
        
        # Get components involved in these edges
        components = []
        for comp_type in layer_def["component_types"]:
            type_components = self.get_components_by_type(comp_type)
            components.extend([c for c in type_components if c.id in component_ids])
        
        return GraphData(
            components=components,
            edges=edges,
            metadata={
                "layer": layer_key,
                "layer_name": layer_def["name"],
            },
        )
    
    def get_subgraph_by_component_type(self, component_type: str) -> GraphData:
        """
        Get subgraph containing only components of a specific type.
        
        This returns components of the specified type along with all
        DEPENDS_ON edges where BOTH endpoints are of that type.
        
        Args:
            component_type: Type to filter by
        
        Returns:
            GraphData with filtered components and edges
        """
        components = self.get_components_by_type(component_type)
        component_ids = {c.id for c in components}
        
        # Get all edges and filter to those within this type
        all_edges = self._get_all_edges()
        edges = [
            e for e in all_edges
            if e.source_id in component_ids and e.target_id in component_ids
        ]
        
        return GraphData(
            components=components,
            edges=edges,
            metadata={"component_type": component_type},
        )
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the graph."""
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node OR n:Topic
        WITH labels(n)[0] AS type, count(n) AS count
        RETURN collect({type: type, count: count}) AS nodes
        """
        
        edge_query = """
        MATCH ()-[r:DEPENDS_ON]->()
        WITH r.dependency_type AS type, count(r) AS count
        RETURN collect({type: type, count: count}) AS edges
        """
        
        with self.driver.session(database=self.database) as session:
            # Node counts
            result = session.run(query)
            record = result.single()
            node_counts = {item["type"]: item["count"] for item in record["nodes"]}
            
            # Edge counts
            result = session.run(edge_query)
            record = result.single()
            edge_counts = {item["type"]: item["count"] for item in record["edges"]}
        
        return {
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
        }
    
    def _get_all_components(self) -> List[ComponentData]:
        """Get all components from the database."""
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node OR n:Topic
        RETURN n.id AS id,
               labels(n)[0] AS type,
               COALESCE(n.weight, 1.0) AS weight,
               properties(n) AS props
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            components = []
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("weight", None)
                
                components.append(ComponentData(
                    id=record["id"],
                    component_type=record["type"],
                    weight=float(record["weight"]),
                    properties=props,
                ))
            return components
    
    def _get_all_edges(self) -> List[EdgeData]:
        """Get all DEPENDS_ON edges from the database."""
        query = """
        MATCH (s)-[r:DEPENDS_ON]->(t)
        RETURN s.id AS source_id,
               t.id AS target_id,
               labels(s)[0] AS source_type,
               labels(t)[0] AS target_type,
               COALESCE(r.dependency_type, 'unknown') AS dependency_type,
               COALESCE(r.weight, 1.0) AS weight,
               properties(r) AS props
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            edges = []
            for record in result:
                props = dict(record["props"])
                props.pop("dependency_type", None)
                props.pop("weight", None)
                
                edges.append(EdgeData(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    source_type=record["source_type"],
                    target_type=record["target_type"],
                    dependency_type=record["dependency_type"],
                    weight=float(record["weight"]),
                    properties=props,
                ))
            return edges
    
    def verify_connection(self) -> bool:
        """Verify the database connection is active."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False
