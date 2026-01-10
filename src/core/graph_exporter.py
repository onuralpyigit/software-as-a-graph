"""
Graph Exporter - Version 7.0

Neo4j client for graph data retrieval.
All graph algorithms are performed using NetworkX after retrieval.

This module handles:
- Connecting to Neo4j database
- Retrieving components (Application, Broker, Node, Topic)
- Retrieving DEPENDS_ON relationships
- Filtering by component type and dependency type

Author: Software-as-a-Graph Research Project
Version: 7.0
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


# Component type constants
COMPONENT_TYPES = ["Application", "Broker", "Node", "Topic"]

# Dependency type constants
DEPENDENCY_TYPES = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]

# Layer definitions for filtering
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
    "complete": {
        "name": "Complete System",
        "component_types": COMPONENT_TYPES,
        "dependency_types": DEPENDENCY_TYPES,
    },
}

# Structural relationship types (for reference)
STRUCTURAL_REL_TYPES = [
    "PUBLISHES_TO", 
    "SUBSCRIBES_TO", 
    "RUNS_ON", 
    "ROUTES", 
    "CONNECTS_TO"
]


@dataclass
class ComponentData:
    """Data for a single component retrieved from database."""
    
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
    """Data for a single DEPENDS_ON edge retrieved from database."""
    
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    dependency_type: str
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "dependency_type": self.dependency_type,
            "relation_type": self.relation_type,
            "weight": self.weight,
            **self.properties,
        }


@dataclass
class GraphData:
    """Complete graph data retrieved from database."""
    
    components: List[ComponentData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
        }
    
    def get_components_by_type(self, comp_type: str) -> List[ComponentData]:
        """Get all components of a specific type."""
        return [c for c in self.components if c.component_type == comp_type]
    
    def get_edges_by_type(self, dep_type: str) -> List[EdgeData]:
        """Get all edges of a specific dependency type."""
        return [e for e in self.edges if e.dependency_type == dep_type]


class GraphExporter:
    """
    Client for retrieving graph data from Neo4j database.
    
    All graph algorithms are performed using NetworkX after data retrieval.
    """
    
    def __init__(
        self,
        uri: str,
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j package not installed. "
                "Install with: pip install neo4j"
            )
        
        self.uri = uri
        self.user = user
        self.password = password
        self.logger = logging.getLogger(__name__)
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the Neo4j driver connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None
    ) -> GraphData:
        """
        Retrieve graph data from Neo4j.
        
        Args:
            component_types: List of component types to retrieve (None = all)
            dependency_types: List of dependency types to retrieve (None = all)
            
        Returns:
            GraphData containing components and edges
        """
        components = self._get_components(component_types)
        component_ids = {c.id for c in components}
        edges = self._get_edges(dependency_types, component_ids)
        
        return GraphData(components=components, edges=edges)
    
    def _get_components(
        self, 
        component_types: Optional[List[str]] = None
    ) -> List[ComponentData]:
        """Retrieve components from Neo4j."""
        
        types = component_types or COMPONENT_TYPES
        types_str = ", ".join(f"'{t}'" for t in types)
        
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN [{types_str}])
        RETURN 
            n.id as id,
            labels(n)[0] as type,
            coalesce(n.weight, 1.0) as weight,
            properties(n) as props
        """
        
        components = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                # Extract known properties from props
                props = dict(record["props"])
                # Remove id and weight from props (already extracted)
                props.pop("id", None)
                props.pop("weight", None)
                
                components.append(ComponentData(
                    id=record["id"],
                    component_type=record["type"],
                    weight=float(record["weight"]),
                    properties=props,
                ))
        
        self.logger.info(f"Retrieved {len(components)} components")
        return components
    
    def _get_edges(
        self,
        dependency_types: Optional[List[str]] = None,
        component_ids: Optional[Set[str]] = None
    ) -> List[EdgeData]:
        """Retrieve DEPENDS_ON edges from Neo4j."""
        
        types = dependency_types or DEPENDENCY_TYPES
        types_str = ", ".join(f"'{t}'" for t in types)
        
        query = f"""
        MATCH (source)-[r:DEPENDS_ON]->(target)
        WHERE r.dependency_type IN [{types_str}]
        RETURN 
            source.id as source_id,
            target.id as target_id,
            labels(source)[0] as source_type,
            labels(target)[0] as target_type,
            r.dependency_type as dependency_type,
            coalesce(r.weight, 1.0) as weight,
            properties(r) as props
        """
        
        edges = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                # Skip if endpoints not in component set
                if component_ids:
                    if record["source_id"] not in component_ids:
                        continue
                    if record["target_id"] not in component_ids:
                        continue
                
                props = dict(record["props"])
                props.pop("dependency_type", None)
                props.pop("weight", None)
                
                edges.append(EdgeData(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    source_type=record["source_type"],
                    target_type=record["target_type"],
                    dependency_type=record["dependency_type"],
                    relation_type="DEPENDS_ON",
                    weight=float(record["weight"]),
                    properties=props,
                ))
        
        self.logger.info(f"Retrieved {len(edges)} edges")
        return edges
    
    def get_layer_data(self, layer: str) -> GraphData:
        """
        Retrieve graph data for a specific layer.
        
        Args:
            layer: Layer name (application, infrastructure, complete, etc.)
            
        Returns:
            GraphData filtered for the specified layer
        """
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}. Valid: {list(LAYER_DEFINITIONS.keys())}")
        
        layer_def = LAYER_DEFINITIONS[layer]
        return self.get_graph_data(
            component_types=layer_def["component_types"],
            dependency_types=layer_def["dependency_types"],
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the graph in the database."""
        
        stats = {}
        
        with self.driver.session() as session:
            # Component counts by type
            for comp_type in COMPONENT_TYPES:
                result = session.run(f"MATCH (n:{comp_type}) RETURN count(n) as c")
                stats[f"{comp_type.lower()}_count"] = result.single()["c"]
            
            # Edge counts by type
            for dep_type in DEPENDENCY_TYPES:
                result = session.run(
                    f"MATCH ()-[r:DEPENDS_ON {{dependency_type: '{dep_type}'}}]->() "
                    f"RETURN count(r) as c"
                )
                stats[f"{dep_type}_count"] = result.single()["c"]
        
        return stats