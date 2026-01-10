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
    
    def get_raw_structural_graph(self) -> Dict[str, Any]:
        """
        Retrieve raw structural graph data for simulation.
        
        Retrieves all components and raw structural relationships:
        - PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
        
        Returns:
            Dict in the same format as JSON input for SimulationGraph
        """
        data = {
            "nodes": [],
            "brokers": [],
            "applications": [],
            "topics": [],
            "publications": [],
            "subscriptions": [],
            "routes": [],
            "runs_on": [],
            "connections": [],
        }
        
        with self.driver.session() as session:
            # 1. Get Nodes (infrastructure)
            result = session.run("""
                MATCH (n:Node)
                RETURN n.id as id, n.name as name, coalesce(n.weight, 1.0) as weight
            """)
            for record in result:
                data["nodes"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "weight": float(record["weight"]),
                })
            
            # 2. Get Brokers
            result = session.run("""
                MATCH (b:Broker)
                RETURN b.id as id, b.name as name, coalesce(b.weight, 1.0) as weight
            """)
            for record in result:
                data["brokers"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "weight": float(record["weight"]),
                })
            
            # 3. Get Applications
            result = session.run("""
                MATCH (a:Application)
                RETURN a.id as id, a.name as name, a.role as role, 
                       coalesce(a.weight, 1.0) as weight
            """)
            for record in result:
                data["applications"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "role": record["role"] or "pubsub",
                    "weight": float(record["weight"]),
                })
            
            # 4. Get Topics with QoS
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.id as id, t.name as name, 
                       coalesce(t.size, 0) as size,
                       coalesce(t.reliability, 'BEST_EFFORT') as reliability,
                       coalesce(t.durability, 'VOLATILE') as durability,
                       coalesce(t.transport_priority, 'LOW') as priority
            """)
            for record in result:
                data["topics"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "size": int(record["size"]),
                    "qos": {
                        "reliability": record["reliability"],
                        "durability": record["durability"],
                        "transport_priority": record["priority"],
                    }
                })
            
            # 5. Get PUBLISHES_TO relationships
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO]->(t:Topic)
                RETURN a.id as app_id, t.id as topic_id
            """)
            for record in result:
                data["publications"].append({
                    "application": record["app_id"],
                    "topic": record["topic_id"],
                })
            
            # 6. Get SUBSCRIBES_TO relationships
            result = session.run("""
                MATCH (a:Application)-[r:SUBSCRIBES_TO]->(t:Topic)
                RETURN a.id as app_id, t.id as topic_id
            """)
            for record in result:
                data["subscriptions"].append({
                    "application": record["app_id"],
                    "topic": record["topic_id"],
                })
            
            # 7. Get ROUTES relationships (Broker -> Topic)
            result = session.run("""
                MATCH (b:Broker)-[r:ROUTES]->(t:Topic)
                RETURN b.id as broker_id, t.id as topic_id
            """)
            for record in result:
                data["routes"].append({
                    "broker": record["broker_id"],
                    "topic": record["topic_id"],
                })
            
            # 8. Get RUNS_ON relationships (App/Broker -> Node)
            result = session.run("""
                MATCH (c)-[r:RUNS_ON]->(n:Node)
                WHERE c:Application OR c:Broker
                RETURN c.id as comp_id, n.id as node_id, labels(c)[0] as comp_type
            """)
            for record in result:
                if record["comp_type"] == "Application":
                    data["runs_on"].append({
                        "application": record["comp_id"],
                        "node": record["node_id"],
                    })
                else:
                    data["runs_on"].append({
                        "broker": record["comp_id"],
                        "node": record["node_id"],
                    })
            
            # 9. Get CONNECTS_TO relationships (Node -> Node)
            result = session.run("""
                MATCH (n1:Node)-[r:CONNECTS_TO]->(n2:Node)
                RETURN n1.id as source, n2.id as target, 
                       coalesce(r.weight, 1.0) as weight
            """)
            for record in result:
                data["connections"].append({
                    "source": record["source"],
                    "target": record["target"],
                    "weight": float(record["weight"]),
                })
        
        self.logger.info(
            f"Retrieved raw graph: {len(data['nodes'])} nodes, "
            f"{len(data['brokers'])} brokers, {len(data['applications'])} apps, "
            f"{len(data['topics'])} topics"
        )
        
        return data