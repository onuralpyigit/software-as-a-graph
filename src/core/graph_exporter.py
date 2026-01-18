"""
Graph Exporter - Version 7.1

Neo4j client for graph data retrieval.
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
COMPONENT_TYPES = ["Application", "Broker", "Node", "Topic", "Library"]

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
    "CONNECTS_TO",
    "USES"
]

@dataclass
class ComponentData:
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
    components: List[ComponentData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
        }
    
    def get_components_by_type(self, comp_type: str) -> List[ComponentData]:
        return [c for c in self.components if c.component_type == comp_type]
    
    def get_edges_by_type(self, dep_type: str) -> List[EdgeData]:
        return [e for e in self.edges if e.dependency_type == dep_type]


class GraphExporter:
    def __init__(self, uri: str, user: str = "neo4j", password: str = "password"):
        if not HAS_NEO4J:
            raise ImportError("neo4j package not installed.")
        
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
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    
    def get_graph_data(self, component_types: Optional[List[str]] = None, dependency_types: Optional[List[str]] = None) -> GraphData:
        components = self._get_components(component_types)
        component_ids = {c.id for c in components}
        edges = self._get_edges(dependency_types, component_ids)
        return GraphData(components=components, edges=edges)
    
    def _get_components(self, component_types: Optional[List[str]] = None) -> List[ComponentData]:
        types = component_types or COMPONENT_TYPES
        types_str = ", ".join(f"'{t}'" for t in types)
        
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN [{types_str}])
        RETURN n.id as id, labels(n)[0] as type, coalesce(n.weight, 1.0) as weight, properties(n) as props
        """
        
        components = []
        with self.driver.session() as session:
            result = session.run(query)
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
    
    def _get_edges(self, dependency_types: Optional[List[str]] = None, component_ids: Optional[Set[str]] = None) -> List[EdgeData]:
        types = dependency_types or DEPENDENCY_TYPES
        types_str = ", ".join(f"'{t}'" for t in types)
        
        query = f"""
        MATCH (source)-[r:DEPENDS_ON]->(target)
        WHERE r.dependency_type IN [{types_str}]
        RETURN source.id as source_id, target.id as target_id, labels(source)[0] as source_type,
               labels(target)[0] as target_type, r.dependency_type as dependency_type,
               coalesce(r.weight, 1.0) as weight, properties(r) as props
        """
        
        edges = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                if component_ids and (record["source_id"] not in component_ids or record["target_id"] not in component_ids):
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
        return edges
    
    def get_layer_data(self, layer: str) -> GraphData:
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}")
        return self.get_graph_data(
            component_types=LAYER_DEFINITIONS[layer]["component_types"],
            dependency_types=LAYER_DEFINITIONS[layer]["dependency_types"],
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        with self.driver.session() as session:
            for comp_type in COMPONENT_TYPES:
                result = session.run(f"MATCH (n:{comp_type}) RETURN count(n) as c")
                stats[f"{comp_type.lower()}_count"] = result.single()["c"]
            for dep_type in DEPENDENCY_TYPES:
                result = session.run(f"MATCH ()-[r:DEPENDS_ON {{dependency_type: '{dep_type}'}}]->() RETURN count(r) as c")
                stats[f"{dep_type}_count"] = result.single()["c"]
        return stats
    
    def get_raw_structural_graph(self) -> Dict[str, Any]:
        data = {
            "nodes": [], "brokers": [], "applications": [], "topics": [], "libraries": [],
            "publications": [], "subscriptions": [], "routes": [], "runs_on": [], "connections": [], "uses": []
        }
        
        with self.driver.session() as session:
            # Entities
            for label, key in [("Node", "nodes"), ("Broker", "brokers"), ("Application", "applications"), ("Library", "libraries")]:
                result = session.run(f"MATCH (n:{label}) RETURN n.id as id, n.name as name, coalesce(n.weight, 1.0) as weight, properties(n) as props")
                for record in result:
                    props = dict(record["props"])
                    # Special handling for Application role
                    role = props.get("role", "pubsub") if label == "Application" else None
                    item = {"id": record["id"], "name": record["name"] or record["id"], "weight": float(record["weight"])}
                    if role: item["role"] = role
                    data[key].append(item)

            # Topics
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.id as id, t.name as name, coalesce(t.size, 0) as size,
                       coalesce(t.reliability, 'BEST_EFFORT') as reliability,
                       coalesce(t.durability, 'VOLATILE') as durability,
                       coalesce(t.transport_priority, 'LOW') as priority
            """)
            for record in result:
                data["topics"].append({
                    "id": record["id"], "name": record["name"] or record["id"], "size": int(record["size"]),
                    "qos": {"reliability": record["reliability"], "durability": record["durability"], "transport_priority": record["priority"]}
                })
            
            # Relationships
            # PUBLISHES_TO (Apps and Libs)
            result = session.run("MATCH (n)-[:PUBLISHES_TO]->(t:Topic) RETURN n.id as src, t.id as tgt")
            for r in result: data["publications"].append({"source": r["src"], "topic": r["tgt"]})
            
            # SUBSCRIBES_TO (Apps and Libs)
            result = session.run("MATCH (n)-[:SUBSCRIBES_TO]->(t:Topic) RETURN n.id as src, t.id as tgt")
            for r in result: data["subscriptions"].append({"source": r["src"], "topic": r["tgt"]})
            
            # ROUTES
            result = session.run("MATCH (b:Broker)-[:ROUTES]->(t:Topic) RETURN b.id as src, t.id as tgt")
            for r in result: data["routes"].append({"broker": r["src"], "topic": r["tgt"]})
            
            # RUNS_ON
            result = session.run("MATCH (c)-[:RUNS_ON]->(n:Node) RETURN c.id as src, n.id as tgt, labels(c)[0] as type")
            for r in result: 
                key = "application" if r["type"] == "Application" else "broker"
                data["runs_on"].append({key: r["src"], "node": r["tgt"]})
                
            # CONNECTS_TO
            result = session.run("MATCH (n1:Node)-[r:CONNECTS_TO]->(n2:Node) RETURN n1.id as src, n2.id as tgt, coalesce(r.weight, 1.0) as w")
            for r in result: data["connections"].append({"source": r["src"], "target": r["tgt"], "weight": float(r["w"])})

            # USES
            result = session.run("MATCH (a)-[:USES]->(b) RETURN a.id as src, b.id as tgt")
            for r in result: data["uses"].append({"source": r["src"], "target": r["tgt"]})
        
        return data