"""
Neo4j Client

Neo4j client for graph data retrieval.
Supports both Derived Dependency graphs and Raw Structural graphs.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

try:
    from neo4j import GraphDatabase, exceptions
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

COMPONENT_TYPES = ["Application", "Broker", "Node", "Topic"]

# Raw structural relationships
STRUCTURAL_REL_TYPES = [
    "PUBLISHES_TO", 
    "SUBSCRIBES_TO", 
    "RUNS_ON", 
    "ROUTES", 
    "CONNECTS_TO"
]

@dataclass
class ComponentData:
    id: str
    component_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeData:
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    relation_type: str # Replaces dependency_type for generic use
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphData:
    components: List[ComponentData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_components_by_type(self, comp_type: str) -> List[ComponentData]:
        return [c for c in self.components if c.component_type == comp_type]

class GraphExporter:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j"):
        if not HAS_NEO4J: raise ImportError("neo4j driver not installed.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.driver.close()

    def get_structural_graph(self) -> GraphData:
        """
        Retrieves the raw topological graph (App, Topic, Node, Broker)
        and their direct relationships (Pub, Sub, Runs_On, etc.)
        WITHOUT derived DEPENDS_ON edges.
        """
        # Get all relevant components
        components = self._run_query_components(
            "MATCH (n) WHERE n:Application OR n:Broker OR n:Node OR n:Topic RETURN n"
        )
        
        # Get structural edges
        # Note: We explicitly fetch specific relationship types
        rel_types = "|".join([f"{r}" for r in STRUCTURAL_REL_TYPES])
        edges = self._run_query_edges(
            f"MATCH (s)-[r:{rel_types}]->(t) RETURN s, t, r"
        )
        
        return GraphData(components=components, edges=edges, metadata={"type": "structural"})

    def _run_query_components(self, query: str) -> List[ComponentData]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            comps = []
            for record in result:
                node = record["n"]
                # Determine primary label (filtering out generic labels if any)
                labels = [l for l in node.labels if l in COMPONENT_TYPES]
                c_type = labels[0] if labels else "Unknown"
                
                props = dict(node)
                weight = props.pop("weight", 1.0)
                c_id = props.pop("id", node.element_id) # Fallback to internal ID if no prop
                
                comps.append(ComponentData(id=c_id, component_type=c_type, weight=weight, properties=props))
            return comps

    def _run_query_edges(self, query: str) -> List[EdgeData]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            edges = []
            for record in result:
                s, t, r = record["s"], record["t"], record["r"]
                
                # Resolve IDs
                s_id = s.get("id", s.element_id)
                t_id = t.get("id", t.element_id)
                
                # Resolve Types
                s_labels = [l for l in s.labels if l in COMPONENT_TYPES]
                t_labels = [l for l in t.labels if l in COMPONENT_TYPES]
                s_type = s_labels[0] if s_labels else "Unknown"
                t_type = t_labels[0] if t_labels else "Unknown"
                
                props = dict(r)
                weight = props.pop("weight", 1.0)
                
                edges.append(EdgeData(
                    source_id=s_id, target_id=t_id,
                    source_type=s_type, target_type=t_type,
                    relation_type=r.type, weight=weight, properties=props
                ))
            return edges