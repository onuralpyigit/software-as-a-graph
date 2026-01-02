"""
Neo4j GDS Client - Version 5.0

Client for Neo4j Graph Data Science library operations.

Provides:
- Graph projections for analysis
- Centrality algorithms (PageRank, Betweenness, Degree)
- Structural analysis (Articulation Points, Bridges)
- Weighted algorithm support

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Generator

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


# =============================================================================
# Constants
# =============================================================================

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
    "full": {
        "name": "Full System",
        "component_types": COMPONENT_TYPES,
        "dependency_types": DEPENDENCY_TYPES,
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CentralityResult:
    """Result from a centrality algorithm."""
    node_id: str
    node_type: str
    score: float
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "score": round(self.score, 6),
            "rank": self.rank,
        }


@dataclass
class ProjectionInfo:
    """Information about a graph projection."""
    name: str
    node_count: int
    relationship_count: int
    node_labels: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "node_count": self.node_count,
            "relationship_count": self.relationship_count,
            "node_labels": self.node_labels,
            "relationship_types": self.relationship_types,
        }


@dataclass 
class StructuralResult:
    """Result from structural analysis (articulation points, bridges)."""
    node_id: str
    node_type: str
    metric_name: str
    value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "metric": self.metric_name,
            "value": self.value,
            **self.metadata,
        }


# =============================================================================
# GDS Client
# =============================================================================

class GDSClient:
    """
    Neo4j Graph Data Science client.
    
    Provides access to GDS algorithms for graph analysis including
    centrality metrics, community detection, and structural analysis.
    
    Example:
        with GDSClient(uri, user, password) as gds:
            # Create projection
            info = gds.create_projection("my_graph")
            
            # Run PageRank
            results = gds.pagerank("my_graph", weighted=True)
            
            # Cleanup
            gds.drop_projection("my_graph")
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize GDS client.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j driver required: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._projections: set = set()
        self.logger = logging.getLogger(__name__)
        
        self._connect()
    
    def __enter__(self) -> GDSClient:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    def _connect(self) -> None:
        """Establish connection."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            raise ConnectionError(f"Cannot connect to Neo4j: {e}")
    
    def close(self) -> None:
        """Close connection and cleanup projections."""
        self._cleanup_projections()
        if self.driver:
            self.driver.close()
            self.driver = None
    
    @contextmanager
    def session(self) -> Generator:
        """Get a database session."""
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    # =========================================================================
    # Projection Management
    # =========================================================================
    
    def create_projection(
        self,
        name: str,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_weights: bool = True,
    ) -> ProjectionInfo:
        """
        Create a graph projection for GDS algorithms.
        
        Args:
            name: Projection name
            component_types: Node types to include (default: all)
            dependency_types: DEPENDS_ON types to include (default: all)
            include_weights: Include relationship weights
        
        Returns:
            ProjectionInfo with projection details
        """
        # Drop if exists
        self.drop_projection(name)
        
        # Build node projection
        node_types = component_types or COMPONENT_TYPES
        node_projection = {t: {} for t in node_types}
        
        # Build relationship projection with optional type filter
        rel_config = {"type": "DEPENDS_ON", "orientation": "NATURAL"}
        if include_weights:
            rel_config["properties"] = {"weight": {"defaultValue": 1.0}}
        
        rel_projection = {"DEPENDS_ON": rel_config}
        
        # Create projection using native projection
        query = """
        CALL gds.graph.project(
            $name,
            $nodeProjection,
            $relProjection
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        
        with self.session() as session:
            result = session.run(
                query,
                name=name,
                nodeProjection=node_projection,
                relProjection=rel_projection,
            ).single()
            
            info = ProjectionInfo(
                name=result["graphName"],
                node_count=result["nodeCount"],
                relationship_count=result["relationshipCount"],
                node_labels=node_types,
                relationship_types=["DEPENDS_ON"],
            )
        
        self._projections.add(name)
        self.logger.info(
            f"Created projection '{name}': {info.node_count} nodes, "
            f"{info.relationship_count} relationships"
        )
        
        return info
    
    def create_layer_projection(
        self,
        name: str,
        layer: str,
        include_weights: bool = True,
    ) -> ProjectionInfo:
        """
        Create projection for a specific layer.
        
        Args:
            name: Projection name
            layer: Layer name (application, infrastructure, app_broker, node_broker, full)
            include_weights: Include weights
        
        Returns:
            ProjectionInfo
        """
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}. Valid: {list(LAYER_DEFINITIONS.keys())}")
        
        layer_def = LAYER_DEFINITIONS[layer]
        
        return self.create_filtered_projection(
            name=name,
            component_types=layer_def["component_types"],
            dependency_types=layer_def["dependency_types"],
            include_weights=include_weights,
        )
    
    def create_filtered_projection(
        self,
        name: str,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_weights: bool = True,
    ) -> ProjectionInfo:
        """
        Create projection with filtered dependency types using Cypher projection.
        
        Args:
            name: Projection name
            component_types: Node types to include
            dependency_types: Dependency types to filter
            include_weights: Include weights
        
        Returns:
            ProjectionInfo
        """
        self.drop_projection(name)
        
        node_types = component_types or COMPONENT_TYPES
        dep_types = dependency_types or DEPENDENCY_TYPES
        
        # Build Cypher projection query
        node_labels = " OR ".join(f"n:{t}" for t in node_types)
        dep_types_str = ", ".join(f"\"{t}\"" for t in dep_types)
        
        weight_prop = ", r.weight AS weight" if include_weights else ""
        
        query = f"""
        CALL gds.graph.project.cypher(
            $name,
            'MATCH (n) WHERE {node_labels} RETURN id(n) AS id',
            'MATCH (a)-[r:DEPENDS_ON]->(b) 
             WHERE r.dependency_type IN [{dep_types_str}]
             AND ({node_labels.replace("n:", "a:")})
             AND ({node_labels.replace("n:", "b:")})
             RETURN id(a) AS source, id(b) AS target{weight_prop}'
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        
        with self.session() as session:
            result = session.run(query, name=name).single()
            
            info = ProjectionInfo(
                name=result["graphName"],
                node_count=result["nodeCount"],
                relationship_count=result["relationshipCount"],
                node_labels=node_types,
                relationship_types=dep_types,
            )
        
        self._projections.add(name)
        return info
    
    def drop_projection(self, name: str) -> bool:
        """Drop a projection if it exists."""
        if not self.projection_exists(name):
            return False
        
        with self.session() as session:
            session.run("CALL gds.graph.drop($name)", name=name)
        
        self._projections.discard(name)
        return True
    
    def projection_exists(self, name: str) -> bool:
        """Check if projection exists."""
        with self.session() as session:
            result = session.run(
                "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                name=name
            ).single()
            return result["exists"] if result else False
    
    def _cleanup_projections(self) -> None:
        """Drop all projections created by this client."""
        for name in list(self._projections):
            try:
                self.drop_projection(name)
            except Exception:
                pass
        self._projections.clear()
    
    # =========================================================================
    # Centrality Algorithms
    # =========================================================================
    
    def pagerank(
        self,
        projection: str,
        weighted: bool = True,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> List[CentralityResult]:
        """
        Run PageRank centrality.
        
        High PageRank = receives dependencies from important components.
        
        Args:
            projection: Graph projection name
            weighted: Use relationship weights
            damping: Damping factor (default 0.85)
            iterations: Maximum iterations
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        config = f"dampingFactor: {damping}, maxIterations: {iterations}"
        if weighted:
            config += ", relationshipWeightProperty: 'weight'"
        
        query = f"""
        CALL gds.pageRank.stream($projection, {{{config}}})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS id, labels(node)[0] AS type, score
        ORDER BY score DESC
        """
        
        return self._run_centrality_query(query, projection, "pagerank")
    
    def betweenness(
        self,
        projection: str,
        weighted: bool = True,
        sampling_size: Optional[int] = None,
    ) -> List[CentralityResult]:
        """
        Run Betweenness centrality.
        
        High betweenness = component is on many shortest paths (bottleneck).
        
        Args:
            projection: Graph projection name
            weighted: Use relationship weights
            sampling_size: Sample size for approximation (None = exact)
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        config_parts = []
        if weighted:
            config_parts.append("relationshipWeightProperty: 'weight'")
        if sampling_size:
            config_parts.append(f"samplingSize: {sampling_size}")
        
        config = ", ".join(config_parts) if config_parts else ""
        config_str = f"{{{config}}}" if config else ""
        
        query = f"""
        CALL gds.betweenness.stream($projection{', ' + config_str if config_str else ''})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS id, labels(node)[0] AS type, score
        ORDER BY score DESC
        """
        
        return self._run_centrality_query(query, projection, "betweenness")
    
    def degree(
        self,
        projection: str,
        weighted: bool = True,
        orientation: str = "UNDIRECTED",
    ) -> List[CentralityResult]:
        """
        Run Degree centrality.
        
        High degree = many direct connections (highly coupled).
        
        Args:
            projection: Graph projection name
            weighted: Use relationship weights (returns weighted degree)
            orientation: NATURAL (out), REVERSE (in), UNDIRECTED (both)
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        config = f"orientation: '{orientation}'"
        if weighted:
            config += ", relationshipWeightProperty: 'weight'"
        
        query = f"""
        CALL gds.degree.stream($projection, {{{config}}})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS id, labels(node)[0] AS type, score
        ORDER BY score DESC
        """
        
        return self._run_centrality_query(query, projection, "degree")
    
    def _run_centrality_query(
        self,
        query: str,
        projection: str,
        metric_name: str,
    ) -> List[CentralityResult]:
        """Execute centrality query and return results."""
        results = []
        
        with self.session() as session:
            for rank, record in enumerate(session.run(query, projection=projection), 1):
                results.append(CentralityResult(
                    node_id=record["id"],
                    node_type=record["type"],
                    score=record["score"],
                    rank=rank,
                ))
        
        self.logger.debug(f"{metric_name}: {len(results)} results")
        return results
    
    # =========================================================================
    # Structural Analysis
    # =========================================================================
    
    def find_articulation_points(self) -> List[StructuralResult]:
        """
        Find articulation points (cut vertices).
        
        Articulation points are nodes whose removal disconnects the graph.
        These are structural single points of failure.
        
        Returns:
            List of articulation point results
        """
        # Use path-based detection: nodes on critical paths
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node
        WITH n
        
        // Count paths through this node
        OPTIONAL MATCH path = (a)-[:DEPENDS_ON*1..3]->(n)-[:DEPENDS_ON*1..3]->(b)
        WHERE a <> n AND b <> n AND a <> b
        
        WITH n, count(DISTINCT path) AS path_count
        WHERE path_count > 0
        
        // Check if node connects otherwise disconnected components
        OPTIONAL MATCH (n)-[:DEPENDS_ON]-(neighbor)
        WITH n, path_count, count(DISTINCT neighbor) AS neighbor_count
        WHERE neighbor_count >= 2
        
        RETURN n.id AS id, labels(n)[0] AS type, path_count, neighbor_count
        ORDER BY path_count DESC
        """
        
        results = []
        with self.session() as session:
            for record in session.run(query):
                results.append(StructuralResult(
                    node_id=record["id"],
                    node_type=record["type"],
                    metric_name="articulation_point",
                    value=1.0,
                    metadata={
                        "path_count": record["path_count"],
                        "neighbor_count": record["neighbor_count"],
                    },
                ))
        
        return results
    
    def find_bridges(self) -> List[Dict[str, Any]]:
        """
        Find bridge edges.
        
        Bridges are edges whose removal disconnects the graph.
        
        Returns:
            List of bridge edge dictionaries
        """
        # Find edges that are the only connection between their endpoints
        query = """
        MATCH (a)-[r:DEPENDS_ON]->(b)
        
        // Check if there's an alternative path
        WHERE NOT EXISTS {
            MATCH path = (a)-[:DEPENDS_ON*2..4]->(b)
            WHERE none(n IN nodes(path) WHERE n = a OR n = b)
        }
        
        // Verify this edge is critical
        WITH a, r, b
        OPTIONAL MATCH (a)-[other:DEPENDS_ON]->(b)
        WHERE other <> r
        WITH a, r, b, count(other) AS alt_count
        WHERE alt_count = 0
        
        RETURN a.id AS source_id, 
               b.id AS target_id,
               labels(a)[0] AS source_type,
               labels(b)[0] AS target_type,
               r.dependency_type AS dependency_type,
               coalesce(r.weight, 1.0) AS weight
        """
        
        bridges = []
        with self.session() as session:
            for record in session.run(query):
                bridges.append({
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "source_type": record["source_type"],
                    "target_type": record["target_type"],
                    "dependency_type": record["dependency_type"],
                    "weight": record["weight"],
                })
        
        return bridges
    
    # =========================================================================
    # Data Retrieval
    # =========================================================================
    
    def get_component_weights(self) -> Dict[str, float]:
        """Get component weights from database."""
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node
        RETURN n.id AS id, coalesce(n.weight, 0.0) AS weight
        """
        
        weights = {}
        with self.session() as session:
            for record in session.run(query):
                weights[record["id"]] = record["weight"]
        
        return weights
    
    def get_edge_weights(
        self,
        dependency_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get DEPENDS_ON edge weights."""
        type_filter = ""
        if dependency_types:
            types_str = ", ".join(f"'{t}'" for t in dependency_types)
            type_filter = f"WHERE r.dependency_type IN [{types_str}]"
        
        query = f"""
        MATCH (a)-[r:DEPENDS_ON]->(b)
        {type_filter}
        RETURN a.id AS source, b.id AS target,
               r.dependency_type AS type,
               coalesce(r.weight, 1.0) AS weight
        """
        
        edges = []
        with self.session() as session:
            for record in session.run(query):
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"],
                    "weight": record["weight"],
                })
        
        return edges
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node OR n:Topic
        WITH labels(n)[0] AS type, count(*) AS count
        RETURN type, count
        ORDER BY count DESC
        """
        
        stats = {"nodes": {}, "relationships": {}}
        
        with self.session() as session:
            for record in session.run(query):
                stats["nodes"][record["type"]] = record["count"]
            
            # Relationship stats
            rel_query = """
            MATCH ()-[r:DEPENDS_ON]->()
            RETURN r.dependency_type AS type, count(*) AS count
            """
            for record in session.run(rel_query):
                stats["relationships"][record["type"]] = record["count"]
        
        stats["total_nodes"] = sum(stats["nodes"].values())
        stats["total_relationships"] = sum(stats["relationships"].values())
        
        return stats
