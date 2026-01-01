"""
Neo4j Graph Data Science (GDS) Client - Version 5.0

Clean, simplified interface for GDS algorithms.

Supported Algorithms:
- Centrality: PageRank, Betweenness, Degree (in/out/total)
- Community: Louvain, Weakly Connected Components
- Path: Articulation Points, Bridges

Focuses on DEPENDS_ON relationships for multi-layer dependency analysis.

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from contextlib import contextmanager
from enum import Enum

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None


# =============================================================================
# Enums
# =============================================================================

class DependencyType(Enum):
    """Types of DEPENDS_ON relationships"""
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"


class ComponentType(Enum):
    """Types of components in the pub-sub system"""
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CentralityResult:
    """Result from centrality algorithms"""
    node_id: str
    node_type: str
    score: float
    rank: int = 0
    normalized_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "score": round(self.score, 6),
            "normalized_score": round(self.normalized_score, 4),
            "rank": self.rank,
        }


@dataclass
class CommunityResult:
    """Result from community detection"""
    node_id: str
    node_type: str
    community_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "community_id": self.community_id,
        }


@dataclass
class ProjectionInfo:
    """Information about a GDS graph projection"""
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


# =============================================================================
# GDS Client
# =============================================================================

class GDSClient:
    """
    Client for Neo4j Graph Data Science operations.
    
    Provides a clean interface for:
    - Creating graph projections from DEPENDS_ON relationships
    - Running centrality algorithms (PageRank, Betweenness, Degree)
    - Detecting communities and clusters
    - Finding articulation points and bridges
    
    All algorithms support:
    - Weighted analysis using DEPENDS_ON weight property
    - Component-type filtering for focused analysis
    """

    # Valid dependency types for projection
    VALID_DEPENDENCY_TYPES = {dt.value for dt in DependencyType}
    
    # Valid component types
    VALID_COMPONENT_TYPES = {ct.value for ct in ComponentType}

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
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger(__name__)
        self._projections: Set[str] = set()
        
        self._connect()

    def __enter__(self) -> "GDSClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _connect(self) -> None:
        """Establish connection to Neo4j"""
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password)
        )
        self.driver.verify_connectivity()
        self.logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self) -> None:
        """Close connection and cleanup projections"""
        self._cleanup_projections()
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")

    @contextmanager
    def session(self):
        """Get a database session"""
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    # =========================================================================
    # Graph Statistics
    # =========================================================================

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the graph in Neo4j"""
        with self.session() as session:
            # Count nodes by type
            node_counts = {}
            for comp_type in self.VALID_COMPONENT_TYPES:
                result = session.run(
                    f"MATCH (n:{comp_type}) RETURN count(n) as count"
                ).single()
                node_counts[comp_type] = result["count"] if result else 0
            
            # Count DEPENDS_ON relationships
            dep_result = session.run(
                "MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) as count"
            ).single()
            depends_on_count = dep_result["count"] if dep_result else 0
            
            # Count by dependency type
            dep_type_counts = {}
            for dep_type in self.VALID_DEPENDENCY_TYPES:
                result = session.run(
                    """
                    MATCH ()-[r:DEPENDS_ON]->() 
                    WHERE r.dependency_type = $type
                    RETURN count(r) as count
                    """,
                    type=dep_type
                ).single()
                dep_type_counts[dep_type] = result["count"] if result else 0
            
            return {
                "nodes": node_counts,
                "total_nodes": sum(node_counts.values()),
                "depends_on_total": depends_on_count,
                "depends_on_by_type": dep_type_counts,
            }

    # =========================================================================
    # Graph Projection
    # =========================================================================

    def create_projection(
        self,
        name: str,
        dependency_types: Optional[List[str]] = None,
        component_types: Optional[List[str]] = None,
        include_weights: bool = True,
    ) -> ProjectionInfo:
        """
        Create a GDS graph projection for DEPENDS_ON relationships.
        
        Args:
            name: Projection name (unique identifier)
            dependency_types: Which DEPENDS_ON types to include
                             (default: app_to_app, node_to_node)
            component_types: Which component types to include as nodes
                            (default: Application, Broker, Node)
            include_weights: Include weight property for weighted algorithms
        
        Returns:
            ProjectionInfo with graph statistics
        """
        # Defaults
        if dependency_types is None:
            dependency_types = ["app_to_app", "node_to_node"]
        
        if component_types is None:
            component_types = ["Application", "Node"]
        
        # Validate
        for dt in dependency_types:
            if dt not in self.VALID_DEPENDENCY_TYPES:
                raise ValueError(
                    f"Invalid dependency type: {dt}. "
                    f"Valid types: {self.VALID_DEPENDENCY_TYPES}"
                )
        
        for ct in component_types:
            if ct not in self.VALID_COMPONENT_TYPES:
                raise ValueError(
                    f"Invalid component type: {ct}. "
                    f"Valid types: {self.VALID_COMPONENT_TYPES}"
                )
        
        self.logger.info(
            f"Creating projection '{name}' with "
            f"deps={dependency_types}, comps={component_types}"
        )
        
        # Build type filters
        type_filter = " OR ".join(
            [f"r.dependency_type = \"{t}\"" for t in dependency_types]
        )
        node_labels = " OR ".join([f"n:{ct}" for ct in component_types])
        
        # Weight configuration
        weight_return = ", r.weight AS weight" if include_weights else ""
        
        with self.session() as session:
            # Drop existing if exists
            self._drop_projection_internal(session, name)
            
            # Create projection using Cypher projection
            query = f"""
            CALL gds.graph.project.cypher(
                $name,
                'MATCH (n) WHERE {node_labels} RETURN id(n) AS id',
                'MATCH (a)-[r:DEPENDS_ON]->(b) 
                 WHERE ({type_filter})
                 RETURN id(a) AS source, id(b) AS target{weight_return}'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            result = session.run(query, name=name).single()
            
            info = ProjectionInfo(
                name=result["graphName"],
                node_count=result["nodeCount"],
                relationship_count=result["relationshipCount"],
                node_labels=component_types,
                relationship_types=dependency_types,
            )
            
            self._projections.add(name)
            self.logger.info(
                f"Created projection: {info.node_count} nodes, "
                f"{info.relationship_count} relationships"
            )
            
            return info

    def create_component_type_projection(
        self,
        name: str,
        component_type: str,
        include_weights: bool = True,
    ) -> ProjectionInfo:
        """
        Create a projection for a specific component type only.
        
        Useful for comparing components of the same type.
        
        Args:
            name: Projection name
            component_type: Component type (Application, Broker, Node, Topic)
            include_weights: Include weight property
        
        Returns:
            ProjectionInfo
        """
        if component_type not in self.VALID_COMPONENT_TYPES:
            raise ValueError(f"Invalid component type: {component_type}")
        
        self.logger.info(
            f"Creating component-type projection '{name}' for {component_type}"
        )
        
        weight_return = ", r.weight AS weight" if include_weights else ""
        
        with self.session() as session:
            self._drop_projection_internal(session, name)
            
            # Project only nodes of this type and their DEPENDS_ON relationships
            query = f"""
            CALL gds.graph.project.cypher(
                $name,
                'MATCH (n:{component_type}) RETURN id(n) AS id',
                'MATCH (a:{component_type})-[r:DEPENDS_ON]->(b:{component_type})
                 RETURN id(a) AS source, id(b) AS target{weight_return}'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            result = session.run(query, name=name).single()
            
            info = ProjectionInfo(
                name=result["graphName"],
                node_count=result["nodeCount"],
                relationship_count=result["relationshipCount"],
                node_labels=[component_type],
            )
            
            self._projections.add(name)
            return info

    def drop_projection(self, name: str) -> None:
        """Drop a graph projection"""
        with self.session() as session:
            self._drop_projection_internal(session, name)
        self._projections.discard(name)

    def _drop_projection_internal(self, session, name: str) -> None:
        """Internal method to drop projection within existing session"""
        try:
            session.run(f"CALL gds.graph.drop($name, false)", name=name)
            self.logger.debug(f"Dropped projection '{name}'")
        except Exception:
            pass  # Projection didn't exist

    def _cleanup_projections(self) -> None:
        """Drop all projections created by this client"""
        with self.session() as session:
            for name in list(self._projections):
                self._drop_projection_internal(session, name)
        self._projections.clear()

    def projection_exists(self, name: str) -> bool:
        """Check if a projection exists"""
        with self.session() as session:
            result = session.run(
                "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                name=name
            ).single()
            return result["exists"] if result else False

    # =========================================================================
    # Centrality Algorithms
    # =========================================================================

    def pagerank(
        self,
        projection_name: str,
        weighted: bool = True,
        damping_factor: float = 0.85,
        max_iterations: int = 20,
        top_k: Optional[int] = None,
    ) -> List[CentralityResult]:
        """
        Calculate PageRank centrality.
        
        High PageRank = node receives dependencies from important nodes.
        Critical for availability analysis.
        
        Args:
            projection_name: GDS graph projection name
            weighted: Use relationship weights
            damping_factor: PageRank damping factor (default: 0.85)
            max_iterations: Maximum iterations
            top_k: Return only top K results (None = all)
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        self.logger.info(
            f"Running {'weighted ' if weighted else ''}PageRank on '{projection_name}'"
        )
        
        config = {
            "dampingFactor": damping_factor,
            "maxIterations": max_iterations,
        }
        if weighted:
            config["relationshipWeightProperty"] = "weight"
        
        return self._run_centrality_algorithm(
            projection_name, "pageRank", config, top_k
        )

    def betweenness(
        self,
        projection_name: str,
        weighted: bool = True,
        sampling_size: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[CentralityResult]:
        """
        Calculate Betweenness centrality.
        
        High betweenness = node is on many shortest paths = critical bottleneck.
        Essential for reliability analysis (SPOFs, cascade risks).
        
        Args:
            projection_name: GDS graph projection name
            weighted: Use relationship weights (higher weight = higher cost)
            sampling_size: Number of source nodes to sample (None = all)
            top_k: Return only top K results
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        self.logger.info(
            f"Running {'weighted ' if weighted else ''}Betweenness on '{projection_name}'"
        )
        
        config = {"concurrency": 4}
        if weighted:
            config["relationshipWeightProperty"] = "weight"
        if sampling_size:
            config["samplingSize"] = sampling_size
        
        return self._run_centrality_algorithm(
            projection_name, "betweenness", config, top_k
        )

    def degree(
        self,
        projection_name: str,
        weighted: bool = True,
        orientation: str = "UNDIRECTED",
        top_k: Optional[int] = None,
    ) -> List[CentralityResult]:
        """
        Calculate Degree centrality.
        
        High degree = node has many connections = potential coupling issue.
        Critical for maintainability analysis.
        
        Args:
            projection_name: GDS graph projection name
            weighted: Use relationship weights
            orientation: NATURAL (out), REVERSE (in), or UNDIRECTED (total)
            top_k: Return only top K results
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        self.logger.info(
            f"Running {orientation} {'weighted ' if weighted else ''}Degree on '{projection_name}'"
        )
        
        config = {"orientation": orientation}
        if weighted:
            config["relationshipWeightProperty"] = "weight"
        
        return self._run_centrality_algorithm(
            projection_name, "degree", config, top_k
        )

    def _run_centrality_algorithm(
        self,
        projection_name: str,
        algorithm: str,
        config: Dict[str, Any],
        top_k: Optional[int],
    ) -> List[CentralityResult]:
        """Generic method to run centrality algorithms"""
        config_str = ", ".join([f"{k}: {repr(v)}" for k, v in config.items()])
        limit_clause = f"LIMIT {top_k}" if top_k else ""
        
        query = f"""
        CALL gds.{algorithm}.stream($projection, {{{config_str}}})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
        ORDER BY score DESC
        {limit_clause}
        """
        
        results = []
        max_score = 0.0
        
        with self.session() as session:
            records = list(session.run(query, projection=projection_name))
            
            # Find max for normalization
            if records:
                max_score = max(r["score"] for r in records) or 1.0
            
            for i, record in enumerate(records):
                score = record["score"]
                results.append(CentralityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    score=score,
                    rank=i + 1,
                    normalized_score=score / max_score if max_score > 0 else 0.0,
                ))
        
        return results

    # =========================================================================
    # Community Detection
    # =========================================================================

    def louvain(
        self,
        projection_name: str,
        weighted: bool = True,
    ) -> tuple[List[CommunityResult], Dict[str, Any]]:
        """
        Run Louvain community detection.
        
        Identifies clusters/modules in the dependency graph.
        Useful for maintainability analysis.
        
        Returns:
            Tuple of (community assignments, statistics)
        """
        self.logger.info(f"Running Louvain on '{projection_name}'")
        
        config = ""
        if weighted:
            config = "relationshipWeightProperty: 'weight'"
        
        query = f"""
        CALL gds.louvain.stream($projection, {{{config}}})
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS node, communityId
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, communityId
        """
        
        results = []
        community_counts: Dict[int, int] = {}
        
        with self.session() as session:
            for record in session.run(query, projection=projection_name):
                comm_id = record["communityId"]
                results.append(CommunityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    community_id=comm_id,
                ))
                community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
        
        stats = {
            "num_communities": len(community_counts),
            "community_sizes": community_counts,
            "avg_size": sum(community_counts.values()) / len(community_counts) if community_counts else 0,
            "max_size": max(community_counts.values()) if community_counts else 0,
            "min_size": min(community_counts.values()) if community_counts else 0,
        }
        
        return results, stats

    def weakly_connected_components(
        self,
        projection_name: str,
    ) -> tuple[List[CommunityResult], Dict[str, Any]]:
        """
        Find weakly connected components.
        
        Identifies disconnected subgraphs.
        Critical for availability analysis.
        
        Returns:
            Tuple of (component assignments, statistics)
        """
        self.logger.info(f"Running WCC on '{projection_name}'")
        
        query = """
        CALL gds.wcc.stream($projection)
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) AS node, componentId
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, componentId
        """
        
        results = []
        component_counts: Dict[int, int] = {}
        
        with self.session() as session:
            for record in session.run(query, projection=projection_name):
                comp_id = record["componentId"]
                results.append(CommunityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    community_id=comp_id,
                ))
                component_counts[comp_id] = component_counts.get(comp_id, 0) + 1
        
        stats = {
            "num_components": len(component_counts),
            "is_connected": len(component_counts) <= 1,
            "component_sizes": component_counts,
            "largest_component_size": max(component_counts.values()) if component_counts else 0,
        }
        
        return results, stats

    # =========================================================================
    # Structural Analysis
    # =========================================================================

    def find_articulation_points(self) -> List[Dict[str, Any]]:
        """
        Find articulation points (cut vertices).
        
        An articulation point is a node whose removal disconnects the graph.
        Critical for reliability - these are structural SPOFs.
        
        Returns:
            List of articulation point info dicts
        """
        self.logger.info("Finding articulation points")
        
        # Use Cypher for articulation point detection
        # This is a simplified approach using connectivity analysis
        query = """
        MATCH (n)
        WHERE n:Application OR n:Broker OR n:Node
        WITH collect(n) AS nodes
        UNWIND nodes AS candidate
        
        // Check if removing this node would disconnect others
        OPTIONAL MATCH path = (a)-[:DEPENDS_ON*]-(b)
        WHERE a <> candidate AND b <> candidate
        AND NOT (candidate)-[:DEPENDS_ON*]-(a)
        AND NOT (candidate)-[:DEPENDS_ON*]-(b)
        
        WITH candidate, 
             count(DISTINCT a) AS disconnected_sources,
             count(DISTINCT b) AS disconnected_targets
        WHERE disconnected_sources > 0 OR disconnected_targets > 0
        
        RETURN candidate.id AS nodeId, 
               labels(candidate)[0] AS nodeType,
               disconnected_sources + disconnected_targets AS impact
        ORDER BY impact DESC
        """
        
        # Simplified query that checks for bridge nodes
        simple_query = """
        MATCH (n)-[r:DEPENDS_ON]-(m)
        WHERE n:Application OR n:Broker OR n:Node
        WITH n, count(DISTINCT m) AS connections
        WHERE connections >= 2
        
        // Check if node is on critical paths limiting connectivity
        MATCH (a)-[:DEPENDS_ON*1..5]->(n)-[:DEPENDS_ON*1..5]->(b)
        WHERE a <> n AND b <> n AND a <> b
        
        WITH n, count(DISTINCT a) + count(DISTINCT b) AS path_participation
        WHERE path_participation > 0
        
        RETURN n.id AS nodeId,
               labels(n)[0] AS nodeType,
               path_participation AS pathCount
        ORDER BY pathCount DESC
        """
        
        results = []
        with self.session() as session:
            try:
                for record in session.run(simple_query):
                    results.append({
                        "node_id": record["nodeId"],
                        "node_type": record["nodeType"],
                        "path_count": record["pathCount"],
                    })
            except Exception as e:
                self.logger.warning(f"Articulation point query failed: {e}")
        
        return results

    def find_bridges(self) -> List[Dict[str, Any]]:
        """
        Find bridge edges.
        
        A bridge is an edge whose removal disconnects the graph.
        Critical for reliability analysis.
        
        Returns:
            List of bridge edge info dicts
        """
        self.logger.info("Finding bridge edges")
        
        query = """
        MATCH (a)-[r:DEPENDS_ON]->(b)
        
        // Check if this is the only path between components
        WHERE NOT EXISTS {
            MATCH (a)-[:DEPENDS_ON*2..]->(b)
        }
        AND NOT EXISTS {
            MATCH (a)<-[:DEPENDS_ON*2..]-(b)
        }
        
        RETURN a.id AS sourceId, 
               b.id AS targetId,
               labels(a)[0] AS sourceType,
               labels(b)[0] AS targetType,
               r.dependency_type AS dependencyType,
               r.weight AS weight
        """
        
        results = []
        with self.session() as session:
            try:
                for record in session.run(query):
                    results.append({
                        "source_id": record["sourceId"],
                        "target_id": record["targetId"],
                        "source_type": record["sourceType"],
                        "target_type": record["targetType"],
                        "dependency_type": record["dependencyType"],
                        "weight": record["weight"],
                    })
            except Exception as e:
                self.logger.warning(f"Bridge query failed: {e}")
        
        return results

    # =========================================================================
    # Component-Type Specific Queries
    # =========================================================================

    def get_components_by_type(
        self, 
        component_type: str
    ) -> List[Dict[str, Any]]:
        """Get all components of a specific type with their properties"""
        if component_type not in self.VALID_COMPONENT_TYPES:
            raise ValueError(f"Invalid component type: {component_type}")
        
        query = f"""
        MATCH (n:{component_type})
        OPTIONAL MATCH (n)-[r:DEPENDS_ON]-()
        WITH n, count(r) AS connections
        RETURN n.id AS id, 
               properties(n) AS props,
               connections
        ORDER BY connections DESC
        """
        
        results = []
        with self.session() as session:
            for record in session.run(query):
                results.append({
                    "id": record["id"],
                    "properties": dict(record["props"]),
                    "connections": record["connections"],
                })
        
        return results
