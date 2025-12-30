"""
Neo4j Graph Data Science (GDS) Client - Version 4.0

Clean interface for GDS algorithms used in pub-sub system analysis:
- Centrality: PageRank, Betweenness, Degree
- Community: Louvain, Weakly Connected Components
- Path: Articulation Points, Bridges

Focuses on DEPENDS_ON relationships for multi-layer dependency analysis.

Usage:
    from src.analysis.gds_client import GDSClient

    with GDSClient(uri, user, password) as gds:
        gds.create_projection("my_graph", ["app_to_app", "node_to_node"])
        results = gds.pagerank("my_graph", weighted=True)
        gds.drop_projection("my_graph")

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None


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

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass
class CommunityResult:
    """Result from community detection"""
    node_id: str
    node_type: str
    community_id: int

    def to_dict(self) -> Dict:
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


# =============================================================================
# GDS Client
# =============================================================================

class GDSClient:
    """
    Client for Neo4j Graph Data Science operations.
    
    Provides methods to:
    - Create graph projections for DEPENDS_ON analysis
    - Run centrality algorithms (PageRank, Betweenness, Degree)
    - Detect communities and clusters
    - Find articulation points and bridges
    - Analyze connectivity
    
    All algorithms support weighted analysis using the DEPENDS_ON weight property.
    """

    # Supported dependency types
    DEPENDENCY_TYPES = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        if not HAS_NEO4J:
            raise ImportError("neo4j driver not installed. Install with: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger(__name__)
        self._projections: List[str] = []
        
        self._connect()

    def __enter__(self) -> GDSClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _connect(self) -> None:
        """Establish connection to Neo4j"""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
        self.logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self) -> None:
        """Close connection and cleanup projections"""
        self.cleanup_projections()
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
    # Graph Projection
    # =========================================================================

    def create_projection(
        self,
        name: str,
        dependency_types: Optional[List[str]] = None,
        include_weights: bool = True,
    ) -> ProjectionInfo:
        """
        Create a GDS graph projection for DEPENDS_ON relationships.
        
        Args:
            name: Projection name
            dependency_types: List of dependency types to include
                             (default: app_to_app, node_to_node)
            include_weights: Include weight property for weighted algorithms
        
        Returns:
            ProjectionInfo with graph statistics
        """
        if dependency_types is None:
            dependency_types = ["app_to_app", "node_to_node"]
        
        # Validate dependency types
        for dt in dependency_types:
            if dt not in self.DEPENDENCY_TYPES:
                raise ValueError(f"Invalid dependency type: {dt}. Valid: {self.DEPENDENCY_TYPES}")
        
        self.logger.info(f"Creating projection '{name}' with types: {dependency_types}")
        
        # Build type filter
        type_filter = " OR ".join([f"r.dependency_type = \"{t}\"" for t in dependency_types])
        
        # Build projection config
        weight_config = ", properties: {weight: {property: 'weight', defaultValue: 1.0}}" if include_weights else ""
        
        with self.session() as session:
            # Drop existing projection if exists
            try:
                session.run(f"CALL gds.graph.drop('{name}', false)")
            except Exception:
                pass
            
            # Create projection using Cypher projection
            query = f"""
            CALL gds.graph.project.cypher(
                '{name}',
                'MATCH (n) WHERE n:Application OR n:Broker OR n:Node RETURN id(n) AS id',
                'MATCH (a)-[r:DEPENDS_ON]->(b) WHERE {type_filter}
                 RETURN id(a) AS source, id(b) AS target, r.weight AS weight'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            result = session.run(query).single()
            
            info = ProjectionInfo(
                name=result["graphName"],
                node_count=result["nodeCount"],
                relationship_count=result["relationshipCount"],
                relationship_types=dependency_types,
            )
            
            self._projections.append(name)
            self.logger.info(f"Created projection: {info.node_count} nodes, {info.relationship_count} relationships")
            
            return info

    def drop_projection(self, name: str) -> None:
        """Drop a graph projection"""
        with self.session() as session:
            try:
                session.run(f"CALL gds.graph.drop('{name}', false)")
                if name in self._projections:
                    self._projections.remove(name)
                self.logger.info(f"Dropped projection '{name}'")
            except Exception as e:
                self.logger.debug(f"Could not drop projection '{name}': {e}")

    def cleanup_projections(self) -> None:
        """Drop all projections created by this client"""
        for name in list(self._projections):
            self.drop_projection(name)

    def projection_exists(self, name: str) -> bool:
        """Check if a projection exists"""
        with self.session() as session:
            result = session.run(
                "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                name=name
            ).single()
            return result["exists"]

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
            top_k: Return only top K results
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        self.logger.info(f"Running {'weighted ' if weighted else ''}PageRank on '{projection_name}'")
        
        weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
        limit_clause = f"LIMIT {top_k}" if top_k else ""
        
        query = f"""
        CALL gds.pageRank.stream('{projection_name}', {{
            {weight_config}
            dampingFactor: {damping_factor},
            maxIterations: {max_iterations}
        }})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
        ORDER BY score DESC
        {limit_clause}
        """
        
        results = []
        with self.session() as session:
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    score=record["score"],
                    rank=i + 1,
                ))
        
        return results

    def betweenness(
        self,
        projection_name: str,
        weighted: bool = True,
        top_k: Optional[int] = None,
    ) -> List[CentralityResult]:
        """
        Calculate Betweenness centrality.
        
        High betweenness = node is on many shortest paths = critical bottleneck.
        Essential for reliability analysis (SPOFs, cascade risks).
        
        Args:
            projection_name: GDS graph projection name
            weighted: Use relationship weights (higher weight = higher cost)
            top_k: Return only top K results
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        self.logger.info(f"Running {'weighted ' if weighted else ''}Betweenness on '{projection_name}'")
        
        weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
        limit_clause = f"LIMIT {top_k}" if top_k else ""
        
        query = f"""
        CALL gds.betweenness.stream('{projection_name}', {{
            {weight_config}
            concurrency: 4
        }})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
        ORDER BY score DESC
        {limit_clause}
        """
        
        results = []
        with self.session() as session:
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    score=record["score"],
                    rank=i + 1,
                ))
        
        return results

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
            weighted: Use relationship weights (returns sum of weights)
            orientation: NATURAL (out), REVERSE (in), or UNDIRECTED
            top_k: Return only top K results
        
        Returns:
            List of CentralityResult sorted by score descending
        """
        self.logger.info(f"Running {'weighted ' if weighted else ''}{orientation} Degree on '{projection_name}'")
        
        weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
        limit_clause = f"LIMIT {top_k}" if top_k else ""
        
        query = f"""
        CALL gds.degree.stream('{projection_name}', {{
            {weight_config}
            orientation: '{orientation}'
        }})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
        ORDER BY score DESC
        {limit_clause}
        """
        
        results = []
        with self.session() as session:
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    score=record["score"],
                    rank=i + 1,
                ))
        
        return results

    # =========================================================================
    # Community Detection
    # =========================================================================

    def louvain(
        self,
        projection_name: str,
        weighted: bool = True,
    ) -> Tuple[List[CommunityResult], Dict[str, Any]]:
        """
        Run Louvain community detection.
        
        Identifies clusters of tightly connected components.
        Low modularity indicates poor system decomposition.
        
        Returns:
            Tuple of (community assignments, statistics)
        """
        self.logger.info(f"Running Louvain community detection on '{projection_name}'")
        
        weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
        
        query = f"""
        CALL gds.louvain.stream('{projection_name}', {{
            {weight_config}
            includeIntermediateCommunities: false
        }})
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS node, communityId
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, communityId
        """
        
        results = []
        community_sizes = {}
        
        with self.session() as session:
            for record in session.run(query):
                results.append(CommunityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    community_id=record["communityId"],
                ))
                cid = record["communityId"]
                community_sizes[cid] = community_sizes.get(cid, 0) + 1
        
        stats = {
            "community_count": len(community_sizes),
            "largest_community": max(community_sizes.values()) if community_sizes else 0,
            "smallest_community": min(community_sizes.values()) if community_sizes else 0,
            "community_sizes": community_sizes,
        }
        
        return results, stats

    def weakly_connected_components(
        self,
        projection_name: str,
    ) -> Tuple[List[CommunityResult], Dict[str, Any]]:
        """
        Find weakly connected components.
        
        Multiple components indicate disconnected system parts.
        Critical for availability analysis.
        
        Returns:
            Tuple of (component assignments, statistics)
        """
        self.logger.info(f"Running WCC on '{projection_name}'")
        
        query = f"""
        CALL gds.wcc.stream('{projection_name}')
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) AS node, componentId
        RETURN node.id AS nodeId, labels(node)[0] AS nodeType, componentId
        """
        
        results = []
        component_sizes = {}
        
        with self.session() as session:
            for record in session.run(query):
                results.append(CommunityResult(
                    node_id=record["nodeId"],
                    node_type=record["nodeType"],
                    community_id=record["componentId"],
                ))
                cid = record["componentId"]
                component_sizes[cid] = component_sizes.get(cid, 0) + 1
        
        stats = {
            "component_count": len(component_sizes),
            "is_connected": len(component_sizes) <= 1,
            "largest_component": max(component_sizes.values()) if component_sizes else 0,
            "component_sizes": list(component_sizes.values()),
        }
        
        return results, stats

    # =========================================================================
    # Structural Analysis
    # =========================================================================

    def find_articulation_points(self) -> List[Dict[str, Any]]:
        """
        Find articulation points (nodes whose removal disconnects the graph).
        
        These are Single Points of Failure (SPOFs).
        Uses native Neo4j query since GDS doesn't have built-in AP detection.
        
        Returns:
            List of articulation point info with impact analysis
        """
        self.logger.info("Finding articulation points")
        
        # Get all DEPENDS_ON connected nodes
        query = """
        // Get the graph structure
        MATCH (a)-[:DEPENDS_ON]->(b)
        WITH collect(DISTINCT a) + collect(DISTINCT b) AS nodes,
             collect({source: a.id, target: b.id}) AS edges
        
        // For each node, check if removal increases components
        UNWIND nodes AS candidate
        WITH candidate, nodes, edges
        
        // Count edges not involving this node
        WITH candidate,
             [e IN edges WHERE e.source <> candidate.id AND e.target <> candidate.id] AS remaining_edges,
             [n IN nodes WHERE n <> candidate] AS remaining_nodes
        
        // Simple connectivity check (this is approximate)
        WITH candidate, size(remaining_nodes) AS node_count, size(remaining_edges) AS edge_count
        WHERE edge_count < node_count - 1  // Potentially disconnecting
        
        RETURN candidate.id AS nodeId,
               labels(candidate)[0] AS nodeType,
               node_count AS remaining_nodes,
               edge_count AS remaining_edges
        """
        
        results = []
        with self.session() as session:
            for record in session.run(query):
                results.append({
                    "node_id": record["nodeId"],
                    "node_type": record["nodeType"],
                    "impact": "Potential articulation point",
                })
        
        # Fallback: identify high-betweenness nodes as potential APs
        if not results:
            self.logger.info("Using betweenness-based AP detection")
            # Create temp projection
            try:
                self.create_projection("_temp_ap", include_weights=False)
                bc_results = self.betweenness("_temp_ap", weighted=False, top_k=10)
                
                if bc_results:
                    max_score = bc_results[0].score
                    threshold = max_score * 0.5  # Top 50% of max betweenness
                    
                    for r in bc_results:
                        if r.score >= threshold and r.score > 0:
                            results.append({
                                "node_id": r.node_id,
                                "node_type": r.node_type,
                                "betweenness": r.score,
                                "impact": f"High betweenness ({r.score:.2f}) - potential SPOF",
                            })
                
                self.drop_projection("_temp_ap")
            except Exception as e:
                self.logger.warning(f"AP detection fallback failed: {e}")
        
        return results

    def find_bridges(self) -> List[Dict[str, Any]]:
        """
        Find bridge edges (edges whose removal disconnects components).
        
        These are critical single-path dependencies.
        
        Returns:
            List of bridge edge info
        """
        self.logger.info("Finding bridge edges")
        
        # Identify edges that are the only path between components
        query = """
        MATCH (a)-[r:DEPENDS_ON]->(b)
        WHERE NOT exists {
            MATCH (a)-[:DEPENDS_ON*2..3]->(b)
        }
        AND NOT exists {
            MATCH (a)<-[:DEPENDS_ON]-(x)-[:DEPENDS_ON]->(b)
            WHERE x <> a AND x <> b
        }
        RETURN a.id AS source, b.id AS target,
               r.dependency_type AS type, r.weight AS weight,
               labels(a)[0] AS sourceType, labels(b)[0] AS targetType
        """
        
        results = []
        with self.session() as session:
            for record in session.run(query):
                results.append({
                    "source": record["source"],
                    "target": record["target"],
                    "source_type": record["sourceType"],
                    "target_type": record["targetType"],
                    "dependency_type": record["type"],
                    "weight": record["weight"],
                })
        
        return results

    # =========================================================================
    # Graph Statistics
    # =========================================================================

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        with self.session() as session:
            # Node counts
            node_result = session.run("""
                MATCH (n)
                WHERE n:Application OR n:Broker OR n:Node
                WITH labels(n)[0] AS label, count(*) AS count
                RETURN collect({label: label, count: count}) AS nodes
            """).single()
            
            # DEPENDS_ON counts
            dep_result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                WITH d.dependency_type AS type, count(*) AS count,
                     avg(d.weight) AS avg_weight, max(d.weight) AS max_weight
                RETURN collect({
                    type: type, count: count,
                    avg_weight: avg_weight, max_weight: max_weight
                }) AS dependencies
            """).single()
            
            return {
                "nodes": {item["label"]: item["count"] for item in node_result["nodes"]},
                "dependencies": {
                    item["type"]: {
                        "count": item["count"],
                        "avg_weight": item["avg_weight"],
                        "max_weight": item["max_weight"],
                    }
                    for item in dep_result["dependencies"]
                },
            }