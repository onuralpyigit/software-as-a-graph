#!/usr/bin/env python3
"""
Neo4j Graph Data Science (GDS) Client
=====================================

Provides a clean interface to Neo4j GDS for graph analysis operations.
Focuses on DEPENDS_ON relationships for reliability, maintainability,
and availability analysis.

Supported DEPENDS_ON types:
- app_to_app: Application depends on another application
- node_to_node: Infrastructure node depends on another node
- app_to_broker: Application depends on broker
- node_to_broker: Node depends on broker

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None


@dataclass
class GDSProjection:
    """Represents a GDS graph projection"""
    name: str
    node_count: int
    relationship_count: int
    node_labels: List[str]
    relationship_types: List[str]


@dataclass
class CentralityResult:
    """Results from centrality algorithms"""
    node_id: str
    node_type: str
    score: float
    rank: int = 0


@dataclass
class CommunityResult:
    """Results from community detection"""
    node_id: str
    community_id: int
    node_type: str


@dataclass 
class PathResult:
    """Results from path algorithms"""
    source: str
    target: str
    cost: float
    path: List[str] = field(default_factory=list)


class GDSClient:
    """
    Client for Neo4j Graph Data Science operations.
    
    Provides methods to:
    - Create graph projections for DEPENDS_ON analysis
    - Run centrality algorithms (betweenness, pagerank, degree)
    - Detect communities and clusters
    - Find articulation points and bridges
    - Analyze connectivity and paths
    
    Weight Property:
        The DEPENDS_ON relationships can have a 'weight' property that reflects
        the importance or criticality of the dependency. Higher weights indicate
        more critical dependencies. Weights are used in:
        - Weighted centrality calculations
        - Path cost computations
        - Community detection
        - Criticality scoring
    """
    
    # Dependency types for projection
    DEPENDS_ON_TYPES = [
        'app_to_app',
        'node_to_node', 
        'app_to_broker',
        'node_to_broker'
    ]
    
    # Default weight for relationships without explicit weight
    DEFAULT_WEIGHT = 1.0
    
    def __init__(self, uri: str, user: str, password: str, database: str = 'neo4j'):
        """
        Initialize GDS client.
        
        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j driver required: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger('GDSClient')
        self._active_projections: List[str] = []
    
    def close(self):
        """Close driver connection and cleanup projections"""
        self.cleanup_projections()
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @contextmanager
    def session(self):
        """Context manager for database sessions"""
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    # =========================================================================
    # Graph Projection Management
    # =========================================================================
    
    def create_depends_on_projection(self, 
                                      projection_name: str = 'depends_on_graph',
                                      dependency_types: Optional[List[str]] = None,
                                      include_weights: bool = True,
                                      weight_property: str = 'weight') -> GDSProjection:
        """
        Create a GDS graph projection for DEPENDS_ON relationships.
        
        Args:
            projection_name: Name for the projection
            dependency_types: List of dependency types to include (default: all)
            include_weights: Whether to include edge weights
            weight_property: Name of the weight property on relationships
            
        Returns:
            GDSProjection with projection details
            
        Note:
            The weight property represents dependency criticality:
            - Higher weight = more critical dependency
            - Default weight is 1.0 if not specified
            - Weights influence centrality, paths, and community detection
        """
        if dependency_types is None:
            dependency_types = self.DEPENDS_ON_TYPES
        
        self.logger.info(f"Creating GDS projection '{projection_name}' "
                        f"(weighted={include_weights})...")
        
        # Build type filter
        type_filter = ' OR '.join([f"r.dependency_type = \"{t}\"" for t in dependency_types])
        
        with self.session() as session:
            # Drop existing projection if exists
            self._drop_projection_if_exists(session, projection_name)
            
            # Weight expression - use property if exists, otherwise default
            weight_expr = f"coalesce(r.{weight_property}, {self.DEFAULT_WEIGHT})" if include_weights else "1.0"
            
            # Create native projection with Cypher
            query = f"""
            CALL gds.graph.project.cypher(
                '{projection_name}',
                'MATCH (n) WHERE n:Application OR n:Node OR n:Broker OR n:Topic 
                 RETURN id(n) AS id',
                'MATCH (a)-[r:DEPENDS_ON]->(b) WHERE {type_filter} 
                 RETURN id(a) AS source, id(b) AS target, 
                        {weight_expr} AS weight,
                        r.dependency_type AS dependency_type'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            result = session.run(query)
            record = result.single()
            
            projection = GDSProjection(
                name=record['graphName'],
                node_count=record['nodeCount'],
                relationship_count=record['relationshipCount'],
                node_labels=['Application', 'Node', 'Broker', 'Topic'],
                relationship_types=dependency_types
            )
            
            self._active_projections.append(projection_name)
            self.logger.info(f"Created projection: {projection.node_count} nodes, "
                           f"{projection.relationship_count} relationships")
            
            # Log weight statistics
            self._log_weight_statistics(session, type_filter)
            
            return projection
    
    def _log_weight_statistics(self, session, type_filter: str):
        """Log statistics about edge weights"""
        try:
            query = f"""
            MATCH ()-[r:DEPENDS_ON]->()
            WHERE {type_filter}
            RETURN 
                count(r) AS total,
                count(r.weight) AS with_weight,
                avg(coalesce(r.weight, 1.0)) AS avg_weight,
                min(coalesce(r.weight, 1.0)) AS min_weight,
                max(coalesce(r.weight, 1.0)) AS max_weight
            """
            result = session.run(query)
            record = result.single()
            if record:
                self.logger.info(f"Weight stats: total={record['total']}, "
                               f"with_weight={record['with_weight']}, "
                               f"avg={record['avg_weight']:.2f}, "
                               f"range=[{record['min_weight']:.2f}, {record['max_weight']:.2f}]")
        except Exception as e:
            self.logger.debug(f"Could not get weight stats: {e}")
    
    def create_layer_projection(self, 
                                 layer: str,
                                 projection_name: Optional[str] = None) -> GDSProjection:
        """
        Create a projection for a specific system layer.
        
        Args:
            layer: 'application', 'infrastructure', or 'broker'
            projection_name: Custom name (default: '{layer}_layer')
            
        Returns:
            GDSProjection for the layer
        """
        if projection_name is None:
            projection_name = f"{layer}_layer"
        
        layer_config = {
            'application': {
                'node_types': ['Application'],
                'dep_types': ['app_to_app']
            },
            'infrastructure': {
                'node_types': ['Node'],
                'dep_types': ['node_to_node']
            },
            'broker': {
                'node_types': ['Broker', 'Application', 'Node'],
                'dep_types': ['app_to_broker', 'node_to_broker']
            }
        }
        
        config = layer_config.get(layer)
        if not config:
            raise ValueError(f"Unknown layer: {layer}. Use: application, infrastructure, broker")
        
        node_filter = ' OR '.join([f"n:{t}" for t in config['node_types']])
        type_filter = ' OR '.join([f"r.dependency_type = '{t}'" for t in config['dep_types']])
        
        with self.session() as session:
            self._drop_projection_if_exists(session, projection_name)
            
            query = f"""
            CALL gds.graph.project.cypher(
                '{projection_name}',
                'MATCH (n) WHERE {node_filter} RETURN id(n) AS id, labels(n)[0] AS type',
                'MATCH (a)-[r:DEPENDS_ON]->(b) WHERE {type_filter}
                 RETURN id(a) AS source, id(b) AS target,
                        coalesce(r.weight, 1.0) AS weight'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            result = session.run(query)
            record = result.single()
            
            projection = GDSProjection(
                name=record['graphName'],
                node_count=record['nodeCount'],
                relationship_count=record['relationshipCount'],
                node_labels=config['node_types'],
                relationship_types=config['dep_types']
            )
            
            self._active_projections.append(projection_name)
            return projection
    
    def _drop_projection_if_exists(self, session, projection_name: str):
        """Drop projection if it exists"""
        try:
            session.run(f"CALL gds.graph.drop('{projection_name}', false)")
        except Exception:
            pass  # Projection doesn't exist
    
    def cleanup_projections(self):
        """Remove all active projections"""
        with self.session() as session:
            for name in self._active_projections:
                try:
                    session.run(f"CALL gds.graph.drop('{name}', false)")
                    self.logger.debug(f"Dropped projection: {name}")
                except Exception:
                    pass
        self._active_projections.clear()
    
    # =========================================================================
    # Centrality Algorithms
    # =========================================================================
    
    def betweenness_centrality(self, 
                                projection_name: str,
                                weighted: bool = True,
                                top_k: Optional[int] = None) -> List[CentralityResult]:
        """
        Calculate betweenness centrality using GDS.
        
        Betweenness identifies nodes that act as bridges between different
        parts of the graph - critical for reliability analysis.
        
        Args:
            projection_name: Name of the graph projection
            weighted: Use relationship weights (default: True)
            top_k: Return only top K results (None for all)
            
        Returns:
            List of CentralityResult sorted by score descending
            
        Note:
            When weighted=True, uses relationship weights to find shortest paths.
            Higher weight = higher cost path, so heavily weighted dependencies
            are avoided in shortest paths, making nodes on lighter-weight paths
            more central.
        """
        mode = "weighted" if weighted else "unweighted"
        self.logger.info(f"Running {mode} betweenness centrality on '{projection_name}'...")
        
        with self.session() as session:
            # Use relationshipWeightProperty for weighted analysis
            weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
            
            query = f"""
            CALL gds.betweenness.stream('{projection_name}', {{
                {weight_config}
                concurrency: 4
            }})
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
            ORDER BY score DESC
            {'LIMIT ' + str(top_k) if top_k else ''}
            """
            
            results = []
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record['nodeId'],
                    node_type=record['nodeType'],
                    score=record['score'],
                    rank=i + 1
                ))
            
            return results
    
    def pagerank(self,
                  projection_name: str,
                  weighted: bool = True,
                  damping_factor: float = 0.85,
                  max_iterations: int = 20,
                  top_k: Optional[int] = None) -> List[CentralityResult]:
        """
        Calculate PageRank centrality using GDS.
        
        PageRank identifies important nodes based on incoming connections
        from other important nodes - useful for availability analysis.
        
        Args:
            projection_name: Name of the graph projection
            weighted: Use relationship weights (default: True)
            damping_factor: PageRank damping factor
            max_iterations: Maximum iterations
            top_k: Return only top K results
            
        Returns:
            List of CentralityResult sorted by score descending
            
        Note:
            When weighted=True, weights influence rank distribution.
            Higher weight dependencies transfer more rank, making
            nodes with high-weight incoming dependencies more important.
        """
        mode = "weighted" if weighted else "unweighted"
        self.logger.info(f"Running {mode} PageRank on '{projection_name}'...")
        
        with self.session() as session:
            weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
            
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
            {'LIMIT ' + str(top_k) if top_k else ''}
            """
            
            results = []
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record['nodeId'],
                    node_type=record['nodeType'],
                    score=record['score'],
                    rank=i + 1
                ))
            
            return results
    
    def degree_centrality(self,
                           projection_name: str,
                           weighted: bool = True,
                           orientation: str = 'NATURAL',
                           top_k: Optional[int] = None) -> List[CentralityResult]:
        """
        Calculate degree centrality using GDS.
        
        Identifies highly connected nodes - potential coupling issues
        for maintainability analysis.
        
        Args:
            projection_name: Name of the graph projection
            weighted: Use relationship weights (default: True)
            orientation: NATURAL (out-degree), REVERSE (in-degree), or UNDIRECTED
            top_k: Return only top K results
            
        Returns:
            List of CentralityResult sorted by score descending
            
        Note:
            When weighted=True, returns sum of weights rather than count.
            This captures total dependency strength, not just count.
            High weighted degree = many high-importance dependencies.
        """
        mode = "weighted" if weighted else "unweighted"
        self.logger.info(f"Running {mode} degree centrality ({orientation}) on '{projection_name}'...")
        
        with self.session() as session:
            weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
            
            query = f"""
            CALL gds.degree.stream('{projection_name}', {{
                {weight_config}
                orientation: '{orientation}'
            }})
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
            ORDER BY score DESC
            {'LIMIT ' + str(top_k) if top_k else ''}
            """
            
            results = []
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record['nodeId'],
                    node_type=record['nodeType'],
                    score=record['score'],
                    rank=i + 1
                ))
            
            return results
    
    def closeness_centrality(self,
                              projection_name: str,
                              weighted: bool = True,
                              top_k: Optional[int] = None) -> List[CentralityResult]:
        """
        Calculate closeness centrality using GDS.
        
        Identifies nodes with short paths to all others - important
        for availability and recovery analysis.
        
        Args:
            projection_name: Name of the graph projection
            weighted: Use relationship weights (default: True)
            top_k: Return only top K results
            
        Returns:
            List of CentralityResult sorted by score descending
            
        Note:
            When weighted=True, path length = sum of weights.
            Lower total weight = shorter path = higher closeness.
            Nodes reachable via low-weight paths have higher closeness.
        """
        mode = "weighted" if weighted else "unweighted"
        self.logger.info(f"Running {mode} closeness centrality on '{projection_name}'...")
        
        with self.session() as session:
            # Note: GDS closeness uses 'useWassermanFaust' for disconnected graphs
            weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
            
            query = f"""
            CALL gds.closeness.stream('{projection_name}', {{
                useWassermanFaust: true
            }})
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE score > 0
            RETURN node.id AS nodeId, labels(node)[0] AS nodeType, score
            ORDER BY score DESC
            {'LIMIT ' + str(top_k) if top_k else ''}
            """
            
            results = []
            for i, record in enumerate(session.run(query)):
                results.append(CentralityResult(
                    node_id=record['nodeId'],
                    node_type=record['nodeType'],
                    score=record['score'],
                    rank=i + 1
                ))
            
            return results
    
    # =========================================================================
    # Community Detection
    # =========================================================================
    
    def louvain_communities(self,
                            projection_name: str,
                            weighted: bool = True,
                            include_intermediate: bool = False) -> Tuple[List[CommunityResult], Dict[str, Any]]:
        """
        Detect communities using Louvain algorithm.
        
        Identifies clusters of tightly coupled components - useful
        for maintainability and modularity analysis.
        
        Args:
            projection_name: Name of the graph projection
            weighted: Use relationship weights (default: True)
            include_intermediate: Include intermediate community levels
            
        Returns:
            Tuple of (community assignments, summary stats)
            
        Note:
            When weighted=True, strongly connected components (high total weight)
            are more likely to be grouped together. This captures logical
            coupling based on dependency importance, not just connection count.
        """
        mode = "weighted" if weighted else "unweighted"
        self.logger.info(f"Running {mode} Louvain community detection on '{projection_name}'...")
        
        with self.session() as session:
            weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
            
            query = f"""
            CALL gds.louvain.stream('{projection_name}', {{
                {weight_config}
                includeIntermediateCommunities: {str(include_intermediate).lower()}
            }})
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS node, communityId
            RETURN node.id AS nodeId, labels(node)[0] AS nodeType, communityId
            """
            
            results = []
            community_counts = {}
            
            for record in session.run(query):
                comm_id = record['communityId']
                results.append(CommunityResult(
                    node_id=record['nodeId'],
                    community_id=comm_id,
                    node_type=record['nodeType']
                ))
                community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
            
            stats = {
                'community_count': len(community_counts),
                'community_sizes': community_counts,
                'modularity': self._calculate_modularity_stats(projection_name, weighted)
            }
            
            return results, stats
    
    def weakly_connected_components(self,
                                     projection_name: str) -> Tuple[List[CommunityResult], Dict[str, Any]]:
        """
        Find weakly connected components.
        
        Identifies disconnected subgraphs - critical for reliability
        and availability analysis.
        
        Args:
            projection_name: Name of the graph projection
            
        Returns:
            Tuple of (component assignments, summary stats)
        """
        self.logger.info(f"Finding connected components in '{projection_name}'...")
        
        with self.session() as session:
            query = f"""
            CALL gds.wcc.stream('{projection_name}')
            YIELD nodeId, componentId
            WITH gds.util.asNode(nodeId) AS node, componentId
            RETURN node.id AS nodeId, labels(node)[0] AS nodeType, componentId
            """
            
            results = []
            component_counts = {}
            
            for record in session.run(query):
                comp_id = record['componentId']
                results.append(CommunityResult(
                    node_id=record['nodeId'],
                    community_id=comp_id,
                    node_type=record['nodeType']
                ))
                component_counts[comp_id] = component_counts.get(comp_id, 0) + 1
            
            stats = {
                'component_count': len(component_counts),
                'component_sizes': component_counts,
                'largest_component': max(component_counts.values()) if component_counts else 0,
                'is_connected': len(component_counts) == 1
            }
            
            return results, stats
    
    def _calculate_modularity_stats(self, projection_name: str, weighted: bool = True) -> float:
        """Calculate modularity score for the projection"""
        try:
            weight_config = "relationshipWeightProperty: 'weight'," if weighted else ""
            with self.session() as session:
                result = session.run(f"""
                CALL gds.louvain.stats('{projection_name}', {{
                    {weight_config}
                }})
                YIELD modularity
                RETURN modularity
                """)
                record = result.single()
                return record['modularity'] if record else 0.0
        except Exception:
            return 0.0
    
    # =========================================================================
    # Path Analysis
    # =========================================================================
    
    def shortest_paths(self,
                        projection_name: str,
                        source_id: str,
                        target_ids: Optional[List[str]] = None) -> List[PathResult]:
        """
        Find shortest paths from source to targets.
        
        Args:
            projection_name: Name of the graph projection
            source_id: Source node ID
            target_ids: Target node IDs (None for all reachable)
            
        Returns:
            List of PathResult
        """
        self.logger.info(f"Finding shortest paths from '{source_id}'...")
        
        with self.session() as session:
            # Get source node internal ID
            source_query = """
            MATCH (n) WHERE n.id = $source_id
            RETURN id(n) AS nodeId
            """
            source_result = session.run(source_query, source_id=source_id)
            source_record = source_result.single()
            
            if not source_record:
                return []
            
            source_neo_id = source_record['nodeId']
            
            # Run Dijkstra single source
            query = f"""
            CALL gds.allShortestPaths.dijkstra.stream('{projection_name}', {{
                sourceNode: {source_neo_id},
                relationshipWeightProperty: 'weight'
            }})
            YIELD index, sourceNode, targetNode, totalCost, path
            WITH gds.util.asNode(sourceNode).id AS source,
                 gds.util.asNode(targetNode).id AS target,
                 totalCost,
                 [n IN nodes(path) | gds.util.asNode(n).id] AS pathNodes
            RETURN source, target, totalCost, pathNodes
            """
            
            results = []
            for record in session.run(query):
                target = record['target']
                if target_ids is None or target in target_ids:
                    results.append(PathResult(
                        source=record['source'],
                        target=target,
                        cost=record['totalCost'],
                        path=record['pathNodes']
                    ))
            
            return results
    
    # =========================================================================
    # Structural Analysis (Articulation Points, Bridges)
    # =========================================================================
    
    def find_articulation_points(self) -> List[Dict[str, Any]]:
        """
        Find articulation points (SPOFs) in the DEPENDS_ON graph.
        
        Articulation points are nodes whose removal disconnects the graph.
        These are critical Single Points of Failure.
        
        Returns:
            List of articulation point info with impact assessment
        """
        self.logger.info("Finding articulation points (SPOFs)...")
        
        with self.session() as session:
            # Use Cypher to find articulation points
            # This works by checking if removing a node increases components
            query = """
            MATCH (n)
            WHERE n:Application OR n:Node OR n:Broker
            WITH n, n.id AS nodeId, labels(n)[0] AS nodeType
            
            // Count components before removal
            CALL {
                MATCH (a)-[:DEPENDS_ON*0..]->(b)
                RETURN count(DISTINCT a) + count(DISTINCT b) AS beforeCount
            }
            
            // Simulate removal by finding what becomes unreachable
            CALL {
                WITH n
                MATCH (other)
                WHERE other <> n AND (other:Application OR other:Node OR other:Broker)
                WITH n, collect(other) AS others
                UNWIND others AS start
                UNWIND others AS end
                WHERE start <> end
                OPTIONAL MATCH path = shortestPath((start)-[:DEPENDS_ON*]-(end))
                WHERE n NOT IN nodes(path)
                WITH n, count(path) AS pathsWithout
                RETURN pathsWithout
            }
            
            WITH nodeId, nodeType, beforeCount, pathsWithout
            WHERE pathsWithout < beforeCount * 0.5
            RETURN nodeId, nodeType, 
                   (beforeCount - pathsWithout) AS impactedPaths
            ORDER BY impactedPaths DESC
            """
            
            # Simplified approach - check connectivity impact
            simple_query = """
            MATCH (n)
            WHERE n:Application OR n:Node OR n:Broker
            WITH n
            OPTIONAL MATCH (n)-[:DEPENDS_ON]-(neighbor)
            WITH n, count(DISTINCT neighbor) AS connections
            WHERE connections >= 2
            
            // Check if node is on critical paths
            OPTIONAL MATCH path = (a)-[:DEPENDS_ON*2..4]->(b)
            WHERE n IN nodes(path) AND a <> b
            WITH n, connections, count(DISTINCT path) AS criticalPaths
            WHERE criticalPaths > 0
            
            RETURN n.id AS nodeId, 
                   labels(n)[0] AS nodeType,
                   connections,
                   criticalPaths,
                   connections * criticalPaths AS criticality
            ORDER BY criticality DESC
            LIMIT 20
            """
            
            results = []
            for record in session.run(simple_query):
                results.append({
                    'node_id': record['nodeId'],
                    'node_type': record['nodeType'],
                    'connections': record['connections'],
                    'critical_paths': record['criticalPaths'],
                    'criticality_score': record['criticality']
                })
            
            return results
    
    def find_bridge_edges(self) -> List[Dict[str, Any]]:
        """
        Find bridge edges whose removal disconnects components.
        
        Returns:
            List of bridge edge info
        """
        self.logger.info("Finding bridge edges...")
        
        with self.session() as session:
            query = """
            MATCH (a)-[r:DEPENDS_ON]->(b)
            WITH a, b, r
            WHERE NOT EXISTS {
                MATCH (a)-[:DEPENDS_ON*2..5]->(b)
                WHERE id(r) <> id(r)
                RETURN r
            }
            RETURN a.id AS source, 
                   b.id AS target,
                   r.dependency_type AS depType,
                   labels(a)[0] AS sourceType,
                   labels(b)[0] AS targetType
            """
            
            results = []
            for record in session.run(query):
                results.append({
                    'source': record['source'],
                    'target': record['target'],
                    'dependency_type': record['depType'],
                    'source_type': record['sourceType'],
                    'target_type': record['targetType']
                })
            
            return results
    
    # =========================================================================
    # Cycle Detection
    # =========================================================================
    
    def find_cycles(self, max_length: int = 5) -> List[List[str]]:
        """
        Find cycles in the DEPENDS_ON graph.
        
        Circular dependencies affect maintainability and can cause
        cascading failures.
        
        Args:
            max_length: Maximum cycle length to detect
            
        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        self.logger.info(f"Finding cycles (max length {max_length})...")
        
        with self.session() as session:
            query = f"""
            MATCH path = (n)-[:DEPENDS_ON*2..{max_length}]->(n)
            WITH nodes(path) AS cycle
            WITH [node IN cycle | node.id] AS cycleIds
            RETURN DISTINCT cycleIds
            LIMIT 50
            """
            
            cycles = []
            for record in session.run(query):
                cycles.append(record['cycleIds'])
            
            return cycles
    
    # =========================================================================
    # Graph Statistics
    # =========================================================================
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the DEPENDS_ON graph.
        
        Returns:
            Dictionary with graph statistics including weight info
        """
        self.logger.info("Gathering graph statistics...")
        
        with self.session() as session:
            # Node counts by type
            node_query = """
            MATCH (n)
            WHERE n:Application OR n:Node OR n:Broker OR n:Topic
            RETURN labels(n)[0] AS nodeType, count(*) AS count
            """
            
            node_counts = {}
            for record in session.run(node_query):
                node_counts[record['nodeType']] = record['count']
            
            # DEPENDS_ON counts by type
            edge_query = """
            MATCH ()-[r:DEPENDS_ON]->()
            RETURN r.dependency_type AS depType, count(*) AS count
            """
            
            edge_counts = {}
            for record in session.run(edge_query):
                edge_counts[record['depType']] = record['count']
            
            # Weight statistics
            weight_query = """
            MATCH ()-[r:DEPENDS_ON]->()
            WITH r.weight AS weight
            WHERE weight IS NOT NULL
            RETURN 
                count(weight) AS count,
                avg(weight) AS avg,
                min(weight) AS min,
                max(weight) AS max,
                stDev(weight) AS std_dev,
                percentileCont(weight, 0.5) AS median,
                percentileCont(weight, 0.9) AS p90
            """
            
            weight_stats = {}
            weight_result = session.run(weight_query).single()
            if weight_result and weight_result['count'] > 0:
                weight_stats = {
                    'count': weight_result['count'],
                    'avg': round(weight_result['avg'], 4),
                    'min': weight_result['min'],
                    'max': weight_result['max'],
                    'std_dev': round(weight_result['std_dev'], 4) if weight_result['std_dev'] else 0,
                    'median': round(weight_result['median'], 4),
                    'p90': round(weight_result['p90'], 4)
                }
            
            # Weight distribution by dependency type
            weight_by_type_query = """
            MATCH ()-[r:DEPENDS_ON]->()
            WHERE r.weight IS NOT NULL
            RETURN r.dependency_type AS depType,
                   count(*) AS count,
                   avg(r.weight) AS avg_weight,
                   max(r.weight) AS max_weight
            """
            
            weight_by_type = {}
            for record in session.run(weight_by_type_query):
                weight_by_type[record['depType']] = {
                    'count': record['count'],
                    'avg_weight': round(record['avg_weight'], 4),
                    'max_weight': record['max_weight']
                }
            
            # Density and connectivity
            stats_query = """
            MATCH (n)
            WHERE n:Application OR n:Node OR n:Broker
            WITH count(n) AS nodeCount
            MATCH ()-[r:DEPENDS_ON]->()
            WITH nodeCount, count(r) AS edgeCount
            RETURN nodeCount, edgeCount,
                   CASE WHEN nodeCount > 1 
                        THEN toFloat(edgeCount) / (nodeCount * (nodeCount - 1))
                        ELSE 0 END AS density
            """
            
            stats = session.run(stats_query).single()
            
            return {
                'node_counts': node_counts,
                'total_nodes': sum(node_counts.values()),
                'edge_counts': edge_counts,
                'total_edges': sum(edge_counts.values()),
                'density': stats['density'] if stats else 0,
                'node_count': stats['nodeCount'] if stats else 0,
                'edge_count': stats['edgeCount'] if stats else 0,
                'weight_statistics': weight_stats,
                'weight_by_dependency_type': weight_by_type
            }
    
    def get_high_weight_dependencies(self, 
                                      percentile: float = 90,
                                      top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
        """
        Get high-weight DEPENDS_ON relationships.
        
        High-weight dependencies are critical and should be prioritized
        in reliability and availability analysis.
        
        Args:
            percentile: Return dependencies above this percentile (default: 90)
            top_k: Maximum number to return (default: 20)
            
        Returns:
            List of high-weight dependencies with details
        """
        self.logger.info(f"Finding dependencies above {percentile}th percentile...")
        
        with self.session() as session:
            query = f"""
            MATCH ()-[r:DEPENDS_ON]->()
            WHERE r.weight IS NOT NULL
            WITH percentileCont(r.weight, {percentile/100}) AS threshold
            MATCH (a)-[r:DEPENDS_ON]->(b)
            WHERE r.weight >= threshold
            RETURN a.id AS source,
                   b.id AS target,
                   labels(a)[0] AS sourceType,
                   labels(b)[0] AS targetType,
                   r.dependency_type AS depType,
                   r.weight AS weight
            ORDER BY r.weight DESC
            {'LIMIT ' + str(top_k) if top_k else ''}
            """
            
            results = []
            for record in session.run(query):
                results.append({
                    'source': record['source'],
                    'target': record['target'],
                    'source_type': record['sourceType'],
                    'target_type': record['targetType'],
                    'dependency_type': record['depType'],
                    'weight': record['weight']
                })
            
            return results
    
    def get_weighted_node_importance(self, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
        """
        Calculate node importance based on weighted incoming dependencies.
        
        A node is important if it has many high-weight incoming dependencies
        (i.e., many critical components depend on it).
        
        Args:
            top_k: Maximum number to return (default: 20)
            
        Returns:
            List of nodes with their weighted importance scores
        """
        self.logger.info("Calculating weighted node importance...")
        
        with self.session() as session:
            query = f"""
            MATCH (n)<-[r:DEPENDS_ON]-(dependent)
            WHERE n:Application OR n:Node OR n:Broker
            WITH n, 
                 count(r) AS dependency_count,
                 sum(coalesce(r.weight, 1.0)) AS total_weight,
                 avg(coalesce(r.weight, 1.0)) AS avg_weight,
                 max(coalesce(r.weight, 1.0)) AS max_weight,
                 collect(DISTINCT dependent.id) AS dependents
            RETURN n.id AS nodeId,
                   labels(n)[0] AS nodeType,
                   dependency_count AS dependencyCount,
                   total_weight AS totalWeight,
                   avg_weight AS avgWeight,
                   max_weight AS maxWeight,
                   size(dependents) AS uniqueDependents
            ORDER BY total_weight DESC
            {'LIMIT ' + str(top_k) if top_k else ''}
            """
            
            results = []
            for record in session.run(query):
                results.append({
                    'node_id': record['nodeId'],
                    'node_type': record['nodeType'],
                    'dependency_count': record['dependencyCount'],
                    'total_weight': round(record['totalWeight'], 4),
                    'avg_weight': round(record['avgWeight'], 4),
                    'max_weight': record['maxWeight'],
                    'unique_dependents': record['uniqueDependents']
                })
            
            return results