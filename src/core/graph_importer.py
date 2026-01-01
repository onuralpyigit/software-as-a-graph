"""
Graph Importer - Version 5.0

Simplified Neo4j importer for pub-sub system graphs with:
- Clean batch processing for large graphs
- Schema management (constraints and indexes)
- QoS-aware DEPENDS_ON relationship derivation
- Four dependency types: app_to_app, node_to_node, app_to_broker, node_to_broker
- Component weight calculation based on relationship criticality
- Progress tracking and statistics

Weight Calculation:
    DEPENDS_ON weight = topic_count + qos_score + size_factor
    
    where:
    - topic_count = number of shared topics
    - qos_score = sum of (durability + reliability + priority) weights per topic
    - size_factor = normalized message size contribution per topic

Component Weight:
    Component weight = sum of incoming DEPENDS_ON weights + sum of outgoing DEPENDS_ON weights
    
    This captures how critical a component is in the dependency graph.

Usage:
    from src.core import GraphImporter

    with GraphImporter(uri="bolt://localhost:7687", password="secret") as importer:
        stats = importer.import_graph(graph_data)
        importer.show_analytics()

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ClientError, ServiceUnavailable

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None  # type: ignore


# =============================================================================
# Constants
# =============================================================================

DEFAULT_BATCH_SIZE = 100

# Schema definitions
CONSTRAINTS = [
    "CREATE CONSTRAINT app_id IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT broker_id IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE",
    "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX app_name IF NOT EXISTS FOR (a:Application) ON (a.name)",
    "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
    "CREATE INDEX broker_name IF NOT EXISTS FOR (b:Broker) ON (b.name)",
    "CREATE INDEX node_name IF NOT EXISTS FOR (n:Node) ON (n.name)",
    "CREATE INDEX depends_on_type IF NOT EXISTS FOR ()-[d:DEPENDS_ON]-() ON (d.dependency_type)",
    "CREATE INDEX depends_on_weight IF NOT EXISTS FOR ()-[d:DEPENDS_ON]-() ON (d.weight)",
]


# =============================================================================
# Graph Importer
# =============================================================================

class GraphImporter:
    """
    Imports pub-sub system graphs into Neo4j.
    
    Features:
    - Batch imports for performance
    - DEPENDS_ON derivation with four dependency types
    - QoS-aware weight calculation
    - Component weight calculation based on relationship weights
    - Schema management with constraints and indexes
    - Context manager protocol
    
    Example:
        with GraphImporter(password="secret") as importer:
            importer.import_graph(graph_data)
            importer.show_analytics()
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
        
        Raises:
            ImportError: If neo4j driver not installed
            ServiceUnavailable: If connection fails
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j driver not installed. pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.logger = logging.getLogger(__name__)
        self.driver = None
        self._stats: Dict[str, Any] = {}
        
        self._connect()

    def __enter__(self) -> GraphImporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
            )
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")

    # =========================================================================
    # Schema Management
    # =========================================================================

    def create_schema(self) -> None:
        """Create database schema with constraints and indexes."""
        self.logger.info("Creating schema...")
        
        with self.driver.session(database=self.database) as session:
            for constraint in CONSTRAINTS:
                try:
                    session.run(constraint)
                except ClientError:
                    pass  # Constraint may already exist
            
            for index in INDEXES:
                try:
                    session.run(index)
                except ClientError:
                    pass  # Index may already exist
        
        self.logger.info("Schema created")

    def clear_database(self) -> None:
        """Clear all data from the database."""
        self.logger.info("Clearing database...")
        
        with self.driver.session(database=self.database) as session:
            session.run("MATCH ()-[r]->() DELETE r")
            session.run("MATCH (n) DELETE n")
        
        self.logger.info("Database cleared")

    # =========================================================================
    # Main Import Method
    # =========================================================================

    def import_graph(
        self,
        graph_data: Dict,
        batch_size: int = DEFAULT_BATCH_SIZE,
        clear_first: bool = False,
        derive_dependencies: bool = True,
        calculate_weights: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, int]:
        """
        Import graph data into Neo4j.
        
        Args:
            graph_data: Dictionary with vertices and relationships
            batch_size: Number of items per batch
            clear_first: Clear database before import
            derive_dependencies: Derive DEPENDS_ON relationships
            calculate_weights: Calculate component weights after import
            progress_callback: Optional callback(stage, current, total)
        
        Returns:
            Dictionary with import statistics
        """
        start_time = time.time()
        
        if clear_first:
            self.clear_database()
        
        self.create_schema()
        
        stats = {
            "nodes": 0,
            "brokers": 0,
            "applications": 0,
            "topics": 0,
            "publishes_to": 0,
            "subscribes_to": 0,
            "routes": 0,
            "runs_on": 0,
            "connects_to": 0,
            "depends_on": 0,
        }
        
        # Import vertices
        nodes = graph_data.get("nodes", [])
        if nodes:
            self._import_nodes(nodes, batch_size)
            stats["nodes"] = len(nodes)
            if progress_callback:
                progress_callback("nodes", len(nodes), len(nodes))
        
        brokers = graph_data.get("brokers", [])
        if brokers:
            self._import_brokers(brokers, batch_size)
            stats["brokers"] = len(brokers)
            if progress_callback:
                progress_callback("brokers", len(brokers), len(brokers))
        
        applications = graph_data.get("applications", [])
        if applications:
            self._import_applications(applications, batch_size)
            stats["applications"] = len(applications)
            if progress_callback:
                progress_callback("applications", len(applications), len(applications))
        
        topics = graph_data.get("topics", [])
        if topics:
            self._import_topics(topics, batch_size)
            stats["topics"] = len(topics)
            if progress_callback:
                progress_callback("topics", len(topics), len(topics))
        
        # Import relationships
        rels = graph_data.get("relationships", {})
        
        publishes = rels.get("publishes_to", [])
        if publishes:
            self._import_relationships("PUBLISHES_TO", publishes, "Application", "Topic", batch_size)
            stats["publishes_to"] = len(publishes)
        
        subscribes = rels.get("subscribes_to", [])
        if subscribes:
            self._import_relationships("SUBSCRIBES_TO", subscribes, "Application", "Topic", batch_size)
            stats["subscribes_to"] = len(subscribes)
        
        routes = rels.get("routes", [])
        if routes:
            self._import_relationships("ROUTES", routes, "Broker", "Topic", batch_size)
            stats["routes"] = len(routes)
        
        runs_on = rels.get("runs_on", [])
        if runs_on:
            self._import_runs_on(runs_on, batch_size)
            stats["runs_on"] = len(runs_on)
        
        connects = rels.get("connects_to", [])
        if connects:
            self._import_relationships("CONNECTS_TO", connects, "Node", "Node", batch_size)
            stats["connects_to"] = len(connects)
        
        # Derive DEPENDS_ON relationships
        if derive_dependencies:
            dep_stats = self.derive_depends_on()
            stats["depends_on"] = sum(dep_stats.values())
            stats["depends_on_by_type"] = dep_stats
        
        # Calculate component weights
        if calculate_weights:
            self.calculate_component_weights()
        
        stats["duration_seconds"] = round(time.time() - start_time, 2)
        self._stats = stats
        
        return stats

    # =========================================================================
    # Vertex Import Methods
    # =========================================================================

    def _import_nodes(self, nodes: List[Dict], batch_size: int) -> None:
        """Import infrastructure nodes."""
        self.logger.info(f"Importing {len(nodes)} nodes...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                session.run(
                    """
                    UNWIND $nodes AS node
                    MERGE (n:Node {id: node.id})
                    SET n.name = coalesce(node.name, node.id),
                        n.weight = 0.0
                    """,
                    nodes=batch,
                )

    def _import_brokers(self, brokers: List[Dict], batch_size: int) -> None:
        """Import brokers."""
        self.logger.info(f"Importing {len(brokers)} brokers...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(brokers), batch_size):
                batch = brokers[i : i + batch_size]
                session.run(
                    """
                    UNWIND $brokers AS broker
                    MERGE (b:Broker {id: broker.id})
                    SET b.name = coalesce(broker.name, broker.id),
                        b.weight = 0.0
                    """,
                    brokers=batch,
                )

    def _import_applications(self, applications: List[Dict], batch_size: int) -> None:
        """Import applications."""
        self.logger.info(f"Importing {len(applications)} applications...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(applications), batch_size):
                batch = applications[i : i + batch_size]
                session.run(
                    """
                    UNWIND $apps AS app
                    MERGE (a:Application {id: app.id})
                    SET a.name = coalesce(app.name, app.id),
                        a.role = coalesce(app.role, 'pubsub'),
                        a.weight = 0.0
                    """,
                    apps=batch,
                )

    def _import_topics(self, topics: List[Dict], batch_size: int) -> None:
        """Import topics with QoS properties."""
        self.logger.info(f"Importing {len(topics)} topics...")
        
        # Normalize topic data
        normalized = []
        for t in topics:
            qos = t.get("qos", {})
            normalized.append({
                "id": t["id"],
                "name": t.get("name", t["id"]),
                "size": t.get("size", 256),
                "qos_durability": qos.get("durability", "VOLATILE"),
                "qos_reliability": qos.get("reliability", "BEST_EFFORT"),
                "qos_transport_priority": qos.get("transport_priority", "MEDIUM"),
            })
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(normalized), batch_size):
                batch = normalized[i : i + batch_size]
                session.run(
                    """
                    UNWIND $topics AS topic
                    MERGE (t:Topic {id: topic.id})
                    SET t.name = coalesce(topic.name, topic.id),
                        t.size = topic.size,
                        t.qos_durability = topic.qos_durability,
                        t.qos_reliability = topic.qos_reliability,
                        t.qos_transport_priority = topic.qos_transport_priority,
                        t.weight = CASE topic.qos_durability
                            WHEN 'PERSISTENT' THEN 0.40
                            WHEN 'TRANSIENT' THEN 0.25
                            WHEN 'TRANSIENT_LOCAL' THEN 0.20
                            ELSE 0.0
                        END +
                        CASE topic.qos_reliability
                            WHEN 'RELIABLE' THEN 0.30
                            ELSE 0.0
                        END +
                        CASE topic.qos_transport_priority
                            WHEN 'URGENT' THEN 0.30
                            WHEN 'HIGH' THEN 0.20
                            WHEN 'MEDIUM' THEN 0.10
                            ELSE 0.0
                        END
                    """,
                    topics=batch,
                )

    # =========================================================================
    # Relationship Import Methods
    # =========================================================================

    def _import_relationships(
        self,
        rel_type: str,
        relationships: List[Dict],
        source_label: str,
        target_label: str,
        batch_size: int,
    ) -> None:
        """Import relationships of a specific type."""
        self.logger.info(f"Importing {len(relationships)} {rel_type} relationships...")
        
        query = f"""
            UNWIND $rels AS rel
            MATCH (s:{source_label} {{id: rel.from}})
            MATCH (t:{target_label} {{id: rel.to}})
            MERGE (s)-[r:{rel_type}]->(t)
        """
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i : i + batch_size]
                # Normalize keys
                normalized = [
                    {"from": r.get("from", r.get("source")), "to": r.get("to", r.get("target"))}
                    for r in batch
                ]
                session.run(query, rels=normalized)

    def _import_runs_on(self, relationships: List[Dict], batch_size: int) -> None:
        """Import RUNS_ON relationships (component can be Application or Broker)."""
        self.logger.info(f"Importing {len(relationships)} RUNS_ON relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i : i + batch_size]
                normalized = [
                    {"from": r.get("from", r.get("source")), "to": r.get("to", r.get("target"))}
                    for r in batch
                ]
                
                # Try Application first, then Broker
                session.run(
                    """
                    UNWIND $rels AS rel
                    OPTIONAL MATCH (a:Application {id: rel.from})
                    OPTIONAL MATCH (b:Broker {id: rel.from})
                    MATCH (n:Node {id: rel.to})
                    FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (a)-[:RUNS_ON]->(n)
                    )
                    FOREACH (_ IN CASE WHEN b IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (b)-[:RUNS_ON]->(n)
                    )
                    """,
                    rels=normalized,
                )

    # =========================================================================
    # DEPENDS_ON Derivation
    # =========================================================================

    def derive_depends_on(self) -> Dict[str, int]:
        """
        Derive all DEPENDS_ON relationships.
        
        Returns:
            Dictionary with counts per dependency type
        """
        self.logger.info("Deriving DEPENDS_ON relationships...")
        
        stats = {
            "app_to_app": self._derive_app_to_app(),
            "app_to_broker": self._derive_app_to_broker(),
            "node_to_node": self._derive_node_to_node(),
            "node_to_broker": self._derive_node_to_broker(),
        }
        
        total = sum(stats.values())
        self.logger.info(f"Derived {total} DEPENDS_ON relationships")
        
        return stats

    def _derive_app_to_app(self) -> int:
        """
        Derive APP_TO_APP dependencies.
        
        Rule: subscriber DEPENDS_ON publisher if they share topics.
        Weight = topic_count + sum(qos_scores) + sum(size_factors)
        """
        query = """
            // Find publisher-subscriber pairs through shared topics
            MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
            WHERE pub <> sub
            
            // Calculate per-topic contributions
            WITH pub, sub, t,
                // QoS score
                CASE t.qos_durability
                    WHEN 'PERSISTENT' THEN 0.40
                    WHEN 'TRANSIENT' THEN 0.25
                    WHEN 'TRANSIENT_LOCAL' THEN 0.20
                    ELSE 0.0
                END +
                CASE t.qos_reliability
                    WHEN 'RELIABLE' THEN 0.30
                    ELSE 0.0
                END +
                CASE t.qos_transport_priority
                    WHEN 'URGENT' THEN 0.30
                    WHEN 'HIGH' THEN 0.20
                    WHEN 'MEDIUM' THEN 0.10
                    ELSE 0.0
                END AS qos_score,
                // Size factor (normalized, capped at 0.5)
                CASE 
                    WHEN t.size IS NULL THEN 0.025
                    ELSE toFloat(t.size) / 10000.0
                END AS size_factor
            
            // Aggregate per pair
            WITH sub, pub,
                 collect(DISTINCT t.id) AS topics,
                 sum(qos_score) AS total_qos,
                 sum(CASE WHEN size_factor > 0.5 THEN 0.5 ELSE size_factor END) AS total_size
            
            // Calculate weight
            WITH sub, pub, topics,
                 size(topics) + total_qos + total_size AS weight
            
            // Create relationship
            MERGE (sub)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(pub)
            SET d.via_topics = topics,
                d.weight = weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    def _derive_app_to_broker(self) -> int:
        """
        Derive APP_TO_BROKER dependencies.
        
        Rule: app DEPENDS_ON broker that routes topics the app uses.
        Weight = topic_count + sum(qos_scores) + sum(size_factors)
        """
        query = """
            // Find apps and brokers through topics
            MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
            
            // Calculate per-topic contributions
            WITH app, broker, t,
                CASE t.qos_durability
                    WHEN 'PERSISTENT' THEN 0.40
                    WHEN 'TRANSIENT' THEN 0.25
                    WHEN 'TRANSIENT_LOCAL' THEN 0.20
                    ELSE 0.0
                END +
                CASE t.qos_reliability
                    WHEN 'RELIABLE' THEN 0.30
                    ELSE 0.0
                END +
                CASE t.qos_transport_priority
                    WHEN 'URGENT' THEN 0.30
                    WHEN 'HIGH' THEN 0.20
                    WHEN 'MEDIUM' THEN 0.10
                    ELSE 0.0
                END AS qos_score,
                CASE 
                    WHEN t.size IS NULL THEN 0.025
                    ELSE toFloat(t.size) / 10000.0
                END AS size_factor
            
            // Aggregate
            WITH app, broker,
                 collect(DISTINCT t.id) AS topics,
                 sum(qos_score) AS total_qos,
                 sum(CASE WHEN size_factor > 0.5 THEN 0.5 ELSE size_factor END) AS total_size
            
            WITH app, broker, topics,
                 size(topics) + total_qos + total_size AS weight
            
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
            SET d.via_topics = topics,
                d.weight = weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    def _derive_node_to_node(self) -> int:
        """
        Derive NODE_TO_NODE dependencies.
        
        Rule: node_A DEPENDS_ON node_B if app on node_A depends on app on node_B.
        Weight = sum of underlying app_to_app weights.
        """
        query = """
            // Find app dependencies that cross nodes
            MATCH (app1:Application)-[:RUNS_ON]->(n1:Node)
            MATCH (app2:Application)-[:RUNS_ON]->(n2:Node)
            MATCH (app1)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(app2)
            WHERE n1 <> n2
            
            // Aggregate by node pair
            WITH n1, n2,
                 collect(DISTINCT {app1: app1.id, app2: app2.id}) AS app_pairs,
                 sum(d.weight) AS total_weight
            
            MERGE (n1)-[dep:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
            SET dep.via_apps = [p IN app_pairs | p.app1 + '->' + p.app2],
                dep.weight = total_weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    def _derive_node_to_broker(self) -> int:
        """
        Derive NODE_TO_BROKER dependencies.
        
        Rule: node DEPENDS_ON broker if any app on node depends on that broker.
        Weight = sum of underlying app_to_broker weights.
        """
        query = """
            // Find node-to-broker dependencies through apps
            MATCH (app:Application)-[:RUNS_ON]->(n:Node)
            MATCH (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker)
            
            // Aggregate
            WITH n, broker,
                 collect(DISTINCT app.id) AS apps,
                 sum(d.weight) AS total_weight
            
            MERGE (n)-[dep:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(broker)
            SET dep.via_apps = apps,
                dep.weight = total_weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    # =========================================================================
    # Component Weight Calculation
    # =========================================================================

    def calculate_component_weights(self) -> None:
        """
        Calculate and update component weights based on DEPENDS_ON relationships.
        
        Weight formula:
            weight(v) = sum(incoming DEPENDS_ON weights) + sum(outgoing DEPENDS_ON weights)
        
        This captures the criticality of each component in the dependency graph.
        Components with many high-weight dependencies are more critical.
        """
        self.logger.info("Calculating component weights...")
        
        # Update Application weights
        self._update_component_weights("Application")
        
        # Update Broker weights
        self._update_component_weights("Broker")
        
        # Update Node weights
        self._update_component_weights("Node")
        
        self.logger.info("Component weights calculated")

    def _update_component_weights(self, label: str) -> None:
        """Update weights for components of a specific type."""
        query = f"""
            MATCH (c:{label})
            OPTIONAL MATCH (c)-[out:DEPENDS_ON]->()
            OPTIONAL MATCH ()-[in:DEPENDS_ON]->(c)
            WITH c,
                 coalesce(sum(out.weight), 0) AS out_weight,
                 coalesce(sum(in.weight), 0) AS in_weight
            SET c.weight = out_weight + in_weight
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query)

    # =========================================================================
    # Statistics and Analytics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get import statistics."""
        return self._stats

    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics."""
        with self.driver.session(database=self.database) as session:
            # Node counts
            node_result = session.run("""
                MATCH (n)
                WITH labels(n)[0] AS label, count(*) AS count
                RETURN collect({label: label, count: count}) AS nodes
            """)
            node_counts = {
                item["label"]: item["count"] 
                for item in node_result.single()["nodes"]
            }
            
            # Relationship counts
            rel_result = session.run("""
                MATCH ()-[r]->()
                WITH type(r) AS rel_type, count(*) AS count
                RETURN collect({type: rel_type, count: count}) AS relationships
            """)
            rel_counts = {
                item["type"]: item["count"]
                for item in rel_result.single()["relationships"]
            }
            
            # DEPENDS_ON by type
            dep_result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                WITH d.dependency_type AS dep_type, count(*) AS count
                RETURN collect({type: dep_type, count: count}) AS depends_on
            """)
            dep_counts = {
                item["type"]: item["count"]
                for item in dep_result.single()["depends_on"]
            }
            
            # Weight statistics
            weight_result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                RETURN 
                    round(avg(d.weight) * 100) / 100 AS avg_weight,
                    round(min(d.weight) * 100) / 100 AS min_weight,
                    round(max(d.weight) * 100) / 100 AS max_weight,
                    round(sum(d.weight) * 100) / 100 AS total_weight
            """)
            weight_stats = weight_result.single()
            
            # Component weight statistics
            comp_weight_result = session.run("""
                MATCH (c)
                WHERE c:Application OR c:Broker OR c:Node
                WITH labels(c)[0] AS label,
                     round(avg(c.weight) * 100) / 100 AS avg_weight,
                     round(max(c.weight) * 100) / 100 AS max_weight
                RETURN collect({label: label, avg: avg_weight, max: max_weight}) AS components
            """)
            comp_weight_stats = {
                item["label"]: {"avg": item["avg"], "max": item["max"]}
                for item in comp_weight_result.single()["components"]
            }
            
            return {
                "nodes": node_counts,
                "relationships": rel_counts,
                "depends_on_by_type": dep_counts,
                "weight_statistics": {
                    "average": weight_stats["avg_weight"],
                    "minimum": weight_stats["min_weight"],
                    "maximum": weight_stats["max_weight"],
                    "total": weight_stats["total_weight"],
                },
                "component_weights": comp_weight_stats,
            }

    def show_analytics(self) -> None:
        """Print sample analytics queries."""
        queries = [
            ("Top Components by Weight", """
                MATCH (c)
                WHERE c:Application OR c:Broker OR c:Node
                RETURN labels(c)[0] AS type, c.name AS component, 
                       round(c.weight * 100) / 100 AS weight
                ORDER BY c.weight DESC
                LIMIT 10
            """),
            ("Most Depended-Upon Components", """
                MATCH ()-[d:DEPENDS_ON]->(target)
                WITH target, count(d) AS dependents, 
                     round(sum(d.weight) * 100) / 100 AS total_weight
                RETURN labels(target)[0] AS type, target.name AS component,
                       dependents, total_weight
                ORDER BY dependents DESC
                LIMIT 10
            """),
            ("DEPENDS_ON Distribution by Type", """
                MATCH ()-[d:DEPENDS_ON]->()
                RETURN d.dependency_type AS type,
                       count(*) AS count,
                       round(avg(d.weight) * 100) / 100 AS avg_weight
                ORDER BY count DESC
            """),
            ("Critical Dependencies (High Weight)", """
                MATCH (a)-[d:DEPENDS_ON]->(b)
                WHERE d.weight > 5.0
                RETURN labels(a)[0] AS source_type, a.name AS source,
                       labels(b)[0] AS target_type, b.name AS target,
                       d.dependency_type AS type, 
                       round(d.weight * 100) / 100 AS weight
                ORDER BY d.weight DESC
                LIMIT 10
            """),
        ]
        
        with self.driver.session(database=self.database) as session:
            for title, query in queries:
                print(f"\n{title}:")
                print("-" * 60)
                try:
                    result = session.run(query)
                    records = list(result)
                    if records:
                        keys = records[0].keys()
                        header = " | ".join(f"{k:20s}" for k in keys)
                        print(header)
                        print("-" * len(header))
                        for record in records:
                            row = " | ".join(f"{str(record[k]):20s}" for k in keys)
                            print(row)
                    else:
                        print("  No results")
                except Exception as e:
                    print(f"  Error: {e}")

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_cypher_queries(self, filepath: str) -> None:
        """Export useful Cypher queries to a file."""
        queries = '''// Software-as-a-Graph: Useful Cypher Queries
// ============================================

// --- Basic Statistics ---

// Count all components
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count ORDER BY count DESC;

// Count all relationships
MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC;

// --- Component Criticality ---

// Top components by weight (criticality)
MATCH (c)
WHERE c:Application OR c:Broker OR c:Node
RETURN labels(c)[0] AS type, c.name AS component, round(c.weight * 100) / 100 AS weight
ORDER BY c.weight DESC LIMIT 20;

// Most depended-upon components (potential single points of failure)
MATCH ()-[d:DEPENDS_ON]->(target)
WITH target, count(d) AS dependents, round(sum(d.weight) * 100) / 100 AS total_weight
RETURN labels(target)[0] AS type, target.name AS component, dependents, total_weight
ORDER BY dependents DESC LIMIT 20;

// --- DEPENDS_ON Analysis ---

// DEPENDS_ON distribution by type
MATCH ()-[d:DEPENDS_ON]->()
RETURN d.dependency_type AS type, count(*) AS count, 
       round(avg(d.weight) * 100) / 100 AS avg_weight,
       round(max(d.weight) * 100) / 100 AS max_weight
ORDER BY count DESC;

// High-weight dependencies (critical paths)
MATCH (a)-[d:DEPENDS_ON]->(b)
WHERE d.weight > 5.0
RETURN a.name AS source, b.name AS target, d.dependency_type AS type, 
       round(d.weight * 100) / 100 AS weight
ORDER BY d.weight DESC LIMIT 20;

// Dependency chains (multi-hop)
MATCH path = (a:Application)-[:DEPENDS_ON*1..3]->(b)
WHERE a <> b
RETURN [n IN nodes(path) | n.name] AS chain, length(path) AS depth
ORDER BY depth DESC LIMIT 20;

// --- Multi-Layer Analysis ---

// Application layer dependencies
MATCH (a1:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
RETURN a1.name AS subscriber, a2.name AS publisher, 
       d.via_topics AS topics, round(d.weight * 100) / 100 AS weight
ORDER BY d.weight DESC LIMIT 20;

// Infrastructure layer dependencies
MATCH (n1:Node)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2:Node)
RETURN n1.name AS source_node, n2.name AS target_node, 
       round(d.weight * 100) / 100 AS weight
ORDER BY d.weight DESC;

// Application-to-Broker dependencies
MATCH (a:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(b:Broker)
RETURN a.name AS application, b.name AS broker, 
       d.via_topics AS topics, round(d.weight * 100) / 100 AS weight
ORDER BY d.weight DESC LIMIT 20;

// --- QoS Analysis ---

// Topics by QoS criticality
MATCH (t:Topic)
RETURN t.name AS topic, 
       t.qos_durability AS durability,
       t.qos_reliability AS reliability, 
       t.qos_transport_priority AS priority,
       round(t.weight * 100) / 100 AS qos_weight
ORDER BY t.weight DESC LIMIT 20;

// Critical topics (high QoS requirements)
MATCH (t:Topic)
WHERE t.qos_reliability = 'RELIABLE' OR t.qos_durability = 'PERSISTENT'
RETURN t.name AS topic, t.qos_durability AS durability, 
       t.qos_reliability AS reliability
ORDER BY t.name;

// --- Infrastructure Analysis ---

// Node load (components per node)
MATCH (c)-[:RUNS_ON]->(n:Node)
WITH n, count(c) AS component_count, collect(c.name) AS components
RETURN n.name AS node, component_count, n.weight AS criticality_weight
ORDER BY component_count DESC;

// Cross-node communication
MATCH (n1:Node)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2:Node)
RETURN n1.name AS source, n2.name AS target, round(d.weight * 100) / 100 AS weight
ORDER BY d.weight DESC;

// --- Single Points of Failure ---

// Brokers that are single points of failure
MATCH (t:Topic)<-[:ROUTES]-(b:Broker)
WITH t, collect(b) AS brokers
WHERE size(brokers) = 1
RETURN t.name AS topic, brokers[0].name AS single_broker, t.qos_reliability AS reliability;

// Nodes with high concentration of critical apps
MATCH (a:Application)-[:RUNS_ON]->(n:Node)
WITH n, collect(a) AS apps, sum(a.weight) AS total_app_weight
WHERE size(apps) > 5
RETURN n.name AS node, size(apps) AS app_count, 
       round(total_app_weight * 100) / 100 AS total_weight
ORDER BY total_weight DESC;
'''
        with open(filepath, "w") as f:
            f.write(queries)
        
        self.logger.info(f"Cypher queries exported to {filepath}")

    def export_graph(self, filepath: str) -> None:
        """Export graph from Neo4j to JSON."""
        with self.driver.session(database=self.database) as session:
            # Export all data
            graph = {
                "metadata": {
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": "neo4j",
                    "database": self.database,
                },
                "nodes": [],
                "brokers": [],
                "applications": [],
                "topics": [],
                "relationships": {
                    "publishes_to": [],
                    "subscribes_to": [],
                    "routes": [],
                    "runs_on": [],
                    "connects_to": [],
                },
                "depends_on": [],
            }
            
            # Export nodes
            for r in session.run("MATCH (n:Node) RETURN n"):
                node = r["n"]
                graph["nodes"].append({
                    "id": node["id"],
                    "name": node["name"],
                    "weight": node.get("weight", 0.0),
                })
            
            # Export brokers
            for r in session.run("MATCH (b:Broker) RETURN b"):
                broker = r["b"]
                graph["brokers"].append({
                    "id": broker["id"],
                    "name": broker["name"],
                    "weight": broker.get("weight", 0.0),
                })
            
            # Export applications
            for r in session.run("MATCH (a:Application) RETURN a"):
                app = r["a"]
                graph["applications"].append({
                    "id": app["id"],
                    "name": app["name"],
                    "role": app.get("role", "pubsub"),
                    "weight": app.get("weight", 0.0),
                })
            
            # Export topics
            for r in session.run("MATCH (t:Topic) RETURN t"):
                topic = r["t"]
                graph["topics"].append({
                    "id": topic["id"],
                    "name": topic["name"],
                    "size": topic.get("size", 256),
                    "weight": topic.get("weight", 0.0),
                    "qos": {
                        "durability": topic.get("qos_durability", "VOLATILE"),
                        "reliability": topic.get("qos_reliability", "BEST_EFFORT"),
                        "transport_priority": topic.get("qos_transport_priority", "MEDIUM"),
                    },
                })
            
            # Export relationships
            rel_queries = {
                "publishes_to": "MATCH (a:Application)-[:PUBLISHES_TO]->(t:Topic) RETURN a.id AS `from`, t.id AS `to`",
                "subscribes_to": "MATCH (a:Application)-[:SUBSCRIBES_TO]->(t:Topic) RETURN a.id AS `from`, t.id AS `to`",
                "routes": "MATCH (b:Broker)-[:ROUTES]->(t:Topic) RETURN b.id AS `from`, t.id AS `to`",
                "runs_on": "MATCH (c)-[:RUNS_ON]->(n:Node) RETURN c.id AS `from`, n.id AS `to`",
                "connects_to": "MATCH (n1:Node)-[:CONNECTS_TO]->(n2:Node) RETURN n1.id AS `from`, n2.id AS `to`",
            }
            
            for key, query in rel_queries.items():
                for r in session.run(query):
                    graph["relationships"][key].append(dict(r))
            
            # Export DEPENDS_ON
            for r in session.run("""
                MATCH (a)-[d:DEPENDS_ON]->(b)
                RETURN a.id AS `from`, b.id AS `to`, 
                       d.dependency_type AS dependency_type,
                       d.weight AS weight,
                       d.via_topics AS via_topics,
                       d.via_apps AS via_apps
            """):
                graph["depends_on"].append({
                    "from": r["from"],
                    "to": r["to"],
                    "dependency_type": r["dependency_type"],
                    "weight": r["weight"],
                    "via_topics": r.get("via_topics"),
                    "via_apps": r.get("via_apps"),
                })
            
            with open(filepath, "w") as f:
                json.dump(graph, f, indent=2)
            
            self.logger.info(f"Graph exported to {filepath}")