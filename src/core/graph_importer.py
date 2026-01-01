"""
Graph Importer - Refactored Version 4.0

Imports pub-sub system graphs into Neo4j with:
- Batch processing for large graphs
- Schema management (constraints and indexes)
- QoS-aware DEPENDS_ON relationship derivation
- Four dependency types: app_to_app, node_to_node, app_to_broker, node_to_broker
- Weight calculation based on topic QoS and message size
- Progress tracking and statistics
- Export capabilities

Weight Calculation:
    DEPENDS_ON weight = base_weight + qos_score + size_factor
    
    where:
    - base_weight = number of shared topics
    - qos_score = sum of (durability + reliability + transport_priority) weights
    - size_factor = normalized message size contribution

Usage:
    from src.core.graph_importer import GraphImporter

    # Basic usage
    with GraphImporter(uri="bolt://localhost:7687", password="secret") as importer:
        importer.import_graph(graph_data)
        stats = importer.get_statistics()

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError, TransientError
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None


class GraphImporter:
    """
    Imports pub-sub system graphs into Neo4j.
    
    Supports:
    - Batch imports for performance
    - Unified DEPENDS_ON derivation with four dependency types
    - QoS-aware weight calculation
    - Schema management with constraints and indexes
    - Statistics and validation
    - Context manager protocol
    """

    # Default batch size for imports
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
    ]

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j driver not installed. Install with: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.logger = logging.getLogger(__name__)
        self.driver = None
        self._stats = {}
        
        self._connect()

    def __enter__(self) -> GraphImporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _connect(self) -> None:
        """Establish connection to Neo4j"""
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
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")

    # =========================================================================
    # Schema Management
    # =========================================================================

    def create_schema(self) -> None:
        """Create database schema with constraints and indexes"""
        self.logger.info("Creating schema...")
        
        with self.driver.session(database=self.database) as session:
            for constraint in self.CONSTRAINTS:
                try:
                    session.run(constraint)
                except ClientError:
                    pass  # Constraint may already exist
            
            for index in self.INDEXES:
                try:
                    session.run(index)
                except ClientError:
                    pass  # Index may already exist
        
        self.logger.info("Schema created")

    def clear_database(self) -> None:
        """Clear all data from the database"""
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
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, int]:
        """
        Import graph data into Neo4j.
        
        Args:
            graph_data: Graph dictionary with vertices and relationships
            batch_size: Batch size for bulk imports
            clear_first: Clear database before import
            derive_dependencies: Derive DEPENDS_ON relationships
            progress_callback: Optional callback(step, current, total)
        
        Returns:
            Dictionary with import counts
        """
        self.logger.info("Starting graph import...")
        start_time = time.time()
        
        if clear_first:
            self.clear_database()
        
        self.create_schema()
        
        # Import vertices
        counts = {}
        
        vertices = [
            ("nodes", graph_data.get("nodes", []), self._import_nodes),
            ("brokers", graph_data.get("brokers", []), self._import_brokers),
            ("topics", graph_data.get("topics", []), self._import_topics),
            ("applications", graph_data.get("applications", []), self._import_applications),
        ]
        
        for name, data, import_fn in vertices:
            if data:
                import_fn(data, batch_size)
                counts[name] = len(data)
                if progress_callback:
                    progress_callback(f"Imported {name}", len(data), len(data))
        
        # Import relationships
        relationships = graph_data.get("relationships", {})
        rel_imports = [
            ("runs_on", relationships.get("runs_on", []), self._import_runs_on),
            ("connects_to", relationships.get("connects_to", []), self._import_connects_to),
            ("routes", relationships.get("routes", []), self._import_routes),
            ("publishes_to", relationships.get("publishes_to", []), self._import_publishes_to),
            ("subscribes_to", relationships.get("subscribes_to", []), self._import_subscribes_to),
        ]
        
        for name, data, import_fn in rel_imports:
            if data:
                import_fn(data, batch_size)
                counts[name] = len(data)
                if progress_callback:
                    progress_callback(f"Imported {name}", len(data), len(data))
        
        # Derive DEPENDS_ON relationships
        if derive_dependencies:
            depends_on_counts = self._derive_all_dependencies()
            counts["depends_on"] = depends_on_counts
            if progress_callback:
                progress_callback("Derived DEPENDS_ON", sum(depends_on_counts.values()), sum(depends_on_counts.values()))
        
        elapsed = time.time() - start_time
        self.logger.info(f"Import completed in {elapsed:.2f}s")
        
        self._stats = counts
        return counts

    # =========================================================================
    # Vertex Import Methods
    # =========================================================================

    def _import_nodes(self, nodes: List[Dict], batch_size: int) -> None:
        """Import infrastructure nodes"""
        self.logger.info(f"Importing {len(nodes)} nodes...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                session.run("""
                    UNWIND $nodes AS node
                    MERGE (n:Node {id: node.id})
                    SET n.name = coalesce(node.name, node.id)
                """, nodes=batch)

    def _import_brokers(self, brokers: List[Dict], batch_size: int) -> None:
        """Import message brokers"""
        self.logger.info(f"Importing {len(brokers)} brokers...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(brokers), batch_size):
                batch = brokers[i:i + batch_size]
                session.run("""
                    UNWIND $brokers AS broker
                    MERGE (b:Broker {id: broker.id})
                    SET b.name = coalesce(broker.name, broker.id)
                """, brokers=batch)

    def _import_topics(self, topics: List[Dict], batch_size: int) -> None:
        """Import topics with QoS settings"""
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
                batch = normalized[i:i + batch_size]
                session.run("""
                    UNWIND $topics AS topic
                    MERGE (t:Topic {id: topic.id})
                    SET t.name = coalesce(topic.name, topic.id),
                        t.size = topic.size,
                        t.qos_durability = topic.qos_durability,
                        t.qos_reliability = topic.qos_reliability,
                        t.qos_transport_priority = topic.qos_transport_priority
                """, topics=batch)

    def _import_applications(self, applications: List[Dict], batch_size: int) -> None:
        """Import applications"""
        self.logger.info(f"Importing {len(applications)} applications...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(applications), batch_size):
                batch = applications[i:i + batch_size]
                session.run("""
                    UNWIND $apps AS app
                    MERGE (a:Application {id: app.id})
                    SET a.name = coalesce(app.name, app.id),
                        a.role = coalesce(app.role, 'pubsub')
                """, apps=batch)

    # =========================================================================
    # Relationship Import Methods
    # =========================================================================

    def _normalize_relationship(self, rel: Dict) -> Dict:
        """Normalize relationship field names"""
        return {
            "from": rel.get("from", rel.get("source")),
            "to": rel.get("to", rel.get("target")),
        }

    def _import_runs_on(self, relationships: List[Dict], batch_size: int) -> None:
        """Import RUNS_ON relationships"""
        self.logger.info(f"Importing {len(relationships)} RUNS_ON relationships...")
        rels = [self._normalize_relationship(r) for r in relationships]
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (component {id: rel.from})
                    MATCH (node:Node {id: rel.to})
                    MERGE (component)-[:RUNS_ON]->(node)
                """, rels=batch)

    def _import_connects_to(self, relationships: List[Dict], batch_size: int) -> None:
        """Import CONNECTS_TO relationships"""
        self.logger.info(f"Importing {len(relationships)} CONNECTS_TO relationships...")
        rels = [self._normalize_relationship(r) for r in relationships]
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (n1:Node {id: rel.from})
                    MATCH (n2:Node {id: rel.to})
                    MERGE (n1)-[:CONNECTS_TO]->(n2)
                """, rels=batch)

    def _import_routes(self, relationships: List[Dict], batch_size: int) -> None:
        """Import ROUTES relationships"""
        self.logger.info(f"Importing {len(relationships)} ROUTES relationships...")
        rels = [self._normalize_relationship(r) for r in relationships]
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (b:Broker {id: rel.from})
                    MATCH (t:Topic {id: rel.to})
                    MERGE (b)-[:ROUTES]->(t)
                """, rels=batch)

    def _import_publishes_to(self, relationships: List[Dict], batch_size: int) -> None:
        """Import PUBLISHES_TO relationships"""
        self.logger.info(f"Importing {len(relationships)} PUBLISHES_TO relationships...")
        rels = [self._normalize_relationship(r) for r in relationships]
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (a:Application {id: rel.from})
                    MATCH (t:Topic {id: rel.to})
                    MERGE (a)-[:PUBLISHES_TO]->(t)
                """, rels=batch)

    def _import_subscribes_to(self, relationships: List[Dict], batch_size: int) -> None:
        """Import SUBSCRIBES_TO relationships"""
        self.logger.info(f"Importing {len(relationships)} SUBSCRIBES_TO relationships...")
        rels = [self._normalize_relationship(r) for r in relationships]
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (a:Application {id: rel.from})
                    MATCH (t:Topic {id: rel.to})
                    MERGE (a)-[:SUBSCRIBES_TO]->(t)
                """, rels=batch)

    # =========================================================================
    # DEPENDS_ON Derivation
    # =========================================================================

    def _derive_all_dependencies(self) -> Dict[str, int]:
        """
        Derive all DEPENDS_ON relationship types.
        
        Returns:
            Dictionary with counts per dependency type
        """
        self.logger.info("Deriving DEPENDS_ON relationships...")
        
        counts = {
            "app_to_app": self._derive_app_to_app(),
            "app_to_broker": self._derive_app_to_broker(),
            "node_to_node": self._derive_node_to_node(),
            "node_to_broker": self._derive_node_to_broker(),
        }
        
        total = sum(counts.values())
        self.logger.info(f"Derived {total} DEPENDS_ON relationships: {counts}")
        
        return counts

    def _derive_app_to_app(self) -> int:
        """
        Derive APP_TO_APP dependencies.
        
        Rule: subscriber DEPENDS_ON publisher if they share topics.
        
        Weight = count(shared_topics) + sum(qos_scores) + sum(size_factors)
        
        QoS score components:
        - PERSISTENT durability: +0.40
        - TRANSIENT durability: +0.25
        - TRANSIENT_LOCAL durability: +0.20
        - RELIABLE reliability: +0.30
        - URGENT transport_priority: +0.30
        - HIGH transport_priority: +0.20
        - MEDIUM transport_priority: +0.10
        
        Size factor: min(size / 10000, 1.0) per topic
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find publisher-subscriber pairs through shared topics
                MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)
                      <-[:SUBSCRIBES_TO]-(sub:Application)
                WHERE pub <> sub
                
                // Calculate QoS criticality score for each topic
                WITH pub, sub, t,
                    // Durability contribution
                    CASE t.qos_durability
                        WHEN 'PERSISTENT' THEN 0.40
                        WHEN 'TRANSIENT' THEN 0.25
                        WHEN 'TRANSIENT_LOCAL' THEN 0.20
                        ELSE 0.0
                    END +
                    // Reliability contribution
                    CASE t.qos_reliability
                        WHEN 'RELIABLE' THEN 0.30
                        ELSE 0.0
                    END +
                    // Priority contribution
                    CASE t.qos_transport_priority
                        WHEN 'URGENT' THEN 0.30
                        WHEN 'HIGH' THEN 0.20
                        WHEN 'MEDIUM' THEN 0.10
                        ELSE 0.0
                    END AS qos_score,
                    // Size contribution (normalized, capped at 0.5)
                    CASE 
                        WHEN t.size IS NULL THEN 0.025
                        ELSE toFloat(t.size) / 10000.0
                    END AS size_factor
                
                // Aggregate per publisher-subscriber pair
                WITH sub, pub,
                     collect(DISTINCT t.id) AS topics,
                     sum(qos_score) AS total_qos,
                     sum(CASE WHEN size_factor > 1.0 THEN 1.0 ELSE size_factor END) AS total_size
                
                // Calculate final weight: topic_count + qos_scores + size_factors
                WITH sub, pub, topics,
                     size(topics) + total_qos + total_size AS weight
                
                // Create DEPENDS_ON: subscriber depends on publisher
                MERGE (sub)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(pub)
                SET d.via_topics = topics,
                    d.weight = weight
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]

    def _derive_app_to_broker(self) -> int:
        """
        Derive APP_TO_BROKER dependencies.
        
        Rule: app DEPENDS_ON broker that routes topics the app uses.
        
        Weight = count(topics) + sum(qos_scores) + sum(size_factors)
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find apps and the brokers routing their topics
                MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
                      <-[:ROUTES]-(broker:Broker)
                
                // Calculate QoS score
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
                     sum(CASE WHEN size_factor > 1.0 THEN 1.0 ELSE size_factor END) AS total_size
                
                WITH app, broker, topics,
                     size(topics) + total_qos + total_size AS weight
                
                MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
                SET d.via_topics = topics,
                    d.weight = weight
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]

    def _derive_node_to_node(self) -> int:
        """
        Derive NODE_TO_NODE dependencies.
        
        Rule: node_X DEPENDS_ON node_Y if app on node_X depends on app on node_Y.
        
        Weight aggregates the weights of underlying app-to-app dependencies.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find nodes through app-to-app dependencies
                MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)
                      -[dep:DEPENDS_ON {dependency_type: 'app_to_app'}]->
                      (a2:Application)-[:RUNS_ON]->(n2:Node)
                WHERE n1 <> n2
                
                // Aggregate dependencies per node pair
                WITH n1, n2,
                     collect(DISTINCT a1.id) AS dependent_apps,
                     collect(DISTINCT a2.id) AS dependency_apps,
                     sum(dep.weight) AS total_weight
                
                // Weight = app_count + aggregated_app_weights
                WITH n1, n2, dependent_apps, dependency_apps,
                     size(dependent_apps) + total_weight AS weight
                
                MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
                SET d.via_apps = dependent_apps,
                    d.target_apps = dependency_apps,
                    d.weight = weight
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]

    def _derive_node_to_broker(self) -> int:
        """
        Derive NODE_TO_BROKER dependencies.
        
        Rule: node DEPENDS_ON broker if apps on node depend on that broker.
        
        Weight aggregates the weights of underlying app-to-broker dependencies.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find nodes that depend on brokers through their apps
                MATCH (n:Node)<-[:RUNS_ON]-(app:Application)
                      -[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker)
                
                // Aggregate
                WITH n, broker,
                     collect(DISTINCT app.id) AS dependent_apps,
                     sum(dep.weight) AS total_weight
                
                WITH n, broker, dependent_apps,
                     size(dependent_apps) + total_weight AS weight
                
                MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(broker)
                SET d.via_apps = dependent_apps,
                    d.weight = weight
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]

    # =========================================================================
    # Statistics and Queries
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics from Neo4j"""
        with self.driver.session(database=self.database) as session:
            # Node counts
            node_result = session.run("""
                MATCH (n)
                WITH labels(n)[0] AS label, count(*) AS count
                RETURN collect({label: label, count: count}) AS nodes
            """)
            node_counts = {item["label"]: item["count"] for item in node_result.single()["nodes"]}
            
            # Relationship counts
            rel_result = session.run("""
                MATCH ()-[r]->()
                WITH type(r) AS rel_type, count(*) AS count
                RETURN collect({type: rel_type, count: count}) AS relationships
            """)
            rel_counts = {item["type"]: item["count"] for item in rel_result.single()["relationships"]}
            
            # DEPENDS_ON by type
            dep_result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                WITH d.dependency_type AS dep_type, count(*) AS count
                RETURN collect({type: dep_type, count: count}) AS depends_on
            """)
            dep_counts = {item["type"]: item["count"] for item in dep_result.single()["depends_on"]}
            
            # Weight statistics
            weight_result = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                RETURN 
                    avg(d.weight) AS avg_weight,
                    min(d.weight) AS min_weight,
                    max(d.weight) AS max_weight
            """)
            weight_stats = weight_result.single()
            
            return {
                "nodes": node_counts,
                "relationships": rel_counts,
                "depends_on_by_type": dep_counts,
                "weight_statistics": {
                    "average": weight_stats["avg_weight"],
                    "minimum": weight_stats["min_weight"],
                    "maximum": weight_stats["max_weight"],
                },
            }

    def run_sample_queries(self) -> None:
        """Execute and print sample queries"""
        queries = [
            ("Top 10 Applications by Dependency Weight", """
                MATCH (a:Application)-[d:DEPENDS_ON]->()
                WITH a, sum(d.weight) AS total_weight, count(d) AS dep_count
                RETURN a.name AS application, round(total_weight, 2) AS weight, dep_count
                ORDER BY total_weight DESC
                LIMIT 10
            """),
            ("Most Depended-Upon Components", """
                MATCH ()-[d:DEPENDS_ON]->(target)
                WITH target, count(d) AS dependents, sum(d.weight) AS total_weight
                RETURN labels(target)[0] AS type, target.name AS component,
                       dependents, round(total_weight, 2) AS weight
                ORDER BY dependents DESC
                LIMIT 10
            """),
            ("DEPENDS_ON Distribution by Type", """
                MATCH ()-[d:DEPENDS_ON]->()
                RETURN d.dependency_type AS type,
                       count(*) AS count,
                       round(avg(d.weight), 2) AS avg_weight
                ORDER BY count DESC
            """),
            ("High-Weight Dependencies (Critical)", """
                MATCH (a)-[d:DEPENDS_ON]->(b)
                WHERE d.weight > 5.0
                RETURN labels(a)[0] AS source_type, a.name AS source,
                       labels(b)[0] AS target_type, b.name AS target,
                       d.dependency_type AS type, round(d.weight, 2) AS weight
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

    def export_to_json(self, filepath: str, include_depends_on: bool = True) -> None:
        """Export graph from Neo4j to JSON"""
        with self.driver.session(database=self.database) as session:
            # Export vertices
            graph = {
                "metadata": {
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": "neo4j",
                    "database": self.database,
                },
                "nodes": list(session.run("MATCH (n:Node) RETURN n.id AS id, n.name AS name")),
                "brokers": list(session.run("MATCH (b:Broker) RETURN b.id AS id, b.name AS name")),
                "applications": list(session.run(
                    "MATCH (a:Application) RETURN a.id AS id, a.name AS name, a.role AS role"
                )),
                "topics": [],
                "relationships": {},
            }
            
            # Topics with QoS
            for r in session.run("""
                MATCH (t:Topic)
                RETURN t.id AS id, t.name AS name, t.size AS size,
                       t.qos_durability AS durability,
                       t.qos_reliability AS reliability,
                       t.qos_transport_priority AS transport_priority
            """):
                graph["topics"].append({
                    "id": r["id"],
                    "name": r["name"],
                    "size": r["size"],
                    "qos": {
                        "durability": r["durability"],
                        "reliability": r["reliability"],
                        "transport_priority": r["transport_priority"],
                    },
                })
            
            # Relationships
            rel_queries = {
                "publishes_to": "MATCH (a:Application)-[:PUBLISHES_TO]->(t:Topic) RETURN a.id AS `from`, t.id AS `to`",
                "subscribes_to": "MATCH (a:Application)-[:SUBSCRIBES_TO]->(t:Topic) RETURN a.id AS `from`, t.id AS `to`",
                "routes": "MATCH (b:Broker)-[:ROUTES]->(t:Topic) RETURN b.id AS `from`, t.id AS `to`",
                "runs_on": "MATCH (c)-[:RUNS_ON]->(n:Node) RETURN c.id AS `from`, n.id AS `to`",
                "connects_to": "MATCH (n1:Node)-[:CONNECTS_TO]->(n2:Node) RETURN n1.id AS `from`, n2.id AS `to`",
            }
            
            for name, query in rel_queries.items():
                graph["relationships"][name] = [dict(r) for r in session.run(query)]
            
            # DEPENDS_ON
            if include_depends_on:
                graph["depends_on"] = []
                for r in session.run("""
                    MATCH (source)-[d:DEPENDS_ON]->(target)
                    RETURN source.id AS `from`, target.id AS `to`,
                           d.dependency_type AS dependency_type,
                           d.weight AS weight,
                           d.via_topics AS via_topics
                """):
                    graph["depends_on"].append(dict(r))
        
        with open(filepath, "w") as f:
            json.dump(graph, f, indent=2, default=str)
        
        self.logger.info(f"Graph exported to {filepath}")

    def export_cypher_queries(self, filepath: str) -> None:
        """Export useful Cypher queries to a file"""
        queries = '''
// ============================================================================
// Software-as-a-Graph: Useful Cypher Queries
// ============================================================================

// --- Basic Statistics ---

// Count all node types
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC;

// Count all relationship types
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY count DESC;

// --- Dependency Analysis ---

// View DEPENDS_ON by type
MATCH ()-[d:DEPENDS_ON]->()
RETURN d.dependency_type AS type, count(*) AS count,
       round(avg(d.weight), 2) AS avg_weight
ORDER BY count DESC;

// Find highest-weight dependencies
MATCH (a)-[d:DEPENDS_ON]->(b)
RETURN a.name AS source, b.name AS target,
       d.dependency_type AS type, round(d.weight, 2) AS weight
ORDER BY d.weight DESC
LIMIT 20;

// Find dependency chains (up to 3 hops)
MATCH path = (a:Application)-[:DEPENDS_ON*1..3]->(b:Application)
WHERE a <> b
RETURN [n IN nodes(path) | n.name] AS chain, length(path) AS depth
ORDER BY depth DESC
LIMIT 20;

// --- Critical Components ---

// Most depended-upon applications
MATCH ()-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a:Application)
WITH a, count(*) AS dependents, sum(d.weight) AS total_weight
RETURN a.name AS application, dependents, round(total_weight, 2) AS weight
ORDER BY dependents DESC
LIMIT 10;

// Single points of failure (only one broker routes critical topics)
MATCH (t:Topic)
WHERE t.qos_reliability = 'RELIABLE' OR t.qos_durability = 'PERSISTENT'
WITH t
MATCH (b:Broker)-[:ROUTES]->(t)
WITH t, collect(b) AS brokers
WHERE size(brokers) = 1
RETURN t.name AS topic, brokers[0].name AS single_broker;

// --- QoS Analysis ---

// Topics by QoS criticality
MATCH (t:Topic)
WITH t,
     CASE t.qos_durability WHEN 'PERSISTENT' THEN 0.4 WHEN 'TRANSIENT' THEN 0.25 ELSE 0 END +
     CASE t.qos_reliability WHEN 'RELIABLE' THEN 0.3 ELSE 0 END +
     CASE t.qos_transport_priority WHEN 'URGENT' THEN 0.3 WHEN 'HIGH' THEN 0.2 WHEN 'MEDIUM' THEN 0.1 ELSE 0 END AS qos_score
RETURN t.name AS topic, t.qos_durability AS durability,
       t.qos_reliability AS reliability, t.qos_transport_priority AS transport_priority,
       round(qos_score, 2) AS criticality_score
ORDER BY qos_score DESC
LIMIT 20;

// --- Infrastructure Analysis ---

// Node load (apps per node)
MATCH (app:Application)-[:RUNS_ON]->(n:Node)
WITH n, count(app) AS app_count, collect(app.name) AS apps
RETURN n.name AS node, app_count, apps
ORDER BY app_count DESC;

// Cross-node dependencies
MATCH (n1:Node)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2:Node)
RETURN n1.name AS source_node, n2.name AS target_node,
       d.via_apps AS dependent_apps, round(d.weight, 2) AS weight
ORDER BY d.weight DESC;
'''
        with open(filepath, "w") as f:
            f.write(queries)
        
        self.logger.info(f"Cypher queries exported to {filepath}")