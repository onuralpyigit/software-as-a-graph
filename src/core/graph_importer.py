"""
Graph Importer

Imports GraphModel data into Neo4j graph database with:
- Batch processing for large graphs
- Schema creation with constraints and indexes
- Unified DEPENDS_ON relationship derivation via Cypher
- Progress tracking and error handling
- Advanced graph analytics queries
- Sample queries for exploration
- Export capabilities
- Retry logic for transient errors
- Idempotent imports using MERGE
- Transaction management for reliability

The importer derives all dependency types directly in Neo4j using Cypher queries,
ensuring consistency between in-memory and database representations.

Usage:
    from src.core.graph_importer import GraphImporter

    # Basic usage
    importer = GraphImporter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    importer.import_graph(graph_data)
    stats = importer.get_statistics()
    importer.run_analytics()
    importer.close()

    # Context manager usage (recommended)
    with GraphImporter(uri="bolt://localhost:7687", user="neo4j", password="password") as importer:
        importer.import_graph(graph_data)
        stats = importer.get_statistics()
"""

import time
import logging
import functools
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError, TransientError, SessionExpired
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None
    ServiceUnavailable = Exception
    ClientError = Exception
    TransientError = Exception
    SessionExpired = Exception


def retry_on_transient_error(max_retries: int = 3, initial_delay: float = 1.0):
    """
    Decorator to retry operations on transient Neo4j errors.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (doubles each attempt)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (TransientError, SessionExpired, ServiceUnavailable) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.getLogger(__name__).warning(
                            f"Transient error in {func.__name__}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise

            raise last_exception
        return wrapper
    return decorator


class GraphImporter:
    """
    Imports pub-sub system graphs into Neo4j.

    Supports:
    - Batch imports for performance
    - Schema management (constraints, indexes)
    - Unified DEPENDS_ON derivation with four dependency types
    - Statistics and validation
    - Context manager protocol
    - Retry logic for transient errors
    - Idempotent imports using MERGE
    - Progress callbacks

    Example:
        with GraphImporter(uri="bolt://localhost:7687", user="neo4j", password="pass") as importer:
            importer.import_graph(graph_data)
    """

    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j",
                 max_retries: int = 3):
        """
        Initialize the Neo4j connection.

        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
            max_retries: Maximum retries for transient errors
        """
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )

        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_retries = max_retries
        self.driver = None
        self.logger = logging.getLogger(__name__)
        self._import_stats = {}

        self._connect()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed"""
        self.close()
        return False  # Don't suppress exceptions

    def _connect(self):
        """Establish connection to Neo4j with retry logic"""
        delay = 1.0
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60
                )
                # Verify connection
                self.driver.verify_connectivity()
                self.logger.info(f"Connected to Neo4j at {self.uri}")
                return
            except ServiceUnavailable as e:
                last_exception = e
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Connection attempt {attempt + 1}/{self.max_retries} failed, "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= 2

        self.logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts")
        raise last_exception

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")
    
    # =========================================================================
    # Schema Management
    # =========================================================================
    
    def create_schema(self):
        """Create database schema with constraints and indexes"""
        self.logger.info("Creating database schema...")
        
        with self.driver.session(database=self.database) as session:
            # Uniqueness constraints
            constraints = [
                "CREATE CONSTRAINT app_id IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT broker_id IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE",
                "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except ClientError as e:
                    self.logger.debug(f"Constraint may already exist: {e}")
            
            # Performance indexes
            indexes = [
                "CREATE INDEX app_name IF NOT EXISTS FOR (a:Application) ON (a.name)",
                "CREATE INDEX app_type IF NOT EXISTS FOR (a:Application) ON (a.app_type)",
                "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
                "CREATE INDEX broker_name IF NOT EXISTS FOR (b:Broker) ON (b.name)",
                "CREATE INDEX broker_type IF NOT EXISTS FOR (b:Broker) ON (b.broker_type)",
                "CREATE INDEX node_name IF NOT EXISTS FOR (n:Node) ON (n.name)",
                "CREATE INDEX node_type IF NOT EXISTS FOR (n:Node) ON (n.node_type)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except ClientError as e:
                    self.logger.debug(f"Index may already exist: {e}")
        
        self.logger.info("✓ Schema created")
    
    def clear_database(self):
        """Clear all data from the database"""
        self.logger.info("Clearing database...")
        
        with self.driver.session(database=self.database) as session:
            # Delete relationships first, then nodes
            session.run("MATCH ()-[r]->() DELETE r")
            session.run("MATCH (n) DELETE n")
        
        self.logger.info("✓ Database cleared")
    
    # =========================================================================
    # Main Import Method
    # =========================================================================

    def import_graph(self,
                     graph_data: Dict,
                     batch_size: int = 100,
                     show_progress: bool = False,
                     clear_first: bool = False,
                     derive_dependencies: bool = True,
                     progress_callback: Optional[Callable[[str, int, int], None]] = None):
        """
        Import graph data into Neo4j.

        Args:
            graph_data: Graph dictionary from JSON or GraphModel.to_dict()
            batch_size: Batch size for bulk imports
            show_progress: Show progress indicators (uses default logging)
            clear_first: Clear existing data before import (default False to avoid accidental data loss)
            derive_dependencies: Whether to derive DEPENDS_ON relationships (default True)
            progress_callback: Optional callback(phase, current, total) for progress updates

        Returns:
            Dict containing import statistics
        """
        self.logger.info("Starting graph import...")
        start_time = time.time()

        # Initialize import statistics
        self._import_stats = {
            'start_time': start_time,
            'phases': {},
            'errors': []
        }

        def report_progress(phase: str, current: int, total: int):
            """Internal progress reporter"""
            if progress_callback:
                progress_callback(phase, current, total)
            if show_progress and total > 0:
                percent = 100 * current / total
                self.logger.info(f"  {phase}: {current}/{total} ({percent:.1f}%)")

        try:
            if clear_first:
                self.clear_database()

            # Schema is always created/verified (idempotent)
            self.create_schema()

            # Import nodes
            nodes = graph_data.get('nodes', [])
            self._import_nodes(nodes, batch_size)
            self._import_stats['phases']['nodes'] = len(nodes)
            report_progress('Nodes', len(nodes), len(nodes))

            apps = graph_data.get('applications', [])
            self._import_applications(apps, batch_size)
            self._import_stats['phases']['applications'] = len(apps)
            report_progress('Applications', len(apps), len(apps))

            topics = graph_data.get('topics', [])
            self._import_topics(topics, batch_size)
            self._import_stats['phases']['topics'] = len(topics)
            report_progress('Topics', len(topics), len(topics))

            brokers = graph_data.get('brokers', [])
            self._import_brokers(brokers, batch_size)
            self._import_stats['phases']['brokers'] = len(brokers)
            report_progress('Brokers', len(brokers), len(brokers))

            # Import explicit relationships
            relationships = graph_data.get('relationships', graph_data.get('edges', {}))

            runs_on = relationships.get('runs_on', [])
            self._import_runs_on(runs_on, batch_size)
            self._import_stats['phases']['runs_on'] = len(runs_on)

            publishes = relationships.get('publishes_to', relationships.get('publishes', []))
            self._import_publishes_to(publishes, batch_size)
            self._import_stats['phases']['publishes_to'] = len(publishes)

            subscribes = relationships.get('subscribes_to', relationships.get('subscribes', []))
            self._import_subscribes_to(subscribes, batch_size)
            self._import_stats['phases']['subscribes_to'] = len(subscribes)

            routes = relationships.get('routes', [])
            self._import_routes(routes, batch_size)
            self._import_stats['phases']['routes'] = len(routes)

            connects = relationships.get('connects_to', [])
            self._import_connects_to(connects, batch_size)
            self._import_stats['phases']['connects_to'] = len(connects)

            # Derive unified DEPENDS_ON relationships
            if derive_dependencies:
                dep_counts = self._derive_all_dependencies()
                self._import_stats['phases']['depends_on'] = dep_counts

            duration = time.time() - start_time
            self._import_stats['duration'] = duration
            self.logger.info(f"✓ Graph imported in {duration:.2f}s")

            # Verify import
            stats = self.get_statistics()
            self._import_stats['final_stats'] = stats

            return stats

        except Exception as e:
            self._import_stats['errors'].append(str(e))
            self.logger.error(f"Import failed: {e}")
            raise
    
    # =========================================================================
    # Node Import Methods
    # =========================================================================

    @retry_on_transient_error(max_retries=3)
    def _import_nodes(self, nodes: List[Dict], batch_size: int):
        """Import infrastructure nodes using MERGE for idempotency"""
        if not nodes:
            return

        self.logger.info(f"Importing {len(nodes)} infrastructure nodes...")

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                # Use explicit transaction for reliability
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $nodes AS node
                        MERGE (n:Node {id: node.id})
                        SET n.name = coalesce(node.name, node.id),
                            n.node_type = coalesce(node.node_type, 'compute'),
                            n.location = node.location,
                            n.zone = node.zone,
                            n.updated_at = datetime()
                    """, nodes=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(nodes)} nodes")

    @retry_on_transient_error(max_retries=3)
    def _import_applications(self, apps: List[Dict], batch_size: int):
        """Import application nodes using MERGE for idempotency"""
        if not apps:
            return

        self.logger.info(f"Importing {len(apps)} applications...")

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(apps), batch_size):
                batch = apps[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $apps AS app
                        MERGE (a:Application {id: app.id})
                        SET a.name = coalesce(app.name, app.id),
                            a.app_type = coalesce(app.type, app.app_type, 'PROSUMER'),
                            a.criticality_weight = app.criticality_weight,
                            a.domain = app.domain,
                            a.updated_at = datetime()
                    """, apps=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(apps)} applications")

    @retry_on_transient_error(max_retries=3)
    def _import_topics(self, topics: List[Dict], batch_size: int):
        """Import topic nodes with QoS, handling missing data gracefully"""
        if not topics:
            return

        self.logger.info(f"Importing {len(topics)} topics...")

        # Pre-process topics to normalize QoS data
        processed_topics = []
        for t in topics:
            qos = t.get('qos', {}) or {}  # Handle None case
            processed_topic = {
                'id': t.get('id'),
                'name': t.get('name'),
                'message_size_bytes': t.get('message_size_bytes'),
                'message_rate_hz': t.get('message_rate_hz'),
                'message_type': t.get('message_type'),
                'qos_reliability': qos.get('reliability', 'best_effort'),
                'qos_durability': qos.get('durability', 'volatile'),
                'qos_deadline_ms': qos.get('deadline_ms'),
                'qos_lifespan_ms': qos.get('lifespan_ms'),
                'qos_history_depth': qos.get('history_depth'),
                'qos_transport_priority': qos.get('transport_priority', 0)
            }
            processed_topics.append(processed_topic)

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(processed_topics), batch_size):
                batch = processed_topics[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $topics AS topic
                        MERGE (t:Topic {id: topic.id})
                        SET t.name = coalesce(topic.name, topic.id),
                            t.message_size_bytes = topic.message_size_bytes,
                            t.message_rate_hz = topic.message_rate_hz,
                            t.message_type = topic.message_type,
                            t.qos_reliability = topic.qos_reliability,
                            t.qos_durability = topic.qos_durability,
                            t.qos_deadline_ms = topic.qos_deadline_ms,
                            t.qos_lifespan_ms = topic.qos_lifespan_ms,
                            t.qos_history_depth = topic.qos_history_depth,
                            t.qos_transport_priority = topic.qos_transport_priority,
                            t.updated_at = datetime()
                    """, topics=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(topics)} topics")

    @retry_on_transient_error(max_retries=3)
    def _import_brokers(self, brokers: List[Dict], batch_size: int):
        """Import broker nodes using MERGE for idempotency"""
        if not brokers:
            return

        self.logger.info(f"Importing {len(brokers)} brokers...")

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(brokers), batch_size):
                batch = brokers[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $brokers AS broker
                        MERGE (b:Broker {id: broker.id})
                        SET b.name = coalesce(broker.name, broker.id),
                            b.broker_type = coalesce(broker.broker_type, 'generic'),
                            b.capacity = broker.capacity,
                            b.current_load = broker.current_load,
                            b.updated_at = datetime()
                    """, brokers=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(brokers)} brokers")
    
    # =========================================================================
    # Relationship Import Methods
    # =========================================================================

    def _normalize_relationship(self, rel: Dict) -> Dict:
        """Normalize relationship dict to use consistent field names"""
        return {
            'from': rel.get('from', rel.get('source')),
            'to': rel.get('to', rel.get('target'))
        }

    @retry_on_transient_error(max_retries=3)
    def _import_runs_on(self, relationships: List[Dict], batch_size: int):
        """Import RUNS_ON relationships using MERGE for idempotency"""
        if not relationships:
            return

        self.logger.info(f"Importing {len(relationships)} RUNS_ON relationships...")

        # Normalize field names
        rels = [self._normalize_relationship(r) for r in relationships]

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                with session.begin_transaction() as tx:
                    # Handle both Application and Broker sources
                    tx.run("""
                        UNWIND $rels AS rel
                        MATCH (source) WHERE source.id = rel.from
                        MATCH (n:Node {id: rel.to})
                        MERGE (source)-[r:RUNS_ON]->(n)
                        SET r.updated_at = datetime()
                    """, rels=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(relationships)} RUNS_ON")

    @retry_on_transient_error(max_retries=3)
    def _import_publishes_to(self, relationships: List[Dict], batch_size: int):
        """Import PUBLISHES_TO relationships using MERGE for idempotency"""
        if not relationships:
            return

        self.logger.info(f"Importing {len(relationships)} PUBLISHES_TO relationships...")

        rels = []
        for r in relationships:
            rel = self._normalize_relationship(r)
            rel['period_ms'] = r.get('period_ms')
            rel['message_size_bytes'] = r.get('message_size_bytes', r.get('msg_size'))
            rels.append(rel)

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $rels AS rel
                        MATCH (a:Application {id: rel.from})
                        MATCH (t:Topic {id: rel.to})
                        MERGE (a)-[r:PUBLISHES_TO]->(t)
                        SET r.period_ms = rel.period_ms,
                            r.message_size_bytes = rel.message_size_bytes,
                            r.updated_at = datetime()
                    """, rels=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(relationships)} PUBLISHES_TO")

    @retry_on_transient_error(max_retries=3)
    def _import_subscribes_to(self, relationships: List[Dict], batch_size: int):
        """Import SUBSCRIBES_TO relationships using MERGE for idempotency"""
        if not relationships:
            return

        self.logger.info(f"Importing {len(relationships)} SUBSCRIBES_TO relationships...")

        rels = [self._normalize_relationship(r) for r in relationships]

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $rels AS rel
                        MATCH (a:Application {id: rel.from})
                        MATCH (t:Topic {id: rel.to})
                        MERGE (a)-[r:SUBSCRIBES_TO]->(t)
                        SET r.updated_at = datetime()
                    """, rels=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(relationships)} SUBSCRIBES_TO")

    @retry_on_transient_error(max_retries=3)
    def _import_routes(self, relationships: List[Dict], batch_size: int):
        """Import ROUTES relationships using MERGE for idempotency"""
        if not relationships:
            return

        self.logger.info(f"Importing {len(relationships)} ROUTES relationships...")

        rels = [self._normalize_relationship(r) for r in relationships]

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $rels AS rel
                        MATCH (b:Broker {id: rel.from})
                        MATCH (t:Topic {id: rel.to})
                        MERGE (b)-[r:ROUTES]->(t)
                        SET r.updated_at = datetime()
                    """, rels=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(relationships)} ROUTES")

    @retry_on_transient_error(max_retries=3)
    def _import_connects_to(self, relationships: List[Dict], batch_size: int):
        """Import explicit CONNECTS_TO relationships (physical topology) using MERGE"""
        if not relationships:
            return

        self.logger.info(f"Importing {len(relationships)} CONNECTS_TO relationships...")

        rels = []
        for r in relationships:
            rel = self._normalize_relationship(r)
            rel['bandwidth_mbps'] = r.get('bandwidth_mbps')
            rel['latency_ms'] = r.get('latency_ms')
            rels.append(rel)

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $rels AS rel
                        MATCH (n1:Node {id: rel.from})
                        MATCH (n2:Node {id: rel.to})
                        MERGE (n1)-[r:CONNECTS_TO]->(n2)
                        SET r.bandwidth_mbps = rel.bandwidth_mbps,
                            r.latency_ms = rel.latency_ms,
                            r.updated_at = datetime()
                    """, rels=batch)
                    tx.commit()

        self.logger.info(f"  ✓ Imported {len(relationships)} CONNECTS_TO")
    
    # =========================================================================
    # Unified DEPENDS_ON Derivation
    # =========================================================================
    
    def _derive_all_dependencies(self):
        """
        Derive all DEPENDS_ON relationships using Cypher queries.
        
        Order of derivation:
        1. APP_TO_APP: From topic subscription patterns
        2. APP_TO_BROKER: From topic routing
        3. NODE_TO_NODE: From application dependencies
        4. NODE_TO_BROKER: From broker placement
        """
        self.logger.info("Deriving unified DEPENDS_ON relationships...")
        
        counts = {}
        
        # 1. APP_TO_APP dependencies
        counts['app_to_app'] = self._derive_app_to_app()
        
        # 2. APP_TO_BROKER dependencies
        counts['app_to_broker'] = self._derive_app_to_broker()
        
        # 3. NODE_TO_NODE dependencies
        counts['node_to_node'] = self._derive_node_to_node()
        
        # 4. NODE_TO_BROKER dependencies
        counts['node_to_broker'] = self._derive_node_to_broker()
        
        total = sum(counts.values())
        self.logger.info(f"  ✓ Derived {counts['app_to_app']} APP_TO_APP dependencies")
        self.logger.info(f"  ✓ Derived {counts['app_to_broker']} APP_TO_BROKER dependencies")
        self.logger.info(f"  ✓ Derived {counts['node_to_node']} NODE_TO_NODE dependencies")
        self.logger.info(f"  ✓ Derived {counts['node_to_broker']} NODE_TO_BROKER dependencies")
        self.logger.info(f"  Total: {total} DEPENDS_ON relationships")
        
        return counts
    
    def _derive_app_to_app(self) -> int:
        """
        Derive APP_TO_APP dependencies from topic subscription patterns.
        
        Rule: subscriber DEPENDS_ON publisher via shared topic
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find all subscriber-publisher pairs through shared topics
                MATCH (subscriber:Application)-[:SUBSCRIBES_TO]->(t:Topic)
                      <-[:PUBLISHES_TO]-(publisher:Application)
                WHERE subscriber.id <> publisher.id
                
                // Collect all shared topics for each pair
                WITH subscriber, publisher, collect(t.id) AS topics
                
                // Calculate weight based on number of shared topics
                WITH subscriber, publisher, topics,
                     1.0 + (size(topics) - 1) * 0.2 AS weight
                
                // Create DEPENDS_ON with aggregated information
                MERGE (subscriber)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)
                SET d.topics = topics,
                    d.weight = CASE WHEN weight > 2.0 THEN 2.0 ELSE weight END,
                    d.derived_at = datetime()
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]
    
    def _derive_app_to_broker(self) -> int:
        """
        Derive APP_TO_BROKER dependencies from topic routing.
        
        Rule: app DEPENDS_ON broker that routes topics the app uses
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find apps and the brokers routing their topics
                MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
                      <-[:ROUTES]-(broker:Broker)
                
                // Collect all topics per app-broker pair
                WITH app, broker, collect(DISTINCT t.id) AS topics
                
                // Calculate weight
                WITH app, broker, topics,
                     1.0 + (size(topics) - 1) * 0.15 AS weight
                
                // Create DEPENDS_ON
                MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
                SET d.topics = topics,
                    d.weight = CASE WHEN weight > 1.8 THEN 1.8 ELSE weight END,
                    d.derived_at = datetime()
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]
    
    def _derive_node_to_node(self) -> int:
        """
        Derive NODE_TO_NODE dependencies from application dependencies.
        
        Rule: node_X DEPENDS_ON node_Y if app on node_X depends on app on node_Y
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find infrastructure dependencies based on app dependencies
                MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)
                      -[dep:DEPENDS_ON {dependency_type: 'app_to_app'}]->
                      (a2:Application)-[:RUNS_ON]->(n2:Node)
                WHERE n1.id <> n2.id
                
                // Aggregate all app pairs for each node pair
                WITH n1, n2, 
                     collect({source: a1.id, target: a2.id, weight: dep.weight}) AS app_deps
                
                // Calculate aggregated weight
                WITH n1, n2, app_deps,
                     reduce(s = 0.0, d IN app_deps | s + d.weight) AS total_weight
                WITH n1, n2, app_deps,
                     1.0 + (total_weight - 1) * 0.3 AS weight
                
                // Create NODE_TO_NODE DEPENDS_ON
                MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
                SET d.app_dependency_count = size(app_deps),
                    d.underlying_apps = [dep IN app_deps | dep.source + '->' + dep.target],
                    d.weight = CASE WHEN weight > 3.0 THEN 3.0 ELSE weight END,
                    d.derived_at = datetime()
                
                RETURN count(*) AS count
            """)
            
            return result.single()["count"]
    
    def _derive_node_to_broker(self) -> int:
        """
        Derive NODE_TO_BROKER dependencies from broker placement.
        
        Rule: node DEPENDS_ON broker if apps on node depend on that broker
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Find nodes that depend on brokers through their apps
                MATCH (n:Node)<-[:RUNS_ON]-(app:Application)
                      -[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker)
                
                // Aggregate dependent apps per node-broker pair
                WITH n, broker, collect(DISTINCT app.id) AS dependent_apps
                
                // Calculate weight
                WITH n, broker, dependent_apps,
                     1.0 + (size(dependent_apps) - 1) * 0.25 AS weight
                
                // Create NODE_TO_BROKER DEPENDS_ON
                MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(broker)
                SET d.dependent_apps = dependent_apps,
                    d.weight = CASE WHEN weight > 2.5 THEN 2.5 ELSE weight END,
                    d.derived_at = datetime()
                
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
            node_counts = session.run("""
                MATCH (n)
                WITH labels(n)[0] AS label, count(*) AS count
                RETURN collect({label: label, count: count}) AS nodes
            """).single()["nodes"]
            
            # Relationship counts
            rel_counts = session.run("""
                MATCH ()-[r]->()
                WITH type(r) AS rel_type, count(*) AS count
                RETURN collect({type: rel_type, count: count}) AS relationships
            """).single()["relationships"]
            
            # DEPENDS_ON by type
            depends_on_counts = session.run("""
                MATCH ()-[d:DEPENDS_ON]->()
                WITH d.dependency_type AS dep_type, count(*) AS count
                RETURN collect({type: dep_type, count: count}) AS depends_on
            """).single()["depends_on"]
            
            return {
                'nodes': {item['label']: item['count'] for item in node_counts},
                'relationships': {item['type']: item['count'] for item in rel_counts},
                'depends_on_by_type': {item['type']: item['count'] for item in depends_on_counts}
            }

    def verify_import(self, expected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that imported data matches expected counts.

        Args:
            expected_data: The original graph data used for import

        Returns:
            Dictionary with verification results including:
            - is_valid: Overall validation status
            - expected: Expected counts from input data
            - actual: Actual counts from database
            - mismatches: List of any mismatches found
        """
        self.logger.info("Verifying import...")

        # Calculate expected counts from input data
        expected = {
            'nodes': len(expected_data.get('nodes', [])),
            'applications': len(expected_data.get('applications', [])),
            'topics': len(expected_data.get('topics', [])),
            'brokers': len(expected_data.get('brokers', []))
        }

        relationships = expected_data.get('relationships', expected_data.get('edges', {}))
        expected['runs_on'] = len(relationships.get('runs_on', []))
        expected['publishes_to'] = len(relationships.get('publishes_to', relationships.get('publishes', [])))
        expected['subscribes_to'] = len(relationships.get('subscribes_to', relationships.get('subscribes', [])))
        expected['routes'] = len(relationships.get('routes', []))
        expected['connects_to'] = len(relationships.get('connects_to', []))

        # Get actual counts from database
        stats = self.get_statistics()
        actual = {
            'nodes': stats.get('nodes', {}).get('Node', 0),
            'applications': stats.get('nodes', {}).get('Application', 0),
            'topics': stats.get('nodes', {}).get('Topic', 0),
            'brokers': stats.get('nodes', {}).get('Broker', 0),
            'runs_on': stats.get('relationships', {}).get('RUNS_ON', 0),
            'publishes_to': stats.get('relationships', {}).get('PUBLISHES_TO', 0),
            'subscribes_to': stats.get('relationships', {}).get('SUBSCRIBES_TO', 0),
            'routes': stats.get('relationships', {}).get('ROUTES', 0),
            'connects_to': stats.get('relationships', {}).get('CONNECTS_TO', 0)
        }

        # Compare and find mismatches
        mismatches = []
        for key in expected:
            exp_val = expected[key]
            act_val = actual.get(key, 0)
            # Allow actual to be >= expected (MERGE may not create duplicates)
            if act_val < exp_val:
                mismatches.append({
                    'type': key,
                    'expected': exp_val,
                    'actual': act_val,
                    'difference': exp_val - act_val
                })

        is_valid = len(mismatches) == 0

        result = {
            'is_valid': is_valid,
            'expected': expected,
            'actual': actual,
            'mismatches': mismatches,
            'statistics': stats
        }

        if is_valid:
            self.logger.info("Import verification passed")
        else:
            self.logger.warning(f"Import verification found {len(mismatches)} mismatches")
            for m in mismatches:
                self.logger.warning(f"  {m['type']}: expected {m['expected']}, got {m['actual']}")

        return result

    def get_import_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health check of the imported graph.

        Returns:
            Dictionary with health metrics including connectivity,
            orphan nodes, missing relationships, etc.
        """
        with self.driver.session(database=self.database) as session:
            health = {}

            # Check for orphan applications (no pub/sub relationships)
            orphan_apps_result = session.run("""
                MATCH (a:Application)
                WHERE NOT (a)-[:PUBLISHES_TO]->() AND NOT (a)-[:SUBSCRIBES_TO]->()
                RETURN count(a) AS count
            """)
            health['orphan_applications'] = orphan_apps_result.single()['count']

            # Check for orphan topics (no publishers)
            orphan_topics_result = session.run("""
                MATCH (t:Topic)
                WHERE NOT ()-[:PUBLISHES_TO]->(t)
                RETURN count(t) AS count
            """)
            health['orphan_topics'] = orphan_topics_result.single()['count']

            # Check for apps without RUNS_ON
            unhosted_apps_result = session.run("""
                MATCH (a:Application)
                WHERE NOT (a)-[:RUNS_ON]->()
                RETURN count(a) AS count
            """)
            health['unhosted_applications'] = unhosted_apps_result.single()['count']

            # Check for unrouted topics
            unrouted_topics_result = session.run("""
                MATCH (t:Topic)
                WHERE NOT ()-[:ROUTES]->(t)
                RETURN count(t) AS count
            """)
            health['unrouted_topics'] = unrouted_topics_result.single()['count']

            # Check for dependency islands (nodes with no dependencies)
            dependency_coverage_result = session.run("""
                MATCH (a:Application)
                OPTIONAL MATCH (a)-[d:DEPENDS_ON]-()
                WITH a, count(d) AS dep_count
                RETURN
                    count(CASE WHEN dep_count = 0 THEN 1 END) AS isolated,
                    count(CASE WHEN dep_count > 0 THEN 1 END) AS connected
            """)
            dep_record = dependency_coverage_result.single()
            health['isolated_applications'] = dep_record['isolated']
            health['connected_applications'] = dep_record['connected']

            # Overall health score (0-100)
            stats = self.get_statistics()
            total_apps = stats.get('nodes', {}).get('Application', 0)
            total_topics = stats.get('nodes', {}).get('Topic', 0)

            if total_apps > 0 and total_topics > 0:
                orphan_app_pct = health['orphan_applications'] / total_apps
                orphan_topic_pct = health['orphan_topics'] / total_topics
                unhosted_pct = health['unhosted_applications'] / total_apps

                # Score: 100 - penalties for issues
                score = 100
                score -= orphan_app_pct * 30  # Up to 30 point penalty
                score -= orphan_topic_pct * 20  # Up to 20 point penalty
                score -= unhosted_pct * 20  # Up to 20 point penalty
                health['health_score'] = max(0, round(score, 1))
            else:
                health['health_score'] = 0

            return health

    def query_dependencies(self, 
                          dependency_type: Optional[str] = None,
                          source_id: Optional[str] = None,
                          target_id: Optional[str] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Query DEPENDS_ON relationships with optional filters.
        
        Args:
            dependency_type: Filter by type (app_to_app, app_to_broker, etc.)
            source_id: Filter by source component ID
            target_id: Filter by target component ID
            limit: Maximum results to return
        """
        with self.driver.session(database=self.database) as session:
            # Build dynamic query
            where_clauses = []
            params = {'limit': limit}
            
            if dependency_type:
                where_clauses.append("d.dependency_type = $dep_type")
                params['dep_type'] = dependency_type
            
            if source_id:
                where_clauses.append("source.id = $source_id")
                params['source_id'] = source_id
            
            if target_id:
                where_clauses.append("target.id = $target_id")
                params['target_id'] = target_id
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "true"
            
            query = f"""
                MATCH (source)-[d:DEPENDS_ON]->(target)
                WHERE {where_clause}
                RETURN source.id AS source,
                       target.id AS target,
                       d.dependency_type AS type,
                       d.weight AS weight,
                       d.topics AS topics
                LIMIT $limit
            """
            
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def get_dependency_chain(self, 
                            start_id: str, 
                            max_depth: int = 5) -> List[Dict]:
        """
        Get transitive dependency chain from a starting component.
        
        Args:
            start_id: Starting component ID
            max_depth: Maximum dependency depth to traverse
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = (start {id: $start_id})-[:DEPENDS_ON*1..$max_depth]->(end)
                WITH path, length(path) AS depth
                RETURN [n IN nodes(path) | n.id] AS chain,
                       [r IN relationships(path) | r.dependency_type] AS types,
                       depth
                ORDER BY depth
            """, start_id=start_id, max_depth=max_depth)
            
            return [dict(record) for record in result]
    
    # =========================================================================
    # Sample Queries
    # =========================================================================
    
    def run_sample_queries(self):
        """Run and display sample queries for exploring the graph"""
        self.logger.info("Running sample queries...")
        
        queries = [
            ("Application Count by Type", """
                MATCH (a:Application)
                RETURN a.app_type AS type, count(*) AS count
                ORDER BY count DESC
            """),
            
            ("Topics with Most Publishers", """
                MATCH (a:Application)-[:PUBLISHES_TO]->(t:Topic)
                WITH t, count(a) AS pub_count
                RETURN t.name AS topic, pub_count
                ORDER BY pub_count DESC
                LIMIT 10
            """),
            
            ("Topics with Most Subscribers", """
                MATCH (a:Application)-[:SUBSCRIBES_TO]->(t:Topic)
                WITH t, count(a) AS sub_count
                RETURN t.name AS topic, sub_count
                ORDER BY sub_count DESC
                LIMIT 10
            """),
            
            ("Applications per Infrastructure Node", """
                MATCH (a:Application)-[:RUNS_ON]->(n:Node)
                WITH n, count(a) AS app_count
                RETURN n.name AS node, n.node_type AS type, app_count
                ORDER BY app_count DESC
            """),
            
            ("Broker Load Distribution", """
                MATCH (b:Broker)-[:ROUTES]->(t:Topic)
                WITH b, count(t) AS topic_count
                RETURN b.name AS broker, b.broker_type AS type, 
                       topic_count, b.current_load AS load
                ORDER BY topic_count DESC
            """),
            
            ("DEPENDS_ON Distribution by Type", """
                MATCH ()-[d:DEPENDS_ON]->()
                RETURN d.dependency_type AS type, count(*) AS count
                ORDER BY count DESC
            """),
            
            ("Applications with Most Dependencies", """
                MATCH (a:Application)-[d:DEPENDS_ON]->()
                WITH a, count(d) AS dep_count
                RETURN a.name AS application, dep_count
                ORDER BY dep_count DESC
                LIMIT 10
            """),
            
            ("Applications Most Depended Upon", """
                MATCH ()-[d:DEPENDS_ON]->(a:Application)
                WITH a, count(d) AS dependents
                RETURN a.name AS application, dependents
                ORDER BY dependents DESC
                LIMIT 10
            """)
        ]
        
        with self.driver.session(database=self.database) as session:
            for title, query in queries:
                print(f"\n{title}:")
                print("-" * 50)
                try:
                    result = session.run(query)
                    records = list(result)
                    if records:
                        # Print header
                        keys = records[0].keys()
                        header = " | ".join(f"{k:20s}" for k in keys)
                        print(header)
                        print("-" * len(header))
                        # Print rows
                        for record in records[:10]:
                            row = " | ".join(f"{str(record[k]):20s}" for k in keys)
                            print(row)
                    else:
                        print("  No results")
                except Exception as e:
                    print(f"  Error: {e}")
    
    # =========================================================================
    # Advanced Analytics
    # =========================================================================
    
    def run_analytics(self) -> Dict[str, Any]:
        """
        Run comprehensive graph analytics.
        
        Returns:
            Dictionary containing all analytics results
        """
        self.logger.info("Running advanced analytics...")
        
        results = {}
        
        # Centrality analysis
        results['centrality'] = self._analyze_centrality()
        
        # Dependency analysis
        results['dependencies'] = self._analyze_dependencies()
        
        # Single Points of Failure
        results['spof_candidates'] = self._find_spof_candidates()
        
        # Topic analysis
        results['topic_analysis'] = self._analyze_topics()
        
        # Infrastructure analysis
        results['infrastructure'] = self._analyze_infrastructure()
        
        # Print summary
        self._print_analytics_summary(results)
        
        return results
    
    def _analyze_centrality(self) -> Dict[str, Any]:
        """Analyze node centrality metrics"""
        with self.driver.session(database=self.database) as session:
            # Degree centrality for applications
            degree_result = session.run("""
                MATCH (a:Application)
                OPTIONAL MATCH (a)-[out:PUBLISHES_TO|SUBSCRIBES_TO|DEPENDS_ON]->()
                OPTIONAL MATCH ()-[in:DEPENDS_ON]->(a)
                WITH a, count(DISTINCT out) AS out_degree, count(DISTINCT in) AS in_degree
                RETURN a.id AS id, a.name AS name, 
                       out_degree, in_degree, 
                       out_degree + in_degree AS total_degree
                ORDER BY total_degree DESC
                LIMIT 20
            """)
            
            return {
                'top_by_degree': [dict(r) for r in degree_result]
            }
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency patterns"""
        with self.driver.session(database=self.database) as session:
            # Dependency depth analysis
            depth_result = session.run("""
                MATCH path = (a:Application)-[:DEPENDS_ON*]->(b:Application)
                WHERE a <> b
                WITH a, b, length(path) AS depth
                RETURN max(depth) AS max_depth,
                       avg(depth) AS avg_depth,
                       count(*) AS total_paths
            """)
            
            depth_record = depth_result.single()
            
            # Circular dependency detection
            circular_result = session.run("""
                MATCH path = (a:Application)-[:DEPENDS_ON*2..5]->(a)
                RETURN [n IN nodes(path) | n.name] AS cycle
                LIMIT 10
            """)
            
            cycles = [dict(r) for r in circular_result]
            
            return {
                'max_dependency_depth': depth_record['max_depth'] if depth_record else 0,
                'avg_dependency_depth': round(depth_record['avg_depth'], 2) if depth_record and depth_record['avg_depth'] else 0,
                'total_dependency_paths': depth_record['total_paths'] if depth_record else 0,
                'circular_dependencies': cycles,
                'has_cycles': len(cycles) > 0
            }
    
    def _find_spof_candidates(self) -> List[Dict]:
        """Find Single Point of Failure candidates"""
        with self.driver.session(database=self.database) as session:
            # Find components that many others depend on
            result = session.run("""
                MATCH (component)-[:DEPENDS_ON]->(target)
                WITH target, count(DISTINCT component) AS dependent_count
                WHERE dependent_count >= 3
                MATCH (target)
                RETURN target.id AS id,
                       target.name AS name,
                       labels(target)[0] AS type,
                       dependent_count
                ORDER BY dependent_count DESC
                LIMIT 20
            """)
            
            return [dict(r) for r in result]
    
    def _analyze_topics(self) -> Dict[str, Any]:
        """Analyze topic usage patterns"""
        with self.driver.session(database=self.database) as session:
            # Topic statistics
            stats_result = session.run("""
                MATCH (t:Topic)
                OPTIONAL MATCH (pub:Application)-[:PUBLISHES_TO]->(t)
                OPTIONAL MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t)
                WITH t, 
                     count(DISTINCT pub) AS publishers,
                     count(DISTINCT sub) AS subscribers
                RETURN avg(publishers) AS avg_publishers,
                       avg(subscribers) AS avg_subscribers,
                       max(publishers) AS max_publishers,
                       max(subscribers) AS max_subscribers,
                       sum(CASE WHEN publishers = 0 THEN 1 ELSE 0 END) AS orphan_topics,
                       count(t) AS total_topics
            """)
            
            stats = stats_result.single()
            
            # God topics (high fan-out)
            god_topics_result = session.run("""
                MATCH (t:Topic)
                OPTIONAL MATCH (pub:Application)-[:PUBLISHES_TO]->(t)
                OPTIONAL MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t)
                WITH t, 
                     count(DISTINCT pub) AS publishers,
                     count(DISTINCT sub) AS subscribers
                WHERE publishers + subscribers > 10
                RETURN t.name AS topic, publishers, subscribers,
                       publishers + subscribers AS total_connections
                ORDER BY total_connections DESC
                LIMIT 10
            """)
            
            return {
                'total_topics': stats['total_topics'] if stats else 0,
                'avg_publishers_per_topic': round(stats['avg_publishers'], 2) if stats and stats['avg_publishers'] else 0,
                'avg_subscribers_per_topic': round(stats['avg_subscribers'], 2) if stats and stats['avg_subscribers'] else 0,
                'max_publishers': stats['max_publishers'] if stats else 0,
                'max_subscribers': stats['max_subscribers'] if stats else 0,
                'orphan_topics': stats['orphan_topics'] if stats else 0,
                'god_topics': [dict(r) for r in god_topics_result]
            }
    
    def _analyze_infrastructure(self) -> Dict[str, Any]:
        """Analyze infrastructure utilization"""
        with self.driver.session(database=self.database) as session:
            # Node utilization
            node_result = session.run("""
                MATCH (n:Node)
                OPTIONAL MATCH (component)-[:RUNS_ON]->(n)
                WITH n, count(component) AS hosted_components
                RETURN n.id AS id,
                       n.name AS name,
                       n.node_type AS type,
                       hosted_components
                ORDER BY hosted_components DESC
            """)
            
            nodes = [dict(r) for r in node_result]
            
            # Identify overloaded nodes
            overloaded = [n for n in nodes if n['hosted_components'] > 10]
            
            # Cross-node dependencies
            cross_node_result = session.run("""
                MATCH (n1:Node)<-[:RUNS_ON]-(a1)-[:DEPENDS_ON]->(a2)-[:RUNS_ON]->(n2:Node)
                WHERE n1 <> n2
                WITH n1, n2, count(*) AS dependency_count
                RETURN n1.name AS from_node, n2.name AS to_node, dependency_count
                ORDER BY dependency_count DESC
                LIMIT 10
            """)
            
            return {
                'nodes': nodes,
                'overloaded_nodes': overloaded,
                'cross_node_dependencies': [dict(r) for r in cross_node_result]
            }
    
    def _print_analytics_summary(self, results: Dict[str, Any]):
        """Print formatted analytics summary"""
        print("\n" + "=" * 70)
        print("GRAPH ANALYTICS SUMMARY")
        print("=" * 70)
        
        # Centrality
        print("\n📊 Top Applications by Degree Centrality:")
        for app in results['centrality']['top_by_degree'][:5]:
            print(f"   {app['name']}: {app['total_degree']} connections "
                  f"(out: {app['out_degree']}, in: {app['in_degree']})")
        
        # Dependencies
        deps = results['dependencies']
        print(f"\n🔗 Dependency Analysis:")
        print(f"   Max Dependency Depth: {deps['max_dependency_depth']}")
        print(f"   Avg Dependency Depth: {deps['avg_dependency_depth']}")
        print(f"   Total Dependency Paths: {deps['total_dependency_paths']}")
        if deps['has_cycles']:
            print(f"   ⚠️  Circular Dependencies Found: {len(deps['circular_dependencies'])}")
        else:
            print(f"   ✓ No Circular Dependencies")
        
        # SPOF
        spof = results['spof_candidates']
        print(f"\n⚠️  Single Point of Failure Candidates ({len(spof)}):")
        for s in spof[:5]:
            print(f"   {s['name']} ({s['type']}): {s['dependent_count']} dependents")
        
        # Topics
        topics = results['topic_analysis']
        print(f"\n📨 Topic Analysis:")
        print(f"   Total Topics: {topics['total_topics']}")
        print(f"   Avg Publishers/Topic: {topics['avg_publishers_per_topic']}")
        print(f"   Avg Subscribers/Topic: {topics['avg_subscribers_per_topic']}")
        print(f"   Orphan Topics: {topics['orphan_topics']}")
        if topics['god_topics']:
            print(f"   God Topics (high connectivity): {len(topics['god_topics'])}")
        
        # Infrastructure
        infra = results['infrastructure']
        print(f"\n🖥️  Infrastructure Analysis:")
        print(f"   Total Nodes: {len(infra['nodes'])}")
        if infra['overloaded_nodes']:
            print(f"   ⚠️  Overloaded Nodes: {len(infra['overloaded_nodes'])}")
        print(f"   Cross-Node Dependencies: {len(infra['cross_node_dependencies'])}")
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_to_json(self, filepath: str, include_derived: bool = True):
        """
        Export graph data from Neo4j to JSON.

        Args:
            filepath: Output file path
            include_derived: Include derived DEPENDS_ON relationships (default True)
        """
        import json

        with self.driver.session(database=self.database) as session:
            # Export infrastructure nodes
            nodes_result = session.run("""
                MATCH (n:Node)
                RETURN n.id AS id, n.name AS name, n.node_type AS node_type,
                       n.location AS location, n.zone AS zone
            """)
            nodes = [dict(r) for r in nodes_result]

            # Export brokers
            brokers_result = session.run("""
                MATCH (b:Broker)
                RETURN b.id AS id, b.name AS name, b.broker_type AS broker_type,
                       b.capacity AS capacity, b.current_load AS current_load
            """)
            brokers = [dict(r) for r in brokers_result]

            # Export applications
            apps_result = session.run("""
                MATCH (a:Application)
                RETURN a.id AS id, a.name AS name, a.app_type AS app_type,
                       a.criticality_weight AS criticality_weight, a.domain AS domain
            """)
            applications = [dict(r) for r in apps_result]

            # Export topics with QoS
            topics_result = session.run("""
                MATCH (t:Topic)
                RETURN t.id AS id, t.name AS name,
                       t.message_type AS message_type,
                       t.message_size_bytes AS message_size_bytes,
                       t.message_rate_hz AS message_rate_hz,
                       t.qos_reliability AS qos_reliability,
                       t.qos_durability AS qos_durability,
                       t.qos_deadline_ms AS qos_deadline_ms,
                       t.qos_lifespan_ms AS qos_lifespan_ms,
                       t.qos_history_depth AS qos_history_depth,
                       t.qos_transport_priority AS qos_transport_priority
            """)
            topics = []
            for r in topics_result:
                topic_data = dict(r)
                # Restructure QoS fields
                qos = {
                    'reliability': topic_data.pop('qos_reliability', None),
                    'durability': topic_data.pop('qos_durability', None),
                    'deadline_ms': topic_data.pop('qos_deadline_ms', None),
                    'lifespan_ms': topic_data.pop('qos_lifespan_ms', None),
                    'history_depth': topic_data.pop('qos_history_depth', None),
                    'transport_priority': topic_data.pop('qos_transport_priority', None)
                }
                # Remove None values
                topic_data['qos'] = {k: v for k, v in qos.items() if v is not None}
                topics.append(topic_data)

            # Export RUNS_ON relationships (both apps and brokers)
            runs_on_result = session.run("""
                MATCH (s)-[r:RUNS_ON]->(n:Node)
                RETURN s.id AS source, n.id AS target
            """)
            runs_on = [{'from': r['source'], 'to': r['target']} for r in runs_on_result]

            # Export PUBLISHES_TO relationships
            publishes_result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO]->(t:Topic)
                RETURN a.id AS source, t.id AS target,
                       r.period_ms AS period_ms, r.message_size_bytes AS message_size_bytes
            """)
            publishes = [
                {
                    'from': r['source'],
                    'to': r['target'],
                    'period_ms': r['period_ms'],
                    'message_size_bytes': r['message_size_bytes']
                }
                for r in publishes_result
            ]

            # Export SUBSCRIBES_TO relationships
            subscribes_result = session.run("""
                MATCH (a:Application)-[r:SUBSCRIBES_TO]->(t:Topic)
                RETURN a.id AS source, t.id AS target
            """)
            subscribes = [{'from': r['source'], 'to': r['target']} for r in subscribes_result]

            # Export ROUTES relationships
            routes_result = session.run("""
                MATCH (b:Broker)-[r:ROUTES]->(t:Topic)
                RETURN b.id AS source, t.id AS target
            """)
            routes = [{'from': r['source'], 'to': r['target']} for r in routes_result]

            # Export CONNECTS_TO relationships (physical topology)
            connects_result = session.run("""
                MATCH (n1:Node)-[r:CONNECTS_TO]->(n2:Node)
                RETURN n1.id AS source, n2.id AS target,
                       r.bandwidth_mbps AS bandwidth_mbps, r.latency_ms AS latency_ms
            """)
            connects = [
                {
                    'from': r['source'],
                    'to': r['target'],
                    'bandwidth_mbps': r['bandwidth_mbps'],
                    'latency_ms': r['latency_ms']
                }
                for r in connects_result
            ]

            # Build relationships dictionary
            relationships = {
                'runs_on': runs_on,
                'publishes_to': publishes,
                'subscribes_to': subscribes,
                'routes': routes,
                'connects_to': connects
            }

            # Export DEPENDS_ON relationships if requested
            if include_derived:
                depends_on_result = session.run("""
                    MATCH (source)-[d:DEPENDS_ON]->(target)
                    RETURN source.id AS source, target.id AS target,
                           d.dependency_type AS dependency_type,
                           d.weight AS weight,
                           d.topics AS topics,
                           labels(source)[0] AS source_type,
                           labels(target)[0] AS target_type
                """)
                depends_on = [
                    {
                        'from': r['source'],
                        'to': r['target'],
                        'dependency_type': r['dependency_type'],
                        'weight': r['weight'],
                        'topics': r['topics'],
                        'source_type': r['source_type'],
                        'target_type': r['target_type']
                    }
                    for r in depends_on_result
                ]
                relationships['depends_on'] = depends_on

            # Build graph dictionary
            graph = {
                'metadata': {
                    'exported_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'source': 'neo4j',
                    'database': self.database,
                    'include_derived': include_derived
                },
                'nodes': nodes,
                'brokers': brokers,
                'applications': applications,
                'topics': topics,
                'relationships': relationships
            }

            with open(filepath, 'w') as f:
                json.dump(graph, f, indent=2, default=str)

        self.logger.info(f"Graph exported to {filepath}")
    
    def export_cypher_queries(self, filepath: str):
        """
        Export useful Cypher queries to a file.
        
        Args:
            filepath: Output file path
        """
        queries = '''
// ============================================================================
// Software-as-a-Graph: Useful Cypher Queries
// ============================================================================

// --- Basic Exploration ---

// View all node types and counts
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC;

// View all relationship types and counts
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY count DESC;

// --- Application Queries ---

// Find all applications with their connections
MATCH (a:Application)
OPTIONAL MATCH (a)-[:PUBLISHES_TO]->(pub:Topic)
OPTIONAL MATCH (a)-[:SUBSCRIBES_TO]->(sub:Topic)
WITH a, collect(DISTINCT pub.name) AS publishes, collect(DISTINCT sub.name) AS subscribes
RETURN a.name AS application, a.app_type AS type,
       size(publishes) AS pub_count, size(subscribes) AS sub_count,
       publishes, subscribes
ORDER BY pub_count + sub_count DESC
LIMIT 20;

// Find applications by type
MATCH (a:Application)
WHERE a.app_type = 'PRODUCER'  // Change to 'CONSUMER' or 'PROSUMER'
RETURN a.name, a.app_type;

// --- Dependency Analysis ---

// View all APP_TO_APP dependencies
MATCH (a1:Application)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
RETURN a1.name AS dependent, a2.name AS dependency, d.topics AS via_topics, d.weight
ORDER BY d.weight DESC;

// Find dependency chains (up to 3 hops)
MATCH path = (a:Application)-[:DEPENDS_ON*1..3]->(b:Application)
WHERE a <> b
RETURN [n IN nodes(path) | n.name] AS chain, length(path) AS depth
ORDER BY depth DESC
LIMIT 20;

// Find circular dependencies
MATCH path = (a:Application)-[:DEPENDS_ON*2..5]->(a)
RETURN [n IN nodes(path) | n.name] AS cycle
LIMIT 10;

// --- Infrastructure Analysis ---

// View infrastructure topology
MATCH (n:Node)
OPTIONAL MATCH (component)-[:RUNS_ON]->(n)
WITH n, collect(component.name) AS hosted
RETURN n.name AS node, n.node_type AS type,
       size(hosted) AS component_count, hosted
ORDER BY component_count DESC;

// Cross-node data flow
MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)-[:PUBLISHES_TO]->(t:Topic)
      <-[:SUBSCRIBES_TO]-(a2:Application)-[:RUNS_ON]->(n2:Node)
WHERE n1 <> n2
RETURN n1.name AS source_node, n2.name AS target_node,
       t.name AS topic, a1.name AS publisher, a2.name AS subscriber
LIMIT 50;

// --- Topic Analysis ---

// Find "god topics" (high connectivity)
MATCH (t:Topic)
OPTIONAL MATCH (pub:Application)-[:PUBLISHES_TO]->(t)
OPTIONAL MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t)
WITH t, count(DISTINCT pub) AS publishers, count(DISTINCT sub) AS subscribers
WHERE publishers + subscribers > 5
RETURN t.name AS topic, publishers, subscribers,
       publishers + subscribers AS total_connections
ORDER BY total_connections DESC;

// Find orphan topics (no publishers)
MATCH (t:Topic)
WHERE NOT ()-[:PUBLISHES_TO]->(t)
RETURN t.name AS orphan_topic;

// --- Broker Analysis ---

// Broker load analysis
MATCH (b:Broker)-[:ROUTES]->(t:Topic)
WITH b, collect(t.name) AS topics, count(t) AS topic_count
RETURN b.name AS broker, b.broker_type AS type,
       topic_count, topics
ORDER BY topic_count DESC;

// --- Critical Components ---

// Find Single Points of Failure candidates
MATCH ()-[d:DEPENDS_ON]->(target)
WITH target, count(*) AS dependent_count
WHERE dependent_count >= 3
RETURN target.name AS component, labels(target)[0] AS type, dependent_count
ORDER BY dependent_count DESC;

// Find applications most depended upon
MATCH (a:Application)<-[d:DEPENDS_ON]-()
WITH a, count(d) AS dependents
RETURN a.name AS application, dependents
ORDER BY dependents DESC
LIMIT 10;

// --- QoS Analysis ---

// Topics by QoS requirements
MATCH (t:Topic)
WHERE t.qos_reliability IS NOT NULL
RETURN t.name AS topic, 
       t.qos_reliability AS reliability,
       t.qos_durability AS durability,
       t.qos_deadline_ms AS deadline_ms
ORDER BY t.qos_deadline_ms ASC;

// Critical topics (reliable + strict deadline)
MATCH (t:Topic)
WHERE t.qos_reliability = 'reliable' AND t.qos_deadline_ms < 100
RETURN t.name AS critical_topic, t.qos_deadline_ms AS deadline
ORDER BY t.qos_deadline_ms;
'''
        
        with open(filepath, 'w') as f:
            f.write(queries)
        
        self.logger.info(f"Cypher queries exported to {filepath}")