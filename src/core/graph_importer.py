"""
Graph Importer
Imports generated pub-sub system graphs into Neo4j database.
Features:
- Automatic schema creation
- Node and relationship import
- Property mapping
- Constraint and index creation
- Batch import for large graphs
- Query examples
- Visualization support
"""

from typing import Dict, List
import logging

# Check if neo4j driver is available
try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j Python driver not installed.")
    print("Install with: pip install neo4j")


class GraphImporter:
    """
    Imports pub-sub graphs into Neo4j
    
    Creates optimized schema with:
    - Node types: Node, Application, Topic, Broker
    - Relationship types: RUNS_ON, PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, DEPENDS_ON
    - Indexes and constraints for performance
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            user: Username
            password: Password
            database: Database name (default: neo4j)
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not installed. Install with: pip install neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] != 1:
                    raise Exception("Connection test failed")
            self.logger.info("✓ Connected to Neo4j")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        self.logger.info("Clearing database...")
        
        with self.driver.session(database=self.database) as session:
            # Delete all relationships and nodes
            session.run("MATCH (n) DETACH DELETE n")
            
        self.logger.info("✓ Database cleared")
    
    def create_schema(self):
        """Create database schema (constraints and indexes)"""
        self.logger.info("Creating schema...")
        
        with self.driver.session(database=self.database) as session:
            # Constraints (unique IDs)
            constraints = [
                "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT app_id IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT broker_id IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    self.logger.debug(f"Constraint creation: {e}")
            
            # Indexes for performance
            indexes = [
                "CREATE INDEX app_type IF NOT EXISTS FOR (a:Application) ON (a.type)",
                "CREATE INDEX app_criticality IF NOT EXISTS FOR (a:Application) ON (a.criticality)",
                "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
                "CREATE INDEX node_zone IF NOT EXISTS FOR (n:Node) ON (n.zone)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    self.logger.debug(f"Index creation: {e}")
        
        self.logger.info("✓ Schema created")
    
    def import_graph(self, graph_data: Dict, batch_size: int = 100):
        """
        Import graph data into Neo4j
        
        Args:
            graph_data: Graph dictionary from JSON
            batch_size: Batch size for bulk imports
        """
        self.logger.info("Importing graph...")
        
        # Import nodes (infrastructure)
        self._import_nodes(graph_data.get('nodes', []), batch_size)
        
        # Import applications
        self._import_applications(graph_data.get('applications', []), batch_size)
        
        # Import topics
        self._import_topics(graph_data.get('topics', []), batch_size)
        
        # Import brokers
        self._import_brokers(graph_data.get('brokers', []), batch_size)
        
        # Import relationships
        relationships = graph_data.get('relationships', {})
        self._import_runs_on(relationships.get('runs_on', []), batch_size)
        self._import_publishes_to(relationships.get('publishes_to', []), batch_size)
        self._import_subscribes_to(relationships.get('subscribes_to', []), batch_size)
        self._import_routes(relationships.get('routes', []), batch_size)
        
        # Derive dependencies (optional)
        self._derive_dependencies()
        
        self.logger.info("✓ Graph imported")
    
    def _import_nodes(self, nodes: List[Dict], batch_size: int):
        """Import infrastructure nodes"""
        if not nodes:
            return
        
        self.logger.info(f"Importing {len(nodes)} nodes...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (n:Node {
                        id: node.id,
                        name: node.name,
                        cpu_capacity: node.cpu_capacity,
                        memory_gb: node.memory_gb,
                        network_bandwidth_mbps: node.network_bandwidth_mbps,
                        zone: node.zone,
                        region: node.region
                    })
                """, nodes=batch)
        
        self.logger.info(f"  ✓ Imported {len(nodes)} nodes")
    
    def _import_applications(self, applications: List[Dict], batch_size: int):
        """Import applications"""
        if not applications:
            return
        
        self.logger.info(f"Importing {len(applications)} applications...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(applications), batch_size):
                batch = applications[i:i + batch_size]
                
                session.run("""
                    UNWIND $apps AS app
                    CREATE (a:Application {
                        id: app.id,
                        name: app.name,
                        type: app.type,
                        criticality: app.criticality,
                        replicas: app.replicas,
                        cpu_request: app.cpu_request,
                        memory_request_mb: app.memory_request_mb
                    })
                """, apps=batch)
        
        self.logger.info(f"  ✓ Imported {len(applications)} applications")
    
    def _import_topics(self, topics: List[Dict], batch_size: int):
        """Import topics"""
        if not topics:
            return
        
        self.logger.info(f"Importing {len(topics)} topics...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(topics), batch_size):
                batch = topics[i:i + batch_size]
                
                # Flatten QoS for easier querying
                processed_batch = []
                for topic in batch:
                    qos = topic.get('qos', {})
                    processed = {
                        'id': topic['id'],
                        'name': topic['name'],
                        'message_size_bytes': topic.get('message_size_bytes', 0),
                        'expected_rate_hz': topic.get('expected_rate_hz', 0),
                        'qos_durability': qos.get('durability', 'VOLATILE'),
                        'qos_reliability': qos.get('reliability', 'BEST_EFFORT'),
                        'qos_history_depth': qos.get('history_depth', 1),
                        'qos_deadline_ms': qos.get('deadline_ms'),
                        'qos_lifespan_ms': qos.get('lifespan_ms'),
                        'qos_transport_priority': qos.get('transport_priority', 'MEDIUM')
                    }
                    processed_batch.append(processed)
                
                session.run("""
                    UNWIND $topics AS topic
                    CREATE (t:Topic {
                        id: topic.id,
                        name: topic.name,
                        message_size_bytes: topic.message_size_bytes,
                        expected_rate_hz: topic.expected_rate_hz,
                        qos_durability: topic.qos_durability,
                        qos_reliability: topic.qos_reliability,
                        qos_history_depth: topic.qos_history_depth,
                        qos_deadline_ms: topic.qos_deadline_ms,
                        qos_lifespan_ms: topic.qos_lifespan_ms,
                        qos_transport_priority: topic.qos_transport_priority
                    })
                """, topics=processed_batch)
        
        self.logger.info(f"  ✓ Imported {len(topics)} topics")
    
    def _import_brokers(self, brokers: List[Dict], batch_size: int):
        """Import message brokers"""
        if not brokers:
            return
        
        self.logger.info(f"Importing {len(brokers)} brokers...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(brokers), batch_size):
                batch = brokers[i:i + batch_size]
                
                session.run("""
                    UNWIND $brokers AS broker
                    CREATE (b:Broker {
                        id: broker.id,
                        name: broker.name,
                        max_topics: broker.max_topics,
                        max_connections: broker.max_connections
                    })
                """, brokers=batch)
        
        self.logger.info(f"  ✓ Imported {len(brokers)} brokers")
    
    def _import_runs_on(self, relationships: List[Dict], batch_size: int):
        """Import RUNS_ON relationships"""
        if not relationships:
            return
        
        self.logger.info(f"Importing {len(relationships)} RUNS_ON relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (a:Application {id: rel.from})
                    MATCH (n:Node {id: rel.to})
                    CREATE (a)-[:RUNS_ON]->(n)
                """, rels=batch)
        
        self.logger.info(f"  ✓ Imported {len(relationships)} RUNS_ON")
    
    def _import_publishes_to(self, relationships: List[Dict], batch_size: int):
        """Import PUBLISHES_TO relationships"""
        if not relationships:
            return
        
        self.logger.info(f"Importing {len(relationships)} PUBLISHES_TO relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (a:Application {id: rel.from})
                    MATCH (t:Topic {id: rel.to})
                    CREATE (a)-[:PUBLISHES_TO {
                        period_ms: rel.period_ms,
                        msg_size: rel.msg_size
                    }]->(t)
                """, rels=batch)
        
        self.logger.info(f"  ✓ Imported {len(relationships)} PUBLISHES_TO")
    
    def _import_subscribes_to(self, relationships: List[Dict], batch_size: int):
        """Import SUBSCRIBES_TO relationships"""
        if not relationships:
            return
        
        self.logger.info(f"Importing {len(relationships)} SUBSCRIBES_TO relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (a:Application {id: rel.from})
                    MATCH (t:Topic {id: rel.to})
                    CREATE (a)-[:SUBSCRIBES_TO]->(t)
                """, rels=batch)
        
        self.logger.info(f"  ✓ Imported {len(relationships)} SUBSCRIBES_TO")
    
    def _import_routes(self, relationships: List[Dict], batch_size: int):
        """Import ROUTES relationships"""
        if not relationships:
            return
        
        self.logger.info(f"Importing {len(relationships)} ROUTES relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (b:Broker {id: rel.from})
                    MATCH (t:Topic {id: rel.to})
                    CREATE (b)-[:ROUTES]->(t)
                """, rels=batch)
        
        self.logger.info(f"  ✓ Imported {len(relationships)} ROUTES")
    
    def _derive_dependencies(self):
        """Derive DEPENDS_ON relationships from pub-sub patterns"""
        self.logger.info("Deriving DEPENDS_ON relationships...")
        
        with self.driver.session(database=self.database) as session:
            # Application depends on publisher if it subscribes to their topic
            result = session.run("""
                MATCH (consumer:Application)-[:SUBSCRIBES_TO]->(t:Topic)
                      <-[:PUBLISHES_TO]-(producer:Application)
                WHERE consumer.id <> producer.id
                MERGE (consumer)-[:DEPENDS_ON]->(producer)
                RETURN count(*) as count
            """)
            
            count = result.single()["count"]
            self.logger.info(f"  ✓ Derived {count} DEPENDS_ON relationships")
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session(database=self.database) as session:
            # Node counts
            node_count = session.run("MATCH (n:Node) RETURN count(n) as count").single()["count"]
            app_count = session.run("MATCH (a:Application) RETURN count(a) as count").single()["count"]
            topic_count = session.run("MATCH (t:Topic) RETURN count(t) as count").single()["count"]
            broker_count = session.run("MATCH (b:Broker) RETURN count(b) as count").single()["count"]
            
            # Relationship counts
            runs_on = session.run("MATCH ()-[r:RUNS_ON]->() RETURN count(r) as count").single()["count"]
            publishes = session.run("MATCH ()-[r:PUBLISHES_TO]->() RETURN count(r) as count").single()["count"]
            subscribes = session.run("MATCH ()-[r:SUBSCRIBES_TO]->() RETURN count(r) as count").single()["count"]
            routes = session.run("MATCH ()-[r:ROUTES]->() RETURN count(r) as count").single()["count"]
            depends = session.run("MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) as count").single()["count"]
            
            return {
                'nodes': {
                    'Node': node_count,
                    'Application': app_count,
                    'Topic': topic_count,
                    'Broker': broker_count,
                    'total': node_count + app_count + topic_count + broker_count
                },
                'relationships': {
                    'RUNS_ON': runs_on,
                    'PUBLISHES_TO': publishes,
                    'SUBSCRIBES_TO': subscribes,
                    'ROUTES': routes,
                    'DEPENDS_ON': depends,
                    'total': runs_on + publishes + subscribes + routes + depends
                }
            }
    
    def run_sample_queries(self):
        """Run sample queries to demonstrate Neo4j capabilities"""
        print("\n" + "=" * 70)
        print("SAMPLE QUERIES")
        print("=" * 70)
        
        with self.driver.session(database=self.database) as session:
            # Query 1: Critical applications
            print("\n1. Critical Applications:")
            result = session.run("""
                MATCH (a:Application)
                WHERE a.criticality = 'CRITICAL'
                RETURN a.id, a.name, a.replicas
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['a.id']}: {record['a.name']} (replicas: {record['a.replicas']})")
            
            # Query 2: Most connected applications
            print("\n2. Most Connected Applications (by topics):")
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
                WITH a, count(DISTINCT t) as topic_count
                RETURN a.id, a.name, topic_count
                ORDER BY topic_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['a.id']}: {record['a.name']} ({record['topic_count']} topics)")
            
            # Query 3: Topics with most subscribers
            print("\n3. Most Popular Topics:")
            result = session.run("""
                MATCH (t:Topic)<-[:SUBSCRIBES_TO]-(a:Application)
                WITH t, count(a) as subscriber_count
                RETURN t.id, t.name, subscriber_count
                ORDER BY subscriber_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['t.id']}: {record['t.name']} ({record['subscriber_count']} subscribers)")
            
            # Query 4: Broker load
            print("\n4. Broker Topic Load:")
            result = session.run("""
                MATCH (b:Broker)-[:ROUTES]->(t:Topic)
                WITH b, count(t) as topic_count
                RETURN b.id, b.name, topic_count, b.max_topics
                ORDER BY topic_count DESC
            """)
            for record in result:
                utilization = (record['topic_count'] / record['b.max_topics'] * 100) if record['b.max_topics'] > 0 else 0
                print(f"   {record['b.id']}: {record['topic_count']}/{record['b.max_topics']} topics ({utilization:.1f}%)")
            
            # Query 5: Application dependencies
            print("\n5. Application Dependencies (chains):")
            result = session.run("""
                MATCH path = (a1:Application)-[:DEPENDS_ON*1..3]->(a2:Application)
                RETURN a1.id, a2.id, length(path) as chain_length
                ORDER BY chain_length DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['a1.id']} → {record['a2.id']} (chain: {record['chain_length']})")
            
            # Query 6: Single points of failure
            print("\n6. Potential Single Points of Failure:")
            result = session.run("""
                MATCH (a:Application)-[:DEPENDS_ON]->(critical:Application)
                WHERE critical.replicas = 1
                WITH critical, count(DISTINCT a) as dependent_count
                WHERE dependent_count > 5
                RETURN critical.id, critical.name, dependent_count, critical.criticality
                ORDER BY dependent_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['critical.id']}: {record['critical.name']} "
                      f"({record['dependent_count']} dependents, {record['critical.criticality']})")
            
            # Query 7: Cross-zone dependencies
            print("\n7. Cross-Zone Dependencies:")
            result = session.run("""
                MATCH (a1:Application)-[:RUNS_ON]->(n1:Node),
                      (a1)-[:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
                WHERE n1.zone <> n2.zone
                RETURN n1.zone, n2.zone, count(*) as dep_count
                ORDER BY dep_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   {record['n1.zone']} → {record['n2.zone']}: {record['dep_count']} dependencies")
            
            # Query 8: High-frequency topics
            print("\n8. High-Frequency Topics:")
            result = session.run("""
                MATCH (t:Topic)
                WHERE t.expected_rate_hz >= 50
                RETURN t.id, t.name, t.expected_rate_hz, t.message_size_bytes
                ORDER BY t.expected_rate_hz DESC
                LIMIT 5
            """)
            for record in result:
                throughput = record['t.expected_rate_hz'] * record['t.message_size_bytes'] / 1024
                print(f"   {record['t.id']}: {record['t.name']} "
                      f"({record['t.expected_rate_hz']} Hz, {throughput:.1f} KB/s)")
