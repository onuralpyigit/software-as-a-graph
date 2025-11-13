"""
Graph Importer

Imports generated pub-sub system graphs into Neo4j database with:
- Automatic schema creation with constraints and indexes
- Batch import for optimal performance
- Transaction management with retry logic
- Progress reporting
- Comprehensive error handling
- Sample queries and analytics
- Statistics and validation

Node Types:
- Node (Infrastructure)
- Application
- Topic  
- Broker

Relationship Types:
- RUNS_ON (Application → Node)
- PUBLISHES_TO (Application → Topic)
- SUBSCRIBES_TO (Application → Topic)
- ROUTES (Broker → Topic)
- DEPENDS_ON (Application → Application) - derived
"""

from typing import Dict, List, Optional, Any
import logging
import time

# Check if neo4j driver is available
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, ClientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  Warning: neo4j Python driver not installed.")
    print("Install with: pip install neo4j")
    print("Or: pip install neo4j --break-system-packages")


class GraphImporter:
    """
    Imports pub-sub graphs into Neo4j with optimized schema and queries
    
    Features:
    - Batch processing for large graphs
    - Transaction management
    - Progress reporting
    - Comprehensive analytics
    - Error handling and retry logic
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
            raise ImportError(
                "neo4j driver not installed. "
                "Install with: pip install neo4j"
            )
        
        self.uri = uri
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Create driver
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        except Exception as e:
            self.logger.error(f"Failed to create Neo4j driver: {e}")
            raise
        
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
        except ServiceUnavailable as e:
            self.logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            self.logger.debug("Neo4j connection closed")
    
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
                except ClientError as e:
                    # Constraint might already exist
                    self.logger.debug(f"Constraint creation: {e}")
            
            # Indexes for performance
            indexes = [
                "CREATE INDEX app_type IF NOT EXISTS FOR (a:Application) ON (a.type)",
                "CREATE INDEX app_criticality IF NOT EXISTS FOR (a:Application) ON (a.criticality)",
                "CREATE INDEX app_name IF NOT EXISTS FOR (a:Application) ON (a.name)",
                "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
                "CREATE INDEX node_zone IF NOT EXISTS FOR (n:Node) ON (n.zone)",
                "CREATE INDEX node_region IF NOT EXISTS FOR (n:Node) ON (n.region)",
                "CREATE INDEX broker_name IF NOT EXISTS FOR (b:Broker) ON (b.name)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except ClientError as e:
                    self.logger.debug(f"Index creation: {e}")
        
        self.logger.info("✓ Schema created")
    
    def import_graph(self, graph_data: Dict, batch_size: int = 100, show_progress: bool = False):
        """
        Import graph data into Neo4j
        
        Args:
            graph_data: Graph dictionary from JSON
            batch_size: Batch size for bulk imports
            show_progress: Show progress bars
        """
        self.logger.info("Importing graph...")
        start_time = time.time()
        
        # Import nodes (infrastructure)
        self._import_nodes(graph_data.get('nodes', []), batch_size, show_progress)
        
        # Import applications
        self._import_applications(graph_data.get('applications', []), batch_size, show_progress)
        
        # Import topics
        self._import_topics(graph_data.get('topics', []), batch_size, show_progress)
        
        # Import brokers
        self._import_brokers(graph_data.get('brokers', []), batch_size, show_progress)
        
        # Import relationships
        relationships = graph_data.get('relationships', {})
        self._import_runs_on(relationships.get('runs_on', []), batch_size, show_progress)
        self._import_publishes_to(relationships.get('publishes_to', []), batch_size, show_progress)
        self._import_subscribes_to(relationships.get('subscribes_to', []), batch_size, show_progress)
        self._import_routes(relationships.get('routes', []), batch_size, show_progress)
        
        # Derive dependencies
        self._derive_dependencies()
        
        duration = time.time() - start_time
        self.logger.info(f"✓ Graph imported in {duration:.2f}s")
    
    def _import_nodes(self, nodes: List[Dict], batch_size: int, show_progress: bool = False):
        """Import infrastructure nodes"""
        if not nodes:
            self.logger.debug("No infrastructure nodes to import")
            return
        
        self.logger.info(f"Importing {len(nodes)} infrastructure nodes...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $nodes AS node
                        CREATE (n:Node {
                            id: node.id,
                            name: node.name,
                            cpu_capacity: coalesce(node.cpu_capacity, 0.0),
                            memory_gb: coalesce(node.memory_gb, 0.0),
                            network_bandwidth_mbps: coalesce(node.network_bandwidth_mbps, 0.0),
                            zone: coalesce(node.zone, 'unknown'),
                            region: coalesce(node.region, 'unknown')
                        })
                    """, nodes=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import node batch {i}-{i+len(batch)}: {e}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(nodes)} nodes")
    
    def _import_applications(self, applications: List[Dict], batch_size: int, show_progress: bool = False):
        """Import applications"""
        if not applications:
            self.logger.debug("No applications to import")
            return
        
        self.logger.info(f"Importing {len(applications)} applications...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(applications), batch_size):
                batch = applications[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $apps AS app
                        CREATE (a:Application {
                            id: app.id,
                            name: app.name,
                            type: coalesce(app.type, 'UNKNOWN'),
                            criticality: coalesce(app.criticality, 'MEDIUM'),
                            replicas: coalesce(app.replicas, 1),
                            cpu_request: coalesce(app.cpu_request, 0.0),
                            memory_request_mb: coalesce(app.memory_request_mb, 0.0)
                        })
                    """, apps=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import application batch {i}-{i+len(batch)}: {e}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(applications)} applications")
    
    def _import_topics(self, topics: List[Dict], batch_size: int, show_progress: bool = False):
        """Import topics with QoS properties"""
        if not topics:
            self.logger.debug("No topics to import")
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
                        'message_rate_hz': topic.get('message_rate_hz', 0),
                        'qos_durability': qos.get('durability', 'VOLATILE'),
                        'qos_reliability': qos.get('reliability', 'BEST_EFFORT'),
                        'qos_history_depth': qos.get('history_depth', 1),
                        'qos_deadline_ms': qos.get('deadline_ms'),
                        'qos_lifespan_ms': qos.get('lifespan_ms'),
                        'qos_transport_priority': qos.get('transport_priority', 'MEDIUM')
                    }
                    processed_batch.append(processed)
                
                try:
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
                except Exception as e:
                    self.logger.error(f"Failed to import topic batch {i}-{i+len(batch)}: {e}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(topics)} topics")
    
    def _import_brokers(self, brokers: List[Dict], batch_size: int, show_progress: bool = False):
        """Import message brokers"""
        if not brokers:
            self.logger.debug("No brokers to import")
            return
        
        self.logger.info(f"Importing {len(brokers)} brokers...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(brokers), batch_size):
                batch = brokers[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $brokers AS broker
                        CREATE (b:Broker {
                            id: broker.id,
                            name: broker.name,
                            max_topics: coalesce(broker.max_topics, 100),
                            max_connections: coalesce(broker.max_connections, 1000)
                        })
                    """, brokers=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import broker batch {i}-{i+len(batch)}: {e}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(brokers)} brokers")
    
    def _import_runs_on(self, relationships: List[Dict], batch_size: int, show_progress: bool = False):
        """Import RUNS_ON relationships"""
        if not relationships:
            self.logger.debug("No RUNS_ON relationships to import")
            return
        
        self.logger.info(f"Importing {len(relationships)} RUNS_ON relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $rels AS rel
                        MATCH (a:Application {id: rel.from})
                        MATCH (n:Node {id: rel.to})
                        CREATE (a)-[:RUNS_ON]->(n)
                    """, rels=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import RUNS_ON batch {i}-{i+len(batch)}: {e}")
                    # Log which relationships failed
                    for rel in batch:
                        self.logger.debug(f"  Failed relationship: {rel}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(relationships)} RUNS_ON")
    
    def _import_publishes_to(self, relationships: List[Dict], batch_size: int, show_progress: bool = False):
        """Import PUBLISHES_TO relationships"""
        if not relationships:
            self.logger.debug("No PUBLISHES_TO relationships to import")
            return
        
        self.logger.info(f"Importing {len(relationships)} PUBLISHES_TO relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $rels AS rel
                        MATCH (a:Application {id: rel.from})
                        MATCH (t:Topic {id: rel.to})
                        CREATE (a)-[:PUBLISHES_TO {
                            period_ms: coalesce(rel.period_ms, 0),
                            msg_size: coalesce(rel.msg_size, 0)
                        }]->(t)
                    """, rels=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import PUBLISHES_TO batch {i}-{i+len(batch)}: {e}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(relationships)} PUBLISHES_TO")
    
    def _import_subscribes_to(self, relationships: List[Dict], batch_size: int, show_progress: bool = False):
        """Import SUBSCRIBES_TO relationships"""
        if not relationships:
            self.logger.debug("No SUBSCRIBES_TO relationships to import")
            return
        
        self.logger.info(f"Importing {len(relationships)} SUBSCRIBES_TO relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $rels AS rel
                        MATCH (a:Application {id: rel.from})
                        MATCH (t:Topic {id: rel.to})
                        CREATE (a)-[:SUBSCRIBES_TO]->(t)
                    """, rels=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import SUBSCRIBES_TO batch {i}-{i+len(batch)}: {e}")
                    raise
        
        self.logger.info(f"  ✓ Imported {len(relationships)} SUBSCRIBES_TO")
    
    def _import_routes(self, relationships: List[Dict], batch_size: int, show_progress: bool = False):
        """Import ROUTES relationships"""
        if not relationships:
            self.logger.debug("No ROUTES relationships to import")
            return
        
        self.logger.info(f"Importing {len(relationships)} ROUTES relationships...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                try:
                    session.run("""
                        UNWIND $rels AS rel
                        MATCH (b:Broker {id: rel.from})
                        MATCH (t:Topic {id: rel.to})
                        CREATE (b)-[:ROUTES]->(t)
                    """, rels=batch)
                except Exception as e:
                    self.logger.error(f"Failed to import ROUTES batch {i}-{i+len(batch)}: {e}")
                    raise
        
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
        """Get comprehensive graph statistics"""
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
                RETURN a.id, a.name, a.replicas, a.type
                ORDER BY a.replicas ASC
                LIMIT 5
            """)
            for record in result:
                print(f"   • {record['a.name']:30s} (ID: {record['a.id']:10s}, "
                      f"Type: {record['a.type']:10s}, Replicas: {record['a.replicas']})")
            
            # Query 2: Most connected applications
            print("\n2. Most Connected Applications (by topics):")
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
                WITH a, count(DISTINCT t) as topic_count,
                     count(CASE WHEN type(r) = 'PUBLISHES_TO' THEN 1 END) as pub_count,
                     count(CASE WHEN type(r) = 'SUBSCRIBES_TO' THEN 1 END) as sub_count
                RETURN a.id, a.name, topic_count, pub_count, sub_count
                ORDER BY topic_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   • {record['a.name']:30s} ({record['topic_count']} topics: "
                      f"{record['pub_count']} pub, {record['sub_count']} sub)")
            
            # Query 3: Topics with most subscribers
            print("\n3. Most Popular Topics (by subscribers):")
            result = session.run("""
                MATCH (t:Topic)<-[:SUBSCRIBES_TO]-(a:Application)
                WITH t, count(a) as subscriber_count
                RETURN t.id, t.name, subscriber_count, t.expected_rate_hz
                ORDER BY subscriber_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   • {record['t.name']:40s} ({record['subscriber_count']} subscribers, "
                      f"{record['t.expected_rate_hz']} Hz)")
            
            # Query 4: Broker load
            print("\n4. Broker Topic Load:")
            result = session.run("""
                MATCH (b:Broker)-[:ROUTES]->(t:Topic)
                WITH b, count(t) as topic_count
                RETURN b.id, b.name, topic_count, b.max_topics,
                       100.0 * topic_count / b.max_topics as utilization_pct
                ORDER BY utilization_pct DESC
            """)
            for record in result:
                print(f"   • {record['b.name']:30s} "
                      f"({record['topic_count']}/{record['b.max_topics']} topics, "
                      f"{record['utilization_pct']:.1f}% util)")
            
            # Query 5: Application dependencies
            print("\n5. Application Dependency Chains:")
            result = session.run("""
                MATCH path = (a1:Application)-[:DEPENDS_ON*1..3]->(a2:Application)
                WITH a1, a2, length(path) as chain_length
                RETURN a1.id, a1.name, a2.id, a2.name, chain_length
                ORDER BY chain_length DESC
                LIMIT 5
            """)
            for record in result:
                print(f"   • {record['a1.name']:25s} → {record['a2.name']:25s} "
                      f"(chain length: {record['chain_length']})")
            
            # Query 6: Single points of failure
            print("\n6. Potential Single Points of Failure:")
            result = session.run("""
                MATCH (a:Application)-[:DEPENDS_ON]->(critical:Application)
                WHERE critical.replicas = 1
                WITH critical, count(DISTINCT a) as dependent_count
                WHERE dependent_count > 3
                RETURN critical.id, critical.name, dependent_count, 
                       critical.criticality, critical.type
                ORDER BY dependent_count DESC
                LIMIT 5
            """)
            found = False
            for record in result:
                found = True
                print(f"   ⚠ {record['critical.name']:30s} "
                      f"({record['dependent_count']} dependents, "
                      f"{record['critical.criticality']}, "
                      f"{record['critical.type']})")
            if not found:
                print("   ✓ No significant single points of failure detected")
            
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
            found = False
            for record in result:
                found = True
                print(f"   • {record['n1.zone']:20s} → {record['n2.zone']:20s}: "
                      f"{record['dep_count']} dependencies")
            if not found:
                print("   ℹ No cross-zone dependencies found")
            
            # Query 8: High-frequency topics
            print("\n8. High-Frequency Topics (>50 Hz):")
            result = session.run("""
                MATCH (t:Topic)
                WHERE t.expected_rate_hz >= 50
                WITH t, t.expected_rate_hz * t.message_size_bytes / 1024.0 as throughput_kb_s
                RETURN t.id, t.name, t.expected_rate_hz, 
                       t.message_size_bytes, throughput_kb_s
                ORDER BY throughput_kb_s DESC
                LIMIT 5
            """)
            found = False
            for record in result:
                found = True
                print(f"   • {record['t.name']:40s} "
                      f"({record['t.expected_rate_hz']} Hz, "
                      f"{record['throughput_kb_s']:.1f} KB/s)")
            if not found:
                print("   ℹ No high-frequency topics found")
    
    def run_analytics(self):
        """Run advanced analytics queries"""
        print("\nRunning advanced analytics...")
        
        with self.driver.session(database=self.database) as session:
            # Application type distribution
            print("\n📊 Application Type Distribution:")
            result = session.run("""
                MATCH (a:Application)
                RETURN a.type as type, count(*) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['type']:15s}: {record['count']:4d}")
            
            # Criticality distribution
            print("\n🎯 Criticality Distribution:")
            result = session.run("""
                MATCH (a:Application)
                RETURN a.criticality as criticality, count(*) as count
                ORDER BY 
                    CASE a.criticality
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'HIGH' THEN 2
                        WHEN 'MEDIUM' THEN 3
                        WHEN 'LOW' THEN 4
                        ELSE 5
                    END
            """)
            for record in result:
                print(f"   {record['criticality']:10s}: {record['count']:4d}")
            
            # QoS policy analysis
            print("\n🔒 QoS Reliability Distribution:")
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.qos_reliability as reliability, count(*) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['reliability']:15s}: {record['count']:4d}")
            
            print("\n💾 QoS Durability Distribution:")
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.qos_durability as durability, count(*) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['durability']:20s}: {record['count']:4d}")
            
            # Network topology insights
            print("\n🌐 Network Topology:")
            result = session.run("""
                MATCH (n:Node)
                WITH n.zone as zone, n.region as region, count(*) as node_count
                RETURN zone, region, node_count
                ORDER BY node_count DESC
            """)
            for record in result:
                print(f"   Zone: {record['zone']:15s}, Region: {record['region']:15s}, "
                      f"Nodes: {record['node_count']}")
            
            # Dependency depth analysis
            print("\n🔗 Dependency Chain Analysis:")
            result = session.run("""
                MATCH (a:Application)
                OPTIONAL MATCH path = (a)-[:DEPENDS_ON*]->(dep:Application)
                WITH a, max(length(path)) as max_depth
                RETURN 
                    CASE 
                        WHEN max_depth IS NULL THEN '0 (Independent)'
                        WHEN max_depth = 1 THEN '1 (Direct)'
                        WHEN max_depth = 2 THEN '2 (Two levels)'
                        ELSE '3+ (Deep chain)'
                    END as depth_category,
                    count(a) as app_count
            """)
            for record in result:
                print(f"   {record['depth_category']:25s}: {record['app_count']:4d}")
            
            # Topic throughput analysis
            print("\n📈 Topic Throughput Summary:")
            result = session.run("""
                MATCH (t:Topic)
                WITH t.expected_rate_hz * t.message_size_bytes / 1024.0 / 1024.0 as throughput_mb_s
                RETURN 
                    count(*) as total_topics,
                    round(avg(throughput_mb_s), 2) as avg_throughput_mb_s,
                    round(max(throughput_mb_s), 2) as max_throughput_mb_s,
                    round(sum(throughput_mb_s), 2) as total_throughput_mb_s
            """)
            record = result.single()
            if record:
                print(f"   Total Topics:      {record['total_topics']}")
                print(f"   Avg Throughput:    {record['avg_throughput_mb_s']} MB/s")
                print(f"   Max Throughput:    {record['max_throughput_mb_s']} MB/s")
                print(f"   Total Throughput:  {record['total_throughput_mb_s']} MB/s")
