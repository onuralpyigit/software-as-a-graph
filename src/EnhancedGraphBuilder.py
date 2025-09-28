#!/usr/bin/env python3
"""
Enhanced GraphBuilder with complete node/edge properties and validation
Integrates with existing implementation while adding missing features
"""

from neo4j import GraphDatabase
import pandas as pd
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random

from src.DatasetGenerator import DatasetGenerator

class EnhancedGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize enhanced graph builder with Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.generator = DatasetGenerator()
        self.dataset = None
        self.rng = np.random.RandomState(42)
        
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def execute_cypher(self, query: str, parameters: Dict = None):
        """Execute Cypher query with parameters"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph"""
        self.execute_cypher("MATCH (n) DETACH DELETE n")
        print("Graph cleared successfully")
    
    def create_enhanced_constraints(self):
        """Create all necessary constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.type)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.status)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.zone)",
            
            # Composite indexes
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.type, a.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.durability, t.reliability)"
        ]
        
        for constraint in constraints:
            try:
                self.execute_cypher(constraint)
            except Exception as e:
                # Index might already exist
                pass
        
        print("✓ Constraints and indexes created")

    def generate_synthetic_dataset(self, config: Dict) -> Dict:
        self.dataset = self.generator.generate_dataset(config)
    
    def import_enhanced_dataset(self, dataset_file: str):
        """Import dataset with enhanced properties and validation"""
        print(f"\nImporting dataset from {dataset_file}")
        
        # Load dataset
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)

        self.import_dataset_to_neo4j()

    def import_dataset_to_neo4j(self):    
        # Clear existing graph
        self.clear_graph()
        
        # Create constraints first
        self.create_enhanced_constraints()
        
        # Print metadata
        metadata = self.dataset.get('metadata', {})
        print(f"Scenario: {metadata.get('scenario', 'Unknown')}")
        print(f"Scale: {metadata.get('scale', 'Unknown')}")
        print(f"Expected nodes: {metadata.get('statistics', {}).get('total_nodes', 'Unknown')}")
        
        # Import in batches for performance
        batch_size = 1000
        
        # Import nodes with enhanced properties
        self._import_infrastructure_nodes_batch(batch_size)
        self._import_applications_batch(batch_size)
        self._import_topics_batch(batch_size)
        self._import_brokers_batch(batch_size)
        
        # Import relationships with enhanced properties
        self._import_relationships_batch(batch_size)
        
        # Derive additional relationships
        self.derive_enhanced_relationships()
        
        # Calculate and update computed properties
        self._update_application_types()
        self._calculate_topic_fanout()
        self._update_node_utilization()
        
        # Validate the imported graph
        is_valid, issues = self.validate_graph()
        if is_valid:
            print("✓ Graph validation passed")
        else:
            print("⚠ Graph validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Print statistics
        self.print_statistics()
        
        return True
    
    def _import_infrastructure_nodes_batch(self, batch_size: int):
        """Import infrastructure nodes with enhanced properties"""
        nodes = self.dataset.get('nodes', [])
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (n:Node {
                    id: item.id,
                    name: item.name,
                    hostname: item.name,
                    type: coalesce(item.type, 'COMPUTE'),
                    
                    // Resources
                    cpu_capacity: item.cpu_capacity,
                    memory_gb: item.memory_gb,
                    storage_gb: coalesce(item.storage_gb, 1000),
                    network_bandwidth_mbps: item.network_bandwidth_mbps,
                    
                    // Location
                    zone: coalesce(item.zone, 'default-zone'),
                    datacenter: coalesce(item.datacenter, 'default-dc'),
                    rack: coalesce(item.rack, 'default-rack'),
                    region: coalesce(item.region, 'us-east-1'),
                    
                    // Status
                    status: coalesce(item.status, 'active'),
                    health_score: coalesce(item.health_score, 1.0),
                    
                    // Utilization (will be calculated)
                    cpu_utilization: 0.0,
                    memory_utilization: 0.0,
                    network_utilization: 0.0,
                    
                    // Metadata
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(nodes))}/{len(nodes)} infrastructure nodes")
    
    def _import_applications_batch(self, batch_size: int):
        """Import applications with complete properties"""
        apps = self.dataset.get('applications', [])
        
        for i in range(0, len(apps), batch_size):
            batch = apps[i:i + batch_size]
            
            # Enhance application properties
            for app in batch:
                # Add missing QoS requirements
                if 'qos_requirements' not in app:
                    app['qos_requirements'] = self._generate_qos_requirements(
                        app.get('criticality_score', 0.5)
                    )
                
                # Add resource requirements
                if 'resource_requirements' not in app:
                    app['resource_requirements'] = self._generate_resource_requirements(
                        app.get('criticality_score', 0.5),
                        app.get('type', 'PROSUMER')
                    )
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (a:Application {
                    id: item.id,
                    name: item.name,
                    type: coalesce(item.type, 'UNKNOWN'),
                    criticality_score: item.criticality_score,
                    
                    // Version info
                    version: coalesce(item.version, '1.0.0'),
                    replicas: coalesce(item.replicas, 1),
                    
                    // QoS Requirements
                    qos_latency_ms: item.qos_requirements.latency_ms,
                    qos_throughput_mbps: item.qos_requirements.throughput_mbps,
                    qos_availability_percent: item.qos_requirements.availability_percent,
                    
                    // Resource Requirements
                    cpu_cores: item.resource_requirements.cpu_cores,
                    memory_gb: item.resource_requirements.memory_gb,
                    storage_gb: item.resource_requirements.storage_gb,
                    
                    // Metadata
                    owner: coalesce(item.owner, 'system'),
                    deployment_date: datetime(),
                    last_update: datetime(),
                    
                    // Metrics (will be calculated)
                    message_rate_in: 0.0,
                    message_rate_out: 0.0,
                    error_rate: 0.0,
                    avg_processing_time_ms: 0.0
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(apps))}/{len(apps)} applications")
    
    def _import_topics_batch(self, batch_size: int):
        """Import topics with enhanced QoS properties"""
        topics = self.dataset.get('topics', [])
        
        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (t:Topic {
                    id: item.id,
                    name: item.name,
                    
                    // QoS Policies
                    durability: item.qos.durability,
                    reliability: item.qos.reliability,
                    transport_priority: item.qos.transport_priority,
                    deadline_ms: item.qos.deadline_ms,
                    lifespan_ms: item.qos.lifespan_ms,
                    history_depth: item.qos.history_depth,
                    
                    // Characteristics
                    criticality_score: item.criticality_score,
                    partition_count: coalesce(item.partition_count, 1),
                    message_pattern: coalesce(item.message_pattern, 'EVENT_DRIVEN'),
                    
                    // Schema
                    schema_type: coalesce(item.schema_type, 'JSON'),
                    schema_version: coalesce(item.schema_version, '1.0'),
                    
                    // Metrics (will be calculated)
                    avg_message_size_kb: 0.0,
                    messages_per_second: 0.0,
                    total_throughput_mbps: 0.0,
                    publisher_count: 0,
                    subscriber_count: 0,
                    fanout_ratio: 0.0,
                    
                    // Metadata
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(topics))}/{len(topics)} topics")
    
    def _import_brokers_batch(self, batch_size: int):
        """Import brokers with complete configuration"""
        brokers = self.dataset.get('brokers', [])
        
        for i in range(0, len(brokers), batch_size):
            batch = brokers[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (b:Broker {
                    id: item.id,
                    name: item.name,
                    type: item.type,
                    
                    // Capacity
                    max_topics: item.max_topics,
                    max_connections: item.max_connections,
                    max_throughput_mbps: item.max_throughput_mbps,
                    
                    // Configuration
                    replication_factor: item.replication_factor,
                    partition_count: item.partition_count,
                    retention_hours: item.retention_hours,
                    
                    // Performance Metrics (will be updated)
                    avg_latency_ms: 0.0,
                    current_load_percent: 0.0,
                    uptime_percent: 100.0,
                    active_connections: 0,
                    topics_count: 0,
                    
                    // Metadata
                    version: item.version,
                    created_at: datetime(),
                    status: 'active'
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(brokers))}/{len(brokers)} brokers")
    
    def _import_relationships_batch(self, batch_size: int):
        """Import relationships with enhanced properties"""
        relationships = self.dataset.get('relationships', {})
        
        # Import RUNS_ON relationships
        runs_on = relationships.get('runs_on', [])
        for i in range(0, len(runs_on), batch_size):
            batch = runs_on[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (source {id: rel.from})
                MATCH (target:Node {id: rel.to})
                CREATE (source)-[r:RUNS_ON {
                    created_at: datetime(),
                    resource_allocation: coalesce(rel.resource_allocation, 1.0)
                }]->(target)
            """, {"batch": batch})
        
        print(f"  Imported {len(runs_on)} RUNS_ON relationships")
        
        # Import PUBLISHES_TO relationships with enhanced properties
        publishes = relationships.get('publishes_to', [])
        for i in range(0, len(publishes), batch_size):
            batch = publishes[i:i + batch_size]
            
            # Enhance each relationship
            for rel in batch:
                rel['pattern'] = self._detect_message_pattern(rel.get('period_ms', 0))
                rel['reliability'] = 'RELIABLE' if rel.get('msg_size', 0) > 1024 else 'BEST_EFFORT'
                rel['priority'] = random.randint(1, 10)
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (a:Application {id: rel.from})
                MATCH (t:Topic {id: rel.to})
                CREATE (a)-[r:PUBLISHES_TO {
                    msg_size: coalesce(rel.msg_size, 512),
                    period_ms: coalesce(rel.period_ms, 1000),
                    msg_rate_hz: CASE 
                        WHEN rel.period_ms > 0 THEN 1000.0 / rel.period_ms 
                        ELSE 0.0 
                    END,
                    pattern: rel.pattern,
                    reliability: rel.reliability,
                    priority: rel.priority,
                    
                    // Metrics
                    messages_sent: 0,
                    messages_failed: 0,
                    avg_latency_ms: 0.0,
                    max_latency_ms: 0.0,
                    error_rate: 0.0,
                    
                    // Metadata
                    created_at: datetime(),
                    last_message_at: datetime()
                }]->(t)
            """, {"batch": batch})
        
        print(f"  Imported {len(publishes)} PUBLISHES_TO relationships")
        
        # Import SUBSCRIBES_TO relationships with enhanced properties
        subscribes = relationships.get('subscribes_to', [])
        for i in range(0, len(subscribes), batch_size):
            batch = subscribes[i:i + batch_size]
            
            # Enhance each relationship
            for rel in batch:
                rel['ack_mode'] = random.choice(['AUTO', 'MANUAL'])
                rel['subscription_type'] = random.choice(['EXCLUSIVE', 'SHARED', 'FAILOVER'])
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (a:Application {id: rel.from})
                MATCH (t:Topic {id: rel.to})
                CREATE (a)-[r:SUBSCRIBES_TO {
                    filter_expression: coalesce(rel.filter, '*'),
                    consumption_rate: coalesce(rel.consumption_rate, 100.0),
                    acknowledgment_mode: rel.ack_mode,
                    offset_management: 'STORED',
                    subscription_type: rel.subscription_type,
                    
                    // Metrics
                    messages_received: 0,
                    messages_processed: 0,
                    messages_failed: 0,
                    avg_processing_time_ms: 0.0,
                    lag: 0,
                    
                    // Metadata
                    created_at: datetime(),
                    last_message_at: datetime()
                }]->(t)
            """, {"batch": batch})
        
        print(f"  Imported {len(subscribes)} SUBSCRIBES_TO relationships")
        
        # Import ROUTES relationships
        routes = relationships.get('routes', [])
        for i in range(0, len(routes), batch_size):
            batch = routes[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (b:Broker {id: rel.from})
                MATCH (t:Topic {id: rel.to})
                CREATE (b)-[r:ROUTES {
                    partition_assignment: coalesce(rel.partitions, '0-*'),
                    routing_policy: coalesce(rel.policy, 'ROUND_ROBIN'),
                    created_at: datetime()
                }]->(t)
            """, {"batch": batch})
        
        print(f"  Imported {len(routes)} ROUTES relationships")
    
    def derive_enhanced_relationships(self):
        """Derive DEPENDS_ON and CONNECTS_TO with calculated metrics"""
        print("\nDeriving enhanced relationships...")
        
        # DEPENDS_ON with strength calculation
        self.execute_cypher("""
            MATCH (a1:Application)-[rp:PUBLISHES_TO]->(t:Topic)<-[rs:SUBSCRIBES_TO]-(a2:Application)
            WITH a1, a2, t, rp, rs,
                 t.criticality_score as topic_criticality,
                 rp.msg_rate_hz * rp.msg_size as throughput_bps,
                 CASE 
                    WHEN t.reliability = 'RELIABLE' AND t.durability = 'PERSISTENT' THEN 1.0
                    WHEN t.reliability = 'RELIABLE' THEN 0.75
                    WHEN t.durability = 'PERSISTENT' THEN 0.75
                    ELSE 0.5
                 END as strength
            MERGE (a2)-[rd:DEPENDS_ON]->(a1)
            SET rd.type = 'DATA',
                rd.strength = strength,
                rd.criticality = CASE
                    WHEN topic_criticality > 0.8 THEN 'CRITICAL'
                    WHEN topic_criticality > 0.5 THEN 'REQUIRED'
                    ELSE 'OPTIONAL'
                END,
                rd.topic = t.name,
                rd.topic_id = t.id,
                rd.throughput_bps = throughput_bps,
                rd.latency_requirement_ms = t.deadline_ms,
                rd.qos_reliability = t.reliability,
                rd.created_at = datetime()
        """)
        
        deps = self.execute_cypher("MATCH ()-[d:DEPENDS_ON]->() RETURN count(d) as count")[0]['count']
        print(f"  Created {deps} DEPENDS_ON relationships")
        
        # CONNECTS_TO with aggregated metrics
        self.execute_cypher("""
            MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)-[d:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
            WHERE n1 <> n2
            WITH n1, n2, 
                 COUNT(DISTINCT d) as num_dependencies,
                 SUM(d.throughput_bps) as total_throughput,
                 MAX(CASE 
                    WHEN d.criticality = 'CRITICAL' THEN 1.0
                    WHEN d.criticality = 'REQUIRED' THEN 0.7
                    ELSE 0.3
                 END) as max_criticality,
                 MIN(d.latency_requirement_ms) as min_latency_req
            MERGE (n1)-[c:CONNECTS_TO]->(n2)
            SET c.num_dependencies = num_dependencies,
                c.total_throughput_bps = total_throughput,
                c.bandwidth_gbps = total_throughput / 1000000000.0,
                c.criticality = max_criticality,
                c.latency_ms = CASE
                    WHEN min_latency_req < 10 THEN 0.5
                    WHEN min_latency_req < 100 THEN 1.0
                    ELSE 5.0
                END,
                c.packet_loss_percent = 0.01,
                c.connection_type = 'DATACENTER',
                c.reliability = 0.999,
                c.connection_strength = num_dependencies * max_criticality,
                c.created_at = datetime()
        """)
        
        conns = self.execute_cypher("MATCH ()-[c:CONNECTS_TO]->() RETURN count(c) as count")[0]['count']
        print(f"  Created {conns} CONNECTS_TO relationships")
    
    def _update_application_types(self):
        """Update application types based on pub/sub relationships"""
        self.execute_cypher("""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO]->()
            WITH a, count(*) as pub_count
            OPTIONAL MATCH (a)-[:SUBSCRIBES_TO]->()
            WITH a, pub_count, count(*) as sub_count
            SET a.type = CASE 
                WHEN pub_count > 0 AND sub_count > 0 THEN 'PROSUMER'
                WHEN pub_count > 0 THEN 'PRODUCER'
                WHEN sub_count > 0 THEN 'CONSUMER'
                ELSE 'UNKNOWN'
            END
        """)
        print("  Updated application types")
    
    def _calculate_topic_fanout(self):
        """Calculate fanout ratio for topics"""
        self.execute_cypher("""
            MATCH (t:Topic)
            OPTIONAL MATCH (t)<-[:PUBLISHES_TO]-(pub:Application)
            WITH t, count(DISTINCT pub) as pub_count
            OPTIONAL MATCH (t)<-[:SUBSCRIBES_TO]-(sub:Application)
            WITH t, pub_count, count(DISTINCT sub) as sub_count
            SET t.publisher_count = pub_count,
                t.subscriber_count = sub_count,
                t.fanout_ratio = CASE 
                    WHEN pub_count > 0 THEN toFloat(sub_count) / pub_count 
                    ELSE 0.0 
                END
        """)
        print("  Calculated topic fanout metrics")
    
    def _update_node_utilization(self):
        """Calculate node resource utilization"""
        self.execute_cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)<-[:RUNS_ON]-(a:Application)
            WITH n, 
                 sum(a.cpu_cores) as used_cpu,
                 sum(a.memory_gb) as used_memory,
                 sum(a.qos_throughput_mbps) as used_bandwidth
            SET n.cpu_utilization = CASE 
                    WHEN n.cpu_capacity > 0 THEN used_cpu / n.cpu_capacity 
                    ELSE 0.0 
                END,
                n.memory_utilization = CASE 
                    WHEN n.memory_gb > 0 THEN used_memory / n.memory_gb 
                    ELSE 0.0 
                END,
                n.network_utilization = CASE 
                    WHEN n.network_bandwidth_mbps > 0 THEN used_bandwidth / n.network_bandwidth_mbps 
                    ELSE 0.0 
                END
        """)
        print("  Updated node utilization metrics")
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Comprehensive graph validation"""
        errors = []
        warnings = []
        
        # Check for orphaned topics
        orphaned = self.execute_cypher("""
            MATCH (t:Topic)
            WHERE NOT (()-[:PUBLISHES_TO]->(t)) OR NOT (()-[:SUBSCRIBES_TO]->(t))
            RETURN collect(t.name) as topics
        """)[0]['topics']
        
        if orphaned:
            warnings.append(f"Found {len(orphaned)} orphaned topics: {orphaned[:5]}")
        
        # Check for isolated applications
        isolated = self.execute_cypher("""
            MATCH (a:Application)
            WHERE NOT (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->()
            RETURN collect(a.name) as apps
        """)[0]['apps']
        
        if isolated:
            errors.append(f"Found {len(isolated)} isolated applications: {isolated[:5]}")
        
        # Check for circular dependencies (depth 2)
        circular = self.execute_cypher("""
            MATCH (a:Application)-[:DEPENDS_ON*2]->(a)
            RETURN collect(DISTINCT a.name) as apps
        """)[0]['apps']
        
        if circular:
            warnings.append(f"Circular dependencies detected: {circular[:5]}")
        
        # Check for overutilized nodes
        overutilized = self.execute_cypher("""
            MATCH (n:Node)
            WHERE n.cpu_utilization > 0.9 OR n.memory_utilization > 0.9
            RETURN collect(n.name) as nodes
        """)[0]['nodes']
        
        if overutilized:
            warnings.append(f"Overutilized nodes: {overutilized}")
        
        # Check QoS compatibility
        qos_issues = self.execute_cypher("""
            MATCH (a1:Application)-[:DEPENDS_ON]->(a2:Application)
            WHERE a1.qos_latency_ms < a2.qos_latency_ms
            RETURN count(*) as count
        """)[0]['count']
        
        if qos_issues > 0:
            warnings.append(f"Found {qos_issues} QoS compatibility issues")
        
        is_valid = len(errors) == 0
        all_issues = errors + warnings
        
        return is_valid, all_issues
    
    def print_statistics(self):
        """Print comprehensive graph statistics"""
        stats = self.execute_cypher("""
            MATCH (a:Application) WITH count(a) as apps
            MATCH (t:Topic) WITH apps, count(t) as topics
            MATCH (n:Node) WITH apps, topics, count(n) as nodes
            MATCH (b:Broker) WITH apps, topics, nodes, count(b) as brokers
            MATCH ()-[p:PUBLISHES_TO]->() WITH apps, topics, nodes, brokers, count(p) as pubs
            MATCH ()-[s:SUBSCRIBES_TO]->() WITH apps, topics, nodes, brokers, pubs, count(s) as subs
            MATCH ()-[d:DEPENDS_ON]->() WITH apps, topics, nodes, brokers, pubs, subs, count(d) as deps
            MATCH ()-[c:CONNECTS_TO]->() WITH apps, topics, nodes, brokers, pubs, subs, deps, count(c) as conns
            RETURN apps, topics, nodes, brokers, pubs, subs, deps, conns
        """)[0]
        
        print("\n📊 Graph Statistics:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Applications: {stats['apps']}")
        print(f"  Topics: {stats['topics']}")
        print(f"  Brokers: {stats['brokers']}")
        print(f"  PUBLISHES_TO: {stats['pubs']}")
        print(f"  SUBSCRIBES_TO: {stats['subs']}")
        print(f"  DEPENDS_ON: {stats['deps']}")
        print(f"  CONNECTS_TO: {stats['conns']}")
        
        # Additional metrics
        metrics = self.execute_cypher("""
            MATCH (a:Application)
            WITH avg(a.criticality_score) as avg_app_criticality,
                 count(CASE WHEN a.type = 'PRODUCER' THEN 1 END) as producers,
                 count(CASE WHEN a.type = 'CONSUMER' THEN 1 END) as consumers,
                 count(CASE WHEN a.type = 'PROSUMER' THEN 1 END) as prosumers
            MATCH (t:Topic)
            WITH avg_app_criticality, producers, consumers, prosumers,
                 avg(t.criticality_score) as avg_topic_criticality,
                 avg(t.fanout_ratio) as avg_fanout
            MATCH (n:Node)
            WITH avg_app_criticality, producers, consumers, prosumers,
                 avg_topic_criticality, avg_fanout,
                 avg(n.cpu_utilization) as avg_cpu_util,
                 avg(n.memory_utilization) as avg_mem_util
            RETURN avg_app_criticality, producers, consumers, prosumers,
                   avg_topic_criticality, avg_fanout, avg_cpu_util, avg_mem_util
        """)[0]
        
        print("\n📈 Key Metrics:")
        print(f"  Application Types: {metrics['producers']} producers, {metrics['consumers']} consumers, {metrics['prosumers']} prosumers")
        print(f"  Avg Application Criticality: {metrics['avg_app_criticality']:.2f}")
        print(f"  Avg Topic Criticality: {metrics['avg_topic_criticality']:.2f}")
        print(f"  Avg Topic Fanout: {metrics['avg_fanout']:.2f}")
        print(f"  Avg CPU Utilization: {metrics['avg_cpu_util']:.1%}")
        print(f"  Avg Memory Utilization: {metrics['avg_mem_util']:.1%}")
    
    # Helper methods
    def _detect_message_pattern(self, period_ms: float) -> str:
        """Detect message pattern from period"""
        if period_ms == 0:
            return 'EVENT_DRIVEN'
        elif period_ms < 100:
            return 'BURST'
        elif period_ms % 1000 == 0:
            return 'PERIODIC'
        else:
            return 'RANDOM'
    
    def _generate_qos_requirements(self, criticality: float) -> Dict:
        """Generate QoS requirements based on criticality"""
        if criticality > 0.8:
            return {
                "latency_ms": random.uniform(10, 50),
                "throughput_mbps": random.uniform(50, 100),
                "availability_percent": 99.9
            }
        elif criticality > 0.5:
            return {
                "latency_ms": random.uniform(50, 200),
                "throughput_mbps": random.uniform(10, 50),
                "availability_percent": 99.0
            }
        else:
            return {
                "latency_ms": random.uniform(200, 1000),
                "throughput_mbps": random.uniform(1, 10),
                "availability_percent": 95.0
            }
    
    def _generate_resource_requirements(self, criticality: float, app_type: str) -> Dict:
        """Generate resource requirements based on criticality and type"""
        base_cpu = 0.5 if app_type == 'CONSUMER' else 1.0
        base_memory = 1.0 if app_type == 'CONSUMER' else 2.0
        
        multiplier = 1.0 + criticality * 2.0
        
        return {
            "cpu_cores": round(base_cpu * multiplier, 1),
            "memory_gb": round(base_memory * multiplier, 1),
            "storage_gb": round(10 * multiplier, 1)
        }
    
    # Export methods for compatibility
    def export_graph(self, file_dir: str):
        """Export graph to CSV files"""
        vertices_file = f"{file_dir}/nodes.csv"
        edges_file = f"{file_dir}/edges.csv"
        
        self.export_vertices_to_csv(vertices_file)
        self.export_edges_to_csv(edges_file)
        
        print(f"Graph exported to {file_dir}")
    
    def export_vertices_to_csv(self, file_path: str):
        """Export vertices to CSV with enhanced properties"""
        records = self.execute_cypher("""
            MATCH (n)
            RETURN n.id AS id, 
                   n.name AS name, 
                   labels(n)[0] AS type,
                   n.criticality_score AS criticality,
                   n.durability AS durability,
                   n.reliability AS reliability,
                   CASE 
                       WHEN 'Application' IN labels(n) THEN n.qos_latency_ms
                       WHEN 'Topic' IN labels(n) THEN n.deadline_ms
                       ELSE null
                   END AS latency,
                   CASE
                       WHEN 'Node' IN labels(n) THEN n.cpu_utilization
                       ELSE null
                   END AS utilization
        """)
        
        vertices_df = pd.DataFrame(records)
        vertices_df.to_csv(file_path, index=False)
        print(f"  Exported {len(vertices_df)} vertices to {file_path}")
    
    def export_edges_to_csv(self, file_path: str):
        """Export edges to CSV with enhanced properties"""
        records = self.execute_cypher("""
            MATCH ()-[r]->()
            RETURN startNode(r).id AS source, 
                   endNode(r).id AS target, 
                   type(r) AS relationship,
                   r.strength AS strength,
                   r.criticality AS criticality,
                   r.throughput_bps AS throughput,
                   r.latency_ms AS latency
        """)
        
        edges_df = pd.DataFrame(records)
        edges_df.to_csv(file_path, index=False)
        print(f"  Exported {len(edges_df)} edges to {file_path}")