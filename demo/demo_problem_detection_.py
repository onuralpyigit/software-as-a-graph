#!/usr/bin/env python3
"""
Direct Neo4j Importer with Enhanced Properties and Problem Pattern Detection
Imports datasets with full property enhancement and identifies anti-patterns
"""

import json
import time
import argparse
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from neo4j import GraphDatabase
import numpy as np

class DirectNeo4jImporter:
    """Direct importer for datasets with enhanced properties and pattern detection"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.import_stats = {}
        self.detected_problems = []
        
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def execute_cypher(self, query: str, parameters: Dict = None):
        """Execute Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def import_dataset_with_detection(self, dataset_file: str, detect_problems: bool = True) -> Dict:
        """
        Import dataset with enhanced properties and optional problem detection
        """
        print(f"\n{'='*70}")
        print(f"IMPORTING DATASET: {dataset_file}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Load dataset
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # Print metadata
        metadata = dataset.get('metadata', {})
        print(f"\n📋 Dataset Information:")
        print(f"  Scenario: {metadata.get('scenario', 'Unknown')}")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        
        if 'antipattern' in metadata:
            print(f"  ⚠️  Anti-pattern: {metadata.get('antipattern')}")
            print(f"  Problems:")
            for problem in metadata.get('problems', []):
                print(f"    - {problem}")
        
        # Clear and setup
        self._clear_and_setup()
        
        # Import with enhancements
        self._import_nodes_enhanced(dataset.get('nodes', []))
        self._import_applications_enhanced(dataset.get('applications', []))
        self._import_topics_enhanced(dataset.get('topics', []))
        self._import_brokers_enhanced(dataset.get('brokers', []))
        self._import_relationships_enhanced(dataset.get('relationships', {}))
        
        # Derive additional relationships
        self._derive_complex_relationships()
        
        # Calculate metrics
        self._calculate_advanced_metrics()
        
        import_time = time.time() - start_time
        
        # Detect problems if requested
        if detect_problems:
            self.detected_problems = self._detect_problem_patterns()
        
        # Generate import report
        report = self._generate_import_report(dataset_file, import_time)
        
        # Print summary
        self._print_import_summary(report)
        
        return report
    
    def _clear_and_setup(self):
        """Clear graph and create constraints/indexes"""
        print("\n🔧 Setting up database...")
        
        # Clear existing data
        self.execute_cypher("MATCH (n) DETACH DELETE n")
        
        # Create constraints
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.execute_cypher(constraint)
            except:
                pass  # Constraint might already exist
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.zone)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[d:DEPENDS_ON]-() ON (d.strength)"
        ]
        
        for index in indexes:
            try:
                self.execute_cypher(index)
            except:
                pass
        
        print("  ✓ Database setup complete")
    
    def _import_nodes_enhanced(self, nodes: List[Dict]):
        """Import nodes with enhanced properties"""
        if not nodes:
            return
        
        print(f"\n📦 Importing {len(nodes)} infrastructure nodes...")
        
        for node in nodes:
            # Add enhanced properties
            enhanced = {
                **node,
                "cpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "network_utilization": 0.0,
                "health_score": 1.0,
                "risk_score": 0.0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            self.execute_cypher("""
                CREATE (n:Node {
                    id: $id,
                    name: $name,
                    cpu_capacity: $cpu_capacity,
                    memory_gb: $memory_gb,
                    network_bandwidth_mbps: $network_bandwidth_mbps,
                    zone: $zone,
                    cpu_utilization: $cpu_utilization,
                    memory_utilization: $memory_utilization,
                    network_utilization: $network_utilization,
                    health_score: $health_score,
                    risk_score: $risk_score,
                    created_at: $created_at,
                    last_updated: $last_updated
                })
            """, enhanced)
        
        self.import_stats['nodes'] = len(nodes)
        print(f"  ✓ Imported {len(nodes)} nodes")
    
    def _import_applications_enhanced(self, apps: List[Dict]):
        """Import applications with enhanced properties and problem flags"""
        if not apps:
            return
        
        print(f"\n📱 Importing {len(apps)} applications...")
        
        for app in apps:
            # Determine application characteristics
            is_critical = app.get('criticality_score', 0) > 0.8
            is_singleton = app.get('replicas', 1) == 1
            has_spof = app.get('single_instance', False)
            
            # Calculate resource requirements
            criticality = app.get('criticality_score', 0.5)
            base_cpu = 1.0 if criticality < 0.5 else 2.0
            base_memory = 2.0 if criticality < 0.5 else 4.0
            
            enhanced = {
                **app,
                "qos_latency_ms": 100 if is_critical else 1000,
                "qos_throughput_mbps": 50 if is_critical else 10,
                "qos_availability_percent": 99.9 if is_critical else 95.0,
                "cpu_cores": base_cpu * (1 + criticality),
                "memory_gb": base_memory * (1 + criticality),
                "storage_gb": 10.0,
                "is_critical": is_critical,
                "is_singleton": is_singleton,
                "has_spof": has_spof,
                "message_rate_in": 0.0,
                "message_rate_out": 0.0,
                "error_rate": 0.0,
                "created_at": datetime.now().isoformat()
            }
            
            # Add problem pattern flags if present
            problem_flags = []
            if app.get('circular_dependency_chain'):
                problem_flags.append('CIRCULAR_DEPENDENCY')
            if app.get('chatty'):
                problem_flags.append('CHATTY_COMMUNICATION')
            if app.get('hidden_dependency'):
                problem_flags.append('HIDDEN_COUPLING')
            if app.get('confused_by_topics'):
                problem_flags.append('TOPIC_SPRAWL')
            if app.get('overwhelmed'):
                problem_flags.append('OVERLOADED')
            
            enhanced['problem_flags'] = ','.join(problem_flags) if problem_flags else 'NONE'
            
            self.execute_cypher("""
                CREATE (a:Application {
                    id: $id,
                    name: $name,
                    criticality_score: $criticality_score,
                    replicas: $replicas,
                    type: $type,
                    owner: $owner,
                    version: $version,
                    qos_latency_ms: $qos_latency_ms,
                    qos_throughput_mbps: $qos_throughput_mbps,
                    qos_availability_percent: $qos_availability_percent,
                    cpu_cores: $cpu_cores,
                    memory_gb: $memory_gb,
                    storage_gb: $storage_gb,
                    is_critical: $is_critical,
                    is_singleton: $is_singleton,
                    has_spof: $has_spof,
                    problem_flags: $problem_flags,
                    message_rate_in: $message_rate_in,
                    message_rate_out: $message_rate_out,
                    error_rate: $error_rate,
                    created_at: $created_at
                })
            """, enhanced)
        
        self.import_stats['applications'] = len(apps)
        print(f"  ✓ Imported {len(apps)} applications")
    
    def _import_topics_enhanced(self, topics: List[Dict]):
        """Import topics with enhanced QoS and problem indicators"""
        if not topics:
            return
        
        print(f"\n📨 Importing {len(topics)} topics...")
        
        for topic in topics:
            # Calculate topic characteristics
            qos = topic.get('qos', {})
            is_god_topic = topic.get('god_topic', False)
            is_hidden = topic.get('hidden', False)
            requires_ordering = topic.get('requires_ordering', False)
            
            # Calculate criticality from QoS
            qos_criticality = self._calculate_qos_criticality(qos)
            
            enhanced = {
                **topic,
                "durability": qos.get('durability', 'VOLATILE'),
                "reliability": qos.get('reliability', 'BEST_EFFORT'),
                "transport_priority": qos.get('transport_priority', 'MEDIUM'),
                "deadline_ms": qos.get('deadline_ms', 1000),
                "lifespan_ms": qos.get('lifespan_ms', 60000),
                "history_depth": qos.get('history_depth', 100),
                "qos_criticality": qos_criticality,
                "is_god_topic": is_god_topic,
                "is_hidden": is_hidden,
                "requires_ordering": requires_ordering,
                "avg_message_size_kb": topic.get('avg_message_size', 0) / 1024 if topic.get('avg_message_size') else 1.0,
                "messages_per_second": topic.get('message_rate', 0),
                "publisher_count": 0,
                "subscriber_count": 0,
                "fanout_ratio": 0.0,
                "created_at": datetime.now().isoformat()
            }
            
            # Add problem indicators
            problems = []
            if is_god_topic:
                problems.append('GOD_TOPIC')
            if topic.get('potentially_duplicate'):
                problems.append('DUPLICATE')
            if topic.get('random_topic'):
                problems.append('RANDOM_NAMING')
            if topic.get('hidden_coupling'):
                problems.append(f"HIDDEN_{topic['hidden_coupling'].upper()}")
            
            enhanced['problem_indicators'] = ','.join(problems) if problems else 'NONE'
            
            self.execute_cypher("""
                CREATE (t:Topic {
                    id: $id,
                    name: $name,
                    criticality_score: $criticality_score,
                    partition_count: $partition_count,
                    message_pattern: $message_pattern,
                    durability: $durability,
                    reliability: $reliability,
                    transport_priority: $transport_priority,
                    deadline_ms: $deadline_ms,
                    lifespan_ms: $lifespan_ms,
                    history_depth: $history_depth,
                    qos_criticality: $qos_criticality,
                    is_god_topic: $is_god_topic,
                    is_hidden: $is_hidden,
                    requires_ordering: $requires_ordering,
                    avg_message_size_kb: $avg_message_size_kb,
                    messages_per_second: $messages_per_second,
                    publisher_count: $publisher_count,
                    subscriber_count: $subscriber_count,
                    fanout_ratio: $fanout_ratio,
                    problem_indicators: $problem_indicators,
                    created_at: $created_at
                })
            """, enhanced)
        
        self.import_stats['topics'] = len(topics)
        print(f"  ✓ Imported {len(topics)} topics")
    
    def _import_brokers_enhanced(self, brokers: List[Dict]):
        """Import brokers with enhanced configuration"""
        if not brokers:
            return
        
        print(f"\n🔄 Importing {len(brokers)} brokers...")
        
        for broker in brokers:
            enhanced = {
                **broker,
                "avg_latency_ms": 0.0,
                "current_load_percent": 0.0,
                "uptime_percent": 100.0,
                "active_connections": 0,
                "topics_count": 0,
                "messages_per_sec": 0.0,
                "bytes_in_per_sec": 0.0,
                "bytes_out_per_sec": 0.0,
                "created_at": datetime.now().isoformat()
            }
            
            self.execute_cypher("""
                CREATE (b:Broker {
                    id: $id,
                    name: $name,
                    protocol: $protocol,
                    max_connections: $max_connections,
                    max_throughput_mbps: $max_throughput_mbps,
                    partition_count: $partition_count,
                    avg_latency_ms: $avg_latency_ms,
                    current_load_percent: $current_load_percent,
                    uptime_percent: $uptime_percent,
                    active_connections: $active_connections,
                    topics_count: $topics_count,
                    messages_per_sec: $messages_per_sec,
                    bytes_in_per_sec: $bytes_in_per_sec,
                    bytes_out_per_sec: $bytes_out_per_sec,
                    created_at: $created_at
                })
            """, enhanced)
        
        self.import_stats['brokers'] = len(brokers)
        print(f"  ✓ Imported {len(brokers)} brokers")
    
    def _import_relationships_enhanced(self, relationships: Dict):
        """Import relationships with enhanced properties"""
        print(f"\n🔗 Importing relationships...")
        
        # RUNS_ON relationships
        runs_on = relationships.get('runs_on', [])
        for rel in runs_on:
            self.execute_cypher("""
                MATCH (source {id: $from})
                MATCH (target:Node {id: $to})
                CREATE (source)-[r:RUNS_ON {
                    created_at: datetime(),
                    resource_allocation: 1.0
                }]->(target)
            """, rel)
        
        print(f"  ✓ RUNS_ON: {len(runs_on)}")
        
        # PUBLISHES_TO relationships
        publishes = relationships.get('publishes_to', [])
        for rel in publishes:
            # Detect patterns
            msg_size = rel.get('msg_size', 512)
            period_ms = rel.get('period_ms', 1000)
            
            pattern = self._detect_message_pattern(period_ms)
            is_chatty = rel.get('chatty', False) or (msg_size < 100 and period_ms < 500)
            
            self.execute_cypher("""
                MATCH (a:Application {id: $from})
                MATCH (t:Topic {id: $to})
                CREATE (a)-[r:PUBLISHES_TO {
                    msg_size: $msg_size,
                    period_ms: $period_ms,
                    msg_rate_hz: CASE WHEN $period_ms > 0 THEN 1000.0 / $period_ms ELSE 0.0 END,
                    bandwidth_bps: $msg_size * CASE WHEN $period_ms > 0 THEN 1000.0 / $period_ms ELSE 0.0 END * 8,
                    pattern: $pattern,
                    is_chatty: $is_chatty,
                    reliability: CASE WHEN $msg_size > 1024 THEN 'RELIABLE' ELSE 'BEST_EFFORT' END,
                    priority: CASE WHEN $period_ms < 100 THEN 10 ELSE 5 END,
                    messages_sent: 0,
                    messages_failed: 0,
                    avg_latency_ms: 0.0,
                    created_at: datetime()
                }]->(t)
            """, {
                **rel,
                'pattern': pattern,
                'is_chatty': is_chatty
            })
        
        print(f"  ✓ PUBLISHES_TO: {len(publishes)}")
        
        # SUBSCRIBES_TO relationships
        subscribes = relationships.get('subscribes_to', [])
        for rel in subscribes:
            self.execute_cypher("""
                MATCH (a:Application {id: $from})
                MATCH (t:Topic {id: $to})
                CREATE (a)-[r:SUBSCRIBES_TO {
                    filter_expression: '*',
                    consumption_rate: 100.0,
                    acknowledgment_mode: 'AUTO',
                    subscription_type: 'SHARED',
                    messages_received: 0,
                    messages_processed: 0,
                    avg_processing_time_ms: 0.0,
                    lag: 0,
                    created_at: datetime()
                }]->(t)
            """, rel)
        
        print(f"  ✓ SUBSCRIBES_TO: {len(subscribes)}")
        
        # ROUTES relationships
        routes = relationships.get('routes', [])
        for rel in routes:
            self.execute_cypher("""
                MATCH (b:Broker {id: $from})
                MATCH (t:Topic {id: $to})
                CREATE (b)-[r:ROUTES {
                    partition_assignment: '0-*',
                    routing_policy: 'ROUND_ROBIN',
                    created_at: datetime()
                }]->(t)
            """, rel)
        
        print(f"  ✓ ROUTES: {len(routes)}")
        
        self.import_stats['relationships'] = {
            'runs_on': len(runs_on),
            'publishes_to': len(publishes),
            'subscribes_to': len(subscribes),
            'routes': len(routes)
        }
    
    def _derive_complex_relationships(self):
        """Derive DEPENDS_ON and CONNECTS_TO relationships with problem detection"""
        print(f"\n🧮 Deriving complex relationships...")
        
        # Derive DEPENDS_ON with enhanced metrics
        self.execute_cypher("""
            MATCH (a1:Application)-[p:PUBLISHES_TO]->(t:Topic)<-[s:SUBSCRIBES_TO]-(a2:Application)
            WITH a1, a2, t, p,
                 CASE 
                    WHEN t.reliability = 'RELIABLE' AND t.durability = 'PERSISTENT' THEN 1.0
                    WHEN t.reliability = 'RELIABLE' THEN 0.75
                    WHEN t.durability = 'PERSISTENT' THEN 0.75
                    ELSE 0.5
                 END as strength,
                 t.criticality_score as topic_criticality,
                 p.bandwidth_bps as bandwidth
            MERGE (a2)-[d:DEPENDS_ON]->(a1)
            SET d.type = 'DATA',
                d.strength = strength,
                d.criticality = CASE
                    WHEN topic_criticality > 0.8 THEN 'CRITICAL'
                    WHEN topic_criticality > 0.5 THEN 'REQUIRED'
                    ELSE 'OPTIONAL'
                END,
                d.topic_name = t.name,
                d.topic_id = t.id,
                d.bandwidth_bps = bandwidth,
                d.latency_requirement_ms = t.deadline_ms,
                d.created_at = datetime()
        """)
        
        deps = self.execute_cypher("MATCH ()-[d:DEPENDS_ON]->() RETURN count(d) as count")[0]['count']
        print(f"  ✓ Created {deps} DEPENDS_ON relationships")
        
        # Derive CONNECTS_TO with network metrics
        self.execute_cypher("""
            MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)-[d:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
            WHERE n1 <> n2
            WITH n1, n2,
                 COUNT(DISTINCT d) as num_deps,
                 SUM(d.bandwidth_bps) as total_bandwidth,
                 MAX(CASE 
                    WHEN d.criticality = 'CRITICAL' THEN 1.0
                    WHEN d.criticality = 'REQUIRED' THEN 0.7
                    ELSE 0.3
                 END) as max_criticality,
                 MIN(d.latency_requirement_ms) as min_latency
            MERGE (n1)-[c:CONNECTS_TO]->(n2)
            SET c.num_dependencies = num_deps,
                c.bandwidth_bps = total_bandwidth,
                c.bandwidth_gbps = total_bandwidth / 1000000000.0,
                c.criticality = max_criticality,
                c.latency_ms = CASE
                    WHEN n1.zone = n2.zone THEN 0.5
                    WHEN n1.datacenter = n2.datacenter THEN 2.0
                    ELSE 10.0
                END,
                c.connection_type = CASE
                    WHEN n1.zone = n2.zone THEN 'LOCAL'
                    WHEN n1.datacenter = n2.datacenter THEN 'DATACENTER'
                    ELSE 'WAN'
                END,
                c.reliability = CASE
                    WHEN n1.zone = n2.zone THEN 0.9999
                    WHEN n1.datacenter = n2.datacenter THEN 0.999
                    ELSE 0.99
                END,
                c.created_at = datetime()
        """)
        
        conns = self.execute_cypher("MATCH ()-[c:CONNECTS_TO]->() RETURN count(c) as count")[0]['count']
        print(f"  ✓ Created {conns} CONNECTS_TO relationships")
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced metrics for all components"""
        print(f"\n📊 Calculating advanced metrics...")
        
        # Update application types
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
        
        # Calculate topic metrics
        self.execute_cypher("""
            MATCH (t:Topic)
            OPTIONAL MATCH (t)<-[:PUBLISHES_TO]-(pub)
            WITH t, count(DISTINCT pub) as pub_count
            OPTIONAL MATCH (t)<-[:SUBSCRIBES_TO]-(sub)
            WITH t, pub_count, count(DISTINCT sub) as sub_count
            SET t.publisher_count = pub_count,
                t.subscriber_count = sub_count,
                t.fanout_ratio = CASE WHEN pub_count > 0 THEN toFloat(sub_count) / pub_count ELSE 0.0 END,
                t.is_orphaned = pub_count = 0 OR sub_count = 0
        """)
        
        # Calculate node utilization
        self.execute_cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)<-[:RUNS_ON]-(a:Application)
            WITH n, 
                 sum(a.cpu_cores) as used_cpu,
                 sum(a.memory_gb) as used_memory,
                 sum(a.qos_throughput_mbps) as used_bandwidth,
                 count(a) as app_count,
                 sum(a.criticality_score) as total_criticality
            SET n.cpu_utilization = CASE WHEN n.cpu_capacity > 0 THEN used_cpu / n.cpu_capacity ELSE 0.0 END,
                n.memory_utilization = CASE WHEN n.memory_gb > 0 THEN used_memory / n.memory_gb ELSE 0.0 END,
                n.network_utilization = CASE WHEN n.network_bandwidth_mbps > 0 THEN used_bandwidth / n.network_bandwidth_mbps ELSE 0.0 END,
                n.app_count = app_count,
                n.criticality_load = total_criticality,
                n.is_overloaded = (used_cpu / n.cpu_capacity > 0.8) OR (used_memory / n.memory_gb > 0.8)
        """)
        
        # Calculate broker load
        self.execute_cypher("""
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH b, count(t) as topic_count, sum(t.messages_per_second) as total_msg_rate
            SET b.topics_count = topic_count,
                b.messages_per_sec = total_msg_rate,
                b.current_load_percent = CASE WHEN b.max_topics > 0 THEN toFloat(topic_count) / b.max_topics * 100 ELSE 0.0 END,
                b.is_overloaded = topic_count > b.max_topics * 0.8
        """)
        
        print("  ✓ Metrics calculation complete")
    
    def _detect_problem_patterns(self) -> List[Dict]:
        """Detect problematic patterns in the imported graph"""
        print(f"\n🔍 Detecting problem patterns...")
        
        problems = []
        
        # 1. Single Points of Failure
        spof = self.execute_cypher("""
            MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
            WHERE a.is_critical = true
            WITH a, collect(n) as nodes
            WHERE size(nodes) = 1
            RETURN a.name as app, a.criticality_score as criticality, nodes[0].name as single_node
        """)
        
        if spof:
            problems.append({
                'pattern': 'SINGLE_POINT_OF_FAILURE',
                'severity': 'CRITICAL',
                'count': len(spof),
                'details': f"{len(spof)} critical applications running on single nodes",
                'affected': [s['app'] for s in spof[:5]]
            })
        
        # 2. God Topics
        god_topics = self.execute_cypher("""
            MATCH (t:Topic)
            WHERE t.fanout_ratio > 10 OR t.subscriber_count > 30 OR t.is_god_topic = true
            RETURN t.name as topic, t.publisher_count as pubs, t.subscriber_count as subs, t.fanout_ratio as fanout
            ORDER BY subs DESC
        """)
        
        if god_topics:
            problems.append({
                'pattern': 'GOD_TOPIC',
                'severity': 'HIGH',
                'count': len(god_topics),
                'details': f"{len(god_topics)} topics with excessive fanout",
                'affected': [f"{g['topic']} (fanout: {g['fanout']:.1f})" for g in god_topics[:3]]
            })
        
        # 3. Circular Dependencies
        circular_deps = self.execute_cypher("""
            MATCH path = (a:Application)-[:DEPENDS_ON*2..5]->(a)
            RETURN DISTINCT a.name as app, length(path) as cycle_length
        """)
        
        if circular_deps:
            problems.append({
                'pattern': 'CIRCULAR_DEPENDENCIES',
                'severity': 'HIGH',
                'count': len(circular_deps),
                'details': f"{len(circular_deps)} applications in circular dependency chains",
                'affected': [f"{c['app']} (cycle length: {c['cycle_length']})" for c in circular_deps[:5]]
            })
        
        # 4. Chatty Communication
        chatty = self.execute_cypher("""
            MATCH (a:Application)-[p:PUBLISHES_TO]->(t:Topic)
            WHERE p.is_chatty = true OR (p.msg_size < 256 AND p.period_ms < 500)
            RETURN a.name as app, count(t) as topic_count, avg(p.msg_size) as avg_size, avg(p.period_ms) as avg_period
        """)
        
        if chatty:
            problems.append({
                'pattern': 'CHATTY_COMMUNICATION',
                'severity': 'MEDIUM',
                'count': len(chatty),
                'details': f"{len(chatty)} applications with chatty communication patterns",
                'affected': [f"{c['app']} (avg msg: {c['avg_size']:.0f}B, period: {c['avg_period']:.0f}ms)" for c in chatty[:3]]
            })
        
        # 5. Hidden Coupling
        hidden = self.execute_cypher("""
            MATCH (a:Application)
            WHERE a.problem_flags CONTAINS 'HIDDEN'
            RETURN a.name as app, a.problem_flags as flags
        """)
        
        if hidden:
            problems.append({
                'pattern': 'HIDDEN_COUPLING',
                'severity': 'MEDIUM',
                'count': len(hidden),
                'details': f"{len(hidden)} applications with hidden dependencies",
                'affected': [h['app'] for h in hidden[:5]]
            })
        
        # 6. Topic Sprawl
        sprawl = self.execute_cypher("""
            MATCH (t1:Topic), (t2:Topic)
            WHERE t1.id < t2.id AND 
                  (t1.problem_indicators CONTAINS 'DUPLICATE' OR 
                   t2.problem_indicators CONTAINS 'DUPLICATE' OR
                   t1.name CONTAINS t2.name OR t2.name CONTAINS t1.name)
            RETURN count(*) as duplicate_pairs
        """)
        
        topic_count = self.execute_cypher("MATCH (t:Topic) RETURN count(t) as count")[0]['count']
        orphaned = self.execute_cypher("MATCH (t:Topic) WHERE t.is_orphaned = true RETURN count(t) as count")[0]['count']
        
        if topic_count > 100 or orphaned > 10:
            problems.append({
                'pattern': 'TOPIC_SPRAWL',
                'severity': 'LOW' if topic_count < 200 else 'MEDIUM',
                'count': topic_count,
                'details': f"{topic_count} total topics, {orphaned} orphaned, {sprawl[0]['duplicate_pairs'] if sprawl else 0} potential duplicates",
                'affected': []
            })
        
        # 7. Resource Overload
        overload = self.execute_cypher("""
            MATCH (n:Node)
            WHERE n.is_overloaded = true
            RETURN n.name as node, n.cpu_utilization as cpu, n.memory_utilization as memory
        """)
        
        if overload:
            problems.append({
                'pattern': 'RESOURCE_OVERLOAD',
                'severity': 'HIGH',
                'count': len(overload),
                'details': f"{len(overload)} nodes with resource overload",
                'affected': [f"{o['node']} (CPU: {o['cpu']*100:.0f}%, Mem: {o['memory']*100:.0f}%)" for o in overload]
            })
        
        return problems
    
    def _generate_import_report(self, dataset_file: str, import_time: float) -> Dict:
        """Generate comprehensive import report"""
        
        # Get graph statistics
        stats = self.execute_cypher("""
            MATCH (n) WITH count(n) as total_nodes
            MATCH ()-[r]->() WITH total_nodes, count(r) as total_edges
            MATCH (a:Application) WITH total_nodes, total_edges, count(a) as apps
            MATCH (t:Topic) WITH total_nodes, total_edges, apps, count(t) as topics
            MATCH (b:Broker) WITH total_nodes, total_edges, apps, topics, count(b) as brokers
            MATCH (n:Node) WITH total_nodes, total_edges, apps, topics, brokers, count(n) as nodes
            RETURN total_nodes, total_edges, apps, topics, brokers, nodes
        """)[0]
        
        # Get problem summary
        problem_summary = {}
        for problem in self.detected_problems:
            severity = problem['severity']
            if severity not in problem_summary:
                problem_summary[severity] = 0
            problem_summary[severity] += 1
        
        report = {
            'dataset_file': dataset_file,
            'import_time': import_time,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'import_stats': self.import_stats,
            'detected_problems': self.detected_problems,
            'problem_summary': problem_summary,
            'health_score': self._calculate_health_score()
        }
        
        return report
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score based on detected problems"""
        if not self.detected_problems:
            return 100.0
        
        # Weight by severity
        severity_weights = {
            'CRITICAL': 30,
            'HIGH': 20,
            'MEDIUM': 10,
            'LOW': 5
        }
        
        total_penalty = 0
        for problem in self.detected_problems:
            severity = problem['severity']
            count = problem.get('count', 1)
            penalty = severity_weights.get(severity, 5) * min(count, 3)
            total_penalty += penalty
        
        health_score = max(0, 100 - total_penalty)
        return round(health_score, 1)
    
    def _print_import_summary(self, report: Dict):
        """Print import summary with problem detection results"""
        
        print(f"\n{'='*70}")
        print("IMPORT SUMMARY")
        print(f"{'='*70}")
        
        stats = report['statistics']
        print(f"\n📈 Graph Statistics:")
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Total Edges: {stats['total_edges']}")
        print(f"  Applications: {stats['apps']}")
        print(f"  Topics: {stats['topics']}")
        print(f"  Brokers: {stats['brokers']}")
        print(f"  Infrastructure Nodes: {stats['nodes']}")
        
        print(f"\n⏱️ Performance:")
        print(f"  Import Time: {report['import_time']:.2f} seconds")
        
        if report['detected_problems']:
            print(f"\n⚠️ Detected Problems:")
            for problem in report['detected_problems']:
                severity_icon = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🔵'
                }.get(problem['severity'], '⚪')
                
                print(f"\n  {severity_icon} {problem['pattern']} ({problem['severity']})")
                print(f"     {problem['details']}")
                if problem['affected']:
                    print(f"     Affected: {', '.join(problem['affected'][:3])}")
                    if len(problem['affected']) > 3:
                        print(f"     ... and {len(problem['affected']) - 3} more")
        
        print(f"\n💯 Health Score: {report['health_score']}%")
        
        if report['health_score'] < 50:
            print("  ⚠️ Critical issues detected - immediate attention required!")
        elif report['health_score'] < 80:
            print("  ⚠️ Several issues detected - review recommended")
        else:
            print("  ✅ System is relatively healthy")
    
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
    
    def _calculate_qos_criticality(self, qos: Dict) -> float:
        """Calculate criticality score from QoS settings"""
        scores = {
            'durability': {
                'VOLATILE': 0.25,
                'TRANSIENT_LOCAL': 0.5,
                'TRANSIENT': 0.75,
                'PERSISTENT': 1.0
            },
            'reliability': {
                'BEST_EFFORT': 0.5,
                'RELIABLE': 1.0
            },
            'transport_priority': {
                'LOW': 0.25,
                'MEDIUM': 0.5,
                'HIGH': 0.75,
                'URGENT': 1.0
            }
        }
        
        durability_score = scores['durability'].get(qos.get('durability', 'VOLATILE'), 0.25)
        reliability_score = scores['reliability'].get(qos.get('reliability', 'BEST_EFFORT'), 0.5)
        priority_score = scores['transport_priority'].get(qos.get('transport_priority', 'MEDIUM'), 0.5)
        
        # Deadline impact
        deadline_ms = qos.get('deadline_ms', 1000)
        deadline_score = max(0, 1 - (deadline_ms / 1000)) if deadline_ms < float('inf') else 0.5
        
        return (durability_score * 0.3 + 
                reliability_score * 0.3 + 
                priority_score * 0.2 + 
                deadline_score * 0.2)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Direct Neo4j Dataset Importer with Problem Pattern Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import a specific dataset with problem detection
  python direct_import.py --dataset dataset_single_point_of_failure.json
  
  # Import all problematic pattern datasets
  python direct_import.py --import-all problematic_datasets/
  
  # Import without problem detection (faster)
  python direct_import.py --dataset dataset.json --no-detect
  
  # Save import report
  python direct_import.py --dataset dataset.json --report report.json
        """
    )
    
    parser.add_argument('--dataset', type=str, help='Path to dataset JSON file')
    parser.add_argument('--import-all', type=str, help='Directory containing multiple datasets')
    parser.add_argument('--no-detect', action='store_true', help='Skip problem pattern detection')
    parser.add_argument('--report', type=str, help='Save import report to file')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--password', type=str, default='password', help='Neo4j password')
    
    args = parser.parse_args()
    
    # Create importer
    importer = DirectNeo4jImporter(args.uri, args.user, args.password)
    
    try:
        if args.import_all:
            # Import all datasets in directory
            import glob
            datasets = glob.glob(os.path.join(args.import_all, '*.json'))
            
            all_reports = []
            for dataset in datasets:
                if 'summary' not in dataset:  # Skip summary file
                    report = importer.import_dataset_with_detection(
                        dataset, 
                        detect_problems=not args.no_detect
                    )
                    all_reports.append(report)
            
            # Save combined report
            if args.report:
                with open(args.report, 'w') as f:
                    json.dump(all_reports, f, indent=2, default=str)
                print(f"\n💾 Reports saved to {args.report}")
        
        elif args.dataset:
            # Import single dataset
            report = importer.import_dataset_with_detection(
                args.dataset,
                detect_problems=not args.no_detect
            )
            
            if args.report:
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\n💾 Report saved to {args.report}")
        
        else:
            parser.print_help()
    
    finally:
        importer.close()


if __name__ == "__main__":
    main()