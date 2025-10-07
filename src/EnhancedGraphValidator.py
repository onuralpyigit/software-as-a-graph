#!/usr/bin/env python3
"""
Enhanced Graph Validator for comprehensive validation of the pub-sub graph
Provides detailed structural, semantic, performance, and scalability validation
"""

import time
from typing import Dict, Optional
import json

class EnhancedGraphValidator:
    def __init__(self, graph_builder):
        """Initialize validator with graph builder instance"""
        self.gb = graph_builder
        self.validation_results = {
            'structural': [],
            'semantic': [],
            'performance': [],
            'scalability': [],
            'summary': {}
        }
    
    def validate_complete(self, output_file: Optional[str] = None) -> Dict:
        """Run all validation checks and generate comprehensive report"""
        print("\n" + "="*50)
        print("RUNNING COMPREHENSIVE GRAPH VALIDATION")
        print("="*50)
        
        # Run all validation categories
        self.validate_structure()
        self.validate_semantics()
        self.validate_performance()
        self.validate_scalability()
        
        # Generate summary
        self._generate_summary()
        
        # Print results
        self._print_results()
        
        # Save to file if requested
        if output_file:
            self._save_results(output_file)
        
        return self.validation_results
    
    def validate_structure(self):
        """Structural validation checks"""
        print("\n📐 Structural Validation...")
        
        checks = [
            self._check_orphaned_components(),
            self._check_dependency_cycles(),
            self._check_broker_coverage(),
            self._check_node_connectivity(),
            self._check_application_distribution(),
            self._check_topic_consistency(),
            self._check_relationship_integrity()
        ]
        
        for check_result in checks:
            self.validation_results['structural'].append(check_result)
            self._print_check_result(check_result)
    
    def validate_semantics(self):
        """Semantic consistency validation"""
        print("\n🧠 Semantic Validation...")
        
        checks = [
            self._check_qos_compatibility(),
            self._check_dependency_strength(),
            self._check_criticality_consistency(),
            self._check_resource_allocation(),
            self._check_message_patterns(),
            self._check_replication_consistency()
        ]
        
        for check_result in checks:
            self.validation_results['semantic'].append(check_result)
            self._print_check_result(check_result)
    
    def validate_performance(self):
        """Performance validation checks"""
        print("\n⚡ Performance Validation...")
        
        checks = [
            self._benchmark_queries(),
            self._check_bottlenecks(),
            self._check_latency_paths(),
            self._check_throughput_capacity()
        ]
        
        for check_result in checks:
            self.validation_results['performance'].append(check_result)
            self._print_check_result(check_result)
    
    def validate_scalability(self):
        """Scalability validation checks"""
        print("\n📈 Scalability Validation...")
        
        checks = [
            self._check_scalability_limits(),
            self._check_partition_tolerance(),
            self._check_failure_resilience(),
            self._check_growth_capacity()
        ]
        
        for check_result in checks:
            self.validation_results['scalability'].append(check_result)
            self._print_check_result(check_result)
    
    # Structural Checks
    def _check_orphaned_components(self) -> Dict:
        """Find orphaned nodes of all types"""
        result = {'check': 'orphaned_components', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check orphaned topics (no publishers OR no subscribers)
        orphaned_topics = self.gb.execute_cypher("""
            MATCH (t:Topic)
            WHERE NOT (()-[:PUBLISHES_TO]->(t)) OR NOT (()-[:SUBSCRIBES_TO]->(t))
            RETURN t.name as name, t.criticality_score as criticality,
                   EXISTS((()-[:PUBLISHES_TO]->(t))) as has_publisher,
                   EXISTS((()-[:SUBSCRIBES_TO]->(t))) as has_subscriber
        """)
        
        if orphaned_topics:
            result['status'] = 'WARN'
            result['issues'].append(f"Found {len(orphaned_topics)} orphaned topics")
            result['metrics']['orphaned_topics'] = len(orphaned_topics)
            result['details'] = orphaned_topics[:5]  # First 5 for brevity
        
        # Check isolated applications
        isolated_apps = self.gb.execute_cypher("""
            MATCH (a:Application)
            WHERE NOT (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->()
            RETURN a.name as name, a.criticality_score as criticality
        """)
        
        if isolated_apps:
            result['status'] = 'FAIL'
            result['issues'].append(f"Found {len(isolated_apps)} isolated applications")
            result['metrics']['isolated_apps'] = len(isolated_apps)
        
        # Check unrouted topics
        unrouted_topics = self.gb.execute_cypher("""
            MATCH (t:Topic)
            WHERE NOT (()-[:ROUTES]->(t))
            RETURN count(t) as count
        """)[0]['count']
        
        if unrouted_topics > 0:
            result['status'] = 'FAIL'
            result['issues'].append(f"Found {unrouted_topics} topics without broker routing")
            result['metrics']['unrouted_topics'] = unrouted_topics
        
        return result
    
    def _check_dependency_cycles(self) -> Dict:
        """Detect circular dependencies at various depths"""
        result = {'check': 'dependency_cycles', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        for depth in [2, 3, 4, 5]:
            cycles = self.gb.execute_cypher(f"""
                MATCH path = (a:Application)-[:DEPENDS_ON*{depth}]->(a)
                RETURN DISTINCT a.name as app, a.criticality_score as criticality, {depth} as cycle_depth
            """)
            
            if cycles:
                result['status'] = 'WARN' if depth > 2 else 'FAIL'
                result['issues'].append(f"Found {len(cycles)} circular dependencies at depth {depth}")
                result['metrics'][f'cycles_depth_{depth}'] = len(cycles)
        
        return result
    
    def _check_broker_coverage(self) -> Dict:
        """Check broker coverage and redundancy"""
        result = {'check': 'broker_coverage', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check broker load distribution
        broker_load = self.gb.execute_cypher("""
            MATCH (b:Broker)-[:ROUTES]->(t:Topic)
            WITH b, count(t) as topic_count
            RETURN b.name as broker, topic_count, b.max_topics as max_topics,
                   toFloat(topic_count) / b.max_topics as utilization
            ORDER BY utilization DESC
        """)
        
        overloaded = [b for b in broker_load if b['utilization'] is not None and b['utilization'] > 0.8]
        if overloaded:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(overloaded)} brokers over 80% capacity")
            result['metrics']['overloaded_brokers'] = len(overloaded)
        
        # Check topic redundancy
        single_broker_topics = self.gb.execute_cypher("""
            MATCH (t:Topic)<-[:ROUTES]-(b:Broker)
            WITH t, count(b) as broker_count
            WHERE broker_count = 1 AND t.criticality_score > 0.7
            RETURN count(t) as count
        """)[0]['count']
        
        if single_broker_topics > 0:
            result['status'] = 'WARN'
            result['issues'].append(f"{single_broker_topics} critical topics with single broker")
            result['metrics']['single_broker_critical_topics'] = single_broker_topics
        
        return result
    
    def _check_node_connectivity(self) -> Dict:
        """Check network connectivity patterns"""
        result = {'check': 'node_connectivity', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Find disconnected node clusters
        disconnected = self.gb.execute_cypher("""
            MATCH (n:Node)
            WHERE NOT (n)-[:CONNECTS_TO]-() AND NOT ()-[:CONNECTS_TO]-(n)
            RETURN count(n) as count
        """)[0]['count']
        
        if disconnected > 0:
            result['status'] = 'WARN'
            result['issues'].append(f"{disconnected} nodes with no network connections")
            result['metrics']['disconnected_nodes'] = disconnected
        
        # Check cross-zone connectivity
        cross_zone = self.gb.execute_cypher("""
            MATCH (n1:Node)-[c:CONNECTS_TO]->(n2:Node)
            WHERE n1.zone <> n2.zone
            RETURN count(c) as cross_zone_connections,
                   avg(c.latency_ms) as avg_latency
        """)[0]
        
        result['metrics']['cross_zone_connections'] = cross_zone['cross_zone_connections']
        result['metrics']['avg_cross_zone_latency'] = cross_zone['avg_latency']
        
        return result
    
    def _check_application_distribution(self) -> Dict:
        """Check application distribution across nodes"""
        result = {'check': 'application_distribution', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check for unbalanced distribution
        distribution = self.gb.execute_cypher("""
            MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
            WITH n, count(a) as app_count
            RETURN avg(app_count) as avg_apps, 
                   stDev(app_count) as std_dev,
                   max(app_count) as max_apps,
                   min(app_count) as min_apps
        """)[0]
        
        if distribution['std_dev'] > distribution['avg_apps'] * 0.5:
            result['status'] = 'WARN'
            result['issues'].append("Unbalanced application distribution across nodes")
        
        result['metrics'] = distribution
        
        # Check critical app distribution
        critical_concentration = self.gb.execute_cypher("""
            MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
            WHERE a.criticality_score > 0.8
            WITH n, count(a) as critical_apps
            WHERE critical_apps > 5
            RETURN n.name as node, critical_apps
        """)
        
        if critical_concentration:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(critical_concentration)} nodes with high concentration of critical apps")
        
        return result
    
    def _check_topic_consistency(self) -> Dict:
        """Check topic configuration consistency"""
        result = {'check': 'topic_consistency', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check for topics with mismatched QoS between publishers and subscribers
        qos_mismatch = self.gb.execute_cypher("""
            MATCH (a1:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(a2:Application)
            WHERE t.reliability = 'BEST_EFFORT' AND a2.criticality_score > 0.8
            RETURN t.name as topic, count(DISTINCT a2) as critical_subscribers
        """)
        
        if qos_mismatch:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(qos_mismatch)} topics with QoS mismatches")
            result['metrics']['qos_mismatches'] = len(qos_mismatch)
        
        return result
    
    def _check_relationship_integrity(self) -> Dict:
        """Check relationship integrity constraints"""
        result = {'check': 'relationship_integrity', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check for duplicate relationships
        duplicates = self.gb.execute_cypher("""
            MATCH (a)-[r1]->(b)
            MATCH (a)-[r2]->(b)
            WHERE id(r1) < id(r2) AND type(r1) = type(r2)
            RETURN type(r1) as rel_type, count(*) as count
        """)
        
        if duplicates:
            result['status'] = 'FAIL'
            result['issues'].append(f"Found duplicate relationships: {duplicates}")
        
        return result
    
    # Semantic Checks
    def _check_qos_compatibility(self) -> Dict:
        """Validate QoS alignment between connected components"""
        result = {'check': 'qos_compatibility', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check latency requirements
        latency_issues = self.gb.execute_cypher("""
            MATCH (a1:Application)-[d:DEPENDS_ON]->(a2:Application)
            WHERE a1.qos_latency_ms < d.latency_requirement_ms
            RETURN a1.name as dependent, a2.name as provider,
                   a1.qos_latency_ms as required_latency,
                   d.latency_requirement_ms as actual_latency
        """)
        
        if latency_issues:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(latency_issues)} latency requirement violations")
            result['metrics']['latency_violations'] = len(latency_issues)
        
        # Check availability requirements
        availability_issues = self.gb.execute_cypher("""
            MATCH (a1:Application)-[:DEPENDS_ON]->(a2:Application)
            WHERE a1.qos_availability_percent > a2.qos_availability_percent
            RETURN count(*) as count
        """)[0]['count']
        
        if availability_issues > 0:
            result['status'] = 'WARN'
            result['issues'].append(f"{availability_issues} availability requirement violations")
            result['metrics']['availability_violations'] = availability_issues
        
        return result
    
    def _check_dependency_strength(self) -> Dict:
        """Analyze dependency strength patterns"""
        result = {'check': 'dependency_strength', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        strength_dist = self.gb.execute_cypher("""
            MATCH ()-[d:DEPENDS_ON]->()
            RETURN d.criticality as criticality, count(*) as count
            ORDER BY criticality DESC
        """)
        
        critical_deps = sum(d['count'] for d in strength_dist if d['criticality'] == 'CRITICAL')
        total_deps = sum(d['count'] for d in strength_dist)
        
        result['metrics']['critical_dependency_ratio'] = critical_deps / total_deps if total_deps > 0 else 0
        result['metrics']['dependency_distribution'] = strength_dist
        
        if result['metrics']['critical_dependency_ratio'] > 0.5:
            result['status'] = 'WARN'
            result['issues'].append("High ratio of critical dependencies")
        
        return result
    
    def _check_criticality_consistency(self) -> Dict:
        """Check criticality score consistency"""
        result = {'check': 'criticality_consistency', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check if critical apps depend on non-critical components
        inconsistent = self.gb.execute_cypher("""
            MATCH (a1:Application)-[:DEPENDS_ON]->(a2:Application)
            WHERE a1.criticality_score > 0.8 AND a2.criticality_score < 0.5
            RETURN a1.name as critical_app, a2.name as non_critical_dep,
                   a1.criticality_score as app_criticality,
                   a2.criticality_score as dep_criticality
        """)
        
        if inconsistent:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(inconsistent)} critical apps depend on non-critical components")
            result['metrics']['criticality_inconsistencies'] = len(inconsistent)
        
        return result
    
    def _check_resource_allocation(self) -> Dict:
        """Check resource allocation and utilization"""
        result = {'check': 'resource_allocation', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check for overallocated nodes
        overallocated = self.gb.execute_cypher("""
            MATCH (n:Node)
            WHERE n.cpu_utilization > 0.9 OR n.memory_utilization > 0.9
            RETURN n.name as node, 
                   n.cpu_utilization as cpu_util,
                   n.memory_utilization as mem_util
        """)
        
        if overallocated:
            result['status'] = 'FAIL'
            result['issues'].append(f"{len(overallocated)} nodes overallocated")
            result['metrics']['overallocated_nodes'] = len(overallocated)
        
        # Check resource distribution
        resource_stats = self.gb.execute_cypher("""
            MATCH (n:Node)
            RETURN avg(n.cpu_utilization) as avg_cpu,
                   avg(n.memory_utilization) as avg_memory,
                   max(n.cpu_utilization) as max_cpu,
                   max(n.memory_utilization) as max_memory
        """)[0]
        
        result['metrics'].update(resource_stats)
        
        return result
    
    def _check_message_patterns(self) -> Dict:
        """Analyze message patterns consistency"""
        result = {'check': 'message_patterns', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        pattern_dist = self.gb.execute_cypher("""
            MATCH (t:Topic)<-[p:PUBLISHES_TO]-(a:Application)
            RETURN p.pattern as pattern, count(*) as count
        """)
        
        result['metrics']['pattern_distribution'] = pattern_dist
        
        # Check for pattern mismatches
        pattern_issues = self.gb.execute_cypher("""
            MATCH (t:Topic)<-[p:PUBLISHES_TO]-(a:Application)
            WHERE t.message_pattern <> p.pattern
            RETURN count(*) as count
        """)[0]['count']
        
        if pattern_issues > 0:
            result['status'] = 'WARN'
            result['issues'].append(f"{pattern_issues} pattern mismatches between topics and publishers")
        
        return result
    
    def _check_replication_consistency(self) -> Dict:
        """Check replication configuration consistency"""
        result = {'check': 'replication_consistency', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check critical apps without sufficient replicas
        under_replicated = self.gb.execute_cypher("""
            MATCH (a:Application)
            WHERE a.criticality_score > 0.8 AND a.replicas < 3
            RETURN a.name as app, a.replicas as replicas, a.criticality_score as criticality
        """)
        
        if under_replicated:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(under_replicated)} critical apps with insufficient replicas")
            result['metrics']['under_replicated_critical_apps'] = len(under_replicated)
        
        return result
    
    # Performance Checks
    def _benchmark_queries(self) -> Dict:
        """Benchmark critical query performance"""
        result = {'check': 'query_performance', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        queries = {
            'find_dependencies': """
                MATCH (a1:Application)-[:DEPENDS_ON*1..3]->(a2:Application)
                RETURN count(*) as paths
            """,
            'critical_topics': """
                MATCH (t:Topic)
                WHERE t.criticality_score > 0.8
                MATCH (t)<-[:PUBLISHES_TO|SUBSCRIBES_TO]-(a:Application)
                RETURN t.name, count(a) as app_count
                ORDER BY app_count DESC LIMIT 10
            """,
            'bottleneck_detection': """
                MATCH (t:Topic)
                MATCH (t)<-[:PUBLISHES_TO]-(pub:Application)
                MATCH (t)<-[:SUBSCRIBES_TO]-(sub:Application)
                WITH t, count(DISTINCT pub) as pubs, count(DISTINCT sub) as subs
                WHERE subs > pubs * 10
                RETURN t.name, pubs, subs
            """,
            'failure_impact': """
                MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
                WHERE a.criticality_score > 0.7
                WITH n, collect(a.name) as critical_apps
                MATCH (a1:Application)-[:RUNS_ON]->(n)
                MATCH (a1)-[:DEPENDS_ON]->(a2:Application)
                RETURN n.name, count(DISTINCT a2) as impacted_apps
            """
        }
        
        for name, query in queries.items():
            start_time = time.time()
            self.gb.execute_cypher(query)
            execution_time = time.time() - start_time
            result['metrics'][f'{name}_time'] = execution_time
            
            # Check against thresholds
            if execution_time > 2.0:
                result['status'] = 'WARN'
                result['issues'].append(f"Query '{name}' took {execution_time:.2f}s")
        
        return result
    
    def _check_bottlenecks(self) -> Dict:
        """Identify potential bottlenecks"""
        result = {'check': 'bottleneck_detection', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Find topics with high fanout
        high_fanout = self.gb.execute_cypher("""
            MATCH (t:Topic)
            WHERE t.fanout_ratio > 10
            RETURN t.name as topic, t.fanout_ratio as fanout,
                   t.publisher_count as publishers,
                   t.subscriber_count as subscribers
            ORDER BY fanout DESC
            LIMIT 10
        """)
        
        if high_fanout:
            result['status'] = 'WARN'
            result['issues'].append(f"Found {len(high_fanout)} topics with high fanout")
            result['metrics']['high_fanout_topics'] = len(high_fanout)
        
        # Find overloaded brokers
        overloaded_brokers = self.gb.execute_cypher("""
            MATCH (b:Broker)-[:ROUTES]->(t:Topic)
            WITH b, count(t) as topic_count, sum(t.messages_per_second) as total_msg_rate
            WHERE topic_count > b.max_topics * 0.8
            RETURN b.name as broker, topic_count, total_msg_rate
        """)
        
        if overloaded_brokers:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(overloaded_brokers)} brokers near capacity")
        
        return result
    
    def _check_latency_paths(self) -> Dict:
        """Analyze latency-critical paths"""
        result = {'check': 'latency_analysis', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Find paths with accumulated latency exceeding requirements
        high_latency_paths = self.gb.execute_cypher("""
            MATCH path = (a1:Application)-[:DEPENDS_ON*1..5]->(a2:Application)
            WHERE a1.qos_latency_ms < 100
            WITH a1, a2, path, reduce(total = 0, r IN relationships(path) | 
                total + coalesce(r.latency_requirement_ms, 0)) as total_latency
            WHERE total_latency > a1.qos_latency_ms
            RETURN a1.name as source, a2.name as target, 
                   length(path) as hops, total_latency
            LIMIT 20
        """)
        
        if high_latency_paths:
            result['status'] = 'WARN'
            result['issues'].append(f"Found {len(high_latency_paths)} high-latency dependency paths")
            result['metrics']['high_latency_paths'] = len(high_latency_paths)
        
        return result
    
    def _check_throughput_capacity(self) -> Dict:
        """Check throughput capacity constraints"""
        result = {'check': 'throughput_capacity', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Check network bandwidth utilization
        bandwidth_issues = self.gb.execute_cypher("""
            MATCH (n1:Node)-[c:CONNECTS_TO]->(n2:Node)
            WHERE c.bandwidth_gbps * 1000 > n1.network_bandwidth_mbps * 0.8
            RETURN n1.name as source, n2.name as target,
                   c.bandwidth_gbps * 1000 as required_mbps,
                   n1.network_bandwidth_mbps as available_mbps
        """)
        
        if bandwidth_issues:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(bandwidth_issues)} network links near capacity")
        
        return result
    
    # Scalability Checks
    def _check_scalability_limits(self) -> Dict:
        """Check current utilization against scalability limits"""
        result = {'check': 'scalability_limits', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Calculate headroom for growth
        headroom = self.gb.execute_cypher("""
            MATCH (n:Node)
            WITH avg(1 - n.cpu_utilization) as cpu_headroom,
                 avg(1 - n.memory_utilization) as memory_headroom
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH cpu_headroom, memory_headroom, b, count(t) as topics_per_broker
            WITH cpu_headroom, memory_headroom, 
                 avg(1 - toFloat(topics_per_broker) / b.max_topics) as broker_headroom
            RETURN cpu_headroom, memory_headroom, broker_headroom
        """)[0]
        
        result['metrics'].update(headroom)
        
        if any(v < 0.2 for v in headroom.values() if v is not None):
            result['status'] = 'WARN'
            result['issues'].append("Less than 20% headroom for growth")
        
        return result
    
    def _check_partition_tolerance(self) -> Dict:
        """Check network partition tolerance"""
        result = {'check': 'partition_tolerance', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Find single points of failure
        spof = self.gb.execute_cypher("""
            MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
            WHERE a.criticality_score > 0.8
            WITH a, collect(DISTINCT n) as nodes
            WHERE size(nodes) = 1
            RETURN count(a) as single_node_critical_apps
        """)[0]['single_node_critical_apps']
        
        if spof > 0:
            result['status'] = 'FAIL'
            result['issues'].append(f"{spof} critical apps with no failover")
            result['metrics']['single_points_of_failure'] = spof
        
        # Check zone distribution
        zone_dist = self.gb.execute_cypher("""
            MATCH (n:Node)
            RETURN n.zone as zone, count(n) as node_count
        """)
        
        if len(zone_dist) < 2:
            result['status'] = 'FAIL'
            result['issues'].append("Insufficient zone distribution for partition tolerance")
        
        return result
    
    def _check_failure_resilience(self) -> Dict:
        """Analyze failure resilience"""
        result = {'check': 'failure_resilience', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Simulate node failure impact
        failure_impact = self.gb.execute_cypher("""
            MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
            WITH n, count(a) as apps_on_node,
                 sum(a.criticality_score) as total_criticality
            RETURN n.name as node, apps_on_node, total_criticality
            ORDER BY total_criticality DESC
            LIMIT 5
        """)
        
        high_impact_nodes = [n for n in failure_impact if n['total_criticality'] > 5.0]
        if high_impact_nodes:
            result['status'] = 'WARN'
            result['issues'].append(f"{len(high_impact_nodes)} nodes with high failure impact")
            result['metrics']['high_impact_nodes'] = high_impact_nodes
        
        return result
    
    def _check_growth_capacity(self) -> Dict:
        """Check capacity for future growth"""
        result = {'check': 'growth_capacity', 'status': 'PASS', 'issues': [], 'metrics': {}}
        
        # Calculate growth metrics
        growth_metrics = self.gb.execute_cypher("""
            MATCH (n:Node)
            WITH count(n) as total_nodes,
                 avg(n.cpu_utilization) as avg_cpu_util,
                 max(n.cpu_utilization) as max_cpu_util
            MATCH (t:Topic)
            WITH total_nodes, avg_cpu_util, max_cpu_util, count(t) as total_topics
            MATCH (a:Application)
            RETURN total_nodes, total_topics, count(a) as total_apps,
                   avg_cpu_util, max_cpu_util
        """)[0]
        
        result['metrics'] = growth_metrics
        
        # Estimate 2x growth capacity
        if growth_metrics['avg_cpu_util'] > 0.5:
            result['status'] = 'WARN'
            result['issues'].append("Insufficient capacity for 2x growth")
        
        return result
    
    # Helper Methods
    def _print_check_result(self, result: Dict):
        """Print individual check result"""
        status_symbols = {
            'PASS': '✅',
            'WARN': '⚠️',
            'FAIL': '❌'
        }
        
        symbol = status_symbols.get(result['status'], '❓')
        print(f"  {symbol} {result['check']}: {result['status']}")
        
        if result.get('issues'):
            for issue in result['issues']:
                print(f"     - {issue}")
    
    def _generate_summary(self):
        """Generate validation summary"""
        summary = {
            'total_checks': 0,
            'passed': 0,
            'warnings': 0,
            'failures': 0,
            'critical_issues': [],
            'recommendations': []
        }
        
        for category in ['structural', 'semantic', 'performance', 'scalability']:
            for check in self.validation_results[category]:
                summary['total_checks'] += 1
                if check['status'] == 'PASS':
                    summary['passed'] += 1
                elif check['status'] == 'WARN':
                    summary['warnings'] += 1
                else:
                    summary['failures'] += 1
                    summary['critical_issues'].extend(check.get('issues', []))
        
        # Generate recommendations
        if summary['failures'] > 0:
            summary['recommendations'].append("Address critical failures before deployment")
        if summary['warnings'] > 5:
            summary['recommendations'].append("Review and mitigate warnings to improve system reliability")
        
        self.validation_results['summary'] = summary
    
    def _print_results(self):
        """Print validation results summary"""
        summary = self.validation_results['summary']
        
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total Checks: {summary['total_checks']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"⚠️ Warnings: {summary['warnings']}")
        print(f"❌ Failures: {summary['failures']}")
        
        if summary['critical_issues']:
            print("\n🚨 Critical Issues:")
            for issue in summary['critical_issues'][:5]:  # Show top 5
                print(f"  - {issue}")
        
        if summary['recommendations']:
            print("\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
        
        # Calculate health score
        health_score = (summary['passed'] / summary['total_checks']) * 100 if summary['total_checks'] > 0 else 0
        print(f"\n📊 Overall Health Score: {health_score:.1f}%")
    
    def _save_results(self, output_file: str):
        """Save validation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        print(f"\n💾 Validation results saved to: {output_file}")