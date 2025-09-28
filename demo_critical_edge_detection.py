import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def create_enhanced_ecommerce_graph():
    """
    Create an enhanced e-commerce graph with detailed edge properties
    """
    G = nx.DiGraph()
    
    # Add nodes (same as before but with additional properties)
    services = [
        {'id': 'order-service', 'type': 'application', 'criticality': 'high', 
         'processing_capacity': 1000, 'current_load': 850},
        {'id': 'payment-service', 'type': 'application', 'criticality': 'critical',
         'processing_capacity': 500, 'current_load': 450},
        {'id': 'inventory-service', 'type': 'application', 'criticality': 'high',
         'processing_capacity': 2000, 'current_load': 1500},
        {'id': 'shipping-service', 'type': 'application', 'criticality': 'high',
         'processing_capacity': 800, 'current_load': 600},
        {'id': 'notification-service', 'type': 'application', 'criticality': 'low',
         'processing_capacity': 5000, 'current_load': 2000},
        {'id': 'fraud-detection', 'type': 'application', 'criticality': 'high',
         'processing_capacity': 300, 'current_load': 250},
        {'id': 'tax-calculation', 'type': 'application', 'criticality': 'critical',
         'processing_capacity': 400, 'current_load': 350},
        {'id': 'pricing-engine', 'type': 'application', 'criticality': 'high',
         'processing_capacity': 1000, 'current_load': 800},
        {'id': 'warehouse-integration', 'type': 'application', 'criticality': 'high',
         'processing_capacity': 500, 'current_load': 400},
        {'id': 'accounting-integration', 'type': 'application', 'criticality': 'medium',
         'processing_capacity': 300, 'current_load': 200}
    ]
    
    for service in services:
        G.add_node(service['id'], **service)
    
    # Add topics with QoS requirements
    topics = [
        {'id': 'orders.created', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 100, 'ordering': 'strict'},
        {'id': 'orders.validated', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 200, 'ordering': 'strict'},
        {'id': 'payments.initiated', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 50, 'ordering': 'strict'},
        {'id': 'payments.processed', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 100, 'ordering': 'strict'},
        {'id': 'inventory.check', 'type': 'topic', 'durability': 'volatile',
         'max_latency_ms': 50, 'ordering': 'none'},
        {'id': 'inventory.reserved', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 150, 'ordering': 'partial'},
        {'id': 'tax.calculated', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 200, 'ordering': 'none'},
        {'id': 'price.updated', 'type': 'topic', 'durability': 'volatile',
         'max_latency_ms': 500, 'ordering': 'none'},
        {'id': 'fraud.check', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 1000, 'ordering': 'strict'},
        {'id': 'shipments.created', 'type': 'topic', 'durability': 'persistent',
         'max_latency_ms': 300, 'ordering': 'partial'}
    ]
    
    for topic in topics:
        G.add_node(topic['id'], **topic)
    
    # Define CRITICAL EDGES with detailed properties
    edges = [
        # CRITICAL PATH: Order Creation → Payment Processing (Synchronous)
        ('order-service', 'orders.created', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 20,
            'p99_latency_ms': 50,
            'error_rate': 0.001,
            'is_synchronous': True,
            'transaction_boundary': True,
            'business_flow': 'order_processing',
            'sla_requirement': 'tier1',
            'alternative_path': False
        }),
        
        # CRITICAL: Orders must reach payment service for processing
        ('orders.created', 'payment-service', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 10,
            'p99_latency_ms': 30,
            'error_rate': 0.0001,
            'is_synchronous': True,
            'transaction_boundary': True,
            'business_flow': 'order_processing',
            'retry_policy': 'exponential_backoff',
            'max_retries': 3
        }),
        
        # CRITICAL: Payment initiation edge
        ('payment-service', 'payments.initiated', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 98,
            'avg_latency_ms': 15,
            'p99_latency_ms': 40,
            'error_rate': 0.002,
            'is_synchronous': True,
            'transaction_boundary': True,
            'business_flow': 'payment_processing',
            'contains_pii': True,
            'encryption_required': True
        }),
        
        # CRITICAL: Tax calculation dependency
        ('payments.initiated', 'tax-calculation', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 98,
            'avg_latency_ms': 25,
            'p99_latency_ms': 80,
            'error_rate': 0.001,
            'is_synchronous': True,
            'transaction_boundary': True,
            'business_flow': 'payment_processing',
            'regulatory_requirement': True
        }),
        
        # CRITICAL: Tax result publication
        ('tax-calculation', 'tax.calculated', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 97,
            'avg_latency_ms': 50,
            'p99_latency_ms': 150,
            'error_rate': 0.001,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'payment_processing'
        }),
        
        # CRITICAL: Payment service needs tax info
        ('tax.calculated', 'payment-service', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 97,
            'avg_latency_ms': 10,
            'p99_latency_ms': 25,
            'error_rate': 0.0001,
            'is_synchronous': True,
            'transaction_boundary': True,
            'business_flow': 'payment_processing',
            'dependency_type': 'blocking'
        }),
        
        # CRITICAL: Final payment processing
        ('payment-service', 'payments.processed', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 95,
            'avg_latency_ms': 30,
            'p99_latency_ms': 100,
            'error_rate': 0.005,
            'is_synchronous': False,
            'transaction_boundary': True,
            'business_flow': 'payment_processing',
            'audit_required': True
        }),
        
        # Inventory check edges
        ('orders.created', 'inventory-service', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 15,
            'p99_latency_ms': 40,
            'error_rate': 0.001,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'inventory_management'
        }),
        
        ('inventory-service', 'inventory.check', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 20,
            'p99_latency_ms': 60,
            'error_rate': 0.002,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'inventory_management'
        }),
        
        # Pricing engine edges
        ('inventory.check', 'pricing-engine', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 25,
            'p99_latency_ms': 75,
            'error_rate': 0.003,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'pricing'
        }),
        
        ('pricing-engine', 'price.updated', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 80,
            'avg_latency_ms': 100,
            'p99_latency_ms': 300,
            'error_rate': 0.01,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'pricing'
        }),
        
        # Fraud detection edges
        ('orders.created', 'fraud-detection', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 50,
            'p99_latency_ms': 200,
            'error_rate': 0.001,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'fraud_check',
            'async_timeout_ms': 5000
        }),
        
        ('fraud-detection', 'fraud.check', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 100,
            'avg_latency_ms': 200,
            'p99_latency_ms': 1000,
            'error_rate': 0.002,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'fraud_check'
        }),
        
        # Other edges
        ('payments.processed', 'inventory-service', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 95,
            'avg_latency_ms': 10,
            'p99_latency_ms': 30,
            'error_rate': 0.001,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'inventory_management'
        }),
        
        ('inventory-service', 'inventory.reserved', {
            'relationship': 'PUBLISHES_TO',
            'msg_rate_per_sec': 93,
            'avg_latency_ms': 25,
            'p99_latency_ms': 80,
            'error_rate': 0.002,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'inventory_management'
        }),
        
        ('inventory.reserved', 'shipping-service', {
            'relationship': 'SUBSCRIBES_TO',
            'msg_rate_per_sec': 93,
            'avg_latency_ms': 15,
            'p99_latency_ms': 45,
            'error_rate': 0.001,
            'is_synchronous': False,
            'transaction_boundary': False,
            'business_flow': 'fulfillment'
        })
    ]
    
    for source, target, attrs in edges:
        G.add_edge(source, target, **attrs)
    
    return G

class CriticalEdgeAnalyzer:
    """
    Comprehensive analyzer for identifying critical edges in pub-sub systems
    """
    
    def __init__(self, graph):
        self.graph = graph
        
    def analyze_all_edges(self):
        """
        Apply multiple rules to identify critical edges
        """
        results = []
        
        for source, target, attrs in self.graph.edges(data=True):
            edge = (source, target)
            
            # Apply each rule
            criticality_checks = {
                'synchronous_blocking': self._check_synchronous_blocking_edge(edge, attrs),
                'transaction_boundary': self._check_transaction_boundary_edge(edge, attrs),
                'no_alternative_path': self._check_no_alternative_path(edge),
                'high_latency_sensitivity': self._check_latency_sensitive_edge(edge, attrs),
                'bottleneck_edge': self._check_bottleneck_edge(edge, attrs),
                'regulatory_compliance': self._check_regulatory_edge(edge, attrs),
                'cascade_trigger': self._check_cascade_trigger_edge(edge),
                'ordering_critical': self._check_ordering_critical_edge(edge, attrs)
            }
            
            # Calculate composite criticality score
            criticality_score = self._calculate_edge_criticality(criticality_checks)
            
            if criticality_score > 0:
                result = {
                    'edge': edge,
                    'source': source,
                    'target': target,
                    'attributes': attrs,
                    'criticality_score': criticality_score,
                    'critical_factors': {k: v for k, v in criticality_checks.items() if v['is_critical']},
                    'impact_analysis': self._analyze_edge_failure_impact(edge, attrs)
                }
                results.append(result)
        
        return sorted(results, key=lambda x: x['criticality_score'], reverse=True)
    
    def _check_synchronous_blocking_edge(self, edge, attrs):
        """
        Rule: Edge is critical if it's part of a synchronous, blocking operation
        """
        is_sync = attrs.get('is_synchronous', False)
        is_transaction = attrs.get('transaction_boundary', False)
        dependency_type = attrs.get('dependency_type', '')
        
        if is_sync and (is_transaction or dependency_type == 'blocking'):
            # Calculate how critical based on latency requirements
            avg_latency = attrs.get('avg_latency_ms', 0)
            p99_latency = attrs.get('p99_latency_ms', 0)
            
            # Check if this edge is on the critical path for latency
            source_node = self.graph.nodes[edge[0]]
            target_node = self.graph.nodes[edge[1]]
            
            criticality = 0.0
            if source_node.get('type') == 'topic':
                max_latency = source_node.get('max_latency_ms', float('inf'))
                if p99_latency > max_latency * 0.3:  # Using >30% of budget
                    criticality = min(1.0, p99_latency / max_latency)
            
            return {
                'is_critical': True,
                'severity': 0.8 + (0.2 * criticality),
                'reason': 'Synchronous blocking edge in transaction flow',
                'details': {
                    'avg_latency': avg_latency,
                    'p99_latency': p99_latency,
                    'blocks_transaction': is_transaction
                }
            }
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_transaction_boundary_edge(self, edge, attrs):
        """
        Rule: Edge is critical if it crosses transaction boundaries
        """
        if attrs.get('transaction_boundary', False):
            business_flow = attrs.get('business_flow', '')
            error_rate = attrs.get('error_rate', 0)
            
            # Higher criticality for lower error tolerance
            criticality = 0.7
            if error_rate < 0.001:  # Very low error tolerance
                criticality = 0.9
            elif error_rate < 0.01:
                criticality = 0.8
            
            # Check if audit or regulatory requirements
            if attrs.get('audit_required', False) or attrs.get('regulatory_requirement', False):
                criticality = min(1.0, criticality + 0.1)
            
            return {
                'is_critical': True,
                'severity': criticality,
                'reason': 'Transaction boundary crossing',
                'details': {
                    'business_flow': business_flow,
                    'error_rate': error_rate,
                    'audit_required': attrs.get('audit_required', False)
                }
            }
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_no_alternative_path(self, edge):
        """
        Rule: Edge is critical if there's no alternative path between source and target
        """
        source, target = edge
        
        # For direct pub-sub relationships, check alternative paths
        if self.graph.nodes[source].get('type') == 'application' and \
           self.graph.nodes[target].get('type') == 'topic':
            # Check if there are other publishers to this topic
            other_publishers = [n for n in self.graph.predecessors(target) 
                              if n != source and self.graph.nodes[n].get('type') == 'application']
            
            if not other_publishers:
                return {
                    'is_critical': True,
                    'severity': 0.9,
                    'reason': 'No alternative publisher for topic',
                    'details': {
                        'sole_publisher': source,
                        'topic': target,
                        'subscriber_count': len(list(self.graph.successors(target)))
                    }
                }
        
        # Check if removing this edge would disconnect the graph
        test_graph = self.graph.copy()
        test_graph.remove_edge(source, target)
        
        # Find if there's still a path between related components
        try:
            # Check if we can still reach key targets
            if self.graph.nodes[target].get('type') == 'application':
                # Find topics this application publishes to
                published_topics = [n for n in self.graph.successors(target) 
                                   if self.graph.nodes[n].get('type') == 'topic']
                
                for topic in published_topics:
                    try:
                        # Check if there's an alternative path to reach this functionality
                        alternative_publishers = [n for n in test_graph.predecessors(topic)
                                                 if test_graph.nodes[n].get('type') == 'application']
                        if not alternative_publishers:
                            return {
                                'is_critical': True,
                                'severity': 0.85,
                                'reason': 'Removing edge breaks critical functionality',
                                'details': {
                                    'affected_functionality': topic,
                                    'no_alternatives': True
                                }
                            }
                    except:
                        pass
        except:
            pass
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_latency_sensitive_edge(self, edge, attrs):
        """
        Rule: Edge is critical if it's on a latency-sensitive path
        """
        p99_latency = attrs.get('p99_latency_ms', 0)
        avg_latency = attrs.get('avg_latency_ms', 0)
        
        # Check if target has strict latency requirements
        target_node = self.graph.nodes[edge[1]]
        
        if target_node.get('type') == 'topic':
            max_latency = target_node.get('max_latency_ms', float('inf'))
            
            if max_latency < 100:  # Sub-100ms requirement
                latency_usage = p99_latency / max_latency
                
                if latency_usage > 0.5:  # Using >50% of latency budget
                    return {
                        'is_critical': True,
                        'severity': min(1.0, latency_usage),
                        'reason': 'Latency-critical edge',
                        'details': {
                            'p99_latency': p99_latency,
                            'max_allowed': max_latency,
                            'latency_budget_usage': f"{latency_usage*100:.1f}%"
                        }
                    }
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_bottleneck_edge(self, edge, attrs):
        """
        Rule: Edge is critical if it's a throughput bottleneck
        """
        msg_rate = attrs.get('msg_rate_per_sec', 0)
        error_rate = attrs.get('error_rate', 0)
        
        # Check target's processing capacity
        target = edge[1]
        if self.graph.nodes[target].get('type') == 'application':
            capacity = self.graph.nodes[target].get('processing_capacity', float('inf'))
            current_load = self.graph.nodes[target].get('current_load', 0)
            
            # Calculate utilization including this edge's contribution
            utilization = (current_load + msg_rate) / capacity if capacity > 0 else 1.0
            
            if utilization > 0.8:  # >80% utilization
                return {
                    'is_critical': True,
                    'severity': min(1.0, utilization),
                    'reason': 'Bottleneck edge - high utilization',
                    'details': {
                        'msg_rate': msg_rate,
                        'target_utilization': f"{utilization*100:.1f}%",
                        'target_capacity': capacity,
                        'error_rate': error_rate
                    }
                }
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_regulatory_edge(self, edge, attrs):
        """
        Rule: Edge is critical if it carries regulatory/compliance data
        """
        if attrs.get('regulatory_requirement', False) or attrs.get('audit_required', False):
            contains_pii = attrs.get('contains_pii', False)
            encryption_required = attrs.get('encryption_required', False)
            
            criticality = 0.7
            if contains_pii:
                criticality += 0.15
            if encryption_required:
                criticality += 0.15
            
            return {
                'is_critical': True,
                'severity': min(1.0, criticality),
                'reason': 'Regulatory/compliance critical edge',
                'details': {
                    'regulatory_requirement': attrs.get('regulatory_requirement', False),
                    'audit_required': attrs.get('audit_required', False),
                    'contains_pii': contains_pii,
                    'encryption_required': encryption_required
                }
            }
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_cascade_trigger_edge(self, edge):
        """
        Rule: Edge is critical if its failure triggers cascading failures
        """
        source, target = edge
        
        # Check how many downstream dependencies exist
        if self.graph.nodes[target].get('type') == 'application':
            # Find all downstream effects
            downstream_nodes = set()
            to_visit = [target]
            visited = set()
            
            while to_visit:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Find what this node publishes to
                for successor in self.graph.successors(current):
                    if self.graph.nodes[successor].get('type') == 'topic':
                        # Find subscribers to this topic
                        subscribers = [s for s in self.graph.successors(successor)
                                     if self.graph.nodes[s].get('type') == 'application']
                        downstream_nodes.update(subscribers)
                        to_visit.extend(subscribers)
            
            if len(downstream_nodes) > 3:
                # Calculate criticality of downstream nodes
                critical_downstream = sum(1 for n in downstream_nodes 
                                         if self.graph.nodes[n].get('criticality') in ['critical', 'high'])
                
                if critical_downstream > 0:
                    return {
                        'is_critical': True,
                        'severity': min(1.0, 0.5 + (critical_downstream * 0.1)),
                        'reason': 'Cascade trigger edge',
                        'details': {
                            'downstream_impact': len(downstream_nodes),
                            'critical_downstream': critical_downstream,
                            'affected_services': list(downstream_nodes)[:5]  # Sample
                        }
                    }
        
        return {'is_critical': False, 'severity': 0}
    
    def _check_ordering_critical_edge(self, edge, attrs):
        """
        Rule: Edge is critical if it must maintain strict ordering
        """
        source, target = edge
        
        # Check if source or target requires strict ordering
        source_node = self.graph.nodes.get(source, {})
        target_node = self.graph.nodes.get(target, {})
        
        source_ordering = source_node.get('ordering', 'none')
        target_ordering = target_node.get('ordering', 'none')
        
        if source_ordering == 'strict' or target_ordering == 'strict':
            # This edge must maintain ordering
            msg_rate = attrs.get('msg_rate_per_sec', 0)
            
            # Higher message rates with strict ordering are more critical
            criticality = 0.6
            if msg_rate > 50:
                criticality = 0.8
            if msg_rate > 100:
                criticality = 0.9
            
            return {
                'is_critical': True,
                'severity': criticality,
                'reason': 'Ordering-critical edge',
                'details': {
                    'ordering_requirement': 'strict',
                    'msg_rate': msg_rate,
                    'source_ordering': source_ordering,
                    'target_ordering': target_ordering
                }
            }
        
        return {'is_critical': False, 'severity': 0}
    
    def _calculate_edge_criticality(self, checks):
        """
        Calculate composite criticality score from all checks
        """
        if not any(check['is_critical'] for check in checks.values()):
            return 0.0
        
        # Weight different factors
        weights = {
            'synchronous_blocking': 1.0,
            'transaction_boundary': 0.9,
            'no_alternative_path': 0.95,
            'high_latency_sensitivity': 0.7,
            'bottleneck_edge': 0.75,
            'regulatory_compliance': 0.8,
            'cascade_trigger': 0.85,
            'ordering_critical': 0.6
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for check_name, check_result in checks.items():
            if check_result['is_critical']:
                weight = weights.get(check_name, 0.5)
                weighted_sum += check_result['severity'] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _analyze_edge_failure_impact(self, edge, attrs):
        """
        Analyze the impact of edge failure
        """
        source, target = edge
        
        impact = {
            'immediate_effects': [],
            'performance_impact': {},
            'business_impact': [],
            'recovery_complexity': 'low'
        }
        
        # Immediate effects
        msg_rate = attrs.get('msg_rate_per_sec', 0)
        impact['immediate_effects'].append(f"Loss of {msg_rate} messages/sec from {source} to {target}")
        
        if attrs.get('is_synchronous', False):
            impact['immediate_effects'].append(f"Blocking failure - {source} cannot proceed")
            impact['recovery_complexity'] = 'high'
        
        if attrs.get('transaction_boundary', False):
            impact['immediate_effects'].append("Transaction boundary broken - consistency at risk")
            impact['recovery_complexity'] = 'high'
        
        # Performance impact
        impact['performance_impact'] = {
            'lost_throughput': msg_rate,
            'latency_impact': attrs.get('p99_latency_ms', 0),
            'error_rate_increase': attrs.get('error_rate', 0)
        }
        
        # Business impact
        business_flow = attrs.get('business_flow', '')
        if business_flow:
            impact['business_impact'].append(f"Business flow '{business_flow}' disrupted")
        
        if attrs.get('regulatory_requirement', False):
            impact['business_impact'].append("Regulatory compliance violation risk")
        
        if attrs.get('audit_required', False):
            impact['business_impact'].append("Audit trail broken")
        
        return impact
    
def demonstrate_critical_edge_detection():
    """
    Demonstrate critical edge detection with detailed analysis
    """
    print("="*80)
    print("CRITICAL EDGE DETECTION IN E-COMMERCE PUB-SUB SYSTEM")
    print("="*80)
    
    # Create the enhanced graph
    graph = create_enhanced_ecommerce_graph()
    
    print(f"\nGraph Statistics:")
    print(f"  - Total Nodes: {graph.number_of_nodes()}")
    print(f"  - Total Edges: {graph.number_of_edges()}")
    
    # Initialize analyzer
    analyzer = CriticalEdgeAnalyzer(graph)
    
    # Analyze all edges
    critical_edges = analyzer.analyze_all_edges()
    
    # Display top critical edges
    print(f"\n{'='*80}")
    print(f"TOP 5 CRITICAL EDGES IDENTIFIED")
    print(f"{'='*80}")
    
    for i, result in enumerate(critical_edges[:5], 1):
        edge = result['edge']
        print(f"\n{i}. CRITICAL EDGE: {edge[0]} → {edge[1]}")
        print("-" * 70)
        print(f"   Criticality Score: {result['criticality_score']:.3f}")
        print(f"   Message Rate: {result['attributes'].get('msg_rate_per_sec', 0)} msg/sec")
        print(f"   Avg Latency: {result['attributes'].get('avg_latency_ms', 0)} ms")
        print(f"   P99 Latency: {result['attributes'].get('p99_latency_ms', 0)} ms")
        print(f"   Error Rate: {result['attributes'].get('error_rate', 0)*100:.3f}%")
        
        print(f"\n   CRITICAL FACTORS:")
        for factor_name, factor_details in result['critical_factors'].items():
            print(f"      • {factor_details['reason']}")
            print(f"        Severity: {factor_details['severity']:.2f}")
            for key, value in factor_details['details'].items():
                print(f"        - {key}: {value}")
        
        print(f"\n   IMPACT ANALYSIS:")
        impact = result['impact_analysis']
        
        print(f"      Immediate Effects:")
        for effect in impact['immediate_effects']:
            print(f"        • {effect}")
        
        if impact['business_impact']:
            print(f"      Business Impact:")
            for effect in impact['business_impact']:
                print(f"        • {effect}")
        
        print(f"      Recovery Complexity: {impact['recovery_complexity']}")
    
    # Analyze critical paths
    print(f"\n{'='*80}")
    print(f"CRITICAL PATH ANALYSIS")
    print(f"{'='*80}")
    
    critical_paths = find_critical_paths(graph, critical_edges)
    
    for i, path_info in enumerate(critical_paths[:3], 1):
        print(f"\n{i}. Critical Path: {' → '.join(path_info['path'])}")
        print(f"   Total Latency: {path_info['total_latency']} ms")
        print(f"   Criticality Score: {path_info['criticality_score']:.3f}")
        print(f"   Business Flow: {path_info['business_flow']}")
        print(f"   Weakest Link: {path_info['weakest_link']}")

def find_critical_paths(graph, critical_edges):
    """
    Identify end-to-end critical paths containing critical edges
    """
    critical_paths = []
    
    # Build a map of critical edges for quick lookup
    critical_edge_map = {edge['edge']: edge for edge in critical_edges}
    
    # Find paths that contain multiple critical edges
    for start_node in graph.nodes():
        if graph.nodes[start_node].get('type') == 'application':
            # Use BFS to find paths containing critical edges
            paths = find_paths_with_critical_edges(graph, start_node, critical_edge_map)
            
            for path in paths:
                if len(path) > 2:  # Non-trivial path
                    path_criticality = calculate_path_criticality(graph, path, critical_edge_map)
                    
                    if path_criticality['score'] > 0.5:
                        critical_paths.append({
                            'path': path,
                            'criticality_score': path_criticality['score'],
                            'total_latency': path_criticality['total_latency'],
                            'business_flow': path_criticality['business_flow'],
                            'weakest_link': path_criticality['weakest_link']
                        })
    
    return sorted(critical_paths, key=lambda x: x['criticality_score'], reverse=True)

def find_paths_with_critical_edges(graph, start_node, critical_edge_map, max_depth=7):
    """
    Find paths containing critical edges using modified BFS
    """
    paths = []
    queue = [(start_node, [start_node], set())]
    
    while queue:
        current, path, visited = queue.pop(0)
        
        if len(path) > max_depth:
            continue
        
        # Check successors
        for successor in graph.successors(current):
            if successor not in visited:
                edge = (current, successor)
                new_path = path + [successor]
                new_visited = visited | {successor}
                
                # If this edge is critical, mark this path as interesting
                if edge in critical_edge_map:
                    paths.append(new_path)
                
                # Continue exploring
                queue.append((successor, new_path, new_visited))
    
    return paths

def calculate_path_criticality(graph, path, critical_edge_map):
    """
    Calculate criticality metrics for a path
    """
    total_latency = 0
    critical_edge_count = 0
    max_criticality = 0
    weakest_link = None
    business_flows = set()
    
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        
        if graph.has_edge(edge[0], edge[1]):
            edge_attrs = graph.edges[edge]
            total_latency += edge_attrs.get('avg_latency_ms', 0)
            
            if edge_attrs.get('business_flow'):
                business_flows.add(edge_attrs['business_flow'])
            
            if edge in critical_edge_map:
                critical_edge_count += 1
                edge_criticality = critical_edge_map[edge]['criticality_score']
                
                if edge_criticality > max_criticality:
                    max_criticality = edge_criticality
                    weakest_link = f"{edge[0]} → {edge[1]}"
    
    # Calculate path criticality score
    path_length = len(path) - 1
    criticality_ratio = critical_edge_count / max(1, path_length)
    
    score = (max_criticality * 0.5) + (criticality_ratio * 0.3) + (min(1.0, total_latency/500) * 0.2)
    
    return {
        'score': score,
        'total_latency': total_latency,
        'business_flow': ', '.join(business_flows) if business_flows else 'mixed',
        'weakest_link': weakest_link
    }

def generate_edge_recommendations(critical_edges):
    """
    Generate specific recommendations for critical edges
    """
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS FOR CRITICAL EDGES")
    print(f"{'='*80}")
    
    for edge_result in critical_edges[:3]:  # Top 3 critical edges
        edge = edge_result['edge']
        factors = edge_result['critical_factors']
        attrs = edge_result['attributes']
        
        print(f"\nFor Edge: {edge[0]} → {edge[1]}")
        print(f"Criticality Score: {edge_result['criticality_score']:.3f}")
        print("\nRecommendations:")
        
        recommendations = []
        
        # Check each critical factor and generate specific recommendations
        if 'synchronous_blocking' in factors:
            recommendations.append({
                'priority': 1,
                'title': 'Convert to Asynchronous Pattern',
                'description': f'Convert synchronous call from {edge[0]} to {edge[1]} to async with callback',
                'implementation': 'Implement async messaging with correlation IDs for request tracking',
                'effort': 'Medium',
                'risk_reduction': 'High',
                'code_example': generate_async_pattern_example(edge)
            })
        
        if 'no_alternative_path' in factors:
            recommendations.append({
                'priority': 1,
                'title': 'Add Redundant Path',
                'description': f'Create alternative route for messages from {edge[0]} to functionality of {edge[1]}',
                'implementation': 'Deploy backup service or alternative topic routing',
                'effort': 'High',
                'risk_reduction': 'Very High',
                'architecture_change': generate_redundant_path_diagram(edge)
            })
        
        if 'bottleneck_edge' in factors:
            details = factors['bottleneck_edge']['details']
            recommendations.append({
                'priority': 2,
                'title': 'Scale Target Capacity',
                'description': f'Increase processing capacity of {edge[1]} (currently at {details["target_utilization"]})',
                'implementation': 'Horizontal scaling with load balancing or vertical scaling',
                'effort': 'Medium',
                'risk_reduction': 'High',
                'scaling_strategy': generate_scaling_strategy(edge[1], details)
            })
        
        if 'high_latency_sensitivity' in factors:
            details = factors['high_latency_sensitivity']['details']
            recommendations.append({
                'priority': 2,
                'title': 'Optimize Latency',
                'description': f'Reduce P99 latency from {details["p99_latency"]}ms to <{details["max_allowed"]*0.5}ms',
                'implementation': 'Cache frequently accessed data, optimize queries, use connection pooling',
                'effort': 'Medium',
                'risk_reduction': 'Medium',
                'optimization_targets': generate_latency_optimizations(edge, details)
            })
        
        if 'transaction_boundary' in factors:
            recommendations.append({
                'priority': 1,
                'title': 'Implement Saga Pattern',
                'description': 'Replace distributed transaction with saga for better fault tolerance',
                'implementation': 'Use choreography or orchestration-based saga',
                'effort': 'High',
                'risk_reduction': 'High',
                'pattern': generate_saga_pattern_example(edge, attrs)
            })
        
        if 'regulatory_compliance' in factors:
            recommendations.append({
                'priority': 1,
                'title': 'Enhance Compliance Controls',
                'description': 'Strengthen audit logging and encryption for regulatory data',
                'implementation': 'Add comprehensive audit trail and end-to-end encryption',
                'effort': 'Medium',
                'risk_reduction': 'Critical',
                'compliance_checklist': generate_compliance_checklist(edge, factors['regulatory_compliance']['details'])
            })
        
        # Sort and display recommendations
        recommendations.sort(key=lambda x: x['priority'])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n   {i}. [{rec['priority']}] {rec['title']}")
            print(f"      Description: {rec['description']}")
            print(f"      Implementation: {rec['implementation']}")
            print(f"      Effort: {rec['effort']} | Risk Reduction: {rec['risk_reduction']}")
            
            # Show additional details if available
            if 'code_example' in rec:
                print(f"      Example Code:\n{rec['code_example']}")
            if 'scaling_strategy' in rec:
                print(f"      Scaling Strategy: {rec['scaling_strategy']}")

def generate_async_pattern_example(edge):
    """
    Generate code example for async pattern
    """
    return f"""
        // Before (Synchronous):
        result = {edge[0]}.publish_and_wait("{edge[1]}", message)
        
        // After (Asynchronous):
        correlation_id = generate_uuid()
        {edge[0]}.publish_async("{edge[1]}", message, correlation_id)
        {edge[0]}.register_callback(correlation_id, handle_response)
    """

def generate_scaling_strategy(target, details):
    """
    Generate scaling strategy based on utilization
    """
    current_util = float(details['target_utilization'].rstrip('%'))
    
    if current_util > 90:
        return "Immediate horizontal scaling: Deploy 2-3 additional instances with load balancing"
    elif current_util > 80:
        return "Proactive scaling: Deploy 1 additional instance and monitor"
    else:
        return "Optimize first: Profile and optimize code, then consider scaling"
    
def generate_redundant_path_diagram(edge):
    """
    Placeholder for architecture diagram generation
    """
    return f"Diagram showing {edge[0]} publishing to an alternative topic or service that {edge[1]} subscribes to."

def generate_latency_optimizations(edge, details):
    """
    Generate latency optimization suggestions
    """
    return f"""
        - Implement caching layer between {edge[0]} and {edge[1]}
        - Optimize database queries in {edge[1]} to reduce processing time
        - Use connection pooling for database access in {edge[1]}
    """

def generate_saga_pattern_example(edge, attrs):
    """
    Generate saga pattern example
    """
    return f"""
        // Saga Orchestration Example:
        begin_saga("OrderProcessing")
            .step("{edge[0]}", "ReserveInventory", on_success="PaymentService.ChargeCustomer", on_failure="CompensateInventory")
            .step("PaymentService", "ChargeCustomer", on_success="ShippingService.ScheduleShipment", on_failure="CompensatePayment")
            .step("ShippingService", "ScheduleShipment", on_success="CompleteSaga", on_failure="CompensateShipment")
        end_saga()
    """

def generate_compliance_checklist(edge, details):
    """
    Generate compliance checklist
    """
    checklist = [
        "Ensure end-to-end encryption is enabled",
        "Implement detailed audit logging for all transactions",
        "Regularly review access controls and permissions",
        "Conduct periodic compliance audits",
        "Maintain data residency requirements"
    ]
    
    if details.get('contains_pii', False):
        checklist.append("Implement data masking for PII fields")
    
    return "\n        - " + "\n        - ".join(checklist)
    
def visualize_critical_edges(graph, critical_edges):
    """
    Visualize the graph with critical edges highlighted
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Prepare edge colors and widths
    edge_criticality_map = {edge['edge']: edge['criticality_score'] 
                            for edge in critical_edges}
    
    edge_colors = []
    edge_widths = []
    edge_styles = []
    
    for edge in graph.edges():
        if edge in edge_criticality_map:
            criticality = edge_criticality_map[edge]
            if criticality > 0.8:
                edge_colors.append('#FF0000')  # Red for very critical
                edge_widths.append(4.0)
                edge_styles.append('solid')
            elif criticality > 0.6:
                edge_colors.append('#FF8C00')  # Orange for critical
                edge_widths.append(3.0)
                edge_styles.append('solid')
            else:
                edge_colors.append('#FFD700')  # Gold for moderate
                edge_widths.append(2.0)
                edge_styles.append('dashed')
        else:
            edge_colors.append('#D3D3D3')  # Light gray for normal
            edge_widths.append(1.0)
            edge_styles.append('dotted')
    
    # Node colors based on type
    node_colors = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type')
        if node_type == 'application':
            criticality = graph.nodes[node].get('criticality', 'low')
            if criticality == 'critical':
                node_colors.append('#8B0000')  # Dark red
            elif criticality == 'high':
                node_colors.append('#FF6B6B')  # Light red
            else:
                node_colors.append('#90EE90')  # Light green
        elif node_type == 'topic':
            node_colors.append('#87CEEB')  # Sky blue
        else:
            node_colors.append('#DDA0DD')  # Plum
    
    # First subplot: Full graph with critical edges
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=500, ax=ax1)
    
    # Draw edges with varying styles
    for i, edge in enumerate(graph.edges()):
        nx.draw_networkx_edges(graph, pos, [edge], 
                               edge_color=[edge_colors[i]], 
                               width=edge_widths[i],
                               style=edge_styles[i],
                               arrows=True, ax=ax1,
                               arrowsize=10,
                               connectionstyle="arc3,rad=0.1")
    
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax1)
    
    ax1.set_title("Full System with Critical Edges Highlighted", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Second subplot: Critical edges subgraph
    critical_edge_list = [edge['edge'] for edge in critical_edges[:10]]
    critical_subgraph = graph.edge_subgraph(critical_edge_list).copy()
    
    pos2 = nx.spring_layout(critical_subgraph, k=2, iterations=50)
    
    # Draw critical subgraph
    nx.draw(critical_subgraph, pos2, ax=ax2,
            node_color=['#FF6B6B' if critical_subgraph.nodes[n].get('type') == 'application' 
                       else '#87CEEB' for n in critical_subgraph.nodes()],
            node_size=700,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='#FF0000',
            width=3,
            arrows=True,
            arrowsize=15)
    
    ax2.set_title("Critical Edges Subgraph", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='#FF0000', linewidth=3, label='Very Critical (>0.8)'),
        plt.Line2D([0], [0], color='#FF8C00', linewidth=2, label='Critical (0.6-0.8)'),
        plt.Line2D([0], [0], color='#FFD700', linewidth=2, label='Moderate (0.4-0.6)'),
        plt.Line2D([0], [0], color='#D3D3D3', linewidth=1, label='Normal (<0.4)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Execute the complete demonstration
if __name__ == "__main__":
    demonstrate_critical_edge_detection()
    
    # Generate graph and analyze
    graph = create_enhanced_ecommerce_graph()
    analyzer = CriticalEdgeAnalyzer(graph)
    critical_edges = analyzer.analyze_all_edges()
    
    # Generate recommendations
    generate_edge_recommendations(critical_edges)
    
    # Visualize
    visualize_critical_edges(graph, critical_edges)