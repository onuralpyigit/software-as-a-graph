#!/usr/bin/env python3
"""
Software-as-a-Graph: End-to-End Demo
=====================================

This comprehensive demo showcases the complete pipeline for analyzing
distributed pub-sub systems using graph-based modeling:

1. GENERATE  - Create realistic pub-sub system graph data
2. BUILD     - Construct NetworkX graph model
3. ANALYZE   - Apply fuzzy logic criticality analysis
4. SIMULATE  - Run traffic simulation with failure injection
5. VISUALIZE - Generate interactive visualizations and reports

Scenarios Demonstrated:
- IoT Smart City deployment
- Financial Trading Platform
- Healthcare Monitoring System

Author: Software-as-a-Graph Research Project
Version: 1.0
"""

import sys
import json
import random
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import networkx as nx
except ImportError:
    print("❌ Error: networkx required. Install with: pip install networkx")
    sys.exit(1)

# Import our modules
try:
    from src.analysis.fuzzy_criticality_analyzer import (
        FuzzyCriticalityAnalyzer,
        FuzzyNodeCriticalityResult,
        FuzzyEdgeCriticalityResult,
        FuzzyCriticalityLevel,
        DefuzzificationMethod,
        analyze_graph_with_fuzzy_logic
    )
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️  Fuzzy logic module not found, will use composite scoring")


# ============================================================================
# Configuration
# ============================================================================

class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


class Scenario(Enum):
    """Available demo scenarios"""
    IOT_SMART_CITY = "iot_smart_city"
    FINANCIAL_TRADING = "financial_trading"
    HEALTHCARE = "healthcare"
    GENERIC = "generic"


# ============================================================================
# Graph Generator
# ============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for graph generation"""
    scenario: Scenario = Scenario.IOT_SMART_CITY
    num_nodes: int = 5
    num_brokers: int = 3
    num_topics: int = 15
    num_applications: int = 25
    connection_density: float = 0.6
    seed: Optional[int] = 42
    inject_spof: bool = True
    inject_god_topic: bool = True


class PubSubGraphGenerator:
    """
    Generates realistic pub-sub system graphs for different scenarios
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        if config.seed:
            random.seed(config.seed)
        
        self.logger = logging.getLogger('generator')
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete pub-sub system graph"""
        self.logger.info(f"Generating {self.config.scenario.value} scenario...")
        
        if self.config.scenario == Scenario.IOT_SMART_CITY:
            return self._generate_iot_smart_city()
        elif self.config.scenario == Scenario.FINANCIAL_TRADING:
            return self._generate_financial_trading()
        elif self.config.scenario == Scenario.HEALTHCARE:
            return self._generate_healthcare()
        else:
            return self._generate_generic()
    
    def _generate_iot_smart_city(self) -> Dict[str, Any]:
        """Generate IoT Smart City scenario"""
        graph = {
            'metadata': {
                'scenario': 'iot_smart_city',
                'generated_at': datetime.now().isoformat(),
                'description': 'Smart City IoT deployment with sensors, gateways, and analytics'
            },
            'nodes': [],
            'brokers': [],
            'topics': [],
            'applications': [],
            'publishes': [],
            'subscribes': [],
            'broker_routes': {}
        }
        
        # Infrastructure nodes (edge servers, cloud servers)
        node_types = ['edge_gateway', 'edge_gateway', 'fog_server', 'cloud_server', 'cloud_server']
        for i, ntype in enumerate(node_types[:self.config.num_nodes]):
            graph['nodes'].append({
                'id': f'N{i+1}',
                'name': f'{ntype.replace("_", " ").title()} {i+1}',
                'type': ntype,
                'ip_address': f'10.0.{i}.1',
                'location': ['district_a', 'district_b', 'datacenter', 'cloud_us', 'cloud_eu'][i % 5]
            })
        
        # Message brokers
        broker_types = ['mosquitto', 'kafka', 'rabbitmq']
        for i in range(self.config.num_brokers):
            node_idx = i % len(graph['nodes'])
            graph['brokers'].append({
                'id': f'B{i+1}',
                'name': f'Broker_{broker_types[i % len(broker_types)]}_{i+1}',
                'type': broker_types[i % len(broker_types)],
                'node': graph['nodes'][node_idx]['id'],
                'max_connections': 10000,
                'protocol': 'MQTT' if 'mosquitto' in broker_types[i % len(broker_types)] else 'AMQP'
            })
        
        # Topics organized by domain
        topic_domains = {
            'traffic': ['vehicle_count', 'speed_data', 'congestion_alerts', 'signal_status'],
            'environment': ['air_quality', 'temperature', 'humidity', 'noise_level'],
            'energy': ['power_consumption', 'grid_status', 'solar_output', 'battery_level'],
            'safety': ['emergency_alerts', 'surveillance_events', 'incident_reports']
        }
        
        topic_idx = 0
        for domain, topics in topic_domains.items():
            for topic_name in topics:
                if topic_idx >= self.config.num_topics:
                    break
                broker_idx = topic_idx % len(graph['brokers'])
                
                # Make first topic a "god topic" if configured
                qos = 'RELIABLE' if domain in ['safety', 'energy'] else 'BEST_EFFORT'
                
                graph['topics'].append({
                    'id': f'T{topic_idx+1}',
                    'name': f'{domain}/{topic_name}',
                    'broker': graph['brokers'][broker_idx]['id'],
                    'domain': domain,
                    'qos': {'reliability': qos, 'deadline_ms': 100 if qos == 'RELIABLE' else 1000}
                })
                topic_idx += 1
        
        # Applications
        app_types = [
            ('sensor', 'publisher'),      # Sensors publish data
            ('gateway', 'both'),          # Gateways relay data
            ('analytics', 'subscriber'),  # Analytics consume data
            ('dashboard', 'subscriber'),  # Dashboards display data
            ('controller', 'both'),       # Controllers send commands
            ('alert_service', 'both'),    # Alert services monitor and notify
            ('data_logger', 'subscriber') # Loggers store data
        ]
        
        for i in range(self.config.num_applications):
            app_type, role = app_types[i % len(app_types)]
            node_idx = i % len(graph['nodes'])
            
            graph['applications'].append({
                'id': f'A{i+1}',
                'name': f'{app_type.title()}_{i+1}',
                'type': app_type,
                'role': role,
                'node': graph['nodes'][node_idx]['id'],
                'criticality': 'high' if app_type in ['controller', 'alert_service'] else 'normal'
            })
        
        # Create publish/subscribe relationships
        self._create_pubsub_relationships(graph)
        
        # Create broker routes
        self._create_broker_routes(graph)
        
        return graph
    
    def _generate_financial_trading(self) -> Dict[str, Any]:
        """Generate Financial Trading Platform scenario"""
        graph = {
            'metadata': {
                'scenario': 'financial_trading',
                'generated_at': datetime.now().isoformat(),
                'description': 'High-frequency trading platform with market data and execution'
            },
            'nodes': [],
            'brokers': [],
            'topics': [],
            'applications': [],
            'publishes': [],
            'subscribes': [],
            'broker_routes': {}
        }
        
        # Trading infrastructure
        for i in range(self.config.num_nodes):
            node_type = ['matching_engine', 'market_data_server', 'execution_server', 
                        'risk_server', 'backup_server'][i % 5]
            graph['nodes'].append({
                'id': f'N{i+1}',
                'name': f'{node_type.replace("_", " ").title()}',
                'type': node_type,
                'ip_address': f'172.16.{i}.1',
                'location': 'primary_dc' if i < 3 else 'backup_dc'
            })
        
        # Message brokers (low-latency)
        for i in range(self.config.num_brokers):
            node_idx = i % len(graph['nodes'])
            graph['brokers'].append({
                'id': f'B{i+1}',
                'name': f'TradingBroker_{i+1}',
                'type': 'aeron' if i == 0 else 'kafka',
                'node': graph['nodes'][node_idx]['id'],
                'latency_us': 10 if i == 0 else 100
            })
        
        # Trading topics
        trading_topics = [
            ('market_data/quotes', 'RELIABLE', 1),
            ('market_data/trades', 'RELIABLE', 1),
            ('market_data/orderbook', 'RELIABLE', 5),
            ('orders/new', 'RELIABLE', 1),
            ('orders/cancel', 'RELIABLE', 1),
            ('orders/modify', 'RELIABLE', 1),
            ('execution/fills', 'RELIABLE', 1),
            ('execution/rejects', 'RELIABLE', 1),
            ('risk/positions', 'RELIABLE', 10),
            ('risk/limits', 'RELIABLE', 10),
            ('risk/alerts', 'RELIABLE', 1),
            ('admin/heartbeat', 'BEST_EFFORT', 1000),
            ('admin/config', 'RELIABLE', 5000),
            ('analytics/pnl', 'BEST_EFFORT', 100),
            ('analytics/metrics', 'BEST_EFFORT', 1000)
        ]
        
        for i, (name, reliability, deadline) in enumerate(trading_topics[:self.config.num_topics]):
            broker_idx = 0 if 'market_data' in name or 'orders' in name else i % len(graph['brokers'])
            graph['topics'].append({
                'id': f'T{i+1}',
                'name': name,
                'broker': graph['brokers'][broker_idx]['id'],
                'qos': {'reliability': reliability, 'deadline_ms': deadline}
            })
        
        # Trading applications
        trading_apps = [
            'MarketDataFeed', 'OrderRouter', 'ExecutionEngine', 'RiskManager',
            'AlgoTrader_1', 'AlgoTrader_2', 'AlgoTrader_3', 'MarketMaker',
            'PositionKeeper', 'PnLCalculator', 'ComplianceMonitor', 'AuditLogger',
            'AdminConsole', 'AlertService', 'BackupService'
        ]
        
        for i, app_name in enumerate(trading_apps[:self.config.num_applications]):
            node_idx = i % len(graph['nodes'])
            is_critical = any(x in app_name for x in ['Execution', 'Risk', 'Order', 'Market'])
            
            graph['applications'].append({
                'id': f'A{i+1}',
                'name': app_name,
                'type': 'trading_component',
                'node': graph['nodes'][node_idx]['id'],
                'criticality': 'critical' if is_critical else 'normal'
            })
        
        self._create_pubsub_relationships(graph)
        self._create_broker_routes(graph)
        
        return graph
    
    def _generate_healthcare(self) -> Dict[str, Any]:
        """Generate Healthcare Monitoring scenario"""
        graph = {
            'metadata': {
                'scenario': 'healthcare',
                'generated_at': datetime.now().isoformat(),
                'description': 'Hospital patient monitoring and alert system'
            },
            'nodes': [],
            'brokers': [],
            'topics': [],
            'applications': [],
            'publishes': [],
            'subscribes': [],
            'broker_routes': {}
        }
        
        # Hospital infrastructure
        node_configs = [
            ('icu_server', 'ICU'),
            ('ward_server_a', 'Ward A'),
            ('ward_server_b', 'Ward B'),
            ('central_server', 'Central'),
            ('backup_server', 'Backup')
        ]
        
        for i, (ntype, location) in enumerate(node_configs[:self.config.num_nodes]):
            graph['nodes'].append({
                'id': f'N{i+1}',
                'name': f'{ntype.replace("_", " ").title()}',
                'type': ntype,
                'ip_address': f'192.168.{i+1}.1',
                'location': location
            })
        
        # Healthcare message brokers
        for i in range(self.config.num_brokers):
            node_idx = min(i, len(graph['nodes']) - 1)
            graph['brokers'].append({
                'id': f'B{i+1}',
                'name': f'HealthBroker_{i+1}',
                'type': 'rabbitmq',
                'node': graph['nodes'][node_idx]['id'],
                'hipaa_compliant': True
            })
        
        # Healthcare topics
        health_topics = [
            ('vitals/heart_rate', 'RELIABLE', 100),
            ('vitals/blood_pressure', 'RELIABLE', 500),
            ('vitals/oxygen_saturation', 'RELIABLE', 100),
            ('vitals/temperature', 'RELIABLE', 1000),
            ('alerts/critical', 'RELIABLE', 10),
            ('alerts/warning', 'RELIABLE', 100),
            ('alerts/info', 'BEST_EFFORT', 1000),
            ('medication/dispensing', 'RELIABLE', 100),
            ('medication/schedule', 'RELIABLE', 1000),
            ('lab/results', 'RELIABLE', 5000),
            ('imaging/status', 'RELIABLE', 5000),
            ('nurse/call', 'RELIABLE', 100),
            ('admin/bed_status', 'BEST_EFFORT', 5000),
            ('admin/staff_location', 'BEST_EFFORT', 1000),
            ('audit/access_log', 'RELIABLE', 10000)
        ]
        
        for i, (name, reliability, deadline) in enumerate(health_topics[:self.config.num_topics]):
            # Critical topics go to primary broker
            broker_idx = 0 if 'critical' in name or 'heart' in name else i % len(graph['brokers'])
            graph['topics'].append({
                'id': f'T{i+1}',
                'name': name,
                'broker': graph['brokers'][broker_idx]['id'],
                'qos': {'reliability': reliability, 'deadline_ms': deadline}
            })
        
        # Healthcare applications
        health_apps = [
            'PatientMonitor_ICU', 'PatientMonitor_WardA', 'PatientMonitor_WardB',
            'VitalSignsCollector', 'AlertManager', 'NurseStation_ICU', 
            'NurseStation_WardA', 'NurseStation_WardB', 'DoctorDashboard',
            'MedicationDispenser', 'LabInterface', 'ImagingInterface',
            'EMRIntegration', 'AuditLogger', 'BackupService'
        ]
        
        for i, app_name in enumerate(health_apps[:self.config.num_applications]):
            node_idx = i % len(graph['nodes'])
            is_critical = any(x in app_name for x in ['ICU', 'Alert', 'Vital', 'Medication'])
            
            graph['applications'].append({
                'id': f'A{i+1}',
                'name': app_name,
                'type': 'healthcare_component',
                'node': graph['nodes'][node_idx]['id'],
                'criticality': 'critical' if is_critical else 'normal'
            })
        
        self._create_pubsub_relationships(graph)
        self._create_broker_routes(graph)
        
        return graph
    
    def _generate_generic(self) -> Dict[str, Any]:
        """Generate generic pub-sub system"""
        graph = {
            'metadata': {
                'scenario': 'generic',
                'generated_at': datetime.now().isoformat()
            },
            'nodes': [],
            'brokers': [],
            'topics': [],
            'applications': [],
            'publishes': [],
            'subscribes': [],
            'broker_routes': {}
        }
        
        # Generate nodes
        for i in range(self.config.num_nodes):
            graph['nodes'].append({
                'id': f'N{i+1}',
                'name': f'Node_{i+1}',
                'ip_address': f'10.0.0.{i+1}'
            })
        
        # Generate brokers
        for i in range(self.config.num_brokers):
            graph['brokers'].append({
                'id': f'B{i+1}',
                'name': f'Broker_{i+1}',
                'node': graph['nodes'][i % len(graph['nodes'])]['id']
            })
        
        # Generate topics
        for i in range(self.config.num_topics):
            graph['topics'].append({
                'id': f'T{i+1}',
                'name': f'topic_{i+1}',
                'broker': graph['brokers'][i % len(graph['brokers'])]['id'],
                'qos': {'reliability': 'RELIABLE' if i % 3 == 0 else 'BEST_EFFORT'}
            })
        
        # Generate applications
        for i in range(self.config.num_applications):
            graph['applications'].append({
                'id': f'A{i+1}',
                'name': f'App_{i+1}',
                'node': graph['nodes'][i % len(graph['nodes'])]['id']
            })
        
        self._create_pubsub_relationships(graph)
        self._create_broker_routes(graph)
        
        return graph
    
    def _create_pubsub_relationships(self, graph: Dict):
        """Create publish/subscribe relationships"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Determine publishers and subscribers based on app role/type
        for app in apps:
            app_id = app['id']
            app_type = app.get('type', 'generic')
            role = app.get('role', 'both')
            
            # Determine number of topics to connect to
            num_pub_topics = random.randint(1, min(3, len(topics)))
            num_sub_topics = random.randint(1, min(5, len(topics)))
            
            if role in ['publisher', 'both']:
                pub_topics = random.sample(topics, num_pub_topics)
                for topic in pub_topics:
                    graph['publishes'].append({
                        'application': app_id,
                        'topic': topic['id']
                    })
            
            if role in ['subscriber', 'both']:
                sub_topics = random.sample(topics, num_sub_topics)
                for topic in sub_topics:
                    graph['subscribes'].append({
                        'topic': topic['id'],
                        'application': app_id
                    })
        
        # If god topic injection is enabled, make first topic have many subscribers
        if self.config.inject_god_topic and topics:
            god_topic = topics[0]
            for app in apps[:min(10, len(apps))]:
                if not any(s['topic'] == god_topic['id'] and s['application'] == app['id'] 
                          for s in graph['subscribes']):
                    graph['subscribes'].append({
                        'topic': god_topic['id'],
                        'application': app['id']
                    })
    
    def _create_broker_routes(self, graph: Dict):
        """Create broker-to-broker routes"""
        brokers = graph['brokers']
        
        if len(brokers) > 1:
            # Create mesh routing between brokers
            for i, broker in enumerate(brokers):
                routes = []
                for j, other in enumerate(brokers):
                    if i != j:
                        routes.append(other['id'])
                graph['broker_routes'][broker['id']] = routes


# ============================================================================
# Graph Builder
# ============================================================================

class GraphBuilder:
    """Builds NetworkX graph from generated data"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger('builder')
    
    def build(self) -> nx.DiGraph:
        """Build complete graph"""
        self.logger.info("Building NetworkX graph model...")
        
        # Add nodes - handle 'type' field carefully
        for node in self.data.get('nodes', []):
            node_copy = {k: v for k, v in node.items() if k != 'id'}
            if 'type' not in node_copy:
                node_copy['type'] = 'Node'
            self.graph.add_node(node['id'], **node_copy)
        
        for broker in self.data.get('brokers', []):
            broker_copy = {k: v for k, v in broker.items() if k != 'id'}
            broker_copy['type'] = 'Broker'
            self.graph.add_node(broker['id'], **broker_copy)
        
        for topic in self.data.get('topics', []):
            topic_copy = {k: v for k, v in topic.items() if k != 'id'}
            topic_copy['type'] = 'Topic'
            self.graph.add_node(topic['id'], **topic_copy)
        
        for app in self.data.get('applications', []):
            app_copy = {k: v for k, v in app.items() if k != 'id'}
            app_copy['type'] = 'Application'
            self.graph.add_node(app['id'], **app_copy)
        
        # Add edges
        for pub in self.data.get('publishes', []):
            self.graph.add_edge(pub['application'], pub['topic'], type='PUBLISHES')
        
        for sub in self.data.get('subscribes', []):
            self.graph.add_edge(sub['topic'], sub['application'], type='SUBSCRIBES')
        
        # Topic to broker (HOSTS_ON)
        for topic in self.data.get('topics', []):
            if 'broker' in topic:
                self.graph.add_edge(topic['id'], topic['broker'], type='HOSTS_ON')
        
        # App/Broker to Node (RUNS_ON)
        for app in self.data.get('applications', []):
            if 'node' in app:
                self.graph.add_edge(app['id'], app['node'], type='RUNS_ON')
        
        for broker in self.data.get('brokers', []):
            if 'node' in broker:
                self.graph.add_edge(broker['id'], broker['node'], type='RUNS_ON')
        
        # Broker routes
        for source, targets in self.data.get('broker_routes', {}).items():
            for target in targets:
                self.graph.add_edge(source, target, type='ROUTES_TO')
        
        self.logger.info(f"Built graph: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        
        return self.graph


# ============================================================================
# Simplified Analyzer (works without fuzzy module)
# ============================================================================

class SimpleAnalyzer:
    """Simple graph analyzer using composite scoring"""
    
    def __init__(self, graph: nx.DiGraph, alpha=0.4, beta=0.3, gamma=0.3):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logger = logging.getLogger('analyzer')
    
    def analyze(self) -> Dict[str, Any]:
        """Run complete analysis"""
        self.logger.info("Analyzing graph structure and criticality...")
        
        results = {
            'structure': self._analyze_structure(),
            'node_criticality': self._analyze_node_criticality(),
            'edge_criticality': self._analyze_edge_criticality(),
            'anti_patterns': self._detect_anti_patterns()
        }
        
        return results
    
    def _analyze_structure(self) -> Dict:
        """Analyze graph structure"""
        G = self.graph
        
        node_types = defaultdict(int)
        for _, data in G.nodes(data=True):
            node_types[data.get('type', 'Unknown')] += 1
        
        return {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': round(nx.density(G), 4),
            'is_connected': nx.is_weakly_connected(G),
            'components': nx.number_weakly_connected_components(G),
            'node_types': dict(node_types)
        }
    
    def _analyze_node_criticality(self) -> Dict:
        """Analyze node criticality"""
        G = self.graph
        
        # Calculate metrics
        betweenness = nx.betweenness_centrality(G, normalized=True)
        max_bc = max(betweenness.values()) if betweenness else 1.0
        
        undirected = G.to_undirected()
        articulation_points = set(nx.articulation_points(undirected))
        
        # Calculate scores
        scores = {}
        for node in G.nodes():
            bc_norm = betweenness.get(node, 0) / max_bc if max_bc > 0 else 0
            is_ap = node in articulation_points
            
            # Simple impact estimation
            degree = G.degree(node)
            max_degree = max(d for _, d in G.degree()) if G.degree() else 1
            impact = degree / max_degree
            
            # Composite score
            score = self.alpha * bc_norm + self.beta * (1.0 if is_ap else 0.0) + self.gamma * impact
            
            # Level classification
            if score >= 0.7:
                level = 'critical'
            elif score >= 0.5:
                level = 'high'
            elif score >= 0.3:
                level = 'medium'
            elif score >= 0.15:
                level = 'low'
            else:
                level = 'minimal'
            
            scores[node] = {
                'score': round(score, 4),
                'level': level,
                'betweenness': round(bc_norm, 4),
                'is_articulation_point': is_ap,
                'impact': round(impact, 4),
                'type': G.nodes[node].get('type', 'Unknown')
            }
        
        # Sort by score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Statistics
        all_scores = [s['score'] for _, s in sorted_nodes]
        level_counts = defaultdict(int)
        for _, s in sorted_nodes:
            level_counts[s['level']] += 1
        
        return {
            'statistics': {
                'total': len(scores),
                'avg_score': round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
                'articulation_points': len(articulation_points),
                'by_level': dict(level_counts)
            },
            'top_critical': [
                {'node': n, **s} for n, s in sorted_nodes[:10]
            ],
            'all_scores': scores
        }
    
    def _analyze_edge_criticality(self) -> Dict:
        """Analyze edge criticality"""
        G = self.graph
        
        edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
        max_ebc = max(edge_betweenness.values()) if edge_betweenness else 1.0
        
        undirected = G.to_undirected()
        bridges = set(nx.bridges(undirected))
        
        edges = []
        for (u, v), score in edge_betweenness.items():
            is_bridge = (u, v) in bridges or (v, u) in bridges
            edges.append({
                'from': u,
                'to': v,
                'score': round(score / max_ebc if max_ebc > 0 else 0, 4),
                'is_bridge': is_bridge,
                'type': G.edges[u, v].get('type', 'Unknown')
            })
        
        edges.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'statistics': {
                'total': len(edges),
                'bridges': sum(1 for e in edges if e['is_bridge'])
            },
            'top_critical': edges[:10]
        }
    
    def _detect_anti_patterns(self) -> List[Dict]:
        """Detect anti-patterns"""
        G = self.graph
        patterns = []
        
        # God topics (topics with many subscribers)
        topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
        for topic in topics:
            subs = len([e for e in G.out_edges(topic) if G.edges[e].get('type') == 'SUBSCRIBES'])
            if subs > 8:
                patterns.append({
                    'type': 'god_topic',
                    'component': topic,
                    'severity': 'high' if subs > 12 else 'medium',
                    'details': f'{subs} subscribers'
                })
        
        # Broker bottlenecks
        brokers = [n for n, d in G.nodes(data=True) if d.get('type') == 'Broker']
        for broker in brokers:
            topics_hosted = len([e for e in G.in_edges(broker) if G.edges[e].get('type') == 'HOSTS_ON'])
            if topics_hosted > 10:
                patterns.append({
                    'type': 'broker_bottleneck',
                    'component': broker,
                    'severity': 'medium',
                    'details': f'hosts {topics_hosted} topics'
                })
        
        return patterns


# ============================================================================
# Lightweight Simulator
# ============================================================================

class SimpleSimulator:
    """Lightweight traffic and failure simulator"""
    
    def __init__(self, graph: nx.DiGraph, speedup: int = 100):
        self.graph = graph
        self.speedup = speedup
        self.logger = logging.getLogger('simulator')
        
        # Extract components
        self.applications = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Application']
        self.topics = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Topic']
        self.brokers = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Broker']
        
        # Statistics
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.latencies = []
        self.failed_components = set()
    
    async def run_baseline(self, duration: float = 30) -> Dict:
        """Run baseline simulation"""
        self.logger.info(f"Running baseline simulation ({duration}s)...")
        return await self._simulate(duration)
    
    async def run_with_failure(self, duration: float = 30, 
                               fail_component: str = None,
                               fail_time: float = 10) -> Dict:
        """Run simulation with failure injection"""
        self.logger.info(f"Running failure simulation ({duration}s, fail {fail_component} at {fail_time}s)...")
        
        self._reset()
        
        start_time = time.time()
        current_time = 0.0
        
        while current_time < duration:
            # Check for failure injection
            if fail_component and current_time >= fail_time and fail_component not in self.failed_components:
                self.failed_components.add(fail_component)
                self.logger.warning(f"⚠️  Injected failure: {fail_component}")
                
                # Find cascade affected components
                affected = self._get_cascade_affected(fail_component)
                for comp in affected:
                    self.failed_components.add(comp)
                    self.logger.warning(f"   → Cascade: {comp}")
            
            # Simulate message flow
            for app in self.applications:
                if app not in self.failed_components:
                    self._simulate_publish(app)
            
            current_time += 1.0 / 10  # 10 Hz simulation rate
            await asyncio.sleep(1.0 / 10 / self.speedup)
        
        elapsed = time.time() - start_time
        
        return self._get_stats(elapsed)
    
    async def _simulate(self, duration: float) -> Dict:
        """Internal simulation loop"""
        self._reset()
        
        start_time = time.time()
        current_time = 0.0
        
        while current_time < duration:
            for app in self.applications:
                if app not in self.failed_components:
                    self._simulate_publish(app)
            
            current_time += 1.0 / 10
            await asyncio.sleep(1.0 / 10 / self.speedup)
        
        elapsed = time.time() - start_time
        return self._get_stats(elapsed)
    
    def _simulate_publish(self, app: str):
        """Simulate message publishing"""
        # Find topics this app publishes to
        for _, target, data in self.graph.out_edges(app, data=True):
            if data.get('type') == 'PUBLISHES' and target not in self.failed_components:
                self.messages_sent += 1
                
                # Find broker for topic
                broker_ok = True
                for _, broker, bdata in self.graph.out_edges(target, data=True):
                    if bdata.get('type') == 'HOSTS_ON':
                        if broker in self.failed_components:
                            broker_ok = False
                        break
                
                if broker_ok:
                    # Calculate latency
                    latency = random.uniform(1, 10)  # 1-10ms
                    self.latencies.append(latency)
                    self.messages_delivered += 1
                else:
                    self.messages_dropped += 1
    
    def _get_cascade_affected(self, component: str) -> List[str]:
        """Get components affected by cascading failure"""
        affected = []
        comp_type = self.graph.nodes[component].get('type')
        
        if comp_type == 'Broker':
            # Broker failure affects hosted topics
            for topic, _, data in self.graph.in_edges(component, data=True):
                if data.get('type') == 'HOSTS_ON':
                    affected.append(topic)
        
        elif comp_type == 'Node':
            # Node failure affects apps/brokers running on it
            for comp, _, data in self.graph.in_edges(component, data=True):
                if data.get('type') == 'RUNS_ON':
                    affected.append(comp)
        
        return affected
    
    def _reset(self):
        """Reset simulation state"""
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.latencies = []
        self.failed_components = set()
    
    def _get_stats(self, elapsed: float) -> Dict:
        """Get simulation statistics"""
        return {
            'elapsed_seconds': round(elapsed, 2),
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_dropped': self.messages_dropped,
            'delivery_rate': round(self.messages_delivered / max(1, self.messages_sent) * 100, 2),
            'avg_latency_ms': round(sum(self.latencies) / max(1, len(self.latencies)), 2),
            'max_latency_ms': round(max(self.latencies) if self.latencies else 0, 2),
            'throughput': round(self.messages_sent / max(0.1, elapsed), 1),
            'failed_components': list(self.failed_components)
        }


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generates analysis reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        scenario: str,
        graph_data: Dict,
        analysis: Dict,
        baseline: Dict,
        failure: Dict,
        fuzzy_results: Optional[Dict] = None
    ) -> str:
        """Generate comprehensive HTML report"""
        
        # Calculate comparison
        comparison = {
            'latency_increase': round(failure['avg_latency_ms'] - baseline['avg_latency_ms'], 2),
            'throughput_decrease': round(baseline['throughput'] - failure['throughput'], 1),
            'delivery_loss': round(baseline['delivery_rate'] - failure['delivery_rate'], 2),
            'messages_lost': failure['messages_dropped'] - baseline['messages_dropped']
        }
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Software-as-a-Graph Analysis Report</title>
    <style>
        :root {{
            --primary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            text-align: center;
            padding: 40px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(52, 152, 219, 0.3);
        }}
        header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        header p {{ color: rgba(255,255,255,0.7); }}
        .scenario-badge {{
            display: inline-block;
            padding: 8px 20px;
            background: var(--primary);
            border-radius: 20px;
            margin-top: 15px;
            font-weight: bold;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            color: var(--primary);
            margin-bottom: 20px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-value {{ font-weight: bold; color: var(--primary); }}
        .metric-value.success {{ color: var(--success); }}
        .metric-value.danger {{ color: var(--danger); }}
        .metric-value.warning {{ color: var(--warning); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ background: rgba(52, 152, 219, 0.2); }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .badge-critical {{ background: var(--danger); }}
        .badge-high {{ background: var(--warning); }}
        .badge-medium {{ background: var(--primary); }}
        .badge-low {{ background: var(--success); }}
        .badge-minimal {{ background: #7f8c8d; }}
        .comparison-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .comparison-item {{
            text-align: center;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }}
        .comparison-value {{ font-size: 1.8rem; font-weight: bold; }}
        .comparison-label {{ font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: 5px; }}
        .full-width {{ grid-column: 1 / -1; }}
        .progress-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{ height: 100%; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Software-as-a-Graph Analysis Report</h1>
            <p>Comprehensive Pub-Sub System Analysis</p>
            <div class="scenario-badge">{scenario.replace('_', ' ').title()}</div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Graph Structure</h2>
                <div class="metric">
                    <span>Total Nodes</span>
                    <span class="metric-value">{analysis['structure']['nodes']}</span>
                </div>
                <div class="metric">
                    <span>Total Edges</span>
                    <span class="metric-value">{analysis['structure']['edges']}</span>
                </div>
                <div class="metric">
                    <span>Density</span>
                    <span class="metric-value">{analysis['structure']['density']}</span>
                </div>
                <div class="metric">
                    <span>Connected</span>
                    <span class="metric-value {'success' if analysis['structure']['is_connected'] else 'danger'}">
                        {'✓ Yes' if analysis['structure']['is_connected'] else '✗ No'}
                    </span>
                </div>
            </div>
            
            <div class="card">
                <h2>🏗️ Component Types</h2>
                {''.join(f'''
                <div class="metric">
                    <span>{t}</span>
                    <span class="metric-value">{c}</span>
                </div>
                ''' for t, c in analysis['structure']['node_types'].items())}
            </div>
            
            <div class="card">
                <h2>🎯 Criticality Summary</h2>
                <div class="metric">
                    <span>Avg Criticality</span>
                    <span class="metric-value">{analysis['node_criticality']['statistics']['avg_score']}</span>
                </div>
                <div class="metric">
                    <span>Articulation Points</span>
                    <span class="metric-value warning">{analysis['node_criticality']['statistics']['articulation_points']}</span>
                </div>
                {''.join(f'''
                <div class="metric">
                    <span>{level.title()}</span>
                    <span class="metric-value">{count}</span>
                </div>
                ''' for level, count in analysis['node_criticality']['statistics']['by_level'].items())}
            </div>
        </div>
        
        <div class="card full-width" style="margin-bottom: 30px;">
            <h2>⚡ Simulation Results: Baseline vs Failure</h2>
            <div class="comparison-grid">
                <div class="comparison-item">
                    <div class="comparison-value {'danger' if comparison['latency_increase'] > 0 else 'success'}">
                        +{comparison['latency_increase']}ms
                    </div>
                    <div class="comparison-label">Latency Increase</div>
                </div>
                <div class="comparison-item">
                    <div class="comparison-value danger">
                        -{comparison['throughput_decrease']}/s
                    </div>
                    <div class="comparison-label">Throughput Loss</div>
                </div>
                <div class="comparison-item">
                    <div class="comparison-value danger">
                        -{comparison['delivery_loss']}%
                    </div>
                    <div class="comparison-label">Delivery Rate Drop</div>
                </div>
                <div class="comparison-item">
                    <div class="comparison-value danger">
                        {comparison['messages_lost']}
                    </div>
                    <div class="comparison-label">Messages Lost</div>
                </div>
            </div>
            
            <table style="margin-top: 20px;">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline</th>
                        <th>After Failure</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Messages Sent</td>
                        <td>{baseline['messages_sent']:,}</td>
                        <td>{failure['messages_sent']:,}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Messages Delivered</td>
                        <td>{baseline['messages_delivered']:,}</td>
                        <td>{failure['messages_delivered']:,}</td>
                        <td class="danger">-{baseline['messages_delivered'] - failure['messages_delivered']:,}</td>
                    </tr>
                    <tr>
                        <td>Delivery Rate</td>
                        <td>{baseline['delivery_rate']}%</td>
                        <td>{failure['delivery_rate']}%</td>
                        <td class="danger">-{comparison['delivery_loss']}%</td>
                    </tr>
                    <tr>
                        <td>Avg Latency</td>
                        <td>{baseline['avg_latency_ms']}ms</td>
                        <td>{failure['avg_latency_ms']}ms</td>
                        <td class="{'danger' if comparison['latency_increase'] > 0 else ''}">+{comparison['latency_increase']}ms</td>
                    </tr>
                    <tr>
                        <td>Throughput</td>
                        <td>{baseline['throughput']}/s</td>
                        <td>{failure['throughput']}/s</td>
                        <td class="danger">-{comparison['throughput_decrease']}/s</td>
                    </tr>
                </tbody>
            </table>
            
            <p style="margin-top: 15px; color: rgba(255,255,255,0.7);">
                <strong>Failed Components:</strong> {', '.join(failure['failed_components']) if failure['failed_components'] else 'None'}
            </p>
        </div>
        
        <div class="card full-width" style="margin-bottom: 30px;">
            <h2>🎯 Top Critical Nodes</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Node</th>
                        <th>Type</th>
                        <th>Score</th>
                        <th>Level</th>
                        <th>Articulation Point</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{i+1}</td>
                        <td><strong>{n['node']}</strong></td>
                        <td>{n['type']}</td>
                        <td>{n['score']}</td>
                        <td><span class="badge badge-{n['level']}">{n['level'].upper()}</span></td>
                        <td>{'✓' if n['is_articulation_point'] else ''}</td>
                    </tr>
                    ''' for i, n in enumerate(analysis['node_criticality']['top_critical'][:10]))}
                </tbody>
            </table>
        </div>
        
        
        {'<div class="card full-width" style="margin-bottom: 30px;"><h2>⚠️ Anti-Patterns Detected</h2>' + ''.join(f'''
            <div class="metric">
                <span><strong>[{p['type'].upper()}]</strong> {p['component']}</span>
                <span class="badge badge-{p['severity']}">{p['severity'].upper()}</span>
            </div>
            <p style="color: rgba(255,255,255,0.6); margin-bottom: 15px; padding-left: 20px;">{p['details']}</p>
        ''' for p in analysis['anti_patterns']) + '</div>' if analysis['anti_patterns'] else ''}
        
        <footer style="text-align: center; padding: 30px; color: rgba(255,255,255,0.5);">
            <p>Generated by Software-as-a-Graph Analysis Framework</p>
            <p>Research: Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems</p>
            <p style="margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>"""
        
        output_path = self.output_dir / f'{scenario}_report.html'
        with open(output_path, 'w') as f:
            f.write(html)
        
        return str(output_path)
    
    def generate_json_report(self, scenario: str, data: Dict) -> str:
        """Generate JSON report"""
        output_path = self.output_dir / f'{scenario}_results.json'
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return str(output_path)


# ============================================================================
# Interactive Visualization
# ============================================================================

def generate_interactive_visualization(
    graph: nx.DiGraph,
    analysis: Dict,
    output_path: Path
) -> str:
    """Generate interactive D3.js visualization"""
    
    # Prepare nodes
    nodes_data = []
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Get criticality
        crit_data = analysis['node_criticality']['all_scores'].get(node, {})
        score = crit_data.get('score', 0)
        level = crit_data.get('level', 'minimal')
        
        # Color by type
        colors = {
            'Application': '#3498db',
            'Broker': '#e74c3c',
            'Topic': '#2ecc71',
            'Node': '#9b59b6'
        }
        
        nodes_data.append({
            'id': node,
            'label': node,
            'type': node_type,
            'color': colors.get(node_type, '#95a5a6'),
            'size': 10 + score * 20,
            'score': score,
            'level': level
        })
    
    # Prepare edges
    edges_data = []
    edge_colors = {
        'PUBLISHES': '#3498db',
        'SUBSCRIBES': '#2ecc71',
        'HOSTS_ON': '#e74c3c',
        'RUNS_ON': '#9b59b6',
        'ROUTES_TO': '#f39c12'
    }
    
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        edges_data.append({
            'source': u,
            'target': v,
            'type': edge_type,
            'color': edge_colors.get(edge_type, '#95a5a6')
        })
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Graph Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ 
            margin: 0; 
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: white;
        }}
        #graph {{ width: 100%; height: 100vh; }}
        #info {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 10px;
            max-width: 300px;
        }}
        #legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="info">
        <h3>Graph Statistics</h3>
        <p>Nodes: {len(graph.nodes())}</p>
        <p>Edges: {len(graph.edges())}</p>
        <p>Click a node for details</p>
        <div id="nodeDetails"></div>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div>Application</div>
        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>Broker</div>
        <div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div>Topic</div>
        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>Node</div>
    </div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes_data)});
        const edges = new vis.DataSet({json.dumps(edges_data)});
        
        const container = document.getElementById('graph');
        const data = {{ nodes, edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                font: {{ color: '#ffffff', size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: 'to',
                smooth: {{ type: 'continuous' }}
            }},
            physics: {{
                stabilization: {{ iterations: 100 }},
                barnesHut: {{ gravitationalConstant: -3000, springLength: 150 }}
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                document.getElementById('nodeDetails').innerHTML = `
                    <hr>
                    <p><strong>${{node.id}}</strong></p>
                    <p>Type: ${{node.type}}</p>
                    <p>Score: ${{node.score.toFixed(4)}}</p>
                    <p>Level: ${{node.level}}</p>
                `;
            }}
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return str(output_path)


# ============================================================================
# Main Demo Runner
# ============================================================================

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*70}{Colors.END}")


def print_step(step: int, total: int, text: str):
    """Print step indicator"""
    print(f"\n{Colors.CYAN}[{step}/{total}] {text}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}  {text}{Colors.END}")


async def run_demo(scenario: Scenario, output_dir: Path):
    """Run complete demo for a scenario"""
    
    print_header(f"DEMO: {scenario.value.replace('_', ' ').upper()}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # ========================================
    # STEP 1: Generate Graph Data
    # ========================================
    print_step(1, 5, "GENERATING GRAPH DATA")
    
    config = GeneratorConfig(
        scenario=scenario,
        num_nodes=5,
        num_brokers=3,
        num_topics=15,
        num_applications=20,
        seed=42
    )
    
    generator = PubSubGraphGenerator(config)
    graph_data = generator.generate()
    
    # Save generated data
    data_path = output_dir / f'{scenario.value}_graph.json'
    with open(data_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print_success(f"Generated {scenario.value} system")
    print_info(f"Nodes: {len(graph_data['nodes'])}")
    print_info(f"Brokers: {len(graph_data['brokers'])}")
    print_info(f"Topics: {len(graph_data['topics'])}")
    print_info(f"Applications: {len(graph_data['applications'])}")
    print_info(f"Saved to: {data_path}")
    
    # ========================================
    # STEP 2: Build Graph Model
    # ========================================
    print_step(2, 5, "BUILDING GRAPH MODEL")
    
    builder = GraphBuilder(graph_data)
    G = builder.build()
    
    print_success("Built NetworkX graph model")
    print_info(f"Graph nodes: {len(G.nodes())}")
    print_info(f"Graph edges: {len(G.edges())}")
    print_info(f"Density: {nx.density(G):.4f}")
    print_info(f"Connected: {nx.is_weakly_connected(G)}")
    
    # ========================================
    # STEP 3: Analyze Graph
    # ========================================
    print_step(3, 5, "ANALYZING GRAPH")
    
    # Try fuzzy analysis first
    fuzzy_results = None
    if FUZZY_AVAILABLE:
        print_info("Using Fuzzy Logic analysis...")
        analyzer = FuzzyCriticalityAnalyzer(DefuzzificationMethod.CENTROID)
        node_results, edge_results = analyze_graph_with_fuzzy_logic(G, calculate_impact=True)
        
        # Convert to analysis format
        analysis = {
            'structure': {
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': round(nx.density(G), 4),
                'is_connected': nx.is_weakly_connected(G),
                'components': nx.number_weakly_connected_components(G),
                'node_types': dict(defaultdict(int, {
                    G.nodes[n].get('type', 'Unknown'): sum(1 for m in G.nodes() 
                        if G.nodes[m].get('type') == G.nodes[n].get('type'))
                    for n in G.nodes()
                }))
            },
            'node_criticality': {
                'statistics': {
                    'total': len(node_results),
                    'avg_score': round(sum(r.fuzzy_criticality_score for r in node_results.values()) / len(node_results), 4),
                    'articulation_points': sum(1 for r in node_results.values() if r.is_articulation_point),
                    'by_level': dict(defaultdict(int, {
                        r.criticality_level.value: sum(1 for s in node_results.values() 
                            if s.criticality_level == r.criticality_level)
                        for r in node_results.values()
                    }))
                },
                'top_critical': sorted([
                    {
                        'node': r.component,
                        'type': r.component_type,
                        'score': round(r.fuzzy_criticality_score, 4),
                        'level': r.criticality_level.value,
                        'is_articulation_point': r.is_articulation_point,
                        'betweenness': round(r.betweenness_centrality_norm, 4),
                        'impact': round(r.impact_score, 4)
                    }
                    for r in node_results.values()
                ], key=lambda x: x['score'], reverse=True)[:10],
                'all_scores': {
                    n: {
                        'score': round(r.fuzzy_criticality_score, 4),
                        'level': r.criticality_level.value,
                        'betweenness': round(r.betweenness_centrality_norm, 4),
                        'is_articulation_point': r.is_articulation_point,
                        'impact': round(r.impact_score, 4),
                        'type': r.component_type
                    }
                    for n, r in node_results.items()
                }
            },
            'edge_criticality': {
                'statistics': {
                    'total': len(edge_results),
                    'bridges': sum(1 for r in edge_results.values() if r.is_bridge)
                },
                'top_critical': sorted([
                    {
                        'from': r.source,
                        'to': r.target,
                        'score': round(r.fuzzy_criticality_score, 4),
                        'is_bridge': r.is_bridge,
                        'type': r.edge_type
                    }
                    for r in edge_results.values()
                ], key=lambda x: x['score'], reverse=True)[:10]
            },
            'anti_patterns': []
        }
        
        # Detect anti-patterns
        topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
        for topic in topics:
            subs = len([e for e in G.out_edges(topic) if G.edges[e].get('type') == 'SUBSCRIBES'])
            if subs > 8:
                analysis['anti_patterns'].append({
                    'type': 'god_topic',
                    'component': topic,
                    'severity': 'high' if subs > 12 else 'medium',
                    'details': f'{subs} subscribers'
                })
        
        fuzzy_results = {'node_results': node_results, 'edge_results': edge_results}
        print_success("Fuzzy logic analysis complete")
    else:
        print_info("Using Composite Scoring analysis...")
        simple_analyzer = SimpleAnalyzer(G)
        analysis = simple_analyzer.analyze()
        print_success("Composite scoring analysis complete")
    
    # Print analysis summary
    print_info(f"Critical nodes: {analysis['node_criticality']['statistics']['by_level'].get('critical', 0)}")
    print_info(f"High criticality: {analysis['node_criticality']['statistics']['by_level'].get('high', 0)}")
    print_info(f"Articulation points: {analysis['node_criticality']['statistics']['articulation_points']}")
    print_info(f"Anti-patterns: {len(analysis.get('anti_patterns', []))}")
    
    # Show top critical nodes
    print(f"\n{Colors.YELLOW}  Top 5 Critical Nodes:{Colors.END}")
    for i, node in enumerate(analysis['node_criticality']['top_critical'][:5], 1):
        ap = " [AP]" if node['is_articulation_point'] else ""
        print(f"    {i}. {node['node']}{ap} - Score: {node['score']:.4f} ({node['level']})")
    
    # ========================================
    # STEP 4: Simulate Traffic & Failures
    # ========================================
    print_step(4, 5, "SIMULATING TRAFFIC & FAILURES")
    
    simulator = SimpleSimulator(G, speedup=500)
    
    # Run baseline
    print_info("Running baseline simulation (30s)...")
    baseline = await simulator.run_baseline(duration=30)
    print_success(f"Baseline: {baseline['messages_delivered']:,} delivered, {baseline['delivery_rate']}% rate")
    
    # Find most critical broker for failure
    critical_broker = None
    for node in analysis['node_criticality']['top_critical']:
        if node['type'] == 'Broker':
            critical_broker = node['node']
            break
    
    if not critical_broker and graph_data['brokers']:
        critical_broker = graph_data['brokers'][0]['id']
    
    # Run with failure
    print_info(f"Running failure simulation (fail {critical_broker} at 10s)...")
    failure = await simulator.run_with_failure(
        duration=30,
        fail_component=critical_broker,
        fail_time=10
    )
    print_success(f"Failure: {failure['messages_delivered']:,} delivered, {failure['delivery_rate']}% rate")
    print_info(f"Messages lost: {failure['messages_dropped'] - baseline['messages_dropped']}")
    print_info(f"Cascade affected: {len(failure['failed_components'])} components")
    
    # ========================================
    # STEP 5: Generate Reports & Visualizations
    # ========================================
    print_step(5, 5, "GENERATING REPORTS & VISUALIZATIONS")
    
    report_gen = ReportGenerator(output_dir)
    
    # Generate HTML report
    html_path = report_gen.generate_html_report(
        scenario.value,
        graph_data,
        analysis,
        baseline,
        failure,
        fuzzy_results
    )
    print_success(f"HTML report: {html_path}")
    
    # Generate JSON report
    json_data = {
        'metadata': graph_data['metadata'],
        'analysis': analysis,
        'simulation': {
            'baseline': baseline,
            'failure': failure
        }
    }
    json_path = report_gen.generate_json_report(scenario.value, json_data)
    print_success(f"JSON report: {json_path}")
    
    # Generate interactive visualization
    viz_path = output_dir / f'{scenario.value}_interactive.html'
    generate_interactive_visualization(G, analysis, viz_path)
    print_success(f"Interactive visualization: {viz_path}")
    
    print_header("DEMO COMPLETE")
    print(f"\n{Colors.GREEN}All artifacts saved to: {output_dir}{Colors.END}\n")
    
    return {
        'graph_data': graph_data,
        'graph': G,
        'analysis': analysis,
        'baseline': baseline,
        'failure': failure,
        'output_dir': output_dir
    }


async def main():
    """Main entry point"""
    print(f"""
{Colors.HEADER}{'='*70}
  SOFTWARE-AS-A-GRAPH: END-TO-END DEMO
  Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems
{'='*70}{Colors.END}

This demo showcases the complete pipeline:
  1. Generate realistic pub-sub system graph data
  2. Build NetworkX graph model  
  3. Analyze criticality (Fuzzy Logic or Composite Scoring)
  4. Simulate traffic and failure scenarios
  5. Generate interactive reports and visualizations
""")
    
    # Create output directory
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Run demos for different scenarios
    scenarios = [
        Scenario.IOT_SMART_CITY,
        Scenario.FINANCIAL_TRADING,
        Scenario.HEALTHCARE
    ]
    
    results = {}
    for scenario in scenarios:
        try:
            result = await run_demo(scenario, output_dir / scenario.value)
            results[scenario.value] = result
        except Exception as e:
            print(f"{Colors.RED}Error in {scenario.value}: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"""
{Colors.HEADER}{'='*70}
  DEMO SUMMARY
{'='*70}{Colors.END}

{Colors.GREEN}✓ Successfully generated {len(results)} scenario demos{Colors.END}

Output files available in: {output_dir.absolute()}

For each scenario:
  • *_graph.json      - Generated system definition
  • *_results.json    - Analysis and simulation results  
  • *_report.html     - Comprehensive analysis report
  • *_interactive.html - Interactive graph visualization

{Colors.CYAN}Try opening the HTML reports in your browser!{Colors.END}
""")


if __name__ == '__main__':
    asyncio.run(main())
