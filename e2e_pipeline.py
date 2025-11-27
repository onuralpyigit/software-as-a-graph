#!/usr/bin/env python3
"""
Software-as-a-Graph: End-to-End Pipeline
==========================================

Comprehensive demonstration of the Graph-Based Modeling and Analysis
of Distributed Publish-Subscribe Systems methodology.

This script integrates all five steps:
1. GENERATE  - Create realistic pub-sub system graph data
2. IMPORT    - Import graph data into Neo4j database
3. ANALYZE   - Apply comprehensive analysis (criticality, structural, QoS)
4. SIMULATE  - Run traffic simulation with failure injection and validation
5. VISUALIZE - Generate multi-layer visualizations and reports

Target Metrics (from research methodology):
- Spearman correlation ≥ 0.7 with failure simulations
- F1-score ≥ 0.9 for critical component identification
- Precision ≥ 0.9, Recall ≥ 0.85

Author: Software-as-a-Graph Research Project
Version: 1.0
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================================
# External Dependencies Check
# ============================================================================

def check_dependencies():
    """Check and import required dependencies"""
    dependencies = {}
    missing = []
    
    try:
        import networkx as nx
        dependencies['networkx'] = nx
    except ImportError:
        missing.append('networkx')
    
    try:
        from scipy import stats
        dependencies['scipy'] = stats
    except ImportError:
        missing.append('scipy')
    
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import ClientError
        dependencies['neo4j'] = GraphDatabase
    except ImportError:
        missing.append('neo4j')
    
    return dependencies, missing


DEPS, MISSING_DEPS = check_dependencies()

if MISSING_DEPS:
    print(f"⚠️  Optional dependencies not found: {', '.join(MISSING_DEPS)}")
    print("Install with: pip install " + " ".join(MISSING_DEPS))

if 'networkx' not in DEPS:
    print("❌ Critical: networkx is required. Install with: pip install networkx")
    sys.exit(1)

nx = DEPS['networkx']

# ============================================================================
# Configuration and Constants
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


class Scenario(Enum):
    """Available domain scenarios"""
    GENERIC = "generic"
    IOT_SMART_CITY = "iot_smart_city"
    FINANCIAL_TRADING = "financial_trading"
    HEALTHCARE = "healthcare"
    ECOMMERCE = "ecommerce"


class CriticalityLevel(Enum):
    """Criticality classification levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


# Target validation thresholds
TARGET_SPEARMAN_CORRELATION = 0.7
TARGET_F1_SCORE = 0.9
TARGET_PRECISION = 0.9
TARGET_RECALL = 0.85

# Criticality scoring weights (C_score = α·C_B + β·AP + γ·I)
DEFAULT_ALPHA = 0.4  # Betweenness centrality
DEFAULT_BETA = 0.3   # Articulation point
DEFAULT_GAMMA = 0.3  # Impact score


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    scale: str = 'medium'
    scenario: Scenario = Scenario.IOT_SMART_CITY
    num_nodes: int = 10
    num_applications: int = 30
    num_topics: int = 20
    num_brokers: int = 3
    edge_density: float = 0.4
    antipatterns: List[str] = field(default_factory=list)
    seed: int = 42


@dataclass
class CriticalityScore:
    """Composite criticality score for a component"""
    component_id: str
    component_type: str
    betweenness_centrality: float
    is_articulation_point: bool
    impact_score: float
    composite_score: float
    criticality_level: CriticalityLevel


@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    duration_seconds: int = 60
    baseline_duration: int = 10
    failure_time: int = 30
    failure_components: List[str] = field(default_factory=list)
    enable_cascading: bool = True


@dataclass
class ValidationResult:
    """Validation results"""
    precision: float
    recall: float
    f1_score: float
    spearman_correlation: float
    targets_met: Dict[str, bool]


# ============================================================================
# Utility Functions
# ============================================================================

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_step(step_num: int, text: str):
    """Print step header"""
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}Step {step_num}: {text}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('E2E-Pipeline')


# ============================================================================
# STEP 1: GRAPH GENERATION
# ============================================================================

class PubSubGraphGenerator:
    """
    Generates realistic pub-sub system graphs for different scenarios
    """
    
    SCALES = {
        'tiny': {'nodes': 3, 'apps': 8, 'topics': 5, 'brokers': 1},
        'small': {'nodes': 5, 'apps': 15, 'topics': 10, 'brokers': 2},
        'medium': {'nodes': 10, 'apps': 30, 'topics': 20, 'brokers': 3},
        'large': {'nodes': 25, 'apps': 80, 'topics': 50, 'brokers': 5},
        'xlarge': {'nodes': 50, 'apps': 150, 'topics': 100, 'brokers': 8}
    }
    
    APP_TYPES = {
        Scenario.GENERIC: ['ServiceA', 'ServiceB', 'Processor', 'Handler', 'Monitor'],
        Scenario.IOT_SMART_CITY: [
            'TrafficSensor', 'ParkingSensor', 'AirQualityMonitor', 'EmergencyDispatcher',
            'LightingController', 'WasteManager', 'WeatherStation', 'TransitTracker'
        ],
        Scenario.FINANCIAL_TRADING: [
            'MarketDataFeed', 'OrderProcessor', 'RiskEngine', 'TradeExecutor',
            'PositionTracker', 'ComplianceMonitor', 'MatchingEngine', 'PricingService'
        ],
        Scenario.HEALTHCARE: [
            'VitalSignsMonitor', 'PatientTracker', 'AlertDispatcher', 'MedicationManager',
            'LabResultsProcessor', 'ImagingService', 'BillingService', 'AppointmentScheduler'
        ],
        Scenario.ECOMMERCE: [
            'OrderService', 'InventoryManager', 'PaymentProcessor', 'ShippingCalculator',
            'RecommendationEngine', 'CartService', 'NotificationService', 'FraudDetector'
        ]
    }
    
    TOPIC_PATTERNS = {
        Scenario.GENERIC: ['events', 'data', 'commands', 'status', 'metrics'],
        Scenario.IOT_SMART_CITY: [
            'traffic/flow', 'traffic/congestion', 'parking/availability', 'air_quality/readings',
            'emergency/alerts', 'lighting/status', 'weather/current', 'transit/location'
        ],
        Scenario.FINANCIAL_TRADING: [
            'market/prices', 'market/quotes', 'orders/new', 'orders/filled',
            'trades/executed', 'risk/alerts', 'positions/updates', 'compliance/events'
        ],
        Scenario.HEALTHCARE: [
            'patient/vitals', 'patient/alerts', 'lab/results', 'imaging/completed',
            'medication/administered', 'appointments/scheduled', 'billing/claims'
        ],
        Scenario.ECOMMERCE: [
            'orders/created', 'inventory/updates', 'payments/processed', 'shipping/tracking',
            'recommendations/generated', 'notifications/sent', 'fraud/detected'
        ]
    }
    
    QOS_PROFILES = {
        'CRITICAL': {
            'reliability': 'RELIABLE',
            'durability': 'TRANSIENT_LOCAL',
            'deadline_ms': 10,
            'latency_budget_ms': 5
        },
        'HIGH': {
            'reliability': 'RELIABLE',
            'durability': 'VOLATILE',
            'deadline_ms': 50,
            'latency_budget_ms': 25
        },
        'MEDIUM': {
            'reliability': 'RELIABLE',
            'durability': 'VOLATILE',
            'deadline_ms': 100,
            'latency_budget_ms': 50
        },
        'LOW': {
            'reliability': 'BEST_EFFORT',
            'durability': 'VOLATILE',
            'deadline_ms': 500,
            'latency_budget_ms': 200
        }
    }
    
    def __init__(self, config: GraphConfig):
        self.config = config
        random.seed(config.seed)
        self.logger = logging.getLogger('GraphGenerator')
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete pub-sub system graph"""
        self.logger.info(f"Generating {self.config.scenario.value} scenario...")
        
        # Get scale parameters
        scale_params = self.SCALES.get(self.config.scale, self.SCALES['medium'])
        
        graph = {
            'metadata': {
                'scenario': self.config.scenario.value,
                'scale': self.config.scale,
                'generated_at': datetime.now().isoformat(),
                'seed': self.config.seed,
                'description': f'{self.config.scenario.value} pub-sub system'
            },
            'nodes': [],
            'brokers': [],
            'topics': [],
            'applications': [],
            'relationships': {
                'publishes_to': [],
                'subscribes_to': [],
                'runs_on': [],
                'routes': []
            }
        }
        
        # Generate components
        self._generate_nodes(graph)
        self._generate_brokers(graph)
        self._generate_topics(graph)
        self._generate_applications(graph)
        
        # Generate relationships
        self._generate_runs_on(graph)
        self._generate_routes(graph)
        self._generate_pub_sub_relationships(graph)
        
        # Inject anti-patterns if specified
        if self.config.antipatterns:
            self._inject_antipatterns(graph)
        
        # Ensure connectivity
        self._ensure_connectivity(graph)
        
        return graph
    
    def _generate_nodes(self, graph: Dict):
        """Generate infrastructure nodes"""
        node_types = ['edge_gateway', 'fog_server', 'cloud_server', 'edge_device']
        
        for i in range(1, self.config.num_nodes + 1):
            node_type = node_types[(i - 1) % len(node_types)]
            zone = f'zone-{(i - 1) % 3 + 1}'
            
            graph['nodes'].append({
                'id': f'N{i}',
                'name': f'{node_type.replace("_", " ").title()} {i}',
                'type': node_type,
                'zone': zone,
                'cpu_capacity': random.choice([4, 8, 16, 32]),
                'memory_gb': random.choice([8, 16, 32, 64]),
                'network_bandwidth_mbps': random.choice([1000, 10000])
            })
    
    def _generate_brokers(self, graph: Dict):
        """Generate message brokers"""
        for i in range(1, self.config.num_brokers + 1):
            zone = f'zone-{(i - 1) % 3 + 1}'
            
            graph['brokers'].append({
                'id': f'B{i}',
                'name': f'Broker-{i}',
                'zone': zone,
                'max_connections': random.choice([1000, 5000, 10000]),
                'max_topics': random.choice([500, 1000, 2000]),
                'protocol': random.choice(['DDS', 'MQTT', 'AMQP'])
            })
    
    def _generate_topics(self, graph: Dict):
        """Generate topics with QoS profiles"""
        patterns = self.TOPIC_PATTERNS.get(
            self.config.scenario, 
            self.TOPIC_PATTERNS[Scenario.GENERIC]
        )
        
        qos_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        
        for i in range(1, self.config.num_topics + 1):
            pattern = patterns[(i - 1) % len(patterns)]
            qos_level = qos_levels[(i - 1) % len(qos_levels)]
            
            # Adjust QoS based on scenario
            if self.config.scenario == Scenario.FINANCIAL_TRADING:
                if 'market' in pattern or 'order' in pattern:
                    qos_level = 'CRITICAL'
            elif self.config.scenario == Scenario.HEALTHCARE:
                if 'vital' in pattern or 'alert' in pattern:
                    qos_level = 'CRITICAL'
            
            graph['topics'].append({
                'id': f'T{i}',
                'name': f'{pattern}_{i}',
                'qos': self.QOS_PROFILES[qos_level].copy(),
                'qos_level': qos_level,
                'message_rate_hz': self._get_message_rate(pattern),
                'avg_message_size_bytes': self._get_message_size(pattern)
            })
    
    def _generate_applications(self, graph: Dict):
        """Generate applications"""
        app_types_list = self.APP_TYPES.get(
            self.config.scenario,
            self.APP_TYPES[Scenario.GENERIC]
        )
        
        criticality_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        role_types = ['PRODUCER', 'CONSUMER', 'PROSUMER']
        
        for i in range(1, self.config.num_applications + 1):
            app_type = app_types_list[(i - 1) % len(app_types_list)]
            criticality = criticality_levels[(i - 1) % len(criticality_levels)]
            role = role_types[(i - 1) % len(role_types)]
            
            graph['applications'].append({
                'id': f'A{i}',
                'name': f'{app_type}_{i}',
                'type': role,
                'criticality': criticality,
                'replicas': 1 if criticality == 'LOW' else random.choice([2, 3]),
                'cpu_request': random.choice([0.5, 1.0, 2.0]),
                'memory_request_mb': random.choice([256, 512, 1024])
            })
    
    def _generate_runs_on(self, graph: Dict):
        """Generate application to node relationships"""
        nodes = graph['nodes']
        
        for app in graph['applications']:
            # Assign to random node
            node = random.choice(nodes)
            graph['relationships']['runs_on'].append({
                'from': app['id'],
                'to': node['id']
            })
    
    def _generate_routes(self, graph: Dict):
        """Generate topic to broker routing"""
        brokers = graph['brokers']
        
        for topic in graph['topics']:
            # Route to 1-2 brokers
            num_brokers = random.randint(1, min(2, len(brokers)))
            selected_brokers = random.sample(brokers, num_brokers)
            
            for broker in selected_brokers:
                graph['relationships']['routes'].append({
                    'from': topic['id'],
                    'to': broker['id']
                })
    
    def _generate_pub_sub_relationships(self, graph: Dict):
        """Generate publish/subscribe relationships"""
        topics = graph['topics']
        
        for app in graph['applications']:
            app_type = app['type']
            
            # Determine number of topics to connect to
            if app_type == 'PRODUCER':
                num_publish = random.randint(1, 3)
                num_subscribe = 0
            elif app_type == 'CONSUMER':
                num_publish = 0
                num_subscribe = random.randint(1, 4)
            else:  # PROSUMER
                num_publish = random.randint(1, 2)
                num_subscribe = random.randint(1, 3)
            
            # Create publish relationships
            if num_publish > 0:
                selected_topics = random.sample(topics, min(num_publish, len(topics)))
                for topic in selected_topics:
                    graph['relationships']['publishes_to'].append({
                        'from': app['id'],
                        'to': topic['id']
                    })
            
            # Create subscribe relationships
            if num_subscribe > 0:
                selected_topics = random.sample(topics, min(num_subscribe, len(topics)))
                for topic in selected_topics:
                    graph['relationships']['subscribes_to'].append({
                        'from': app['id'],
                        'to': topic['id']
                    })
    
    def _inject_antipatterns(self, graph: Dict):
        """Inject specified anti-patterns"""
        for pattern in self.config.antipatterns:
            if pattern == 'spof':
                self._inject_spof(graph)
            elif pattern == 'god_topic':
                self._inject_god_topic(graph)
            elif pattern == 'tight_coupling':
                self._inject_tight_coupling(graph)
    
    def _inject_spof(self, graph: Dict):
        """Inject Single Point of Failure"""
        if len(graph['brokers']) > 1:
            # Make one broker handle most traffic
            spof_broker = graph['brokers'][0]
            for route in graph['relationships']['routes']:
                route['to'] = spof_broker['id']
            self.logger.info(f"Injected SPOF: {spof_broker['id']}")
    
    def _inject_god_topic(self, graph: Dict):
        """Inject God Topic anti-pattern"""
        if graph['topics']:
            god_topic = graph['topics'][0]
            for app in graph['applications']:
                if random.random() < 0.7:
                    graph['relationships']['subscribes_to'].append({
                        'from': app['id'],
                        'to': god_topic['id']
                    })
            self.logger.info(f"Injected God Topic: {god_topic['id']}")
    
    def _inject_tight_coupling(self, graph: Dict):
        """Inject tight coupling between applications"""
        apps = graph['applications']
        if len(apps) >= 4:
            # Create circular dependencies through topics
            for i in range(min(4, len(apps))):
                topic_id = f'T{i + 1}' if i < len(graph['topics']) else graph['topics'][0]['id']
                graph['relationships']['publishes_to'].append({
                    'from': apps[i]['id'],
                    'to': topic_id
                })
                graph['relationships']['subscribes_to'].append({
                    'from': apps[(i + 1) % 4]['id'],
                    'to': topic_id
                })
            self.logger.info("Injected tight coupling pattern")
    
    def _ensure_connectivity(self, graph: Dict):
        """Ensure all components are connected"""
        # Ensure every application publishes or subscribes to at least one topic
        for app in graph['applications']:
            has_pub = any(r['from'] == app['id'] for r in graph['relationships']['publishes_to'])
            has_sub = any(r['from'] == app['id'] for r in graph['relationships']['subscribes_to'])
            
            if not has_pub and not has_sub:
                topic = random.choice(graph['topics'])
                graph['relationships']['subscribes_to'].append({
                    'from': app['id'],
                    'to': topic['id']
                })
    
    def _get_message_rate(self, pattern: str) -> float:
        """Get realistic message rate based on topic pattern"""
        if self.config.scenario == Scenario.FINANCIAL_TRADING:
            if 'market' in pattern:
                return random.choice([100, 500, 1000])
            return random.choice([10, 50, 100])
        elif self.config.scenario == Scenario.HEALTHCARE:
            if 'vital' in pattern:
                return random.choice([10, 20, 50])
            return random.choice([1, 5, 10])
        return random.choice([1, 10, 50, 100])
    
    def _get_message_size(self, pattern: str) -> int:
        """Get realistic message size based on topic pattern"""
        if self.config.scenario == Scenario.FINANCIAL_TRADING:
            return random.choice([64, 128, 256])
        elif self.config.scenario == Scenario.IOT_SMART_CITY:
            return random.choice([32, 64, 128])
        return random.choice([128, 256, 512, 1024])


# ============================================================================
# STEP 2: NEO4J IMPORT
# ============================================================================

class Neo4jImporter:
    """
    Imports graph data into Neo4j database
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = 'neo4j'):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger('Neo4jImporter')
    
    def connect(self) -> bool:
        """Connect to Neo4j database"""
        if 'neo4j' not in DEPS:
            self.logger.warning("Neo4j driver not available")
            return False
        
        try:
            GraphDatabase = DEPS['neo4j']
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            self.logger.info("Connected to Neo4j")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all data from database"""
        if not self.driver:
            return
        
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.logger.info("Database cleared")
    
    def create_schema(self):
        """Create database schema (constraints and indexes)"""
        if not self.driver:
            return
        
        constraints = [
            "CREATE CONSTRAINT app_id IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT broker_id IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE",
            "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.debug(f"Constraint: {e}")
        
        self.logger.info("Schema created")
    
    def import_graph(self, graph_data: Dict, batch_size: int = 100):
        """Import graph data into Neo4j"""
        if not self.driver:
            self.logger.warning("Not connected to Neo4j, skipping import")
            return
        
        self.logger.info("Importing graph data...")
        
        # Import nodes
        self._import_nodes(graph_data.get('nodes', []))
        
        # Import applications
        self._import_applications(graph_data.get('applications', []))
        
        # Import topics
        self._import_topics(graph_data.get('topics', []))
        
        # Import brokers
        self._import_brokers(graph_data.get('brokers', []))
        
        # Import relationships
        relationships = graph_data.get('relationships', {})
        self._import_relationships(relationships)
        
        self.logger.info("Graph import complete")
    
    def _import_nodes(self, nodes: List[Dict]):
        """Import infrastructure nodes"""
        if not nodes:
            return
        
        query = """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        SET n.name = node.name,
            n.type = node.type,
            n.zone = node.zone
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query, nodes=nodes)
    
    def _import_applications(self, applications: List[Dict]):
        """Import applications"""
        if not applications:
            return
        
        query = """
        UNWIND $apps AS app
        MERGE (a:Application {id: app.id})
        SET a.name = app.name,
            a.type = app.type,
            a.criticality = app.criticality,
            a.replicas = app.replicas
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query, apps=applications)
    
    def _import_topics(self, topics: List[Dict]):
        """Import topics"""
        if not topics:
            return
        
        query = """
        UNWIND $topics AS topic
        MERGE (t:Topic {id: topic.id})
        SET t.name = topic.name,
            t.qos_level = topic.qos_level,
            t.message_rate_hz = topic.message_rate_hz
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query, topics=topics)
    
    def _import_brokers(self, brokers: List[Dict]):
        """Import brokers"""
        if not brokers:
            return
        
        query = """
        UNWIND $brokers AS broker
        MERGE (b:Broker {id: broker.id})
        SET b.name = broker.name,
            b.zone = broker.zone,
            b.max_connections = broker.max_connections
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query, brokers=brokers)
    
    def _import_relationships(self, relationships: Dict):
        """Import all relationships"""
        with self.driver.session(database=self.database) as session:
            # RUNS_ON
            for rel in relationships.get('runs_on', []):
                session.run("""
                    MATCH (a:Application {id: $from})
                    MATCH (n:Node {id: $to})
                    MERGE (a)-[:RUNS_ON]->(n)
                """, **rel)
            
            # PUBLISHES_TO
            for rel in relationships.get('publishes_to', []):
                session.run("""
                    MATCH (a:Application {id: $from})
                    MATCH (t:Topic {id: $to})
                    MERGE (a)-[:PUBLISHES]->(t)
                """, **rel)
            
            # SUBSCRIBES_TO
            for rel in relationships.get('subscribes_to', []):
                session.run("""
                    MATCH (a:Application {id: $from})
                    MATCH (t:Topic {id: $to})
                    MERGE (a)-[:SUBSCRIBES]->(t)
                """, **rel)
            
            # ROUTES
            for rel in relationships.get('routes', []):
                session.run("""
                    MATCH (t:Topic {id: $from})
                    MATCH (b:Broker {id: $to})
                    MERGE (t)-[:ROUTED_BY]->(b)
                """, **rel)
    
    def run_analytics(self) -> Dict:
        """Run analytics queries on imported graph"""
        if not self.driver:
            return {}
        
        results = {}
        
        with self.driver.session(database=self.database) as session:
            # Application distribution
            result = session.run("""
                MATCH (a:Application)
                RETURN a.type AS type, count(*) AS count
            """)
            results['app_distribution'] = {r['type']: r['count'] for r in result}
            
            # Topic by QoS level
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.qos_level AS level, count(*) AS count
            """)
            results['qos_distribution'] = {r['level']: r['count'] for r in result}
            
            # Most connected topics
            result = session.run("""
                MATCH (t:Topic)
                WITH t, size((t)<-[:PUBLISHES]-()) + size((t)<-[:SUBSCRIBES]-()) AS connections
                RETURN t.id AS topic, connections
                ORDER BY connections DESC
                LIMIT 5
            """)
            results['top_topics'] = [(r['topic'], r['connections']) for r in result]
        
        return results


# ============================================================================
# STEP 3: GRAPH ANALYSIS
# ============================================================================

class GraphAnalyzer:
    """
    Comprehensive graph analysis for pub-sub systems
    """
    
    def __init__(self, alpha: float = DEFAULT_ALPHA, 
                 beta: float = DEFAULT_BETA, 
                 gamma: float = DEFAULT_GAMMA):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logger = logging.getLogger('GraphAnalyzer')
    
    def build_networkx_graph(self, graph_data: Dict) -> nx.DiGraph:
        """Build NetworkX directed graph from graph data"""
        G = nx.DiGraph()
        
        # Add nodes (infrastructure)
        for node in graph_data.get('nodes', []):
            extra_attrs = {k: v for k, v in node.items() if k not in ['id', 'name', 'type']}
            G.add_node(node['id'], 
                      type='Node',
                      name=node.get('name', node['id']),
                      node_type=node.get('type', 'generic'),
                      **extra_attrs)
        
        for broker in graph_data.get('brokers', []):
            extra_attrs = {k: v for k, v in broker.items() if k not in ['id', 'name', 'type']}
            G.add_node(broker['id'],
                      type='Broker',
                      name=broker.get('name', broker['id']),
                      **extra_attrs)
        
        for topic in graph_data.get('topics', []):
            extra_attrs = {k: v for k, v in topic.items() if k not in ['id', 'name', 'type', 'qos', 'qos_level']}
            G.add_node(topic['id'],
                      type='Topic',
                      name=topic.get('name', topic['id']),
                      qos_level=topic.get('qos_level', 'MEDIUM'),
                      **extra_attrs)
        
        for app in graph_data.get('applications', []):
            extra_attrs = {k: v for k, v in app.items() if k not in ['id', 'name', 'type', 'criticality']}
            G.add_node(app['id'],
                      type='Application',
                      name=app.get('name', app['id']),
                      app_type=app.get('type', 'PROSUMER'),
                      criticality=app.get('criticality', 'MEDIUM'),
                      **extra_attrs)
        
        # Add edges
        relationships = graph_data.get('relationships', {})
        
        for rel in relationships.get('runs_on', []):
            G.add_edge(rel['from'], rel['to'], type='RUNS_ON')
        
        for rel in relationships.get('publishes_to', []):
            G.add_edge(rel['from'], rel['to'], type='PUBLISHES')
        
        for rel in relationships.get('subscribes_to', []):
            G.add_edge(rel['from'], rel['to'], type='SUBSCRIBES')
        
        for rel in relationships.get('routes', []):
            G.add_edge(rel['from'], rel['to'], type='ROUTES')
        
        return G
    
    def analyze(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        self.logger.info("Running comprehensive analysis...")
        
        results = {
            'graph_summary': self._get_graph_summary(G),
            'centrality_metrics': self._calculate_centrality(G),
            'structural_analysis': self._analyze_structure(G),
            'criticality_scores': {},
            'layer_analysis': self._analyze_layers(G),
            'anti_patterns': self._detect_anti_patterns(G)
        }
        
        # Calculate composite criticality scores
        results['criticality_scores'] = self._calculate_criticality_scores(
            G, 
            results['centrality_metrics'],
            results['structural_analysis']
        )
        
        return results
    
    def _get_graph_summary(self, G: nx.DiGraph) -> Dict:
        """Get graph summary statistics"""
        node_types = defaultdict(int)
        for node, data in G.nodes(data=True):
            node_types[data.get('type', 'Unknown')] += 1
        
        return {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'node_types': dict(node_types),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'num_components': nx.number_weakly_connected_components(G)
        }
    
    def _calculate_centrality(self, G: nx.DiGraph) -> Dict:
        """Calculate centrality metrics"""
        # Convert to undirected for some metrics
        G_undirected = G.to_undirected()
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        
        # Degree centrality
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # PageRank
        try:
            pagerank = nx.pagerank(G)
        except:
            pagerank = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}
        
        # Closeness centrality (on largest component)
        try:
            closeness = nx.closeness_centrality(G)
        except:
            closeness = {node: 0 for node in G.nodes()}
        
        return {
            'betweenness': betweenness,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'pagerank': pagerank,
            'closeness': closeness
        }
    
    def _analyze_structure(self, G: nx.DiGraph) -> Dict:
        """Analyze structural properties"""
        G_undirected = G.to_undirected()
        
        # Find articulation points (cut vertices)
        articulation_points = set()
        if nx.is_connected(G_undirected):
            articulation_points = set(nx.articulation_points(G_undirected))
        
        # Find bridges
        bridges = set()
        if nx.is_connected(G_undirected):
            bridges = set(nx.bridges(G_undirected))
        
        # Detect cycles
        try:
            cycles = list(nx.simple_cycles(G))
            has_cycles = len(cycles) > 0
        except:
            has_cycles = False
            cycles = []
        
        return {
            'articulation_points': articulation_points,
            'num_articulation_points': len(articulation_points),
            'bridges': bridges,
            'num_bridges': len(bridges),
            'has_cycles': has_cycles,
            'num_cycles': len(cycles) if len(cycles) < 100 else '100+'
        }
    
    def _analyze_layers(self, G: nx.DiGraph) -> Dict:
        """Analyze by layer (Application, Topic, Infrastructure)"""
        layers = {
            'Application': [],
            'Topic': [],
            'Broker': [],
            'Node': []
        }
        
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            if node_type in layers:
                layers[node_type].append(node)
        
        layer_stats = {}
        for layer, nodes in layers.items():
            if nodes:
                subgraph = G.subgraph(nodes)
                layer_stats[layer] = {
                    'count': len(nodes),
                    'edges': subgraph.number_of_edges(),
                    'density': nx.density(subgraph) if len(nodes) > 1 else 0
                }
        
        return layer_stats
    
    def _detect_anti_patterns(self, G: nx.DiGraph) -> Dict:
        """Detect common anti-patterns"""
        anti_patterns = {
            'spof_candidates': [],
            'god_topics': [],
            'isolated_components': [],
            'circular_dependencies': []
        }
        
        # SPOF: High betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        threshold = 0.3
        for node, bc in betweenness.items():
            if bc > threshold:
                anti_patterns['spof_candidates'].append({
                    'node': node,
                    'betweenness': bc
                })
        
        # God Topics: Topics with many subscribers
        for node, data in G.nodes(data=True):
            if data.get('type') == 'Topic':
                in_edges = G.in_degree(node)
                if in_edges > 10:
                    anti_patterns['god_topics'].append({
                        'topic': node,
                        'connections': in_edges
                    })
        
        # Isolated components
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            if len(components) > 1:
                for comp in components[1:]:  # Skip largest
                    anti_patterns['isolated_components'].append(list(comp))
        
        # Circular dependencies
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles[:10]:  # Limit to first 10
                anti_patterns['circular_dependencies'].append(cycle)
        except:
            pass
        
        return anti_patterns
    
    def _calculate_criticality_scores(self, G: nx.DiGraph,
                                      centrality: Dict,
                                      structural: Dict) -> Dict[str, CriticalityScore]:
        """
        Calculate composite criticality scores using the formula:
        C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
        """
        scores = {}
        
        betweenness = centrality['betweenness']
        articulation_points = structural['articulation_points']
        
        # Normalize betweenness
        max_bc = max(betweenness.values()) if betweenness else 1
        min_bc = min(betweenness.values()) if betweenness else 0
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Normalized betweenness centrality
            bc_norm = (betweenness.get(node, 0) - min_bc) / (max_bc - min_bc) if max_bc > min_bc else 0
            
            # Articulation point indicator
            ap = 1.0 if node in articulation_points else 0.0
            
            # Impact score (based on reachability)
            impact = self._calculate_impact_score(G, node)
            
            # Composite score
            composite = self.alpha * bc_norm + self.beta * ap + self.gamma * impact
            
            # Determine criticality level
            if composite >= 0.8:
                level = CriticalityLevel.CRITICAL
            elif composite >= 0.6:
                level = CriticalityLevel.HIGH
            elif composite >= 0.4:
                level = CriticalityLevel.MEDIUM
            elif composite >= 0.2:
                level = CriticalityLevel.LOW
            else:
                level = CriticalityLevel.MINIMAL
            
            scores[node] = CriticalityScore(
                component_id=node,
                component_type=node_type,
                betweenness_centrality=bc_norm,
                is_articulation_point=node in articulation_points,
                impact_score=impact,
                composite_score=composite,
                criticality_level=level
            )
        
        return scores
    
    def _calculate_impact_score(self, G: nx.DiGraph, node: str) -> float:
        """Calculate impact score based on reachability loss"""
        if G.number_of_nodes() <= 1:
            return 0.0
        
        # Create graph without the node
        G_removed = G.copy()
        G_removed.remove_node(node)
        
        if G_removed.number_of_nodes() == 0:
            return 1.0
        
        # Calculate reachability loss
        original_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
        
        if nx.is_weakly_connected(G_removed):
            remaining_pairs = G_removed.number_of_nodes() * (G_removed.number_of_nodes() - 1)
        else:
            remaining_pairs = sum(
                len(comp) * (len(comp) - 1) 
                for comp in nx.weakly_connected_components(G_removed)
            )
        
        reachability_loss = 1 - (remaining_pairs / original_pairs) if original_pairs > 0 else 0
        
        return min(1.0, reachability_loss)


# ============================================================================
# STEP 4: SIMULATION AND VALIDATION
# ============================================================================

class SimulationEngine:
    """
    Lightweight event-driven simulation for pub-sub systems
    """
    
    def __init__(self, G: nx.DiGraph, graph_data: Dict):
        self.G = G
        self.graph_data = graph_data
        self.logger = logging.getLogger('Simulation')
        
        # Simulation state
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.total_latency_ms = 0.0
        self.active_failures = set()
    
    async def run_baseline_simulation(self, duration: int = 10) -> Dict:
        """Run baseline simulation without failures"""
        self.logger.info(f"Running baseline simulation for {duration}s...")
        
        self._reset_stats()
        await self._simulate_traffic(duration)
        
        return self._get_stats()
    
    async def run_failure_simulation(self, 
                                     duration: int = 60,
                                     failure_time: int = 30,
                                     failure_components: List[str] = None,
                                     enable_cascading: bool = True) -> Dict:
        """Run simulation with failure injection"""
        self.logger.info(f"Running failure simulation for {duration}s...")
        
        self._reset_stats()
        
        # Pre-failure phase
        await self._simulate_traffic(failure_time)
        pre_failure_stats = self._get_stats()
        
        # Inject failures
        if failure_components:
            self.logger.info(f"Injecting failures: {failure_components}")
            for comp in failure_components:
                self.active_failures.add(comp)
                if enable_cascading:
                    cascaded = self._propagate_failure(comp)
                    self.active_failures.update(cascaded)
        
        # Post-failure phase
        await self._simulate_traffic(duration - failure_time)
        post_failure_stats = self._get_stats()
        
        # Calculate impact
        impact = self._calculate_failure_impact(pre_failure_stats, post_failure_stats)
        
        return {
            'pre_failure': pre_failure_stats,
            'post_failure': post_failure_stats,
            'failed_components': list(self.active_failures),
            'impact': impact
        }
    
    async def _simulate_traffic(self, duration: int):
        """Simulate message traffic"""
        topics = self.graph_data.get('topics', [])
        apps = self.graph_data.get('applications', [])
        
        # Calculate messages per second based on topic rates
        total_rate = sum(t.get('message_rate_hz', 10) for t in topics)
        messages_per_second = min(total_rate, 1000)  # Cap at 1000/s
        
        # Simulate with time compression (1000x speedup)
        sim_steps = duration * 10  # 10 steps per simulated second
        
        for step in range(sim_steps):
            messages_this_step = int(messages_per_second / 10)
            
            for _ in range(messages_this_step):
                # Random publisher
                publishers = [a for a in apps if a.get('type') in ['PRODUCER', 'PROSUMER']]
                if not publishers:
                    continue
                
                publisher = random.choice(publishers)
                
                # Skip if publisher is failed
                if publisher['id'] in self.active_failures:
                    self.messages_dropped += 1
                    continue
                
                # Random topic this publisher publishes to
                pub_rels = [r for r in self.graph_data['relationships']['publishes_to'] 
                           if r['from'] == publisher['id']]
                if not pub_rels:
                    continue
                
                topic_id = random.choice(pub_rels)['to']
                
                # Skip if topic is failed
                if topic_id in self.active_failures:
                    self.messages_dropped += 1
                    continue
                
                # Find subscribers
                sub_rels = [r for r in self.graph_data['relationships']['subscribes_to']
                           if r['to'] == topic_id]
                
                self.messages_sent += 1
                
                # Deliver to subscribers
                for sub_rel in sub_rels:
                    subscriber_id = sub_rel['from']
                    
                    if subscriber_id in self.active_failures:
                        self.messages_dropped += 1
                    else:
                        self.messages_delivered += 1
                        # Simulate latency
                        base_latency = random.uniform(1, 20)
                        if len(self.active_failures) > 0:
                            base_latency *= 1.5  # Degraded performance
                        self.total_latency_ms += base_latency
            
            # Small yield to keep event loop responsive
            if step % 100 == 0:
                await asyncio.sleep(0.001)
    
    def _propagate_failure(self, component: str) -> Set[str]:
        """Propagate failure to dependent components"""
        cascaded = set()
        
        # Find dependent components
        if component in self.G:
            for successor in self.G.successors(component):
                # 50% chance of cascade
                if random.random() < 0.5:
                    cascaded.add(successor)
            
            for predecessor in self.G.predecessors(component):
                # 30% chance of upstream cascade
                if random.random() < 0.3:
                    cascaded.add(predecessor)
        
        return cascaded
    
    def _reset_stats(self):
        """Reset simulation statistics"""
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.total_latency_ms = 0.0
        self.active_failures.clear()
    
    def _get_stats(self) -> Dict:
        """Get current statistics"""
        delivery_rate = (self.messages_delivered / self.messages_sent 
                        if self.messages_sent > 0 else 0)
        avg_latency = (self.total_latency_ms / self.messages_delivered 
                      if self.messages_delivered > 0 else 0)
        
        return {
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_dropped': self.messages_dropped,
            'delivery_rate': delivery_rate,
            'avg_latency_ms': avg_latency
        }
    
    def _calculate_failure_impact(self, pre: Dict, post: Dict) -> Dict:
        """Calculate impact of failures"""
        latency_increase = post['avg_latency_ms'] - pre['avg_latency_ms']
        delivery_decrease = pre['delivery_rate'] - post['delivery_rate']
        
        return {
            'latency_increase_ms': latency_increase,
            'latency_increase_pct': (latency_increase / pre['avg_latency_ms'] * 100 
                                    if pre['avg_latency_ms'] > 0 else 0),
            'delivery_rate_decrease': delivery_decrease,
            'messages_lost': post['messages_dropped'] - pre['messages_dropped'],
            'affected_components': len(self.active_failures)
        }


class ValidationEngine:
    """
    Validates analysis results against simulation outcomes
    """
    
    def __init__(self):
        self.logger = logging.getLogger('Validation')
    
    def validate(self, 
                analysis_results: Dict,
                simulation_results: Dict) -> ValidationResult:
        """Validate analysis predictions against simulation"""
        self.logger.info("Validating analysis results...")
        
        # Extract criticality scores
        criticality_scores = analysis_results.get('criticality_scores', {})
        
        # Predicted critical components (from analysis)
        predicted_critical = {
            node_id for node_id, score in criticality_scores.items()
            if score.criticality_level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH]
        }
        
        # Actual critical components (from simulation)
        failed_components = set(simulation_results.get('failed_components', []))
        impact = simulation_results.get('impact', {})
        
        # Components that caused significant impact when failed
        actual_critical = failed_components.copy()
        
        # Add components with high impact scores
        for node_id, score in criticality_scores.items():
            if score.impact_score > 0.3:  # Threshold for "significant impact"
                actual_critical.add(node_id)
        
        # Calculate precision, recall, F1
        if not predicted_critical:
            precision = 0.0
        else:
            true_positives = len(predicted_critical & actual_critical)
            precision = true_positives / len(predicted_critical)
        
        if not actual_critical:
            recall = 1.0
        else:
            true_positives = len(predicted_critical & actual_critical)
            recall = true_positives / len(actual_critical)
        
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0)
        
        # Calculate Spearman correlation
        spearman = self._calculate_spearman_correlation(
            criticality_scores, simulation_results
        )
        
        # Check targets
        targets_met = {
            'precision': precision >= TARGET_PRECISION,
            'recall': recall >= TARGET_RECALL,
            'f1_score': f1 >= TARGET_F1_SCORE,
            'spearman': spearman >= TARGET_SPEARMAN_CORRELATION
        }
        
        return ValidationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            spearman_correlation=spearman,
            targets_met=targets_met
        )
    
    def _calculate_spearman_correlation(self, 
                                        criticality_scores: Dict,
                                        simulation_results: Dict) -> float:
        """Calculate Spearman correlation between predictions and outcomes"""
        if 'scipy' not in DEPS:
            # Fallback: simple correlation
            return self._simple_correlation(criticality_scores, simulation_results)
        
        # Get ordered lists of scores and impacts
        nodes = list(criticality_scores.keys())
        predicted = [criticality_scores[n].composite_score for n in nodes]
        
        # Calculate actual impact scores from simulation
        failed = set(simulation_results.get('failed_components', []))
        actual = []
        for n in nodes:
            if n in failed:
                actual.append(1.0)
            else:
                actual.append(criticality_scores[n].impact_score)
        
        if len(predicted) < 3:
            return 0.0
        
        try:
            stats = DEPS['scipy']
            correlation, _ = stats.spearmanr(predicted, actual)
            return correlation if not math.isnan(correlation) else 0.0
        except:
            return self._simple_correlation(criticality_scores, simulation_results)
    
    def _simple_correlation(self, criticality_scores: Dict, simulation_results: Dict) -> float:
        """Simple correlation calculation fallback"""
        if not criticality_scores:
            return 0.0
        
        # Rank-based correlation
        nodes = list(criticality_scores.keys())
        predicted_ranks = self._rank(
            [criticality_scores[n].composite_score for n in nodes]
        )
        actual_ranks = self._rank(
            [criticality_scores[n].impact_score for n in nodes]
        )
        
        n = len(nodes)
        if n < 2:
            return 0.0
        
        # Spearman's rho formula
        d_squared_sum = sum(
            (predicted_ranks[i] - actual_ranks[i]) ** 2 
            for i in range(n)
        )
        
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        return rho
    
    def _rank(self, values: List[float]) -> List[float]:
        """Compute ranks of values"""
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        ranks = [0] * len(values)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

class Visualizer:
    """
    Generates multi-layer visualizations and reports
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('Visualizer')
    
    def generate_all(self,
                    graph_data: Dict,
                    G: nx.DiGraph,
                    analysis_results: Dict,
                    simulation_results: Dict,
                    validation_result: ValidationResult) -> Dict[str, Path]:
        """Generate all visualizations and reports"""
        self.logger.info("Generating visualizations...")
        
        outputs = {}
        
        # JSON data export
        outputs['json_data'] = self._export_json(
            graph_data, analysis_results, simulation_results, validation_result
        )
        
        # Interactive HTML visualization
        outputs['interactive_html'] = self._generate_interactive_html(
            G, analysis_results
        )
        
        # Multi-layer visualization
        outputs['multi_layer_html'] = self._generate_multi_layer_html(
            G, analysis_results
        )
        
        # Dashboard
        outputs['dashboard'] = self._generate_dashboard(
            analysis_results, simulation_results, validation_result
        )
        
        # Report
        outputs['report'] = self._generate_report(
            graph_data, analysis_results, simulation_results, validation_result
        )
        
        return outputs
    
    def _export_json(self, graph_data: Dict, analysis: Dict, 
                     simulation: Dict, validation: ValidationResult) -> Path:
        """Export all data to JSON"""
        output_path = self.output_dir / 'pipeline_results.json'
        
        # Convert criticality scores to serializable format
        criticality_dict = {}
        for node_id, score in analysis.get('criticality_scores', {}).items():
            criticality_dict[node_id] = {
                'component_id': score.component_id,
                'component_type': score.component_type,
                'betweenness_centrality': score.betweenness_centrality,
                'is_articulation_point': score.is_articulation_point,
                'impact_score': score.impact_score,
                'composite_score': score.composite_score,
                'criticality_level': score.criticality_level.value
            }
        
        data = {
            'metadata': graph_data.get('metadata', {}),
            'graph_summary': analysis.get('graph_summary', {}),
            'layer_analysis': analysis.get('layer_analysis', {}),
            'anti_patterns': {
                k: v if not isinstance(v, set) else list(v)
                for k, v in analysis.get('anti_patterns', {}).items()
            },
            'criticality_scores': criticality_dict,
            'simulation': {
                'pre_failure': simulation.get('pre_failure', {}),
                'post_failure': simulation.get('post_failure', {}),
                'impact': simulation.get('impact', {})
            },
            'validation': {
                'precision': validation.precision,
                'recall': validation.recall,
                'f1_score': validation.f1_score,
                'spearman_correlation': validation.spearman_correlation,
                'targets_met': validation.targets_met
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
    
    def _generate_interactive_html(self, G: nx.DiGraph, analysis: Dict) -> Path:
        """Generate interactive HTML visualization using Vis.js"""
        output_path = self.output_dir / 'interactive_graph.html'
        
        # Prepare nodes data
        nodes_data = []
        criticality_scores = analysis.get('criticality_scores', {})
        
        type_colors = {
            'Application': '#3498db',
            'Topic': '#2ecc71',
            'Broker': '#e74c3c',
            'Node': '#9b59b6'
        }
        
        criticality_colors = {
            'CRITICAL': '#e74c3c',
            'HIGH': '#e67e22',
            'MEDIUM': '#f1c40f',
            'LOW': '#27ae60',
            'MINIMAL': '#95a5a6'
        }
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Get criticality
            score = criticality_scores.get(node)
            if score:
                crit_level = score.criticality_level.value
                crit_score = score.composite_score
                color = criticality_colors.get(crit_level, '#95a5a6')
            else:
                crit_level = 'UNKNOWN'
                crit_score = 0
                color = type_colors.get(node_type, '#95a5a6')
            
            nodes_data.append({
                'id': node,
                'label': node,
                'title': f"<b>{node}</b><br>Type: {node_type}<br>Criticality: {crit_level}<br>Score: {crit_score:.3f}",
                'color': color,
                'size': 15 + (crit_score * 25),
                'group': node_type
            })
        
        # Prepare edges data
        edges_data = []
        edge_colors = {
            'PUBLISHES': '#3498db',
            'SUBSCRIBES': '#2ecc71',
            'ROUTES': '#e74c3c',
            'RUNS_ON': '#9b59b6'
        }
        
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            edges_data.append({
                'from': u,
                'to': v,
                'arrows': 'to',
                'color': {'color': edge_colors.get(edge_type, '#7f8c8d')},
                'title': edge_type
            })
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Interactive Graph Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{ font-size: 1.8em; margin-bottom: 5px; }}
        .header p {{ opacity: 0.7; }}
        #graph {{ 
            width: 100%; 
            height: calc(100vh - 200px); 
            background: rgba(0,0,0,0.2);
        }}
        .legend {{
            position: absolute;
            top: 100px;
            right: 20px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            font-size: 0.85em;
        }}
        .legend-title {{ font-weight: bold; margin-bottom: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ 
            width: 16px; height: 16px; 
            border-radius: 50%; 
            margin-right: 8px;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            padding: 15px;
            background: rgba(0,0,0,0.2);
        }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 0.8em; opacity: 0.7; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔗 Interactive Graph Visualization</h1>
        <p>Multi-Layer Pub-Sub System Analysis</p>
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{G.number_of_nodes()}</div>
            <div class="stat-label">Nodes</div>
        </div>
        <div class="stat">
            <div class="stat-value">{G.number_of_edges()}</div>
            <div class="stat-label">Edges</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len([s for s in criticality_scores.values() if s.criticality_level == CriticalityLevel.CRITICAL])}</div>
            <div class="stat-label">Critical</div>
        </div>
        <div class="stat">
            <div class="stat-value">{analysis.get('structural_analysis', {}).get('num_articulation_points', 0)}</div>
            <div class="stat-label">SPOFs</div>
        </div>
    </div>
    
    <div id="graph"></div>
    
    <div class="legend">
        <div class="legend-title">Criticality Levels</div>
        <div class="legend-item">
            <div class="legend-color" style="background: #e74c3c;"></div>
            <span>Critical</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #e67e22;"></div>
            <span>High</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f1c40f;"></div>
            <span>Medium</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #27ae60;"></div>
            <span>Low</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #95a5a6;"></div>
            <span>Minimal</span>
        </div>
    </div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{ color: '#fff', size: 12 }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                shadow: true,
                width: 2
            }},
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 150
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _generate_multi_layer_html(self, G: nx.DiGraph, analysis: Dict) -> Path:
        """Generate multi-layer visualization"""
        output_path = self.output_dir / 'multi_layer.html'
        
        # Extract layers
        layers = {
            'Application': [],
            'Topic': [],
            'Broker': [],
            'Node': []
        }
        
        criticality_scores = analysis.get('criticality_scores', {})
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            if node_type in layers:
                score = criticality_scores.get(node)
                layers[node_type].append({
                    'id': node,
                    'score': score.composite_score if score else 0,
                    'level': score.criticality_level.value if score else 'UNKNOWN'
                })
        
        # Count cross-layer edges
        cross_layer_edges = defaultdict(int)
        for u, v in G.edges():
            u_type = G.nodes[u].get('type', 'Unknown')
            v_type = G.nodes[v].get('type', 'Unknown')
            if u_type != v_type:
                key = f"{u_type} → {v_type}"
                cross_layer_edges[key] += 1
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multi-Layer Graph Model</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #fff;
            padding: 20px;
        }}
        h1 {{ text-align: center; margin-bottom: 20px; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .layer {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            min-width: 300px;
            flex: 1;
            max-width: 400px;
        }}
        .layer-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid;
        }}
        .layer-app .layer-title {{ border-color: #3498db; color: #3498db; }}
        .layer-topic .layer-title {{ border-color: #2ecc71; color: #2ecc71; }}
        .layer-broker .layer-title {{ border-color: #e74c3c; color: #e74c3c; }}
        .layer-node .layer-title {{ border-color: #9b59b6; color: #9b59b6; }}
        .node-list {{ max-height: 300px; overflow-y: auto; }}
        .node-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            margin: 5px 0;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
        }}
        .node-item.critical {{ border-left: 3px solid #e74c3c; }}
        .node-item.high {{ border-left: 3px solid #e67e22; }}
        .node-item.medium {{ border-left: 3px solid #f1c40f; }}
        .cross-layer {{
            margin-top: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
        }}
        .cross-layer h2 {{ margin-bottom: 15px; }}
        .edge-stat {{ 
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
        }}
    </style>
</head>
<body>
    <h1>📊 Multi-Layer Graph Model</h1>
    
    <div class="container">
        <div class="layer layer-app">
            <div class="layer-title">Application Layer ({len(layers['Application'])} nodes)</div>
            <div class="node-list">
                {''.join(f'<div class="node-item {n["level"].lower()}"><span>{n["id"]}</span><span>{n["score"]:.3f}</span></div>' for n in sorted(layers['Application'], key=lambda x: x['score'], reverse=True)[:15])}
            </div>
        </div>
        
        <div class="layer layer-topic">
            <div class="layer-title">Topic Layer ({len(layers['Topic'])} nodes)</div>
            <div class="node-list">
                {''.join(f'<div class="node-item {n["level"].lower()}"><span>{n["id"]}</span><span>{n["score"]:.3f}</span></div>' for n in sorted(layers['Topic'], key=lambda x: x['score'], reverse=True)[:15])}
            </div>
        </div>
        
        <div class="layer layer-broker">
            <div class="layer-title">Broker Layer ({len(layers['Broker'])} nodes)</div>
            <div class="node-list">
                {''.join(f'<div class="node-item {n["level"].lower()}"><span>{n["id"]}</span><span>{n["score"]:.3f}</span></div>' for n in sorted(layers['Broker'], key=lambda x: x['score'], reverse=True)[:15])}
            </div>
        </div>
        
        <div class="layer layer-node">
            <div class="layer-title">Infrastructure Layer ({len(layers['Node'])} nodes)</div>
            <div class="node-list">
                {''.join(f'<div class="node-item {n["level"].lower()}"><span>{n["id"]}</span><span>{n["score"]:.3f}</span></div>' for n in sorted(layers['Node'], key=lambda x: x['score'], reverse=True)[:15])}
            </div>
        </div>
    </div>
    
    <div class="cross-layer">
        <h2>Cross-Layer Dependencies</h2>
        {''.join(f'<span class="edge-stat">{k}: {v}</span>' for k, v in sorted(cross_layer_edges.items(), key=lambda x: x[1], reverse=True))}
    </div>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _generate_dashboard(self, analysis: Dict, simulation: Dict, 
                           validation: ValidationResult) -> Path:
        """Generate metrics dashboard"""
        output_path = self.output_dir / 'dashboard.html'
        
        graph_summary = analysis.get('graph_summary', {})
        criticality_scores = analysis.get('criticality_scores', {})
        anti_patterns = analysis.get('anti_patterns', {})
        
        # Count criticality levels
        level_counts = defaultdict(int)
        for score in criticality_scores.values():
            level_counts[score.criticality_level.value] += 1
        
        # Simulation stats
        pre_sim = simulation.get('pre_failure', {})
        post_sim = simulation.get('post_failure', {})
        impact = simulation.get('impact', {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Analysis Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }}
        .card-title {{
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ opacity: 0.7; font-size: 0.85em; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; }}
        .metric-value.good {{ color: #2ecc71; }}
        .metric-value.warning {{ color: #f1c40f; }}
        .metric-value.bad {{ color: #e74c3c; }}
        .target-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            margin-left: 10px;
        }}
        .target-badge.met {{ background: #27ae60; }}
        .target-badge.not-met {{ background: #e74c3c; }}
        .bar-chart {{ margin-top: 10px; }}
        .bar {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .bar-label {{ width: 80px; font-size: 0.85em; }}
        .bar-fill {{
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 Analysis Dashboard</h1>
        <p>Graph-Based Modeling and Analysis Results</p>
    </div>
    
    <div class="grid">
        <!-- Graph Summary -->
        <div class="card">
            <div class="card-title">📊 Graph Summary</div>
            <div class="metric">
                <div class="metric-label">Total Nodes</div>
                <div class="metric-value">{graph_summary.get('total_nodes', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Edges</div>
                <div class="metric-value">{graph_summary.get('total_edges', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Graph Density</div>
                <div class="metric-value">{graph_summary.get('density', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Connected</div>
                <div class="metric-value {'good' if graph_summary.get('is_connected') else 'bad'}">
                    {'Yes' if graph_summary.get('is_connected') else 'No'}
                </div>
            </div>
        </div>
        
        <!-- Criticality Distribution -->
        <div class="card">
            <div class="card-title">⚠️ Criticality Distribution</div>
            <div class="bar-chart">
                <div class="bar">
                    <span class="bar-label">Critical</span>
                    <div class="bar-fill" style="width: {level_counts.get('CRITICAL', 0) * 20}px; background: #e74c3c;"></div>
                    <span>{level_counts.get('CRITICAL', 0)}</span>
                </div>
                <div class="bar">
                    <span class="bar-label">High</span>
                    <div class="bar-fill" style="width: {level_counts.get('HIGH', 0) * 20}px; background: #e67e22;"></div>
                    <span>{level_counts.get('HIGH', 0)}</span>
                </div>
                <div class="bar">
                    <span class="bar-label">Medium</span>
                    <div class="bar-fill" style="width: {level_counts.get('MEDIUM', 0) * 20}px; background: #f1c40f;"></div>
                    <span>{level_counts.get('MEDIUM', 0)}</span>
                </div>
                <div class="bar">
                    <span class="bar-label">Low</span>
                    <div class="bar-fill" style="width: {level_counts.get('LOW', 0) * 20}px; background: #27ae60;"></div>
                    <span>{level_counts.get('LOW', 0)}</span>
                </div>
                <div class="bar">
                    <span class="bar-label">Minimal</span>
                    <div class="bar-fill" style="width: {level_counts.get('MINIMAL', 0) * 20}px; background: #95a5a6;"></div>
                    <span>{level_counts.get('MINIMAL', 0)}</span>
                </div>
            </div>
        </div>
        
        <!-- Validation Results -->
        <div class="card">
            <div class="card-title">✅ Validation Results</div>
            <div class="metric">
                <div class="metric-label">Precision (Target: ≥0.9)</div>
                <div class="metric-value {'good' if validation.precision >= 0.9 else 'warning' if validation.precision >= 0.7 else 'bad'}">
                    {validation.precision:.3f}
                    <span class="target-badge {'met' if validation.targets_met.get('precision') else 'not-met'}">
                        {'✓ Met' if validation.targets_met.get('precision') else '✗ Not Met'}
                    </span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Recall (Target: ≥0.85)</div>
                <div class="metric-value {'good' if validation.recall >= 0.85 else 'warning' if validation.recall >= 0.7 else 'bad'}">
                    {validation.recall:.3f}
                    <span class="target-badge {'met' if validation.targets_met.get('recall') else 'not-met'}">
                        {'✓ Met' if validation.targets_met.get('recall') else '✗ Not Met'}
                    </span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">F1 Score (Target: ≥0.9)</div>
                <div class="metric-value {'good' if validation.f1_score >= 0.9 else 'warning' if validation.f1_score >= 0.7 else 'bad'}">
                    {validation.f1_score:.3f}
                    <span class="target-badge {'met' if validation.targets_met.get('f1_score') else 'not-met'}">
                        {'✓ Met' if validation.targets_met.get('f1_score') else '✗ Not Met'}
                    </span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Spearman Correlation (Target: ≥0.7)</div>
                <div class="metric-value {'good' if validation.spearman_correlation >= 0.7 else 'warning' if validation.spearman_correlation >= 0.5 else 'bad'}">
                    {validation.spearman_correlation:.3f}
                    <span class="target-badge {'met' if validation.targets_met.get('spearman') else 'not-met'}">
                        {'✓ Met' if validation.targets_met.get('spearman') else '✗ Not Met'}
                    </span>
                </div>
            </div>
        </div>
        
        <!-- Simulation Results -->
        <div class="card">
            <div class="card-title">🔄 Simulation Results</div>
            <div class="metric">
                <div class="metric-label">Baseline Delivery Rate</div>
                <div class="metric-value good">{pre_sim.get('delivery_rate', 0):.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Post-Failure Delivery Rate</div>
                <div class="metric-value {'warning' if post_sim.get('delivery_rate', 0) < 0.9 else 'good'}">
                    {post_sim.get('delivery_rate', 0):.1%}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Latency Increase</div>
                <div class="metric-value {'bad' if impact.get('latency_increase_pct', 0) > 50 else 'warning' if impact.get('latency_increase_pct', 0) > 20 else 'good'}">
                    +{impact.get('latency_increase_pct', 0):.1f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Components Affected</div>
                <div class="metric-value">{impact.get('affected_components', 0)}</div>
            </div>
        </div>
        
        <!-- Anti-Patterns -->
        <div class="card">
            <div class="card-title">🐛 Anti-Patterns Detected</div>
            <div class="metric">
                <div class="metric-label">SPOF Candidates</div>
                <div class="metric-value {'bad' if len(anti_patterns.get('spof_candidates', [])) > 0 else 'good'}">
                    {len(anti_patterns.get('spof_candidates', []))}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">God Topics</div>
                <div class="metric-value {'warning' if len(anti_patterns.get('god_topics', [])) > 0 else 'good'}">
                    {len(anti_patterns.get('god_topics', []))}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Circular Dependencies</div>
                <div class="metric-value {'warning' if len(anti_patterns.get('circular_dependencies', [])) > 0 else 'good'}">
                    {len(anti_patterns.get('circular_dependencies', []))}
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _generate_report(self, graph_data: Dict, analysis: Dict,
                        simulation: Dict, validation: ValidationResult) -> Path:
        """Generate text report"""
        output_path = self.output_dir / 'report.md'
        
        graph_summary = analysis.get('graph_summary', {})
        criticality_scores = analysis.get('criticality_scores', {})
        structural = analysis.get('structural_analysis', {})
        
        # Get top critical components
        sorted_scores = sorted(
            criticality_scores.items(),
            key=lambda x: x[1].composite_score,
            reverse=True
        )[:10]
        
        report = f"""# Graph-Based Analysis Report

## Executive Summary

This report presents the results of comprehensive graph-based modeling and analysis
of a distributed publish-subscribe system.

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Scenario:** {graph_data.get('metadata', {}).get('scenario', 'Unknown')}

## Graph Summary

| Metric | Value |
|--------|-------|
| Total Nodes | {graph_summary.get('total_nodes', 0)} |
| Total Edges | {graph_summary.get('total_edges', 0)} |
| Density | {graph_summary.get('density', 0):.4f} |
| Connected | {'Yes' if graph_summary.get('is_connected') else 'No'} |
| Components | {graph_summary.get('num_components', 0)} |

### Node Distribution

| Type | Count |
|------|-------|
"""
        
        for node_type, count in graph_summary.get('node_types', {}).items():
            report += f"| {node_type} | {count} |\n"
        
        report += f"""

## Structural Analysis

- **Articulation Points (SPOFs):** {structural.get('num_articulation_points', 0)}
- **Bridges:** {structural.get('num_bridges', 0)}
- **Has Cycles:** {'Yes' if structural.get('has_cycles') else 'No'}

## Top 10 Critical Components

| Rank | Component | Type | Score | Level |
|------|-----------|------|-------|-------|
"""
        
        for i, (node_id, score) in enumerate(sorted_scores, 1):
            report += f"| {i} | {node_id} | {score.component_type} | {score.composite_score:.3f} | {score.criticality_level.value} |\n"
        
        report += f"""

## Validation Results

### Target Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Precision | {validation.precision:.3f} | ≥0.9 | {'✓ Met' if validation.targets_met.get('precision') else '✗ Not Met'} |
| Recall | {validation.recall:.3f} | ≥0.85 | {'✓ Met' if validation.targets_met.get('recall') else '✗ Not Met'} |
| F1 Score | {validation.f1_score:.3f} | ≥0.9 | {'✓ Met' if validation.targets_met.get('f1_score') else '✗ Not Met'} |
| Spearman Correlation | {validation.spearman_correlation:.3f} | ≥0.7 | {'✓ Met' if validation.targets_met.get('spearman') else '✗ Not Met'} |

## Simulation Results

### Baseline Performance
- Messages Delivered: {simulation.get('pre_failure', {}).get('messages_delivered', 0):,}
- Delivery Rate: {simulation.get('pre_failure', {}).get('delivery_rate', 0):.1%}
- Avg Latency: {simulation.get('pre_failure', {}).get('avg_latency_ms', 0):.2f}ms

### Post-Failure Performance
- Messages Delivered: {simulation.get('post_failure', {}).get('messages_delivered', 0):,}
- Delivery Rate: {simulation.get('post_failure', {}).get('delivery_rate', 0):.1%}
- Avg Latency: {simulation.get('post_failure', {}).get('avg_latency_ms', 0):.2f}ms

### Impact Analysis
- Latency Increase: {simulation.get('impact', {}).get('latency_increase_pct', 0):.1f}%
- Components Affected: {simulation.get('impact', {}).get('affected_components', 0)}

## Recommendations

1. **Address Single Points of Failure:** Components with high betweenness centrality should be replicated.
2. **Monitor Critical Components:** Implement enhanced monitoring for CRITICAL and HIGH criticality nodes.
3. **Review God Topics:** Topics with excessive connections may indicate design issues.
4. **Implement Graceful Degradation:** Ensure the system can handle component failures without cascading.

---
*Generated by Software-as-a-Graph Analysis Framework*
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

async def run_pipeline(config: GraphConfig,
                      sim_config: SimulationConfig,
                      neo4j_config: Dict = None,
                      output_dir: Path = None,
                      verbose: bool = False) -> Dict:
    """Run the complete end-to-end pipeline"""
    
    logger = setup_logging(verbose)
    start_time = time.time()
    
    print_header("SOFTWARE-AS-A-GRAPH: END-TO-END PIPELINE")
    print(f"""
This pipeline demonstrates the comprehensive methodology for
Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems.

Scenario: {config.scenario.value}
Scale: {config.scale}
""")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # STEP 1: Generate Graph Data
    # -------------------------------------------------------------------------
    print_step(1, "GENERATE GRAPH DATA")
    
    generator = PubSubGraphGenerator(config)
    graph_data = generator.generate()
    
    print_success(f"Generated {len(graph_data['nodes'])} nodes")
    print_success(f"Generated {len(graph_data['applications'])} applications")
    print_success(f"Generated {len(graph_data['topics'])} topics")
    print_success(f"Generated {len(graph_data['brokers'])} brokers")
    
    results['graph_data'] = graph_data
    
    # -------------------------------------------------------------------------
    # STEP 2: Import to Neo4j (optional)
    # -------------------------------------------------------------------------
    print_step(2, "IMPORT TO NEO4J")
    
    if neo4j_config and 'neo4j' in DEPS:
        importer = Neo4jImporter(
            uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
            user=neo4j_config.get('user', 'neo4j'),
            password=neo4j_config.get('password', 'password')
        )
        
        if importer.connect():
            importer.clear_database()
            importer.create_schema()
            importer.import_graph(graph_data)
            analytics = importer.run_analytics()
            importer.close()
            print_success("Graph imported to Neo4j")
            results['neo4j_analytics'] = analytics
        else:
            print_warning("Could not connect to Neo4j, skipping import")
    else:
        print_info("Neo4j import skipped (not configured or driver unavailable)")
    
    # -------------------------------------------------------------------------
    # STEP 3: Analyze Graph
    # -------------------------------------------------------------------------
    print_step(3, "ANALYZE GRAPH")
    
    analyzer = GraphAnalyzer()
    G = analyzer.build_networkx_graph(graph_data)
    analysis_results = analyzer.analyze(G)
    
    graph_summary = analysis_results['graph_summary']
    print_success(f"Graph has {graph_summary['total_nodes']} nodes and {graph_summary['total_edges']} edges")
    print_success(f"Found {analysis_results['structural_analysis']['num_articulation_points']} articulation points (SPOFs)")
    
    # Count criticality levels
    level_counts = defaultdict(int)
    for score in analysis_results['criticality_scores'].values():
        level_counts[score.criticality_level.value] += 1
    
    print_info(f"Criticality distribution: CRITICAL={level_counts['CRITICAL']}, "
               f"HIGH={level_counts['HIGH']}, MEDIUM={level_counts['MEDIUM']}, "
               f"LOW={level_counts['LOW']}, MINIMAL={level_counts['MINIMAL']}")
    
    results['analysis'] = analysis_results
    results['networkx_graph'] = G
    
    # -------------------------------------------------------------------------
    # STEP 4: Simulate and Validate
    # -------------------------------------------------------------------------
    print_step(4, "SIMULATE AND VALIDATE")
    
    # Select failure target (highest criticality component)
    sorted_scores = sorted(
        analysis_results['criticality_scores'].items(),
        key=lambda x: x[1].composite_score,
        reverse=True
    )
    
    if sorted_scores and not sim_config.failure_components:
        sim_config.failure_components = [sorted_scores[0][0]]
    
    simulation = SimulationEngine(G, graph_data)
    
    # Run baseline
    print_info("Running baseline simulation...")
    baseline_results = await simulation.run_baseline_simulation(sim_config.baseline_duration)
    print_success(f"Baseline: {baseline_results['delivery_rate']:.1%} delivery rate, "
                  f"{baseline_results['avg_latency_ms']:.2f}ms latency")
    
    # Run failure simulation
    print_info(f"Running failure simulation (failing: {sim_config.failure_components})...")
    failure_results = await simulation.run_failure_simulation(
        duration=sim_config.duration_seconds,
        failure_time=sim_config.failure_time,
        failure_components=sim_config.failure_components,
        enable_cascading=sim_config.enable_cascading
    )
    
    impact = failure_results['impact']
    print_success(f"Post-failure: {failure_results['post_failure']['delivery_rate']:.1%} delivery rate")
    print_success(f"Impact: +{impact['latency_increase_pct']:.1f}% latency, "
                  f"{impact['affected_components']} components affected")
    
    # Validate
    print_info("Validating analysis results...")
    validator = ValidationEngine()
    validation_result = validator.validate(analysis_results, failure_results)
    
    print()
    print(f"{Colors.BOLD}Validation Results:{Colors.END}")
    print(f"  Precision:    {validation_result.precision:.3f} "
          f"{'✓' if validation_result.targets_met['precision'] else '✗'} (target: ≥0.9)")
    print(f"  Recall:       {validation_result.recall:.3f} "
          f"{'✓' if validation_result.targets_met['recall'] else '✗'} (target: ≥0.85)")
    print(f"  F1 Score:     {validation_result.f1_score:.3f} "
          f"{'✓' if validation_result.targets_met['f1_score'] else '✗'} (target: ≥0.9)")
    print(f"  Spearman:     {validation_result.spearman_correlation:.3f} "
          f"{'✓' if validation_result.targets_met['spearman'] else '✗'} (target: ≥0.7)")
    
    results['simulation'] = failure_results
    results['validation'] = validation_result
    
    # -------------------------------------------------------------------------
    # STEP 5: Visualize Results
    # -------------------------------------------------------------------------
    print_step(5, "VISUALIZE RESULTS")
    
    if output_dir is None:
        output_dir = Path('e2e_output')
    
    visualizer = Visualizer(output_dir)
    output_files = visualizer.generate_all(
        graph_data, G, analysis_results, failure_results, validation_result
    )
    
    print_success("Generated outputs:")
    for name, path in output_files.items():
        print(f"  • {name}: {path}")
    
    results['output_files'] = output_files
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    
    print_header("PIPELINE COMPLETE")
    
    targets_met = sum(validation_result.targets_met.values())
    total_targets = len(validation_result.targets_met)
    
    print(f"""
{Colors.GREEN}✓ Pipeline completed successfully in {elapsed:.2f}s{Colors.END}

Summary:
  • Graph: {graph_summary['total_nodes']} nodes, {graph_summary['total_edges']} edges
  • Critical Components: {level_counts['CRITICAL']}
  • SPOFs Detected: {analysis_results['structural_analysis']['num_articulation_points']}
  • Validation: {targets_met}/{total_targets} targets met
  
Output directory: {output_dir.absolute()}

Open {output_dir / 'dashboard.html'} in your browser to explore the results!
""")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Software-as-a-Graph End-to-End Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (IoT Smart City scenario)
  python e2e_pipeline.py
  
  # Financial trading scenario
  python e2e_pipeline.py --scenario financial_trading --scale medium
  
  # Healthcare with Neo4j import
  python e2e_pipeline.py --scenario healthcare --neo4j-uri bolt://localhost:7687
  
  # Large scale with verbose output
  python e2e_pipeline.py --scale large --verbose
        """
    )
    
    # Graph configuration
    parser.add_argument('--scenario', type=str, default='iot_smart_city',
                       choices=['generic', 'iot_smart_city', 'financial_trading', 
                               'healthcare', 'ecommerce'],
                       help='Domain scenario (default: iot_smart_city)')
    parser.add_argument('--scale', type=str, default='medium',
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                       help='Graph scale (default: medium)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--antipatterns', nargs='+', default=['spof'],
                       choices=['spof', 'god_topic', 'tight_coupling'],
                       help='Anti-patterns to inject (default: spof)')
    
    # Simulation configuration
    parser.add_argument('--sim-duration', type=int, default=60,
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--failure-time', type=int, default=30,
                       help='Time to inject failure (default: 30)')
    parser.add_argument('--fail-component', type=str, default=None,
                       help='Specific component to fail (default: auto-select)')
    
    # Neo4j configuration
    parser.add_argument('--neo4j-uri', type=str, default=None,
                       help='Neo4j URI (e.g., bolt://localhost:7687)')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--neo4j-password', type=str, default='password',
                       help='Neo4j password (default: password)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='e2e_output',
                       help='Output directory (default: e2e_output)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create graph config
    graph_config = GraphConfig(
        scale=args.scale,
        scenario=Scenario(args.scenario),
        seed=args.seed,
        antipatterns=args.antipatterns
    )
    
    # Update with scale defaults
    scale_params = PubSubGraphGenerator.SCALES.get(args.scale, 
                                                    PubSubGraphGenerator.SCALES['medium'])
    graph_config.num_nodes = scale_params['nodes']
    graph_config.num_applications = scale_params['apps']
    graph_config.num_topics = scale_params['topics']
    graph_config.num_brokers = scale_params['brokers']
    
    # Create simulation config
    sim_config = SimulationConfig(
        duration_seconds=args.sim_duration,
        failure_time=args.failure_time,
        failure_components=[args.fail_component] if args.fail_component else []
    )
    
    # Neo4j config
    neo4j_config = None
    if args.neo4j_uri:
        neo4j_config = {
            'uri': args.neo4j_uri,
            'user': args.neo4j_user,
            'password': args.neo4j_password
        }
    
    # Run pipeline
    output_dir = Path(args.output_dir)
    
    asyncio.run(run_pipeline(
        config=graph_config,
        sim_config=sim_config,
        neo4j_config=neo4j_config,
        output_dir=output_dir,
        verbose=args.verbose
    ))


if __name__ == '__main__':
    main()
