"""
Graph Generator
Generates realistic DDS pub-sub system graphs with various scales and scenarios.
Supports anti-pattern injection for testing purposes.
"""

import random
from typing import Dict, List
from dataclasses import asdict, dataclass
from datetime import datetime
import logging

@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    scale: str  # small, medium, large, xlarge
    scenario: str  # generic, iot, financial, ecommerce, analytics
    num_nodes: int
    num_applications: int
    num_topics: int
    num_brokers: int
    edge_density: float  # 0.0 - 1.0
    high_availability: bool
    antipatterns: List[str]
    seed: int

class GraphGenerator:
    """
    Generates realistic DDS pub-sub system graphs
    
    Supports multiple scales and scenarios with realistic patterns
    """
    
    # Predefined scales
    SCALES = {
        'tiny': {'nodes': 3, 'apps': 5, 'topics': 3, 'brokers': 1},
        'small': {'nodes': 5, 'apps': 10, 'topics': 8, 'brokers': 2},
        'medium': {'nodes': 15, 'apps': 50, 'topics': 25, 'brokers': 3},
        'large': {'nodes': 50, 'apps': 200, 'topics': 100, 'brokers': 8},
        'xlarge': {'nodes': 100, 'apps': 500, 'topics': 250, 'brokers': 15},
        'extreme': {'nodes': 200, 'apps': 1000, 'topics': 500, 'brokers': 20}
    }
    
    # Application types by domain
    APP_TYPES = {
        'generic': [
            'ServiceA', 'ServiceB', 'ServiceC', 'DataProcessor',
            'EventHandler', 'MessageRouter', 'Aggregator', 'Monitor'
        ],
        'iot': [
            'SensorCollector', 'DeviceManager', 'TelemetryAggregator',
            'CommandDispatcher', 'StatusMonitor', 'AlertProcessor',
            'DataForwarder', 'FirmwareUpdater', 'DiagnosticsEngine'
        ],
        'financial': [
            'OrderProcessor', 'MarketDataFeed', 'RiskEngine',
            'TradeExecutor', 'PositionTracker', 'ComplianceMonitor',
            'PricingEngine', 'AuditLogger', 'ReportGenerator'
        ],
        'ecommerce': [
            'OrderService', 'InventoryManager', 'PaymentProcessor',
            'ShippingCalculator', 'RecommendationEngine', 'CartService',
            'ReviewAggregator', 'NotificationService', 'FraudDetector'
        ],
        'analytics': [
            'DataCollector', 'StreamProcessor', 'MetricsAggregator',
            'LogAnalyzer', 'ReportGenerator', 'MLPipeline',
            'FeatureExtractor', 'ModelInferencer', 'ResultsPublisher'
        ]
    }
    
    # Topic patterns by domain
    TOPIC_PATTERNS = {
        'generic': [
            'events', 'commands', 'status', 'data', 'metrics',
            'alerts', 'config', 'health'
        ],
        'iot': [
            'telemetry', 'device/status', 'sensor/data', 'command',
            'alert', 'diagnostic', 'firmware', 'config'
        ],
        'financial': [
            'market/data', 'order', 'trade', 'position', 'risk',
            'price', 'quote', 'settlement', 'audit'
        ],
        'ecommerce': [
            'order', 'inventory', 'payment', 'shipping', 'user',
            'product', 'cart', 'review', 'notification'
        ],
        'analytics': [
            'raw/data', 'processed', 'aggregated', 'metrics',
            'model/input', 'model/output', 'report', 'alert'
        ]
    }
    
    def __init__(self, config: GraphConfig):
        """Initialize graph generator with configuration"""
        self.config = config
        random.seed(config.seed)
        self.logger = logging.getLogger(__name__)
        
        # Get app types and topic patterns for scenario
        self.app_types = self.APP_TYPES.get(
            config.scenario,
            self.APP_TYPES['generic']
        )
        self.topic_patterns = self.TOPIC_PATTERNS.get(
            config.scenario,
            self.TOPIC_PATTERNS['generic']
        )
    
    def generate(self) -> Dict:
        """
        Generate complete graph
        
        Returns:
            Dictionary with graph data in standard format
        """
        self.logger.info(f"Generating {self.config.scale} scale "
                        f"{self.config.scenario} system...")
        
        graph = {
            'metadata': self._generate_metadata(),
            'nodes': self._generate_nodes(),
            'applications': self._generate_applications(),
            'topics': self._generate_topics(),
            'brokers': self._generate_brokers(),
            'relationships': {
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'runs_on': []
            }
        }
        
        # Generate relationships
        self._generate_runs_on(graph)
        self._generate_pub_sub(graph)
        self._generate_routes(graph)
        
        # Apply anti-patterns if requested
        if self.config.antipatterns:
            self._apply_antipatterns(graph)
        
        self.logger.info(f"Generated: {len(graph['nodes'])} nodes, "
                        f"{len(graph['applications'])} apps, "
                        f"{len(graph['topics'])} topics, "
                        f"{len(graph['brokers'])} brokers")
        
        return graph
    
    def _generate_metadata(self) -> Dict:
        """Generate metadata section"""
        return {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'scenario': self.config.scenario,
            'scale': self.config.scale,
            'config': asdict(self.config),
            'description': f"{self.config.scale.title()} scale {self.config.scenario} system"
        }
    
    def _generate_nodes(self) -> List[Dict]:
        """Generate infrastructure nodes"""
        nodes = []
        
        # Determine node specs based on scale
        if self.config.num_nodes <= 10:
            cpu_range = (4, 16)
            mem_range = (8, 32)
            bw_range = (1000, 5000)
        elif self.config.num_nodes <= 50:
            cpu_range = (8, 32)
            mem_range = (16, 64)
            bw_range = (5000, 10000)
        else:
            cpu_range = (16, 64)
            mem_range = (32, 128)
            bw_range = (10000, 40000)
        
        # Generate zones for HA
        num_zones = min(3, max(1, self.config.num_nodes // 10)) if self.config.high_availability else 1
        num_regions = min(3, max(1, self.config.num_nodes // 30)) if self.config.high_availability else 1
        
        for i in range(1, self.config.num_nodes + 1):
            node = {
                'id': f'N{i}',
                'name': f'Node{i}',
                'cpu_capacity': float(random.randint(*cpu_range)),
                'memory_gb': float(random.randint(*mem_range)),
                'network_bandwidth_mbps': float(random.randint(*bw_range)),
                'zone': f'zone-{(i-1) % num_zones + 1}',
                'region': f'region-{(i-1) // (self.config.num_nodes // num_regions + 1) + 1}'
            }
            nodes.append(node)
        
        return nodes
    
    def _generate_applications(self) -> List[Dict]:
        """Generate applications"""
        applications = []
        
        for i in range(1, self.config.num_applications + 1):
            # Determine application type
            base_name = random.choice(self.app_types)
            
            # Determine producer/consumer/prosumer
            app_type_prob = random.random()
            if app_type_prob < 0.3:
                app_type = 'PRODUCER'
            elif app_type_prob < 0.6:
                app_type = 'CONSUMER'
            else:
                app_type = 'PROSUMER'
            
            # Determine criticality
            criticality = random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
            
            # Replicas for HA
            replicas = 1
            if self.config.high_availability and criticality in ['HIGH', 'CRITICAL']:
                replicas = random.choice([2, 3, 5])
            
            app = {
                'id': f'A{i}',
                'name': f'{base_name}{i}',
                'type': app_type,
                'criticality': criticality,
                'replicas': replicas,
                'cpu_request': random.uniform(0.5, 4.0),
                'memory_request_mb': random.uniform(256, 2048)
            }
            applications.append(app)
        
        return applications
    
    def _generate_topics(self) -> List[Dict]:
        """Generate topics with QoS policies"""
        topics = []
        
        for i in range(1, self.config.num_topics + 1):
            # Generate topic name
            pattern = random.choice(self.topic_patterns)
            topic_name = f'{pattern}/{i}'
            
            # Generate QoS based on scenario
            qos = self._generate_qos(pattern)
            
            topic = {
                'id': f'T{i}',
                'name': topic_name,
                'qos': qos,
                'message_size_bytes': random.choice([128, 256, 512, 1024, 2048, 4096]),
                'expected_rate_hz': random.choice([1, 5, 10, 20, 50, 100])
            }
            topics.append(topic)
        
        return topics
    
    def _generate_qos(self, pattern: str) -> Dict:
        """Generate realistic QoS policies"""
        
        # Default QoS
        qos = {
            'durability': 'VOLATILE',
            'reliability': 'BEST_EFFORT',
            'history_depth': 1,
            'deadline_ms': None,
            'lifespan_ms': None,
            'transport_priority': 'MEDIUM'
        }
        
        # Scenario-specific QoS
        if self.config.scenario == 'financial':
            qos.update({
                'durability': random.choice(['TRANSIENT_LOCAL', 'PERSISTENT']),
                'reliability': 'RELIABLE',
                'history_depth': random.choice([10, 20, 50]),
                'deadline_ms': random.choice([10, 50, 100]),
                'transport_priority': random.choice(['HIGH', 'URGENT'])
            })
        
        elif self.config.scenario == 'iot':
            if 'telemetry' in pattern or 'sensor' in pattern:
                qos.update({
                    'durability': 'VOLATILE',
                    'reliability': 'BEST_EFFORT',
                    'history_depth': 5,
                    'deadline_ms': 100
                })
            elif 'command' in pattern:
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 10,
                    'deadline_ms': 50,
                    'transport_priority': 'HIGH'
                })
        
        elif self.config.scenario == 'analytics':
            if 'raw' in pattern:
                qos.update({
                    'reliability': 'BEST_EFFORT',
                    'history_depth': 1
                })
            elif 'processed' in pattern or 'aggregated' in pattern:
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 20
                })
        
        return qos
    
    def _generate_brokers(self) -> List[Dict]:
        """Generate message brokers"""
        brokers = []
        
        for i in range(1, self.config.num_brokers + 1):
            broker = {
                'id': f'B{i}',
                'name': f'Broker{i}',
                'max_topics': random.randint(50, 200),
                'max_connections': random.randint(100, 1000)
            }
            brokers.append(broker)
        
        return brokers
    
    def _generate_runs_on(self, graph: Dict):
        """Generate application-to-node deployment"""
        apps = graph['applications']
        nodes = graph['nodes']
        
        # Distribute applications across nodes
        for app in apps:
            # For replicated apps, distribute across zones
            if app['replicas'] > 1:
                # Get nodes from different zones
                zones = {}
                for node in nodes:
                    zone = node['zone']
                    if zone not in zones:
                        zones[zone] = []
                    zones[zone].append(node['id'])
                
                # Distribute replicas across zones
                selected_nodes = []
                for zone, zone_nodes in list(zones.items())[:app['replicas']]:
                    if zone_nodes:
                        selected_nodes.append(random.choice(zone_nodes))
                
                for node_id in selected_nodes:
                    graph['relationships']['runs_on'].append({
                        'from': app['id'],
                        'to': node_id
                    })
            else:
                # Single replica - random placement
                node = random.choice(nodes)
                graph['relationships']['runs_on'].append({
                    'from': app['id'],
                    'to': node['id']
                })
    
    def _generate_pub_sub(self, graph: Dict):
        """Generate publish-subscribe relationships"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Calculate edge density
        max_pub = max(1, int(len(topics) * 0.3))  # Max 30% of topics
        max_sub = max(1, int(len(topics) * self.config.edge_density))
        
        for app in apps:
            app_type = app['type']
            
            # Publishers
            if app_type in ['PRODUCER', 'PROSUMER']:
                num_topics = random.randint(1, min(max_pub, len(topics)))
                pub_topics = random.sample(topics, num_topics)
                
                for topic in pub_topics:
                    graph['relationships']['publishes_to'].append({
                        'from': app['id'],
                        'to': topic['id'],
                        'period_ms': int(1000 / topic['expected_rate_hz']),
                        'msg_size': topic['message_size_bytes']
                    })
            
            # Subscribers
            if app_type in ['CONSUMER', 'PROSUMER']:
                num_topics = random.randint(1, min(max_sub, len(topics)))
                sub_topics = random.sample(topics, num_topics)
                
                for topic in sub_topics:
                    graph['relationships']['subscribes_to'].append({
                        'from': app['id'],
                        'to': topic['id']
                    })
    
    def _generate_routes(self, graph: Dict):
        """Generate broker-to-topic routing"""
        brokers = graph['brokers']
        topics = graph['topics']
        
        # Distribute topics across brokers
        topics_per_broker = len(topics) // len(brokers)
        
        for i, topic in enumerate(topics):
            broker_idx = i % len(brokers)
            broker = brokers[broker_idx]
            
            graph['relationships']['routes'].append({
                'from': broker['id'],
                'to': topic['id']
            })
    
    def _apply_antipatterns(self, graph: Dict):
        """Apply anti-patterns to the graph"""
        
        for antipattern in self.config.antipatterns:
            self.logger.info(f"Applying anti-pattern: {antipattern}")
            
            if antipattern == 'spof':
                self._apply_spof(graph)
            elif antipattern == 'broker_overload':
                self._apply_broker_overload(graph)
            elif antipattern == 'god_object':
                self._apply_god_object(graph)
            elif antipattern == 'single_broker':
                self._apply_single_broker(graph)
            elif antipattern == 'tight_coupling':
                self._apply_tight_coupling(graph)
    
    def _apply_spof(self, graph: Dict):
        """Apply Single Point of Failure anti-pattern"""
        # Force one critical app with no replicas
        critical_apps = [a for a in graph['applications'] 
                        if a['criticality'] == 'CRITICAL']
        if critical_apps:
            app = random.choice(critical_apps)
            app['replicas'] = 1
            
            # Make many apps depend on it
            for other_app in graph['applications'][:10]:
                if other_app['id'] != app['id']:
                    # Ensure dependency through topics
                    pub_topics = [r['to'] for r in graph['relationships']['publishes_to'] 
                                 if r['from'] == app['id']]
                    if pub_topics:
                        topic = random.choice(pub_topics)
                        if not any(r['from'] == other_app['id'] and r['to'] == topic 
                                  for r in graph['relationships']['subscribes_to']):
                            graph['relationships']['subscribes_to'].append({
                                'from': other_app['id'],
                                'to': topic
                            })
    
    def _apply_broker_overload(self, graph: Dict):
        """Apply broker overload anti-pattern"""
        # Route all topics through first broker
        if graph['brokers']:
            broker = graph['brokers'][0]
            graph['relationships']['routes'] = [
                {'from': broker['id'], 'to': t['id']}
                for t in graph['topics']
            ]
    
    def _apply_god_object(self, graph: Dict):
        """Apply God Object anti-pattern"""
        # Make one app publish/subscribe to most topics
        if graph['applications'] and graph['topics']:
            god_app = graph['applications'][0]
            god_app['type'] = 'PROSUMER'
            
            # Subscribe to 80% of topics
            for topic in random.sample(graph['topics'], 
                                      int(len(graph['topics']) * 0.8)):
                if not any(r['from'] == god_app['id'] and r['to'] == topic['id']
                          for r in graph['relationships']['subscribes_to']):
                    graph['relationships']['subscribes_to'].append({
                        'from': god_app['id'],
                        'to': topic['id']
                    })
    
    def _apply_single_broker(self, graph: Dict):
        """Apply single broker anti-pattern"""
        # Reduce to one broker
        if graph['brokers']:
            graph['brokers'] = [graph['brokers'][0]]
            graph['relationships']['routes'] = [
                {'from': graph['brokers'][0]['id'], 'to': t['id']}
                for t in graph['topics']
            ]
    
    def _apply_tight_coupling(self, graph: Dict):
        """Apply tight coupling anti-pattern"""
        # Create circular dependencies through topics
        if len(graph['applications']) >= 3:
            apps = graph['applications'][:3]
            topics_needed = 3
            
            if len(graph['topics']) < topics_needed:
                return
            
            topics = graph['topics'][:topics_needed]
            
            # A1 publishes T1, subscribes T3
            # A2 publishes T2, subscribes T1
            # A3 publishes T3, subscribes T2
            for i in range(3):
                pub_topic = topics[i]
                sub_topic = topics[(i + 2) % 3]
                
                # Add publish
                if not any(r['from'] == apps[i]['id'] and r['to'] == pub_topic['id']
                          for r in graph['relationships']['publishes_to']):
                    graph['relationships']['publishes_to'].append({
                        'from': apps[i]['id'],
                        'to': pub_topic['id'],
                        'period_ms': 100,
                        'msg_size': 1024
                    })
                
                # Add subscribe
                if not any(r['from'] == apps[i]['id'] and r['to'] == sub_topic['id']
                          for r in graph['relationships']['subscribes_to']):
                    graph['relationships']['subscribes_to'].append({
                        'from': apps[i]['id'],
                        'to': sub_topic['id']
                    })