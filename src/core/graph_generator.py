"""
Graph Generator - Simplified Version 3.0

Generates pub-sub system graphs with the simplified model:

Vertices:
- Application: {id, name, role (pub|sub|pubsub)}
- Broker: {id, name}
- Topic: {id, name, size, qos {durability, reliability, transport_priority}}
- Node: {id, name}

Edges:
- PUBLISHES_TO (App → Topic): {from, to}
- SUBSCRIBES_TO (App → Topic): {from, to}
- ROUTES (Broker → Topic): {from, to}
- RUNS_ON (App/Broker → Node): {from, to}
- CONNECTS_TO (Node → Node): {from, to}

Features:
- Multiple scales (tiny to extreme)
- Domain-specific scenarios
- Anti-pattern injection
- Reproducible with seeds

Author: Research Team
Version: 3.0
"""

import random
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict


@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    # Required
    scale: Optional[str] = None # tiny, small, medium, large, xlarge, extreme
    scenario: Optional[str] = "generic" # generic, iot, financial, ecommerce, analytics, smart_city, healthcare, autonomous_vehicle, gaming
    
    # Optional overrides
    num_nodes: Optional[int] = None
    num_applications: Optional[int] = None
    num_topics: Optional[int] = None
    num_brokers: Optional[int] = None
    
    # Anti-patterns to inject
    antipatterns: List[str] = field(default_factory=list)
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        valid_scales = ['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme']
        valid_scenarios = ['generic', 'iot', 'financial', 'ecommerce', 'analytics', 
                          'smart_city', 'healthcare', 'autonomous_vehicle', 'gaming']
        valid_antipatterns = ['spof', 'broker_overload', 'god_topic', 'single_broker',
                             'tight_coupling', 'chatty', 'bottleneck', 'circular_dependency']
        
        if self.scale is not None and self.scale not in valid_scales:
            raise ValueError(f"Invalid scale '{self.scale}'. Valid: {valid_scales}")
        if self.scenario is not None and self.scenario not in valid_scenarios:
            raise ValueError(f"Invalid scenario '{self.scenario}'. Valid: {valid_scenarios}")
        
        for ap in self.antipatterns:
            if ap not in valid_antipatterns:
                raise ValueError(f"Invalid antipattern '{ap}'. Valid: {valid_antipatterns}")


class GraphGenerator:
    """
    Generates realistic pub-sub system graphs
    
    Supports:
    - Multiple scales (tiny to extreme)
    - Domain-specific scenarios
    - Anti-pattern injection
    - Reproducible generation
    """
    
    # Scale presets
    SCALES = {
        'tiny': {'nodes': 3, 'apps': 5, 'topics': 3, 'brokers': 1},
        'small': {'nodes': 5, 'apps': 12, 'topics': 8, 'brokers': 2},
        'medium': {'nodes': 15, 'apps': 50, 'topics': 30, 'brokers': 4},
        'large': {'nodes': 50, 'apps': 200, 'topics': 120, 'brokers': 10},
        'xlarge': {'nodes': 100, 'apps': 500, 'topics': 300, 'brokers': 20},
        'extreme': {'nodes': 200, 'apps': 1000, 'topics': 600, 'brokers': 30}
    }
    
    # Application name patterns by domain
    APP_NAMES = {
        'generic': [
            'ServiceHandler', 'DataProcessor', 'EventRouter', 'MessageGateway',
            'StateManager', 'CacheService', 'QueueWorker', 'Aggregator',
            'Transformer', 'Validator', 'Logger', 'Monitor'
        ],
        'iot': [
            'SensorCollector', 'DeviceManager', 'TelemetryAggregator',
            'CommandDispatcher', 'StatusMonitor', 'AlertProcessor',
            'DataForwarder', 'FirmwareManager', 'DiagnosticsEngine',
            'EdgeGateway', 'ProtocolBridge', 'DataFilter'
        ],
        'financial': [
            'OrderProcessor', 'MarketDataFeed', 'RiskEngine',
            'TradeExecutor', 'PositionTracker', 'ComplianceMonitor',
            'PricingEngine', 'AuditLogger', 'ReportGenerator',
            'OrderBookManager', 'MatchingEngine', 'SettlementService'
        ],
        'ecommerce': [
            'OrderService', 'InventoryManager', 'PaymentGateway',
            'ShippingCalculator', 'RecommendationEngine', 'CartService',
            'ReviewAggregator', 'NotificationService', 'FraudDetector',
            'ProductCatalog', 'SearchIndexer', 'PriceOptimizer'
        ],
        'analytics': [
            'DataIngester', 'StreamProcessor', 'BatchProcessor',
            'MetricsCollector', 'DashboardService', 'AlertManager',
            'ReportGenerator', 'MLPipeline', 'FeatureStore',
            'DataValidator', 'SchemaRegistry', 'QueryEngine'
        ],
        'smart_city': [
            'TrafficController', 'ParkingManager', 'PublicTransitTracker',
            'EnvironmentMonitor', 'EnergyManager', 'WasteCollector',
            'EmergencyDispatcher', 'CitizenPortal', 'StreetLightController',
            'WaterFlowMonitor', 'AirQualitySensor', 'NoiseMonitor'
        ],
        'healthcare': [
            'PatientMonitor', 'VitalSignsCollector', 'MedicationDispenser',
            'AppointmentScheduler', 'LabResultProcessor', 'DiagnosisEngine',
            'ClinicalDecisionSupport', 'EMRIntegration', 'BillingProcessor',
            'InsuranceVerifier', 'PharmacyInterface', 'ImagingService'
        ],
        'autonomous_vehicle': [
            'LidarProcessor', 'CameraFusion', 'RadarProcessor',
            'PathPlanner', 'MotionController', 'LocalizationService',
            'ObstacleDetector', 'TrafficSignReader', 'V2XCommunicator',
            'TelemetryRecorder', 'DiagnosticsLogger', 'SafetyMonitor'
        ],
        'gaming': [
            'GameStateManager', 'PlayerSessionHandler', 'MatchMaker',
            'LeaderboardService', 'InventoryManager', 'ChatService',
            'AchievementTracker', 'ReplayRecorder', 'AntiCheatEngine',
            'AnalyticsCollector', 'SocialGraphService', 'NotificationHub'
        ]
    }
    
    # Topic name patterns by domain
    TOPIC_PATTERNS = {
        'generic': [
            'events/{type}', 'data/{category}', 'commands/{target}',
            'status/{component}', 'logs/{level}', 'metrics/{source}'
        ],
        'iot': [
            'sensors/{type}/data', 'devices/{id}/status', 'devices/{id}/commands',
            'telemetry/aggregated', 'alerts/{severity}', 'config/{device}'
        ],
        'financial': [
            'market/{exchange}/quotes', 'orders/{type}', 'trades/executed',
            'positions/{account}', 'risk/metrics', 'compliance/alerts'
        ],
        'ecommerce': [
            'orders/{status}', 'inventory/updates', 'payments/{status}',
            'shipping/status', 'products/catalog', 'recommendations'
        ],
        'analytics': [
            'ingest/raw', 'stream/processed', 'metrics/{domain}',
            'alerts/{severity}', 'reports/generated', 'ml/predictions'
        ],
        'smart_city': [
            'traffic/{intersection}', 'parking/{zone}', 'transit/{route}',
            'environment/readings', 'energy/consumption', 'emergency/dispatch'
        ],
        'healthcare': [
            'patient/{id}/vitals', 'patient/alerts', 'medication/dispensed',
            'lab/results', 'imaging/status', 'scheduling/updates'
        ],
        'autonomous_vehicle': [
            'perception/{sensor}', 'planning/path', 'control/commands',
            'localization/position', 'v2x/messages', 'diagnostics/status'
        ],
        'gaming': [
            'game/{session}/state', 'player/actions', 'match/events',
            'chat/messages', 'inventory/updates', 'achievements/unlocked'
        ]
    }
    
    # QoS profiles by domain
    QOS_PROFILES = {
        'generic': {
            'default': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'MEDIUM'},
            'reliable': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'MEDIUM'},
            'persistent': {'durability': 'PERSISTENT', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'}
        },
        'iot': {
            'telemetry': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'MEDIUM'},
            'commands': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'},
            'alerts': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'URGENT'}
        },
        'financial': {
            'market_data': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'URGENT'},
            'orders': {'durability': 'PERSISTENT', 'reliability': 'RELIABLE', 'transport_priority': 'URGENT'},
            'audit': {'durability': 'PERSISTENT', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'}
        },
        'ecommerce': {
            'orders': {'durability': 'PERSISTENT', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'},
            'inventory': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'MEDIUM'},
            'recommendations': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'LOW'}
        },
        'analytics': {
            'raw': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'LOW'},
            'processed': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'MEDIUM'},
            'alerts': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'}
        },
        'smart_city': {
            'sensors': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'MEDIUM'},
            'emergency': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'URGENT'},
            'control': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'}
        },
        'healthcare': {
            'vitals': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'},
            'alerts': {'durability': 'PERSISTENT', 'reliability': 'RELIABLE', 'transport_priority': 'URGENT'},
            'records': {'durability': 'PERSISTENT', 'reliability': 'RELIABLE', 'transport_priority': 'MEDIUM'}
        },
        'autonomous_vehicle': {
            'perception': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'URGENT'},
            'control': {'durability': 'VOLATILE', 'reliability': 'RELIABLE', 'transport_priority': 'URGENT'},
            'safety': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'URGENT'}
        },
        'gaming': {
            'state': {'durability': 'VOLATILE', 'reliability': 'RELIABLE', 'transport_priority': 'HIGH'},
            'chat': {'durability': 'TRANSIENT_LOCAL', 'reliability': 'RELIABLE', 'transport_priority': 'MEDIUM'},
            'events': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT', 'transport_priority': 'MEDIUM'}
        }
    }
    
    # Message size ranges by domain (bytes)
    MESSAGE_SIZES = {
        'generic': (64, 4096),
        'iot': (32, 1024),
        'financial': (128, 2048),
        'ecommerce': (256, 8192),
        'analytics': (1024, 65536),
        'smart_city': (64, 2048),
        'healthcare': (128, 16384),
        'autonomous_vehicle': (256, 131072),
        'gaming': (64, 4096)
    }
    
    def __init__(self, config: GraphConfig):
        """Initialize generator with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        random.seed(config.seed)
        
        # Get scale defaults
        scale_defaults = self.SCALES.get(config.scale, self.SCALES['medium'])
        self.num_nodes = config.num_nodes or scale_defaults['nodes']
        self.num_apps = config.num_applications or scale_defaults['apps']
        self.num_topics = config.num_topics or scale_defaults['topics']
        self.num_brokers = config.num_brokers or scale_defaults['brokers']
        
        # Get domain-specific patterns
        self.app_names = self.APP_NAMES.get(config.scenario, self.APP_NAMES['generic'])
        self.topic_patterns = self.TOPIC_PATTERNS.get(config.scenario, self.TOPIC_PATTERNS['generic'])
        self.qos_profiles = self.QOS_PROFILES.get(config.scenario, self.QOS_PROFILES['generic'])
        self.msg_size_range = self.MESSAGE_SIZES.get(config.scenario, self.MESSAGE_SIZES['generic'])
        
        self.logger.info(f"Initialized GraphGenerator: scale={config.scale}, scenario={config.scenario}")
    
    def generate(self) -> Dict:
        """
        Generate a complete pub-sub system graph
        
        Returns:
            Dictionary with vertices and edges
        """
        self.logger.info(f"Generating {self.config.scale} scale {self.config.scenario} system...")
        
        start_time = datetime.utcnow()
        
        # Generate vertices
        nodes = self._generate_nodes()
        brokers = self._generate_brokers()
        topics = self._generate_topics()
        applications = self._generate_applications()
        
        # Initialize edges
        edges = {
            'publishes_to': [],
            'subscribes_to': [],
            'routes': [],
            'runs_on': [],
            'connects_to': []
        }
        
        # Generate edges
        self._generate_connects_to(nodes, edges)
        self._generate_runs_on(applications, brokers, nodes, edges)
        self._generate_routes(brokers, topics, edges)
        self._generate_pub_sub(applications, topics, edges)
        
        # Apply anti-patterns
        antipatterns_applied = {}
        for ap in self.config.antipatterns:
            result = self._apply_antipattern(ap, applications, topics, brokers, edges)
            if result:
                antipatterns_applied[ap] = result
        
        # Ensure connectivity
        self._ensure_connectivity(applications, topics, edges)
        
        # Build final graph
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        graph = {
            'metadata': {
                'id': f"graph_{self.config.scenario}_{self.config.scale}_{self.config.seed}",
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'generator_version': '3.0',
                'scale': self.config.scale,
                'scenario': self.config.scenario,
                'seed': self.config.seed,
                'generation_time_seconds': generation_time,
                'antipatterns_applied': antipatterns_applied if antipatterns_applied else None
            },
            'applications': applications,
            'brokers': brokers,
            'topics': topics,
            'nodes': nodes,
            'relationships': edges
        }
        
        # Calculate metrics
        graph['metrics'] = self._calculate_metrics(graph)
        
        self.logger.info(
            f"Generated: {len(nodes)} nodes, {len(applications)} apps, "
            f"{len(topics)} topics, {len(brokers)} brokers in {generation_time:.3f}s"
        )
        
        return graph
    
    def _generate_nodes(self) -> List[Dict]:
        """Generate Node vertices"""
        nodes = []
        for i in range(1, self.num_nodes + 1):
            nodes.append({
                'id': f'N{i}',
                'name': f'Node{i}'
            })
        return nodes
    
    def _generate_brokers(self) -> List[Dict]:
        """Generate Broker vertices"""
        brokers = []
        for i in range(1, self.num_brokers + 1):
            brokers.append({
                'id': f'B{i}',
                'name': f'Broker{i}'
            })
        return brokers
    
    def _generate_topics(self) -> List[Dict]:
        """Generate Topic vertices"""
        topics = []
        
        for i in range(1, self.num_topics + 1):
            # Generate topic name from pattern
            pattern = self.topic_patterns[(i - 1) % len(self.topic_patterns)]
            name = self._instantiate_topic_name(pattern, i)
            
            # Generate QoS based on topic pattern
            qos = self._generate_qos_for_topic(name)
            
            # Generate message size
            size = random.randint(self.msg_size_range[0], self.msg_size_range[1])
            
            topics.append({
                'id': f'T{i}',
                'name': name,
                'size': size,
                'qos': qos
            })
        
        return topics
    
    def _instantiate_topic_name(self, pattern: str, index: int) -> str:
        """Create concrete topic name from pattern"""
        name = pattern
        
        # Replace placeholders
        replacements = {
            '{type}': random.choice(['created', 'updated', 'deleted', 'processed']),
            '{category}': random.choice(['user', 'system', 'audit', 'metrics']),
            '{target}': random.choice(['service', 'device', 'user', 'system']),
            '{component}': random.choice(['api', 'db', 'cache', 'queue']),
            '{level}': random.choice(['info', 'warn', 'error', 'debug']),
            '{source}': random.choice(['app', 'infra', 'network', 'security']),
            '{id}': f'{index:04d}',
            '{severity}': random.choice(['critical', 'warning', 'info']),
            '{device}': random.choice(['sensor', 'actuator', 'gateway']),
            '{exchange}': random.choice(['NYSE', 'NASDAQ', 'LSE']),
            '{account}': f'acct{random.randint(1, 100)}',
            '{status}': random.choice(['pending', 'confirmed', 'shipped']),
            '{domain}': random.choice(['sales', 'marketing', 'ops']),
            '{intersection}': f'int{random.randint(1, 50)}',
            '{zone}': f'zone{random.randint(1, 10)}',
            '{route}': f'route{random.randint(1, 20)}',
            '{sensor}': random.choice(['lidar', 'camera', 'radar', 'gps']),
            '{session}': f'sess{random.randint(1, 100)}'
        }
        
        for placeholder, value in replacements.items():
            if placeholder in name:
                name = name.replace(placeholder, value, 1)
        
        return name
    
    def _generate_qos_for_topic(self, topic_name: str) -> Dict:
        """Generate QoS policy based on topic name and scenario"""
        name_lower = topic_name.lower()
        
        # Match keywords to QoS profiles
        keyword_profile_map = {
            'alert': 'alerts' if 'alerts' in self.qos_profiles else 'reliable',
            'command': 'commands' if 'commands' in self.qos_profiles else 'reliable',
            'order': 'orders' if 'orders' in self.qos_profiles else 'persistent',
            'trade': 'orders' if 'orders' in self.qos_profiles else 'persistent',
            'vital': 'vitals' if 'vitals' in self.qos_profiles else 'reliable',
            'patient': 'vitals' if 'vitals' in self.qos_profiles else 'reliable',
            'emergency': 'emergency' if 'emergency' in self.qos_profiles else 'reliable',
            'control': 'control' if 'control' in self.qos_profiles else 'reliable',
            'safety': 'safety' if 'safety' in self.qos_profiles else 'reliable',
            'perception': 'perception' if 'perception' in self.qos_profiles else 'default',
            'telemetry': 'telemetry' if 'telemetry' in self.qos_profiles else 'default',
            'sensor': 'sensors' if 'sensors' in self.qos_profiles else 'default',
            'market': 'market_data' if 'market_data' in self.qos_profiles else 'default',
            'audit': 'audit' if 'audit' in self.qos_profiles else 'persistent',
            'state': 'state' if 'state' in self.qos_profiles else 'reliable',
            'chat': 'chat' if 'chat' in self.qos_profiles else 'reliable'
        }
        
        # Find matching profile
        profile_key = 'default'
        for keyword, profile in keyword_profile_map.items():
            if keyword in name_lower:
                if profile in self.qos_profiles:
                    profile_key = profile
                    break
        
        # Get profile or default
        profile = self.qos_profiles.get(profile_key)
        if not profile:
            profile = list(self.qos_profiles.values())[0]
        
        return {
            'durability': profile.get('durability', 'VOLATILE'),
            'reliability': profile.get('reliability', 'BEST_EFFORT'),
            'transport_priority': profile.get('transport_priority', 'MEDIUM')
        }
    
    def _generate_applications(self) -> List[Dict]:
        """Generate Application vertices"""
        applications = []
        
        # Distribute roles: ~30% pub, ~40% sub, ~30% pubsub
        roles = []
        num_pub = int(self.num_apps * 0.3)
        num_sub = int(self.num_apps * 0.4)
        num_pubsub = self.num_apps - num_pub - num_sub
        
        roles = ['pub'] * num_pub + ['sub'] * num_sub + ['pubsub'] * num_pubsub
        random.shuffle(roles)
        
        for i in range(1, self.num_apps + 1):
            base_name = self.app_names[(i - 1) % len(self.app_names)]
            role = roles[i - 1] if i - 1 < len(roles) else 'pubsub'
            
            applications.append({
                'id': f'A{i}',
                'name': base_name if i <= len(self.app_names) else f'{base_name}_{i}',
                'role': role
            })
        
        return applications
    
    def _generate_connects_to(self, nodes: List[Dict], edges: Dict):
        """Generate CONNECTS_TO edges (Node → Node)"""
        if len(nodes) < 2:
            return
        
        # Create connected topology (ring + some cross-links)
        for i, node in enumerate(nodes):
            # Ring connection
            next_idx = (i + 1) % len(nodes)
            edges['connects_to'].append({
                'from': node['id'],
                'to': nodes[next_idx]['id']
            })
            
            # Random cross-link
            if len(nodes) > 3 and random.random() < 0.3:
                other_idx = random.randint(0, len(nodes) - 1)
                if other_idx != i and other_idx != next_idx:
                    edges['connects_to'].append({
                        'from': node['id'],
                        'to': nodes[other_idx]['id']
                    })
    
    def _generate_runs_on(self, applications: List[Dict], brokers: List[Dict], 
                         nodes: List[Dict], edges: Dict):
        """Generate RUNS_ON edges (App/Broker → Node)"""
        if not nodes:
            return
        
        # Place applications on nodes
        for app in applications:
            node = random.choice(nodes)
            edges['runs_on'].append({
                'from': app['id'],
                'to': node['id']
            })
        
        # Place brokers on nodes (distribute evenly)
        for i, broker in enumerate(brokers):
            node = nodes[i % len(nodes)]
            edges['runs_on'].append({
                'from': broker['id'],
                'to': node['id']
            })
    
    def _generate_routes(self, brokers: List[Dict], topics: List[Dict], edges: Dict):
        """Generate ROUTES edges (Broker → Topic)"""
        if not brokers or not topics:
            return
        
        # Distribute topics across brokers
        for i, topic in enumerate(topics):
            broker = brokers[i % len(brokers)]
            edges['routes'].append({
                'from': broker['id'],
                'to': topic['id']
            })
    
    def _generate_pub_sub(self, applications: List[Dict], topics: List[Dict], edges: Dict):
        """Generate PUBLISHES_TO and SUBSCRIBES_TO edges"""
        if not applications or not topics:
            return
        
        for app in applications:
            role = app['role']
            
            # Publishers
            if role in ['pub', 'pubsub']:
                num_topics = random.randint(1, min(5, len(topics)))
                pub_topics = random.sample(topics, num_topics)
                
                for topic in pub_topics:
                    edges['publishes_to'].append({
                        'from': app['id'],
                        'to': topic['id']
                    })
            
            # Subscribers
            if role in ['sub', 'pubsub']:
                num_topics = random.randint(1, min(8, len(topics)))
                sub_topics = random.sample(topics, num_topics)
                
                for topic in sub_topics:
                    edges['subscribes_to'].append({
                        'from': app['id'],
                        'to': topic['id']
                    })
    
    def _apply_antipattern(self, antipattern: str, applications: List[Dict], 
                          topics: List[Dict], brokers: List[Dict], edges: Dict) -> Optional[Dict]:
        """Apply an anti-pattern and return info"""
        
        if antipattern == 'spof':
            return self._apply_spof(topics, edges)
        elif antipattern == 'broker_overload':
            return self._apply_broker_overload(brokers, topics, edges)
        elif antipattern == 'god_topic':
            return self._apply_god_topic(applications, topics, edges)
        elif antipattern == 'single_broker':
            return self._apply_single_broker(brokers, topics, edges)
        elif antipattern == 'tight_coupling':
            return self._apply_tight_coupling(applications, topics, edges)
        elif antipattern == 'chatty':
            return self._apply_chatty(topics)
        elif antipattern == 'bottleneck':
            return self._apply_bottleneck(applications, topics, edges)
        elif antipattern == 'circular_dependency':
            return self._apply_circular_dependency(applications, topics, edges)
        
        return None
    
    def _apply_spof(self, topics: List[Dict], edges: Dict) -> Dict:
        """SPOF: Critical topic with single publisher"""
        if not topics:
            return {}
        
        # Find topic with most subscribers
        topic_sub_count = defaultdict(int)
        for sub in edges['subscribes_to']:
            topic_sub_count[sub['to']] += 1
        
        if not topic_sub_count:
            spof_topic = random.choice(topics)
        else:
            spof_topic_id = max(topic_sub_count, key=topic_sub_count.get)
            spof_topic = next((t for t in topics if t['id'] == spof_topic_id), topics[0])
        
        # Ensure only one publisher
        pubs = [p for p in edges['publishes_to'] if p['to'] == spof_topic['id']]
        if len(pubs) > 1:
            keep = random.choice(pubs)
            edges['publishes_to'] = [p for p in edges['publishes_to'] 
                                     if p['to'] != spof_topic['id'] or p == keep]
        
        return {'topic': spof_topic['id']}
    
    def _apply_broker_overload(self, brokers: List[Dict], topics: List[Dict], edges: Dict) -> Dict:
        """Broker overload: 80% of topics on one broker"""
        if len(brokers) < 2 or not topics:
            return {}
        
        overloaded = random.choice(brokers)
        num_topics = int(len(topics) * 0.8)
        overload_topics = random.sample(topics, num_topics)
        
        # Remove existing routes for these topics
        overload_ids = {t['id'] for t in overload_topics}
        edges['routes'] = [r for r in edges['routes'] if r['to'] not in overload_ids]
        
        # Add all to overloaded broker
        for topic in overload_topics:
            edges['routes'].append({
                'from': overloaded['id'],
                'to': topic['id']
            })
        
        return {'broker': overloaded['id'], 'topic_count': num_topics}
    
    def _apply_god_topic(self, applications: List[Dict], topics: List[Dict], edges: Dict) -> Dict:
        """God topic: Single topic with 75%+ subscribers"""
        if not topics or not applications:
            return {}
        
        god_topic = random.choice(topics)
        subscribers = [a for a in applications if a['role'] in ['sub', 'pubsub']]
        
        num_subs = int(len(subscribers) * 0.75)
        god_subscribers = random.sample(subscribers, min(num_subs, len(subscribers)))
        
        for app in god_subscribers:
            if not any(s['from'] == app['id'] and s['to'] == god_topic['id'] 
                      for s in edges['subscribes_to']):
                edges['subscribes_to'].append({
                    'from': app['id'],
                    'to': god_topic['id']
                })
        
        return {'topic': god_topic['id'], 'subscriber_count': len(god_subscribers)}
    
    def _apply_single_broker(self, brokers: List[Dict], topics: List[Dict], edges: Dict) -> Dict:
        """Single broker: All topics through one broker"""
        if not brokers or not topics:
            return {}
        
        single = brokers[0]
        edges['routes'] = []
        
        for topic in topics:
            edges['routes'].append({
                'from': single['id'],
                'to': topic['id']
            })
        
        return {'broker': single['id']}
    
    def _apply_tight_coupling(self, applications: List[Dict], topics: List[Dict], edges: Dict) -> Dict:
        """Tight coupling: Circular pub-sub dependencies"""
        if len(applications) < 3 or len(topics) < 3:
            return {}
        
        # Create cycle: A1 → T1 ← A2, A2 → T2 ← A3, A3 → T3 ← A1
        cycle_apps = random.sample(applications, min(5, len(applications)))
        cycle_topics = random.sample(topics, min(len(cycle_apps), len(topics)))
        
        for i, app in enumerate(cycle_apps):
            topic = cycle_topics[i % len(cycle_topics)]
            next_app = cycle_apps[(i + 1) % len(cycle_apps)]
            
            # Current publishes
            if not any(p['from'] == app['id'] and p['to'] == topic['id'] 
                      for p in edges['publishes_to']):
                edges['publishes_to'].append({'from': app['id'], 'to': topic['id']})
            
            # Next subscribes
            if not any(s['from'] == next_app['id'] and s['to'] == topic['id'] 
                      for s in edges['subscribes_to']):
                edges['subscribes_to'].append({'from': next_app['id'], 'to': topic['id']})
        
        return {'cycle_apps': [a['id'] for a in cycle_apps]}
    
    def _apply_chatty(self, topics: List[Dict]) -> Dict:
        """Chatty: Small message sizes on many topics"""
        if not topics:
            return {}
        
        num_chatty = max(1, len(topics) // 3)
        chatty_topics = random.sample(topics, num_chatty)
        
        for topic in chatty_topics:
            topic['size'] = random.randint(8, 64)  # Very small messages
        
        return {'topics': [t['id'] for t in chatty_topics]}
    
    def _apply_bottleneck(self, applications: List[Dict], topics: List[Dict], edges: Dict) -> Dict:
        """Bottleneck: Many apps depend on single topic"""
        if not topics or not applications:
            return {}
        
        bottleneck = random.choice(topics)
        subscribers = [a for a in applications if a['role'] in ['sub', 'pubsub']]
        
        num_subs = int(len(subscribers) * 0.8)
        for app in random.sample(subscribers, min(num_subs, len(subscribers))):
            if not any(s['from'] == app['id'] and s['to'] == bottleneck['id'] 
                      for s in edges['subscribes_to']):
                edges['subscribes_to'].append({
                    'from': app['id'],
                    'to': bottleneck['id']
                })
        
        return {'topic': bottleneck['id']}
    
    def _apply_circular_dependency(self, applications: List[Dict], topics: List[Dict], edges: Dict) -> Dict:
        """Circular dependency: A → B → C → A through topics"""
        if len(applications) < 3 or len(topics) < 3:
            return {}
        
        cycle = random.sample(applications, 3)
        cycle_topics = random.sample(topics, 3)
        
        # A1 publishes T1, A2 subscribes T1
        # A2 publishes T2, A3 subscribes T2  
        # A3 publishes T3, A1 subscribes T3
        for i in range(3):
            pub_app = cycle[i]
            sub_app = cycle[(i + 1) % 3]
            topic = cycle_topics[i]
            
            edges['publishes_to'].append({'from': pub_app['id'], 'to': topic['id']})
            edges['subscribes_to'].append({'from': sub_app['id'], 'to': topic['id']})
        
        return {'cycle': [a['id'] for a in cycle]}
    
    def _ensure_connectivity(self, applications: List[Dict], topics: List[Dict], edges: Dict):
        """Ensure all apps have at least one pub or sub connection"""
        if not applications or not topics:
            return
        
        connected = set()
        for pub in edges['publishes_to']:
            connected.add(pub['from'])
        for sub in edges['subscribes_to']:
            connected.add(sub['from'])
        
        for app in applications:
            if app['id'] not in connected:
                topic = random.choice(topics)
                if app['role'] in ['pub', 'pubsub']:
                    edges['publishes_to'].append({'from': app['id'], 'to': topic['id']})
                else:
                    edges['subscribes_to'].append({'from': app['id'], 'to': topic['id']})
    
    def _calculate_metrics(self, graph: Dict) -> Dict:
        """Calculate graph metrics"""
        edges = graph['relationships']
        topics = graph['topics']
        apps = graph['applications']
        
        # Topic metrics
        topic_pubs = defaultdict(int)
        topic_subs = defaultdict(int)
        for pub in edges['publishes_to']:
            topic_pubs[pub['to']] += 1
        for sub in edges['subscribes_to']:
            topic_subs[sub['to']] += 1
        
        # Role distribution
        role_counts = defaultdict(int)
        for app in apps:
            role_counts[app['role']] += 1
        
        # QoS distribution
        qos_counts = {
            'durability': defaultdict(int),
            'reliability': defaultdict(int),
            'transport_priority': defaultdict(int)
        }
        for topic in topics:
            qos = topic['qos']
            qos_counts['durability'][qos['durability']] += 1
            qos_counts['reliability'][qos['reliability']] += 1
            qos_counts['transport_priority'][qos['transport_priority']] += 1
        
        return {
            'vertex_counts': {
                'applications': len(apps),
                'brokers': len(graph['brokers']),
                'topics': len(topics),
                'nodes': len(graph['nodes'])
            },
            'edge_counts': {
                'publishes_to': len(edges['publishes_to']),
                'subscribes_to': len(edges['subscribes_to']),
                'routes': len(edges['routes']),
                'runs_on': len(edges['runs_on']),
                'connects_to': len(edges['connects_to']),
                'total': sum(len(v) for v in edges.values())
            },
            'pub_sub': {
                'unique_publishers': len(set(p['from'] for p in edges['publishes_to'])),
                'unique_subscribers': len(set(s['from'] for s in edges['subscribes_to'])),
                'avg_pubs_per_topic': sum(topic_pubs.values()) / len(topics) if topics else 0,
                'avg_subs_per_topic': sum(topic_subs.values()) / len(topics) if topics else 0,
                'max_fanout': max(topic_subs.values()) if topic_subs else 0
            },
            'role_distribution': dict(role_counts),
            'qos_distribution': {k: dict(v) for k, v in qos_counts.items()}
        }


def create_graph(scale: str = 'small', scenario: str = 'generic', 
                seed: int = 42, **kwargs) -> Dict:
    """
    Convenience function to create a graph
    
    Args:
        scale: Scale preset (tiny, small, medium, large, xlarge, extreme)
        scenario: Domain scenario
        seed: Random seed
        **kwargs: Additional GraphConfig parameters
    
    Returns:
        Generated graph dictionary
    """
    config = GraphConfig(scale=scale, scenario=scenario, seed=seed, **kwargs)
    generator = GraphGenerator(config)
    return generator.generate()