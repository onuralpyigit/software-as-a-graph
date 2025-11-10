"""
Graph Generator
Generates complex pub-sub system graphs with various scales and scenarios.
Supports sophisticated anti-pattern injection and domain-specific patterns.

Key Improvements:
1. Fixed incomplete methods in original implementation
2. More realistic pub-sub topology patterns
3. Sophisticated anti-pattern injection
4. Better QoS policy generation based on domain
5. Realistic message flow patterns
6. Circular dependency detection
7. Hidden coupling patterns
8. Zone-aware and region-aware deployments
"""

import random
import math
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from collections import defaultdict


@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    scale: str  # tiny, small, medium, large, xlarge, extreme
    scenario: str  # generic, iot, financial, ecommerce, analytics, smart_city, healthcare
    num_nodes: int
    num_applications: int
    num_topics: int
    num_brokers: int
    edge_density: float = 0.3  # 0.0 - 1.0
    high_availability: bool = False
    multi_zone: bool = False
    num_zones: int = 3
    num_regions: int = 1
    antipatterns: List[str] = field(default_factory=list)
    seed: int = 42
    realistic_topology: bool = True  # Use realistic pub-sub patterns


class GraphGenerator:
    """
    Enhanced Graph Generator with realistic DDS pub-sub system patterns
    
    Supports:
    - Multiple scales (tiny to extreme)
    - Domain-specific scenarios
    - Realistic QoS policies
    - Sophisticated anti-patterns
    - Zone/region-aware deployment
    - Realistic message flow patterns
    """
    
    # Predefined scales - expanded
    SCALES = {
        'tiny': {'nodes': 3, 'apps': 5, 'topics': 3, 'brokers': 1},
        'small': {'nodes': 5, 'apps': 10, 'topics': 8, 'brokers': 2},
        'medium': {'nodes': 15, 'apps': 50, 'topics': 25, 'brokers': 3},
        'large': {'nodes': 50, 'apps': 200, 'topics': 100, 'brokers': 8},
        'xlarge': {'nodes': 100, 'apps': 500, 'topics': 250, 'brokers': 15},
        'extreme': {'nodes': 200, 'apps': 1000, 'topics': 500, 'brokers': 20}
    }
    
    # Application types by domain - expanded
    APP_TYPES = {
        'generic': [
            'ServiceA', 'ServiceB', 'ServiceC', 'DataProcessor',
            'EventHandler', 'MessageRouter', 'Aggregator', 'Monitor',
            'Gateway', 'Transformer', 'Validator', 'Logger'
        ],
        'iot': [
            'SensorCollector', 'DeviceManager', 'TelemetryAggregator',
            'CommandDispatcher', 'StatusMonitor', 'AlertProcessor',
            'DataForwarder', 'FirmwareUpdater', 'DiagnosticsEngine',
            'EdgeGateway', 'ProtocolConverter', 'DataFilter'
        ],
        'financial': [
            'OrderProcessor', 'MarketDataFeed', 'RiskEngine',
            'TradeExecutor', 'PositionTracker', 'ComplianceMonitor',
            'PricingEngine', 'AuditLogger', 'ReportGenerator',
            'OrderBookManager', 'MatchingEngine', 'SettlementService'
        ],
        'ecommerce': [
            'OrderService', 'InventoryManager', 'PaymentProcessor',
            'ShippingCalculator', 'RecommendationEngine', 'CartService',
            'ReviewAggregator', 'NotificationService', 'FraudDetector',
            'ProductCatalog', 'SearchService', 'PromotionEngine'
        ],
        'analytics': [
            'DataCollector', 'StreamProcessor', 'MetricsAggregator',
            'LogAnalyzer', 'ReportGenerator', 'MLPipeline',
            'FeatureExtractor', 'ModelInferencer', 'ResultsPublisher',
            'DataWarehouse', 'ETLService', 'DashboardService'
        ],
        'smart_city': [
            'TrafficMonitor', 'ParkingService', 'StreetLightController',
            'WasteManagement', 'EmergencyDispatch', 'WeatherStation',
            'AirQualityMonitor', 'PublicTransitTracker', 'EventCoordinator',
            'CameraAnalyzer', 'IncidentDetector', 'ResourceOptimizer'
        ],
        'healthcare': [
            'PatientMonitor', 'VitalSignsCollector', 'AlertDispatcher',
            'DiagnosticsEngine', 'RecordManager', 'AppointmentScheduler',
            'MedicationTracker', 'LabResultsProcessor', 'ImagingService',
            'EmergencyCoordinator', 'ComplianceChecker', 'BillingService'
        ]
    }
    
    # Topic patterns by domain - expanded with hierarchies
    TOPIC_PATTERNS = {
        'generic': [
            'events', 'commands', 'status', 'data', 'metrics',
            'alerts', 'config', 'health', 'logs', 'notifications'
        ],
        'iot': [
            'telemetry/raw', 'telemetry/aggregated', 'device/status', 
            'device/config', 'sensor/data', 'sensor/calibration',
            'command/control', 'command/firmware', 'alert/threshold',
            'alert/anomaly', 'diagnostic/health', 'diagnostic/performance'
        ],
        'financial': [
            'market/data/L1', 'market/data/L2', 'order/new', 'order/cancel',
            'order/modify', 'trade/execution', 'trade/confirmation',
            'position/update', 'risk/limit', 'risk/breach', 
            'price/quote', 'price/indicative', 'settlement/instruction',
            'audit/trail', 'compliance/alert'
        ],
        'ecommerce': [
            'order/created', 'order/updated', 'order/cancelled',
            'inventory/update', 'inventory/low_stock', 'payment/initiated',
            'payment/completed', 'payment/failed', 'shipping/dispatched',
            'shipping/delivered', 'user/registered', 'user/updated',
            'product/added', 'product/updated', 'cart/modified',
            'review/submitted', 'notification/email', 'notification/sms'
        ],
        'analytics': [
            'raw/events', 'raw/logs', 'processed/aggregated',
            'processed/enriched', 'aggregated/hourly', 'aggregated/daily',
            'metrics/system', 'metrics/business', 'model/input/features',
            'model/output/predictions', 'report/scheduled', 'report/adhoc',
            'alert/threshold', 'alert/anomaly'
        ],
        'smart_city': [
            'traffic/flow', 'traffic/congestion', 'parking/availability',
            'parking/occupancy', 'lighting/status', 'lighting/control',
            'waste/fill_level', 'waste/collection', 'emergency/incident',
            'emergency/dispatch', 'weather/current', 'weather/forecast',
            'air_quality/reading', 'transit/location', 'transit/schedule'
        ],
        'healthcare': [
            'patient/vitals', 'patient/alerts', 'patient/admission',
            'patient/discharge', 'vital_signs/ecg', 'vital_signs/bp',
            'vital_signs/spo2', 'alert/critical', 'alert/warning',
            'lab/results', 'lab/requested', 'imaging/ordered',
            'imaging/completed', 'medication/administered', 
            'appointment/scheduled', 'billing/claim'
        ]
    }
    
    # Application type distribution (PRODUCER, CONSUMER, PROSUMER)
    APP_TYPE_DISTRIBUTION = {
        'generic': {'PRODUCER': 0.3, 'CONSUMER': 0.3, 'PROSUMER': 0.4},
        'iot': {'PRODUCER': 0.4, 'CONSUMER': 0.2, 'PROSUMER': 0.4},
        'financial': {'PRODUCER': 0.2, 'CONSUMER': 0.3, 'PROSUMER': 0.5},
        'ecommerce': {'PRODUCER': 0.25, 'CONSUMER': 0.25, 'PROSUMER': 0.5},
        'analytics': {'PRODUCER': 0.35, 'CONSUMER': 0.15, 'PROSUMER': 0.5},
        'smart_city': {'PRODUCER': 0.45, 'CONSUMER': 0.2, 'PROSUMER': 0.35},
        'healthcare': {'PRODUCER': 0.3, 'CONSUMER': 0.3, 'PROSUMER': 0.4}
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
        self.app_type_dist = self.APP_TYPE_DISTRIBUTION.get(
            config.scenario,
            self.APP_TYPE_DISTRIBUTION['generic']
        )
        
        # Track components for relationship generation
        self.topic_hierarchy: Dict[str, List[str]] = {}
        self.critical_topics: Set[str] = set()
        self.critical_apps: Set[str] = set()
    
    def generate(self) -> Dict:
        """
        Generate complete graph with realistic patterns
        
        Returns:
            Dictionary with graph data in standard format
        """
        self.logger.info(f"Generating {self.config.scale} scale "
                        f"{self.config.scenario} system...")
        
        graph = {
            'metadata': self._generate_metadata(),
            'nodes': self._generate_nodes(),
            'brokers': self._generate_brokers(),
            'topics': self._generate_topics(),
            'applications': self._generate_applications(),
            'relationships': {
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'runs_on': []
            }
        }
        
        # Generate relationships with realistic patterns
        self._generate_runs_on(graph)
        self._generate_routes(graph)
        
        if self.config.realistic_topology:
            self._generate_realistic_pub_sub(graph)
        else:
            self._generate_random_pub_sub(graph)
        
        # Apply anti-patterns if requested
        if self.config.antipatterns:
            self._apply_antipatterns(graph)
        
        # Post-process to ensure consistency
        self._ensure_connectivity(graph)
        self._calculate_derived_metrics(graph)
        
        self.logger.info(f"Generated: {len(graph['nodes'])} nodes, "
                        f"{len(graph['applications'])} apps, "
                        f"{len(graph['topics'])} topics, "
                        f"{len(graph['brokers'])} brokers")
        
        return graph
    
    def _generate_metadata(self) -> Dict:
        """Generate graph metadata"""
        return {
            'generated_at': datetime.utcnow().isoformat(),
            'generator_version': '2.0',
            'config': asdict(self.config),
            'description': f"{self.config.scale} scale {self.config.scenario} system"
        }
    
    def _generate_nodes(self) -> List[Dict]:
        """Generate physical nodes with zone/region awareness"""
        nodes = []
        
        zones = [f'zone-{i+1}' for i in range(self.config.num_zones)] if self.config.multi_zone else ['default']
        regions = [f'region-{i+1}' for i in range(self.config.num_regions)]
        
        node_types = ['VM', 'Container', 'Bare Metal', 'Edge Device', 'Cloud Instance']
        
        for i in range(1, self.config.num_nodes + 1):
            zone = zones[i % len(zones)]
            region = regions[i % len(regions)]
            
            node = {
                'id': f'N{i}',
                'name': f'Node{i}',
                'zone': zone,
                'region': region,
                'node_type': random.choice(node_types),
                'capacity': random.uniform(500, 5000),
                'cpu_cores': random.choice([4, 8, 16, 32, 64]),
                'memory_gb': random.choice([16, 32, 64, 128, 256]),
                'network_bandwidth_mbps': random.choice([1000, 10000, 25000, 100000]),
                'uptime_seconds': random.uniform(0, 86400 * 365),
                'os': random.choice(['Linux', 'Ubuntu', 'RHEL', 'Windows Server']),
                'ip_address': f'192.168.{i//256}.{i%256}'
            }
            nodes.append(node)
        
        return nodes
    
    def _generate_brokers(self) -> List[Dict]:
        """Generate brokers with realistic capacity planning"""
        brokers = []
        nodes = list(range(1, self.config.num_nodes + 1))
        
        # Calculate capacity based on total topics
        avg_topics_per_broker = math.ceil(self.config.num_topics / self.config.num_brokers)
        avg_apps_per_broker = math.ceil(self.config.num_applications / self.config.num_brokers)
        
        for i in range(1, self.config.num_brokers + 1):
            # Assign broker to node (distribute across nodes)
            node_id = f'N{nodes[(i-1) % len(nodes)]}'
            
            # Vary capacity slightly
            capacity_multiplier = random.uniform(0.8, 1.2)
            
            broker = {
                'id': f'B{i}',
                'name': f'Broker{i}',
                'node_id': node_id,
                'max_topics': int(avg_topics_per_broker * capacity_multiplier * 1.5),
                'max_applications': int(avg_apps_per_broker * capacity_multiplier * 1.5),
                'port': 7400 + i,
                'protocol': random.choice(['DDS', 'RTPS', 'DDS-RTPS']),
                'transport': random.choice(['UDP', 'TCP', 'Shared Memory']),
                'discovery_service': f'DiscoveryService{i}',
                'capacity_cpu': random.uniform(0.5, 8.0),
                'capacity_memory_mb': random.uniform(512, 4096)
            }
            brokers.append(broker)
        
        return brokers
    
    def _generate_topics(self) -> List[Dict]:
        """Generate topics with realistic QoS and hierarchies"""
        topics = []
        
        # Build topic hierarchy for realistic patterns
        self._build_topic_hierarchy()
        
        for i in range(1, self.config.num_topics + 1):
            # Select pattern - favor hierarchical if available
            if '/' in self.topic_patterns[0]:  # Hierarchical
                pattern = random.choice(self.topic_patterns)
            else:
                pattern = random.choice(self.topic_patterns)
                if random.random() < 0.3:  # 30% chance of sub-topic
                    pattern = f'{pattern}/sub{random.randint(1,3)}'
            
            topic_name = f'{pattern}_{i}'
            
            # Generate QoS based on scenario and topic type
            qos = self._generate_qos_for_topic(pattern, topic_name)
            
            # Determine criticality based on QoS and pattern
            criticality = self._determine_topic_criticality(pattern, qos)
            
            # Message characteristics
            msg_size = self._get_message_size_for_scenario(pattern)
            msg_rate = self._get_message_rate_for_scenario(pattern)
            
            topic = {
                'id': f'T{i}',
                'name': topic_name,
                'pattern': pattern,
                'qos': qos,
                'criticality': criticality,
                'message_size_bytes': msg_size,
                'expected_rate_hz': msg_rate,
                'max_message_size_bytes': int(msg_size * 1.5),
                'content_filter': self._generate_content_filter(pattern)
            }
            
            topics.append(topic)
            
            if criticality in ['HIGH', 'CRITICAL']:
                self.critical_topics.add(f'T{i}')
        
        return topics
    
    def _build_topic_hierarchy(self):
        """Build topic hierarchy for realistic parent-child patterns"""
        self.topic_hierarchy = defaultdict(list)
        
        for pattern in self.topic_patterns:
            if '/' in pattern:
                parts = pattern.split('/')
                parent = parts[0]
                self.topic_hierarchy[parent].append(pattern)
    
    def _generate_qos_for_topic(self, pattern: str, topic_name: str) -> Dict:
        """Generate realistic QoS policies based on scenario and topic pattern"""
        
        # Default QoS
        qos = {
            'durability': 'VOLATILE',
            'reliability': 'BEST_EFFORT',
            'history_depth': 1,
            'deadline_ms': None,
            'lifespan_ms': None,
            'transport_priority': 'MEDIUM'
        }
        
        # Scenario-specific QoS logic
        if self.config.scenario == 'financial':
            # Financial: high reliability, low latency
            qos.update({
                'durability': random.choice(['TRANSIENT_LOCAL', 'PERSISTENT']),
                'reliability': 'RELIABLE',
                'history_depth': random.choice([10, 20, 50, 100]),
                'deadline_ms': random.choice([5, 10, 50, 100]),
                'transport_priority': random.choice(['HIGH', 'URGENT'])
            })
            
            # Orders and trades need persistent storage
            if 'order' in pattern or 'trade' in pattern:
                qos['durability'] = 'PERSISTENT'
                qos['history_depth'] = 100
                qos['deadline_ms'] = 10
        
        elif self.config.scenario == 'iot':
            # IoT: mixed requirements
            if 'telemetry' in pattern or 'sensor' in pattern:
                # Sensor data: best effort, volatile
                qos.update({
                    'durability': 'VOLATILE',
                    'reliability': 'BEST_EFFORT',
                    'history_depth': 1,
                    'deadline_ms': random.choice([100, 500, 1000]),
                    'transport_priority': 'LOW'
                })
            elif 'command' in pattern or 'control' in pattern:
                # Commands: reliable, transient
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 10,
                    'deadline_ms': random.choice([50, 100, 200]),
                    'transport_priority': 'HIGH'
                })
            elif 'alert' in pattern:
                # Alerts: reliable, urgent
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 20,
                    'deadline_ms': 50,
                    'transport_priority': 'URGENT'
                })
        
        elif self.config.scenario == 'healthcare':
            # Healthcare: safety-critical
            if 'vital' in pattern or 'alert' in pattern:
                qos.update({
                    'durability': 'PERSISTENT',
                    'reliability': 'RELIABLE',
                    'history_depth': 50,
                    'deadline_ms': random.choice([50, 100]),
                    'transport_priority': 'URGENT',
                    'lifespan_ms': 3600000  # 1 hour
                })
            elif 'patient' in pattern:
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 20,
                    'deadline_ms': 500,
                    'transport_priority': 'HIGH'
                })
        
        elif self.config.scenario == 'analytics':
            # Analytics: high throughput, eventual consistency
            if 'raw' in pattern:
                qos.update({
                    'durability': 'VOLATILE',
                    'reliability': 'BEST_EFFORT',
                    'history_depth': 1,
                    'deadline_ms': None,
                    'transport_priority': 'LOW'
                })
            elif 'aggregated' in pattern or 'report' in pattern:
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 10,
                    'deadline_ms': 5000,
                    'transport_priority': 'MEDIUM'
                })
        
        elif self.config.scenario == 'smart_city':
            # Smart city: sensor data vs control
            if 'traffic' in pattern or 'parking' in pattern:
                qos.update({
                    'durability': 'VOLATILE',
                    'reliability': 'BEST_EFFORT',
                    'history_depth': 5,
                    'deadline_ms': 1000,
                    'transport_priority': 'MEDIUM'
                })
            elif 'emergency' in pattern:
                qos.update({
                    'durability': 'PERSISTENT',
                    'reliability': 'RELIABLE',
                    'history_depth': 50,
                    'deadline_ms': 100,
                    'transport_priority': 'URGENT'
                })
            elif 'control' in pattern:
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 10,
                    'deadline_ms': 200,
                    'transport_priority': 'HIGH'
                })
        
        elif self.config.scenario == 'ecommerce':
            # E-commerce: transactional
            if 'order' in pattern or 'payment' in pattern:
                qos.update({
                    'durability': 'PERSISTENT',
                    'reliability': 'RELIABLE',
                    'history_depth': 50,
                    'deadline_ms': 1000,
                    'transport_priority': 'HIGH',
                    'lifespan_ms': 86400000  # 24 hours
                })
            elif 'inventory' in pattern:
                qos.update({
                    'durability': 'TRANSIENT_LOCAL',
                    'reliability': 'RELIABLE',
                    'history_depth': 20,
                    'deadline_ms': 500,
                    'transport_priority': 'MEDIUM'
                })
        
        return qos
    
    def _determine_topic_criticality(self, pattern: str, qos: Dict) -> str:
        """Determine topic criticality based on QoS and pattern"""
        score = 0
        
        # QoS-based scoring
        if qos['durability'] == 'PERSISTENT':
            score += 3
        elif qos['durability'] in ['TRANSIENT', 'TRANSIENT_LOCAL']:
            score += 2
        
        if qos['reliability'] == 'RELIABLE':
            score += 2
        
        if qos.get('deadline_ms'):
            if qos['deadline_ms'] < 100:
                score += 3
            elif qos['deadline_ms'] < 500:
                score += 2
            else:
                score += 1
        
        if qos['transport_priority'] == 'URGENT':
            score += 3
        elif qos['transport_priority'] == 'HIGH':
            score += 2
        
        # Pattern-based scoring
        critical_keywords = ['alert', 'emergency', 'critical', 'order', 'trade', 'vital', 'command']
        for keyword in critical_keywords:
            if keyword in pattern.lower():
                score += 2
                break
        
        # Convert score to criticality level
        if score >= 8:
            return 'CRITICAL'
        elif score >= 5:
            return 'HIGH'
        elif score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_message_size_for_scenario(self, pattern: str) -> int:
        """Get realistic message sizes based on scenario and pattern"""
        
        if self.config.scenario == 'financial':
            # Financial: compact messages
            if 'quote' in pattern or 'price' in pattern:
                return random.choice([64, 128, 256])
            else:
                return random.choice([256, 512, 1024])
        
        elif self.config.scenario == 'iot':
            # IoT: small sensor readings, larger commands
            if 'sensor' in pattern or 'telemetry' in pattern:
                return random.choice([32, 64, 128])
            elif 'command' in pattern:
                return random.choice([128, 256, 512])
            else:
                return random.choice([256, 512])
        
        elif self.config.scenario == 'analytics':
            # Analytics: large batches
            if 'raw' in pattern:
                return random.choice([1024, 2048, 4096, 8192])
            else:
                return random.choice([512, 1024, 2048])
        
        else:
            # Default distribution
            return random.choice([128, 256, 512, 1024, 2048, 4096])
    
    def _get_message_rate_for_scenario(self, pattern: str) -> float:
        """Get realistic message rates based on scenario and pattern"""
        
        if self.config.scenario == 'financial':
            # Financial: very high frequency
            if 'market' in pattern:
                return random.choice([100, 500, 1000, 5000])
            elif 'order' in pattern:
                return random.choice([10, 50, 100, 500])
            else:
                return random.choice([1, 5, 10, 50])
        
        elif self.config.scenario == 'iot':
            # IoT: high frequency telemetry
            if 'telemetry' in pattern or 'sensor' in pattern:
                return random.choice([10, 20, 50, 100])
            elif 'command' in pattern:
                return random.choice([0.1, 1, 5])
            else:
                return random.choice([1, 5, 10])
        
        elif self.config.scenario == 'healthcare':
            # Healthcare: continuous monitoring
            if 'vital' in pattern:
                return random.choice([1, 5, 10, 20])
            else:
                return random.choice([0.1, 0.5, 1, 5])
        
        else:
            # Default distribution
            return random.choice([0.1, 1, 5, 10, 20, 50, 100])
    
    def _generate_content_filter(self, pattern: str) -> Optional[str]:
        """Generate realistic content filters for topics"""
        if random.random() < 0.3:  # 30% of topics have filters
            if self.config.scenario == 'financial':
                return f"symbol = 'AAPL' OR symbol = 'GOOGL'"
            elif self.config.scenario == 'iot':
                return f"device_id IN ('dev001', 'dev002', 'dev003')"
            else:
                return f"priority > {random.choice([5, 7, 9])}"
        return None
    
    def _generate_applications(self) -> List[Dict]:
        """Generate applications with realistic distributions"""
        applications = []
        
        # Calculate type distribution
        num_producers = int(self.config.num_applications * self.app_type_dist['PRODUCER'])
        num_consumers = int(self.config.num_applications * self.app_type_dist['CONSUMER'])
        num_prosumers = self.config.num_applications - num_producers - num_consumers
        
        types = (['PRODUCER'] * num_producers + 
                ['CONSUMER'] * num_consumers + 
                ['PROSUMER'] * num_prosumers)
        random.shuffle(types)
        
        for i in range(1, self.config.num_applications + 1):
            # Select base name from domain-specific types
            base_name = random.choice(self.app_types)
            app_type = types[i-1] if i-1 < len(types) else 'PROSUMER'
            
            # Determine criticality
            criticality = random.choices(
                ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                weights=[0.3, 0.4, 0.2, 0.1]
            )[0]
            
            # Replicas for HA
            replicas = 1
            if self.config.high_availability and criticality in ['HIGH', 'CRITICAL']:
                replicas = random.choice([2, 3, 5])
            
            # Resource requirements scale with criticality
            cpu_multiplier = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 2.0, 'CRITICAL': 4.0}[criticality]
            mem_multiplier = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 1.5, 'CRITICAL': 2.0}[criticality]
            
            app = {
                'id': f'A{i}',
                'name': f'{base_name}_{i}',
                'type': app_type,
                'criticality': criticality,
                'replicas': replicas,
                'cpu_request': round(random.uniform(0.5, 4.0) * cpu_multiplier, 2),
                'memory_request_mb': int(random.uniform(256, 2048) * mem_multiplier),
                'restart_policy': 'Always' if criticality in ['HIGH', 'CRITICAL'] else 'OnFailure',
                'health_check_enabled': criticality in ['HIGH', 'CRITICAL'],
                'monitoring_level': criticality
            }
            
            applications.append(app)
            
            if criticality in ['HIGH', 'CRITICAL']:
                self.critical_apps.add(f'A{i}')
        
        return applications
    
    def _generate_runs_on(self, graph: Dict):
        """Generate runs_on relationships with zone awareness"""
        apps = graph['applications']
        nodes = graph['nodes']
        
        for app in apps:
            if app['replicas'] > 1:
                # HA: distribute replicas across zones/nodes
                selected_nodes = []
                
                if self.config.multi_zone:
                    # Select nodes from different zones
                    zones_used = set()
                    for node in random.sample(nodes, min(app['replicas'], len(nodes))):
                        if node['zone'] not in zones_used:
                            selected_nodes.append(node['id'])
                            zones_used.add(node['zone'])
                else:
                    # Just distribute across different nodes
                    selected_nodes = [n['id'] for n in random.sample(nodes, min(app['replicas'], len(nodes)))]
                
                # Create runs_on for each replica
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
    
    def _generate_routes(self, graph: Dict):
        """Generate broker-to-topic routing with load balancing"""
        brokers = graph['brokers']
        topics = graph['topics']
        
        if len(brokers) == 1:
            # Single broker - all topics
            for topic in topics:
                graph['relationships']['routes'].append({
                    'from': brokers[0]['id'],
                    'to': topic['id']
                })
        else:
            # Distribute topics across brokers
            # Critical topics get multiple brokers for redundancy
            for i, topic in enumerate(topics):
                # Primary broker
                primary_broker = brokers[i % len(brokers)]
                graph['relationships']['routes'].append({
                    'from': primary_broker['id'],
                    'to': topic['id'],
                    'role': 'primary'
                })
                
                # Secondary broker for critical topics
                if topic['id'] in self.critical_topics and len(brokers) > 1:
                    secondary_idx = (i + 1) % len(brokers)
                    secondary_broker = brokers[secondary_idx]
                    graph['relationships']['routes'].append({
                        'from': secondary_broker['id'],
                        'to': topic['id'],
                        'role': 'secondary'
                    })
    
    def _generate_realistic_pub_sub(self, graph: Dict):
        """Generate realistic pub-sub relationships based on domain patterns"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Group topics by pattern prefix
        topic_groups = defaultdict(list)
        for topic in topics:
            prefix = topic['pattern'].split('/')[0] if '/' in topic['pattern'] else topic['pattern']
            topic_groups[prefix].append(topic)
        
        for app in apps:
            app_type = app['type']
            app_name = app['name'].lower()
            
            # Publishers
            if app_type in ['PRODUCER', 'PROSUMER']:
                # Select topics based on app name and topic patterns
                relevant_topics = self._find_relevant_topics_for_app(app_name, topics, 'publish')
                
                # Number of topics to publish (typically 1-3)
                num_pub = random.randint(1, min(3, len(relevant_topics))) if relevant_topics else 1
                if not relevant_topics:
                    relevant_topics = topics
                
                pub_topics = random.sample(relevant_topics, min(num_pub, len(relevant_topics)))
                
                for topic in pub_topics:
                    # Calculate realistic message rate
                    base_rate = topic['expected_rate_hz']
                    actual_rate = base_rate * random.uniform(0.8, 1.2)
                    
                    graph['relationships']['publishes_to'].append({
                        'from': app['id'],
                        'to': topic['id'],
                        'period_ms': int(1000 / actual_rate) if actual_rate > 0 else None,
                        'msg_size': topic['message_size_bytes'],
                        'burst_allowed': random.random() < 0.3
                    })
            
            # Subscribers
            if app_type in ['CONSUMER', 'PROSUMER']:
                # Select topics based on app name and topic patterns
                relevant_topics = self._find_relevant_topics_for_app(app_name, topics, 'subscribe')
                
                # Number of topics to subscribe (1-5, consumers typically subscribe to more)
                max_sub = 5 if app_type == 'CONSUMER' else 3
                num_sub = random.randint(1, min(max_sub, len(relevant_topics))) if relevant_topics else 1
                if not relevant_topics:
                    relevant_topics = topics
                
                sub_topics = random.sample(relevant_topics, min(num_sub, len(relevant_topics)))
                
                for topic in sub_topics:
                    graph['relationships']['subscribes_to'].append({
                        'from': app['id'],
                        'to': topic['id'],
                        'queue_size': random.choice([10, 20, 50, 100]),
                        'take_history': random.random() < 0.3
                    })
    
    def _find_relevant_topics_for_app(self, app_name: str, topics: List[Dict], action: str) -> List[Dict]:
        """Find topics relevant to an application based on naming patterns"""
        relevant = []
        
        # Extract keywords from app name
        keywords = set()
        for word in ['sensor', 'telemetry', 'order', 'market', 'command', 
                     'inventory', 'payment', 'alert', 'monitor', 'processor',
                     'traffic', 'parking', 'vital', 'patient', 'data']:
            if word in app_name:
                keywords.add(word)
        
        # Match topics
        for topic in topics:
            topic_pattern = topic['pattern'].lower()
            topic_name = topic['name'].lower()
            
            # Check for keyword matches
            for keyword in keywords:
                if keyword in topic_pattern or keyword in topic_name:
                    relevant.append(topic)
                    break
        
        return relevant
    
    def _generate_random_pub_sub(self, graph: Dict):
        """Generate random pub-sub relationships (fallback)"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Calculate edge density
        max_pub = max(1, int(len(topics) * 0.3))
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
    
    def _apply_antipatterns(self, graph: Dict):
        """Apply anti-patterns to the graph"""
        for antipattern in self.config.antipatterns:
            if antipattern == 'spof':
                self._apply_spof_antipattern(graph)
            elif antipattern == 'broker_overload':
                self._apply_broker_overload_antipattern(graph)
            elif antipattern == 'god_object':
                self._apply_god_object_antipattern(graph)
            elif antipattern == 'single_broker':
                self._apply_single_broker_antipattern(graph)
            elif antipattern == 'tight_coupling':
                self._apply_tight_coupling_antipattern(graph)
            elif antipattern == 'chatty_communication':
                self._apply_chatty_communication_antipattern(graph)
            elif antipattern == 'bottleneck':
                self._apply_bottleneck_antipattern(graph)
            else:
                self.logger.warning(f"Unknown antipattern: {antipattern}")
    
    def _apply_spof_antipattern(self, graph: Dict):
        """Create single point of failure by making critical app non-redundant"""
        apps = graph['applications']
        
        # Find a critical app with replicas
        critical_apps = [a for a in apps if a['criticality'] in ['HIGH', 'CRITICAL'] and a['replicas'] > 1]
        
        if critical_apps:
            # Make one critical app a SPOF
            spof_app = random.choice(critical_apps)
            spof_app['replicas'] = 1
            spof_app['criticality'] = 'CRITICAL'
            
            # Remove extra runs_on relationships
            runs_on = graph['relationships']['runs_on']
            first_run = next(r for r in runs_on if r['from'] == spof_app['id'])
            graph['relationships']['runs_on'] = [
                r for r in runs_on if r['from'] != spof_app['id'] or r == first_run
            ]
            
            # Make many apps depend on it
            topics_published = [r['to'] for r in graph['relationships']['publishes_to'] 
                              if r['from'] == spof_app['id']]
            
            if topics_published:
                # Add more subscribers to these topics
                other_apps = [a for a in apps if a['id'] != spof_app['id']]
                for _ in range(min(5, len(other_apps))):
                    app = random.choice(other_apps)
                    topic = random.choice(topics_published)
                    
                    # Add subscription if not exists
                    existing = any(r['from'] == app['id'] and r['to'] == topic 
                                 for r in graph['relationships']['subscribes_to'])
                    if not existing:
                        graph['relationships']['subscribes_to'].append({
                            'from': app['id'],
                            'to': topic
                        })
            
            self.logger.info(f"Applied SPOF antipattern to {spof_app['id']}")
    
    def _apply_broker_overload_antipattern(self, graph: Dict):
        """Overload a single broker with most topics"""
        brokers = graph['brokers']
        topics = graph['topics']
        
        if len(brokers) < 2:
            return
        
        # Select one broker to overload
        overloaded_broker = random.choice(brokers)
        
        # Route 80% of topics through this broker
        num_overload = int(len(topics) * 0.8)
        overload_topics = random.sample(topics, num_overload)
        
        # Clear existing routes for these topics
        routes = graph['relationships']['routes']
        graph['relationships']['routes'] = [
            r for r in routes if r['to'] not in [t['id'] for t in overload_topics]
        ]
        
        # Add new routes to overloaded broker
        for topic in overload_topics:
            graph['relationships']['routes'].append({
                'from': overloaded_broker['id'],
                'to': topic['id'],
                'role': 'primary'
            })
        
        self.logger.info(f"Applied broker_overload antipattern to {overloaded_broker['id']}")
    
    def _apply_god_object_antipattern(self, graph: Dict):
        """Create a god object that subscribes to most topics"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Select or create a god app
        god_app = random.choice([a for a in apps if a['type'] in ['CONSUMER', 'PROSUMER']])
        god_app['name'] = f"GodObject_{god_app['id']}"
        
        # Subscribe to 80% of topics
        num_subs = int(len(topics) * 0.8)
        god_topics = random.sample(topics, num_subs)
        
        # Remove existing subscriptions
        subs = graph['relationships']['subscribes_to']
        graph['relationships']['subscribes_to'] = [
            s for s in subs if s['from'] != god_app['id']
        ]
        
        # Add new subscriptions
        for topic in god_topics:
            graph['relationships']['subscribes_to'].append({
                'from': god_app['id'],
                'to': topic['id']
            })
        
        self.logger.info(f"Applied god_object antipattern to {god_app['id']}")
    
    def _apply_single_broker_antipattern(self, graph: Dict):
        """Reduce to a single broker"""
        brokers = graph['brokers']
        
        if len(brokers) <= 1:
            return
        
        # Keep only first broker
        single_broker = brokers[0]
        graph['brokers'] = [single_broker]
        
        # Route all topics through it
        topics = graph['topics']
        graph['relationships']['routes'] = []
        
        for topic in topics:
            graph['relationships']['routes'].append({
                'from': single_broker['id'],
                'to': topic['id'],
                'role': 'primary'
            })
        
        self.logger.info(f"Applied single_broker antipattern - only {single_broker['id']} remains")
    
    def _apply_tight_coupling_antipattern(self, graph: Dict):
        """Create circular dependencies"""
        apps = graph['applications']
        topics = graph['topics']
        
        if len(apps) < 3 or len(topics) < 3:
            return
        
        # Select 3-5 apps for circular dependency
        num_apps = min(5, len(apps), len(topics))
        cycle_apps = random.sample(apps, num_apps)
        cycle_topics = random.sample(topics, num_apps)
        
        # Create cycle: A1 -> T1 -> A2 -> T2 -> A3 -> T3 -> A1
        for i in range(num_apps):
            curr_app = cycle_apps[i]
            curr_topic = cycle_topics[i]
            next_app = cycle_apps[(i + 1) % num_apps]
            
            # Curr app publishes to curr topic
            if not any(r['from'] == curr_app['id'] and r['to'] == curr_topic['id'] 
                      for r in graph['relationships']['publishes_to']):
                graph['relationships']['publishes_to'].append({
                    'from': curr_app['id'],
                    'to': curr_topic['id'],
                    'period_ms': 1000,
                    'msg_size': 512
                })
            
            # Next app subscribes to curr topic
            if not any(r['from'] == next_app['id'] and r['to'] == curr_topic['id'] 
                      for r in graph['relationships']['subscribes_to']):
                graph['relationships']['subscribes_to'].append({
                    'from': next_app['id'],
                    'to': curr_topic['id']
                })
        
        self.logger.info(f"Applied tight_coupling antipattern with {num_apps} apps in cycle")
    
    def _apply_chatty_communication_antipattern(self, graph: Dict):
        """Create excessive message frequency"""
        # Find or create high-frequency publishers
        pubs = [r for r in graph['relationships']['publishes_to']]
        
        # Make 20% of publishers very chatty (>1000 Hz)
        num_chatty = max(1, int(len(pubs) * 0.2))
        chatty_pubs = random.sample(pubs, num_chatty)
        
        for pub in chatty_pubs:
            pub['period_ms'] = 1  # 1000 Hz
            pub['burst_allowed'] = True
            pub['msg_size'] = random.choice([64, 128, 256])  # Small messages
        
        self.logger.info(f"Applied chatty_communication antipattern to {num_chatty} publishers")
    
    def _apply_bottleneck_antipattern(self, graph: Dict):
        """Create a bottleneck topic that all apps use"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Select a topic as bottleneck
        bottleneck_topic = random.choice(topics)
        bottleneck_topic['name'] = f"Bottleneck_{bottleneck_topic['id']}"
        
        # Make 70% of apps publish to it
        num_publishers = int(len(apps) * 0.7)
        publishers = random.sample([a for a in apps if a['type'] in ['PRODUCER', 'PROSUMER']], 
                                   min(num_publishers, len(apps)))
        
        for pub_app in publishers:
            if not any(r['from'] == pub_app['id'] and r['to'] == bottleneck_topic['id'] 
                      for r in graph['relationships']['publishes_to']):
                graph['relationships']['publishes_to'].append({
                    'from': pub_app['id'],
                    'to': bottleneck_topic['id'],
                    'period_ms': 100,
                    'msg_size': 1024
                })
        
        # Make 70% of apps subscribe to it
        num_subscribers = int(len(apps) * 0.7)
        subscribers = random.sample([a for a in apps if a['type'] in ['CONSUMER', 'PROSUMER']], 
                                   min(num_subscribers, len(apps)))
        
        for sub_app in subscribers:
            if not any(r['from'] == sub_app['id'] and r['to'] == bottleneck_topic['id'] 
                      for r in graph['relationships']['subscribes_to']):
                graph['relationships']['subscribes_to'].append({
                    'from': sub_app['id'],
                    'to': bottleneck_topic['id']
                })
        
        self.logger.info(f"Applied bottleneck antipattern to {bottleneck_topic['id']}")
    
    def _ensure_connectivity(self, graph: Dict):
        """Ensure all components are connected"""
        apps = graph['applications']
        topics = graph['topics']
        
        # Find disconnected apps
        connected_apps = set()
        for pub in graph['relationships']['publishes_to']:
            connected_apps.add(pub['from'])
        for sub in graph['relationships']['subscribes_to']:
            connected_apps.add(sub['from'])
        
        disconnected = [a for a in apps if a['id'] not in connected_apps]
        
        # Connect disconnected apps
        for app in disconnected:
            topic = random.choice(topics)
            
            if app['type'] in ['PRODUCER', 'PROSUMER']:
                graph['relationships']['publishes_to'].append({
                    'from': app['id'],
                    'to': topic['id'],
                    'period_ms': 1000,
                    'msg_size': 512
                })
            
            if app['type'] in ['CONSUMER', 'PROSUMER']:
                topic = random.choice(topics)
                graph['relationships']['subscribes_to'].append({
                    'from': app['id'],
                    'to': topic['id']
                })
    
    def _calculate_derived_metrics(self, graph: Dict):
        """Calculate and add derived metrics"""
        # Calculate topic fanout/fanin
        topic_publishers = defaultdict(int)
        topic_subscribers = defaultdict(int)
        
        for pub in graph['relationships']['publishes_to']:
            topic_publishers[pub['to']] += 1
        
        for sub in graph['relationships']['subscribes_to']:
            topic_subscribers[sub['from']] += 1
        
        for topic in graph['topics']:
            topic['num_publishers'] = topic_publishers[topic['id']]
            topic['num_subscribers'] = topic_subscribers[topic['id']]
            topic['fanout'] = topic['num_subscribers']
        
        # Calculate app connectivity
        app_publishes = defaultdict(int)
        app_subscribes = defaultdict(int)
        
        for pub in graph['relationships']['publishes_to']:
            app_publishes[pub['from']] += 1
        
        for sub in graph['relationships']['subscribes_to']:
            app_subscribes[sub['from']] += 1
        
        for app in graph['applications']:
            app['num_topics_published'] = app_publishes[app['id']]
            app['num_topics_subscribed'] = app_subscribes[app['id']]
