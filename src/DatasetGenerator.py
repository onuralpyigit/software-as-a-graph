"""
Multi-Scale Dataset Generator for Distributed Publish-Subscribe Systems
Generates realistic datasets for small, medium, and large-scale deployments
"""
import random
import numpy as np
from datetime import datetime
from typing import Dict, List

class DatasetGenerator:
    """Generates realistic pub-sub system datasets at different scales"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Realistic application types for different domains
        self.app_types = {
            'financial': ['PaymentGateway', 'FraudDetector', 'RiskAnalyzer', 'TransactionProcessor', 
                         'AccountManager', 'AuditLogger', 'ComplianceMonitor'],
            'ecommerce': ['OrderProcessor', 'InventoryManager', 'RecommendationEngine', 
                         'CartService', 'PricingEngine', 'ShippingCalculator', 'ReviewService'],
            'iot': ['SensorAggregator', 'DeviceManager', 'AlertProcessor', 'DataCollector',
                   'CommandDispatcher', 'StatusMonitor', 'FirmwareUpdater'],
            'analytics': ['MetricsCollector', 'LogAggregator', 'AnalyticsEngine', 'ReportGenerator',
                         'DataTransformer', 'StreamProcessor', 'MLPipeline'],
            'core': ['AuthService', 'UserProfileService', 'NotificationService', 'ConfigManager',
                    'ServiceRegistry', 'HealthMonitor', 'RateLimiter']
        }
        
        # Topic patterns by domain
        self.topic_patterns = {
            'user': ['profile', 'preferences', 'activity', 'session', 'authentication'],
            'order': ['created', 'updated', 'fulfilled', 'cancelled', 'payment'],
            'inventory': ['stock', 'reserved', 'depleted', 'restocked', 'transfer'],
            'payment': ['initiated', 'processed', 'failed', 'refunded', 'verified'],
            'metrics': ['cpu', 'memory', 'latency', 'throughput', 'errors'],
            'system': ['health', 'config', 'deployment', 'scaling', 'alerts'],
            'device': ['telemetry', 'command', 'status', 'firmware', 'diagnostic']
        }
    
    def generate_small_scale(self) -> Dict:
        """
        Small-scale: Microservices for a startup
        - 3 nodes (development/staging environment)
        - 15 applications
        - 25 topics
        - 1 broker
        """
        config = {
            'num_nodes': 3,
            'num_apps': 15,
            'num_topics': 25,
            'num_brokers': 1,
            'scenario': 'Small Startup Microservices'
        }
        
        # Small scale characteristics
        # - Higher criticality variance (less redundancy)
        # - Simpler topic structure
        # - More concentrated deployments
        
        return self.generate_dataset(config)
    
    def generate_medium_scale(self) -> Dict:
        """
        Medium-scale: E-commerce platform
        - 10 nodes (production cluster)
        - 50 applications
        - 100 topics
        - 3 brokers
        """
        config = {
            'num_nodes': 10,
            'num_apps': 50,
            'num_topics': 100,
            'num_brokers': 3,
            'scenario': 'E-commerce Platform'
        }
        
        # Medium scale characteristics
        # - Balanced criticality distribution
        # - Multi-broker setup for reliability
        # - Clear service boundaries
        
        return self.generate_dataset(config)
    
    def generate_large_scale(self) -> Dict:
        """
        Large-scale: Enterprise IoT platform
        - 25 nodes (multi-region deployment)
        - 150 applications
        - 300 topics
        - 8 brokers
        """
        config = {
            'num_nodes': 25,
            'num_apps': 150,
            'num_topics': 300,
            'num_brokers': 8,
            'scenario': 'Enterprise IoT Platform'
        }
        
        # Large scale characteristics
        # - Regional distribution
        # - High redundancy for critical services
        # - Complex topic hierarchies
        
        return self.generate_dataset(config)
    
    def generate_dataset(self, config: Dict) -> Dict:
        """Generate dataset based on provided configuration"""
        dataset = self._generate_base_structure(config)
        return self._finalize_dataset(dataset, config)
    
    def _generate_base_structure(self, config: Dict) -> Dict:
        """Generate base structure for any scale"""
        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'config': config,
                'version': '2.0',
                'scenario': config.get('scenario', 'Generic Pub-Sub System')
            },
            'nodes': [],
            'applications': [],
            'topics': [],
            'brokers': [],
            'relationships': {
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'runs_on': []
            }
        }
        
        # Generate infrastructure nodes
        dataset['nodes'] = self._generate_nodes(config['num_nodes'])
        
        # Generate brokers
        dataset['brokers'] = self._generate_brokers(config['num_brokers'])
        
        # Generate applications with realistic distribution
        dataset['applications'] = self._generate_applications(config['num_apps'])
        
        # Generate topics with QoS policies
        dataset['topics'] = self._generate_topics(config['num_topics'])
        
        # Generate relationships
        self._generate_relationships(dataset, config)
        
        return dataset
    
    def _generate_nodes(self, num_nodes: int) -> List[Dict]:
        """Generate infrastructure nodes with realistic specs"""
        nodes = []
        
        # Node specifications based on scale
        if num_nodes <= 5:  # Small scale
            cpu_range = (4, 16)
            memory_range = (8, 32)
            bandwidth_range = (1000, 5000)
        elif num_nodes <= 15:  # Medium scale
            cpu_range = (8, 32)
            memory_range = (16, 64)
            bandwidth_range = (5000, 10000)
        else:  # Large scale
            cpu_range = (16, 64)
            memory_range = (32, 128)
            bandwidth_range = (10000, 40000)
        
        for i in range(1, num_nodes + 1):
            node = {
                'id': f'N{i}',
                'name': f'Node{i}',
                'cpu_capacity': float(self.rng.integers(*cpu_range)),
                'memory_gb': float(self.rng.integers(*memory_range)),
                'network_bandwidth_mbps': float(self.rng.integers(*bandwidth_range)),
                'zone': f'zone-{(i-1) % 3 + 1}' if num_nodes > 10 else 'zone-1',
                'region': f'region-{(i-1) // 10 + 1}' if num_nodes > 20 else 'region-1'
            }
            nodes.append(node)
        
        return nodes
    
    def _generate_brokers(self, num_brokers: int) -> List[Dict]:
        """Generate message brokers"""
        brokers = []
        protocols = ['Kafka', 'RabbitMQ', 'MQTT', 'Pulsar', 'NATS']
        
        for i in range(1, num_brokers + 1):
            broker = {
                'id': f'B{i}',
                'name': f'Broker{i}',
                'protocol': protocols[i % len(protocols)] if num_brokers > 3 else 'Kafka',
                'max_throughput_mbps': float(self.rng.integers(5000, 50000)),
                'max_connections': int(self.rng.integers(1000, 10000)),
                'partition_count': int(self.rng.integers(10, 100))
            }
            brokers.append(broker)
        
        return brokers
    
    def _generate_applications(self, num_apps: int) -> List[Dict]:
        """Generate applications with realistic types and criticality"""
        apps = []
        
        # Determine app distribution based on scale
        if num_apps <= 20:
            domains = ['core', 'analytics']
        elif num_apps <= 60:
            domains = ['core', 'ecommerce', 'analytics']
        else:
            domains = ['core', 'ecommerce', 'iot', 'analytics', 'financial']
        
        for i in range(1, num_apps + 1):
            domain = random.choice(domains)
            app_type = random.choice(self.app_types[domain])
            
            # Criticality based on app type
            if app_type in ['AuthService', 'PaymentGateway', 'TransactionProcessor']:
                criticality = self.rng.uniform(0.8, 1.0)
                impact = random.choice(['HIGH', 'VERY_HIGH'])
            elif app_type in ['MetricsCollector', 'LogAggregator', 'HealthMonitor']:
                criticality = self.rng.uniform(0.3, 0.6)
                impact = random.choice(['LOW', 'MEDIUM'])
            else:
                criticality = self.rng.uniform(0.4, 0.8)
                impact = random.choice(['MEDIUM', 'HIGH'])
            
            app = {
                'id': f'A{i}',
                'name': f'{app_type}{i}',
                'criticality_score': float(criticality),
                'business_impact': impact,
                'domain': domain,
                'replicas': int(self.rng.integers(1, 5)) if criticality > 0.7 else 1
            }
            apps.append(app)
        
        return apps
    
    def _generate_topics(self, num_topics: int) -> List[Dict]:
        """Generate topics with realistic QoS policies"""
        topics = []
        
        for i in range(1, num_topics + 1):
            domain = random.choice(list(self.topic_patterns.keys()))
            pattern = random.choice(self.topic_patterns[domain])
            
            # QoS based on domain
            if domain in ['payment', 'order']:
                durability = random.choice(['TRANSIENT', 'PERSISTENT'])
                reliability = 'RELIABLE'
                priority = random.choice(['HIGH', 'URGENT'])
                deadline = float(self.rng.integers(10, 500))
            elif domain in ['metrics', 'device']:
                durability = random.choice(['VOLATILE', 'TRANSIENT_LOCAL'])
                reliability = random.choice(['BEST_EFFORT', 'RELIABLE'])
                priority = random.choice(['LOW', 'MEDIUM'])
                deadline = float(self.rng.integers(100, 2000))
            else:
                durability = random.choice(['VOLATILE', 'TRANSIENT_LOCAL', 'TRANSIENT', 'PERSISTENT'])
                reliability = random.choice(['BEST_EFFORT', 'RELIABLE'])
                priority = random.choice(['LOW', 'MEDIUM', 'HIGH', 'URGENT'])
                deadline = float(self.rng.integers(50, 1000))
            
            qos = {
                'durability': durability,
                'reliability': reliability,
                'transport_priority': priority,
                'deadline_ms': deadline,
                'lifespan_ms': float(self.rng.integers(1000, 60000)),
                'history_depth': int(self.rng.integers(1, 100))
            }
            
            # Calculate criticality
            criticality = self._calculate_qos_criticality(qos)
            
            topic = {
                'id': f'T{i}',
                'name': f'{domain}/{pattern}_{i}',
                'qos': qos,
                'criticality_score': criticality,
                'partition_count': int(self.rng.integers(1, 16))
            }
            topics.append(topic)
        
        return topics
    
    def _calculate_qos_criticality(self, qos: Dict) -> float:
        """Calculate topic criticality based on QoS"""
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
        
        durability_score = scores['durability'][qos['durability']]
        reliability_score = scores['reliability'][qos['reliability']]
        priority_score = scores['transport_priority'][qos['transport_priority']]
        
        # Deadline impact
        deadline_score = max(0, 1 - (qos['deadline_ms'] / 1000))
        
        return float(
            durability_score * 0.3 +
            reliability_score * 0.3 +
            priority_score * 0.2 +
            deadline_score * 0.2
        )
    
    def _generate_relationships(self, dataset: Dict, config: Dict):
        """Generate realistic relationship patterns"""
        apps = dataset['applications']
        topics = dataset['topics']
        nodes = dataset['nodes']
        brokers = dataset['brokers']
        
        # Assign apps to nodes (considering zones for HA)
        for app in apps:
            if app['replicas'] > 1 and len(nodes) > 3:
                # Distribute replicas across zones
                selected_nodes = random.sample(nodes, min(app['replicas'], len(nodes)))
            else:
                selected_nodes = [random.choice(nodes)]
            
            for node in selected_nodes:
                dataset['relationships']['runs_on'].append({
                    'from': app['id'],
                    'to': node['id']
                })
        
        # Generate pub-sub relationships with realistic patterns
        for topic in topics:
            # Number of publishers (typically fewer)
            num_publishers = self.rng.integers(1, min(4, len(apps)))
            publishers = random.sample(apps, num_publishers)
            
            for pub in publishers:
                dataset['relationships']['publishes_to'].append({
                    'from': pub['id'],
                    'to': topic['id'],
                    'msg_size': int(self.rng.integers(100, 10000)),
                    'period_ms': float(self.rng.integers(10, 1000))
                })
            
            # Number of subscribers (typically more)
            num_subscribers = self.rng.integers(1, min(8, len(apps)))
            subscribers = random.sample(apps, num_subscribers)
            
            for sub in subscribers:
                dataset['relationships']['subscribes_to'].append({
                    'from': sub['id'],
                    'to': topic['id']
                })
        
        # Assign topics to brokers
        if brokers:
            for i, topic in enumerate(topics):
                # Primary broker
                primary_broker = brokers[i % len(brokers)]
                dataset['relationships']['routes'].append({
                    'from': primary_broker['id'],
                    'to': topic['id']
                })
                
                # Replica brokers for critical topics
                if topic['criticality_score'] > 0.7 and len(brokers) > 1:
                    replica_broker = random.choice([b for b in brokers if b != primary_broker])
                    dataset['relationships']['routes'].append({
                        'from': replica_broker['id'],
                        'to': topic['id']
                    })
    
    def _finalize_dataset(self, dataset: Dict, config: Dict) -> Dict:
        """Add final touches and validation"""
        # Add statistics
        dataset['metadata']['statistics'] = {
            'total_nodes': len(dataset['nodes']),
            'total_applications': len(dataset['applications']),
            'total_topics': len(dataset['topics']),
            'total_brokers': len(dataset['brokers']),
            'total_publishes': len(dataset['relationships']['publishes_to']),
            'total_subscribes': len(dataset['relationships']['subscribes_to']),
            'avg_publishers_per_topic': len(dataset['relationships']['publishes_to']) / len(dataset['topics']),
            'avg_subscribers_per_topic': len(dataset['relationships']['subscribes_to']) / len(dataset['topics'])
        }
        
        return dataset