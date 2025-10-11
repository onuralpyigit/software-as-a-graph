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

        self.base_metadata = {
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "purpose": "testing_antipatterns"
        }
        
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
        return self._finalize_dataset(dataset)
    
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
    
    def _finalize_dataset(self, dataset: Dict) -> Dict:
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
    
    def generate_single_point_of_failure(self) -> Dict:
        """
        Generate dataset with Single Point of Failure pattern
        - Critical services on single node
        - No replication for critical components
        - Single broker for all critical topics
        """
        dataset = {
            "metadata": {
                **self.base_metadata,
                "scenario": "Single Point of Failure Pattern",
                "description": "System with multiple SPOFs - critical services without redundancy",
                "antipattern": "single_point_of_failure",
                "problems": [
                    "All critical services on single node",
                    "Single broker handling all traffic",
                    "No replica sets for critical applications",
                    "No failover mechanisms"
                ]
            },
            "nodes": [
                {
                    "id": "N1",
                    "name": "critical-node-single",
                    "cpu_capacity": 64,
                    "memory_gb": 128,
                    "network_bandwidth_mbps": 10000,
                    "zone": "single-zone",
                    "datacenter": "primary-dc",
                    "type": "CRITICAL",
                    "status": "active"
                },
                {
                    "id": "N2",
                    "name": "worker-node-1",
                    "cpu_capacity": 16,
                    "memory_gb": 32,
                    "network_bandwidth_mbps": 1000,
                    "zone": "single-zone",
                    "datacenter": "primary-dc",
                    "type": "WORKER",
                    "status": "active"
                },
                {
                    "id": "N3",
                    "name": "worker-node-2",
                    "cpu_capacity": 16,
                    "memory_gb": 32,
                    "network_bandwidth_mbps": 1000,
                    "zone": "single-zone",
                    "datacenter": "primary-dc",
                    "type": "WORKER",
                    "status": "active"
                }
            ],
            "brokers": [
                {
                    "id": "B1",
                    "name": "single-broker",
                    "type": "KAFKA",
                    "max_topics": 1000,
                    "max_connections": 10000,
                    "max_throughput_mbps": 5000,
                    "replication_factor": 1,  # No replication!
                    "partition_count": 1,     # Single partition!
                    "retention_hours": 24,
                    "version": "3.5.0"
                }
            ],
            "applications": [],
            "topics": [],
            "relationships": {
                "runs_on": [],
                "publishes_to": [],
                "subscribes_to": [],
                "routes": []
            }
        }
        
        # Add critical applications - all on single node
        critical_services = [
            "payment-processor", "auth-service", "order-manager",
            "inventory-controller", "user-database", "session-manager",
            "api-gateway", "notification-hub", "billing-engine", "audit-logger"
        ]
        
        for i, service in enumerate(critical_services, 1):
            app = {
                "id": f"A{i}",
                "name": f"{service}-critical",
                "criticality_score": 0.95,  # Very critical
                "replicas": 1,  # No replicas!
                "type": "CRITICAL",
                "owner": "core-team",
                "version": "1.0.0",
                "single_instance": True  # Flag for SPOF
            }
            dataset["applications"].append(app)
            
            # All critical apps run on single node
            dataset["relationships"]["runs_on"].append({
                "from": f"A{i}",
                "to": "N1"  # All on same node!
            })
        
        # Add non-critical apps distributed on worker nodes
        for i in range(11, 21):
            app = {
                "id": f"A{i}",
                "name": f"worker-service-{i}",
                "criticality_score": 0.3,
                "replicas": 2,
                "type": "WORKER",
                "owner": "app-team",
                "version": "1.0.0"
            }
            dataset["applications"].append(app)
            
            # Distribute on worker nodes
            dataset["relationships"]["runs_on"].append({
                "from": f"A{i}",
                "to": f"N{2 if i % 2 == 0 else 3}"
            })
        
        # Add topics - all routed through single broker
        critical_topics = [
            "payments/process", "auth/validate", "orders/create",
            "inventory/update", "users/profile", "sessions/manage",
            "notifications/send", "billing/charge", "audit/log", "system/health"
        ]
        
        for i, topic_name in enumerate(critical_topics, 1):
            topic = {
                "id": f"T{i}",
                "name": topic_name,
                "qos": {
                    "durability": "PERSISTENT",
                    "reliability": "RELIABLE",
                    "transport_priority": "URGENT",
                    "deadline_ms": 50,
                    "lifespan_ms": 86400000,
                    "history_depth": 1000
                },
                "criticality_score": 0.9,
                "partition_count": 1,  # Single partition!
                "message_pattern": "EVENT_DRIVEN"
            }
            dataset["topics"].append(topic)
            
            # All topics through single broker
            dataset["relationships"]["routes"].append({
                "from": "B1",
                "to": f"T{i}"
            })
        
        # Create pub/sub relationships
        for i in range(1, 11):
            # Each critical app publishes to its topic
            dataset["relationships"]["publishes_to"].append({
                "from": f"A{i}",
                "to": f"T{i}",
                "msg_size": 1024,
                "period_ms": 100
            })
            
            # Other critical apps subscribe (creating dependencies)
            for j in range(1, 11):
                if i != j and random.random() < 0.3:
                    dataset["relationships"]["subscribes_to"].append({
                        "from": f"A{j}",
                        "to": f"T{i}"
                    })
        
        # Single broker runs on critical node (another SPOF!)
        dataset["relationships"]["runs_on"].append({
            "from": "B1",
            "to": "N1"
        })
        
        return dataset
    
    def generate_god_topic_pattern(self) -> Dict:
        """
        Generate dataset with God Topic pattern
        - One or few topics handling everything
        - All services publish/subscribe to same topics
        - No topic specialization
        """
        dataset = {
            "metadata": {
                **self.base_metadata,
                "scenario": "God Topic Anti-Pattern",
                "description": "System where few topics handle all communication - no separation of concerns",
                "antipattern": "god_topic",
                "problems": [
                    "Massive fan-out on single topic",
                    "Mixed message types in same topic",
                    "No domain separation",
                    "Performance bottleneck"
                ]
            },
            "nodes": [],
            "brokers": [],
            "applications": [],
            "topics": [],
            "relationships": {
                "runs_on": [],
                "publishes_to": [],
                "subscribes_to": [],
                "routes": []
            }
        }

        # Generate infrastructure nodes
        dataset['nodes'] = self._generate_nodes(5)
        
        # Generate brokers
        dataset['brokers'] = self._generate_brokers(3)
        
        # Create the "God Topics"
        god_topics = [
            {
                "id": "T1",
                "name": "GLOBAL_EVENT_BUS",  # The main god topic
                "qos": {
                    "durability": "PERSISTENT",
                    "reliability": "RELIABLE",
                    "transport_priority": "HIGH",
                    "deadline_ms": 100,
                    "lifespan_ms": 86400000,
                    "history_depth": 10000
                },
                "criticality_score": 0.99,  # Everything depends on it!
                "partition_count": 50,  # Heavily partitioned due to load
                "message_pattern": "MIXED",  # All patterns mixed
                "god_topic": True
            },
            {
                "id": "T2",
                "name": "SYSTEM_COMMANDS",  # Another god topic
                "qos": {
                    "durability": "PERSISTENT",
                    "reliability": "RELIABLE",
                    "transport_priority": "URGENT",
                    "deadline_ms": 50,
                    "lifespan_ms": 86400000,
                    "history_depth": 5000
                },
                "criticality_score": 0.95,
                "partition_count": 30,
                "message_pattern": "MIXED",
                "god_topic": True
            }
        ]
        
        dataset["topics"] = god_topics
        
        # Add many other underutilized topics (showing the imbalance)
        for i in range(3, 21):
            topic = {
                "id": f"T{i}",
                "name": f"specific/topic/{i}",
                "qos": {
                    "durability": "VOLATILE",
                    "reliability": "BEST_EFFORT",
                    "transport_priority": "LOW",
                    "deadline_ms": 5000,
                    "lifespan_ms": 10000,
                    "history_depth": 10
                },
                "criticality_score": 0.2,
                "partition_count": 1,
                "message_pattern": "EVENT_DRIVEN"
            }
            dataset["topics"].append(topic)
        
        # Create 50 applications - ALL use the god topics
        services = [
            "user", "order", "payment", "inventory", "shipping",
            "notification", "analytics", "reporting", "billing", "customer"
        ]
        
        app_id = 1
        for service in services:
            for instance in range(1, 6):  # 5 instances of each
                app = {
                    "id": f"A{app_id}",
                    "name": f"{service}-service-{instance}",
                    "criticality_score": random.uniform(0.5, 0.9),
                    "replicas": 2,
                    "type": "PROSUMER",
                    "owner": f"{service}-team",
                    "version": "1.0.0"
                }
                dataset["applications"].append(app)
                
                # Distribute across nodes
                dataset["relationships"]["runs_on"].append({
                    "from": f"A{app_id}",
                    "to": f"N{(app_id % 5) + 1}"
                })
                
                # EVERYONE publishes to god topics!
                dataset["relationships"]["publishes_to"].append({
                    "from": f"A{app_id}",
                    "to": "T1",  # God topic
                    "msg_size": random.randint(512, 4096),
                    "period_ms": random.randint(10, 1000)
                })
                
                # EVERYONE subscribes to god topics!
                dataset["relationships"]["subscribes_to"].append({
                    "from": f"A{app_id}",
                    "to": "T1"  # God topic
                })
                
                # Some also use the command god topic
                if random.random() < 0.6:
                    dataset["relationships"]["publishes_to"].append({
                        "from": f"A{app_id}",
                        "to": "T2",
                        "msg_size": random.randint(256, 1024),
                        "period_ms": random.randint(100, 5000)
                    })
                    dataset["relationships"]["subscribes_to"].append({
                        "from": f"A{app_id}",
                        "to": "T2"
                    })
                
                # Rarely use specific topics (showing the problem)
                if random.random() < 0.1:
                    specific_topic = f"T{random.randint(3, 20)}"
                    dataset["relationships"]["publishes_to"].append({
                        "from": f"A{app_id}",
                        "to": specific_topic,
                        "msg_size": 256,
                        "period_ms": 10000
                    })
                
                app_id += 1
        
        # Route all topics through brokers
        for topic in dataset["topics"]:
            if topic["id"] in ["T1", "T2"]:
                # God topics need all brokers
                for broker_id in range(1, 4):
                    dataset["relationships"]["routes"].append({
                        "from": f"B{broker_id}",
                        "to": topic["id"]
                    })
            else:
                # Other topics randomly assigned
                dataset["relationships"]["routes"].append({
                    "from": f"B{random.randint(1, 3)}",
                    "to": topic["id"]
                })
        
        # Place brokers on nodes
        for i in range(1, 4):
            dataset["relationships"]["runs_on"].append({
                "from": f"B{i}",
                "to": f"N{i}"
            })
        
        return dataset
    
    def generate_circular_dependencies(self) -> Dict:
        """
        Generate dataset with Circular Dependencies
        - Services depend on each other in cycles
        - A -> B -> C -> A patterns
        - Deadlock-prone architecture
        """
        dataset = {
            "metadata": {
                **self.base_metadata,
                "scenario": "Circular Dependencies Anti-Pattern",
                "description": "System with multiple circular dependency chains causing potential deadlocks",
                "antipattern": "circular_dependencies",
                "problems": [
                    "Circular dependency chains",
                    "Potential deadlocks",
                    "Difficult to maintain and update",
                    "Cascading failures"
                ]
            },
            "nodes": self._generate_nodes(4),
            "brokers": self._generate_brokers(2),
            "applications": [],
            "topics": [],
            "relationships": {
                "runs_on": [],
                "publishes_to": [],
                "subscribes_to": [],
                "routes": []
            }
        }
        
        # Create circular dependency chains
        circular_chains = [
            # Chain 1: Order -> Payment -> Inventory -> Order
            [("A1", "order-service"), ("A2", "payment-service"), ("A3", "inventory-service")],
            
            # Chain 2: User -> Session -> Auth -> User
            [("A4", "user-service"), ("A5", "session-service"), ("A6", "auth-service")],
            
            # Chain 3: Shipping -> Tracking -> Notification -> Shipping
            [("A7", "shipping-service"), ("A8", "tracking-service"), ("A9", "notification-service")],
            
            # Chain 4: Analytics -> Reporting -> Billing -> Analytics
            [("A10", "analytics-service"), ("A11", "reporting-service"), ("A12", "billing-service")],
            
            # Chain 5: Complex multi-service cycle
            [("A13", "api-gateway"), ("A14", "cache-service"), ("A15", "database-service"), 
             ("A16", "queue-service"), ("A17", "worker-service")]
        ]
        
        topic_id = 1
        
        for chain_idx, chain in enumerate(circular_chains):
            chain_topics = []
            
            # Create applications in the chain
            for app_id, app_name in chain:
                app = {
                    "id": app_id,
                    "name": app_name,
                    "criticality_score": 0.7 + (chain_idx * 0.05),
                    "replicas": 2,
                    "type": "PROSUMER",
                    "owner": "platform-team",
                    "version": "1.0.0",
                    "circular_dependency_chain": chain_idx + 1
                }
                dataset["applications"].append(app)
                
                # Distribute across nodes
                node_id = ((int(app_id[1:]) - 1) % 4) + 1
                dataset["relationships"]["runs_on"].append({
                    "from": app_id,
                    "to": f"N{node_id}"
                })
                
                # Create topic for this service
                topic = {
                    "id": f"T{topic_id}",
                    "name": f"{app_name}/events",
                    "qos": {
                        "durability": "PERSISTENT",
                        "reliability": "RELIABLE",
                        "transport_priority": "HIGH",
                        "deadline_ms": 200,
                        "lifespan_ms": 3600000,
                        "history_depth": 100
                    },
                    "criticality_score": 0.7,
                    "partition_count": 3,
                    "message_pattern": "EVENT_DRIVEN"
                }
                dataset["topics"].append(topic)
                chain_topics.append((app_id, f"T{topic_id}"))
                
                # Route through brokers
                dataset["relationships"]["routes"].append({
                    "from": f"B{(topic_id % 2) + 1}",
                    "to": f"T{topic_id}"
                })
                
                topic_id += 1
            
            # Create circular dependencies
            for i in range(len(chain_topics)):
                current_app, current_topic = chain_topics[i]
                next_app, next_topic = chain_topics[(i + 1) % len(chain_topics)]
                
                # Current app publishes to its topic
                dataset["relationships"]["publishes_to"].append({
                    "from": current_app,
                    "to": current_topic,
                    "msg_size": 1024,
                    "period_ms": 500
                })
                
                # Next app in chain subscribes to current topic
                dataset["relationships"]["subscribes_to"].append({
                    "from": next_app,
                    "to": current_topic
                })
                
                # Create additional cross-dependencies for complexity
                if random.random() < 0.3:
                    other_idx = (i + 2) % len(chain_topics)
                    other_app, other_topic = chain_topics[other_idx]
                    dataset["relationships"]["subscribes_to"].append({
                        "from": other_app,
                        "to": current_topic
                    })
        
        # Add some inter-chain dependencies (making it worse!)
        for i in range(5):
            chain1_idx = random.randint(0, len(circular_chains) - 1)
            chain2_idx = random.randint(0, len(circular_chains) - 1)
            if chain1_idx != chain2_idx:
                app1 = circular_chains[chain1_idx][0][0]
                app2 = circular_chains[chain2_idx][0][0]
                topic_idx = random.randint(1, topic_id - 1)
                
                dataset["relationships"]["subscribes_to"].append({
                    "from": app1,
                    "to": f"T{topic_idx}"
                })
        
        # Place brokers
        for i in range(1, 3):
            dataset["relationships"]["runs_on"].append({
                "from": f"B{i}",
                "to": f"N{i}"
            })
        
        return dataset
    
    def generate_chatty_communication(self) -> Dict:
        """
        Generate dataset with Chatty Communication pattern
        - Excessive small messages
        - High frequency polling
        - No batching or aggregation
        """
        dataset = {
            "metadata": {
                **self.base_metadata,
                "scenario": "Chatty Communication Anti-Pattern",
                "description": "System with excessive small, frequent messages causing network congestion",
                "antipattern": "chatty_communication",
                "problems": [
                    "Excessive network overhead",
                    "High message rates with small payloads",
                    "No message batching",
                    "Polling instead of event-driven"
                ]
            },
            "nodes": self._generate_nodes(6),
            "brokers": self._generate_brokers(4),
            "applications": [],
            "topics": [],
            "relationships": {
                "runs_on": [],
                "publishes_to": [],
                "subscribes_to": [],
                "routes": []
            }
        }
        
        # Create chatty services
        chatty_patterns = [
            {
                "name": "heartbeat",
                "count": 10,
                "msg_size": 32,  # Tiny messages
                "period_ms": 100,  # Very frequent
                "pattern": "PERIODIC"
            },
            {
                "name": "status-poller",
                "count": 8,
                "msg_size": 64,
                "period_ms": 50,  # Excessive polling
                "pattern": "POLLING"
            },
            {
                "name": "metric-reporter",
                "count": 12,
                "msg_size": 128,
                "period_ms": 200,
                "pattern": "PERIODIC"
            },
            {
                "name": "health-checker",
                "count": 15,
                "msg_size": 48,
                "period_ms": 250,
                "pattern": "PERIODIC"
            },
            {
                "name": "sync-service",
                "count": 6,
                "msg_size": 256,
                "period_ms": 10,  # Extremely chatty
                "pattern": "BURST"
            }
        ]
        
        app_id = 1
        topic_id = 1
        
        for pattern in chatty_patterns:
            for i in range(pattern["count"]):
                # Create application
                app = {
                    "id": f"A{app_id}",
                    "name": f"{pattern['name']}-{i+1}",
                    "criticality_score": 0.3,  # Usually not critical
                    "replicas": 1,
                    "type": "PRODUCER" if pattern["name"] != "sync-service" else "PROSUMER",
                    "owner": "monitoring-team",
                    "version": "1.0.0",
                    "chatty": True,
                    "message_overhead_ratio": 0.9  # 90% overhead
                }
                dataset["applications"].append(app)
                
                # Distribute across nodes
                dataset["relationships"]["runs_on"].append({
                    "from": f"A{app_id}",
                    "to": f"N{((app_id - 1) % 6) + 1}"
                })
                
                # Create multiple topics per service (fragmentation)
                for j in range(3):  # Each service uses 3 topics
                    topic = {
                        "id": f"T{topic_id}",
                        "name": f"{pattern['name']}/{i+1}/channel{j}",
                        "qos": {
                            "durability": "VOLATILE",
                            "reliability": "BEST_EFFORT",
                            "transport_priority": "LOW",
                            "deadline_ms": 1000,
                            "lifespan_ms": 1000,  # Very short
                            "history_depth": 5
                        },
                        "criticality_score": 0.2,
                        "partition_count": 1,
                        "message_pattern": pattern["pattern"],
                        "avg_message_size": pattern["msg_size"],
                        "message_rate": 1000 / pattern["period_ms"]
                    }
                    dataset["topics"].append(topic)
                    
                    # Publish with high frequency, small messages
                    dataset["relationships"]["publishes_to"].append({
                        "from": f"A{app_id}",
                        "to": f"T{topic_id}",
                        "msg_size": pattern["msg_size"],
                        "period_ms": pattern["period_ms"],
                        "chatty": True
                    })
                    
                    # Route through brokers
                    dataset["relationships"]["routes"].append({
                        "from": f"B{((topic_id - 1) % 4) + 1}",
                        "to": f"T{topic_id}"
                    })
                    
                    topic_id += 1
                
                app_id += 1
        
        # Add subscriber services that have to handle all this chatter
        aggregator_services = ["log-aggregator", "metric-aggregator", "event-processor", "data-lake"]
        
        for service in aggregator_services:
            app = {
                "id": f"A{app_id}",
                "name": service,
                "criticality_score": 0.6,
                "replicas": 3,  # Need more replicas due to load
                "type": "CONSUMER",
                "owner": "data-team",
                "version": "1.0.0",
                "overwhelmed": True  # Flag indicating overload
            }
            dataset["applications"].append(app)
            
            dataset["relationships"]["runs_on"].append({
                "from": f"A{app_id}",
                "to": f"N{((app_id - 1) % 6) + 1}"
            })
            
            # Subscribe to many chatty topics
            for tid in range(1, min(topic_id, 50)):  # Subscribe to first 50 topics
                if random.random() < 0.7:  # Subscribe to 70% of topics
                    dataset["relationships"]["subscribes_to"].append({
                        "from": f"A{app_id}",
                        "to": f"T{tid}"
                    })
            
            app_id += 1
        
        # Place brokers
        for i in range(1, 5):
            dataset["relationships"]["runs_on"].append({
                "from": f"B{i}",
                "to": f"N{(i % 6) + 1}"
            })
        
        return dataset
    
    def generate_hidden_coupling(self) -> Dict:
        """
        Generate dataset with Hidden Coupling pattern
        - Implicit dependencies through shared data
        - Temporal coupling
        - Hidden state dependencies
        """
        dataset = {
            "metadata": {
                **self.base_metadata,
                "scenario": "Hidden Coupling Anti-Pattern",
                "description": "System with implicit dependencies and hidden coupling between services",
                "antipattern": "hidden_coupling",
                "problems": [
                    "Implicit dependencies not visible in topology",
                    "Shared mutable state",
                    "Temporal coupling between services",
                    "Race conditions"
                ]
            },
            "nodes": self._generate_nodes(5),
            "brokers": self._generate_brokers(3),
            "applications": [],
            "topics": [],
            "relationships": {
                "runs_on": [],
                "publishes_to": [],
                "subscribes_to": [],
                "routes": []
            }
        }
        
        # Create services with hidden dependencies
        
        # Group 1: Services that share hidden state through cache
        cache_dependent_services = [
            ("A1", "user-profile-service"),
            ("A2", "preference-service"),
            ("A3", "recommendation-service"),
            ("A4", "personalization-service")
        ]
        
        # Group 2: Services with temporal coupling (order matters)
        temporally_coupled = [
            ("A5", "data-preprocessor"),
            ("A6", "feature-extractor"),
            ("A7", "model-trainer"),
            ("A8", "prediction-service")
        ]
        
        # Group 3: Services with hidden shared database
        db_coupled = [
            ("A9", "order-writer"),
            ("A10", "order-reader"),
            ("A11", "order-validator"),
            ("A12", "order-archiver")
        ]
        
        # Shared hidden state topics (not obvious from names)
        hidden_topics = [
            {
                "id": "T1",
                "name": "cache/invalidation",  # Hidden coupling point
                "hidden": True,
                "coupling_type": "cache_coherence"
            },
            {
                "id": "T2",
                "name": "internal/state/sync",  # Another hidden coupling
                "hidden": True,
                "coupling_type": "state_synchronization"
            },
            {
                "id": "T3",
                "name": "system/timestamps",  # Temporal coupling
                "hidden": True,
                "coupling_type": "temporal_dependency"
            }
        ]
        
        # Add all hidden topics
        for ht in hidden_topics:
            topic = {
                "id": ht["id"],
                "name": ht["name"],
                "qos": {
                    "durability": "TRANSIENT_LOCAL",
                    "reliability": "RELIABLE",
                    "transport_priority": "HIGH",
                    "deadline_ms": 50,
                    "lifespan_ms": 60000,
                    "history_depth": 100
                },
                "criticality_score": 0.8,  # Critical but not obvious
                "partition_count": 1,  # Single partition causes ordering issues
                "message_pattern": "EVENT_DRIVEN",
                "hidden_coupling": ht["coupling_type"]
            }
            dataset["topics"].append(topic)
            
            # Route through brokers
            dataset["relationships"]["routes"].append({
                "from": "B1",
                "to": ht["id"]
            })
        
        topic_id = 4
        
        # Process cache-dependent services
        for app_id, app_name in cache_dependent_services:
            app = {
                "id": app_id,
                "name": app_name,
                "criticality_score": 0.7,
                "replicas": 2,
                "type": "PROSUMER",
                "owner": "platform-team",
                "version": "1.0.0",
                "hidden_dependency": "shared_cache"
            }
            dataset["applications"].append(app)
            
            dataset["relationships"]["runs_on"].append({
                "from": app_id,
                "to": f"N{(int(app_id[1:]) % 5) + 1}"
            })
            
            # All secretly depend on cache invalidation
            dataset["relationships"]["subscribes_to"].append({
                "from": app_id,
                "to": "T1"  # Hidden coupling through cache
            })

        return dataset