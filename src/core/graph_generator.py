"""
Graph Generator

Generates realistic, complex pub-sub software system graphs at multiple scales
with domain-specific patterns and anti-patterns.

Key Features:
- 6 scale presets (tiny to extreme)
- 8 domain scenarios with realistic data flow patterns
- 8 sophisticated anti-patterns for validation testing
- QoS-aware topic generation
- Zone/region-aware deployment
- Realistic message flow topologies
- Semantic topic-application matching
- Hierarchical data flow patterns (edge→gateway→cloud)
- Network topology patterns (star, ring, hierarchical, mesh)
- Integration with unified DEPENDS_ON model

Research Applications:
- Criticality analysis validation
- Anti-pattern detection testing
- Scalability benchmarking
- Failure simulation scenarios
"""

import random
import math
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum
import logging

from .graph_model import (
    QoSReliability, QoSDurability, QosTransportPriority, DependencyType
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GraphConfig:
    """
    Configuration for graph generation.
    
    Attributes:
        scale: Size preset (tiny, small, medium, large, xlarge, extreme)
        scenario: Domain scenario for realistic patterns
        num_nodes: Number of infrastructure nodes (overrides scale if set)
        num_applications: Number of applications (overrides scale if set)
        num_topics: Number of topics (overrides scale if set)
        num_brokers: Number of brokers (overrides scale if set)
        edge_density: Connection density 0.0-1.0 (affects pub/sub connections)
        antipatterns: List of anti-patterns to inject
        seed: Random seed for reproducibility
    """
    scale: str = 'medium'
    scenario: str = 'generic'
    num_nodes: Optional[int] = None
    num_applications: Optional[int] = None
    num_topics: Optional[int] = None
    num_brokers: Optional[int] = None
    edge_density: float = 0.3
    antipatterns: List[str] = field(default_factory=list)
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        valid_scales = ['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme']
        if self.scale not in valid_scales:
            raise ValueError(f"Invalid scale '{self.scale}'. Must be one of {valid_scales}")
        
        if not 0.0 <= self.edge_density <= 1.0:
            raise ValueError(f"edge_density must be between 0.0 and 1.0")


class AntiPatternType(Enum):
    """Types of architectural anti-patterns that can be injected"""
    SPOF = "spof"                           # Single Point of Failure
    GOD_TOPIC = "god_topic"                 # Topic with too many connections
    BROKER_OVERLOAD = "broker_overload"     # Broker handling too many topics
    TIGHT_COUPLING = "tight_coupling"       # Excessive app-to-app dependencies
    CHATTY_COMMUNICATION = "chatty"         # Too many small messages
    CIRCULAR_DEPENDENCY = "circular"        # Circular dependency chains
    BOTTLENECK = "bottleneck"               # Infrastructure bottleneck
    HIDDEN_COUPLING = "hidden_coupling"     # Implicit dependencies via shared topics


class NetworkTopology(Enum):
    """Network topology patterns for infrastructure connectivity"""
    FULL_MESH = "full_mesh"      # All nodes connected (unrealistic for large systems)
    HIERARCHICAL = "hierarchical" # Tree structure: edge → compute → cloud
    STAR = "star"                 # Hub-and-spoke: central nodes connect to others
    RING = "ring"                 # Ring topology with redundant connections
    HYBRID = "hybrid"             # Combination of patterns (most realistic)


class DataFlowPattern(Enum):
    """Data flow patterns in pub-sub systems"""
    RANDOM = "random"             # Random connections (current behavior)
    PIPELINE = "pipeline"         # Linear: producer → processor → consumer
    FANOUT = "fanout"             # One producer → many consumers
    FANIN = "fanin"               # Many producers → one aggregator
    HIERARCHICAL = "hierarchical" # Tiered: edge → gateway → cloud
    MIXED = "mixed"               # Realistic mix of all patterns


# =============================================================================
# Graph Generator
# =============================================================================

class GraphGenerator:
    """
    Enhanced Graph Generator for realistic DDS/pub-sub system patterns.
    
    Generates graphs that model real-world distributed systems with:
    - Realistic application roles and communication patterns
    - Domain-specific topic hierarchies
    - QoS policies matching use cases
    - Infrastructure deployment patterns
    - Optional anti-patterns for testing detection algorithms
    """
    
    # =========================================================================
    # Scale Presets
    # =========================================================================
    
    SCALES = {
        'tiny': {
            'nodes': 3,
            'apps': 6,
            'topics': 4,
            'brokers': 1,
            'description': 'Minimal system for unit testing'
        },
        'small': {
            'nodes': 5,
            'apps': 15,
            'topics': 10,
            'brokers': 2,
            'description': 'Small development system'
        },
        'medium': {
            'nodes': 15,
            'apps': 50,
            'topics': 30,
            'brokers': 3,
            'description': 'Typical production system'
        },
        'large': {
            'nodes': 50,
            'apps': 200,
            'topics': 120,
            'brokers': 8,
            'description': 'Large enterprise system'
        },
        'xlarge': {
            'nodes': 100,
            'apps': 500,
            'topics': 300,
            'brokers': 15,
            'description': 'Very large distributed system'
        },
        'extreme': {
            'nodes': 250,
            'apps': 1500,
            'topics': 800,
            'brokers': 30,
            'description': 'Extreme scale for performance testing'
        }
    }
    
    # =========================================================================
    # Domain Scenarios
    # =========================================================================
    
    SCENARIOS = {
        'generic': {
            'description': 'Generic microservices system',
            'app_types': [
                ('Service', 'PROSUMER', 0.15),
                ('Gateway', 'PROSUMER', 0.10),
                ('Processor', 'PROSUMER', 0.15),
                ('Handler', 'CONSUMER', 0.12),
                ('Publisher', 'PRODUCER', 0.10),
                ('Aggregator', 'PROSUMER', 0.10),
                ('Monitor', 'CONSUMER', 0.08),
                ('Logger', 'CONSUMER', 0.08),
                ('Validator', 'PROSUMER', 0.06),
                ('Router', 'PROSUMER', 0.06)
            ],
            'topic_patterns': [
                ('events/{domain}', 0.25),
                ('commands/{domain}', 0.20),
                ('data/{domain}/stream', 0.20),
                ('notifications/{type}', 0.15),
                ('metrics/{source}', 0.10),
                ('logs/{level}', 0.10)
            ],
            'qos_profile': 'balanced'
        },
        'iot': {
            'description': 'IoT sensor network',
            'app_types': [
                ('SensorCollector', 'PRODUCER', 0.20),
                ('DeviceManager', 'PROSUMER', 0.12),
                ('TelemetryAggregator', 'PROSUMER', 0.12),
                ('EdgeGateway', 'PROSUMER', 0.10),
                ('CommandDispatcher', 'PRODUCER', 0.08),
                ('AlertProcessor', 'CONSUMER', 0.10),
                ('DataForwarder', 'PROSUMER', 0.08),
                ('StatusMonitor', 'CONSUMER', 0.08),
                ('FirmwareUpdater', 'PRODUCER', 0.06),
                ('DiagnosticsEngine', 'PROSUMER', 0.06)
            ],
            'topic_patterns': [
                ('sensors/{type}/{device_id}/data', 0.25),
                ('telemetry/{zone}/aggregate', 0.15),
                ('commands/{device_type}/control', 0.15),
                ('alerts/{severity}', 0.12),
                ('status/{device_id}', 0.12),
                ('firmware/{device_type}/updates', 0.08),
                ('diagnostics/{zone}', 0.08),
                ('config/{device_type}', 0.05)
            ],
            'qos_profile': 'iot'
        },
        'financial': {
            'description': 'Financial trading system',
            'app_types': [
                ('MarketDataFeed', 'PRODUCER', 0.12),
                ('OrderProcessor', 'PROSUMER', 0.15),
                ('RiskEngine', 'CONSUMER', 0.12),
                ('TradeExecutor', 'PROSUMER', 0.10),
                ('PositionTracker', 'CONSUMER', 0.10),
                ('PricingEngine', 'PROSUMER', 0.10),
                ('MatchingEngine', 'PROSUMER', 0.08),
                ('ComplianceMonitor', 'CONSUMER', 0.08),
                ('AuditLogger', 'CONSUMER', 0.08),
                ('SettlementService', 'PROSUMER', 0.07)
            ],
            'topic_patterns': [
                ('market/{exchange}/{symbol}/quotes', 0.20),
                ('orders/{desk}/new', 0.15),
                ('orders/{desk}/status', 0.12),
                ('trades/{desk}/executed', 0.12),
                ('risk/{portfolio}/exposure', 0.10),
                ('positions/{account}/updates', 0.10),
                ('compliance/alerts', 0.08),
                ('settlement/instructions', 0.08),
                ('audit/events', 0.05)
            ],
            'qos_profile': 'financial'
        },
        'ecommerce': {
            'description': 'E-commerce platform',
            'app_types': [
                ('OrderService', 'PROSUMER', 0.15),
                ('InventoryManager', 'PROSUMER', 0.12),
                ('PaymentProcessor', 'PROSUMER', 0.12),
                ('CartService', 'PROSUMER', 0.10),
                ('ProductCatalog', 'PRODUCER', 0.10),
                ('RecommendationEngine', 'PROSUMER', 0.08),
                ('NotificationService', 'CONSUMER', 0.10),
                ('ShippingCalculator', 'PROSUMER', 0.08),
                ('FraudDetector', 'CONSUMER', 0.08),
                ('ReviewAggregator', 'PROSUMER', 0.07)
            ],
            'topic_patterns': [
                ('orders/{region}/created', 0.18),
                ('orders/{region}/status', 0.12),
                ('inventory/{warehouse}/updates', 0.15),
                ('payments/{provider}/events', 0.12),
                ('cart/{session}/updates', 0.10),
                ('products/updates', 0.10),
                ('notifications/{channel}', 0.10),
                ('shipping/rates', 0.08),
                ('fraud/alerts', 0.05)
            ],
            'qos_profile': 'transactional'
        },
        'autonomous_vehicle': {
            'description': 'Autonomous vehicle system (ROS2-like)',
            'app_types': [
                ('SensorFusion', 'PROSUMER', 0.12),
                ('PerceptionNode', 'PROSUMER', 0.15),
                ('PlanningNode', 'PROSUMER', 0.12),
                ('ControlNode', 'CONSUMER', 0.10),
                ('LocalizationNode', 'PROSUMER', 0.10),
                ('MappingNode', 'PROSUMER', 0.08),
                ('NavigationNode', 'PROSUMER', 0.10),
                ('SafetyMonitor', 'CONSUMER', 0.10),
                ('DiagnosticsNode', 'CONSUMER', 0.08),
                ('V2XCommunicator', 'PROSUMER', 0.05)
            ],
            'topic_patterns': [
                ('/sensors/{type}/raw', 0.20),
                ('/perception/objects', 0.15),
                ('/perception/lanes', 0.10),
                ('/localization/pose', 0.12),
                ('/planning/trajectory', 0.12),
                ('/control/commands', 0.10),
                ('/safety/status', 0.08),
                ('/diagnostics/{subsystem}', 0.08),
                ('/v2x/messages', 0.05)
            ],
            'qos_profile': 'realtime'
        },
        'smart_city': {
            'description': 'Smart city infrastructure',
            'app_types': [
                ('TrafficController', 'PROSUMER', 0.15),
                ('EnvironmentSensor', 'PRODUCER', 0.15),
                ('EnergyManager', 'PROSUMER', 0.12),
                ('PublicSafetyMonitor', 'CONSUMER', 0.10),
                ('WasteManager', 'PROSUMER', 0.08),
                ('ParkingService', 'PROSUMER', 0.10),
                ('TransitTracker', 'PROSUMER', 0.10),
                ('EmergencyDispatcher', 'PROSUMER', 0.08),
                ('CitizenNotifier', 'CONSUMER', 0.07),
                ('DataAggregator', 'CONSUMER', 0.05)
            ],
            'topic_patterns': [
                ('traffic/{intersection}/flow', 0.18),
                ('environment/{zone}/air_quality', 0.12),
                ('environment/{zone}/noise', 0.08),
                ('energy/{district}/consumption', 0.12),
                ('transit/{line}/position', 0.12),
                ('parking/{zone}/availability', 0.10),
                ('safety/{district}/incidents', 0.10),
                ('emergency/dispatch', 0.08),
                ('waste/{zone}/levels', 0.05),
                ('notifications/public', 0.05)
            ],
            'qos_profile': 'smart_city'
        },
        'healthcare': {
            'description': 'Healthcare monitoring system',
            'app_types': [
                ('PatientMonitor', 'PRODUCER', 0.18),
                ('VitalSignsAnalyzer', 'PROSUMER', 0.15),
                ('AlertManager', 'PROSUMER', 0.12),
                ('MedicationTracker', 'PROSUMER', 0.10),
                ('NurseStation', 'CONSUMER', 0.10),
                ('PhysicianDashboard', 'CONSUMER', 0.08),
                ('EMRIntegrator', 'PROSUMER', 0.10),
                ('LabResultsProcessor', 'PROSUMER', 0.07),
                ('EquipmentMonitor', 'PRODUCER', 0.05),
                ('ComplianceLogger', 'CONSUMER', 0.05)
            ],
            'topic_patterns': [
                ('vitals/{patient_id}/realtime', 0.20),
                ('vitals/{patient_id}/summary', 0.12),
                ('alerts/{severity}/{unit}', 0.15),
                ('medications/{patient_id}/schedule', 0.10),
                ('lab/{patient_id}/results', 0.10),
                ('equipment/{unit}/status', 0.10),
                ('nursing/{unit}/tasks', 0.08),
                ('emr/{patient_id}/updates', 0.10),
                ('compliance/audit', 0.05)
            ],
            'qos_profile': 'healthcare'
        },
        'gaming': {
            'description': 'Online multiplayer gaming backend',
            'app_types': [
                ('GameServer', 'PROSUMER', 0.18),
                ('MatchMaker', 'PROSUMER', 0.12),
                ('PlayerService', 'PROSUMER', 0.12),
                ('SessionManager', 'PROSUMER', 0.10),
                ('LeaderboardService', 'PROSUMER', 0.08),
                ('ChatService', 'PROSUMER', 0.10),
                ('InventoryService', 'PROSUMER', 0.08),
                ('AnalyticsCollector', 'CONSUMER', 0.08),
                ('AntiCheatEngine', 'CONSUMER', 0.07),
                ('NotificationPusher', 'CONSUMER', 0.07)
            ],
            'topic_patterns': [
                ('game/{session_id}/state', 0.20),
                ('game/{session_id}/events', 0.15),
                ('matchmaking/queue', 0.12),
                ('player/{player_id}/updates', 0.12),
                ('chat/{channel}/messages', 0.12),
                ('leaderboard/{game_mode}', 0.08),
                ('inventory/{player_id}/changes', 0.08),
                ('analytics/events', 0.08),
                ('anticheat/reports', 0.05)
            ],
            'qos_profile': 'gaming'
        }
    }
    
    # =========================================================================
    # QoS Profiles
    # =========================================================================
    
    QOS_PROFILES = {
        'balanced': {
            'reliability_dist': [(QoSReliability.RELIABLE, 0.6), (QoSReliability.BEST_EFFORT, 0.4)],
            'durability_dist': [(QoSDurability.VOLATILE, 0.5), (QoSDurability.TRANSIENT_LOCAL, 0.4), 
                               (QoSDurability.PERSISTENT, 0.1)],
            'deadline_range': (50, 1000),
            'priority_dist': [(QosTransportPriority.MEDIUM, 0.6), (QosTransportPriority.HIGH, 0.3),
                             (QosTransportPriority.URGENT, 0.1)]
        },
        'iot': {
            'reliability_dist': [(QoSReliability.BEST_EFFORT, 0.7), (QoSReliability.RELIABLE, 0.3)],
            'durability_dist': [(QoSDurability.VOLATILE, 0.8), (QoSDurability.TRANSIENT_LOCAL, 0.2)],
            'deadline_range': (100, 5000),
            'priority_dist': [(QosTransportPriority.LOW, 0.3), (QosTransportPriority.MEDIUM, 0.5),
                             (QosTransportPriority.HIGH, 0.2)]
        },
        'financial': {
            'reliability_dist': [(QoSReliability.RELIABLE, 0.95), (QoSReliability.BEST_EFFORT, 0.05)],
            'durability_dist': [(QoSDurability.PERSISTENT, 0.4), (QoSDurability.TRANSIENT_LOCAL, 0.5),
                               (QoSDurability.VOLATILE, 0.1)],
            'deadline_range': (1, 50),
            'priority_dist': [(QosTransportPriority.HIGH, 0.4), (QosTransportPriority.URGENT, 0.4),
                             (QosTransportPriority.MEDIUM, 0.2)]
        },
        'transactional': {
            'reliability_dist': [(QoSReliability.RELIABLE, 0.85), (QoSReliability.BEST_EFFORT, 0.15)],
            'durability_dist': [(QoSDurability.TRANSIENT_LOCAL, 0.5), (QoSDurability.PERSISTENT, 0.3),
                               (QoSDurability.VOLATILE, 0.2)],
            'deadline_range': (100, 2000),
            'priority_dist': [(QosTransportPriority.MEDIUM, 0.5), (QosTransportPriority.HIGH, 0.4),
                             (QosTransportPriority.URGENT, 0.1)]
        },
        'realtime': {
            'reliability_dist': [(QoSReliability.BEST_EFFORT, 0.6), (QoSReliability.RELIABLE, 0.4)],
            'durability_dist': [(QoSDurability.VOLATILE, 0.9), (QoSDurability.TRANSIENT_LOCAL, 0.1)],
            'deadline_range': (5, 100),
            'priority_dist': [(QosTransportPriority.HIGH, 0.5), (QosTransportPriority.URGENT, 0.3),
                             (QosTransportPriority.MEDIUM, 0.2)]
        },
        'smart_city': {
            'reliability_dist': [(QoSReliability.RELIABLE, 0.7), (QoSReliability.BEST_EFFORT, 0.3)],
            'durability_dist': [(QoSDurability.TRANSIENT_LOCAL, 0.5), (QoSDurability.VOLATILE, 0.3),
                               (QoSDurability.PERSISTENT, 0.2)],
            'deadline_range': (100, 5000),
            'priority_dist': [(QosTransportPriority.MEDIUM, 0.5), (QosTransportPriority.HIGH, 0.35),
                             (QosTransportPriority.URGENT, 0.15)]
        },
        'healthcare': {
            'reliability_dist': [(QoSReliability.RELIABLE, 0.9), (QoSReliability.BEST_EFFORT, 0.1)],
            'durability_dist': [(QoSDurability.PERSISTENT, 0.5), (QoSDurability.TRANSIENT_LOCAL, 0.4),
                               (QoSDurability.VOLATILE, 0.1)],
            'deadline_range': (10, 500),
            'priority_dist': [(QosTransportPriority.HIGH, 0.4), (QosTransportPriority.URGENT, 0.35),
                             (QosTransportPriority.MEDIUM, 0.25)]
        },
        'gaming': {
            'reliability_dist': [(QoSReliability.BEST_EFFORT, 0.7), (QoSReliability.RELIABLE, 0.3)],
            'durability_dist': [(QoSDurability.VOLATILE, 0.8), (QoSDurability.TRANSIENT_LOCAL, 0.2)],
            'deadline_range': (16, 100),  # ~60fps to 10fps
            'priority_dist': [(QosTransportPriority.HIGH, 0.5), (QosTransportPriority.MEDIUM, 0.3),
                             (QosTransportPriority.URGENT, 0.2)]
        }
    }
    
    # =========================================================================
    # Semantic Keyword Mappings (App Type → Topic Keywords)
    # =========================================================================

    # Maps application type keywords to relevant topic pattern keywords
    # Used for semantic matching between apps and topics
    SEMANTIC_MAPPINGS = {
        # Generic services
        'Service': ['events', 'data', 'commands', 'status'],
        'Gateway': ['events', 'commands', 'notifications', 'status'],
        'Processor': ['data', 'events', 'stream', 'processing'],
        'Handler': ['events', 'commands', 'notifications'],
        'Publisher': ['events', 'data', 'notifications'],
        'Aggregator': ['data', 'metrics', 'aggregate', 'summary'],
        'Monitor': ['metrics', 'status', 'health', 'diagnostics'],
        'Logger': ['logs', 'audit', 'events', 'debug'],
        'Validator': ['validation', 'events', 'data'],
        'Router': ['commands', 'events', 'routing'],

        # IoT specific
        'SensorCollector': ['sensors', 'telemetry', 'data', 'raw'],
        'DeviceManager': ['devices', 'status', 'commands', 'config'],
        'TelemetryAggregator': ['telemetry', 'aggregate', 'summary'],
        'EdgeGateway': ['sensors', 'telemetry', 'commands', 'edge'],
        'CommandDispatcher': ['commands', 'control', 'dispatch'],
        'AlertProcessor': ['alerts', 'notifications', 'critical'],
        'DataForwarder': ['data', 'forward', 'stream'],
        'StatusMonitor': ['status', 'health', 'monitoring'],
        'FirmwareUpdater': ['firmware', 'updates', 'config'],
        'DiagnosticsEngine': ['diagnostics', 'health', 'debug'],

        # Financial specific
        'MarketDataFeed': ['market', 'quotes', 'prices', 'feed'],
        'OrderProcessor': ['orders', 'new', 'status', 'execution'],
        'RiskEngine': ['risk', 'exposure', 'limits', 'compliance'],
        'TradeExecutor': ['trades', 'execution', 'orders'],
        'PositionTracker': ['positions', 'updates', 'portfolio'],
        'PricingEngine': ['pricing', 'quotes', 'market'],
        'MatchingEngine': ['orders', 'matching', 'execution'],
        'ComplianceMonitor': ['compliance', 'audit', 'alerts'],
        'AuditLogger': ['audit', 'logs', 'events', 'compliance'],
        'SettlementService': ['settlement', 'instructions', 'clearing'],

        # E-commerce specific
        'OrderService': ['orders', 'created', 'status'],
        'InventoryManager': ['inventory', 'updates', 'stock'],
        'PaymentProcessor': ['payments', 'events', 'transactions'],
        'CartService': ['cart', 'updates', 'session'],
        'ProductCatalog': ['products', 'catalog', 'updates'],
        'RecommendationEngine': ['recommendations', 'products', 'user'],
        'NotificationService': ['notifications', 'email', 'sms', 'push'],
        'ShippingCalculator': ['shipping', 'rates', 'delivery'],
        'FraudDetector': ['fraud', 'alerts', 'transactions'],
        'ReviewAggregator': ['reviews', 'ratings', 'products'],

        # Autonomous vehicle specific
        'SensorFusion': ['sensors', 'fusion', 'perception'],
        'PerceptionNode': ['perception', 'objects', 'lanes', 'sensors'],
        'PlanningNode': ['planning', 'trajectory', 'path'],
        'ControlNode': ['control', 'commands', 'actuator'],
        'LocalizationNode': ['localization', 'pose', 'position'],
        'MappingNode': ['mapping', 'map', 'environment'],
        'NavigationNode': ['navigation', 'trajectory', 'route'],
        'SafetyMonitor': ['safety', 'status', 'emergency'],
        'DiagnosticsNode': ['diagnostics', 'health', 'status'],
        'V2XCommunicator': ['v2x', 'messages', 'vehicle'],

        # Smart city specific
        'TrafficController': ['traffic', 'flow', 'intersection'],
        'EnvironmentSensor': ['environment', 'air_quality', 'noise'],
        'EnergyManager': ['energy', 'consumption', 'power'],
        'PublicSafetyMonitor': ['safety', 'incidents', 'emergency'],
        'WasteManager': ['waste', 'levels', 'collection'],
        'ParkingService': ['parking', 'availability', 'zones'],
        'TransitTracker': ['transit', 'position', 'schedule'],
        'EmergencyDispatcher': ['emergency', 'dispatch', 'response'],
        'CitizenNotifier': ['notifications', 'public', 'alerts'],
        'DataAggregator': ['data', 'aggregate', 'analytics'],

        # Healthcare specific
        'PatientMonitor': ['vitals', 'patient', 'realtime'],
        'VitalSignsAnalyzer': ['vitals', 'analysis', 'alerts'],
        'AlertManager': ['alerts', 'notifications', 'critical'],
        'MedicationTracker': ['medications', 'schedule', 'patient'],
        'NurseStation': ['nursing', 'tasks', 'alerts'],
        'PhysicianDashboard': ['patient', 'vitals', 'results'],
        'EMRIntegrator': ['emr', 'updates', 'records'],
        'LabResultsProcessor': ['lab', 'results', 'patient'],
        'EquipmentMonitor': ['equipment', 'status', 'maintenance'],
        'ComplianceLogger': ['compliance', 'audit', 'hipaa'],

        # Gaming specific
        'GameServer': ['game', 'state', 'events', 'session'],
        'MatchMaker': ['matchmaking', 'queue', 'lobby'],
        'PlayerService': ['player', 'updates', 'profile'],
        'SessionManager': ['session', 'game', 'connection'],
        'LeaderboardService': ['leaderboard', 'scores', 'ranking'],
        'ChatService': ['chat', 'messages', 'channel'],
        'InventoryService': ['inventory', 'items', 'player'],
        'AnalyticsCollector': ['analytics', 'events', 'telemetry'],
        'AntiCheatEngine': ['anticheat', 'reports', 'violations'],
        'NotificationPusher': ['notifications', 'push', 'player']
    }

    # Data flow tiers for hierarchical patterns
    DATA_FLOW_TIERS = {
        'edge': {
            'app_keywords': ['Sensor', 'Collector', 'Edge', 'Device', 'Monitor'],
            'publishes_to_tier': 'gateway',
            'tier_level': 0
        },
        'gateway': {
            'app_keywords': ['Gateway', 'Aggregator', 'Forwarder', 'Dispatcher'],
            'publishes_to_tier': 'cloud',
            'tier_level': 1
        },
        'cloud': {
            'app_keywords': ['Processor', 'Engine', 'Service', 'Analyzer', 'Manager'],
            'publishes_to_tier': None,  # Terminal tier
            'tier_level': 2
        }
    }

    # =========================================================================
    # Infrastructure Templates
    # =========================================================================

    NODE_TYPES = {
        'edge': {'prefix': 'edge', 'weight': 0.3, 'capacity_range': (10, 50), 'tier': 0},
        'compute': {'prefix': 'compute', 'weight': 0.4, 'capacity_range': (50, 200), 'tier': 1},
        'cloud': {'prefix': 'cloud', 'weight': 0.2, 'capacity_range': (200, 1000), 'tier': 2},
        'gateway': {'prefix': 'gateway', 'weight': 0.1, 'capacity_range': (100, 500), 'tier': 1}
    }
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def __init__(self, config: GraphConfig):
        """
        Initialize the graph generator.
        
        Args:
            config: GraphConfig instance with generation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        
        # Get scale parameters
        scale_params = self.SCALES.get(config.scale, self.SCALES['medium'])
        
        # Use config overrides or scale defaults
        self.num_nodes = config.num_nodes or scale_params['nodes']
        self.num_apps = config.num_applications or scale_params['apps']
        self.num_topics = config.num_topics or scale_params['topics']
        self.num_brokers = config.num_brokers or scale_params['brokers']
        
        # Get scenario configuration
        self.scenario_config = self.SCENARIOS.get(config.scenario, self.SCENARIOS['generic'])
        self.qos_profile = self.QOS_PROFILES.get(
            self.scenario_config['qos_profile'], 
            self.QOS_PROFILES['balanced']
        )
        
        # Storage for generated components
        self.nodes: List[Dict] = []
        self.brokers: List[Dict] = []
        self.applications: List[Dict] = []
        self.topics: List[Dict] = []
        
        # Relationship storage
        self.runs_on: List[Dict] = []
        self.publishes_to: List[Dict] = []
        self.subscribes_to: List[Dict] = []
        self.routes: List[Dict] = []
        self.connects_to: List[Dict] = []
        self.depends_on: List[Dict] = []  # Unified dependency relationships

        # Tracking for anti-patterns
        self.injected_antipatterns: List[Dict] = []

        # App and topic categorization for semantic matching
        self.app_tiers: Dict[str, str] = {}  # app_id -> tier name
        self.topic_keywords: Dict[str, List[str]] = {}  # topic_id -> keywords
    
    # =========================================================================
    # Main Generation Method
    # =========================================================================
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate a complete pub-sub system graph.
        
        Returns:
            Dictionary containing the complete graph definition
        """
        self.logger.info(f"Generating {self.config.scale} scale {self.config.scenario} system...")
        
        # Reset state
        self._reset_state()
        
        # Generate components
        self._generate_infrastructure()
        self._generate_brokers()
        self._generate_topics()
        self._generate_applications()
        
        # Generate relationships
        self._generate_runs_on_relationships()
        self._generate_routes_relationships()
        self._generate_connects_to_relationships()
        self._generate_pubsub_relationships()

        # Generate unified DEPENDS_ON relationships
        self._generate_depends_on_relationships()

        # Inject anti-patterns if configured
        if self.config.antipatterns:
            self._inject_antipatterns()

        # Build final graph
        graph = self._build_graph_dict()
        
        self.logger.info(
            f"Generated graph: {len(self.nodes)} nodes, {len(self.applications)} apps, "
            f"{len(self.topics)} topics, {len(self.brokers)} brokers"
        )
        
        return graph
    
    def _reset_state(self):
        """Reset all state for fresh generation"""
        self.nodes = []
        self.brokers = []
        self.applications = []
        self.topics = []
        self.runs_on = []
        self.publishes_to = []
        self.subscribes_to = []
        self.routes = []
        self.connects_to = []
        self.depends_on = []
        self.injected_antipatterns = []
        self.app_tiers = {}
        self.topic_keywords = {}
    
    # =========================================================================
    # Infrastructure Generation
    # =========================================================================
    
    def _generate_infrastructure(self):
        """Generate infrastructure nodes"""
        self.logger.debug(f"Generating {self.num_nodes} infrastructure nodes...")
        
        # Distribute nodes across types
        type_weights = [t['weight'] for t in self.NODE_TYPES.values()]
        type_names = list(self.NODE_TYPES.keys())
        
        # Calculate nodes per type
        nodes_per_type = self._distribute_by_weights(self.num_nodes, type_weights)
        
        node_id = 0
        for node_type, count in zip(type_names, nodes_per_type):
            type_config = self.NODE_TYPES[node_type]
            
            for i in range(count):
                node = {
                    'id': f"{type_config['prefix']}_{node_id}",
                    'name': f"{node_type.title()} Node {node_id}",
                    'node_type': node_type
                }
                
                self.nodes.append(node)
                node_id += 1
    
    def _generate_brokers(self):
        """Generate message brokers"""
        self.logger.debug(f"Generating {self.num_brokers} brokers...")
        
        broker_types = ['dds', 'kafka', 'rabbitmq', 'mqtt', 'ros2']
        scenario_broker_type = {
            'iot': 'mqtt',
            'financial': 'kafka',
            'autonomous_vehicle': 'ros2',
            'smart_city': 'mqtt',
            'healthcare': 'kafka',
            'gaming': 'kafka'
        }.get(self.config.scenario, 'dds')
        
        for i in range(self.num_brokers):
            # Primary broker type from scenario, others mixed
            broker_type = scenario_broker_type if i == 0 else random.choice(broker_types)
            
            broker = {
                'id': f"broker_{i}",
                'name': f"Broker {i} ({broker_type.upper()})",
                'broker_type': broker_type
            }
            self.brokers.append(broker)
    
    # =========================================================================
    # Topic Generation
    # =========================================================================
    
    def _generate_topics(self):
        """Generate topics with QoS policies and keyword extraction for semantic matching"""
        self.logger.debug(f"Generating {self.num_topics} topics...")

        # Get topic patterns for scenario
        patterns = self.scenario_config['topic_patterns']
        pattern_weights = [p[1] for p in patterns]
        pattern_templates = [p[0] for p in patterns]

        # Generate topic names from patterns
        for i in range(self.num_topics):
            # Select pattern based on weights
            pattern = self._weighted_choice(pattern_templates, pattern_weights)

            # Fill in pattern variables
            topic_name = self._fill_topic_pattern(pattern, i)

            # Generate QoS based on profile
            qos = self._generate_qos()

            # Extract keywords for semantic matching
            keywords = self._extract_topic_keywords(topic_name, pattern)

            topic = {
                'id': f"topic_{i}",
                'name': topic_name,
                'qos': qos,
                'message_size_bytes': random.choice([64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536]),
                'message_rate_hz': random.choice([1, 10, 20, 50, 100, 200, 500, 1000]),
                'keywords': keywords  # Store keywords for matching
            }
            self.topics.append(topic)

            # Store keywords for semantic matching
            self.topic_keywords[topic['id']] = keywords
    
    def _fill_topic_pattern(self, pattern: str, index: int) -> str:
        """Fill in topic pattern placeholders"""
        replacements = {
            '{domain}': random.choice(['users', 'orders', 'products', 'inventory', 'payments']),
            '{type}': random.choice(['temperature', 'pressure', 'humidity', 'motion', 'light']),
            '{device_id}': f"device_{index % 100}",
            '{device_type}': random.choice(['sensor', 'actuator', 'controller', 'gateway']),
            '{zone}': f"zone_{index % 10}",
            '{severity}': random.choice(['info', 'warning', 'critical']),
            '{exchange}': random.choice(['NYSE', 'NASDAQ', 'LSE', 'TSE']),
            '{symbol}': f"SYM{index % 50}",
            '{desk}': random.choice(['equities', 'fixed_income', 'derivatives', 'fx']),
            '{portfolio}': f"portfolio_{index % 20}",
            '{account}': f"account_{index % 100}",
            '{region}': random.choice(['us-east', 'us-west', 'eu', 'asia']),
            '{warehouse}': f"warehouse_{index % 5}",
            '{provider}': random.choice(['stripe', 'paypal', 'square']),
            '{session}': f"session_{index}",
            '{channel}': random.choice(['email', 'sms', 'push']),
            '{intersection}': f"intersection_{index % 50}",
            '{district}': f"district_{index % 10}",
            '{line}': random.choice(['red', 'blue', 'green', 'orange']),
            '{patient_id}': f"patient_{index % 200}",
            '{unit}': random.choice(['icu', 'er', 'ward_a', 'ward_b']),
            '{subsystem}': random.choice(['lidar', 'camera', 'radar', 'gps']),
            '{session_id}': f"game_{index % 100}",
            '{player_id}': f"player_{index % 1000}",
            '{game_mode}': random.choice(['ranked', 'casual', 'tournament']),
            '{level}': random.choice(['debug', 'info', 'warn', 'error']),
            '{source}': f"source_{index % 20}"
        }
        
        result = pattern
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, str(value))
        
        return result
    
    def _extract_topic_keywords(self, topic_name: str, pattern: str) -> List[str]:
        """
        Extract keywords from topic name and pattern for semantic matching.

        Args:
            topic_name: The filled topic name
            pattern: The original pattern template

        Returns:
            List of keywords that can be used for semantic matching
        """
        keywords = set()

        # Extract keywords from pattern (before variable substitution)
        pattern_parts = pattern.replace('/', ' ').replace('{', ' ').replace('}', ' ').split()
        for part in pattern_parts:
            # Clean and add meaningful parts
            part = part.lower().strip('_')
            if len(part) > 2 and not part.isdigit():
                keywords.add(part)

        # Extract keywords from topic name
        name_parts = topic_name.replace('/', ' ').replace('_', ' ').split()
        for part in name_parts:
            part = part.lower()
            if len(part) > 2 and not part.isdigit():
                keywords.add(part)

        # Add domain-specific keywords based on common patterns
        keyword_expansions = {
            'sensor': ['sensors', 'telemetry', 'data', 'raw'],
            'telemetry': ['sensors', 'data', 'metrics'],
            'command': ['commands', 'control'],
            'alert': ['alerts', 'notifications', 'warning'],
            'order': ['orders', 'trades', 'transactions'],
            'status': ['status', 'health', 'state'],
            'event': ['events', 'notifications'],
            'metric': ['metrics', 'telemetry', 'monitoring'],
            'position': ['positions', 'localization', 'location'],
            'vital': ['vitals', 'health', 'patient'],
            'game': ['game', 'session', 'player'],
            'traffic': ['traffic', 'flow', 'intersection'],
            'market': ['market', 'quotes', 'prices'],
            'inventory': ['inventory', 'stock', 'items']
        }

        expanded_keywords = set()
        for kw in keywords:
            if kw in keyword_expansions:
                expanded_keywords.update(keyword_expansions[kw])
        keywords.update(expanded_keywords)

        return list(keywords)

    def _generate_message_type(self, pattern: str) -> str:
        """Generate appropriate message type based on topic pattern"""
        if 'sensor' in pattern or 'telemetry' in pattern:
            return 'SensorData'
        elif 'command' in pattern or 'control' in pattern:
            return 'Command'
        elif 'alert' in pattern:
            return 'Alert'
        elif 'order' in pattern:
            return 'OrderMessage'
        elif 'status' in pattern:
            return 'StatusUpdate'
        elif 'event' in pattern:
            return 'Event'
        elif 'metric' in pattern:
            return 'Metric'
        else:
            return 'GenericMessage'
    
    def _generate_qos(self) -> Dict[str, Any]:
        """Generate QoS policy based on profile"""
        profile = self.qos_profile
        
        reliability = self._weighted_choice(
            [r[0] for r in profile['reliability_dist']],
            [r[1] for r in profile['reliability_dist']]
        )
        
        durability = self._weighted_choice(
            [d[0] for d in profile['durability_dist']],
            [d[1] for d in profile['durability_dist']]
        )
        
        priority = self._weighted_choice(
            [p[0] for p in profile['priority_dist']],
            [p[1] for p in profile['priority_dist']]
        )
        
        deadline_range = profile['deadline_range']
        deadline = random.randint(deadline_range[0], deadline_range[1])
        
        return {
            'reliability': reliability.value,
            'durability': durability.value,
            'deadline_ms': deadline,
            'transport_priority': priority.value,
            'history_depth': random.choice([1, 5, 10, 100])
        }
    
    # =========================================================================
    # Application Generation
    # =========================================================================
    
    def _generate_applications(self):
        """Generate applications with domain-specific types and tier classification"""
        self.logger.debug(f"Generating {self.num_apps} applications...")

        # Get app types and weights for scenario
        app_types = self.scenario_config['app_types']
        type_names = [t[0] for t in app_types]
        type_roles = [t[1] for t in app_types]
        type_weights = [t[2] for t in app_types]

        for i in range(self.num_apps):
            # Select app type based on weights
            idx = self._weighted_choice_index(type_weights)
            app_type = type_names[idx]
            app_role = type_roles[idx]

            # Generate unique name
            type_count = sum(1 for a in self.applications if app_type in a['name'])

            # Classify app into tier based on app type keywords
            tier = self._classify_app_tier(app_type)

            app = {
                'id': f"app_{i}",
                'name': f"{app_type}_{type_count}",
                'app_type': app_role,
                'tier': tier  # Add tier for hierarchical data flow
            }
            self.applications.append(app)

            # Store tier classification
            self.app_tiers[app['id']] = tier

    def _classify_app_tier(self, app_type: str) -> str:
        """
        Classify an application into a data flow tier based on its type.

        Args:
            app_type: The application type name

        Returns:
            Tier name: 'edge', 'gateway', or 'cloud'
        """
        app_type_lower = app_type.lower()

        # Check each tier's keywords
        for tier_name, tier_config in self.DATA_FLOW_TIERS.items():
            for keyword in tier_config['app_keywords']:
                if keyword.lower() in app_type_lower:
                    return tier_name

        # Default classification based on role patterns
        if any(kw in app_type_lower for kw in ['sensor', 'collector', 'device', 'monitor']):
            return 'edge'
        elif any(kw in app_type_lower for kw in ['gateway', 'aggregator', 'forwarder', 'dispatcher']):
            return 'gateway'
        else:
            return 'cloud'

    def _get_semantic_keywords_for_app(self, app: Dict) -> List[str]:
        """
        Get semantic keywords for an application based on its type.

        Args:
            app: Application dictionary

        Returns:
            List of keywords this app should be interested in
        """
        app_name = app['name']

        # Extract base type (before the underscore and number)
        base_type = app_name.rsplit('_', 1)[0] if '_' in app_name else app_name

        # Look up in semantic mappings
        if base_type in self.SEMANTIC_MAPPINGS:
            return self.SEMANTIC_MAPPINGS[base_type]

        # Try partial matches
        for type_key, keywords in self.SEMANTIC_MAPPINGS.items():
            if type_key.lower() in base_type.lower() or base_type.lower() in type_key.lower():
                return keywords

        # Default keywords based on role
        role = app.get('app_type', 'PROSUMER')
        if role == 'PRODUCER':
            return ['events', 'data', 'status', 'notifications']
        elif role == 'CONSUMER':
            return ['events', 'commands', 'alerts', 'updates']
        else:
            return ['events', 'data', 'status', 'commands']
    
    def _generate_criticality_weight(self, app_type: str, role: str) -> float:
        """Generate criticality weight based on app characteristics"""
        # Base weight by role
        base_weights = {
            'PRODUCER': 1.2,
            'CONSUMER': 0.9,
            'PROSUMER': 1.0
        }
        base = base_weights.get(role, 1.0)
        
        # Adjust by app type keywords
        critical_keywords = ['Monitor', 'Safety', 'Risk', 'Compliance', 'Alert', 'Emergency']
        high_keywords = ['Engine', 'Processor', 'Manager', 'Gateway']
        
        if any(kw in app_type for kw in critical_keywords):
            base *= 1.5
        elif any(kw in app_type for kw in high_keywords):
            base *= 1.2
        
        # Add some randomness
        return round(base * random.uniform(0.9, 1.1), 2)
    
    # =========================================================================
    # Relationship Generation
    # =========================================================================
    
    def _generate_runs_on_relationships(self):
        """Generate RUNS_ON relationships (apps/brokers -> nodes)"""
        self.logger.debug("Generating RUNS_ON relationships...")
        
        # Distribute applications across nodes
        # Prefer compute nodes for most apps, edge nodes for sensors/gateways
        for app in self.applications:
            # Select node based on app type
            if any(kw in app['name'] for kw in ['Sensor', 'Edge', 'Gateway', 'Device']):
                # Prefer edge nodes
                candidates = [n for n in self.nodes if n['node_type'] == 'edge']
            elif any(kw in app['name'] for kw in ['Cloud', 'Analytics', 'Logger', 'Aggregator']):
                # Prefer cloud nodes
                candidates = [n for n in self.nodes if n['node_type'] == 'cloud']
            else:
                # Prefer compute nodes
                candidates = [n for n in self.nodes if n['node_type'] == 'compute']
            
            # Fallback to any node if no candidates
            if not candidates:
                candidates = self.nodes
            
            node = random.choice(candidates)
            self.runs_on.append({
                'from': app['id'],
                'to': node['id']
            })
        
        # Place brokers on appropriate nodes
        for broker in self.brokers:
            # Brokers typically on compute or cloud nodes
            candidates = [n for n in self.nodes if n['node_type'] in ['compute', 'cloud']]
            if not candidates:
                candidates = self.nodes
            
            node = random.choice(candidates)
            self.runs_on.append({
                'from': broker['id'],
                'to': node['id']
            })
    
    def _generate_routes_relationships(self):
        """Generate ROUTES relationships (brokers -> topics)"""
        self.logger.debug("Generating ROUTES relationships...")
        
        for i, topic in enumerate(self.topics):
            # Primary broker
            primary_broker = self.brokers[i % len(self.brokers)]
            self.routes.append({
                'from': primary_broker['id'],
                'to': topic['id']
            })

    def _generate_connects_to_relationships(self):
        """
        Generate CONNECTS_TO relationships with realistic network topologies.

        Uses a hybrid topology that combines:
        - Hierarchical connections between tiers (edge → compute → cloud)
        - Intra-tier connections for redundancy
        - Hub connections through gateway nodes
        """
        self.logger.debug("Generating CONNECTS_TO relationships with hybrid topology...")

        if len(self.nodes) < 2:
            return

        # Group nodes by type/tier
        edge_nodes = [n for n in self.nodes if n['node_type'] == 'edge']
        compute_nodes = [n for n in self.nodes if n['node_type'] == 'compute']
        cloud_nodes = [n for n in self.nodes if n['node_type'] == 'cloud']
        gateway_nodes = [n for n in self.nodes if n['node_type'] == 'gateway']

        connections_set: Set[Tuple[str, str]] = set()

        def add_connection(node1_id: str, node2_id: str):
            """Add bidirectional connection avoiding duplicates"""
            if node1_id != node2_id:
                key = tuple(sorted([node1_id, node2_id]))
                connections_set.add(key)

        # 1. Hierarchical inter-tier connections
        # Edge nodes connect to compute/gateway nodes
        for edge in edge_nodes:
            # Each edge connects to 1-2 compute or gateway nodes
            targets = compute_nodes + gateway_nodes
            if targets:
                num_connections = min(2, len(targets))
                for target in random.sample(targets, num_connections):
                    add_connection(edge['id'], target['id'])

        # Compute/Gateway nodes connect to cloud nodes
        for compute in compute_nodes + gateway_nodes:
            if cloud_nodes:
                # Each compute connects to 1-2 cloud nodes
                num_connections = min(2, len(cloud_nodes))
                for cloud in random.sample(cloud_nodes, num_connections):
                    add_connection(compute['id'], cloud['id'])

        # 2. Intra-tier redundancy connections
        # Connect some nodes within the same tier for fault tolerance

        # Edge nodes: sparse ring-like connections
        if len(edge_nodes) > 1:
            for i in range(len(edge_nodes)):
                # Connect to next node in ring
                next_idx = (i + 1) % len(edge_nodes)
                add_connection(edge_nodes[i]['id'], edge_nodes[next_idx]['id'])

        # Compute nodes: more connected (small-world like)
        if len(compute_nodes) > 1:
            for i, node in enumerate(compute_nodes):
                # Connect to 2-3 nearby compute nodes
                for j in range(1, min(3, len(compute_nodes))):
                    target_idx = (i + j) % len(compute_nodes)
                    if target_idx != i:
                        add_connection(node['id'], compute_nodes[target_idx]['id'])

        # Cloud nodes: highly connected (data center fabric)
        if len(cloud_nodes) > 1:
            # Near full mesh for cloud nodes (they're in data centers)
            for i in range(len(cloud_nodes)):
                for j in range(i + 1, len(cloud_nodes)):
                    add_connection(cloud_nodes[i]['id'], cloud_nodes[j]['id'])

        # Gateway nodes: connect to each other
        if len(gateway_nodes) > 1:
            for i in range(len(gateway_nodes)):
                for j in range(i + 1, len(gateway_nodes)):
                    add_connection(gateway_nodes[i]['id'], gateway_nodes[j]['id'])

        # 3. Ensure connectivity - add random connections if graph is too sparse
        all_node_ids = [n['id'] for n in self.nodes]
        min_connections = len(self.nodes) - 1  # At least a spanning tree

        if len(connections_set) < min_connections:
            # Add random connections to ensure connectivity
            attempts = 0
            while len(connections_set) < min_connections and attempts < 100:
                n1, n2 = random.sample(all_node_ids, 2)
                add_connection(n1, n2)
                attempts += 1

        # Convert set to list of relationships
        for node1_id, node2_id in connections_set:
            self.connects_to.append({
                'from': node1_id,
                'to': node2_id
            })
    
    def _generate_pubsub_relationships(self):
        """
        Generate PUBLISHES_TO and SUBSCRIBES_TO relationships with semantic matching.

        Uses keyword matching to create realistic connections between apps and topics,
        and implements hierarchical data flow patterns (edge → gateway → cloud).
        """
        self.logger.debug("Generating pub/sub relationships with semantic matching...")

        # Group apps by role and tier
        publishers = [a for a in self.applications if a['app_type'] in ['PRODUCER', 'PROSUMER']]
        subscribers = [a for a in self.applications if a['app_type'] in ['CONSUMER', 'PROSUMER']]

        # Score topics for each app based on semantic matching
        app_topic_scores: Dict[str, List[Tuple[str, float]]] = {}

        for app in self.applications:
            app_keywords = self._get_semantic_keywords_for_app(app)
            topic_scores = []

            for topic in self.topics:
                topic_kws = self.topic_keywords.get(topic['id'], [])
                score = self._calculate_semantic_match_score(app_keywords, topic_kws)
                topic_scores.append((topic['id'], score))

            # Sort by score descending
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            app_topic_scores[app['id']] = topic_scores

        # Generate publishing relationships with semantic preference
        for app in publishers:
            app_tier = self.app_tiers.get(app['id'], 'cloud')

            # Get topic scores for this app
            topic_scores = app_topic_scores.get(app['id'], [])

            # Determine number of topics to publish to
            base_num_topics = max(1, int(len(self.topics) * self.config.edge_density * 0.2))

            # Adjust based on tier (edge apps typically publish to fewer topics)
            if app_tier == 'edge':
                num_topics = max(1, base_num_topics // 2)
            elif app_tier == 'gateway':
                num_topics = base_num_topics
            else:  # cloud
                num_topics = max(1, int(base_num_topics * 1.2))

            num_topics = min(num_topics, len(self.topics))

            # Select topics preferring high-scoring semantic matches
            # Use weighted random selection favoring better matches
            selected_topics = self._select_topics_by_score(topic_scores, num_topics)

            for topic_id in selected_topics:
                topic = next((t for t in self.topics if t['id'] == topic_id), None)
                if topic:
                    self.publishes_to.append({
                        'from': app['id'],
                        'to': topic_id,
                        'period_ms': int(1000 / topic.get('message_rate_hz', 100)),
                        'message_size_bytes': topic.get('message_size_bytes', 256)
                    })

        # Generate subscription relationships with semantic preference
        for app in subscribers:
            app_tier = self.app_tiers.get(app['id'], 'cloud')

            # Get topic scores for this app
            topic_scores = app_topic_scores.get(app['id'], [])

            # Determine number of topics to subscribe to
            base_num_topics = max(1, int(len(self.topics) * self.config.edge_density * 0.25))

            # Adjust based on tier (cloud/gateway apps typically subscribe to more topics)
            if app_tier == 'edge':
                num_topics = max(1, base_num_topics // 2)  # Edge devices often just produce
            elif app_tier == 'gateway':
                num_topics = int(base_num_topics * 1.5)  # Gateways aggregate from many
            else:  # cloud
                num_topics = base_num_topics

            num_topics = min(num_topics, len(self.topics))

            # Select topics preferring high-scoring semantic matches
            selected_topics = self._select_topics_by_score(topic_scores, num_topics)

            # Get topics this app already publishes to (to avoid self-subscription)
            published_topics = {rel['to'] for rel in self.publishes_to if rel['from'] == app['id']}

            for topic_id in selected_topics:
                if topic_id not in published_topics:
                    self.subscribes_to.append({
                        'from': app['id'],
                        'to': topic_id
                    })

        # Ensure every app has at least one connection
        self._ensure_all_apps_connected()

        # Add hierarchical data flow patterns (edge → gateway → cloud)
        self._add_hierarchical_data_flows()

    # =========================================================================
    # DEPENDS_ON Relationship Generation
    # =========================================================================

    def _calculate_topic_qos_criticality(self, topic: Dict) -> float:
        """
        Calculate criticality weight based on topic QoS policies.

        Higher scores indicate more critical dependencies:
        - Reliable topics create stronger dependencies than best-effort
        - Persistent/durable topics are more critical than volatile
        - Lower deadlines indicate time-critical communication
        - Higher transport priority indicates business criticality

        Args:
            topic: Topic dictionary with 'qos' field

        Returns:
            Criticality score between 0.1 and 1.0
        """
        qos = topic.get('qos', {})
        score = 0.0

        # Reliability contributes up to 0.30
        # RELIABLE topics create stronger dependencies - failure affects data integrity
        reliability = qos.get('reliability', 'best_effort')
        if reliability in ['reliable', 'RELIABLE']:
            score += 0.30
        else:  # best_effort
            score += 0.05

        # Durability contributes up to 0.25
        # PERSISTENT topics indicate critical data that must survive failures
        durability = qos.get('durability', 'volatile')
        durability_scores = {
            'persistent': 0.25,
            'PERSISTENT': 0.25,
            'transient': 0.15,
            'TRANSIENT': 0.15,
            'transient_local': 0.10,
            'TRANSIENT_LOCAL': 0.10,
            'volatile': 0.02,
            'VOLATILE': 0.02
        }
        score += durability_scores.get(durability, 0.02)

        # Deadline contributes up to 0.25
        # Tighter deadlines = more time-critical = higher dependency weight
        deadline_ms = qos.get('deadline_ms')
        if deadline_ms is not None:
            if deadline_ms <= 10:
                score += 0.25  # Ultra-low latency (real-time systems)
            elif deadline_ms <= 50:
                score += 0.20  # Low latency (financial, gaming)
            elif deadline_ms <= 100:
                score += 0.15  # Medium latency
            elif deadline_ms <= 500:
                score += 0.08  # Standard latency
            else:
                score += 0.03  # Relaxed deadline

        # Transport priority contributes up to 0.20
        # Higher priority = more business-critical
        priority = qos.get('transport_priority', 1)
        if isinstance(priority, str):
            priority_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'URGENT': 3}
            priority = priority_map.get(priority.upper(), 1)
        priority_scores = {0: 0.02, 1: 0.08, 2: 0.14, 3: 0.20}
        score += priority_scores.get(priority, 0.08)

        # Ensure score is in valid range [0.1, 1.0]
        return max(0.1, min(1.0, score))

    def _generate_depends_on_relationships(self):
        """
        Generate unified DEPENDS_ON relationships across all layers with QoS-based weighting.

        This creates the semantic dependency graph used for criticality analysis:
        - APP_TO_APP: Application A depends on B if B publishes to a topic A subscribes to
          Weight is determined by the topic's QoS criticality (reliability, durability, deadline, priority)
        - APP_TO_BROKER: Application depends on broker routing its topics
          Weight is the maximum QoS criticality of topics routed through the broker
        - NODE_TO_NODE: Derived from app dependencies of hosted applications
          Weight aggregates the QoS-weighted app dependencies
        - NODE_TO_BROKER: Node depends on broker if hosted apps use it
        """
        self.logger.debug("Generating unified DEPENDS_ON relationships with QoS weighting...")

        # Build topic lookup by ID for QoS access
        topic_by_id: Dict[str, Dict] = {t['id']: t for t in self.topics}

        # Build lookup structures
        topic_publishers: Dict[str, List[str]] = defaultdict(list)  # topic_id -> [app_ids]
        topic_subscribers: Dict[str, List[str]] = defaultdict(list)  # topic_id -> [app_ids]
        topic_brokers: Dict[str, List[str]] = defaultdict(list)  # topic_id -> [broker_ids]
        app_to_node: Dict[str, str] = {}  # app_id -> node_id
        broker_to_node: Dict[str, str] = {}  # broker_id -> node_id

        # Build publisher/subscriber maps
        for rel in self.publishes_to:
            topic_publishers[rel['to']].append(rel['from'])
        for rel in self.subscribes_to:
            topic_subscribers[rel['to']].append(rel['from'])
        for rel in self.routes:
            topic_brokers[rel['to']].append(rel['from'])

        # Build placement maps
        for rel in self.runs_on:
            source = rel['from']
            if source.startswith('app_') or any(prefix in source for prefix in ['spof_', 'coupling_', 'cycle_']):
                app_to_node[source] = rel['to']
            elif source.startswith('broker_'):
                broker_to_node[source] = rel['to']

        # 1. Generate APP_TO_APP dependencies with QoS-based weights
        # A subscriber depends on all publishers to the topics it subscribes to
        # Weight reflects the criticality of the topic's QoS policies
        for topic_id, subscribers in topic_subscribers.items():
            publishers = topic_publishers.get(topic_id, [])
            topic = topic_by_id.get(topic_id, {})
            qos_weight = self._calculate_topic_qos_criticality(topic)

            for subscriber in subscribers:
                for publisher in publishers:
                    if subscriber != publisher:  # No self-dependency
                        self.depends_on.append({
                            'from': subscriber,
                            'to': publisher,
                            'type': DependencyType.APP_TO_APP.value,
                            'via_topic': topic_id,
                            'weight': round(qos_weight, 3),
                            'qos_factors': {
                                'reliability': topic.get('qos', {}).get('reliability', 'unknown'),
                                'durability': topic.get('qos', {}).get('durability', 'unknown'),
                                'deadline_ms': topic.get('qos', {}).get('deadline_ms'),
                                'priority': topic.get('qos', {}).get('transport_priority', 'unknown')
                            }
                        })

        # 2. Generate APP_TO_BROKER dependencies with max QoS weight
        # Apps depend on brokers that route topics they publish/subscribe to
        # Weight is the maximum QoS criticality across all topics the app uses through this broker
        app_broker_deps: Dict[Tuple[str, str], float] = {}  # (app_id, broker_id) -> max_weight

        for topic_id, brokers in topic_brokers.items():
            topic = topic_by_id.get(topic_id, {})
            qos_weight = self._calculate_topic_qos_criticality(topic)

            # Publishers depend on brokers
            for publisher in topic_publishers.get(topic_id, []):
                for broker in brokers:
                    key = (publisher, broker)
                    app_broker_deps[key] = max(app_broker_deps.get(key, 0), qos_weight)

            # Subscribers depend on brokers
            for subscriber in topic_subscribers.get(topic_id, []):
                for broker in brokers:
                    key = (subscriber, broker)
                    app_broker_deps[key] = max(app_broker_deps.get(key, 0), qos_weight)

        for (app_id, broker_id), weight in app_broker_deps.items():
            self.depends_on.append({
                'from': app_id,
                'to': broker_id,
                'type': DependencyType.APP_TO_BROKER.value,
                'weight': round(weight, 3)
            })

        # 3. Generate NODE_TO_NODE dependencies (derived from app dependencies)
        # Weight aggregates QoS-weighted app dependencies between nodes
        node_deps: Dict[Tuple[str, str], List[float]] = defaultdict(list)  # (from_node, to_node) -> [weights]

        for dep in self.depends_on:
            if dep['type'] == DependencyType.APP_TO_APP.value:
                from_node = app_to_node.get(dep['from'])
                to_node = app_to_node.get(dep['to'])
                if from_node and to_node and from_node != to_node:
                    node_deps[(from_node, to_node)].append(dep['weight'])

        for (from_node, to_node), weights in node_deps.items():
            # Aggregate weight: use weighted average with count factor
            # More dependencies with higher weights = higher node-level criticality
            avg_weight = sum(weights) / len(weights)
            count_factor = min(1.0, len(weights) / 5.0)  # Normalize by count
            aggregated_weight = (avg_weight * 0.7) + (count_factor * 0.3)

            self.depends_on.append({
                'from': from_node,
                'to': to_node,
                'type': DependencyType.NODE_TO_NODE.value,
                'weight': round(aggregated_weight, 3),
                'derived_count': len(weights),
                'avg_qos_weight': round(avg_weight, 3)
            })

        # 4. Generate NODE_TO_BROKER dependencies
        # Weight is max of app-to-broker weights for apps on this node
        node_broker_deps: Dict[Tuple[str, str], float] = {}

        for dep in self.depends_on:
            if dep['type'] == DependencyType.APP_TO_BROKER.value:
                app_node = app_to_node.get(dep['from'])
                broker_node = broker_to_node.get(dep['to'])
                if app_node and broker_node and app_node != broker_node:
                    key = (app_node, broker_node)
                    node_broker_deps[key] = max(node_broker_deps.get(key, 0), dep['weight'])

        for (node_id, broker_node), weight in node_broker_deps.items():
            self.depends_on.append({
                'from': node_id,
                'to': broker_node,
                'type': DependencyType.NODE_TO_BROKER.value,
                'weight': round(weight, 3)
            })

        self.logger.debug(f"Generated {len(self.depends_on)} DEPENDS_ON relationships with QoS weighting")

    def _calculate_semantic_match_score(self, app_keywords: List[str], topic_keywords: List[str]) -> float:
        """
        Calculate semantic match score between app and topic keywords.

        Args:
            app_keywords: Keywords associated with the app type
            topic_keywords: Keywords extracted from the topic

        Returns:
            Score between 0.0 and 1.0 indicating match quality
        """
        if not app_keywords or not topic_keywords:
            return 0.1  # Small baseline score

        app_set = set(kw.lower() for kw in app_keywords)
        topic_set = set(kw.lower() for kw in topic_keywords)

        # Calculate Jaccard similarity
        intersection = len(app_set & topic_set)
        union = len(app_set | topic_set)

        if union == 0:
            return 0.1

        jaccard = intersection / union

        # Boost score if there's exact keyword match
        exact_matches = len(app_set & topic_set)
        boost = min(0.3, exact_matches * 0.1)

        return min(1.0, jaccard + boost + 0.1)  # Add small baseline

    def _select_topics_by_score(self, topic_scores: List[Tuple[str, float]], num_topics: int) -> List[str]:
        """
        Select topics using weighted random selection based on scores.

        Args:
            topic_scores: List of (topic_id, score) tuples sorted by score
            num_topics: Number of topics to select

        Returns:
            List of selected topic IDs
        """
        if not topic_scores:
            return []

        # Use scores as weights for selection
        topics = [t[0] for t in topic_scores]
        scores = [max(0.1, t[1]) for t in topic_scores]  # Ensure minimum weight

        selected = []
        available_topics = list(zip(topics, scores))

        for _ in range(min(num_topics, len(available_topics))):
            if not available_topics:
                break

            # Weighted random selection
            total_weight = sum(s for _, s in available_topics)
            r = random.uniform(0, total_weight)
            cumulative = 0

            for i, (topic, score) in enumerate(available_topics):
                cumulative += score
                if r <= cumulative:
                    selected.append(topic)
                    available_topics.pop(i)
                    break

        return selected

    def _ensure_all_apps_connected(self):
        """Ensure every application has at least one pub/sub connection."""
        connected_apps = set()
        for rel in self.publishes_to:
            connected_apps.add(rel['from'])
        for rel in self.subscribes_to:
            connected_apps.add(rel['from'])

        for app in self.applications:
            if app['id'] not in connected_apps:
                # Use semantic matching to find best topic
                app_keywords = self._get_semantic_keywords_for_app(app)
                best_topic = None
                best_score = -1

                for topic in self.topics:
                    topic_kws = self.topic_keywords.get(topic['id'], [])
                    score = self._calculate_semantic_match_score(app_keywords, topic_kws)
                    if score > best_score:
                        best_score = score
                        best_topic = topic

                if best_topic is None:
                    best_topic = random.choice(self.topics)

                if app['app_type'] in ['PRODUCER', 'PROSUMER']:
                    self.publishes_to.append({
                        'from': app['id'],
                        'to': best_topic['id'],
                        'period_ms': int(1000 / best_topic.get('message_rate_hz', 100)),
                        'message_size_bytes': best_topic.get('message_size_bytes', 256)
                    })
                else:
                    self.subscribes_to.append({
                        'from': app['id'],
                        'to': best_topic['id']
                    })

    def _add_hierarchical_data_flows(self):
        """
        Add hierarchical data flow patterns to create realistic tiered architecture.

        Creates connections: edge apps → gateway apps → cloud apps
        """
        # Group apps by tier
        edge_apps = [a for a in self.applications if self.app_tiers.get(a['id']) == 'edge']
        gateway_apps = [a for a in self.applications if self.app_tiers.get(a['id']) == 'gateway']
        cloud_apps = [a for a in self.applications if self.app_tiers.get(a['id']) == 'cloud']

        # For IoT/smart city scenarios, create clear tier connections
        if self.config.scenario in ['iot', 'smart_city', 'healthcare', 'autonomous_vehicle']:
            # Ensure some gateway apps subscribe to topics published by edge apps
            edge_published_topics = {rel['to'] for rel in self.publishes_to
                                     if rel['from'] in {a['id'] for a in edge_apps}}

            for topic_id in edge_published_topics:
                # Assign some gateway apps to subscribe
                for gateway in gateway_apps[:min(2, len(gateway_apps))]:
                    if not any(s['from'] == gateway['id'] and s['to'] == topic_id
                               for s in self.subscribes_to):
                        self.subscribes_to.append({
                            'from': gateway['id'],
                            'to': topic_id
                        })

            # Ensure some cloud apps subscribe to topics published by gateway apps
            gateway_published_topics = {rel['to'] for rel in self.publishes_to
                                        if rel['from'] in {a['id'] for a in gateway_apps}}

            for topic_id in gateway_published_topics:
                for cloud in cloud_apps[:min(2, len(cloud_apps))]:
                    if not any(s['from'] == cloud['id'] and s['to'] == topic_id
                               for s in self.subscribes_to):
                        self.subscribes_to.append({
                            'from': cloud['id'],
                            'to': topic_id
                        })
    
    # =========================================================================
    # Anti-Pattern Injection
    # =========================================================================
    
    def _inject_antipatterns(self):
        """Inject configured anti-patterns into the graph"""
        self.logger.info(f"Injecting anti-patterns: {self.config.antipatterns}")
        
        for pattern in self.config.antipatterns:
            pattern_lower = pattern.lower()
            
            if pattern_lower == 'spof':
                self._inject_spof()
            elif pattern_lower == 'god_topic':
                self._inject_god_topic()
            elif pattern_lower == 'broker_overload':
                self._inject_broker_overload()
            elif pattern_lower == 'tight_coupling':
                self._inject_tight_coupling()
            elif pattern_lower in ['chatty', 'chatty_communication']:
                self._inject_chatty_communication()
            elif pattern_lower in ['circular', 'circular_dependency']:
                self._inject_circular_dependency()
            elif pattern_lower == 'bottleneck':
                self._inject_bottleneck()
            elif pattern_lower == 'hidden_coupling':
                self._inject_hidden_coupling()
            else:
                self.logger.warning(f"Unknown anti-pattern: {pattern}")
    
    def _create_antipattern_topic(self, topic_id: str, topic_name: str,
                                     reliability: str = 'reliable',
                                     durability: str = 'volatile',
                                     deadline_ms: int = 100,
                                     priority: int = 1,
                                     message_size: int = 256,
                                     message_rate: int = 100) -> Dict:
        """
        Create a properly formatted topic for anti-pattern injection.

        Ensures all required fields are present for consistency with regular topics.
        """
        keywords = self._extract_topic_keywords(topic_name, topic_name)
        return {
            'id': topic_id,
            'name': topic_name,
            'qos': {
                'reliability': reliability,
                'durability': durability,
                'deadline_ms': deadline_ms,
                'transport_priority': priority,
                'history_depth': 10
            },
            'message_size_bytes': message_size,
            'message_rate_hz': message_rate,
            'keywords': keywords
        }

    def _inject_spof(self):
        """Inject Single Point of Failure pattern"""
        # Create a critical app that many others depend on
        spof_app = {
            'id': 'spof_critical_service',
            'name': 'CriticalSPOFService',
            'app_type': 'PROSUMER',
            'tier': 'cloud',
            'criticality_weight': 2.0
        }
        self.applications.append(spof_app)
        self.app_tiers[spof_app['id']] = 'cloud'

        # Create a topic that this SPOF publishes to
        spof_topic = self._create_antipattern_topic(
            topic_id='spof_critical_topic',
            topic_name='critical/spof/data',
            reliability='reliable',
            durability='persistent',
            deadline_ms=10,
            priority=3,
            message_size=1024,
            message_rate=100
        )
        self.topics.append(spof_topic)
        self.topic_keywords[spof_topic['id']] = spof_topic['keywords']
        
        # SPOF publishes to this topic
        self.publishes_to.append({
            'from': spof_app['id'],
            'to': spof_topic['id'],
            'period_ms': 10,
            'message_size_bytes': 1024
        })
        
        # Make many apps subscribe to it (creating the SPOF)
        num_dependents = min(len(self.applications) // 2, 20)
        dependents = random.sample(
            [a for a in self.applications if a['id'] != spof_app['id']],
            num_dependents
        )
        
        for app in dependents:
            self.subscribes_to.append({
                'from': app['id'],
                'to': spof_topic['id']
            })
        
        # Place SPOF on single node
        node = random.choice(self.nodes)
        self.runs_on.append({
            'from': spof_app['id'],
            'to': node['id']
        })
        
        self.injected_antipatterns.append({
            'type': 'spof',
            'component': spof_app['id'],
            'dependents': [a['id'] for a in dependents],
            'description': f"SPOF: {spof_app['name']} with {len(dependents)} dependent apps"
        })
    
    def _inject_god_topic(self):
        """Inject God Topic pattern (topic with too many connections)"""
        god_topic = self._create_antipattern_topic(
            topic_id='god_topic',
            topic_name='central/everything/events',
            reliability='reliable',
            durability='transient_local',
            deadline_ms=100,
            priority=2,
            message_size=512,
            message_rate=200
        )
        self.topics.append(god_topic)
        self.topic_keywords[god_topic['id']] = god_topic['keywords']
        
        # Many publishers
        num_publishers = min(len(self.applications) // 3, 15)
        publishers = random.sample(
            [a for a in self.applications if a['app_type'] in ['PRODUCER', 'PROSUMER']],
            num_publishers
        )
        
        for pub in publishers:
            self.publishes_to.append({
                'from': pub['id'],
                'to': god_topic['id'],
                'period_ms': random.choice([50, 100, 200]),
                'message_size_bytes': 512
            })
        
        # Many subscribers
        num_subscribers = min(len(self.applications) // 2, 25)
        subscribers = random.sample(
            [a for a in self.applications if a['app_type'] in ['CONSUMER', 'PROSUMER']],
            num_subscribers
        )
        
        for sub in subscribers:
            if sub['id'] not in [p['from'] for p in self.publishes_to if p['to'] == god_topic['id']]:
                self.subscribes_to.append({
                    'from': sub['id'],
                    'to': god_topic['id']
                })
        
        self.injected_antipatterns.append({
            'type': 'god_topic',
            'component': god_topic['id'],
            'publishers': len(publishers),
            'subscribers': len(subscribers),
            'description': f"God Topic with {len(publishers)} publishers and {len(subscribers)} subscribers"
        })
    
    def _inject_broker_overload(self):
        """Inject Broker Overload pattern"""
        if not self.brokers:
            return
        
        # Select one broker to overload
        overloaded_broker = self.brokers[0]
        overloaded_broker['current_load'] = 0.95  # Near capacity
        
        # Route most topics through this broker
        num_topics_to_route = int(len(self.topics) * 0.8)
        topics_to_route = random.sample(self.topics, num_topics_to_route)
        
        for topic in topics_to_route:
            # Remove existing routes for this topic
            self.routes = [r for r in self.routes if r['to'] != topic['id']]
            # Add route through overloaded broker
            self.routes.append({
                'from': overloaded_broker['id'],
                'to': topic['id']
            })
        
        self.injected_antipatterns.append({
            'type': 'broker_overload',
            'component': overloaded_broker['id'],
            'topics_routed': num_topics_to_route,
            'load': overloaded_broker['current_load'],
            'description': f"Broker {overloaded_broker['name']} routing {num_topics_to_route} topics at {overloaded_broker['current_load']*100:.0f}% load"
        })
    
    def _inject_tight_coupling(self):
        """Inject Tight Coupling pattern (many direct dependencies)"""
        # Create a cluster of tightly coupled apps
        cluster_size = min(8, len(self.applications) // 4)
        cluster_apps = random.sample(self.applications, cluster_size)

        # Create shared topics for tight coupling
        coupling_topics = []
        for i in range(cluster_size - 1):
            topic = self._create_antipattern_topic(
                topic_id=f'coupling_topic_{i}',
                topic_name=f'coupling/internal/channel_{i}',
                reliability='reliable',
                durability='volatile',
                deadline_ms=50,
                priority=1,
                message_size=256,
                message_rate=50
            )
            self.topics.append(topic)
            self.topic_keywords[topic['id']] = topic['keywords']
            coupling_topics.append(topic)
        
        # Create dense pub/sub mesh
        for i, app in enumerate(cluster_apps):
            # Each app publishes to next topic
            if i < len(coupling_topics):
                self.publishes_to.append({
                    'from': app['id'],
                    'to': coupling_topics[i]['id'],
                    'period_ms': 20,
                    'message_size_bytes': 256
                })
            
            # Each app subscribes to previous topics
            for j in range(max(0, i-2), i):
                if j < len(coupling_topics):
                    self.subscribes_to.append({
                        'from': app['id'],
                        'to': coupling_topics[j]['id']
                    })
        
        self.injected_antipatterns.append({
            'type': 'tight_coupling',
            'components': [a['id'] for a in cluster_apps],
            'coupling_topics': [t['id'] for t in coupling_topics],
            'description': f"Tight coupling cluster of {cluster_size} apps with {len(coupling_topics)} shared topics"
        })
    
    def _inject_chatty_communication(self):
        """Inject Chatty Communication pattern"""
        # Select apps to make chatty
        num_chatty = min(5, len(self.applications) // 5)
        chatty_apps = random.sample(
            [a for a in self.applications if a['app_type'] in ['PRODUCER', 'PROSUMER']],
            num_chatty
        )
        
        for app in chatty_apps:
            # Find their publishing relationships and make them chatty
            for pub_rel in self.publishes_to:
                if pub_rel['from'] == app['id']:
                    pub_rel['period_ms'] = 1  # Very frequent
                    pub_rel['message_size_bytes'] = 32  # Very small messages
        
        self.injected_antipatterns.append({
            'type': 'chatty_communication',
            'components': [a['id'] for a in chatty_apps],
            'description': f"{num_chatty} apps with chatty communication (1ms period, 32 byte messages)"
        })
    
    def _inject_circular_dependency(self):
        """Inject Circular Dependency pattern"""
        # Create a cycle: A -> B -> C -> A
        cycle_size = min(4, len(self.applications) // 4)
        cycle_apps = random.sample(self.applications, cycle_size)

        # Create topics for the cycle
        cycle_topics = []
        for i in range(cycle_size):
            topic = self._create_antipattern_topic(
                topic_id=f'cycle_topic_{i}',
                topic_name=f'cycle/step_{i}',
                reliability='reliable',
                durability='volatile',
                deadline_ms=100,
                priority=1,
                message_size=256,
                message_rate=10
            )
            self.topics.append(topic)
            self.topic_keywords[topic['id']] = topic['keywords']
            cycle_topics.append(topic)
        
        # Create the cycle
        for i in range(cycle_size):
            # App i publishes to topic i
            self.publishes_to.append({
                'from': cycle_apps[i]['id'],
                'to': cycle_topics[i]['id'],
                'period_ms': 100,
                'message_size_bytes': 256
            })
            
            # App (i+1) % cycle_size subscribes to topic i
            next_app = cycle_apps[(i + 1) % cycle_size]
            self.subscribes_to.append({
                'from': next_app['id'],
                'to': cycle_topics[i]['id']
            })
        
        self.injected_antipatterns.append({
            'type': 'circular_dependency',
            'components': [a['id'] for a in cycle_apps],
            'cycle_topics': [t['id'] for t in cycle_topics],
            'description': f"Circular dependency chain of {cycle_size} apps"
        })
    
    def _inject_bottleneck(self):
        """Inject Infrastructure Bottleneck pattern"""
        if len(self.nodes) < 2:
            return
        
        # Select one node to be the bottleneck
        bottleneck_node = self.nodes[0]
        
        # Move many apps to this node
        num_apps_to_move = int(len(self.applications) * 0.6)
        apps_to_move = random.sample(self.applications, num_apps_to_move)
        
        for app in apps_to_move:
            # Remove existing RUNS_ON for this app
            self.runs_on = [r for r in self.runs_on if r['from'] != app['id']]
            # Add to bottleneck node
            self.runs_on.append({
                'from': app['id'],
                'to': bottleneck_node['id']
            })
        
        self.injected_antipatterns.append({
            'type': 'bottleneck',
            'component': bottleneck_node['id'],
            'apps_hosted': num_apps_to_move,
            'description': f"Infrastructure bottleneck: {bottleneck_node['name']} hosting {num_apps_to_move} apps"
        })
    
    def _inject_hidden_coupling(self):
        """Inject Hidden Coupling pattern (shared topics creating implicit dependencies)"""
        # Create a "hidden" topic that multiple unrelated apps use
        hidden_topic = self._create_antipattern_topic(
            topic_id='hidden_coupling_topic',
            topic_name='system/internal/shared_state',
            reliability='reliable',
            durability='transient_local',
            deadline_ms=200,
            priority=1,
            message_size=1024,
            message_rate=2
        )
        self.topics.append(hidden_topic)
        self.topic_keywords[hidden_topic['id']] = hidden_topic['keywords']
        
        # Select random apps to create hidden coupling
        num_coupled = min(10, len(self.applications) // 3)
        coupled_apps = random.sample(self.applications, num_coupled)
        
        # Half publish, half subscribe - creating hidden dependencies
        mid = num_coupled // 2
        for app in coupled_apps[:mid]:
            self.publishes_to.append({
                'from': app['id'],
                'to': hidden_topic['id'],
                'period_ms': 500,
                'message_size_bytes': 1024
            })
        
        for app in coupled_apps[mid:]:
            self.subscribes_to.append({
                'from': app['id'],
                'to': hidden_topic['id']
            })
        
        self.injected_antipatterns.append({
            'type': 'hidden_coupling',
            'component': hidden_topic['id'],
            'coupled_apps': [a['id'] for a in coupled_apps],
            'description': f"Hidden coupling via {hidden_topic['name']} connecting {num_coupled} apps"
        })
    
    # =========================================================================
    # Output Building
    # =========================================================================
    
    def _build_graph_dict(self) -> Dict[str, Any]:
        """Build the final graph dictionary"""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '3.0.0',  # Version bump for new features
                'config': {
                    'scale': self.config.scale,
                    'scenario': self.config.scenario,
                    'seed': self.config.seed,
                    'antipatterns': self.config.antipatterns
                },
                'features': [
                    'semantic_topic_matching',
                    'hierarchical_data_flow',
                    'realistic_network_topology',
                    'unified_depends_on_model'
                ]
            },
            'nodes': self.nodes,
            'brokers': self.brokers,
            'applications': self.applications,
            'topics': self.topics,
            'relationships': {
                'runs_on': self.runs_on,
                'publishes_to': self.publishes_to,
                'subscribes_to': self.subscribes_to,
                'routes': self.routes,
                'connects_to': self.connects_to,
                'depends_on': self.depends_on  # Unified dependency relationships
            },
            'statistics': self._calculate_statistics(),
            'injected_antipatterns': self.injected_antipatterns
        }
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate graph statistics"""
        # Publisher/subscriber stats
        publishers_count = len(set(r['from'] for r in self.publishes_to))
        subscribers_count = len(set(r['from'] for r in self.subscribes_to))

        # Topics stats
        topic_pub_counts = defaultdict(int)
        topic_sub_counts = defaultdict(int)
        for r in self.publishes_to:
            topic_pub_counts[r['to']] += 1
        for r in self.subscribes_to:
            topic_sub_counts[r['to']] += 1

        avg_pubs_per_topic = sum(topic_pub_counts.values()) / max(1, len(self.topics))
        avg_subs_per_topic = sum(topic_sub_counts.values()) / max(1, len(self.topics))

        # Node stats
        apps_per_node = defaultdict(int)
        for r in self.runs_on:
            if r['from'].startswith('app_') or any(r['from'].startswith(p) for p in ['spof_', 'coupling_']):
                apps_per_node[r['to']] += 1

        # Dependency stats by type
        depends_on_by_type = defaultdict(int)
        for dep in self.depends_on:
            depends_on_by_type[dep.get('type', 'unknown')] += 1

        # App tier distribution
        tier_counts = defaultdict(int)
        for tier in self.app_tiers.values():
            tier_counts[tier] += 1

        return {
            'total_nodes': len(self.nodes),
            'total_brokers': len(self.brokers),
            'total_applications': len(self.applications),
            'total_topics': len(self.topics),
            'total_publishes': len(self.publishes_to),
            'total_subscribes': len(self.subscribes_to),
            'total_routes': len(self.routes),
            'total_connects': len(self.connects_to),
            'total_depends_on': len(self.depends_on),
            'unique_publishers': publishers_count,
            'unique_subscribers': subscribers_count,
            'avg_publishers_per_topic': round(avg_pubs_per_topic, 2),
            'avg_subscribers_per_topic': round(avg_subs_per_topic, 2),
            'avg_apps_per_node': round(sum(apps_per_node.values()) / max(1, len(self.nodes)), 2),
            'antipatterns_injected': len(self.injected_antipatterns),
            'depends_on_by_type': dict(depends_on_by_type),
            'app_tiers': dict(tier_counts)
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _distribute_by_weights(self, total: int, weights: List[float]) -> List[int]:
        """Distribute total count according to weights"""
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]
        
        result = [int(total * w) for w in normalized]
        
        # Distribute remainder
        remainder = total - sum(result)
        for i in range(remainder):
            result[i % len(result)] += 1
        
        return result
    
    def _weighted_choice(self, items: List, weights: List[float]):
        """Select an item based on weights"""
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for item, weight in zip(items, weights):
            cumulative += weight
            if r <= cumulative:
                return item
        return items[-1]
    
    def _weighted_choice_index(self, weights: List[float]) -> int:
        """Return index of weighted random choice"""
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return i
        return len(weights) - 1


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_graph(
    scale: str = 'medium',
    scenario: str = 'generic',
    antipatterns: Optional[List[str]] = None,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to generate a graph.
    
    Args:
        scale: Size preset (tiny, small, medium, large, xlarge, extreme)
        scenario: Domain scenario
        antipatterns: List of anti-patterns to inject
        seed: Random seed
        **kwargs: Additional config parameters
    
    Returns:
        Generated graph dictionary
    """
    config = GraphConfig(
        scale=scale,
        scenario=scenario,
        antipatterns=antipatterns or [],
        seed=seed,
        **kwargs
    )
    generator = GraphGenerator(config)
    return generator.generate()