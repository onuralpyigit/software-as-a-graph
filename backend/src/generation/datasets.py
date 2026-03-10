"""
Dataset definitions for realistic component generation.
"""
from typing import Dict, List, Any, Tuple
import random
from src.core.models import QoSPolicy

# Predefined datasets for various domains mapping to scenario.md
DOMAIN_DATASETS: Dict[str, Dict[str, List[str]]] = {
    "av": {
        "nodes": ["vision-compute", "motion-controller", "sensor-hub", "nav-computer", "lidar-processor", "main-brain"],
        "apps": ["object-detector", "path-planner", "motor-driver", "lidar-driver", "camera-driver", "slam-node", "collision-avoider", "state-estimator"],
        "topics": ["camera.rgb", "lidar.scan", "odom", "cmd_vel", "joint_states", "tf", "goal_pose", "map"],
        "libraries": ["cv-bridge", "tf-lib", "nav-core", "sensor-msgs", "geometry-msgs"],
        "brokers": ["ros-master", "zenoh-router"]
    },
    "iot": {
        "nodes": ["street-hub", "sensor-gateway", "edge-compute", "city-db", "traffic-controller", "weather-station"],
        "apps": ["traffic-monitor", "air-quality-sensor", "smart-light-controller", "power-meter", "data-aggregator", "alert-manager"],
        "topics": ["traffic.flow", "air.quality", "lights.status", "lights.cmd", "power.usage", "alerts.city"],
        "libraries": ["mqtt-client", "coap-lib", "edge-analytics", "sensor-drivers"],
        "brokers": ["emqx-cluster", "mosquitto-edge"]
    },
    "finance": {
        "nodes": ["trading-node", "ledger-node", "market-data-node", "risk-node", "gateway-node"],
        "apps": ["matching-engine", "order-router", "risk-evaluator", "market-data-feed", "ledger-writer", "fraud-detector"],
        "topics": ["market.quotes", "market.trades", "orders.new", "orders.executed", "risk.alerts", "ledger.entries"],
        "libraries": ["fix-protocol", "crypto-lib", "math-models", "hft-utils"],
        "brokers": ["kafka-trading", "solace-router"]
    },
    "healthcare": {
        "nodes": ["his-server", "pacs-archive", "clinical-gateway", "wearable-hub", "lab-system"],
        "apps": ["emr-service", "pacs-viewer", "vitals-monitor", "lab-results-processor", "appointment-scheduler", "billing-service"],
        "topics": ["patient.vitals", "hl7.orders", "hl7.results", "dicom.images", "appointments.created", "billing.events"],
        "libraries": ["hl7-parser", "dicom-toolkit", "phi-encryptor", "fhir-client"],
        "brokers": ["rabbitmq-clinical", "kafka-analytics"]
    },
    "hub-and-spoke": {
        "nodes": ["central-hub", "spoke-node-1", "spoke-node-2", "spoke-node-3"],
        "apps": ["hub-router", "spoke-service-a", "spoke-service-b", "spoke-service-c"],
        "topics": ["hub.ingress", "hub.egress", "spoke.data"],
        "libraries": ["rpc-client", "hub-utils"],
        "brokers": ["activemq-main", "activemq-backup"]
    },
    "microservices": {
        "nodes": ["k8s-worker-1", "k8s-worker-2", "k8s-worker-3", "db-node"],
        "apps": ["auth-svc", "user-svc", "payment-svc", "product-svc", "email-svc", "search-svc"],
        "topics": ["user.events", "payment.events", "product.events", "email.queue"],
        "libraries": ["grpc-lib", "service-mesh-proxy", "tracing-lib", "metrics-sdk"],
        "brokers": ["kafka-events", "redis-streams"]
    },
    "enterprise": {
        "nodes": ["esb-node", "erp-node", "crm-node", "hr-node", "warehouse-node", "api-gateway"],
        "apps": ["esb-router", "erp-sync", "crm-manager", "hr-portal", "inventory-tracker", "invoice-service"],
        "topics": ["erp.updates", "crm.leads", "hr.onboarding", "warehouse.stock", "invoice.generated"],
        "libraries": ["soap-client", "xml-parser", "enterprise-auth", "legacy-db-driver"],
        "brokers": ["ibm-mq", "kafka-enterprise"]
    }
}

# Default QoS mappings based on topic name patterns within a scenario/domain
# Format: List of (pattern, (durability, reliability, transport_priority))
QOS_MAPPINGS = {
    "av": [
        ("camera", ("TRANSIENT_LOCAL", "RELIABLE", "HIGH")),
        ("lidar", ("TRANSIENT_LOCAL", "RELIABLE", "HIGH")),
        ("cmd", ("TRANSIENT_LOCAL", "RELIABLE", "HIGHEST")),
        ("goal", ("TRANSIENT_LOCAL", "RELIABLE", "HIGH")),
        ("default", ("TRANSIENT_LOCAL", "RELIABLE", "HIGH"))
    ],
    "iot": [
        ("traffic", ("VOLATILE", "BEST_EFFORT", "LOW")),
        ("air", ("VOLATILE", "BEST_EFFORT", "LOW")),
        ("power", ("VOLATILE", "BEST_EFFORT", "LOW")),
        ("cmd", ("VOLATILE", "RELIABLE", "MEDIUM")),
        ("alert", ("TRANSIENT_LOCAL", "RELIABLE", "HIGH")),
        ("default", ("VOLATILE", "BEST_EFFORT", "LOW"))
    ],
    "finance": [
        ("quotes", ("VOLATILE", "BEST_EFFORT", "HIGH")),
        ("trades", ("PERSISTENT", "RELIABLE", "CRITICAL")),
        ("orders", ("PERSISTENT", "RELIABLE", "CRITICAL")),
        ("ledger", ("PERSISTENT", "RELIABLE", "CRITICAL")),
        ("risk", ("TRANSIENT_LOCAL", "RELIABLE", "HIGH")),
        ("default", ("PERSISTENT", "RELIABLE", "HIGH"))
    ],
    "healthcare": [
        ("vitals", ("VOLATILE", "BEST_EFFORT", "MEDIUM")),
        ("hl7", ("PERSISTENT", "RELIABLE", "HIGH")),
        ("dicom", ("PERSISTENT", "RELIABLE", "MEDIUM")),
        ("billing", ("PERSISTENT", "RELIABLE", "HIGH")),
        ("appointments", ("PERSISTENT", "RELIABLE", "MEDIUM")),
        ("default", ("PERSISTENT", "RELIABLE", "HIGH"))
    ],
    "hub-and-spoke": [
        ("default", ("TRANSIENT_LOCAL", "RELIABLE", "MEDIUM"))
    ],
    "microservices": [
        ("default", ("TRANSIENT_LOCAL", "RELIABLE", "MEDIUM"))
    ],
    "enterprise": [
        ("default", ("TRANSIENT_LOCAL", "RELIABLE", "MEDIUM"))
    ]
}

def get_scenario_for_topic(topic_name: str, domain: str, scenario: str = None) -> str:
    """Determine the scenario class for a given topic to assign QoS."""
    if scenario and scenario in QOS_MAPPINGS:
        return scenario
        
    if domain in QOS_MAPPINGS:
        return domain
        
    return "enterprise"

def get_qos_for_topic(topic_name: str, domain: str, scenario: str = None) -> Tuple[str, str, str]:
    """Return (durability, reliability, transport_priority) based on topic name and scenario."""
    active_scenario = get_scenario_for_topic(topic_name, domain, scenario)
    mappings = QOS_MAPPINGS.get(active_scenario, QOS_MAPPINGS["enterprise"])
    
    topic_lower = topic_name.lower()
    for pattern, qos_tuple in mappings:
        if pattern != "default" and pattern in topic_lower:
            return qos_tuple
            
    # Return default for this scenario
    for pattern, qos_tuple in mappings:
        if pattern == "default":
            return qos_tuple
            
    return ("TRANSIENT_LOCAL", "RELIABLE", "MEDIUM")


class DomainDataset:
    """Helper to sample realistic names from a domain dataset."""
    
    def __init__(self, domain: str, rng: random.Random):
        self.domain = domain
        self.rng = rng
        self.dataset = DOMAIN_DATASETS.get(domain, DOMAIN_DATASETS["enterprise"])
        
        # Track used root names to append numbers if we exhaust the list
        self.used_counts: Dict[str, Dict[str, int]] = {
            "nodes": {}, "brokers": {}, "topics": {}, "apps": {}, "libraries": {}
        }
        
    def _get_name(self, category: str, fallback_prefix: str) -> str:
        options = self.dataset.get(category, [])
        if not options:
            return f"{fallback_prefix}-{self.rng.randint(1000, 9999)}"
            
        base_name = self.rng.choice(options)
        count = self.used_counts[category].get(base_name, 0)
        self.used_counts[category][base_name] = count + 1
        
        if count == 0:
            return base_name
        return f"{base_name}-{count}"
        
    def get_node_name(self) -> str:
        return self._get_name("nodes", "node")
        
    def get_broker_name(self) -> str:
        return self._get_name("brokers", "broker")
        
    def get_topic_name(self) -> str:
        return self._get_name("topics", "topic")
        
    def get_app_name(self) -> str:
        return self._get_name("apps", "app")
        
    def get_library_name(self) -> str:
        return self._get_name("libraries", "lib")
