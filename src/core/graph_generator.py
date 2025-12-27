"""
Graph Generator - Refactored Version 4.0

Generates realistic pub-sub system graphs with:
- Configurable scale presets (tiny to extreme)
- Domain-specific scenarios (IoT, financial, healthcare, etc.)
- Anti-pattern injection for testing
- Reproducible generation with seeds

Usage:
    from src.core.graph_generator import generate_graph, GraphConfig
    
    # Simple usage
    graph = generate_graph(scale="medium", scenario="iot")
    
    # With configuration
    config = GraphConfig(scale="large", scenario="financial", seed=42)
    generator = GraphGenerator(config)
    graph = generator.generate()

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

# Scale presets: (nodes, apps, topics, brokers)
SCALE_PRESETS = {
    "tiny": (2, 5, 3, 1),
    "small": (4, 15, 10, 2),
    "medium": (8, 40, 25, 4),
    "large": (16, 100, 60, 8),
    "xlarge": (32, 250, 150, 16),
    "extreme": (64, 500, 300, 32),
}

# QoS profiles by scenario
QOS_PROFILES = {
    "default": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "priority": "MEDIUM"},
    "reliable": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "priority": "MEDIUM"},
    "persistent": {"durability": "PERSISTENT", "reliability": "RELIABLE", "priority": "HIGH"},
    "realtime": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "priority": "URGENT"},
    "critical": {"durability": "PERSISTENT", "reliability": "RELIABLE", "priority": "URGENT"},
}

# Scenario-specific topic templates
SCENARIO_TOPICS = {
    "generic": [
        ("events/{type}", "default"),
        ("commands/{action}", "reliable"),
        ("status/{component}", "default"),
        ("data/{stream}", "default"),
    ],
    "iot": [
        ("sensor/{type}/data", "default"),
        ("device/{id}/status", "default"),
        ("alerts/{priority}", "reliable"),
        ("commands/{device}", "reliable"),
        ("telemetry/{zone}", "default"),
    ],
    "financial": [
        ("market/{symbol}/quotes", "realtime"),
        ("orders/{type}", "persistent"),
        ("trades/{venue}", "persistent"),
        ("risk/{metric}", "reliable"),
        ("audit/{event}", "persistent"),
    ],
    "healthcare": [
        ("patient/{id}/vitals", "critical"),
        ("alerts/clinical", "critical"),
        ("devices/{type}/data", "reliable"),
        ("orders/medication", "persistent"),
        ("records/{type}", "persistent"),
    ],
    "autonomous_vehicle": [
        ("perception/{sensor}", "realtime"),
        ("planning/trajectory", "realtime"),
        ("control/commands", "critical"),
        ("safety/alerts", "critical"),
        ("localization/pose", "realtime"),
    ],
    "smart_city": [
        ("traffic/{intersection}", "reliable"),
        ("parking/{zone}/status", "default"),
        ("energy/{grid}/usage", "reliable"),
        ("emergency/dispatch", "critical"),
        ("weather/forecast", "default"),
    ],
}


@dataclass
class GraphConfig:
    """Configuration for graph generation"""
    scale: str = "medium"
    scenario: str = "generic"
    seed: int = 42
    
    # Optional overrides
    num_nodes: Optional[int] = None
    num_applications: Optional[int] = None
    num_topics: Optional[int] = None
    num_brokers: Optional[int] = None
    
    # Anti-patterns to inject
    antipatterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.scale not in SCALE_PRESETS:
            raise ValueError(f"Invalid scale '{self.scale}'. Valid: {list(SCALE_PRESETS.keys())}")
        
        valid_scenarios = list(SCENARIO_TOPICS.keys())
        if self.scenario not in valid_scenarios:
            raise ValueError(f"Invalid scenario '{self.scenario}'. Valid: {valid_scenarios}")
        
        valid_antipatterns = ["spof", "god_topic", "chatty", "bottleneck"]
        for ap in self.antipatterns:
            if ap not in valid_antipatterns:
                raise ValueError(f"Invalid antipattern '{ap}'. Valid: {valid_antipatterns}")


# =============================================================================
# Graph Generator
# =============================================================================

class GraphGenerator:
    """
    Generates realistic pub-sub system graphs.
    
    The generator creates:
    1. Infrastructure nodes
    2. Message brokers distributed across nodes
    3. Topics with scenario-appropriate QoS settings
    4. Applications with publisher/subscriber roles
    5. All necessary relationships
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        random.seed(config.seed)
        
        # Get counts from scale or overrides
        preset = SCALE_PRESETS[config.scale]
        self.num_nodes = config.num_nodes or preset[0]
        self.num_apps = config.num_applications or preset[1]
        self.num_topics = config.num_topics or preset[2]
        self.num_brokers = config.num_brokers or preset[3]
        
        # Get scenario templates
        self.topic_templates = SCENARIO_TOPICS.get(config.scenario, SCENARIO_TOPICS["generic"])

    def generate(self) -> Dict[str, Any]:
        """
        Generate the complete graph.
        
        Returns:
            Dictionary with vertices, relationships, and metadata
        """
        start_time = datetime.utcnow()
        
        # Generate vertices
        nodes = self._generate_nodes()
        brokers = self._generate_brokers()
        topics = self._generate_topics()
        applications = self._generate_applications()
        
        # Generate relationships
        relationships = {
            "publishes_to": [],
            "subscribes_to": [],
            "routes": [],
            "runs_on": [],
            "connects_to": [],
        }
        
        self._generate_infrastructure(nodes, brokers, relationships)
        self._generate_routing(brokers, topics, relationships)
        self._generate_pubsub(applications, topics, relationships)
        self._ensure_connectivity(applications, topics, relationships)
        
        # Apply anti-patterns
        antipatterns_applied = {}
        for ap in self.config.antipatterns:
            result = self._apply_antipattern(ap, applications, topics, relationships)
            if result:
                antipatterns_applied[ap] = result
        
        # Build final graph
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        graph = {
            "metadata": {
                "id": f"graph_{self.config.scenario}_{self.config.scale}_{self.config.seed}",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "generator_version": "4.0",
                "scale": self.config.scale,
                "scenario": self.config.scenario,
                "seed": self.config.seed,
                "generation_time_seconds": generation_time,
                "antipatterns_applied": antipatterns_applied or None,
            },
            "applications": applications,
            "brokers": brokers,
            "topics": topics,
            "nodes": nodes,
            "relationships": relationships,
        }
        
        # Add metrics
        graph["metrics"] = self._calculate_metrics(graph)
        
        self.logger.info(
            f"Generated: {len(nodes)} nodes, {len(applications)} apps, "
            f"{len(topics)} topics, {len(brokers)} brokers in {generation_time:.3f}s"
        )
        
        return graph

    # -------------------------------------------------------------------------
    # Vertex Generation
    # -------------------------------------------------------------------------

    def _generate_nodes(self) -> List[Dict]:
        """Generate infrastructure nodes"""
        return [{"id": f"N{i+1}", "name": f"Node{i+1}"} for i in range(self.num_nodes)]

    def _generate_brokers(self) -> List[Dict]:
        """Generate message brokers"""
        return [{"id": f"B{i+1}", "name": f"Broker{i+1}"} for i in range(self.num_brokers)]

    def _generate_topics(self) -> List[Dict]:
        """Generate topics with scenario-appropriate QoS"""
        topics = []
        for i in range(self.num_topics):
            # Select template based on index
            template, qos_profile = self.topic_templates[i % len(self.topic_templates)]
            
            # Generate topic name from template
            name = template.format(
                type=f"type{i+1}",
                id=f"id{i+1}",
                action=f"action{i+1}",
                component=f"component{i+1}",
                stream=f"stream{i+1}",
                device=f"device{i+1}",
                zone=f"zone{i+1}",
                priority=["low", "medium", "high"][i % 3],
                symbol=f"sym{i+1}",
                venue=f"venue{i+1}",
                metric=f"metric{i+1}",
                event=f"event{i+1}",
                sensor=["lidar", "camera", "radar"][i % 3],
                intersection=f"int{i+1}",
                grid=f"grid{i+1}",
            )
            
            # Get QoS profile
            qos = QOS_PROFILES.get(qos_profile, QOS_PROFILES["default"]).copy()
            
            # Vary message size based on topic type
            base_size = 256
            if "data" in name or "vitals" in name:
                base_size = 1024
            elif "commands" in name or "control" in name:
                base_size = 128
            elif "quotes" in name or "perception" in name:
                base_size = 512
            
            size = base_size + random.randint(-base_size // 4, base_size // 2)
            
            topics.append({
                "id": f"T{i+1}",
                "name": name,
                "size": max(64, size),
                "qos": qos,
            })
        
        return topics

    def _generate_applications(self) -> List[Dict]:
        """Generate applications with role distribution"""
        applications = []
        
        # Role distribution: ~30% pub, ~40% sub, ~30% pubsub
        roles = ["pub"] * int(self.num_apps * 0.3)
        roles += ["sub"] * int(self.num_apps * 0.4)
        roles += ["pubsub"] * (self.num_apps - len(roles))
        random.shuffle(roles)
        
        scenario_prefix = {
            "iot": ["Sensor", "Controller", "Gateway", "Monitor", "Actuator"],
            "financial": ["Trading", "Risk", "Market", "Settlement", "Audit"],
            "healthcare": ["Monitor", "Alert", "Device", "Record", "Order"],
            "autonomous_vehicle": ["Perception", "Planning", "Control", "Safety", "Localization"],
            "smart_city": ["Traffic", "Energy", "Parking", "Emergency", "Weather"],
        }.get(self.config.scenario, ["App", "Service", "Worker", "Handler", "Processor"])
        
        for i in range(self.num_apps):
            prefix = scenario_prefix[i % len(scenario_prefix)]
            applications.append({
                "id": f"A{i+1}",
                "name": f"{prefix}{i+1}",
                "role": roles[i],
            })
        
        return applications

    # -------------------------------------------------------------------------
    # Relationship Generation
    # -------------------------------------------------------------------------

    def _generate_infrastructure(self, nodes: List[Dict], brokers: List[Dict], 
                                  relationships: Dict) -> None:
        """Generate RUNS_ON and CONNECTS_TO relationships"""
        node_ids = [n["id"] for n in nodes]
        
        # Distribute brokers across nodes (round-robin with some randomness)
        for i, broker in enumerate(brokers):
            node_id = node_ids[i % len(node_ids)]
            relationships["runs_on"].append({"from": broker["id"], "to": node_id})
        
        # Create node mesh connectivity (each node connects to 2-3 others)
        for i, node in enumerate(nodes):
            # Connect to next node (ring topology base)
            next_idx = (i + 1) % len(nodes)
            if next_idx != i:
                relationships["connects_to"].append({
                    "from": node["id"],
                    "to": nodes[next_idx]["id"],
                })
            
            # Add random connections for mesh
            if len(nodes) > 3:
                extra_connections = random.randint(1, min(2, len(nodes) - 2))
                candidates = [n["id"] for n in nodes if n["id"] != node["id"]]
                for target in random.sample(candidates, min(extra_connections, len(candidates))):
                    if not any(r["from"] == node["id"] and r["to"] == target 
                              for r in relationships["connects_to"]):
                        relationships["connects_to"].append({
                            "from": node["id"],
                            "to": target,
                        })

    def _generate_routing(self, brokers: List[Dict], topics: List[Dict],
                          relationships: Dict) -> None:
        """Generate ROUTES relationships (broker -> topic)"""
        broker_ids = [b["id"] for b in brokers]
        
        for topic in topics:
            # Each topic routed by 1-2 brokers
            num_brokers = min(random.randint(1, 2), len(broker_ids))
            for broker_id in random.sample(broker_ids, num_brokers):
                relationships["routes"].append({
                    "from": broker_id,
                    "to": topic["id"],
                })

    def _generate_pubsub(self, applications: List[Dict], topics: List[Dict],
                         relationships: Dict) -> None:
        """Generate PUBLISHES_TO, SUBSCRIBES_TO, and RUNS_ON relationships"""
        topic_ids = [t["id"] for t in topics]
        
        for app in applications:
            role = app["role"]
            app_id = app["id"]
            
            # Determine number of topics based on role
            if role == "pub":
                num_pub = random.randint(1, max(1, len(topic_ids) // 5))
                num_sub = 0
            elif role == "sub":
                num_pub = 0
                num_sub = random.randint(1, max(1, len(topic_ids) // 4))
            else:  # pubsub
                num_pub = random.randint(1, max(1, len(topic_ids) // 6))
                num_sub = random.randint(1, max(1, len(topic_ids) // 5))
            
            # Create publish relationships
            pub_topics = random.sample(topic_ids, min(num_pub, len(topic_ids)))
            for topic_id in pub_topics:
                relationships["publishes_to"].append({"from": app_id, "to": topic_id})
            
            # Create subscribe relationships (avoid subscribing to own publications)
            available_for_sub = [t for t in topic_ids if t not in pub_topics]
            if available_for_sub:
                sub_topics = random.sample(available_for_sub, min(num_sub, len(available_for_sub)))
                for topic_id in sub_topics:
                    relationships["subscribes_to"].append({"from": app_id, "to": topic_id})

    def _ensure_connectivity(self, applications: List[Dict], topics: List[Dict],
                             relationships: Dict) -> None:
        """Ensure all topics have at least one publisher and subscriber"""
        app_ids = [a["id"] for a in applications]
        pub_apps = [a["id"] for a in applications if a["role"] in ("pub", "pubsub")]
        sub_apps = [a["id"] for a in applications if a["role"] in ("sub", "pubsub")]
        
        published_topics = {r["to"] for r in relationships["publishes_to"]}
        subscribed_topics = {r["to"] for r in relationships["subscribes_to"]}
        
        for topic in topics:
            topic_id = topic["id"]
            
            # Ensure at least one publisher
            if topic_id not in published_topics and pub_apps:
                publisher = random.choice(pub_apps)
                relationships["publishes_to"].append({"from": publisher, "to": topic_id})
            
            # Ensure at least one subscriber
            if topic_id not in subscribed_topics and sub_apps:
                subscriber = random.choice(sub_apps)
                relationships["subscribes_to"].append({"from": subscriber, "to": topic_id})
        
        # Distribute apps across nodes
        nodes_with_apps = {r["to"] for r in relationships["runs_on"] 
                          if r["from"].startswith("A")}
        all_nodes = {r["to"] for r in relationships["runs_on"]}
        
        for app in applications:
            if not any(r["from"] == app["id"] for r in relationships["runs_on"]):
                # Assign to least loaded node
                node_loads = defaultdict(int)
                for r in relationships["runs_on"]:
                    if r["from"].startswith("A"):
                        node_loads[r["to"]] += 1
                
                if all_nodes:
                    target_node = min(all_nodes, key=lambda n: node_loads.get(n, 0))
                    relationships["runs_on"].append({"from": app["id"], "to": target_node})

    # -------------------------------------------------------------------------
    # Anti-patterns
    # -------------------------------------------------------------------------

    def _apply_antipattern(self, antipattern: str, applications: List[Dict],
                           topics: List[Dict], relationships: Dict) -> Optional[Dict]:
        """Apply an anti-pattern to the graph"""
        
        if antipattern == "god_topic":
            # Create a topic that many apps depend on
            god_topic = topics[0] if topics else None
            if god_topic:
                app_ids = [a["id"] for a in applications[:min(10, len(applications))]]
                for app_id in app_ids:
                    if not any(r["from"] == app_id and r["to"] == god_topic["id"] 
                              for r in relationships["subscribes_to"]):
                        relationships["subscribes_to"].append({
                            "from": app_id, "to": god_topic["id"]
                        })
                return {"topic": god_topic["id"], "subscribers": len(app_ids)}
        
        elif antipattern == "spof":
            # Make one broker route critical topics
            if len(topics) > 3:
                spof_broker = random.choice([b["id"] for b in self._generate_brokers()[:1]])
                critical_topics = [t["id"] for t in topics[:3]]
                # Remove other broker routes for these topics
                relationships["routes"] = [
                    r for r in relationships["routes"]
                    if r["to"] not in critical_topics or r["from"] == spof_broker
                ]
                return {"broker": spof_broker, "critical_topics": critical_topics}
        
        elif antipattern == "chatty":
            # Make some apps publish to many topics
            if applications and topics:
                chatty_app = applications[0]
                topic_ids = [t["id"] for t in topics]
                for topic_id in topic_ids[:min(10, len(topic_ids))]:
                    if not any(r["from"] == chatty_app["id"] and r["to"] == topic_id
                              for r in relationships["publishes_to"]):
                        relationships["publishes_to"].append({
                            "from": chatty_app["id"], "to": topic_id
                        })
                return {"app": chatty_app["id"], "topics": len(topic_ids)}
        
        return None

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def _calculate_metrics(self, graph: Dict) -> Dict:
        """Calculate graph metrics"""
        apps = graph["applications"]
        topics = graph["topics"]
        edges = graph["relationships"]
        
        # Publisher/subscriber analysis
        topic_pubs = defaultdict(int)
        topic_subs = defaultdict(int)
        for pub in edges["publishes_to"]:
            topic_pubs[pub["to"]] += 1
        for sub in edges["subscribes_to"]:
            topic_subs[sub["to"]] += 1
        
        # Role distribution
        role_counts = defaultdict(int)
        for app in apps:
            role_counts[app["role"]] += 1
        
        return {
            "vertex_counts": {
                "applications": len(apps),
                "brokers": len(graph["brokers"]),
                "topics": len(topics),
                "nodes": len(graph["nodes"]),
                "total": len(apps) + len(graph["brokers"]) + len(topics) + len(graph["nodes"]),
            },
            "edge_counts": {
                "publishes_to": len(edges["publishes_to"]),
                "subscribes_to": len(edges["subscribes_to"]),
                "routes": len(edges["routes"]),
                "runs_on": len(edges["runs_on"]),
                "connects_to": len(edges["connects_to"]),
                "total": sum(len(v) for v in edges.values()),
            },
            "pub_sub": {
                "avg_pubs_per_topic": sum(topic_pubs.values()) / len(topics) if topics else 0,
                "avg_subs_per_topic": sum(topic_subs.values()) / len(topics) if topics else 0,
                "max_fanout": max(topic_subs.values()) if topic_subs else 0,
            },
            "role_distribution": dict(role_counts),
        }


# =============================================================================
# Convenience Function
# =============================================================================

def generate_graph(
    scale: str = "medium",
    scenario: str = "generic",
    seed: int = 42,
    antipatterns: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to generate a graph.
    
    Args:
        scale: Size preset (tiny, small, medium, large, xlarge, extreme)
        scenario: Domain scenario (generic, iot, financial, healthcare, etc.)
        seed: Random seed for reproducibility
        antipatterns: List of anti-patterns to inject
        **kwargs: Additional GraphConfig parameters
    
    Returns:
        Generated graph dictionary
    
    Example:
        graph = generate_graph(scale="medium", scenario="iot")
    """
    config = GraphConfig(
        scale=scale,
        scenario=scenario,
        seed=seed,
        antipatterns=antipatterns or [],
        **kwargs,
    )
    generator = GraphGenerator(config)
    return generator.generate()