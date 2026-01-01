"""
Graph Generator - Version 5.0

Generates realistic pub-sub system graphs for testing and validation.

Features:
- Multiple scale presets (tiny, small, medium, large, xlarge)
- Domain-specific scenarios (iot, financial, healthcare, etc.)
- Configurable anti-patterns for testing
- QoS-aware topic generation
- Deterministic generation with seed support

Usage:
    from src.core import generate_graph
    
    graph = generate_graph(scale="medium", scenario="iot", seed=42)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# Scale Presets
# =============================================================================

SCALE_PRESETS = {
    "tiny": {"apps": 5, "brokers": 1, "topics": 8, "nodes": 2},
    "small": {"apps": 10, "brokers": 2, "topics": 20, "nodes": 4},
    "medium": {"apps": 30, "brokers": 4, "topics": 60, "nodes": 8},
    "large": {"apps": 100, "brokers": 8, "topics": 200, "nodes": 20},
    "xlarge": {"apps": 300, "brokers": 16, "topics": 600, "nodes": 50},
}

# Scenario-specific configurations
SCENARIOS = {
    "generic": {
        "topic_prefixes": ["data", "events", "commands", "status"],
        "app_prefixes": ["service", "worker", "processor", "handler"],
        "qos_distribution": {"reliable": 0.3, "persistent": 0.2},
    },
    "iot": {
        "topic_prefixes": ["sensor", "telemetry", "actuator", "alert", "device"],
        "app_prefixes": ["collector", "aggregator", "controller", "monitor"],
        "qos_distribution": {"reliable": 0.4, "persistent": 0.3},
    },
    "financial": {
        "topic_prefixes": ["market", "orders", "trades", "quotes", "risk"],
        "app_prefixes": ["trading", "pricing", "risk", "execution", "analytics"],
        "qos_distribution": {"reliable": 0.8, "persistent": 0.5},
    },
    "healthcare": {
        "topic_prefixes": ["patient", "vitals", "alerts", "records", "monitor"],
        "app_prefixes": ["monitor", "recorder", "alerter", "analytics"],
        "qos_distribution": {"reliable": 0.9, "persistent": 0.7},
    },
    "autonomous_vehicle": {
        "topic_prefixes": ["lidar", "camera", "radar", "cmd_vel", "odom", "nav"],
        "app_prefixes": ["perception", "planning", "control", "localization"],
        "qos_distribution": {"reliable": 0.6, "persistent": 0.1},
    },
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GraphConfig:
    """Configuration for graph generation."""
    
    # Scale
    num_applications: int = 30
    num_brokers: int = 4
    num_topics: int = 60
    num_nodes: int = 8
    
    # Connectivity
    pub_probability: float = 0.15
    sub_probability: float = 0.25
    min_publishers_per_topic: int = 1
    min_subscribers_per_topic: int = 1
    
    # Scenario
    scenario: str = "generic"
    
    # QoS distribution
    reliable_probability: float = 0.3
    persistent_probability: float = 0.2
    
    # Anti-patterns
    antipatterns: List[str] = field(default_factory=list)
    
    # Randomization
    seed: Optional[int] = None

    @classmethod
    def from_scale(cls, scale: str, scenario: str = "generic", **kwargs) -> GraphConfig:
        """Create config from scale preset."""
        preset = SCALE_PRESETS.get(scale, SCALE_PRESETS["medium"])
        scenario_config = SCENARIOS.get(scenario, SCENARIOS["generic"])
        
        return cls(
            num_applications=preset["apps"],
            num_brokers=preset["brokers"],
            num_topics=preset["topics"],
            num_nodes=preset["nodes"],
            scenario=scenario,
            reliable_probability=scenario_config["qos_distribution"]["reliable"],
            persistent_probability=scenario_config["qos_distribution"]["persistent"],
            **kwargs,
        )


# =============================================================================
# Graph Generator
# =============================================================================

class GraphGenerator:
    """Generates realistic pub-sub system graphs."""
    
    def __init__(self, config: GraphConfig) -> None:
        """Initialize generator with configuration."""
        self.config = config
        self.rng = random.Random(config.seed)
        self.scenario_config = SCENARIOS.get(config.scenario, SCENARIOS["generic"])
    
    def generate(self) -> Dict[str, Any]:
        """Generate a complete graph."""
        # Generate components
        nodes = self._generate_nodes()
        brokers = self._generate_brokers()
        topics = self._generate_topics()
        applications = self._generate_applications()
        
        # Generate relationships
        runs_on = self._generate_runs_on(applications, brokers, nodes)
        routes = self._generate_routes(brokers, topics)
        pub_sub = self._generate_pub_sub(applications, topics)
        connects = self._generate_connects(nodes)
        
        return {
            "metadata": {
                "scale": self._get_scale_name(),
                "scenario": self.config.scenario,
                "seed": self.config.seed,
            },
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": applications,
            "relationships": {
                "runs_on": runs_on,
                "routes": routes,
                "publishes_to": pub_sub["publishes"],
                "subscribes_to": pub_sub["subscribes"],
                "connects_to": connects,
            },
        }
    
    def _get_scale_name(self) -> str:
        """Determine scale name from config."""
        for name, preset in SCALE_PRESETS.items():
            if preset["apps"] == self.config.num_applications:
                return name
        return "custom"
    
    def _generate_nodes(self) -> List[Dict]:
        """Generate infrastructure nodes."""
        return [
            {"id": f"N{i}", "name": f"Node {i}"}
            for i in range(self.config.num_nodes)
        ]
    
    def _generate_brokers(self) -> List[Dict]:
        """Generate brokers."""
        return [
            {"id": f"B{i}", "name": f"Broker {i}"}
            for i in range(self.config.num_brokers)
        ]
    
    def _generate_topics(self) -> List[Dict]:
        """Generate topics with QoS."""
        prefixes = self.scenario_config["topic_prefixes"]
        topics = []
        
        for i in range(self.config.num_topics):
            prefix = self.rng.choice(prefixes)
            
            # Determine QoS
            reliability = (
                "RELIABLE"
                if self.rng.random() < self.config.reliable_probability
                else "BEST_EFFORT"
            )
            
            if self.rng.random() < self.config.persistent_probability:
                durability = "PERSISTENT"
            elif self.rng.random() < 0.3:
                durability = "TRANSIENT"
            else:
                durability = "VOLATILE"
            
            priority = self.rng.choice(["LOW", "MEDIUM", "MEDIUM", "HIGH", "URGENT"])
            size = self.rng.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
            
            topics.append({
                "id": f"T{i}",
                "name": f"/{prefix}/{prefix}_{i}",
                "size": size,
                "qos": {
                    "durability": durability,
                    "reliability": reliability,
                    "transport_priority": priority,
                },
            })
        
        return topics
    
    def _generate_applications(self) -> List[Dict]:
        """Generate applications."""
        prefixes = self.scenario_config["app_prefixes"]
        apps = []
        
        for i in range(self.config.num_applications):
            prefix = self.rng.choice(prefixes)
            role = self.rng.choice(["pub", "sub", "pubsub"])
            
            apps.append({
                "id": f"A{i}",
                "name": f"{prefix}_{i}",
                "role": role,
            })
        
        return apps
    
    def _generate_runs_on(
        self,
        applications: List[Dict],
        brokers: List[Dict],
        nodes: List[Dict],
    ) -> List[Dict]:
        """Generate RUNS_ON relationships."""
        runs_on = []
        
        # Applications run on nodes
        for app in applications:
            node = self.rng.choice(nodes)
            runs_on.append({"from": app["id"], "to": node["id"]})
        
        # Brokers run on nodes
        for broker in brokers:
            node = self.rng.choice(nodes)
            runs_on.append({"from": broker["id"], "to": node["id"]})
        
        return runs_on
    
    def _generate_routes(
        self,
        brokers: List[Dict],
        topics: List[Dict],
    ) -> List[Dict]:
        """Generate ROUTES relationships (broker -> topic)."""
        routes = []
        
        # Each topic is routed by at least one broker
        for topic in topics:
            # Pick 1-2 brokers to route this topic
            num_brokers = self.rng.randint(1, min(2, len(brokers)))
            selected = self.rng.sample(brokers, num_brokers)
            
            for broker in selected:
                routes.append({"from": broker["id"], "to": topic["id"]})
        
        return routes
    
    def _generate_pub_sub(
        self,
        applications: List[Dict],
        topics: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Generate PUBLISHES_TO and SUBSCRIBES_TO relationships."""
        publishes = []
        subscribes = []
        
        for topic in topics:
            # Ensure minimum publishers
            publishers = []
            pub_candidates = [a for a in applications if a["role"] in ("pub", "pubsub")]
            
            if pub_candidates:
                num_pubs = max(
                    self.config.min_publishers_per_topic,
                    int(len(pub_candidates) * self.config.pub_probability),
                )
                num_pubs = min(num_pubs, len(pub_candidates))
                publishers = self.rng.sample(pub_candidates, num_pubs)
            
            for app in publishers:
                publishes.append({"from": app["id"], "to": topic["id"]})
            
            # Ensure minimum subscribers
            sub_candidates = [a for a in applications if a["role"] in ("sub", "pubsub")]
            
            if sub_candidates:
                num_subs = max(
                    self.config.min_subscribers_per_topic,
                    int(len(sub_candidates) * self.config.sub_probability),
                )
                num_subs = min(num_subs, len(sub_candidates))
                subscribers = self.rng.sample(sub_candidates, num_subs)
            else:
                subscribers = []
            
            for app in subscribers:
                subscribes.append({"from": app["id"], "to": topic["id"]})
        
        return {"publishes": publishes, "subscribes": subscribes}
    
    def _generate_connects(self, nodes: List[Dict]) -> List[Dict]:
        """Generate CONNECTS_TO relationships (mesh network)."""
        connects = []
        
        # Create a connected mesh
        for i, node in enumerate(nodes):
            # Connect to next node (ring topology base)
            if i < len(nodes) - 1:
                connects.append({"from": node["id"], "to": nodes[i + 1]["id"]})
            
            # Add some random connections
            for other in nodes:
                if node["id"] != other["id"] and self.rng.random() < 0.2:
                    connects.append({"from": node["id"], "to": other["id"]})
        
        return connects


# =============================================================================
# Convenience Function
# =============================================================================

def generate_graph(
    scale: str = "medium",
    scenario: str = "generic",
    seed: Optional[int] = None,
    antipatterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a pub-sub system graph.
    
    Args:
        scale: Size preset (tiny, small, medium, large, xlarge)
        scenario: Domain scenario (generic, iot, financial, healthcare, etc.)
        seed: Random seed for reproducibility
        antipatterns: List of anti-patterns to inject
    
    Returns:
        Dictionary with graph data ready for import
    
    Example:
        graph = generate_graph(scale="medium", scenario="iot", seed=42)
    """
    config = GraphConfig.from_scale(
        scale=scale,
        scenario=scenario,
        seed=seed,
        antipatterns=antipatterns or [],
    )
    
    generator = GraphGenerator(config)
    return generator.generate()