"""
Graph Generator

Generates realistic pub-sub system graphs with QoS properties.
"""

import random
from typing import Dict, Any, List, Union

from ..models import (
    Application,
    Broker,
    Node,
    Topic,
    Library,
    QoSPolicy,
    ApplicationType,
)

# Scale presets - defines component counts for different graph sizes
SCALE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":   {"apps": 5,   "topics": 5,   "brokers": 1, "nodes": 2,  "libs": 2},
    "small":  {"apps": 15,  "topics": 10,  "brokers": 2, "nodes": 4,  "libs": 5},
    "medium": {"apps": 50,  "topics": 30,  "brokers": 3, "nodes": 8,  "libs": 10},
    "large":  {"apps": 150, "topics": 100, "brokers": 6, "nodes": 20, "libs": 30},
    "xlarge": {"apps": 500, "topics": 300, "brokers": 10, "nodes": 50, "libs": 100},
}

# Valid options for random generation (derived from enums/constants)
DURABILITY_OPTIONS = list(QoSPolicy.DURABILITY_SCORES.keys())
RELIABILITY_OPTIONS = list(QoSPolicy.RELIABILITY_SCORES.keys())
PRIORITY_OPTIONS = list(QoSPolicy.PRIORITY_SCORES.keys())
ROLE_OPTIONS = ["pub", "sub", "pubsub"]
APP_TYPE_OPTIONS = [t.value for t in ApplicationType]


class GraphGenerator:
    """Generates realistic pub-sub system graphs with configurable scale."""
    
    def __init__(self, scale: str = "medium", seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.scale_config = self._get_scale_config(scale)

    def _get_scale_config(self, scale: str) -> Dict[str, int]:
        """Returns configuration for graph generation based on scale."""
        return SCALE_PRESETS.get(scale, SCALE_PRESETS["medium"])

    def generate(self) -> Dict[str, Any]:
        """Generate a complete graph with all components and relationships."""
        c = self.scale_config
        
        # 1. Generate Vertices using Models
        nodes = [Node(id=f"N{i}", name=f"Node-{i}") for i in range(c["nodes"])]
        brokers = [Broker(id=f"B{i}", name=f"Broker-{i}") for i in range(c["brokers"])]
        
        topics = [
            Topic(
                id=f"T{i}",
                name=f"Topic-{i}",
                size=self.rng.randint(64, 65536),
                qos=QoSPolicy(
                    durability=self.rng.choice(DURABILITY_OPTIONS),
                    reliability=self.rng.choice(RELIABILITY_OPTIONS),
                    transport_priority=self.rng.choice(PRIORITY_OPTIONS),
                ),
            )
            for i in range(c["topics"])
        ]

        apps = [
            Application(
                id=f"A{i}",
                name=f"App-{i}",
                role=self.rng.choice(ROLE_OPTIONS),
                app_type=self.rng.choice(APP_TYPE_OPTIONS),
                criticality=self.rng.choice([True, False]),
                version=f"{self.rng.randint(1, 3)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            )
            for i in range(c["apps"])
        ]

        libs = [
            Library(
                id=f"L{i}",
                name=f"Lib-{i}",
                version=f"{self.rng.randint(0, 2)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            )
            for i in range(c["libs"])
        ]

        # 2. Generate Relationships

        # RUNS_ON: Apps/Brokers -> Nodes
        runs_on = []
        for comp in apps + brokers:
            host = self.rng.choice(nodes)
            runs_on.append(self._make_edge(comp, host))

        # ROUTES: Brokers -> Topics
        routes = []
        for topic in topics:
            broker = self.rng.choice(brokers)
            routes.append(self._make_edge(broker, topic))

        # PUB/SUB: Apps/Libraries -> Topics
        publishes = []
        subscribes = []

        # Both Applications and Libraries can publish/subscribe
        potential_clients = apps + libs
        
        for topic in topics:
            # Randomly select a subset of clients for this topic
            k_pubs = self.rng.randint(1, max(2, min(5, len(potential_clients))))
            k_subs = self.rng.randint(1, max(2, min(8, len(potential_clients))))
            
            pubs = self.rng.sample(potential_clients, k=k_pubs)
            subs = self.rng.sample(potential_clients, k=k_subs)
            
            for p in pubs:
                publishes.append(self._make_edge(p, topic))
            for s in subs:
                subscribes.append(self._make_edge(s, topic))

        # USES: Application -> Library, Library -> Library
        uses = []
        
        # Apps using Libraries
        for app in apps:
            # Each app uses between 0 and 3 libraries
            if libs:
                n_uses = self.rng.randint(0, min(3, len(libs)))
                targets = self.rng.sample(libs, k=n_uses)
                for t in targets:
                    uses.append(self._make_edge(app, t))

        # Libraries using other Libraries (simple random edges, potential cycles allowed for stress test)
        for lib in libs:
            if len(libs) > 1 and self.rng.random() < 0.2:
                other = self.rng.choice(libs)
                if other != lib:
                    uses.append(self._make_edge(lib, other))

        # CONNECTS_TO: Mesh links between Nodes
        connects = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.rng.random() < 0.3:
                    connects.append(self._make_edge(nodes[i], nodes[j]))

        # Serialize to JSON-compatible dict
        return {
            "metadata": {"scale": str(self.scale_config), "seed": str(self.rng.getrandbits(32))},
            "nodes": [n.to_dict() for n in nodes], 
            "brokers": [b.to_dict() for b in brokers], 
            "topics": [t.to_dict() for t in topics], 
            "applications": [a.to_dict() for a in apps],
            "libraries": [l.to_dict() for l in libs],
            "relationships": {
                "runs_on": runs_on, 
                "routes": routes,
                "publishes_to": publishes, 
                "subscribes_to": subscribes,
                "connects_to": connects,
                "uses": uses
            }
        }

    @staticmethod
    def _make_edge(src: Union[Application, Broker, Node, Topic, Library],
                   tgt: Union[Application, Broker, Node, Topic, Library]) -> Dict[str, str]:
        """Create an edge dictionary from source to target."""
        return {"from": src.id, "to": tgt.id}


def generate_graph(scale: str = "medium", **kwargs: Any) -> Dict[str, Any]:
    """Convenience function to generate a graph with the given scale."""
    return GraphGenerator(scale=scale, **kwargs).generate()
