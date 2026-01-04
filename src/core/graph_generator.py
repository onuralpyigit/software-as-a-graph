"""
Graph Generator

Generates realistic pub-sub system graphs with QoS properties
to enable weight calculation during import.
"""

import random
from typing import Dict, Any, List
from .graph_model import Application, Broker, Node, Topic, QoSPolicy

class GraphGenerator:
    def __init__(self, scale: str = "medium", seed: int = 42):
        self.rng = random.Random(seed)
        self.scale_config = self._get_scale_config(scale)

    def _get_scale_config(self, scale: str) -> Dict[str, int]:
        presets = {
            "tiny":   {"apps": 5,   "topics": 5,   "brokers": 1, "nodes": 2},
            "small":  {"apps": 15,  "topics": 10,  "brokers": 2, "nodes": 4},
            "medium": {"apps": 50,  "topics": 30,  "brokers": 3, "nodes": 8},
            "large":  {"apps": 150, "topics": 100, "brokers": 6, "nodes": 20},
            "xlarge": {"apps": 500, "topics": 300, "brokers": 10, "nodes": 50},
        }
        return presets.get(scale, presets["medium"])

    def generate(self) -> Dict[str, Any]:
        c = self.scale_config
        
        # 1. Generate Vertices using Models
        nodes = [Node(id=f"N{i}", name=f"Node-{i}") for i in range(c["nodes"])]
        brokers = [Broker(id=f"B{i}", name=f"Broker-{i}") for i in range(c["brokers"])]
        
        topics = []
        for i in range(c["topics"]):
            qos = QoSPolicy(
                durability=self.rng.choice(["VOLATILE", "TRANSIENT_LOCAL", "TRANSIENT", "PERSISTENT"]),
                reliability=self.rng.choice(["BEST_EFFORT", "RELIABLE"]),
                transport_priority=self.rng.choice(["LOW", "MEDIUM", "HIGH", "URGENT"])
            )
            topics.append(Topic(
                id=f"T{i}", 
                name=f"Topic-{i}", 
                size=self.rng.randint(64, 8192), 
                qos=qos
            ))

        apps = []
        for i in range(c["apps"]):
            apps.append(Application(
                id=f"A{i}", 
                name=f"App-{i}", 
                role=self.rng.choice(["pub", "sub", "pubsub"])
            ))

        # 2. Generate Basic Relationships
        # Helper to simplify edge creation
        def make_edge(src, tgt): return {"from": src.id, "to": tgt.id}

        # RUNS_ON: Apps/Brokers -> Nodes
        runs_on = []
        for comp in apps + brokers:
            host = self.rng.choice(nodes)
            runs_on.append(make_edge(comp, host))

        # ROUTES: Brokers -> Topics
        routes = []
        for topic in topics:
            broker = self.rng.choice(brokers)
            routes.append(make_edge(broker, topic))

        # PUB/SUB: Apps -> Topics
        publishes = []
        subscribes = []
        for topic in topics:
            pubs = self.rng.sample(apps, k=self.rng.randint(1, min(3, len(apps))))
            subs = self.rng.sample(apps, k=self.rng.randint(1, min(5, len(apps))))
            
            for p in pubs: publishes.append(make_edge(p, topic))
            for s in subs: subscribes.append(make_edge(s, topic))

        # CONNECTS_TO: Mesh links between Nodes
        connects = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.rng.random() < 0.3:
                    connects.append(make_edge(nodes[i], nodes[j]))

        # Serialize to JSON-compatible dict
        return {
            "metadata": {"scale": str(self.scale_config), "seed": str(self.rng.getrandbits(32))},
            "nodes": [n.to_dict() for n in nodes], 
            "brokers": [b.to_dict() for b in brokers], 
            "topics": [t.to_dict() for t in topics], 
            "applications": [a.to_dict() for a in apps],
            "relationships": {
                "runs_on": runs_on, 
                "routes": routes,
                "publishes_to": publishes, 
                "subscribes_to": subscribes,
                "connects_to": connects
            }
        }

def generate_graph(scale="medium", **kwargs) -> Dict[str, Any]:
    return GraphGenerator(scale=scale, **kwargs).generate()