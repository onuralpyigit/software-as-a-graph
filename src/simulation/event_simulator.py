"""
Event Simulator

Simulates event propagation using raw Pub-Sub topology.
Path: Publisher -> Topic -> Subscriber
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from .simulation_graph import SimulationGraph

@dataclass
class EventScenario:
    source_node: str
    description: str

@dataclass
class EventResult:
    scenario: str
    source: str
    reached_subscribers: List[str]
    affected_topics: List[str]
    hops: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "source": self.source,
            "reached_count": len(self.reached_subscribers),
            "topics_traversed": len(self.affected_topics),
            "max_hops": self.hops,
            "reached_subscribers": self.reached_subscribers
        }

class EventSimulator:
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

    def simulate(self, scenario: EventScenario) -> EventResult:
        self.graph.reset()
        source = scenario.source_node
        
        if source not in self.graph.graph:
            self.logger.error(f"Source {source} not found")
            return EventResult(scenario.description, source, [], [], 0)

        # Logic: Publisher -> Topic -> Subscriber
        # 1. Find Topics the source PUBLISHES_TO
        # (Source)-[:PUBLISHES_TO]->(Topic)
        # Note: In SimulationGraph.get_successors_by_type, we look for outgoing edges.
        # Graph Model: (App)-[:PUBLISHES_TO]->(Topic)
        published_topics = self.graph.get_successors_by_type(source, "PUBLISHES_TO")
        
        reached_subscribers = set()
        hops = 0
        
        if published_topics:
            hops = 1
            for topic in published_topics:
                # Mark load
                self.graph.graph.nodes[topic]["load"] += 1
                
                # 2. Find Subscribers that SUBSCRIBE_TO this Topic
                # Graph Model: (App)-[:SUBSCRIBES_TO]->(Topic)
                # So we look for PREDECESSORS of the Topic via SUBSCRIBES_TO
                subscribers = self.graph.get_predecessors_by_type(topic, "SUBSCRIBES_TO")
                
                if subscribers:
                    hops = 2
                    for sub in subscribers:
                        # Don't count the source if it subscribes to its own topic (echo)
                        if sub != source:
                            reached_subscribers.add(sub)
                            self.graph.graph.nodes[sub]["load"] += 1

        return EventResult(
            scenario=scenario.description,
            source=source,
            reached_subscribers=list(reached_subscribers),
            affected_topics=published_topics,
            hops=hops
        )