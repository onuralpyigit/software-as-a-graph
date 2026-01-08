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
    subscriber_coverage: float # % of total subscribers in system
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "source": self.source,
            "metrics": {
                "reached_count": len(self.reached_subscribers),
                "topics_traversed": len(self.affected_topics),
                "max_hops": self.hops,
                "subscriber_coverage": round(self.subscriber_coverage, 4)
            },
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
            return EventResult(scenario.description, source, [], [], 0, 0.0)

        # 1. Identify Published Topics (Publisher -> Topic)
        published_topics = self.graph.get_successors_by_type(source, "PUBLISHES_TO")
        
        reached_subscribers = set()
        hops = 0
        
        if published_topics:
            hops = 1
            for topic in published_topics:
                # 2. Identify Subscribers (Topic -> Subscriber)
                # Note: In our model, SUBSCRIBES_TO is (Sub)->(Topic), so we look for predecessors of Topic
                subscribers = self.graph.get_predecessors_by_type(topic, "SUBSCRIBES_TO")
                
                if subscribers:
                    hops = 2
                    for sub in subscribers:
                        if sub != source: # Avoid echo
                            reached_subscribers.add(sub)
        
        # Calculate coverage
        all_apps = [n for n, d in self.graph.graph.nodes(data=True) if d.get("type") == "Application"]
        total_potential_subs = len(all_apps) - 1 # Exclude self
        
        coverage = 0.0
        if total_potential_subs > 0:
            coverage = len(reached_subscribers) / total_potential_subs

        return EventResult(
            scenario=scenario.description,
            source=source,
            reached_subscribers=list(reached_subscribers),
            affected_topics=published_topics,
            hops=hops,
            subscriber_coverage=coverage
        )