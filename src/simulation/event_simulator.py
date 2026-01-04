"""
Event Simulator

Simulates event propagation using raw Pub-Sub topology.
Path: Publisher -> Topic -> Subscriber
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Set
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
            "max_hops": self.hops
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

        # 1. Identify Topics published to by the Source App
        # (Source)-[:PUBLISHES_TO]->(Topic)
        published_topics = self.graph.get_successors_by_type(source, "PUBLISHES_TO")
        
        reached_subscribers = set()
        hops = 0
        
        if published_topics:
            hops = 1
            # 2. Identify Subscribers for these Topics
            # (Topic)<-[:SUBSCRIBES_TO]-(Subscriber)
            # In our directed graph, the edge is usually Subscriber->Topic for 'SUBSCRIBES_TO'.
            # We need to find nodes that have an outgoing SUBSCRIBES_TO edge to these topics.
            # i.e., Predecessors of Topic via SUBSCRIBES_TO.
            
            for topic in published_topics:
                # Mark Topic as active/loaded
                self.graph.graph.nodes[topic]["load"] += 1
                
                subscribers = self.graph.get_predecessors_by_type(topic, "SUBSCRIBES_TO")
                if subscribers:
                    hops = 2
                    reached_subscribers.update(subscribers)
                    
                    for sub in subscribers:
                        self.graph.graph.nodes[sub]["load"] += 1

        return EventResult(
            scenario=scenario.description,
            source=source,
            reached_subscribers=list(reached_subscribers),
            affected_topics=published_topics,
            hops=hops
        )