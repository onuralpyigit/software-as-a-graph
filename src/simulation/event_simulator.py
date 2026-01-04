"""
Event Simulator

Simulates event propagation (Pub-Sub message flow).
Identifies reachability and potential load bottlenecks.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
from .simulation_graph import SimulationGraph

@dataclass
class EventScenario:
    source_node: str
    description: str
    event_payload_size: float = 1.0

@dataclass
class EventResult:
    scenario: str
    source: str
    reached_nodes: List[str]
    unreachable_nodes: List[str]
    max_hops: int
    bottlenecks: List[str] # Nodes with high fan-in in the path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "source": self.source,
            "coverage": {
                "reached": len(self.reached_nodes),
                "unreachable": len(self.unreachable_nodes)
            },
            "max_hops": self.max_hops,
            "bottlenecks": self.bottlenecks
        }

class EventSimulator:
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

    def simulate(self, scenario: EventScenario) -> EventResult:
        """Run an event propagation simulation."""
        self.graph.reset()
        G = self.graph.graph
        
        if scenario.source_node not in G:
            self.logger.error(f"Source node {scenario.source_node} not found")
            return EventResult(scenario.description, scenario.source_node, [], [], 0, [])

        # BFS for Event Reachability
        # Flow follows direction of edges for message passing (Publisher -> Topic -> Subscriber)
        # Note: If graph is DEPENDS_ON, flow might be reverse of dependency.
        # Assuming DEPENDS_ON (A -> B) means A needs B.
        # Data flow usually goes B -> A (Service B provides data to A).
        # We will assume REVERSE of dependency for event flow unless strictly specified.
        # For Pub-Sub: Publisher -> Topic (Depends? usually Pub depends on Topic to send)
        # To be safe: We will assume standard graph direction is dependency. 
        # Event flow = Reverse of Dependency (Data flows from dependency to dependent).
        
        flow_graph = G.reverse() if G.is_directed() else G
        
        visited = {scenario.source_node}
        queue = [(scenario.source_node, 0)]
        reached_order = []
        max_hops = 0
        node_load = {n: 0 for n in G.nodes}
        
        while queue:
            curr, dist = queue.pop(0)
            max_hops = max(max_hops, dist)
            reached_order.append(curr)
            
            # Record load (Fan-in simulation)
            node_load[curr] += scenario.event_payload_size
            
            for neighbor in flow_graph.neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # Identify Bottlenecks (nodes in path with high degree/load)
        # Simple heuristic: Top 10% of visited nodes by original in-degree (fan-in)
        reached_nodes_set = set(reached_order)
        unreachable = [n for n in G.nodes if n not in reached_nodes_set]
        
        # Bottleneck detection: High degree nodes in the flow path
        potential_bottlenecks = []
        for n in reached_order:
            if G.degree(n) > 3: # Arbitrary threshold for demo
                potential_bottlenecks.append(n)

        return EventResult(
            scenario=scenario.description,
            source=scenario.source_node,
            reached_nodes=reached_order,
            unreachable_nodes=unreachable,
            max_hops=max_hops,
            bottlenecks=potential_bottlenecks[:5] # Top 5
        )