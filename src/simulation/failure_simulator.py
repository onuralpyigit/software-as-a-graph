"""
Failure Simulator

Simulates cascading failures in the system.
Uses probabilistic propagation based on edge weights.
"""

import random
import logging
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
from .simulation_graph import SimulationGraph

@dataclass
class FailureScenario:
    target_nodes: List[str]
    description: str

@dataclass
class FailureResult:
    scenario: str
    initial_failures: List[str]
    cascaded_failures: List[str]
    surviving_nodes: int
    total_impact: int
    propagation_steps: int
    affected_components: Dict[str, List[str]] = field(default_factory=dict) # Grouped by type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "initial_failures": self.initial_failures,
            "cascaded_failures": self.cascaded_failures,
            "stats": {
                "initial_count": len(self.initial_failures),
                "cascade_count": len(self.cascaded_failures),
                "total_impact": self.total_impact,
                "steps": self.propagation_steps
            },
            "affected_by_type": self.affected_components
        }

class FailureSimulator:
    def __init__(self, graph: SimulationGraph, propagation_threshold: float = 0.5):
        self.graph = graph
        self.threshold = propagation_threshold
        self.logger = logging.getLogger(__name__)

    def simulate(self, scenario: FailureScenario) -> FailureResult:
        """Run a cascading failure simulation."""
        self.graph.reset()
        G = self.graph.graph
        
        # Initial failures
        failed = set()
        queue = []
        
        # Validate initial nodes
        for node in scenario.target_nodes:
            if node in G:
                failed.add(node)
                G.nodes[node]["state"] = "failed"
                queue.append(node)
            else:
                self.logger.warning(f"Node {node} not found in graph.")

        initial_count = len(failed)
        steps = 0
        cascaded = []

        # BFS for Cascade
        while queue:
            steps += 1
            current_batch = queue[:]
            queue = []
            
            for node in current_batch:
                # Find dependents (who depends on 'node'?)
                # In DEPENDS_ON (A -> B), A depends on B.
                # If B fails, A might fail.
                # So we look for predecessors in the directed graph (A -> node).
                if G.is_directed():
                    dependents = list(G.predecessors(node))
                else:
                    dependents = list(G.neighbors(node))
                
                for dep in dependents:
                    if dep not in failed:
                        # Probability logic: weight * random factor
                        # High weight dependency = High chance of failure
                        edge_weight = self.graph.get_edge_weight(dep, node)
                        prob = edge_weight * random.uniform(0.5, 1.0)
                        
                        if prob > self.threshold:
                            failed.add(dep)
                            cascaded.append(dep)
                            G.nodes[dep]["state"] = "failed"
                            queue.append(dep)

        # Categorize results
        affected_by_type = {}
        for f_node in failed:
            n_type = G.nodes[f_node].get("type", "unknown")
            if n_type not in affected_by_type:
                affected_by_type[n_type] = []
            affected_by_type[n_type].append(f_node)

        return FailureResult(
            scenario=scenario.description,
            initial_failures=scenario.target_nodes,
            cascaded_failures=cascaded,
            surviving_nodes=G.number_of_nodes() - len(failed),
            total_impact=len(failed),
            propagation_steps=steps,
            affected_components=affected_by_type
        )