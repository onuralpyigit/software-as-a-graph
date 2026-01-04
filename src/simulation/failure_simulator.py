"""
Failure Simulator

Simulates cascading structural failures.
Logic:
1. Node Failure -> Hosted Components (Apps/Brokers) Fail (RUNS_ON)
2. Broker Failure -> Routed Topics become unreachable (ROUTES)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from .simulation_graph import SimulationGraph

@dataclass
class FailureScenario:
    target_node: str
    description: str

@dataclass
class FailureResult:
    scenario: str
    initial_failure: str
    cascaded_failures: List[str]
    impact_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "initial": self.initial_failure,
            "cascaded": self.cascaded_failures,
            "impact": self.impact_counts
        }

class FailureSimulator:
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

    def simulate(self, scenario: FailureScenario) -> FailureResult:
        self.graph.reset()
        target = scenario.target_node
        G = self.graph.graph
        
        if target not in G:
            return FailureResult(scenario.description, target, [], {})

        # Set initial state
        G.nodes[target]["state"] = "failed"
        failed_set = {target}
        cascaded = []
        
        # 1. Structural Cascade (Hard Dependencies)
        # Find components that RUNS_ON the target
        # Edge: (App)-[:RUNS_ON]->(Node)
        # If Node fails, we look for Predecessors via RUNS_ON
        hosted_components = self.graph.get_predecessors_by_type(target, "RUNS_ON")
        
        for comp in hosted_components:
            if comp not in failed_set:
                failed_set.add(comp)
                cascaded.append(comp)
                G.nodes[comp]["state"] = "failed"
        
        # 2. Functional Cascade (Broker -> Topic)
        # If a Broker fails, Topics it ROUTES might be affected.
        # Edge: (Broker)-[:ROUTES]->(Topic)
        # If Broker fails, Successors via ROUTES are affected.
        # Note: We don't mark Topic as "failed" (it's logical), but "unreachable".
        # For this sim, we'll list them as cascaded impact.
        if G.nodes[target].get("type") == "Broker" or any(G.nodes[c].get("type") == "Broker" for c in cascaded):
            # Check for topics routed by failed brokers
            failed_brokers = [n for n in failed_set if G.nodes[n].get("type") == "Broker"]
            for b in failed_brokers:
                routed_topics = self.graph.get_successors_by_type(b, "ROUTES")
                for t in routed_topics:
                    if t not in failed_set:
                        failed_set.add(t)
                        cascaded.append(t)
                        G.nodes[t]["state"] = "unreachable"

        # Categorize Impact
        impact = {}
        for n in failed_set:
            ctype = G.nodes[n].get("type", "Unknown")
            impact[ctype] = impact.get(ctype, 0) + 1

        return FailureResult(
            scenario=scenario.description,
            initial_failure=target,
            cascaded_failures=cascaded,
            impact_counts=impact
        )