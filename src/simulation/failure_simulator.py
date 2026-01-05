"""
Failure Simulator

Simulates cascading failures and calculates impact scores based on the PhD methodology.

Logic:
1. Cascade Propagation: Based on DEPENDS_ON weights, thresholds, and probabilities.
2. Impact Calculation: Reachability Loss, Fragmentation, Cascade Extent.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Set
from .simulation_graph import SimulationGraph

@dataclass
class FailureScenario:
    target_node: str
    description: str
    cascade_threshold: float = 0.5  # Default from Report Section 4.4.2
    cascade_probability: float = 0.7 # Default from Report Section 4.4.2
    max_depth: int = 5               # Default from Report Section 4.4.2

@dataclass
class FailureResult:
    scenario: str
    initial_failure: str
    cascaded_failures: List[str]
    
    # Metrics from Report Section 4.4.1 [Formula 7]
    reachability_loss: float
    fragmentation: float
    cascade_extent: float
    impact_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "initial_failure": self.initial_failure,
            "cascaded_count": len(self.cascaded_failures),
            "metrics": {
                "reachability_loss": round(self.reachability_loss, 4),
                "fragmentation": round(self.fragmentation, 4),
                "cascade_extent": round(self.cascade_extent, 4),
                "total_impact_score": round(self.impact_score, 4)
            },
            "cascaded_nodes": self.cascaded_failures
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
            self.logger.warning(f"Target node {target} not found in graph.")
            return FailureResult(scenario.description, target, [], 0.0, 0.0, 0.0, 0.0)

        # --- 1. Cascade Propagation (Report Section 4.4.2) ---
        failed_set = set()
        queue = [(target, 0)] # (node_id, depth)
        
        # Mark initial failure
        failed_set.add(target)
        G.nodes[target]["state"] = "failed"
        
        cascaded_failures = []

        while queue:
            current_node, depth = queue.pop(0)
            
            if depth >= scenario.max_depth:
                continue

            # Check dependencies: If 'current_node' fails, who depends on it?
            # In dependency graph: Dependent -> Dependency.
            # If Dependency (current_node) fails, Dependent (predecessor) might fail.
            dependents = self.graph.get_predecessors_by_type(current_node, "DEPENDS_ON")
            
            for dep in dependents:
                if dep in failed_set:
                    continue
                
                # Get Dependency Strength (Formula 1) stored on edge
                weight = self.graph.get_edge_weight(dep, current_node)
                
                # Cascade Logic: Threshold & Probability
                if weight >= scenario.cascade_threshold:
                    if random.random() < scenario.cascade_probability:
                        failed_set.add(dep)
                        cascaded_failures.append(dep)
                        G.nodes[dep]["state"] = "failed"
                        queue.append((dep, depth + 1))

        # --- 2. Impact Calculation (Report Section 4.4.1) ---
        total_nodes = G.number_of_nodes()
        remaining_components = total_nodes - 1 # Excluding the initial failure trigger
        
        if remaining_components <= 0:
            cascade_extent = 1.0
        else:
            # Cascade Extent = |cascade_failures| / |remaining_components|
            cascade_extent = len(cascaded_failures) / remaining_components

        # Reachability Loss = |broken_paths| / |total_paths|
        initial_paths = self.graph.initial_paths
        total_paths_count = len(initial_paths)
        
        if total_paths_count == 0:
            reachability_loss = 0.0
        else:
            # A path is broken if any node in the (Pub, Topic, Sub) triple is failed
            broken_paths = 0
            for pub, topic, sub in initial_paths:
                if pub in failed_set or topic in failed_set or sub in failed_set:
                    broken_paths += 1
            reachability_loss = broken_paths / total_paths_count

        # Fragmentation = (C_after - C_before) / (n - 1)
        c_before = self.graph.initial_component_count
        c_after = self.graph.get_connected_components_count(active_only=True)
        
        if total_nodes <= 1:
            fragmentation = 0.0
        else:
            fragmentation = max(0, (c_after - c_before) / (total_nodes - 1))

        # Composite Impact Score (Formula 7)
        # Impact(v) = 0.5 * reachability + 0.3 * fragmentation + 0.2 * cascade
        impact_score = (0.5 * reachability_loss) + \
                       (0.3 * fragmentation) + \
                       (0.2 * cascade_extent)

        return FailureResult(
            scenario=scenario.description,
            initial_failure=target,
            cascaded_failures=cascaded_failures,
            reachability_loss=reachability_loss,
            fragmentation=fragmentation,
            cascade_extent=cascade_extent,
            impact_score=impact_score
        )