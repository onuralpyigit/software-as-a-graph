"""
Failure Simulator

Simulates cascading failures using RAW structural relationships.

Propagation Logic:
1. Physical Propagation (Hard): Node Failure -> Hosted Apps Failure (RUNS_ON).
2. Network Propagation (Soft/Probabilistic): Node -> Connected Node (CONNECTS_TO).
3. Impact Analysis: Evaluates Reachability Loss (Pub-Sub paths) and Fragmentation.
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
    layer: str = "complete" # application, infrastructure, complete
    cascade_threshold: float = 0.5
    cascade_probability: float = 0.7
    max_depth: int = 5

@dataclass
class FailureResult:
    scenario: str
    initial_failure: str
    failure_type: str  # e.g., "Node", "Application"
    cascaded_failures: List[str]
    reachability_loss: float
    fragmentation: float
    impact_score: float
    params: Dict[str, Any] # Store simulation params for dataset completeness

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "initial_failure": self.initial_failure,
            "failure_type": self.failure_type,
            "metrics": {
                "reachability_loss": round(self.reachability_loss, 4),
                "fragmentation": round(self.fragmentation, 4),
                "total_impact_score": round(self.impact_score, 4)
            },
            "cascaded_nodes": self.cascaded_failures,
            "cascaded_count": len(self.cascaded_failures)
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens result for CSV/DataFrame export."""
        return {
            "node_id": self.initial_failure,
            "node_type": self.failure_type,
            "reachability_loss": round(self.reachability_loss, 4),
            "fragmentation": round(self.fragmentation, 4),
            "impact_score": round(self.impact_score, 4),
            "cascaded_count": len(self.cascaded_failures),
            "cascade_threshold": self.params.get("threshold"),
            "cascade_probability": self.params.get("probability"),
            "cascade_depth": self.params.get("depth")
        }

class FailureSimulator:
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

    def simulate(self, scenario: FailureScenario) -> FailureResult:
        self.graph.reset()
        target = scenario.target_node
        G = self.graph.graph
        
        # Check if target exists
        if target not in G:
            self.logger.warning(f"Target node {target} not found in graph.")
            return FailureResult(scenario.description, target, "unknown", [], 0.0, 0.0, 0.0, {})

        # Determine target type
        target_type = G.nodes[target].get("type", "unknown")

        failed_set = set()
        queue = [(target, 0)] 
        failed_set.add(target)
        G.nodes[target]["state"] = "failed"
        
        cascaded_failures = []

        while queue:
            curr, depth = queue.pop(0)
            
            # --- 1. Physical Propagation (Hard Dependency) ---
            # If 'curr' is a Node, everything running on it fails immediately.
            # (App)-[:RUNS_ON]->(Node) => Node fails, App fails.
            hosted_apps = self.graph.get_hosted_components(curr)
            for app in hosted_apps:
                if app not in failed_set:
                    failed_set.add(app)
                    cascaded_failures.append(app)
                    G.nodes[app]["state"] = "failed"
                    # Apps generally don't physically host things, so no queue append
                    # unless implementing App->App logical cascades here.

            # --- 2. Network/Structural Propagation (Soft/Probabilistic) ---
            if depth < scenario.max_depth:
                # Use CONNECTS_TO for infrastructure cascades
                neighbors = self.graph.get_successors_by_type(curr, "CONNECTS_TO")
                
                for neighbor in neighbors:
                    if neighbor in failed_set: continue
                    
                    # Check edge weight for threshold
                    edge_data = G.get_edge_data(curr, neighbor)
                    weight = edge_data.get("weight", 1.0) if edge_data else 1.0
                    
                    if weight >= scenario.cascade_threshold:
                        if random.random() < scenario.cascade_probability:
                            failed_set.add(neighbor)
                            cascaded_failures.append(neighbor)
                            G.nodes[neighbor]["state"] = "failed"
                            queue.append((neighbor, depth + 1))

        # --- 3. Impact Calculation ---
        # 3a. Reachability Loss (App Layer Impact)
        total_paths = len(self.graph.initial_paths)
        if total_paths == 0:
            reachability_loss = 0.0
        else:
            current_paths = self.graph.get_pub_sub_paths(active_only=True)
            reachability_loss = 1.0 - (len(current_paths) / total_paths)

        # 3b. Fragmentation (Infra/System Layer Impact)
        total_nodes = G.number_of_nodes()
        c_before = self.graph.initial_component_count
        c_after = self.graph.get_connected_components_count(active_only=True)
        
        if total_nodes <= 1:
            fragmentation = 0.0
        else:
            fragmentation = max(0, (c_after - c_before) / (total_nodes - 1))

        # 3c. Composite Score
        impact_score = (0.6 * reachability_loss) + (0.4 * fragmentation)

        return FailureResult(
            scenario=scenario.description,
            initial_failure=target,
            failure_type=target_type,
            cascaded_failures=cascaded_failures,
            reachability_loss=reachability_loss,
            fragmentation=fragmentation,
            impact_score=impact_score,
            params={
                "threshold": scenario.cascade_threshold,
                "probability": scenario.cascade_probability,
                "depth": scenario.max_depth
            }
        )