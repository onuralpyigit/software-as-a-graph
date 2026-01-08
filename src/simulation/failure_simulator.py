"""
Failure Simulator

Simulates cascading failures using RAW structural relationships.
Provides distinct impact metrics for Application and Infrastructure layers.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any
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
    failure_type: str
    cascaded_failures: List[str]
    
    # Impact Metrics
    app_reachability_loss: float     # % of broken Pub-Sub paths
    infra_fragmentation: float       # % increase in graph components
    impact_score: float       # Composite score
    
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "target": {
                "id": self.initial_failure,
                "type": self.failure_type
            },
            "impact": {
                "reachability_loss": round(self.app_reachability_loss, 4),
                "fragmentation": round(self.infra_fragmentation, 4),
                "composite_score": round(self.impact_score, 4)
            },
            "cascade": {
                "count": len(self.cascaded_failures),
                "nodes": self.cascaded_failures
            }
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens result for CSV/DataFrame export."""
        return {
            "node_id": self.initial_failure,
            "node_type": self.failure_type,
            "reachability_loss": round(self.app_reachability_loss, 4),
            "fragmentation": round(self.infra_fragmentation, 4),
            "impact_score": round(self.impact_score, 4),
            "cascaded_count": len(self.cascaded_failures),
            "threshold": self.params.get("threshold"),
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
            return self._empty_result(scenario, "Not Found")

        target_type = G.nodes[target].get("type", "unknown")
        
        # --- Propagation Phase ---
        failed_set = set()
        queue = [(target, 0)] 
        failed_set.add(target)
        G.nodes[target]["state"] = "failed"
        
        cascaded_failures = []

        while queue:
            curr, depth = queue.pop(0)
            
            # 1. Physical Propagation (Hard): Node -> Hosted App
            # Only relevant if 'curr' is a Compute Node
            hosted_apps = self.graph.get_hosted_components(curr)
            for app in hosted_apps:
                if app not in failed_set:
                    failed_set.add(app)
                    cascaded_failures.append(app)
                    G.nodes[app]["state"] = "failed"
                    # Apps don't host things, so no further queue push for this branch

            # 2. Network Propagation (Soft): Node -> Node via CONNECTS_TO
            # Only traverse if within depth limit
            if depth < scenario.max_depth:
                neighbors = self.graph.get_successors_by_type(curr, "CONNECTS_TO")
                for neighbor in neighbors:
                    if neighbor in failed_set: continue
                    
                    edge_data = G.get_edge_data(curr, neighbor)
                    weight = edge_data.get("weight", 1.0) if edge_data else 1.0
                    
                    if weight >= scenario.cascade_threshold:
                        if random.random() < scenario.cascade_probability:
                            failed_set.add(neighbor)
                            cascaded_failures.append(neighbor)
                            G.nodes[neighbor]["state"] = "failed"
                            queue.append((neighbor, depth + 1))

        # --- Impact Calculation Phase ---
        
        # 1. Application Layer: Reachability Loss
        # (How many Pub->Sub paths are broken?)
        total_paths = len(self.graph.initial_paths)
        if total_paths == 0:
            reach_loss = 0.0
        else:
            remaining_paths = self.graph.get_pub_sub_paths(active_only=True)
            reach_loss = 1.0 - (len(remaining_paths) / total_paths)

        # 2. Infrastructure Layer: Fragmentation
        # (How disjoint did the graph become?)
        total_nodes = self.graph.initial_node_count
        c_before = self.graph.initial_component_count
        c_after = self.graph.get_connected_components_count(active_only=True)
        
        if total_nodes <= 1:
            frag = 0.0
        else:
            # Normalized increase in components
            frag = max(0, (c_after - c_before) / (total_nodes - 1))

        # 3. Composite Score
        # Weighting depends on target type slightly, but generally 50/50
        impact_score = (0.5 * reach_loss) + (0.5 * frag)

        return FailureResult(
            scenario=scenario.description,
            initial_failure=target,
            failure_type=target_type,
            cascaded_failures=cascaded_failures,
            app_reachability_loss=reach_loss,
            infra_fragmentation=frag,
            impact_score=impact_score,
            params={
                "threshold": scenario.cascade_threshold,
                "probability": scenario.cascade_probability,
                "depth": scenario.max_depth
            }
        )

    def _empty_result(self, scenario, reason) -> FailureResult:
        return FailureResult(scenario.description, scenario.target_node, reason, [], 0.0, 0.0, 0.0, {})