"""
Compromise Propagation Simulator

Models adversarial compromise propagation over the trusted dependency graph
to compute the Vulnerability ground truth IV(v).

Algorithm
---------
For each component v:
  1. Build G^T: reversed DEPENDS_ON edges. v ---> u means "v compromise can spread to u".
  2. Traverse G^T from v.
  3. Stop condition (trust threshold):
     If edge_weight < theta_trust, propagation STOPS.
     (Low-QoS edges represented as BEST_EFFORT are not trusted enough to carry automated compromise).
  4. Collect reached nodes and compute:
     - AttackReach(v): fraction of reachable components
     - WeightedAttackImpact(v): importance-weighted sum of contaminated components
     - HighValueContamination(v): distance-discounted sum of high-value components reached
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)

@dataclass
class CompromisePropagationResult:
    component_id: str
    compromised_components: List[str] = field(default_factory=list)
    attack_reach: float = 0.0
    weighted_attack_impact: float = 0.0
    high_value_contamination: float = 0.0
    # For Attack Path Enumeration (optional)
    critical_paths: List[List[str]] = field(default_factory=list)
    theta_trust: float = 0.30

class CompromisePropagationSimulator:
    def __init__(self, theta_trust: float = 0.30, max_path_len: int = 4):
        self.theta_trust = theta_trust
        self.max_path_len = max_path_len

    def simulate_all(
        self,
        component_ids: List[str],
        dependency_edges: List[Tuple[str, str, float]],
        component_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, CompromisePropagationResult]:
        n = len(component_ids)
        if n < 2:
            return {cid: CompromisePropagationResult(component_id=cid) for cid in component_ids}

        cw = component_weights or {}
        
        # G^T: tgt -> src because src depends on tgt. If tgt is compromised, it spreads to src.
        gt_adj: Dict[str, List[Tuple[str, float]]] = {cid: [] for cid in component_ids}
        
        for (src, tgt, w) in dependency_edges:
            if tgt in gt_adj:
                gt_adj[tgt].append((src, w))

        total_weight = sum(cw.get(cid, 1.0) for cid in component_ids)
        if total_weight == 0:
            total_weight = n

        results: Dict[str, CompromisePropagationResult] = {}
        for v in component_ids:
            results[v] = self._propagate_compromise(v, n, gt_adj, cw, total_weight)
            
        return results

    def _propagate_compromise(
        self,
        source: str,
        total_n: int,
        gt_adj: Dict[str, List[Tuple[str, float]]],
        component_weights: Dict[str, float],
        total_weight: float
    ) -> CompromisePropagationResult:
        reached: Set[str] = set()
        depth_map: Dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])
        
        # For attack path enumeration to high-value targets (weight >= 0.8)
        # We'll store simple parents mapping to reconstruct paths
        parents: Dict[str, List[str]] = {source: []}
        
        while queue:
            current = queue.popleft()
            current_depth = depth_map[current]
            
            for (neighbor, edge_w) in gt_adj.get(current, []):
                # Trust Filter: If edge weight < theta_trust, compromise does not propagate
                if edge_w < self.theta_trust:
                    continue
                    
                if neighbor not in reached and neighbor != source:
                    reached.add(neighbor)
                    depth_map[neighbor] = current_depth + 1
                    parents[neighbor] = [current]
                    queue.append(neighbor)
                elif neighbor != source and current_depth + 1 == depth_map.get(neighbor, 0):
                    # Multiple shortest paths
                    parents[neighbor].append(current)

        # Metrics
        n_reached = len(reached)
        n_others = max(total_n - 1, 1)
        attack_reach = n_reached / n_others
        
        weighted_sum = sum(component_weights.get(u, 1.0) for u in reached)
        weighted_attack_impact = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # High value contamination (discounted by distance: weight / 2^(depth-1))
        # depth 1: full weight | depth 2: half weight | depth 3: quarter weight
        hvc_sum = 0.0
        for u in reached:
            w_u = component_weights.get(u, 1.0)
            d = depth_map[u]
            if d > 0:
                hvc_sum += w_u / (2 ** (d - 1))
        
        high_value_contamination = hvc_sum / total_weight if total_weight > 0 else 0.0

        # Optional: Reconstruct attack paths for high value targets
        critical_paths = []
        for u in reached:
            if component_weights.get(u, 0.0) >= 0.8 and depth_map[u] <= self.max_path_len:
                # Top down DFS from u to source using parents
                paths = self._reconstruct_paths(u, source, parents)
                critical_paths.extend(paths)

        return CompromisePropagationResult(
            component_id=source,
            compromised_components=sorted(reached),
            attack_reach=attack_reach,
            weighted_attack_impact=weighted_attack_impact,
            high_value_contamination=high_value_contamination,
            critical_paths=critical_paths,
            theta_trust=self.theta_trust
        )

    def _reconstruct_paths(self, target: str, source: str, parents: Dict[str, List[str]]) -> List[List[str]]:
        if target == source:
            return [[source]]
        paths = []
        for p in parents.get(target, []):
            for path in self._reconstruct_paths(p, source, parents):
                paths.append(path + [target])
        return paths

    def sensitivity_grid(
        self,
        component_ids: List[str],
        dependency_edges: List[Tuple[str, str, float]],
        component_weights: Optional[Dict[str, float]] = None,
        theta_trust_values: Optional[List[float]] = None
    ) -> Dict[float, Dict[str, CompromisePropagationResult]]:
        if theta_trust_values is None:
            theta_trust_values = [0.10, 0.30, 0.50]
        
        grid_results = {}
        for tt in theta_trust_values:
            sim = CompromisePropagationSimulator(theta_trust=tt)
            grid_results[tt] = sim.simulate_all(
                component_ids=component_ids,
                dependency_edges=dependency_edges,
                component_weights=component_weights
            )
        return grid_results
