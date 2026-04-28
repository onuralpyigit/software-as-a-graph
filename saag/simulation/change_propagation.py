"""
Change Propagation Simulator

Models development-time change propagation for the Maintainability ground truth IM(v).

This is a fundamentally different simulation mode from the failure cascade simulator:

  - FAILURE simulation: removes component v from G_structural, propagates via
    physical/logical/application cascade rules, measures runtime damage.

  - CHANGE PROPAGATION: marks component v as CHANGED in G_analysis, traverses
    G^T (transposed DEPENDS_ON graph) to find components that must adapt, stops
    at loose-coupling and stable-interface boundaries.

The conceptual difference maps exactly to the two different operational questions:
  - Failure: "What happens at runtime if v crashes?"
  - Change:  "What must change at development time if v's interface changes?"

Algorithm
---------
For each component v:
  1. Build G^T: reversed DEPENDS_ON edges from the analysis dependency graph.
     If u --[DEPENDS_ON]--> v in G_analysis, then v ---> u in G^T.
     This means: "v changing may force u to adapt."
  2. BFS from v on G^T.
  3. At each node u encountered, decide whether to stop propagation:
     a. Loose-coupling stop: the edge weight w(u→v original edge) < θ_loose.
        Low-weight VOLATILE/BEST_EFFORT dependencies are loosely contracted;
        the dependent can likely absorb the change without modification.
     b. Stable-interface stop: Instability(u) < θ_stable.
        A highly stable component (many afferent, few efferent) acts as an
        interface boundary that absorbs change obligations.
  4. Collect reached nodes and compute:
     - ChangeReach(v): fraction of all other components that must adapt
     - WeightedChangeImpact(v): importance-weighted adaptation cost
     - NormalizedChangeDepth(v): relative BFS depth (normalised post-pass)

Complexity: O(|V| * (|V| + |E|)) — identical to one exhaustive failure sim run.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ChangePropagationResult:
    """Result of change propagation analysis for a single component."""
    component_id: str
    reached_components: List[str] = field(default_factory=list)
    change_reach: float = 0.0
    weighted_change_impact: float = 0.0
    # Raw BFS depth (not yet normalised)
    max_bfs_depth: int = 0
    # Final normalised depth (populated in post-pass)
    normalized_change_depth: float = 0.0
    # Parameter record for reproducibility
    theta_loose: float = 0.20
    theta_stable: float = 0.20


# =============================================================================
# Simulator
# =============================================================================

class ChangePropagationSimulator:
    """
    Computes the IM(v) Maintainability ground truth via change propagation BFS
    on the transposed DEPENDS_ON graph.

    Args:
        theta_loose:  Loose-coupling stop threshold. Edges with weight < theta_loose
                      are treated as loosely contracted — propagation stops at them.
                      Default: 0.20 (VOLATILE + BEST_EFFORT edges stop propagation).
        theta_stable: Stable-interface stop threshold. Components with
                      Instability(u) < theta_stable are treated as stable
                      interface boundaries that absorb change obligations.
                      Default: 0.20 (highly stable components stop propagation).
    """

    def __init__(
        self,
        theta_loose: float = 0.20,
        theta_stable: float = 0.20,
    ) -> None:
        self.theta_loose = theta_loose
        self.theta_stable = theta_stable

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def simulate_all(
        self,
        component_ids: List[str],
        dependency_edges: List[Tuple[str, str, float]],  # (source, target, weight)
        component_weights: Optional[Dict[str, float]] = None,
        component_in_degrees: Optional[Dict[str, int]] = None,
        component_out_degrees: Optional[Dict[str, int]] = None,
    ) -> Dict[str, ChangePropagationResult]:
        """
        Run change propagation for all components and return per-component results.

        Args:
            component_ids:        All component IDs in the analysis layer.
            dependency_edges:     List of (source, target, weight) DEPENDS_ON edges.
                                  Edge `(u, v, w)` means u depends on v with weight w.
            component_weights:    Per-component QoS weight w(v); used for WeightedChangeImpact.
                                  Defaults to 1.0 for all if not provided.
            component_in_degrees: Raw in-degree per component (for instability calculation).
            component_out_degrees:Raw out-degree per component (for instability calculation).

        Returns:
            Dict mapping component_id -> ChangePropagationResult (with IM sub-metrics filled).
        """
        n = len(component_ids)
        if n < 2:
            return {cid: ChangePropagationResult(component_id=cid) for cid in component_ids}

        # Defaults
        cw = component_weights or {}
        in_deg = component_in_degrees or {}
        out_deg = component_out_degrees or {}

        # Build G^T: transposed DEPENDS_ON graph
        # Original: u --[w]--> v  means "u depends on v"
        # G^T:      v --[w]--> u  means "v changing may force u to adapt"
        gt_adj: Dict[str, List[Tuple[str, float]]] = {cid: [] for cid in component_ids}
        # Also keep original edge weights for stop-condition lookup
        edge_weight_map: Dict[Tuple[str, str], float] = {}  # (u, v) -> w

        for (src, tgt, w) in dependency_edges:
            # Store original edge weight src→tgt
            edge_weight_map[(src, tgt)] = w
            # Add to G^T: tgt → src
            if tgt in gt_adj:
                gt_adj[tgt].append((src, w))

        # Pre-compute instability per component
        instability: Dict[str, float] = {}
        _eps = 1e-9
        for cid in component_ids:
            od = out_deg.get(cid, 0)
            id_ = in_deg.get(cid, 0)
            instability[cid] = od / (id_ + od + _eps)

        # Total weight for WeightedChangeImpact denominator
        total_weight = sum(cw.get(cid, 1.0) for cid in component_ids)
        if total_weight == 0:
            total_weight = n

        # Run BFS per component
        results: Dict[str, ChangePropagationResult] = {}
        for v in component_ids:
            res = self._bfs_change_propagation(
                v, n, gt_adj, edge_weight_map, instability, cw, total_weight,
            )
            results[v] = res

        # Post-pass: normalise change depth
        max_depth = max((r.max_bfs_depth for r in results.values()), default=0)
        if max_depth > 0:
            for r in results.values():
                r.normalized_change_depth = r.max_bfs_depth / max_depth
        else:
            for r in results.values():
                r.normalized_change_depth = 0.0

        logger.debug(
            "ChangePropagationSimulator: processed %d components, "
            "max_depth=%d, θ_loose=%.2f, θ_stable=%.2f",
            n, max_depth, self.theta_loose, self.theta_stable,
        )
        return results

    # ------------------------------------------------------------------
    # BFS for a single source component
    # ------------------------------------------------------------------

    def _bfs_change_propagation(
        self,
        source: str,
        total_n: int,
        gt_adj: Dict[str, List[Tuple[str, float]]],
        edge_weight_map: Dict[Tuple[str, str], float],
        instability: Dict[str, float],
        component_weights: Dict[str, float],
        total_weight: float,
    ) -> ChangePropagationResult:
        """
        BFS on G^T from `source`, applying stop conditions.

        For each neighbor u encountered (meaning 'u may need to adapt to source changing'):
          Stop propagation THROUGH u if:
            1. The edge weight w(u → source in original graph) < θ_loose
               (loose-coupling stop: VOLATILE/BEST_EFFORT dependency)
            2. Instability(u) < θ_stable
               (stable-interface stop: u has far more afferent than efferent coupling)

          Note: u is STILL counted as reached even if we stop propagation through
          it — the question is whether u must adapt, not whether it propagates
          further. The stop condition only gates whether we continue BFS through u.
        """
        reached: Set[str] = set()
        depth_map: Dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])
        # Track max edge weight along path to each reached node
        path_weight_map: Dict[str, float] = {source: 0.0}

        while queue:
            current = queue.popleft()
            current_depth = depth_map[current]

            for (neighbor, edge_w) in gt_adj.get(current, []):
                if neighbor in reached or neighbor == source:
                    continue

                # The edge connecting current to neighbor in G_analysis is:
                # neighbor --[original_w]--> current  (neighbor depends on current)
                # edge_w already carries this weight from the G^T construction.
                original_edge_w = edge_w

                # Always count the neighbor as reached (u must potentially adapt)
                reached.add(neighbor)
                depth_map[neighbor] = current_depth + 1
                path_weight_map[neighbor] = max(
                    path_weight_map.get(current, 0.0),
                    original_edge_w,
                )

                # --- Stop conditions (gate further propagation through neighbor) ---
                stop_loose = original_edge_w < self.theta_loose
                stop_stable = instability.get(neighbor, 0.5) < self.theta_stable

                if not (stop_loose or stop_stable):
                    queue.append(neighbor)

        # --- Compute sub-metrics ---
        n_reached = len(reached)
        n_others = max(total_n - 1, 1)

        change_reach = n_reached / n_others

        # WeightedChangeImpact: sum of component weights of reached nodes,
        # weighted by maximum path edge weight (tight contract = higher cost)
        weighted_sum = sum(
            component_weights.get(u, 1.0) * path_weight_map.get(u, 0.0)
            for u in reached
        )
        weighted_change_impact = weighted_sum / total_weight if total_weight > 0 else 0.0

        max_bfs_depth = max(
            (depth_map[u] for u in reached), default=0
        )

        return ChangePropagationResult(
            component_id=source,
            reached_components=sorted(reached),
            change_reach=change_reach,
            weighted_change_impact=weighted_change_impact,
            max_bfs_depth=max_bfs_depth,
            theta_loose=self.theta_loose,
            theta_stable=self.theta_stable,
        )

    # ------------------------------------------------------------------
    # Sensitivity grid
    # ------------------------------------------------------------------

    def sensitivity_grid(
        self,
        component_ids: List[str],
        dependency_edges: List[Tuple[str, str, float]],
        component_weights: Optional[Dict[str, float]] = None,
        component_in_degrees: Optional[Dict[str, int]] = None,
        component_out_degrees: Optional[Dict[str, int]] = None,
        theta_loose_values: Optional[List[float]] = None,
        theta_stable_values: Optional[List[float]] = None,
    ) -> Dict[Tuple[float, float], Dict[str, ChangePropagationResult]]:
        """
        Run change propagation across a grid of (θ_loose, θ_stable) values.

        Returns a dict mapping (θ_loose, θ_stable) -> per-component results.
        Used for the sensitivity analysis required by the validation framework.

        Default grid:
            θ_loose  ∈ {0.10, 0.20, 0.30}
            θ_stable ∈ {0.10, 0.20, 0.30}
        """
        if theta_loose_values is None:
            theta_loose_values = [0.10, 0.20, 0.30]
        if theta_stable_values is None:
            theta_stable_values = [0.10, 0.20, 0.30]

        grid_results: Dict[Tuple[float, float], Dict[str, ChangePropagationResult]] = {}
        for tl in theta_loose_values:
            for ts in theta_stable_values:
                sim = ChangePropagationSimulator(theta_loose=tl, theta_stable=ts)
                grid_results[(tl, ts)] = sim.simulate_all(
                    component_ids=component_ids,
                    dependency_edges=dependency_edges,
                    component_weights=component_weights,
                    component_in_degrees=component_in_degrees,
                    component_out_degrees=component_out_degrees,
                )
        return grid_results
