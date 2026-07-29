"""Cascade-simulation ground truth I(v) for the validation CLI."""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .scoring import NodeScores


def simulate_cascade(G: nx.DiGraph, origin: str, depth_limit: int = 5, seed: int = 42) -> Tuple[float, int, int]:
    """
    LEGACY WRAPPER: Now uses central FaultInjector for consistency.
    """
    from saag.simulation.fault_injector import FaultInjector
    injector = FaultInjector(graph=G, seeds=[seed], cascade_depth_limit=depth_limit)
    # _inject_node is a private helper that runs a single node injection
    rec = injector._inject_node(origin)
    return rec.impact_score, rec.cascade_depth, rec.total_impacted_subscribers


def derive_ground_truth(
    G: nx.DiGraph,
    scores: Dict[str, NodeScores],
    depth_limit: int = 5,
    seed: int = 42,
    n_repeats: int = 5,
) -> Dict[str, NodeScores]:
    """
    Run cascade simulation for every node and record I(v).

    Uses `n_repeats` stochastic runs per node; I(v) = mean impact. The reported
    depth is the worst case observed and the affected count is the mean, so all
    three figures summarise every seed rather than only the last one.
    """
    rng_seeds = [seed + i * 37 for i in range(n_repeats)]

    for v, ns in scores.items():
        impacts, depths, affected_counts = [], [], []
        for s in rng_seeds:
            impact, depth, affected = simulate_cascade(G, v, depth_limit, seed=s)
            impacts.append(impact)
            depths.append(depth)
            affected_counts.append(affected)
        ns.I = float(np.mean(impacts))
        ns.cascade_depth = int(max(depths)) if depths else 0
        ns.nodes_affected = int(round(float(np.mean(affected_counts)))) if affected_counts else 0
    return scores
