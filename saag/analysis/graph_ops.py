"""
Shared NetworkX helpers for the Analyze stage.

These three primitives are used identically by ``structural_analyzer`` (the
Step-2 metric engine) and ``statistics`` (the cross-cutting descriptive stats
module), so they live here rather than being duplicated in both.
"""

from __future__ import annotations

from typing import Set, Tuple

import networkx as nx


def build_distance_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Return a copy of *G* with edge weights inverted for distance-based metrics.

    DEPENDS_ON weights represent dependency strength (importance), whereas
    betweenness centrality interprets weights as distances. Inversion makes
    strongly-weighted dependencies "close", so shortest paths prefer them.

        w_distance = 1.0 / w_importance
    """
    G_dist = G.copy()
    for _u, _v, data in G_dist.edges(data=True):
        w = data.get("weight", 1.0)
        data["weight"] = 1.0 / w if w > 0 else 1.0
    return G_dist


def articulation_points_disconnected(U: nx.Graph) -> Set[str]:
    """Articulation points across all connected components of *U*."""
    pts: Set[str] = set()
    for comp in nx.connected_components(U):
        sub = U.subgraph(comp)
        if len(sub) >= 3:
            pts.update(nx.articulation_points(sub))
    return pts


def bridges_disconnected(U: nx.Graph) -> Set[Tuple[str, str]]:
    """Bridges across all connected components of *U*."""
    br: Set[Tuple[str, str]] = set()
    for comp in nx.connected_components(U):
        sub = U.subgraph(comp)
        if len(sub) >= 2:
            br.update(nx.bridges(sub))
    return br
