"""
Fast bottleneck analysis for the /stats/bottleneck endpoint.

Computes only the four metrics needed for the bottleneck score formula:

    score = 0.40·betweenness + 0.25·ap_c_directed
          + 0.20·blast_radius_norm + 0.15·bridge_ratio

Performance improvements over running full StructuralAnalyzer.analyze():

* Approximate betweenness (k-pivot sampling) for graphs > 512 nodes.
* ap_c_directed via BFS-per-AP without graph copies → O(|AP|·avg_BFS_size).
* blast_radius via SCC condensation → O(V+E) on the condensation DAG.
* Directed APs via BFS-with-exclusion → no per-node graph allocation.
* Skips: PageRank (×2), closeness (×2), eigenvector (×2),
  edge betweenness, pubsub metrics, QoS profile,
  code quality normalisation, RCM ordering.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Set

import networkx as nx

from src.analysis.structural_analyzer import extract_layer_subgraph
from src.core.layers import AnalysisLayer, get_layer_definition

_logger = logging.getLogger(__name__)

_APPROX_THRESHOLD = 512   # nodes above which approx betweenness is used
_APPROX_K         = 512   # pivot count for approximation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_for_bottleneck(
    graph_data: Any,
    layer: AnalysisLayer = AnalysisLayer.SYSTEM,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-component metrics needed by ``compute_bottleneck_stats_from_structural``.

    Returns a dict ``{component_id: metric_dict}`` with the same keys that
    ``StructuralMetrics.to_dict()`` would produce for the fields consumed by
    the bottleneck scoring function (betweenness, ap_c_directed, blast_radius,
    bridge_ratio, is_articulation_point, is_directed_ap, name, type, weight,
    cascade_depth, pubsub_betweenness).
    """
    defn = get_layer_definition(layer)
    G = extract_layer_subgraph(graph_data, layer)

    _logger.info(
        "Fast bottleneck analysis [%s]: %d nodes, %d edges",
        layer.value, G.number_of_nodes(), G.number_of_edges(),
    )

    if G.number_of_nodes() == 0:
        return {}

    analyze_types = defn.types_to_analyze
    n_nodes = G.number_of_nodes()

    # --- Betweenness (approximate for large graphs) ---
    G_dist = _build_distance_graph(G)
    if n_nodes > _APPROX_THRESHOLD:
        k = min(_APPROX_K, n_nodes)
        _logger.info("Using approximate betweenness (k=%d) for n=%d", k, n_nodes)
        betweenness = nx.betweenness_centrality(G_dist, k=k, weight="weight", normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G_dist, weight="weight", normalized=True)

    # --- Degree ---
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # --- Resilience (undirected) ---
    U = G.to_undirected()
    art_points = (
        set(nx.articulation_points(U))
        if nx.is_connected(U)
        else _art_points_disconnected(U)
    )
    bridges = (
        set(nx.bridges(U))
        if nx.is_connected(U)
        else _bridges_disconnected(U)
    )
    node_bridge_count: Dict[str, int] = {}
    for (u, v) in bridges:
        node_bridge_count[u] = node_bridge_count.get(u, 0) + 1
        node_bridge_count[v] = node_bridge_count.get(v, 0) + 1

    # --- ap_c_directed (BFS-per-AP, no graph copies) ---
    ap_c_out = _compute_ap_c_fast(U, art_points, n_nodes)
    U_T = G.reverse(copy=False).to_undirected()
    art_points_T = (
        set(nx.articulation_points(U_T))
        if nx.is_connected(U_T)
        else _art_points_disconnected(U_T)
    )
    ap_c_in = _compute_ap_c_fast(U_T, art_points_T, n_nodes)
    ap_c_directed: Dict[str, float] = {
        nid: max(ap_c_out.get(nid, 0.0), ap_c_in.get(nid, 0.0))
        for nid in G.nodes
    }

    # --- Blast radius (SCC condensation) ---
    blast_radius = _compute_blast_radius_fast(G)

    # --- Directed APs (BFS with exclusion, no graph copies) ---
    directed_aps = _compute_directed_aps_fast(G)

    # --- Assemble result dicts ---
    result: Dict[str, Dict[str, Any]] = {}
    for nid in G.nodes:
        ntype = G.nodes[nid].get("component_type", "Unknown")
        if ntype not in analyze_types:
            continue

        raw_in    = in_deg.get(nid, 0)
        raw_out   = out_deg.get(nid, 0)
        total_raw = raw_in + raw_out
        bc        = node_bridge_count.get(nid, 0)

        result[nid] = {
            "id": nid,
            "name": G.nodes[nid].get("name", nid),
            "type": ntype,
            "betweenness": betweenness.get(nid, 0.0),
            "ap_c_directed": ap_c_directed.get(nid, 0.0),
            "blast_radius": blast_radius.get(nid, 0),
            "bridge_count": bc,
            "bridge_ratio": bc / total_raw if total_raw > 0 else 0.0,
            "is_articulation_point": nid in art_points,
            "is_directed_ap": nid in directed_aps,
            "is_isolated": total_raw == 0,
            "weight": G.nodes[nid].get("weight", 1.0),
            "dependency_weight_in": 0.0,
            "dependency_weight_out": 0.0,
            # Fields expected by compute_bottleneck_stats_from_structural
            # that are not computed in this fast path — zero-filled.
            "cascade_depth": 0,
            "pubsub_betweenness": 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_distance_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of G with weights inverted for distance-based betweenness."""
    G_dist = G.copy()
    for u, v, data in G_dist.edges(data=True):
        w = data.get("weight", 1.0)
        data["weight"] = 1.0 / w if w > 0 else 1.0
    return G_dist


def _art_points_disconnected(U: nx.Graph) -> Set[str]:
    """Articulation points across all connected components."""
    pts: Set[str] = set()
    for comp in nx.connected_components(U):
        sub = U.subgraph(comp)
        if len(sub) >= 3:
            pts.update(nx.articulation_points(sub))
    return pts


def _bridges_disconnected(U: nx.Graph) -> Set:
    """Bridges across all connected components."""
    br = set()
    for comp in nx.connected_components(U):
        sub = U.subgraph(comp)
        if len(sub) >= 2:
            br.update(nx.bridges(sub))
    return br


def _compute_ap_c_fast(
    G_undir: nx.Graph,
    art_points: Set[str],
    n: int,
) -> Dict[str, float]:
    """
    Compute ap_c = 1 − largest_component/(n−1) for all nodes.

    Non-articulation points short-circuit to 0.0.  For articulation points,
    BFS from each neighbour of the AP (skipping the AP itself) finds the
    resulting component sizes without copying the graph.

    Complexity: O(|AP| · avg_BFS_size) instead of O(V · (V+E)).
    """
    result: Dict[str, float] = {v: 0.0 for v in G_undir.nodes}
    if n <= 2 or not art_points:
        return result

    ccs: List[Set[str]] = list(nx.connected_components(G_undir))
    cc_of_node: Dict[str, int] = {node: i for i, cc in enumerate(ccs) for node in cc}
    cc_sizes = [len(cc) for cc in ccs]

    for v in art_points:
        v_cc_idx = cc_of_node.get(v)
        if v_cc_idx is None:
            continue

        other_max = max(
            (s for i, s in enumerate(cc_sizes) if i != v_cc_idx), default=0
        )

        sub = G_undir.subgraph(ccs[v_cc_idx])
        visited: Set[str] = {v}
        component_sizes: List[int] = []

        for start in sub.neighbors(v):
            if start in visited:
                continue
            queue: deque = deque([start])
            visited.add(start)
            size = 1
            while queue:
                curr = queue.popleft()
                for nb in sub.neighbors(curr):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
                        size += 1
            component_sizes.append(size)

        local_max = max(component_sizes, default=0)
        largest   = max(local_max, other_max)
        result[v] = 1.0 - largest / (n - 1)

    return result


def _compute_blast_radius_fast(G: nx.DiGraph) -> Dict[str, int]:
    """
    Compute blast_radius[v] = reachable nodes from v (excluding v) via SCC condensation.

    blast_radius[v] = (|SCC(v)| − 1) + Σ |SCC(w)| for w in descendants(SCC(v))

    Complexity: O(V+E) for condensation + O(V_c·(V_c+E_c)) for DAG traversal.
    """
    if G.number_of_nodes() == 0:
        return {}

    dag = nx.condensation(G)

    scc_blast: Dict[int, int] = {}
    for scc_id in dag.nodes:
        members    = dag.nodes[scc_id]["members"]
        own_size   = len(members)
        downstream = sum(
            len(dag.nodes[rid]["members"])
            for rid in nx.descendants(dag, scc_id)
        )
        scc_blast[scc_id] = (own_size - 1) + downstream

    blast_radius: Dict[str, int] = {}
    for scc_id in dag.nodes:
        for node in dag.nodes[scc_id]["members"]:
            blast_radius[node] = scc_blast[scc_id]

    return blast_radius


def _compute_directed_aps_fast(G: nx.DiGraph) -> Set[str]:
    """
    Identify directed articulation points via BFS with node exclusion.

    A node v is a directed AP if its removal reduces the set of nodes
    reachable from all in-degree-0 roots.  Uses explicit exclusion in BFS
    rather than graph copies, eliminating O(V+E) allocation overhead per node.

    Complexity: O(|base_reachable| · (V+E)) time, O(V+E) space.
    """
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        return set()

    base_reachable: Set[str] = set()
    for r in roots:
        if r not in base_reachable:
            stack = [r]
            base_reachable.add(r)
            while stack:
                curr = stack.pop()
                for nb in G.successors(curr):
                    if nb not in base_reachable:
                        base_reachable.add(nb)
                        stack.append(nb)

    if not base_reachable:
        return set()

    target = len(base_reachable) - 1
    directed_aps: Set[str] = set()

    for v in base_reachable:
        reachable: Set[str] = set()
        stack = []
        for r in roots:
            if r != v and r not in reachable:
                reachable.add(r)
                stack.append(r)
        while stack:
            curr = stack.pop()
            for nb in G.successors(curr):
                if nb != v and nb not in reachable:
                    reachable.add(nb)
                    stack.append(nb)
        if len(reachable) < target:
            directed_aps.add(v)

    return directed_aps
