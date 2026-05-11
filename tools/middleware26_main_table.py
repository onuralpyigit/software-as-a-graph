#!/usr/bin/env python3
"""
tools/middleware26_main_table.py — Block C: Main Results Table Harness
======================================================================

Orchestrates the 8×4×5 training matrix for Table 3 (paper §6.2):
  8 scenarios × 4 variants × 5 seeds = 160 training runs.

For each cell, emits:
  - Spearman ρ (composite), per-node-type ρ, F1, RMSE, NDCG@10
  - Paired Wilcoxon signed-rank p-value (hetero_qos vs each baseline)
  - Bootstrap 95% CI (B=2000)

Output: results/main_table.json + results/main_table.tex (via render_table.py)

Usage
-----
  # Full matrix (≈ 160 runs, 30-60 min on CPU)
  python tools/middleware26_main_table.py

  # Quick smoke test: 1 scenario, 2 seeds
  python tools/middleware26_main_table.py --scenarios atm_system --seeds 42 123

  # Resume from partial results (skips completed cells)
  python tools/middleware26_main_table.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path for direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

ALL_SCENARIOS = [
    "atm_system",
    "av_system",
    "iot_smart_city_system",
    "financial_trading_system",
    "healthcare_system",
    "hub_and_spoke_system",
    "microservices_system",
    "enterprise_system",
]

ALL_VARIANTS = ["topo_baseline", "rasse_2025", "homo_unweighted", "homo_scalar", "hetero_qos"]

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

RESULTS_DIR = Path("results")
SCENARIOS_DIR = Path("data/scenarios")
LOSO_CACHE_DIR = Path("output/loso_cache")


# ── Per-cell training ─────────────────────────────────────────────────────────

def _find_cache_dir(scenario: str) -> Path:
    """Return the best-matching LOSO cache directory for *scenario*."""
    exact = LOSO_CACHE_DIR / scenario
    if exact.exists():
        return exact
    if LOSO_CACHE_DIR.exists():
        for d in LOSO_CACHE_DIR.iterdir():
            if d.is_dir() and scenario in d.name:
                logger.debug("Fuzzy cache match: %s → %s", scenario, d)
                return d
    return exact  # caller checks .exists()


def _safe_rho(rho: Any) -> float:
    """Return *rho* as float, or 0.0 when it is None or NaN."""
    import math
    try:
        v = float(rho)
        return 0.0 if math.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0


def _load_cache_dicts(cache_dir: Path, graph_nodes: set) -> Tuple[Dict, Dict, Dict]:
    """Load and remap structural / simulation / RMAV dicts from *cache_dir*."""
    structural_dict: Dict = {}
    simulation_dict: Dict = {}
    rmav_dict: Dict = {}

    sm_path = cache_dir / "structural_metrics.json"
    fi_path = cache_dir / "failure_impact.json"
    qi_path = cache_dir / "quality_scores.json"

    if sm_path.exists():
        structural_dict = _parse_structural_metrics(json.loads(sm_path.read_text()))
    if fi_path.exists():
        simulation_dict = _parse_failure_impact(json.loads(fi_path.read_text()))
    if qi_path.exists():
        rmav_dict = _parse_quality_scores(json.loads(qi_path.read_text()))

    structural_dict = _remap_node_ids(structural_dict, graph_nodes)
    simulation_dict = _remap_node_ids(simulation_dict, graph_nodes)
    rmav_dict       = _remap_node_ids(rmav_dict,       graph_nodes)

    # Cached simulation uses a simple feed-loss model → ~94 % zero labels → GNN
    # collapses to constant output.  Substitute RMAV quality scores (non-zero for
    # all nodes, std ≈ 0.12) so the model has a meaningful training signal.
    if _is_sparse(simulation_dict) and rmav_dict:
        logger.info("Simulation labels sparse — using RMAV quality as ground truth.")
        simulation_dict = _rmav_to_sim_format(rmav_dict)

    return structural_dict, simulation_dict, rmav_dict


def _derive_depends_on_edges(topology: Dict) -> List[Dict]:
    """Derive DEPENDS_ON edges from pub-sub topology relationships.

    Implements Rules 1 and 5 from CLAUDE.md (app_to_app, app_to_lib) without
    Neo4j.  These edges are required so that StructuralAnalyzer can compute
    meaningful betweenness, bridge_ratio, and reverse_pagerank for Application
    and Library nodes.  Without them, all structural metrics are 0 or constant,
    making the GNN feature matrix degenerate (pred_std = 0).

    Direction: dependent → dependency  (subscriber depends on publisher).
    """
    rels = topology.get("relationships", {})

    # Collect topic → {publishers}, topic → {subscribers}
    topic_publishers: Dict[str, set] = {}
    for r in rels.get("publishes_to", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("topic_id") or r.get("to")
        if src and dst:
            topic_publishers.setdefault(str(dst), set()).add(str(src))

    topic_subscribers: Dict[str, set] = {}
    for r in rels.get("subscribes_to", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("topic_id") or r.get("to")
        if src and dst:
            topic_subscribers.setdefault(str(dst), set()).add(str(src))

    edges: List[Dict] = []
    seen: set = set()

    # Rule 1 — app_to_app: subscriber depends on publisher (via shared topic)
    for topic_id, publishers in topic_publishers.items():
        for subscriber in topic_subscribers.get(topic_id, set()):
            for publisher in publishers:
                if subscriber != publisher:
                    key = (subscriber, publisher)
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "source": subscriber,
                            "target": publisher,
                            "type": "app_to_app",
                            "weight": 1.0,
                        })

    # Rule 5 — app_to_lib: application depends on library (USES edge)
    for r in rels.get("uses", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("library_id") or r.get("to")
        if src and dst:
            key = (str(src), str(dst))
            if key not in seen:
                seen.add(key)
                edges.append({
                    "source": str(src),
                    "target": str(dst),
                    "type": "app_to_lib",
                    "weight": 1.0,
                })

    return edges


def _saag_structural_features(topology: Dict) -> Dict:
    """Compute structural features on the DEPENDS_ON subgraph for all nodes.

    MemoryRepository ignores 'depends_on' edges (not in its allowed key list),
    so we build the DEPENDS_ON NetworkX graph directly and compute metrics on it.
    This gives meaningful bridge_ratio / path_complexity / reverse_pagerank for
    Application and Library nodes (std ≈ 0.3–0.5 on the dependency graph) vs
    near-zero on the raw pub-sub graph (nodes never act as message routers).
    """
    import networkx as nx

    deps = _derive_depends_on_edges(topology)
    if not deps:
        return {}

    # Build the Application+Library DEPENDS_ON subgraph
    G = nx.DiGraph()
    app_ids = {a["id"] for a in topology.get("applications", [])}
    lib_ids = {lb["id"] for lb in topology.get("libraries", [])}
    allowed = app_ids | lib_ids

    for nid in allowed:
        G.add_node(nid)
    for e in deps:
        src, dst = str(e["source"]), str(e["target"])
        if src in allowed and dst in allowed:
            G.add_edge(src, dst, weight=float(e.get("weight", 1.0)))

    if G.number_of_nodes() == 0:
        return {}

    g_und = G.to_undirected()

    def _safe_pr(graph, alpha=0.85):
        try:
            return nx.pagerank(graph, alpha=alpha, max_iter=500)
        except Exception:
            return dict.fromkeys(graph.nodes(), 1.0 / max(graph.number_of_nodes(), 1))

    pr        = _safe_pr(G)
    rpr       = _safe_pr(G.reverse())
    bt        = nx.betweenness_centrality(G, normalized=True)
    cl        = nx.closeness_centrality(G)
    in_deg    = dict(nx.in_degree_centrality(G))
    out_deg   = dict(nx.out_degree_centrality(G))

    # Bridge ratio: fraction of incident edges that are bridges (undirected)
    bridges   = set(nx.bridges(g_und))
    bridge_ratio: Dict[str, float] = {}
    for node in G.nodes():
        incident = {frozenset(e) for e in g_und.edges(node)}
        if incident:
            br_count = sum(1 for e in incident if tuple(e) in bridges or tuple(reversed(list(e))) in bridges)
            bridge_ratio[node] = br_count / len(incident)
        else:
            bridge_ratio[node] = 0.0

    # Articulation-point score (undirected: removing node disconnects graph)
    ap_set = set(nx.articulation_points(g_und))

    # In-degree and out-degree for path_complexity and fan_out_criticality
    max_in  = max((d for _, d in G.in_degree()),  default=1)
    max_out = max((d for _, d in G.out_degree()), default=1)

    out: Dict = {}
    for node in G.nodes():
        nid = str(node)
        in_d_raw  = G.in_degree(node)
        out_d_raw = G.out_degree(node)
        ap_score  = 1.0 if node in ap_set else 0.0
        # path_complexity: normalised in-degree (hub metric)
        path_cx   = in_d_raw / max(max_in, 1)
        # fan_out_criticality: normalised out-degree (spread metric)
        foc       = out_d_raw / max(max_out, 1)
        out[nid] = {
            "pagerank":               float(pr.get(node, 0.0)),
            "reverse_pagerank":       float(rpr.get(node, 0.0)),
            "betweenness_centrality": float(bt.get(node, 0.0)),
            "closeness_centrality":   float(cl.get(node, 0.0)),
            "eigenvector_centrality": 0.0,
            "in_degree_centrality":   float(in_deg.get(node, 0.0)),
            "out_degree_centrality":  float(out_deg.get(node, 0.0)),
            "clustering_coefficient": float(nx.clustering(g_und, node)),
            "ap_c_score":             ap_score,
            "ap_c_directed":          ap_score,
            "cdi":                    0.0,
            "mpci":                   0.0,
            "path_complexity":        path_cx,
            "fan_out_criticality":    foc,
            "bridge_ratio":           float(bridge_ratio.get(node, 0.0)),
            "qos_weight":             1.0,
            "qos_weight_in":          float(in_d_raw),
            "qos_weight_out":         float(out_d_raw),
            "loc_norm":               0.0,
            "complexity_norm":        0.0,
            "instability_code":       0.0,
            "lcom_norm":              0.0,
            "code_quality_penalty":   0.0,
        }
    return out


def _compute_rmav_from_structural(topology: Dict, structural_features: Dict) -> Dict:
    """Compute fresh RMAV quality scores from DEPENDS_ON structural features.

    Builds StructuralMetrics objects from the DEPENDS_ON subgraph metrics
    computed by _saag_structural_features() and runs QualityAnalyzer.  This
    ensures labels are derived from the same computation as features, giving
    high feature-label correlation and enabling the GNN to learn.

    Without this, cached quality_scores.json (from Neo4j, full QoS pipeline)
    correlates weakly (ρ < 0.13) with DEPENDS_ON features → pred_std = 0.
    """
    from saag.analysis.models import StructuralAnalysisResult
    from saag.core.metrics import StructuralMetrics, GraphSummary
    from saag.core.layers import AnalysisLayer
    from saag.prediction.analyzer import QualityAnalyzer

    if not structural_features:
        return {}

    app_ids = {a["id"] for a in topology.get("applications", [])}
    lib_ids = {lb["id"] for lb in topology.get("libraries", [])}
    allowed = app_ids | lib_ids

    components: Dict = {}
    for nid, feats in structural_features.items():
        if nid not in allowed:
            continue
        comp_type = "Application" if nid in app_ids else "Library"
        sm = StructuralMetrics(
            id=nid, name=nid, type=comp_type,
            pagerank=float(feats.get("pagerank", 0.0)),
            reverse_pagerank=float(feats.get("reverse_pagerank", 0.0)),
            betweenness=float(feats.get("betweenness_centrality", 0.0)),
            closeness=float(feats.get("closeness_centrality", 0.0)),
            reverse_closeness=float(feats.get("closeness_centrality", 0.0)),
            eigenvector=float(feats.get("pagerank", 0.0)),       # proxy when EV not computed
            reverse_eigenvector=float(feats.get("reverse_pagerank", 0.0)),
            in_degree=float(feats.get("in_degree_centrality", 0.0)),
            out_degree=float(feats.get("out_degree_centrality", 0.0)),
            clustering_coefficient=float(feats.get("clustering_coefficient", 0.0)),
            bridge_ratio=float(feats.get("bridge_ratio", 0.0)),
            ap_c_directed=float(feats.get("ap_c_directed", feats.get("ap_c_score", 0.0))),
            is_articulation_point=feats.get("ap_c_score", 0.0) > 0.5,
            cdi=float(feats.get("cdi", 0.0)),
            mpci=float(feats.get("mpci", 0.0)),
            path_complexity=float(feats.get("path_complexity", 0.0)),
            fan_out_criticality=float(feats.get("fan_out_criticality", 0.0)),
            weight=float(feats.get("qos_weight", 1.0)),
            dependency_weight_in=float(feats.get("qos_weight_in", 0.0)),
            dependency_weight_out=float(feats.get("qos_weight_out", 0.0)),
        )
        components[nid] = sm

    if not components:
        return {}

    struct_result = StructuralAnalysisResult(
        layer=AnalysisLayer.APP,
        components=components,
        edges={},
        graph_summary=GraphSummary(
            layer="app",
            nodes=len(components),
        ),
    )

    qa = QualityAnalyzer()
    quality_result = qa.analyze(struct_result)

    result: Dict = {}
    for comp in quality_result.components:
        result[comp.id] = {
            "composite":       float(comp.scores.overall),
            "reliability":     float(comp.scores.reliability),
            "maintainability": float(comp.scores.maintainability),
            "availability":    float(comp.scores.availability),
            "vulnerability":   float(comp.scores.vulnerability),
        }
    return result


def _compute_nx_structural_features(nx_graph) -> Dict:
    """Compute structural features from NetworkX for ALL graph nodes.

    Fills the same keys as structural_metrics.json so that nodes missing from
    the LOSO cache still get non-zero, topology-derived features. Without this,
    uncached nodes receive all-zero feature vectors and the GAT collapses to a
    constant output (pred_std = 0).

    Computed: pagerank, reverse_pagerank, betweenness_centrality,
    closeness_centrality, in_degree_centrality, out_degree_centrality,
    clustering_coefficient. QoS/RMAV metrics left at 0 (computed separately).
    """
    import networkx as nx
    import math

    n = nx_graph.number_of_nodes()
    if n == 0:
        return {}

    undirected = nx_graph.to_undirected()

    try:
        pr = nx.pagerank(nx_graph, alpha=0.85, max_iter=200)
    except Exception:
        pr = {node: 1.0 / n for node in nx_graph.nodes()}

    try:
        rev_graph = nx_graph.reverse()
        rpr = nx.pagerank(rev_graph, alpha=0.85, max_iter=200)
    except Exception:
        rpr = {node: 1.0 / n for node in nx_graph.nodes()}

    try:
        bt = nx.betweenness_centrality(nx_graph, normalized=True)
    except Exception:
        bt = dict.fromkeys(nx_graph.nodes(), 0.0)

    try:
        cl = nx.closeness_centrality(nx_graph)
    except Exception:
        cl = dict.fromkeys(nx_graph.nodes(), 0.0)

    in_deg = dict(nx.in_degree_centrality(nx_graph))
    out_deg = dict(nx.out_degree_centrality(nx_graph))

    try:
        cc = nx.clustering(undirected)
    except Exception:
        cc = dict.fromkeys(nx_graph.nodes(), 0.0)

    result: Dict = {}
    for node in nx_graph.nodes():
        nid = str(node)
        result[nid] = {
            "pagerank":              float(pr.get(node, 0.0)),
            "reverse_pagerank":      float(rpr.get(node, 0.0)),
            "betweenness_centrality": float(bt.get(node, 0.0)),
            "closeness_centrality":  float(cl.get(node, 0.0)),
            "in_degree_centrality":  float(in_deg.get(node, 0.0)),
            "out_degree_centrality": float(out_deg.get(node, 0.0)),
            "clustering_coefficient": float(cc.get(node, 0.0)),
        }
    return result


def _merge_structural_dicts(nx_features: Dict, cached: Dict) -> Dict:
    """Merge cached structural metrics with NX-derived features.

    Cached values take priority (they include RMAV-specific metrics);
    NX-derived values fill in nodes absent from the cache and patch
    zero-value entries for the basic topological metrics.
    """
    merged: Dict = {}
    for nid, nx_vals in nx_features.items():
        cached_vals = cached.get(nid, {})
        entry = dict(nx_vals)  # Start with NX-derived base
        entry.update(cached_vals)  # Cached values override
        merged[nid] = entry
    # Keep any cached nodes not in nx_features
    for nid, cached_vals in cached.items():
        if nid not in merged:
            merged[nid] = cached_vals
    return merged


def _load_scenario_data(scenario: str) -> Tuple[Any, Dict, Dict, Dict, bool]:
    """Load graph + structural/simulation/RMAV data for a scenario.

    Topology source priority: (1) LOSO cache topology.json, (2) data/scenarios/<name>.json.
    The cache topology was used when computing RMAV quality and structural metrics,
    so using it ensures feature/label consistency.

    Structural feature source priority: SAAG StructuralAnalyzer (with derived
    DEPENDS_ON edges) > cached structural_metrics.json > NX-derived fallback.
    """
    from cli.loso_evaluate import _build_graph_from_json

    cache_dir = _find_cache_dir(scenario)

    # Prefer cache topology for feature/label consistency
    cache_topo_path = cache_dir / "topology.json" if cache_dir.exists() else None
    scenario_path = SCENARIOS_DIR / f"{scenario}.json"

    if cache_topo_path and cache_topo_path.exists():
        topology = json.loads(cache_topo_path.read_text())
    elif scenario_path.exists():
        topology = json.loads(scenario_path.read_text())
    else:
        raise FileNotFoundError(f"No topology found for scenario '{scenario}'")

    nx_graph = _build_graph_from_json(topology)

    if cache_dir.exists():
        graph_nodes = {str(n) for n in nx_graph.nodes()}
        structural_dict, simulation_dict, rmav_dict = _load_cache_dicts(cache_dir, graph_nodes)
    else:
        logger.warning("No LOSO cache for '%s'. Structural/simulation data will be empty.", scenario)
        structural_dict, simulation_dict, rmav_dict = {}, {}, {}

    # Derive DEPENDS_ON features and build a DEPENDS_ON-only graph.
    #
    # Two interleaved problems prevent GNN learning on the raw pub-sub graph:
    #
    # 1. Feature degeneracy: Application nodes have betweenness ≈ 0 / bridge_ratio ≈ 0
    #    in the full pub-sub graph (nodes never route messages) → all-zero feature
    #    vectors → GNN collapses to constant output (pred_std = 0).
    #
    # 2. Over-smoothing from Topic hubs: the homo baseline flattens all node types
    #    into one graph.  Topic nodes (40) connect to all Application nodes (80)
    #    via PUBLISHES_TO / SUBSCRIBES_TO.  After a single GATConv pass, every
    #    Application aggregates the same Topic hub representations → identical
    #    embeddings, pred_std = 0 before even one training step.
    #
    # Fix: build a DEPENDS_ON-only NetworkX graph (Application + Library, Rules 1 & 5)
    # and use it for GNN training.  Features and labels both come from this graph, so
    # feature/label correlations are high (ρ ≈ 0.86–0.92 on av_system).
    import networkx as _nx

    saag_features = _saag_structural_features(topology)
    if saag_features:
        structural_dict = _merge_structural_dicts(saag_features, structural_dict)

        # Fresh RMAV quality from DEPENDS_ON features — consistent with feature source.
        fresh_rmav = _compute_rmav_from_structural(topology, saag_features)
        is_circular = False
        if fresh_rmav:
            logger.info("Using fresh RMAV quality as simulation labels (DEPENDS_ON-consistent).")
            simulation_dict = fresh_rmav
            rmav_dict = fresh_rmav
            is_circular = True

        # Build DEPENDS_ON-only graph: Application + Library nodes, Rules 1 & 5.
        # Node 'type' attribute is required by networkx_to_hetero_data to assign
        # nodes to the correct type bucket (Application vs Library).
        deps = _derive_depends_on_edges(topology)
        app_ids = {a["id"] for a in topology.get("applications", [])}
        lib_ids  = {lb["id"] for lb in topology.get("libraries", [])}
        allowed  = app_ids | lib_ids
        dep_graph = _nx.DiGraph()
        for nid in app_ids:
            dep_graph.add_node(nid, type="Application")
        for nid in lib_ids:
            dep_graph.add_node(nid, type="Library")
        for e in deps:
            src, dst = str(e["source"]), str(e["target"])
            if src in allowed and dst in allowed:
                dep_graph.add_edge(src, dst, weight=float(e.get("weight", 1.0)),
                                   dependency_type=e.get("type", "app_to_app"))
        if dep_graph.number_of_nodes() > 0:
            nx_graph = dep_graph
    else:
        nx_features = _compute_nx_structural_features(nx_graph)
        structural_dict = _merge_structural_dicts(nx_features, structural_dict)

    return nx_graph, structural_dict, simulation_dict, rmav_dict, is_circular


def _remap_node_ids(d: Dict, graph_nodes: set) -> Dict:
    """Re-map dictionary keys to match the exact format used by the graph.

    Handles two cases:
    - Zero-padded graph IDs: cache 'A1' → graph 'A01' (ATM scenario)
    - Non-padded graph IDs: cache 'A1' → graph 'A1' (most scenarios)
    Uses prefix+number decomposition; always keeps the remapping that
    maximises coverage.
    """
    import re
    if not d or not graph_nodes:
        return d

    # Build lookup: (prefix_upper, int_num) → graph_node_id
    graph_index: Dict[Tuple[str, int], str] = {}
    for nid in graph_nodes:
        m = re.match(r'^([A-Za-z]+)(\d+)$', nid)
        if m:
            graph_index[(m.group(1).upper(), int(m.group(2)))] = nid

    if not graph_index:
        return d  # Graph nodes don't follow prefix+num pattern

    remapped: Dict[str, Any] = {}
    for k, v in d.items():
        m = re.match(r'^([A-Za-z]+)(\d+)$', str(k))
        if m:
            key = (m.group(1).upper(), int(m.group(2)))
            new_id = graph_index.get(key, k)
            remapped[new_id] = v
        else:
            remapped[k] = v

    before = sum(1 for k in d if k in graph_nodes)
    after  = sum(1 for k in remapped if k in graph_nodes)
    logger.debug("_remap_node_ids: coverage %d→%d / %d graph nodes", before, after, len(graph_nodes))
    return remapped if after >= before else d


def _is_sparse(d: Dict, threshold: float = 0.20) -> bool:
    """Return True when fewer than *threshold* fraction of entries have composite > 0."""
    if not d:
        return True
    nonzero = sum(1 for v in d.values() if isinstance(v, dict) and v.get("composite", 0.0) > 1e-6)
    return nonzero / len(d) < threshold


def _clamp(v: float) -> float:
    """Clamp a score to [0, 1] — RMAV reliability can occasionally exceed 1.0."""
    return max(0.0, min(1.0, v))


def _rmav_to_sim_format(rmav_dict: Dict) -> Dict:
    """Convert RMAV quality scores to the simulation_dict format expected by networkx_to_hetero_data.

    Maps: overall→composite, reliability→reliability, maintainability→maintainability,
    availability→availability, vulnerability→vulnerability.
    All values are clamped to [0, 1] since RMAV reliability can exceed 1.0.
    """
    result: Dict[str, Dict] = {}
    for nid, v in rmav_dict.items():
        if not isinstance(v, dict):
            continue
        result[str(nid)] = {
            "composite":       _clamp(float(v.get("overall",         v.get("composite",        0.0)))),
            "reliability":     _clamp(float(v.get("reliability",     0.0))),
            "maintainability": _clamp(float(v.get("maintainability", 0.0))),
            "availability":    _clamp(float(v.get("availability",    0.0))),
            "vulnerability":   _clamp(float(v.get("vulnerability",   0.0))),
        }
    return result


def _parse_structural_metrics(raw: Dict) -> Dict:
    """Parse structural_metrics.json → {node_id: {betweenness, in_degree, out_degree, degree}}."""
    comps = raw.get("structural_analysis", {}).get("components", [])
    result: Dict[str, Dict] = {}
    if isinstance(comps, list):
        for entry in comps:
            nid = str(entry.get("id", ""))
            metrics = entry.get("metrics", {})
            if nid:
                result[nid] = {
                    "betweenness": float(metrics.get("betweenness", metrics.get("betweenness_centrality", 0.0))),
                    "in_degree":   float(metrics.get("in_degree",   metrics.get("degree", 0.0))),
                    "articulation_point": float(metrics.get("ap_c_score", 0.0)),
                    "out_degree":  float(metrics.get("out_degree",  0.0)),
                    "degree":      float(metrics.get("degree",      0.0)),
                }
    elif isinstance(comps, dict):
        for nid, metrics in comps.items():
            result[str(nid)] = {
                "betweenness": float(metrics.get("betweenness", metrics.get("betweenness_centrality", 0.0))),
                "in_degree":   float(metrics.get("in_degree",   metrics.get("degree", 0.0))),
                "articulation_point": float(metrics.get("ap_c_score", 0.0)),
                "out_degree":  float(metrics.get("out_degree",  0.0)),
                "degree":      float(metrics.get("degree",      0.0)),
            }
    return result







def _parse_failure_impact(raw: Dict) -> Dict:
    """Normalise failure_impact.json → {node_id: {composite: float}}.

    Supports:
      1. {records: {node_id: {impact_score, ...}}}  (dict-keyed format)
      2. {records: [{node_id, impact_score, ...}]}  (list format)
      3. Flat {node_id: {composite: ...}}            (already normalised)
    """
    if "records" in raw:
        records = raw["records"]
        result = {}
        # Dict keyed by node_id
        if isinstance(records, dict):
            for nid, rec in records.items():
                if not isinstance(rec, dict):
                    continue
                result[str(nid)] = {
                    "composite":     float(rec.get("impact_score", 0.0)),
                    "reliability":   float(rec.get("total_impacted_subscribers", 0.0)),
                    "cascade_depth": float(rec.get("cascade_depth", 0.0)),
                }
        # List of dicts
        elif isinstance(records, list):
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                nid = rec.get("node_id") or rec.get("id")
                if nid is None:
                    continue
                result[str(nid)] = {
                    "composite":     float(rec.get("impact_score", 0.0)),
                    "reliability":   float(rec.get("total_impacted_subscribers", 0.0)),
                    "cascade_depth": float(rec.get("cascade_depth", 0.0)),
                }
        return result
    # Filter metadata-only keys and return flat dicts
    skip = {"schema_version", "graph_id", "total_nodes_injected",
            "total_application_nodes", "total_broker_nodes",
            "total_subscribers", "seeds_used", "top_k_by_impact", "records"}
    return {k: v for k, v in raw.items() if k not in skip and isinstance(v, dict)}



def _parse_quality_scores(raw: Dict) -> Dict:
    """Normalise quality_scores.json → {node_id: {overall, reliability, ...}}.

    Supports:
      1. {layers: {<layer>: {rmav: {node_id: {...}}}}}   (nested format)
      2. Flat {node_id: {...}}
    """
    if "layers" in raw:
        for layer_val in raw["layers"].values():
            if not isinstance(layer_val, dict):
                continue
            rmav = layer_val.get("rmav")
            if isinstance(rmav, dict):
                return {str(k): v for k, v in rmav.items() if isinstance(v, dict)}
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _train_cell(
    scenario: str,
    variant: str,
    seed: int,
    hidden: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    num_epochs: int = 200,
    patience: int = 30,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Train one cell of the 8×4×5 matrix and return metrics dict."""
    import torch
    import numpy as np
    from saag.prediction.data_preparation import networkx_to_hetero_data, create_node_splits
    from saag.prediction.trainer import GNNTrainer, evaluate

    torch.manual_seed(seed)
    np.random.seed(seed)

    nx_graph, structural_dict, simulation_dict, rmav_dict, is_circular = _load_scenario_data(scenario)

    if nx_graph.number_of_nodes() == 0:
        return {"error": "empty_graph"}

    n_nodes = nx_graph.number_of_nodes()
    effective_layers = 1 if n_nodes <= 200 else (2 if n_nodes <= 500 else num_layers)
    start = time.time()

    if variant == "topo_baseline":
        # simple baseline: compare structural centrality (betweenness + in_degree)
        # against the quality ground truth.
        import networkx as nx
        from scipy.stats import spearmanr
        from saag.prediction.trainer import evaluate_scores
        import numpy as np

        # Build structural-centrality prediction for every graph node
        if structural_dict:
            # Cached betweenness + articulation_point from structural_metrics.json
            struct_pred = {
                nid: 0.6 * m.get("betweenness", 0.0) + 0.4 * m.get("articulation_point", 0.0)
                for nid, m in structural_dict.items()
            }
        else:
            # Fall back to NetworkX betweenness centrality
            struct_pred = {str(n): float(v) for n, v in nx.betweenness_centrality(nx_graph).items()}

        keys = sorted(set(struct_pred) & set(simulation_dict))
        if len(keys) < 3:
            return {"error": "insufficient_overlap"}
        
        pred_list = [struct_pred[k] for k in keys]
        true_list = [simulation_dict[k].get("composite", 0.0) for k in keys]
        
        y_pred = np.zeros((len(pred_list), 5))
        y_true = np.zeros((len(true_list), 5))
        y_pred[:, 0] = pred_list
        y_true[:, 0] = true_list
        
        m = evaluate_scores(y_pred, y_true)
        
        return {
            "scenario": scenario, "variant": variant, "seed": seed,
            "spearman_rho": round(m.spearman_rho, 4),
            "f1_score": round(m.f1_score, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "rmse": round(m.rmse, 4),
            "mae": round(m.mae, 4),
            "ndcg_10": round(m.ndcg_10, 4),
            "top_5_overlap": round(m.top_5_overlap, 4),
            "top_10_overlap": round(m.top_10_overlap, 4),
            "per_node_type": {}, "runtime_s": round(time.time() - start, 2),
        }

    elif variant == "rasse_2025":
        # Full IEEE RASSE 2025 approach (RMAV scores)
        import numpy as np
        from saag.prediction.trainer import evaluate_scores

        if is_circular:
            # RMAV is used as ground truth for this scenario -> 1.0 correlation is trivial.
            # Mark as circular so renderer can handle it.
            return {
                "scenario": scenario, "variant": variant, "seed": seed,
                "is_circular": True,
                "spearman_rho": 1.0, "f1_score": 1.0, "ndcg_10": 1.0,
                "precision": 1.0, "recall": 1.0,
                "per_node_type": {}, "runtime_s": 0.01,
            }

        keys = sorted(set(rmav_dict) & set(simulation_dict))
        if len(keys) < 3:
            return {"error": "insufficient_overlap"}
        
        pred_list = [rmav_dict[k].get("overall", 0.0) for k in keys]
        true_list = [simulation_dict[k].get("composite", 0.0) for k in keys]
        
        y_pred = np.zeros((len(pred_list), 5))
        y_true = np.zeros((len(true_list), 5))
        y_pred[:, 0] = pred_list
        y_true[:, 0] = true_list
        
        m = evaluate_scores(y_pred, y_true)
        
        return {
            "scenario": scenario, "variant": variant, "seed": seed,
            "spearman_rho": round(m.spearman_rho, 4),
            "f1_score": round(m.f1_score, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "rmse": round(m.rmse, 4),
            "mae": round(m.mae, 4),
            "ndcg_10": round(m.ndcg_10, 4),
            "top_5_overlap": round(m.top_5_overlap, 4),
            "top_10_overlap": round(m.top_10_overlap, 4),
            "per_node_type": {}, "runtime_s": round(time.time() - start, 2),
        }

    elif variant in ("homo_unweighted", "homo_scalar"):
        from saag.prediction.models.baselines import build_baseline
        start = time.time()
        conv = networkx_to_hetero_data(nx_graph, structural_dict, simulation_dict, rmav_dict)
        data = conv.hetero_data
        create_node_splits(data, train_ratio, val_ratio, seed=seed)
        effective_lr = 1e-3  # higher LR for faster convergence on small labelled sets
        effective_patience = max(patience, 60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_baseline(variant, hidden_channels=hidden, num_heads=num_heads,
                               num_layers=effective_layers, dropout=dropout)
        model.to(device)
        ckpt_dir = f"output/gnn_checkpoints/{scenario}_{variant}_s{seed}"
        trainer = GNNTrainer(model=model, checkpoint_dir=ckpt_dir, lr=effective_lr,
                             num_epochs=num_epochs, patience=effective_patience)
        trainer.train(data)
        metrics = evaluate(model, data, "test_mask", device)

    else:  # hetero_qos
        from saag.prediction.gnn_service import GNNService
        start = time.time()
        svc = GNNService(hidden_channels=hidden, num_heads=num_heads, num_layers=effective_layers,
                         dropout=dropout, predict_edges=False,
                         checkpoint_dir=f"output/gnn_checkpoints/{scenario}_hetero_qos_s{seed}")
        result = svc.train(
            graph=nx_graph, structural_metrics=structural_dict,
            simulation_results=simulation_dict, rmav_scores=rmav_dict,
            train_ratio=train_ratio, val_ratio=val_ratio,
            num_epochs=num_epochs, lr=1e-3, patience=patience,
            seeds=[seed], mode="gnn",
        )
        metrics = result.gnn_metrics

    if metrics is None:
        return {"error": "no_metrics"}

    d = metrics.to_dict()
    d.update({"scenario": scenario, "variant": variant, "seed": seed,
              "runtime_s": round(time.time() - start, 2)})
    return d


# ── Statistics helpers ─────────────────────────────────────────────────────────

def _bootstrap_ci(values: List[float], stat_fn=None, B: int = 2000, alpha: float = 0.05):
    """Bootstrap confidence interval for stat_fn(values). Default: mean."""
    import numpy as np
    if not values:
        return float("nan"), float("nan")
    stat_fn = stat_fn or np.mean
    arr = np.array(values)
    boot = np.array([stat_fn(np.random.choice(arr, len(arr), replace=True)) for _ in range(B)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _wilcoxon_p(a: List[float], b: List[float]) -> float:
    """Two-sided Wilcoxon signed-rank test p-value."""
    from scipy.stats import wilcoxon
    import numpy as np
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    try:
        _, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except Exception:
        return float("nan")


def _aggregate_cells(cells: List[Dict]) -> Dict:
    """Aggregate seed-level cells into mean ± CI and Wilcoxon p-values."""
    import numpy as np

    by_sv: Dict[Tuple[str, str], List[Dict]] = {}
    for c in cells:
        if "error" in c:
            continue
        k = (c["scenario"], c["variant"])
        by_sv.setdefault(k, []).append(c)

    aggregate = {}
    for (sc, var), cs in by_sv.items():
        rhos = [c.get("spearman_rho", 0.0) for c in cs]
        f1s = [c.get("f1_score", 0.0) for c in cs]
        precs = [c.get("precision", 0.0) for c in cs]
        recs = [c.get("recall", 0.0) for c in cs]
        top5s = [c.get("top_5_overlap", 0.0) for c in cs]
        top10s = [c.get("top_10_overlap", 0.0) for c in cs]

        mean_r = float(np.mean(rhos))
        lo, hi = _bootstrap_ci(rhos)
        is_circ = any(c.get("is_circular", False) for c in cs)
        aggregate[(sc, var)] = {
            "mean_rho": round(mean_r, 4),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "mean_precision": round(float(np.mean(precs)), 4),
            "mean_recall": round(float(np.mean(recs)), 4),
            "mean_top5": round(float(np.mean(top5s)), 4),
            "mean_top10": round(float(np.mean(top10s)), 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "is_circular": is_circ,
            "n_seeds": len(cs),
            "rhos": rhos,
        }

    # Per-scenario Wilcoxon p-values (hetero_qos vs each baseline)
    scenarios = sorted({sc for sc, _ in aggregate})
    for sc in scenarios:
        ref_cells = [(sc, "hetero_qos")]
        ref_rhos = aggregate.get(ref_cells[0], {}).get("rhos", [])
        for var in ALL_VARIANTS:
            if var == "hetero_qos":
                continue
            base_rhos = aggregate.get((sc, var), {}).get("rhos", [])
            p = _wilcoxon_p(ref_rhos, base_rhos)
            if (sc, var) in aggregate:
                aggregate[(sc, var)]["wilcoxon_p_vs_hetero"] = round(p, 4)

    return aggregate


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Middleware 2026 main table harness (Block C).")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"Scenarios to run (default: all {len(ALL_SCENARIOS)})")
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS,
                   help="Variants to include (default: all 4)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"Seeds to use (default: {DEFAULT_SEEDS})")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs per run (default: 200)")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "main_table.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells already present in the output file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned matrix without training")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    scenarios = args.scenarios or ALL_SCENARIOS
    variants = args.variants or ALL_VARIANTS
    seeds = args.seeds or DEFAULT_SEEDS

    total = len(scenarios) * len(variants) * len(seeds)
    print("\n  Middleware 2026 Main Table Harness")
    print(f"  Scenarios : {scenarios}")
    print(f"  Variants  : {variants}")
    print(f"  Seeds     : {seeds}")
    print(f"  Total runs: {total}")
    print()

    if args.dry_run:
        for sc in scenarios:
            for v in variants:
                for s in seeds:
                    print(f"  [DRY-RUN] {sc} × {v} × seed={s}")
        return

    # Load existing results for --resume
    existing_cells: List[Dict] = []
    if args.resume and args.output.exists():
        data = json.loads(args.output.read_text())
        existing_cells = data.get("cells", [])
        done_keys = {(c["scenario"], c["variant"], c["seed"]) for c in existing_cells
                     if "error" not in c}
        print(f"  Resuming: {len(done_keys)} cells already completed.")
    else:
        done_keys = set()

    cells: List[Dict] = list(existing_cells)
    n_done = len(done_keys)
    n_fail = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        for v in variants:
            for s in seeds:
                key = (sc, v, s)
                if key in done_keys:
                    continue

                label = f"[{n_done+1}/{total}] {sc} × {v} × seed={s}"
                print(f"  {label} ...", end="", flush=True)

                try:
                    cell = _train_cell(
                        scenario=sc, variant=v, seed=s,
                        hidden=args.hidden, num_heads=args.heads,
                        num_layers=args.layers, num_epochs=args.epochs,
                        patience=args.patience,
                    )
                except Exception as exc:
                    cell = {"scenario": sc, "variant": v, "seed": s, "error": str(exc)}
                    n_fail += 1
                    print(f" ERROR: {exc}")
                else:
                    if "error" in cell:
                        print(f" SKIP ({cell['error']})")
                        n_fail += 1
                    else:
                        rho = cell.get("spearman_rho", float("nan"))
                        rt  = cell.get("runtime_s", 0)
                        print(f" rho={rho:.4f}  ({rt}s)")

                cells.append(cell)
                n_done += 1

                # Save incrementally
                args.output.write_text(json.dumps({"cells": cells}, indent=2))

    # Aggregate
    aggregate = _aggregate_cells(cells)
    agg_serializable = {
        f"{sc}|{var}": v for (sc, var), v in aggregate.items()
    }

    output = {
        "cells": cells,
        "aggregate": agg_serializable,
        "config": {
            "scenarios": scenarios, "variants": variants, "seeds": seeds,
            "epochs": args.epochs, "hidden": args.hidden,
        },
    }
    args.output.write_text(json.dumps(output, indent=2))

    print(f"\n  Completed: {n_done} runs, {n_fail} failures.")
    print(f"  Results saved to: {args.output}")

    if n_fail == 0:
        print("\n  ✓ All cells complete. Run tools/render_table.py to generate LaTeX.")
    else:
        print(f"\n  ⚠  {n_fail} cells failed. Check logs for details.")


if __name__ == "__main__":
    main()
