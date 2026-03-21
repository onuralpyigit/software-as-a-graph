"""
GNN Data Preparation
====================
Converts the Software-as-a-Graph framework's internal representations into
PyTorch Geometric (PyG) HeteroData objects suitable for training and inference.

Design principles
-----------------
* **Heterogeneous-first**: The pub-sub multi-layer graph has five distinct
  node types (Application, Broker, Topic, Node, Library) and seven edge types
  (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES,
  DEPENDS_ON).  All type information is preserved as separate PyG stores.

* **Feature parity with RMAV**: The 13 topological metrics that feed the RMAV
  quality scorer are reused directly as node features so GNN predictions are
  directly comparable to RMAV predictions under the same validation protocol.

* **Multi-task labels**: All five simulation ground-truth dimensions
  (I*(v), IR(v), IM(v), IA(v), IV(v)) are stored as multi-column label
  tensors, enabling multi-task learning or single-task ablations.

* **Edge labels**: Edge criticality scores are derived from the endpoint nodes'
  composite impact scores (max pooling), enabling link-level criticality
  prediction as a companion task.

Node feature vector (dim = 27)
-------------------------------
Index  Metric
  0    PageRank (PR)
  1    Reverse PageRank (RPR)
  2    Betweenness Centrality (BT)
  3    Closeness Centrality (CL)
  4    Eigenvector Centrality (EV)
  5    In-Degree normalised (DG_in)
  6    Out-Degree normalised (DG_out)
  7    Clustering Coefficient (CC)
  8    AP_c undirected
  9    Bridge Ratio (BR)
 10    QoS aggregate weight (w)
 11    QoS weighted in-degree (w_in)
 12    QoS weighted out-degree (w_out)
 13    MPCI
 14    Fan-Out Criticality (FOC)
 15    AP_c directed
 16    CDI
 17    Normalised LOC (loc_norm)
 18    Normalised Complexity (complexity_norm)
 19    Instability I = Ce/(Ca+Ce) (instability_code)
 20    Normalised LCOM (lcom_norm)
 21    Code Quality Penalty (CQP)
 22-26 Node-type one-hot (Application, Broker, Topic, Node, Library)

Edge feature vector (dim = 8)
------------------------------
Index  Feature
  0    QoS-derived edge weight (normalised)
  1-7  Edge-type one-hot (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES,
                          RUNS_ON, CONNECTS_TO, USES, DEPENDS_ON)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

NODE_TYPES: List[str] = ["Application", "Broker", "Topic", "Node", "Library"]
EDGE_TYPES: List[str] = [
    "PUBLISHES_TO",
    "SUBSCRIBES_TO",
    "ROUTES",
    "RUNS_ON",
    "CONNECTS_TO",
    "USES",
    "DEPENDS_ON",
]

NODE_TYPE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(NODE_TYPES)}
EDGE_TYPE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(EDGE_TYPES)}

# The 22 topological metrics extracted from the structural analysis result.
# Order matches Step 3: Prediction doc (indices 0-21)
TOPOLOGICAL_METRIC_KEYS: List[str] = [
    "pagerank",              # 0
    "reverse_pagerank",      # 1
    "betweenness_centrality", # 2
    "closeness_centrality",   # 3
    "eigenvector_centrality", # 4
    "in_degree_centrality",   # 5
    "out_degree_centrality",  # 6
    "clustering_coefficient", # 7
    "ap_c_score",            # 8: AP_c undirected (proxied in Step 2)
    "bridge_ratio",          # 9
    "qos_weight",            # 10
    "qos_weight_in",         # 11: w_in (QADS)
    "qos_weight_out",        # 12: w_out
    "mpci",                  # 13: New Tier 1
    "fan_out_criticality",   # 14
    "ap_c_directed",         # 15
    "cdi",                   # 16
    "loc_norm",              # 17: Code Quality
    "complexity_norm",       # 18
    "instability_code",      # 19
    "lcom_norm",             # 20
    "code_quality_penalty",  # 21
]

NODE_FEATURE_DIM = len(TOPOLOGICAL_METRIC_KEYS) + len(NODE_TYPES)  # 27
EDGE_FEATURE_DIM = 1 + len(EDGE_TYPES)                              # 8

# Label column indices in the (N, 5) label matrix
LABEL_COLS = {
    "composite": 0,
    "reliability": 1,
    "maintainability": 2,
    "availability": 3,
    "vulnerability": 4,
}

# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class GraphConversionResult:
    """Output of :func:`networkx_to_hetero_data`."""

    # Core PyG HeteroData object — import lazily to avoid hard dependency
    hetero_data: object

    # Mapping from (node_type, local_index) → global node name (string ID)
    node_id_map: Dict[str, List[str]] = field(default_factory=dict)

    # Reverse: global node name → (node_type, local_index)
    node_name_to_idx: Dict[str, Tuple[str, int]] = field(default_factory=dict)

    # Mapping from (src_type, rel, dst_type) → list of (src_name, dst_name)
    edge_name_map: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = field(
        default_factory=dict
    )

    # Number of labelled nodes (nodes with simulation ground truth)
    num_labelled_nodes: int = 0

    # List of node types actually present in this graph
    present_node_types: List[str] = field(default_factory=list)


def networkx_to_hetero_data(
    graph: nx.DiGraph,
    structural_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    simulation_results: Optional[Dict[str, Dict[str, float]]] = None,
    rmav_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> GraphConversionResult:
    """Convert a NetworkX DiGraph to a PyG HeteroData object.

    Parameters
    ----------
    graph:
        The full structural NetworkX graph produced by Step 1 of the pipeline.
        Nodes must carry a ``type`` attribute (one of NODE_TYPES).
        Edges must carry a ``type`` attribute (one of EDGE_TYPES) and
        optionally a ``weight`` attribute (float).
    structural_metrics:
        ``{node_name: {metric_key: value}}`` dict produced by the
        ``StructuralAnalyzer``.  When provided, the 13 topological metrics
        are used as node features.  When absent, features default to zeros
        (inference mode without pre-computed metrics).
    simulation_results:
        ``{node_name: {composite, reliability, maintainability,
                        availability, vulnerability}}`` dict produced by
        the ``SimulationService``.  Used as training labels.
    rmav_scores:
        ``{node_name: {overall, reliability, maintainability,
                        availability, vulnerability}}`` produced by
        the ``AnalysisService``.  Stored as an alternative label tensor
        (``y_rmav``) for comparison / ensemble purposes.

    Returns
    -------
    GraphConversionResult
        Contains the HeteroData object plus index maps needed to map
        GNN predictions back to named components.
    """
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for GNN functionality. "
            "Install with: pip install torch-geometric"
        ) from exc

    result = GraphConversionResult(hetero_data=HeteroData())
    data: HeteroData = result.hetero_data  # type: ignore[assignment]

    # ── 1. Partition nodes by type ────────────────────────────────────────────
    type_to_nodes: Dict[str, List[str]] = {t: [] for t in NODE_TYPES}

    for node, attrs in graph.nodes(data=True):
        # Support both 'type' (used in fallback/simple graphs) 
        # and 'component_type' (used by StructuralAnalyzer/AnalysisService)
        node_type = attrs.get("type") or attrs.get("component_type") or "Application"
        
        if node_type not in type_to_nodes:
            logger.warning(
                "Unknown node type '%s' for node '%s'; treating as Application.",
                node_type, node,
            )
            node_type = "Application"
        type_to_nodes[node_type].append(node)

    # Build index maps
    for node_type, nodes in type_to_nodes.items():
        if not nodes:
            continue
        result.node_id_map[node_type] = nodes
        result.present_node_types.append(node_type)
        for local_idx, name in enumerate(nodes):
            result.node_name_to_idx[name] = (node_type, local_idx)

    # ── 2. Build node feature tensors per type ────────────────────────────────
    for node_type in result.present_node_types:
        nodes = result.node_id_map[node_type]
        n = len(nodes)
        feat_matrix = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)

        for local_idx, name in enumerate(nodes):
            # Topological metrics (indices 0-12)
            if structural_metrics and name in structural_metrics:
                metrics = structural_metrics[name]
                for col, key in enumerate(TOPOLOGICAL_METRIC_KEYS):
                    feat_matrix[local_idx, col] = float(metrics.get(key, 0.0))

            # Node type one-hot (indices 18-22)
            type_col = len(TOPOLOGICAL_METRIC_KEYS) + NODE_TYPE_INDEX.get(node_type, 0)
            feat_matrix[local_idx, type_col] = 1.0

        data[node_type].x = torch.from_numpy(feat_matrix)
        data[node_type].num_nodes = n

        # ── Node labels (simulation ground truth) ─────────────────────────────
        if simulation_results:
            label_matrix = np.zeros((n, 5), dtype=np.float32)
            labelled_count = 0
            for local_idx, name in enumerate(nodes):
                sim = simulation_results.get(name)
                if sim is not None:
                    label_matrix[local_idx, 0] = float(sim.get("composite", 0.0))
                    label_matrix[local_idx, 1] = float(sim.get("reliability", 0.0))
                    label_matrix[local_idx, 2] = float(sim.get("maintainability", 0.0))
                    label_matrix[local_idx, 3] = float(sim.get("availability", 0.0))
                    label_matrix[local_idx, 4] = float(sim.get("vulnerability", 0.0))
                    labelled_count += 1
            data[node_type].y = torch.from_numpy(label_matrix)
            result.num_labelled_nodes += labelled_count

        # ── RMAV scores (for ensemble / comparison) ───────────────────────────
        if rmav_scores:
            rmav_matrix = np.zeros((n, 5), dtype=np.float32)
            for local_idx, name in enumerate(nodes):
                rmav = rmav_scores.get(name)
                if rmav is not None:
                    rmav_matrix[local_idx, 0] = float(rmav.get("overall", 0.0))
                    rmav_matrix[local_idx, 1] = float(rmav.get("reliability", 0.0))
                    rmav_matrix[local_idx, 2] = float(rmav.get("maintainability", 0.0))
                    rmav_matrix[local_idx, 3] = float(rmav.get("availability", 0.0))
                    rmav_matrix[local_idx, 4] = float(rmav.get("vulnerability", 0.0))
            data[node_type].y_rmav = torch.from_numpy(rmav_matrix)

    # ── 3. Build edge index and feature tensors per relation ──────────────────
    # Group edges by (src_type, edge_type, dst_type)
    rel_edges: Dict[Tuple[str, str, str], Tuple[List[int], List[int], List[List[float]]]] = {}

    for src, dst, attrs in graph.edges(data=True):
        if src not in result.node_name_to_idx or dst not in result.node_name_to_idx:
            continue  # skip edges whose endpoints weren't indexed

        src_type, src_local = result.node_name_to_idx[src]
        dst_type, dst_local = result.node_name_to_idx[dst]
        edge_type = attrs.get("type", "DEPENDS_ON")

        rel_key = (src_type, edge_type, dst_type)

        if rel_key not in rel_edges:
            rel_edges[rel_key] = ([], [], [])
            result.edge_name_map[rel_key] = []

        rel_edges[rel_key][0].append(src_local)
        rel_edges[rel_key][1].append(dst_local)
        result.edge_name_map[rel_key].append((src, dst))

        # Edge features: [weight, type_onehot x7]
        weight = float(attrs.get("weight", 1.0))
        type_onehot = [0.0] * len(EDGE_TYPES)
        if edge_type in EDGE_TYPE_INDEX:
            type_onehot[EDGE_TYPE_INDEX[edge_type]] = 1.0
        rel_edges[rel_key][2].append([weight] + type_onehot)

    # Write edge stores into HeteroData
    for (src_type, edge_type, dst_type), (srcs, dsts, feats) in rel_edges.items():
        rel = (src_type, edge_type, dst_type)
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_attr = torch.tensor(feats, dtype=torch.float32)

        data[rel].edge_index = edge_index
        data[rel].edge_attr = edge_attr

        # Edge labels: max(I*(src), I*(dst)) — used for link criticality prediction
        if simulation_results:
            src_nodes = result.node_id_map[src_type]
            dst_nodes = result.node_id_map[dst_type]
            edge_labels = np.zeros((len(srcs), 5), dtype=np.float32)
            for i, (s_idx, d_idx) in enumerate(zip(srcs, dsts)):
                s_name = src_nodes[s_idx]
                d_name = dst_nodes[d_idx]
                s_sim = simulation_results.get(s_name, {})
                d_sim = simulation_results.get(d_name, {})
                for col, key in enumerate(
                    ["composite", "reliability", "maintainability", "availability", "vulnerability"]
                ):
                    edge_labels[i, col] = max(
                        float(s_sim.get(key, 0.0)),
                        float(d_sim.get(key, 0.0)),
                    )
            data[rel].y_edge = torch.from_numpy(edge_labels)

    logger.info(
        "Graph converted: %d node types, %d relation types, %d labelled nodes.",
        len(result.present_node_types),
        len(rel_edges),
        result.num_labelled_nodes,
    )
    if result.num_labelled_nodes == 0 and simulation_results:
        logger.warning(
            "ZERO labelled nodes found! Check if component IDs in simulation results "
            "match those in the graph. Sample graph nodes: %s, Sample sim keys: %s",
            list(graph.nodes())[:5], list(simulation_results.keys())[:5]
        )
    return result


# ── Training split utilities ───────────────────────────────────────────────────

def create_node_splits(
    hetero_data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Add train/val/test boolean masks to every node store in-place.

    When only a single graph is available (transductive setting), we randomly
    split labelled nodes into train/val/test sets for each node type
    independently.

    Parameters
    ----------
    hetero_data:
        The HeteroData object returned by :func:`networkx_to_hetero_data`.
    train_ratio, val_ratio:
        Fractions of labelled nodes assigned to train and validation sets.
        Remaining nodes form the test set.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    for store in hetero_data.node_stores:
        n = store.num_nodes
        if n == 0:
            continue

        indices = rng.permutation(n)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[torch.from_numpy(indices[:n_train])] = True
        val_mask[torch.from_numpy(indices[n_train: n_train + n_val])] = True
        test_mask[torch.from_numpy(indices[n_train + n_val:])] = True

        store.train_mask = train_mask
        store.val_mask = val_mask
        store.test_mask = test_mask


def extract_simulation_dict(simulation_results: Union[list, dict]) -> Dict[str, Dict[str, float]]:
    """Normalise Simulation output to common flat dict format."""
    out: Dict[str, Dict[str, float]] = {}
    
    # Handle the new SimulationReport dict structure
    if isinstance(simulation_results, dict) and "component_criticality" in simulation_results:
        for c in simulation_results["component_criticality"]:
            name = c.get("id")
            out[name] = {
                "composite": float(c.get("combined_impact", 0.0)),
                "reliability": float(c.get("failure_impact", 0.0)), # Map failure_impact to reliability ground truth
                "maintainability": 0.0, # Not available in summary
                "availability": float(c.get("failure_impact", 0.0)), # Often same or similar in summary
                "vulnerability": 0.0,
            }
        return out

    # Handle standard list of FailureResult objects/dicts
    results_list = simulation_results
    if isinstance(simulation_results, dict) and "results" in simulation_results:
        results_list = simulation_results["results"]

    if not isinstance(results_list, list):
        return out

    for r in results_list:
        if hasattr(r, "target_id"):
            name = r.target_id
            impact = r.impact
            out[name] = {
                "composite": float(impact.composite_impact),
                "reliability": float(impact.reliability_impact),
                "maintainability": float(impact.maintainability_impact),
                "availability": float(impact.availability_impact),
                "vulnerability": float(impact.vulnerability_impact),
            }
        elif isinstance(r, dict):
            name = r.get("target_id")
            if not name: continue
            impact = r.get("impact", r) # Might be nested or flat
            out[name] = {
                "composite": float(impact.get("composite_impact", 0.0)),
                "reliability": float(impact.get("reliability_impact", 0.0)),
                "maintainability": float(impact.get("maintainability_impact", 0.0)),
                "availability": float(impact.get("availability_impact", 0.0)),
                "vulnerability": float(impact.get("vulnerability_impact", 0.0)),
            }
    return out


def extract_structural_metrics_dict(structural_result) -> Dict[str, Dict[str, float]]:
    """Normalise StructuralAnalyzer output to the flat dict expected by this module.

    Handles both the ``StructuralAnalysisResult`` dataclass returned by the
    ``AnalysisService`` and raw dict representations.
    """
    out: Dict[str, Dict[str, float]] = {}

    def _from_component(comp):
        def _get(obj, attr, default=0.0):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        return {
            "pagerank": float(_get(comp, "pagerank")),
            "reverse_pagerank": float(_get(comp, "reverse_pagerank")),
            "betweenness_centrality": float(_get(comp, "betweenness")),
            "closeness_centrality": float(_get(comp, "closeness")),
            "eigenvector_centrality": float(_get(comp, "eigenvector")),
            "in_degree_centrality": float(_get(comp, "in_degree")),
            "out_degree_centrality": float(_get(comp, "out_degree")),
            "clustering_coefficient": float(_get(comp, "clustering_coefficient")),
            "ap_c_score": float(_get(comp, "ap_c_directed")),
            "ap_c_directed": float(_get(comp, "ap_c_directed")),
            "cdi": float(_get(comp, "cdi")),
            "mpci": float(_get(comp, "mpci")),
            "fan_out_criticality": float(_get(comp, "fan_out_criticality")),
            "bridge_ratio": float(_get(comp, "bridge_ratio")),
            "qos_weight": float(_get(comp, "weight", 1.0)),
            "qos_weight_in": float(_get(comp, "dependency_weight_in")),
            "qos_weight_out": float(_get(comp, "dependency_weight_out")),
            "loc_norm": float(_get(comp, "loc_norm")),
            "complexity_norm": float(_get(comp, "complexity_norm")),
            "instability_code": float(_get(comp, "instability_code")),
            "lcom_norm": float(_get(comp, "lcom_norm")),
            "code_quality_penalty": float(_get(comp, "code_quality_penalty")),
        }

    # Handle StructuralAnalysisResult with .components list
    if hasattr(structural_result, "components"):
        # Support both components list and dict
        components = structural_result.components
        if hasattr(components, "values"):
            components = components.values()
        
        for comp in components:
            name = getattr(comp, "component_id", getattr(comp, "name", str(getattr(comp, "id", comp))))
            out[name] = _from_component(comp)
    # Handle dict of {name: component_dict} or nested structural dict
    elif isinstance(structural_result, dict):
        components = structural_result.get("components", structural_result)
        # If it was a nested dict with 'components' field, it might be a list or a dict
        if isinstance(components, dict):
             for name, comp in components.items():
                out[name] = _from_component(comp)
        elif isinstance(components, list):
            for comp in components:
                name = getattr(comp, "component_id", getattr(comp, "name", str(getattr(comp, "id", comp))))
                if isinstance(comp, dict):
                    name = comp.get("component_id", comp.get("name", comp.get("id", str(comp))))
                out[name] = _from_component(comp)
        else:
            # Fallback for unexpected formats
            for name, comp in structural_result.items():
                out[name] = _from_component(comp)

    return out


def extract_rmav_scores_dict(quality_result) -> Dict[str, Dict[str, float]]:
    """Normalise QualityAnalyzer output (RMAV scores) to a flat dict."""
    out: Dict[str, Dict[str, float]] = {}

    if hasattr(quality_result, "components"):
        for comp in quality_result.components:
            name = getattr(comp, "component_id", getattr(comp, "name", str(comp)))
            scores = getattr(comp, "scores", None)
            if scores is not None:
                out[name] = {
                    "overall": float(getattr(scores, "overall", 0.0)),
                    "reliability": float(getattr(scores, "reliability", 0.0)),
                    "maintainability": float(getattr(scores, "maintainability", 0.0)),
                    "availability": float(getattr(scores, "availability", 0.0)),
                    "vulnerability": float(getattr(scores, "vulnerability", 0.0)),
                }
    # Handle dict of {name: scores_dict} or nested quality dict
    elif isinstance(quality_result, dict):
        components = quality_result.get("components", quality_result)
        if isinstance(components, list):
            for comp in components:
                name = getattr(comp, "component_id", getattr(comp, "name", str(comp)))
                if isinstance(comp, dict):
                    name = comp.get("component_id", comp.get("name", comp.get("id", str(comp))))
                scores = getattr(comp, "scores", None)
                if isinstance(comp, dict):
                    scores = comp.get("scores", comp)
                if scores is not None:
                    out[name] = {
                        "overall": float(getattr(scores, "overall", 0.0) if not isinstance(scores, dict) else scores.get("overall", 0.0)),
                        "reliability": float(getattr(scores, "reliability", 0.0) if not isinstance(scores, dict) else scores.get("reliability", 0.0)),
                        "maintainability": float(getattr(scores, "maintainability", 0.0) if not isinstance(scores, dict) else scores.get("maintainability", 0.0)),
                        "availability": float(getattr(scores, "availability", 0.0) if not isinstance(scores, dict) else scores.get("availability", 0.0)),
                        "vulnerability": float(getattr(scores, "vulnerability", 0.0) if not isinstance(scores, dict) else scores.get("vulnerability", 0.0)),
                    }
        elif isinstance(components, dict):
            for name, scores in components.items():
                if isinstance(scores, dict):
                    out[name] = {
                        "overall": float(scores.get("overall", 0.0)),
                        "reliability": float(scores.get("reliability", 0.0)),
                        "maintainability": float(scores.get("maintainability", 0.0)),
                        "availability": float(scores.get("availability", 0.0)),
                        "vulnerability": float(scores.get("vulnerability", 0.0)),
                    }

    return out
