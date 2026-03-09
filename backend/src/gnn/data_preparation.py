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

Node feature vector (dim = 23)
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
  8    Continuous AP score (AP_c)
  9    Bridge Ratio (BR)
 10    QoS aggregate weight (w)
 11    QoS weighted in-degree (w_in)
 12    QoS weighted out-degree (w_out)
 13    Normalised LOC (loc_norm)
 14    Normalised Complexity (complexity_norm)
 15    Instability I = Ce/(Ca+Ce) (instability_code)
 16    Normalised LCOM (lcom_norm)
 17    Code Quality Penalty (CQP)
 18-22 Node-type one-hot (Application, Broker, Topic, Node, Library)

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

# The 13 topological metrics extracted from the structural analysis result.
TOPOLOGICAL_METRIC_KEYS: List[str] = [
    "pagerank",
    "reverse_pagerank",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "clustering_coefficient",
    "ap_c_score",
    "bridge_ratio",
    "qos_weight",
    "qos_weight_in",
    "qos_weight_out",
    "loc_norm",
    "complexity_norm",
    "instability_code",
    "lcom_norm",
    "code_quality_penalty",
]

NODE_FEATURE_DIM = len(TOPOLOGICAL_METRIC_KEYS) + len(NODE_TYPES)  # 23
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


def extract_simulation_dict(simulation_results: list) -> Dict[str, Dict[str, float]]:
    """Normalise SimulationService output to the flat dict expected by this module.

    The ``SimulationService.run_failure_simulation_exhaustive()`` method returns
    a list of ``FailureResult`` dataclass instances.  This function converts
    them into the ``{node_name: {composite, reliability, ...}}`` format.

    Parameters
    ----------
    simulation_results:
        List of ``FailureResult`` objects (or dicts with the same fields).
    """
    out: Dict[str, Dict[str, float]] = {}
    for r in simulation_results:
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
            name = r["target_id"]
            out[name] = {
                "composite": float(r.get("composite_impact", 0.0)),
                "reliability": float(r.get("reliability_impact", 0.0)),
                "maintainability": float(r.get("maintainability_impact", 0.0)),
                "availability": float(r.get("availability_impact", 0.0)),
                "vulnerability": float(r.get("vulnerability_impact", 0.0)),
            }
    return out


def extract_structural_metrics_dict(structural_result) -> Dict[str, Dict[str, float]]:
    """Normalise StructuralAnalyzer output to the flat dict expected by this module.

    Handles both the ``StructuralAnalysisResult`` dataclass returned by the
    ``AnalysisService`` and raw dict representations.
    """
    out: Dict[str, Dict[str, float]] = {}

    def _from_component(comp):
        return {
            "pagerank": float(getattr(comp, "pagerank", 0.0)),
            "reverse_pagerank": float(getattr(comp, "reverse_pagerank", 0.0)),
            "betweenness_centrality": float(getattr(comp, "betweenness_centrality", 0.0)),
            "closeness_centrality": float(getattr(comp, "closeness_centrality", 0.0)),
            "eigenvector_centrality": float(getattr(comp, "eigenvector_centrality", 0.0)),
            "in_degree_centrality": float(getattr(comp, "in_degree_centrality", 0.0)),
            "out_degree_centrality": float(getattr(comp, "out_degree_centrality", 0.0)),
            "clustering_coefficient": float(getattr(comp, "clustering_coefficient", 0.0)),
            "ap_c_score": float(getattr(comp, "ap_c_score", 0.0)),
            "bridge_ratio": float(getattr(comp, "bridge_ratio", 0.0)),
            "qos_weight": float(getattr(comp, "qos_weight", 1.0)),
            "qos_weight_in": float(getattr(comp, "qos_weight_in", 0.0)),
            "qos_weight_out": float(getattr(comp, "qos_weight_out", 0.0)),
            "loc_norm": float(getattr(comp, "loc_norm", 0.0)),
            "complexity_norm": float(getattr(comp, "complexity_norm", 0.0)),
            "instability_code": float(getattr(comp, "instability_code", 0.0)),
            "lcom_norm": float(getattr(comp, "lcom_norm", 0.0)),
            "code_quality_penalty": float(getattr(comp, "code_quality_penalty", 0.0)),
        }

    # Handle StructuralAnalysisResult with .components list
    if hasattr(structural_result, "components"):
        for comp in structural_result.components:
            name = getattr(comp, "component_id", getattr(comp, "name", str(comp)))
            out[name] = _from_component(comp)
    # Handle dict of {name: component_dict}
    elif isinstance(structural_result, dict):
        for name, comp in structural_result.items():
            if isinstance(comp, dict):
                out[name] = {k: float(comp.get(k, 0.0)) for k in TOPOLOGICAL_METRIC_KEYS}
            else:
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
    elif isinstance(quality_result, dict):
        for name, scores in quality_result.items():
            if isinstance(scores, dict):
                out[name] = {
                    k: float(scores.get(k, 0.0))
                    for k in ["overall", "reliability", "maintainability", "availability", "vulnerability"]
                }

    return out
