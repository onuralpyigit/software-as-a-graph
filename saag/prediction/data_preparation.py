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

* **Runtime-enriched features**: Broker, Topic, and Node types carry
  additional infrastructure/runtime features (max_connections, subscriber/
  publisher counts, cpu_cores, memory_gb) beyond the base topology metrics.

* **Multi-task labels**: All five simulation ground-truth dimensions
  (I*(v), IR(v), IM(v), IA(v), IV(v)) are stored as multi-column label
  tensors, enabling multi-task learning or single-task ablations.

Architectural decision: Application.criticality vs Topic.topic_qos_criticality
------------------------------------------------------------------------------
Two fields named ``criticality`` exist in the system, and they are *not* the
same concept.  They must NOT share a feature dimension:

  ``Application.criticality`` (bool)
      Process-level ground truth.  Set by the two-pass topology-aware
      assignment in the generator (structurally central apps → True).
      Represents whether the *service* is mission-critical.
      Used as a classification label for the Application node type.

  ``Topic.criticality`` (str: minimal/low/medium/high/critical)
      QoS-channel urgency, derived from the QoS weight score
      (0.3·Rel + 0.4·Dur + 0.3·Pri) with ≈17% label-noise injection
      so the GNN must use graph context to resolve ambiguous cases.
      Encoded as an ordinal integer 0–4 (``topic_qos_criticality_ord``).
      Appears **only** in the Topic NodeStorage (dim 21) — it is absent
      from Application, Broker, Node, and Library stores.

Consequences:
  * No feature-dimension collision in the heterograph.
  * Attention heads on the PUBLISHES_TO / SUBSCRIBES_TO bipartite
    subgraph can independently learn what frequency and QoS urgency
    contribute to topic-level impact, without entangling the boolean
    service-criticality signal from the Application store.
  * Frequency normalization is always **per-scenario** (z-score of
    log1p(Hz) over the Topic nodes present in the current graph) so
    the normalizer does not leak scenario identity across LOSO folds.

Node feature vector (heterogeneous)
------------------------------------
Each node type HAS ITS OWN feature dimension.

Base Topological Metrics (dim = 18) - all node types:
  0  PageRank (PR)
  1  Reverse PageRank (RPR)
  2  Betweenness Centrality (BT)
  3  Closeness Centrality (CL)
  4  Eigenvector Centrality (EV)
  5  In-Degree normalised (DG_in)
  6  Out-Degree normalised (DG_out)
  7  Clustering Coefficient (CC)
  8  AP_c undirected
  9  Bridge Ratio (BR)
 10  QoS aggregate weight (w)
 11  QoS weighted in-degree (w_in)
 12  QoS weighted out-degree (w_out)
 13  MPCI
 14  Path Complexity (path_complexity)
 15  Fan-Out Criticality (FOC)
 16  AP_c directed
 17  CDI

Code Quality Metrics (dim = 5) - Application and Library ONLY (indices 18-22):
 18  Normalised LOC (loc_norm)
 19  Normalised Complexity (complexity_norm)
 20  Instability I = Ce/(Ca+Ce) (instability_code)
 21  Normalised LCOM (lcom_norm)
 22  Code Quality Penalty (CQP)

Infrastructure/Runtime Extras:
  Broker (index 18):   max_connections_norm
  Node   (18, 19):     cpu_cores_norm, memory_gb_norm
  Topic  (18–21):      subscriber_count_norm, publisher_count_norm,
                       log1p_frequency_norm (per-scenario z-score),
                       topic_qos_criticality_ord (ordinal 0–4)

Total dimensions:
  Application, Library: 23
  Broker:               19
  Topic:                22   ← +2 vs previous (frequency + QoS criticality)
  Node:                 20

Edge feature vector (dim = 16)   [expanded for Middleware 2026 Q-HGL]
------------------------------------------------------------------------
  0   QoS aggregate weight w(e)  (normalised scalar, unchanged)
  1   path_count_norm            (log2-scaled coupling intensity, unchanged)
  2–8 Edge-type one-hot          (PUBLISHES_TO … DEPENDS_ON, 7 dims, unchanged)
  9   reliability_score          (0.0 BEST_EFFORT / 1.0 RELIABLE)
  10  durability_score           (VOLATILE=0.0 / TRANSIENT_LOCAL=0.5 /
                                  TRANSIENT=0.6 / PERSISTENT=1.0)
  11  priority_score             (LOW=0.0 / MEDIUM=0.33 / HIGH=0.66 / URGENT=1.0)
  12  has_deadline               (1.0 if a finite deadline_ns is set, else 0.0)
  13  deadline_ns_log            (log10(1 + deadline_ns / 1e6), 0.0 if absent)
  14  max_blocking_ms_log        (log10(1 + max_blocking_ms), 0.0 if absent)
  15  qos_heterogeneity_flag     (1.0 if topic QoS differs from scenario-level
                                  majority profile, else 0.0)

Dims 9–15 are non-zero only for PUBLISHES_TO / SUBSCRIBES_TO edges, where
QoS profiles are semantically meaningful.  All other edge types receive zeros
for these dimensions, preserving backward numerical compatibility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Base topological metrics applicable to all node types (18 dims)
BASE_METRIC_KEYS: List[str] = [
    "pagerank",              # 0
    "reverse_pagerank",      # 1
    "betweenness_centrality", # 2
    "closeness_centrality",   # 3
    "eigenvector_centrality", # 4
    "in_degree_centrality",   # 5
    "out_degree_centrality",  # 6
    "clustering_coefficient", # 7
    "ap_c_score",            # 8
    "bridge_ratio",          # 9
    "qos_weight",            # 10
    "qos_weight_in",         # 11
    "qos_weight_out",        # 12
    "mpci",                  # 13
    "path_complexity",       # 14
    "fan_out_criticality",   # 15
    "ap_c_directed",         # 16
    "cdi",                   # 17
]

# Code quality metrics for Application and Library (appended to base, indices 18-22)
CQ_METRIC_KEYS: List[str] = [
    "loc_norm",              # 18
    "complexity_norm",       # 19
    "instability_code",      # 20
    "lcom_norm",             # 21
    "code_quality_penalty",  # 22
]

TOPOLOGICAL_METRIC_KEYS = BASE_METRIC_KEYS + CQ_METRIC_KEYS  # 23 dims for App/Lib

# Infrastructure/runtime extra feature keys (appended to BASE_METRIC_KEYS per type)
NODE_INFRA_KEYS: List[str] = ["cpu_cores_norm", "memory_gb_norm"]      # indices 18-19
BROKER_EXTRA_KEYS: List[str] = ["max_connections_norm"]                  # index 18
# Topic-type-only extras (indices 18-21).
# IMPORTANT: these keys must NEVER appear in Application/Broker/Node/Library
# feature dicts — doing so would create a spurious shared embedding dimension
# between semantically distinct node types.  See the module docstring for the
# Application.criticality vs Topic.topic_qos_criticality design decision.
TOPIC_RUNTIME_KEYS: List[str] = [
    "subscriber_count_norm",       # 18 — pub/sub fan-in (topology)
    "publisher_count_norm",        # 19 — pub/sub fan-out (topology)
    "log1p_frequency_norm",        # 20 — per-scenario z-score of log1p(Hz)
    "topic_qos_criticality_ord",   # 21 — QoS urgency ordinal {0=minimal…4=critical}
]

# Ordinal mapping for Topic.criticality labels (5-level QoS urgency scale).
# Separate from Application.criticality which is a bool (process-level ground truth).
TOPIC_CRITICALITY_ORD: Dict[str, float] = {
    "minimal":  0.0,
    "low":      1.0,
    "medium":   2.0,
    "high":     3.0,
    "critical": 4.0,
}

# Per-type feature key mapping used during feature extraction
KEYS_BY_TYPE: Dict[str, List[str]] = {
    "Application": TOPOLOGICAL_METRIC_KEYS,
    "Library":     TOPOLOGICAL_METRIC_KEYS,
    "Broker":      BASE_METRIC_KEYS + BROKER_EXTRA_KEYS,
    "Topic":       BASE_METRIC_KEYS + TOPIC_RUNTIME_KEYS,
    "Node":        BASE_METRIC_KEYS + NODE_INFRA_KEYS,
}

NODE_TYPE_TO_DIM: Dict[str, int] = {
    "Application": 23,
    "Library":     23,
    "Broker":      19,   # +1: max_connections_norm
    "Topic":       22,   # +4: subscriber_count_norm, publisher_count_norm,
                         #      log1p_frequency_norm, topic_qos_criticality_ord
    "Node":        20,   # +2: cpu_cores_norm, memory_gb_norm
}

# Base dims (unchanged): weight + path_count_norm + 7 type one-hot
_EDGE_BASE_DIM = 2 + len(EDGE_TYPES)  # = 9
# QoS decomposition dims added for Middleware 2026 Q-HGL contribution:
#   reliability_score, durability_score, priority_score,
#   has_deadline, deadline_ns_log, max_blocking_ms_log, qos_heterogeneity_flag
_QOS_EXTRA_DIM = 7
EDGE_FEATURE_DIM = _EDGE_BASE_DIM + _QOS_EXTRA_DIM  # = 16

# Label column indices in the (N, 5) label matrix
LABEL_COLS = {
    "composite": 0,
    "reliability": 1,
    "maintainability": 2,
    "availability": 3,
    "vulnerability": 4,
}

# ── QoS edge feature helpers ───────────────────────────────────────────────────

# Canonical score tables (mirrors QoSPolicy in saag/core/models.py)
_RELIABILITY_SCORE: Dict[str, float] = {"BEST_EFFORT": 0.0, "RELIABLE": 1.0}
_DURABILITY_SCORE: Dict[str, float] = {
    "VOLATILE": 0.0, "TRANSIENT_LOCAL": 0.5, "TRANSIENT": 0.6, "PERSISTENT": 1.0
}
_PRIORITY_SCORE: Dict[str, float] = {
    "LOW": 0.0, "MEDIUM": 0.33, "HIGH": 0.66, "URGENT": 1.0
}

# Edge types for which QoS attributes are semantically meaningful
_QOS_EDGE_TYPES = {"PUBLISHES_TO", "SUBSCRIBES_TO", "DEPENDS_ON"}


def _compute_qos_heterogeneity_flags(
    graph: nx.DiGraph,
) -> Dict[Tuple[str, str], float]:
    """Return a per-edge flag indicating QoS heterogeneity relative to the
    scenario-level majority profile.

    For every PUBLISHES_TO / SUBSCRIBES_TO edge we compare its QoS triple
    (reliability, durability, priority) against the modal profile across all
    such edges in the graph.  The flag is 1.0 if the edge deviates, 0.0 otherwise.
    """
    profiles: List[Tuple[str, str, str]] = []
    edge_profiles: Dict[Tuple[str, str], Tuple[str, str, str]] = {}

    for src, dst, attrs in graph.edges(data=True):
        if attrs.get("type", "") not in _QOS_EDGE_TYPES:
            continue
        qp = attrs.get("qos_profile") or {}
        triple = (
            str(qp.get("reliability", "BEST_EFFORT")).upper(),
            str(qp.get("durability", "VOLATILE")).upper(),
            str(qp.get("transport_priority", qp.get("priority", "MEDIUM"))).upper(),
        )
        profiles.append(triple)
        edge_profiles[(src, dst)] = triple

    if not profiles:
        return {}

    # Modal (majority) profile
    from collections import Counter
    modal_triple = Counter(profiles).most_common(1)[0][0]

    flags: Dict[Tuple[str, str], float] = {}
    for (src, dst), triple in edge_profiles.items():
        flags[(src, dst)] = 0.0 if triple == modal_triple else 1.0
    return flags


def _extract_qos_edge_features(
    attrs: Dict[str, Any],
    edge_type: str,
    heterogeneity_flag: float = 0.0,
    qos_enabled: bool = True,
) -> List[float]:
    """Return the 7 QoS-specific edge feature values (dims 9-15).

    Non-zero only for PUBLISHES_TO / SUBSCRIBES_TO edges.
    All values are in [0, 1] or log-scaled to a bounded range.
    """
    if not qos_enabled or edge_type not in _QOS_EDGE_TYPES:
        return [0.0] * _QOS_EXTRA_DIM

    qp: Dict[str, Any] = attrs.get("qos_profile") or {}

    reliability = _RELIABILITY_SCORE.get(
        str(qp.get("reliability", "BEST_EFFORT")).upper(), 0.0
    )
    durability = _DURABILITY_SCORE.get(
        str(qp.get("durability", "VOLATILE")).upper(), 0.0
    )
    priority_key = str(
        qp.get("transport_priority", qp.get("priority", "MEDIUM"))
    ).upper()
    priority = _PRIORITY_SCORE.get(priority_key, 0.33)

    # Deadline handling: raw nanoseconds → log10-scaled ms, clamped
    deadline_ns = float(qp.get("deadline_ns", qp.get("deadline", 0)) or 0)
    has_deadline = 1.0 if deadline_ns > 0 else 0.0
    deadline_log = math.log10(1.0 + deadline_ns / 1e6) if deadline_ns > 0 else 0.0
    deadline_log = min(deadline_log, 10.0) / 10.0  # normalise to [0,1]

    # Max blocking time: raw ms → log10-scaled
    blocking_ms = float(qp.get("max_blocking_ms", qp.get("max_blocking", 0)) or 0)
    blocking_log = math.log10(1.0 + blocking_ms) if blocking_ms > 0 else 0.0
    blocking_log = min(blocking_log, 5.0) / 5.0   # normalise to [0,1]

    return [
        reliability,       # dim 9
        durability,        # dim 10
        priority,          # dim 11
        has_deadline,      # dim 12
        deadline_log,      # dim 13
        blocking_log,      # dim 14
        heterogeneity_flag,  # dim 15
    ]


# ── Infrastructure feature helpers ─────────────────────────────────────────────

def _normalize_infra_features(
    graph: nx.DiGraph,
    structural_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    qos_enabled: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute normalized infrastructure/runtime features per node.

    Derives:
    - Node: cpu_cores_norm, memory_gb_norm  (from graph attrs or structural_metrics)
    - Broker: max_connections_norm          (from graph attrs or structural_metrics)
    - Topic: subscriber_count_norm, publisher_count_norm  (from graph edge topology)
    - Topic: log1p_frequency_norm (per-scenario z-score of log1p(Hz))
    - Topic: topic_qos_criticality_ord (ordinal 0–4 of the 5-level QoS label)

    All values are normalized to [0, 1] via per-graph max, EXCEPT
    log1p_frequency_norm which uses z-score normalization limited to the
    Topic nodes present in *this* graph (per-scenario, not global).

    Per-scenario frequency normalization rationale
    -----------------------------------------------
    Frequency spans ~5 orders of magnitude across domains (0.001 Hz healthcare
    to 10 000 Hz HFT).  Normalizing globally (across all scenarios in the
    training corpus) would encode the scenario's domain identity in the
    standardised values, leaking label information into features across LOSO
    folds.  Normalizing per-scenario ensures each graph's frequency distribution
    is centred at 0 regardless of domain, preserving relative ordering within
    a scenario while eliminating cross-scenario scale differences.
    """
    node_cpu: Dict[str, float] = {}
    node_mem: Dict[str, float] = {}
    broker_conn: Dict[str, float] = {}
    topic_subs: Dict[str, float] = {}
    topic_pubs: Dict[str, float] = {}
    topic_freq_raw: Dict[str, float] = {}    # raw Hz values for all Topic nodes
    topic_crit_ord: Dict[str, float] = {}   # ordinal criticality values

    for n, attrs in graph.nodes(data=True):
        nt = attrs.get("type") or attrs.get("component_type") or "Application"
        sm = (structural_metrics or {}).get(n, {})
        if nt == "Node":
            cpu = float(attrs.get("cpu_cores", sm.get("cpu_cores", 0)) or 0)
            mem = float(attrs.get("memory_gb", sm.get("memory_gb", 0)) or 0)
            node_cpu[n] = cpu
            node_mem[n] = mem
        elif nt == "Broker":
            conn = float(attrs.get("max_connections", sm.get("max_connections", 0)) or 0)
            broker_conn[n] = conn
        elif nt == "Topic":
            # Collect raw frequency for per-scenario z-score (computed after this loop).
            freq_raw = float(
                attrs.get("frequency", attrs.get("topic_frequency", 0.0)) or 0.0
            )
            topic_freq_raw[n] = freq_raw
            # Criticality ordinal (Topic.criticality, NOT Application.criticality).
            crit_str = str(
                attrs.get("criticality", attrs.get("topic_criticality", "minimal"))
            ).lower()
            topic_crit_ord[n] = TOPIC_CRITICALITY_ORD.get(crit_str, 0.0) if qos_enabled else 0.0

    # PUBLISHES_TO: Application → Topic; SUBSCRIBES_TO: Application → Topic.
    # Both counts are from the Topic's perspective (how many publishers/subscribers it has).
    for src, dst, attrs in graph.edges(data=True):
        etype = attrs.get("type", "")
        if etype == "SUBSCRIBES_TO":
            topic_subs[dst] = topic_subs.get(dst, 0.0) + 1.0
        elif etype == "PUBLISHES_TO":
            topic_pubs[dst] = topic_pubs.get(dst, 0.0) + 1.0

    max_cpu = max(node_cpu.values(), default=1.0) or 1.0
    max_mem = max(node_mem.values(), default=1.0) or 1.0
    max_conn = max(broker_conn.values(), default=1.0) or 1.0
    max_subs = max(topic_subs.values(), default=1.0) or 1.0
    max_pubs = max(topic_pubs.values(), default=1.0) or 1.0

    # --- Per-scenario log1p z-score for topic frequency -------------------
    # Apply log1p first (compresses the 0.001–10 000 Hz span to ~0–9.2),
    # then z-score within this graph so the distribution is centred at μ=0
    # regardless of domain.  Clamp to [-3, 3] to bound outliers.
    log1p_vals = {n: math.log1p(hz) for n, hz in topic_freq_raw.items()}
    if log1p_vals:
        mu = sum(log1p_vals.values()) / len(log1p_vals)
        variance = sum((v - mu) ** 2 for v in log1p_vals.values()) / len(log1p_vals)
        sigma = max(math.sqrt(variance), 1e-6)  # guard against zero-variance graphs
        log1p_freq_norm: Dict[str, float] = {
            n: max(-3.0, min(3.0, (v - mu) / sigma))
            for n, v in log1p_vals.items()
        }
    else:
        log1p_freq_norm = {}

    # Normalise topic_qos_criticality_ord to [0, 1] (max value is 4.0).
    max_crit_ord = 4.0

    # Dynamic masking (G4 covariate shift mitigation):
    # If all topic criticalities in this graph are identical (zero variance),
    # it means we cannot justify a real ground-truth distribution for this scenario
    # and the field is flat (typically all minimal). We mask the field to a uniform 0.0
    # to prevent inductive covariate shift when training/testing across scenarios.
    crit_vals = list(topic_crit_ord.values())
    if len(crit_vals) > 0 and len(set(crit_vals)) <= 1:
        topic_crit_ord = {n: 0.0 for n in topic_crit_ord}

    infra: Dict[str, Dict[str, float]] = {}
    for n in node_cpu:
        infra[n] = {
            "cpu_cores_norm": node_cpu[n] / max_cpu,
            "memory_gb_norm": node_mem.get(n, 0.0) / max_mem,
        }
    for n, v in broker_conn.items():
        infra[n] = {"max_connections_norm": v / max_conn}
    all_topic_nodes = set(topic_subs) | set(topic_pubs) | set(topic_freq_raw)
    for n in all_topic_nodes:
        infra[n] = {
            "subscriber_count_norm": topic_subs.get(n, 0.0) / max_subs,
            "publisher_count_norm": topic_pubs.get(n, 0.0) / max_pubs,
            "log1p_frequency_norm": log1p_freq_norm.get(n, 0.0),
            "topic_qos_criticality_ord": topic_crit_ord.get(n, 0.0) / max_crit_ord,
        }
    return infra


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class GraphConversionResult:
    """Output of :func:`networkx_to_hetero_data`."""

    hetero_data: object

    node_id_map: Dict[str, List[str]] = field(default_factory=dict)

    node_name_to_idx: Dict[str, Tuple[str, int]] = field(default_factory=dict)

    edge_name_map: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = field(
        default_factory=dict
    )

    num_labelled_nodes: int = 0

    present_node_types: List[str] = field(default_factory=list)


def networkx_to_hetero_data(
    graph: nx.DiGraph,
    structural_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    simulation_results: Optional[Dict[str, Dict[str, float]]] = None,
    rmav_scores: Optional[Dict[str, Dict[str, float]]] = None,
    qos_enabled: bool = True,
) -> GraphConversionResult:
    """Convert a NetworkX DiGraph to a PyG HeteroData object.

    Parameters
    ----------
    graph:
        The full structural NetworkX graph from Step 1.
        Nodes must carry a ``type`` attribute (one of NODE_TYPES).
        Edges must carry a ``type`` attribute (one of EDGE_TYPES) and
        optionally ``weight`` and ``path_count`` attributes.
    structural_metrics:
        ``{node_name: {metric_key: value}}`` from ``StructuralAnalyzer``.
    simulation_results:
        ``{node_name: {composite, reliability, maintainability,
                        availability, vulnerability}}`` training labels.
    rmav_scores:
        ``{node_name: {overall, reliability, maintainability,
                        availability, vulnerability}}`` for ensemble blending.

    Returns
    -------
    GraphConversionResult
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
        node_type = attrs.get("type") or attrs.get("component_type")
        if node_type is None:
            logger.warning(
                "Node '%s' has no 'type' or 'component_type'; defaulting to 'Application'.",
                node,
            )
            node_type = "Application"
        if node_type not in type_to_nodes:
            logger.warning(
                "Unknown node type '%s' for node '%s'; treating as Application.",
                node_type, node,
            )
            node_type = "Application"
        type_to_nodes[node_type].append(node)

    for node_type, nodes in type_to_nodes.items():
        if not nodes:
            continue
        result.node_id_map[node_type] = nodes
        result.present_node_types.append(node_type)
        for local_idx, name in enumerate(nodes):
            result.node_name_to_idx[name] = (node_type, local_idx)

    # Pre-compute infrastructure/runtime features (Topic counts, Node/Broker infra)
    infra_features = _normalize_infra_features(graph, structural_metrics, qos_enabled=qos_enabled)

    # ── 2. Build node feature tensors per type ────────────────────────────────
    for node_type in result.present_node_types:
        nodes = result.node_id_map[node_type]
        n = len(nodes)
        dim = NODE_TYPE_TO_DIM.get(node_type, len(BASE_METRIC_KEYS))
        feat_matrix = np.zeros((n, dim), dtype=np.float32)

        keys_to_use = KEYS_BY_TYPE.get(node_type, BASE_METRIC_KEYS)

        import os
        decouple_features = os.environ.get("DECOUPLE_FEATURES") in ("1", "true", "True")

        for local_idx, name in enumerate(nodes):
            base_source = (structural_metrics or {}).get(name, {})
            infra_source = infra_features.get(name, {})
            for col, key in enumerate(keys_to_use):
                # Base metrics come from structural_metrics; infra keys from infra_source
                val = base_source.get(key, infra_source.get(key, 0.0))
                if not qos_enabled and key in ("qos_weight", "qos_weight_in", "qos_weight_out"):
                    val = 0.0
                if decouple_features and key in (
                    "pagerank", "reverse_pagerank", "betweenness_centrality",
                    "closeness_centrality", "eigenvector_centrality",
                    "in_degree_centrality", "out_degree_centrality",
                    "clustering_coefficient", "ap_c_score", "ap_c_directed",
                    "cdi", "mpci", "path_complexity", "fan_out_criticality", "bridge_ratio"
                ):
                    val = 0.0
                feat_matrix[local_idx, col] = float(val)

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

        # ── RMAV scores (for ensemble / consistency regularization) ───────────
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
    try:
        bridges = set(nx.bridges(graph.to_undirected()))
    except Exception:
        bridges = set()

    # Pre-compute per-edge QoS heterogeneity flags (single graph pass)
    _hetero_flags: Dict[Tuple[str, str], float] = _compute_qos_heterogeneity_flags(graph)

    rel_edges: Dict[Tuple[str, str, str], Tuple[List[int], List[int], List[List[float]]]] = {}

    for src, dst, attrs in graph.edges(data=True):
        if src not in result.node_name_to_idx or dst not in result.node_name_to_idx:
            continue

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

        # ── Edge features: 16-d (base 9 + QoS decomposition 7) ───────────────
        weight = float(attrs.get("weight", 1.0))
        path_count_raw = float(attrs.get("path_count", 1) or 1)
        path_count_norm = math.log2(1.0 + path_count_raw) / math.log2(17.0)
        type_onehot = [0.0] * len(EDGE_TYPES)
        if edge_type in EDGE_TYPE_INDEX:
            type_onehot[EDGE_TYPE_INDEX[edge_type]] = 1.0

        # QoS decomposition dims 9-15 (non-zero only for pub/sub edges)
        hetero_flag = _hetero_flags.get((src, dst), 0.0)
        qos_dims = _extract_qos_edge_features(attrs, edge_type, hetero_flag, qos_enabled=qos_enabled)

        rel_edges[rel_key][2].append([weight, path_count_norm] + type_onehot + qos_dims)

    for (src_type, edge_type, dst_type), (srcs, dsts, feats) in rel_edges.items():
        rel = (src_type, edge_type, dst_type)
        data[rel].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        data[rel].edge_attr = torch.tensor(feats, dtype=torch.float32)

        # Edge labels: grounded in structural bridge property (Issue G3 workaround)
        if simulation_results:
            src_nodes = result.node_id_map[src_type]
            dst_nodes = result.node_id_map[dst_type]
            edge_labels = np.zeros((len(srcs), 5), dtype=np.float32)
            for i, (s_idx, d_idx) in enumerate(zip(srcs, dsts)):
                s_name = src_nodes[s_idx]
                d_name = dst_nodes[d_idx]
                is_bridge = (s_name, d_name) in bridges or (d_name, s_name) in bridges
                bridge_multiplier = 1.0 if is_bridge else 0.1
                s_sim = simulation_results.get(s_name, {})
                for col, key in enumerate(
                    ["composite", "reliability", "maintainability", "availability", "vulnerability"]
                ):
                    edge_labels[i, col] = float(s_sim.get(key, 0.0)) * bridge_multiplier
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

    Uses **stratified splitting** when ground-truth labels exist: labelled nodes
    (|y_composite| > 1e-6) are split proportionally across train/val/test first,
    ensuring each split contains labelled nodes for meaningful ρ evaluation.
    Unlabelled nodes are then distributed to fill the remaining capacity.
    Falls back to uniform random split when no ``y`` attribute is present.
    """
    rng = np.random.default_rng(seed)

    for store in hetero_data.node_stores:
        n = store.num_nodes
        if n == 0:
            continue

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)

        # ── Stratified split when labels available ──────────────────────────
        if hasattr(store, "y") and store.y.numel() > 0:
            y_comp = store.y[:, 0].detach().numpy()
            labelled_idx   = np.where(np.abs(y_comp) > 1e-6)[0]
            unlabelled_idx = np.where(np.abs(y_comp) <= 1e-6)[0]

            if len(labelled_idx) >= 3:
                lab_shuffled = rng.permutation(labelled_idx)
                n_lab_train = max(1, int(len(lab_shuffled) * train_ratio))
                n_lab_val   = max(1, int(len(lab_shuffled) * val_ratio))
                lab_train = lab_shuffled[:n_lab_train]
                lab_val   = lab_shuffled[n_lab_train: n_lab_train + n_lab_val]
                lab_test  = lab_shuffled[n_lab_train + n_lab_val:]

                unlab_shuffled = rng.permutation(unlabelled_idx)
                n_total_train  = max(1, int(n * train_ratio))
                n_total_val    = max(1, int(n * val_ratio))
                n_need_train   = max(0, n_total_train - len(lab_train))
                n_need_val     = max(0, n_total_val   - len(lab_val))

                unlab_train = unlab_shuffled[:n_need_train]
                unlab_val   = unlab_shuffled[n_need_train: n_need_train + n_need_val]
                unlab_test  = unlab_shuffled[n_need_train + n_need_val:]

                all_train = np.concatenate([lab_train, unlab_train])
                all_val   = np.concatenate([lab_val,   unlab_val])
                all_test  = np.concatenate([lab_test,  unlab_test])

                train_mask[torch.from_numpy(all_train.astype(np.int64))] = True
                val_mask[torch.from_numpy(all_val.astype(np.int64))]     = True
                test_mask[torch.from_numpy(all_test.astype(np.int64))]   = True

                store.train_mask = train_mask
                store.val_mask   = val_mask
                store.test_mask  = test_mask
                continue

        # ── Fallback: uniform random split ──────────────────────────────────
        indices = rng.permutation(n)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        train_mask[torch.from_numpy(indices[:n_train].astype(np.int64))]               = True
        val_mask[torch.from_numpy(indices[n_train: n_train + n_val].astype(np.int64))] = True
        test_mask[torch.from_numpy(indices[n_train + n_val:].astype(np.int64))]        = True
        store.train_mask = train_mask
        store.val_mask   = val_mask
        store.test_mask  = test_mask



def normalize_labels_robust(hetero_data) -> None:
    """In-place IQR normalization of .y label tensors, preserving zeros.

    Computes global median and IQR over non-zero labelled nodes, then maps non-zero
    labels through (x - median) / IQR, clamps to [-3, 3], and applies sigmoid
    to keep values in (0, 1) for use with the sigmoid output heads. Zeros remain 0.0.
    """
    all_labels = [
        hetero_data[nt].y
        for nt in hetero_data.node_types
        if hasattr(hetero_data[nt], "y") and hetero_data[nt].y.numel() > 0
    ]
    if not all_labels:
        return
    concat = torch.cat(all_labels, dim=0)   # (N_total, 5)
    
    # Mask out original zeros to preserve zero structure
    non_zero_mask = concat[:, 0].abs() > 1e-6
    if not non_zero_mask.any():
        return
        
    non_zero_concat = concat[non_zero_mask]
    q25 = torch.quantile(non_zero_concat, 0.25, dim=0)
    q75 = torch.quantile(non_zero_concat, 0.75, dim=0)
    iqr = (q75 - q25).clamp(min=1e-6)
    median = torch.median(non_zero_concat, dim=0).values
    
    for nt in hetero_data.node_types:
        store = hetero_data[nt]
        if hasattr(store, "y") and store.y.numel() > 0:
            nz = store.y[:, 0].abs() > 1e-6
            if nz.any():
                scaled = torch.sigmoid(((store.y[nz] - median) / iqr).clamp(-3.0, 3.0))
                new_y = store.y.clone()
                new_y[nz] = scaled
                store.y = new_y


# ── Extraction utilities ───────────────────────────────────────────────────────

def extract_simulation_dict(simulation_results: Union[list, dict]) -> Dict[str, Dict[str, float]]:
    """Normalise Simulation output to common flat dict format."""
    out: Dict[str, Dict[str, float]] = {}

    if isinstance(simulation_results, dict) and "component_criticality" in simulation_results:
        for c in simulation_results["component_criticality"]:
            name = c.get("id")
            out[name] = {
                "composite": float(c.get("combined_impact", 0.0)),
                "reliability": float(c.get("failure_impact", 0.0)),
                "maintainability": 0.0,
                "availability": float(c.get("failure_impact", 0.0)),
                "vulnerability": 0.0,
            }
        return out

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
            if not name:
                continue
            impact = r.get("impact", r)
            out[name] = {
                "composite": float(impact.get("composite_impact", 0.0)),
                "reliability": float(impact.get("reliability_impact", 0.0)),
                "maintainability": float(impact.get("maintainability_impact", 0.0)),
                "availability": float(impact.get("availability_impact", 0.0)),
                "vulnerability": float(impact.get("vulnerability_impact", 0.0)),
            }
    return out


def extract_structural_metrics_dict(structural_result) -> Dict[str, Dict[str, float]]:
    """Normalise StructuralAnalyzer output to the flat dict expected by this module."""
    if isinstance(structural_result, dict) and "structural_analysis" in structural_result:
        structural_result = structural_result["structural_analysis"]
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
            "path_complexity": float(_get(comp, "path_complexity")),
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
            # Infrastructure attributes (used by _normalize_infra_features)
            "cpu_cores": float(_get(comp, "cpu_cores", 0)),
            "memory_gb": float(_get(comp, "memory_gb", 0)),
            "max_connections": float(_get(comp, "max_connections", 0)),
        }

    if hasattr(structural_result, "components"):
        components = structural_result.components
        if hasattr(components, "values"):
            components = components.values()
        for comp in components:
            name = getattr(comp, "component_id", getattr(comp, "name", str(getattr(comp, "id", comp))))
            out[name] = _from_component(comp)
    elif isinstance(structural_result, dict):
        components = structural_result.get("components", structural_result)
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
