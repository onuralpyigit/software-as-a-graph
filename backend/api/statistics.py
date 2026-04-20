"""
Statistics calculator module.

This module provides functions to calculate descriptive statistics
(mean, median, std, boxplot) for various system components.

Can be used standalone with dataset.json to compute all statistics
without needing the reporter module:

    import json
    from api.statistics import extract_cross_cutting_data, compute_all_extras_statistics

    with open("dataset.json") as f:
        data = json.load(f)

    cc = extract_cross_cutting_data(data)
    stats = compute_all_extras_statistics(cc)
"""

import math
import statistics as stats
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a metric."""
    count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    q1: float
    q3: float
    iqr: float
    outlier_upper: float  # Upper IQR fence
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
            "q1": round(self.q1, 4),
            "q3": round(self.q3, 4),
            "iqr": round(self.iqr, 4),
            "outlier_upper": round(self.outlier_upper, 4),
        }


@dataclass
class CategoricalStats:
    """Statistics for categorical (distribution) metrics."""
    total_count: int        # Total number of items
    category_count: int     # Number of distinct categories
    mode: str               # Most frequent category
    mode_count: int         # Count of most frequent category
    mode_percentage: float  # Percentage of most frequent category
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "category_count": self.category_count,
            "mode": self.mode,
            "mode_count": self.mode_count,
            "mode_percentage": round(self.mode_percentage, 2),
        }


def calculate_categorical_stats(category_counts: Dict[str, int]) -> CategoricalStats:
    """Calculate statistics for categorical variables."""
    if not category_counts:
        return CategoricalStats(
            total_count=0, category_count=0, mode="N/A", 
            mode_count=0, mode_percentage=0.0
        )
    
    total = sum(category_counts.values())
    category_count = len(category_counts)
    
    # Find mode (most frequent category)
    mode = max(category_counts, key=category_counts.get)
    mode_count = category_counts[mode]
    mode_pct = (mode_count / total * 100) if total > 0 else 0.0
    
    return CategoricalStats(
        total_count=total,
        category_count=category_count,
        mode=mode,
        mode_count=mode_count,
        mode_percentage=mode_pct
    )


def calculate_descriptive_stats(values: List[float]) -> DescriptiveStats:
    """Calculate descriptive statistics for a list of values."""
    if not values:
        return DescriptiveStats(
            count=0, mean=0, median=0, std=0,
            min_val=0, max_val=0, q1=0, q3=0, iqr=0,
            outlier_upper=0
        )
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    mean_val = stats.mean(sorted_vals)
    median_val = stats.median(sorted_vals)
    std_val = stats.stdev(sorted_vals) if n > 1 else 0.0
    
    # Quartiles
    if n < 4:
        q1 = median_val
        q3 = median_val
    else:
        q1 = _percentile(sorted_vals, 25)
        q3 = _percentile(sorted_vals, 75)
    
    iqr = q3 - q1
    outlier_upper = q3 + 1.5 * iqr
    
    return DescriptiveStats(
        count=n,
        mean=mean_val,
        median=median_val,
        std=std_val,
        min_val=sorted_vals[0],
        max_val=sorted_vals[-1],
        q1=q1,
        q3=q3,
        iqr=iqr,
        outlier_upper=outlier_upper
    )


def _percentile(data: List[float], p: float) -> float:
    """Calculate p-th percentile using linear interpolation."""
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


# ===================== OUTLIER DETECTION =====================


def find_1d_outliers_iqr(values: List[float]) -> Tuple[float, float, float]:
    """Compute IQR fences for 1D outlier detection.

    Returns:
        (lower_fence, upper_fence, iqr)
    """
    if len(values) < 4:
        return float('-inf'), float('inf'), 0.0
    q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, iqr


def calculate_outliers(
    ranked_list: List[Dict],
    outlier_stats: "DescriptiveStats",
    *,
    is_categorical: bool = False,
) -> List[Tuple[str, float]]:
    """Calculate outliers using IQR method from a ranked list and stats.

    Args:
        ranked_list: List of dicts with 'name', 'value', optional 'version'.
        outlier_stats: DescriptiveStats with outlier_lower/outlier_upper.
        is_categorical: If True, returns empty (no outliers for categorical).

    Returns:
        List of (display_name, value) tuples.
    """
    if is_categorical or outlier_stats.count == 0:
        return []

    upper_bound = outlier_stats.outlier_upper

    outliers: List[Tuple[str, float]] = []
    for item in ranked_list:
        value = item.get("value", 0)
        name = item.get("name", item.get("id", "?"))
        version = item.get("version", "")
        display_name = f"{name} ({version})" if version and version != "NOT_FOUND" else str(name)
        if value > upper_bound:
            outliers.append((display_name, value))
    return outliers


# ===================== DATA EXTRACTION =====================


def extract_cross_cutting_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and pre-process data needed for cross-cutting charts from raw JSON.

    This is the main data extraction function that preprocesses the raw
    aggregated JSON into lookup dictionaries usable by both chart and
    statistics functions.
    """
    nodes = raw_data.get("nodes", [])
    apps = raw_data.get("applications", [])
    topics = raw_data.get("topics", [])
    relationships = raw_data.get("relationships", {})

    runs_on = relationships.get("runs_on", [])
    publishes_to = relationships.get("publishes_to", [])
    subscribes_to = relationships.get("subscribes_to", [])

    node_names = {n["id"]: n.get("name", n["id"]) for n in nodes}
    app_names = {a["id"]: a.get("name", a["id"]) for a in apps}

    app_node: Dict[str, str] = {}
    for rel in runs_on:
        app_id = rel.get("from")
        node_id = rel.get("to")
        if app_id and node_id:
            app_node[app_id] = node_id

    topic_names = {t["id"]: t.get("name", t["id"]) for t in topics}
    topic_sizes: Dict[str, float] = {}
    topic_qos: Dict[str, Dict[str, str]] = {}
    for t in topics:
        size = t.get("size", 0)
        topic_sizes[t["id"]] = max(0, size) if size is not None else 0
        topic_qos[t["id"]] = t.get("qos", {})

    app_pub_count: Dict[str, int] = {a["id"]: 0 for a in apps}
    app_sub_count: Dict[str, int] = {a["id"]: 0 for a in apps}
    for rel in publishes_to:
        aid = rel.get("from")
        if aid in app_pub_count:
            app_pub_count[aid] += 1
    for rel in subscribes_to:
        aid = rel.get("from")
        if aid in app_sub_count:
            app_sub_count[aid] += 1

    topic_pub_count: Dict[str, int] = {t["id"]: 0 for t in topics}
    topic_sub_count: Dict[str, int] = {t["id"]: 0 for t in topics}
    for rel in publishes_to:
        tid = rel.get("to")
        if tid in topic_pub_count:
            topic_pub_count[tid] += 1
    for rel in subscribes_to:
        tid = rel.get("to")
        if tid in topic_sub_count:
            topic_sub_count[tid] += 1

    return {
        "nodes": nodes,
        "apps": apps,
        "topics": topics,
        "runs_on": runs_on,
        "publishes_to": publishes_to,
        "subscribes_to": subscribes_to,
        "node_names": node_names,
        "app_names": app_names,
        "app_node": app_node,
        "topic_names": topic_names,
        "topic_sizes": topic_sizes,
        "topic_qos": topic_qos,
        "app_pub_count": app_pub_count,
        "app_sub_count": app_sub_count,
        "topic_pub_count": topic_pub_count,
        "topic_sub_count": topic_sub_count,
        "app_criticality": {a["id"]: a.get("criticality", False) for a in apps},
        "app_role": {a["id"]: a.get("role", "NOT_FOUND") for a in apps},
        "app_domain": {
            a["id"]: (a.get("system_hierarchy") or {}).get("domain_name", "NOT_FOUND")
            for a in apps
        },
        "libs": raw_data.get("libraries", []),
        "lib_names": {
            lib["id"]: lib.get("name", lib["id"])
            for lib in raw_data.get("libraries", [])
        },
        "uses": relationships.get("uses", []),
    }


# ===================== PER-CHART STATISTICS =====================


def compute_topic_bandwidth_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute bandwidth statistics for Topic Size vs Subscriber chart."""
    sizes, subs, pubs, labels, ids = [], [], [], [], []
    for tid in cc["topic_sizes"]:
        sizes.append(cc["topic_sizes"][tid])
        subs.append(cc["topic_sub_count"].get(tid, 0))
        pubs.append(cc["topic_pub_count"].get(tid, 0))
        labels.append(cc["topic_names"].get(tid, tid))
        ids.append(tid)

    bandwidth_sub = [s * sc for s, sc in zip(sizes, subs)]
    bandwidth_pub = [s * pc for s, pc in zip(sizes, pubs)]
    bandwidth_pubsub = [s * (pc + sc) for s, pc, sc in zip(sizes, pubs, subs)]
    # Keep legacy "bandwidth" as sub-based for backward compatibility
    bandwidth = bandwidth_sub

    nonzero_bw = [b for b in bandwidth_sub if b > 0]
    outlier_indices: List[int] = []
    iqr_upper = 0.0
    iqr_val = 0.0
    if len(nonzero_bw) >= 4:
        _, iqr_upper, iqr_val = find_1d_outliers_iqr(nonzero_bw)
        outlier_indices = [
            i for i in range(len(bandwidth_sub))
            if bandwidth_sub[i] > iqr_upper and bandwidth_sub[i] > 0
        ]
        outlier_indices.sort(key=lambda i: bandwidth_sub[i], reverse=True)

    summary: Dict[str, Any] = {}
    if sizes:
        size_arr = np.array(sizes, dtype=float)
        sub_arr = np.array(subs, dtype=float)
        pub_arr = np.array(pubs, dtype=float)
        summary = {
            "total_topics": len(sizes),
            "size_mean": float(np.mean(size_arr)),
            "size_median": float(np.median(size_arr)),
            "size_max": float(np.max(size_arr)),
            "sub_mean": float(np.mean(sub_arr)),
            "sub_median": float(np.median(sub_arr)),
            "sub_max": int(np.max(sub_arr)),
            "pub_mean": float(np.mean(pub_arr)),
            "pub_median": float(np.median(pub_arr)),
            "pub_max": int(np.max(pub_arr)),
            "zero_sub_count": sum(1 for s in subs if s == 0),
            "bw_mean": float(np.mean(nonzero_bw)) if nonzero_bw else 0,
            "bw_median": float(np.median(nonzero_bw)) if nonzero_bw else 0,
            "outlier_count": len(outlier_indices),
        }

    return {
        "sizes": sizes, "subs": subs, "pubs": pubs, "labels": labels, "ids": ids,
        "bandwidth": bandwidth, "bandwidth_sub": bandwidth_sub,
        "bandwidth_pub": bandwidth_pub, "bandwidth_pubsub": bandwidth_pubsub,
        "nonzero_bw": nonzero_bw,
        "outlier_indices": outlier_indices,
        "iqr_upper": iqr_upper, "iqr": iqr_val,
        "summary": summary,
    }


def compute_qos_risk_stats(
    cc: Dict[str, Any],
    risk_weight_fn: Callable[[str, str], float],
    w2name: Optional[Dict[str, Dict[float, str]]] = None,
) -> Dict[str, Any]:
    """Compute QoS risk statistics for QoS scatter chart.

    Args:
        cc: Cross-cutting data from extract_cross_cutting_data.
        risk_weight_fn: Callable(dimension, value) -> float weight.
        w2name: Optional weight-to-name maps per dimension for display.
    """
    topic_data: List[Dict[str, Any]] = []
    for tid, qos in cc["topic_qos"].items():
        size = cc["topic_sizes"].get(tid, 0)
        tname = cc["topic_names"].get(tid, tid)
        dur_w = risk_weight_fn("durability", str(qos.get("durability", "NOT_FOUND")))
        rel_w = risk_weight_fn("reliability", str(qos.get("reliability", "NOT_FOUND")))
        tp_w = risk_weight_fn("transport_priority", str(qos.get("transport_priority", "NOT_FOUND")))
        risk = (dur_w * rel_w * tp_w) * math.log2(size + 1)

        dur_name = w2name["durability"].get(dur_w, str(qos.get("durability", "-"))) if w2name else str(dur_w)
        rel_name = w2name["reliability"].get(rel_w, str(qos.get("reliability", "-"))) if w2name else str(rel_w)
        tp_name = w2name["transport_priority"].get(tp_w, str(qos.get("transport_priority", "-"))) if w2name else str(tp_w)

        topic_data.append({
            "name": tname, "size": size, "risk": risk,
            "durability": dur_w, "reliability": rel_w, "transport_priority": tp_w,
            "dur_name": dur_name, "rel_name": rel_name, "tp_name": tp_name,
        })

    topic_data.sort(key=lambda d: d["risk"], reverse=True)
    risk_values = [td["risk"] for td in topic_data if td["risk"] > 0]
    top_outliers: List[Dict[str, Any]] = []
    risk_upper = 0.0
    risk_iqr = 0.0
    if len(risk_values) >= 4:
        _, risk_upper, risk_iqr = find_1d_outliers_iqr(risk_values)
        top_outliers = [td for td in topic_data if td["risk"] > risk_upper]
    else:
        top_outliers = topic_data[:10]

    summary: Dict[str, Any] = {
        "total_topics": len(topic_data),
        "outlier_count": len(top_outliers),
    }
    if risk_values:
        risk_arr = np.array(risk_values, dtype=float)
        summary.update({
            "risk_mean": float(np.mean(risk_arr)),
            "risk_median": float(np.median(risk_arr)),
            "risk_std": float(np.std(risk_arr)),
        })

    qos_distribution: Dict[str, Dict[float, int]] = {}
    for dim_key in ("durability", "reliability", "transport_priority"):
        w_counts: Dict[float, int] = {}
        for td in topic_data:
            w = td[dim_key]
            w_counts[w] = w_counts.get(w, 0) + 1
        qos_distribution[dim_key] = w_counts
    summary["qos_distribution"] = qos_distribution

    return {
        "topic_data": topic_data,
        "top_outliers": top_outliers,
        "risk_upper": risk_upper,
        "risk_iqr": risk_iqr,
        "summary": summary,
    }


def compute_app_balance_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pub/sub balance statistics for App Balance chart."""
    pubs, subs, labels, ids = [], [], [], []
    for aid in cc["app_pub_count"]:
        pubs.append(cc["app_pub_count"][aid])
        subs.append(cc["app_sub_count"].get(aid, 0))
        labels.append(cc["app_names"].get(aid, aid))
        ids.append(aid)

    io_load = [p + s for p, s in zip(pubs, subs)]
    nonzero_io = [v for v in io_load if v > 0]
    outlier_indices: List[int] = []
    io_upper = 0.0
    io_iqr = 0.0
    if len(nonzero_io) >= 4:
        _, io_upper, io_iqr = find_1d_outliers_iqr(nonzero_io)
        outlier_indices = [
            i for i in range(len(io_load))
            if io_load[i] > io_upper and io_load[i] > 0
        ]
        outlier_indices.sort(key=lambda i: io_load[i], reverse=True)

    summary: Dict[str, Any] = {}
    if pubs:
        pub_arr = np.array(pubs, dtype=float)
        sub_arr = np.array(subs, dtype=float)
        mean_p = float(np.mean(pub_arr))
        mean_s = float(np.mean(sub_arr))
        summary = {
            "total_apps": len(pubs),
            "pub_mean": mean_p,
            "pub_median": float(np.median(pub_arr)),
            "pub_max": int(np.max(pub_arr)),
            "sub_mean": mean_s,
            "sub_median": float(np.median(sub_arr)),
            "sub_max": int(np.max(sub_arr)),
            "q_high_io": sum(1 for p, s in zip(pubs, subs) if p > mean_p and s > mean_s),
            "q_consumer": sum(1 for p, s in zip(pubs, subs) if p <= mean_p and s > mean_s),
            "q_producer": sum(1 for p, s in zip(pubs, subs) if p > mean_p and s <= mean_s),
            "q_low": sum(1 for p, s in zip(pubs, subs) if p <= mean_p and s <= mean_s),
            "zero_activity": sum(1 for p, s in zip(pubs, subs) if p == 0 and s == 0),
            "outlier_count": len(outlier_indices),
        }

    return {
        "pubs": pubs, "subs": subs, "labels": labels, "ids": ids,
        "io_load": io_load,
        "outlier_indices": outlier_indices,
        "io_upper": io_upper, "io_iqr": io_iqr,
        "summary": summary,
    }


def compute_topic_fanout_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute fanout statistics for Topic Fanout chart."""
    pubs, subs, labels, ids = [], [], [], []
    for tid in cc["topic_pub_count"]:
        pubs.append(cc["topic_pub_count"][tid])
        subs.append(cc["topic_sub_count"].get(tid, 0))
        labels.append(cc["topic_names"].get(tid, tid))
        ids.append(tid)

    fanout = [p * s for p, s in zip(pubs, subs)]
    nonzero_fanout = [v for v in fanout if v > 0]
    outlier_indices: List[int] = []
    fanout_upper = 0.0
    fanout_iqr = 0.0
    if len(nonzero_fanout) >= 4:
        _, fanout_upper, fanout_iqr = find_1d_outliers_iqr(nonzero_fanout)
        outlier_indices = [
            i for i in range(len(fanout))
            if fanout[i] > fanout_upper and fanout[i] > 0
        ]
        outlier_indices.sort(key=lambda i: fanout[i], reverse=True)

    summary: Dict[str, Any] = {}
    if pubs:
        pub_arr = np.array(pubs, dtype=float)
        sub_arr = np.array(subs, dtype=float)
        summary = {
            "total_topics": len(pubs),
            "pub_mean": float(np.mean(pub_arr)),
            "pub_median": float(np.median(pub_arr)),
            "pub_max": int(np.max(pub_arr)),
            "sub_mean": float(np.mean(sub_arr)),
            "sub_median": float(np.median(sub_arr)),
            "sub_max": int(np.max(sub_arr)),
            "one_to_many": sum(1 for p, s in zip(pubs, subs) if p == 1 and s > 1),
            "many_to_one": sum(1 for p, s in zip(pubs, subs) if p > 1 and s == 1),
            "many_to_many": sum(1 for p, s in zip(pubs, subs) if p > 1 and s > 1),
            "orphan": sum(1 for p, s in zip(pubs, subs) if p == 0 or s == 0),
            "fanout_mean": float(np.mean(nonzero_fanout)) if nonzero_fanout else 0,
            "fanout_max": int(np.max(nonzero_fanout)) if nonzero_fanout else 0,
            "outlier_count": len(outlier_indices),
        }

    return {
        "pubs": pubs, "subs": subs, "labels": labels, "ids": ids,
        "fanout": fanout, "nonzero_fanout": nonzero_fanout,
        "outlier_indices": outlier_indices,
        "fanout_upper": fanout_upper, "fanout_iqr": fanout_iqr,
        "summary": summary,
    }


def _build_communication_matrix(
    cc: Dict[str, Any],
    entity_key_fn: Callable[[str], Optional[str]],
    entity_ids: List[str],
) -> "np.ndarray":
    """Build an NxN communication matrix for a set of entities.

    Cell (i,j) = number of topics where an entity-i app publishes
    and an entity-j app subscribes.

    Args:
        cc: Cross-cutting data.
        entity_key_fn: Maps app_id -> entity identifier (node_id or domain).
        entity_ids: Ordered list of entity identifiers.
    """
    idx = {eid: i for i, eid in enumerate(entity_ids)}
    n = len(entity_ids)

    topic_pub_entities: Dict[str, set] = {}
    topic_sub_entities: Dict[str, set] = {}

    for rel in cc["publishes_to"]:
        app_id = rel.get("from")
        topic_id = rel.get("to")
        eid = entity_key_fn(app_id)
        if eid and topic_id:
            topic_pub_entities.setdefault(topic_id, set()).add(eid)

    for rel in cc["subscribes_to"]:
        app_id = rel.get("from")
        topic_id = rel.get("to")
        eid = entity_key_fn(app_id)
        if eid and topic_id:
            topic_sub_entities.setdefault(topic_id, set()).add(eid)

    matrix = np.zeros((n, n))
    all_topics = set(topic_pub_entities.keys()) | set(topic_sub_entities.keys())
    for tid in all_topics:
        pub_ents = topic_pub_entities.get(tid, set())
        sub_ents = topic_sub_entities.get(tid, set())
        for pe in pub_ents:
            for se in sub_ents:
                if pe in idx and se in idx:
                    matrix[idx[pe]][idx[se]] += 1
    return matrix


def _compute_matrix_stats(
    matrix: "np.ndarray",
    labels: List[str],
    n: int,
) -> Dict[str, Any]:
    """Compute IQR outliers and summary statistics for a communication matrix."""
    nonzero_vals = matrix[matrix > 0].flatten()
    outlier_pairs: List[Tuple[str, str, int, float]] = []
    iqr_upper = 0.0
    iqr_val = 0.0

    if len(nonzero_vals) >= 4:
        _, iqr_upper, iqr_val = find_1d_outliers_iqr(nonzero_vals.tolist())
        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                if val > iqr_upper and val > 0:
                    dev = (val - iqr_upper) / iqr_val if iqr_val > 0 else 0.0
                    outlier_pairs.append((labels[i], labels[j], int(val), dev))
        outlier_pairs.sort(key=lambda x: x[2], reverse=True)

    total_cells = n * n
    nonzero_count = int(np.count_nonzero(matrix))
    diag_total = sum(int(matrix[i][i]) for i in range(n))
    off_diag_total = int(matrix.sum()) - diag_total
    cross_pairs = sum(1 for i in range(n) for j in range(n) if i != j and matrix[i][j] > 0)

    summary: Dict[str, Any] = {
        "entity_count": n,
        "total_cells": total_cells,
        "nonzero_count": nonzero_count,
        "active_pct": (nonzero_count / total_cells * 100) if total_cells > 0 else 0,
        "intra_total": diag_total,
        "inter_total": off_diag_total,
        "cross_pairs": cross_pairs,
        "outlier_count": len(outlier_pairs),
    }
    if len(nonzero_vals) > 0:
        summary["cell_mean"] = float(np.mean(nonzero_vals))
        summary["cell_median"] = float(np.median(nonzero_vals))
        summary["cell_max"] = int(np.max(nonzero_vals))

    return {
        "outlier_pairs": outlier_pairs,
        "iqr_upper": iqr_upper,
        "iqr": iqr_val,
        "summary": summary,
    }


def compute_cross_node_heatmap_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute cross-node communication heatmap statistics."""
    node_ids = [n["id"] for n in cc["nodes"]]
    n = len(node_ids)

    if n == 0:
        return {"node_ids": [], "labels": [], "matrix": np.zeros((0, 0)),
                "outlier_pairs": [], "iqr_upper": 0, "iqr": 0, "summary": {},
                "per_node": {}}

    labels = [cc["node_names"].get(nid, nid)[:20] for nid in node_ids]
    app_node = cc["app_node"]
    matrix = _build_communication_matrix(
        cc, lambda aid: app_node.get(aid), node_ids)

    # Build per-node topic lists: which topics are published/subscribed from each node
    node_pub_topics: Dict[str, Dict[str, str]] = {nid: {} for nid in node_ids}
    node_sub_topics: Dict[str, Dict[str, str]] = {nid: {} for nid in node_ids}
    node_apps: Dict[str, List[Dict[str, str]]] = {nid: [] for nid in node_ids}

    for app in cc["apps"]:
        aid = app["id"]
        nid = app_node.get(aid)
        if nid and nid in node_apps:
            node_apps[nid].append({"id": aid, "name": cc["app_names"].get(aid, aid)})

    for rel in cc["publishes_to"]:
        aid, tid = rel.get("from"), rel.get("to")
        nid = app_node.get(aid)
        if nid and nid in node_pub_topics and tid:
            node_pub_topics[nid][tid] = cc["topic_names"].get(tid, tid)

    for rel in cc["subscribes_to"]:
        aid, tid = rel.get("from"), rel.get("to")
        nid = app_node.get(aid)
        if nid and nid in node_sub_topics and tid:
            node_sub_topics[nid][tid] = cc["topic_names"].get(tid, tid)

    topic_sizes = cc["topic_sizes"]

    per_node = {
        nid: {
            "label": cc["node_names"].get(nid, nid),
            "apps": node_apps[nid],
            "pub_topics": [
                {"id": tid, "name": name, "size_kb": topic_sizes.get(tid, 0)}
                for tid, name in sorted(node_pub_topics[nid].items(), key=lambda x: x[1])
            ],
            "sub_topics": [
                {"id": tid, "name": name, "size_kb": topic_sizes.get(tid, 0)}
                for tid, name in sorted(node_sub_topics[nid].items(), key=lambda x: x[1])
            ],
        }
        for nid in node_ids
    }

    # Size-weighted matrix: cell(i,j) = sum of topic sizes (KB) for shared pub→sub topics
    idx = {nid: i for i, nid in enumerate(node_ids)}
    matrix_kb = np.zeros((n, n))
    # Build topic -> publishing nodes and subscribing nodes
    topic_pub_nodes: Dict[str, set] = {}
    topic_sub_nodes: Dict[str, set] = {}
    for nid in node_ids:
        for t in per_node[nid]["pub_topics"]:
            topic_pub_nodes.setdefault(t["id"], set()).add(nid)
        for t in per_node[nid]["sub_topics"]:
            topic_sub_nodes.setdefault(t["id"], set()).add(nid)
    for tid, pub_nodes in topic_pub_nodes.items():
        sub_nodes = topic_sub_nodes.get(tid, set())
        size = topic_sizes.get(tid, 0)
        for pn in pub_nodes:
            for sn in sub_nodes:
                if pn in idx and sn in idx:
                    matrix_kb[idx[pn]][idx[sn]] += size

    matrix_stats = _compute_matrix_stats(matrix, labels, n)
    return {
        "node_ids": node_ids, "labels": labels, "matrix": matrix, "matrix_kb": matrix_kb,
        "per_node": per_node,
        **matrix_stats,
    }


def compute_node_comm_load_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute node communication load statistics."""
    node_ids = [n["id"] for n in cc["nodes"]]
    node_labels = [cc["node_names"].get(nid, nid) for nid in node_ids]

    node_total_pub = {nid: 0 for nid in node_ids}
    node_total_sub = {nid: 0 for nid in node_ids}
    for aid, nid in cc["app_node"].items():
        if nid in node_total_pub:
            node_total_pub[nid] += cc["app_pub_count"].get(aid, 0)
            node_total_sub[nid] += cc["app_sub_count"].get(aid, 0)

    pub_vals = [node_total_pub[nid] for nid in node_ids]
    sub_vals = [node_total_sub[nid] for nid in node_ids]
    totals = [p + s for p, s in zip(pub_vals, sub_vals)]

    sorted_idx = sorted(range(len(totals)), key=lambda i: totals[i], reverse=True)
    sorted_labels = [node_labels[i][:25] for i in sorted_idx]
    sorted_ids = [node_ids[i] for i in sorted_idx]
    sorted_pub = [pub_vals[i] for i in sorted_idx]
    sorted_sub = [sub_vals[i] for i in sorted_idx]
    all_totals = [sorted_pub[i] + sorted_sub[i] for i in range(len(sorted_labels))]

    outliers: List[Tuple[str, int, int, int, float]] = []
    iqr_upper = 0.0
    iqr_val = 0.0
    if len(all_totals) >= 4:
        _, iqr_upper, iqr_val = find_1d_outliers_iqr(all_totals)
        for i in range(len(sorted_labels)):
            t = all_totals[i]
            if t > iqr_upper:
                dev = (t - iqr_upper) / iqr_val if iqr_val > 0 else 0.0
                outliers.append((sorted_labels[i], sorted_pub[i], sorted_sub[i], t, dev))
        outliers.sort(key=lambda x: x[3], reverse=True)

    summary: Dict[str, Any] = {"node_count": len(sorted_labels), "outlier_count": len(outliers)}
    if all_totals:
        total_arr = np.array(all_totals, dtype=float)
        pub_total = sum(sorted_pub)
        sub_total = sum(sorted_sub)
        summary.update({
            "pub_total": pub_total, "sub_total": sub_total,
            "load_mean": float(np.mean(total_arr)),
            "load_median": float(np.median(total_arr)),
            "load_std": float(np.std(total_arr)),
            "cv": float(np.std(total_arr) / np.mean(total_arr) * 100) if np.mean(total_arr) > 0 else 0,
            "zero_load": sum(1 for t in all_totals if t == 0),
        })

    return {
        "node_ids": node_ids, "sorted_labels": sorted_labels, "sorted_ids": sorted_ids,
        "sorted_pub": sorted_pub, "sorted_sub": sorted_sub,
        "all_totals": all_totals,
        "outliers": outliers, "iqr_upper": iqr_upper, "iqr": iqr_val,
        "summary": summary,
    }


def compute_domain_comm_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute domain-to-domain communication matrix statistics."""
    app_domain = cc["app_domain"]
    domain_set = sorted({d for d in app_domain.values() if d != "NOT_FOUND"})

    if len(domain_set) < 2:
        return {"domain_set": domain_set, "labels": [], "matrix": np.zeros((0, 0)),
                "outlier_pairs": [], "iqr_upper": 0, "iqr": 0, "summary": {}}

    n = len(domain_set)
    labels = [d[:25] for d in domain_set]
    matrix = _build_communication_matrix(
        cc, lambda aid: app_domain.get(aid, "NOT_FOUND") if app_domain.get(aid, "NOT_FOUND") != "NOT_FOUND" else None,
        domain_set)

    matrix_stats = _compute_matrix_stats(matrix, labels, n)
    return {
        "domain_set": domain_set, "labels": labels, "matrix": matrix,
        **matrix_stats,
    }


def compute_criticality_io_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute criticality × I/O load statistics."""
    app_criticality = cc["app_criticality"]
    app_pub = cc["app_pub_count"]
    app_sub = cc["app_sub_count"]
    app_names = cc["app_names"]

    crit_pubs, crit_subs, crit_labels, crit_ids = [], [], [], []
    norm_pubs, norm_subs, norm_labels, norm_ids = [], [], [], []

    for aid in app_pub:
        p = app_pub[aid]
        s = app_sub.get(aid, 0)
        name = app_names.get(aid, aid)
        if app_criticality.get(aid, False):
            crit_pubs.append(p)
            crit_subs.append(s)
            crit_labels.append(name)
            crit_ids.append(aid)
        else:
            norm_pubs.append(p)
            norm_subs.append(s)
            norm_labels.append(name)
            norm_ids.append(aid)

    crit_io = [p + s for p, s in zip(crit_pubs, crit_subs)]
    nonzero_crit = [v for v in crit_io if v > 0]
    outliers: List[Tuple[str, int, int, int]] = []
    iqr_upper = 0.0
    if len(nonzero_crit) >= 4:
        _, iqr_upper, _ = find_1d_outliers_iqr(nonzero_crit)
        outliers = [
            (crit_labels[i], crit_pubs[i], crit_subs[i], crit_io[i])
            for i in range(len(crit_io)) if crit_io[i] > iqr_upper
        ]
        outliers.sort(key=lambda x: x[3], reverse=True)

    total_apps = len(crit_pubs) + len(norm_pubs)
    summary: Dict[str, Any] = {
        "total_apps": total_apps,
        "crit_count": len(crit_pubs),
        "crit_pct": (len(crit_pubs) / total_apps * 100) if total_apps > 0 else 0,
        "outlier_count": len(outliers),
    }
    if crit_io:
        crit_arr = np.array(crit_io, dtype=float)
        summary.update({
            "crit_io_mean": float(np.mean(crit_arr)),
            "crit_io_median": float(np.median(crit_arr)),
            "crit_io_max": int(np.max(crit_arr)),
        })
    norm_io = [p + s for p, s in zip(norm_pubs, norm_subs)]
    if norm_io:
        norm_arr = np.array(norm_io, dtype=float)
        summary.update({
            "norm_io_mean": float(np.mean(norm_arr)),
            "norm_io_median": float(np.median(norm_arr)),
            "norm_io_max": int(np.max(norm_arr)),
        })
    if crit_io and norm_io:
        crit_mean = float(np.mean(crit_io))
        norm_mean = float(np.mean(norm_io))
        if norm_mean > 0:
            summary["crit_norm_ratio"] = crit_mean / norm_mean

    return {
        "crit_pubs": crit_pubs, "crit_subs": crit_subs, "crit_labels": crit_labels, "crit_ids": crit_ids,
        "norm_pubs": norm_pubs, "norm_subs": norm_subs, "norm_labels": norm_labels, "norm_ids": norm_ids,
        "crit_io": crit_io,
        "outliers": outliers, "iqr_upper": iqr_upper,
        "summary": summary,
    }


def compute_lib_dependency_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute library dependency density statistics."""
    uses = cc["uses"]
    lib_names = cc["lib_names"]
    app_names = cc["app_names"]

    if not uses:
        return {
            "active_ids": [], "display_ids": [], "labels": [],
            "in_vals": [], "out_vals": [],
            "outliers": [], "iqr_upper": 0, "iqr": 0,
            "summary": {"total_relations": 0},
        }

    all_ids = set(app_names.keys()) | set(lib_names.keys())
    entity_names = {**app_names, **lib_names}

    in_degree: Dict[str, int] = {eid: 0 for eid in all_ids}
    out_degree: Dict[str, int] = {eid: 0 for eid in all_ids}
    for rel in uses:
        src = rel.get("from")
        dst = rel.get("to")
        if src in all_ids and dst in all_ids:
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[dst] = in_degree.get(dst, 0) + 1

    active_ids = [eid for eid in all_ids if in_degree.get(eid, 0) > 0 or out_degree.get(eid, 0) > 0]
    active_ids.sort(key=lambda eid: in_degree.get(eid, 0) + out_degree.get(eid, 0), reverse=True)

    display_ids = active_ids[:40]
    labels = [entity_names.get(eid, eid)[:30] for eid in display_ids]
    in_vals = [in_degree.get(eid, 0) for eid in display_ids]
    out_vals = [out_degree.get(eid, 0) for eid in display_ids]

    all_in = [in_degree.get(eid, 0) for eid in active_ids]
    nonzero_in = [v for v in all_in if v > 0]
    outliers: List[Tuple[str, int, int]] = []
    iqr_upper = 0.0
    iqr_val = 0.0
    if len(nonzero_in) >= 4:
        _, iqr_upper, iqr_val = find_1d_outliers_iqr(nonzero_in)
        outliers = [
            (entity_names.get(eid, eid), in_degree.get(eid, 0), out_degree.get(eid, 0))
            for eid in active_ids if in_degree.get(eid, 0) > iqr_upper
        ]
        outliers.sort(key=lambda x: x[1], reverse=True)

    all_out = [out_degree.get(eid, 0) for eid in active_ids]
    nonzero_out = [v for v in all_out if v > 0]

    lib_count = sum(1 for eid in active_ids if eid in lib_names)
    app_count = len(active_ids) - lib_count

    summary: Dict[str, Any] = {
        "total_relations": len(uses),
        "active_count": len(active_ids),
        "app_count": app_count,
        "lib_count": lib_count,
        "outlier_count": len(outliers),
    }
    if nonzero_in:
        summary["in_mean"] = float(np.mean(nonzero_in))
        summary["in_max"] = int(np.max(nonzero_in))
    if nonzero_out:
        summary["out_mean"] = float(np.mean(nonzero_out))
        summary["out_max"] = int(np.max(nonzero_out))

    return {
        "active_ids": active_ids, "display_ids": display_ids,
        "labels": labels, "in_vals": in_vals, "out_vals": out_vals,
        "entity_names": entity_names, "lib_names_set": set(lib_names.keys()),
        "outliers": outliers, "iqr_upper": iqr_upper, "iqr": iqr_val,
        "summary": summary,
    }


def compute_node_critical_density_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute node critical application density statistics."""
    node_ids = [n["id"] for n in cc["nodes"]]
    node_labels = [cc["node_names"].get(nid, nid) for nid in node_ids]
    app_criticality = cc["app_criticality"]
    app_node = cc["app_node"]

    node_crit = {nid: 0 for nid in node_ids}
    node_norm = {nid: 0 for nid in node_ids}
    for aid, nid in app_node.items():
        if nid in node_crit:
            if app_criticality.get(aid, False):
                node_crit[nid] += 1
            else:
                node_norm[nid] += 1

    crit_vals = [node_crit[nid] for nid in node_ids]
    norm_vals = [node_norm[nid] for nid in node_ids]
    totals = [c + n for c, n in zip(crit_vals, norm_vals)]

    sorted_idx = sorted(range(len(totals)), key=lambda i: totals[i], reverse=True)
    sorted_labels = [node_labels[i][:25] for i in sorted_idx]
    sorted_ids = [node_ids[i] for i in sorted_idx]
    sorted_crit = [crit_vals[i] for i in sorted_idx]
    sorted_norm = [norm_vals[i] for i in sorted_idx]

    total_crit = sum(crit_vals)
    total_norm = sum(norm_vals)
    total_all = total_crit + total_norm

    crit_ratios = [c / (c + n) * 100 if (c + n) > 0 else 0 for c, n in zip(crit_vals, norm_vals)]
    max_ratio_idx = max(range(len(crit_ratios)), key=lambda i: crit_ratios[i]) if crit_ratios else 0

    summary: Dict[str, Any] = {
        "node_count": len(node_ids),
        "total_all": total_all,
        "total_crit": total_crit,
        "total_norm": total_norm,
        "system_crit_pct": (total_crit / total_all * 100) if total_all > 0 else 0,
        "zero_crit": sum(1 for c in crit_vals if c == 0),
    }
    if crit_vals:
        crit_arr = np.array(crit_vals, dtype=float)
        summary["crit_per_node_mean"] = float(np.mean(crit_arr))
        summary["crit_per_node_max"] = int(np.max(crit_arr))
    if crit_ratios and node_labels:
        summary["max_ratio_node"] = node_labels[max_ratio_idx]
        summary["max_ratio_pct"] = crit_ratios[max_ratio_idx]

    return {
        "node_ids": node_ids, "node_labels": node_labels,
        "sorted_labels": sorted_labels, "sorted_ids": sorted_ids,
        "sorted_crit": sorted_crit, "sorted_norm": sorted_norm,
        "crit_vals": crit_vals, "norm_vals": norm_vals,
        "summary": summary,
    }


def compute_domain_diversity_stats(cc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute domain application and topic diversity statistics."""
    app_domain = cc["app_domain"]
    app_pub = cc["app_pub_count"]
    app_sub = cc["app_sub_count"]

    domain_set = sorted({d for d in app_domain.values() if d != "NOT_FOUND"})
    if len(domain_set) < 2:
        return {"domain_set": domain_set, "labels": [], "app_counts": [],
                "topic_counts": [], "io_vals": [], "ranked": [],
                "summary": {}}

    domain_apps: Dict[str, set] = {d: set() for d in domain_set}
    domain_topics: Dict[str, set] = {d: set() for d in domain_set}
    domain_io: Dict[str, int] = {d: 0 for d in domain_set}

    for aid, dom in app_domain.items():
        if dom in domain_apps:
            domain_apps[dom].add(aid)
            domain_io[dom] += app_pub.get(aid, 0) + app_sub.get(aid, 0)

    for rel in cc["publishes_to"]:
        aid = rel.get("from")
        tid = rel.get("to")
        dom = app_domain.get(aid, "NOT_FOUND")
        if dom in domain_topics and tid:
            domain_topics[dom].add(tid)

    for rel in cc["subscribes_to"]:
        aid = rel.get("from")
        tid = rel.get("to")
        dom = app_domain.get(aid, "NOT_FOUND")
        if dom in domain_topics and tid:
            domain_topics[dom].add(tid)

    labels = list(domain_set)
    app_counts = [len(domain_apps[d]) for d in labels]
    topic_counts = [len(domain_topics[d]) for d in labels]
    io_vals = [domain_io[d] for d in labels]

    ranked = sorted(
        zip(labels, app_counts, topic_counts, io_vals),
        key=lambda x: x[3], reverse=True,
    )

    summary: Dict[str, Any] = {"css_count": len(labels)}
    if app_counts:
        summary["app_mean"] = float(np.mean(app_counts))
        summary["app_max"] = int(np.max(app_counts))
    if topic_counts:
        summary["topic_mean"] = float(np.mean(topic_counts))
        summary["topic_max"] = int(np.max(topic_counts))
    if io_vals:
        summary["io_mean"] = float(np.mean(io_vals))
        summary["io_max"] = int(np.max(io_vals))

    return {
        "domain_set": domain_set, "labels": labels,
        "app_counts": app_counts, "topic_counts": topic_counts,
        "io_vals": io_vals, "ranked": ranked,
        "summary": summary,
    }


# ===================== TOP-LEVEL ENTRY POINT =====================


def compute_all_extras_statistics(
    cc: Dict[str, Any],
    risk_weight_fn: Optional[Callable[[str, str], float]] = None,
    w2name: Optional[Dict[str, Dict[float, str]]] = None,
) -> Dict[str, Any]:
    """Compute all extras chart statistics from cross-cutting data.

    This is the main entry point for standalone statistics computation.
    Given cross-cutting data (from ``extract_cross_cutting_data``), this
    function computes all statistics used by the extras charts.

    Args:
        cc: Cross-cutting data dict from ``extract_cross_cutting_data``.
        risk_weight_fn: Callable(dimension, value) -> float weight for QoS
            risk scoring.  When *None* the QoS chart is skipped.
        w2name: Weight-to-display-name maps per QoS dimension.

    Returns:
        Dict keyed by chart identifier with each chart's statistics.
    """
    result: Dict[str, Any] = {
        "topic_bandwidth": compute_topic_bandwidth_stats(cc),
        "app_balance": compute_app_balance_stats(cc),
        "topic_fanout": compute_topic_fanout_stats(cc),
        "cross_node_heatmap": compute_cross_node_heatmap_stats(cc),
        "node_comm_load": compute_node_comm_load_stats(cc),
        "domain_comm": compute_domain_comm_stats(cc),
        "criticality_io": compute_criticality_io_stats(cc),
        "lib_dependency": compute_lib_dependency_stats(cc),
        "node_critical_density": compute_node_critical_density_stats(cc),
        "domain_diversity": compute_domain_diversity_stats(cc),
    }

    if risk_weight_fn is not None:
        result["qos_risk"] = compute_qos_risk_stats(cc, risk_weight_fn, w2name)

    return result
