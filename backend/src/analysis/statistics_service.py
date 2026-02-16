"""
Statistics Service

Provides comprehensive graph statistics and metrics.
All business logic for statistics computation lives here;
the API router is a thin adapter that delegates to these methods.
"""
import logging
import time
import statistics as py_stats
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Union

from src.core import create_repository
from src.core.interfaces import IGraphRepository
from src.core.models import GraphData
from src.analysis import statistics as stats_logic

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _patch_graph_data(graph_data: GraphData) -> GraphData:
    """
    Ensure every ComponentData object has a ``type`` attribute
    (alias for ``component_type``).
    """
    for component in graph_data.components:
        if not hasattr(component, "type"):
            component.type = component.component_type
    return graph_data


def _component_info(graph_data: GraphData) -> Dict[str, Dict[str, str]]:
    """Build a ``{id: {name, type}}`` lookup from component data."""
    return {
        c.id: {"name": c.properties.get("name", c.id), "type": c.type}
        for c in graph_data.components
    }


def _gini_coefficient(values: List[float]) -> float:
    """Approximate Gini coefficient for a list of non-negative values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if n == 0 or total == 0:
        return 0.0
    cumsum = sum((2 * (i + 1) - n - 1) * w for i, w in enumerate(sorted_vals))
    return cumsum / (n * total)


def _health_from_concentration(concentration: float) -> Dict[str, str]:
    """Return health / category / interpretation based on a concentration score."""
    if concentration > 0.7:
        return {"health": "poor", "category": "concentrated",
                "interpretation": "High concentration - few items dominate importance"}
    if concentration > 0.5:
        return {"health": "moderate", "category": "moderately_concentrated",
                "interpretation": "Moderate concentration - some imbalance in importance"}
    if concentration > 0.3:
        return {"health": "fair", "category": "balanced",
                "interpretation": "Balanced distribution with some key items"}
    return {"health": "good", "category": "distributed",
            "interpretation": "Even distribution - importance is well spread"}


# ── Service ──────────────────────────────────────────────────────────────

class StatisticsService:
    """
    Service for calculating graph statistics and metrics.

    Accepts either a Neo4j URI string (creates its own repository) or
    an existing ``IGraphRepository`` instance.
    """

    def __init__(
        self,
        uri_or_repo: Union[str, IGraphRepository],
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if isinstance(uri_or_repo, str):
            self.repository: IGraphRepository = create_repository(uri_or_repo, user, password)
            self._own_repo = True
        else:
            self.repository = uri_or_repo
            self._own_repo = False

    def close(self) -> None:
        """Close repository if we created it."""
        if self._own_repo:
            self.repository.close()

    # ── helpers ───────────────────────────────────────────────────────

    def _graph_data(self, node_type: Optional[str] = None) -> GraphData:
        """Fetch and patch graph data from the repository."""
        return _patch_graph_data(self.repository.get_graph_data())

    # ── public methods ───────────────────────────────────────────────

    def get_graph_stats(self) -> Dict[str, Any]:
        """Overall graph statistics from the repository."""
        return self.repository.get_statistics()

    def get_degree_distribution(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """Degree distribution: in/out/total, hubs, isolated nodes."""
        gd = self._graph_data()
        return stats_logic.get_degree_distribution(gd, node_type=node_type)

    def get_connectivity_density(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """Graph density and top-connected components."""
        gd = self._graph_data()
        return stats_logic.get_connectivity_density(gd, node_type=node_type)

    # ── clustering ───────────────────────────────────────────────────

    def get_clustering_coefficient(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Clustering coefficient with rich per-node breakdown.

        Returns base stats from ``stats_logic`` enriched with local
        coefficient distribution, categorised node lists, etc.
        """
        gd = self._graph_data()
        stats = stats_logic.get_clustering_coefficient(gd, node_type=node_type)

        info = _component_info(gd)
        component_types = {c.id: c.type for c in gd.components}

        # Build neighbour sets
        neighbors: Dict[str, set] = defaultdict(set)
        for edge in gd.edges:
            neighbors[edge.source_id].add(edge.target_id)
            neighbors[edge.target_id].add(edge.source_id)

        # Local clustering coefficients
        local_coefficients: Dict[str, float] = {}
        for node in info:
            node_neighbors = neighbors.get(node, set())
            degree = len(node_neighbors)
            if degree < 2:
                local_coefficients[node] = 0.0
                continue
            triangles = 0
            neighbor_list = list(node_neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in neighbors[neighbor_list[i]]:
                        triangles += 1
            possible = degree * (degree - 1) / 2
            local_coefficients[node] = triangles / possible if possible > 0 else 0.0

        # Optional type filter
        filtered = (
            {k: v for k, v in local_coefficients.items() if component_types.get(k) == node_type}
            if node_type
            else local_coefficients
        )

        coeff_values = list(filtered.values())
        non_zero = [c for c in coeff_values if c > 0]

        stats["global_clustering"] = stats.get("avg_clustering_coefficient", 0)
        stats["average_clustering"] = stats.get("avg_clustering_coefficient", 0)
        stats["max_coefficient"] = round(max(coeff_values), 4) if coeff_values else 0
        stats["min_coefficient"] = round(min(non_zero), 4) if non_zero else 0
        stats["median_coefficient"] = round(py_stats.median(coeff_values), 4) if coeff_values else 0
        stats["std_coefficient"] = round(py_stats.stdev(non_zero), 4) if len(non_zero) > 1 else 0

        high = [(nid, c) for nid, c in filtered.items() if c >= 0.5]
        medium = [(nid, c) for nid, c in filtered.items() if 0.2 <= c < 0.5]
        low = [(nid, c) for nid, c in filtered.items() if 0 < c < 0.2]
        zero = [(nid, c) for nid, c in filtered.items() if c == 0]

        stats["high_clustering_count"] = len(high)
        stats["medium_clustering_count"] = len(medium)
        stats["low_clustering_count"] = len(low)
        stats["zero_clustering_count"] = len(zero)
        stats["total_nodes"] = len(filtered)

        stats["high_clustering_nodes"] = [
            {"id": nid, "name": info[nid]["name"], "type": info[nid]["type"],
             "coefficient": round(c, 4), "degree": len(neighbors[nid])}
            for nid, c in sorted(high, key=lambda x: x[1], reverse=True)[:20]
        ]
        stats["zero_clustering_nodes"] = [
            {"id": nid, "name": info[nid]["name"], "type": info[nid]["type"],
             "degree": len(neighbors[nid])}
            for nid, _ in zero[:20]
        ]
        return stats

    # ── dependency depth ─────────────────────────────────────────────

    def get_dependency_depth(self) -> Dict[str, Any]:
        """Dependency depth via BFS, with categorisation and root/leaf lists."""
        gd = self._graph_data()
        stats = stats_logic.get_dependency_depth(gd)

        info = _component_info(gd)
        outgoing: Dict[str, list] = defaultdict(list)
        incoming: Dict[str, list] = defaultdict(list)
        for edge in gd.edges:
            outgoing[edge.source_id].append(edge.target_id)
            incoming[edge.target_id].append(edge.source_id)

        node_depths: Dict[str, int] = {}
        for node in info:
            max_d, visited = 0, {node}
            queue: deque = deque([(node, 0)])
            while queue:
                curr, d = queue.popleft()
                max_d = max(max_d, d)
                for neighbor in outgoing[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
            node_depths[node] = max_d

        depth_values = list(node_depths.values())
        stats["median_depth"] = round(py_stats.median(depth_values), 2) if depth_values else 0
        stats["min_depth"] = min(depth_values) if depth_values else 0
        stats["std_depth"] = round(py_stats.stdev(depth_values), 2) if len(depth_values) > 1 else 0

        shallow = [(nid, d) for nid, d in node_depths.items() if d == 0]
        low_d = [(nid, d) for nid, d in node_depths.items() if 1 <= d <= 2]
        med_d = [(nid, d) for nid, d in node_depths.items() if 3 <= d <= 5]
        high_d = [(nid, d) for nid, d in node_depths.items() if d > 5]

        stats["shallow_count"] = len(shallow)
        stats["low_depth_count"] = len(low_d)
        stats["medium_depth_count"] = len(med_d)
        stats["high_depth_count"] = len(high_d)
        stats["total_nodes"] = len(node_depths)

        depth_dist: Dict[str, int] = {}
        for d in depth_values:
            depth_dist[str(d)] = depth_dist.get(str(d), 0) + 1
        stats["depth_distribution"] = depth_dist

        stats["root_nodes"] = [
            {"id": nid, "name": info[nid]["name"], "type": info[nid]["type"],
             "depth": node_depths[nid], "dependencies": len(outgoing[nid])}
            for nid in info if not incoming[nid]
        ][:20]

        stats["leaf_nodes"] = [
            {"id": nid, "name": info[nid]["name"], "type": info[nid]["type"],
             "depth": node_depths[nid], "dependents": len(incoming[nid])}
            for nid in info if not outgoing[nid]
        ][:20]

        max_depth = stats.get("max_depth", 0)
        if max_depth <= 2:
            stats["interpretation"] = "Shallow dependencies - simple architecture"
            stats["category"] = "shallow"
        elif max_depth <= 5:
            stats["interpretation"] = "Moderate depth - balanced complexity"
            stats["category"] = "moderate"
        elif max_depth <= 10:
            stats["interpretation"] = "Deep dependencies - complex architecture"
            stats["category"] = "deep"
        else:
            stats["interpretation"] = "Very deep dependencies - review architecture"
            stats["category"] = "very_deep"

        return stats

    # ── component isolation ──────────────────────────────────────────

    def get_component_isolation(self) -> Dict[str, Any]:
        """Identify isolated, source, sink, and bidirectional components."""
        gd = self._graph_data()
        stats = stats_logic.get_component_isolation(gd)

        total = stats.get("total_components", 0)
        stats["source_percentage"] = round(stats.get("source_count", 0) / total * 100, 2) if total else 0
        stats["sink_percentage"] = round(stats.get("sink_count", 0) / total * 100, 2) if total else 0
        stats["bidirectional_percentage"] = round(stats.get("bidirectional_count", 0) / total * 100, 2) if total else 0

        isolated_pct = stats.get("isolated_percentage", 0)
        if isolated_pct > 30:
            stats.update(health="poor", category="highly_isolated",
                         interpretation="High isolation indicates poor integration")
        elif isolated_pct > 15:
            stats.update(health="moderate", category="moderately_isolated",
                         interpretation="Moderate isolation - review integration points")
        elif isolated_pct > 5:
            stats.update(health="fair", category="low_isolation",
                         interpretation="Low isolation - good connectivity")
        else:
            stats.update(health="good", category="well_connected",
                         interpretation="Excellent connectivity")

        info = _component_info(gd)
        inc: Dict[str, int] = defaultdict(int)
        outc: Dict[str, int] = defaultdict(int)
        for edge in gd.edges:
            outc[edge.source_id] += 1
            inc[edge.target_id] += 1

        isolated, sources, sinks, bidir = [], [], [], []
        for cid in info:
            data = {"id": cid, "name": info[cid]["name"], "type": info[cid]["type"],
                    "in_degree": inc[cid], "out_degree": outc[cid]}
            if inc[cid] == 0 and outc[cid] == 0:
                isolated.append(data)
            elif inc[cid] == 0:
                sources.append(data)
            elif outc[cid] == 0:
                sinks.append(data)
            else:
                bidir.append(data)

        stats["isolated_components"] = isolated[:20]
        stats["top_sources"] = sorted(sources, key=lambda x: x["out_degree"], reverse=True)[:20]
        stats["top_sinks"] = sorted(sinks, key=lambda x: x["in_degree"], reverse=True)[:20]
        stats["top_bidirectional"] = sorted(bidir, key=lambda x: x["in_degree"] + x["out_degree"], reverse=True)[:20]
        stats["total_nodes"] = stats.get("total_components", 0)
        return stats

    # ── message flow patterns ────────────────────────────────────────

    def get_message_flow_patterns(self) -> Dict[str, Any]:
        """Analyse pub/sub message-flow patterns from graph structure."""
        t0 = time.time()
        gd = self._graph_data()
        basic_stats = self.repository.get_statistics()

        topics = [c for c in gd.components if c.type == "Topic"]
        brokers = [c for c in gd.components if c.type == "Broker"]
        applications = [c for c in gd.components if c.type == "Application"]

        pub_edges = [e for e in gd.edges if e.relation_type == "PUBLISHES_TO"]
        sub_edges = [e for e in gd.edges if e.relation_type == "SUBSCRIBES_TO"]

        topic_pubs: Dict[str, int] = defaultdict(int)
        topic_subs: Dict[str, int] = defaultdict(int)
        app_pubs: Dict[str, int] = defaultdict(int)
        app_subs: Dict[str, int] = defaultdict(int)

        for e in pub_edges:
            topic_pubs[e.target_id] += 1
            app_pubs[e.source_id] += 1
        for e in sub_edges:
            topic_subs[e.target_id] += 1
            app_subs[e.source_id] += 1

        hot_topics = sorted(
            [{"id": t.id, "name": t.properties.get("name", t.id),
              "publishers": topic_pubs.get(t.id, 0),
              "subscribers": topic_subs.get(t.id, 0),
              "total_activity": topic_pubs.get(t.id, 0) + topic_subs.get(t.id, 0),
              "brokers": 0}
             for t in topics if topic_pubs.get(t.id, 0) + topic_subs.get(t.id, 0) > 0],
            key=lambda x: x["total_activity"], reverse=True,
        )[:20]

        top_publishers = sorted(
            [{"id": a.id, "name": a.properties.get("name", a.id),
              "publications": app_pubs[a.id]}
             for a in applications if app_pubs.get(a.id, 0) > 0],
            key=lambda x: x["publications"], reverse=True,
        )[:20]

        top_subscribers = sorted(
            [{"id": a.id, "name": a.properties.get("name", a.id),
              "subscriptions": app_subs[a.id]}
             for a in applications if app_subs.get(a.id, 0) > 0],
            key=lambda x: x["subscriptions"], reverse=True,
        )[:20]

        isolated_apps = [
            {"id": a.id, "name": a.properties.get("name", a.id)}
            for a in applications
            if app_pubs.get(a.id, 0) == 0 and app_subs.get(a.id, 0) == 0
        ]

        active = len([a for a in applications if app_pubs.get(a.id, 0) or app_subs.get(a.id, 0)])
        avg_pubs = sum(topic_pubs.values()) / len(topics) if topics else 0
        avg_subs = sum(topic_subs.values()) / len(topics) if topics else 0

        if len(isolated_apps) > len(applications) * 0.5:
            health, category = "poor", "isolated"
            interpretation = "Many applications are isolated from pub-sub system"
        elif len(hot_topics) < 3 and len(topics) > 10:
            health, category = "moderate", "sparse"
            interpretation = "Low message flow activity relative to system size"
        elif avg_pubs > 5 or avg_subs > 5:
            health, category = "fair", "bottleneck"
            interpretation = "Some topics may be overloaded with connections"
        else:
            health, category = "good", "balanced"
            interpretation = "Message flow appears well distributed"

        return {
            "total_topics": len(topics),
            "total_brokers": len(brokers),
            "total_applications": len(applications),
            "active_applications": active,
            "avg_publishers_per_topic": round(avg_pubs, 2),
            "avg_subscribers_per_topic": round(avg_subs, 2),
            "avg_topics_per_broker": 0,
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "hot_topics": hot_topics,
            "broker_utilization": [],
            "isolated_applications": isolated_apps[:20],
            "top_publishers": top_publishers,
            "top_subscribers": top_subscribers,
            "computation_time_ms": round((time.time() - t0) * 1000, 2),
        }

    # ── component redundancy ─────────────────────────────────────────

    def get_component_redundancy(self) -> Dict[str, Any]:
        """Identify SPOFs, bridge components and compute resilience score."""
        t0 = time.time()
        gd = self._graph_data()

        info = _component_info(gd)
        incoming: Dict[str, list] = defaultdict(list)
        outgoing: Dict[str, list] = defaultdict(list)
        for edge in gd.edges:
            outgoing[edge.source_id].append(edge.target_id)
            incoming[edge.target_id].append(edge.source_id)

        spofs: List[Dict[str, Any]] = []
        for comp_id in info:
            out_deg = len(outgoing[comp_id])
            in_deg = len(incoming[comp_id])
            if out_deg > 5:
                spofs.append({
                    "id": comp_id, "name": info[comp_id]["name"],
                    "type": info[comp_id]["type"],
                    "in_degree": in_deg, "out_degree": out_deg,
                    "is_critical": True, "reason": f"High fan-out ({out_deg} dependents)",
                })
            elif in_deg > 3 and out_deg > 3:
                spofs.append({
                    "id": comp_id, "name": info[comp_id]["name"],
                    "type": info[comp_id]["type"],
                    "in_degree": in_deg, "out_degree": out_deg,
                    "is_critical": True, "reason": f"Bridge component ({in_deg} in, {out_deg} out)",
                })

        spofs.sort(key=lambda x: x["out_degree"], reverse=True)
        spofs = spofs[:20]

        redundant = [cid for cid in info if len(incoming[cid]) > 2]
        total = len(info)
        spof_pct = (len(spofs) / total * 100) if total else 0
        red_pct = (len(redundant) / total * 100) if total else 0
        resilience = max(0, min(100, 100 - spof_pct + red_pct * 0.3))

        if resilience >= 70:
            health, interp = "good", "High resilience - good redundancy and few SPOFs"
        elif resilience >= 50:
            health, interp = "fair", "Moderate resilience - some redundancy present"
        elif resilience >= 30:
            health, interp = "moderate", "Limited resilience - notable SPOFs identified"
        else:
            health, interp = "poor", "Low resilience - many SPOFs and limited redundancy"

        if len(spofs) > total * 0.3:
            category = "fragile"
        elif len(redundant) > total * 0.5:
            category = "resilient"
        elif len(redundant) > total * 0.2:
            category = "moderate"
        else:
            category = "vulnerable"

        return {
            "total_components": total,
            "spof_count": len(spofs),
            "spof_percentage": round(spof_pct, 2),
            "redundant_count": len(redundant),
            "redundancy_percentage": round(red_pct, 2),
            "resilience_score": round(resilience, 1),
            "interpretation": interp,
            "category": category,
            "health": health,
            "single_points_of_failure": spofs,
            "bridge_components": spofs[:10],
            "computation_time_ms": round((time.time() - t0) * 1000, 2),
        }

    # ── node weight distribution ─────────────────────────────────────

    def get_node_weight_distribution(self) -> Dict[str, Any]:
        """Node importance distribution based on degree centrality."""
        t0 = time.time()
        gd = self._graph_data()

        info = _component_info(gd)
        degree_map: Dict[str, int] = defaultdict(int)
        for edge in gd.edges:
            degree_map[edge.source_id] += 1
            degree_map[edge.target_id] += 1

        max_degree = max(degree_map.values()) if degree_map else 1
        weights = {cid: degree_map[cid] / max_degree for cid in info}

        wv = list(weights.values())
        concentration = _gini_coefficient(wv)
        health_info = _health_from_concentration(concentration)

        top_components = [
            {"id": cid, "name": info[cid]["name"], "type": info[cid]["type"],
             "weight": round(w, 4)}
            for cid, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
        ]

        by_type: Dict[str, list] = defaultdict(list)
        for cid, ci in info.items():
            by_type[ci["type"]].append(weights[cid])
        type_stats = {}
        for t, tw in by_type.items():
            type_stats[t] = {
                "count": len(tw),
                "total_weight": round(sum(tw), 3),
                "avg_weight": round(py_stats.mean(tw), 3),
                "median_weight": round(py_stats.median(tw), 3),
                "min_weight": round(min(tw), 3),
                "max_weight": round(max(tw), 3),
                "std_weight": round(py_stats.stdev(tw), 3) if len(tw) > 1 else 0,
            }

        return {
            "total_components": len(info),
            "total_weight": round(sum(wv), 3),
            "avg_weight": round(py_stats.mean(wv), 4) if wv else 0,
            "median_weight": round(py_stats.median(wv), 4) if wv else 0,
            "min_weight": round(min(wv), 4) if wv else 0,
            "max_weight": round(max(wv), 4) if wv else 0,
            "std_weight": round(py_stats.stdev(wv), 4) if len(wv) > 1 else 0,
            "weight_concentration": round(concentration, 4),
            **health_info,
            "very_high_count": sum(1 for w in wv if w >= 0.8),
            "high_count": sum(1 for w in wv if 0.6 <= w < 0.8),
            "medium_count": sum(1 for w in wv if 0.3 <= w < 0.6),
            "low_count": sum(1 for w in wv if 0.1 <= w < 0.3),
            "very_low_count": sum(1 for w in wv if w < 0.1),
            "top_components": top_components,
            "type_stats": type_stats,
            "computation_time_ms": round((time.time() - t0) * 1000, 2),
        }

    # ── edge weight distribution ─────────────────────────────────────

    def get_edge_weight_distribution(self) -> Dict[str, Any]:
        """Edge weight distribution with type-based breakdown."""
        t0 = time.time()
        gd = self._graph_data()

        component_names = {c.id: c.properties.get("name", c.id) for c in gd.components}

        edge_weights: List[float] = []
        edge_data_list: List[Dict] = []
        type_weights: Dict[str, list] = defaultdict(list)

        for edge in gd.edges:
            weight = edge.properties.get("weight", 1.0) if hasattr(edge, "properties") else 1.0
            edge_weights.append(weight)
            edge_data_list.append({"edge": edge, "weight": weight})
            edge_type = getattr(edge, "relation_type", "DEPENDS_ON")
            type_weights[edge_type].append(weight)

        if not edge_weights:
            edge_weights = [0]

        concentration = _gini_coefficient(edge_weights)
        health_info = _health_from_concentration(concentration)
        max_w = max(edge_weights)

        type_stats = {}
        for et, ws in type_weights.items():
            type_stats[et] = {
                "count": len(ws),
                "total_weight": round(sum(ws), 3),
                "avg_weight": round(py_stats.mean(ws), 3),
                "median_weight": round(py_stats.median(ws), 3),
                "min_weight": round(min(ws), 3),
                "max_weight": round(max(ws), 3),
                "std_weight": round(py_stats.stdev(ws), 3) if len(ws) > 1 else 0,
            }

        top_edges = []
        for item in sorted(edge_data_list, key=lambda x: x["weight"], reverse=True)[:20]:
            e = item["edge"]
            top_edges.append({
                "source": e.source_id, "target": e.target_id,
                "source_name": component_names.get(e.source_id, e.source_id),
                "target_name": component_names.get(e.target_id, e.target_id),
                "type": e.relation_type, "weight": round(item["weight"], 4),
            })

        return {
            "total_edges": len(edge_weights),
            "total_weight": round(sum(edge_weights), 3),
            "avg_weight": round(py_stats.mean(edge_weights), 4),
            "median_weight": round(py_stats.median(edge_weights), 4),
            "min_weight": round(min(edge_weights), 4),
            "max_weight": round(max_w, 4),
            "std_weight": round(py_stats.stdev(edge_weights), 4) if len(edge_weights) > 1 else 0,
            "weight_concentration": round(concentration, 4),
            **health_info,
            "very_high_count": sum(1 for w in edge_weights if w >= 0.8 * max_w),
            "high_count": sum(1 for w in edge_weights if 0.6 * max_w <= w < 0.8 * max_w),
            "medium_count": sum(1 for w in edge_weights if 0.3 * max_w <= w < 0.6 * max_w),
            "low_count": sum(1 for w in edge_weights if 0.1 * max_w <= w < 0.3 * max_w),
            "very_low_count": sum(1 for w in edge_weights if w < 0.1 * max_w),
            "top_edges": top_edges,
            "type_stats": type_stats,
            "computation_time_ms": round((time.time() - t0) * 1000, 2),
        }
