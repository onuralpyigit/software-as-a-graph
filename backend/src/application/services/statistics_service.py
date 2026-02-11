from typing import Dict, Any, List, Optional
import logging
import statistics
import time
from collections import defaultdict, deque

from src.core.graph_exporter import GraphExporter
from src.infrastructure.repositories.graph_query_repo import GraphQueryRepository

logger = logging.getLogger(__name__)

class StatisticsService:
    """
    Service for calculating graph statistics and metrics.
    Encapsulates logic for degree distribution, connectivity, clustering, etc.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password

    def _get_repository(self, exporter: GraphExporter) -> GraphQueryRepository:
        return GraphQueryRepository(exporter)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics including structural relationships."""
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            structural_counts = repo.get_structural_stats()
            
            # Count by type
            type_counts = {}
            for component in graph_data.components:
                comp_type = component.component_type
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            
            # Count by dependency type (derived DEPENDS_ON edges)
            dep_counts = {}
            for edge in graph_data.edges:
                dep_type = edge.dependency_type
                dep_counts[dep_type] = dep_counts.get(dep_type, 0) + 1
            
            return {
                "total_nodes": len(graph_data.components),
                "total_edges": len(graph_data.edges),
                "total_structural_edges": sum(structural_counts.values()),
                "node_counts": type_counts,
                "edge_counts": dep_counts,
                "structural_edge_counts": structural_counts
            }

    def get_degree_distribution(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get fast degree distribution statistics.
        Computes in-degree, out-degree, hub nodes, and isolation.
        """
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            
            # Calculate degree for ALL components first
        in_degree: Dict[str, int] = {}
        out_degree: Dict[str, int] = {}
        component_types: Dict[str, str] = {}
        component_names: Dict[str, str] = {}
        
        # Initialize ALL components with 0 degree
        for component in graph_data.components:
            comp_id = component.id
            in_degree[comp_id] = 0
            out_degree[comp_id] = 0
            component_types[comp_id] = component.component_type
            component_names[comp_id] = component.properties.get('name', comp_id)
        
        # Count degrees from ALL edges in the system
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            out_degree[source] = out_degree.get(source, 0) + 1
            in_degree[target] = in_degree.get(target, 0) + 1
        
        # Calculate total degree for ALL components
        total_degree_all = {comp_id: in_degree[comp_id] + out_degree[comp_id] 
                           for comp_id in in_degree.keys()}
        
        # Calculate hub threshold based on ALL components (before filtering)
        all_degrees = list(total_degree_all.values())
        if all_degrees:
            all_mean = statistics.mean(all_degrees)
            all_std = statistics.stdev(all_degrees) if len(all_degrees) > 1 else 0
            hub_threshold = all_mean + 2 * all_std
        else:
            hub_threshold = 0
        
        # Now filter to only the requested node type for display
        if node_type:
            filtered_ids = {comp_id for comp_id, comp_type in component_types.items() 
                          if comp_type == node_type}
            in_degree = {k: v for k, v in in_degree.items() if k in filtered_ids}
            out_degree = {k: v for k, v in out_degree.items() if k in filtered_ids}
            component_types = {k: v for k, v in component_types.items() if k in filtered_ids}
            component_names = {k: v for k, v in component_names.items() if k in filtered_ids}
        
        # Calculate total degree for filtered components
        total_degree = {comp_id: in_degree[comp_id] + out_degree[comp_id] 
                       for comp_id in in_degree.keys()}
        
        # Helper function to compute stats
        def compute_stats(degree_dict: Dict[str, int]) -> Dict[str, float]:
            values = list(degree_dict.values())
            if not values:
                return {"mean": 0, "median": 0, "max": 0, "min": 0, "std": 0}
            
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        in_stats = compute_stats(in_degree)
        out_stats = compute_stats(out_degree)
        total_stats = compute_stats(total_degree)
        
        # Identify hub nodes using the system-wide threshold
        hub_nodes = []
        for comp_id, degree in total_degree.items():
            if degree > hub_threshold:
                hub_nodes.append({
                    "id": comp_id,
                    "name": component_names.get(comp_id, comp_id),
                    "degree": degree,
                    "type": component_types.get(comp_id, "Unknown")
                })
        
        hub_nodes.sort(key=lambda x: x["degree"], reverse=True)
        
        # Count isolated nodes
        isolated_count = 0
        for comp_id in total_degree.keys():
            if total_degree_all.get(comp_id, 0) == 0:
                isolated_count += 1
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "in_degree": in_stats,
            "out_degree": out_stats,
            "total_degree": total_stats,
            "hub_nodes": hub_nodes,
            "isolated_nodes": isolated_count,
            "total_nodes": len(total_degree),
            "hub_threshold": round(hub_threshold, 2),
            "computation_time_ms": round(computation_time, 2)
        }

    def get_connectivity_density(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """Get connectivity density statistics."""
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            
            total_nodes = len(graph_data.components)
        total_edges = len(graph_data.edges)
        
        max_possible_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 0
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate degree for ALL components
        component_degrees = {}
        component_info = {}
        component_types = {}
        
        for component in graph_data.components:
            comp_id = component.id
            component_degrees[comp_id] = 0
            component_types[comp_id] = component.component_type
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            if source in component_degrees:
                component_degrees[source] += 1
            if target in component_degrees:
                component_degrees[target] += 1
        
        # Filter if requested
        if node_type:
            filtered_ids = {comp_id for comp_id, comp_type in component_types.items() 
                          if comp_type == node_type}
            component_degrees = {k: v for k, v in component_degrees.items() if k in filtered_ids}
            component_info = {k: v for k, v in component_info.items() if k in filtered_ids}
        
        # Top 10 dense components
        most_dense_components = []
        sorted_components = sorted(component_degrees.items(), key=lambda x: x[1], reverse=True)
        
        for comp_id, degree in sorted_components[:10]:
            if degree > 0:
                info = component_info.get(comp_id, {})
                most_dense_components.append({
                    "id": comp_id,
                    "name": info.get("name", comp_id),
                    "type": info.get("type", "Unknown"),
                    "degree": degree,
                    "density_contribution": round((degree / (2 * total_edges)) * 100, 2) if total_edges > 0 else 0
                })
        
        if density < 0.05:
            interpretation = "Sparse - Low coupling, good modularity"
            category = "sparse"
        elif density < 0.15:
            interpretation = "Moderate - Balanced connectivity"
            category = "moderate"
        elif density < 0.30:
            interpretation = "Dense - High coupling"
            category = "dense"
        else:
            interpretation = "Very Dense - Very high coupling"
            category = "very_dense"
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "density": round(density, 6),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "max_possible_edges": max_possible_edges,
            "interpretation": interpretation,
            "category": category,
            "most_dense_components": most_dense_components,
            "computation_time_ms": round(computation_time, 2)
        }
    
    def get_clustering_coefficient(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """Get clustering coefficient statistics."""
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            
            neighbors = defaultdict(set)
        component_info = {}
        component_types = {}
        
        for component in graph_data.components:
            comp_id = component.id
            component_types[comp_id] = component.component_type
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            neighbors[source].add(target)
            neighbors[target].add(source)
        
        local_coefficients = {}
        for node in component_info.keys():
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
        
        avg_clustering = sum(local_coefficients.values()) / len(local_coefficients) if local_coefficients else 0.0
        
        # Filter if requested
        if node_type:
            filtered_ids = {comp_id for comp_id, comp_type in component_types.items() 
                          if comp_type == node_type}
            local_coefficients_filtered = {k: v for k, v in local_coefficients.items() if k in filtered_ids}
            component_info_filtered = {k: v for k, v in component_info.items() if k in filtered_ids}
        else:
            local_coefficients_filtered = local_coefficients
            component_info_filtered = component_info
        
        highly_clustered = []
        sorted_nodes = sorted(
            [(node, coef) for node, coef in local_coefficients_filtered.items() if coef > 0],
            key=lambda x: (x[1], len(neighbors[x[0]])),
            reverse=True
        )
        
        for node, coef in sorted_nodes[:10]:
            info = component_info.get(node, {})
            highly_clustered.append({
                "id": node,
                "name": info.get("name", node),
                "type": info.get("type", "Unknown"),
                "coefficient": round(coef, 4),
                "degree": len(neighbors[node]),
                "triangles": int(coef * len(neighbors[node]) * (len(neighbors[node]) - 1) / 2)
            })
            
        non_zero_coefficients = [c for c in local_coefficients.values() if c > 0]
        if non_zero_coefficients:
            max_coef = max(local_coefficients.values())
            min_coef = min([c for c in local_coefficients.values() if c > 0], default=0)
            median_coef = statistics.median(non_zero_coefficients)
            std_coef = statistics.stdev(non_zero_coefficients) if len(non_zero_coefficients) > 1 else 0
        else:
            max_coef = min_coef = median_coef = std_coef = 0.0
            
        zero_clustering = sum(1 for c in local_coefficients.values() if c == 0)
        low_clustering = sum(1 for c in local_coefficients.values() if 0 < c < 0.3)
        medium_clustering = sum(1 for c in local_coefficients.values() if 0.3 <= c < 0.7)
        high_clustering = sum(1 for c in local_coefficients.values() if c >= 0.7)
        
        if avg_clustering < 0.1:
            interpretation = "Low clustering - components operate independently"
            category = "low"
        elif avg_clustering < 0.3:
            interpretation = "Moderate clustering - some component grouping"
            category = "moderate"
        elif avg_clustering < 0.6:
            interpretation = "High clustering - strong component grouping"
            category = "high"
        else:
            interpretation = "Very high clustering - tightly connected groups"
            category = "very_high"
            
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "avg_clustering_coefficient": round(avg_clustering, 6),
            "max_coefficient": round(max_coef, 6),
            "min_coefficient": round(min_coef, 6),
            "median_coefficient": round(median_coef, 6),
            "std_coefficient": round(std_coef, 6),
            "interpretation": interpretation,
            "category": category,
            "zero_clustering_count": zero_clustering,
            "low_clustering_count": low_clustering,
            "medium_clustering_count": medium_clustering,
            "high_clustering_count": high_clustering,
            "total_nodes": len(component_info),
            "highly_clustered_components": highly_clustered,
            "computation_time_ms": round(computation_time, 2)
        }

    def get_dependency_depth(self) -> Dict[str, Any]:
        """Get dependency depth statistics."""
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            
            outgoing = defaultdict(list)
        incoming = defaultdict(list)
        component_info = {}
        
        for component in graph_data.components:
            comp_id = component.id
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            outgoing[source].append(target)
            incoming[target].append(source)
        
        node_depths = {}
        for node in component_info.keys():
            max_depth = 0
            visited = {node}
            queue = deque([(node, 0)])
            while queue:
                current, depth = queue.popleft()
                max_depth = max(max_depth, depth)
                for neighbor in outgoing[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            node_depths[node] = max_depth
            
        if node_depths:
            depths = list(node_depths.values())
            avg_depth = statistics.mean(depths)
            max_depth = max(depths)
            min_depth = min(depths)
            median_depth = statistics.median(depths)
            std_depth = statistics.stdev(depths) if len(depths) > 1 else 0
        else:
            avg_depth = max_depth = min_depth = median_depth = std_depth = 0
        
        deepest_components = []
        sorted_nodes = sorted(
            node_depths.items(),
            key=lambda x: (x[1], len(outgoing[x[0]])),
            reverse=True
        )
        
        for node, depth in sorted_nodes[:10]:
            info = component_info.get(node, {})
            deepest_components.append({
                "id": node,
                "name": info.get("name", node),
                "type": info.get("type", "Unknown"),
                "depth": depth,
                "dependencies": len(outgoing[node]),
                "dependents": len(incoming[node])
            })
            
        depth_distribution = defaultdict(int)
        for depth in depths:
            depth_distribution[depth] += 1
            
        shallow = sum(1 for d in depths if d == 0)
        low_depth = sum(1 for d in depths if 0 < d <= 2)
        medium_depth = sum(1 for d in depths if 2 < d <= 5)
        high_depth = sum(1 for d in depths if d > 5)

        root_nodes = []
        leaf_nodes = []
        isolated_nodes = []
        
        for node in component_info.keys():
            out_count = len(outgoing[node])
            in_count = len(incoming[node])
            info = component_info[node]
            
            if out_count == 0 and in_count == 0:
                isolated_nodes.append({
                    "id": node,
                    "name": info.get("name", node),
                    "type": info.get("type", "Unknown"),
                    "dependencies": 0,
                    "dependents": 0
                })
            elif out_count == 0 and in_count > 0:
                leaf_nodes.append({
                    "id": node,
                    "name": info.get("name", node),
                    "type": info.get("type", "Unknown"),
                    "dependents": in_count
                })
            elif in_count == 0 and out_count > 0:
                root_nodes.append({
                    "id": node,
                    "name": info.get("name", node),
                    "type": info.get("type", "Unknown"),
                    "dependencies": out_count
                })
        
        if len(root_nodes) == 0:
            candidates = []
            for node in component_info.keys():
                in_count = len(incoming[node])
                out_count = len(outgoing[node])
                if out_count > 0:
                    info = component_info[node]
                    candidates.append({
                        "id": node,
                        "name": info.get("name", node),
                        "type": info.get("type", "Unknown"),
                        "dependencies": out_count,
                        "dependents": in_count
                    })
            root_nodes = sorted(candidates, key=lambda x: (x["dependents"], -x["dependencies"]))[:10]
        else:
            root_nodes = sorted(root_nodes, key=lambda x: x["dependencies"], reverse=True)[:10]
            
        leaf_nodes = sorted(leaf_nodes, key=lambda x: x["dependents"], reverse=True)[:10]
        
        if max_depth == 0:
            interpretation = "No dependencies - completely isolated components"
            category = "isolated"
        elif max_depth <= 2:
            interpretation = "Shallow dependencies - simple, flat architecture"
            category = "shallow"
        elif max_depth <= 5:
            interpretation = "Moderate depth - balanced architecture"
            category = "moderate"
        elif max_depth <= 10:
            interpretation = "Deep dependencies - complex architecture"
            category = "deep"
        else:
            interpretation = "Very deep dependencies - highly complex"
            category = "very_deep"
            
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "avg_depth": round(avg_depth, 3),
            "max_depth": max_depth,
            "min_depth": min_depth,
            "median_depth": round(median_depth, 3),
            "std_depth": round(std_depth, 3),
            "interpretation": interpretation,
            "category": category,
            "shallow_count": shallow,
            "low_depth_count": low_depth,
            "medium_depth_count": medium_depth,
            "high_depth_count": high_depth,
            "total_nodes": len(component_info),
            "deepest_components": deepest_components,
            "root_nodes": root_nodes,
            "leaf_nodes": leaf_nodes,
            "depth_distribution": dict(depth_distribution),
            "computation_time_ms": round(computation_time, 2)
        }

    def get_component_isolation(self) -> Dict[str, Any]:
        """Get component isolation statistics."""
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            
            incoming = defaultdict(set)
        outgoing = defaultdict(set)
        component_info = {}
        
        for component in graph_data.components:
            comp_id = component.id
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            outgoing[source].add(target)
            incoming[target].add(source)
            
        isolated_components = []
        source_components = []
        sink_components = []
        bidirectional_components = []
        
        for comp_id in component_info.keys():
            in_count = len(incoming[comp_id])
            out_count = len(outgoing[comp_id])
            info = component_info[comp_id]
            comp_data = {
                "id": comp_id,
                "name": info.get("name", comp_id),
                "type": info.get("type", "Unknown"),
                "in_degree": in_count,
                "out_degree": out_count
            }
            
            if in_count == 0 and out_count == 0:
                isolated_components.append(comp_data)
            elif in_count == 0 and out_count > 0:
                source_components.append(comp_data)
            elif in_count > 0 and out_count == 0:
                sink_components.append(comp_data)
            else:
                bidirectional_components.append(comp_data)
                
        source_components.sort(key=lambda x: x["out_degree"], reverse=True)
        sink_components.sort(key=lambda x: x["in_degree"], reverse=True)
        bidirectional_components.sort(key=lambda x: x["in_degree"] + x["out_degree"], reverse=True)
        
        total = len(component_info)
        isolated_pct = (len(isolated_components) / total * 100) if total > 0 else 0
        source_pct = (len(source_components) / total * 100) if total > 0 else 0
        sink_pct = (len(sink_components) / total * 100) if total > 0 else 0
        bidirectional_pct = (len(bidirectional_components) / total * 100) if total > 0 else 0
        
        if isolated_pct > 50:
            interpretation = "Highly fragmented"
            category = "fragmented"
            health = "poor"
        elif source_pct > 50:
            interpretation = "Top-heavy architecture"
            category = "top_heavy"
            health = "moderate"
        elif sink_pct > 50:
            interpretation = "Foundation-heavy"
            category = "foundation_heavy"
            health = "moderate"
        elif bidirectional_pct > 70:
            interpretation = "Well-connected architecture"
            category = "connected"
            health = "good"
        elif isolated_pct > 20:
            interpretation = "Significant fragmentation"
            category = "partially_fragmented"
            health = "fair"
        else:
            interpretation = "Balanced architecture"
            category = "balanced"
            health = "good"
            
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "total_components": total,
            "isolated_count": len(isolated_components),
            "source_count": len(source_components),
            "sink_count": len(sink_components),
            "bidirectional_count": len(bidirectional_components),
            "isolated_percentage": round(isolated_pct, 2),
            "source_percentage": round(source_pct, 2),
            "sink_percentage": round(sink_pct, 2),
            "bidirectional_percentage": round(bidirectional_pct, 2),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "isolated_components": isolated_components[:20],
            "top_sources": source_components[:10],
            "top_sinks": sink_components[:10],
            "top_bidirectional": bidirectional_components[:10],
            "computation_time_ms": round(computation_time, 2)
        }

    def get_message_flow_patterns(self) -> Dict[str, Any]:
        """Get message flow pattern statistics."""
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            publishes, subscribes, topic_brokers, all_apps = repo.get_pub_sub_data()
            
        topic_stats = defaultdict(lambda: {
            "publishers": set(),
            "subscribers": set(),
            "brokers": set(),
            "name": "",
            "pub_weight": 0,
            "sub_weight": 0
        })
        
        for pub in publishes:
            topic_id = pub["topic_id"]
            topic_stats[topic_id]["publishers"].add(pub["app_id"])
            topic_stats[topic_id]["name"] = pub["topic_name"]
            topic_stats[topic_id]["pub_weight"] += pub["weight"]
        
        for sub in subscribes:
            topic_id = sub["topic_id"]
            topic_stats[topic_id]["subscribers"].add(sub["app_id"])
            topic_stats[topic_id]["name"] = sub["topic_name"]
            topic_stats[topic_id]["sub_weight"] += sub["weight"]
            
        for tb in topic_brokers:
            topic_id = tb["topic_id"]
            topic_stats[topic_id]["brokers"].add(tb["broker_id"])
            if not topic_stats[topic_id]["name"]:
                topic_stats[topic_id]["name"] = tb["topic_name"]
                
        broker_stats = defaultdict(lambda: {
            "topics": set(),
            "total_publishers": 0,
            "total_subscribers": 0,
            "name": ""
        })
        
        for tb in topic_brokers:
            broker_id = tb["broker_id"]
            topic_id = tb["topic_id"]
            broker_stats[broker_id]["topics"].add(topic_id)
            broker_stats[broker_id]["name"] = tb["broker_name"]
            if topic_id in topic_stats:
                broker_stats[broker_id]["total_publishers"] += len(topic_stats[topic_id]["publishers"])
                broker_stats[broker_id]["total_subscribers"] += len(topic_stats[topic_id]["subscribers"])
                
        app_pub_count = defaultdict(int)
        app_sub_count = defaultdict(int)
        app_names = {}
        for app in all_apps:
            app_names[app["id"]] = app["name"]
        for pub in publishes:
            app_pub_count[pub["app_id"]] += 1
        for sub in subscribes:
            app_sub_count[sub["app_id"]] += 1
            
        hot_topics = []
        for topic_id, stats in topic_stats.items():
            total_activity = len(stats["publishers"]) + len(stats["subscribers"])
            topic_name = stats["name"]
            if isinstance(topic_name, dict):
                topic_name = topic_name.get("name", topic_id)
            hot_topics.append({
                "id": str(topic_id),
                "name": str(topic_name),
                "publishers": len(stats["publishers"]),
                "subscribers": len(stats["subscribers"]),
                "total_activity": total_activity,
                "brokers": len(stats["brokers"]),
                "pub_weight": round(stats["pub_weight"], 2),
                "sub_weight": round(stats["sub_weight"], 2)
            })
        hot_topics.sort(key=lambda x: x["total_activity"], reverse=True)
        
        broker_utilization = []
        for broker_id, stats in broker_stats.items():
            total_load = stats["total_publishers"] + stats["total_subscribers"]
            broker_name = stats["name"]
            if isinstance(broker_name, dict):
                broker_name = broker_name.get("name", broker_id)
            broker_utilization.append({
                "id": str(broker_id),
                "name": str(broker_name),
                "topics": len(stats["topics"]),
                "publishers": stats["total_publishers"],
                "subscribers": stats["total_subscribers"],
                "total_load": total_load
            })
        broker_utilization.sort(key=lambda x: x["total_load"], reverse=True)
        
        isolated_apps = []
        for app in all_apps:
            app_id = app["id"]
            if app_pub_count[app_id] == 0 and app_sub_count[app_id] == 0:
                app_name = app.get("name", app_id)
                if isinstance(app_name, dict):
                    app_name = app_name.get("name", app_id)
                isolated_apps.append({
                    "id": str(app_id),
                    "name": str(app_name)
                })
                
        top_publishers = []
        for app_id, count in app_pub_count.items():
            if count > 0:
                app_name = app_names.get(app_id, app_id)
                if isinstance(app_name, dict):
                    app_name = app_name.get("name", app_id)
                top_publishers.append({
                    "id": str(app_id),
                    "name": str(app_name),
                    "topics": count,
                    "subscriptions": app_sub_count[app_id]
                })
        top_publishers.sort(key=lambda x: x["topics"], reverse=True)
        
        top_subscribers = []
        for app_id, count in app_sub_count.items():
            if count > 0:
                app_name = app_names.get(app_id, app_id)
                if isinstance(app_name, dict):
                    app_name = app_name.get("name", app_id)
                top_subscribers.append({
                    "id": str(app_id),
                    "name": str(app_name),
                    "topics": count,
                    "publications": app_pub_count[app_id]
                })
        top_subscribers.sort(key=lambda x: x["topics"], reverse=True)
        
        total_topics = len(topic_stats)
        total_brokers = len(broker_stats)
        total_apps = len(all_apps)
        active_apps = len(set(app_pub_count.keys()) | set(app_sub_count.keys()))
        isolated_count = len(isolated_apps)
        
        avg_publishers_per_topic = sum(len(s["publishers"]) for s in topic_stats.values()) / total_topics if total_topics > 0 else 0
        avg_subscribers_per_topic = sum(len(s["subscribers"]) for s in topic_stats.values()) / total_topics if total_topics > 0 else 0
        avg_topics_per_broker = sum(len(s["topics"]) for s in broker_stats.values()) / total_brokers if total_brokers > 0 else 0
        
        if isolated_count / total_apps > 0.3 if total_apps > 0 else False:
            interpretation = "High isolation"
            category = "isolated"
            health = "poor"
        elif avg_publishers_per_topic < 1.5 and avg_subscribers_per_topic < 1.5:
            interpretation = "Sparse communication"
            category = "sparse"
            health = "moderate"
        elif len(hot_topics) > 0 and hot_topics[0]["total_activity"] > total_apps * 0.3:
            interpretation = "Bottleneck detected"
            category = "bottleneck"
            health = "fair"
        else:
            interpretation = "Balanced communication"
            category = "balanced"
            health = "good"
            
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "total_topics": total_topics,
            "total_brokers": total_brokers,
            "total_applications": total_apps,
            "active_applications": active_apps,
            "isolated_applications": isolated_count,
            "avg_publishers_per_topic": round(avg_publishers_per_topic, 2),
            "avg_subscribers_per_topic": round(avg_subscribers_per_topic, 2),
            "avg_topics_per_broker": round(avg_topics_per_broker, 2),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "hot_topics": hot_topics[:10],
            "broker_utilization": broker_utilization[:10],
            "isolated_applications": isolated_apps[:20],
            "top_publishers": top_publishers[:10],
            "top_subscribers": top_subscribers[:10],
            "computation_time_ms": round(computation_time, 2)
        }

    def get_component_redundancy(self) -> Dict[str, Any]:
        """Get component redundancy statistics."""
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
            
            adjacency = defaultdict(set)
        reverse_adjacency = defaultdict(set)
        component_info = {}
        
        for component in graph_data.components:
            comp_id = component.id
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            adjacency[source].add(target)
            reverse_adjacency[target].add(source)
            
        def find_articulation_points():
            visited = set()
            disc = {}
            low = {}
            parent = {}
            ap = set()
            time_counter = [0]
            
            def dfs(u):
                children = 0
                visited.add(u)
                disc[u] = low[u] = time_counter[0]
                time_counter[0] += 1
                
                for v in adjacency[u]:
                    if v not in visited:
                        children += 1
                        parent[v] = u
                        dfs(v)
                        low[u] = min(low[u], low[v])
                        if parent.get(u) is None and children > 1:
                            ap.add(u)
                        if parent.get(u) is not None and low[v] >= disc[u]:
                            ap.add(u)
                    elif v != parent.get(u):
                        low[u] = min(low[u], disc[v])
            
            for node in component_info.keys():
                if node not in visited:
                    dfs(node)
            return ap
            
        articulation_points = find_articulation_points()
        
        def identify_bridge_components():
            bridge_scores = defaultdict(int)
            sample_nodes = list(component_info.keys())[:min(50, len(component_info))]
            for start in sample_nodes:
                visited = set([start])
                queue = deque([(start, [start])])
                paths_through = defaultdict(int)
                while queue:
                    node, path = queue.popleft()
                    if len(path) > 10:
                        continue
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [neighbor]
                            queue.append((neighbor, new_path))
                            for comp in new_path[1:-1]:
                                paths_through[comp] += 1
                for comp, count in paths_through.items():
                    bridge_scores[comp] += count
            sorted_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_bridges[:20]
            
        bridge_components = identify_bridge_components()
        
        total_components = len(component_info)
        spof_count = len(articulation_points)
        spof_percentage = (spof_count / total_components * 100) if total_components > 0 else 0
        
        redundant_count = 0
        for comp_id in component_info.keys():
            if len(adjacency[comp_id]) > 1 or len(reverse_adjacency[comp_id]) > 1:
                redundant_count += 1
        redundancy_percentage = (redundant_count / total_components * 100) if total_components > 0 else 0
        
        if spof_percentage > 20:
            interpretation = "High risk - many single points of failure"
            resilience_score = "low"
        elif spof_percentage > 10:
            interpretation = "Moderate risk - some single points of failure"
            resilience_score = "moderate"
        elif redundancy_percentage > 70:
            interpretation = "High resilience - good redundancy"
            resilience_score = "high"
        else:
            interpretation = "Balanced resilience"
            resilience_score = "medium"
            
        computation_time = (time.time() - start_time) * 1000
        
        formatted_spofs = []
        for spof in list(articulation_points)[:20]:
            info = component_info.get(spof, {})
            formatted_spofs.append({
                "id": spof,
                "name": info.get("name", spof),
                "type": info.get("type", "Unknown")
            })
            
        formatted_bridges = []
        for bridge, score in bridge_components:
            info = component_info.get(bridge, {})
            formatted_bridges.append({
                "id": bridge,
                "name": info.get("name", bridge),
                "type": info.get("type", "Unknown"),
                "paths_traversed": score
            })
            
        return {
            "success": True,
            "stats": {
                "total_components": total_components,
                "spof_count": spof_count,
                "spof_percentage": round(spof_percentage, 2),
                "redundant_count": redundant_count,
                "redundancy_percentage": round(redundancy_percentage, 2),
                "resilience_score": resilience_score,
                "interpretation": interpretation,
                "potential_spofs": formatted_spofs,
                "bridge_components": formatted_bridges
            },
            "computation_time_ms": round(computation_time, 2)
        }

    def get_node_weight_distribution(self) -> Dict[str, Any]:
        """Get node weight distribution statistics."""
        import statistics
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
        
        # Collect weights by component type
        weights_by_type = {}
        all_weights = []
        component_details = []
        
        for component in graph_data.components:
            comp_id = component.id
            comp_type = component.component_type
            weight = component.weight
            name = component.properties.get('name', comp_id)
            
            all_weights.append(weight)
            
            if comp_type not in weights_by_type:
                weights_by_type[comp_type] = []
            weights_by_type[comp_type].append(weight)
            
            component_details.append({
                "id": comp_id,
                "name": name,
                "type": comp_type,
                "weight": round(weight, 4)
            })
        
        # Sort components by weight
        component_details.sort(key=lambda x: x["weight"], reverse=True)
        
        # Overall statistics
        if all_weights:
            total_weight = sum(all_weights)
            avg_weight = statistics.mean(all_weights)
            median_weight = statistics.median(all_weights)
            std_weight = statistics.stdev(all_weights) if len(all_weights) > 1 else 0
            min_weight = min(all_weights)
            max_weight = max(all_weights)
        else:
            total_weight = avg_weight = median_weight = std_weight = min_weight = max_weight = 0
        
        # Statistics by type
        type_stats = {}
        for comp_type, weights in weights_by_type.items():
            if weights:
                type_stats[comp_type] = {
                    "count": len(weights),
                    "total_weight": round(sum(weights), 4),
                    "avg_weight": round(statistics.mean(weights), 4),
                    "median_weight": round(statistics.median(weights), 4),
                    "min_weight": round(min(weights), 4),
                    "max_weight": round(max(weights), 4),
                    "std_weight": round(statistics.stdev(weights), 4) if len(weights) > 1 else 0
                }
        
        # Weight distribution categories
        very_high_threshold = avg_weight + 2 * std_weight if std_weight > 0 else avg_weight * 2
        high_threshold = avg_weight + std_weight if std_weight > 0 else avg_weight * 1.5
        low_threshold = avg_weight - std_weight if std_weight > 0 else avg_weight * 0.5
        very_low_threshold = avg_weight - 2 * std_weight if std_weight > 0 else avg_weight * 0.25
        
        very_high_count = sum(1 for w in all_weights if w >= very_high_threshold)
        high_count = sum(1 for w in all_weights if high_threshold <= w < very_high_threshold)
        medium_count = sum(1 for w in all_weights if low_threshold <= w < high_threshold)
        low_count = sum(1 for w in all_weights if very_low_threshold <= w < low_threshold)
        very_low_count = sum(1 for w in all_weights if w < very_low_threshold)
        
        # Top weighted components (top 15)
        top_components = component_details[:15]
        
        # Calculate weight concentration (what % of total weight is in top 20% of components)
        sorted_weights = sorted(all_weights, reverse=True)
        top_20_percent_count = max(1, int(len(sorted_weights) * 0.2))
        top_20_weight = sum(sorted_weights[:top_20_percent_count])
        weight_concentration = (top_20_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Interpretation
        if weight_concentration > 80:
            interpretation = "Highly concentrated - small number of critical components hold most importance"
            category = "concentrated"
            health = "moderate"
        elif weight_concentration > 60:
            interpretation = "Moderately concentrated - some components significantly more important than others"
            category = "moderate"
            health = "fair"
        elif weight_concentration > 40:
            interpretation = "Balanced distribution - importance spread across many components"
            category = "balanced"
            health = "good"
        else:
            interpretation = "Evenly distributed - most components have similar importance levels"
            category = "even"
            health = "good"
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "total_components": len(graph_data.components),
            "total_weight": round(total_weight, 4),
            "avg_weight": round(avg_weight, 4),
            "median_weight": round(median_weight, 4),
            "min_weight": round(min_weight, 4),
            "max_weight": round(max_weight, 4),
            "std_weight": round(std_weight, 4),
            "weight_concentration": round(weight_concentration, 2),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "very_high_count": very_high_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "very_low_count": very_low_count,
            "top_components": top_components,
            "type_stats": type_stats,
            "computation_time_ms": round(computation_time, 2)
        }

    def get_edge_weight_distribution(self) -> Dict[str, Any]:
        """Get edge weight distribution statistics."""
        import statistics
        start_time = time.time()
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = self._get_repository(exporter)
            graph_data = repo.get_graph_data()
        
        # Collect weights by dependency type
        weights_by_type = {}
        all_weights = []
        edge_details = []
        
        # Build component names map for better display
        component_names = {}
        for component in graph_data.components:
            component_names[component.id] = component.properties.get('name', component.id)
        
        for edge in graph_data.edges:
            source_id = edge.source_id
            target_id = edge.target_id
            dep_type = edge.dependency_type
            weight = edge.weight
            
            all_weights.append(weight)
            
            if dep_type not in weights_by_type:
                weights_by_type[dep_type] = []
            weights_by_type[dep_type].append(weight)
            
            edge_details.append({
                "source": source_id,
                "target": target_id,
                "source_name": component_names.get(source_id, source_id),
                "target_name": component_names.get(target_id, target_id),
                "type": dep_type,
                "weight": round(weight, 4)
            })
        
        # Sort edges by weight
        edge_details.sort(key=lambda x: x["weight"], reverse=True)
        
        # Overall statistics
        if all_weights:
            total_weight = sum(all_weights)
            avg_weight = statistics.mean(all_weights)
            median_weight = statistics.median(all_weights)
            std_weight = statistics.stdev(all_weights) if len(all_weights) > 1 else 0
            min_weight = min(all_weights)
            max_weight = max(all_weights)
        else:
            total_weight = avg_weight = median_weight = std_weight = min_weight = max_weight = 0
        
        # Statistics by dependency type
        type_stats = {}
        for dep_type, weights in weights_by_type.items():
            if weights:
                type_stats[dep_type] = {
                    "count": len(weights),
                    "total_weight": round(sum(weights), 4),
                    "avg_weight": round(statistics.mean(weights), 4),
                    "median_weight": round(statistics.median(weights), 4),
                    "min_weight": round(min(weights), 4),
                    "max_weight": round(max(weights), 4),
                    "std_weight": round(statistics.stdev(weights), 4) if len(weights) > 1 else 0
                }
        
        # Weight distribution categories
        very_high_threshold = avg_weight + 2 * std_weight if std_weight > 0 else avg_weight * 2
        high_threshold = avg_weight + std_weight if std_weight > 0 else avg_weight * 1.5
        low_threshold = avg_weight - std_weight if std_weight > 0 else avg_weight * 0.5
        very_low_threshold = avg_weight - 2 * std_weight if std_weight > 0 else avg_weight * 0.25
        
        very_high_count = sum(1 for w in all_weights if w >= very_high_threshold)
        high_count = sum(1 for w in all_weights if high_threshold <= w < very_high_threshold)
        medium_count = sum(1 for w in all_weights if low_threshold <= w < high_threshold)
        low_count = sum(1 for w in all_weights if very_low_threshold <= w < low_threshold)
        very_low_count = sum(1 for w in all_weights if w < very_low_threshold)
        
        # Top weighted edges (top 15)
        top_edges = edge_details[:15]
        
        # Calculate weight concentration (what % of total weight is in top 20% of edges)
        sorted_weights = sorted(all_weights, reverse=True)
        top_20_percent_count = max(1, int(len(sorted_weights) * 0.2))
        top_20_weight = sum(sorted_weights[:top_20_percent_count])
        weight_concentration = (top_20_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Interpretation
        if weight_concentration > 80:
            interpretation = "Highly concentrated - few critical dependencies carry most weight"
            category = "concentrated"
            health = "moderate"
        elif weight_concentration > 60:
            interpretation = "Moderately concentrated - some dependencies significantly stronger than others"
            category = "moderate"
            health = "fair"
        elif weight_concentration > 40:
            interpretation = "Balanced distribution - dependency importance spread across many edges"
            category = "balanced"
            health = "good"
        else:
            interpretation = "Evenly distributed - most dependencies have similar importance levels"
            category = "even"
            health = "good"
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "total_edges": len(graph_data.edges),
            "total_weight": round(total_weight, 4),
            "avg_weight": round(avg_weight, 4),
            "median_weight": round(median_weight, 4),
            "min_weight": round(min_weight, 4),
            "max_weight": round(max_weight, 4),
            "std_weight": round(std_weight, 4),
            "weight_concentration": round(weight_concentration, 2),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "very_high_count": very_high_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "very_low_count": very_low_count,
            "top_edges": top_edges,
            "type_stats": type_stats,
            "computation_time_ms": round(computation_time, 2)
        }
