"""
Statistics calculations for graph metrics.
Ported from legacy StatisticsService.
"""
import statistics
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

from src.core.models import GraphData

def get_degree_distribution(graph_data: GraphData, node_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate degree distribution statistics.
    """
    start_time = time.time()
    
    in_degree: Dict[str, int] = {}
    out_degree: Dict[str, int] = {}
    component_types: Dict[str, str] = {}
    component_names: Dict[str, str] = {}
    
    for component in graph_data.components:
        comp_id = component.id
        in_degree[comp_id] = 0
        out_degree[comp_id] = 0
        component_types[comp_id] = component.type
        component_names[comp_id] = component.properties.get('name', comp_id)
    
    for edge in graph_data.edges:
        source = edge.source_id
        target = edge.target_id
        if source in out_degree:
            out_degree[source] += 1
        if target in in_degree:
            in_degree[target] += 1
    
    total_degree_all = {comp_id: in_degree[comp_id] + out_degree[comp_id] 
                       for comp_id in in_degree.keys()}
    
    all_degrees = list(total_degree_all.values())
    if all_degrees:
        all_mean = statistics.mean(all_degrees)
        all_std = statistics.stdev(all_degrees) if len(all_degrees) > 1 else 0
        hub_threshold = all_mean + 2 * all_std
    else:
        hub_threshold = 0
    
    if node_type:
        filtered_ids = {comp_id for comp_id, comp_type in component_types.items() 
                      if comp_type == node_type}
        in_degree = {k: v for k, v in in_degree.items() if k in filtered_ids}
        out_degree = {k: v for k, v in out_degree.items() if k in filtered_ids}
        component_types = {k: v for k, v in component_types.items() if k in filtered_ids}
        component_names = {k: v for k, v in component_names.items() if k in filtered_ids}
    
    total_degree = {comp_id: in_degree[comp_id] + out_degree[comp_id] 
                   for comp_id in in_degree.keys()}
    
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
    
    isolated_count = sum(1 for d in total_degree_all.values() if d == 0)
    
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

def get_connectivity_density(graph_data: GraphData, node_type: Optional[str] = None) -> Dict[str, Any]:
    """Calculate connectivity density statistics."""
    start_time = time.time()
    
    total_nodes = len(graph_data.components)
    total_edges = len(graph_data.edges)
    
    max_possible_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 0
    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    component_degrees = {c.id: 0 for c in graph_data.components}
    component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in graph_data.components}
    component_types = {c.id: c.type for c in graph_data.components}
    
    for edge in graph_data.edges:
        if edge.source_id in component_degrees: component_degrees[edge.source_id] += 1
        if edge.target_id in component_degrees: component_degrees[edge.target_id] += 1
    
    if node_type:
        filtered_ids = {cid for cid, ct in component_types.items() if ct == node_type}
        component_degrees = {k: v for k, v in component_degrees.items() if k in filtered_ids}
        component_info = {k: v for k, v in component_info.items() if k in filtered_ids}
    
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
    
    if density < 0.05: interpretation, category = "Sparse - Low coupling, good modularity", "sparse"
    elif density < 0.15: interpretation, category = "Moderate - Balanced connectivity", "moderate"
    elif density < 0.30: interpretation, category = "Dense - High coupling", "dense"
    else: interpretation, category = "Very Dense - Very high coupling", "very_dense"
    
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

def get_clustering_coefficient(graph_data: GraphData, node_type: Optional[str] = None) -> Dict[str, Any]:
    """Calculate clustering coefficient statistics."""
    start_time = time.time()
    
    neighbors = defaultdict(set)
    component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in graph_data.components}
    component_types = {c.id: c.type for c in graph_data.components}
    
    for edge in graph_data.edges:
        neighbors[edge.source_id].add(edge.target_id)
        neighbors[edge.target_id].add(edge.source_id)
    
    local_coefficients = {}
    for node in component_info:
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
    
    if node_type:
        filtered_ids = {cid for cid, ct in component_types.items() if ct == node_type}
        # We still use the global context for triangles, but filter the results
        local_coefficients_filtered = {k: v for k, v in local_coefficients.items() if k in filtered_ids}
    else:
        local_coefficients_filtered = local_coefficients
    
    highly_clustered = []
    sorted_nodes = sorted([(n, c) for n, c in local_coefficients_filtered.items() if c > 0],
                          key=lambda x: (x[1], len(neighbors[x[0]])), reverse=True)
    
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
    
    non_zero = [c for c in local_coefficients.values() if c > 0]
    std_coef = statistics.stdev(non_zero) if len(non_zero) > 1 else 0
    
    if avg_clustering < 0.1: interpretation, category = "Low clustering", "low"
    elif avg_clustering < 0.3: interpretation, category = "Moderate clustering", "moderate"
    else: interpretation, category = "High clustering", "high"
    
    return {
        "avg_clustering_coefficient": round(avg_clustering, 6),
        "interpretation": interpretation,
        "category": category,
        "highly_clustered_components": highly_clustered,
        "computation_time_ms": round((time.time() - start_time) * 1000, 2)
    }

def get_dependency_depth(graph_data: GraphData) -> Dict[str, Any]:
    """Calculate dependency depth statistics."""
    start_time = time.time()
    
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in graph_data.components}
    
    for edge in graph_data.edges:
        outgoing[edge.source_id].append(edge.target_id)
        incoming[edge.target_id].append(edge.source_id)
    
    node_depths = {}
    for node in component_info:
        max_d = 0
        visited = {node}
        queue = deque([(node, 0)])
        while queue:
            curr, d = queue.popleft()
            max_d = max(max_d, d)
            for neighbor in outgoing[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        node_depths[node] = max_d
        
    avg_depth = statistics.mean(node_depths.values()) if node_depths else 0
    max_depth = max(node_depths.values()) if node_depths else 0
    
    deepest = []
    sorted_nodes = sorted(node_depths.items(), key=lambda x: (x[1], len(outgoing[x[0]])), reverse=True)
    for n, d in sorted_nodes[:10]:
        info = component_info[n]
        deepest.append({
            "id": n, "name": info["name"], "type": info["type"], "depth": d,
            "dependencies": len(outgoing[n]), "dependents": len(incoming[n])
        })
    
    return {
        "avg_depth": round(avg_depth, 3),
        "max_depth": max_depth,
        "deepest_components": deepest,
        "computation_time_ms": round((time.time() - start_time) * 1000, 2)
    }

def get_component_isolation(graph_data: GraphData) -> Dict[str, Any]:
    """Calculate component isolation statistics."""
    start_time = time.time()
    
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in graph_data.components}
    
    for edge in graph_data.edges:
        outgoing[edge.source_id] += 1
        incoming[edge.target_id] += 1
        
    isolated, sources, sinks, bidirectional = [], [], [], []
    for cid in component_info:
        inc = incoming[cid]
        outc = outgoing[cid]
        data = {"id": cid, "name": component_info[cid]["name"], "type": component_info[cid]["type"], "in_degree": inc, "out_degree": outc}
        if inc == 0 and outc == 0: isolated.append(data)
        elif inc == 0: sources.append(data)
        elif outc == 0: sinks.append(data)
        else: bidirectional.append(data)
        
    total = len(component_info)
    return {
        "total_components": total,
        "isolated_count": len(isolated),
        "source_count": len(sources),
        "sink_count": len(sinks),
        "bidirectional_count": len(bidirectional),
        "isolated_percentage": round(len(isolated)/total*100, 2) if total > 0 else 0,
        "computation_time_ms": round((time.time() - start_time) * 1000, 2)
    }
