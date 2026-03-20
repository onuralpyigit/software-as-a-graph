
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def format_statistics_response(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a stats dict in a success envelope."""
    return {
        "success": True,
        "stats": stats,
        "computation_time_ms": stats.get("computation_time_ms", 0),
    }

def format_empty_statistics_response(defaults: Dict[str, Any], error: str = "Feature temporarily unavailable") -> Dict[str, Any]:
    """Return a graceful-degradation response with zero-value defaults."""
    return {
        "success": False, 
        "error": error, 
        "stats": defaults, 
        "computation_time_ms": 0
    }

# ── Default Stat Factories ──────────────────────────────────────────────

def get_degree_distribution_defaults() -> Dict[str, Any]:
    return {
        "in_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
        "out_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
        "total_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
        "hub_nodes": [], "isolated_nodes": 0, "total_nodes": 0,
        "hub_threshold": 0,
    }

def get_connectivity_density_defaults() -> Dict[str, Any]:
    return {
        "density": 0, "total_nodes": 0, "total_edges": 0,
        "max_possible_edges": 0, "interpretation": "Data unavailable",
        "category": "unavailable", "most_dense_components": [],
    }

def get_clustering_coefficient_defaults() -> Dict[str, Any]:
    return {
        "avg_clustering_coefficient": 0, "global_clustering": 0,
        "average_clustering": 0, "max_coefficient": 0, "median_coefficient": 0,
        "min_coefficient": 0, "std_coefficient": 0,
        "high_clustering_count": 0, "medium_clustering_count": 0,
        "low_clustering_count": 0, "zero_clustering_count": 0,
        "total_nodes": 0, "high_clustering_nodes": [], "zero_clustering_nodes": [],
    }

def get_dependency_depth_defaults() -> Dict[str, Any]:
    return {
        "max_depth": 0, "avg_depth": 0, "median_depth": 0, "min_depth": 0,
        "std_depth": 0, "interpretation": "Data unavailable", "category": "unavailable",
        "depth_distribution": {}, "shallow_count": 0, "low_depth_count": 0,
        "medium_depth_count": 0, "high_depth_count": 0, "total_nodes": 0,
        "root_nodes": [], "leaf_nodes": [], "deepest_components": [],
    }

def get_component_isolation_defaults() -> Dict[str, Any]:
    return {
        "isolated_count": 0, "isolated_percentage": 0,
        "source_count": 0, "source_percentage": 0,
        "sink_count": 0, "sink_percentage": 0,
        "bidirectional_count": 0, "bidirectional_percentage": 0,
        "category": "unavailable", "interpretation": "Data unavailable",
        "health": "unknown", "top_sources": [], "top_sinks": [],
        "isolated_components": [], "total_nodes": 0,
    }

def get_message_flow_patterns_defaults() -> Dict[str, Any]:
    return {
        "total_topics": 0, "total_brokers": 0, "total_applications": 0,
        "active_applications": 0, "avg_publishers_per_topic": 0,
        "avg_subscribers_per_topic": 0, "avg_topics_per_broker": 0,
        "interpretation": "Data unavailable", "category": "error",
        "health": "unknown", "hot_topics": [], "broker_utilization": [],
        "isolated_applications": [], "top_publishers": [], "top_subscribers": [],
    }

def get_component_redundancy_defaults() -> Dict[str, Any]:
    return {
        "total_components": 0, "spof_count": 0, "spof_percentage": 0,
        "redundant_count": 0, "redundancy_percentage": 0, "resilience_score": 0,
        "interpretation": "Data unavailable", "category": "error",
        "health": "unknown", "single_points_of_failure": [], "bridge_components": [],
    }

def get_weight_distribution_defaults() -> Dict[str, Any]:
    return {
        "total_components": 0, "total_edges": 0, "total_weight": 0, "avg_weight": 0,
        "median_weight": 0, "min_weight": 0, "max_weight": 0,
        "std_weight": 0, "weight_concentration": 0,
        "interpretation": "Data unavailable", "category": "error",
        "health": "unknown", "very_high_count": 0, "high_count": 0,
        "medium_count": 0, "low_count": 0, "very_low_count": 0,
        "top_components": [], "top_edges": [], "type_stats": {},
    }
