
from typing import Dict, Any, List, Optional

def format_generation_response(graph_data: Dict[str, Any], scale: str) -> Dict[str, Any]:
    """Format the response for graph generation."""
    return {
        "success": True,
        "message": f"Graph generated successfully with scale '{scale}'",
        "metadata": graph_data.get("metadata", {}),
        "stats": {
            "nodes": len(graph_data.get("nodes", [])),
            "brokers": len(graph_data.get("brokers", [])),
            "topics": len(graph_data.get("topics", [])),
            "applications": len(graph_data.get("applications", []))
        },
        "graph_data": graph_data
    }

def format_import_response(stats: Dict[str, Any], message: str = "Graph imported successfully") -> Dict[str, Any]:
    """Format the response for graph import."""
    return {
        "success": True,
        "message": message,
        "stats": stats
    }

def format_generate_import_response(graph_data: Dict[str, Any], stats: Dict[str, Any], scale: str, seed: int) -> Dict[str, Any]:
    """Format the response for combined generate and import."""
    return {
        "success": True,
        "message": f"Graph generated (scale={scale}) and imported successfully",
        "generation": {
            "scale": scale,
            "seed": seed,
            "metadata": graph_data.get("metadata", {})
        },
        "import_stats": stats
    }

def format_export_response(graph_data: Any, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Format the response for graph export."""
    return {
        "success": True,
        "export_format": "analysis",
        "components": [c.to_dict() for c in graph_data.components],
        "edges": [e.to_dict() for e in graph_data.edges],
        "stats": stats
    }

def format_limited_export_response(components: List[Any], edges: List[Any], node_limit: int, edge_limit: Optional[int]) -> Dict[str, Any]:
    """Format the response for limited graph export."""
    return {
        "success": True,
        "export_format": "analysis",
        "components": [c.to_dict() for c in components],
        "edges": [e.to_dict() for e in edges],
        "stats": {
            "component_count": len(components),
            "edge_count": len(edges),
            "node_limit": node_limit,
            "edge_limit": edge_limit,
            "limited": True
        }
    }

def format_neo4j_export_response(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the response for Neo4j JSON export."""
    return {
        "success": True,
        "export_format": "persistence",
        "message": "Graph data exported successfully",
        "graph_data": graph_data,
        "stats": {
            "nodes": len(graph_data.get("nodes", [])),
            "brokers": len(graph_data.get("brokers", [])),
            "topics": len(graph_data.get("topics", [])),
            "applications": len(graph_data.get("applications", [])),
            "libraries": len(graph_data.get("libraries", []))
        }
    }

def format_node_connections_response(node_id: str, depth: int, components: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format the response for node connections query."""
    return {
        "success": True,
        "node_id": node_id,
        "depth": depth,
        "components": components,
        "edges": edges,
        "stats": {
            "connected_nodes": len(components),
            "edges": len(edges)
        }
    }

def format_topology_response(node_id: Optional[str], components: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format the response for topology data query."""
    return {
        "success": True,
        "node_id": node_id,
        "components": components,
        "edges": edges,
        "stats": {
            "nodes": len(components),
            "edges": len(edges)
        }
    }
