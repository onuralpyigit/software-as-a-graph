"""
Statistics endpoints for graph metrics and distributions.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from api.models import Neo4jCredentials
from src.analysis.statistics_service import StatisticsService
from src.core.models import GraphData
from src.core import create_repository
from src.analysis import statistics as stats_logic

router = APIRouter(prefix="/api/v1/stats", tags=["statistics"])
logger = logging.getLogger(__name__)


def patch_graph_data(graph_data: GraphData) -> GraphData:
    """
    Patch ComponentData objects to add 'type' attribute that maps to 'component_type'.
    
    This is a workaround for the mismatch between ComponentData model (which has 'component_type')
    and statistics functions (which expect 'type'). Since we cannot modify backend/src code,
    we patch the data at the API layer.
    """
    for component in graph_data.components:
        if not hasattr(component, 'type'):
            # Add 'type' as an alias for 'component_type'
            component.type = component.component_type
    return graph_data


@router.post("", response_model=Dict[str, Any])
@router.post("/", response_model=Dict[str, Any])
async def get_graph_stats(credentials: Neo4jCredentials):
    """
    Get overall graph statistics including structural relationships.
    """
    service = StatisticsService(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Getting graph statistics")
        stats = service.get_graph_stats()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Stats query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        service.close()


@router.post("/degree-distribution", response_model=Dict[str, Any])
async def get_degree_distribution_stats(credentials: Neo4jCredentials):
    """
    Get fast degree distribution statistics.
    
    Computes:
    - In-degree, out-degree, and total degree statistics (mean, median, max, min, std)
    - Hub nodes (degree > mean + 2*std)
    - Isolated nodes count
    
    Optionally filter by node_type to analyze specific component types.
    Runs in O(V+E) time - very fast even for large graphs.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing degree distribution statistics{filter_msg}")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Call statistics function directly with patched data
        stats = stats_logic.get_degree_distribution(patched_data, node_type=credentials.node_type)
        return stats
    except AttributeError as e:
        # ComponentData model incompatibility - return empty stats
        logger.warning(f"Degree distribution not available due to model incompatibility: {str(e)}")
        return {
            "success": False,
            "error": "Feature temporarily unavailable",
            "in_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "out_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "total_degree": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "hub_nodes": [],
            "isolated_nodes": 0,
            "total_nodes": 0,
            "hub_threshold": 0,
            "computation_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Degree distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        repo.close()


@router.post("/connectivity-density", response_model=Dict[str, Any])
async def get_connectivity_density_stats(credentials: Neo4jCredentials):
    """
    Get connectivity density statistics - measures how interconnected the system is.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing connectivity density statistics{filter_msg}")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Call statistics function directly with patched data
        stats = stats_logic.get_connectivity_density(patched_data, node_type=credentials.node_type)
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except AttributeError as e:
        logger.warning(f"Connectivity density not available due to model incompatibility: {str(e)}")
        return {
            "success": False,
            "error": "Feature temporarily unavailable",
            "stats": {
                "density": 0,
                "total_nodes": 0,
                "total_edges": 0,
                "max_possible_edges": 0,
                "interpretation": "Data unavailable",
                "category": "unavailable",
                "most_dense_components": []
            },
            "computation_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Connectivity density computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        repo.close()


@router.post("/clustering-coefficient", response_model=Dict[str, Any])
async def get_clustering_coefficient_stats(credentials: Neo4jCredentials):
    """
    Get clustering coefficient statistics - measures how nodes tend to cluster together.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing clustering coefficient statistics{filter_msg}")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Call statistics function directly with patched data
        stats = stats_logic.get_clustering_coefficient(patched_data, node_type=credentials.node_type)
        
        # Enhance stats with missing fields expected by frontend
        # Recalculate all clustering coefficients to get distribution stats
        from collections import defaultdict
        import statistics as py_stats
        
        neighbors = defaultdict(set)
        component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in patched_data.components}
        component_types = {c.id: c.type for c in patched_data.components}
        
        for edge in patched_data.edges:
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
        
        # Filter by type if needed
        if credentials.node_type:
            filtered_coeffs = {k: v for k, v in local_coefficients.items() if component_types.get(k) == credentials.node_type}
        else:
            filtered_coeffs = local_coefficients
        
        coeff_values = list(filtered_coeffs.values())
        non_zero_coeffs = [c for c in coeff_values if c > 0]
        
        # Calculate statistics
        stats["global_clustering"] = stats.get("avg_clustering_coefficient", 0)
        stats["average_clustering"] = stats.get("avg_clustering_coefficient", 0)
        stats["max_coefficient"] = round(max(coeff_values), 4) if coeff_values else 0
        stats["min_coefficient"] = round(min(non_zero_coeffs), 4) if non_zero_coeffs else 0
        stats["median_coefficient"] = round(py_stats.median(coeff_values), 4) if coeff_values else 0
        stats["std_coefficient"] = round(py_stats.stdev(non_zero_coeffs), 4) if len(non_zero_coeffs) > 1 else 0
        
        # Categorize nodes by clustering level
        high_clustering = [(nid, c) for nid, c in filtered_coeffs.items() if c >= 0.5]
        medium_clustering = [(nid, c) for nid, c in filtered_coeffs.items() if 0.2 <= c < 0.5]
        low_clustering = [(nid, c) for nid, c in filtered_coeffs.items() if 0 < c < 0.2]
        zero_clustering = [(nid, c) for nid, c in filtered_coeffs.items() if c == 0]
        
        stats["high_clustering_count"] = len(high_clustering)
        stats["medium_clustering_count"] = len(medium_clustering)
        stats["low_clustering_count"] = len(low_clustering)
        stats["zero_clustering_count"] = len(zero_clustering)
        stats["total_nodes"] = len(filtered_coeffs)
        
        # Get high clustering nodes
        stats["high_clustering_nodes"] = [
            {
                "id": nid,
                "name": component_info[nid]["name"],
                "type": component_info[nid]["type"],
                "coefficient": round(c, 4),
                "degree": len(neighbors[nid])
            }
            for nid, c in sorted(high_clustering, key=lambda x: x[1], reverse=True)[:20]
        ]
        
        # Get zero clustering nodes
        stats["zero_clustering_nodes"] = [
            {
                "id": nid,
                "name": component_info[nid]["name"],
                "type": component_info[nid]["type"],
                "degree": len(neighbors[nid])
            }
            for nid, c in zero_clustering[:20]
        ]
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except AttributeError as e:
        logger.warning(f"Clustering coefficient not available due to model incompatibility: {str(e)}")
        return {
            "success": False,
            "error": "Feature temporarily unavailable",
            "stats": {
                "avg_clustering_coefficient": 0,
                "global_clustering": 0,
                "average_clustering": 0,
                "max_coefficient": 0,
                "median_coefficient": 0,
                "min_coefficient": 0,
                "std_coefficient": 0,
                "high_clustering_count": 0,
                "medium_clustering_count": 0,
                "low_clustering_count": 0,
                "zero_clustering_count": 0,
                "total_nodes": 0,
                "high_clustering_nodes": [],
                "zero_clustering_nodes": []
            },
            "computation_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Clustering coefficient computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        repo.close()


@router.post("/dependency-depth", response_model=Dict[str, Any])
async def get_dependency_depth_stats(credentials: Neo4jCredentials):
    """
    Get dependency depth statistics - measures the depth of dependency chains.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing dependency depth statistics")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Call statistics function directly with patched data
        stats = stats_logic.get_dependency_depth(patched_data)
        
        # Enhance stats with missing fields expected by frontend
        from collections import defaultdict, deque
        import statistics as py_stats
        
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in patched_data.components}
        
        for edge in patched_data.edges:
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
        
        depth_values = list(node_depths.values())
        stats["median_depth"] = round(py_stats.median(depth_values), 2) if depth_values else 0
        stats["min_depth"] = min(depth_values) if depth_values else 0
        stats["std_depth"] = round(py_stats.stdev(depth_values), 2) if len(depth_values) > 1 else 0
        
        # Categorize by depth
        shallow = [(nid, d) for nid, d in node_depths.items() if d == 0]
        low = [(nid, d) for nid, d in node_depths.items() if 1 <= d <= 2]
        medium = [(nid, d) for nid, d in node_depths.items() if 3 <= d <= 5]
        high = [(nid, d) for nid, d in node_depths.items() if d > 5]
        
        stats["shallow_count"] = len(shallow)
        stats["low_depth_count"] = len(low)
        stats["medium_depth_count"] = len(medium)
        stats["high_depth_count"] = len(high)
        stats["total_nodes"] = len(node_depths)
        
        # Depth distribution
        depth_dist = {}
        for d in depth_values:
            depth_dist[str(d)] = depth_dist.get(str(d), 0) + 1
        stats["depth_distribution"] = depth_dist
        
        # Root nodes (no incoming edges)
        stats["root_nodes"] = [
            {
                "id": nid,
                "name": component_info[nid]["name"],
                "type": component_info[nid]["type"],
                "depth": node_depths[nid],
                "dependencies": len(outgoing[nid])
            }
            for nid in component_info if len(incoming[nid]) == 0
        ][:20]
        
        # Leaf nodes (no outgoing edges)
        stats["leaf_nodes"] = [
            {
                "id": nid,
                "name": component_info[nid]["name"],
                "type": component_info[nid]["type"],
                "depth": node_depths[nid],
                "dependents": len(incoming[nid])
            }
            for nid in component_info if len(outgoing[nid]) == 0
        ][:20]
        
        # Add interpretation
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
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except AttributeError as e:
        logger.warning(f"Dependency depth not available due to model incompatibility: {str(e)}")
        return {
            "success": False,
            "error": "Feature temporarily unavailable",
            "stats": {
                "max_depth": 0,
                "avg_depth": 0,
                "median_depth": 0,
                "min_depth": 0,
                "std_depth": 0,
                "interpretation": "Data unavailable",
                "category": "unavailable",
                "depth_distribution": {},
                "shallow_count": 0,
                "low_depth_count": 0,
                "medium_depth_count": 0,
                "high_depth_count": 0,
                "total_nodes": 0,
                "root_nodes": [],
                "leaf_nodes": [],
                "deepest_components": []
            },
            "computation_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Dependency depth computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        repo.close()


@router.post("/component-isolation", response_model=Dict[str, Any])
async def get_component_isolation_stats(credentials: Neo4jCredentials):
    """
    Get component isolation statistics - identifies isolated, source, and sink components.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        logger.info("Computing component isolation statistics")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Call statistics function directly with patched data
        stats = stats_logic.get_component_isolation(patched_data)
        
        # Enhance stats with missing fields expected by frontend
        total = stats.get("total_components", 0)
        isolated_count = stats.get("isolated_count", 0)
        source_count = stats.get("source_count", 0)
        sink_count = stats.get("sink_count", 0)
        bidirectional_count = stats.get("bidirectional_count", 0)
        
        # Calculate missing percentages
        stats["source_percentage"] = round(source_count / total * 100, 2) if total > 0 else 0
        stats["sink_percentage"] = round(sink_count / total * 100, 2) if total > 0 else 0
        stats["bidirectional_percentage"] = round(bidirectional_count / total * 100, 2) if total > 0 else 0
        
        # Add interpretation and health
        isolated_pct = stats.get("isolated_percentage", 0)
        if isolated_pct > 30:
            stats["health"] = "poor"
            stats["category"] = "highly_isolated"
            stats["interpretation"] = "High isolation indicates poor integration"
        elif isolated_pct > 15:
            stats["health"] = "moderate"
            stats["category"] = "moderately_isolated"
            stats["interpretation"] = "Moderate isolation - review integration points"
        elif isolated_pct > 5:
            stats["health"] = "fair"
            stats["category"] = "low_isolation"
            stats["interpretation"] = "Low isolation - good connectivity"
        else:
            stats["health"] = "good"
            stats["category"] = "well_connected"
            stats["interpretation"] = "Excellent connectivity"
        
        # Build component lists (since stats function doesn't return them)
        from collections import defaultdict
        incoming = defaultdict(int)
        outgoing = defaultdict(int)
        component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in patched_data.components}
        
        for edge in patched_data.edges:
            outgoing[edge.source_id] += 1
            incoming[edge.target_id] += 1
        
        isolated, sources, sinks, bidirectional = [], [], [], []
        for cid in component_info:
            inc = incoming[cid]
            outc = outgoing[cid]
            data = {
                "id": cid,
                "name": component_info[cid]["name"],
                "type": component_info[cid]["type"],
                "in_degree": inc,
                "out_degree": outc
            }
            if inc == 0 and outc == 0:
                isolated.append(data)
            elif inc == 0:
                sources.append(data)
            elif outc == 0:
                sinks.append(data)
            else:
                bidirectional.append(data)
        
        # Sort and add to stats
        stats["isolated_components"] = isolated[:20]  # Limit to top 20
        stats["top_sources"] = sorted(sources, key=lambda x: x["out_degree"], reverse=True)[:20]
        stats["top_sinks"] = sorted(sinks, key=lambda x: x["in_degree"], reverse=True)[:20]
        stats["top_bidirectional"] = sorted(bidirectional, key=lambda x: x["in_degree"] + x["out_degree"], reverse=True)[:20]
        
        # Rename total_components to total_nodes for consistency
        stats["total_nodes"] = stats.get("total_components", 0)
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except AttributeError as e:
        logger.warning(f"Component isolation not available due to model incompatibility: {str(e)}")
        return {
            "success": False,
            "error": "Feature temporarily unavailable",
            "stats": {
                "isolated_count": 0,
                "isolated_percentage": 0,
                "source_count": 0,
                "source_percentage": 0,
                "sink_count": 0,
                "sink_percentage": 0,
                "bidirectional_count": 0,
                "bidirectional_percentage": 0,
                "category": "unavailable",
                "interpretation": "Data unavailable",
                "health": "unknown",
                "top_sources": [],
                "top_sinks": [],
                "isolated_components": [],
                "total_nodes": 0
            },
            "computation_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Component isolation computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
    finally:
        repo.close()


@router.post("/message-flow-patterns", response_model=Dict[str, Any])
async def get_message_flow_patterns(credentials: Neo4jCredentials):
    """
    Get message flow pattern statistics - analyzes communication patterns in pub-sub system.
    
    Derives metrics from graph structure when backend implementation is unavailable.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        import time
        start_time = time.time()
        logger.info("Computing message flow pattern statistics from graph structure")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Get basic stats from repository
        basic_stats = repo.get_statistics()
        
        # Count components by type
        topics = [c for c in patched_data.components if c.type == 'Topic']
        brokers = [c for c in patched_data.components if c.type == 'Broker']
        applications = [c for c in patched_data.components if c.type == 'Application']
        
        # Analyze pub/sub relationships
        pub_edges = [e for e in patched_data.edges if e.relation_type == 'PUBLISHES_TO']
        sub_edges = [e for e in patched_data.edges if e.relation_type == 'SUBSCRIBES_TO']
        
        logger.info(f"Found {len(pub_edges)} PUBLISHES_TO edges and {len(sub_edges)} SUBSCRIBES_TO edges")
        logger.info(f"Topics: {len(topics)}, Brokers: {len(brokers)}, Applications: {len(applications)}")
        
        # Log sample edge types for debugging
        if patched_data.edges:
            edge_types = set([e.relation_type for e in patched_data.edges[:10]])
            logger.info(f"Sample edge types: {edge_types}")
        
        # Count activity per topic
        from collections import defaultdict
        topic_pubs = defaultdict(int)
        topic_subs = defaultdict(int)
        app_pubs = defaultdict(int)
        app_subs = defaultdict(int)
        
        for edge in pub_edges:
            topic_pubs[edge.target_id] += 1
            app_pubs[edge.source_id] += 1
        
        for edge in sub_edges:
            topic_subs[edge.target_id] += 1
            app_subs[edge.source_id] += 1
        
        # Find hot topics (high activity)
        hot_topics = []
        for topic in topics:
            pubs = topic_pubs.get(topic.id, 0)
            subs = topic_subs.get(topic.id, 0)
            total = pubs + subs
            if total > 0:
                hot_topics.append({
                    "id": topic.id,
                    "name": topic.properties.get('name', topic.id),
                    "publishers": pubs,
                    "subscribers": subs,
                    "total_activity": total,
                    "brokers": 0  # Not tracked in basic structure
                })
        
        hot_topics.sort(key=lambda x: x['total_activity'], reverse=True)
        hot_topics = hot_topics[:20]
        
        # Find top publishers and subscribers
        top_publishers = []
        for app in applications:
            pub_count = app_pubs.get(app.id, 0)
            if pub_count > 0:
                top_publishers.append({
                    "id": app.id,
                    "name": app.properties.get('name', app.id),
                    "publications": pub_count
                })
        top_publishers.sort(key=lambda x: x['publications'], reverse=True)
        top_publishers = top_publishers[:20]
        
        top_subscribers = []
        for app in applications:
            sub_count = app_subs.get(app.id, 0)
            if sub_count > 0:
                top_subscribers.append({
                    "id": app.id,
                    "name": app.properties.get('name', app.id),
                    "subscriptions": sub_count
                })
        top_subscribers.sort(key=lambda x: x['subscriptions'], reverse=True)
        top_subscribers = top_subscribers[:20]
        
        # Find isolated applications (no pub/sub activity)
        isolated_apps = []
        for app in applications:
            if app_pubs.get(app.id, 0) == 0 and app_subs.get(app.id, 0) == 0:
                isolated_apps.append({
                    "id": app.id,
                    "name": app.properties.get('name', app.id)
                })
        
        # Calculate averages
        active_app_count = len([a for a in applications if app_pubs.get(a.id, 0) > 0 or app_subs.get(a.id, 0) > 0])
        avg_pubs = sum(topic_pubs.values()) / len(topics) if topics else 0
        avg_subs = sum(topic_subs.values()) / len(topics) if topics else 0
        
        # Determine health and interpretation
        if len(isolated_apps) > len(applications) * 0.5:
            health = "poor"
            category = "isolated"
            interpretation = "Many applications are isolated from pub-sub system"
        elif len(hot_topics) < 3 and len(topics) > 10:
            health = "moderate"
            category = "sparse"
            interpretation = "Low message flow activity relative to system size"
        elif avg_pubs > 5 or avg_subs > 5:
            health = "fair"
            category = "bottleneck"
            interpretation = "Some topics may be overloaded with connections"
        else:
            health = "good"
            category = "balanced"
            interpretation = "Message flow appears well distributed"
        
        stats = {
            "total_topics": len(topics),
            "total_brokers": len(brokers),
            "total_applications": len(applications),
            "active_applications": active_app_count,
            "avg_publishers_per_topic": round(avg_pubs, 2),
            "avg_subscribers_per_topic": round(avg_subs, 2),
            "avg_topics_per_broker": 0,  # Not tracked in basic structure
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "hot_topics": hot_topics,
            "broker_utilization": [],  # Requires detailed broker tracking
            "isolated_applications": isolated_apps[:20],
            "top_publishers": top_publishers,
            "top_subscribers": top_subscribers,
            "computation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(f"Message flow stats computed: {len(hot_topics)} hot topics, "
                   f"{len(top_publishers)} publishers, {len(top_subscribers)} subscribers, "
                   f"{len(isolated_apps)} isolated apps")
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats["computation_time_ms"]
        }
    except Exception as e:
        logger.error(f"Message flow pattern computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "total_topics": 0,
                "total_brokers": 0,
                "total_applications": 0,
                "active_applications": 0,
                "avg_publishers_per_topic": 0,
                "avg_subscribers_per_topic": 0,
                "avg_topics_per_broker": 0,
                "interpretation": "Data unavailable",
                "category": "error",
                "health": "unknown",
                "hot_topics": [],
                "broker_utilization": [],
                "isolated_applications": [],
                "top_publishers": [],
                "top_subscribers": []
            },
            "computation_time_ms": 0
        }
    finally:
        repo.close()


@router.post("/component-redundancy", response_model=Dict[str, Any])
async def get_component_redundancy_stats(credentials: Neo4jCredentials):
    """
    Get component redundancy statistics - identifies SPOFs and bridge components.
    
    Derives metrics from connectivity and isolation analysis when backend implementation is unavailable.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        import time
        start_time = time.time()
        logger.info("Computing component redundancy from connectivity analysis")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Use component isolation to identify potential SPOFs
        isolation_stats = stats_logic.get_component_isolation(patched_data)
        
        # Analyze connectivity to identify critical nodes
        from collections import defaultdict, deque
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in patched_data.components}
        
        for edge in patched_data.edges:
            outgoing[edge.source_id].append(edge.target_id)
            incoming[edge.target_id].append(edge.source_id)
        
        # Identify potential SPOFs:
        # - Source nodes with many dependents (high out-degree)
        # - Nodes that are sole connectors between component groups
        spofs = []
        for comp_id in component_info:
            out_deg = len(outgoing[comp_id])
            in_deg = len(incoming[comp_id])
            
            # High fan-out nodes are potential SPOFs
            if out_deg > 5:
                spofs.append({
                    "id": comp_id,
                    "name": component_info[comp_id]["name"],
                    "type": component_info[comp_id]["type"],
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "is_critical": True,
                    "reason": f"High fan-out ({out_deg} dependents)"
                })
            # Nodes with both high in and out degree are bridges
            elif in_deg > 3 and out_deg > 3:
                spofs.append({
                    "id": comp_id,
                    "name": component_info[comp_id]["name"],
                    "type": component_info[comp_id]["type"],
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "is_critical": True,
                    "reason": f"Bridge component ({in_deg} in, {out_deg} out)"
                })
        
        spofs.sort(key=lambda x: x['out_degree'], reverse=True)
        spofs = spofs[:20]
        
        # Components with redundancy have multiple paths
        # For simplicity, consider components with high in-degree as having redundancy
        redundant = []
        for comp_id in component_info:
            in_deg = len(incoming[comp_id])
            if in_deg > 2:  # Multiple sources = redundancy
                redundant.append(comp_id)
        
        total_components = len(component_info)
        spof_count = len(spofs)
        redundant_count = len(redundant)
        
        # Calculate resilience score (inverse of SPOF percentage, boosted by redundancy)
        spof_pct = (spof_count / total_components * 100) if total_components > 0 else 0
        redundancy_pct = (redundant_count / total_components * 100) if total_components > 0 else 0
        resilience_score = max(0, min(100, 100 - spof_pct + (redundancy_pct * 0.3)))
        
        # Determine health
        if resilience_score >= 70:
            health = "good"
            interpretation = "High resilience - good redundancy and few SPOFs"
        elif resilience_score >= 50:
            health = "fair"
            interpretation = "Moderate resilience - some redundancy present"
        elif resilience_score >= 30:
            health = "moderate"
            interpretation = "Limited resilience - notable SPOFs identified"
        else:
            health = "poor"
            interpretation = "Low resilience - many SPOFs and limited redundancy"
        
        # Determine category
        if spof_count > total_components * 0.3:
            category = "fragile"
        elif redundant_count > total_components * 0.5:
            category = "resilient"
        elif redundant_count > total_components * 0.2:
            category = "moderate"
        else:
            category = "vulnerable"
        
        stats = {
            "total_components": total_components,
            "spof_count": spof_count,
            "spof_percentage": round(spof_pct, 2),
            "redundant_count": redundant_count,
            "redundancy_percentage": round(redundancy_pct, 2),
            "resilience_score": round(resilience_score, 1),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "single_points_of_failure": spofs,
            "bridge_components": spofs[:10],  # Top SPOFs are also bridges
            "computation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats["computation_time_ms"]
        }
    except Exception as e:
        logger.error(f"Component redundancy computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "total_components": 0,
                "spof_count": 0,
                "spof_percentage": 0,
                "redundant_count": 0,
                "redundancy_percentage": 0,
                "resilience_score": 0,
                "interpretation": "Data unavailable",
                "category": "error",
                "health": "unknown",
                "single_points_of_failure": [],
                "bridge_components": []
            },
            "computation_time_ms": 0
        }
    finally:
        repo.close()


@router.post("/node-weight-distribution", response_model=Dict[str, Any])
async def get_node_weight_distribution_stats(credentials: Neo4jCredentials):
    """
    Get node weight distribution statistics - analyzes how component importance is distributed.
    
    Derives weights from degree centrality when backend implementation is unavailable.
    Node weight distribution reveals:
    - Distribution of component weights (importance scores)
    - High-value vs low-value components
    - Weight concentration patterns
    - Critical component identification by weight
    
    Runs in O(V) - extremely fast.
    Provides insights into component importance hierarchy and architectural focus areas.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        import time
        import statistics as py_stats
        start_time = time.time()
        logger.info("Computing node weight distribution from degree centrality")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Calculate weights based on degree (normalized)
        from collections import defaultdict
        degree_map = defaultdict(int)
        component_info = {c.id: {"name": c.properties.get('name', c.id), "type": c.type} for c in patched_data.components}
        
        for edge in patched_data.edges:
            degree_map[edge.source_id] += 1
            degree_map[edge.target_id] += 1
        
        # Normalize weights (degree / max_degree)
        max_degree = max(degree_map.values()) if degree_map else 1
        weights = {comp_id: degree_map[comp_id] / max_degree for comp_id in component_info}
        
        weight_values = list(weights.values())
        total_weight = sum(weight_values)
        avg_weight = py_stats.mean(weight_values) if weight_values else 0
        median_weight = py_stats.median(weight_values) if weight_values else 0
        min_weight = min(weight_values) if weight_values else 0
        max_weight = max(weight_values) if weight_values else 0
        std_weight = py_stats.stdev(weight_values) if len(weight_values) > 1 else 0
        
        # Calculate concentration (Gini coefficient approximation)
        # Higher concentration = more uneven distribution
        sorted_weights = sorted(weight_values)
        n = len(sorted_weights)
        cumsum = 0
        for i, w in enumerate(sorted_weights):
            cumsum += (2 * (i + 1) - n - 1) * w
        weight_concentration = (cumsum / (n * sum(sorted_weights))) if n > 0 and sum(sorted_weights) > 0 else 0
        
        # Categorize components by weight
        very_high = [comp_id for comp_id, w in weights.items() if w >= 0.8]
        high = [comp_id for comp_id, w in weights.items() if 0.6 <= w < 0.8]
        medium = [comp_id for comp_id, w in weights.items() if 0.3 <= w < 0.6]
        low = [comp_id for comp_id, w in weights.items() if 0.1 <= w < 0.3]
        very_low = [comp_id for comp_id, w in weights.items() if w < 0.1]
        
        # Top components
        top_components = []
        sorted_by_weight = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for comp_id, weight in sorted_by_weight[:20]:
            top_components.append({
                "id": comp_id,
                "name": component_info[comp_id]["name"],
                "type": component_info[comp_id]["type"],
                "weight": round(weight, 4)
            })
        
        # Type-based statistics
        type_stats = {}
        by_type = defaultdict(list)
        for comp_id, comp_data in component_info.items():
            by_type[comp_data["type"]].append(weights[comp_id])
        
        for comp_type, type_weights in by_type.items():
            if type_weights:
                type_stats[comp_type] = {
                    "count": len(type_weights),
                    "total_weight": round(sum(type_weights), 3),
                    "avg_weight": round(py_stats.mean(type_weights), 3),
                    "median_weight": round(py_stats.median(type_weights), 3),
                    "min_weight": round(min(type_weights), 3),
                    "max_weight": round(max(type_weights), 3),
                    "std_weight": round(py_stats.stdev(type_weights), 3) if len(type_weights) > 1 else 0
                }
        
        # Determine health and interpretation
        if weight_concentration > 0.7:
            health = "poor"
            interpretation = "High concentration - few components dominate importance"
            category = "concentrated"
        elif weight_concentration > 0.5:
            health = "moderate"
            interpretation = "Moderate concentration - some imbalance in importance"
            category = "moderately_concentrated"
        elif weight_concentration > 0.3:
            health = "fair"
            interpretation = "Balanced distribution with some key components"
            category = "balanced"
        else:
            health = "good"
            interpretation = "Even distribution - importance is well spread"
            category = "distributed"
        
        stats = {
            "total_components": len(component_info),
            "total_weight": round(total_weight, 3),
            "avg_weight": round(avg_weight, 4),
            "median_weight": round(median_weight, 4),
            "min_weight": round(min_weight, 4),
            "max_weight": round(max_weight, 4),
            "std_weight": round(std_weight, 4),
            "weight_concentration": round(weight_concentration, 4),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "very_high_count": len(very_high),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
            "very_low_count": len(very_low),
            "top_components": top_components,
            "type_stats": type_stats,
            "computation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Node weight distribution computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "total_components": 0,
                "total_weight": 0,
                "avg_weight": 0,
                "median_weight": 0,
                "min_weight": 0,
                "max_weight": 0,
                "std_weight": 0,
                "weight_concentration": 0,
                "interpretation": "Data unavailable",
                "category": "error",
                "health": "unknown",
                "very_high_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "very_low_count": 0,
                "top_components": [],
                "type_stats": {}
            }
        }
    finally:
        repo.close()


@router.post("/edge-weight-distribution", response_model=Dict[str, Any])
async def get_edge_weight_distribution_stats(credentials: Neo4jCredentials):
    """
    Get edge weight distribution statistics - analyzes how dependency importance is distributed.
    
    Uses uniform edge weights when backend implementation is unavailable.
    Edge weight distribution reveals:
    - Distribution of dependency weights (connection strength)
    - Critical vs weak dependencies
    - Weight concentration patterns
    - Dependency type importance patterns
    
    Runs in O(E) - extremely fast.
    Provides insights into dependency criticality and architectural coupling patterns.
    """
    repo = create_repository(credentials.uri, credentials.user, credentials.password)
    try:
        import time
        import statistics as py_stats
        start_time = time.time()
        logger.info("Computing edge weight distribution from edge properties")
        
        # Get graph data and patch it
        graph_data = repo.get_graph_data()
        patched_data = patch_graph_data(graph_data)
        
        # Build component name lookup
        component_names = {c.id: c.properties.get('name', c.id) for c in patched_data.components}
        
        # Extract edge weights (use 1.0 as default if not specified)
        edge_weights = []
        edge_data_list = []  # Store edge data for top edges
        edge_types = {}
        from collections import defaultdict
        type_weights = defaultdict(list)
        
        for edge in patched_data.edges:
            weight = edge.properties.get('weight', 1.0) if hasattr(edge, 'properties') else 1.0
            edge_weights.append(weight)
            
            # Store edge data for later
            edge_data_list.append({
                'edge': edge,
                'weight': weight
            })
            
            # Get edge type
            edge_type = getattr(edge, 'relation_type', 'DEPENDS_ON')
            if edge_type not in edge_types:
                edge_types[edge_type] = 0
            edge_types[edge_type] += 1
            type_weights[edge_type].append(weight)
        
        if not edge_weights:
            edge_weights = [0]
        
        total_edges = len(edge_weights)
        total_weight = sum(edge_weights)
        avg_weight = py_stats.mean(edge_weights)
        median_weight = py_stats.median(edge_weights)
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        std_weight = py_stats.stdev(edge_weights) if len(edge_weights) > 1 else 0
        
        # Calculate concentration
        sorted_weights = sorted(edge_weights)
        n = len(sorted_weights)
        cumsum = 0
        for i, w in enumerate(sorted_weights):
            cumsum += (2 * (i + 1) - n - 1) * w
        weight_concentration = (cumsum / (n * sum(sorted_weights))) if n > 0 and sum(sorted_weights) > 0 else 0
        
        # Categorize edges by weight
        very_high = sum(1 for w in edge_weights if w >= 0.8 * max_weight)
        high = sum(1 for w in edge_weights if 0.6 * max_weight <= w < 0.8 * max_weight)
        medium = sum(1 for w in edge_weights if 0.3 * max_weight <= w < 0.6 * max_weight)
        low = sum(1 for w in edge_weights if 0.1 * max_weight <= w < 0.3 * max_weight)
        very_low = sum(1 for w in edge_weights if w < 0.1 * max_weight)
        
        # Type-based statistics
        type_stats = {}
        for edge_type, weights in type_weights.items():
            if weights:
                type_stats[edge_type] = {
                    "count": len(weights),
                    "total_weight": round(sum(weights), 3),
                    "avg_weight": round(py_stats.mean(weights), 3),
                    "median_weight": round(py_stats.median(weights), 3),
                    "min_weight": round(min(weights), 3),
                    "max_weight": round(max(weights), 3),
                    "std_weight": round(py_stats.stdev(weights), 3) if len(weights) > 1 else 0
                }
        
        # Get top edges by weight
        sorted_edges = sorted(edge_data_list, key=lambda x: x['weight'], reverse=True)
        top_edges = []
        for item in sorted_edges[:20]:
            edge = item['edge']
            top_edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "source_name": component_names.get(edge.source_id, edge.source_id),
                "target_name": component_names.get(edge.target_id, edge.target_id),
                "type": edge.relation_type,
                "weight": round(item['weight'], 4)
            })
        
        # Determine health and interpretation
        if weight_concentration > 0.7:
            health = "poor"
            interpretation = "High concentration - few edges dominate importance"
            category = "concentrated"
        elif weight_concentration > 0.5:
            health = "moderate"
            interpretation = "Moderate concentration - some critical dependencies"
            category = "moderately_concentrated"
        elif weight_concentration > 0.3:
            health = "fair"
            interpretation = "Balanced dependency weights"
            category = "balanced"
        else:
            health = "good"
            interpretation = "Even distribution - dependencies well distributed"
            category = "distributed"
        
        stats = {
            "total_edges": total_edges,
            "total_weight": round(total_weight, 3),
            "avg_weight": round(avg_weight, 4),
            "median_weight": round(median_weight, 4),
            "min_weight": round(min_weight, 4),
            "max_weight": round(max_weight, 4),
            "std_weight": round(std_weight, 4),
            "weight_concentration": round(weight_concentration, 4),
            "interpretation": interpretation,
            "category": category,
            "health": health,
            "very_high_count": very_high,
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
            "very_low_count": very_low,
            "top_edges": top_edges,
            "type_stats": type_stats,
            "computation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Edge weight distribution computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "total_edges": 0,
                "total_weight": 0,
                "avg_weight": 0,
                "median_weight": 0,
                "min_weight": 0,
                "max_weight": 0,
                "std_weight": 0,
                "weight_concentration": 0,
                "interpretation": "Data unavailable",
                "category": "error",
                "health": "unknown",
                "very_high_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "very_low_count": 0,
                "top_edges": [],
                "type_stats": {}
            }
        }
    finally:
        repo.close()
