
from typing import Dict, Any, List, Optional
from saag.core.models import ComponentData, GraphData

def reconstruct_metadata_dict(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct graph metadata from flattened properties.
    Handles mapping from storage-optimized keys back to nested structure.
    """
    if not props:
        return {}
        
    return {
        "scale": {
            "apps": props.get("scale_apps", 0),
            "topics": props.get("scale_topics", 0),
            "brokers": props.get("scale_brokers", 0),
            "nodes": props.get("scale_nodes", 0),
            "libs": props.get("scale_libs", 0)
        },
        "seed": props.get("seed"),
        "generation_mode": props.get("generation_mode"),
        "domain": props.get("domain"),
        "scenario": props.get("scenario")
    }

def reconstruct_component_dict(comp: ComponentData) -> Dict[str, Any]:
    """
    Reconstruct a component dictionary with nested sub-objects (system_hierarchy, code_metrics)
    from ComponentData properties.
    """
    props = comp.properties
    # Base fields
    res = {"id": comp.id, "name": props.get("name", comp.id), "weight": comp.weight}
    
    # 1. System Hierarchy reconstruction
    sh = {}
    for key in ["csms_name", "css_name", "csc_name", "csci_name"]:
        if val := props.get(key):
            if val != "": # Don't export empty strings
                sh[key] = val
    if sh:
        res["system_hierarchy"] = sh

    # 2. Code Metrics reconstruction (Size, Complexity, Cohesion, Coupling, Quality)
    cm = {"size": {}, "complexity": {}, "cohesion": {}, "coupling": {}, "quality": {}}
    
    # Mapping definition for un-flattening
    metrics_mapping = {
        "size": {
            "cm_total_loc": "total_loc", "cm_total_classes": "total_classes", 
            "cm_total_methods": "total_methods", "cm_total_fields": "total_fields"
        },
        "complexity": {
            "cm_total_wmc": "total_wmc", "cm_avg_wmc": "avg_wmc", "cm_max_wmc": "max_wmc"
        },
        "cohesion": {
            "cm_avg_lcom": "avg_lcom", "cm_max_lcom": "max_lcom"
        },
        "coupling": {
            "cm_avg_cbo": "avg_cbo", "cm_max_cbo": "max_cbo", "cm_avg_rfc": "avg_rfc",
            "cm_max_rfc": "max_rfc", "cm_avg_fanin": "avg_fanin", "cm_max_fanin": "max_fanin",
            "cm_avg_fanout": "avg_fanout", "cm_max_fanout": "max_fanout"
        },
        "quality": {
            "sqale_debt_ratio": "sqale_debt_ratio", "bugs": "bugs", 
            "vulnerabilities": "vulnerabilities", "duplicated_lines_density": "duplicated_lines_density"
        }
    }
    
    for section, fields in metrics_mapping.items():
        for flat_key, nest_key in fields.items():
            if flat_key in props:
                cm[section][nest_key] = props[flat_key]
    
    # Filter out empty sections
    cm = {k: v for k, v in cm.items() if v}
    if cm:
        res["code_metrics"] = cm

    # 3. Type-specific logic and property normalization
    if comp.component_type == "Topic":
        res["size"] = props.get("size", 256)
        # Preserve uppercase QoS values — the canonical format and weight calculations
        # both use uppercase (RELIABLE, TRANSIENT_LOCAL, HIGH, etc.). Lowercasing here
        # would cause silent weight mismatches on round-trip import.
        res["qos"] = {
            "reliability": props.get("qos_reliability", "BEST_EFFORT"),
            "durability": props.get("qos_durability", "VOLATILE"),
            "transport_priority": props.get("qos_transport_priority", "MEDIUM"),
        }
    elif comp.component_type == "Application":
        res.update({
            "role": props.get("role", "pubsub"),
            "app_type": props.get("app_type", "service"),
            "criticality": props.get("criticality", "LOW"),
        })
        if props.get("version"): res["version"] = props["version"]
    elif comp.component_type == "Library":
        if props.get("version"): res["version"] = props["version"]
    elif comp.component_type == "Node":
        for key in ["ip_address", "cpu_cores", "memory_gb", "os_type"]:
            if key in props: res[key] = props[key]
    elif comp.component_type == "Broker":
        for key in ["type", "max_connections", "host"]:
            if key in props: res[key] = props[key]
    
    return res

def reconstruct_export_payload(graph_data: GraphData, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble the final export JSON payload from GraphData and metadata.
    Ensures consistent category mapping and relationship placement.
    """
    data = {
        "metadata": metadata,
        "nodes": [], "brokers": [], "topics": [], 
        "applications": [], "libraries": [],
        "relationships": {
            "runs_on": [], "routes": [], "publishes_to": [],
            "subscribes_to": [], "connects_to": [], "uses": [],
            "depends_on": [] # include pre-computed dependencies
        }
    }
    
    # Category mapping for components
    type_to_category = {
        "Node": "nodes",
        "Broker": "brokers",
        "Topic": "topics",
        "Application": "applications",
        "Library": "libraries"
    }
    
    # Process components
    for comp in graph_data.components:
        category = type_to_category.get(comp.component_type)
        if not category:
            continue
            
        comp_dict = reconstruct_component_dict(comp)
        data[category].append(comp_dict)
        
    # Process edges
    for edge in graph_data.edges:
        rel_key = edge.relation_type.lower()
        if rel_key in data["relationships"]:
            edge_dict = {"from": edge.source_id, "to": edge.target_id, "weight": edge.weight}
            
            # Include metadata for DEPENDS_ON
            if edge.relation_type == "DEPENDS_ON":
                edge_dict.update({
                    "dependency_type": edge.dependency_type,
                    "path_count": edge.path_count
                })
            
            data["relationships"][rel_key].append(edge_dict)
            
    return data

def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested metadata into storage-optimized properties.
    """
    if not metadata:
        return {}
        
    scale = metadata.get("scale", {})
    return {
        "seed": metadata.get("seed"),
        "generation_mode": metadata.get("generation_mode", "unknown"),
        "domain": metadata.get("domain"),
        "scenario": metadata.get("scenario"),
        "scale_apps": scale.get("apps", 0),
        "scale_topics": scale.get("topics", 0),
        "scale_brokers": scale.get("brokers", 0),
        "scale_nodes": scale.get("nodes", 0),
        "scale_libs": scale.get("libs", 0),
    }

def flatten_component(comp: Dict[str, Any], comp_type: str) -> Dict[str, Any]:
    """
    Flatten nested component data (code_metrics, system_hierarchy) 
    into storage-optimized properties.
    """
    # 1. Base fields
    res = {
        "id": comp["id"],
        "name": comp.get("name", comp["id"]),
    }
    if "weight" in comp:
        res["weight"] = comp["weight"]

    # 2. System Hierarchy
    sh = comp.get("system_hierarchy") or {}
    for key in ["csms_name", "css_name", "csc_name", "csci_name"]:
        res[key] = sh.get(key, "")

    # 3. Code Metrics
    cm = comp.get("code_metrics") or {}
    metrics_mapping = {
        "size": {
            "cm_total_loc": "total_loc", "cm_total_classes": "total_classes", 
            "cm_total_methods": "total_methods", "cm_total_fields": "total_fields"
        },
        "complexity": {
            "cm_total_wmc": "total_wmc", "cm_avg_wmc": "avg_wmc", "cm_max_wmc": "max_wmc"
        },
        "cohesion": {
            "cm_avg_lcom": "avg_lcom", "cm_max_lcom": "max_lcom"
        },
        "coupling": {
            "cm_avg_cbo": "avg_cbo", "cm_max_cbo": "max_cbo", "cm_avg_rfc": "avg_rfc",
            "cm_max_rfc": "max_rfc", "cm_avg_fanin": "avg_fanin", "cm_max_fanin": "max_fanin",
            "cm_avg_fanout": "avg_fanout", "cm_max_fanout": "max_fanout"
        },
        "quality": {
            "sqale_debt_ratio": "sqale_debt_ratio", "bugs": "bugs", 
            "vulnerabilities": "vulnerabilities", "duplicated_lines_density": "duplicated_lines_density"
        }
    }
    
    for section, fields in metrics_mapping.items():
        section_data = cm.get(section, {})
        # Support both nested (cm[section][key]) and legacy flat (comp[key]) if needed,
        # but here we focus on canonical nested input.
        for flat_key, nest_key in fields.items():
            res[flat_key] = section_data.get(nest_key, 0)
            # Support floats for average/ratio fields
            if "avg" in nest_key or "ratio" in nest_key or "density" in nest_key:
                res[flat_key] = float(res[flat_key])

    # 3.1 Additional flat aliases for common analysis fields (Issue 7 hardening)
    # These ensure compatibility with extract_layer_subgraph and basic Cypher queries.
    if cm:
        res["loc"] = cm.get("size", {}).get("total_loc", 0)
        res["cyclomatic_complexity"] = float(cm.get("complexity", {}).get("avg_wmc", 0.0))
        res["lcom"] = float(cm.get("cohesion", {}).get("avg_lcom", 0.0))
        res["coupling_afferent"] = int(cm.get("coupling", {}).get("avg_fanin", 0))
        res["coupling_efferent"] = int(cm.get("coupling", {}).get("avg_fanout", 0))

    # 4. Type-specific properties
    if comp_type == "Topic":
        qos = comp.get("qos", comp.get("qos_policy", {}))
        res.update({
            "size": comp.get("size", 256),
            "qos_reliability": qos.get("reliability", "BEST_EFFORT"),
            "qos_durability": qos.get("durability", "VOLATILE"),
            "qos_transport_priority": qos.get("transport_priority", "MEDIUM"),
        })
    elif comp_type == "Application":
        res.update({
            "role": comp.get("role", "pubsub"),
            "app_type": comp.get("app_type", "service"),
            "criticality": comp.get("criticality", "LOW"),
            "version": comp.get("version", ""),
        })
    elif comp_type == "Library":
        res["version"] = comp.get("version", "")
    elif comp_type == "Node":
        res.update({
            "ip_address": comp.get("ip_address", ""),
            "cpu_cores": comp.get("cpu_cores", 0),
            "memory_gb": comp.get("memory_gb", 0),
            "os_type": comp.get("os_type", "linux"),
        })
    elif comp_type == "Broker":
        res.update({
            "type": comp.get("type", "mqtt"),
            "max_connections": comp.get("max_connections", 0),
            "host": comp.get("host", ""),
        })

    return res
