"""
In-Memory Graph Repository Adapter

Implements IGraphRepository using in-memory storage for testing.
"""

from typing import Dict, Any, List, Optional
import copy

from src.core.ports.graph_repository import IGraphRepository
from src.core.models import GraphData, ComponentData, EdgeData
from src.core.utils import serialization


class MemoryRepository:
    """
    In-memory adapter implementing IGraphRepository.
    
    Useful for testing without Neo4j dependency.
    """
    
    def __init__(self) -> None:
        """Initialize in-memory repository."""
        self.data: Dict[str, Any] = {
            "metadata": {},
            "nodes": [],
            "brokers": [],
            "topics": [],
            "applications": [],
            "libraries": [],
            "relationships": {
                "runs_on": [],
                "routes": [],
                "publishes_to": [],
                "subscribes_to": [],
                "connects_to": [],
                "uses": [],
            },
        }
        self.derived_stats: Dict[str, int] = {}

    def close(self) -> None:
        """No-op for in-memory repository."""
        pass

    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """
        Save graph data to in-memory storage, performing normalization/flattening
        to simulate Neo4j persistence behavior.
        """
        if clear:
            # We initialize with empty structures but preserve metadata handling
            self.data = {
                "metadata": {},
                "nodes": [], "brokers": [], "topics": [], "applications": [], "libraries": [],
                "relationships": {
                    "runs_on": [], "routes": [], "publishes_to": [],
                    "subscribes_to": [], "connects_to": [], "uses": [],
                },
            }

        # 1. Normalize and store Metadata
        if "metadata" in data:
            self.data["metadata"] = serialization.flatten_metadata(data["metadata"])

        # 2. Normalize and store Components
        mapping = {
            "nodes": "Node", "brokers": "Broker", "topics": "Topic", 
            "applications": "Application", "libraries": "Library"
        }
        for key, comp_type in mapping.items():
            for item in data.get(key, []):
                flattened = serialization.flatten_component(item, comp_type)
                # MERGE behavior (simple overwrite for memory)
                existing = next((x for x in self.data[key] if x["id"] == flattened["id"]), None)
                if existing:
                    existing.update(flattened)
                else:
                    self.data[key].append(flattened)
        
        # 3. Store Relationships (simple copy for memory)
        for key in self.data["relationships"]:
            items = data.get("relationships", {}).get(key, [])
            for item in items:
                # Basic normalization for source/target keys
                rel = {
                    "from": item.get("from", item.get("source")),
                    "to": item.get("to", item.get("target")),
                    "weight": item.get("weight", 1.0)
                }
                self.data["relationships"][key].append(rel)

    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False,
    ) -> GraphData:
        """Retrieve graph data with optional type filtering."""
        components = []
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            ctype = key[:-1].capitalize()  # nodes -> Node
            if key == "libraries": ctype = "Library"
            
            if component_types and ctype not in component_types:
                continue
                
            for item in self.data.get(key, []):
                components.append(ComponentData(
                    id=item["id"],
                    component_type=ctype,
                    weight=item.get("weight", 1.0),
                    properties={k: v for k, v in item.items() if k not in ["id", "weight"]}
                ))

        # Build a lookup map of ID -> Type for edges
        id_to_type = {c.id: c.component_type for c in components}

        # Build component weight lookup for Phase-5-equivalent propagation
        comp_weights: Dict[str, float] = {c.id: c.weight for c in components}

        # Map raw relationship types to DEPENDS_ON subtypes (mirrors neo4j_repo derivation rules).
        # Note: runs_on is a structural edge and does NOT map to any DEPENDS_ON subtype
        # (node_to_node is derived from app-level dependencies in Rule 3, not from RUNS_ON directly).
        _dep_type_map: Dict[str, str] = {
            "publishes_to":  "app_to_app",
            "subscribes_to": "app_to_app",
            "uses":          "app_to_lib",    # Rule 5: app depends on library (was wrongly app_to_app)
            "routes":        "app_to_broker",
            "connects_to":   "node_to_node",
            # "runs_on" intentionally omitted — structural only, not a DEPENDS_ON dependency
        }

        edges = []
        for rel_type, items in self.data.get("relationships", {}).items():
            dep_type = _dep_type_map.get(rel_type)
            if dep_type is None:
                continue  # skip structural-only edges (runs_on) and unknown types

            if dependency_types and dep_type not in dependency_types:
                continue

            for item in items:
                src_id = item.get("from", item.get("source_id"))
                tgt_id = item.get("to", item.get("target_id"))

                # Phase-5-equivalent weight propagation:
                # app_to_lib edges inherit the source application's weight (Rule 5).
                raw_weight = item.get("weight", 1.0)
                if dep_type == "app_to_lib" and src_id in comp_weights:
                    raw_weight = comp_weights[src_id] or 0.01

                edges.append(EdgeData(
                    source_id=src_id,
                    target_id=tgt_id,
                    source_type=id_to_type.get(src_id, "Unknown"),
                    target_type=id_to_type.get(tgt_id, "Unknown"),
                    dependency_type=dep_type,
                    relation_type=rel_type,
                    weight=raw_weight,
                    properties={k: v for k, v in item.items() if k not in ["source_id", "target_id", "weight"]}
                ))
        
        return GraphData(components=components, edges=edges)

    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data for a specific layer."""
        return self.get_graph_data()

    def get_statistics(self) -> Dict[str, int]:
        """Retrieve graph statistics."""
        stats = {}
        total_nodes = 0
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            count = len(self.data.get(key, []))
            stats[f"{key[:-1]}_count"] = count
            total_nodes += count
        
        total_rels = 0
        for key in self.data.get("relationships", {}):
            total_rels += len(self.data["relationships"][key])
            
        stats["total_nodes"] = total_nodes
        stats["total_relationships"] = total_rels
        return stats

    def export_json(self) -> Dict[str, Any]:
        """
        Export graph as JSON (compatible with data generation format).
        Consolidated via get_graph_data to ensure logic consistency and fidelity parity
        with the Neo4j persistence layer.
        """
        graph_data = self.get_graph_data(include_raw=True)
        # MemoryRepository stores metadata in a format already compatible with reconstruction
        # but we pass it through anyway for future-proofing and consistency.
        return serialization.reconstruct_export_payload(graph_data, self.data["metadata"])
