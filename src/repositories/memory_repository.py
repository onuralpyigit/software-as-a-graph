from typing import Dict, Any, List, Optional
import copy

from src.models import QoSPolicy
from src.models import GraphData, ComponentData, EdgeData

class InMemoryGraphRepository:
    """
    In-memory implementation of GraphRepository for testing.
    """

    def __init__(self) -> None:
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
        pass

    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        if clear:
            self.data = copy.deepcopy(data)
        else:
            # Simple merge logic for testing
            for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
                self.data[key].extend(data.get(key, []))
            
            for key in self.data["relationships"]:
                self.data["relationships"][key].extend(data.get("relationships", {}).get(key, []))



    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
    ) -> GraphData:
        # Simple implementation that returns everything or filters by ID if needed.
        # For unit tests, we often just want to verify data was saved.
        
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

        edges = [] # In-memory derivation of DEPENDS_ON is complex, skipping for basic unit tests
        
        return GraphData(components=components, edges=edges)

    def get_layer_data(self, layer: str) -> GraphData:
        return self.get_graph_data()

    def get_statistics(self) -> Dict[str, int]:
        stats = {}
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            stats[f"{key[:-1]}_count"] = len(self.data.get(key, []))
        return stats

    def export_json(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)
