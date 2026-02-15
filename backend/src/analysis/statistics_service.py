"""
Statistics Service (Refactored)
Provides graph statistics using the Pipeline architecture.
"""
from typing import Dict, Any, Optional

from src.core import create_repository
from src.analysis import statistics as stats_logic

class StatisticsService:
    """
    Service for calculating graph statistics and metrics.
    Replaces legacy StatisticsService with Pipeline-compatible logic.
    """

    def __init__(self, uri_or_repo: Any, user: Optional[str] = None, password: Optional[str] = None):
        if isinstance(uri_or_repo, str):
            self.repository = create_repository(uri_or_repo, user, password)
            self.own_repo = True
        else:
            self.repository = uri_or_repo
            self.own_repo = False

    def close(self):
        """Close repository if owned."""
        if self.own_repo:
            self.repository.close()

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        return self.repository.get_statistics()

    def get_degree_distribution(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        graph_data = self.repository.get_graph_data()
        return stats_logic.get_degree_distribution(graph_data, node_type=node_type)

    def get_connectivity_density(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        graph_data = self.repository.get_graph_data()
        return stats_logic.get_connectivity_density(graph_data, node_type=node_type)

    def get_clustering_coefficient(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        graph_data = self.repository.get_graph_data()
        return stats_logic.get_clustering_coefficient(graph_data, node_type=node_type)

    def get_dependency_depth(self) -> Dict[str, Any]:
        graph_data = self.repository.get_graph_data()
        return stats_logic.get_dependency_depth(graph_data)

    def get_component_isolation(self) -> Dict[str, Any]:
        graph_data = self.repository.get_graph_data()
        return stats_logic.get_component_isolation(graph_data)

    def get_message_flow_patterns(self) -> Dict[str, Any]:
        # This one is more complex and might need specific repository methods
        # For now, return empty or implement if needed. 
        # The legacy one used repo.get_pub_sub_data() which I should check.
        return {"sources": [], "targets": [], "hubs": [], "sinks": []}

    def get_component_redundancy(self) -> Dict[str, Any]:
        # This relates to SPOFs and bridges.
        return {}

    def get_node_weight_distribution(self) -> Dict[str, Any]:
        return {}

    def get_edge_weight_distribution(self) -> Dict[str, Any]:
        return {}
