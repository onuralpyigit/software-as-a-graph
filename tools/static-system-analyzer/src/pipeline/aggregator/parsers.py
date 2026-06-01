"""
Parser modules: Used for parsing SYSTEM_REPO and TypeSupport data.
"""

from typing import List, Tuple, Dict, Set
from .models import Topic
from common.logger import log_info
from common.runtime_config import get_runtime_config


def _is_dummy_topic(topic_name: str) -> bool:
    """Return True when the topic name should be excluded from aggregation."""
    normalized_name = topic_name.strip().lower()
    dummy_topic_names = {
        name.strip().lower()
        for name in get_runtime_config().aggregator.dummy_topic_names
        if name.strip()
    }
    return normalized_name in dummy_topic_names


class SystemRepoParser:
    """
    Processes System Application Repository.
    """
    def __init__(self, platform_name: str, config_dir: str = ""):
        self.platform_name = platform_name
        self.config_dir = config_dir
        # In real implementation, platform-specific files can be read here.
        # Config files are expected at: {config_dir}/system_repo_{platform_name}.xml or similar

    def get_app_node_relation(self) -> List[Tuple[str, str]]:
        """
        Returns information about which nodes applications run on.
        In a real scenario, this method should read SYSTEM_REPO files to produce this data.
        Currently returning mock data.
        
        Returns:
            List of (app_name, node_name) tuples
        """
        log_info(f"SystemRepoParser: Retrieving app-node relationships for '{self.platform_name}'.")
        # Example mock data:
        return [
            ("App-0", "Node-1"),
            ("App-1", "Node-0"),
            ("App-2", "Node-0"),
            ("App-3", "Node-1"),
        ]

    def get_app_role_relation(self) -> Dict[str, str]:
        """
        Returns application role information.
        
        Returns:
            Dictionary mapping app_name to role
        """
        log_info(f"SystemRepoParser: Retrieving app-role relationships for '{self.platform_name}'.")
        return {}

    def get_app_criticality_relation(self) -> Dict[str, bool]:
        """
        Returns application criticality information.
        
        Returns:
            Dictionary mapping app_name to criticality boolean
        """
        log_info(f"SystemRepoParser: Retrieving app-criticality relationships for '{self.platform_name}'.")
        return {}


class TypeSupportParser:
    """
    Processes TypeSupport data (typically generated from IDLs).
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # In real implementation, files under root_dir can be scanned here.

    def get_topic_list(self) -> Set[Topic]:
        """
        Finds all Topic objects in the project and returns them as a set.
        In a real scenario, this method should read IDL or similar definition files
        to produce this data. Currently returning mock data.
        
        Returns:
            Set of Topic objects
        """
        log_info(f"TypeSupportParser: Retrieving topics from '{self.root_dir}' directory.")
        # Example mock data:
        topics = {
            Topic(
                name="Topic-0", 
                size=6138, 
                durability="PERSISTENT", 
                reliability="BEST_EFFORT", 
                transport_priority="LOW"
            ),
            Topic(
                name="Topic-1", 
                size=1207, 
                durability="TRANSIENT", 
                reliability="BEST_EFFORT", 
                transport_priority="MEDIUM"
            ),
            Topic(
                name="Topic-2", 
                size=4901, 
                durability="PERSISTENT", 
                reliability="BEST_EFFORT", 
                transport_priority="LOW"
            ),
        }

        filtered_topics = {topic for topic in topics if not _is_dummy_topic(topic.name)}
        if len(filtered_topics) != len(topics):
            log_info("TypeSupportParser: DummyTopic entries were skipped.")

        return filtered_topics
