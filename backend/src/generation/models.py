"""
Statistical Data Models for Graph Generation
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.core.models import QoSPolicy

SCALE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":   {"apps": 5,   "topics": 5,   "brokers": 1, "nodes": 2,  "libs": 2},
    "small":  {"apps": 15,  "topics": 10,  "brokers": 2, "nodes": 4,  "libs": 5},
    "medium": {"apps": 50,  "topics": 30,  "brokers": 3, "nodes": 8,  "libs": 10},
    "large":  {"apps": 150, "topics": 100, "brokers": 6, "nodes": 20, "libs": 30},
    "xlarge": {"apps": 500, "topics": 300, "brokers": 10, "nodes": 50, "libs": 100},
}

DURABILITY_OPTIONS = list(QoSPolicy.DURABILITY_SCORES.keys())
RELIABILITY_OPTIONS = list(QoSPolicy.RELIABILITY_SCORES.keys())
PRIORITY_OPTIONS = list(QoSPolicy.PRIORITY_SCORES.keys())

ROLE_OPTIONS = ["pub", "sub", "pubsub"]
APP_TYPE_OPTIONS = ["sensor", "actuator", "controller", "monitor", "gateway", "processor"]


@dataclass
class StatisticalMetric:
    """Represents statistical parameters for a metric."""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalMetric":
        return cls(
            count=data.get("count", 0),
            mean=float(data.get("mean", 0)),
            median=float(data.get("median", 0)),
            std=float(data.get("std", 0)),
            min=float(data.get("min", 0)),
            max=float(data.get("max", 0)),
            q1=float(data.get("q1", 0)),
            q3=float(data.get("q3", 0)),
            iqr=float(data.get("iqr", 0)),
        )


@dataclass
class NodeStats:
    applications_per_node: StatisticalMetric = field(default_factory=StatisticalMetric)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeStats":
        return cls(
            applications_per_node=StatisticalMetric.from_dict(data.get("applications_per_node", {}))
        )


@dataclass
class CategoricalDistribution:
    """Base class for categorical distribution statistics."""
    total_count: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    mode: str = ""
    mode_count: int = 0
    mode_percentage: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoricalDistribution":
        return cls(
            total_count=data.get("total_count", 0),
            category_counts=data.get("category_counts", {}),
            mode=data.get("mode", ""),
            mode_count=data.get("mode_count", 0),
            mode_percentage=float(data.get("mode_percentage", 0)),
        )
    
    def to_weighted_list(self, default_options: List[str]) -> List[str]:
        """Convert category_counts to weighted list for random sampling."""
        result = []
        for category, count in self.category_counts.items():
            result.extend([category] * count)
        return result if result else default_options


@dataclass
class AppRoleDistribution(CategoricalDistribution):
    """Distribution of application roles."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppRoleDistribution":
        base = CategoricalDistribution.from_dict(data)
        return cls(
            total_count=base.total_count,
            category_counts=base.category_counts,
            mode=base.mode,
            mode_count=base.mode_count,
            mode_percentage=base.mode_percentage,
        )
    
    def to_weighted_list(self, default_options: List[str] = None) -> List[str]:
        return super().to_weighted_list(default_options or ROLE_OPTIONS)


@dataclass
class AppCriticalityDistribution(CategoricalDistribution):
    """Distribution of application criticality."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppCriticalityDistribution":
        base = CategoricalDistribution.from_dict(data)
        return cls(
            total_count=base.total_count,
            category_counts=base.category_counts,
            mode=base.mode,
            mode_count=base.mode_count,
            mode_percentage=base.mode_percentage,
        )
    
    def to_weighted_list(self, default_options: List[str] = None) -> List[bool]:
        """Convert to weighted list with boolean values."""
        result = []
        for category, count in self.category_counts.items():
            if category == "critical":
                result.extend([True] * count)
            else:
                result.extend([False] * count)
        return result if result else [True, False]


@dataclass
class ApplicationStats:
    direct_publish_count: StatisticalMetric = field(default_factory=StatisticalMetric)
    direct_subscribe_count: StatisticalMetric = field(default_factory=StatisticalMetric)
    total_publish_count_including_libraries: StatisticalMetric = field(default_factory=StatisticalMetric)
    total_subscribe_count_including_libraries: StatisticalMetric = field(default_factory=StatisticalMetric)
    app_role_distribution: Optional[AppRoleDistribution] = None
    app_criticality_distribution: Optional[AppCriticalityDistribution] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationStats":
        return cls(
            direct_publish_count=StatisticalMetric.from_dict(data.get("direct_publish_count", {})),
            direct_subscribe_count=StatisticalMetric.from_dict(data.get("direct_subscribe_count", {})),
            total_publish_count_including_libraries=StatisticalMetric.from_dict(
                data.get("total_publish_count_including_libraries", {})
            ),
            total_subscribe_count_including_libraries=StatisticalMetric.from_dict(
                data.get("total_subscribe_count_including_libraries", {})
            ),
            app_role_distribution=AppRoleDistribution.from_dict(
                data.get("app_role_distribution", {})
            ) if "app_role_distribution" in data else None,
            app_criticality_distribution=AppCriticalityDistribution.from_dict(
                data.get("app_criticality_distribution", {})
            ) if "app_criticality_distribution" in data else None,
        )


@dataclass
class LibraryStats:
    applications_using_this_library: StatisticalMetric = field(default_factory=StatisticalMetric)
    direct_publish_count: StatisticalMetric = field(default_factory=StatisticalMetric)
    direct_subscribe_count: StatisticalMetric = field(default_factory=StatisticalMetric)
    total_publish_count_including_libraries: StatisticalMetric = field(default_factory=StatisticalMetric)
    total_subscribe_count_including_libraries: StatisticalMetric = field(default_factory=StatisticalMetric)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LibraryStats":
        return cls(
            applications_using_this_library=StatisticalMetric.from_dict(
                data.get("applications_using_this_library", {})
            ),
            direct_publish_count=StatisticalMetric.from_dict(data.get("direct_publish_count", {})),
            direct_subscribe_count=StatisticalMetric.from_dict(data.get("direct_subscribe_count", {})),
            total_publish_count_including_libraries=StatisticalMetric.from_dict(
                data.get("total_publish_count_including_libraries", {})
            ),
            total_subscribe_count_including_libraries=StatisticalMetric.from_dict(
                data.get("total_subscribe_count_including_libraries", {})
            ),
        )


@dataclass
class TopicStats:
    topic_size_bytes: StatisticalMetric = field(default_factory=StatisticalMetric)
    applications_publishing_to_this_topic: StatisticalMetric = field(default_factory=StatisticalMetric)
    applications_subscribing_to_this_topic: StatisticalMetric = field(default_factory=StatisticalMetric)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicStats":
        return cls(
            topic_size_bytes=StatisticalMetric.from_dict(data.get("topic_size_bytes", {})),
            applications_publishing_to_this_topic=StatisticalMetric.from_dict(
                data.get("applications_publishing_to_this_topic", {})
            ),
            applications_subscribing_to_this_topic=StatisticalMetric.from_dict(
                data.get("applications_subscribing_to_this_topic", {})
            ),
        )


@dataclass
class QosDurabilityDistribution(CategoricalDistribution):
    """Distribution of QoS Durability values across infograms."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QosDurabilityDistribution":
        base = CategoricalDistribution.from_dict(data)
        return cls(
            total_count=base.total_count,
            category_counts=base.category_counts,
            mode=base.mode,
            mode_count=base.mode_count,
            mode_percentage=base.mode_percentage,
        )
    
    def to_weighted_list(self, default_options: List[str] = None) -> List[str]:
        return super().to_weighted_list(default_options or DURABILITY_OPTIONS)


@dataclass
class QosReliabilityDistribution(CategoricalDistribution):
    """Distribution of QoS Reliability values across infograms."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QosReliabilityDistribution":
        base = CategoricalDistribution.from_dict(data)
        return cls(
            total_count=base.total_count,
            category_counts=base.category_counts,
            mode=base.mode,
            mode_count=base.mode_count,
            mode_percentage=base.mode_percentage,
        )
    
    def to_weighted_list(self, default_options: List[str] = None) -> List[str]:
        return super().to_weighted_list(default_options or RELIABILITY_OPTIONS)


@dataclass
class QosTransportPriorityDistribution(CategoricalDistribution):
    """Distribution of QoS Transport Priority values across infograms."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QosTransportPriorityDistribution":
        base = CategoricalDistribution.from_dict(data)
        return cls(
            total_count=base.total_count,
            category_counts=base.category_counts,
            mode=base.mode,
            mode_count=base.mode_count,
            mode_percentage=base.mode_percentage,
        )
    
    def to_weighted_list(self, default_options: List[str] = None) -> List[str]:
        return super().to_weighted_list(default_options or PRIORITY_OPTIONS)


@dataclass
class QosStats:
    """QoS-related statistics."""
    qos_durability_distribution: Optional[QosDurabilityDistribution] = None
    qos_reliability_distribution: Optional[QosReliabilityDistribution] = None
    qos_transport_priority_distribution: Optional[QosTransportPriorityDistribution] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QosStats":
        return cls(
            qos_durability_distribution=QosDurabilityDistribution.from_dict(
                data.get("qos_durability_distribution", {})
            ) if "qos_durability_distribution" in data else None,
            qos_reliability_distribution=QosReliabilityDistribution.from_dict(
                data.get("qos_reliability_distribution", {})
            ) if "qos_reliability_distribution" in data else None,
            qos_transport_priority_distribution=QosTransportPriorityDistribution.from_dict(
                data.get("qos_transport_priority_distribution", {})
            ) if "qos_transport_priority_distribution" in data else None,
        )


@dataclass
class GraphConfig:
    """Configuration for graph generation with statistical distributions."""
    nodes: int = 8
    apps: int = 50
    topics: int = 30
    brokers: int = 3
    libs: int = 10
    seed: int = 42
    
    node_stats: Optional[NodeStats] = None
    application_stats: Optional[ApplicationStats] = None
    library_stats: Optional[LibraryStats] = None
    topic_stats: Optional[TopicStats] = None
    qos_stats: Optional[QosStats] = None
    use_statistics: bool = False
    
    @classmethod
    def from_scale(cls, scale: str, seed: int = 42) -> "GraphConfig":
        preset = SCALE_PRESETS.get(scale, SCALE_PRESETS["medium"])
        return cls(
            apps=preset["apps"],
            topics=preset["topics"],
            brokers=preset["brokers"],
            nodes=preset["nodes"],
            libs=preset["libs"],
            seed=seed,
            use_statistics=False,
        )
    
    @classmethod
    def from_yaml(cls, data: Dict[str, Any]) -> "GraphConfig":
        graph_data = data.get("graph", data)
        counts = graph_data.get("counts", {})
        has_stats = any(key in graph_data for key in ["node_stats", "application_stats", "library_stats", "topic_stats", "qos_stats"])
        
        if has_stats:
            return cls(
                apps=counts.get("applications", graph_data.get("apps", 50)),
                topics=counts.get("topics", graph_data.get("topics", 30)),
                brokers=counts.get("brokers", graph_data.get("brokers", 3)),
                nodes=counts.get("nodes", graph_data.get("nodes", 8)),
                libs=counts.get("libraries", graph_data.get("libs", 10)),
                seed=graph_data.get("seed", 42),
                node_stats=NodeStats.from_dict(graph_data.get("node_stats", {})) if "node_stats" in graph_data else None,
                application_stats=ApplicationStats.from_dict(graph_data.get("application_stats", {})) if "application_stats" in graph_data else None,
                library_stats=LibraryStats.from_dict(graph_data.get("library_stats", {})) if "library_stats" in graph_data else None,
                topic_stats=TopicStats.from_dict(graph_data.get("topic_stats", {})) if "topic_stats" in graph_data else None,
                qos_stats=QosStats.from_dict(graph_data.get("qos_stats", {})) if "qos_stats" in graph_data else None,
                use_statistics=True,
            )
        else:
            return cls(
                apps=graph_data.get("apps", 50),
                topics=graph_data.get("topics", 30),
                brokers=graph_data.get("brokers", 3),
                nodes=graph_data.get("nodes", 8),
                libs=graph_data.get("libs", 10),
                seed=graph_data.get("seed", 42),
                use_statistics=False,
            )
    
    def to_scale_dict(self) -> Dict[str, int]:
        """Convert to scale config dict (excludes seed)."""
        return {
            "apps": self.apps,
            "topics": self.topics,
            "brokers": self.brokers,
            "nodes": self.nodes,
            "libs": self.libs,
        }
