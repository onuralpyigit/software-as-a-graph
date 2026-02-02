#!/usr/bin/env python3
"""
CLI script to generate pub-sub graph data with statistical distribution support.

Example usage:
    python generate_graph.py --scale medium --output output/graph.json --seed 42
    python generate_graph.py --config input/graph_config.yaml --output output/graph.json
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml


# =============================================================================
# CONSTANTS
# =============================================================================

SCALE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":   {"apps": 5,   "topics": 5,   "brokers": 1, "nodes": 2,  "libs": 2},
    "small":  {"apps": 15,  "topics": 10,  "brokers": 2, "nodes": 4,  "libs": 5},
    "medium": {"apps": 50,  "topics": 30,  "brokers": 3, "nodes": 8,  "libs": 10},
    "large":  {"apps": 150, "topics": 100, "brokers": 6, "nodes": 20, "libs": 30},
    "xlarge": {"apps": 500, "topics": 300, "brokers": 10, "nodes": 50, "libs": 100},
}

DURABILITY_OPTIONS = ["volatile", "transient_local", "transient", "persistent"]
RELIABILITY_OPTIONS = ["best_effort", "reliable"]
PRIORITY_OPTIONS = ["low", "medium", "high", "critical"]
ROLE_OPTIONS = ["pub", "sub", "pubsub"]
APP_TYPE_OPTIONS = ["sensor", "actuator", "controller", "monitor", "gateway", "processor"]


# =============================================================================
# STATISTICAL DATA CLASSES
# =============================================================================

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
    infogram_size_bytes: StatisticalMetric = field(default_factory=StatisticalMetric)
    applications_publishing_to_this_infogram: StatisticalMetric = field(default_factory=StatisticalMetric)
    applications_subscribing_to_this_infogram: StatisticalMetric = field(default_factory=StatisticalMetric)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicStats":
        return cls(
            infogram_size_bytes=StatisticalMetric.from_dict(data.get("infogram_size_bytes", {})),
            applications_publishing_to_this_infogram=StatisticalMetric.from_dict(
                data.get("applications_publishing_to_this_infogram", {})
            ),
            applications_subscribing_to_this_infogram=StatisticalMetric.from_dict(
                data.get("applications_subscribing_to_this_infogram", {})
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


def load_config(path: Path) -> GraphConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return GraphConfig.from_yaml(data)


# =============================================================================
# GRAPH GENERATOR
# =============================================================================

class StatisticalGraphGenerator:
    """Generates graphs using statistical distributions from config."""
    
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        # Track pub/sub counts per entity for recursive calculation
        self._direct_pub_counts: Dict[str, int] = {}
        self._direct_sub_counts: Dict[str, int] = {}
        self._uses_graph: Dict[str, List[str]] = {}  # entity_id -> [lib_ids it uses]

    def _sample_from_distribution(self, metric: StatisticalMetric, as_int: bool = True) -> float:
        """Sample a value from a statistical distribution."""
        if metric.std <= 0:
            value = metric.mean
        else:
            value = self.rng.gauss(metric.mean, metric.std)
        
        value = max(metric.min, min(metric.max, value))
        
        if as_int:
            return int(round(value))
        return value

    def _generate_values_from_distribution(self, metric: StatisticalMetric, count: int, as_int: bool = True) -> List[float]:
        return [self._sample_from_distribution(metric, as_int) for _ in range(count)]

    def _make_edge(self, src_id: str, tgt_id: str) -> Dict[str, str]:
        return {"from": src_id, "to": tgt_id}

    def _get_all_used_libs_recursive(self, entity_id: str, visited: Optional[set] = None) -> set:
        """Get all libraries used by an entity (app or lib), recursively."""
        if visited is None:
            visited = set()
        
        if entity_id in visited:
            return set()  # Avoid cycles
        
        visited.add(entity_id)
        result = set()
        
        for lib_id in self._uses_graph.get(entity_id, []):
            result.add(lib_id)
            # Recursively get libs used by this lib
            result.update(self._get_all_used_libs_recursive(lib_id, visited))
        
        return result

    def _get_inherited_pub_count(self, entity_id: str) -> int:
        """Get total pub count from all recursively used libraries."""
        used_libs = self._get_all_used_libs_recursive(entity_id)
        return sum(self._direct_pub_counts.get(lib_id, 0) for lib_id in used_libs)

    def _get_inherited_sub_count(self, entity_id: str) -> int:
        """Get total sub count from all recursively used libraries."""
        used_libs = self._get_all_used_libs_recursive(entity_id)
        return sum(self._direct_sub_counts.get(lib_id, 0) for lib_id in used_libs)

    def generate(self) -> Dict[str, Any]:
        c = self.config
        
        # 1. Generate base entities
        nodes = [{"id": f"N{i}", "name": f"Node-{i}"} for i in range(c.nodes)]
        brokers = [{"id": f"B{i}", "name": f"Broker-{i}"} for i in range(c.brokers)]
        
        # Topics with statistical size and QoS distribution
        topics = []
        
        # Prepare QoS distribution lists if available
        durability_pool = None
        reliability_pool = None
        priority_pool = None
        
        if c.qos_stats:
            if c.qos_stats.qos_durability_distribution:
                durability_pool = c.qos_stats.qos_durability_distribution.to_weighted_list()
                self.rng.shuffle(durability_pool)
            if c.qos_stats.qos_reliability_distribution:
                reliability_pool = c.qos_stats.qos_reliability_distribution.to_weighted_list()
                self.rng.shuffle(reliability_pool)
            if c.qos_stats.qos_transport_priority_distribution:
                priority_pool = c.qos_stats.qos_transport_priority_distribution.to_weighted_list()
                self.rng.shuffle(priority_pool)
        
        for i in range(c.topics):
            # Size from distribution
            if c.topic_stats and c.topic_stats.infogram_size_bytes.mean > 0:
                size = self._sample_from_distribution(c.topic_stats.infogram_size_bytes)
            else:
                size = self.rng.randint(64, 65536)
            
            # Durability from distribution or random
            if durability_pool and i < len(durability_pool):
                durability = durability_pool[i]
            elif durability_pool:
                durability = self.rng.choice(durability_pool)
            else:
                durability = self.rng.choice(DURABILITY_OPTIONS)
            
            # Reliability from distribution or random
            if reliability_pool and i < len(reliability_pool):
                reliability = reliability_pool[i]
            elif reliability_pool:
                reliability = self.rng.choice(reliability_pool)
            else:
                reliability = self.rng.choice(RELIABILITY_OPTIONS)
            
            # Transport priority from distribution or random
            if priority_pool and i < len(priority_pool):
                transport_priority = priority_pool[i]
            elif priority_pool:
                transport_priority = self.rng.choice(priority_pool)
            else:
                transport_priority = self.rng.choice(PRIORITY_OPTIONS)
            
            topics.append({
                "id": f"T{i}",
                "name": f"Topic-{i}",
                "size": size,
                "qos": {
                    "durability": durability,
                    "reliability": reliability,
                    "transport_priority": transport_priority,
                },
            })

        # Applications - with role and criticality distribution
        apps = []
        
        # Prepare role and criticality pools if available
        role_pool = None
        criticality_pool = None
        
        if c.application_stats:
            if c.application_stats.app_role_distribution:
                role_pool = c.application_stats.app_role_distribution.to_weighted_list()
                self.rng.shuffle(role_pool)
            if c.application_stats.app_criticality_distribution:
                criticality_pool = c.application_stats.app_criticality_distribution.to_weighted_list()
                self.rng.shuffle(criticality_pool)
        
        for i in range(c.apps):
            # Role from distribution or random
            if role_pool and i < len(role_pool):
                role = role_pool[i]
            elif role_pool:
                role = self.rng.choice(role_pool)
            else:
                role = self.rng.choice(ROLE_OPTIONS)
            
            # Criticality from distribution or random
            if criticality_pool and i < len(criticality_pool):
                criticality = criticality_pool[i]
            elif criticality_pool:
                criticality = self.rng.choice(criticality_pool)
            else:
                criticality = self.rng.choice([True, False])
            
            apps.append({
                "id": f"A{i}",
                "name": f"App-{i}",
                "role": role,
                "app_type": self.rng.choice(APP_TYPE_OPTIONS),
                "criticality": criticality,
                "version": f"{self.rng.randint(1, 3)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            })

        # Libraries
        libs = [
            {
                "id": f"L{i}",
                "name": f"Lib-{i}",
                "version": f"{self.rng.randint(0, 2)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            }
            for i in range(c.libs)
        ]

        # 2. Generate Relationships
        runs_on = []
        
        # RUNS_ON: Distribute apps across nodes based on applications_per_node stat
        if c.node_stats and c.node_stats.applications_per_node.mean > 0:
            app_counts_per_node = self._generate_values_from_distribution(
                c.node_stats.applications_per_node, len(nodes)
            )
            app_index = 0
            for node_idx, target_count in enumerate(app_counts_per_node):
                for _ in range(target_count):
                    if app_index < len(apps):
                        runs_on.append(self._make_edge(apps[app_index]["id"], nodes[node_idx]["id"]))
                        app_index += 1
            while app_index < len(apps):
                host = self.rng.choice(nodes)
                runs_on.append(self._make_edge(apps[app_index]["id"], host["id"]))
                app_index += 1
        else:
            for app in apps:
                host = self.rng.choice(nodes)
                runs_on.append(self._make_edge(app["id"], host["id"]))

        # Brokers also run on nodes
        for broker in brokers:
            host = self.rng.choice(nodes)
            runs_on.append(self._make_edge(broker["id"], host["id"]))

        # ROUTES: Brokers -> Topics
        routes = []
        for topic in topics:
            broker = self.rng.choice(brokers)
            routes.append(self._make_edge(broker["id"], topic["id"]))

        # PUB/SUB relationships - Now using total_including_libraries metrics
        # Strategy:
        # 1. First create USES relationships (lib->lib and app->lib)
        # 2. Then create library direct pub/sub
        # 3. Then create app direct pub/sub based on total - inherited
        
        publishes = []
        subscribes = []
        uses = []
        
        # Initialize tracking structures
        for lib in libs:
            self._uses_graph[lib["id"]] = []
            self._direct_pub_counts[lib["id"]] = 0
            self._direct_sub_counts[lib["id"]] = 0
        for app in apps:
            self._uses_graph[app["id"]] = []
            self._direct_pub_counts[app["id"]] = 0
            self._direct_sub_counts[app["id"]] = 0

        # STEP 1: Create Library -> Library dependencies first (needed for recursive calculation)
        for lib in libs:
            if len(libs) > 1 and self.rng.random() < 0.3:
                # Each lib can use 0-2 other libs
                n_lib_deps = self.rng.randint(0, min(2, len(libs) - 1))
                other_libs = [l for l in libs if l["id"] != lib["id"]]
                if n_lib_deps > 0 and other_libs:
                    targets = self.rng.sample(other_libs, k=min(n_lib_deps, len(other_libs)))
                    for t in targets:
                        uses.append(self._make_edge(lib["id"], t["id"]))
                        self._uses_graph[lib["id"]].append(t["id"])

        # STEP 2: Create Library direct pub/sub
        if c.library_stats and c.library_stats.direct_publish_count.mean > 0:
            for lib in libs:
                pub_count = self._sample_from_distribution(c.library_stats.direct_publish_count)
                pub_count = min(pub_count, len(topics))
                if pub_count > 0:
                    pub_topics = self.rng.sample(topics, k=pub_count)
                    for t in pub_topics:
                        publishes.append(self._make_edge(lib["id"], t["id"]))
                    self._direct_pub_counts[lib["id"]] = pub_count
                
                sub_count = self._sample_from_distribution(c.library_stats.direct_subscribe_count)
                sub_count = min(sub_count, len(topics))
                if sub_count > 0:
                    sub_topics = self.rng.sample(topics, k=sub_count)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(lib["id"], t["id"]))
                    self._direct_sub_counts[lib["id"]] = sub_count

        # STEP 3: Create Application -> Library USES relationships
        if c.library_stats and c.library_stats.applications_using_this_library.mean > 0:
            for lib in libs:
                usage_count = self._sample_from_distribution(c.library_stats.applications_using_this_library)
                usage_count = min(usage_count, len(apps))
                if usage_count > 0:
                    using_apps = self.rng.sample(apps, k=usage_count)
                    for app in using_apps:
                        if lib["id"] not in self._uses_graph[app["id"]]:
                            uses.append(self._make_edge(app["id"], lib["id"]))
                            self._uses_graph[app["id"]].append(lib["id"])
        else:
            for app in apps:
                if libs:
                    n_uses = self.rng.randint(0, min(3, len(libs)))
                    targets = self.rng.sample(libs, k=n_uses)
                    for t in targets:
                        if t["id"] not in self._uses_graph[app["id"]]:
                            uses.append(self._make_edge(app["id"], t["id"]))
                            self._uses_graph[app["id"]].append(t["id"])

        # STEP 4: Create Application pub/sub using total_including_libraries metrics
        # total_pub = direct_pub + inherited_pub (from all recursively used libs)
        # So: direct_pub = total_pub - inherited_pub
        
        if c.application_stats and c.application_stats.total_publish_count_including_libraries.mean > 0:
            # Use total metrics - calculate direct from total minus inherited
            for app in apps:
                # Calculate inherited counts from used libraries
                inherited_pub = self._get_inherited_pub_count(app["id"])
                inherited_sub = self._get_inherited_sub_count(app["id"])
                
                # Sample total targets
                total_pub_target = self._sample_from_distribution(
                    c.application_stats.total_publish_count_including_libraries
                )
                total_sub_target = self._sample_from_distribution(
                    c.application_stats.total_subscribe_count_including_libraries
                )
                
                # Calculate direct counts (total - inherited, minimum 0)
                direct_pub_count = max(0, total_pub_target - inherited_pub)
                direct_sub_count = max(0, total_sub_target - inherited_sub)
                
                # Limit to available topics
                direct_pub_count = min(direct_pub_count, len(topics))
                direct_sub_count = min(direct_sub_count, len(topics))
                
                if direct_pub_count > 0:
                    pub_topics = self.rng.sample(topics, k=direct_pub_count)
                    for t in pub_topics:
                        publishes.append(self._make_edge(app["id"], t["id"]))
                    self._direct_pub_counts[app["id"]] = direct_pub_count
                
                if direct_sub_count > 0:
                    sub_topics = self.rng.sample(topics, k=direct_sub_count)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(app["id"], t["id"]))
                    self._direct_sub_counts[app["id"]] = direct_sub_count
                    
        elif c.application_stats and c.application_stats.direct_publish_count.mean > 0:
            # Fallback: Use direct metrics if total metrics not available
            for app in apps:
                pub_count = self._sample_from_distribution(c.application_stats.direct_publish_count)
                pub_count = min(pub_count, len(topics))
                if pub_count > 0:
                    pub_topics = self.rng.sample(topics, k=pub_count)
                    for t in pub_topics:
                        publishes.append(self._make_edge(app["id"], t["id"]))
                    self._direct_pub_counts[app["id"]] = pub_count
                
                sub_count = self._sample_from_distribution(c.application_stats.direct_subscribe_count)
                sub_count = min(sub_count, len(topics))
                if sub_count > 0:
                    sub_topics = self.rng.sample(topics, k=sub_count)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(app["id"], t["id"]))
                    self._direct_sub_counts[app["id"]] = sub_count
                    
        elif c.topic_stats and c.topic_stats.applications_publishing_to_this_infogram.mean > 0:
            # Topic-centric approach
            for topic in topics:
                pub_count = self._sample_from_distribution(c.topic_stats.applications_publishing_to_this_infogram)
                pub_count = min(pub_count, len(apps))
                if pub_count > 0:
                    pubs = self.rng.sample(apps, k=pub_count)
                    for p in pubs:
                        publishes.append(self._make_edge(p["id"], topic["id"]))
                        self._direct_pub_counts[p["id"]] = self._direct_pub_counts.get(p["id"], 0) + 1
                
                sub_count = self._sample_from_distribution(c.topic_stats.applications_subscribing_to_this_infogram)
                sub_count = min(sub_count, len(apps))
                if sub_count > 0:
                    subs = self.rng.sample(apps, k=sub_count)
                    for s in subs:
                        subscribes.append(self._make_edge(s["id"], topic["id"]))
                        self._direct_sub_counts[s["id"]] = self._direct_sub_counts.get(s["id"], 0) + 1
        else:
            # Legacy fallback
            for topic in topics:
                k_pubs = self.rng.randint(1, max(2, min(5, len(apps))))
                k_subs = self.rng.randint(1, max(2, min(8, len(apps))))
                pubs = self.rng.sample(apps, k=k_pubs)
                subs = self.rng.sample(apps, k=k_subs)
                for p in pubs:
                    publishes.append(self._make_edge(p["id"], topic["id"]))
                for s in subs:
                    subscribes.append(self._make_edge(s["id"], topic["id"]))

        # CONNECTS_TO: Mesh links between Nodes
        connects = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.rng.random() < 0.3:
                    connects.append(self._make_edge(nodes[i]["id"], nodes[j]["id"]))

        return {
            "metadata": {
                "scale": {"apps": c.apps, "topics": c.topics, "brokers": c.brokers, "nodes": c.nodes, "libs": c.libs},
                "seed": c.seed,
                "generation_mode": "statistical" if c.use_statistics else "legacy",
            },
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": apps,
            "libraries": libs,
            "relationships": {
                "runs_on": runs_on,
                "routes": routes,
                "publishes_to": publishes,
                "subscribes_to": subscribes,
                "connects_to": connects,
                "uses": uses
            }
        }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main entry point for graph generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Pub-Sub Graph Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--scale",
        default=None,
        choices=["tiny", "small", "medium", "large", "xlarge"],
        help="Scale of the graph to generate (preset)",
    )
    config_group.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (ignored if --config is used)",
    )
    args = parser.parse_args()

    # Determine configuration source
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
            sys.exit(1)
        try:
            config = load_config(args.config)
            print(f"Loading configuration from '{args.config}'...")
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
        config_desc = f"config={args.config}"
    else:
        scale = args.scale or "medium"
        config = GraphConfig.from_scale(scale, args.seed)
        config_desc = f"scale={scale}, seed={args.seed}"
    
    print(f"Generating graph ({config_desc})...")
    print(f"Mode: {'statistical' if config.use_statistics else 'legacy'}")

    try:
        generator = StatisticalGraphGenerator(config)
        data = generator.generate()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Success! Saved to {args.output}")
        print(
            f"Stats: "
            f"{len(data['nodes'])} Nodes, "
            f"{len(data['applications'])} Apps, "
            f"{len(data.get('libraries', []))} Libs, "
            f"{len(data['topics'])} Topics, "
            f"{len(data['brokers'])} Brokers"
        )
        print(f"Relationships: {sum(len(v) for v in data['relationships'].values())} edges")

    except Exception as e:
        print(f"Error generating graph: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()