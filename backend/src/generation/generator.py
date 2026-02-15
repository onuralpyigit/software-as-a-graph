"""
Statistical Graph Generator
"""
import random
from typing import Dict, Any, List, Optional, Union, Tuple

from src.core.models import (
    Application,
    Broker,
    Node,
    Topic,
    Library,
    QoSPolicy,
)
from .models import (
    GraphConfig,
    StatisticalMetric,
    DURABILITY_OPTIONS,
    RELIABILITY_OPTIONS,
    PRIORITY_OPTIONS,
    ROLE_OPTIONS,
    APP_TYPE_OPTIONS,
)


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

    def _make_edge(self, src: Any, tgt: Any) -> Dict[str, str]:
        return {"from": src.id, "to": tgt.id}

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
        
        # 1. Generate base entities using Core Models
        nodes = [Node(id=f"N{i}", name=f"Node-{i}") for i in range(c.nodes)]
        brokers = [Broker(id=f"B{i}", name=f"Broker-{i}") for i in range(c.brokers)]
        
        topics: List[Topic] = []
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
            if c.topic_stats and c.topic_stats.topic_size_bytes.mean > 0:
                size = self._sample_from_distribution(c.topic_stats.topic_size_bytes)
            else:
                size = self.rng.randint(64, 65536)
            
            if durability_pool and i < len(durability_pool):
                durability = durability_pool[i]
            elif durability_pool:
                durability = self.rng.choice(durability_pool)
            else:
                durability = self.rng.choice(DURABILITY_OPTIONS)
            
            if reliability_pool and i < len(reliability_pool):
                reliability = reliability_pool[i]
            elif reliability_pool:
                reliability = self.rng.choice(reliability_pool)
            else:
                reliability = self.rng.choice(RELIABILITY_OPTIONS)
            
            if priority_pool and i < len(priority_pool):
                transport_priority = priority_pool[i]
            elif priority_pool:
                transport_priority = self.rng.choice(priority_pool)
            else:
                transport_priority = self.rng.choice(PRIORITY_OPTIONS)
            
            topics.append(Topic(
                id=f"T{i}",
                name=f"Topic-{i}",
                size=size,
                qos=QoSPolicy(
                    durability=durability,
                    reliability=reliability,
                    transport_priority=transport_priority,
                )
            ))

        apps: List[Application] = []
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
            if role_pool and i < len(role_pool):
                role = role_pool[i]
            elif role_pool:
                role = self.rng.choice(role_pool)
            else:
                role = self.rng.choice(ROLE_OPTIONS)
            
            if criticality_pool and i < len(criticality_pool):
                criticality = criticality_pool[i]
            elif criticality_pool:
                criticality = self.rng.choice(criticality_pool)
            else:
                criticality = self.rng.choice([True, False])
            
            apps.append(Application(
                id=f"A{i}",
                name=f"App-{i}",
                role=role,
                app_type=self.rng.choice(APP_TYPE_OPTIONS),
                criticality=criticality,
                version=f"{self.rng.randint(1, 3)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            ))

        libs = [
            Library(
                id=f"L{i}",
                name=f"Lib-{i}",
                version=f"{self.rng.randint(0, 2)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
            )
            for i in range(c.libs)
        ]

        # 2. Relationships
        runs_on = []
        if c.node_stats and c.node_stats.applications_per_node.mean > 0:
            app_counts_per_node = self._generate_values_from_distribution(
                c.node_stats.applications_per_node, len(nodes)
            )
            app_index = 0
            for node_idx, target_count in enumerate(app_counts_per_node):
                for _ in range(target_count):
                    if app_index < len(apps):
                        runs_on.append(self._make_edge(apps[app_index], nodes[node_idx]))
                        app_index += 1
            while app_index < len(apps):
                host = self.rng.choice(nodes)
                runs_on.append(self._make_edge(apps[app_index], host))
                app_index += 1
        else:
            for app in apps:
                host = self.rng.choice(nodes)
                runs_on.append(self._make_edge(app, host))

        for broker in brokers:
            host = self.rng.choice(nodes)
            runs_on.append(self._make_edge(broker, host))

        routes = []
        for topic in topics:
            broker = self.rng.choice(brokers)
            routes.append(self._make_edge(broker, topic))

        publishes = []
        subscribes = []
        uses = []
        
        for lib in libs:
            self._uses_graph[lib.id] = []
            self._direct_pub_counts[lib.id] = 0
            self._direct_sub_counts[lib.id] = 0
        for app in apps:
            self._uses_graph[app.id] = []
            self._direct_pub_counts[app.id] = 0
            self._direct_sub_counts[app.id] = 0

        for lib in libs:
            if len(libs) > 1 and self.rng.random() < 0.3:
                n_lib_deps = self.rng.randint(0, min(2, len(libs) - 1))
                other_libs = [l for l in libs if l.id != lib.id]
                if n_lib_deps > 0 and other_libs:
                    targets = self.rng.sample(other_libs, k=min(n_lib_deps, len(other_libs)))
                    for t in targets:
                        uses.append(self._make_edge(lib, t))
                        self._uses_graph[lib.id].append(t.id)

        if c.library_stats and c.library_stats.direct_publish_count.mean > 0:
            for lib in libs:
                pub_count = self._sample_from_distribution(c.library_stats.direct_publish_count)
                pub_count = min(pub_count, len(topics))
                if pub_count > 0:
                    pub_topics = self.rng.sample(topics, k=pub_count)
                    for t in pub_topics:
                        publishes.append(self._make_edge(lib, t))
                    self._direct_pub_counts[lib.id] = pub_count
                
                sub_count = self._sample_from_distribution(c.library_stats.direct_subscribe_count)
                sub_count = min(sub_count, len(topics))
                if sub_count > 0:
                    sub_topics = self.rng.sample(topics, k=sub_count)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(lib, t))
                    self._direct_sub_counts[lib.id] = sub_count

        if c.library_stats and c.library_stats.applications_using_this_library.mean > 0:
            for lib in libs:
                usage_count = self._sample_from_distribution(c.library_stats.applications_using_this_library)
                usage_count = min(usage_count, len(apps))
                if usage_count > 0:
                    using_apps = self.rng.sample(apps, k=usage_count)
                    for app in using_apps:
                        if lib.id not in self._uses_graph[app.id]:
                            uses.append(self._make_edge(app, lib))
                            self._uses_graph[app.id].append(lib.id)
        else:
            for app in apps:
                if libs:
                    n_uses = self.rng.randint(0, min(3, len(libs)))
                    targets = self.rng.sample(libs, k=n_uses)
                    for t in targets:
                        if t.id not in self._uses_graph[app.id]:
                            uses.append(self._make_edge(app, t))
                            self._uses_graph[app.id].append(t.id)

        if c.application_stats and c.application_stats.total_publish_count_including_libraries.mean > 0:
            for app in apps:
                inherited_pub = self._get_inherited_pub_count(app.id)
                inherited_sub = self._get_inherited_sub_count(app.id)
                
                total_pub_target = self._sample_from_distribution(
                    c.application_stats.total_publish_count_including_libraries
                )
                total_sub_target = self._sample_from_distribution(
                    c.application_stats.total_subscribe_count_including_libraries
                )
                
                direct_pub_count = max(0, total_pub_target - inherited_pub)
                direct_sub_count = max(0, total_sub_target - inherited_sub)
                
                direct_pub_count = min(direct_pub_count, len(topics))
                direct_sub_count = min(direct_sub_count, len(topics))
                
                if direct_pub_count > 0:
                    pub_topics = self.rng.sample(topics, k=direct_pub_count)
                    for t in pub_topics:
                        publishes.append(self._make_edge(app, t))
                    self._direct_pub_counts[app.id] = direct_pub_count
                
                if direct_sub_count > 0:
                    sub_topics = self.rng.sample(topics, k=direct_sub_count)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(app, t))
                    self._direct_sub_counts[app.id] = direct_sub_count
                    
        elif c.application_stats and c.application_stats.direct_publish_count.mean > 0:
            for app in apps:
                pub_count = self._sample_from_distribution(c.application_stats.direct_publish_count)
                pub_count = min(pub_count, len(topics))
                if pub_count > 0:
                    pub_topics = self.rng.sample(topics, k=pub_count)
                    for t in pub_topics:
                        publishes.append(self._make_edge(app, t))
                    self._direct_pub_counts[app.id] = pub_count
                
                sub_count = self._sample_from_distribution(c.application_stats.direct_subscribe_count)
                sub_count = min(sub_count, len(topics))
                if sub_count > 0:
                    sub_topics = self.rng.sample(topics, k=sub_count)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(app, t))
                    self._direct_sub_counts[app.id] = sub_count
                    
        elif c.topic_stats and c.topic_stats.applications_publishing_to_this_topic.mean > 0:
            for topic in topics:
                pub_count = self._sample_from_distribution(c.topic_stats.applications_publishing_to_this_topic)
                pub_count = min(pub_count, len(apps))
                if pub_count > 0:
                    pubs = self.rng.sample(apps, k=pub_count)
                    for p in pubs:
                        publishes.append(self._make_edge(p, topic))
                        self._direct_pub_counts[p.id] = self._direct_pub_counts.get(p.id, 0) + 1
                
                sub_count = self._sample_from_distribution(c.topic_stats.applications_subscribing_to_this_topic)
                sub_count = min(sub_count, len(apps))
                if sub_count > 0:
                    subs = self.rng.sample(apps, k=sub_count)
                    for s in subs:
                        subscribes.append(self._make_edge(s, topic))
                        self._direct_sub_counts[s.id] = self._direct_sub_counts.get(s.id, 0) + 1
        else:
            for topic in topics:
                k_pubs = self.rng.randint(1, max(2, min(5, len(apps))))
                k_subs = self.rng.randint(1, max(2, min(8, len(apps))))
                pubs = self.rng.sample(apps, k=k_pubs)
                subs = self.rng.sample(apps, k=k_subs)
                for p in pubs:
                    publishes.append(self._make_edge(p, topic))
                for s in subs:
                    subscribes.append(self._make_edge(s, topic))

        connects = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.rng.random() < 0.3:
                    connects.append(self._make_edge(nodes[i], nodes[j]))

        return {
            "metadata": {
                "scale": c.to_scale_dict(),
                "seed": c.seed,
                "generation_mode": "statistical" if c.use_statistics else "legacy",
            },
            "nodes": [n.to_dict() for n in nodes],
            "brokers": [b.to_dict() for b in brokers],
            "topics": [t.to_dict() for t in topics],
            "applications": [a.to_dict() for a in apps],
            "libraries": [l.to_dict() for l in libs],
            "relationships": {
                "runs_on": runs_on,
                "routes": routes,
                "publishes_to": publishes,
                "subscribes_to": subscribes,
                "connects_to": connects,
                "uses": uses
            }
        }
