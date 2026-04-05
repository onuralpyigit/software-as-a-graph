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
from .datasets import DomainDataset, get_qos_for_topic, get_app_type_for_name, get_lib_archetype_for_name, get_generic_system_hierarchy


# --- Code-metrics generation parameters by app_type ---
# Each entry maps to ranges for the full code_metrics structure.
# Format: {loc_lo, loc_hi, classes_per_kloc, methods_per_class, fields_per_class,
#          avg_wmc_lo, avg_wmc_hi, avg_lcom_lo, avg_lcom_hi,
#          avg_cbo_lo, avg_cbo_hi, avg_rfc_lo, avg_rfc_hi,
#          avg_fanin_lo, avg_fanin_hi, avg_fanout_lo, avg_fanout_hi}
_CODE_METRICS_PARAMS: Dict[str, Dict[str, Any]] = {
    "sensor":     {"loc": (100,  500),  "cls_per_kloc": (8, 15), "meth_per_cls": (4, 8),  "fld_per_cls": (2, 4),
                   "avg_wmc": (4, 10),   "avg_lcom": (5, 25),
                   "avg_cbo": (2, 6),    "avg_rfc": (8, 20),
                   "avg_fanin": (1, 4),  "avg_fanout": (2, 5)},
    "actuator":   {"loc": (100,  400),  "cls_per_kloc": (8, 14), "meth_per_cls": (4, 7),  "fld_per_cls": (2, 4),
                   "avg_wmc": (4, 9),    "avg_lcom": (5, 20),
                   "avg_cbo": (2, 5),    "avg_rfc": (8, 18),
                   "avg_fanin": (1, 3),  "avg_fanout": (2, 5)},
    "monitor":    {"loc": (300, 1000),  "cls_per_kloc": (10, 18), "meth_per_cls": (5, 9),  "fld_per_cls": (2, 5),
                   "avg_wmc": (8, 16),   "avg_lcom": (10, 40),
                   "avg_cbo": (4, 10),   "avg_rfc": (14, 32),
                   "avg_fanin": (1, 5),  "avg_fanout": (3, 8)},
    "controller": {"loc": (400, 2000),  "cls_per_kloc": (9, 16),  "meth_per_cls": (5, 10), "fld_per_cls": (2, 5),
                   "avg_wmc": (10, 20),  "avg_lcom": (12, 50),
                   "avg_cbo": (5, 14),   "avg_rfc": (16, 40),
                   "avg_fanin": (1, 6),  "avg_fanout": (3, 10)},
    "gateway":    {"loc": (800, 3000),  "cls_per_kloc": (8, 14),  "meth_per_cls": (6, 12), "fld_per_cls": (3, 6),
                   "avg_wmc": (12, 28),  "avg_lcom": (20, 65),
                   "avg_cbo": (6, 16),   "avg_rfc": (20, 50),
                   "avg_fanin": (2, 8),  "avg_fanout": (4, 14)},
    "processor":  {"loc": (600, 2500),  "cls_per_kloc": (8, 15),  "meth_per_cls": (5, 11), "fld_per_cls": (2, 5),
                   "avg_wmc": (10, 25),  "avg_lcom": (15, 55),
                   "avg_cbo": (5, 14),   "avg_rfc": (18, 45),
                   "avg_fanin": (1, 7),  "avg_fanout": (3, 12)},
}
# Fallback used when app_type is not in _CODE_METRICS_PARAMS (should not happen
# in normal generation since APP_TYPE_OPTIONS is exhaustive, but guards against
# future additions to APP_TYPE_OPTIONS that are not yet reflected here).
_DEFAULT_CODE_METRICS_PARAMS: Dict[str, Any] = {
    "loc": (400, 2000),  "cls_per_kloc": (9, 16),  "meth_per_cls": (5, 10), "fld_per_cls": (2, 5),
    "avg_wmc": (8, 22),  "avg_lcom": (10, 50),
    "avg_cbo": (4, 12),  "avg_rfc": (14, 38),
    "avg_fanin": (1, 6), "avg_fanout": (3, 10),
}

# --- Code-metrics generation parameters by library archetype ---
_LIB_CODE_METRICS_PARAMS: Dict[str, Dict[str, Any]] = {
    "utility":    {"loc": (50,   500),  "cls_per_kloc": (10, 20), "meth_per_cls": (3, 7),  "fld_per_cls": (1, 3),
                   "avg_wmc": (3, 8),    "avg_lcom": (3, 15),
                   "avg_cbo": (1, 4),    "avg_rfc": (5, 14),
                   "avg_fanin": (2, 8),  "avg_fanout": (1, 4)},
    "framework":  {"loc": (500, 6000),  "cls_per_kloc": (6, 12),  "meth_per_cls": (6, 14), "fld_per_cls": (3, 7),
                   "avg_wmc": (10, 30),  "avg_lcom": (15, 60),
                   "avg_cbo": (5, 16),   "avg_rfc": (16, 50),
                   "avg_fanin": (3, 12), "avg_fanout": (2, 8)},
    "driver":     {"loc": (100,  900),  "cls_per_kloc": (8, 16),  "meth_per_cls": (4, 8),  "fld_per_cls": (2, 5),
                   "avg_wmc": (5, 12),   "avg_lcom": (5, 25),
                   "avg_cbo": (2, 7),    "avg_rfc": (8, 22),
                   "avg_fanin": (2, 6),  "avg_fanout": (1, 5)},
    "middleware": {"loc": (200, 2500),  "cls_per_kloc": (7, 14),  "meth_per_cls": (5, 10), "fld_per_cls": (2, 5),
                   "avg_wmc": (7, 20),   "avg_lcom": (10, 45),
                   "avg_cbo": (4, 12),   "avg_rfc": (12, 35),
                   "avg_fanin": (2, 8),  "avg_fanout": (2, 6)},
    "protocol":   {"loc": (150, 1200),  "cls_per_kloc": (8, 16),  "meth_per_cls": (4, 9),  "fld_per_cls": (2, 4),
                   "avg_wmc": (5, 14),   "avg_lcom": (5, 30),
                   "avg_cbo": (3, 8),    "avg_rfc": (10, 28),
                   "avg_fanin": (2, 7),  "avg_fanout": (1, 5)},
}
# Weighted random selection of lib archetype (utility and driver are most common)
_LIB_ARCHETYPE_WEIGHTS = ["utility"] * 4 + ["framework"] * 2 + ["driver"] * 3 + ["middleware"] * 2 + ["protocol"] * 2

# --- Criticality levels for statistical generation ---
CRITICALITY_OPTIONS = [True, False]


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

    def _generate_code_metrics(self, app_type: str) -> Dict[str, Any]:
        """Generate a full code_metrics dict for an Application."""
        params = _CODE_METRICS_PARAMS.get(app_type, _DEFAULT_CODE_METRICS_PARAMS)
        return self._build_code_metrics(params)

    def _generate_lib_code_metrics(self, archetype: Optional[str] = None) -> Dict[str, Any]:
        """Generate a full code_metrics dict for a Library."""
        if not archetype or archetype not in _LIB_CODE_METRICS_PARAMS:
            archetype = self.rng.choice(_LIB_ARCHETYPE_WEIGHTS)
        params = _LIB_CODE_METRICS_PARAMS[archetype]
        return self._build_code_metrics(params)

    def _build_code_metrics(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Build a code_metrics dict from parameter ranges."""
        rng = self.rng
        total_loc = rng.randint(*p["loc"])
        cls_per_kloc = rng.uniform(*p["cls_per_kloc"])
        total_classes = max(1, int(round(total_loc / 1000.0 * cls_per_kloc)))
        meth_per_cls = rng.uniform(*p["meth_per_cls"])
        total_methods = max(1, int(round(total_classes * meth_per_cls)))
        fld_per_cls = rng.uniform(*p["fld_per_cls"])
        total_fields = max(0, int(round(total_classes * fld_per_cls)))

        avg_wmc = round(rng.uniform(*p["avg_wmc"]), 2)
        total_wmc = int(round(avg_wmc * total_classes))
        max_wmc = int(round(avg_wmc * rng.uniform(1.8, 3.0)))

        avg_lcom = round(rng.uniform(*p["avg_lcom"]), 2)
        max_lcom = round(avg_lcom * rng.uniform(2.0, 4.5), 1)

        avg_cbo = round(rng.uniform(*p["avg_cbo"]), 2)
        max_cbo = int(round(avg_cbo * rng.uniform(1.5, 2.5)))
        avg_rfc = round(rng.uniform(*p["avg_rfc"]), 2)
        max_rfc = int(round(avg_rfc * rng.uniform(1.5, 2.5)))
        avg_fanin = round(rng.uniform(*p["avg_fanin"]), 2)
        max_fanin = int(round(avg_fanin * rng.uniform(2.0, 4.5)))
        avg_fanout = round(rng.uniform(*p["avg_fanout"]), 2)
        max_fanout = int(round(avg_fanout * rng.uniform(2.0, 3.0)))

        return {
            "size": {
                "total_loc": total_loc,
                "total_classes": total_classes,
                "total_methods": total_methods,
                "total_fields": total_fields,
            },
            "complexity": {
                "total_wmc": total_wmc,
                "avg_wmc": avg_wmc,
                "max_wmc": max_wmc,
            },
            "cohesion": {
                "avg_lcom": avg_lcom,
                "max_lcom": max_lcom,
            },
            "coupling": {
                "avg_cbo": avg_cbo,
                "max_cbo": max_cbo,
                "avg_rfc": avg_rfc,
                "max_rfc": max_rfc,
                "avg_fanin": avg_fanin,
                "max_fanin": max_fanin,
                "avg_fanout": avg_fanout,
                "max_fanout": max_fanout,
            },
        }

    def generate(self) -> Dict[str, Any]:
        c = self.config
        
        # 1. Generate base entities using Core Models
        domain_ds = None
        if c.domain:
            domain_ds = DomainDataset(c.domain, self.rng)
            
        nodes = [
            Node(id=f"N{i}", name=domain_ds.get_node_name() if domain_ds else f"Node-{i}") 
            for i in range(c.nodes)
        ]
        brokers = [
            Broker(id=f"B{i}", name=domain_ds.get_broker_name() if domain_ds else f"Broker-{i}") 
            for i in range(c.brokers)
        ]
        
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
            
            topic_name = domain_ds.get_topic_name() if domain_ds else f"Topic-{i}"
            
            if domain_ds:
                durability, reliability, transport_priority = get_qos_for_topic(
                    topic_name, c.domain, c.scenario
                )
            else:
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
                name=topic_name,
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
                criticality = self.rng.choice(CRITICALITY_OPTIONS)
            
            app_name = domain_ds.get_app_name() if domain_ds else f"App-{i}"
            if domain_ds:
                app_type = get_app_type_for_name(app_name)
            else:
                app_type = self.rng.choice(APP_TYPE_OPTIONS)
                
            code_metrics = self._generate_code_metrics(app_type)
            hierarchy = domain_ds.get_system_hierarchy() if domain_ds else get_generic_system_hierarchy(self.rng)
            apps.append(Application(
                id=f"A{i}",
                name=app_name,
                role=role,
                app_type=app_type,
                criticality=criticality,
                version=f"{self.rng.randint(1, 3)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
                system_hierarchy=hierarchy,
                code_metrics=code_metrics,
            ))

        libs: List[Library] = []
        for i in range(c.libs):
            lib_name = domain_ds.get_library_name() if domain_ds else f"Lib-{i}"
            if domain_ds:
                archetype = get_lib_archetype_for_name(lib_name)
                lib_code_metrics = self._generate_lib_code_metrics(archetype)
            else:
                lib_code_metrics = self._generate_lib_code_metrics()

            lib_hierarchy = domain_ds.get_system_hierarchy() if domain_ds else get_generic_system_hierarchy(self.rng)
            libs.append(Library(
                id=f"L{i}",
                name=lib_name,
                version=f"{self.rng.randint(0, 2)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
                system_hierarchy=lib_hierarchy,
                code_metrics=lib_code_metrics,
            ))

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
            # Assign each topic to 1 or 2 brokers.
            # With len(brokers)>=2, there is a 30% chance of a second broker (redundancy).
            # This makes broker failure impact proportional to structural importance
            # rather than being a random lottery, improving Spearman ρ in validation.
            primary_broker = self.rng.choice(brokers)
            routes.append(self._make_edge(primary_broker, topic))
            if len(brokers) >= 2 and self.rng.random() < 0.30:
                other_brokers = [b for b in brokers if b.id != primary_broker.id]
                if other_brokers:
                    secondary_broker = self.rng.choice(other_brokers)
                    routes.append(self._make_edge(secondary_broker, topic))

        # Guard: ensure every broker routes at least one topic so that no broker
        # is invisible to betweenness and ROUTES-based metrics (which would
        # produce anomalously low RMAV scores).  This can happen when the topic
        # count is small relative to the broker count (e.g. custom YAML configs
        # with many brokers and few topics).  Assign each stranded broker to a
        # topic in round-robin order so the result is deterministic given the seed.
        routed_broker_ids = {edge["from"] for edge in routes}
        unrouted_brokers = [b for b in brokers if b.id not in routed_broker_ids]
        for idx, broker in enumerate(unrouted_brokers):
            topic = topics[idx % len(topics)]
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
                "generation_mode": "statistical" if c.use_statistics else "random",
                "domain": c.domain,
                "scenario": c.scenario,
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
