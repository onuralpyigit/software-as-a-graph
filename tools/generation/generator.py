"""
Statistical Graph Generator
"""
import random
import logging
from collections import Counter
from typing import Dict, Any, List, Optional, Union, Tuple

from saag.core.models import (
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
    APP_TYPE_OPTIONS,
    APP_PRIORITY_OPTIONS,
    APP_HOTSTANDBY_OPTIONS,
    APP_USER_ROLE_OPTIONS,
)
from .datasets import (
    DomainDataset,
    get_qos_for_topic,
    get_app_type_for_name,
    get_lib_archetype_for_name,
    get_generic_system_hierarchy,
    SYSTEM_HIERARCHY_POOLS,
    GENERIC_HIERARCHY_POOL,
)


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

# Maps app_type → preferred QoS attributes for topic selection bias.
# Used by _partition_topics_by_qos_affinity() to steer which topics are drawn into
# the "preferred" tier of _sample_biased(), making QoS semantics structurally
# coherent: gateways/controllers prefer RELIABLE/HIGH topics; sensors stay on
# BEST_EFFORT/LOW topics.  Falls back to the cluster pool when the preferred
# set is empty (e.g. all topics in the cluster share the same QoS level).
_APP_TYPE_QOS_AFFINITY: Dict[str, Dict[str, List[str]]] = {
    "gateway":    {"reliability": ["RELIABLE"],                "priority": ["HIGH", "HIGHEST", "CRITICAL"]},
    "controller": {"reliability": ["RELIABLE"],                "priority": ["HIGH", "HIGHEST", "CRITICAL"]},
    "processor":  {"reliability": ["RELIABLE", "BEST_EFFORT"], "priority": ["MEDIUM", "HIGH"]},
    "monitor":    {"reliability": ["RELIABLE", "BEST_EFFORT"], "priority": ["LOW", "MEDIUM"]},
    "actuator":   {"reliability": ["RELIABLE"],                "priority": ["MEDIUM", "HIGH"]},
    "sensor":     {"reliability": ["BEST_EFFORT"],             "priority": ["LOW", "MEDIUM"]},
}

# Maps app_type → (can_publish, can_subscribe) messaging capability.
# Derived from domain semantics:
#   sensor    — originates data, never consumes
#   actuator  — receives commands, never produces
#   monitor   — observes data streams, never produces
#   controller — issues commands and reads state feedback
#   gateway   — bridges data flows in both directions
#   processor — transforms and re-emits data
#   service   — general-purpose; bidirectional by default
_APP_TYPE_MESSAGING_CAPABILITY: Dict[str, tuple] = {
    "sensor":     (True,  False),   # publish only
    "actuator":   (False, True),    # subscribe only
    "monitor":    (False, True),    # subscribe only
    "controller": (True,  True),    # both
    "gateway":    (True,  True),    # both
    "processor":  (True,  True),    # both
    "service":    (True,  True),    # both
}

def _can_publish(app_type: str) -> bool:
    """Return True if an app of this type is allowed to publish to topics."""
    return _APP_TYPE_MESSAGING_CAPABILITY.get(app_type, (True, True))[0]

def _can_subscribe(app_type: str) -> bool:
    """Return True if an app of this type is allowed to subscribe to topics."""
    return _APP_TYPE_MESSAGING_CAPABILITY.get(app_type, (True, True))[1]

# --- Criticality levels for statistical generation ---
CRITICALITY_OPTIONS = [True, False]

# ---------------------------------------------------------------------------
# Per-domain frequency bounds (Hz) for log-uniform sampling.
# Each entry is (lo_hz, hi_hz) for ``random.uniform(log10(lo), log10(hi))``.
# References:
#   ATM (ICAO SWIM/DDS):  radar/ADS-B updates 1–100 Hz, control msgs up to 200 Hz
#   AV (ROS2 pubsub):     sensor streams 10–100 Hz, HD-map infrequent ≈ 0.1 Hz
#   HFT (market data):    tick feeds 100–10 000 Hz
#   IoT smart-city:       environmental sensors 0.01–5 Hz
#   Healthcare/PACS:      clinical events 0.001–2 Hz
#   Financial/enterprise: order/CRM events 0.001–10 Hz
#   Hub-and-spoke/micro:  service calls 0.01–50 Hz
# Domains not listed fall back to the generic range (0.1–100 Hz).
# ---------------------------------------------------------------------------
_DOMAIN_FREQ_BOUNDS: Dict[str, tuple] = {
    "air_traffic_management": (1.0,   200.0),
    "autonomous_vehicle":     (0.1,   100.0),
    "financial_trading":      (100.0, 10000.0),
    "iot_smart_city":         (0.01,  5.0),
    "healthcare":             (0.001, 2.0),
    "enterprise":             (0.001, 10.0),
    "hub_and_spoke":          (0.01,  50.0),
    "microservices":          (0.01,  50.0),
}
_DOMAIN_FREQ_BOUNDS_DEFAULT: tuple = (0.1, 100.0)

# Label-noise rate for Topic.criticality ground-truth injection.
# ~17 % of QoS-rule-derived labels are flipped uniformly so the GNN cannot
# trivially recover criticality from QoS features alone.
_CRITICALITY_NOISE_RATE: float = 0.17

# Ordered criticality levels (used to pick a random *different* label on flip).
_CRITICALITY_LABELS = ["minimal", "low", "medium", "high", "critical"]


class StatisticalGraphGenerator:
    """Generates graphs using statistical distributions from config."""
    
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        # Track pub/sub counts per entity for recursive calculation
        self._direct_pub_counts: Dict[str, int] = {}
        self._direct_sub_counts: Dict[str, int] = {}
        self._uses_graph: Dict[str, List[str]] = {}  # entity_id -> [lib_ids it uses]
        self.logger = logging.getLogger(__name__)

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

    def _generate_lib_code_metrics(self, archetype: Optional[str] = None, name_rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Generate a full code_metrics dict for a Library."""
        if not archetype or archetype not in _LIB_CODE_METRICS_PARAMS:
            rng_to_use = name_rng if name_rng else self.rng
            archetype = rng_to_use.choice(_LIB_ARCHETYPE_WEIGHTS)
        params = _LIB_CODE_METRICS_PARAMS[archetype]
        return self._build_code_metrics(params)

    def _sample_biased(
        self,
        count: int,
        all_items: list,
        cluster_items: list,
        p_intra: float = 0.65,
    ) -> list:
        """Sample *count* items with intra-cluster bias.

        Approximately *p_intra* of the samples come from *cluster_items* (when
        available); the remainder are drawn from *all_items* without
        replacement across the combined result.  Falls back to plain uniform
        sampling when *cluster_items* is empty or *count* is zero.
        """
        if not cluster_items or count <= 0:
            return self.rng.sample(all_items, k=min(count, len(all_items)))

        intra_count = min(round(count * p_intra), len(cluster_items), count)
        intra_sample = self.rng.sample(cluster_items, k=intra_count)

        extra_count = count - intra_count
        if extra_count > 0:
            intra_ids = {id(x) for x in intra_sample}
            extra_pool = [x for x in all_items if id(x) not in intra_ids]
            extra_sample = self.rng.sample(extra_pool, k=min(extra_count, len(extra_pool)))
            return intra_sample + extra_sample
        return intra_sample

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
        max_wmc = max(int(round(avg_wmc)), int(round(avg_wmc * rng.lognormvariate(0.8, 0.5))))

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

    # ------------------------------------------------------------------
    # Structural-quality helpers (called inside generate())
    # ------------------------------------------------------------------

    def _build_cluster_to_nodes(
        self,
        nodes: List[Node],
        cluster_domains: List[str],
    ) -> Dict[str, List[Node]]:
        """Randomly partition nodes into per-cluster subsets.

        Each node is assigned to one cluster via random.choices so the
        partition is seeded and deterministic.  Clusters that receive no
        nodes fall back to the full node list so no app/broker is ever
        stranded without a placement target.
        """
        cluster_to_nodes: Dict[str, List[Node]] = {d: [] for d in cluster_domains}
        assignments = self.rng.choices(cluster_domains, k=len(nodes))
        for node, cluster in zip(nodes, assignments):
            cluster_to_nodes[cluster].append(node)
        # Guard: empty cluster bins fall back to all nodes
        for cluster in cluster_domains:
            if not cluster_to_nodes[cluster]:
                cluster_to_nodes[cluster] = list(nodes)
        return cluster_to_nodes

    def _assign_apps_to_nodes(
        self,
        apps: List[Application],
        nodes: List[Node],
        app_cluster_domain: List[str],
        cluster_to_nodes: Dict[str, List[Node]],
        p_collocate: float = 0.70,
    ) -> List[Dict[str, str]]:
        """Assign apps to nodes with cluster affinity.

        Apps with hotstandby=True get a second RUNS_ON edge to a distinct
        node, providing the "active + standby" infra pair.

        With probability *p_collocate* the host is drawn from the app's
        cluster node subset; otherwise it is drawn uniformly from all nodes.
        This makes node-level structural metrics (betweenness, SPOF
        detection) realistic: functionally related apps share infrastructure.
        """
        runs_on = []
        for i, app in enumerate(apps):
            cluster = app_cluster_domain[i]
            if self.rng.random() < p_collocate:
                host = self.rng.choice(cluster_to_nodes[cluster])
            else:
                host = self.rng.choice(nodes)
            runs_on.append(self._make_edge(app, host))
            if app.hotstandby:
                # Pick a second distinct node for the standby pair
                candidates = [n for n in nodes if n.id != host.id]
                if candidates:
                    standby_host = self.rng.choice(candidates)
                    runs_on.append(self._make_edge(app, standby_host))
        return runs_on

    def _partition_topics_by_qos_affinity(
        self,
        topics: List[Topic],
        app_type: str,
    ) -> Tuple[List[Topic], List[Topic]]:
        """Return (preferred_topics, other_topics) based on app_type QoS affinity.

        Preferred topics match the reliability and priority tiers in
        _APP_TYPE_QOS_AFFINITY[app_type].  Falls back to (all_topics, [])
        when no affinity is defined or no topics match, so _sample_biased()
        behaviour is unchanged for unknown app types or uniform QoS clusters.
        """
        affinity = _APP_TYPE_QOS_AFFINITY.get(app_type)
        if not affinity or not topics:
            return list(topics), []
        pref_rel = set(affinity["reliability"])
        pref_pri = set(affinity["priority"])
        preferred = [
            t for t in topics
            if t.qos.reliability in pref_rel and t.qos.transport_priority in pref_pri
        ]
        if not preferred:
            return list(topics), []
        other = [t for t in topics if t not in preferred]
        return preferred, other

    def _rewrite_broker_placement(
        self,
        brokers: List[Broker],
        nodes: List[Node],
        routes: List[Dict[str, str]],
        topic_id_to_cluster: Dict[str, str],
        cluster_to_nodes: Dict[str, List[Node]],
    ) -> List[Dict[str, str]]:
        """Place each broker on a node in its plurality cluster.

        The plurality cluster is determined by the cluster distribution of
        topics the broker routes.  This co-locates brokers with the apps
        they serve, improving infra-layer SPOF and betweenness accuracy.
        Should be called *after* the routes list (including stranded-broker
        guard) is complete so every broker has at least one routed topic.
        """
        broker_cluster_votes: Dict[str, Counter] = {b.id: Counter() for b in brokers}
        for edge in routes:
            broker_id = edge["from"]
            topic_id = edge["to"]
            cluster = topic_id_to_cluster.get(topic_id)
            if cluster and broker_id in broker_cluster_votes:
                broker_cluster_votes[broker_id][cluster] += 1

        runs_on = []
        for broker in brokers:
            votes = broker_cluster_votes[broker.id]
            if votes:
                plurality_cluster = votes.most_common(1)[0][0]
                cluster_nodes = cluster_to_nodes.get(plurality_cluster, nodes)
                host = self.rng.choice(cluster_nodes)
            else:
                host = self.rng.choice(nodes)
            runs_on.append(self._make_edge(broker, host))
        return runs_on

    def _validate_role_constraints(
        self,
        apps: List[Application],
        publishes: List[Dict[str, str]],
        subscribes: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Remove edges that violate app messaging capability constraints.

        Apps whose app_type allows only publishing must not appear in *subscribes*;
        apps whose app_type allows only subscribing must not appear in *publishes*.
        Library edges are not affected (libraries have no app_type field).
        """
        pub_only_ids = {a.id for a in apps if _can_publish(a.app_type) and not _can_subscribe(a.app_type)}
        sub_only_ids = {a.id for a in apps if _can_subscribe(a.app_type) and not _can_publish(a.app_type)}

        clean_pub = [e for e in publishes if e["from"] not in sub_only_ids]
        clean_sub = [e for e in subscribes if e["from"] not in pub_only_ids]

        removed_pub = len(publishes) - len(clean_pub)
        removed_sub = len(subscribes) - len(clean_sub)
        if removed_pub or removed_sub:
            self.logger.debug(
                "Messaging capability enforcement: removed %d publish and %d subscribe edges.",
                removed_pub, removed_sub,
            )
        return clean_pub, clean_sub

    def _sample_topic_frequency(self, domain: Optional[str], rng: Optional[random.Random] = None) -> float:
        """Sample a topic frequency from a per-domain log-uniform distribution.

        Log-uniform (i.e. uniform on the log10 axis) is appropriate because
        topic frequencies span several orders of magnitude across domains.  A
        plain uniform draw over [lo, hi] would massively over-sample the
        high-frequency end and suppress the low-frequency tail that carries the
        domain-differentiating signal LOSO needs to reward.

        Args:
            domain: Domain key matching ``_DOMAIN_FREQ_BOUNDS`` (e.g.
                ``"air_traffic_management"``).  Falls back to the generic range
                when the domain is unknown or ``None``.
            rng: Random instance to use.  Defaults to ``self.rng``; callers
                that want to avoid perturbing the main topology RNG stream
                should pass a dedicated ``random.Random`` object.

        Returns:
            Frequency in Hz, rounded to 3 significant figures.
        """
        import math
        _rng = rng if rng is not None else self.rng
        lo, hi = _DOMAIN_FREQ_BOUNDS.get(domain or "", _DOMAIN_FREQ_BOUNDS_DEFAULT)
        log_val = _rng.uniform(math.log10(lo), math.log10(hi))
        freq = 10.0 ** log_val
        # Round to 3 significant figures to avoid spurious decimal precision.
        if freq >= 1.0:
            return round(freq, max(0, 2 - int(math.floor(math.log10(freq)))))
        return round(freq, 3)

    def _derive_topic_criticality_with_noise(
        self,
        qos: "QoSPolicy",
        noise_rate: float = _CRITICALITY_NOISE_RATE,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Derive a criticality label from QoS and inject controlled label noise.

        The base label is produced by the same threshold table used in
        ``Topic.__post_init__`` (``CRITICALITY_THRESHOLDS``).  Then, with
        probability *noise_rate*, the label is replaced by a uniformly
        chosen *different* label from ``_CRITICALITY_LABELS``.

        Rationale for noise injection:
            Without noise, ``criticality`` is a deterministic function of
            QoS weights.  Any model that sees the QoS features can recover the
            label perfectly via a lookup table — the prediction task collapses
            to a dictionary read, not structural graph learning.  The noise
            channel forces the GNN to exploit multi-hop graph context (fan-out,
            betweenness, cluster membership) to resolve ambiguous labels,
            which is precisely the generalisation the paper claims.

        Args:
            qos: The ``QoSPolicy`` of the topic being constructed.
            noise_rate: Fraction of labels to flip.  Default is
                ``_CRITICALITY_NOISE_RATE`` (≈ 17 %).
            rng: Random instance to use.  Defaults to ``self.rng``; callers
                that want to avoid perturbing the main topology RNG stream
                should pass a dedicated ``random.Random`` object.

        Returns:
            A criticality label string (one of ``_CRITICALITY_LABELS``).
        """
        from saag.core.models import CRITICALITY_THRESHOLDS
        _rng = rng if rng is not None else self.rng
        # --- derive base label via threshold table ---
        qos_score = qos.calculate_weight()
        base_label = "critical"  # fallback if all thresholds exceeded
        for threshold, label in CRITICALITY_THRESHOLDS:
            if qos_score <= threshold:
                base_label = label
                break

        # --- inject noise ---
        if _rng.random() < noise_rate:
            alternatives = [lbl for lbl in _CRITICALITY_LABELS if lbl != base_label]
            return _rng.choice(alternatives)
        return base_label

    def _assign_criticality_two_pass(
        self,
        apps: List[Application],
        publishes: List[Dict[str, str]],
        subscribes: List[Dict[str, str]],
        criticality_pool: Optional[List[bool]],
    ) -> None:
        """Assign criticality in-place after topology is built.

        Ranks apps by structural degree proxy (pub_count + sub_count) and
        assigns critical=True to the top-N highest-degree apps, where N is
        derived from *criticality_pool* (or 10 % of apps as fallback).  Ties
        are broken with a seeded jitter so the result is deterministic but
        not index-position biased.

        This intentionally changes seeded output vs. the previous approach
        (random assignment before topology existed) because it produces
        topologically coherent criticality: structurally central components
        are more likely to be labelled critical.
        """
        pub_count: Counter = Counter(e["from"] for e in publishes)
        sub_count: Counter = Counter(e["from"] for e in subscribes)

        if criticality_pool is not None:
            n_critical = sum(1 for x in criticality_pool if x)
        else:
            n_critical = max(1, round(len(apps) * 0.10))
        n_critical = min(n_critical, len(apps))

        # Sort descending by degree proxy + small seeded jitter to break ties
        ranked = sorted(
            apps,
            key=lambda a: (
                pub_count.get(a.id, 0) + sub_count.get(a.id, 0)
                + self.rng.uniform(0.0, 0.3)
            ),
            reverse=True,
        )
        critical_ids = {a.id for a in ranked[:n_critical]}
        for app in apps:
            app.criticality = app.id in critical_ids

    def generate(self) -> Dict[str, Any]:
        c = self.config
        name_rng = random.Random(c.seed + 12345)
        # Dedicated RNG for topic attribute sampling (frequency, criticality).
        # Isolated from self.rng so that topic attribute draws do NOT perturb
        # the main topology RNG stream (broker routes, pub/sub edges, node
        # placement), preserving reproducibility of all previously seeded tests.
        topic_attr_rng = random.Random(c.seed + 99999)
        # 1. Generate base entities using Core Models
        domain_ds = None
        if c.domain:
            domain_ds = DomainDataset(c.domain, name_rng)
            
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
                    durability = name_rng.choice(durability_pool)
                else:
                    durability = name_rng.choice(DURABILITY_OPTIONS)
                
                if reliability_pool and i < len(reliability_pool):
                    reliability = reliability_pool[i]
                elif reliability_pool:
                    reliability = name_rng.choice(reliability_pool)
                else:
                    reliability = name_rng.choice(RELIABILITY_OPTIONS)
                
                if priority_pool and i < len(priority_pool):
                    transport_priority = priority_pool[i]
                elif priority_pool:
                    transport_priority = name_rng.choice(priority_pool)
                else:
                    transport_priority = name_rng.choice(PRIORITY_OPTIONS)
            
            qos_policy = QoSPolicy(
                durability=durability,
                reliability=reliability,
                transport_priority=transport_priority,
            )
            topics.append(Topic(
                id=f"T{i}",
                name=topic_name,
                size=size,
                qos=qos_policy,
                # Inject domain-aware frequency and noisy criticality as
                # generator-supplied ground truth.  This prevents the prediction
                # task from collapsing to a QoS lookup (leakage path).
                # Both calls use the isolated topic_attr_rng so the main
                # topology RNG stream (self.rng) remains unperturbed.
                frequency=self._sample_topic_frequency(c.domain, rng=topic_attr_rng),
                criticality=self._derive_topic_criticality_with_noise(qos_policy, rng=topic_attr_rng),
            ))

        # === Pass 1: hierarchy cluster pre-assignment ===
        # Partition apps and topics into clusters keyed by css_name (the
        # third level of the MIL-STD-498 hierarchy).  Apps in the same cluster
        # share a css_name and will preferentially pub/sub to topics in the
        # same cluster (Pass 2 below, p_intra = 0.65), making the hierarchy
        # signal structurally meaningful for coupling analysis instead of being
        # an independently-sampled label with no topological effect.
        _hier_pool = SYSTEM_HIERARCHY_POOLS.get(c.domain, GENERIC_HIERARCHY_POOL)
        _cluster_domains: List[str] = _hier_pool["domain"]
        _n_clusters = len(_cluster_domains)

        # Uses random weighted assignment to create natural cluster skew.
        _app_cluster_domain: List[str] = self.rng.choices(_cluster_domains, k=c.apps)
        _lib_cluster_domain: List[str] = self.rng.choices(_cluster_domains, k=c.libs)
        
        _app_id_to_cluster: Dict[str, str] = {
            f"A{i}": _app_cluster_domain[i] for i in range(c.apps)
        }

        # Assign topics to clusters and build the reverse lookup.
        _cluster_to_topics: Dict[str, List[Topic]] = {d: [] for d in _cluster_domains}
        _topic_id_to_cluster: Dict[str, str] = {}
        for idx, topic in enumerate(topics):
            cluster_d = _cluster_domains[idx % _n_clusters]
            _cluster_to_topics[cluster_d].append(topic)
            _topic_id_to_cluster[topic.id] = cluster_d

        apps: List[Application] = []
        criticality_pool = None
        
        if c.application_stats:
            if c.application_stats.app_criticality_distribution:
                criticality_pool = c.application_stats.app_criticality_distribution.to_weighted_list()
                self.rng.shuffle(criticality_pool)
        
        _cluster_to_libs: Dict[str, List[Library]] = {d: [] for d in _cluster_domains}
        
        for i in range(c.apps):
            # Criticality is assigned after topology is built (two-pass approach
            # in _assign_criticality_two_pass).  Use placeholder False here.
            app_name = domain_ds.get_app_name() if domain_ds else f"App-{i}"
            if domain_ds:
                app_type = get_app_type_for_name(app_name)
            else:
                app_type = name_rng.choice(APP_TYPE_OPTIONS)
                
            code_metrics = self._generate_code_metrics(app_type)
            # Use the pre-assigned cluster css_name so hierarchy reflects
            # actual structural grouping rather than an independent random draw.
            hierarchy = {
                "csc_name": name_rng.choice(_hier_pool["component"]),
                "csci_name": name_rng.choice(_hier_pool["config_item"]),
                "css_name": _app_cluster_domain[i],
                "csms_name": name_rng.choice(_hier_pool["system"]),
            }

            priority = self.rng.choice(APP_PRIORITY_OPTIONS)
            hotstandby = self.rng.choice(APP_HOTSTANDBY_OPTIONS)
            num_roles = self.rng.randint(1, 3)
            role = sorted(self.rng.sample(APP_USER_ROLE_OPTIONS, num_roles))
            apps.append(Application(
                id=f"A{i}",
                name=app_name,
                app_type=app_type,
                role=role,
                criticality=False,  # assigned after topology by _assign_criticality_two_pass
                priority=priority,
                hotstandby=hotstandby,
                version=f"{self.rng.randint(1, 3)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
                system_hierarchy=hierarchy,
                code_metrics=code_metrics,
            ))

        # Build the cluster → apps map now that all app objects exist.
        _cluster_to_apps: Dict[str, List[Application]] = {d: [] for d in _cluster_domains}
        for i, app in enumerate(apps):
            _cluster_to_apps[_app_cluster_domain[i]].append(app)

        libs: List[Library] = []
        for i in range(c.libs):
            lib_name = domain_ds.get_library_name() if domain_ds else f"Lib-{i}"
            if domain_ds:
                archetype = get_lib_archetype_for_name(lib_name)
                lib_code_metrics = self._generate_lib_code_metrics(archetype, name_rng)
            else:
                lib_code_metrics = self._generate_lib_code_metrics(None, name_rng)

            lib_hierarchy = domain_ds.get_system_hierarchy() if domain_ds else get_generic_system_hierarchy(name_rng)
            if not domain_ds:
                lib_hierarchy["css_name"] = _lib_cluster_domain[i]
                
            libs.append(Library(
                id=f"L{i}",
                name=lib_name,
                version=f"{self.rng.randint(0, 2)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}",
                system_hierarchy=lib_hierarchy,
                code_metrics=lib_code_metrics,
            ))
            _cluster_to_libs[_lib_cluster_domain[i]].append(libs[-1])

        # 2. Relationships
        # Build a shared cluster→node partition used for both app and broker placement.
        _cluster_to_nodes = self._build_cluster_to_nodes(nodes, _cluster_domains)

        # Apps: cluster-affine node assignment (70 % of apps land on a node
        # from their functional cluster; 30 % are placed anywhere).  This
        # replaces the previous sequential/random assignment and makes node-level
        # structural metrics (betweenness, SPOF detection) realistic.
        runs_on = self._assign_apps_to_nodes(
            apps, nodes, _app_cluster_domain, _cluster_to_nodes
        )

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

        # Brokers: cluster-affine placement based on plurality of routed topics.
        # Called after routes (including the stranded-broker guard) are complete
        # so every broker has at least one topic to vote its cluster.
        broker_runs_on = self._rewrite_broker_placement(
            brokers, nodes, routes, _topic_id_to_cluster, _cluster_to_nodes
        )
        runs_on.extend(broker_runs_on)

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
                _lib_cluster_apps = _cluster_to_apps[_lib_cluster_domain[libs.index(lib)]]
                usage_count = self._sample_from_distribution(c.library_stats.applications_using_this_library)
                usage_count = min(usage_count, len(apps))
                if usage_count > 0:
                    using_apps = self._sample_biased(usage_count, apps, _lib_cluster_apps, p_intra=c.intra_cluster_coupling)
                    for app in using_apps:
                        if lib.id not in self._uses_graph[app.id]:
                            uses.append(self._make_edge(app, lib))
                            self._uses_graph[app.id].append(lib.id)
        else:
            for app in apps:
                _app_cluster_libs = _cluster_to_libs[_app_id_to_cluster[app.id]]
                if libs:
                    max_uses = min(max(3, int(0.1 * len(libs))), len(libs))
                    n_uses = self.rng.randint(0, max_uses)
                    targets = self._sample_biased(n_uses, libs, _app_cluster_libs, p_intra=c.intra_cluster_coupling)
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
                
                # Enforce messaging capability constraint derived from app_type
                if not _can_publish(app.app_type): direct_pub_count = 0
                if not _can_subscribe(app.app_type): direct_sub_count = 0

                # Pass 2: use cluster-biased sampling so apps in the same
                # hierarchy cluster preferentially share topics (p_intra=0.65).
                # QoS affinity further steers the preferred pool: gateway/controller
                # apps draw from RELIABLE/HIGH topics first; sensors from BEST_EFFORT.
                _app_cluster_topics = _cluster_to_topics[_app_id_to_cluster[app.id]]
                _qos_preferred, _ = self._partition_topics_by_qos_affinity(
                    _app_cluster_topics, app.app_type
                )
                _pub_pool = _qos_preferred if _qos_preferred else _app_cluster_topics
                _sub_pool = _qos_preferred if _qos_preferred else _app_cluster_topics

                if direct_pub_count > 0:
                    pub_topics = self._sample_biased(direct_pub_count, topics, _pub_pool, p_intra=c.intra_cluster_coupling)
                    for t in pub_topics:
                        publishes.append(self._make_edge(app, t))
                    self._direct_pub_counts[app.id] = direct_pub_count

                if direct_sub_count > 0:
                    sub_topics = self._sample_biased(direct_sub_count, topics, _sub_pool, p_intra=c.intra_cluster_coupling)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(app, t))
                    self._direct_sub_counts[app.id] = direct_sub_count

        elif c.application_stats and c.application_stats.direct_publish_count.mean > 0:
            for app in apps:
                _app_cluster_topics = _cluster_to_topics[_app_id_to_cluster[app.id]]
                _qos_preferred, _ = self._partition_topics_by_qos_affinity(
                    _app_cluster_topics, app.app_type
                )
                _pub_pool = _qos_preferred if _qos_preferred else _app_cluster_topics
                _sub_pool = _qos_preferred if _qos_preferred else _app_cluster_topics

                pub_count = self._sample_from_distribution(c.application_stats.direct_publish_count)
                pub_count = min(pub_count, len(topics))
                if not _can_publish(app.app_type): pub_count = 0

                if pub_count > 0:
                    pub_topics = self._sample_biased(pub_count, topics, _pub_pool, p_intra=c.intra_cluster_coupling)
                    for t in pub_topics:
                        publishes.append(self._make_edge(app, t))
                    self._direct_pub_counts[app.id] = pub_count

                sub_count = self._sample_from_distribution(c.application_stats.direct_subscribe_count)
                sub_count = min(sub_count, len(topics))
                if not _can_subscribe(app.app_type): sub_count = 0

                if sub_count > 0:
                    sub_topics = self._sample_biased(sub_count, topics, _sub_pool, p_intra=c.intra_cluster_coupling)
                    for t in sub_topics:
                        subscribes.append(self._make_edge(app, t))
                    self._direct_sub_counts[app.id] = sub_count
                    
        elif c.topic_stats and c.topic_stats.applications_publishing_to_this_topic.mean > 0:
            valid_pubs_all = [a for a in apps if _can_publish(a.app_type)]
            valid_subs_all = [a for a in apps if _can_subscribe(a.app_type)]
            for topic in topics:
                _topic_cluster_apps = _cluster_to_apps[_topic_id_to_cluster[topic.id]]
                _topic_cluster_pubs = [a for a in _topic_cluster_apps if _can_publish(a.app_type)]
                _topic_cluster_subs = [a for a in _topic_cluster_apps if _can_subscribe(a.app_type)]
                
                pub_count = self._sample_from_distribution(c.topic_stats.applications_publishing_to_this_topic)
                pub_count = min(pub_count, len(valid_pubs_all))
                if pub_count > 0:
                    pubs = self._sample_biased(pub_count, valid_pubs_all, _topic_cluster_pubs, p_intra=c.intra_cluster_coupling)
                    for p in pubs:
                        publishes.append(self._make_edge(p, topic))
                        self._direct_pub_counts[p.id] = self._direct_pub_counts.get(p.id, 0) + 1

                sub_count = self._sample_from_distribution(c.topic_stats.applications_subscribing_to_this_topic)
                sub_count = min(sub_count, len(valid_subs_all))
                if sub_count > 0:
                    subs = self._sample_biased(sub_count, valid_subs_all, _topic_cluster_subs, p_intra=c.intra_cluster_coupling)
                    for s in subs:
                        subscribes.append(self._make_edge(s, topic))
                        self._direct_sub_counts[s.id] = self._direct_sub_counts.get(s.id, 0) + 1
        else:
            valid_pubs_all_fb = [a for a in apps if _can_publish(a.app_type)]
            valid_subs_all_fb = [a for a in apps if _can_subscribe(a.app_type)]
            for topic in topics:
                _topic_cluster_apps = _cluster_to_apps[_topic_id_to_cluster[topic.id]]
                _topic_cluster_pubs = [a for a in _topic_cluster_apps if _can_publish(a.app_type)]
                _topic_cluster_subs = [a for a in _topic_cluster_apps if _can_subscribe(a.app_type)]
                
                k_pubs = self.rng.randint(1, max(2, min(5, len(valid_pubs_all_fb))))
                k_subs = self.rng.randint(1, max(2, min(8, len(valid_subs_all_fb))))
                pubs = self._sample_biased(k_pubs, valid_pubs_all_fb, _topic_cluster_pubs, p_intra=c.intra_cluster_coupling)
                subs = self._sample_biased(k_subs, valid_subs_all_fb, _topic_cluster_subs, p_intra=c.intra_cluster_coupling)
                for p in pubs:
                    publishes.append(self._make_edge(p, topic))
                for s in subs:
                    subscribes.append(self._make_edge(s, topic))

        connects = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.rng.random() < c.connection_density:
                    connects.append(self._make_edge(nodes[i], nodes[j]))
        
        # Guard against zero-mesh tiny topologies explicitly
        if len(nodes) >= 2 and not connects:
            i, j = self.rng.sample(range(len(nodes)), 2)
            connects.append(self._make_edge(nodes[i], nodes[j]))
            
        # Ensure zero-degree apps receive at least 1 edge mapping to prevent inflated F1s
        connected_apps = {e["from"] for e in publishes}.union(e["from"] for e in subscribes)
        isolated_apps = [a for a in apps if a.id not in connected_apps]
        for app in isolated_apps:
            t = self.rng.choice(topics)
            if _can_publish(app.app_type):
                publishes.append(self._make_edge(app, t))
            elif _can_subscribe(app.app_type):
                subscribes.append(self._make_edge(app, t))
                
        # Enforce role constraints: remove any edges that violate pub/sub role
        # (e.g. a "pub" app appearing in subscribes due to the topic_stats path).
        publishes, subscribes = self._validate_role_constraints(apps, publishes, subscribes)

        # Two-pass criticality: now that topology is known, assign critical=True
        # to the structurally most central apps (highest pub+sub degree), honouring
        # the target count from the scenario YAML's criticality distribution.
        self._assign_criticality_two_pass(apps, publishes, subscribes, criticality_pool)

        graph_dict = {
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
        return validate_and_clean_schema(graph_dict)


def validate_and_clean_schema(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the schema of the generated graph and clean up duplicate edges.
    
    Ensures all necessary top-level fields are present, deduplicates all edges
    in the relationships block, and validates that all edge references point
    to existing entities.
    """
    # 1. Verify top-level structure
    required_keys = {"metadata", "nodes", "brokers", "topics", "applications", "libraries", "relationships"}
    for k in required_keys:
        if k not in graph_data:
            raise ValueError(f"Generated graph is missing required top-level key: '{k}'")
            
    # 2. Collect all valid entity IDs to check for duplicates and dangling edges
    valid_ids = set()
    categories = ["nodes", "brokers", "topics", "applications", "libraries"]
    for category in categories:
        for entity in graph_data[category]:
            entity_id = entity.get("id")
            if not entity_id:
                raise ValueError(f"Entity in '{category}' is missing 'id' attribute: {entity}")
            if entity_id in valid_ids:
                raise ValueError(f"Duplicate entity ID found in generated graph: '{entity_id}'")
            valid_ids.add(entity_id)
            
    # 3. Deduplicate and validate relationships
    relationships = graph_data["relationships"]
    required_rels = {"runs_on", "routes", "publishes_to", "subscribes_to", "connects_to", "uses"}
    for rel_type in required_rels:
        if rel_type not in relationships:
            raise ValueError(f"Relationships dict is missing required key: '{rel_type}'")
            
        edges = relationships[rel_type]
        if not isinstance(edges, list):
            raise ValueError(f"Relationship key '{rel_type}' must map to a list of edges")
            
        seen_edges = set()
        deduped_edges = []
        for idx, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise ValueError(f"Edge at index {idx} in '{rel_type}' is not a dictionary")
            if "from" not in edge or "to" not in edge:
                raise ValueError(f"Edge at index {idx} in '{rel_type}' is missing 'from' or 'to' key")
                
            from_id = edge["from"]
            to_id = edge["to"]
            
            # Check for dangling reference
            if from_id not in valid_ids:
                raise ValueError(f"Edge '{from_id}' -> '{to_id}' in '{rel_type}' references non-existent source ID '{from_id}'")
            if to_id not in valid_ids:
                raise ValueError(f"Edge '{from_id}' -> '{to_id}' in '{rel_type}' references non-existent target ID '{to_id}'")
                
            # Deduplicate edge pair
            edge_key = (from_id, to_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                deduped_edges.append(edge)
                
        relationships[rel_type] = deduped_edges
        
    return graph_data
