"""
Structural metrics calculator module.

Implements all structural metrics defined in the paper:

Application-level:
    R(a)   - Reach: unique apps interacting with a through pub/sub
    AMP(a) - Amplification: R(a) / (|PUB(a)| + 1)
    RA(a)  - Role Asymmetry: (|PUB(a)| - |SUB(a)|) / (|PUB(a)| + |SUB(a)| + 1)
    TC(a)  - Topic Context Diversity: distinct topic categories
    LE(a)  - Library Exposure: |USES(a)|

Topic-level:
    C(t)   - Coverage: |SUB(t)| + |PUB(t)|
    I(t)   - Imbalance: ||SUB(t)| - |PUB(t)|| / (|SUB(t)| + |PUB(t)| + 1)
    PS(t)  - Physical Spread: distinct nodes of connected apps
    LCR(t) - Low Connectivity Ratio: ratio of low-connectivity apps

Node-level:
    ND(n)  - Node Density: |RUNS(n)|
    NID(n) - Node Interaction Density: interacting app pairs on node

Library-level:
    LC(l)  - Library Coverage: |USES(l)|
    LCon(l)- Library Concentration: max apps using l on any single node
"""

from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class ApplicationMetrics:
    """Structural metrics for a single application."""
    id: str
    name: str
    reach: int = 0
    amplification: float = 0.0
    role_asymmetry: float = 0.0
    topic_context_diversity: int = 0
    library_exposure: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "R": self.reach,
            "AMP": round(self.amplification, 4),
            "RA": round(self.role_asymmetry, 4),
            "TC": self.topic_context_diversity,
            "LE": self.library_exposure,
        }


@dataclass
class TopicMetrics:
    """Structural metrics for a single topic."""
    id: str
    name: str
    coverage: int = 0
    imbalance: float = 0.0
    physical_spread: int = 0
    low_connectivity_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "C": self.coverage,
            "I": round(self.imbalance, 4),
            "PS": self.physical_spread,
            "LCR": round(self.low_connectivity_ratio, 4),
        }


@dataclass
class NodeMetrics:
    """Structural metrics for a single node."""
    id: str
    name: str
    density: int = 0
    interaction_density: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "ND": self.density,
            "NID": self.interaction_density,
        }


@dataclass
class LibraryMetrics:
    """Structural metrics for a single library."""
    id: str
    name: str
    coverage: int = 0
    concentration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "LC": self.coverage,
            "LCon": self.concentration,
        }


@dataclass
class AllMetrics:
    """Container for all structural metrics in the system."""
    applications: List[ApplicationMetrics] = field(default_factory=list)
    topics: List[TopicMetrics] = field(default_factory=list)
    nodes: List[NodeMetrics] = field(default_factory=list)
    libraries: List[LibraryMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applications": [a.to_dict() for a in self.applications],
            "topics": [t.to_dict() for t in self.topics],
            "nodes": [n.to_dict() for n in self.nodes],
            "libraries": [l.to_dict() for l in self.libraries],
        }


def calculate_all_metrics(data: Dict[str, Any], k: int = 3) -> AllMetrics:
    """
    Calculate all structural metrics from aggregated JSON data.

    Args:
        data: Parsed JSON from aggregator output.
        k: Upper bound for low-connectivity classification in LCR metric.

    Returns:
        AllMetrics containing metrics for all component types.
    """
    apps = data.get("applications", [])
    topics = data.get("topics", [])
    nodes = data.get("nodes", [])
    libs = data.get("libraries", [])
    rels = data.get("relationships", {})

    pub_to = rels.get("publishes_to", [])
    sub_to = rels.get("subscribes_to", [])
    runs_on = rels.get("runs_on", [])
    uses_rels = rels.get("uses", [])

    # ---- build lookup structures ----

    # PUB(a) — topics that app a publishes to
    app_pub_topics: Dict[str, Set[str]] = {a["id"]: set() for a in apps}
    for r in pub_to:
        if r["from"] in app_pub_topics:
            app_pub_topics[r["from"]].add(r["to"])

    # SUB(a) — topics that app a subscribes to
    app_sub_topics: Dict[str, Set[str]] = {a["id"]: set() for a in apps}
    for r in sub_to:
        if r["from"] in app_sub_topics:
            app_sub_topics[r["from"]].add(r["to"])

    # PUB(t) — apps that publish to topic t
    topic_pub_apps: Dict[str, Set[str]] = {t["id"]: set() for t in topics}
    for r in pub_to:
        if r["to"] in topic_pub_apps:
            topic_pub_apps[r["to"]].add(r["from"])

    # SUB(t) — apps that subscribe to topic t
    topic_sub_apps: Dict[str, Set[str]] = {t["id"]: set() for t in topics}
    for r in sub_to:
        if r["to"] in topic_sub_apps:
            topic_sub_apps[r["to"]].add(r["from"])

    # RUNS(n) — apps running on node n
    node_apps: Dict[str, Set[str]] = {n["id"]: set() for n in nodes}
    for r in runs_on:
        if r["to"] in node_apps:
            node_apps[r["to"]].add(r["from"])

    # app → node mapping
    app_node: Dict[str, str] = {}
    for r in runs_on:
        app_set = {a["id"] for a in apps}
        if r["from"] in app_set:
            app_node[r["from"]] = r["to"]

    # USES(a) — libs used by app a
    app_libs: Dict[str, Set[str]] = {a["id"]: set() for a in apps}
    for r in uses_rels:
        if r["from"] in app_libs:
            app_libs[r["from"]].add(r["to"])

    # USES(l) — apps using lib l
    lib_apps: Dict[str, Set[str]] = {l["id"]: set() for l in libs}
    for r in uses_rels:
        if r["to"] in lib_apps:
            lib_apps[r["to"]].add(r["from"])

    # topic category lookup
    topic_category: Dict[str, str] = {}
    for t in topics:
        topic_category[t["id"]] = _extract_category(t.get("name", ""))

    result = AllMetrics()

    # ---- Application metrics ----
    result.applications = _calc_app_metrics(
        apps, app_pub_topics, app_sub_topics,
        topic_pub_apps, topic_sub_apps,
        app_libs, topic_category,
    )

    # ---- Topic metrics ----
    result.topics = _calc_topic_metrics(
        topics, topic_pub_apps, topic_sub_apps,
        app_pub_topics, app_sub_topics, app_node, k,
    )

    # ---- Node metrics ----
    result.nodes = _calc_node_metrics(
        nodes, node_apps,
        app_pub_topics, app_sub_topics,
        topic_pub_apps, topic_sub_apps,
    )

    # ---- Library metrics ----
    result.libraries = _calc_lib_metrics(libs, lib_apps, node_apps)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_category(topic_name: str) -> str:
    """Extract category from topic name using hierarchical prefix.

    Examples:
        "sensor/temperature/cabin" → "sensor"
        "CMD.navigation.waypoint"  → "CMD"
        "Topic-0"                  → "Topic"
        "my_topic_name"            → "my"
    """
    for sep in ("/", ".", "-", "_"):
        if sep in topic_name:
            parts = topic_name.split(sep)
            if len(parts) > 1:
                return parts[0]
    return topic_name


# ---------------------------------------------------------------------------
# Application-level metrics
# ---------------------------------------------------------------------------

def _calc_app_metrics(
    apps: List[Dict],
    app_pub_topics: Dict[str, Set[str]],
    app_sub_topics: Dict[str, Set[str]],
    topic_pub_apps: Dict[str, Set[str]],
    topic_sub_apps: Dict[str, Set[str]],
    app_libs: Dict[str, Set[str]],
    topic_category: Dict[str, str],
) -> List[ApplicationMetrics]:
    results: List[ApplicationMetrics] = []

    for app in apps:
        aid = app["id"]
        aname = app.get("name", aid)
        m = ApplicationMetrics(id=aid, name=aname)

        pub_topics = app_pub_topics.get(aid, set())
        sub_topics = app_sub_topics.get(aid, set())

        # R(a) — Reach
        reached: Set[str] = set()
        for t in pub_topics:
            reached |= topic_sub_apps.get(t, set())
        for t in sub_topics:
            reached |= topic_pub_apps.get(t, set())
        reached.discard(aid)  # exclude self
        m.reach = len(reached)

        # AMP(a) — Amplification
        m.amplification = m.reach / (len(pub_topics) + 1)

        # RA(a) — Role Asymmetry
        npub = len(pub_topics)
        nsub = len(sub_topics)
        m.role_asymmetry = (npub - nsub) / (npub + nsub + 1)

        # TC(a) — Topic Context Diversity
        categories: Set[str] = set()
        for t in pub_topics | sub_topics:
            cat = topic_category.get(t, "")
            if cat:
                categories.add(cat)
        m.topic_context_diversity = len(categories)

        # LE(a) — Library Exposure
        m.library_exposure = len(app_libs.get(aid, set()))

        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Topic-level metrics
# ---------------------------------------------------------------------------

def _calc_topic_metrics(
    topics: List[Dict],
    topic_pub_apps: Dict[str, Set[str]],
    topic_sub_apps: Dict[str, Set[str]],
    app_pub_topics: Dict[str, Set[str]],
    app_sub_topics: Dict[str, Set[str]],
    app_node: Dict[str, str],
    k: int,
) -> List[TopicMetrics]:
    results: List[TopicMetrics] = []

    for topic in topics:
        tid = topic["id"]
        tname = topic.get("name", tid)
        m = TopicMetrics(id=tid, name=tname)

        pubs = topic_pub_apps.get(tid, set())
        subs = topic_sub_apps.get(tid, set())

        # C(t) — Coverage
        m.coverage = len(subs) + len(pubs)

        # I(t) — Imbalance
        m.imbalance = abs(len(subs) - len(pubs)) / (len(subs) + len(pubs) + 1)

        # PS(t) — Physical Spread
        involved_nodes: Set[str] = set()
        for a in pubs | subs:
            n = app_node.get(a)
            if n:
                involved_nodes.add(n)
        m.physical_spread = len(involved_nodes)

        # LCR(t) — Low Connectivity Ratio
        all_connected = pubs | subs
        low_conn_count = 0
        for a in all_connected:
            total_topics = len(app_pub_topics.get(a, set()) | app_sub_topics.get(a, set()))
            if total_topics <= k:
                low_conn_count += 1
        m.low_connectivity_ratio = low_conn_count / (len(all_connected) + 1)

        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Node-level metrics
# ---------------------------------------------------------------------------

def _calc_node_metrics(
    nodes: List[Dict],
    node_apps: Dict[str, Set[str]],
    app_pub_topics: Dict[str, Set[str]],
    app_sub_topics: Dict[str, Set[str]],
    topic_pub_apps: Dict[str, Set[str]],
    topic_sub_apps: Dict[str, Set[str]],
) -> List[NodeMetrics]:
    results: List[NodeMetrics] = []

    for node in nodes:
        nid = node["id"]
        nname = node.get("name", nid)
        m = NodeMetrics(id=nid, name=nname)

        apps_on_node = node_apps.get(nid, set())

        # ND(n) — Node Density
        m.density = len(apps_on_node)

        # NID(n) — Node Interaction Density
        # Count pairs (a_i, a_j) on same node where
        # ∃t: (a_i ∈ PUB(t) ∧ a_j ∈ SUB(t)) ∨ (a_j ∈ PUB(t) ∧ a_i ∈ SUB(t))
        interaction_count = 0
        apps_list = list(apps_on_node)
        for i, ai in enumerate(apps_list):
            for aj in apps_list[i + 1:]:
                if _apps_interact(ai, aj, app_pub_topics, app_sub_topics,
                                  topic_pub_apps, topic_sub_apps):
                    interaction_count += 1
        m.interaction_density = interaction_count

        results.append(m)

    return results


def _apps_interact(
    ai: str, aj: str,
    app_pub_topics: Dict[str, Set[str]],
    app_sub_topics: Dict[str, Set[str]],
    topic_pub_apps: Dict[str, Set[str]],
    topic_sub_apps: Dict[str, Set[str]],
) -> bool:
    """Check if two apps interact through at least one topic (pub↔sub)."""
    # ai publishes, aj subscribes
    for t in app_pub_topics.get(ai, set()):
        if aj in topic_sub_apps.get(t, set()):
            return True
    # aj publishes, ai subscribes
    for t in app_pub_topics.get(aj, set()):
        if ai in topic_sub_apps.get(t, set()):
            return True
    return False


# ---------------------------------------------------------------------------
# Library-level metrics
# ---------------------------------------------------------------------------

def _calc_lib_metrics(
    libs: List[Dict],
    lib_apps: Dict[str, Set[str]],
    node_apps: Dict[str, Set[str]],
) -> List[LibraryMetrics]:
    results: List[LibraryMetrics] = []

    for lib in libs:
        lid = lib["id"]
        lname = lib.get("name", lid)
        m = LibraryMetrics(id=lid, name=lname)

        users = lib_apps.get(lid, set())

        # LC(l) — Library Coverage
        m.coverage = len(users)

        # LCon(l) — Library Concentration
        # max over all nodes of |RUNS(n) ∩ USES(l)|
        max_conc = 0
        for nid, apps_on_node in node_apps.items():
            overlap = len(apps_on_node & users)
            if overlap > max_conc:
                max_conc = overlap
        m.concentration = max_conc

        results.append(m)

    return results
