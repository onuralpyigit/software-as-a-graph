"""
saag/prescription/rules.py

Compiles the transformation policy Delta(G) from structural analysis and
predicted risk. Pure: a function of its inputs, with no repository access and no
simulation, so a rule can be exercised against a hand-built GraphData.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from saag.core.models import GraphData

from .models import NodeReallocation, PrescriptionPolicy, QosUpgrade, TopicSplit

#: Criticality tiers that make a component eligible for remediation.
_CRITICAL_LEVELS = ("CRITICAL", "HIGH")

#: RMAV dimensions plus the aggregate, each of which can flag a component.
_RMAV_DIMENSIONS = ("reliability", "maintainability", "availability", "security", "overall")


def compile_policy(
    graph_data: GraphData,
    analysis_result: Any,
    prediction_result: Optional[Any] = None,
) -> PrescriptionPolicy:
    """Evaluate structural anomalies and predicted risks into a policy."""
    critical = _predicted_critical(prediction_result) | _scored_critical(
        analysis_result, prediction_result)
    spof, god = _smells(analysis_result, prediction_result)
    publishers, subscribers, hosted = _endpoints(graph_data)

    policy = PrescriptionPolicy()
    for component in graph_data.components:
        if component.component_type == "Topic":
            _topic_edits(policy, component, publishers, subscribers, critical, god)
        elif component.component_type == "Node":
            _node_edits(policy, component, hosted, critical, spof)
    return policy


# ── Risk extraction ───────────────────────────────────────────────────────────

def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off an object or a mapping.

    Problems and predictions arrive either as domain objects or as decoded JSON
    depending on the caller, and an object whose attribute is set but empty must
    not be mistaken for a mapping.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _predicted_critical(prediction_result: Optional[Any]) -> Set[str]:
    """Component ids the Predict stage scored CRITICAL/HIGH.

    Accepts the three shapes the stage is reachable in: the SDK facade with
    ``all_components``, a decoded JSON payload with ``node_scores``, and the raw
    GNN result behind a facade's ``.raw``.
    """
    if not prediction_result:
        return set()

    components = _field(prediction_result, "all_components", []) or []
    if components:
        return {
            c.id for c in components
            if c.criticality_level.upper() in _CRITICAL_LEVELS
        }

    if isinstance(prediction_result, dict):
        node_scores = prediction_result.get("node_scores", {})
    else:
        node_scores = _field(_field(prediction_result, "raw"), "node_scores", {}) or {}

    if not isinstance(node_scores, dict):
        return set()
    return {
        comp_id for comp_id, info in node_scores.items()
        if str(_field(info, "criticality_level", "MINIMAL")).upper() in _CRITICAL_LEVELS
    }


def _scored_critical(analysis_result: Any, prediction_result: Optional[Any]) -> Set[str]:
    """Component ids flagged CRITICAL/HIGH on any RMAV dimension.

    ``analysis_result.quality`` is always None now that Analyze (Step 2) is
    structural-only, so the RMAV scores are taken from the Predict stage's raw
    result instead.
    """
    quality = getattr(analysis_result, "quality", None)
    if quality is None:
        raw_prediction = _field(prediction_result, "raw", prediction_result)
        if hasattr(raw_prediction, "components"):
            quality = raw_prediction

    if quality is None or not hasattr(quality, "components"):
        return _legacy_scored_critical(analysis_result)

    critical = set()
    for component in quality.components:
        for dimension in _RMAV_DIMENSIONS:
            level = getattr(component.levels, dimension, None)
            if level and getattr(level, "name", str(level)).upper() in _CRITICAL_LEVELS:
                critical.add(component.id)
                break
    return critical


def _legacy_scored_critical(analysis_result: Any) -> Set[str]:
    """Fallback for analysis results that still carry their own scored components."""
    components = getattr(analysis_result, "all_components", None)
    if not components:
        return set()

    critical = set()
    for component in components:
        levels = getattr(
            component, "criticality_levels", getattr(component, "levels", {}))
        if component.criticality_level.upper() in _CRITICAL_LEVELS or any(
            str(level).upper() in _CRITICAL_LEVELS for level in levels.values()
        ):
            critical.add(component.id)
    return critical


def _smells(analysis_result: Any, prediction_result: Optional[Any]) -> Tuple[Set[str], Set[str]]:
    """Split detected anti-patterns into (SPOF, god-component) entity ids.

    Anti-patterns are matched on the problem's display name rather than a
    pattern-ID field. `tests/test_antipattern_detector.py` guards the catalog
    names this depends on — renaming a `PatternSpec` silently unlinks the
    operator it feeds.
    """
    problems = (
        _field(prediction_result, "problems")
        or _field(analysis_result, "problems")
        or []
    )

    spof: Set[str] = set()
    god: Set[str] = set()
    for problem in problems:
        name = _field(problem, "name", "") or ""
        entity_id = _field(problem, "entity_id", "") or ""
        if "SPOF" in name or "Single Point of Failure" in name:
            spof.add(entity_id)
        if "God Component" in name or "Bottleneck" in name or "Hub" in name:
            god.add(entity_id)
    return spof, god


def _endpoints(
    graph_data: GraphData,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """One pass over the edges for publishers, subscribers and hosted processes."""
    publishers: Dict[str, List[str]] = {}
    subscribers: Dict[str, List[str]] = {}
    hosted: Dict[str, List[str]] = {}

    buckets = {
        "PUBLISHES_TO": publishers,
        "SUBSCRIBES_TO": subscribers,
        "RUNS_ON": hosted,
    }
    for edge in graph_data.edges:
        bucket = buckets.get(edge.relation_type)
        if bucket is not None:
            bucket.setdefault(edge.target_id, []).append(edge.source_id)
    return publishers, subscribers, hosted


# ── The three remediation operators ───────────────────────────────────────────

def _topic_edits(
    policy: PrescriptionPolicy,
    topic: Any,
    publishers: Dict[str, List[str]],
    subscribers: Dict[str, List[str]],
    critical: Set[str],
    god: Set[str],
) -> None:
    """Rules 1 and 3: split congested topics, harden critical channels."""
    pubs = publishers.get(topic.id, [])
    subs = subscribers.get(topic.id, [])
    endpoints = (*pubs, *subs)

    # Rule 1 — Logical Subgraph Refactoring. A topic is congested when it
    # multiplexes several publishers into several subscribers, or when it fans
    # several publishers into a component that is itself critical or a god node.
    is_congested = len(pubs) > 1 and len(subs) > 1
    is_critical_multi = topic.id in critical and len(pubs) > 1
    feeds_god = len(pubs) > 1 and any(e in god for e in endpoints)
    if is_congested or is_critical_multi or feeds_god:
        policy.topic_splits.append(TopicSplit(
            topic=topic.id,
            publishers=sorted(set(pubs)),
            subscribers=sorted(set(subs)),
        ))

    # Rule 3 — Middleware Transport Contract Hardening. Already-hardened topics
    # are skipped, so a compiled upgrade always changes something.
    is_critical_channel = topic.id in critical or any(e in critical for e in endpoints)
    reliability = topic.properties.get("qos_reliability", "BEST_EFFORT")
    durability = topic.properties.get("qos_durability", "VOLATILE")
    if is_critical_channel and (reliability != "RELIABLE" or durability == "VOLATILE"):
        policy.qos_upgrades.append(QosUpgrade(
            topic=topic.id,
            original_reliability=reliability,
            original_durability=durability,
        ))


def _node_edits(
    policy: PrescriptionPolicy,
    node: Any,
    hosted: Dict[str, List[str]],
    critical: Set[str],
    spof: Set[str],
) -> None:
    """Rule 2: Physical Locality Anti-Affinity.

    When a host or anything it runs is a SPOF or critical and it co-locates
    several processes, every process but the lexicographically first is moved
    onto its own node.
    """
    processes = hosted.get(node.id, [])
    if len(processes) <= 1:
        return

    at_risk = (
        node.id in spof
        or node.id in critical
        or any(p in spof or p in critical for p in processes)
    )
    if not at_risk:
        return

    for process in sorted(processes)[1:]:
        policy.node_reallocations.append(NodeReallocation(
            component=process,
            from_node=node.id,
            to_node=f"{node.id}_{process}",
        ))
