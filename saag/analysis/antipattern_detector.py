"""
saag/analysis/antipattern_detector.py — Unified Architectural Anti-Pattern Engine
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from saag.analysis.models import DetectedProblem

logger = logging.getLogger("antipattern_detector")

# =============================================================================
# Anti-Pattern Catalog Definitions
# =============================================================================

@dataclass(frozen=True)
class PatternSpec:
    """Static metadata for a single anti-pattern in the catalog."""
    id: str
    name: str
    severity: str           # "CRITICAL" | "HIGH" | "MEDIUM"
    category: str           # "Availability" | "Reliability" | "Maintainability" | "Security" | "Architecture"
    description: str        # What the pattern IS
    risk: str               # Why it is dangerous
    recommendation: str     # How to fix it


# NOTE: PatternSpec.name must not contain "Bottleneck" or "Hub" substrings unless the pattern
# is intentionally meant to route into the prescription stage's god_components remediation
# bucket, which substring-matches on `name` (see saag/prescription/rules.py, `_smells`).
CATALOG: Dict[str, PatternSpec] = {

    # ── Availability ─────────────────────────────────────────────────────────

    "SPOF": PatternSpec(
        id="SPOF",
        name="Single Point of Failure (SPOF)",
        severity="CRITICAL",
        category="Availability",
        description=(
            "A component whose removal disconnects the dependency graph, making "
            "downstream subscribers unreachable."
        ),
        risk="Any failure or maintenance event for this component halts all dependent data flows.",
        recommendation="Introduce redundancy, active-passive failover, or clustered broker configurations."
    ),
    "BRIDGE_EDGE": PatternSpec(
        id="BRIDGE_EDGE",
        name="Bridge Edge (Critical Link)",
        severity="HIGH",
        category="Availability",
        description="A dependency edge whose removal increases the number of connected components.",
        risk="Loss of this single link partitions the system into isolated clusters.",
        recommendation="Add redundant connections or alternative paths."
    ),
    "BOTTLENECK_EDGE": PatternSpec(
        id="BOTTLENECK_EDGE",
        name="Bottleneck Dependency",
        severity="HIGH",
        category="Availability",
        description="An edge carrying an anomalously high percentage of shortest paths.",
        risk="Potential performance bottleneck and shared failure point.",
        recommendation="Consider caching, load balancing, or async patterns."
    ),
    "BROKER_OVERLOAD": PatternSpec(
        id="BROKER_OVERLOAD",
        name="Broker Saturation",
        severity="HIGH",
        category="Availability",
        description=(
            "A broker handling a disproportionate share of message routing relative to its "
            "peers, or the sole broker in the system."
        ),
        risk=(
            "Resource exhaustion on this broker propagates immediately to every producer and "
            "consumer that depends on it; both an availability and a performance risk."
        ),
        recommendation=(
            "Partition the topic namespace across brokers, deploy a broker cluster, or "
            "introduce hierarchical/edge brokers to reduce the central broker's connection count."
        )
    ),

    # ── Reliability ──────────────────────────────────────────────────────────

    "FAILURE_HUB": PatternSpec(
        id="FAILURE_HUB",
        name="Critical Failure Propagation Hub",
        severity="CRITICAL",
        category="Reliability",
        description="A component with critical reliability risk whose failure cascades widely.",
        risk="A failure here triggers a mass outage across many downstream dependents.",
        recommendation="Health checks, circuit breakers in dependents, and retry policies."
    ),
    "CONCENTRATION_RISK": PatternSpec(
        id="CONCENTRATION_RISK",
        name="Concentration Risk",
        severity="MEDIUM",
        category="Reliability",
        description="Top 3 components hold > 50% of transitive system importance (PageRank).",
        risk="The system is fragile because its correct operation depends too heavily on a few nodes.",
        recommendation="Distribute load via domain partitioning or message brokers."
    ),
    "DEEP_PIPELINE": PatternSpec(
        id="DEEP_PIPELINE",
        name="Deep Processing Pipeline",
        severity="HIGH",
        category="Reliability",
        description=(
            "A long directed source-to-sink chain in the application dependency graph "
            "exceeding the layer's adaptive depth threshold."
        ),
        risk=(
            "Latency amplification across stages, multiplicative reliability degradation "
            "along the chain, and observability collapse for root-cause tracing."
        ),
        recommendation=(
            "Merge adjacent stages sharing an ownership boundary, parallelize independent "
            "stages, and instrument per-stage latency SLOs."
        )
    ),
    "TOPIC_FANOUT": PatternSpec(
        id="TOPIC_FANOUT",
        name="Topic Fan-Out Explosion",
        severity="MEDIUM",
        category="Reliability",
        description="A topic with an anomalously large subscriber count relative to other topics in the system.",
        risk=(
            "Broker resource amplification per publish, broadcast blast radius on any "
            "topic-level issue, and subscriber-lag proliferation for durable topics."
        ),
        recommendation=(
            "Segment the topic by semantic sub-topic, introduce a topic router, or replace "
            "fan-out with a shared-state store where appropriate."
        )
    ),
    "QOS_MISMATCH": PatternSpec(
        id="QOS_MISMATCH",
        name="QoS Policy Mismatch",
        severity="MEDIUM",
        category="Reliability",
        description=(
            "A dependency edge where the publisher's QoS weight is substantially lower than "
            "the subscriber's expected QoS weight."
        ),
        risk=(
            "Silent connectivity failures (ROS 2/DDS discovery mismatch) or missing delivery "
            "guarantees (MQTT) that manifest only under failure or high load."
        ),
        recommendation=(
            "Establish a QoS policy registry with CI validation, introduce a QoS-bridging "
            "relay component, or standardize on predefined QoS profiles."
        )
    ),

    # ── Maintainability ──────────────────────────────────────────────────────

    "GOD_COMPONENT": PatternSpec(
        id="GOD_COMPONENT",
        name="God Component / Central Bottleneck",
        severity="CRITICAL",
        category="Maintainability",
        description="A component with extreme betweenness centrality and high coupling.",
        risk="God components are risky to change and hard to test due to too many responsibilities.",
        recommendation="Decompose into smaller services using the Strangler Fig pattern."
    ),
    "HUB_AND_SPOKE": PatternSpec(
        id="HUB_AND_SPOKE",
        name="Hub-and-Spoke Anti-Pattern",
        severity="MEDIUM",
        category="Maintainability",
        description="A hub node where its neighbors do not communicate directly (low clustering).",
        risk="Create bottlenecks and single-failure-point behaviors in local clusters.",
        recommendation="Add direct links between neighbors for redundant paths."
    ),
    "CHATTY_PAIR": PatternSpec(
        id="CHATTY_PAIR",
        name="Chatty Pair",
        severity="MEDIUM",
        category="Maintainability",
        description=(
            "Two application components with a reciprocal, high-weight dependency in both "
            "directions via separate topics."
        ),
        risk=(
            "Hidden logical coupling behind pub-sub indirection: the pair cannot be deployed "
            "or reasoned about independently despite the appearance of decoupling."
        ),
        recommendation=(
            "Introduce a mediator component, apply event-carried state transfer, or replace "
            "conversational exchange with Tell-Don't-Ask."
        )
    ),
    "ORPHANED_TOPIC": PatternSpec(
        id="ORPHANED_TOPIC",
        name="Orphaned Topic",
        severity="MEDIUM",
        category="Maintainability",
        description="A topic with no publishers or no subscribers.",
        risk=(
            "Publisher-only orphans waste broker resources on data nobody consumes; "
            "subscriber-only orphans leave components waiting indefinitely for data that "
            "never arrives."
        ),
        recommendation=(
            "Connect or remove publisher-only orphans; diagnose the missing publisher for "
            "subscriber-only orphans (deployment, naming, or version mismatch)."
        )
    ),
    "UNSTABLE_INTERFACE": PatternSpec(
        id="UNSTABLE_INTERFACE",
        name="Unstable Interface",
        severity="MEDIUM",
        category="Maintainability",
        description=(
            "A component with high maintainability risk driven by near-equal, "
            "high-path-complexity in/out coupling (high CouplingRisk_enh)."
        ),
        risk=(
            "Simultaneously absorbs every upstream change and propagates every one of its own "
            "changes downstream: the system's highest-friction point for independent "
            "deployability."
        ),
        recommendation=(
            "Apply the Stable Abstractions Principle, introduce schema versioning/a schema "
            "registry, and invert unstable dependencies onto a shared aggregation topic."
        )
    ),

    # NOTE: the "Security" category (TARGET, EXPOSURE) was retired along with
    # the Vulnerability/Security dimension — see saag/core/quality_model.py.

    # ── Architecture ──────────────────────────────────────────────────────────

    "CYCLE": PatternSpec(
        id="CYCLE",
        name="Dependency Cycle",
        severity="HIGH",
        category="Architecture",
        description="Circular dependency detected between two or more components.",
        risk="Oscillating message amplification and impossible-to-test feedback loops.",
        recommendation="Break the cycle via interfaces or event-driven decoupling."
    ),
    "CHAIN": PatternSpec(
        id="CHAIN",
        name="Chain Topology",
        severity="MEDIUM",
        category="Architecture",
        description="Fragile chain of components where any failure isolates the entire sequence.",
        risk="Reliability is limited by the product of every node in the sequence.",
        recommendation="Introduce redundant paths or bypasses to reduce sequence depth."
    ),
    "ISOLATED": PatternSpec(
        id="ISOLATED",
        name="Isolated Component",
        severity="MEDIUM",
        category="Architecture",
        description="Component has no dependencies in this layer.",
        risk="May be orphaned, misconfigured, or pending integration.",
        recommendation="Verify deployment, configuration, and integration status."
    ),
    "SYSTEMIC_RISK": PatternSpec(
        id="SYSTEMIC_RISK",
        name="Systemic Risk Pattern",
        severity="CRITICAL",
        category="Architecture",
        description="Significant portion of the system is classified as CRITICAL.",
        risk="Fundamental architectural issues require comprehensive review.",
        recommendation="Architecture review and remediation roadmap."
    ),
    "COMPOUND_RISK": PatternSpec(
        id="COMPOUND_RISK",
        name="Compound Architectural Risk",
        severity="CRITICAL",
        category="Architecture",
        description="Component is simultaneously a structural SPOF and a high-criticality God Component or Failure Hub.",
        risk="Extremely dangerous: the component is critical, hard to change, and its failure isolates the system.",
        recommendation="Urgent: Prioritize decoupling and redundancy for this specific component."
    ),
}


def _fence_ratio(value: float, fence: float) -> Optional[float]:
    """Metric over the threshold it crossed; the delta gate's margin reads it.

    None when the threshold is not positive, so no margin can be applied.
    """
    return value / fence if fence > 0 else None


# =============================================================================
# Detector Engine
# =============================================================================

class AntiPatternDetector:
    """
    Unified engine for detecting architectural anti-patterns and bad smells.
    The single catalog and detection path; ``ProblemDetector`` is a thin
    back-compat wrapper around it.
    """

    def __init__(self, active_patterns: Optional[List[str]] = None) -> None:
        self._active = set(active_patterns) if active_patterns else set(CATALOG.keys())
        unknown = self._active - set(CATALOG.keys())
        if unknown:
            raise ValueError(f"Unknown patterns: {unknown}")
        #: Pattern IDs whose detector raised during the most recent detect()
        #: call. Empty (not just absent-findings) means every active detector
        #: ran cleanly.
        self.failed_patterns: List[str] = []

    def detect(self, layer_result: Any, layer: str) -> List[DetectedProblem]:
        """Run all active detectors against a LayerAnalysisResult."""
        problems: List[DetectedProblem] = []
        # Reset per call: a crashing detector must be distinguishable from a
        # clean scan that found nothing for that pattern. Callers that care
        # (e.g. the predict CLI's CI gate) read this after detect() returns.
        self.failed_patterns: List[str] = []
        components = layer_result.components if layer_result else []
        if not components:
            return problems

        edges = getattr(layer_result, "edges", [])
        stats = self._compute_distribution_stats(components, edges)

        detectors = {
            "SPOF":               self._detect_spof,
            "BRIDGE_EDGE":        self._detect_bridge_edge,
            "FAILURE_HUB":        self._detect_failure_hub,
            "CONCENTRATION_RISK": self._detect_concentration_risk,
            "GOD_COMPONENT":      self._detect_god_component,
            "HUB_AND_SPOKE":      self._detect_hub_and_spoke,
            "BOTTLENECK_EDGE":    self._detect_bottleneck_edge,
            "CYCLE":              self._detect_cycle,
            "CHAIN":              self._detect_chain,
            "ISOLATED":           self._detect_isolated,
            "SYSTEMIC_RISK":      self._detect_systemic_risk,
            "BROKER_OVERLOAD":    self._detect_broker_overload,
            "TOPIC_FANOUT":       self._detect_topic_fanout,
            "QOS_MISMATCH":       self._detect_qos_mismatch,
            "DEEP_PIPELINE":      self._detect_deep_pipeline,
            "CHATTY_PAIR":        self._detect_chatty_pair,
            "ORPHANED_TOPIC":     self._detect_orphaned_topic,
            "UNSTABLE_INTERFACE": self._detect_unstable_interface,
        }

        for pid, detector_fn in detectors.items():
            if pid not in self._active:
                continue
            try:
                found = detector_fn(layer_result, layer, stats)
                problems.extend(found)
            except Exception as exc:
                logger.exception("Detector %s failed: %s", pid, exc)
                self.failed_patterns.append(pid)

        # Issue #13: Compound risk post-pass
        if "COMPOUND_RISK" in self._active:
            problems.extend(self._detect_compound_risk(problems))

        _severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        problems.sort(key=lambda p: (_severity_order.get(p.severity, 9), p.entity_id))
        return problems

    def summarize(self, problems: List[DetectedProblem]) -> Dict[str, Any]:
        """Summarize detected problems by category and severity."""
        summary = {
            "total": len(problems),
            "by_severity": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "by_category": {},
            "affected_entities": len(set(p.entity_id for p in problems))
        }
        for p in problems:
            summary["by_severity"][p.severity] = summary["by_severity"].get(p.severity, 0) + 1
            summary["by_category"][p.category] = summary["by_category"].get(p.category, 0) + 1
        return summary

    # ── Helpers ──────────────────────────────────────────────────────

    #: Minimum per-type sample size for stratified box-plot fences, matching
    #: QualityClassifier._MIN_GROUP in saag/analysis/classifier.py.
    _MIN_TYPE_GROUP: int = 8

    def _make_problem(self, pid: str, entity_id: str, evidence: Dict[str, Any], 
                       entity_type: str = "Component") -> DetectedProblem:
        spec = CATALOG[pid]
        return DetectedProblem(
            entity_id=entity_id, entity_type=entity_type, 
            category=spec.category, severity=spec.severity,
            name=spec.name, description=spec.description,
            recommendation=spec.recommendation, evidence=evidence,
            pattern_id=pid,
        )

    @staticmethod
    def _compute_distribution_stats(components, edges=None) -> Dict[str, Any]:
        """Compute pooled and per-type stratified box-plot fences for metrics."""
        def _boxplot_fence(values: List[float]) -> Tuple[float, float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0, 1.0
            sv = sorted(values)
            n = len(sv)
            q1, median, q3 = sv[n // 4], sv[n // 2], sv[min(3 * n // 4, n - 1)]
            return q1, median, q3, q3 + 1.5 * (q3 - q1)

        overall_scores = [c.scores.overall for c in components]
        avail_scores = [c.scores.availability for c in components]
        rel_scores = [c.scores.reliability for c in components]
        bt_scores = [getattr(c.structural, "betweenness", 0.0) for c in components]
        total_degrees = [getattr(c.structural, 'in_degree_raw', 0) + getattr(c.structural, 'out_degree_raw', 0) for c in components]
        in_degrees = [getattr(c.structural, 'in_degree_raw', 0) for c in components]
        out_degrees = [getattr(c.structural, 'out_degree_raw', 0) for c in components]

        _, _, q3_q, fence_q = _boxplot_fence(overall_scores)
        _, _, q3_deg, fence_deg = _boxplot_fence(total_degrees)
        _, _, q3_rel, fence_rel = _boxplot_fence(rel_scores)
        _, _, q3_avail, fence_avail = _boxplot_fence(avail_scores)
        _, _, _, fence_bt = _boxplot_fence(bt_scores)
        if len(components) < 4:
            # Degenerate tiny-sample fallback for betweenness fence
            fence_bt = 0.3

        median_in = statistics.median(in_degrees) if in_degrees else 0
        median_out = statistics.median(out_degrees) if out_degrees else 0

        # Per-type stratified fences (mirrors QualityClassifier._MIN_GROUP = 8)
        by_type: Dict[str, list] = {}
        for c in components:
            by_type.setdefault(c.type, []).append(c)

        fence_rel_by_type: Dict[str, float] = {}
        fence_avail_by_type: Dict[str, float] = {}
        fence_bt_by_type: Dict[str, float] = {}
        median_in_by_type: Dict[str, float] = {}
        for ctype, group in by_type.items():
            if len(group) >= AntiPatternDetector._MIN_TYPE_GROUP:
                _, _, _, fence_rel_by_type[ctype] = _boxplot_fence([c.scores.reliability for c in group])
                _, _, _, fence_avail_by_type[ctype] = _boxplot_fence([c.scores.availability for c in group])
                _, _, _, fence_bt_by_type[ctype] = _boxplot_fence([getattr(c.structural, "betweenness", 0.0) for c in group])
                median_in_by_type[ctype] = statistics.median(getattr(c.structural, "in_degree_raw", 0) for c in group)

        # TOPIC_FANOUT: fence computed over Topic-type components only, using raw
        # subscriber_count (not the DEPENDS_ON-projected out_degree_raw, which is always 0
        # for Topics), so Application-type degree distributions don't pollute the threshold.
        topic_sub_counts = [
            getattr(c.structural, "topic_subscriber_count", 0) for c in components if c.type == "Topic"
        ]
        _, _, q3_topic_out, fence_topic_out_raw = _boxplot_fence(topic_sub_counts)
        fence_topic_out = max(fence_topic_out_raw, 5)  # doc's floor of 5 subscribers

        # BOTTLENECK_EDGE: adaptive fence over the edge-betweenness distribution
        # (docs/antipatterns.md §5.5: edge_BT(u,v) > Q3_edge_BT + 1.5 × IQR_edge_BT)
        edge_betweennesses = [getattr(e.structural, "betweenness", 0) for e in (edges or [])]
        _, _, q3_edge_bt, fence_edge_bt = _boxplot_fence(edge_betweennesses)

        return {
            "fence_q": fence_q, "q3_degree": q3_deg, "fence_degree": fence_deg,
            "fence_rel": fence_rel, "fence_avail": fence_avail, "fence_bt": fence_bt,
            "fence_rel_by_type": fence_rel_by_type,
            "fence_avail_by_type": fence_avail_by_type,
            "fence_bt_by_type": fence_bt_by_type,
            "median_in": median_in, "median_in_by_type": median_in_by_type,
            "median_out": median_out, "total_count": len(components),
            "fence_topic_out": fence_topic_out, "fence_edge_bt": fence_edge_bt
        }

    # ── Detectors ────────────────────────────────────────────────────

    def _detect_spof(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        # docs/antipatterns.md §5.1: SPOF(v) <-> AP_c(v) > 0 OR A(v) > upper_fence(A)
        pooled_fence_avail = stats.get("fence_avail", 1.0)
        fence_avail_by_type = stats.get("fence_avail_by_type", {})
        for c in lr.components:
            fence_avail = fence_avail_by_type.get(c.type, pooled_fence_avail)
            # Issue #10: Use directed AP if available
            is_spof = getattr(c.structural, 'is_directed_ap', False)
            if is_spof:
                out.append(self._make_problem("SPOF", c.id, {"availability_level": c.levels.availability.value, "is_directed": True, "trigger": "articulation_point"}))
            elif getattr(c.structural, 'is_articulation_point', False):
                 out.append(self._make_problem("SPOF", c.id, {"availability_level": c.levels.availability.value, "is_directed": False, "trigger": "articulation_point"}))
            elif c.scores.availability > fence_avail:
                out.append(self._make_problem("SPOF", c.id, {"availability_level": c.levels.availability.value, "availability_score": c.scores.availability, "trigger": "availability_fence",
                                                             "fence_ratio": _fence_ratio(c.scores.availability, fence_avail)}))
        return out

    def _detect_bridge_edge(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for e in getattr(lr, "edges", []):
            if getattr(e.structural, 'is_bridge', False):
                out.append(self._make_problem("BRIDGE_EDGE", e.id, {"is_bridge": True}, entity_type="Edge"))
        return out

    def _detect_failure_hub(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        # DEPENDS_ON edges point dependent -> dependency, so a failure hub whose
        # outage cascades widely has high Reliability risk R(v) AND high in_degree_raw.
        pooled_rel_fence = stats.get("fence_rel", 0.8)
        fence_rel_by_type = stats.get("fence_rel_by_type", {})
        pooled_median_in = stats.get("median_in", 0)
        median_in_by_type = stats.get("median_in_by_type", {})

        for c in lr.components:
            rel_fence = fence_rel_by_type.get(c.type, pooled_rel_fence)
            median_in = median_in_by_type.get(c.type, pooled_median_in)
            if c.scores.reliability > rel_fence and c.structural.in_degree_raw > median_in:
                out.append(self._make_problem("FAILURE_HUB", c.id, {
                    "reliability": c.scores.reliability,
                    "in_degree": c.structural.in_degree_raw,
                    "fence_ratio": _fence_ratio(c.scores.reliability, rel_fence),
                }))
        return out

    def _detect_concentration_risk(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        comps = lr.components
        if len(comps) < 5: return []
        sorted_pr = sorted(comps, key=lambda c: getattr(c.structural, 'pagerank', 0), reverse=True)
        top3_pr = sum(getattr(c.structural, 'pagerank', 0) for c in sorted_pr[:3])
        if top3_pr > 0.5:
            return [self._make_problem("CONCENTRATION_RISK", "SYSTEM", {"top3_pagerank": top3_pr}, entity_type="System")]
        return []

    def _detect_god_component(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        from saag.core.criticality import CriticalityLevel
        pooled_bt_fence = stats.get("fence_bt", 0.3)
        fence_bt_by_type = stats.get("fence_bt_by_type", {})
        for c in lr.components:
            bt_fence = fence_bt_by_type.get(c.type, pooled_bt_fence)
            betweenness = getattr(c.structural, 'betweenness', 0.0)
            if betweenness > 0.0 and betweenness > bt_fence and c.levels.maintainability >= CriticalityLevel.CRITICAL:
                out.append(self._make_problem("GOD_COMPONENT", c.id, {
                    "betweenness": betweenness,
                    "bt_fence": bt_fence,
                    "fence_ratio": _fence_ratio(betweenness, bt_fence),
                }))
        return out

    def _detect_hub_and_spoke(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for c in lr.components:
            cc = getattr(c.structural, 'clustering_coefficient', 1.0)
            deg = getattr(c.structural, 'in_degree_raw', 0) + getattr(c.structural, 'out_degree_raw', 0)
            if cc < 0.1 and deg > 3:
                out.append(self._make_problem("HUB_AND_SPOKE", c.id, {"clustering": cc, "degree": deg}))
        return out

    def _detect_bottleneck_edge(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        fence_edge_bt = stats.get("fence_edge_bt", 0.2)
        for e in getattr(lr, "edges", []):
            betweenness = getattr(e.structural, 'betweenness', 0)
            if betweenness > fence_edge_bt:
                out.append(self._make_problem("BOTTLENECK_EDGE", e.id, {"betweenness": betweenness,
                                                                        "fence_ratio": _fence_ratio(betweenness, fence_edge_bt)},
                                              entity_type="Edge"))
        return out

    def _detect_cycle(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        import networkx as nx
        # Issue #11: Use original graph if available to catch cycles lost in closure
        G = getattr(lr, "graph", None)
        if G is None:
            G = nx.DiGraph()
            for e in getattr(lr, "edges", []):
                G.add_edge(e.source, e.target)
        
        for scc in nx.strongly_connected_components(G):
            if len(scc) >= 2:
                # Filter for non-trivial cycles (NetworkX might return single apps if they have self-loops)
                if len(scc) > 1 or G.has_edge(list(scc)[0], list(scc)[0]):
                    out.append(self._make_problem("CYCLE", " -> ".join(sorted(scc)), {"size": len(scc)}, entity_type="Architecture"))
        return out

    def _detect_chain(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        import networkx as nx
        G = nx.DiGraph()
        for e in getattr(lr, "edges", []):
            G.add_edge(e.source, e.target)
        chain_nodes = [v for v in G.nodes if G.in_degree(v) <= 1 and G.out_degree(v) <= 1]
        H = G.subgraph(chain_nodes)
        out = []
        for component in nx.weakly_connected_components(H):
            if len(component) >= 4:
                ordered = sorted(component)
                out.append(self._make_problem("CHAIN", f"CHAIN-{''.join(ordered[:3])}", {"length": len(component)}, entity_type="Architecture"))
        return out

    def _detect_isolated(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for c in lr.components:
            # A switch/router with no node_to_device edge is one no host is forced
            # through -- redundancy, not a disconnected component.
            if c.type == "NetworkDevice":
                continue
            if getattr(c.structural, 'is_isolated', False):
                out.append(self._make_problem("ISOLATED", c.id, {"isolated": True}))
        return out

    def _detect_systemic_risk(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        from saag.core.criticality import CriticalityLevel
        crit_count = sum(1 for c in lr.components if c.levels.overall == CriticalityLevel.CRITICAL)
        total = stats["total_count"]
        if total > 0 and (crit_count / total) > 0.2:
            return [self._make_problem("SYSTEMIC_RISK", "SYSTEM", {"critical_ratio": crit_count/total,
                                                                   "fence_ratio": _fence_ratio(crit_count / total, 0.2)},
                                       entity_type="System")]
        return []

    def _detect_broker_overload(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        brokers = [c for c in lr.components if c.type == "Broker"]
        if not brokers:
            return out
        if len(brokers) == 1:
            b = brokers[0]
            out.append(self._make_problem("BROKER_OVERLOAD", b.id, {"sole_broker": True, "availability": b.scores.availability}))
            return out
        median_a = statistics.median(b.scores.availability for b in brokers)
        for b in brokers:
            if median_a > 0 and b.scores.availability >= 2 * median_a:
                out.append(self._make_problem("BROKER_OVERLOAD", b.id, {
                    "availability": b.scores.availability, "median_broker_availability": median_a,
                    "fence_ratio": _fence_ratio(b.scores.availability, 2 * median_a),
                }))
        return out

    def _detect_topic_fanout(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        fence = stats.get("fence_topic_out", 5)
        for c in lr.components:
            if c.type != "Topic":
                continue
            sub_count = getattr(c.structural, "topic_subscriber_count", 0)
            if sub_count > fence:
                out.append(self._make_problem("TOPIC_FANOUT", c.id, {"subscriber_count": sub_count, "fence": fence}))
        return out

    def _detect_orphaned_topic(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for c in lr.components:
            if c.type != "Topic":
                continue
            subs = getattr(c.structural, "topic_subscriber_count", 0)
            pubs = getattr(c.structural, "topic_publisher_count", 0)
            # A topic with zero publishers AND zero subscribers is fully disconnected and
            # already covered by ISOLATED; only flag single-sided orphans here.
            if subs == 0 and pubs > 0:
                out.append(self._make_problem("ORPHANED_TOPIC", c.id, {
                    "orphan_type": "publisher_only", "publisher_count": pubs, "subscriber_count": 0
                }))
            elif pubs == 0 and subs > 0:
                out.append(self._make_problem("ORPHANED_TOPIC", c.id, {
                    "orphan_type": "subscriber_only", "publisher_count": 0, "subscriber_count": subs
                }))
        return out

    def _detect_qos_mismatch(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        by_id = {c.id: c for c in lr.components}
        for e in getattr(lr, "edges", []):
            if getattr(e, "dependency_type", "") != "app_to_app":
                continue
            # DEPENDS_ON (app_to_app) points dependent (subscriber) -> dependency (publisher).
            sub, pub = by_id.get(e.source), by_id.get(e.target)
            if sub is None or pub is None:
                continue
            w_sub = getattr(sub.structural, "weight", 1.0)
            w_pub = getattr(pub.structural, "weight", 1.0)
            if w_pub < w_sub - 0.3:
                out.append(self._make_problem("QOS_MISMATCH", e.id, {
                    "w_publisher": w_pub, "w_subscriber": w_sub, "gap": w_sub - w_pub
                }, entity_type="Edge"))
        return out

    def _detect_chatty_pair(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        edge_score: Dict[Tuple[str, str], float] = {}
        for e in getattr(lr, "edges", []):
            score = getattr(e.scores, "overall", None) if hasattr(e, "scores") else None
            if score is None:
                score = getattr(getattr(e, "structural", None), "weight", 0.0)
            edge_score[(e.source, e.target)] = score

        seen = set()
        for (u, v), fwd in edge_score.items():
            if (u, v) in seen or (v, u) in seen:
                continue
            rev = edge_score.get((v, u))
            if rev is None:
                continue
            seen.add((u, v))
            seen.add((v, u))
            if fwd * rev > 0.25:
                out.append(self._make_problem("CHATTY_PAIR", f"{u}<->{v}", {
                    "score_fwd": fwd, "score_rev": rev, "product": fwd * rev
                }, entity_type="Edge"))
        return out

    @staticmethod
    def _p75_path_length(G) -> int:
        import networkx as nx
        lengths = [
            d
            for src in G.nodes
            for tgt, d in nx.single_source_shortest_path_length(G, src).items()
            if tgt != src
        ]
        if not lengths:
            return 0
        sl = sorted(lengths)
        return sl[int(0.75 * (len(sl) - 1))]

    #: Upper bound on DEEP_PIPELINE findings emitted per scan to prevent alert flooding.
    _MAX_DEEP_PIPELINES: int = 100

    def _enumerate_deep_pipelines(
        self, G, sources, sinks, tau: int, max_findings: int = _MAX_DEEP_PIPELINES
    ) -> List[DetectedProblem]:
        """Find longest source-to-sink pipelines in polynomial time via SCC condensation.

        Condensing strongly connected components turns any directed graph (including
        cyclic ones) into a DAG C in O(V + E). Longest-path DP on C then runs in
        O(|sources| * (V + E)) and emits at most one canonical longest path per
        (source_scc, sink_scc) pair — eliminating the exponential blowup of
        nx.all_simple_paths on dense/cyclic topologies.
        """
        import networkx as nx

        C = nx.condensation(G)
        if C.number_of_nodes() == 0:
            return []

        mapping = C.graph["mapping"]  # original node -> scc_id
        # Canonical representative node per SCC (deterministic across runs)
        scc_rep: Dict[int, str] = {
            scc_id: min(str(n) for n in C.nodes[scc_id]["members"])
            for scc_id in C.nodes
        }

        source_sccs = sorted({mapping[s] for s in sources if s in mapping})
        sink_sccs = {mapping[t] for t in sinks if t in mapping}
        if not source_sccs or not sink_sccs:
            return []

        topo_order = list(nx.topological_sort(C))
        candidates: List[Tuple[int, str, List[str]]] = []

        for src_scc in source_sccs:
            dist: Dict[int, int] = {src_scc: 0}
            pred: Dict[int, Optional[int]] = {src_scc: None}
            for u in topo_order:
                if u not in dist:
                    continue
                du = dist[u]
                for v in C.successors(u):
                    cand_d = du + 1
                    if cand_d > dist.get(v, -1) or (
                        cand_d == dist.get(v, -1)
                        and pred.get(v) is not None
                        and scc_rep[u] < scc_rep[pred[v]]
                    ):
                        dist[v] = cand_d
                        pred[v] = u

            for tgt_scc in sink_sccs:
                hops = dist.get(tgt_scc, -1)
                if hops < tau:
                    continue
                # Reconstruct canonical path
                scc_path: List[int] = []
                cur: Optional[int] = tgt_scc
                while cur is not None:
                    scc_path.append(cur)
                    cur = pred.get(cur)
                scc_path.reverse()
                node_path = [scc_rep[s] for s in scc_path]
                path_str = " -> ".join(node_path)
                candidates.append((hops, path_str, node_path))

        # Deterministic ordering: longest pipelines first, then lexicographical path
        candidates.sort(key=lambda item: (-item[0], item[1]))
        out: List[DetectedProblem] = []
        seen_paths = set()
        for hops, path_str, _ in candidates:
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            out.append(self._make_problem(
                "DEEP_PIPELINE", path_str,
                {"hops": hops, "tau": tau}, entity_type="Architecture"
            ))
            if len(out) >= max_findings:
                break
        return out

    def _detect_deep_pipeline(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        import networkx as nx
        G = getattr(lr, "graph", None)
        if G is None:
            G = nx.DiGraph()
            for e in getattr(lr, "edges", []):
                G.add_edge(e.source, e.target)
        if G.number_of_nodes() == 0:
            return []

        sources = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        sinks = [n for n in G.nodes if G.out_degree(n) == 0 and G.in_degree(n) > 0]
        if not sources or not sinks:
            return []

        tau = max(5, self._p75_path_length(G))
        return self._enumerate_deep_pipelines(G, sources, sinks, tau)

    def _detect_unstable_interface(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for c in lr.components:
            m_score = c.scores.maintainability
            coupling_risk_enh = getattr(c.structural, "coupling_risk_enh", 0.0)
            if m_score > 0.80 and coupling_risk_enh > 0.80:
                out.append(self._make_problem("UNSTABLE_INTERFACE", c.id, {
                    "maintainability": m_score, "coupling_risk_enh": coupling_risk_enh
                }))
        return out

    def _detect_compound_risk(self, problems: List[DetectedProblem]) -> List[DetectedProblem]:
        """Issue #13: Post-pass to identify nodes with multiple critical problems."""
        _name_to_id = {spec.name: pid for pid, spec in CATALOG.items()}
        by_entity: Dict[str, List[Tuple[Optional[str], str]]] = {}
        for p in problems:
            if p.entity_type == "Component":
                pid = p.pattern_id or _name_to_id.get(p.name)
                by_entity.setdefault(p.entity_id, []).append((pid, p.name))

        out = []
        for eid, entries in by_entity.items():
            pids = {pid for pid, _ in entries if pid is not None}
            is_spof = "SPOF" in pids
            is_god_or_hub = bool(pids & {"GOD_COMPONENT", "FAILURE_HUB"})

            if is_spof and is_god_or_hub:
                out.append(self._make_problem(
                    "COMPOUND_RISK",
                    eid,
                    {"risks": [name for _, name in entries]},
                    entity_type="Component"
                ))
        return out
