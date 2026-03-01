#!/usr/bin/env python3
"""
bin/detect_antipatterns.py — Pub-Sub Architectural Anti-Pattern & Bad Smell Detector
=====================================================================================
Catalogs and detects architectural anti-patterns and bad smells in distributed
publish-subscribe systems using graph topology analysis.

Each finding tells you:
  • WHAT kind of architectural risk is present (named pattern)
  • WHICH components are involved
  • WHY it is dangerous (root-cause explanation)
  • HOW to fix it (concrete refactoring recommendation)
  • WHICH RMAV quality dimension it primarily degrades

Anti-Pattern Catalog (12 patterns across 3 severity tiers):

  CRITICAL ─────────────────────────────────────────────────────────────────
    SPOF               Single Point of Failure — structural graph cut vertex
    SYSTEMIC_RISK      Correlated failure cluster — CRITICAL clique
    CYCLIC_DEPENDENCY  Circular pub-sub feedback loop (SCC > 1)

  HIGH ─────────────────────────────────────────────────────────────────────
    GOD_COMPONENT      Dependency magnet — absorbs too many responsibilities
    BOTTLENECK_EDGE    High-traffic bridge with no redundant path
    BROKER_OVERLOAD    Broker saturation — disproportionate routing share
    DEEP_PIPELINE      Excessive processing chain depth — latency amplifier

  MEDIUM ───────────────────────────────────────────────────────────────────
    TOPIC_FANOUT       Topic fan-out explosion — broadcast blast radius
    CHATTY_PAIR        Bidirectional tight coupling through topics
    QOS_MISMATCH       Publisher/subscriber QoS incompatibility
    ORPHANED_TOPIC     Topic with no publishers OR no subscribers
    UNSTABLE_INTERFACE High churn potential — extreme coupling imbalance

Usage:
    # Detect all patterns in the system layer
    python bin/detect_antipatterns.py --layer system

    # Detect patterns in the application layer, show only CRITICAL/HIGH
    python bin/detect_antipatterns.py --layer app --severity critical,high

    # Detect a specific subset of patterns
    python bin/detect_antipatterns.py --all --pattern spof,broker_overload,cyclic_dependency

    # Print the full anti-pattern catalog (no Neo4j needed)
    python bin/detect_antipatterns.py --catalog

    # Export findings to JSON for downstream tooling
    python bin/detect_antipatterns.py --layer system --output results/smells.json

    # Full scan across all layers, verbose, export JSON
    python bin/detect_antipatterns.py --all --output results/smells.json --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from project root as `python bin/<script>`
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.core import create_repository, AnalysisLayer
from src.analysis import AnalysisService
from src.cli.console import ConsoleDisplay

logger = logging.getLogger("detect_antipatterns")


# =============================================================================
# Anti-Pattern Catalog Definitions
# =============================================================================

@dataclass(frozen=True)
class PatternSpec:
    """Static metadata for a single anti-pattern in the catalog."""
    id: str
    name: str
    severity: str           # "CRITICAL" | "HIGH" | "MEDIUM"
    rmav_dimension: str     # Primary quality dimension degraded
    description: str        # What the pattern IS
    risk: str               # Why it is dangerous
    recommendation: str     # How to fix it
    references: List[str]   # Related literature / patterns


CATALOG: Dict[str, PatternSpec] = {

    # ── CRITICAL ────────────────────────────────────────────────────────────

    "SPOF": PatternSpec(
        id="SPOF",
        name="Single Point of Failure",
        severity="CRITICAL",
        rmav_dimension="Availability",
        description=(
            "A component whose removal disconnects the dependency graph, making "
            "downstream subscribers unreachable. Detected when AP_c > 0 (non-zero "
            "continuous articulation point score)."
        ),
        risk=(
            "Any failure or maintenance event for this component halts all dependent "
            "data flows. The system has no redundant path to restore connectivity, "
            "making this the highest-priority architectural risk."
        ),
        recommendation=(
            "1. Introduce a redundant replica behind a load balancer or active-passive "
            "failover pair. "
            "2. For brokers: deploy a clustered broker configuration (e.g., Kafka "
            "partition replication, RabbitMQ mirrored queues). "
            "3. For application SPOFs: extract the critical function into a stateless "
            "microservice that can be horizontally scaled. "
            "4. Add health-check and circuit-breaker patterns around this component."
        ),
        references=["RMAV: Availability (QSPOF)", "IEEE 1012 Reliability"],
    ),

    "SYSTEMIC_RISK": PatternSpec(
        id="SYSTEMIC_RISK",
        name="Systemic Risk Cluster",
        severity="CRITICAL",
        rmav_dimension="Reliability",
        description=(
            "Three or more CRITICAL-tier components that share mutual DEPENDS_ON "
            "edges, forming a densely connected failure clique. A failure in any one "
            "member can propagate to all others almost simultaneously."
        ),
        risk=(
            "Correlated failures cascade through the entire cluster. The blast radius "
            "is multiplicative: if component A fails and triggers B, and B triggers C, "
            "the system experiences a compound outage that is far more severe than "
            "any single-component failure."
        ),
        recommendation=(
            "1. Introduce anti-corruption layer (ACL) boundaries between cluster "
            "members to break direct dependency chains. "
            "2. Convert synchronous DEPENDS_ON edges between cluster members into "
            "asynchronous pub-sub relationships with a dedicated internal topic, "
            "adding back-pressure and retry semantics. "
            "3. Apply bulkhead isolation: deploy each cluster member in separate "
            "process/container groups with independent resource pools. "
            "4. Implement saga patterns for multi-step workflows spanning cluster members."
        ),
        references=["RMAV: Reliability (RPR, DG_in)", "Michael Nygard: Release It!"],
    ),

    "CYCLIC_DEPENDENCY": PatternSpec(
        id="CYCLIC_DEPENDENCY",
        name="Cyclic Dependency Loop",
        severity="CRITICAL",
        rmav_dimension="Maintainability",
        description=(
            "A strongly connected component (SCC) with more than one node in the "
            "application dependency layer. Applications form a publish-subscribe "
            "feedback loop: A publishes to a topic that B subscribes to, and B "
            "publishes to a topic that A subscribes to (directly or transitively)."
        ),
        risk=(
            "Cyclic dependencies create oscillating message amplification, potential "
            "infinite publish loops under fault conditions, and make the system "
            "impossible to test or reason about in isolation. Maintenance changes "
            "to any member ripple unpredictably around the cycle."
        ),
        recommendation=(
            "1. Break the cycle by converting one edge into a unidirectional "
            "event-driven notification (e.g., replace a direct subscription with a "
            "domain event that a third orchestrator component handles). "
            "2. Introduce an aggregator/reducer component that consumes from both "
            "ends of the cycle and produces a single authoritative output topic. "
            "3. Apply the Dependency Inversion Principle: extract the shared "
            "abstraction that both sides depend on into a separate topic schema "
            "owned by neither. "
            "4. Add rate-limiting and loop-detection middleware on cyclic topics."
        ),
        references=["RMAV: Maintainability (BT, CC)", "Martin Fowler: Refactoring"],
    ),

    # ── HIGH ────────────────────────────────────────────────────────────────

    "GOD_COMPONENT": PatternSpec(
        id="GOD_COMPONENT",
        name="God Component",
        severity="HIGH",
        rmav_dimension="Maintainability",
        description=(
            "A component with both an anomalously high composite quality score "
            "Q(v) (above the box-plot upper fence Q3 + 1.5×IQR) AND a total degree "
            "(in + out) above the 75th percentile. It acts as a dependency magnet, "
            "accumulating responsibilities far beyond its architectural role."
        ),
        risk=(
            "God components are simultaneously the most likely to fail (high "
            "complexity), the most impactful when they do (many dependents), and the "
            "hardest to change (high coupling risk). They violate the Single "
            "Responsibility Principle at the architectural level."
        ),
        recommendation=(
            "1. Decompose using the Strangler Fig pattern: extract cohesive subsets "
            "of the component's publish/subscribe responsibilities into new, "
            "purpose-built application components. "
            "2. Identify topic ownership boundaries — each topic should have a clear "
            "single-publisher contract. If this component publishes to many "
            "semantically unrelated topics, split it. "
            "3. Apply domain-driven design bounded contexts: each bounded context "
            "gets its own set of topics and applications. "
            "4. Introduce a façade topic layer that decouples the internal complexity "
            "from external subscribers."
        ),
        references=["RMAV: Maintainability (BT, CouplingRisk)", "DDD Bounded Contexts"],
    ),

    "BOTTLENECK_EDGE": PatternSpec(
        id="BOTTLENECK_EDGE",
        name="Bottleneck Edge",
        severity="HIGH",
        rmav_dimension="Availability",
        description=(
            "A dependency edge (typically an app→broker or app→app DEPENDS_ON link) "
            "whose betweenness centrality exceeds Q3 + 1.5×IQR of the edge "
            "betweenness distribution. All or most message traffic between two "
            "architectural regions flows through this single connection."
        ),
        risk=(
            "Bottleneck edges create throughput ceilings and availability "
            "vulnerabilities. QoS degradation on a bottleneck edge — even briefly — "
            "propagates as a cascade delay to all downstream consumers. Network "
            "partitions isolating this edge partition the entire system."
        ),
        recommendation=(
            "1. Add parallel edge redundancy: introduce a second broker or relay "
            "path between the same endpoint pairs with load-balanced routing. "
            "2. Apply topic partitioning/sharding to distribute traffic across "
            "multiple parallel edges. "
            "3. Introduce a message bus abstraction layer between the two regions "
            "so that multiple brokers can serve the same logical topic. "
            "4. Monitor this edge with dedicated latency/throughput SLOs and "
            "add back-pressure signalling."
        ),
        references=["RMAV: Availability (BR, AP_c)", "SRE: Service Level Objectives"],
    ),

    "BROKER_OVERLOAD": PatternSpec(
        id="BROKER_OVERLOAD",
        name="Broker Saturation",
        severity="HIGH",
        rmav_dimension="Availability",
        description=(
            "A broker component whose betweenness centrality in the middleware layer "
            "is disproportionately high relative to other brokers (≥ 2× the median "
            "broker betweenness, or the sole broker serving a large cluster of "
            "applications). Equivalent to the 'Hub-and-Spoke' or 'God Broker' "
            "anti-pattern at the infrastructure level."
        ),
        risk=(
            "The overloaded broker becomes a single-threaded bottleneck for all "
            "message routing in its region. Resource exhaustion (CPU, memory, "
            "socket connections) on this broker stalls all producer and consumer "
            "applications that rely on it, regardless of their individual health."
        ),
        recommendation=(
            "1. Partition the broker's topic namespace across multiple broker "
            "instances using consistent hashing or range-based assignment. "
            "2. Deploy a broker cluster with horizontal scaling (e.g., Kafka "
            "multi-partition topics, ROS 2 domain segmentation, MQTT bridge "
            "cluster). "
            "3. Introduce a hierarchical broker topology: local edge brokers "
            "aggregate traffic before forwarding to a central broker. "
            "4. Set hard limits on maximum topics-per-broker and enforce them "
            "via deployment policy."
        ),
        references=["RMAV: Availability (QSPOF, BR)", "EIP: Message Broker Pattern"],
    ),

    "DEEP_PIPELINE": PatternSpec(
        id="DEEP_PIPELINE",
        name="Deep Processing Pipeline",
        severity="HIGH",
        rmav_dimension="Reliability",
        description=(
            "A directed chain of application components in the dependency graph "
            "with depth ≥ 5 hops (longest shortest path in the application-layer "
            "subgraph exceeds the system's 75th percentile path length). Each "
            "hop adds latency, a new failure point, and transformation complexity."
        ),
        risk=(
            "End-to-end latency grows linearly with pipeline depth. A single slow "
            "or failed stage stalls all downstream stages. Debugging requires "
            "tracing state through many intermediate transformations. This pattern "
            "is particularly dangerous in real-time systems (ROS 2, financial "
            "trading) with strict latency budgets."
        ),
        recommendation=(
            "1. Flatten the pipeline by merging adjacent transformation stages "
            "that share the same data ownership boundary. "
            "2. Introduce parallel fan-out branches where stages are logically "
            "independent, reducing the critical path length. "
            "3. Use a content-based router or message enricher at an earlier "
            "stage to pre-compute data needed downstream, eliminating intermediate "
            "request-response hops. "
            "4. Set latency SLOs per stage and enforce them with timeouts and "
            "fallback publishers."
        ),
        references=["RMAV: Reliability (RPR, CDPot)", "EIP: Pipes and Filters"],
    ),

    # ── MEDIUM ───────────────────────────────────────────────────────────────

    "TOPIC_FANOUT": PatternSpec(
        id="TOPIC_FANOUT",
        name="Topic Fan-Out Explosion",
        severity="MEDIUM",
        rmav_dimension="Reliability",
        description=(
            "A topic vertex with a subscriber count (out-degree in the topic "
            "projection) that exceeds Q3 + 1.5×IQR of the topic out-degree "
            "distribution. A single message published to this topic triggers "
            "delivery to an anomalously large number of consumers."
        ),
        risk=(
            "Any quality issue — late arrival, malformed message, schema change — "
            "on this topic has a system-wide blast radius. Broker memory and "
            "network bandwidth spikes on every publish. Subscriber lag on a "
            "high-fanout topic is hard to diagnose because it manifests in many "
            "unrelated application logs simultaneously."
        ),
        recommendation=(
            "1. Apply topic segmentation: split the overloaded topic into "
            "domain-specific sub-topics (e.g., 'sensor.raw' → 'sensor.vision', "
            "'sensor.lidar', 'sensor.imu'). Subscribers opt into only what "
            "they need. "
            "2. Introduce a topic aggregator pattern: a lightweight router "
            "application subscribes to the broad topic and republishes to "
            "specific sub-topics based on message content. "
            "3. Evaluate whether all N subscribers actually need every message, "
            "or whether a shared cache/state-store pattern is more appropriate. "
            "4. Apply QoS policies (e.g., BEST_EFFORT for non-critical fanout "
            "consumers) to reduce broker acknowledgement overhead."
        ),
        references=["RMAV: Reliability (DG_in, RPR)", "EIP: Publish-Subscribe Channel"],
    ),

    "CHATTY_PAIR": PatternSpec(
        id="CHATTY_PAIR",
        name="Chatty Pair (Tight Bidirectional Coupling)",
        severity="MEDIUM",
        rmav_dimension="Maintainability",
        description=(
            "Two application components with a bidirectional high-weight dependency "
            "relationship: each depends on the other through topics (A → topic_1 → B "
            "and B → topic_2 → A), with both edges carrying high QoS weights. "
            "Detected when the symmetric weight score (w_in × w_out) for both "
            "nodes exceeds the 75th percentile."
        ),
        risk=(
            "Chatty pairs create logical coupling that masquerades as decoupling. "
            "The two components cannot be independently deployed, scaled, or tested. "
            "They share an implicit shared-state contract through their topics. "
            "A change to one component's message schema always requires a "
            "coordinated change in the other."
        ),
        recommendation=(
            "1. Introduce a mediator component that owns the shared state and "
            "serves as the single authority both components query/update via "
            "separate one-directional topics. "
            "2. Replace the bidirectional dependency with an event-carried state "
            "transfer pattern: one component broadcasts its full state as events; "
            "the other reacts without needing to reply. "
            "3. Apply the Tell-Don't-Ask principle: ensure each component "
            "publishes decisions/actions rather than requesting information from "
            "the other."
        ),
        references=["RMAV: Maintainability (CouplingRisk, CC)", "EIP: Correlation Identifier"],
    ),

    "QOS_MISMATCH": PatternSpec(
        id="QOS_MISMATCH",
        name="QoS Policy Mismatch",
        severity="MEDIUM",
        rmav_dimension="Reliability",
        description=(
            "A DEPENDS_ON edge where the publisher's QoS weight is significantly "
            "lower than the subscriber's QoS weight (w_publisher < w_subscriber "
            "by a margin > 0.3). The publisher offers weaker guarantees (e.g., "
            "BEST_EFFORT) than the subscriber requires (e.g., RELIABLE + TRANSIENT_LOCAL)."
        ),
        risk=(
            "QoS mismatches silently degrade the effective reliability of "
            "communication. In ROS 2 and DDS systems, incompatible QoS policies "
            "result in the connection not being established at all — a runtime "
            "discovery failure with no compile-time warning. In MQTT, a publisher "
            "at QoS 0 cannot satisfy a subscriber expecting QoS 2 guarantees."
        ),
        recommendation=(
            "1. Align publisher and subscriber QoS policies during system design, "
            "not at runtime. Establish a QoS policy registry and enforce it via "
            "automated validation in the CI pipeline. "
            "2. If the publisher cannot upgrade its QoS (e.g., a hardware driver "
            "limited to BEST_EFFORT), introduce a QoS bridge component that "
            "buffers and re-publishes with elevated guarantees. "
            "3. For ROS 2: use rmw_implementation-agnostic QoS profiles "
            "(sensor_data, services_default) to avoid accidental mismatches. "
            "4. Add static analysis of topic QoS in your CI/CD build step."
        ),
        references=["RMAV: Reliability (w_in, w_out)", "ROS 2 QoS Policies"],
    ),

    "ORPHANED_TOPIC": PatternSpec(
        id="ORPHANED_TOPIC",
        name="Orphaned Topic",
        severity="MEDIUM",
        rmav_dimension="Maintainability",
        description=(
            "A topic node with no publishers (in-degree = 0 in the structural "
            "PUBLISHES_TO graph) OR no subscribers (out-degree = 0 in the "
            "SUBSCRIBES_TO graph). The topic exists in the system model but "
            "carries no functional communication."
        ),
        risk=(
            "Orphaned topics are dead architecture. Publisher-only orphans "
            "waste broker resources and indicate that intended consumers were "
            "never connected or were removed without cleaning up the topic. "
            "Subscriber-only orphans indicate broken expectations — components "
            "waiting for data that will never arrive, potentially blocking "
            "dependent processing pipelines indefinitely."
        ),
        recommendation=(
            "1. For publisher-only orphans: either connect the intended subscriber "
            "or delete the topic and its publisher-side code. "
            "2. For subscriber-only orphans: trace the missing publisher to a "
            "deployment gap (service not started, topic name mismatch, namespace "
            "misconfiguration) or remove the dead subscriber. "
            "3. Enforce a topic lifecycle policy: topics that have been idle "
            "(no messages) for more than N days are automatically flagged for "
            "review and removal. "
            "4. Add topic existence validation to your integration test suite."
        ),
        references=["RMAV: Maintainability (DG_in, DG_out)", "EIP: Dead Letter Channel"],
    ),

    "UNSTABLE_INTERFACE": PatternSpec(
        id="UNSTABLE_INTERFACE",
        name="Unstable Interface (Extreme Coupling Imbalance)",
        severity="MEDIUM",
        rmav_dimension="Maintainability",
        description=(
            "A component with a CouplingRisk score above 0.8 (near 1.0 = maximum "
            "instability) — indicating a severe imbalance between efferent coupling "
            "(how much it depends on others) and afferent coupling (how many depend "
            "on it). Very high instability (DG_out >> DG_in) means many outgoing "
            "dependencies with little incoming stability pressure."
        ),
        risk=(
            "Highly unstable components are change-prone: they depend on many "
            "others and therefore absorb every interface change from their "
            "dependencies. Because few others depend on them in return, there is "
            "little architectural pressure to stabilize their contract. In practice, "
            "these components are frequently modified and become a source of "
            "integration failures."
        ),
        recommendation=(
            "1. Apply Robert Martin's Stable Abstractions Principle: components "
            "with high afferent coupling should define stable interfaces (abstract "
            "topic schemas); highly unstable components should depend on those "
            "stable abstractions rather than concrete implementations. "
            "2. Introduce a schema registry (e.g., Confluent Schema Registry, "
            "ROS 2 message types versioning) to make the dependency explicit and "
            "manageable. "
            "3. Consider inverting the dependency: instead of the unstable component "
            "subscribing to N topics from stable publishers, have the stable "
            "publishers emit to a unified aggregation topic that the unstable "
            "component owns."
        ),
        references=["RMAV: Maintainability (CouplingRisk)", "Martin: Stable Dependencies Principle"],
    ),
}


# =============================================================================
# Detection Result Model
# =============================================================================

@dataclass
class DetectedSmell:
    """A single detected anti-pattern instance in a specific layer."""
    pattern_id: str
    pattern_name: str
    severity: str
    rmav_dimension: str
    layer: str
    component_ids: List[str]
    metric_evidence: Dict[str, Any]   # Raw metric values that triggered detection
    description: str
    risk: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SmellReport:
    """Full detection report across one or more layers."""
    generated_at: str
    layers_analyzed: List[str]
    total_components: int
    total_smells: int
    by_severity: Dict[str, int]
    by_pattern: Dict[str, int]
    by_layer: Dict[str, int]
    smells: List[DetectedSmell]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# =============================================================================
# Extended Smell Detector
# =============================================================================

class SmellDetector:
    """
    Detects all 12 catalog anti-patterns from graph analysis results.

    Uses QualityAnalysisResult (from AnalysisService) as its primary input.
    Structural layer data is accessed via the component score objects and
    graph summary stats already computed by the analysis pipeline.
    """

    def __init__(self, active_patterns: Optional[List[str]] = None) -> None:
        """
        Args:
            active_patterns: If given, only patterns in this list are checked.
                             Use None to run all patterns.
        """
        self._active = set(active_patterns) if active_patterns else set(CATALOG.keys())
        # Validate requested patterns
        unknown = self._active - set(CATALOG.keys())
        if unknown:
            raise ValueError(f"Unknown pattern IDs: {sorted(unknown)}. "
                             f"Valid IDs: {sorted(CATALOG.keys())}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, layer_result, layer: str) -> List[DetectedSmell]:
        """
        Run all active detectors against a LayerAnalysisResult.

        Args:
            layer_result: LayerAnalysisResult from AnalysisService.analyze_layer()
            layer:        Layer name string ("app", "infra", "mw", "system")

        Returns:
            Sorted list of DetectedSmell, CRITICAL first.
        """
        smells: List[DetectedSmell] = []
        components = layer_result.quality.components
        if not components:
            return smells

        # Pre-compute distribution statistics used by multiple detectors
        stats = self._compute_distribution_stats(components)

        detectors = {
            "SPOF":              self._detect_spof,
            "SYSTEMIC_RISK":     self._detect_systemic_risk,
            "CYCLIC_DEPENDENCY": self._detect_cyclic_dependency,
            "GOD_COMPONENT":     self._detect_god_component,
            "BOTTLENECK_EDGE":   self._detect_bottleneck_edge,
            "BROKER_OVERLOAD":   self._detect_broker_overload,
            "DEEP_PIPELINE":     self._detect_deep_pipeline,
            "TOPIC_FANOUT":      self._detect_topic_fanout,
            "CHATTY_PAIR":       self._detect_chatty_pair,
            "QOS_MISMATCH":      self._detect_qos_mismatch,
            "ORPHANED_TOPIC":    self._detect_orphaned_topic,
            "UNSTABLE_INTERFACE":self._detect_unstable_interface,
        }

        for pid, detector_fn in detectors.items():
            if pid not in self._active:
                continue
            try:
                new_smells = detector_fn(layer_result, layer, stats)
                smells.extend(new_smells)
            except Exception as exc:
                logger.warning("Detector %s raised an error: %s", pid, exc, exc_info=True)

        _severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        smells.sort(key=lambda s: (_severity_order.get(s.severity, 9), s.pattern_id))
        return smells

    # ------------------------------------------------------------------
    # Distribution Statistics (box-plot parameters)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_distribution_stats(components) -> Dict[str, Any]:
        """Compute Q1, Q3, IQR fence thresholds for key metric fields."""
        import statistics

        def _boxplot_fence(values: List[float]) -> Tuple[float, float, float, float]:
            """Returns (q1, median, q3, upper_fence)."""
            if not values:
                return 0.0, 0.0, 0.0, 1.0
            sv = sorted(values)
            n = len(sv)
            q1 = sv[n // 4]
            median = sv[n // 2]
            q3 = sv[min(3 * n // 4, n - 1)]
            iqr = q3 - q1
            return q1, median, q3, q3 + 1.5 * iqr

        overall_scores  = [c.scores.overall          for c in components]
        in_degrees      = [getattr(c, "in_degree",  getattr(c.scores, "reliability",  0)) for c in components]
        out_degrees     = [getattr(c, "out_degree", getattr(c.scores, "vulnerability", 0)) for c in components]
        total_degrees   = [a + b for a, b in zip(in_degrees, out_degrees)]
        coupling_risks  = [getattr(c.scores, "maintainability", 0) for c in components]
        availability    = [c.scores.availability for c in components]

        q1_q, med_q, q3_q, fence_q = _boxplot_fence(overall_scores)
        _, _, q3_deg, fence_deg     = _boxplot_fence(total_degrees)
        _, _, q3_avail, fence_avail = _boxplot_fence(availability)

        # Broker-specific: betweenness via availability proxy
        broker_comps = [c for c in components
                        if getattr(c, "type", "").lower() in ("broker", "middleware")]
        broker_availabilities = [c.scores.availability for c in broker_comps]
        median_broker_avail = (
            statistics.median(broker_availabilities) if broker_availabilities else 0.0
        )

        return {
            "q1_q":            q1_q,
            "q3_q":            q3_q,
            "fence_q":         fence_q,
            "q3_degree":       q3_deg,
            "fence_degree":    fence_deg,
            "q3_avail":        q3_avail,
            "fence_avail":     fence_avail,
            "median_broker_avail": median_broker_avail,
            "n":               len(components),
        }

    # ------------------------------------------------------------------
    # Individual Detectors
    # ------------------------------------------------------------------

    def _make_smell(self, pid: str, layer: str, component_ids: List[str],
                    evidence: Dict[str, Any]) -> DetectedSmell:
        spec = CATALOG[pid]
        return DetectedSmell(
            pattern_id=pid,
            pattern_name=spec.name,
            severity=spec.severity,
            rmav_dimension=spec.rmav_dimension,
            layer=layer,
            component_ids=component_ids,
            metric_evidence=evidence,
            description=spec.description,
            risk=spec.risk,
            recommendation=spec.recommendation,
        )

    # ── CRITICAL ──────────────────────────────────────────────────────

    def _detect_spof(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        smells = []
        for c in lr.quality.components:
            ap_score = getattr(c, "ap_score", 0.0)
            is_ap    = getattr(c, "is_articulation_point", False)
            # Prefer explicit flag; fall back to availability > fence
            if is_ap or ap_score > 0 or c.scores.availability > stats["fence_avail"]:
                smells.append(self._make_smell(
                    "SPOF", layer, [c.id],
                    {"ap_score": round(ap_score, 4),
                     "availability": round(c.scores.availability, 4),
                     "overall_q": round(c.scores.overall, 4)},
                ))
        return smells

    def _detect_systemic_risk(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect cliques of ≥3 CRITICAL components."""
        critical_ids = {
            c.id for c in lr.quality.components
            if c.levels.overall.value.upper() == "CRITICAL"
        }
        if len(critical_ids) < 3:
            return []
        # Build adjacency from edge results
        edges = getattr(lr.quality, "edges", [])
        adj: Dict[str, set] = {cid: set() for cid in critical_ids}
        for e in edges:
            if e.source in critical_ids and e.target in critical_ids:
                adj[e.source].add(e.target)
                adj[e.target].add(e.source)
        # Find cliques of size ≥ 3 (greedy — exact Bron-Kerbosch for full rigor)
        found_clusters: List[List[str]] = []
        visited = set()
        for cid in sorted(critical_ids):
            if cid in visited:
                continue
            cluster = {cid}
            for neighbor in adj.get(cid, []):
                if neighbor in critical_ids:
                    cluster.add(neighbor)
            if len(cluster) >= 3:
                key = frozenset(cluster)
                if key not in visited:
                    found_clusters.append(sorted(cluster))
                    visited.update(cluster)
        smells = []
        for cluster in found_clusters:
            smells.append(self._make_smell(
                "SYSTEMIC_RISK", layer, cluster,
                {"cluster_size": len(cluster),
                 "critical_count": len(critical_ids)},
            ))
        return smells

    def _detect_cyclic_dependency(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect SCCs with > 1 node in application layer."""
        try:
            import networkx as nx
        except ImportError:
            logger.debug("networkx not available; skipping CYCLIC_DEPENDENCY detection.")
            return []
        G = nx.DiGraph()
        edges = getattr(lr.quality, "edges", [])
        for e in edges:
            G.add_edge(e.source, e.target)
        smells = []
        for scc in nx.strongly_connected_components(G):
            if len(scc) >= 2:
                cids = sorted(scc)
                smells.append(self._make_smell(
                    "CYCLIC_DEPENDENCY", layer, cids,
                    {"scc_size": len(cids),
                     "members": cids},
                ))
        return smells

    # ── HIGH ──────────────────────────────────────────────────────────

    def _detect_god_component(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        smells = []
        fence_q   = stats["fence_q"]
        q3_degree = stats["q3_degree"]
        for c in lr.quality.components:
            if c.scores.overall <= fence_q:
                continue
            in_d  = getattr(c, "in_degree_raw",  0)
            out_d = getattr(c, "out_degree_raw", 0)
            total = in_d + out_d
            if total > q3_degree:
                smells.append(self._make_smell(
                    "GOD_COMPONENT", layer, [c.id],
                    {"overall_q": round(c.scores.overall, 4),
                     "fence_q": round(fence_q, 4),
                     "total_degree": total,
                     "q3_degree": round(q3_degree, 4),
                     "maintainability": round(c.scores.maintainability, 4)},
                ))
        return smells

    def _detect_bottleneck_edge(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect edges with betweenness above the box-plot fence."""
        smells = []
        edges = getattr(lr.quality, "edges", [])
        if not edges:
            return smells
        edge_scores = [e.scores.overall for e in edges]
        if not edge_scores:
            return smells
        import statistics as _s
        sv = sorted(edge_scores)
        n  = len(sv)
        q3 = sv[min(3 * n // 4, n - 1)]
        iqr = q3 - sv[n // 4]
        fence = q3 + 1.5 * iqr
        for e in edges:
            if e.scores.overall > fence:
                smells.append(self._make_smell(
                    "BOTTLENECK_EDGE", layer,
                    [e.source, e.target],
                    {"edge_score": round(e.scores.overall, 4),
                     "fence": round(fence, 4),
                     "source": e.source,
                     "target": e.target},
                ))
        return smells

    def _detect_broker_overload(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        smells = []
        broker_comps = [c for c in lr.quality.components
                        if getattr(c, "type", "").lower() in ("broker", "middleware")]
        if len(broker_comps) < 2:
            # Single broker is always overloaded by definition
            if len(broker_comps) == 1:
                b = broker_comps[0]
                total = len(lr.quality.components)
                smells.append(self._make_smell(
                    "BROKER_OVERLOAD", layer, [b.id],
                    {"broker_count": 1,
                     "total_components": total,
                     "availability": round(b.scores.availability, 4),
                     "note": "Sole broker in layer — hub-and-spoke topology"},
                ))
            return smells
        median_avail = stats["median_broker_avail"]
        for b in broker_comps:
            if median_avail > 0 and b.scores.availability >= 2.0 * median_avail:
                smells.append(self._make_smell(
                    "BROKER_OVERLOAD", layer, [b.id],
                    {"availability": round(b.scores.availability, 4),
                     "median_broker_availability": round(median_avail, 4),
                     "ratio": round(b.scores.availability / median_avail, 2)},
                ))
        return smells

    def _detect_deep_pipeline(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect excessive pipeline depth using graph diameter."""
        try:
            import networkx as nx
        except ImportError:
            return []
        G = nx.DiGraph()
        edges = getattr(lr.quality, "edges", [])
        for e in edges:
            G.add_edge(e.source, e.target)
        if not G.nodes:
            return []
        # Compute longest shortest path (eccentricity proxy via BFS)
        smells = []
        DEPTH_THRESHOLD = 5
        # Find all simple paths that are unusually long
        visited_chains = set()
        for node in G.nodes:
            if G.in_degree(node) == 0:          # start from source nodes
                for target in G.nodes:
                    if target == node:
                        continue
                    try:
                        paths = list(nx.all_simple_paths(G, node, target, cutoff=DEPTH_THRESHOLD + 2))
                        for path in paths:
                            if len(path) - 1 >= DEPTH_THRESHOLD:
                                key = (path[0], path[-1])
                                if key not in visited_chains:
                                    visited_chains.add(key)
                                    smells.append(self._make_smell(
                                        "DEEP_PIPELINE", layer,
                                        list(path),
                                        {"depth_hops": len(path) - 1,
                                         "threshold": DEPTH_THRESHOLD,
                                         "chain_start": path[0],
                                         "chain_end": path[-1]},
                                    ))
                    except Exception:
                        pass
        return smells

    # ── MEDIUM ────────────────────────────────────────────────────────

    def _detect_topic_fanout(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect topics with subscriber counts above the box-plot fence."""
        smells = []
        # In the quality result, topics appear as component type "topic"
        topic_comps = [c for c in lr.quality.components
                       if getattr(c, "type", "").lower() == "topic"]
        if not topic_comps:
            return smells
        out_degrees = [getattr(c, "out_degree_raw", 0) for c in topic_comps]
        if not out_degrees or max(out_degrees) == 0:
            return smells
        sv = sorted(out_degrees)
        n  = len(sv)
        q3 = sv[min(3 * n // 4, n - 1)]
        iqr = q3 - sv[n // 4]
        fence = q3 + 1.5 * iqr
        for tc in topic_comps:
            od = getattr(tc, "out_degree_raw", 0)
            if od > max(fence, 5):      # minimum threshold of 5 subscribers
                smells.append(self._make_smell(
                    "TOPIC_FANOUT", layer, [tc.id],
                    {"subscriber_count": od,
                     "fanout_fence": round(fence, 1),
                     "reliability": round(tc.scores.reliability, 4)},
                ))
        return smells

    def _detect_chatty_pair(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect bidirectional high-weight edges between the same pair."""
        edges = getattr(lr.quality, "edges", [])
        edge_map: Dict[str, float] = {}
        for e in edges:
            edge_map[f"{e.source}→{e.target}"] = e.scores.overall
        smells = []
        seen = set()
        for e in edges:
            reverse_key = f"{e.target}→{e.source}"
            pair_key    = frozenset([e.source, e.target])
            if reverse_key in edge_map and pair_key not in seen:
                seen.add(pair_key)
                fwd_score = e.scores.overall
                rev_score = edge_map[reverse_key]
                combined  = fwd_score * rev_score
                if combined > 0.25:     # both edges must be moderately weighted
                    smells.append(self._make_smell(
                        "CHATTY_PAIR", layer,
                        sorted([e.source, e.target]),
                        {"forward_edge_score": round(fwd_score, 4),
                         "reverse_edge_score": round(rev_score, 4),
                         "coupling_product": round(combined, 4)},
                    ))
        return smells

    def _detect_qos_mismatch(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """
        Detect edges where publisher QoS weight < subscriber QoS weight by >0.3.
        Uses the edge vulnerability (V) score as a proxy for QoS weight gap.
        """
        edges = getattr(lr.quality, "edges", [])
        smells = []
        # Build component weight index
        comp_weights = {c.id: c.scores.vulnerability for c in lr.quality.components}
        for e in edges:
            src_v = comp_weights.get(e.source, 0.5)
            tgt_v = comp_weights.get(e.target, 0.5)
            # Publisher (source) has lower QoS than subscriber (target)
            gap = tgt_v - src_v
            if gap > 0.30:
                smells.append(self._make_smell(
                    "QOS_MISMATCH", layer,
                    [e.source, e.target],
                    {"publisher": e.source,
                     "subscriber": e.target,
                     "publisher_qos_proxy": round(src_v, 4),
                     "subscriber_qos_proxy": round(tgt_v, 4),
                     "qos_gap": round(gap, 4)},
                ))
        return smells

    def _detect_orphaned_topic(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect topic nodes with zero in-degree OR zero out-degree."""
        topic_comps = [c for c in lr.quality.components
                       if getattr(c, "type", "").lower() == "topic"]
        smells = []
        for tc in topic_comps:
            in_d  = getattr(tc, "in_degree_raw",  0)
            out_d = getattr(tc, "out_degree_raw", 0)
            if in_d == 0:
                smells.append(self._make_smell(
                    "ORPHANED_TOPIC", layer, [tc.id],
                    {"in_degree": 0,
                     "out_degree": out_d,
                     "orphan_type": "no_publisher"},
                ))
            elif out_d == 0:
                smells.append(self._make_smell(
                    "ORPHANED_TOPIC", layer, [tc.id],
                    {"in_degree": in_d,
                     "out_degree": 0,
                     "orphan_type": "no_subscriber"},
                ))
        return smells

    def _detect_unstable_interface(self, lr, layer: str, stats: Dict) -> List[DetectedSmell]:
        """Detect components with CouplingRisk (via maintainability) > 0.8."""
        smells = []
        for c in lr.quality.components:
            coupling = c.scores.maintainability     # M(v) captures CouplingRisk
            if coupling > 0.80:
                smells.append(self._make_smell(
                    "UNSTABLE_INTERFACE", layer, [c.id],
                    {"coupling_risk_proxy": round(coupling, 4),
                     "threshold": 0.80,
                     "maintainability": round(c.scores.maintainability, 4)},
                ))
        return smells


# =============================================================================
# Console Display Helpers
# =============================================================================

class SmellConsoleDisplay:
    """Rich terminal rendering for smell detection results."""

    SEVERITY_COLORS = {
        "CRITICAL": "\033[91m",   # bright red
        "HIGH":     "\033[93m",   # yellow
        "MEDIUM":   "\033[94m",   # blue
    }
    RMAV_ICONS = {
        "Reliability":     "R",
        "Maintainability": "M",
        "Availability":    "A",
        "Vulnerability":   "V",
    }
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    GRAY   = "\033[90m"
    DIM    = "\033[2m"

    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{self.RESET}"

    def _severity_color(self, sev: str) -> str:
        return self.SEVERITY_COLORS.get(sev, "")

    def print_banner(self) -> None:
        line = "═" * 70
        print(f"\n{self._c(line, self.CYAN)}")
        print(self._c(
            "  Software-as-a-Graph  ·  Pub-Sub Anti-Pattern & Bad Smell Detector",
            self.CYAN + self.BOLD))
        print(f"{self._c(line, self.CYAN)}\n")

    def print_catalog(self) -> None:
        """Print the full anti-pattern catalog in a readable format."""
        self.print_banner()
        print(self._c("  ANTI-PATTERN CATALOG  (12 patterns across 3 severity tiers)\n",
                      self.BOLD))
        for tier, sev in [("CRITICAL", "CRITICAL"), ("HIGH", "HIGH"), ("MEDIUM", "MEDIUM")]:
            color = self._severity_color(sev)
            print(self._c(f"  ── {tier} ──────────────────────────────────────────────────────",
                          color))
            for pid, spec in CATALOG.items():
                if spec.severity != sev:
                    continue
                rmav_icon = self.RMAV_ICONS.get(spec.rmav_dimension, "?")
                print(f"\n  {self._c(f'[{pid}]', color + self.BOLD)}"
                      f"  {self._c(spec.name, self.BOLD)}"
                      f"  {self._c(f'[RMAV:{rmav_icon}]', self.GRAY)}")
                # Word-wrap description
                words = spec.description.split()
                line_buf, col = "  ", 2
                for w in words:
                    if col + len(w) + 1 > 72:
                        print(self._c(line_buf, self.GRAY))
                        line_buf, col = "  ", 2
                    line_buf += w + " "
                    col += len(w) + 1
                if line_buf.strip():
                    print(self._c(line_buf, self.GRAY))
                print(f"  {self._c('Recommendation:', self.BOLD)} "
                      f"{spec.recommendation[:120]}...")
            print()

    def print_report(self, report: SmellReport, severity_filter: Optional[List[str]] = None) -> None:
        """Print the full detection report to stdout."""
        self.print_banner()

        # ── Summary KPIs ──────────────────────────────────────────────────
        print(self._c("  SCAN SUMMARY", self.BOLD))
        print(f"  Layers analyzed:     {', '.join(report.layers_analyzed)}")
        print(f"  Components scanned:  {report.total_components}")
        print(f"  Total smells found:  {self._c(str(report.total_smells), self.BOLD)}")
        print()
        for sev in ("CRITICAL", "HIGH", "MEDIUM"):
            count = report.by_severity.get(sev, 0)
            color = self._severity_color(sev)
            bar = self._c("█" * count, color) if count else self._c("─", self.GRAY)
            print(f"  {self._c(f'{sev:<10}', color)}  {bar}  {self._c(str(count), color + self.BOLD)}")
        print()

        # ── Pattern breakdown ─────────────────────────────────────────────
        if report.by_pattern:
            print(self._c("  BY PATTERN", self.BOLD))
            for pid, count in sorted(report.by_pattern.items(),
                                     key=lambda x: -x[1]):
                spec = CATALOG.get(pid)
                sev_color = self._severity_color(spec.severity) if spec else ""
                label = f"{pid:<24}" if spec else f"{pid:<24}"
                print(f"  {self._c(label, sev_color)}  {count}")
            print()

        # ── Individual findings ───────────────────────────────────────────
        smells = report.smells
        if severity_filter:
            sf = [s.upper() for s in severity_filter]
            smells = [s for s in smells if s.severity in sf]

        if not smells:
            print(self._c("  ✓  No smells found matching the active filters.\n", self.GREEN))
            return

        print(self._c(f"  FINDINGS ({len(smells)})", self.BOLD))
        print()

        prev_sev = None
        for i, smell in enumerate(smells, 1):
            if smell.severity != prev_sev:
                color = self._severity_color(smell.severity)
                print(self._c(
                    f"  {'─' * 3} {smell.severity} {'─' * (62 - len(smell.severity))}",
                    color))
                print()
                prev_sev = smell.severity

            color  = self._severity_color(smell.severity)
            rmav_i = self.RMAV_ICONS.get(smell.rmav_dimension, "?")

            print(f"  {self._c(f'#{i:02d}', self.BOLD)}  "
                  f"{self._c(f'[{smell.pattern_id}]', color + self.BOLD)}"
                  f"  {self._c(smell.pattern_name, self.BOLD)}"
                  f"  {self._c(f'[RMAV:{rmav_i}]', self.GRAY)}"
                  f"  {self._c(f'layer={smell.layer}', self.DIM)}")

            comps_display = ", ".join(smell.component_ids[:5])
            if len(smell.component_ids) > 5:
                comps_display += f" … (+{len(smell.component_ids) - 5} more)"
            print(f"       {self._c('Components:', self.BOLD)} {comps_display}")

            # Evidence
            ev_parts = [f"{k}={v}" for k, v in list(smell.metric_evidence.items())[:4]]
            print(f"       {self._c('Evidence:  ', self.BOLD)} {self._c(', '.join(ev_parts), self.GRAY)}")

            # Risk (first 140 chars)
            risk_short = smell.risk[:140].rstrip() + ("…" if len(smell.risk) > 140 else "")
            print(f"       {self._c('Risk:      ', self.BOLD)} {risk_short}")

            # Recommendation (first numbered point only)
            rec_first = smell.recommendation.split("2.")[0].strip()
            print(f"       {self._c('Fix:       ', self.BOLD)} {rec_first}")
            print()

    def print_success(self, msg: str) -> None:
        print(f"  {self._c('✓', self.GREEN)} {msg}")

    def print_error(self, msg: str) -> None:
        red = "\033[91m"
        print(f"  {self._c('✗', red)} {msg}")


# =============================================================================
# CLI — Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect_antipatterns",
        description="Pub-Sub Anti-Pattern & Bad Smell Detector — graph topology analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --layer system                        Scan complete system layer
  %(prog)s --layer app --severity critical,high  Only CRITICAL and HIGH smells
  %(prog)s --all                                 Scan all four layers
  %(prog)s --all --pattern spof,broker_overload  Specific patterns only
  %(prog)s --catalog                             Print full catalog (no Neo4j needed)
  %(prog)s --layer system --output smells.json   Export findings to JSON
  %(prog)s --layer system --use-ahp              Use AHP-derived RMAV weights

pattern IDs (case-insensitive):
  spof, systemic_risk, cyclic_dependency,
  god_component, bottleneck_edge, broker_overload, deep_pipeline,
  topic_fanout, chatty_pair, qos_mismatch, orphaned_topic, unstable_interface
""",
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw", "system"],
        default="system",
        help="Analysis layer to scan (default: system)",
    )
    mode.add_argument(
        "--all", "-a",
        action="store_true",
        help="Scan all four layers (app, infra, mw, system)",
    )
    mode.add_argument(
        "--catalog",
        action="store_true",
        help="Print full anti-pattern catalog and exit (no Neo4j required)",
    )

    # ── Neo4j connection ──────────────────────────────────────────────────
    neo4j = parser.add_argument_group("Neo4j connection")
    neo4j.add_argument("--uri",      default="bolt://localhost:7687", help="Neo4j Bolt URI")
    neo4j.add_argument("--user",     default="neo4j",                help="Neo4j username")
    neo4j.add_argument("--password", default="password",             help="Neo4j password")

    # ── Detection options ─────────────────────────────────────────────────
    detection = parser.add_argument_group("Detection options")
    detection.add_argument(
        "--pattern", "-P",
        metavar="ID[,ID…]",
        help="Comma-separated list of pattern IDs to run (default: all)",
    )
    detection.add_argument(
        "--severity", "-S",
        metavar="LEVEL[,LEVEL…]",
        help="Filter output to these severity levels, e.g. critical,high",
    )
    detection.add_argument(
        "--use-ahp",
        action="store_true",
        help="Use AHP-derived RMAV weights instead of default equal weights",
    )

    # ── Output ────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument("--output", "-o", metavar="FILE",  help="Export findings to JSON file")
    out.add_argument("--json",         action="store_true", help="Print JSON to stdout")
    out.add_argument("--quiet",  "-q", action="store_true", help="Suppress human-readable output")
    out.add_argument("--verbose","-v", action="store_true", help="Enable debug logging")

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser  = build_parser()
    args    = parser.parse_args()
    display = SmellConsoleDisplay()

    # ── Logging ───────────────────────────────────────────────────────────
    log_level = (
        logging.DEBUG   if args.verbose else
        logging.WARNING if args.quiet   else
        logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Catalog mode (no Neo4j needed) ───────────────────────────────────
    if args.catalog:
        display.print_catalog()
        return 0

    # ── Parse active patterns ─────────────────────────────────────────────
    active_patterns: Optional[List[str]] = None
    if args.pattern:
        active_patterns = [p.strip().upper() for p in args.pattern.split(",") if p.strip()]
        unknown = set(active_patterns) - set(CATALOG.keys())
        if unknown:
            display.print_error(f"Unknown pattern IDs: {sorted(unknown)}")
            display.print_error(f"Valid IDs: {sorted(CATALOG.keys())}")
            return 1

    # ── Parse severity filter ─────────────────────────────────────────────
    severity_filter: Optional[List[str]] = None
    if args.severity:
        severity_filter = [s.strip().upper() for s in args.severity.split(",") if s.strip()]
        valid_sevs = {"CRITICAL", "HIGH", "MEDIUM"}
        bad_sevs = set(severity_filter) - valid_sevs
        if bad_sevs:
            display.print_error(f"Unknown severity levels: {sorted(bad_sevs)}. "
                                f"Use: critical, high, medium")
            return 1

    # ── Layers to scan ───────────────────────────────────────────────────
    layers_to_scan = (
        ["app", "infra", "mw", "system"] if args.all else [args.layer]
    )

    # ── Connect to Neo4j ─────────────────────────────────────────────────
    try:
        repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    except Exception as exc:
        display.print_error(f"Cannot connect to Neo4j at {args.uri}: {exc}")
        display.print_error("Ensure Neo4j is running and data has been imported.")
        return 1

    try:
        analyzer = AnalysisService(repo, use_ahp=args.use_ahp)
        detector = SmellDetector(active_patterns=active_patterns)

        all_smells: List[DetectedSmell] = []
        total_components = 0

        for layer in layers_to_scan:
            logger.info("Analyzing layer: %s", layer)
            try:
                layer_result = analyzer.analyze_layer(layer)
            except Exception as exc:
                logger.warning("Analysis failed for layer %s: %s", layer, exc)
                if args.verbose:
                    logger.exception("Layer analysis error")
                continue

            total_components += len(layer_result.quality.components)
            layer_smells = detector.detect(layer_result, layer)
            all_smells.extend(layer_smells)
            logger.info("Layer %s: %d components, %d smells detected",
                        layer, len(layer_result.quality.components), len(layer_smells))

        # ── Build report ─────────────────────────────────────────────────
        by_severity: Dict[str, int] = {}
        by_pattern:  Dict[str, int] = {}
        by_layer:    Dict[str, int] = {}
        for s in all_smells:
            by_severity[s.severity]   = by_severity.get(s.severity, 0) + 1
            by_pattern[s.pattern_id]  = by_pattern.get(s.pattern_id, 0) + 1
            by_layer[s.layer]         = by_layer.get(s.layer, 0) + 1

        report = SmellReport(
            generated_at=datetime.utcnow().isoformat() + "Z",
            layers_analyzed=layers_to_scan,
            total_components=total_components,
            total_smells=len(all_smells),
            by_severity=by_severity,
            by_pattern=by_pattern,
            by_layer=by_layer,
            smells=all_smells,
        )

        # ── Human-readable output ─────────────────────────────────────────
        if not args.quiet:
            display.print_report(report, severity_filter=severity_filter)

        # ── JSON stdout ───────────────────────────────────────────────────
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))

        # ── File export ───────────────────────────────────────────────────
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(report.to_dict(), fh, indent=2, default=str)
            if not args.quiet:
                display.print_success(f"Report saved → {args.output}")

        # Exit code: 0 if clean, 2 if any CRITICAL found
        if by_severity.get("CRITICAL", 0) > 0:
            return 2
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        display.print_error(f"Unexpected error: {exc}")
        if args.verbose:
            logger.exception("Fatal error")
        return 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
