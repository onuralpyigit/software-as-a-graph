"""
saag/analysis/antipattern_detector.py — Unified Architectural Anti-Pattern Engine
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from saag.prediction.models import DetectedProblem

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
    "BOTTLENECK_EDGE": PatternSpec(
        id="BOTTLENECK_EDGE",
        name="Bottleneck Dependency",
        severity="MEDIUM",
        category="Maintainability",
        description="An edge carrying an anomalously high percentage of shortest paths.",
        risk="Potential performance bottleneck and shared failure point.",
        recommendation="Consider caching, load balancing, or async patterns."
    ),

    # ── Security ─────────────────────────────────────────────────────────────

    "TARGET": PatternSpec(
        id="TARGET",
        name="High Value Target",
        severity="CRITICAL",
        category="Security",
        description="Highly connected component with critical vulnerability classification.",
        risk="A breach here provides an attacker with high reachability into the system.",
        recommendation="Zero Trust policies, audit logging, and network isolation."
    ),
    "EXPOSURE": PatternSpec(
        id="EXPOSURE",
        name="High Exposure Surface",
        severity="HIGH",
        category="Security",
        description="Easily reachable component (high closeness) with a large attack surface.",
        risk="Easier target for initial penetration or lateral movement.",
        recommendation="Restrict incoming connections and validate all inputs via API gateways."
    ),

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


# =============================================================================
# Detector Engine
# =============================================================================

class AntiPatternDetector:
    """
    Unified engine for detecting architectural anti-patterns and bad smells.
    Consolidates SmellDetector and ProblemDetector catalogs and logic.
    """

    def __init__(self, active_patterns: Optional[List[str]] = None) -> None:
        self._active = set(active_patterns) if active_patterns else set(CATALOG.keys())
        unknown = self._active - set(CATALOG.keys())
        if unknown:
            raise ValueError(f"Unknown patterns: {unknown}")

    def detect(self, layer_result: Any, layer: str) -> List[DetectedProblem]:
        """Run all active detectors against a LayerAnalysisResult."""
        problems: List[DetectedProblem] = []
        components = layer_result.components if layer_result else []
        if not components:
            return problems

        stats = self._compute_distribution_stats(components)

        detectors = {
            "SPOF":               self._detect_spof,
            "BRIDGE_EDGE":        self._detect_bridge_edge,
            "FAILURE_HUB":        self._detect_failure_hub,
            "CONCENTRATION_RISK": self._detect_concentration_risk,
            "GOD_COMPONENT":      self._detect_god_component,
            "HUB_AND_SPOKE":      self._detect_hub_and_spoke,
            "BOTTLENECK_EDGE":    self._detect_bottleneck_edge,
            "TARGET":             self._detect_target,
            "EXPOSURE":           self._detect_exposure,
            "CYCLE":              self._detect_cycle,
            "CHAIN":              self._detect_chain,
            "ISOLATED":           self._detect_isolated,
            "SYSTEMIC_RISK":      self._detect_systemic_risk,
        }

        for pid, detector_fn in detectors.items():
            if pid not in self._active:
                continue
            try:
                found = detector_fn(layer_result, layer, stats)
                problems.extend(found)
            except Exception as exc:
                logger.warning("Detector %s failed: %s", pid, exc, exc_info=True)

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

    def _make_problem(self, pid: str, entity_id: str, evidence: Dict[str, Any], 
                       entity_type: str = "Component") -> DetectedProblem:
        spec = CATALOG[pid]
        return DetectedProblem(
            entity_id=entity_id, entity_type=entity_type, 
            category=spec.category, severity=spec.severity,
            name=spec.name, description=spec.description,
            recommendation=spec.recommendation, evidence=evidence
        )

    @staticmethod
    def _compute_distribution_stats(components) -> Dict[str, Any]:
        """Compute box-plot fences for metrics."""
        def _boxplot_fence(values: List[float]) -> Tuple[float, float, float, float]:
            if not values: return 0.0, 0.0, 0.0, 1.0
            sv = sorted(values)
            n = len(sv)
            q1, median, q3 = sv[n // 4], sv[n // 2], sv[min(3 * n // 4, n - 1)]
            return q1, median, q3, q3 + 1.5 * (q3 - q1)

        overall_scores = [c.scores.overall for c in components]
        avail_scores = [c.scores.availability for c in components]
        rel_scores = [c.scores.reliability for c in components]
        vuln_scores = [c.scores.vulnerability for c in components]
        total_degrees = [getattr(c.structural, 'in_degree_raw', 0) + getattr(c.structural, 'out_degree_raw', 0) for c in components]
        out_degrees = [getattr(c.structural, 'out_degree_raw', 0) for c in components]

        _, _, q3_q, fence_q = _boxplot_fence(overall_scores)
        _, _, q3_deg, fence_deg = _boxplot_fence(total_degrees)
        _, _, q3_rel, fence_rel = _boxplot_fence(rel_scores)
        _, _, q3_avail, fence_avail = _boxplot_fence(avail_scores)
        _, _, q3_vuln, fence_vuln = _boxplot_fence(vuln_scores)
        
        median_out = statistics.median(out_degrees) if out_degrees else 0

        return {
            "fence_q": fence_q, "q3_degree": q3_deg, "fence_degree": fence_deg,
            "fence_rel": fence_rel, "fence_avail": fence_avail, "fence_vuln": fence_vuln,
            "median_out": median_out, "total_count": len(components)
        }

    # ── Detectors ────────────────────────────────────────────────────

    def _detect_spof(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for c in lr.components:
            # Issue #10: Use directed AP if available
            is_spof = getattr(c.structural, 'is_directed_ap', False)
            if is_spof:
                out.append(self._make_problem("SPOF", c.id, {"availability_level": c.levels.availability.value, "is_directed": True}))
            elif getattr(c.structural, 'is_articulation_point', False):
                 out.append(self._make_problem("SPOF", c.id, {"availability_level": c.levels.availability.value, "is_directed": False}))
        return out

    def _detect_bridge_edge(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for e in getattr(lr, "edges", []):
            if getattr(e.structural, 'is_bridge', False):
                out.append(self._make_problem("BRIDGE_EDGE", e.id, {"is_bridge": True}, entity_type="Edge"))
        return out

    def _detect_failure_hub(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        # Issue #12: Failure Hub should be based on Reliability R(v) and out-degree
        rel_fence = stats.get("fence_rel", 0.8)
        median_out = stats.get("median_out", 1)
        
        for c in lr.components:
            if c.scores.reliability > rel_fence and c.structural.out_degree_raw > median_out:
                out.append(self._make_problem("FAILURE_HUB", c.id, {
                    "reliability": c.scores.reliability, 
                    "out_degree": c.structural.out_degree_raw
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
        for c in lr.components:
            betweenness = getattr(c.structural, 'betweenness', 0)
            if betweenness > 0.3 and c.levels.maintainability >= CriticalityLevel.CRITICAL:
                out.append(self._make_problem("GOD_COMPONENT", c.id, {"betweenness": betweenness}))
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
        for e in getattr(lr, "edges", []):
            betweenness = getattr(e.structural, 'betweenness', 0)
            if betweenness > 0.2:
                out.append(self._make_problem("BOTTLENECK_EDGE", e.id, {"betweenness": betweenness}, entity_type="Edge"))
        return out

    def _detect_target(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        from saag.core.criticality import CriticalityLevel
        for c in lr.components:
            if c.levels.vulnerability >= CriticalityLevel.CRITICAL:
                out.append(self._make_problem("TARGET", c.id, {"vulnerability": c.scores.vulnerability}))
        return out

    def _detect_exposure(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        from saag.core.criticality import CriticalityLevel
        for c in lr.components:
            closeness = getattr(c.structural, 'closeness', 0)
            if c.levels.vulnerability == CriticalityLevel.HIGH and closeness > 0.6:
                out.append(self._make_problem("EXPOSURE", c.id, {"closeness": closeness}))
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
                out.append(self._make_problem("CHAIN", f"CHAIN-{''.join(list(component)[:3])}", {"length": len(component)}, entity_type="Architecture"))
        return out

    def _detect_isolated(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        out = []
        for c in lr.components:
            if getattr(c.structural, 'is_isolated', False):
                out.append(self._make_problem("ISOLATED", c.id, {"isolated": True}))
        return out

    def _detect_systemic_risk(self, lr, layer: str, stats: Dict) -> List[DetectedProblem]:
        from saag.core.criticality import CriticalityLevel
        crit_count = sum(1 for c in lr.components if c.levels.overall == CriticalityLevel.CRITICAL)
        total = stats["total_count"]
        if total > 0 and (crit_count / total) > 0.2:
            return [self._make_problem("SYSTEMIC_RISK", "SYSTEM", {"critical_ratio": crit_count/total}, entity_type="System")]
        return []

    def _detect_compound_risk(self, problems: List[DetectedProblem]) -> List[DetectedProblem]:
        """Issue #13: Post-pass to identify nodes with multiple critical problems."""
        by_entity: Dict[str, List[str]] = {}
        for p in problems:
            if p.entity_type == "Component":
                by_entity.setdefault(p.entity_id, []).append(p.name)
        
        out = []
        for eid, names in by_entity.items():
            is_spof = any("SPOF" in n for n in names)
            is_god = any("God" in n for n in names) or any("Hub" in n for n in names)
            
            if is_spof and is_god:
                out.append(self._make_problem(
                    "COMPOUND_RISK", 
                    eid, 
                    {"risks": names}, 
                    entity_type="Component"
                ))
        return out
