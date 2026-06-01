"""
Combined anomaly scoring module.

Implements the composite scoring system defined in the paper:

1. Pattern-based anomaly score OS^P(x):
   For each pattern p, weight = 1 / |{x' : p(x')}|
   OS^P(x) = Σ_p  (1 / count_matching) · I[p(x)]

2. Single-dimension contribution UNI(x):
   u_M(x) ∈ [0,1]  — upper-tail extremity for metric M
   c_M(x) = min(u_M(x), τ)
   UNI(x) = Σ_{M ∈ M_x} c_M(x)

3. Final score:
   Score(x) = OS^P(x) + λ · UNI(x)
"""

from typing import Dict, List, Any, Set
from dataclasses import dataclass, field

from .metrics import (
    AllMetrics,
    ApplicationMetrics,
    TopicMetrics,
    NodeMetrics,
    LibraryMetrics,
)
from .patterns import PatternResults, QuartileBounds


# Default parameters
DEFAULT_TAU = 0.3       # Cap for single-dimension contribution
DEFAULT_LAMBDA = 0.1    # Weight for single-dimension term


@dataclass
class ComponentScore:
    """Anomaly score for a single component."""
    id: str
    name: str
    pattern_score: float = 0.0          # OS^P(x)
    uni_score: float = 0.0              # UNI(x)
    total_score: float = 0.0            # Score(x)
    matched_patterns: List[str] = field(default_factory=list)
    metric_extremities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pattern_score": round(self.pattern_score, 6),
            "uni_score": round(self.uni_score, 6),
            "total_score": round(self.total_score, 6),
            "matched_patterns": self.matched_patterns,
            "metric_extremities": {
                k: round(v, 4) for k, v in self.metric_extremities.items()
            },
        }


@dataclass
class ScoringResults:
    """Combined scoring results for all component types."""
    parameters: Dict[str, float] = field(default_factory=dict)
    applications: List[ComponentScore] = field(default_factory=list)
    topics: List[ComponentScore] = field(default_factory=list)
    nodes: List[ComponentScore] = field(default_factory=list)
    libraries: List[ComponentScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters,
            "applications": [s.to_dict() for s in self.applications],
            "topics": [s.to_dict() for s in self.topics],
            "nodes": [s.to_dict() for s in self.nodes],
            "libraries": [s.to_dict() for s in self.libraries],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_scores(
    metrics: AllMetrics,
    patterns: PatternResults,
    tau: float = DEFAULT_TAU,
    lam: float = DEFAULT_LAMBDA,
) -> ScoringResults:
    """
    Calculate combined anomaly scores for all components.

    Args:
        metrics: AllMetrics with calculated structural metrics.
        patterns: PatternResults with detected patterns and quartiles.
        tau: Cap for single-dimension contribution (default 0.3).
        lam: Weight for single-dimension term (default 0.1).

    Returns:
        ScoringResults with ranked scores per component type.
    """
    results = ScoringResults(
        parameters={"tau": tau, "lambda": lam}
    )

    # ---- Application scores ----
    results.applications = _score_apps(metrics, patterns, tau, lam)

    # ---- Topic scores ----
    results.topics = _score_topics(metrics, patterns, tau, lam)

    # ---- Node scores ----
    results.nodes = _score_nodes(metrics, patterns, tau, lam)

    # ---- Library scores ----
    results.libraries = _score_libs(metrics, patterns, tau, lam)

    return results


# ---------------------------------------------------------------------------
# Upper-tail extremity
# ---------------------------------------------------------------------------

def _upper_tail_extremity(value: float, bounds: QuartileBounds) -> float:
    """
    Compute upper-tail extremity u_M(x) ∈ [0, 1].

    u_M(x) = (M(x) - Q3) / (max - Q3)  if M(x) > Q3 and max > Q3
            = 0                           otherwise
    """
    if bounds.max_val <= bounds.q3:
        return 0.0
    if value <= bounds.q3:
        return 0.0
    return min((value - bounds.q3) / (bounds.max_val - bounds.q3), 1.0)


# ---------------------------------------------------------------------------
# Pattern-based score calculation
# ---------------------------------------------------------------------------

def _pattern_score_for_component(
    component_id: str,
    pattern_dict: Dict[str, list],
) -> tuple:
    """
    Calculate OS^P for a single component.

    Returns:
        (score, list_of_matched_pattern_names)
    """
    score = 0.0
    matched = []

    for pattern_name, matches in pattern_dict.items():
        match_count = len(matches)
        if match_count == 0:
            continue

        # Check if this component is in the match list
        is_match = any(m.id == component_id for m in matches)
        if is_match:
            score += 1.0 / match_count
            matched.append(pattern_name)

    return score, matched


# ---------------------------------------------------------------------------
# Application scoring
# ---------------------------------------------------------------------------

def _score_apps(
    metrics: AllMetrics,
    patterns: PatternResults,
    tau: float,
    lam: float,
) -> List[ComponentScore]:
    # Metric names for apps
    app_metric_keys = ["R", "AMP", "RA", "TC", "LE"]

    scores = []
    for app in metrics.applications:
        cs = ComponentScore(id=app.id, name=app.name)

        # Pattern-based score
        cs.pattern_score, cs.matched_patterns = _pattern_score_for_component(
            app.id, patterns.app_patterns
        )

        # Single-dimension contribution
        metric_values = {
            "R": float(app.reach),
            "AMP": app.amplification,
            "RA": app.role_asymmetry,
            "TC": float(app.topic_context_diversity),
            "LE": float(app.library_exposure),
        }
        uni = 0.0
        for mk in app_metric_keys:
            bounds = patterns.app_quartiles.get(mk)
            if bounds:
                u = _upper_tail_extremity(metric_values[mk], bounds)
                c = min(u, tau)
                cs.metric_extremities[mk] = u
                uni += c
        cs.uni_score = uni

        # Combined score
        cs.total_score = cs.pattern_score + lam * cs.uni_score

        scores.append(cs)

    # Sort by total_score descending
    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Topic scoring
# ---------------------------------------------------------------------------

def _score_topics(
    metrics: AllMetrics,
    patterns: PatternResults,
    tau: float,
    lam: float,
) -> List[ComponentScore]:
    topic_metric_keys = ["C", "I", "PS", "LCR"]

    scores = []
    for topic in metrics.topics:
        cs = ComponentScore(id=topic.id, name=topic.name)

        cs.pattern_score, cs.matched_patterns = _pattern_score_for_component(
            topic.id, patterns.topic_patterns
        )

        metric_values = {
            "C": float(topic.coverage),
            "I": topic.imbalance,
            "PS": float(topic.physical_spread),
            "LCR": topic.low_connectivity_ratio,
        }
        uni = 0.0
        for mk in topic_metric_keys:
            bounds = patterns.topic_quartiles.get(mk)
            if bounds:
                u = _upper_tail_extremity(metric_values[mk], bounds)
                c = min(u, tau)
                cs.metric_extremities[mk] = u
                uni += c
        cs.uni_score = uni

        cs.total_score = cs.pattern_score + lam * cs.uni_score

        scores.append(cs)

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Node scoring
# ---------------------------------------------------------------------------

def _score_nodes(
    metrics: AllMetrics,
    patterns: PatternResults,
    tau: float,
    lam: float,
) -> List[ComponentScore]:
    node_metric_keys = ["ND", "NID"]

    scores = []
    for node in metrics.nodes:
        cs = ComponentScore(id=node.id, name=node.name)

        cs.pattern_score, cs.matched_patterns = _pattern_score_for_component(
            node.id, patterns.node_patterns
        )

        metric_values = {
            "ND": float(node.density),
            "NID": float(node.interaction_density),
        }
        uni = 0.0
        for mk in node_metric_keys:
            bounds = patterns.node_quartiles.get(mk)
            if bounds:
                u = _upper_tail_extremity(metric_values[mk], bounds)
                c = min(u, tau)
                cs.metric_extremities[mk] = u
                uni += c
        cs.uni_score = uni

        cs.total_score = cs.pattern_score + lam * cs.uni_score

        scores.append(cs)

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Library scoring
# ---------------------------------------------------------------------------

def _score_libs(
    metrics: AllMetrics,
    patterns: PatternResults,
    tau: float,
    lam: float,
) -> List[ComponentScore]:
    lib_metric_keys = ["LC", "LCon"]

    scores = []
    for lib in metrics.libraries:
        cs = ComponentScore(id=lib.id, name=lib.name)

        cs.pattern_score, cs.matched_patterns = _pattern_score_for_component(
            lib.id, patterns.lib_patterns
        )

        metric_values = {
            "LC": float(lib.coverage),
            "LCon": float(lib.concentration),
        }
        uni = 0.0
        for mk in lib_metric_keys:
            bounds = patterns.lib_quartiles.get(mk)
            if bounds:
                u = _upper_tail_extremity(metric_values[mk], bounds)
                c = min(u, tau)
                cs.metric_extremities[mk] = u
                uni += c
        cs.uni_score = uni

        cs.total_score = cs.pattern_score + lam * cs.uni_score

        scores.append(cs)

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores
