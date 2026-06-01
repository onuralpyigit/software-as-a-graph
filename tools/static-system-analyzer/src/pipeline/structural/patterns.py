"""
Structural anomaly pattern detection module.

Detects structural patterns based on relative quartile interpretation
as defined in the paper.

Relative interpretation:
    M(x)↑  ⟺  M(x) ≥ Q3(M)         (relatively high)
    M(x)↓  ⟺  M(x) ≤ Q1(M)         (relatively low)
    When Q1 = Q3: only absolute extremes (min/max) are evaluated.

Application-level patterns:
    WR  - Wide Reach:                    R↑ ∧ AMP↑
    RS  - Role Skew:                     RA↑ ∨ RA↓
    CS  - Context Spread:                TC↑
    SD  - Shared Dependency Exposure:    LE↑

Topic-level patterns:
    CB  - Communication Backbone:        C↑ ∧ I↓
    DC  - Directional Concentration:     I↑
    PA  - Peripheral Aggregator:         LCR↑

Node-level patterns:
    IH  - Interaction Hotspot:           ND↑ ∧ NID↑

Library-level patterns:
    WUL - Widely Used Library:           LC↑
    CL  - Concentrated Library:          LCon↑
"""

from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field

from .metrics import (
    AllMetrics,
    ApplicationMetrics,
    TopicMetrics,
    NodeMetrics,
    LibraryMetrics,
)


@dataclass
class QuartileBounds:
    """Quartile boundaries for a metric."""
    q1: float
    q3: float
    min_val: float
    max_val: float
    degenerate: bool = False  # True when Q1 == Q3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q1": round(self.q1, 4),
            "q3": round(self.q3, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
            "degenerate": self.degenerate,
        }


@dataclass
class PatternMatch:
    """A single pattern match for a component."""
    id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name}


@dataclass
class PatternResults:
    """All pattern detection results."""
    # Quartile bounds per metric
    app_quartiles: Dict[str, QuartileBounds] = field(default_factory=dict)
    topic_quartiles: Dict[str, QuartileBounds] = field(default_factory=dict)
    node_quartiles: Dict[str, QuartileBounds] = field(default_factory=dict)
    lib_quartiles: Dict[str, QuartileBounds] = field(default_factory=dict)

    # Pattern matches
    app_patterns: Dict[str, List[PatternMatch]] = field(default_factory=dict)
    topic_patterns: Dict[str, List[PatternMatch]] = field(default_factory=dict)
    node_patterns: Dict[str, List[PatternMatch]] = field(default_factory=dict)
    lib_patterns: Dict[str, List[PatternMatch]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quartiles": {
                "applications": {k: v.to_dict() for k, v in self.app_quartiles.items()},
                "topics": {k: v.to_dict() for k, v in self.topic_quartiles.items()},
                "nodes": {k: v.to_dict() for k, v in self.node_quartiles.items()},
                "libraries": {k: v.to_dict() for k, v in self.lib_quartiles.items()},
            },
            "patterns": {
                "applications": {k: [m.to_dict() for m in v] for k, v in self.app_patterns.items()},
                "topics": {k: [m.to_dict() for m in v] for k, v in self.topic_patterns.items()},
                "nodes": {k: [m.to_dict() for m in v] for k, v in self.node_patterns.items()},
                "libraries": {k: [m.to_dict() for m in v] for k, v in self.lib_patterns.items()},
            },
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_patterns(metrics: AllMetrics) -> PatternResults:
    """
    Detect all structural anomaly patterns from computed metrics.

    Args:
        metrics: AllMetrics with calculated structural metrics.

    Returns:
        PatternResults with quartile bounds and pattern matches.
    """
    results = PatternResults()

    # ---- Application patterns ----
    app_values = _extract_app_metric_values(metrics.applications)
    for metric_name, values in app_values.items():
        results.app_quartiles[metric_name] = _compute_quartile_bounds(values)

    results.app_patterns = _detect_app_patterns(
        metrics.applications, results.app_quartiles
    )

    # ---- Topic patterns ----
    topic_values = _extract_topic_metric_values(metrics.topics)
    for metric_name, values in topic_values.items():
        results.topic_quartiles[metric_name] = _compute_quartile_bounds(values)

    results.topic_patterns = _detect_topic_patterns(
        metrics.topics, results.topic_quartiles
    )

    # ---- Node patterns ----
    node_values = _extract_node_metric_values(metrics.nodes)
    for metric_name, values in node_values.items():
        results.node_quartiles[metric_name] = _compute_quartile_bounds(values)

    results.node_patterns = _detect_node_patterns(
        metrics.nodes, results.node_quartiles
    )

    # ---- Library patterns ----
    lib_values = _extract_lib_metric_values(metrics.libraries)
    for metric_name, values in lib_values.items():
        results.lib_quartiles[metric_name] = _compute_quartile_bounds(values)

    results.lib_patterns = _detect_lib_patterns(
        metrics.libraries, results.lib_quartiles
    )

    return results


# ---------------------------------------------------------------------------
# Quartile computation
# ---------------------------------------------------------------------------

def _compute_quartile_bounds(values: List[float]) -> QuartileBounds:
    """Compute Q1, Q3, min, max for a list of values."""
    if not values:
        return QuartileBounds(q1=0, q3=0, min_val=0, max_val=0, degenerate=True)

    sorted_v = sorted(values)
    n = len(sorted_v)
    min_val = sorted_v[0]
    max_val = sorted_v[-1]

    if n < 4:
        q1 = sorted_v[0]
        q3 = sorted_v[-1]
    else:
        q1 = _percentile(sorted_v, 25)
        q3 = _percentile(sorted_v, 75)

    degenerate = (q1 == q3)

    return QuartileBounds(
        q1=q1, q3=q3, min_val=min_val, max_val=max_val, degenerate=degenerate
    )


def _percentile(data: List[float], p: float) -> float:
    """Compute p-th percentile using linear interpolation."""
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(data) - 1)
    return data[f] + (k - f) * (data[c] - data[f])


# ---------------------------------------------------------------------------
# Relative interpretation helpers
# ---------------------------------------------------------------------------

def _is_high(value: float, bounds: QuartileBounds) -> bool:
    """M(x)↑ — relatively high."""
    if bounds.degenerate:
        # Only absolute max qualifies, and only if it's above the common value
        return value == bounds.max_val and bounds.max_val > bounds.q3
    return value >= bounds.q3


def _is_low(value: float, bounds: QuartileBounds) -> bool:
    """M(x)↓ — relatively low."""
    if bounds.degenerate:
        # Only absolute min qualifies, and only if it's below the common value
        return value == bounds.min_val and bounds.min_val < bounds.q1
    return value <= bounds.q1


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def _extract_app_metric_values(apps: List[ApplicationMetrics]) -> Dict[str, List[float]]:
    return {
        "R": [float(a.reach) for a in apps],
        "AMP": [a.amplification for a in apps],
        "RA": [a.role_asymmetry for a in apps],
        "TC": [float(a.topic_context_diversity) for a in apps],
        "LE": [float(a.library_exposure) for a in apps],
    }


def _extract_topic_metric_values(topics: List[TopicMetrics]) -> Dict[str, List[float]]:
    return {
        "C": [float(t.coverage) for t in topics],
        "I": [t.imbalance for t in topics],
        "PS": [float(t.physical_spread) for t in topics],
        "LCR": [t.low_connectivity_ratio for t in topics],
    }


def _extract_node_metric_values(nodes: List[NodeMetrics]) -> Dict[str, List[float]]:
    return {
        "ND": [float(n.density) for n in nodes],
        "NID": [float(n.interaction_density) for n in nodes],
    }


def _extract_lib_metric_values(libs: List[LibraryMetrics]) -> Dict[str, List[float]]:
    return {
        "LC": [float(l.coverage) for l in libs],
        "LCon": [float(l.concentration) for l in libs],
    }


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def _detect_app_patterns(
    apps: List[ApplicationMetrics],
    quartiles: Dict[str, QuartileBounds],
) -> Dict[str, List[PatternMatch]]:
    """Detect application-level structural patterns."""
    wr: List[PatternMatch] = []   # Wide Reach
    rs: List[PatternMatch] = []   # Role Skew
    cs: List[PatternMatch] = []   # Context Spread
    sd: List[PatternMatch] = []   # Shared Dependency Exposure

    for a in apps:
        pm = PatternMatch(id=a.id, name=a.name)

        # WR: R↑ ∧ AMP↑
        if (_is_high(float(a.reach), quartiles["R"]) and
                _is_high(a.amplification, quartiles["AMP"])):
            wr.append(pm)

        # RS: RA↑ ∨ RA↓
        if (_is_high(a.role_asymmetry, quartiles["RA"]) or
                _is_low(a.role_asymmetry, quartiles["RA"])):
            rs.append(pm)

        # CS: TC↑
        if _is_high(float(a.topic_context_diversity), quartiles["TC"]):
            cs.append(pm)

        # SD: LE↑
        if _is_high(float(a.library_exposure), quartiles["LE"]):
            sd.append(pm)

    return {"WR": wr, "RS": rs, "CS": cs, "SD": sd}


def _detect_topic_patterns(
    topics: List[TopicMetrics],
    quartiles: Dict[str, QuartileBounds],
) -> Dict[str, List[PatternMatch]]:
    """Detect topic-level structural patterns."""
    cb: List[PatternMatch] = []   # Communication Backbone
    dc: List[PatternMatch] = []   # Directional Concentration
    pa: List[PatternMatch] = []   # Peripheral Aggregator

    for t in topics:
        pm = PatternMatch(id=t.id, name=t.name)

        # CB: C↑ ∧ I↓
        if (_is_high(float(t.coverage), quartiles["C"]) and
                _is_low(t.imbalance, quartiles["I"])):
            cb.append(pm)

        # DC: I↑
        if _is_high(t.imbalance, quartiles["I"]):
            dc.append(pm)

        # PA: LCR↑
        if _is_high(t.low_connectivity_ratio, quartiles["LCR"]):
            pa.append(pm)

    return {"CB": cb, "DC": dc, "PA": pa}


def _detect_node_patterns(
    nodes: List[NodeMetrics],
    quartiles: Dict[str, QuartileBounds],
) -> Dict[str, List[PatternMatch]]:
    """Detect node-level structural patterns."""
    ih: List[PatternMatch] = []  # Interaction Hotspot

    for n in nodes:
        pm = PatternMatch(id=n.id, name=n.name)

        # IH: ND↑ ∧ NID↑
        if (_is_high(float(n.density), quartiles["ND"]) and
                _is_high(float(n.interaction_density), quartiles["NID"])):
            ih.append(pm)

    return {"IH": ih}


def _detect_lib_patterns(
    libs: List[LibraryMetrics],
    quartiles: Dict[str, QuartileBounds],
) -> Dict[str, List[PatternMatch]]:
    """Detect library-level structural patterns."""
    wul: List[PatternMatch] = []  # Widely Used Library
    cl: List[PatternMatch] = []   # Concentrated Library

    for l in libs:
        pm = PatternMatch(id=l.id, name=l.name)

        # WUL: LC↑
        if _is_high(float(l.coverage), quartiles["LC"]):
            wul.append(pm)

        # CL: LCon↑
        if _is_high(float(l.concentration), quartiles["LCon"]):
            cl.append(pm)

    return {"WUL": wul, "CL": cl}
