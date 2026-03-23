"""
Explanation Templates

Contains dictionary-based templates for standard human-readable explanations.
These templates are interpolated by the ExplanationEngine using metric evidence.
"""

# Default fallbacks when a specific metric template isn't found
DEFAULT_DIMENSION_TEMPLATES = {
    "reliability": {
        "driving_metric": "Reverse PageRank / In-Degree",
        "plain_meaning": "This component has many incoming dependencies ({in_degree_raw} direct).",
        "risk_sentence": "Unplanned downtime here will broadly impact downstream services.",
    },
    "maintainability": {
        "driving_metric": "Betweenness / Out-Degree",
        "plain_meaning": "It acts as a structural bridge with {out_degree_raw} outgoing dependencies.",
        "risk_sentence": "Modifying this component carries a high risk of unintended side-effects.",
    },
    "availability": {
        "driving_metric": "SPOF / Bridge Ratio",
        "plain_meaning": "It is an articulation point in the architecture.",
        "risk_sentence": "If this component fails, sections of the system will be entirely disconnected.",
    },
    "vulnerability": {
        "driving_metric": "Reverse Eigenvector / Exposure",
        "plain_meaning": "It is transitively exposed to many other high-risk components.",
        "risk_sentence": "Security or performance issues here will rapidly compound.",
    }
}

PATTERN_TEMPLATES = {
    "Total Hub": {
        "one_line": "{id} is a critical hub — it concentrates reliability, "
                    "availability, and change risk simultaneously.",
        "top_risk": "A single failure here activates three independent failure modes at once. "
                    "It is the highest-priority component in the system.",
        "priority_action": "Introduce a redundant replica and circuit breakers before deployment.",
    },
    "SPOF": {
        "one_line": "{id} is a structural single point of failure.",
        "top_risk": "Removing it disconnects {fragmented_pct:.0%} of the system's "
                    "dependency graph. No redundant path exists.",
        "priority_action": "Deploy a failover replica or add an alternative routing path.",
    },
    "Reliability Hub": {
        "one_line": "{id} is a reliability hub — failure cascades broadly.",
        "top_risk": "{cascade_count} downstream components depend on it transitively. "
                    "Its failure silences their data flows.",
        "priority_action": "Add circuit breakers and health checks in all direct dependents.",
    },
    "Bottleneck": {
        "one_line": "{id} is a change bottleneck — interface changes are expensive.",
        "top_risk": "{coupling_count} components must be updated whenever its interface changes. "
                    "This is the highest maintenance cost in the system.",
        "priority_action": "Extract a stable interface; reduce outgoing dependency count.",
    },
    "Attack Target": {
        "one_line": "{id} is a high-value attack target.",
        "top_risk": "It receives high-priority traffic from {in_degree_raw} publishers and "
                    "is reachable from most of the system. Compromise propagates broadly.",
        "priority_action": "Apply access controls, input validation, and network isolation.",
    },
    "Fragile Hub": {
        "one_line": "{id} combines reliability risk with availability risk.",
        "top_risk": "It is simultaneously a cascade source and a structural SPOF. "
                    "Failure stops data flows and disconnects the graph.",
        "priority_action": "Redundancy is essential — both for availability and to reduce blast radius.",
    },
    "Exposed Bottleneck": {
        "one_line": "{id} is a change bottleneck that is also an attractive attack target.",
        "top_risk": "High coupling makes it difficult to change safely; high attack surface "
                    "makes it a priority for hardening.",
        "priority_action": "Harden the interface first, then reduce outgoing coupling.",
    },
    "Composite Risk": {
        "one_line": "{id} has elevated risk across multiple dimensions without a single dominant cause.",
        "top_risk": "No one metric dominates, but the combination across R, M, A, V puts it "
                    "above the system's population threshold.",
        "priority_action": "Review coupling and redundancy together; no single fix resolves this.",
    },
}
