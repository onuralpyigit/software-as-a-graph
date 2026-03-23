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
        "one_line": "This component is deeply entrenched, critically exposed, and a single point of failure.",
        "priority_action": "Introduce redundancy and decouple its most heavily used endpoints.",
        "full_remediation": [
            "Deploy active-active redundancy.",
            "Implement circuit breakers on incoming traffic.",
            "Split its responsibilities into smaller, domain-specific services."
        ],
        "dimensions": {
            "reliability": {
                "driving_metric": "Reverse PageRank (RPR)",
                "plain_meaning": "{in_degree_raw} downstream components depend on it transitively.",
                "risk_sentence": "A failure here cascades to the entire dependent stack."
            },
            "maintainability": {
                "driving_metric": "Betweenness Centrality (BC)",
                "plain_meaning": "It sits on {bridge_ratio_pct}% of the system's structural bridges.",
                "risk_sentence": "Changes here are highly likely to break independent modules."
            },
            "availability": {
                "driving_metric": "Articulation Point (AP)",
                "plain_meaning": "It represents a critical single point of failure.",
                "risk_sentence": "Without it, the dependent subsystems cannot communicate."
            },
            "vulnerability": {
                "driving_metric": "Reverse Eigenvector",
                "plain_meaning": "It is widely exposed to unauthenticated or risky paths.",
                "risk_sentence": "An exploit here gives access to a massive portion of the architecture."
            }
        }
    },
    "Reliability Hub": {
        "one_line": "Many downstream components rely on this for their normal operation.",
        "priority_action": "Add strict SLAs and aggressive caching to its outputs.",
        "full_remediation": [
            "Implement caching for read-heavy operations.",
            "Enforce strict API versioning to prevent breaking downstream."
        ],
        "dimensions": {
            "reliability": {
                "driving_metric": "Reverse PageRank (RPR)",
                "plain_meaning": "{in_degree_raw} components depend on it directly or indirectly.",
                "risk_sentence": "A degradation here causes widespread cascading latency or failure."
            }
        }
    },
    "Bottleneck": {
        "one_line": "This component orchestrates too many disparate dependencies.",
        "priority_action": "Refactor to apply the Dependency Inversion Principle.",
        "full_remediation": [
            "Use event-driven architecture to invert dependencies.",
            "Decompose the monolith into focused bounded contexts."
        ],
        "dimensions": {
            "maintainability": {
                "driving_metric": "Betweenness Centrality & Out-Degree",
                "plain_meaning": "It coordinates {out_degree_raw} external boundaries.",
                "risk_sentence": "Any localized change forces a re-test of the entire workflow."
            }
        }
    },
    "SPOF": {
        "one_line": "This component represents a critical structural disconnect risk.",
        "priority_action": "Deploy multiple active instances across availability zones.",
        "full_remediation": [
            "Implement active-active or active-passive deployment.",
            "Eliminate persistent local state to allow clustering."
        ],
        "dimensions": {
            "availability": {
                "driving_metric": "QoS Weighted SPOF",
                "plain_meaning": "It is isolated as an articulation point.",
                "risk_sentence": "A discrete crash completely halts data flow in this region."
            }
        }
    },
    "Attack Target": {
        "one_line": "Highly exposed to the edges of the application, representing a prime vulnerability.",
        "priority_action": "Audit access controls and implement zero-trust policies.",
        "full_remediation": [
            "Enforce strict input validation on all entry points.",
            "Run isolated in a highly restrictive network policy."
        ],
        "dimensions": {
            "vulnerability": {
                "driving_metric": "Reverse Closeness / Exposure",
                "plain_meaning": "It is only a few hops away from untrusted entry points.",
                "risk_sentence": "If compromised, it offers immediate lateral movement to core systems."
            }
        }
    },
    "Fragile Hub": {
        "one_line": "A heavily relied-upon hub that is simultaneously a single point of failure.",
        "priority_action": "Immediate horizontal scaling and load balancing.",
        "full_remediation": [
            "Scale horizontally behind a resilient load balancer.",
            "Introduce bulkheads to restrict failure blast radius."
        ],
        "dimensions": {
            "reliability": {
                "driving_metric": "Reverse PageRank (RPR)",
                "plain_meaning": "It supports {in_degree_raw} incoming dependencies.",
                "risk_sentence": "Its workload is heavy, increasing the chance of overload."
            },
            "availability": {
                "driving_metric": "Articulation Point",
                "plain_meaning": "There are no alternate redundant fallback routes.",
                "risk_sentence": "When it overloads and crashes, the graph separates."
            }
        }
    },
    "Exposed Bottleneck": {
        "one_line": "A high-friction maintenance area that is also widely exposed to external input.",
        "priority_action": "Isolate the component behind a strict API gateway.",
        "full_remediation": [
            "Implement a Facade to untangle direct integrations.",
            "Audit all incoming paths for injection risks."
        ],
        "dimensions": {
            "maintainability": {
                "driving_metric": "Betweenness Centrality",
                "plain_meaning": "It merges {in_degree_raw} incoming with {out_degree_raw} outgoing edges.",
                "risk_sentence": "Developer cognitive load here is extremely high."
            },
            "vulnerability": {
                "driving_metric": "Weighted Exposure",
                "plain_meaning": "It processes payloads from diverse, untrusted sources.",
                "risk_sentence": "Complex logic combined with high exposure often hides zero-days."
            }
        }
    },
    "Composite Risk": {
        "one_line": "This component exhibits a combination of structural risks across multiple dimensions.",
        "priority_action": "Investigate structural metrics to determine explicit decoupled fixes.",
        "full_remediation": [
            "Review code quality metrics and refactor.",
            "Ensure operational monitoring is exceptionally tight."
        ],
        "dimensions": {}
    }
}
