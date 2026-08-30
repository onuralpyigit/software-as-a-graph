"""
api/presenters/triage_presenter.py

Presenter for Triage and Stakeholder-Oriented Root-Cause Attribution.
Formats Top-K shortlisted components and maps architectural remediation
actions to distinct stakeholder roles:
- DevOps / SRE: Host anti-affinity, broker/service replication, cluster topology
- Architect: Topic splitting, circuit breakers, transport QoS upgrades, decoupling
- Developer: Code complexity refactoring, coupling reduction, dead dependency cleanup
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from saag.analysis.triage import TriageResult, TriageEntry


def format_triage_entry(entry: TriageEntry) -> Dict[str, Any]:
    """Format a single TriageEntry dictionary for JSON serialization."""
    return entry.to_dict()


def format_triage_result(result: TriageResult) -> Dict[str, Any]:
    """Format a TriageResult object into a standardized API response dictionary."""
    return result.to_dict()


def categorize_by_stakeholder(
    triage_result: TriageResult,
    name_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Groups triage entries and recommended remediation actions by stakeholder role.

    Returns:
        Dict mapping stakeholder roles ('devops_sre', 'architect', 'developer')
        to their respective component items and prioritized actions.
    """
    lookup = name_lookup or {}
    role_buckets: Dict[str, List[Dict[str, Any]]] = {
        "devops_sre": [],
        "architect": [],
        "developer": [],
    }

    role_mapping = {
        "SRE": "devops_sre",
        "DevOps": "devops_sre",
        "Architect": "architect",
        "Software Architect": "architect",
        "Developer": "developer",
        "Software Engineer": "developer",
    }

    for entry in triage_result.entries:
        item = {
            "component_id": entry.component_id,
            "component_name": lookup.get(entry.component_id, entry.component_id),
            "rank": entry.rank,
            "ranking_score": round(entry.ranking_score, 4),
            "component_type": entry.component_type,
            "pattern": entry.pattern,
            "criticality_level": entry.level,
            "priority_action": entry.priority_action,
            "elevated_dimensions": entry.elevated_dimensions,
        }

        assigned_roles = set()
        for role_str in entry.roles:
            canonical_role = role_mapping.get(role_str, "architect")
            assigned_roles.add(canonical_role)

        if not assigned_roles:
            assigned_roles.add("architect")

        for role_key in assigned_roles:
            if role_key in role_buckets:
                role_buckets[role_key].append(item)

    return {
        "layer": triage_result.layer,
        "k": triage_result.k,
        "ranking_source": triage_result.ranking_source,
        "population": triage_result.population,
        "stakeholders": {
            "devops_sre": {
                "role_name": "DevOps / Site Reliability Engineers",
                "focus": "Infrastructure locality, host anti-affinity, broker replication, and SPOF mitigation",
                "count": len(role_buckets["devops_sre"]),
                "items": role_buckets["devops_sre"],
            },
            "architect": {
                "role_name": "System & Software Architects",
                "focus": "Pub-sub topology decoupling, topic splitting, transport QoS policies, and circuit breakers",
                "count": len(role_buckets["architect"]),
                "items": role_buckets["architect"],
            },
            "developer": {
                "role_name": "Software Engineers / Developers",
                "focus": "Internal code complexity, coupling metrics (MPCI), and component modularity",
                "count": len(role_buckets["developer"]),
                "items": role_buckets["developer"],
            },
        },
    }
