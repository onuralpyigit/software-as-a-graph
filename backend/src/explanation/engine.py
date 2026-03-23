"""
backend/src/explanation/engine.py — Explanation logic for architectural quality.

Takes machine-readable outputs and produces human-readable text.
"""
from dataclasses import dataclass
from typing import List, Dict, Any

from src.core.metrics import ComponentQuality, CriticalityLevel
from src.prediction.models import QualityAnalysisResult, DetectedProblem
from src.analysis.smells import AntiPatternReport

from src.explanation.templates import PATTERN_TEMPLATES, DEFAULT_DIMENSION_TEMPLATES


@dataclass
class DimensionExplanation:
    """Human-readable explanation of a single RMAV dimension's score."""
    dimension: str           # "Reliability"
    score: float             # 0.891
    level: str               # "CRITICAL"
    driving_metric: str      # "Reverse PageRank (RPR)"
    driving_value: float     # 0.87 (normalized)
    plain_meaning: str       # "14 downstream components depend on it transitively."
    risk_sentence: str       # "A failure here cascades to the entire planning stack."


@dataclass
class ComponentExplanation:
    """Full explanation for one component."""
    component_id: str
    pattern: str             # "Total Hub" | "SPOF" | "Bottleneck" …
    level: str               # "CRITICAL"
    one_line: str            # "App_Controller fails in three independent ways."
    top_risk: str            # "A single failure here activates three independent failure modes at once."
    dimensions: List[DimensionExplanation]   # one per elevated dimension
    priority_action: str     # single most impactful remediation step
    anti_patterns: List[str]     # IDs of any triggered catalog entries


@dataclass
class RemediationStep:
    """System-level single remediation action with associated components."""
    action: str
    components: List[str]
    priority: int


@dataclass
class SystemReport:
    """System-level explanation across all components."""
    total_components: int
    critical_count: int
    high_count: int
    deployment_blocked: bool     # True if any CRITICAL anti-pattern found
    reason: str                  # Reason for deployment block or status
    top_risk_summary: str        # 2-3 sentence plain English summary
    by_stakeholder: Dict[str, List[str]]  # role → list of action items
    component_explanations: List[ComponentExplanation]
    remediation_plan: List[RemediationStep]  # ordered, deduplicated


DIMENSION_DRIVERS = {
    "Reliability": [
        ("reverse_pagerank", "Reverse PageRank", "fraction of the system reachable from this component"),
        ("in_degree", "In-degree centrality", "number of components directly depending on it"),
        ("cdpot", "Cascade Depth Potential", "depth × breadth of its failure cascade"),
    ],
    "Maintainability": [
        ("betweenness", "Betweenness centrality", "routing traffic passing through it"),
        ("dependency_weight_out", "Weighted out-degree", "number and priority of interface dependencies"),
        ("coupling_risk", "Coupling Risk", "instability relative to its dependents"),
    ],
    "Availability": [
        ("ap_c_directed", "Directed SPOF score", "fraction of connectivity lost if it is removed"),
        ("bridge_ratio", "Bridge ratio", "fraction of its connections that are irreplaceable"),
        ("cdi", "Connectivity Degradation Index", "increase in average path length without it"),
    ],
    "Vulnerability": [
        ("dependency_weight_in", "QoS-weighted in-degree (QADS)", "attack surface — high-priority traffic flowing into it"),
        ("reverse_closeness", "Reverse Closeness", "how quickly compromise propagates to its dependents"),
        ("reverse_eigenvector", "Reverse Eigenvector", "connection quality to high-value downstream targets"),
    ],
}

def identify_driver(component: ComponentQuality, dimension: str) -> tuple[str, float, str]:
    """Return (metric_name, value, plain_meaning) for the top contributor."""
    metrics = DIMENSION_DRIVERS.get(dimension, [])
    if not metrics:
        return ("Unknown", 0.0, "unknown metric driver")
    
    values = [(name, getattr(component.structural, attr, 0.0), meaning)
              for attr, name, meaning in metrics]
    return max(values, key=lambda x: x[1])


class ExplanationEngine:
    """
    Engine for translating ComponentQuality and AntiPattern reports into 
    structured, human-readable explanations.
    """

    def explain_component(self, quality: ComponentQuality,
                          smells: List[DetectedProblem]) -> ComponentExplanation:
        """Generate human-readable explanations for a single component."""
        
        pattern = quality.profile.pattern if quality.profile else "Composite Risk"
        template_group = PATTERN_TEMPLATES.get(pattern, PATTERN_TEMPLATES["Composite Risk"])
        
        # Prepare context for interpolation
        ctx = quality.structural.to_dict()
        ctx["in_degree_raw"] = quality.structural.in_degree_raw
        ctx["out_degree_raw"] = quality.structural.out_degree_raw
        bridge_ratio = getattr(quality.structural, "bridge_ratio", 0.0)
        ctx["bridge_ratio_pct"] = round(bridge_ratio * 100, 1)
        
        # Additional context for Pattern-Specific Templates
        ctx["id"] = quality.id
        ctx["fragmented_pct"] = bridge_ratio
        ctx["cascade_count"] = quality.structural.in_degree_raw
        ctx["coupling_count"] = quality.structural.total_degree_raw

        dim_explanations: List[DimensionExplanation] = []
        
        # We only explain dimensions that are at least HIGH or explicitly flagged in the profile.
        # But we do it systematically via the scores.
        dimension_map = {
            "reliability": (quality.scores.reliability, quality.levels.reliability.value.upper()),
            "maintainability": (quality.scores.maintainability, quality.levels.maintainability.value.upper()),
            "availability": (quality.scores.availability, quality.levels.availability.value.upper()),
            "vulnerability": (quality.scores.vulnerability, quality.levels.vulnerability.value.upper()),
        }
        
        for dim, (score, level_str) in dimension_map.items():
            level_enum = CriticalityLevel(level_str.lower())
            is_elevated = level_enum >= CriticalityLevel.HIGH
            
            # Additional check to see if profile explicitly flagged it
            profile_flag = getattr(quality.profile, f"{dim[0]}_crit") if quality.profile else False
            
            if is_elevated or profile_flag:
                dim_template = template_group.get("dimensions", {}).get(dim)
                if not dim_template:
                    dim_template = DEFAULT_DIMENSION_TEMPLATES[dim]

                # Identify primary driver dynamically
                drv_metric_name, drv_val, drv_meaning = identify_driver(quality, dim.capitalize())

                template_meaning = dim_template["plain_meaning"].format(**ctx)
                plain_meaning = f"{template_meaning} Primary driver: {drv_meaning}."
                risk_sentence = dim_template["risk_sentence"].format(**ctx)

                dim_exp = DimensionExplanation(
                    dimension=dim.capitalize(),
                    score=round(score, 3),
                    level=level_str,
                    driving_metric=drv_metric_name,
                    driving_value=round(drv_val, 3),
                    plain_meaning=plain_meaning,
                    risk_sentence=risk_sentence
                )
                dim_explanations.append(dim_exp)

        # Build list of triggered anti-pattern descriptions
        anti_pattern_ids = [s.name for s in smells]
        
        # Final population
        component_exp = ComponentExplanation(
            component_id=quality.id,
            pattern=pattern,
            level=quality.levels.overall.value.upper(),
            one_line=template_group.get("one_line", "{id} requires attention across multiple dimensions.").format(**ctx),
            top_risk=template_group.get("top_risk", "Multiple systemic vulnerabilities detected.").format(**ctx),
            dimensions=dim_explanations,
            priority_action=template_group.get("priority_action", "Review architecture and decouple.").format(**ctx),
            anti_patterns=anti_pattern_ids
        )
        return component_exp

    def explain_system(self, quality_result: QualityAnalysisResult,
                       smell_report: AntiPatternReport) -> SystemReport:
        """Generate a system-level report with executive summaries and action plans."""
        from .templates import STAKEHOLDER_MAPPING
        
        all_components = quality_result.components
        total = len(all_components)
        
        critical_comps = [c for c in all_components if c.levels.overall == CriticalityLevel.CRITICAL]
        high_comps = [c for c in all_components if c.levels.overall == CriticalityLevel.HIGH]
        
        component_explanations = []
        for c in all_components:
            if c.levels.overall >= CriticalityLevel.HIGH:
                my_smells = [s for s in smell_report.problems if s.entity_id == c.id]
                component_explanations.append(self.explain_component(c, my_smells))
        
        # Check if deployment blocked
        critical_smells = [s for s in smell_report.problems if s.severity.upper() == "CRITICAL"]
        deployment_blocked = len(critical_smells) > 0
        
        # Reason for block
        if deployment_blocked:
            block_names = sorted(list(set(s.name for s in critical_smells)))
            reason = f"{len(critical_smells)} CRITICAL anti-patterns detected ({', '.join(block_names)})."
        else:
            reason = "No critical architectural blockers detected."

        # Summary text
        top_risk_summary = f"{len(critical_comps)} CRITICAL components require action before deployment. "
        top_risk_summary += f"{len(high_comps)} HIGH components should be addressed in the current sprint. "
        
        if critical_comps:
            risk_drivers = []
            for c in critical_comps[:2]:
                exp = self.explain_component(c, [])
                for dim in exp.dimensions:
                    if dim.level == "CRITICAL":
                        risk_drivers.append(dim.plain_meaning)
            if risk_drivers:
                top_risk_summary += f"The system's primary architectural risk is concentration: {risk_drivers[0]} " \
                                   f"and have limited structural redundancy."

        # Aggregate remediation into prioritized plan AND stakeholder map
        by_stakeholder = {
            "Reliability Engineer": [],
            "Software Architect": [],
            "DevOps / SRE": [],
            "Security Engineer": []
        }
        
        action_map = {}
        for exp in component_explanations:
            act = exp.priority_action
            
            # Identify stakeholder
            role = STAKEHOLDER_MAPPING["patterns"].get(exp.pattern)
            if not role:
                # Fallback to dimensions
                roles = []
                for dim in exp.dimensions:
                    if dim.level in ("CRITICAL", "HIGH"):
                        roles.append(STAKEHOLDER_MAPPING["dimensions"].get(dim.dimension))
                role = roles[0] if roles else "Software Architect"
            
            if act not in by_stakeholder[role]:
                by_stakeholder[role].append(act)

            if act not in action_map:
                prio = 1 if exp.level == "CRITICAL" else 2
                action_map[act] = RemediationStep(
                    action=act,
                    components=[],
                    priority=prio
                )
            if exp.component_id not in action_map[act].components:
                action_map[act].components.append(exp.component_id)

        sorted_steps = sorted(list(action_map.values()), key=lambda x: (x.priority, -len(x.components)))

        return SystemReport(
            total_components=total,
            critical_count=len(critical_comps),
            high_count=len(high_comps),
            deployment_blocked=deployment_blocked,
            reason=reason,
            top_risk_summary=top_risk_summary,
            by_stakeholder=by_stakeholder,
            component_explanations=component_explanations,
            remediation_plan=sorted_steps
        )
