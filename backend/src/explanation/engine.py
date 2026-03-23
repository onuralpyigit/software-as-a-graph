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
    severity: str            # "CRITICAL"
    one_line: str            # "App_Controller fails in three independent ways."
    dimensions: List[DimensionExplanation]   # one per elevated dimension
    priority_action: str     # single most impactful remediation step
    full_remediation: List[str]  # ordered list of steps
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
    top_risk_summary: str        # 2-3 sentence plain English summary
    by_stakeholder: Dict[str, List[str]]  # role → list of action items
    component_explanations: List[ComponentExplanation]
    remediation_plan: List[RemediationStep]  # ordered, deduplicated


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

        dim_explanations: List[DimensionExplanation] = []
        
        # We only explain dimensions that are at least HIGH or explicitly flagged in the profile.
        # But we do it systematically via the scores.
        dimension_map = {
            "reliability": (quality.scores.reliability, quality.levels.reliability.value.upper(), quality.structural.reverse_pagerank),
            "maintainability": (quality.scores.maintainability, quality.levels.maintainability.value.upper(), quality.structural.betweenness),
            "availability": (quality.scores.availability, quality.levels.availability.value.upper(), quality.structural.ap_c_directed),
            "vulnerability": (quality.scores.vulnerability, quality.levels.vulnerability.value.upper(), quality.structural.reverse_eigenvector),
        }
        
        for dim, (score, level_str, drv_val) in dimension_map.items():
            level_enum = CriticalityLevel(level_str.lower())
            is_elevated = level_enum >= CriticalityLevel.HIGH
            
            # Additional check to see if profile explicitly flagged it
            profile_flag = getattr(quality.profile, f"{dim[0]}_crit") if quality.profile else False
            
            if is_elevated or profile_flag:
                dim_template = template_group.get("dimensions", {}).get(dim)
                if not dim_template:
                    dim_template = DEFAULT_DIMENSION_TEMPLATES[dim]

                plain_meaning = dim_template["plain_meaning"].format(**ctx)
                risk_sentence = dim_template["risk_sentence"].format(**ctx)

                dim_exp = DimensionExplanation(
                    dimension=dim.capitalize(),
                    score=round(score, 3),
                    level=level_str,
                    driving_metric=dim_template["driving_metric"],
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
            severity=quality.levels.overall.value.upper(),
            one_line=template_group.get("one_line", f"This component requires attention across {len(dim_explanations)} dimensions."),
            dimensions=dim_explanations,
            priority_action=template_group.get("priority_action", "Review architecture and decouple."),
            full_remediation=template_group.get("full_remediation", []),
            anti_patterns=anti_pattern_ids
        )
        return component_exp

    def explain_system(self, quality_result: QualityAnalysisResult,
                       smell_report: AntiPatternReport) -> SystemReport:
        """Generate a system-level report with executive summaries and action plans."""
        
        all_components = quality_result.components
        total = len(all_components)
        
        critical_comps = [c for c in all_components if c.levels.overall == CriticalityLevel.CRITICAL]
        high_comps = [c for c in all_components if c.levels.overall == CriticalityLevel.HIGH]
        
        component_explanations = []
        for c in all_components:
            if c.levels.overall >= CriticalityLevel.HIGH:
                my_smells = [s for s in smell_report.problems if s.entity_id == c.id]
                component_explanations.append(self.explain_component(c, my_smells))
        
        # Check if deployment blocked: any CRITICAL severity smell (or CRITICAL components?)
        # Specification says: "True if any CRITICAL anti-pattern found"
        deployment_blocked = any(s.severity.upper() == "CRITICAL" for s in smell_report.problems)

        # Generate top risk summary
        if critical_comps:
            crit_names = [c.id for c in critical_comps[:2]]
            top_risk_summary = f"The system architecture is at critical risk due to {' and '.join(crit_names)}. " \
                               f"These structural bottlenecks threaten both stability and scale."
        elif high_comps:
            top_risk_summary = "System health is moderate, but several high-risk hubs are beginning to emerge. " \
                               "Refactoring now will prevent future cascading failures."
        else:
            top_risk_summary = "The architecture is structurally sound with no high-risk bottlenecks or SPOFs."

        # Aggregate remediation into prioritized plan
        # We group by action to avoid repeats
        action_map = {}
        for exp in component_explanations:
            act = exp.priority_action
            if act not in action_map:
                prio = 1 if exp.severity == "CRITICAL" else 2
                action_map[act] = RemediationStep(
                    action=act,
                    components=[],
                    priority=prio
                )
            action_map[act].components.append(exp.component_id)

        # Sort the step map by priority (1=Critical, 2=High), then length of components desc
        sorted_steps = sorted(list(action_map.values()), key=lambda x: (x.priority, -len(x.components)))

        # Naive stakeholder mapping based on dimension presence
        by_stakeholder = {
            "Architect": ["Review overall structural decouplings and new bounded contexts."],
            "DevOps": ["Evaluate single-points-of-failure for active-active deployment."],
            "Security": ["Audit highly exposed routing components and enforce zero-trust."]
        }
        
        # Add specific notes based on findings
        if deployment_blocked:
            by_stakeholder["DevOps"].insert(0, "URGENT: Deployment halt required to resolve critical anti-patterns.")

        return SystemReport(
            total_components=total,
            critical_count=len(critical_comps),
            high_count=len(high_comps),
            deployment_blocked=deployment_blocked,
            top_risk_summary=top_risk_summary,
            by_stakeholder=by_stakeholder,
            component_explanations=component_explanations,
            remediation_plan=sorted_steps
        )
