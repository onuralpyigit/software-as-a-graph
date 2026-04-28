"""
CLI Formatting Utilities for Explanations
"""
from typing import List, Optional
import sys

from .engine import ComponentExplanation, DimensionExplanation, ExplanationEngine
from saag.prediction.models import QualityAnalysisResult, DetectedProblem

class CLIFormatter:
    """
    Formats architectural explanations into human-readable CLI cards.
    """

    @staticmethod
    def format_component_card(exp: ComponentExplanation) -> str:
        """
        Renders a single component explanation as a card.
        """
        width = 60
        border = "━" * width
        
        # Header
        level_str = f"[{exp.level}]"
        header = f"  {level_str:<12} {exp.component_id}   —  {exp.pattern}"
        
        lines = [
            border,
            header,
            border,
            "",
            f"  {exp.one_line}",
            "",
            f"  {exp.top_risk}",
            "",
            "  Breakdown:"
        ]
        
        for dim in exp.dimensions:
            dim_header = f"    {dim.dimension:<16} ({dim.score:.3f} — {dim.level})"
            lines.append(dim_header)
            
            # Driver info
            import textwrap
            prefix = "      Driven by: "
            driver_info = f"{dim.driving_metric} {dim.driving_value:.2f} — {dim.plain_meaning}"
            
            wrapped = textwrap.wrap(
                driver_info, 
                width=width-len(prefix)-2,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix)
            )
            lines.extend([f"  {wl}" for wl in wrapped])
            lines.append("")
            
        lines.append("  Priority Action:")
        import textwrap
        action_wrapped = textwrap.wrap(exp.priority_action, width=width-4, initial_indent="    ", subsequent_indent="    ")
        lines.extend(action_wrapped)
        
        return "\n".join(lines)

    @staticmethod
    def format_system_report(report: any, layer: str) -> str:
        """
        Renders a system-level report as a footer block.
        """
        width = 60
        border = "━" * width
        
        lines = [
            border,
            f"  System Report — {layer} layer  ({report.total_components} components)",
            border,
            "",
            f"  Deployment status: {'BLOCKED' if report.deployment_blocked else 'GO / PROCEED'}",
            f"  Reason: {report.reason}",
            "",
            "  Summary:",
            f"    {report.critical_count} CRITICAL components require action before deployment.",
            f"    {report.high_count} HIGH components should be addressed in the current sprint.",
        ]
        
        # Wrap the top risk summary
        import textwrap
        summary_wrapped = textwrap.wrap(report.top_risk_summary, width=width-4, initial_indent="    ", subsequent_indent="    ")
        lines.extend(summary_wrapped)
        lines.append("")
        
        lines.append("  By stakeholder:")
        for stakeholder, actions in report.by_stakeholder.items():
            if not actions:
                continue
            lines.append("")
            lines.append(f"    {stakeholder} ({len(actions)} items):")
            for i, action in enumerate(actions, 1):
                # Simple wrapping for stakeholder actions
                action_wrapped = textwrap.wrap(action, width=width-9, initial_indent=f"      {i}. ", subsequent_indent="         ")
                lines.extend(action_wrapped)
        
        lines.append("")
        lines.append("  Prioritised remediation backlog:")
        for i, step in enumerate(report.remediation_plan[:10], 1):
            level = "CRITICAL — deploy blocker" if step.priority == 1 else "HIGH"
            p_label = f"P{i} [{level}]:"
            # Wrap the action line
            action_wrapped = textwrap.wrap(
                step.action, 
                width=width-2, 
                initial_indent=f"    {p_label:<32} ", 
                subsequent_indent=" " * 37
            )
            lines.extend(action_wrapped)
            
        return "\n".join(lines)

    @staticmethod
    def print_critical_report(
        quality_result: QualityAnalysisResult, 
        problems: Optional[List[DetectedProblem]] = None,
        limit_top: int = 5
    ) -> None:
        """
        Generates and prints cards for top CRITICAL and HIGH components,
        followed by a system-level summary.
        """
        engine = ExplanationEngine()
        
        # We only care about HIGH and CRITICAL components for the card output
        targets = [c for c in quality_result.components if c.levels.overall.value.upper() in ("CRITICAL", "HIGH")]
        
        if not targets:
            print("\n  ✅ No CRITICAL or HIGH risk components detected.")
            # Even if no critical pieces, we might want the system summary if there were smells
            # but for now let's keep it simple.
            return

        # Sort by score descending (they should already be sorted but being safe)
        targets.sort(key=lambda x: x.scores.overall, reverse=True)
        
        print(f"\n  [Architectural Risk Report — Top {min(len(targets), limit_top)} Components]")
        
        # Problems mapping for faster lookup
        problems_by_id = {}
        smell_report_mock = type('obj', (object,), {'problems': problems or []}) # Quick shim
        if problems:
            for p in problems:
                if p.entity_id not in problems_by_id:
                    problems_by_id[p.entity_id] = []
                problems_by_id[p.entity_id].append(p)

        for comp in targets[:limit_top]:
            comp_problems = problems_by_id.get(comp.id, [])
            explanation = engine.explain_component(comp, comp_problems)
            print("\n" + CLIFormatter.format_component_card(explanation))
            
        # Print System Report
        from saag.analysis.smells import AntiPatternReport
        # If we have a list of problems but no AntiPatternReport, we wrap it
        if not isinstance(problems, AntiPatternReport):
            # This is a bit of a hack for the existing call signature
            # In a real scenario we'd pass the actual report
            report_for_engine = AntiPatternReport(problems=problems or [], summary={})
        else:
            report_for_engine = problems
            
        system_report = engine.explain_system(quality_result, report_for_engine)
        print("\n" + CLIFormatter.format_system_report(system_report, quality_result.layer))
