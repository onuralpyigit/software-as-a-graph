"""
CLI Formatting Utilities for Explanations
"""
from typing import List, Optional
import sys

from .engine import ComponentExplanation, DimensionExplanation, ExplanationEngine
from src.prediction.models import QualityAnalysisResult, DetectedProblem

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
    def print_critical_report(
        quality_result: QualityAnalysisResult, 
        problems: Optional[List[DetectedProblem]] = None,
        limit_top: int = 5
    ) -> None:
        """
        Generates and prints cards for top CRITICAL and HIGH components.
        """
        engine = ExplanationEngine()
        
        # We only care about HIGH and CRITICAL components for the card output
        targets = [c for c in quality_result.components if c.levels.overall.value.upper() in ("CRITICAL", "HIGH")]
        
        if not targets:
            print("\n  ✅ No CRITICAL or HIGH risk components detected.")
            return

        # Sort by score descending (they should already be sorted but being safe)
        targets.sort(key=lambda x: x.scores.overall, reverse=True)
        
        print(f"\n  [Architectural Risk Report — Top {min(len(targets), limit_top)} Components]")
        
        # Problems mapping for faster lookup
        problems_by_id = {}
        if problems:
            for p in problems:
                if p.entity_id not in problems_by_id:
                    problems_by_id[p.entity_id] = []
                problems_by_id[p.entity_id].append(p)

        for comp in targets[:limit_top]:
            comp_problems = problems_by_id.get(comp.id, [])
            explanation = engine.explain_component(comp, comp_problems)
            print("\n" + CLIFormatter.format_component_card(explanation))
