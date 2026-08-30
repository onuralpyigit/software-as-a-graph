"""
saag/usecases/diagnostic.py

Pathway A (Diagnostic / ISO-RM) Use Case.
Executes deterministic, standards-grounded architectural quality attribution
(ISO/IEC 25010 / ISO/IEC 25019) over the structural graph:
- Fault Tolerance (cascade depth, propagation paths, fan-out criticality)
- Availability (Single Points of Failure exposure, bridge ratios)
- Maintainability (multi-path coupling complexity, code quality penalties)
- Architectural Anti-Pattern Detection & Natural-Language Explanations
"""
from typing import Any, List, Optional, Tuple

from saag.analysis.models import (
    DetectedProblem,
    ProblemSummary,
    QualityAnalysisResult,
    StructuralAnalysisResult,
)


class DiagnosticUseCase:
    """
    Use Case for Pathway A: Standards-grounded Architectural Diagnosis (ISO/IEC RM).
    
    INDEPENDENCE GUARANTEE:
    This use case operates deterministically on structural analysis results and
    declared QoS topology metrics. It never accesses raw simulation runtime data,
    preserving the pre-deployment evaluation guarantee.
    """

    def __init__(self, prediction_service: Optional[Any] = None, analysis_service: Optional[Any] = None):
        self.prediction_service = prediction_service
        self.analysis_service = analysis_service

    def execute(
        self,
        layer: str = "system",
        structural_result: Optional[StructuralAnalysisResult] = None,
        detect_problems: bool = True,
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
        **kwargs,
    ) -> Tuple[QualityAnalysisResult, Optional[List[DetectedProblem]], Optional[ProblemSummary], Optional[Any]]:
        """
        Execute Pathway A diagnostic attribution on a structural result or by querying analysis_service.
        
        Returns:
            Tuple of (quality_result, detected_problems, problem_summary, explanation)
        """
        if structural_result is None:
            if self.analysis_service is None:
                raise ValueError("Either structural_result or an initialized analysis_service must be provided.")
            layer_res = self.analysis_service.analyze_layer(layer)
            structural_result = layer_res.structural

        # Use prediction_service if available (holds QualityAnalyzer orchestration), else instantiate analyzer
        if self.prediction_service is not None:
            quality = self.prediction_service.predict_quality(
                structural_result,
                run_sensitivity=run_sensitivity,
                **kwargs,
            )
        else:
            from saag.analysis.analyzer import QualityAnalyzer
            analyzer = QualityAnalyzer()
            quality = analyzer.analyze(structural_result, run_sensitivity=run_sensitivity)

        problems: Optional[List[DetectedProblem]] = None
        summary: Optional[ProblemSummary] = None
        explanation: Optional[Any] = None

        if detect_problems:
            from saag.analysis.antipattern_detector import AntiPatternDetector
            from saag.analysis.problem_detector import ProblemDetector
            from saag.analysis.smells import AntiPatternReport
            from saag.explanation.engine import ExplanationEngine

            detector = AntiPatternDetector(active_patterns=active_patterns)
            problems = detector.detect(quality, layer=layer)
            summary = ProblemDetector().summarize(problems)
            smell_report = AntiPatternReport(
                problems=problems,
                summary=summary.to_dict() if hasattr(summary, "to_dict") else summary,
            )
            explanation = ExplanationEngine().explain_system(quality, smell_report)

            quality.problems = problems
            quality.problem_summary = summary
            quality.explanation = explanation
            quality.prediction_mode = "rm"
            quality.failed_patterns = detector.failed_patterns

        return quality, problems, summary, explanation


# Backward-compatible alias
DiagnosticGraphUseCase = DiagnosticUseCase
