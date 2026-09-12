"""
Prediction Service

Orchestrates the unified Prediction Step (Step 3): rule-based RM scoring
(always), ML/GNN scoring (when a trained checkpoint is available), and
anti-pattern detection + explanations on top of the result. This replaces
the legacy "Quality Scoring" step that used to live inside Analyze (Step 2).
"""

from typing import Any, List, Optional, Tuple, Union
import logging
from pathlib import Path

from saag.analysis.analyzer import QualityAnalyzer
from saag.analysis.models import (
    DetectedProblem,
    ProblemSummary,
    QualityAnalysisResult,
    StructuralAnalysisResult,
)
from saag.analysis.problem_detector import ProblemDetector

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for running the unified Prediction Step: rule-based RM scoring,
    ML/GNN scoring, anti-pattern detection, and explanation generation.

    When a GNN checkpoint is available and ``prefer_gnn=True`` (default),
    :meth:`predict_quality_with_gnn` returns GNN predictions.  The rule-based
    RM path is always computed and serves as regularisation input during
    training and as a fallback when no checkpoint exists. Anti-patterns and
    explanations are always derived from the deterministic RM scores,
    since they are inherently rule-based (structural thresholds).
    """

    def __init__(
        self,
        use_ahp: bool = False,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        equal_weights: bool = False,
        ahp_shrinkage: float = 0.7,
        gnn_checkpoint_dir: Optional[str] = None,
        prefer_gnn: bool = True,
    ):
        self.use_ahp = use_ahp
        self.normalization_method = normalization_method
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit
        self.equal_weights = equal_weights
        self.ahp_shrinkage = ahp_shrinkage
        self.gnn_checkpoint_dir = gnn_checkpoint_dir
        self.prefer_gnn = prefer_gnn

    # ── Rule-based RM scoring ───────────────────────────────────────────────

    def predict_quality(
        self,
        structural_result: StructuralAnalysisResult,
        run_sensitivity: bool = False,
        sensitivity_perturbations: int = 200,
        sensitivity_noise: float = 0.05,
    ) -> QualityAnalysisResult:
        """Run deterministic RM quality scoring on a structural analysis result."""
        analyzer = QualityAnalyzer(
            normalization_method=self.normalization_method,
            winsorize=self.winsorize,
            winsorize_limit=self.winsorize_limit,
            use_ahp=self.use_ahp,
            equal_weights=self.equal_weights,
            ahp_shrinkage=self.ahp_shrinkage,
        )
        return analyzer.analyze(
            structural_result,
            run_sensitivity=run_sensitivity,
            sensitivity_perturbations=sensitivity_perturbations,
            sensitivity_noise=sensitivity_noise,
        )

    def detect_problems(
        self,
        quality_result: QualityAnalysisResult,
        active_patterns: Optional[List[str]] = None,
    ) -> List[DetectedProblem]:
        """Detect architectural problems from quality results."""
        return ProblemDetector(active_patterns=active_patterns).detect(quality_result)

    def summarize_problems(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """Summarize detected problems."""
        return ProblemDetector().summarize(problems)

    def predict_structural(
        self,
        structural_result: StructuralAnalysisResult,
        graph=None,
        use_qos: bool = True,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
        diagnose: bool = True,
    ) -> Any:
        """Execute training-free topological prediction (Topo or Topo-QoS)."""
        from .structural_predictor import TopoPredictor, TopoQoSPredictor
        from .gnn_service import GNNAnalysisResult, GNNCriticalityScore
        from .data_preparation import extract_structural_metrics_dict

        rm_result = self.predict_quality(structural_result, run_sensitivity=run_sensitivity)
        rm_result.prediction_mode = "rm"
        if diagnose:
            problems, problem_summary, explanation = self._attach_problems_and_explanation(
                rm_result, layer=layer, active_patterns=active_patterns
            )
        else:
            problems, problem_summary, explanation = None, None, None

        predictor = TopoQoSPredictor() if use_qos else TopoPredictor()
        struct_metrics = extract_structural_metrics_dict(structural_result)
        target_graph = graph if graph is not None else getattr(structural_result, "graph", None)
        scores = predictor.predict(target_graph, struct_metrics)

        mode_name = "topo_qos" if use_qos else "topo"
        source_name = "Topo-QoS" if use_qos else "Topo"

        from saag.analysis.classifier import BoxPlotClassifier
        box_clf = BoxPlotClassifier(k_factor=1.5)
        raw_vals = [scores[nid] for nid in scores] if scores else [0.0]
        stats = box_clf.compute_stats(raw_vals)

        node_scores = {}
        for nid, val in scores.items():
            lvl = box_clf.classify_score(val, stats)
            node_scores[nid] = GNNCriticalityScore(
                component=nid,
                composite_score=float(val),
                reliability_score=float(val),
                maintainability_score=float(val),
                source=source_name,
                criticality_level=lvl.name,
            )

        res = GNNAnalysisResult(
            node_scores=node_scores,
            prediction_mode=mode_name,
            layer=layer,
            rm_result=rm_result,
        )
        res.problems = problems
        res.problem_summary = problem_summary
        res.explanation = explanation
        res.failed_patterns = getattr(rm_result, "failed_patterns", [])
        return res

    def predict_dual(
        self,
        structural_result: StructuralAnalysisResult,
        graph,
        simulation_results=None,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
        diagnose: bool = True,
        k: int = 10,
        divergence_threshold: int = 5,
    ) -> Any:
        """Operationalizes JSS Section 8.1: runs HGT-QoS and Topo-QoS concurrently."""
        from .structural_predictor import DualEnginePredictor, TopoQoSPredictor
        from .data_preparation import extract_structural_metrics_dict

        # 1. Run GNN prediction
        gnn_res = self.predict_quality_with_gnn(
            structural_result=structural_result,
            graph=graph,
            simulation_results=simulation_results,
            layer=layer,
            active_patterns=active_patterns,
            run_sensitivity=run_sensitivity,
            diagnose=diagnose,
            predictor_mode="gnn",
        )

        # 2. Extract GNN scores
        if hasattr(gnn_res, "node_scores") and gnn_res.node_scores:
            gnn_scores = {nid: float(ns.composite_score) for nid, ns in gnn_res.node_scores.items()}
        else:
            gnn_scores = {c.id: float(c.scores.overall) for c in getattr(gnn_res, "components", [])}

        # 3. Run Dual-Engine evaluation
        dual_predictor = DualEnginePredictor(
            topo_predictor=TopoQoSPredictor(),
            divergence_threshold=divergence_threshold,
        )
        struct_metrics = extract_structural_metrics_dict(structural_result)
        dual_res = dual_predictor.evaluate_dual(
            gnn_scores=gnn_scores,
            graph_or_flow=graph,
            structural_metrics=struct_metrics,
            k=k,
        )

        gnn_res.prediction_mode = "dual"
        gnn_res.dual_result = dual_res
        return gnn_res

    def predict_quality_with_gnn(
        self,
        structural_result: StructuralAnalysisResult,
        graph,
        simulation_results=None,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
        diagnose: bool = True,
        predictor_mode: str = "gnn",
    ) -> Union[QualityAnalysisResult, Any]:
        """Return predictions according to predictor_mode ('gnn', 'rm', 'topo', 'topo_qos', 'dual').

        RM scores are always computed — they serve as the consistency
        regularisation target for the GNN and as a fallback when no
        checkpoint is present. When ``diagnose`` is True (default), Step 4's
        anti-pattern detection and explanation are also run on the RM scores
        and attached to whichever result is ultimately returned.
        """
        if predictor_mode == "topo":
            return self.predict_structural(
                structural_result, graph=graph, use_qos=False,
                layer=layer, active_patterns=active_patterns,
                run_sensitivity=run_sensitivity, diagnose=diagnose,
            )
        elif predictor_mode == "topo_qos":
            return self.predict_structural(
                structural_result, graph=graph, use_qos=True,
                layer=layer, active_patterns=active_patterns,
                run_sensitivity=run_sensitivity, diagnose=diagnose,
            )
        elif predictor_mode == "dual":
            return self.predict_dual(
                structural_result, graph=graph, simulation_results=simulation_results,
                layer=layer, active_patterns=active_patterns,
                run_sensitivity=run_sensitivity, diagnose=diagnose,
            )

        rm_result = self.predict_quality(structural_result, run_sensitivity=run_sensitivity)
        rm_result.prediction_mode = "rm"
        if diagnose:
            problems, problem_summary, explanation = self._attach_problems_and_explanation(
                rm_result, layer=layer, active_patterns=active_patterns
            )
        else:
            problems, problem_summary, explanation = None, None, None

        if predictor_mode == "rm" or not self.prefer_gnn:
            logger.debug("RM scoring requested (mode='rm'); returning RM scores.")
            return rm_result

        if not (self.gnn_checkpoint_dir and self._has_checkpoint(self.gnn_checkpoint_dir)):
            logger.warning(
                "GNN prediction requested but no checkpoint found at '%s'; "
                "returning RM scores instead.", self.gnn_checkpoint_dir
            )
            return rm_result

        try:
            from .gnn_service import GNNService
            from .data_preparation import (
                extract_structural_metrics_dict,
                extract_rm_scores_dict,
            )
            gnn_svc = GNNService.from_checkpoint(self.gnn_checkpoint_dir, graph=graph)
            gnn_result = gnn_svc.predict(
                graph=graph,
                structural_metrics=extract_structural_metrics_dict(structural_result),
                rm_scores=extract_rm_scores_dict(rm_result),
                eval_labels=simulation_results,
                mode="gnn",
            )
        except Exception:
            logger.warning("GNN inference failed; falling back to RM scores.", exc_info=True)
            return rm_result

        gnn_result.problems = problems
        gnn_result.problem_summary = problem_summary
        gnn_result.explanation = explanation
        gnn_result.failed_patterns = getattr(rm_result, "failed_patterns", [])
        gnn_result.rm_result = rm_result
        return gnn_result

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _has_checkpoint(directory: str) -> bool:
        """Return True if directory contains a loadable GNN checkpoint."""
        p = Path(directory)
        return (p / "service_config.json").exists() and (
            (p / "node_model.pt").exists() or (p / "best_model.pt").exists()
        )

    def _attach_problems_and_explanation(
        self,
        quality_result: QualityAnalysisResult,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
    ) -> Tuple[List[DetectedProblem], ProblemSummary, Any]:
        """Run anti-pattern detection and explanation on RM scores, and attach
        them to *quality_result*. Returns them so a GNN result can reuse them."""
        from saag.analysis.antipattern_detector import AntiPatternDetector
        from saag.analysis.smells import AntiPatternReport
        from saag.explanation.engine import ExplanationEngine

        # An empty list is an explicit opt-out, distinct from None ("run the whole
        # catalogue"). AntiPatternDetector treats both as falsy and would run
        # everything, so the caller's opt-out has to be honoured here.
        failed_patterns: List[str] = []
        if active_patterns is not None and len(active_patterns) == 0:
            problems = []
        else:
            detector = AntiPatternDetector(active_patterns=active_patterns)
            problems = detector.detect(quality_result, layer=layer)
            failed_patterns = detector.failed_patterns
        problem_summary = self.summarize_problems(problems)
        smell_report = AntiPatternReport(
            problems=problems,
            summary=problem_summary.to_dict()
            if hasattr(problem_summary, "to_dict")
            else problem_summary,
        )
        explanation = ExplanationEngine().explain_system(quality_result, smell_report)

        quality_result.problems = problems
        quality_result.problem_summary = problem_summary
        quality_result.explanation = explanation
        quality_result.prediction_mode = "rm"
        quality_result.failed_patterns = failed_patterns
        return problems, problem_summary, explanation
