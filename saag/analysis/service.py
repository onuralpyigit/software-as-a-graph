from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .structural_analyzer import StructuralAnalyzer
from .models import CrossLayerInsight, MultiLayerAnalysisResult, LayerAnalysisResult
from saag.core.layers import AnalysisLayer, get_layer_definition
from saag.core.ports.graph_repository import IGraphRepository

_PREDICT_KWARG_KEYS = ["use_ahp", "normalization_method", "winsorize", "winsorize_limit", "equal_weights", "ahp_shrinkage"]


class AnalysisService:
    """
    Service for running structural graph analysis (Step 2).
    Orchestrates StructuralAnalyzer only — RMAV/GNN scoring, anti-pattern
    detection, and explanations are produced by the Predict stage (Step 3),
    see saag.prediction.service.PredictionService.
    """

    def __init__(
        self,
        repository: IGraphRepository,
        **kwargs
    ):
        self.repository = repository
        self._analysis_kwargs = kwargs

    def _predict_layer(self, layer: str) -> LayerAnalysisResult:
        """Run Step 2 (structural) then Step 3 (RMAV + anti-patterns) for one layer.

        Convenience helper for methods that need criticality data
        (cross-layer insights, critical-component/edge lookups).
        """
        from saag.prediction.service import PredictionService

        layer_result = self.analyze_layer(layer)
        pred_kwargs = {k: v for k, v in self._analysis_kwargs.items() if k in _PREDICT_KWARG_KEYS}
        predictor = PredictionService(**pred_kwargs)
        quality_result = predictor.predict(layer_result.structural, layer=layer)
        layer_result.quality = quality_result
        layer_result.problems = quality_result.problems
        layer_result.problem_summary = quality_result.problem_summary
        layer_result.explanation = quality_result.explanation
        return layer_result

    def analyze_all_layers(self) -> MultiLayerAnalysisResult:
        """Analyze all primary graph layers and compute cross-layer insights.

        Spans Step 2 (structural) and Step 3 (Predict), since cross-layer
        insights require RMAV criticality levels.
        """
        layers = ["app", "infra", "mw", "system"]
        results = {}
        for layer in layers:
            results[layer] = self._predict_layer(layer)

        insights = self._compute_cross_layer_insights(results)

        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=insights,
        )

    # ------------------------------------------------------------------
    # Cross-layer insight computation
    # ------------------------------------------------------------------

    def _compute_cross_layer_insights(
        self, results: Dict[str, LayerAnalysisResult]
    ) -> List[CrossLayerInsight]:
        """
        Derive insights that span multiple layers.

        Three insight types are produced:

        1. compound_critical  — component is CRITICAL or HIGH in ≥2 distinct layers.
        2. systemic_spof      — component is an articulation point in ≥2 distinct layers.
        3. layer_concentration — a layer where >30 % of analysed components are CRITICAL.
        """
        from saag.core.criticality import CriticalityLevel

        insights: List[CrossLayerInsight] = []

        # -------------------------------------------------------------------
        # 1 & 2: Per-component cross-layer signals
        # -------------------------------------------------------------------
        # Map: component_id → {layer: ComponentQuality}
        comp_by_layer: Dict[str, Dict[str, Any]] = {}

        for layer_name, layer_result in results.items():
            for cq in layer_result.quality.components:
                comp_by_layer.setdefault(cq.id, {})[layer_name] = cq

        for comp_id, layer_map in comp_by_layer.items():
            if len(layer_map) < 2:
                continue  # only appears in one layer — no cross-layer signal

            # Collect any layer where level is CRITICAL or HIGH
            high_layers = [
                lname for lname, cq in layer_map.items()
                if cq.levels.overall >= CriticalityLevel.HIGH
            ]
            # Collect any layer where component is an articulation point
            spof_layers = [
                lname for lname, cq in layer_map.items()
                if cq.structural.is_articulation_point
            ]

            # Derive name from any available entry
            sample_cq = next(iter(layer_map.values()))
            comp_name = getattr(sample_cq.structural, "name", comp_id)

            if len(high_layers) >= 2:
                severity = (
                    "CRITICAL"
                    if any(
                        layer_map[l].levels.overall == CriticalityLevel.CRITICAL
                        for l in high_layers
                    )
                    else "HIGH"
                )
                insights.append(CrossLayerInsight(
                    component_id=comp_id,
                    csc_name=comp_name,
                    insight_type="compound_critical",
                    layers_affected=sorted(high_layers),
                    severity=severity,
                    description=(
                        f"{comp_name} is classified {severity} in "
                        f"{len(high_layers)} layers ({', '.join(sorted(high_layers))}), "
                        "indicating compound risk that spans architectural boundaries."
                    ),
                ))

            if len(spof_layers) >= 2:
                insights.append(CrossLayerInsight(
                    component_id=comp_id,
                    csc_name=comp_name,
                    insight_type="systemic_spof",
                    layers_affected=sorted(spof_layers),
                    severity="CRITICAL",
                    description=(
                        f"{comp_name} is a structural articulation point in "
                        f"{len(spof_layers)} layers ({', '.join(sorted(spof_layers))}). "
                        "Its failure would disconnect subgraphs at multiple architectural levels."
                    ),
                ))

        # -------------------------------------------------------------------
        # 3: Layer-level concentration
        # -------------------------------------------------------------------
        for layer_name, layer_result in results.items():
            components = layer_result.quality.components
            if not components:
                continue
            critical_count = sum(
                1 for cq in components
                if cq.levels.overall == CriticalityLevel.CRITICAL
            )
            fraction = critical_count / len(components)
            if fraction > 0.30:
                insights.append(CrossLayerInsight(
                    component_id="",
                    csc_name="",
                    insight_type="layer_concentration",
                    layers_affected=[layer_name],
                    severity="HIGH",
                    description=(
                        f"Layer '{layer_name}' has {critical_count}/{len(components)} "
                        f"({fraction:.0%}) components classified as CRITICAL — "
                        "high systemic risk concentration in this architectural tier."
                    ),
                ))

        # Sort: CRITICAL first, then by number of affected layers descending
        def _sort_key(i: CrossLayerInsight) -> tuple:
            sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
            return (sev_order.get(i.severity, 9), -len(i.layers_affected))

        insights.sort(key=_sort_key)
        return insights

    def analyze_system(self, layer: str = "system", context: Optional[str] = None) -> Dict[str, Any]:
        """Run analysis on a specific layer or the full system and return a dict."""
        res = self.analyze_layer(layer)
        return res.to_dict()

    def analyze_layer(self, layer: str) -> LayerAnalysisResult:
        """
        Run structural analysis (Step 2) on a specific layer and return the
        LayerAnalysisResult with only ``structural`` populated.

        Pre-analysis: DEPENDS_ON relationships are derived from the structural
        graph before structural analysis begins. RMAV/GNN quality scoring,
        anti-pattern detection, and explanations are produced separately by
        the Predict stage (Step 3) — see saag.prediction.service.PredictionService.
        """
        try:
            layer_enum = AnalysisLayer.from_string(layer)
        except ValueError:
            layer_enum = AnalysisLayer.SYSTEM

        layer_def = get_layer_definition(layer_enum)

        # Pre-analysis stage: derive DEPENDS_ON edges and finalise their weights.
        self.repository.derive_dependencies()

        graph_data = self.repository.get_graph_data()
        structural_analyzer = StructuralAnalyzer()
        struct_result = structural_analyzer.analyze(graph_data, layer=layer_enum)

        return LayerAnalysisResult(
            layer=layer_enum.value,
            layer_name=layer_def.name,
            description=layer_def.description,
            structural=struct_result,
        )

    def analyze_by_type(self, component_type: str) -> Dict[str, Any]:
        """Run analysis+prediction and filter by component type."""
        # Reuse the Predict-enriched result then filter
        result = self._predict_layer("system")
        
        # Filter components
        filtered_components = [c for c in result.components if c.type == component_type]
        
        # Calculate averages
        if filtered_components:
            avg_reliability = sum(c.scores.reliability for c in filtered_components) / len(filtered_components)
            avg_maintainability = sum(c.scores.maintainability for c in filtered_components) / len(filtered_components)
            avg_availability = sum(c.scores.availability for c in filtered_components) / len(filtered_components)
            avg_overall = sum(c.scores.overall for c in filtered_components) / len(filtered_components)
        else:
            avg_reliability = avg_maintainability = avg_availability = avg_overall = 0.0
            
        return {
            "component_type": component_type,
            "count": len(filtered_components),
            "average_scores": {
                "reliability": avg_reliability,
                "maintainability": avg_maintainability,
                "availability": avg_availability,
                "overall": avg_overall
            },
            "components": [
                {
                    "id": c.id,
                    "scores": {
                        "reliability": c.scores.reliability,
                        "maintainability": c.scores.maintainability,
                        "availability": c.scores.availability,
                        "overall": c.scores.overall
                    },
                    "issues": [p.description for p in getattr(result, "problems", []) if p.entity_id == c.id]
                }
                for c in filtered_components
            ]
        }

    def get_critical_components(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most critical components based on analysis+prediction."""
        result = self._predict_layer("system")
        
        components = sorted(
            result.components,
            key=lambda c: c.scores.overall,
            reverse=True
        )[:limit]
        
        return [
            {
                "id": c.id,
                "type": c.type,
                "criticality_level": c.levels.overall.value,
                "criticality_levels": {
                    "reliability": c.levels.reliability.value,
                    "maintainability": c.levels.maintainability.value,
                    "availability": c.levels.availability.value,
                    "security": c.levels.security.value,
                    "overall": c.levels.overall.value
                },
                "overall_score": c.scores.overall,
                "scores": {
                    "reliability": c.scores.reliability,
                    "maintainability": c.scores.maintainability,
                    "availability": c.scores.availability,
                    "security": c.scores.security
                }
            }
            for c in components
        ]

    def get_critical_edges(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most critical edges based on analysis+prediction."""
        result = self._predict_layer("system")
        
        edges = sorted(
            result.edges,
            key=lambda e: e.scores.overall,
            reverse=True
        )[:limit]
        
        return [
            {
                "source": e.source,
                "target": e.target,
                "type": e.dependency_type,
                "criticality_level": e.level.value,
                "overall_score": e.scores.overall,
                "scores": {
                    "reliability": e.scores.reliability,
                    "maintainability": e.scores.maintainability,
                    "availability": e.scores.availability,
                    "security": e.scores.security
                }
            }
            for e in edges
        ]
    def export_results(self, results: Any, output_path: str) -> None:
        """Export analysis results to a JSON file."""
        import json
        from saag.core.utils import CustomJSONEncoder
        
        # Use CustomJSONEncoder if available to handle any non-serializable objects
        with open(output_path, 'w') as f:
            if hasattr(results, 'to_dict'):
                json.dump(results.to_dict(), f, indent=4)
            else:
                json.dump(results, f, indent=4)
