"""
Visualization Data Collector

Collects and aggregates data from analysis, simulation, and validation
services into LayerData structures for dashboard generation.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

from .models import LayerData, ComponentDetail, LAYER_DEFINITIONS
from src.analysis.antipattern_detector import AntiPatternDetector


class LayerDataCollector:
    """
    Collects and aggregates data from various services for visualization.
    """

    def __init__(
        self,
        analysis_service: Any,
        prediction_service: Any,
        simulation_service: Any,
        validation_service: Any,
        repository: Any,
    ):
        self.analysis_service = analysis_service
        self.prediction_service = prediction_service
        self.simulation_service = simulation_service
        self.validation_service = validation_service
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def collect_layer_data(
        self, layer: str, include_validation: bool = True, antipatterns_file: Optional[str] = None
    ) -> LayerData:
        """
        Collect all data for a specific layer.
        """
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}")

        layer_def = LAYER_DEFINITIONS[layer]
        data = LayerData(layer=layer, name=layer_def["name"])

        # 1. Structural & Quality Analysis
        analysis = self._collect_analysis_data(data, layer)

        # 2. Simulation Results
        self._collect_simulation_data(data, layer)

        # 3. Validation Results
        if include_validation:
            self._collect_validation_data(data, layer)

        # 4. Anti-Patterns
        self._collect_antipattern_data(data, layer, antipatterns_file)

        # 5. Build enriched network data
        if analysis:
            self._build_network_data(data, analysis)

        return data

    def _collect_component_details(self, result: Any, scatter_data: List[Tuple], smells: Dict[str, List[str]] = None) -> List[ComponentDetail]:
        """Collect detailed component information."""
        details = []
        smells = smells or {}
        
        def safe_float(val: Any) -> float:
            try:
                # Handle MagicMocks in tests
                if hasattr(val, '_mock_return_value') and not isinstance(val, (int, float)):
                    return 0.0
                return float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        for comp in result.quality.components:
            # Extract basic quality
            overall = safe_float(comp.scores.overall)
            impact = safe_float(getattr(comp, 'impact', 0.0))
            
            detail = ComponentDetail(
                id=comp.id,
                name=getattr(comp.structural, 'name', comp.id),
                type=comp.type,
                reliability=safe_float(comp.scores.reliability),
                maintainability=safe_float(comp.scores.maintainability),
                availability=safe_float(comp.scores.availability),
                vulnerability=safe_float(comp.scores.vulnerability),
                overall=overall,
                level=getattr(comp.levels.overall, 'name', 'MINIMAL'),
                impact=impact,
                anti_patterns=smells.get(comp.id, [])
            )
            
            # Enrich with MPCI, FOC, SPOF if available in analysis
            detail.mpci = safe_float(getattr(comp.metrics, 'mpci', 0.0)) # Corrected from comp.scores to comp.metrics
            detail.foc = safe_float(getattr(comp.metrics, 'foc', 0.0)) # Corrected from comp.scores to comp.metrics
            detail.spof = bool(getattr(comp.structural, 'is_articulation_point', False)) # Corrected from is_spof to is_articulation_point
            
            details.append(detail)
            
            # Populate scatter data if Q > 0 and it has impact
            if overall > 0:
                scatter_data.append((detail.id, overall, impact, detail.level))
                
        return details

    def _collect_analysis_data(self, data: LayerData, layer: str) -> None:
        """
        Run analysis and populate graph statistics, quality scores,
        and classification counts.
        """
        try:
            analysis = self.analysis_service.analyze_layer(layer)
            prediction = analysis.quality
            
            # Structural Stats
            data.nodes = analysis.structural.graph_summary.nodes
            data.edges = analysis.structural.graph_summary.edges
            data.density = analysis.structural.graph_summary.density
            data.connected_components = analysis.structural.graph_summary.num_components
            data.component_counts = analysis.structural.graph_summary.node_types or {}

            # Quality Classification Counts
            data.spof_count = analysis.structural.graph_summary.num_articulation_points
            data.problems_count = len(analysis.problems)
            
            # Carry over the system explanation
            if hasattr(analysis, "explanation") and analysis.explanation:
                data.explanation = analysis.explanation.to_dict()

            # Build component details with RMAV breakdown            # 3. Quality Metrics & Ranking
            sorted_comps = sorted(
                prediction.components,
                key=lambda x: _safe_float(x.scores.overall),
                reverse=True,
            )

            # Reset counts
            data.critical_count = 0
            data.high_count = 0
            data.medium_count = 0
            data.low_count = 0
            data.minimal_count = 0
            
            # New: Full component details with RMAV breakdown
            data.component_details = []
            data.scatter_data = []
            
            for c in sorted_comps:
                level = (
                    c.levels.overall.name
                    if hasattr(c.levels.overall, "name")
                    else str(c.levels.overall)
                )
                
                # Update counts manually
                if level == "CRITICAL": data.critical_count += 1
                elif level == "HIGH": data.high_count += 1
                elif level == "MEDIUM": data.medium_count += 1
                elif level == "LOW": data.low_count += 1
                elif level == "MINIMAL": data.minimal_count += 1
                
                # Map component explanation if present in system report
                c_explanation = None
                if data.explanation and "component_explanations" in data.explanation:
                    for ce in data.explanation["component_explanations"]:
                        if ce["component_id"] == c.id:
                            c_explanation = ce
                            break

                overall = _safe_float(c.scores.overall)
                detail = ComponentDetail(
                    id=c.id,
                    name=c.structural.name if hasattr(c, "structural") else c.id,
                    type=c.type,
                    reliability=_safe_float(c.scores.reliability),
                    maintainability=_safe_float(c.scores.maintainability),
                    availability=_safe_float(c.scores.availability),
                    vulnerability=_safe_float(c.scores.vulnerability),
                    overall=overall,
                    level=level,
                    mpci=_safe_float(getattr(c.scores, "mpci", 0.0)),
                    foc=_safe_float(getattr(c.scores, "foc", 0.0)),
                    spof=bool(getattr(c.structural, "is_articulation_point", False)),
                    explanation=c_explanation
                )
                data.component_details.append(detail)
                
                # Populate scatter data (Predicted Q vs Ground Truth I)
                data.scatter_data.append([c.id, overall, 0.0, level])

            data.component_names = {
                c.id: c.structural.name
                for c in prediction.components
                if hasattr(c, "structural")
            }
            data.rcm_order = analysis.structural.rcm_order
            
            # Legacy fields for backward compat
            data.top_components = [
                {"id": d.id, "type": d.type, "score": d.overall, "level": d.level}
                for d in data.component_details[:10]
            ]
            
            # Return enriched analysis-like object for downstream (used by _build_network_data)
            # We wrap it back into a structure that looks like LayerAnalysisResult for compatibility
            from src.analysis.models import LayerAnalysisResult
            return analysis

        except Exception as e:
            self.logger.error(f"Analysis failed for layer {layer}: {e}")
            self.logger.exception("Analysis failure details")

    def _collect_simulation_data(self, data: LayerData, layer: str) -> None:
        """Run simulation and populate impact metrics."""
        try:
            layer_metrics = self.simulation_service.analyze_layer(layer)
            data.event_throughput = layer_metrics.event_throughput
            data.event_delivery_rate = layer_metrics.event_delivery_rate
            data.avg_impact = layer_metrics.avg_reachability_loss
            data.max_impact = layer_metrics.max_impact

            # Enrich component details with impact scores
            if hasattr(layer_metrics, "component_impacts"):
                impact_map = {
                    c_id: impact
                    for c_id, impact in layer_metrics.component_impacts.items()
                }
                for detail in data.component_details:
                    if detail.id in impact_map:
                        detail.impact = impact_map[detail.id]
                
                # Update scatter data with actual impact (Ground Truth)
                for i, scatter_item in enumerate(data.scatter_data):
                    c_id = scatter_item[0]
                    if c_id in impact_map:
                        data.scatter_data[i][2] = impact_map[c_id]

        except Exception as e:
            self.logger.error(f"Simulation failed for layer {layer}: {e}")

    def _collect_validation_data(self, data: LayerData, layer: str) -> None:
        """Run validation and populate correlation/classification metrics."""
        try:
            # Note: ValidationService in new structure returns PipelineResult which has .layers (dict)
            pipeline_res = self.validation_service.validate_layers(layers=[layer])
            val_result = pipeline_res.layers.get(layer)
            
            if val_result:
                data.spearman = val_result.spearman
                data.f1_score = val_result.f1_score
                data.precision = val_result.precision
                data.recall = val_result.recall
                data.validation_passed = val_result.passed

                # Top-K overlap if available
                if hasattr(val_result, "top_5_overlap"):
                    data.top5_overlap = val_result.top_5_overlap
                elif hasattr(val_result, "top5_overlap"):
                    data.top5_overlap = val_result.top5_overlap
                if hasattr(val_result, "top_10_overlap"):
                    data.top10_overlap = val_result.top_10_overlap
                elif hasattr(val_result, "top10_overlap"):
                    data.top10_overlap = val_result.top10_overlap

                # Per-dimension rhos
                data.reliability_spearman = getattr(val_result, "reliability_spearman", 0.0)
                data.maintainability_spearman = getattr(val_result, "maintainability_spearman", 0.0)
                data.availability_spearman = getattr(val_result, "availability_spearman", 0.0)
                data.vulnerability_spearman = getattr(val_result, "vulnerability_spearman", 0.0)
                data.composite_spearman = getattr(val_result, "composite_spearman", 0.0)
                data.predictive_gain = getattr(val_result, "predictive_gain", 0.0)
                
                # Gates
                data.gates = getattr(val_result, "gates", {})

                # Full dimensional scatter data
                if hasattr(val_result, "dimensional_scatter"):
                    data.reliability_scatter = val_result.dimensional_scatter.get("reliability", [])
                    data.maintainability_scatter = val_result.dimensional_scatter.get("maintainability", [])
                    data.availability_scatter = val_result.dimensional_scatter.get("availability", [])
                    data.vulnerability_scatter = val_result.dimensional_scatter.get("vulnerability", [])
                    # The composite scatter is also in dimensional_scatter["composite"]
                    if "composite" in val_result.dimensional_scatter:
                        data.scatter_data = val_result.dimensional_scatter["composite"]
                
                # Confidence intervals
                if hasattr(val_result, "confidence_intervals"):
                    data.reliability_ci = val_result.confidence_intervals.get("reliability")
                    data.maintainability_ci = val_result.confidence_intervals.get("maintainability")
                    data.availability_ci = val_result.confidence_intervals.get("availability")
                    data.vulnerability_ci = val_result.confidence_intervals.get("vulnerability")
                    data.composite_ci = val_result.confidence_intervals.get("composite")

        except Exception as e:
            self.logger.error(f"Validation failed for layer {layer}: {e}")

    def _collect_antipattern_data(self, data: LayerData, layer: str, antipatterns_file: Optional[str] = None) -> None:
        """
        Run anti-pattern detection or load from file.
        """
        import json
        from pathlib import Path

        try:
            smells = []
            if antipatterns_file and Path(antipatterns_file).exists():
                self.logger.info(f"Loading anti-patterns from {antipatterns_file}")
                with open(antipatterns_file, "r") as f:
                    report_dict = json.load(f)
                    
                # Handle SmellReport structure or direct list
                raw_smells = report_dict.get("smells", report_dict) if isinstance(report_dict, dict) else report_dict
                if not isinstance(raw_smells, list):
                    self.logger.warning(f"Invalid anti-pattern data format in {antipatterns_file}")
                    raw_smells = []
                
                # Filter for current layer
                for s in raw_smells:
                    if s.get("layer") == layer:
                        smells.append(s)
                
                data.anti_patterns = smells
            else:
                # Real-time detection fallback
                # Re-running analysis to get the full result for detector
                analysis_result = self.analysis_service.analyze_layer(layer)
                detector = AntiPatternDetector()
                detected = detector.detect(analysis_result.quality, layer)
                smells = [s.to_dict() for s in detected]
                data.anti_patterns = smells
            data.problems_count = len(smells)
            
            # Map smells back to components in details
            smell_map: Dict[str, List[str]] = {}
            for s in smells:
                # s can be a dict (from JSON) or DetectedSmell object (from detector)
                pattern_id = s.get("pattern_id") if isinstance(s, dict) else s.pattern_id
                comp_ids = s.get("component_ids", []) if isinstance(s, dict) else s.component_ids
                
                for c_id in comp_ids:
                    if c_id not in smell_map:
                        smell_map[c_id] = []
                    smell_map[c_id].append(pattern_id)
            
            for detail in data.component_details:
                if detail.id in smell_map:
                    detail.anti_patterns = smell_map[detail.id]
                    
        except Exception as e:
            self.logger.error(f"Anti-pattern detection failed for layer {layer}: {e}")

    def _build_scatter_data(self, data: LayerData) -> None:
        """
        Build Q(v) vs I(v) scatter plot data from component details.
        """
        data.scatter_data = []
        for detail in data.component_details:
            q_score = detail.overall
            i_score = detail.impact
            if q_score == q_score and q_score > 0:  # NaN check
                data.scatter_data.append(
                    (detail.id, q_score, i_score, detail.level)
                )

    def _build_network_data(self, data: LayerData, analysis: Any) -> None:
        """Build network nodes and edges for interactive visualization."""
        data.network_nodes = []
        
        # Map component ID to detail for tooltip enrichment
        detail_map = {d.id: d for d in data.component_details}
        
        for c in analysis.quality.components:
            score = c.scores.overall if c.scores.overall is not None else 0.0
            if score != score:  # NaN check
                score = 0.0
            value = score * 30 + 10
            level = (
                c.levels.overall.name
                if hasattr(c.levels.overall, "name")
                else str(c.levels.overall)
            )

            # Build rich tooltip
            detail = detail_map.get(c.id)
            title_parts = [
                f"<b>{c.id}</b>",
                f"Name: {c.structural.name}",
                f"Type: {c.type}",
                f"Score: {score:.3f} ({level})",
            ]
            
            if detail:
                title_parts.append(f"MPCI: {detail.mpci:.3f}")
                title_parts.append(f"FOC: {detail.foc:.3f}")
                if detail.impact > 0:
                    title_parts.append(f"Impact: {detail.impact:.3f}")
                    title_parts.append(f"Cascade: {detail.cascade_depth} layers")
                if detail.anti_patterns:
                    title_parts.append(f"Anti-Patterns: {', '.join(detail.anti_patterns)}")

            data.network_nodes.append(
                {
                    "id": c.id,
                    "label": f"{c.id}\n({c.structural.name})",
                    "group": level,
                    "type": c.type,
                    "shape": c.structural.vis_shape if hasattr(c.structural, "vis_shape") else "ellipse",
                    "level": level,
                    "value": value,
                    "title": "<br>".join(title_parts),
                }
            )

        data.network_edges = []
        if hasattr(analysis.quality, "edges"):
            for edge in analysis.quality.edges:
                weight = edge.weight if hasattr(edge, "weight") else 1.0
                if weight != weight:  # NaN
                    weight = 1.0
                dep_type = (
                    edge.dependency_type
                    if hasattr(edge, "dependency_type")
                    else "DEPENDS_ON"
                )
                data.network_edges.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "weight": weight,
                        "relation_type": "DEPENDS_ON",
                        "dependency_type": dep_type,
                    }
                )

        try:
            # Expecting repository to have get_raw_edges or similar
            if hasattr(self.repository, "get_raw_edges"):
                raw_edges = self.repository.get_raw_edges(data.layer)
                if raw_edges:
                    for edge in raw_edges:
                        data.network_edges.append(
                            {
                                "source": edge.get("source", ""),
                                "target": edge.get("target", ""),
                                "weight": edge.get("weight", 1.0),
                                "relation_type": edge.get("type", "STRUCTURAL"),
                                "dependency_type": edge.get("type", "structural"),
                            }
                        )
        except Exception:
            pass


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 for NaN or None."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        return f if f == f else 0.0  # NaN check
    except (TypeError, ValueError):
        return 0.0
