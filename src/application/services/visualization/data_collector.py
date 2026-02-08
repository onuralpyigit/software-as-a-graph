"""
Visualization Data Collector

Collects and aggregates data from analysis, simulation, and validation
services into LayerData structures for dashboard generation.

Implements the data pipeline specified in Definition 9 of visualization.md:
    S_l = (G_l, M_l, Q_l, F_l, V_l) for each layer l
"""
import logging
from typing import Dict, List, Any, Optional

from src.application.services.visualization.layer_data import (
    LayerData, ComponentDetail, LAYER_DEFINITIONS,
)
from src.application.services.analysis_service import AnalysisService
from src.application.services.simulation_service import SimulationService
from src.application.services.validation_service import ValidationService


class LayerDataCollector:
    """
    Collects and aggregates data from various services for visualization.

    Each collect method maps to a specific methodology step:
        _collect_analysis_data  → Steps 1-3 (graph, metrics, quality)
        _collect_simulation_data → Step 4 (failure impact)
        _collect_validation_data → Step 5 (statistical validation)
    """

    def __init__(
        self,
        analysis_service: AnalysisService,
        simulation_service: SimulationService,
        validation_service: ValidationService,
        repository: Any,  # IGraphRepository
    ):
        self.analysis_service = analysis_service
        self.simulation_service = simulation_service
        self.validation_service = validation_service
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def collect_layer_data(
        self, layer: str, include_validation: bool = True
    ) -> LayerData:
        """
        Collect all data for a specific layer.

        This is the main entry point for the data collection pipeline.
        It orchestrates calls to analysis, simulation, and validation
        services, populating a single LayerData instance.
        """
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}")

        layer_def = LAYER_DEFINITIONS[layer]
        data = LayerData(layer=layer, name=layer_def["name"])

        # 1. Structural & Quality Analysis (Steps 1-3)
        self._collect_analysis_data(data, layer)

        # 2. Simulation Results (Step 4)
        self._collect_simulation_data(data, layer)

        # 3. Validation Results (Step 5)
        if include_validation:
            self._collect_validation_data(data, layer)

        # 4. Build scatter plot data (Steps 3+4 combined)
        self._build_scatter_data(data)

        return data

    # ─── Step 1-3: Analysis Data ─────────────────────────────────────────

    def _collect_analysis_data(self, data: LayerData, layer: str) -> None:
        """
        Run analysis and populate graph statistics, quality scores,
        and classification counts.
        """
        try:
            analysis = self.analysis_service.analyze_layer(layer)

            # Step 1: Structural Stats
            data.nodes = analysis.structural.graph_summary.nodes
            data.edges = analysis.structural.graph_summary.edges
            data.density = analysis.structural.graph_summary.density
            data.connected_components = analysis.structural.graph_summary.num_components
            data.component_counts = analysis.structural.graph_summary.node_types or {}

            # Step 3: Quality Classification Counts
            for comp in analysis.quality.components:
                level = str(comp.levels.overall)
                if hasattr(comp.levels.overall, "name"):
                    level = comp.levels.overall.name

                if level == "CRITICAL":
                    data.critical_count += 1
                elif level == "HIGH":
                    data.high_count += 1
                elif level == "MEDIUM":
                    data.medium_count += 1
                elif level == "LOW":
                    data.low_count += 1
                elif level == "MINIMAL":
                    data.minimal_count += 1

            data.spof_count = analysis.structural.graph_summary.num_articulation_points
            data.problems_count = len(analysis.problems)

            # Steps 2+3: Build component details with RMAV breakdown
            sorted_comps = sorted(
                analysis.quality.components,
                key=lambda c: c.scores.overall if c.scores.overall == c.scores.overall else 0,
                reverse=True,
            )

            # Legacy top_components (backward compat)
            data.top_components = [
                {
                    "id": c.id,
                    "type": c.type,
                    "score": c.scores.overall,
                    "level": (
                        c.levels.overall.name
                        if hasattr(c.levels.overall, "name")
                        else str(c.levels.overall)
                    ),
                }
                for c in sorted_comps[:10]
            ]

            # New: Full component details with RMAV breakdown
            data.component_details = []
            for c in sorted_comps:
                level = (
                    c.levels.overall.name
                    if hasattr(c.levels.overall, "name")
                    else str(c.levels.overall)
                )
                detail = ComponentDetail(
                    id=c.id,
                    name=c.structural.name if hasattr(c, "structural") else c.id,
                    type=c.type,
                    reliability=_safe_float(c.scores.reliability),
                    maintainability=_safe_float(c.scores.maintainability),
                    availability=_safe_float(c.scores.availability),
                    vulnerability=_safe_float(c.scores.vulnerability),
                    overall=_safe_float(c.scores.overall),
                    level=level,
                )
                data.component_details.append(detail)

            data.component_names = {
                c.id: c.structural.name
                for c in analysis.quality.components
                if hasattr(c, "structural")
            }

            # Step 1: Network Graph Data
            self._build_network_data(data, analysis)

        except Exception as e:
            self.logger.error(f"Analysis failed for layer {layer}: {e}")
            self.logger.exception("Analysis failure details")

    # ─── Step 4: Simulation Data ─────────────────────────────────────────

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

        except Exception as e:
            self.logger.error(f"Simulation failed for layer {layer}: {e}")

    # ─── Step 5: Validation Data ─────────────────────────────────────────

    def _collect_validation_data(self, data: LayerData, layer: str) -> None:
        """Run validation and populate correlation/classification metrics."""
        try:
            val_result = self.validation_service.validate_layers(
                layers=[layer]
            ).layers.get(layer)
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

        except Exception as e:
            self.logger.error(f"Validation failed for layer {layer}: {e}")

    # ─── Combined: Scatter Plot Data (Steps 3+4) ────────────────────────

    def _build_scatter_data(self, data: LayerData) -> None:
        """
        Build Q(v) vs I(v) scatter plot data from component details.

        Each point: (component_id, Q(v), I(v), level)
        Only includes components that have both quality and impact scores.
        """
        data.scatter_data = []
        for detail in data.component_details:
            q_score = detail.overall
            i_score = detail.impact
            # Include all components (even those with zero impact)
            if q_score == q_score and q_score > 0:  # NaN check
                data.scatter_data.append(
                    (detail.id, q_score, i_score, detail.level)
                )

    # ─── Network Graph Construction ──────────────────────────────────────

    def _build_network_data(self, data: LayerData, analysis: Any) -> None:
        """Build network nodes and edges for interactive visualization."""
        # Network nodes
        data.network_nodes = []
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

            data.network_nodes.append(
                {
                    "id": c.id,
                    "label": f"{c.id}\n({c.structural.name})",
                    "group": level,
                    "type": c.type,
                    "level": level,
                    "value": value,
                    "title": (
                        f"{c.id}<br>"
                        f"Name: {c.structural.name}<br>"
                        f"Type: {c.type}<br>"
                        f"Score: {score:.3f}<br>"
                        f"Level: {level}"
                    ),
                }
            )

        # Network edges
        data.network_edges = []

        # 1. Derived DEPENDS_ON edges from analysis
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

        # 2. Raw structural edges from repository
        try:
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
            pass  # Raw edges are optional enhancement


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 for NaN or None."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        return f if f == f else 0.0  # NaN check
    except (TypeError, ValueError):
        return 0.0