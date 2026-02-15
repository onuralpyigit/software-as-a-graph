"""
Visualization Application Service

Orchestrates the Step 6 visualization pipeline, transforming multi-layer
analysis results into interactive decision-support dashboards.

Architecture follows the hexagonal pattern:
    VisualizationService (application layer)
    ├── LayerDataCollector  → collects data from analysis/simulation/validation ports
    ├── ChartGenerator      → produces chart HTML snippets
    └── DashboardGenerator  → assembles the final HTML document

Dashboard structure implements §6.5 of visualization.md:
    1. Overview section (aggregate KPIs across all layers)
    2. Layer Comparison section (cross-layer metrics)
    3. Per-layer detail sections (KPIs, charts, tables, network, matrix)
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .models import LayerData, LAYER_DEFINITIONS
from .charts import ChartGenerator
from .dashboard import DashboardGenerator
from .collector import LayerDataCollector

# Services for type hinting (using Any to avoid circular dependencies)
from typing import Any


class VisualizationService:
    """
    Orchestrates multi-layer analysis and visualization pipeline.

    Generates comprehensive HTML dashboards by collecting data from all
    preceding methodology steps and assembling visualizations according
    to the taxonomy defined in §6.4 of visualization.md.
    """

    def __init__(
        self,
        analysis_service: Any,
        simulation_service: Any,
        validation_service: Any,
        repository: Any,  # IGraphRepository
        chart_generator: Optional[ChartGenerator] = None,
    ):
        self.analysis_service = analysis_service
        self.simulation_service = simulation_service
        self.validation_service = validation_service
        self.repository = repository
        self.charts = chart_generator or ChartGenerator()

        # Initialize collector
        self.collector = LayerDataCollector(
            analysis_service,
            simulation_service,
            validation_service,
            repository,
        )

        self.logger = logging.getLogger(__name__)

    def generate_dashboard(
        self,
        output_file: str = "dashboard.html",
        layers: Optional[List[str]] = None,
        include_network: bool = True,
        include_matrix: bool = True,
        include_validation: bool = True,
    ) -> str:
        """
        Generate a comprehensive multi-layer analysis dashboard.

        Implements the pipeline from Definition 10 (visualization.md):
            1. Collect data for all selected layers
            2. Add overview section (§6.5 Section 1)
            3. Add layer comparison section (§6.5 Section 2)
            4. Add per-layer detail sections (§6.5 Section 3)
            5. Write self-contained HTML output

        Args:
            output_file: Path for the output HTML file.
            layers: List of layers to include. Defaults to all layers.
            include_network: Whether to include interactive network graphs.
            include_matrix: Whether to include dependency matrices.
            include_validation: Whether to run and include validation metrics.

        Returns:
            Path to the generated dashboard file.
        """
        if layers is None:
            layers = ["app", "infra", "mw", "system"]

        self.logger.info(f"Generating dashboard for layers: {layers}")
        dash = DashboardGenerator("Software-as-a-Graph Analysis Dashboard")

        # ── Phase 1: Data Collection ─────────────────────────────────────
        layer_data: Dict[str, LayerData] = {}
        for layer in layers:
            if layer not in LAYER_DEFINITIONS:
                self.logger.warning(f"Unknown layer: {layer}, skipping")
                continue

            self.logger.info(f"Processing layer: {layer}")
            try:
                data = self.collector.collect_layer_data(layer, include_validation)
                layer_data[layer] = data

                # Log scalability advice
                if data.recommend_matrix_only:
                    self.logger.info(
                        f"  Layer {layer}: {data.nodes} nodes — "
                        f"recommending matrix-only visualization"
                    )
            except Exception as e:
                self.logger.error(f"Failed to collect data for layer {layer}: {e}")

        # ── Phase 2: Dashboard Assembly ──────────────────────────────────
        self._add_overview_section(dash, layer_data)
        self._add_comparison_section(dash, layer_data)

        for layer, data in layer_data.items():
            # Auto-disable network graph for large systems (§6.11)
            effective_network = include_network and not data.recommend_matrix_only
            if include_network and data.recommend_matrix_only:
                self.logger.info(
                    f"  Layer {layer}: auto-disabling network graph "
                    f"({data.nodes} nodes > 200 threshold)"
                )
            self._add_layer_section(
                dash, data, effective_network, include_matrix
            )

        # ── Phase 3: Write Output ────────────────────────────────────────
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = dash.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        file_size_kb = output_path.stat().st_size / 1024
        self.logger.info(
            f"Dashboard generated: {output_path} ({file_size_kb:.0f} KB)"
        )
        return str(output_path)

    # ─── §6.5 Section 1: Overview ────────────────────────────────────────

    def _add_overview_section(
        self, dash: DashboardGenerator, layer_data: Dict[str, LayerData]
    ) -> None:
        """
        Add the overview section with aggregate KPIs and distributions.

        Implements progressive disclosure Level 1 (§6.3.1):
        practitioners can assess system health in ~5 seconds.
        """
        dash.start_section("📊 Overview", "overview")

        # ── KPI Cards ────────────────────────────────────────────────────
        total_critical = sum(d.critical_count for d in layer_data.values())
        total_spofs = sum(d.spof_count for d in layer_data.values())
        total_problems = sum(d.problems_count for d in layer_data.values())

        kpis = {
            "Layers Analyzed": len(layer_data),
            "Total Nodes": sum(d.nodes for d in layer_data.values()),
            "Total Edges": sum(d.edges for d in layer_data.values()),
            "Critical Components": total_critical,
            "SPOFs Detected": total_spofs,
            "Anti-Patterns": total_problems,
        }

        styles = {}
        if total_critical > 0:
            styles["Critical Components"] = "danger"
        if total_spofs > 0:
            styles["SPOFs Detected"] = "warning"
        if total_problems > 0:
            styles["Anti-Patterns"] = "warning"

        dash.add_kpis(kpis, styles)

        # ── Validation Summary Table ─────────────────────────────────────
        validated_layers = [d for d in layer_data.values() if d.has_validation]
        if validated_layers:
            dash.add_subsection("Validation Summary")
            headers = [
                "Layer", "Spearman ρ", "F1 Score", "Precision",
                "Recall", "Top-5", "Top-10", "Status",
            ]
            rows = []
            for d in validated_layers:
                top5_str = f"{d.top5_overlap:.0%}" if d.top5_overlap > 0 else "—"
                top10_str = f"{d.top10_overlap:.0%}" if d.top10_overlap > 0 else "—"
                rows.append([
                    LAYER_DEFINITIONS[d.layer]["name"],
                    f"{d.spearman:.4f}",
                    f"{d.f1_score:.4f}",
                    f"{d.precision:.4f}",
                    f"{d.recall:.4f}",
                    top5_str,
                    top10_str,
                    "PASSED" if d.validation_passed else "FAILED",
                ])
            dash.add_table(headers, rows, "Validation Results by Layer")

        dash.end_section()

    # ─── §6.5 Section 2: Layer Comparison ────────────────────────────────

    def _add_comparison_section(
        self, dash: DashboardGenerator, layer_data: Dict[str, LayerData]
    ) -> None:
        """
        Add the layer comparison section with cross-layer charts and tables.

        Supports Workflow 3: Architecture Assessment (§6.7) —
        practitioners identify which layer carries the most risk.
        """
        if len(layer_data) < 2:
            return

        dash.start_section("📈 Layer Comparison", "comparison")

        charts = []

        # Criticality distribution by layer
        crit_data = {
            LAYER_DEFINITIONS[l]["name"]: {
                "Critical": d.critical_count,
                "High": d.high_count,
                "Medium": d.medium_count,
            }
            for l, d in layer_data.items()
        }
        chart = self.charts.grouped_bar_chart(
            crit_data,
            "Criticality Distribution by Layer",
            ylabel="Component Count",
        )
        if chart:
            charts.append(chart)

        # Validation metrics by layer
        val_data = {
            LAYER_DEFINITIONS[l]["name"]: {
                "Spearman ρ": d.spearman,
                "F1 Score": d.f1_score,
            }
            for l, d in layer_data.items()
            if d.has_validation
        }
        if val_data:
            chart = self.charts.grouped_bar_chart(
                val_data,
                "Validation Metrics by Layer",
                ylabel="Score",
            )
            if chart:
                charts.append(chart)

        dash.add_charts(charts)

        # Cross-layer statistics table
        headers = [
            "Layer", "Nodes", "Edges", "Density",
            "Critical", "SPOFs", "Max Impact", "Scale",
        ]
        rows = [
            [
                LAYER_DEFINITIONS[d.layer]["name"],
                d.nodes,
                d.edges,
                f"{d.density:.4f}",
                d.critical_count,
                d.spof_count,
                f"{d.max_impact:.4f}",
                d.scale_category.upper(),
            ]
            for d in layer_data.values()
        ]
        dash.add_table(headers, rows, "Layer Statistics")
        dash.end_section()

    # ─── §6.5 Section 3: Per-Layer Detail ────────────────────────────────

    def _add_layer_section(
        self,
        dash: DashboardGenerator,
        data: LayerData,
        include_network: bool,
        include_matrix: bool = True,
    ) -> None:
        """
        Add a comprehensive detail section for a specific layer.

        Implements progressive disclosure Levels 2-5 (§6.3.1):
            Level 2: Distribution charts and KPIs
            Level 3: Component tables with RMAV breakdown
            Level 4: Interactive network graph
            Level 5: Dependency matrix
        """
        layer_def = LAYER_DEFINITIONS[data.layer]
        dash.start_section(
            f"{layer_def['icon']} {layer_def['name']}", data.layer
        )

        # ── Level 1-2: KPIs + Distribution ──────────────────────────────
        kpis = {
            "Nodes": data.nodes,
            "Edges": data.edges,
            "Density": f"{data.density:.4f}",
            "Critical": data.critical_count,
            "SPOFs": data.spof_count,
            "Anti-Patterns": data.problems_count,
        }
        styles = {}
        if data.critical_count > 0:
            styles["Critical"] = "danger"
        if data.spof_count > 0:
            styles["SPOFs"] = "warning"
        dash.add_kpis(kpis, styles)

        # ── Charts ──────────────────────────────────────────────────────
        charts = []

        # §6.4.2: Criticality distribution pie
        chart = self.charts.criticality_distribution(
            data.classification_distribution,
            f"Criticality Distribution — {layer_def['name']}",
        )
        if chart:
            charts.append(chart)

        # §6.4.2: Component types pie
        if data.component_counts:
            chart = self.charts.pie_chart(
                data.component_counts,
                f"Component Types — {layer_def['name']}",
            )
            if chart:
                charts.append(chart)

        # §6.4.3: Impact ranking bar
        if data.top_components:
            top_data = [
                (c["id"], c["score"], c["level"])
                for c in data.top_components
            ]
            chart = self.charts.impact_ranking(
                top_data,
                f"Top Components by Q(v) — {layer_def['name']}",
                names=data.component_names,
            )
            if chart:
                charts.append(chart)

        # §6.4.3: RMAV stacked breakdown (NEW)
        if data.component_details:
            chart = self.charts.rmav_breakdown(
                data.component_details,
                f"RMAV Quality Breakdown — {layer_def['name']}",
            )
            if chart:
                charts.append(chart)

        # §6.4.4: Correlation scatter plot (NEW)
        if data.scatter_data and len(data.scatter_data) >= 3:
            chart = self.charts.correlation_scatter(
                data.scatter_data,
                f"Prediction Correlation — {layer_def['name']}",
                spearman=data.spearman,
            )
            if chart:
                charts.append(chart)

        dash.add_charts(charts)

        # ── Level 3: Component Tables ───────────────────────────────────
        if data.component_details:
            dash.add_subsection("Component Quality Scores (RMAV)")
            headers = [
                "Component", "Name", "Type",
                "R(v)", "M(v)", "A(v)", "V(v)", "Q(v)",
                "Level",
            ]
            rows = [
                [
                    c.id,
                    c.name,
                    c.type,
                    f"{c.reliability:.3f}",
                    f"{c.maintainability:.3f}",
                    f"{c.availability:.3f}",
                    f"{c.vulnerability:.3f}",
                    f"{c.overall:.3f}",
                    c.level,
                ]
                for c in data.component_details[:15]
            ]
            dash.add_table(headers, rows)
        elif data.top_components:
            # Fallback to legacy top_components format
            dash.add_subsection("Top Components by Quality Score")
            headers = ["Component", "Name", "Type", "Q(v)", "Level"]
            rows = [
                [
                    c["id"],
                    data.component_names.get(c["id"], c["id"]),
                    c["type"],
                    f"{c['score']:.4f}",
                    c["level"],
                ]
                for c in data.top_components
            ]
            dash.add_table(headers, rows)

        # ── Simulation Results ──────────────────────────────────────────
        if data.has_simulation:
            dash.add_subsection("Simulation Results")
            metrics = {
                "Event Throughput": f"{data.event_throughput} msgs",
                "Delivery Rate": f"{data.event_delivery_rate:.1f}%",
                "Avg Reachability Loss": f"{data.avg_impact * 100:.1f}%",
                "Max Impact": f"{data.max_impact:.4f}",
            }
            dash.add_metrics_box(metrics, "Simulation Metrics")

        # ── Validation Metrics ──────────────────────────────────────────
        if data.has_validation:
            dash.add_subsection("Validation Metrics")
            metrics = {
                "Spearman ρ": data.spearman,
                "F1 Score": data.f1_score,
                "Precision": data.precision,
                "Recall": data.recall,
                "Status": "PASSED" if data.validation_passed else "FAILED",
            }
            highlights = {
                "Spearman ρ": data.spearman >= 0.70,
                "F1 Score": data.f1_score >= 0.80,
                "Precision": data.precision >= 0.80,
                "Recall": data.recall >= 0.80,
            }

            # Add Top-K overlap if available
            if data.top5_overlap > 0:
                metrics["Top-5 Overlap"] = f"{data.top5_overlap:.0%}"
                highlights["Top-5 Overlap"] = data.top5_overlap >= 0.40
            if data.top10_overlap > 0:
                metrics["Top-10 Overlap"] = f"{data.top10_overlap:.0%}"
                highlights["Top-10 Overlap"] = data.top10_overlap >= 0.50

            dash.add_metrics_box(metrics, "Validation Metrics", highlights)

        # ── Level 4: Interactive Network Graph (§6.4.5) ──────────────────
        if include_network and data.network_nodes:
            dash.add_subsection("Network Topology")
            if data.recommend_matrix_only:
                dash.add_subsection(
                    f"⚠️ Large graph ({data.nodes} nodes) — "
                    "network graph disabled for performance. "
                    "Use dependency matrix below."
                )
            else:
                dash.add_network_graph(
                    f"network-{data.layer}",
                    data.network_nodes,
                    data.network_edges,
                    f"Interactive Graph — {layer_def['name']}",
                )

        # ── Level 5: Dependency Matrix (§6.4.6) ─────────────────────────
        if include_matrix and data.network_nodes and len(data.network_nodes) > 1:
            dash.add_subsection("Dependency Matrix")
            dash.add_matrix_view(
                f"matrix-{data.layer}",
                data.network_nodes,
                data.network_edges,
                f"Adjacency Matrix — {layer_def['name']}",
            )

        dash.end_section()