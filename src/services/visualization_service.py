"""
Visualization Application Service
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.models.visualization.layer_data import LayerData, LAYER_DEFINITIONS
from src.services.visualization.chart_generator import ChartGenerator
from src.services.visualization.dashboard_generator import DashboardGenerator
from src.services.analysis_service import AnalysisService
from src.services.simulation_service import SimulationService
from src.services.validation_service import ValidationService


class VisualizationService:
    """
    Orchestrates multi-layer analysis and visualization pipeline.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisService,
        simulation_service: SimulationService,
        validation_service: ValidationService,
        chart_generator: Optional[ChartGenerator] = None
    ):
        self.analysis_service = analysis_service
        self.simulation_service = simulation_service
        self.validation_service = validation_service
        self.charts = chart_generator or ChartGenerator()
        self.logger = logging.getLogger(__name__)

    def generate_dashboard(
        self,
        output_file: str = "dashboard.html",
        layers: Optional[List[str]] = None,
        include_network: bool = True,
        include_matrix: bool = True,
        include_validation: bool = True
    ) -> str:
        """Generate a comprehensive dashboard."""
        if layers is None:
            layers = ["app", "infra", "system"]
        
        self.logger.info(f"Generating dashboard for layers: {layers}")
        dash = DashboardGenerator("Software-as-a-Graph Analysis Dashboard")
        
        layer_data: Dict[str, LayerData] = {}
        for layer in layers:
            if layer not in LAYER_DEFINITIONS:
                self.logger.warning(f"Unknown layer: {layer}, skipping")
                continue
            
            self.logger.info(f"Processing layer: {layer}")
            layer_data[layer] = self._collect_layer_data(layer, include_validation)
        
        self._add_overview_section(dash, layer_data)
        self._add_comparison_section(dash, layer_data)
        
        for layer, data in layer_data.items():
            self._add_layer_section(dash, data, include_network, include_matrix)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html = dash.generate()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.logger.info(f"Dashboard generated: {output_path}")
        return str(output_path)
    
    def _collect_layer_data(self, layer: str, include_validation: bool) -> LayerData:
        """Collect all data for a layer."""
        layer_def = LAYER_DEFINITIONS[layer]
        data = LayerData(layer=layer, name=layer_def["name"])
        
        # Run analysis
        try:
            analysis = self.analysis_service.analyze_layer(layer)
            data.nodes = analysis.structural.graph_summary.nodes
            data.edges = analysis.structural.graph_summary.edges
            data.density = analysis.structural.graph_summary.density
            data.connected_components = analysis.structural.graph_summary.num_components
            data.component_counts = analysis.structural.graph_summary.node_types or {}
            
            for comp in analysis.quality.components:
                level = comp.levels.overall.name if hasattr(comp.levels.overall, 'name') else str(comp.levels.overall)
                if level == "CRITICAL": data.critical_count += 1
                elif level == "HIGH": data.high_count += 1
                elif level == "MEDIUM": data.medium_count += 1
                elif level == "LOW": data.low_count += 1
                elif level == "MINIMAL": data.minimal_count += 1
            
            data.spof_count = analysis.structural.graph_summary.num_articulation_points
            data.problems_count = len(analysis.problems)
            
            sorted_comps = sorted(analysis.quality.components, key=lambda c: c.scores.overall, reverse=True)
            data.top_components = [
                {
                    "id": c.id,
                    "type": c.type,
                    "score": c.scores.overall,
                    "level": c.levels.overall.name if hasattr(c.levels.overall, 'name') else str(c.levels.overall),
                }
                for c in sorted_comps[:10]
            ]
            
            data.component_names = {c.id: c.structural.name for c in analysis.quality.components}
            
            # Network nodes
            data.network_nodes = []
            for c in analysis.quality.components:
                score = c.scores.overall if c.scores.overall is not None else 0.0
                if score != score: score = 0.0
                value = score * 30 + 10
                level = c.levels.overall.name if hasattr(c.levels.overall, 'name') else str(c.levels.overall)
                data.network_nodes.append({
                    "id": c.id,
                    "label": f"{c.id}\n({c.structural.name})",
                    "group": level,
                    "type": c.type,
                    "level": level,
                    "value": value,
                    "title": f"{c.id}<br>Name: {c.structural.name}<br>Type: {c.type}<br>Score: {score:.3f}<br>Level: {level}",
                })
            
            # Network edges - includes both DEPENDS_ON and raw structural edges
            data.network_edges = []
            
            # 1. Add DEPENDS_ON edges from analysis
            for (source, target), edge_metrics in analysis.structural.edges.items():
                weight = 1.0
                if hasattr(edge_metrics, 'weight') and edge_metrics.weight is not None:
                    weight = edge_metrics.weight
                    if weight != weight: weight = 1.0
                dep_type = getattr(edge_metrics, 'dependency_type', 'default') or 'default'
                edge_data = {
                    "source": source, 
                    "target": target,
                    "weight": weight,
                    "dependency_type": dep_type,
                    "relation_type": "DEPENDS_ON",
                    "title": f"DEPENDS_ON<br>Weight: {weight:.3f}<br>Type: {dep_type}"
                }
                data.network_edges.append(edge_data)
            
            # 2. Add raw structural edges (USES, PUBLISHES_TO, etc.)
            node_ids = {n["id"] for n in data.network_nodes}
            try:
                repository = self.analysis_service._repository
                if repository:
                    raw_graph = repository.get_graph_data(include_raw=True)
                    for edge in raw_graph.edges:
                        # Skip DEPENDS_ON edges (already added above)
                        if edge.relation_type == "DEPENDS_ON":
                            continue
                        # Only include edges where both nodes are in our graph
                        if edge.source_id in node_ids and edge.target_id in node_ids:
                            weight = edge.weight if edge.weight == edge.weight else 1.0
                            edge_data = {
                                "source": edge.source_id,
                                "target": edge.target_id,
                                "weight": weight,
                                "dependency_type": edge.relation_type.lower(),
                                "relation_type": edge.relation_type,
                                "title": f"{edge.relation_type}<br>Weight: {weight:.3f}"
                            }
                            data.network_edges.append(edge_data)
            except Exception as e:
                self.logger.warning(f"Could not fetch raw edges: {e}")
                
        except Exception as e:
            self.logger.error(f"Analysis failed for layer {layer}: {e}")
        
        # Run simulation
        try:
            layer_metrics = self.simulation_service.analyze_layer(layer)
            data.event_throughput = layer_metrics.event_throughput
            data.event_delivery_rate = layer_metrics.event_delivery_rate
            data.avg_impact = layer_metrics.avg_reachability_loss
            data.max_impact = layer_metrics.max_impact
        except Exception as e:
            self.logger.error(f"Simulation failed for layer {layer}: {e}")
        
        # Run validation
        if include_validation:
            try:
                val_result = self.validation_service.validate_layers(layers=[layer]).layers.get(layer)
                if val_result:
                    data.spearman = val_result.spearman
                    data.f1_score = val_result.f1_score
                    data.precision = val_result.precision
                    data.recall = val_result.recall
                    data.validation_passed = val_result.passed
            except Exception as e:
                self.logger.error(f"Validation failed for layer {layer}: {e}")
        
        return data

    def _add_overview_section(self, dash: DashboardGenerator, layer_data: Dict[str, LayerData]) -> None:
        """Add the overview section."""
        dash.start_section("📊 Overview", "overview")
        
        kpis = {
            "Layers Analyzed": len(layer_data),
            "Total Nodes": sum(d.nodes for d in layer_data.values()),
            "Total Edges": sum(d.edges for d in layer_data.values()),
            "Critical Components": sum(d.critical_count for d in layer_data.values()),
            "SPOFs Detected": sum(d.spof_count for d in layer_data.values()),
            "Problems Found": sum(d.problems_count for d in layer_data.values()),
        }
        
        styles = {}
        if kpis["Critical Components"] > 0: styles["Critical Components"] = "danger"
        if kpis["SPOFs Detected"] > 0: styles["SPOFs Detected"] = "warning"
        if kpis["Problems Found"] > 0: styles["Problems Found"] = "warning"
        
        dash.add_kpis(kpis, styles)
        
        if any(d.spearman > 0 for d in layer_data.values()):
            dash.add_subsection("Validation Summary")
            headers = ["Layer", "Spearman ρ", "F1 Score", "Precision", "Recall", "Status"]
            rows = [
                [LAYER_DEFINITIONS[d.layer]["name"], f"{d.spearman:.4f}", f"{d.f1_score:.4f}", f"{d.precision:.4f}", f"{d.recall:.4f}", "PASSED" if d.validation_passed else "FAILED"]
                for d in layer_data.values() if d.spearman > 0
            ]
            dash.add_table(headers, rows, "Validation Results by Layer")
        
        dash.end_section()

    def _add_comparison_section(self, dash: DashboardGenerator, layer_data: Dict[str, LayerData]) -> None:
        """Add the layer comparison section."""
        if len(layer_data) < 2: return
        
        dash.start_section("📈 Layer Comparison", "comparison")
        
        charts = []
        crit_data = {LAYER_DEFINITIONS[l]["name"]: {"Critical": d.critical_count, "High": d.high_count, "Medium": d.medium_count} for l, d in layer_data.items()}
        chart = self.charts.grouped_bar_chart(crit_data, "Criticality Distribution by Layer", ylabel="Component Count")
        if chart: charts.append(chart)
        
        val_data = {LAYER_DEFINITIONS[l]["name"]: {"Spearman": d.spearman, "F1": d.f1_score} for l, d in layer_data.items() if d.spearman > 0}
        if val_data:
            chart = self.charts.grouped_bar_chart(val_data, "Validation Metrics by Layer", ylabel="Score")
            if chart: charts.append(chart)
        
        dash.add_charts(charts)
        
        headers = ["Layer", "Nodes", "Edges", "Density", "Critical", "SPOFs", "Max Impact"]
        rows = [[LAYER_DEFINITIONS[d.layer]["name"], d.nodes, d.edges, f"{d.density:.4f}", d.critical_count, d.spof_count, f"{d.max_impact:.4f}"] for d in layer_data.values()]
        dash.add_table(headers, rows, "Layer Statistics")
        dash.end_section()

    def _add_layer_section(self, dash: DashboardGenerator, data: LayerData, include_network: bool, include_matrix: bool = True) -> None:
        """Add a section for a specific layer."""
        layer_def = LAYER_DEFINITIONS[data.layer]
        dash.start_section(f"{layer_def['icon']} {layer_def['name']}", data.layer)
        
        kpis = {"Nodes": data.nodes, "Edges": data.edges, "Density": f"{data.density:.4f}", "Critical": data.critical_count, "SPOFs": data.spof_count, "Problems": data.problems_count}
        styles = {}
        if data.critical_count > 0: styles["Critical"] = "danger"
        if data.spof_count > 0: styles["SPOFs"] = "warning"
        dash.add_kpis(kpis, styles)
        
        charts = []
        crit_counts = {"CRITICAL": data.critical_count, "HIGH": data.high_count, "MEDIUM": data.medium_count, "LOW": data.low_count, "MINIMAL": data.minimal_count}
        chart = self.charts.criticality_distribution(crit_counts, f"Criticality Distribution - {layer_def['name']}")
        if chart: charts.append(chart)
        
        if data.component_counts:
            chart = self.charts.pie_chart(data.component_counts, f"Component Types - {layer_def['name']}")
            if chart: charts.append(chart)
        
        if data.top_components:
            top_data = [(c["id"], c["score"], c["level"]) for c in data.top_components]
            chart = self.charts.impact_ranking(top_data, f"Top Components - {layer_def['name']}", names=data.component_names)
            if chart: charts.append(chart)
        
        dash.add_charts(charts)
        
        if data.top_components:
            dash.add_subsection("Top Components by Quality Score")
            headers = ["Component", "Name", "Type", "Score", "Level"]
            rows = [[c["id"], data.component_names.get(c["id"], c["id"]), c["type"], f"{c['score']:.4f}", c["level"]] for c in data.top_components]
            dash.add_table(headers, rows)
        
        if data.event_throughput > 0 or data.max_impact > 0:
            dash.add_subsection("Simulation Results")
            metrics = {"Event Throughput": f"{data.event_throughput} msgs", "Delivery Rate": f"{data.event_delivery_rate:.1f}%", "Avg Reachability Loss": f"{data.avg_impact * 100:.1f}%", "Max Impact": f"{data.max_impact:.4f}"}
            dash.add_metrics_box(metrics, "Simulation Metrics")
        
        if data.spearman > 0:
            dash.add_subsection("Validation Metrics")
            metrics = {"Spearman ρ": data.spearman, "F1 Score": data.f1_score, "Precision": data.precision, "Recall": data.recall, "Status": "PASSED" if data.validation_passed else "FAILED"}
            highlights = {"Spearman ρ": data.spearman >= 0.70, "F1 Score": data.f1_score >= 0.80, "Precision": data.precision >= 0.80, "Recall": data.recall >= 0.80}
            dash.add_metrics_box(metrics, "Validation Metrics", highlights)
        
        if include_network and data.network_nodes:
            dash.add_subsection("Network Topology")
            dash.add_network_graph(f"network-{data.layer}", data.network_nodes, data.network_edges, f"Interactive Graph - {layer_def['name']}")
        
        if include_matrix and data.network_nodes and len(data.network_nodes) > 1:
            dash.add_subsection("Dependency Matrix")
            dash.add_matrix_view(f"matrix-{data.layer}", data.network_nodes, data.network_edges, f"Adjacency Matrix - {layer_def['name']}")
        
        dash.end_section()
