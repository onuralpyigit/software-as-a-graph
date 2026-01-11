"""
Graph Visualizer

Orchestrates the multi-layer analysis and visualization pipeline.
Integrates Analysis (Prediction), Simulation (Ground Truth), and Validation (Statistics).
Generates a comprehensive dashboard for Application, Infrastructure, and Complete levels.
"""

import logging
from typing import Dict, Any, List, Optional

from src.analysis.analyzer import GraphAnalyzer
from src.simulation.simulator import Simulator
from src.analysis.quality_analyzer import CriticalityLevel
from src.validation.validator import Validator
from src.validation.metrics import ValidationTargets

from .charts import ChartGenerator
from .dashboard import DashboardGenerator

class GraphVisualizer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.logger = logging.getLogger(__name__)
        self.charts = ChartGenerator()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass

    def generate_dashboard(self, output_file: str = "dashboard.html", layers: List[str] = None):
        """
        Generates the analysis dashboard.
        
        Args:
            output_file: Path to save HTML file
            layers: List of layers to analyze ["application", "infrastructure", "complete"]
        """
        dash = DashboardGenerator("Software-as-a-Graph Analysis & Validation Report")
        layers = layers or ["complete", "application", "infrastructure"]
        
        # Use context managers for robust resource handling
        with GraphAnalyzer(self.uri, self.user, self.password) as analyzer, \
             Simulator(self.uri, self.user, self.password) as simulator:
            
            validator = Validator(ValidationTargets()) # Default targets
            
            for i, layer in enumerate(layers, 1):
                title = f"{i}. {layer.title()} System Analysis"
                self.logger.info(f"Processing [{layer}] layer...")
                
                try:
                    self._process_layer(
                        dash=dash, 
                        analyzer=analyzer, 
                        simulator=simulator, 
                        validator=validator,
                        layer=layer, 
                        title=title
                    )
                except Exception as e:
                    self.logger.error(f"Failed to process layer {layer}: {e}", exc_info=True)
                    dash.start_section(title)
                    dash.add_kpis({"Error": "Analysis Failed"})
                    dash.end_section()

        # Save output
        with open(output_file, "w") as f:
            f.write(dash.generate())
        self.logger.info(f"Dashboard generated successfully: {output_file}")
        return output_file

    def _process_layer(self, dash: DashboardGenerator, analyzer: GraphAnalyzer, simulator: Simulator, validator: Validator, layer: str, title: str):
        """
        Executes Analysis, Simulation, and Validation for a specific layer.
        Populates a dashboard section with KPIs, Charts, and Tables.
        """
        dash.start_section(title)
        
        # 1. Analysis (Prediction)
        # Always use analyze_layer to get standardized LayerAnalysisResult dataclass
        analysis_res = analyzer.analyze_layer(layer)
        
        structural = analysis_res.structural
        quality = analysis_res.quality
        problems = analysis_res.problems
        components = quality.components

        if not components:
            dash.add_kpis({"Status": "No Components Found"})
            dash.end_section()
            return

        # 2. Simulation (Ground Truth)
        # Run exhaustive failure simulation to get actual impact scores
        sim_results = simulator.run_failure_simulation_exhaustive(layer=layer)
        impact_map = {res.target_id: res.impact.composite_impact for res in sim_results}
        
        # 3. Validation (Comparison)
        pred_scores = {c.id: c.scores.overall for c in components}
        actual_scores = impact_map 
        comp_types = {c.id: c.type for c in components}
        
        val_result = validator.validate(pred_scores, actual_scores, comp_types, context=layer)
        overall_stats = val_result.overall
        
        # 4. Dashboard Population
        
        # --- Section A: KPIs ---
        critical_count = len([c for c in components if c.levels.overall == CriticalityLevel.CRITICAL])
        avg_impact = sum(impact_map.values()) / len(impact_map) if impact_map else 0
        density = structural.graph_summary.density
        
        dash.add_kpis({
            "Nodes / Edges": f"{structural.graph_summary.nodes} / {structural.graph_summary.edges}",
            "Graph Density": f"{density:.3f}",
            "Critical Nodes": critical_count,
            "Problems Found": len(problems),
            "Avg Impact (Sim)": f"{avg_impact:.3f}"
        })
        
        # --- Section B: Structural & Quality Charts ---
        charts = []
        
        # Topology stats
        charts.append(self.charts.plot_graph_statistics({
            "Nodes": structural.graph_summary.nodes,
            "Edges": structural.graph_summary.edges,
            "SPOFs": structural.graph_summary.num_articulation_points
        }, "Graph Topology Stats"))

        # Criticality Distribution
        crit_counts = {l.name: len([c for c in components if c.levels.overall == l]) for l in CriticalityLevel}
        charts.append(self.charts.plot_criticality_distribution(crit_counts, "Predicted Criticality Distribution"))

        # Problem Severity
        if problems:
            severity_counts = {}
            for p in problems:
                severity_counts[p.severity] = severity_counts.get(p.severity, 0) + 1
            charts.append(self.charts.plot_problem_severity(severity_counts, "Architectural Problems by Severity"))

        # --- Section C: Validation Charts ---
        scatter_ids, scatter_pred, scatter_act = [], [], []
        for cid in pred_scores:
            if cid in actual_scores:
                scatter_ids.append(cid)
                scatter_pred.append(pred_scores[cid])
                scatter_act.append(actual_scores[cid])
        
        charts.append(self.charts.plot_validation_scatter(
            scatter_pred, scatter_act, scatter_ids, f"Prediction vs Reality ({layer})"
        ))
        
        charts.append(self.charts.plot_validation_metrics({
            "Spearman (Rho)": overall_stats.correlation.spearman, 
            "F1 Score": overall_stats.classification.f1_score, 
            "RMSE": overall_stats.error.rmse,
            "Ranking Overlap": overall_stats.ranking.top_5_overlap
        }, "Validation Metrics"))

        dash.add_charts([c for c in charts if c])

        # --- Section D: Metrics Table ---
        dash.add_metrics_table({
            "Validation Status": "PASSED" if overall_stats.passed else "FAILED",
            "Samples Validated": overall_stats.sample_size,
            "Spearman Correlation": overall_stats.correlation.spearman,
            "F1 Score": overall_stats.classification.f1_score,
            "RMSE (Error)": overall_stats.error.rmse,
            "Top-5 Overlap": overall_stats.ranking.top_5_overlap
        })

        # --- Section E: Top Critical Components Table ---
        # Sort by predicted score
        top_critical = sorted(components, key=lambda x: x.scores.overall, reverse=True)[:10]
        
        headers = ["ID", "Type", "Pred Score", "Actual Impact", "Reliability", "Maintainability", "Risk"]
        rows = []
        for c in top_critical:
            act = impact_map.get(c.id, 0.0)
            rows.append([
                c.id, 
                c.type,
                f"{c.scores.overall:.3f}",
                f"{act:.3f}",
                f"{c.scores.reliability:.2f}",
                f"{c.scores.maintainability:.2f}",
                c.levels.overall.value.upper()
            ])
        
        dash.add_table(headers, rows)
        
        # --- Section F: Detected Problems Table ---
        if problems:
            dash.sections.append("<h3>Top Detected Architectural Problems</h3>")
            prob_headers = ["Component", "Problem Type", "Severity", "Description"]
            prob_rows = []
            # Sort by priority/severity
            sorted_probs = sorted(problems, key=lambda p: p.priority, reverse=True)[:10]
            for p in sorted_probs:
                prob_rows.append([
                    p.entity_id,
                    p.entity_type,
                    p.severity,
                    p.description
                ])
            dash.add_table(prob_headers, prob_rows)

        dash.end_section()