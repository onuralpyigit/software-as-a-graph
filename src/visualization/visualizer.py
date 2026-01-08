"""
Graph Visualizer

Orchestrates the multi-layer analysis and visualization pipeline.
Integrates Analysis (Prediction), Simulation (Ground Truth), and Validation (Statistics).
Generates a comprehensive dashboard for Application, Infrastructure, and Complete levels.
"""

import logging
from typing import Dict, Any, List

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

    def generate_dashboard(self, output_file: str = "dashboard.html"):
        dash = DashboardGenerator("Software-as-a-Graph Analysis & Validation Report")
        
        # Use context managers for robust resource handling
        with GraphAnalyzer(self.uri, self.user, self.password) as analyzer, \
             Simulator(self.uri, self.user, self.password) as simulator:
            
            validator = Validator(ValidationTargets()) # Default targets
            
            # --- 1. Complete System Level ---
            self.logger.info("Processing [Complete System] layer...")
            self._process_layer(
                dash=dash, 
                analyzer=analyzer, 
                simulator=simulator, 
                validator=validator,
                layer="complete", 
                title="1. Complete System Overview"
            )
            
            # --- 2. Application Level ---
            self.logger.info("Processing [Application] layer...")
            self._process_layer(
                dash=dash, 
                analyzer=analyzer, 
                simulator=simulator, 
                validator=validator,
                layer="application", 
                title="2. Application Layer Analysis"
            )
            
            # --- 3. Infrastructure Level ---
            self.logger.info("Processing [Infrastructure] layer...")
            self._process_layer(
                dash=dash, 
                analyzer=analyzer, 
                simulator=simulator, 
                validator=validator,
                layer="infrastructure", 
                title="3. Infrastructure Layer Analysis"
            )
            
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
        
        # A. Analysis (Prediction)
        if layer == "complete":
            analysis_res = analyzer.analyze()
        else:
            analysis_res = analyzer.analyze_layer(layer)
        
        components = analysis_res["results"].components
        if not components:
            dash.add_kpis({"Status": "No Data Found"})
            dash.end_section()
            return

        # B. Simulation (Ground Truth)
        # We run simulation for the specific layer to get relevant impact metrics
        # (Reachability for App, Fragmentation for Infra)
        sim_results = simulator.run_exhaustive_failure_sim(layer=layer)
        impact_map = {res.initial_failure: res.impact_score for res in sim_results}
        
        # C. Validation (Comparison)
        # Align data
        pred_scores = {c.id: c.scores.overall for c in components}
        actual_scores = impact_map # Already id->score
        comp_types = {c.id: c.type for c in components}
        
        val_result = validator.validate(pred_scores, actual_scores, comp_types, context=layer)
        overall_stats = val_result.overall
        
        # D. Dashboard Population
        
        # 1. KPIs
        critical_count = len([c for c in components if c.levels.overall == CriticalityLevel.CRITICAL])
        avg_impact = sum(impact_map.values()) / len(impact_map) if impact_map else 0
        
        dash.add_kpis({
            "Nodes Analyzed": len(components),
            "Critical Components": critical_count,
            "Avg Predicted Score": f"{sum(pred_scores.values())/len(pred_scores):.3f}" if pred_scores else "0.00",
            "Avg Actual Impact": f"{avg_impact:.3f}"
        })
        
        # 2. Validation Metrics Table
        dash.add_metrics_table({
            "Status": "PASSED" if overall_stats.passed else "FAILED",
            "Spearman Correlation (Rho)": overall_stats.correlation.spearman,
            "F1 Score (Classification)": overall_stats.classification.f1_score,
            "RMSE (Error)": overall_stats.error.rmse,
            "Top-5 Overlap": overall_stats.ranking.top_5_overlap
        })

        # 3. Charts
        # Data prep for scatter
        scatter_ids = []
        scatter_pred = []
        scatter_act = []
        for cid in pred_scores:
            if cid in actual_scores:
                scatter_ids.append(cid)
                scatter_pred.append(pred_scores[cid])
                scatter_act.append(actual_scores[cid])
        
        charts = [
            self.charts.plot_criticality_distribution(
                {l.name: len([c for c in components if c.levels.overall == l]) for l in CriticalityLevel}, 
                "Predicted Criticality"
            ),
            self.charts.plot_validation_scatter(
                scatter_pred, scatter_act, scatter_ids, f"{layer.capitalize()} Layer Validation"
            ),
            self.charts.plot_validation_metrics(
                {
                    "Rho": overall_stats.correlation.spearman, 
                    "F1": overall_stats.classification.f1_score, 
                    "RMSE": overall_stats.error.rmse
                }, 
                "Statistical Performance"
            )
        ]
        
        # Layer specific charts (Top Critical Quality Breakdown)
        top_critical = sorted(components, key=lambda x: x.scores.overall, reverse=True)[:5]
        charts.append(self.charts.plot_quality_comparison(top_critical, "Top 5 Critical Components Quality"))
        
        dash.add_charts(charts)

        # 4. Detailed Table (Top 10)
        headers = ["ID", "Type", "Predicted Score", "Actual Impact", "Reliability", "Maintainability", "Risk"]
        rows = []
        for c in top_critical: # Using top 5 here, could be top 10
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
        dash.end_section()