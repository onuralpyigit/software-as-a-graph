"""
Graph Visualizer

Orchestrates the multi-layer analysis and visualization pipeline.
Combines Analysis (Prediction) and Simulation (Ground Truth) to create a comprehensive dashboard.
"""

import logging
import os
from typing import Dict, Any, List

from src.core.graph_exporter import GraphExporter
from src.analysis.analyzer import GraphAnalyzer
from src.simulation.simulator import Simulator
from src.analysis.quality_analyzer import CriticalityLevel

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
        dash = DashboardGenerator("Software-as-a-Graph Analysis Report")
        
        # We use context managers properly for the analyzers/simulators
        with GraphAnalyzer(self.uri, self.user, self.password) as analyzer, \
             Simulator(self.uri, self.user, self.password) as simulator:
            
            # --- 1. Complete System Overview & Validation ---
            self.logger.info("Generating System Overview...")
            self._generate_system_overview(dash, analyzer, simulator)
            
            # --- 2. Application Layer Detail ---
            self.logger.info("Generating Application Layer View...")
            self._generate_layer_view(dash, analyzer, "application", "Application Layer")
            
            # --- 3. Infrastructure Layer Detail ---
            self.logger.info("Generating Infrastructure Layer View...")
            self._generate_layer_view(dash, analyzer, "infrastructure", "Infrastructure Layer")
            
        # Save output
        with open(output_file, "w") as f:
            f.write(dash.generate())
        self.logger.info(f"Dashboard saved to {output_file}")
        return output_file

    def _generate_system_overview(self, dash: DashboardGenerator, analyzer: GraphAnalyzer, simulator: Simulator):
        """Generates the main system stats and validation section."""
        dash.start_section("System Overview & Validation")
        
        # A. Analysis
        full_res = analyzer.analyze() # Default context is complete system
        stats = full_res["stats"]
        qual = full_res["results"]
        
        # B. Validation (Run Simulation)
        self.logger.info("Running validation simulation (Exhaustive)...")
        # Note: In a huge graph, we might want to sample, but for project scale exhaustive is fine.
        sim_results = simulator.run_exhaustive_failure_sim(layer="complete")
        
        # Map Actual Impacts
        impact_map = {res.initial_failure: res.impact_score for res in sim_results}
        
        # Prepare Validation Data
        pred_scores = []
        actual_scores = []
        ids = []
        
        for comp in qual.components:
            if comp.id in impact_map:
                pred_scores.append(comp.scores.overall)
                actual_scores.append(impact_map[comp.id])
                ids.append(comp.id)

        # C. KPIs
        avg_impact = sum(actual_scores) / len(actual_scores) if actual_scores else 0
        dash.add_kpis({
            "Total Nodes": stats["nodes"],
            "Total Edges": stats["edges"],
            "Critical Components": len([c for c in qual.components if c.levels.overall == CriticalityLevel.CRITICAL]),
            "Avg System Impact": f"{avg_impact:.3f}"
        })
        
        # D. Charts
        # 1. Criticality Distribution
        level_counts = {l.name: 0 for l in CriticalityLevel}
        for c in qual.components:
            level_counts[c.levels.overall.name] += 1
            
        # 2. Validation Scatter
        charts = [
            self.charts.plot_graph_statistics({"Nodes": stats["nodes"], "Edges": stats["edges"]}, "Graph Topology"),
            self.charts.plot_criticality_distribution(level_counts, "Predicted Criticality Distribution"),
            self.charts.plot_validation_scatter(pred_scores, actual_scores, ids, "Validation: Prediction vs Reality")
        ]
        dash.add_charts(charts)
        dash.end_section()

    def _generate_layer_view(self, dash: DashboardGenerator, analyzer: GraphAnalyzer, layer: str, title: str):
        """Generates details for a specific layer."""
        dash.start_section(title)
        
        # Analysis for specific layer
        res = analyzer.analyze_layer(layer)
        components = res["results"].components
        
        if not components:
            dash.add_kpis({"Status": "No Data Found"})
            dash.end_section()
            return

        # Top Critical Components
        critical = sorted(components, key=lambda x: x.scores.overall, reverse=True)
        top_5 = critical[:5]
        
        # Charts
        charts = [
            self.charts.plot_quality_comparison(top_5, f"Top 5 Critical {layer.capitalize()} Components")
        ]
        dash.add_charts(charts)
        
        # Table
        headers = ["ID", "Type", "Reliability", "Maintainability", "Availability", "Risk Level"]
        rows = []
        for c in top_5:
            rows.append([
                c.id, c.type,
                f"{c.scores.reliability:.2f}",
                f"{c.scores.maintainability:.2f}",
                f"{c.scores.availability:.2f}",
                c.levels.overall.value.upper()
            ])
            
        dash.add_table(headers, rows)
        dash.end_section()