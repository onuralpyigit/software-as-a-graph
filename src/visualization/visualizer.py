"""
Graph Visualizer

Orchestrates the multi-layer analysis pipeline and generates the dashboard.
Integrates Prediction (Analysis) and Ground Truth (Simulation) for Validation.

Workflow:
1. System Overview & Validation (Prediction vs Impact)
2. Application Layer Detail
3. Infrastructure Layer Detail
4. Simulation Impact Ranking
"""

import logging
from typing import List, Dict, Any

from src.core.graph_exporter import GraphExporter
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer, CriticalityLevel
from src.simulation.simulation_graph import SimulationGraph
from src.simulation.failure_simulator import FailureSimulator, FailureScenario

from .charts import ChartGenerator
from .dashboard import DashboardGenerator

class GraphVisualizer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)
        self.charts = ChartGenerator()
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.exporter.close()

    def generate_dashboard(self, output_file: str = "dashboard.html"):
        self.logger.info("Starting Multi-Layer Visualization Pipeline...")
        dash = DashboardGenerator("Graph-Based Modeling and Analysis Report")
        
        # --- Data Acquisition & Analysis ---
        full_data = self.exporter.get_graph_data() # Derived Graph (App dependencies)
        structural_graph = self.exporter.get_structural_graph() # Physical Graph (Nodes, etc)
        
        # 1. Prediction (Topological Metrics)
        struct_an = StructuralAnalyzer()
        qual_an = QualityAnalyzer()
        
        full_struct = struct_an.analyze(full_data)
        full_qual = qual_an.analyze(full_struct)
        
        # 2. Ground Truth (Failure Simulation)
        self.logger.info("Running Validation Simulations...")
        sim_graph = SimulationGraph(structural_graph)
        simulator = FailureSimulator(sim_graph)
        
        # Map ID -> Predicted (Cscore) and Actual (Impact)
        validation_data = {"ids": [], "pred": [], "actual": []}
        impact_map = {}
        
        # Run simulation for a subset (or all) to generate correlation chart
        # For visualization speed, we limit if too large, but report implies rigorous validation
        target_components = full_qual.components 
        
        for i, comp in enumerate(target_components):
            # Predicted Score (Composite Criticality)
            c_score = comp.scores.overall
            
            # Actual Impact (Formula 7)
            res = simulator.simulate(FailureScenario(comp.id, "DashboardSim"))
            impact = res.impact_score
            
            validation_data["ids"].append(comp.id)
            validation_data["pred"].append(c_score)
            validation_data["actual"].append(impact)
            impact_map[comp.id] = impact

        # --- Section 1: Executive Summary & Validation ---
        dash.add_section_header("Executive Summary & Validation")
        dash.add_kpis({
            "Total Components": len(full_data.components),
            "Critical (Predicted)": len([c for c in full_qual.components if c.level == CriticalityLevel.CRITICAL]),
            "Avg Resilience": f"{1.0 - (sum(validation_data['actual'])/max(1, len(validation_data['actual']))):.2f}",
            "Validation Runs": len(validation_data["ids"])
        })
        
        # Charts: Criticality Distribution & Validation Correlation
        dist_data = {l.name: 0 for l in CriticalityLevel}
        for c in full_qual.components:
            dist_data[c.level.name] += 1
            
        dash.add_charts([
            self.charts.plot_criticality_distribution(dist_data, "Criticality Level Distribution (Box-Plot)"),
            self.charts.plot_validation_scatter(
                validation_data["pred"], 
                validation_data["actual"], 
                validation_data["ids"],
                "Validation: Predicted Criticality vs Actual Impact"
            )
        ])
        dash.close_section()

        # --- Section 2: Application Layer ---
        self.logger.info("Visualizing Application Layer...")
        app_data = self.exporter.get_layer("application")
        if app_data.component_count > 0:
            app_struct = struct_an.analyze(app_data)
            app_qual = qual_an.analyze(app_struct)
            
            dash.add_section_header("Application Layer")
            
            # Top Critical Apps
            top_apps = sorted(app_qual.components, key=lambda x: x.scores.overall, reverse=True)[:5]
            
            dash.add_charts([
                self.charts.plot_quality_comparison(
                    [c.id for c in top_apps],
                    {
                        "Reliability": [c.scores.reliability for c in top_apps],
                        "Maintainability": [c.scores.maintainability for c in top_apps],
                        "Availability": [c.scores.availability for c in top_apps]
                    },
                    "Top 5 Critical Applications (Quality Attributes)"
                )
            ])
            
            # Table
            headers = ["App ID", "Type", "Cscore", "Reliability", "Maintainability", "Criticality"]
            rows = []
            for c in top_apps:
                rows.append([
                    c.id, c.type, 
                    f"{c.scores.overall:.2f}", f"{c.scores.reliability:.2f}", 
                    f"{c.scores.maintainability:.2f}", c.level.value.upper()
                ])
            dash.add_table(headers, rows)
            dash.close_section()

        # --- Section 3: Infrastructure Layer ---
        self.logger.info("Visualizing Infrastructure Layer...")
        infra_data = self.exporter.get_layer("infrastructure")
        if infra_data.component_count > 0:
            infra_struct = struct_an.analyze(infra_data)
            infra_qual = qual_an.analyze(infra_struct)
            
            dash.add_section_header("Infrastructure Layer")
            
            # Identify Bottlenecks (High Betweenness/Centrality)
            top_infra = sorted(infra_qual.components, key=lambda x: x.scores.overall, reverse=True)[:5]
            
            dash.add_charts([
                self.charts.plot_impact_ranking(
                    {c.id: impact_map.get(c.id, 0) for c in top_infra},
                    "Infrastructure Failure Impact (Simulation)",
                    top_n=5
                )
            ])
            
            rows = [[n.id, f"{n.scores.overall:.2f}", f"{impact_map.get(n.id,0):.2f}", n.level.value.upper()] for n in top_infra]
            dash.add_table(["Node ID", "Centrality Score", "Simulated Impact", "Risk Level"], rows)
            dash.close_section()

        # Save
        html = dash.generate()
        with open(output_file, "w") as f:
            f.write(html)
        self.logger.info(f"Dashboard generated: {output_file}")
        
        return output_file