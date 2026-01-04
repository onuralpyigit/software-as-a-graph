"""
Graph Visualizer

Orchestrates the analysis pipeline and generates the dashboard.
Separates analysis into:
1. System Overview
2. Application Layer (Apps, Brokers)
3. Infrastructure Layer (Nodes)
4. Simulation Results
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
        self.logger.info("Starting Visualization Pipeline...")
        dash = DashboardGenerator()
        
        # --- 1. System Overview ---
        self.logger.info("Analyzing System Overview...")
        full_data = self.exporter.get_graph_data()
        
        struct_an = StructuralAnalyzer()
        qual_an = QualityAnalyzer()
        
        full_struct = struct_an.analyze(full_data)
        full_qual = qual_an.analyze(full_struct)
        
        dash.add_section_header("System Overview")
        dash.add_kpis({
            "Total Components": len(full_data.components),
            "Total Edges": len(full_data.edges),
            "Critical Components": len([c for c in full_qual.components if c.level == CriticalityLevel.CRITICAL]),
            "Avg Quality Score": f"{sum(c.scores.overall for c in full_qual.components) / max(1, len(full_qual.components)):.2f}"
        })
        
        # Component Distribution Chart
        dist_data = {}
        for c in full_data.components:
            dist_data[c.component_type] = dist_data.get(c.component_type, 0) + 1
        
        dash.add_charts([
            self.charts.plot_distribution(dist_data, "Component Type Distribution")
        ])
        dash.close_section()

        # --- 2. Application Layer ---
        self.logger.info("Analyzing Application Layer...")
        app_data = self.exporter.get_layer("application")
        if app_data.component_count > 0:
            app_struct = struct_an.analyze(app_data)
            app_qual = qual_an.analyze(app_struct)
            
            dash.add_section_header("Application Layer Analysis")
            dash.add_kpis({
                "Apps & Brokers": len(app_data.components),
                "App Dependencies": len(app_data.edges),
                "Avg App Reliability": f"{sum(c.scores.reliability for c in app_qual.components)/len(app_qual.components):.2f}"
            })
            
            # Top 5 Critical Apps Chart
            top_apps = sorted(app_qual.components, key=lambda x: x.scores.overall, reverse=True)[:5]
            dash.add_charts([
                self.charts.plot_quality_comparison(
                    [c.id for c in top_apps],
                    {
                        "Reliability": [c.scores.reliability for c in top_apps],
                        "Maintainability": [c.scores.maintainability for c in top_apps],
                        "Availability": [c.scores.availability for c in top_apps]
                    },
                    "Top 5 Critical Apps Quality Breakdown"
                )
            ])
            
            # Table
            headers = ["ID", "Type", "Overall Q", "Reliability", "Maintainability", "Criticality"]
            rows = []
            for c in top_apps:
                rows.append([
                    c.id, c.type, 
                    f"{c.scores.overall:.2f}", f"{c.scores.reliability:.2f}", 
                    f"{c.scores.maintainability:.2f}", c.level.value
                ])
            dash.add_table(headers, rows)
            dash.close_section()

        # --- 3. Infrastructure Layer ---
        self.logger.info("Analyzing Infrastructure Layer...")
        infra_data = self.exporter.get_layer("infrastructure")
        if infra_data.component_count > 0:
            infra_struct = struct_an.analyze(infra_data)
            infra_qual = qual_an.analyze(infra_struct)
            
            dash.add_section_header("Infrastructure Layer Analysis")
            dash.add_kpis({
                "Compute Nodes": len(infra_data.components),
                "Network Links": len(infra_data.edges),
                "Avg Availability": f"{sum(c.scores.availability for c in infra_qual.components)/len(infra_qual.components):.2f}"
            })
            
            # Nodes Table
            top_nodes = sorted(infra_qual.components, key=lambda x: x.scores.overall, reverse=True)[:5]
            rows = [[n.id, f"{n.scores.overall:.2f}", f"{n.scores.availability:.2f}", n.level.value] for n in top_nodes]
            dash.add_table(["Node ID", "Overall Q", "Availability", "Criticality"], rows)
            dash.close_section()

        # --- 4. Simulation Results ---
        self.logger.info("Running Simulations...")
        # We simulate failure of the Top 10 Critical Components from the Full System
        sim_graph = SimulationGraph(self.exporter.get_structural_graph())
        simulator = FailureSimulator(sim_graph)
        
        top_critical = sorted(full_qual.components, key=lambda x: x.scores.overall, reverse=True)[:10]
        impact_scores = {}
        
        for comp in top_critical:
            res = simulator.simulate(FailureScenario([comp.id], "Viz Sim"))
            total_impact = sum(res.impact_counts.values())
            impact_scores[comp.id] = total_impact

        dash.add_section_header("Simulation & Impact Analysis")
        dash.add_charts([
            self.charts.plot_impact_ranking(impact_scores, "Projected Failure Impact (Nodes Affected)")
        ])
        
        # Impact Table
        rows = []
        for cid, imp in sorted(impact_scores.items(), key=lambda x: x[1], reverse=True):
            rows.append([cid, imp, f"{(imp/len(full_data.components))*100:.1f}%"])
        dash.add_table(["Component ID", "Cascading Failures", "System % Affected"], rows)
        dash.close_section()

        # Save
        html = dash.generate()
        with open(output_file, "w") as f:
            f.write(html)
        self.logger.info(f"Dashboard saved to {output_file}")
        
        return output_file