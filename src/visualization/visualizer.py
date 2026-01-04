"""
Graph Visualizer

Orchestrates the pipeline:
1. Connects to Neo4j
2. Runs Structural & Quality Analysis
3. Runs Simulation
4. Generates Dashboard
"""

import logging
from typing import Dict, Any, List

from src.core.graph_exporter import GraphExporter
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
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
        """Run full pipeline and generate dashboard HTML."""
        self.logger.info("Starting Visualization Pipeline")
        
        # 1. Fetch Data
        graph_data = self.exporter.get_graph_data()
        self.logger.info(f"Loaded {len(graph_data.components)} components from Neo4j")

        # 2. Run Analysis (R, M, A, Q Scores)
        struct_analyzer = StructuralAnalyzer()
        quality_analyzer = QualityAnalyzer()
        
        struct_res = struct_analyzer.analyze(graph_data)
        quality_res = quality_analyzer.analyze(struct_res)
        
        # 3. Run Simulation (Failure Impact)
        sim_graph = SimulationGraph(graph_data)
        simulator = FailureSimulator(sim_graph)
        
        # Simulate impact for top 10 critical nodes (by Q score) to save time
        # In full production, you might simulate all.
        top_critical = sorted(quality_res.components, key=lambda x: x.scores.overall, reverse=True)[:10]
        impact_scores = {}
        
        for comp in top_critical:
            scenario = FailureScenario([comp.id], "Viz Sim")
            res = simulator.simulate(scenario)
            impact_scores[comp.id] = res.total_impact

        # 4. Construct Dashboard
        dash = DashboardGenerator()
        
        # KPI Section
        kpis = {
            "Total Components": len(graph_data.components),
            "Total Edges": len(graph_data.edges),
            "Avg Quality Score": f"{sum(c.scores.overall for c in quality_res.components)/len(quality_res.components):.2f}",
            "Critical Nodes": len([c for c in quality_res.components if c.level.value == "CRITICAL"])
        }
        dash.add_kpi_section(kpis)

        # Charts Section
        # a. Component Distribution
        types = {}
        for c in graph_data.components:
            types[c.component_type] = types.get(c.component_type, 0) + 1
        
        # b. Quality Scores Breakdown (Top 5)
        top_5 = quality_res.components[:5]
        names = [c.id for c in top_5]
        grouped_data = {
            "Reliability": [c.scores.reliability for c in top_5],
            "Maintainability": [c.scores.maintainability for c in top_5],
            "Availability": [c.scores.availability for c in top_5]
        }

        chart_objs = []
        chart_objs.append(self.charts.plot_pie(types, "Component Types"))
        chart_objs.append(self.charts.plot_grouped_bar(names, grouped_data, "Top 5 Components Quality Breakdown"))
        chart_objs.append(self.charts.plot_horizontal_bar(impact_scores, "Simulated Failure Impact (Nodes Affected)"))
        
        dash.add_charts_section("Analysis Visualization", [c for c in chart_objs if c])

        # Table Section
        headers = ["ID", "Type", "Reliability", "Maintainability", "Availability", "Overall Q", "Criticality"]
        rows = []
        for c in quality_res.components[:15]: # Show top 15
            rows.append([
                c.id, c.type,
                f"{c.scores.reliability:.2f}",
                f"{c.scores.maintainability:.2f}",
                f"{c.scores.availability:.2f}",
                f"{c.scores.overall:.2f}",
                c.level.value
            ])
            
        dash.add_table_section("Detailed Quality Analysis", headers, rows)

        # Save
        html = dash.generate()
        with open(output_file, "w") as f:
            f.write(html)
        
        return output_file