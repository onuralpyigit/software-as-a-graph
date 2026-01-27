"""
Graph Visualizer

Orchestrates multi-layer analysis and visualization pipeline.
Integrates Analysis, Simulation, and Validation results into
a comprehensive HTML dashboard.

Features:
    - Multi-layer analysis (app, infra, mw-app, mw-infra, system)
    - Graph statistics and structural metrics
    - Criticality classification and problem detection
    - Simulation results (event + failure)
    - Validation metrics (correlation, classification, ranking)
    - Interactive network visualization
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .charts import ChartGenerator, CRITICALITY_COLORS
from .dashboard import DashboardGenerator


# Layer definitions
LAYER_DEFINITIONS = {
    "app": {
        "name": "Application Layer",
        "description": "Application-to-application dependencies",
        "icon": "📱",
    },
    "infra": {
        "name": "Infrastructure Layer",
        "description": "Node-to-node connections",
        "icon": "🖥️",
    },
    "mw-app": {
        "name": "Middleware-Application Layer",
        "description": "Applications and Brokers",
        "icon": "🔗",
    },
    "mw-infra": {
        "name": "Middleware-Infrastructure Layer",
        "description": "Nodes and Brokers",
        "icon": "⚙️",
    },
    "system": {
        "name": "Complete System",
        "description": "All components and dependencies",
        "icon": "🌐",
    },
}


@dataclass
class LayerData:
    """Aggregated data for a single layer."""
    layer: str
    name: str
    
    # Graph statistics
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    connected_components: int = 0
    
    # Component breakdown
    component_counts: Dict[str, int] = field(default_factory=dict)
    
    # Analysis results
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    minimal_count: int = 0
    spof_count: int = 0
    problems_count: int = 0
    
    # Top components
    top_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Simulation results
    avg_impact: float = 0.0
    max_impact: float = 0.0
    event_throughput: int = 0
    event_delivery_rate: float = 0.0
    
    # Validation results
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    validation_passed: bool = False
    
    # Network graph data
    network_nodes: List[Dict[str, Any]] = field(default_factory=list)
    network_edges: List[Dict[str, Any]] = field(default_factory=list)

    # Name mapping
    component_names: Dict[str, str] = field(default_factory=dict)


class GraphVisualizer:
    """
    Generates multi-layer analysis dashboards.
    
    Integrates:
        - Graph Analysis (structural metrics, quality scores)
        - Failure Simulation (impact assessment)
        - Validation (correlation, classification)
        - Interactive Visualization (vis.js network)
    
    Example:
        >>> with GraphVisualizer(uri="bolt://localhost:7687") as viz:
        ...     viz.generate_dashboard(
        ...         output_file="dashboard.html",
        ...         layers=["app", "infra", "system"]
        ...     )
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        repository: Optional[Any] = None  # GraphRepository
    ):
        """
        Initialize the visualizer.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            repository: Optional injected GraphRepository
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.repository = repository
        
        self.logger = logging.getLogger(__name__)
        self.charts = ChartGenerator()
        
        # Lazy-loaded modules
        self._analyzer = None
        self._simulator = None
        self._validator = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.repository and hasattr(self.repository, 'close'):
            self.repository.close()
    
    @property
    def analyzer(self):
        """Lazy-load analyzer."""
        if self._analyzer is None:
            try:
                from ..analysis import GraphAnalyzer
                self._analyzer = GraphAnalyzer(
                    uri=self.uri,
                    user=self.user,
                    password=self.password,
                    repository=self.repository
                )
            except ImportError:
                self.logger.warning("Analysis module not available")
        return self._analyzer
    
    @property
    def simulator(self):
        """Lazy-load simulator."""
        if self._simulator is None:
            try:
                from ..simulation import Simulator
                self._simulator = Simulator(
                    uri=self.uri,
                    user=self.user,
                    password=self.password,
                    repository=self.repository
                )
            except ImportError:
                self.logger.warning("Simulation module not available")
        return self._simulator
    
    @property
    def validator(self):
        """Lazy-load validator."""
        if self._validator is None:
            try:
                from ..validation import Validator
                self._validator = Validator()
            except ImportError:
                self.logger.warning("Validation module not available")
        return self._validator
    
    def generate_dashboard(
        self,
        output_file: str = "dashboard.html",
        layers: Optional[List[str]] = None,
        include_network: bool = True,
        include_validation: bool = True
    ) -> str:
        """
        Generate a comprehensive dashboard.
        
        Args:
            output_file: Output HTML file path
            layers: Layers to analyze (default: app, infra, system)
            include_network: Include interactive network visualization
            include_validation: Include validation metrics
            
        Returns:
            Path to generated HTML file
        """
        if layers is None:
            layers = ["app", "infra", "system"]
        
        self.logger.info(f"Generating dashboard for layers: {layers}")
        
        # Create dashboard
        dash = DashboardGenerator("Software-as-a-Graph Analysis Dashboard")
        
        # Collect data for all layers
        layer_data: Dict[str, LayerData] = {}
        
        for layer in layers:
            if layer not in LAYER_DEFINITIONS:
                self.logger.warning(f"Unknown layer: {layer}, skipping")
                continue
            
            self.logger.info(f"Processing layer: {layer}")
            layer_data[layer] = self._collect_layer_data(layer, include_validation)
        
        # Add overview section
        self._add_overview_section(dash, layer_data)
        
        # Add layer comparison section
        self._add_comparison_section(dash, layer_data)
        
        # Add individual layer sections
        for layer, data in layer_data.items():
            self._add_layer_section(dash, data, include_network)
        
        # Write output
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
        if self.analyzer:
            try:
                analysis = self.analyzer.analyze_layer(layer)
                
                # Graph statistics
                data.nodes = analysis.structural.graph_summary.nodes
                data.edges = analysis.structural.graph_summary.edges
                data.density = analysis.structural.graph_summary.density
                data.connected_components = analysis.structural.graph_summary.num_components
                
                # Component breakdown
                data.component_counts = analysis.structural.graph_summary.node_types or {}
                
                # Criticality counts
                for comp in analysis.quality.components:
                    level = comp.levels.overall.name if hasattr(comp.levels.overall, 'name') else str(comp.levels.overall)
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
                
                # SPOF count
                data.spof_count = analysis.structural.graph_summary.num_articulation_points
                
                # Problems
                data.problems_count = len(analysis.problems)
                
                # Top components
                sorted_comps = sorted(
                    analysis.quality.components,
                    key=lambda c: c.scores.overall,
                    reverse=True
                )
                
                data.top_components = [
                    {
                        "id": c.id,
                        "type": c.type,
                        "score": c.scores.overall,
                        "level": c.levels.overall.name if hasattr(c.levels.overall, 'name') else str(c.levels.overall),
                    }
                    for c in sorted_comps[:10]
                ]
                
                # Extract names
                data.component_names = {c.id: c.structural.name for c in analysis.quality.components}
                
                # Network graph data
                data.network_nodes = []
                for c in analysis.quality.components:
                    # Sanitize score
                    score = c.scores.overall if c.scores.overall is not None else 0.0
                    # Check for NaN (float('nan') != float('nan'))
                    if score != score:
                        score = 0.0
                    
                    # Calculate value based on sanitized score
                    value = score * 30 + 10
                    
                    data.network_nodes.append({
                        "id": c.id,
                        "label": f"{c.id}\n({c.structural.name})",
                        "group": c.levels.overall.name if hasattr(c.levels.overall, 'name') else c.type,
                        "value": value,
                        "title": f"{c.id}<br>Name: {c.structural.name}<br>Type: {c.type}<br>Score: {score:.3f}",
                    })
                
                # Build edges from structural data
                data.network_edges = self._build_network_edges(analysis)
                
            except Exception as e:
                self.logger.error(f"Analysis failed for layer {layer}: {e}")
        
        # Run simulation
        if self.simulator:
            try:
                # Get layer metrics
                layer_metrics = self.simulator.analyze_layer(layer)
                
                data.event_throughput = layer_metrics.event_throughput
                data.event_delivery_rate = layer_metrics.event_delivery_rate
                data.avg_impact = layer_metrics.avg_reachability_loss
                data.max_impact = layer_metrics.max_impact
                
            except Exception as e:
                self.logger.error(f"Simulation failed for layer {layer}: {e}")
        
        # Run validation
        if include_validation and self.validator and self.analyzer and self.simulator:
            try:
                # Get predicted and actual scores
                analysis = self.analyzer.analyze_layer(layer)
                sim_results = self.simulator.run_failure_simulation_exhaustive(layer=layer)
                
                predicted = {c.id: c.scores.overall for c in analysis.quality.components}
                actual = {r.target_id: r.impact.composite_impact for r in sim_results}
                types = {c.id: c.type for c in analysis.quality.components}
                
                val_result = self.validator.validate(
                    predicted_scores=predicted,
                    actual_scores=actual,
                    component_types=types,
                    layer=layer
                )
                
                data.spearman = val_result.overall.correlation.spearman
                data.f1_score = val_result.overall.classification.f1_score
                data.precision = val_result.overall.classification.precision
                data.recall = val_result.overall.classification.recall
                data.validation_passed = val_result.passed
                
            except Exception as e:
                self.logger.error(f"Validation failed for layer {layer}: {e}")
        
        return data
    
    def _build_network_edges(self, analysis) -> List[Dict[str, Any]]:
        """Build network edges from analysis data."""
        edges = []
        
        # Try to get edges from structural analysis
        # edges is a Dict[Tuple[str, str], EdgeMetrics]
        try:
            for (source, target), edge_metrics in analysis.structural.edges.items():
                edge_data = {
                    "source": source,
                    "target": target,
                }
                # Add title with weight if available
                if hasattr(edge_metrics, 'weight') and edge_metrics.weight is not None:
                    weight = edge_metrics.weight
                    # Check for NaN
                    if weight != weight:
                        weight = 0.0
                    edge_data["title"] = f"Weight: {weight:.3f}"
                edges.append(edge_data)
        except Exception as e:
            self.logger.error(f"Could not build network edges: {e}", exc_info=True)
        
        return edges
    
    def _add_overview_section(self, dash: DashboardGenerator, layer_data: Dict[str, LayerData]) -> None:
        """Add the overview section."""
        dash.start_section("📊 Overview", "overview")
        
        # Aggregate KPIs
        total_nodes = sum(d.nodes for d in layer_data.values())
        total_edges = sum(d.edges for d in layer_data.values())
        total_critical = sum(d.critical_count for d in layer_data.values())
        total_spofs = sum(d.spof_count for d in layer_data.values())
        total_problems = sum(d.problems_count for d in layer_data.values())
        
        layers_passed = sum(1 for d in layer_data.values() if d.validation_passed)
        
        kpis = {
            "Layers Analyzed": len(layer_data),
            "Total Nodes": total_nodes,
            "Total Edges": total_edges,
            "Critical Components": total_critical,
            "SPOFs Detected": total_spofs,
            "Problems Found": total_problems,
        }
        
        styles = {}
        if total_critical > 0:
            styles["Critical Components"] = "danger"
        if total_spofs > 0:
            styles["SPOFs Detected"] = "warning"
        if total_problems > 0:
            styles["Problems Found"] = "warning"
        
        dash.add_kpis(kpis, styles)
        
        # Validation summary
        if any(d.spearman > 0 for d in layer_data.values()):
            dash.add_subsection("Validation Summary")
            
            validation_data = {
                layer: {
                    "Spearman ρ": (d.spearman, 0.70, d.spearman >= 0.70),
                    "F1 Score": (d.f1_score, 0.80, d.f1_score >= 0.80),
                }
                for layer, d in layer_data.items()
                if d.spearman > 0
            }
            
            headers = ["Layer", "Spearman ρ", "F1 Score", "Precision", "Recall", "Status"]
            rows = [
                [
                    LAYER_DEFINITIONS[d.layer]["name"],
                    f"{d.spearman:.4f}",
                    f"{d.f1_score:.4f}",
                    f"{d.precision:.4f}",
                    f"{d.recall:.4f}",
                    "PASSED" if d.validation_passed else "FAILED",
                ]
                for d in layer_data.values()
                if d.spearman > 0
            ]
            
            dash.add_table(headers, rows, "Validation Results by Layer")
        
        dash.end_section()
    
    def _add_comparison_section(self, dash: DashboardGenerator, layer_data: Dict[str, LayerData]) -> None:
        """Add the layer comparison section."""
        if len(layer_data) < 2:
            return
        
        dash.start_section("📈 Layer Comparison", "comparison")
        
        # Comparison charts
        charts = []
        
        # Criticality distribution comparison
        crit_data = {
            LAYER_DEFINITIONS[layer]["name"]: {
                "Critical": d.critical_count,
                "High": d.high_count,
                "Medium": d.medium_count,
            }
            for layer, d in layer_data.items()
        }
        
        chart = self.charts.grouped_bar_chart(
            crit_data,
            "Criticality Distribution by Layer",
            ylabel="Component Count"
        )
        if chart:
            charts.append(chart)
        
        # Validation metrics comparison
        val_data = {
            LAYER_DEFINITIONS[layer]["name"]: {
                "Spearman": d.spearman,
                "F1": d.f1_score,
            }
            for layer, d in layer_data.items()
            if d.spearman > 0
        }
        
        if val_data:
            chart = self.charts.grouped_bar_chart(
                val_data,
                "Validation Metrics by Layer",
                ylabel="Score"
            )
            if chart:
                charts.append(chart)
        
        dash.add_charts(charts)
        
        # Comparison table
        headers = ["Layer", "Nodes", "Edges", "Density", "Critical", "SPOFs", "Max Impact"]
        rows = [
            [
                LAYER_DEFINITIONS[d.layer]["name"],
                d.nodes,
                d.edges,
                f"{d.density:.4f}",
                d.critical_count,
                d.spof_count,
                f"{d.max_impact:.4f}",
            ]
            for d in layer_data.values()
        ]
        
        dash.add_table(headers, rows, "Layer Statistics")
        
        dash.end_section()
    
    def _add_layer_section(
        self,
        dash: DashboardGenerator,
        data: LayerData,
        include_network: bool
    ) -> None:
        """Add a section for a specific layer."""
        layer_def = LAYER_DEFINITIONS[data.layer]
        
        dash.start_section(
            f"{layer_def['icon']} {layer_def['name']}",
            data.layer
        )
        
        # KPIs
        kpis = {
            "Nodes": data.nodes,
            "Edges": data.edges,
            "Density": f"{data.density:.4f}",
            "Critical": data.critical_count,
            "SPOFs": data.spof_count,
            "Problems": data.problems_count,
        }
        
        styles = {}
        if data.critical_count > 0:
            styles["Critical"] = "danger"
        if data.spof_count > 0:
            styles["SPOFs"] = "warning"
        
        dash.add_kpis(kpis, styles)
        
        # Charts
        charts = []
        
        # Criticality distribution
        crit_counts = {
            "CRITICAL": data.critical_count,
            "HIGH": data.high_count,
            "MEDIUM": data.medium_count,
            "LOW": data.low_count,
            "MINIMAL": data.minimal_count,
        }
        chart = self.charts.criticality_distribution(
            crit_counts,
            f"Criticality Distribution - {layer_def['name']}"
        )
        if chart:
            charts.append(chart)
        
        # Component type distribution
        if data.component_counts:
            chart = self.charts.pie_chart(
                data.component_counts,
                f"Component Types - {layer_def['name']}"
            )
            if chart:
                charts.append(chart)
        
        # Top components ranking
        if data.top_components:
            top_data = [(c["id"], c["score"], c["level"]) for c in data.top_components]
            chart = self.charts.impact_ranking(
                top_data,
                f"Top Components - {layer_def['name']}",
                names=data.component_names
            )
            if chart:
                charts.append(chart)
        
        dash.add_charts(charts)
        
        # Top components table
        if data.top_components:
            dash.add_subsection("Top Components by Quality Score")
            
            headers = ["Component", "Name", "Type", "Score", "Level"]
            rows = [
                [c["id"], data.component_names.get(c["id"], c["id"]), c["type"], f"{c['score']:.4f}", c["level"]]
                for c in data.top_components
            ]
            
            dash.add_table(headers, rows)
        
        # Simulation metrics
        if data.event_throughput > 0 or data.max_impact > 0:
            dash.add_subsection("Simulation Results")
            
            metrics = {
                "Event Throughput": f"{data.event_throughput} msgs",
                "Delivery Rate": f"{data.event_delivery_rate:.1f}%",
                "Avg Reachability Loss": f"{data.avg_impact * 100:.1f}%",
                "Max Impact": f"{data.max_impact:.4f}",
            }
            
            dash.add_metrics_box(metrics, "Simulation Metrics")
        
        # Validation metrics
        if data.spearman > 0:
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
            
            dash.add_metrics_box(metrics, "Validation Metrics", highlights)
        
        # Network visualization
        if include_network and data.network_nodes:
            dash.add_subsection("Network Topology")
            
            dash.add_network_graph(
                f"network-{data.layer}",
                data.network_nodes,
                data.network_edges,
                f"Interactive Graph - {layer_def['name']}"
            )
        
        dash.end_section()