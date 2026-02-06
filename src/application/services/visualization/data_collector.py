"""
Visualization Data Collector
"""
import logging
from typing import Dict, List, Any, Optional

from src.domain.models.visualization.layer_data import LayerData, LAYER_DEFINITIONS
from src.application.services.analysis_service import AnalysisService
from src.application.services.simulation_service import SimulationService
from src.application.services.validation_service import ValidationService

class LayerDataCollector:
    """
    Collects and aggregates data from various services for visualization.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisService,
        simulation_service: SimulationService,
        validation_service: ValidationService,
        repository: Any  # IGraphRepository
    ):
        self.analysis_service = analysis_service
        self.simulation_service = simulation_service
        self.validation_service = validation_service
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def collect_layer_data(self, layer: str, include_validation: bool = True) -> LayerData:
        """
        Collect all data for a specific layer.
        """
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}")
            
        layer_def = LAYER_DEFINITIONS[layer]
        data = LayerData(layer=layer, name=layer_def["name"])
        
        # 1. Structural & Quality Analysis
        self._collect_analysis_data(data, layer)
        
        # 2. Simulation Results
        self._collect_simulation_data(data, layer)
        
        # 3. Validation Results
        if include_validation:
            self._collect_validation_data(data, layer)
            
        return data

    def _collect_analysis_data(self, data: LayerData, layer: str) -> None:
        """Run analysis and populate data."""
        try:
            analysis = self.analysis_service.analyze_layer(layer)
            
            # Structural Stats
            data.nodes = analysis.structural.graph_summary.nodes
            data.edges = analysis.structural.graph_summary.edges
            data.density = analysis.structural.graph_summary.density
            data.connected_components = analysis.structural.graph_summary.num_components
            data.component_counts = analysis.structural.graph_summary.node_types or {}
            
            # Quality Stats
            for comp in analysis.quality.components:
                level = str(comp.levels.overall)
                # Handle enum string conversion if needed
                if hasattr(comp.levels.overall, 'name'):
                    level = comp.levels.overall.name
                
                if level == "CRITICAL": data.critical_count += 1
                elif level == "HIGH": data.high_count += 1
                elif level == "MEDIUM": data.medium_count += 1
                elif level == "LOW": data.low_count += 1
                elif level == "MINIMAL": data.minimal_count += 1
            
            data.spof_count = analysis.structural.graph_summary.num_articulation_points
            data.problems_count = len(analysis.problems)
            
            # Top Components
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
            
            # 4. Network Graph Data
            self._build_network_data(data, analysis)
            
        except Exception as e:
            self.logger.error(f"Analysis failed for layer {layer}: {e}")
            self.logger.exception("Analysis failure details")

    def _collect_simulation_data(self, data: LayerData, layer: str) -> None:
        """Run simulation and populate data."""
        try:
            layer_metrics = self.simulation_service.analyze_layer(layer)
            data.event_throughput = layer_metrics.event_throughput
            data.event_delivery_rate = layer_metrics.event_delivery_rate
            data.avg_impact = layer_metrics.avg_reachability_loss
            data.max_impact = layer_metrics.max_impact
        except Exception as e:
            self.logger.error(f"Simulation failed for layer {layer}: {e}")

    def _collect_validation_data(self, data: LayerData, layer: str) -> None:
        """Run validation and populate data."""
        try:
            # Note: Validation service might need updates if it doesn't support 'mw' directly yet
            # forcing single layer validation
            val_result = self.validation_service.validate_layers(layers=[layer]).layers.get(layer)
            if val_result:
                data.spearman = val_result.spearman
                data.f1_score = val_result.f1_score
                data.precision = val_result.precision
                data.recall = val_result.recall
                data.validation_passed = val_result.passed
        except Exception as e:
            self.logger.error(f"Validation failed for layer {layer}: {e}")

    def _build_network_data(self, data: LayerData, analysis: Any) -> None:
        """Build network nodes and edges for visualization."""
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
        
        # Network edges
        data.network_edges = []
        
        # 1. Add DEPENDS_ON edges from analysis (derived)
        for (source, target), edge_metrics in analysis.structural.edges.items():
            weight = 1.0
            if hasattr(edge_metrics, 'weight') and edge_metrics.weight is not None:
                weight = edge_metrics.weight
                if weight != weight: weight = 1.0
            
            dep_type = getattr(edge_metrics, 'dependency_type', 'default') or 'default'
            
            data.network_edges.append({
                "source": source, 
                "target": target,
                "weight": weight,
                "dependency_type": dep_type,
                "relation_type": "DEPENDS_ON",
                "title": f"DEPENDS_ON<br>Weight: {weight:.3f}<br>Type: {dep_type}"
            })
        
        # 2. Add raw structural edges if repository is available
        node_ids = {n["id"] for n in data.network_nodes}
        try:
            if self.repository:
                raw_graph = self.repository.get_graph_data(include_raw=True)
                for edge in raw_graph.edges:
                    # Skip DEPENDS_ON edges (already added above)
                    if edge.relation_type == "DEPENDS_ON":
                        continue
                    # Only include edges where both nodes are in our graph
                    if edge.source_id in node_ids and edge.target_id in node_ids:
                        weight = edge.weight if edge.weight == edge.weight else 1.0
                        data.network_edges.append({
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "weight": weight,
                            "dependency_type": edge.relation_type.lower(),
                            "relation_type": edge.relation_type,
                            "title": f"{edge.relation_type}<br>Weight: {weight:.3f}"
                        })
        except Exception as e:
            self.logger.warning(f"Could not fetch raw edges: {e}")
