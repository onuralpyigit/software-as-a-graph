"""
Simulation Application Service

Orchestrates Event and Failure simulations using the Simulation Domain Model.
Replaces the legacy Simulator facade.
"""

from __future__ import annotations
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.application.ports.graph_repository import GraphRepository
from src.domain.models.simulation.graph import SimulationGraph
from src.domain.models.simulation.metrics import LayerMetrics, ComponentCriticality, SimulationReport
from src.domain.services.simulation.event_simulator import EventSimulator, EventScenario, EventResult
from src.domain.services.simulation.failure_simulator import FailureSimulator, FailureScenario, FailureResult

# Import classifier from analysis module (optional)
try:
    from src.domain.services.analysis.classifier import BoxPlotClassifier
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False


class SimulationService:
    """
    Application Service for running simulations.
    
    Orchestrates:
        - Event-driven simulation
        - Failure simulation
        - Multi-layer analysis
        - Criticality classification
    """
    
    def __init__(self, repository: GraphRepository):
        """
        Initialize the simulation service.
        
        Args:
            repository: GraphRepository to load graph data
        """
        self.repository = repository
        self.logger = logging.getLogger(__name__)
        self._graph: Optional[SimulationGraph] = None
        self._classifier = BoxPlotClassifier() if HAS_CLASSIFIER else None
    
    def __enter__(self):
        """Context manager entry."""
        self._load_graph()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass # Repository lifecycle is managed by Container

    @property
    def graph(self) -> SimulationGraph:
        """Get the simulation graph, loading it if necessary."""
        if self._graph is None:
            self._load_graph()
        if self._graph is None: # Should be loaded by _load_graph
             raise ValueError("Failed to load simulation graph")
        return self._graph

    def _load_graph(self) -> None:
        """Load graph from repository."""
        self.logger.info("Loading simulation graph from repository")
        # include_raw=True is required for simulation (PUBLISHES_TO etc.)
        graph_data = self.repository.get_graph_data(include_raw=True)
        self._graph = SimulationGraph(graph_data=graph_data)
    
    # =========================================================================
    # Event Simulation
    # =========================================================================
    
    def run_event_simulation(
        self,
        source_app: str,
        num_messages: int = 100,
        duration: float = 10.0,
        **kwargs
    ) -> EventResult:
        """Run event simulation from a specific source application."""
        scenario = EventScenario(
            source_app=source_app,
            description=f"Event simulation: {source_app}",
            num_messages=num_messages,
            duration=duration,
            **kwargs
        )
        
        simulator = EventSimulator(self.graph)
        return simulator.simulate(scenario)
    
    def run_event_simulation_all(
        self,
        num_messages: int = 50,
        duration: float = 5.0,
        **kwargs
    ) -> Dict[str, EventResult]:
        """Run event simulation from all publisher applications."""
        scenario = EventScenario(
            source_app="",
            num_messages=num_messages,
            duration=duration,
            **kwargs
        )
        
        simulator = EventSimulator(self.graph)
        return simulator.simulate_all_publishers(scenario)

    # =========================================================================
    # Failure Simulation
    # =========================================================================
    
    def run_failure_simulation(
        self,
        target_id: str,
        layer: str = "system",
        cascade_probability: float = 1.0,
        **kwargs
    ) -> FailureResult:
        """Run failure simulation for a specific component."""
        scenario = FailureScenario(
            target_id=target_id,
            description=f"Failure simulation: {target_id}",
            layer=layer,
            cascade_probability=cascade_probability,
            **kwargs
        )
        
        simulator = FailureSimulator(self.graph)
        return simulator.simulate(scenario)
    
    def run_failure_simulation_exhaustive(
        self,
        layer: str = "system",
        cascade_probability: float = 1.0
    ) -> List[FailureResult]:
        """Run failure simulation for all components in a layer."""
        scenario = FailureScenario(
            target_id="",
            layer=layer,
            cascade_probability=cascade_probability,
        )
        
        simulator = FailureSimulator(self.graph)
        return simulator.simulate_exhaustive(scenario, layer=layer)

    # =========================================================================
    # Layer Analysis & Reporting
    # =========================================================================

    def analyze_layer(self, layer: str = "system") -> LayerMetrics:
        """Run combined event and failure analysis for a layer."""
        metrics = LayerMetrics(layer=layer)
        
        # Get components for this layer
        layer_comps = self.graph.get_components_by_layer(layer)
        metrics.total_components = len(layer_comps)
        
        # === Event Simulation ===
        event_results = self.run_event_simulation_all(num_messages=50, duration=5.0)
        
        if event_results:
            total_published = sum(r.metrics.messages_published for r in event_results.values())
            total_delivered = sum(r.metrics.messages_delivered for r in event_results.values())
            total_dropped = sum(r.metrics.messages_dropped for r in event_results.values())
            
            latencies = []
            for r in event_results.values():
                if r.metrics.messages_delivered > 0:
                    latencies.append(r.metrics.avg_latency)
            
            metrics.event_throughput = total_published
            metrics.event_delivery_rate = (total_delivered / total_published * 100) if total_published > 0 else 0
            metrics.event_drop_rate = (total_dropped / total_published * 100) if total_published > 0 else 0
            metrics.event_avg_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0
        
        # === Failure Simulation ===
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        
        if failure_results:
            reach_losses = [r.impact.reachability_loss for r in failure_results]
            fragmentations = [r.impact.fragmentation for r in failure_results]
            throughput_losses = [r.impact.throughput_loss for r in failure_results]
            
            metrics.avg_reachability_loss = sum(reach_losses) / len(reach_losses) if reach_losses else 0
            metrics.avg_fragmentation = sum(fragmentations) / len(fragmentations) if fragmentations else 0
            metrics.avg_throughput_loss = sum(throughput_losses) / len(throughput_losses) if throughput_losses else 0
            metrics.max_impact = max(r.impact.composite_impact for r in failure_results) if failure_results else 0
            
            # Count SPOFs (components with cascade > 0)
            metrics.spof_count = sum(1 for r in failure_results if r.impact.cascade_count > 0)
        
        return metrics

    def classify_components(
        self,
        layer: str = "system",
        k_factor: float = 1.5
    ) -> List[ComponentCriticality]:
        """Classify components by criticality based on simulation results."""
        # Run simulations
        event_results = self.run_event_simulation_all(num_messages=50, duration=5.0)
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        
        # Build component scores
        component_scores: Dict[str, ComponentCriticality] = {}
        
        # Process event results
        total_messages = sum(r.metrics.messages_published for r in event_results.values())
        
        for app_id, result in event_results.items():
            if total_messages > 0:
                event_impact = result.metrics.messages_published / total_messages
            else:
                event_impact = 0.0
            
            comp = self.graph.components.get(app_id)
            component_scores[app_id] = ComponentCriticality(
                id=app_id,
                type=comp.type if comp else "Unknown",
                event_impact=event_impact,
                message_throughput=result.metrics.messages_published,
            )
        
        # Process failure results
        for result in failure_results:
            comp_id = result.target_id
            failure_impact = result.impact.composite_impact
            
            if comp_id in component_scores:
                component_scores[comp_id].failure_impact = failure_impact
                component_scores[comp_id].cascade_count = result.impact.cascade_count
                component_scores[comp_id].reachability_loss = result.impact.reachability_loss
            else:
                comp = self.graph.components.get(comp_id)
                component_scores[comp_id] = ComponentCriticality(
                    id=comp_id,
                    type=comp.type if comp else "Unknown",
                    failure_impact=failure_impact,
                    cascade_count=result.impact.cascade_count,
                    reachability_loss=result.impact.reachability_loss,
                )
        
        # Calculate combined impact
        for crit in component_scores.values():
            crit.combined_impact = 0.5 * crit.event_impact + 0.5 * crit.failure_impact
        
        # Classify using box-plot method
        if self._classifier and component_scores:
            scores = [{"id": c.id, "score": c.combined_impact} for c in component_scores.values()]
            result = self._classifier.classify(scores, metric_name="criticality")
            
            level_map = {item.id: item.level.value for item in result.items}
            for comp_id, crit in component_scores.items():
                crit.level = level_map.get(comp_id, "minimal")
        else:
            # Simple threshold-based classification
            sorted_scores = sorted(component_scores.values(), key=lambda x: x.combined_impact, reverse=True)
            n = len(sorted_scores)
            for i, crit in enumerate(sorted_scores):
                pct = i / n if n > 0 else 0
                if pct < 0.1:
                    crit.level = "critical"
                elif pct < 0.25:
                    crit.level = "high"
                elif pct < 0.5:
                    crit.level = "medium"
                elif pct < 0.75:
                    crit.level = "low"
                else:
                    crit.level = "minimal"
        
        # Sort by combined impact
        result_list = list(component_scores.values())
        result_list.sort(key=lambda x: x.combined_impact, reverse=True)
        
        return result_list

    def generate_report(
        self,
        layers: Optional[List[str]] = None
    ) -> SimulationReport:
        """Generate comprehensive simulation report."""
        if layers is None:
            layers = ["app", "infra", "system"]
        
        self.logger.info(f"Generating simulation report for layers: {layers}")
        
        # Graph summary
        graph_summary = self.graph.get_summary()
        
        # Per-layer metrics
        layer_metrics: Dict[str, LayerMetrics] = {}
        for layer in layers:
            self.logger.info(f"Analyzing layer: {layer}")
            layer_metrics[layer] = self.analyze_layer(layer)
        
        # Classify components
        component_criticality = self.classify_components(layer="system")
        
        # Update layer metrics with criticality counts
        for crit in component_criticality:
            for layer, metrics in layer_metrics.items():
                layer_comps = self.graph.get_components_by_layer(layer)
                if crit.id in layer_comps:
                    if crit.level == "critical":
                        metrics.critical_count += 1
                    elif crit.level == "high":
                        metrics.high_count += 1
                    elif crit.level == "medium":
                        metrics.medium_count += 1
        
        # Top critical components
        top_critical = [
            {
                "id": c.id,
                "type": c.type,
                "level": c.level,
                "combined_impact": c.combined_impact,
                "cascade_count": c.cascade_count,
            }
            for c in component_criticality[:10]
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(layer_metrics, component_criticality)
        
        return SimulationReport(
            timestamp=datetime.now().isoformat(),
            graph_summary=graph_summary,
            layer_metrics=layer_metrics,
            component_criticality=component_criticality,
            top_critical=top_critical,
            recommendations=recommendations,
            component_names={c.id: c.properties.get("name", c.id) for c in self.graph.components.values()},
        )
    
    def _generate_recommendations(
        self,
        layer_metrics: Dict[str, LayerMetrics],
        component_criticality: List[ComponentCriticality]
    ) -> List[str]:
        """Generate actionable recommendations based on simulation results."""
        recommendations = []
        
        # Check for critical components
        critical_comps = [c for c in component_criticality if c.level == "critical"]
        if critical_comps:
            comp_ids = [c.id for c in critical_comps[:5]]
            recommendations.append(
                f"CRITICAL: {len(critical_comps)} components identified as critical: "
                f"{', '.join(comp_ids)}. Implement redundancy immediately."
            )
        
        # Check for SPOFs
        total_spofs = sum(m.spof_count for m in layer_metrics.values())
        if total_spofs > 0:
            recommendations.append(
                f"SPOF: {total_spofs} single points of failure detected. "
                f"Add backup instances or alternative paths."
            )
        
        # Check message drop rate
        for layer, metrics in layer_metrics.items():
            if metrics.event_drop_rate > 10:
                recommendations.append(
                    f"HIGH DROP RATE: Layer '{layer}' has {metrics.event_drop_rate:.1f}% message drops. "
                    f"Review broker capacity and topic configuration."
                )
        
        # Check fragmentation
        for layer, metrics in layer_metrics.items():
            if metrics.avg_fragmentation > 0.3:
                recommendations.append(
                    f"FRAGILE TOPOLOGY: Layer '{layer}' shows {metrics.avg_fragmentation*100:.1f}% "
                    f"average fragmentation. Increase connectivity."
                )
        
        # Check for high latency
        for layer, metrics in layer_metrics.items():
            if metrics.event_avg_latency_ms > 100:
                recommendations.append(
                    f"HIGH LATENCY: Layer '{layer}' has {metrics.event_avg_latency_ms:.1f}ms average latency. "
                    f"Optimize routing or add brokers."
                )
        
        if not recommendations:
            recommendations.append(
                "HEALTHY: No critical issues detected. Continue monitoring."
            )
        
        return recommendations
