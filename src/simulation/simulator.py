"""
Simulator Facade

Orchestrates Event and Failure simulations on the raw graph model.
Provides unified API for simulation, metrics collection, and criticality classification.

Features:
    - Event-driven simulation (pub-sub message flow, throughput, latency)
    - Failure simulation (cascade propagation, impact assessment)
    - Multi-layer analysis (app, infra, mw-app, mw-infra, system)
    - Criticality classification based on simulation results
    - Comprehensive reporting and export

Layers:
    - app: Application layer (Applications only)
    - infra: Infrastructure layer (Nodes only)
    - mw-app: Middleware-Application (Applications + Brokers)
    - mw-infra: Middleware-Infrastructure (Nodes + Brokers)
    - system: Complete system (all components)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .simulation_graph import SimulationGraph, ComponentState
from .event_simulator import EventSimulator, EventScenario, EventResult, RuntimeMetrics
from .failure_simulator import FailureSimulator, FailureScenario, FailureResult, ImpactMetrics, CascadeRule


# Import classifier from analysis module (optional)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.classifier import BoxPlotClassifier, CriticalityLevel
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False
    CriticalityLevel = None


@dataclass
class LayerMetrics:
    """
    Aggregated metrics for a single analysis layer.
    
    Combines event simulation metrics (throughput, latency, drops)
    with failure simulation metrics (impact, cascade, fragmentation).
    """
    layer: str
    
    # Event simulation metrics
    event_throughput: int = 0            # Total messages processed
    event_delivery_rate: float = 0.0     # % messages delivered
    event_drop_rate: float = 0.0         # % messages dropped
    event_avg_latency_ms: float = 0.0    # Average latency in ms
    
    # Failure simulation metrics
    avg_reachability_loss: float = 0.0   # Average path loss
    avg_fragmentation: float = 0.0       # Average component loss
    avg_throughput_loss: float = 0.0     # Average capacity loss
    max_impact: float = 0.0              # Highest single-point impact
    
    # Criticality counts
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    spof_count: int = 0
    
    # Component count
    total_components: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "event_metrics": {
                "throughput": self.event_throughput,
                "delivery_rate_percent": round(self.event_delivery_rate, 2),
                "drop_rate_percent": round(self.event_drop_rate, 2),
                "avg_latency_ms": round(self.event_avg_latency_ms, 3),
            },
            "failure_metrics": {
                "avg_reachability_loss_percent": round(self.avg_reachability_loss * 100, 2),
                "avg_fragmentation_percent": round(self.avg_fragmentation * 100, 2),
                "avg_throughput_loss_percent": round(self.avg_throughput_loss * 100, 2),
                "max_impact": round(self.max_impact, 4),
            },
            "criticality": {
                "total_components": self.total_components,
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "spof_count": self.spof_count,
            },
        }


@dataclass
class ComponentCriticality:
    """
    Criticality assessment for a component based on simulation results.
    
    Combines event impact (message handling importance) with
    failure impact (cascade and reachability effects).
    """
    id: str
    type: str
    
    # Simulation-based scores
    event_impact: float = 0.0         # Impact from event simulation
    failure_impact: float = 0.0       # Impact from failure simulation
    combined_impact: float = 0.0      # Weighted combination
    
    # Classification level
    level: str = "minimal"
    
    # Supporting metrics
    cascade_count: int = 0
    message_throughput: int = 0
    reachability_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "scores": {
                "event_impact": round(self.event_impact, 4),
                "failure_impact": round(self.failure_impact, 4),
                "combined_impact": round(self.combined_impact, 4),
            },
            "level": self.level,
            "metrics": {
                "cascade_count": self.cascade_count,
                "message_throughput": self.message_throughput,
                "reachability_loss_percent": round(self.reachability_loss * 100, 2),
            },
        }


@dataclass
class SimulationReport:
    """
    Comprehensive simulation report combining all analyses.
    """
    timestamp: str
    graph_summary: Dict[str, Any]
    
    # Per-layer metrics
    layer_metrics: Dict[str, LayerMetrics]
    
    # Component criticality (classified by simulation)
    component_criticality: List[ComponentCriticality]
    
    # Top critical components
    top_critical: List[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[str]
    
    # Name mapping for display
    component_names: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "graph_summary": self.graph_summary,
            "layer_metrics": {k: v.to_dict() for k, v in self.layer_metrics.items()},
            "component_criticality": [c.to_dict() for c in self.component_criticality],
            "top_critical": self.top_critical,
            "recommendations": self.recommendations,
        }
    
    def get_critical_components(self) -> List[ComponentCriticality]:
        """Get all CRITICAL level components."""
        return [c for c in self.component_criticality if c.level == "critical"]
    
    def get_high_priority(self) -> List[ComponentCriticality]:
        """Get CRITICAL and HIGH level components."""
        return [c for c in self.component_criticality if c.level in ("critical", "high")]


class Simulator:
    """
    Main simulator facade for pub-sub system analysis.
    
    Orchestrates:
        - Event-driven simulation for throughput/latency analysis
        - Failure simulation for impact/cascade analysis
        - Multi-layer analysis (app, infra, mw-app, mw-infra, system)
        - Criticality classification based on simulation results
    
    Example:
        >>> with Simulator(uri="bolt://localhost:7687") as sim:
        ...     # Run event simulation
        ...     event_result = sim.run_event_simulation("App1")
        ...     
        ...     # Run failure simulation
        ...     failure_result = sim.run_failure_simulation("Broker1")
        ...     
        ...     # Generate comprehensive report
        ...     report = sim.generate_report(layers=["app", "infra", "system"])
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        Initialize the simulator.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        
        self.logger = logging.getLogger(__name__)
        
        # Graph instance (lazy loaded)
        self._graph: Optional[SimulationGraph] = None
        
        # Classifier for criticality classification
        self._classifier = BoxPlotClassifier() if HAS_CLASSIFIER else None
    
    def __enter__(self):
        """Context manager entry."""
        self._load_graph()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    def _load_graph(self) -> None:
        """Load graph from configured source."""
        if self._graph is not None:
            return
        
        self._graph = SimulationGraph(
            uri=self.uri,
            user=self.user,
            password=self.password
        )
    
    @property
    def graph(self) -> SimulationGraph:
        """Get the simulation graph (lazy loading)."""
        if self._graph is None:
            self._load_graph()
        return self._graph
    
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
        """
        Run event simulation from a specific source application.
        
        Args:
            source_app: Source application ID
            num_messages: Number of messages to simulate
            duration: Simulation duration in seconds
            **kwargs: Additional EventScenario parameters
            
        Returns:
            EventResult with throughput, latency, and drop metrics
        """
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
        """
        Run event simulation from all publisher applications.
        
        Args:
            num_messages: Messages per publisher
            duration: Duration per simulation
            **kwargs: Additional EventScenario parameters
            
        Returns:
            Dict mapping app_id to EventResult
        """
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
        """
        Run failure simulation for a specific component.
        
        Args:
            target_id: Component ID to fail
            layer: Analysis layer
            cascade_probability: Probability of cascade propagation
            **kwargs: Additional FailureScenario parameters
            
        Returns:
            FailureResult with cascade and impact analysis
        """
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
        """
        Run failure simulation for all components in a layer.
        
        Args:
            layer: Analysis layer
            cascade_probability: Probability of cascade propagation
            
        Returns:
            List of FailureResult sorted by impact (highest first)
        """
        scenario = FailureScenario(
            target_id="",
            layer=layer,
            cascade_probability=cascade_probability,
        )
        
        simulator = FailureSimulator(self.graph)
        return simulator.simulate_exhaustive(scenario, layer=layer)
    
    # =========================================================================
    # Layer Analysis
    # =========================================================================
    
    def analyze_layer(self, layer: str = "system") -> LayerMetrics:
        """
        Run combined event and failure analysis for a layer.
        
        Args:
            layer: Analysis layer (app, infra, mw-app, mw-infra, system)
            
        Returns:
            LayerMetrics with combined simulation results
        """
        metrics = LayerMetrics(layer=layer)
        
        # Get components for this layer
        layer_comps = self.graph.get_components_by_layer(layer)
        metrics.total_components = len(layer_comps)
        
        # === Event Simulation ===
        self.logger.info(f"Running event simulation for layer: {layer}")
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
        self.logger.info(f"Running failure simulation for layer: {layer}")
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        
        if failure_results:
            reach_losses = [r.impact.reachability_loss for r in failure_results]
            fragmentations = [r.impact.fragmentation for r in failure_results]
            throughput_losses = [r.impact.throughput_loss for r in failure_results]
            
            metrics.avg_reachability_loss = sum(reach_losses) / len(reach_losses)
            metrics.avg_fragmentation = sum(fragmentations) / len(fragmentations)
            metrics.avg_throughput_loss = sum(throughput_losses) / len(throughput_losses)
            metrics.max_impact = max(r.impact.composite_impact for r in failure_results)
            
            # Count SPOFs (components with cascade > 0)
            metrics.spof_count = sum(1 for r in failure_results if r.impact.cascade_count > 0)
        
        return metrics
    
    # =========================================================================
    # Criticality Classification
    # =========================================================================
    
    def classify_components(
        self,
        layer: str = "system",
        k_factor: float = 1.5
    ) -> List[ComponentCriticality]:
        """
        Classify components by criticality based on simulation results.
        
        Uses box-plot classification on combined event + failure impact.
        
        Args:
            layer: Analysis layer
            k_factor: IQR multiplier for outlier detection
            
        Returns:
            List of ComponentCriticality sorted by combined impact
        """
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
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(
        self,
        layers: Optional[List[str]] = None
    ) -> SimulationReport:
        """
        Generate comprehensive simulation report.
        
        Args:
            layers: Layers to analyze (default: app, infra, system)
            
        Returns:
            SimulationReport with all analyses
        """
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
    
    def export_report(self, report: SimulationReport, output_path: str) -> None:
        """Export report to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Report exported to: {path}")