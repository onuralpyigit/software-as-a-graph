"""
Simulator Facade

Orchestrates Event and Failure simulations on the raw graph model.
Provides unified API for simulation, reporting, and criticality classification.

Features:
- Event-driven simulation (pub-sub message flow)
- Failure simulation (cascade propagation)
- Multi-layer analysis (application, infrastructure, complete)
- Criticality classification based on simulation results
- Comprehensive reporting and export

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .simulation_graph import SimulationGraph
from .event_simulator import EventSimulator, EventScenario, EventResult, RuntimeMetrics
from .failure_simulator import (
    FailureSimulator, 
    FailureScenario, 
    FailureResult, 
    ImpactMetrics,
    BatchFailureSimulator
)

# Import classifier from analysis module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from analysis.classifier import BoxPlotClassifier, CriticalityLevel, ClassificationResult
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False
    CriticalityLevel = None


@dataclass
class LayerMetrics:
    """Metrics for a single analysis layer."""
    layer: str
    
    # Event simulation metrics
    event_throughput: float = 0.0
    event_delivery_rate: float = 0.0
    event_drop_rate: float = 0.0
    event_avg_latency: float = 0.0
    
    # Failure simulation metrics
    avg_reachability_loss: float = 0.0
    avg_fragmentation: float = 0.0
    avg_throughput_loss: float = 0.0
    max_impact: float = 0.0
    
    # Critical components
    critical_count: int = 0
    high_count: int = 0
    spof_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "event_metrics": {
                "throughput": round(self.event_throughput, 2),
                "delivery_rate_percent": round(self.event_delivery_rate, 2),
                "drop_rate_percent": round(self.event_drop_rate, 2),
                "avg_latency_ms": round(self.event_avg_latency, 3),
            },
            "failure_metrics": {
                "avg_reachability_loss_percent": round(self.avg_reachability_loss * 100, 2),
                "avg_fragmentation_percent": round(self.avg_fragmentation * 100, 2),
                "avg_throughput_loss_percent": round(self.avg_throughput_loss * 100, 2),
                "max_impact": round(self.max_impact, 4),
            },
            "criticality": {
                "critical_count": self.critical_count,
                "high_count": self.high_count,
                "spof_count": self.spof_count,
            },
        }


@dataclass
class ComponentCriticality:
    """Criticality assessment for a component based on simulation."""
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
    """Comprehensive simulation report."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "graph_summary": self.graph_summary,
            "layer_metrics": {
                k: v.to_dict() for k, v in self.layer_metrics.items()
            },
            "component_criticality": [c.to_dict() for c in self.component_criticality],
            "top_critical": self.top_critical,
            "recommendations": self.recommendations,
        }


class Simulator:
    """
    Main simulator facade for pub-sub system analysis.
    
    Orchestrates:
    - Event-driven simulation for throughput/latency analysis
    - Failure simulation for impact/cascade analysis
    - Multi-layer analysis
    - Criticality classification
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
        
        # Classifier instance
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
            description=f"Event simulation from {source_app}",
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
        
        Returns:
            Dict mapping app_id to EventResult
        """
        scenario = EventScenario(
            source_app="",  # Will be overridden
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
        layer: str = "complete",
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7,
        **kwargs
    ) -> FailureResult:
        """
        Run failure simulation for a specific component.
        
        Args:
            target_id: Component ID to fail
            layer: Analysis layer (application, infrastructure, complete)
            cascade_threshold: Weight threshold for cascade propagation
            cascade_probability: Probability of cascade propagation
            **kwargs: Additional FailureScenario parameters
            
        Returns:
            FailureResult with cascade and impact analysis
        """
        scenario = FailureScenario(
            target_id=target_id,
            description=f"Failure simulation: {target_id}",
            layer=layer,
            cascade_threshold=cascade_threshold,
            cascade_probability=cascade_probability,
            **kwargs
        )
        
        simulator = FailureSimulator(self.graph)
        return simulator.simulate(scenario)
    
    def run_failure_simulation_exhaustive(
        self,
        layer: str = "complete",
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7
    ) -> List[FailureResult]:
        """
        Run failure simulation for all components in a layer.
        
        Returns:
            List of FailureResult sorted by impact
        """
        scenario_template = FailureScenario(
            target_id="",  # Will be overridden
            layer=layer,
            cascade_threshold=cascade_threshold,
            cascade_probability=cascade_probability,
        )
        
        simulator = FailureSimulator(self.graph)
        return simulator.simulate_exhaustive(scenario_template)
    
    # =========================================================================
    # Combined Analysis
    # =========================================================================
    
    def analyze_layer(self, layer: str = "complete") -> LayerMetrics:
        """
        Run combined event and failure analysis for a layer.
        
        Args:
            layer: Analysis layer
            
        Returns:
            LayerMetrics with combined simulation results
        """
        metrics = LayerMetrics(layer=layer)
        
        # === Event Simulation ===
        event_results = self.run_event_simulation_all(num_messages=50, duration=5.0)
        
        if event_results:
            total_published = sum(r.metrics.messages_published for r in event_results.values())
            total_delivered = sum(r.metrics.messages_delivered for r in event_results.values())
            total_dropped = sum(r.metrics.messages_dropped for r in event_results.values())
            
            avg_latency_sum = sum(
                r.metrics.avg_latency for r in event_results.values() 
                if r.metrics.messages_delivered > 0
            )
            latency_count = sum(
                1 for r in event_results.values() 
                if r.metrics.messages_delivered > 0
            )
            
            metrics.event_throughput = total_published
            metrics.event_delivery_rate = (total_delivered / total_published * 100) if total_published > 0 else 0
            metrics.event_drop_rate = (total_dropped / total_published * 100) if total_published > 0 else 0
            metrics.event_avg_latency = (avg_latency_sum / latency_count * 1000) if latency_count > 0 else 0
        
        # === Failure Simulation ===
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        
        if failure_results:
            reach_losses = [r.impact.reachability_loss for r in failure_results]
            fragmentations = [r.impact.fragmentation for r in failure_results]
            throughput_losses = [r.impact.throughput_loss for r in failure_results]
            
            metrics.avg_reachability_loss = sum(reach_losses) / len(reach_losses)
            metrics.avg_fragmentation = sum(fragmentations) / len(fragmentations)
            metrics.avg_throughput_loss = sum(throughput_losses) / len(throughput_losses)
            metrics.max_impact = max(r.impact.composite_impact for r in failure_results)
            
            # Count SPOFs (components with high cascade)
            metrics.spof_count = sum(1 for r in failure_results if r.impact.cascade_count > 2)
        
        return metrics
    
    def classify_components(
        self,
        layer: str = "complete",
        k_factor: float = 1.5
    ) -> List[ComponentCriticality]:
        """
        Classify components by criticality based on simulation results.
        
        Uses box-plot classification on combined event + failure impact.
        
        Args:
            layer: Analysis layer
            k_factor: Box-plot IQR multiplier for outlier detection
            
        Returns:
            List of ComponentCriticality sorted by combined impact
        """
        criticality_list = []
        
        # === Collect Event Impacts ===
        event_results = self.run_event_simulation_all(num_messages=50, duration=5.0)
        event_impacts = {}
        
        for app_id, result in event_results.items():
            event_impacts.update(result.component_impacts)
        
        # === Collect Failure Impacts ===
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        failure_impacts = {r.target_id: r.impact.composite_impact for r in failure_results}
        failure_metrics = {r.target_id: r for r in failure_results}
        
        # === Combine Impacts ===
        all_components = set(event_impacts.keys()) | set(failure_impacts.keys())
        
        combined_data = []
        for comp_id in all_components:
            e_impact = event_impacts.get(comp_id, 0.0)
            f_impact = failure_impacts.get(comp_id, 0.0)
            
            # Combined impact: weighted average
            combined = 0.4 * e_impact + 0.6 * f_impact
            
            comp_info = self.graph.components.get(comp_id)
            comp_type = comp_info.type if comp_info else "Unknown"
            
            # Get failure metrics
            f_result = failure_metrics.get(comp_id)
            cascade_count = f_result.impact.cascade_count if f_result else 0
            reach_loss = f_result.impact.reachability_loss if f_result else 0.0
            
            # Get event metrics
            msg_throughput = sum(
                r.metrics.component_messages.get(comp_id, 0)
                for r in event_results.values()
            )
            
            criticality_list.append(ComponentCriticality(
                id=comp_id,
                type=comp_type,
                event_impact=e_impact,
                failure_impact=f_impact,
                combined_impact=combined,
                cascade_count=cascade_count,
                message_throughput=msg_throughput,
                reachability_loss=reach_loss,
            ))
            
            combined_data.append({"id": comp_id, "score": combined})
        
        # === Classify using Box-Plot ===
        if self._classifier and combined_data:
            classifier = BoxPlotClassifier(k_factor=k_factor)
            classification = classifier.classify(combined_data, metric_name="combined_impact")
            
            level_map = {item.id: item.level.value for item in classification.items}
            
            for crit in criticality_list:
                crit.level = level_map.get(crit.id, "minimal")
        
        # Sort by combined impact (highest first)
        criticality_list.sort(key=lambda x: x.combined_impact, reverse=True)
        
        return criticality_list
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self) -> SimulationReport:
        """
        Generate comprehensive simulation report.
        
        Includes:
        - Graph summary
        - Per-layer metrics
        - Component criticality classification
        - Recommendations
        """
        self.logger.info("Generating simulation report...")
        
        # === Graph Summary ===
        graph_summary = self.graph.get_summary()
        
        # === Layer Metrics ===
        layer_metrics = {}
        for layer in ["application", "infrastructure", "complete"]:
            self.logger.info(f"  Analyzing layer: {layer}")
            try:
                layer_metrics[layer] = self.analyze_layer(layer)
            except Exception as e:
                self.logger.warning(f"  Failed to analyze layer {layer}: {e}")
                layer_metrics[layer] = LayerMetrics(layer=layer)
        
        # === Component Criticality ===
        self.logger.info("  Classifying components...")
        criticality = self.classify_components(layer="complete")
        
        # Update layer metrics with classification counts
        for layer, metrics in layer_metrics.items():
            layer_comps = self._get_layer_components(layer)
            metrics.critical_count = sum(
                1 for c in criticality 
                if c.level == "critical" and c.id in layer_comps
            )
            metrics.high_count = sum(
                1 for c in criticality 
                if c.level == "high" and c.id in layer_comps
            )
        
        # === Top Critical Components ===
        top_critical = [
            {
                "id": c.id,
                "type": c.type,
                "level": c.level,
                "combined_impact": round(c.combined_impact, 4),
                "failure_impact": round(c.failure_impact, 4),
                "cascade_count": c.cascade_count,
            }
            for c in criticality[:10]
        ]
        
        # === Recommendations ===
        recommendations = self._generate_recommendations(criticality, layer_metrics)
        
        return SimulationReport(
            timestamp=datetime.now().isoformat(),
            graph_summary=graph_summary,
            layer_metrics=layer_metrics,
            component_criticality=criticality,
            top_critical=top_critical,
            recommendations=recommendations,
        )
    
    def _get_layer_components(self, layer: str) -> Set[str]:
        """Get component IDs for a layer."""
        if layer == "application":
            apps = set(self.graph.get_components_by_type("Application"))
            return apps
        elif layer == "infrastructure":
            nodes = set(self.graph.get_components_by_type("Node"))
            return nodes
        else:
            return set(self.graph.components.keys())
    
    def _generate_recommendations(
        self,
        criticality: List[ComponentCriticality],
        layer_metrics: Dict[str, LayerMetrics]
    ) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Check for critical components
        critical_comps = [c for c in criticality if c.level == "critical"]
        if critical_comps:
            comp_list = ", ".join(c.id for c in critical_comps[:3])
            recommendations.append(
                f"CRITICAL: {len(critical_comps)} components identified as critical outliers "
                f"({comp_list}). Implement redundancy and monitoring."
            )
        
        # Check for high cascade count
        high_cascade = [c for c in criticality if c.cascade_count > 3]
        if high_cascade:
            recommendations.append(
                f"SPOF RISK: {len(high_cascade)} components cause significant cascade failures. "
                f"Consider decoupling and failover mechanisms."
            )
        
        # Check application layer
        app_metrics = layer_metrics.get("application")
        if app_metrics:
            if app_metrics.event_drop_rate > 5:
                recommendations.append(
                    f"MESSAGE DROPS: Application layer shows {app_metrics.event_drop_rate:.1f}% "
                    f"message drop rate. Review broker capacity and topic QoS settings."
                )
            if app_metrics.avg_reachability_loss > 0.2:
                recommendations.append(
                    f"REACHABILITY: Average reachability loss is {app_metrics.avg_reachability_loss * 100:.1f}%. "
                    f"Add redundant publishers and routing paths."
                )
        
        # Check infrastructure layer
        infra_metrics = layer_metrics.get("infrastructure")
        if infra_metrics:
            if infra_metrics.spof_count > 0:
                recommendations.append(
                    f"INFRASTRUCTURE: {infra_metrics.spof_count} infrastructure SPOFs detected. "
                    f"Review node placement and network topology."
                )
        
        if not recommendations:
            recommendations.append(
                "No critical issues detected. System appears resilient to simulated failures."
            )
        
        return recommendations
    
    def export_report(
        self,
        report: SimulationReport,
        output_path: str,
        include_details: bool = True
    ) -> None:
        """
        Export simulation report to JSON file.
        
        Args:
            report: SimulationReport to export
            output_path: Path to output file
            include_details: Whether to include full component list
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = report.to_dict()
        
        if not include_details:
            # Reduce size by limiting component list
            data["component_criticality"] = data["component_criticality"][:50]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Report exported to: {path.absolute()}")
    
    def export_impact_dataset(
        self,
        output_path: str,
        layer: str = "complete"
    ) -> None:
        """
        Export impact dataset for validation.
        
        Creates CSV with component_id, component_type, impact_score.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        
        lines = ["component_id,component_type,impact_score,reachability_loss,cascade_count"]
        
        for r in failure_results:
            lines.append(
                f"{r.target_id},{r.target_type},"
                f"{r.impact.composite_impact:.4f},"
                f"{r.impact.reachability_loss:.4f},"
                f"{r.impact.cascade_count}"
            )
        
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        
        self.logger.info(f"Impact dataset exported to: {path.absolute()}")