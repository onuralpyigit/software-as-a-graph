from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Direct imports from domain models and services
from src.core.graph_exporter import GraphExporter
from src.domain.models.simulation.graph import SimulationGraph
from src.domain.services.event_simulator import EventSimulator, EventScenario
from src.domain.services.failure_simulator import FailureSimulator, FailureScenario, CascadeRule
from src.domain.models.simulation.metrics import (
    SimulationReport, LayerMetrics, ComponentCriticality, EdgeCriticality
)
from src.domain.models.simulation.types import ComponentState

logger = logging.getLogger(__name__)

from src.application.ports.inbound_ports import ISimulationUseCase

class SimulationService(ISimulationUseCase):
    """
    Service for running system simulations (events, failures).
    Orchestrates EventSimulator and FailureSimulator directly.
    """

    def __init__(self, repository):
        self.repository = repository

    def _get_graph(self) -> SimulationGraph:
        """Helper to fetch data and create SimulationGraph."""
        graph_data = self.repository.get_graph_data(include_raw=True)
        return SimulationGraph(graph_data)

    def run_event_simulation(self, source_app: str, num_messages: int = 100, duration: float = 10.0, **kwargs) -> Any:
        """Run event simulation from a source application."""
        graph = self._get_graph()
        simulator = EventSimulator(graph)
        
        scenario = EventScenario(
            source_app=source_app,
            num_messages=num_messages,
            duration=duration
        )
        
        result = simulator.simulate(scenario)
        return result

    def run_event_simulation_all(self, num_messages: int = 100, duration: float = 10.0, layer: str = "system", **kwargs) -> Dict[str, Any]:
        """Run event simulation for all publishers."""
        graph = self._get_graph()
        simulator = EventSimulator(graph)
        
        results = simulator.simulate_all_publishers(
            EventScenario(source_app="all", num_messages=num_messages, duration=duration)
        )
        return results

    def run_failure_simulation(self, target_id: str, layer: str = "system", cascade_probability: float = 1.0, **kwargs) -> Any:
        """Run failure simulation for a target component."""
        graph = self._get_graph()
        simulator = FailureSimulator(graph)
        
        scenario = FailureScenario(
            target_id=target_id,
            layer=layer,
            cascade_probability=cascade_probability,
            cascade_rule=CascadeRule.ALL
        )
        
        result = simulator.simulate(scenario)
        return result

    def run_failure_simulation_monte_carlo(self, target_id: str, layer: str = "system", cascade_probability: float = 1.0, n_trials: int = 100, **kwargs) -> Any:
        """Run Monte Carlo failure simulation."""
        graph = self._get_graph()
        simulator = FailureSimulator(graph)
        
        result = simulator.simulate_monte_carlo(
            target_id=target_id,
            layer=layer,
            cascade_probability=cascade_probability,
            n_trials=n_trials
        )
        return result

    def run_failure_simulation_exhaustive(self, layer: str = "system", cascade_probability: float = 1.0, **kwargs) -> List[Any]:
        """Run exhaustive failure analysis for all components in a layer."""
        graph = self._get_graph()
        simulator = FailureSimulator(graph)
        
        results = simulator.simulate_exhaustive(
            scenario_template=FailureScenario(
                target_id="template", # ignored
                cascade_probability=cascade_probability,
                cascade_rule=CascadeRule.ALL
            ),
            layer=layer
        )
        return results

    def generate_report(self, layers: List[str] = ["app", "infra", "mw", "system"], classify_edges: bool = False) -> SimulationReport:
        """Generate comprehensive simulation report."""
        graph = self._get_graph()
        fail_sim = FailureSimulator(graph)
        event_sim = EventSimulator(graph)
        
        report_timestamp = datetime.now().isoformat()
        
        # 1. Graph Summary
        graph_summary = graph.get_summary()
        
        # 2. Per-layer analysis
        layer_metrics_map = {}
        all_comp_criticality = []
        all_edge_criticality = [] # Simulation doesn't explicitly score edges yet, keeping empty or inferred
        
        for layer in layers:
            logger.info(f"Generating simulation report for layer: {layer}")
            
            # --- Failure Simulation ---
            fail_results = fail_sim.simulate_exhaustive(layer=layer)
            
            # --- Event Simulation ---
            # Run a standard event scenario for all publishers in this layer to gauge throughput/latency
            # Only relevant if layer has publishers (app, mw, system)
            event_results_map = {}
            if layer in ["app", "mw", "system"]:
                event_template = EventScenario(source_app="template", num_messages=50, duration=5.0)
                # Filter to run only for publishers IN this layer? 
                # simulate_all_publishers runs for ALL in graph. We can filter results later.
                # To save time, we might want to run it once for the whole graph and reuse.
                # For now, let's run it once per layer loop but that's inefficient.
                # Better: Run once outside loop.
                pass 

        # Optimization: Run event sim once for the whole system
        event_metrics_system = event_sim.simulate_all_publishers(
             EventScenario(source_app="template", num_messages=50, duration=5.0)
        )
        
        # Now populate layer metrics
        for layer in layers:
            # Failure metrics aggregation
            # We need to re-fetch failure results if not cached? 
            # Well, simulate_exhaustive is fast enough if baseline is cached (which it handles internally per call)
            # But we called it above inside the loop.
            fail_results = fail_sim.simulate_exhaustive(layer=layer)
            
            l_metrics = LayerMetrics(layer=layer)
            
            # Failure aggregates
            if fail_results:
                l_metrics.total_components = len(fail_results)
                l_metrics.max_impact = max((r.impact.composite_impact for r in fail_results), default=0.0)
                top_fail = max(fail_results, key=lambda r: r.impact.composite_impact) if fail_results else None
                l_metrics.max_impact_component = top_fail.target_id if top_fail else ""
                
                l_metrics.avg_reachability_loss = sum(r.impact.reachability_loss for r in fail_results) / len(fail_results)
                l_metrics.avg_fragmentation = sum(r.impact.fragmentation for r in fail_results) / len(fail_results)
                l_metrics.avg_throughput_loss = sum(r.impact.throughput_loss for r in fail_results) / len(fail_results)
                
                # Criticality counts based on failure impact thresholds (approximate)
                for r in fail_results:
                    imp = r.impact.composite_impact
                    if imp > 0.8: l_metrics.critical_count += 1
                    elif imp > 0.6: l_metrics.high_count += 1
                    elif imp > 0.4: l_metrics.medium_count += 1
                    elif imp > 0.2: l_metrics.low_count += 1
                    else: l_metrics.minimal_count += 1
                    
                    # SPOF check (fragmentation > 0 means it broke the graph)
                    if r.impact.fragmentation > 0.01:
                        l_metrics.spof_count += 1
                        
            # Event aggregates (filter system results by layer)
            layer_comps = set(graph.get_components_by_layer(layer))
            layer_event_results = [res for app, res in event_metrics_system.items() if app in layer_comps]
            
            if layer_event_results:
                l_metrics.event_throughput = sum(r.metrics.messages_published for r in layer_event_results)
                l_metrics.event_delivered = sum(r.metrics.messages_delivered for r in layer_event_results)
                l_metrics.event_dropped = sum(r.metrics.messages_dropped for r in layer_event_results)
                
                avg_delivery = sum(r.metrics.delivery_rate for r in layer_event_results) / len(layer_event_results)
                l_metrics.event_delivery_rate = avg_delivery
                
                avg_drop = sum(r.metrics.drop_rate for r in layer_event_results) / len(layer_event_results)
                l_metrics.event_drop_rate = avg_drop
                
                # Latency averages
                total_latency = sum(r.metrics.avg_latency for r in layer_event_results)
                l_metrics.event_avg_latency_ms = (total_latency / len(layer_event_results)) * 1000
                
                total_p99 = sum(r.metrics.p99_latency for r in layer_event_results)
                l_metrics.event_p99_latency_ms = (total_p99 / len(layer_event_results)) * 1000
                
                l_metrics.event_throughput_per_sec = sum(r.metrics.throughput for r in layer_event_results)

            layer_metrics_map[layer] = l_metrics
            
            # Build Component Criticality objects
            # We map failure results -> ComponentCriticality
            for r in fail_results:
                comp_id = r.target_id
                
                # Find corresponding event result
                ev_res = event_metrics_system.get(comp_id)
                ev_impact = 0.0
                throughput = 0
                if ev_res:
                    # Normalize event impact based on message volume/throughput relative to others?
                    # For now just use delivery rate as inverse impact? No, that's quality.
                    # Impact is "how important is this component".
                    # In simulation, importance is how many messages it handles.
                    throughput = ev_res.metrics.messages_published
                    ev_impact = min(1.0, throughput / 1000.0) # Normalize arbitrary cap
                
                fail_imp = r.impact.composite_impact
                combined = (fail_imp * 0.7) + (ev_impact * 0.3) # Heavy weight on failure impact
                
                crit_level = "minimal"
                if combined > 0.8: crit_level = "critical"
                elif combined > 0.6: crit_level = "high"
                elif combined > 0.4: crit_level = "medium"
                elif combined > 0.2: crit_level = "low"
                
                cc = ComponentCriticality(
                    id=comp_id,
                    type=r.target_type,
                    event_impact=ev_impact,
                    failure_impact=fail_imp,
                    combined_impact=combined,
                    level=crit_level,
                    cascade_count=r.impact.cascade_count,
                    cascade_depth=r.impact.cascade_depth,
                    message_throughput=throughput,
                    reachability_loss=r.impact.reachability_loss,
                    throughput_loss=r.impact.throughput_loss,
                    affected_topics=r.impact.affected_topics,
                    affected_subscribers=r.impact.affected_subscribers
                )
                
                # Avoid duplicates if component appears in multiple layers (unlikely with sets but possible)
                if not any(c.id == comp_id for c in all_comp_criticality):
                    all_comp_criticality.append(cc)

        # Sort top critical
        all_comp_criticality.sort(key=lambda c: c.combined_impact, reverse=True)
        top_critical = [c.to_dict() for c in all_comp_criticality[:10]]
        
        # Recommendations
        recommendations = []
        if any(c.reachability_loss > 0.5 for c in all_comp_criticality):
            recommendations.append("High reachability loss detected. Consider adding redundant paths for critical flows.")
        if any(c.cascade_depth > 5 for c in all_comp_criticality):
            recommendations.append("Deep failure cascades detected. Implement circuit breakers to limit propagation.")
        
        return SimulationReport(
            timestamp=report_timestamp,
            graph_summary=graph_summary,
            layer_metrics=layer_metrics_map,
            component_criticality=all_comp_criticality,
            edge_criticality=all_edge_criticality,
            top_critical=top_critical,
            recommendations=recommendations,
            # Auxiliary data
            component_names={c.id: c.properties.get("name", c.id) for c in graph.components.values()},
            library_usage=graph.get_library_usage(),
            node_allocations=graph.get_node_allocations(),
            broker_routing=graph.get_broker_routing()
        )
    def classify_components(self, layer: str = "system", k_factor: float = 1.5) -> List[Any]:
        """Classify components by criticality based on simulation results."""
        report = self.generate_report(layers=[layer])
        return report.component_criticality

    def classify_edges(self, layer: str = "system", k_factor: float = 1.5) -> List[Any]:
        """Classify edges by criticality."""
        report = self.generate_report(layers=[layer])
        return report.edge_criticality
