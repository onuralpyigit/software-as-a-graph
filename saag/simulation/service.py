from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .graph import SimulationGraph
from .event_simulator import EventSimulator
from .failure_simulator import FailureSimulator
from .models import (
    SimulationReport,
    LayerMetrics,
    ComponentCriticality,
    EdgeCriticality,
    ComponentState,
    CascadeRule,
    EventScenario,
    FailureScenario,
    FailureMode,
    FailureResult,
)
from saag.core.ports.graph_repository import IGraphRepository

logger = logging.getLogger(__name__)

class SimulationService:
    """
    Service for running system simulations (events, failures).
    Orchestrates EventSimulator and FailureSimulator directly.
    """

    def __init__(self, repository: IGraphRepository):
        self.repository = repository

    def _get_graph(self) -> SimulationGraph:
        """Helper to fetch data and create SimulationGraph."""
        graph_data = self.repository.get_graph_data(include_raw=True)
        return SimulationGraph(graph_data)

    def run_event_simulation(
        self,
        source_app: str,
        num_messages: int = 100,
        duration: float = 10.0,
        failure_rate: float = 0.0,
        failure_targets: list = None,
        mean_recovery_time: float = 0.0,
        poisson_arrivals: bool = False,
        **kwargs,
    ) -> Any:
        """Run event simulation from a source application."""
        graph = self._get_graph()
        simulator = EventSimulator(graph)

        scenario = EventScenario(
            source_app=source_app,
            num_messages=num_messages,
            duration=duration,
            failure_rate=failure_rate,
            failure_targets=failure_targets,
            mean_recovery_time=mean_recovery_time,
            poisson_arrivals=poisson_arrivals,
        )

        return simulator.simulate(scenario)

    def run_event_simulation_all(
        self,
        num_messages: int = 100,
        duration: float = 10.0,
        layer: str = "system",
        failure_rate: float = 0.0,
        failure_targets: list = None,
        mean_recovery_time: float = 0.0,
        poisson_arrivals: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run event simulation for all publishers."""
        graph = self._get_graph()
        simulator = EventSimulator(graph)

        template = EventScenario(
            source_app="all",
            num_messages=num_messages,
            duration=duration,
            failure_rate=failure_rate,
            failure_targets=failure_targets,
            mean_recovery_time=mean_recovery_time,
            poisson_arrivals=poisson_arrivals,
        )

        return simulator.simulate_all_publishers(template)

    def run_failure_simulation(self, target_ids: List[str], layer: str = "system",
                              cascade_rule: CascadeRule = CascadeRule.ALL,
                              cascade_probability: float = 1.0,
                              library_cascade_probability: Optional[float] = None,
                              propagation_threshold: float = 0.2,
                              failure_mode: FailureMode = FailureMode.CRASH) -> FailureResult:
        """Run a single failure simulation for one or more targets."""
        graph = self._get_graph()
        sim = FailureSimulator(graph, propagation_threshold=propagation_threshold)

        scenario = FailureScenario(
            target_ids=target_ids,
            failure_mode=failure_mode,
            cascade_rule=cascade_rule,
            cascade_probability=cascade_probability,
            library_cascade_probability=library_cascade_probability
        )
        return sim.simulate(scenario)

    def run_failure_simulation_monte_carlo(self, target_id: str, layer: str = "system",
                                          cascade_probability: float = 1.0,
                                          library_cascade_probability: Optional[float] = None,
                                          propagation_threshold: float = 0.2,
                                          failure_mode: FailureMode = FailureMode.CRASH,
                                          n_trials: int = 100) -> Any:
        """Run Monte Carlo failure simulation."""
        graph = self._get_graph()
        sim = FailureSimulator(graph, propagation_threshold=propagation_threshold)

        scenario = FailureScenario(
            target_ids=[target_id],
            failure_mode=failure_mode,
            cascade_probability=cascade_probability,
            library_cascade_probability=library_cascade_probability,
            cascade_rule=CascadeRule.ALL
        )
        return sim.simulate_monte_carlo(scenario, n_trials=n_trials)

    def run_failure_simulation_exhaustive(self, layer: str = "system", cascade_probability: float = 1.0,
                                         library_cascade_probability: Optional[float] = None,
                                         propagation_threshold: float = 0.2,
                                         failure_mode: FailureMode = FailureMode.CRASH,
                                         seed: Optional[int] = 42, **kwargs) -> List[Any]:
        """Run exhaustive failure analysis for all components in a layer.

        `seed` makes the sweep reproducible; pass None for free-running behaviour.
        """
        graph = self._get_graph()
        fail_sim = FailureSimulator(graph, propagation_threshold=propagation_threshold)

        # --- Stage A: Discrete-event baseline flows ---
        event_sim = EventSimulator(graph)
        event_results = event_sim.simulate_all_publishers(
            EventScenario(source_app="all", num_messages=50, duration=5.0)
        )
        all_flows = []
        for res in event_results.values():
            all_flows.extend(res.successful_flows)
        fail_sim.set_baseline_flows(all_flows)

        # --- Stage B + C: Main loop and Post-passes ---
        n_trials = kwargs.get("n_trials", 1)
        return fail_sim.simulate_exhaustive(
            scenario_template=FailureScenario(
                target_ids=["template"],
                failure_mode=failure_mode,
                cascade_probability=cascade_probability,
                library_cascade_probability=library_cascade_probability,
                cascade_rule=CascadeRule.ALL
            ),
            layer=layer,
            n_trials=n_trials,
            seed=seed
        )

    def run_failure_simulation_pairwise(self, layer: str = "system", cascade_probability: float = 1.0,
                                      library_cascade_probability: Optional[float] = None,
                                      propagation_threshold: float = 0.2,
                                      failure_mode: FailureMode = FailureMode.CRASH) -> List[Any]:
        """Run pairwise failure analysis for all component pairs in a layer."""
        graph = self._get_graph()
        sim = FailureSimulator(graph, propagation_threshold=propagation_threshold)

        return sim.simulate_pairwise(
            scenario_template=FailureScenario(
                target_ids=["template"],
                failure_mode=failure_mode,
                cascade_probability=cascade_probability,
                library_cascade_probability=library_cascade_probability,
                cascade_rule=CascadeRule.ALL
            ),
            layer=layer
        )

    def generate_report(self, layers: List[str] = ["app", "infra", "mw", "system"], classify_edges: bool = False) -> SimulationReport:
        """Generate comprehensive simulation report across layers without redundant execution passes."""
        graph = self._get_graph()
        fail_sim = FailureSimulator(graph)
        event_sim = EventSimulator(graph)
        
        report_timestamp = datetime.now().isoformat()
        graph_summary = graph.get_summary()
        
        layer_metrics_map = {}
        all_comp_criticality = []
        all_edge_criticality = []
        
        # Step 1: Establish baseline flows via discrete-event simulation ONCE globally
        event_metrics_system = event_sim.simulate_all_publishers(
             EventScenario(source_app="template", num_messages=50, duration=5.0)
        )
        
        all_flows = []
        for res in event_metrics_system.values():
            all_flows.extend(res.successful_flows)
        fail_sim.set_baseline_flows(all_flows)
        
        # Step 2: Loop layers and calculate execution profiles exactly once per scope
        for layer in layers:
            logger.info(f"Generating simulation metrics for layer scope: {layer}")
            
            # Canonical single-pass exhaustive failure sweep
            fail_results = fail_sim.simulate_exhaustive(layer=layer)
            l_metrics = LayerMetrics(layer=layer)
            
            if fail_results:
                l_metrics.total_components = len(fail_results)
                l_metrics.max_impact = max((r.impact.composite_impact for r in fail_results), default=0.0)
                top_fail = max(fail_results, key=lambda r: r.impact.composite_impact) if fail_results else None
                l_metrics.max_impact_component = top_fail.target_id if top_fail else ""
                
                l_metrics.avg_reachability_loss = sum(r.impact.reachability_loss for r in fail_results) / len(fail_results)
                l_metrics.avg_fragmentation = sum(r.impact.fragmentation for r in fail_results) / len(fail_results)
                l_metrics.avg_throughput_loss = sum(r.impact.throughput_loss for r in fail_results) / len(fail_results)
                
                for r in fail_results:
                    imp = r.impact.composite_impact
                    if imp > 0.8: l_metrics.critical_count += 1
                    elif imp > 0.6: l_metrics.high_count += 1
                    elif imp > 0.4: l_metrics.medium_count += 1
                    elif imp > 0.2: l_metrics.low_count += 1
                    else: l_metrics.minimal_count += 1
                    
                    if r.impact.fragmentation > 0.01:
                        l_metrics.spof_count += 1
                        
            # Aggregate event parameters by layer boundaries
            layer_comps = set(graph.get_components_by_layer(layer))
            layer_event_results = [res for app, res in event_metrics_system.items() if app in layer_comps]
            
            if layer_event_results:
                l_metrics.event_throughput = sum(r.metrics.messages_published for r in layer_event_results)
                l_metrics.event_delivered = sum(r.metrics.messages_delivered for r in layer_event_results)
                l_metrics.event_dropped = sum(r.metrics.messages_dropped for r in layer_event_results)
                
                l_metrics.event_delivery_rate = sum(r.metrics.delivery_rate for r in layer_event_results) / len(layer_event_results)
                l_metrics.event_drop_rate = sum(r.metrics.drop_rate for r in layer_event_results) / len(layer_event_results)
                l_metrics.event_avg_latency_ms = (sum(r.metrics.avg_latency for r in layer_event_results) / len(layer_event_results)) * 1000
                l_metrics.event_p99_latency_ms = (sum(r.metrics.p99_latency for r in layer_event_results) / len(layer_event_results)) * 1000
                l_metrics.event_throughput_per_sec = sum(r.metrics.throughput for r in layer_event_results)

            layer_metrics_map[layer] = l_metrics
            
            # Map failure profiles directly to global criteria metrics safely
            for r in fail_results:
                comp_id = r.target_id
                if any(c.id == comp_id for c in all_comp_criticality):
                    continue  # Ensure idempotency across layered metrics definitions
                
                ev_res = event_metrics_system.get(comp_id)
                ev_impact = 0.0
                throughput = 0
                if ev_res:
                    throughput = ev_res.metrics.messages_published
                    ev_impact = min(1.0, throughput / 1000.0)
                
                fail_imp = r.impact.composite_impact
                combined = (fail_imp * 0.7) + (ev_impact * 0.3)
                
                crit_level = "minimal"
                if combined > 0.8: crit_level = "critical"
                elif combined > 0.6: crit_level = "high"
                elif combined > 0.4: crit_level = "medium"
                elif combined > 0.2: crit_level = "low"
                
                all_comp_criticality.append(ComponentCriticality(
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
                ))

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
            csc_names={c.id: c.properties.get("name", c.id) for c in graph.components.values()},
            library_usage=graph.get_library_usage(),
            node_allocations=graph.get_node_allocations(),
            broker_routing=graph.get_broker_routing()
        )

    def analyze_layer(self, layer: str) -> LayerMetrics:
        """Alias for generate_report for a single layer, for visualization compatibility."""
        report = self.generate_report(layers=[layer])
        return report.layer_metrics.get(layer, LayerMetrics(layer=layer))

    def classify_components(self, layer: str = "system", k_factor: float = 1.5) -> List[Any]:
        """Classify components by criticality based on simulation results."""
        report = self.generate_report(layers=[layer])
        return report.component_criticality

    def classify_edges(self, layer: str = "system", k_factor: float = 1.5) -> List[Any]:
        """Classify edges by criticality."""
        report = self.generate_report(layers=[layer])
        return report.edge_criticality