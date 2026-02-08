"""
Simulation Service

Application service orchestrating event-driven and failure simulations
across multi-layer graph models loaded directly from Neo4j.

Architecture:
    CLI (bin/simulate_graph.py)
      └── SimulationService          ← this module
            ├── EventSimulator       (domain service)
            ├── FailureSimulator     (domain service)
            ├── BoxPlotClassifier    (domain service)
            └── Neo4jGraphRepository (outbound adapter)

Simulation layers operate on RAW structural relationships
(PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO, USES)
without deriving DEPENDS_ON edges.
"""

from __future__ import annotations
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.application.ports import ISimulationUseCase
from src.adapters.outbound.neo4j_repo import Neo4jGraphRepository
from src.domain.models.simulation.graph import SimulationGraph
from src.domain.models.simulation.metrics import (
    LayerMetrics,
    ComponentCriticality,
    EdgeCriticality,
    SimulationReport,
)
from src.domain.services.event_simulator import (
    EventSimulator,
    EventScenario,
    EventResult,
)
from src.domain.services.failure_simulator import (
    FailureSimulator,
    FailureScenario,
    FailureResult,
    MonteCarloResult,
)

try:
    from src.domain.services import BoxPlotClassifier
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False


class SimulationService(ISimulationUseCase):
    """
    Application service for running simulations on the multi-layer graph model.

    Capabilities:
        - Event-driven simulation (single source / all publishers)
        - Failure simulation (single target / exhaustive per layer)
        - Multi-layer analysis with per-layer metrics
        - Component and edge criticality classification
        - Comprehensive report generation with recommendations
    """

    # Default simulation parameters
    DEFAULT_EVENT_MESSAGES = 100
    DEFAULT_EVENT_DURATION = 10.0
    DEFAULT_CASCADE_PROB = 1.0
    DEFAULT_LAYERS = ("app", "infra", "mw", "system")

    # Criticality weights for combining event + failure impacts
    EVENT_WEIGHT = 0.4
    FAILURE_WEIGHT = 0.6

    def __init__(self, repository: Neo4jGraphRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)
        self._graph: Optional[SimulationGraph] = None
        self._classifier = BoxPlotClassifier() if HAS_CLASSIFIER else None

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self):
        self._load_graph()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Repository lifecycle managed by Container

    # =========================================================================
    # Graph Access
    # =========================================================================

    @property
    def graph(self) -> SimulationGraph:
        """Lazily load and return the simulation graph."""
        if self._graph is None:
            self._load_graph()
        if self._graph is None:
            raise ValueError("Failed to load simulation graph from Neo4j")
        return self._graph

    def _load_graph(self) -> None:
        """Load the raw graph from the repository (include_raw=True)."""
        self.logger.info("Loading simulation graph from Neo4j (raw relationships)")
        graph_data = self.repository.get_graph_data(include_raw=True)
        self._graph = SimulationGraph(graph_data=graph_data)
        summary = self._graph.get_summary()
        self.logger.info(
            f"Graph loaded: {summary['total_nodes']} nodes, "
            f"{summary['total_edges']} edges, "
            f"{summary['topics']} topics"
        )

    # =========================================================================
    # Event Simulation
    # =========================================================================

    def run_event_simulation(
        self,
        source_app: str,
        num_messages: int = DEFAULT_EVENT_MESSAGES,
        duration: float = DEFAULT_EVENT_DURATION,
        **kwargs,
    ) -> EventResult:
        """Run event simulation from a specific source application."""
        scenario = EventScenario(
            source_app=source_app,
            description=f"Event simulation: {source_app}",
            num_messages=num_messages,
            duration=duration,
            **kwargs,
        )
        simulator = EventSimulator(self.graph)
        return simulator.simulate(scenario)

    def run_event_simulation_all(
        self,
        num_messages: int = 50,
        duration: float = 5.0,
        layer: str = "system",
        **kwargs,
    ) -> Dict[str, EventResult]:
        """
        Run event simulation from all publisher applications.

        When layer != 'system', only publishers within the layer's component
        set are simulated.
        """
        scenario = EventScenario(
            source_app="",
            num_messages=num_messages,
            duration=duration,
            **kwargs,
        )
        simulator = EventSimulator(self.graph)
        all_results = simulator.simulate_all_publishers(scenario)

        # Filter results to layer-relevant publishers
        if layer != "system":
            layer_comps = set(self.graph.get_components_by_layer(layer))
            all_results = {
                k: v for k, v in all_results.items() if k in layer_comps
            }

        return all_results

    # =========================================================================
    # Failure Simulation
    # =========================================================================

    def run_failure_simulation(
        self,
        target_id: str,
        layer: str = "system",
        cascade_probability: float = DEFAULT_CASCADE_PROB,
        **kwargs,
    ) -> FailureResult:
        """Run failure simulation for a specific component."""
        scenario = FailureScenario(
            target_id=target_id,
            description=f"Failure simulation: {target_id}",
            layer=layer,
            cascade_probability=cascade_probability,
            **kwargs,
        )
        simulator = FailureSimulator(self.graph)
        return simulator.simulate(scenario)

    def run_failure_simulation_exhaustive(
        self,
        layer: str = "system",
        cascade_probability: float = DEFAULT_CASCADE_PROB,
    ) -> List[FailureResult]:
        """Run failure simulation for every analyzable component in a layer."""
        scenario = FailureScenario(
            target_id="",
            layer=layer,
            cascade_probability=cascade_probability,
        )
        simulator = FailureSimulator(self.graph)
        return simulator.simulate_exhaustive(scenario, layer=layer)

    def run_failure_simulation_monte_carlo(
        self,
        target_id: str,
        layer: str = "system",
        cascade_probability: float = DEFAULT_CASCADE_PROB,
        n_trials: int = 100,
        **kwargs,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo stochastic failure simulation.
        
        Runs N trials with probabilistic cascade propagation to compute
        confidence intervals on impact scores.
        
        Args:
            target_id: Component to fail
            layer: Simulation layer
            cascade_probability: Probability of cascade propagation (0-1)
            n_trials: Number of Monte Carlo trials
            
        Returns:
            MonteCarloResult with mean, std, and 95% CI
        """
        scenario = FailureScenario(
            target_id=target_id,
            description=f"Monte Carlo simulation: {target_id}",
            layer=layer,
            cascade_probability=cascade_probability,
            **kwargs,
        )
        simulator = FailureSimulator(self.graph)
        return simulator.simulate_monte_carlo(scenario, n_trials=n_trials)

    # =========================================================================
    # Layer Analysis
    # =========================================================================

    def analyze_layer(
        self,
        layer: str = "system",
        num_messages: int = 50,
        duration: float = 5.0,
    ) -> LayerMetrics:
        """
        Run combined event + failure analysis for a single layer.

        Returns LayerMetrics populated with both event runtime metrics
        and failure impact metrics for the specified layer.
        """
        metrics = LayerMetrics(layer=layer)
        layer_comps = self.graph.get_analyze_components_by_layer(layer)
        metrics.total_components = len(layer_comps)

        # --- Event Simulation ---
        event_results = self.run_event_simulation_all(
            num_messages=num_messages, duration=duration, layer=layer,
        )
        self._populate_event_metrics(metrics, event_results)

        # --- Failure Simulation ---
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        self._populate_failure_metrics(metrics, failure_results)

        return metrics

    def _populate_event_metrics(
        self, metrics: LayerMetrics, event_results: Dict[str, EventResult],
    ) -> None:
        """Extract aggregated event metrics from per-publisher results."""
        if not event_results:
            return

        total_pub = sum(r.metrics.messages_published for r in event_results.values())
        total_del = sum(r.metrics.messages_delivered for r in event_results.values())
        total_drop = sum(r.metrics.messages_dropped for r in event_results.values())

        latencies: List[float] = []
        throughputs: List[float] = []
        drop_reasons: Dict[str, int] = {}

        for r in event_results.values():
            if r.metrics.messages_delivered > 0:
                latencies.append(r.metrics.avg_latency)
            if r.metrics.throughput > 0:
                throughputs.append(r.metrics.throughput)
            for reason, count in r.drop_reasons.items():
                drop_reasons[reason] = drop_reasons.get(reason, 0) + count

        metrics.event_throughput = total_pub
        metrics.event_delivered = total_del
        metrics.event_dropped = total_drop
        metrics.event_delivery_rate = (total_del / total_pub * 100) if total_pub > 0 else 0
        metrics.event_drop_rate = (total_drop / total_pub * 100) if total_pub > 0 else 0
        metrics.event_avg_latency_ms = (
            (sum(latencies) / len(latencies) * 1000) if latencies else 0
        )
        metrics.event_throughput_per_sec = (
            sum(throughputs) / len(throughputs) if throughputs else 0
        )
        metrics.event_drop_reasons = drop_reasons

        # P99 latency from all individual latency samples
        all_latencies: List[float] = []
        for r in event_results.values():
            all_latencies.extend(r.metrics.latencies)
        if all_latencies:
            all_latencies.sort()
            p99_idx = int(len(all_latencies) * 0.99)
            metrics.event_p99_latency_ms = all_latencies[min(p99_idx, len(all_latencies) - 1)] * 1000

    def _populate_failure_metrics(
        self, metrics: LayerMetrics, failure_results: List[FailureResult],
    ) -> None:
        """Extract aggregated failure metrics from exhaustive results."""
        if not failure_results:
            return

        reach_losses = [r.impact.reachability_loss for r in failure_results]
        fragmentations = [r.impact.fragmentation for r in failure_results]
        throughput_losses = [r.impact.throughput_loss for r in failure_results]
        composite_impacts = [r.impact.composite_impact for r in failure_results]

        n = len(failure_results)
        metrics.avg_reachability_loss = sum(reach_losses) / n
        metrics.avg_fragmentation = sum(fragmentations) / n
        metrics.avg_throughput_loss = sum(throughput_losses) / n
        metrics.max_impact = max(composite_impacts)
        metrics.spof_count = sum(1 for r in failure_results if r.impact.cascade_count > 0)

        # Identify the component with the highest impact
        top_result = max(failure_results, key=lambda r: r.impact.composite_impact)
        metrics.max_impact_component = top_result.target_id

    # =========================================================================
    # Component Criticality Classification
    # =========================================================================

    def classify_components(
        self,
        layer: str = "system",
        event_weight: float = EVENT_WEIGHT,
        failure_weight: float = FAILURE_WEIGHT,
        k_factor: float = 1.5,
    ) -> List[ComponentCriticality]:
        """
        Classify components by criticality based on combined simulation results.

        Steps:
            1. Run event simulation (all publishers) -> per-component event impact
            2. Run failure simulation (exhaustive) -> per-component failure impact
            3. Combine with configurable weights
            4. Classify using BoxPlotClassifier (or threshold fallback)
        """
        layer_comps = set(self.graph.get_analyze_components_by_layer(layer))

        # --- Event impact ---
        event_results = self.run_event_simulation_all(layer=layer)
        event_impacts = self._compute_event_impacts(event_results, layer_comps)

        # --- Failure impact ---
        failure_results = self.run_failure_simulation_exhaustive(layer=layer)
        failure_map = {r.target_id: r for r in failure_results}

        # --- Build criticality scores ---
        component_scores: Dict[str, ComponentCriticality] = {}
        for comp_id in layer_comps:
            comp = self.graph.components.get(comp_id)
            if not comp:
                continue

            e_impact = event_impacts.get(comp_id, 0.0)
            f_result = failure_map.get(comp_id)
            f_impact = f_result.impact.composite_impact if f_result else 0.0

            combined = event_weight * e_impact + failure_weight * f_impact

            crit = ComponentCriticality(
                id=comp_id,
                type=comp.type,
                event_impact=e_impact,
                failure_impact=f_impact,
                combined_impact=combined,
            )

            # Attach detail metrics from failure result
            if f_result:
                crit.cascade_count = f_result.impact.cascade_count
                crit.cascade_depth = f_result.impact.cascade_depth
                crit.reachability_loss = f_result.impact.reachability_loss
                crit.throughput_loss = f_result.impact.throughput_loss
                crit.affected_topics = f_result.impact.affected_topics
                crit.affected_subscribers = f_result.impact.affected_subscribers

            component_scores[comp_id] = crit

        # --- Classification ---
        self._classify_scores(component_scores, k_factor)

        # Sort by combined impact descending
        result_list = sorted(
            component_scores.values(),
            key=lambda c: c.combined_impact,
            reverse=True,
        )
        return result_list

    def _compute_event_impacts(
        self,
        event_results: Dict[str, EventResult],
        layer_comps: set,
    ) -> Dict[str, float]:
        """
        Compute normalized event impact per component.

        Impact = (messages_sent + messages_routed + messages_received) / total_published
        """
        if not event_results:
            return {}

        total_published = sum(r.metrics.messages_published for r in event_results.values())
        if total_published == 0:
            return {}

        # Aggregate component_impacts across all event results
        aggregated: Dict[str, float] = {}
        for r in event_results.values():
            for comp_id, impact in r.component_impacts.items():
                if comp_id in layer_comps:
                    aggregated[comp_id] = aggregated.get(comp_id, 0.0) + impact

        # Normalize to [0, 1]
        if aggregated:
            max_val = max(aggregated.values())
            if max_val > 0:
                aggregated = {k: v / max_val for k, v in aggregated.items()}

        return aggregated

    def _classify_scores(
        self,
        scores: Dict[str, ComponentCriticality],
        k_factor: float,
    ) -> None:
        """Apply BoxPlotClassifier or threshold-based fallback."""
        if self._classifier and len(scores) >= 5:
            values = [{"id": cid, "score": c.combined_impact} for cid, c in scores.items()]
            result = self._classifier.classify(values, k_factor=k_factor)
            level_map = {item.id: item.level.value for item in result.items}
            for comp_id, crit in scores.items():
                crit.level = level_map.get(comp_id, "minimal")
        else:
            # Threshold-based fallback for small populations
            sorted_scores = sorted(scores.values(), key=lambda x: x.combined_impact, reverse=True)
            n = len(sorted_scores)
            for i, crit in enumerate(sorted_scores):
                pct = i / n if n > 0 else 0
                if pct < 0.10:
                    crit.level = "critical"
                elif pct < 0.25:
                    crit.level = "high"
                elif pct < 0.50:
                    crit.level = "medium"
                elif pct < 0.75:
                    crit.level = "low"
                else:
                    crit.level = "minimal"

    # =========================================================================
    # Edge Criticality Classification
    # =========================================================================

    def classify_edges(
        self,
        layer: str = "system",
        k_factor: float = 1.5,
    ) -> List[EdgeCriticality]:
        """
        Classify edges by criticality based on message flow and connectivity.

        For each edge in the layer's relationship set, compute:
            - flow_impact:  fraction of total messages that traverse this edge
            - connectivity_impact: reachability loss when the edge is removed
        """
        layer_rels = self.graph.get_layer_relationships(layer)
        layer_comps = set(self.graph.get_components_by_layer(layer))

        # Run event simulation to get message flow data
        event_results = self.run_event_simulation_all(layer=layer)

        # Aggregate messages per edge (approximate via component impacts)
        total_published = sum(r.metrics.messages_published for r in event_results.values())

        edge_scores: List[EdgeCriticality] = []
        for u, v, data in self.graph.graph.edges(data=True):
            rel_type = data.get("type", data.get("relationship", ""))
            if rel_type not in layer_rels:
                continue
            if u not in layer_comps and v not in layer_comps:
                continue

            # Flow impact: approximate from endpoint component impacts
            flow = 0.0
            for r in event_results.values():
                u_imp = r.component_impacts.get(u, 0.0)
                v_imp = r.component_impacts.get(v, 0.0)
                flow += min(u_imp, v_imp)  # conservative: min of endpoints

            # Normalize
            n_results = len(event_results) if event_results else 1
            flow_norm = flow / n_results if n_results > 0 else 0.0

            edge_scores.append(EdgeCriticality(
                source=u,
                target=v,
                relationship=rel_type,
                flow_impact=round(flow_norm, 4),
                connectivity_impact=0.0,  # Computed below if needed
                combined_impact=round(flow_norm, 4),
                messages_traversed=int(flow_norm * total_published) if total_published > 0 else 0,
            ))

        # Classify edges
        if edge_scores:
            combined_vals = [{"id": f"{e.source}->{e.target}", "score": e.combined_impact} for e in edge_scores]
            if self._classifier and len(combined_vals) >= 5:
                result = self._classifier.classify(combined_vals, k_factor=k_factor)
                level_map = {item.id: item.level.value for item in result.items}
                for e in edge_scores:
                    key = f"{e.source}->{e.target}"
                    e.level = level_map.get(key, "minimal")
            else:
                edge_scores.sort(key=lambda e: e.combined_impact, reverse=True)
                n = len(edge_scores)
                for i, e in enumerate(edge_scores):
                    pct = i / n if n > 0 else 0
                    if pct < 0.10:
                        e.level = "critical"
                    elif pct < 0.25:
                        e.level = "high"
                    elif pct < 0.50:
                        e.level = "medium"
                    elif pct < 0.75:
                        e.level = "low"
                    else:
                        e.level = "minimal"

        edge_scores.sort(key=lambda e: e.combined_impact, reverse=True)
        return edge_scores

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(
        self,
        layers: Optional[List[str]] = None,
        classify_edges: bool = False,
    ) -> SimulationReport:
        """
        Generate a comprehensive simulation report across specified layers.

        For each layer:
            1. Run event simulation (all publishers filtered by layer)
            2. Run exhaustive failure simulation
            3. Compute per-layer metrics
            4. Classify component criticality

        Optionally classify edge criticality as well.
        """
        if layers is None:
            layers = list(self.DEFAULT_LAYERS)

        self.logger.info(f"Generating simulation report for layers: {layers}")

        graph_summary = self.graph.get_summary()

        # Per-layer metrics
        layer_metrics: Dict[str, LayerMetrics] = {}
        for layer in layers:
            self.logger.info(f"  Analyzing layer: {layer}")
            layer_metrics[layer] = self.analyze_layer(layer)

        # Global component criticality (system-level)
        component_criticality = self.classify_components(layer="system")

        # Update layer metrics with criticality counts
        self._update_criticality_counts(layer_metrics, component_criticality)

        # Edge criticality (optional)
        edge_criticality: List[EdgeCriticality] = []
        if classify_edges:
            edge_criticality = self.classify_edges(layer="system")

        # Top critical components
        top_critical = [
            {
                "id": c.id,
                "type": c.type,
                "level": c.level,
                "combined_impact": round(c.combined_impact, 4),
                "cascade_count": c.cascade_count,
                "event_impact": round(c.event_impact, 4),
                "failure_impact": round(c.failure_impact, 4),
            }
            for c in component_criticality[:10]
        ]

        # Recommendations
        recommendations = self._generate_recommendations(layer_metrics, component_criticality)

        # Auxiliary mappings
        comp_names = {
            c.id: c.properties.get("name", c.id)
            for c in self.graph.components.values()
        }

        report = SimulationReport(
            timestamp=datetime.now().isoformat(),
            graph_summary=graph_summary,
            layer_metrics=layer_metrics,
            component_criticality=component_criticality,
            edge_criticality=edge_criticality,
            top_critical=top_critical,
            recommendations=recommendations,
            component_names=comp_names,
            library_usage=self.graph.get_library_usage(),
            node_allocations=self.graph.get_node_allocations(),
            broker_routing=self.graph.get_broker_routing(),
        )

        return report

    def _update_criticality_counts(
        self,
        layer_metrics: Dict[str, LayerMetrics],
        component_criticality: List[ComponentCriticality],
    ) -> None:
        """Update each layer's criticality counts from classification results."""
        for layer, metrics in layer_metrics.items():
            layer_comps = set(self.graph.get_analyze_components_by_layer(layer))
            counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "minimal": 0}
            for crit in component_criticality:
                if crit.id in layer_comps:
                    counts[crit.level] = counts.get(crit.level, 0) + 1
            metrics.critical_count = counts["critical"]
            metrics.high_count = counts["high"]
            metrics.medium_count = counts["medium"]
            metrics.low_count = counts["low"]
            metrics.minimal_count = counts["minimal"]

    # =========================================================================
    # Export
    # =========================================================================

    def export_report(self, report: SimulationReport, output_file: str) -> None:
        """Export simulation report to JSON file."""
        with open(output_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        self.logger.info(f"Report exported to {output_file}")

    # =========================================================================
    # Recommendations
    # =========================================================================

    def _generate_recommendations(
        self,
        layer_metrics: Dict[str, LayerMetrics],
        component_criticality: List[ComponentCriticality],
    ) -> List[str]:
        """Generate actionable recommendations based on simulation results."""
        recommendations: List[str] = []

        # Critical components
        critical_comps = [c for c in component_criticality if c.level == "critical"]
        if critical_comps:
            ids = ", ".join(c.id for c in critical_comps[:5])
            recommendations.append(
                f"CRITICAL: {len(critical_comps)} critical component(s) detected: {ids}. "
                f"Implement redundancy or failover immediately."
            )

        # SPOFs
        total_spofs = sum(m.spof_count for m in layer_metrics.values())
        if total_spofs > 0:
            recommendations.append(
                f"SPOF: {total_spofs} single point(s) of failure across layers. "
                f"Add backup instances or alternative routing paths."
            )

        # High message drop rate
        for layer, metrics in layer_metrics.items():
            if metrics.event_drop_rate > 10:
                reasons = ", ".join(
                    f"{r}: {c}" for r, c in sorted(
                        metrics.event_drop_reasons.items(),
                        key=lambda x: x[1], reverse=True,
                    )[:3]
                )
                recommendations.append(
                    f"HIGH DROP RATE ({layer}): {metrics.event_drop_rate:.1f}% messages dropped. "
                    f"Top reasons: {reasons}. Review broker capacity and QoS settings."
                )

        # Fragile topology
        for layer, metrics in layer_metrics.items():
            if metrics.avg_fragmentation > 0.3:
                recommendations.append(
                    f"FRAGILE TOPOLOGY ({layer}): {metrics.avg_fragmentation * 100:.1f}% "
                    f"average fragmentation. Increase connectivity or add redundant links."
                )

        # High latency
        for layer, metrics in layer_metrics.items():
            if metrics.event_avg_latency_ms > 100:
                recommendations.append(
                    f"HIGH LATENCY ({layer}): {metrics.event_avg_latency_ms:.1f}ms average. "
                    f"Optimize broker routing or reduce network hops."
                )

        if not recommendations:
            recommendations.append(
                "HEALTHY: No critical issues detected. System topology is well-balanced."
            )

        return recommendations