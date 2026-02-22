"""
Failure Simulator

Simulates component failures and their cascading effects on the system.
Works directly on RAW structural relationships without DEPENDS_ON derivation.

Impact Metrics:
    - Reachability Loss: Percentage of broken pub-sub paths (broker-aware)
    - Infrastructure Fragmentation: Graph connectivity loss (connected components)
    - Throughput Loss: QoS-weighted reduction in message delivery capacity
    - Cascade Count: Number of components affected by cascade

Cascade Rules:
    - Physical: Node failure -> hosted components fail (RUNS_ON)
    - Logical: Broker failure -> topics become unreachable;
               Publisher failure -> subscriber starvation
    - Network: Network partition via CONNECTS_TO
"""

from __future__ import annotations
import logging
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict

from .graph import SimulationGraph
from .models import ComponentState, FailureMode, CascadeRule, FailureScenario, ImpactMetrics, CascadeEvent, FailureResult, MonteCarloResult

class FailureSimulator:
    """
    Simulates component failures and cascade propagation.
    
    Works on RAW structural relationships:
        - RUNS_ON: Physical cascade (node -> hosted components)
        - ROUTES: Logical cascade (broker -> topics -> subscriber starvation)
        - CONNECTS_TO: Network cascade (node -> connected nodes)
        - PUBLISHES_TO / SUBSCRIBES_TO: Application cascade (publisher loss)
    
    Example:
        >>> graph = SimulationGraph(graph_data=data)
        >>> sim = FailureSimulator(graph)
        >>> result = sim.simulate(FailureScenario(target_id="Broker1"))
        >>> print(f"Impact: {result.impact.composite_impact}")
    """
    
    def __init__(self, graph: SimulationGraph):
        """
        Initialize the failure simulator.
        
        Args:
            graph: SimulationGraph instance
        """
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        
        # Random generator
        self._rng = random.Random()
        
        # Baseline metrics (computed once per exhaustive run, or per simulate call)
        self._initial_paths: int = 0
        self._initial_components: int = 0
        self._initial_connected_components: int = 1
        self._initial_total_weight: float = 0.0
        self._baseline_computed: bool = False
    
    def simulate(self, scenario: FailureScenario) -> FailureResult:
        """
        Run a failure simulation.
        
        Args:
            scenario: Configuration for the simulation
            
        Returns:
            FailureResult with impact metrics and cascade analysis
        """
        # Reset graph state
        self.graph.reset()
        
        if scenario.seed is not None:
            self._rng.seed(scenario.seed)
        
        # Validate targets
        valid_targets = []
        for tid in scenario.target_ids:
            if tid in self.graph.components:
                valid_targets.append(tid)
            else:
                self.logger.warning(f"Target '{tid}' not found, skipping.")
        
        if not valid_targets:
            return self._empty_result_multi(scenario, "No valid targets found")
        
        self.logger.info(f"Simulating failure: {valid_targets}")
        
        # Capture initial state (skip if already cached by exhaustive run)
        if not self._baseline_computed:
            self._compute_baseline()
        
        # Fail the targets
        failed_set = set()
        cascade_sequence = []
        for tid in valid_targets:
            target_comp = self.graph.components[tid]
            self.graph.fail_component(tid)
            failed_set.add(tid)
            cascade_sequence.append(CascadeEvent(
                component_id=tid,
                component_type=target_comp.type,
                cause="initial_failure",
                depth=0
            ))
        
        # Propagate cascade starting from all initial failed targets
        max_depth = self._propagate_cascade_multi(
            scenario, 
            valid_targets, 
            failed_set, 
            cascade_sequence
        )
        
        # Calculate impact metrics
        # For multi-target, we use the first target as the primary identifier if needed
        primary_target = valid_targets[0]
        impact = self._calculate_impact(primary_target, failed_set)
        impact.cascade_count = len(failed_set) - len(valid_targets)
        impact.cascade_depth = max_depth
        
        # Calculate per-layer impacts
        layer_impacts = self._calculate_layer_impacts(failed_set)
        
        # Determine directly related components (combined list)
        related = []
        for tid in valid_targets:
            comp = self.graph.components[tid]
            related.extend(self._get_related_components(tid, comp.type))

        return FailureResult(
            target_id="+".join(valid_targets), # Combined ID for multi-failure
            target_type="Multi" if len(valid_targets) > 1 else self.graph.components[valid_targets[0]].type,
            scenario=scenario.description or f"Failure: {', '.join(valid_targets)}",
            impact=impact,
            cascaded_failures=[c for c in failed_set if c not in valid_targets],
            cascade_sequence=cascade_sequence,
            layer_impacts=layer_impacts,
            related_components=related,
            component_names={c.id: c.properties.get("name", c.id) for c in self.graph.components.values()},
        )
    
    def simulate_exhaustive(
        self,
        scenario_template: Optional[FailureScenario] = None,
        layer: str = "system"
    ) -> List[FailureResult]:
        """
        Run failure simulation for all components in a layer.
        
        Computes baseline once and reuses it across all simulations
        for efficiency.
        
        Args:
            scenario_template: Base scenario configuration
            layer: Layer to analyze
            
        Returns:
            List of FailureResult sorted by impact (highest first)
        """
        results = []
        
        # Get components to analyze for the layer
        component_ids = self.graph.get_analyze_components_by_layer(layer)
        
        self.logger.info(f"Running exhaustive failure analysis: {len(component_ids)} components in layer '{layer}'")
        
        # Compute baseline once (C5 fix: avoid recomputing per simulation)
        self.graph.reset()
        self._compute_baseline()
        self._baseline_computed = True
        
        try:
            for comp_id in component_ids:
                scenario = FailureScenario(
                    target_ids=[comp_id],
                    description=f"Exhaustive failure: {comp_id}",
                    layer=layer,
                    cascade_rule=scenario_template.cascade_rule if scenario_template else CascadeRule.ALL,
                    cascade_probability=scenario_template.cascade_probability if scenario_template else 1.0,
                    max_cascade_depth=scenario_template.max_cascade_depth if scenario_template else 10,
                )
                
                result = self.simulate(scenario)
                results.append(result)
        finally:
            # Always clear the cached baseline flag
            self._baseline_computed = False
        
        # Sort by composite impact (highest first)
        results.sort(key=lambda r: r.impact.composite_impact, reverse=True)
        
        return results

    def simulate_pairwise(
        self,
        scenario_template: Optional[FailureScenario] = None,
        layer: str = "app"
    ) -> List[FailureResult]:
        """
        Run pairwise failure simulation for components in a layer.
        
        Simulates initial failure of all pairs (v1, v2) to detect
        superadditive impact and redundancy failure.
        
        Args:
            scenario_template: Base scenario configuration
            layer: Layer to analyze
            
        Returns:
            List of FailureResult sorted by joint impact
        """
        results = []
        component_ids = self.graph.get_analyze_components_by_layer(layer)
        n = len(component_ids)
        
        self.logger.info(f"Running pairwise failure analysis: {n*(n-1)//2} pairs in layer '{layer}'")
        
        self.graph.reset()
        self._compute_baseline()
        self._baseline_computed = True
        
        try:
            for i in range(n):
                for j in range(i + 1, n):
                    v1, v2 = component_ids[i], component_ids[j]
                    scenario = FailureScenario(
                        target_ids=[v1, v2],
                        description=f"Pairwise failure: {v1}+{v2}",
                        layer=layer,
                        cascade_rule=scenario_template.cascade_rule if scenario_template else CascadeRule.ALL,
                        cascade_probability=scenario_template.cascade_probability if scenario_template else 1.0,
                    )
                    
                    result = self.simulate(scenario)
                    results.append(result)
        finally:
            self._baseline_computed = False
            
        results.sort(key=lambda r: r.impact.composite_impact, reverse=True)
        return results
    
    def simulate_monte_carlo(
        self,
        scenario: FailureScenario,
        n_trials: int = 100,
    ) -> MonteCarloResult:
        """
        Run N stochastic simulations with cascade_probability < 1.0
        and return the distribution of I(v).
        
        Useful for generating confidence intervals on impact scores
        when cascade propagation is probabilistic.
        
        Args:
            scenario: Base scenario (cascade_probability should be < 1.0)
            n_trials: Number of Monte Carlo trials
            
        Returns:
            MonteCarloResult with mean, std, and 95% CI
        """
        impacts: List[float] = []
        
        for trial in range(n_trials):
            trial_scenario = FailureScenario(
                target_id=scenario.target_id,
                description=f"Monte Carlo trial {trial}",
                failure_mode=scenario.failure_mode,
                cascade_rule=scenario.cascade_rule,
                cascade_probability=scenario.cascade_probability,
                max_cascade_depth=scenario.max_cascade_depth,
                layer=scenario.layer,
                seed=trial,
            )
            result = self.simulate(trial_scenario)
            impacts.append(result.impact.composite_impact)
        
        sorted_impacts = sorted(impacts)
        ci_low = sorted_impacts[max(0, int(0.025 * n_trials))]
        ci_high = sorted_impacts[min(n_trials - 1, int(0.975 * n_trials))]
        
        return MonteCarloResult(
            target_id=scenario.target_id,
            n_trials=n_trials,
            mean_impact=statistics.mean(impacts),
            std_impact=statistics.stdev(impacts) if n_trials > 1 else 0.0,
            ci_95=(ci_low, ci_high),
            trial_impacts=impacts,
        )
    
    def _compute_baseline(self) -> None:
        """Compute and cache baseline metrics from the current (healthy) graph state."""
        self._initial_paths = len(self.graph.get_pub_sub_paths(active_only=True))
        self._initial_components = len([
            c for c in self.graph.components.values()
            if c.type in ("Application", "Broker", "Node") and c.state == ComponentState.ACTIVE
        ])
        self._initial_connected_components = self.graph.count_active_connected_components()
        self._initial_total_weight = self._compute_total_topic_weight()
    
    def _compute_total_topic_weight(self) -> float:
        """Compute total QoS-weighted topic capacity."""
        total = 0.0
        for topic_id, topic_info in self.graph.topics.items():
            total += getattr(topic_info, 'weight', 1.0)
        return total if total > 0 else float(len(self.graph.topics))
    
    def _propagate_cascade_multi(
        self,
        scenario: FailureScenario,
        initial_targets: List[str],
        failed_set: Set[str],
        cascade_sequence: List[CascadeEvent]
    ) -> int:
        """
        Propagate failure cascade from multiple initial targets.
        """
        max_depth = 0
        queue: List[Tuple[str, int]] = [(tid, 0) for tid in initial_targets]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= scenario.max_cascade_depth:
                continue
            
            max_depth = max(max_depth, depth)
            current_comp = self.graph.components.get(current_id)
            if not current_comp:
                continue
            
            current_type = current_comp.type
            
            # === Physical Cascade (Node -> Hosted Components) ===
            if scenario.cascade_rule in (CascadeRule.PHYSICAL, CascadeRule.ALL):
                if current_type == "Node":
                    hosted = self.graph.get_hosted_components(current_id)
                    for comp_id in hosted:
                        if comp_id not in failed_set:
                            if self._rng.random() < scenario.cascade_probability:
                                failed_set.add(comp_id)
                                self.graph.fail_component(comp_id)
                                comp = self.graph.components.get(comp_id)
                                cascade_sequence.append(CascadeEvent(
                                    component_id=comp_id,
                                    component_type=comp.type if comp else "Unknown",
                                    cause=f"hosted_on:{current_id}",
                                    depth=depth + 1
                                ))
                                queue.append((comp_id, depth + 1))
            
            # === Logical Cascade ===
            if scenario.cascade_rule in (CascadeRule.LOGICAL, CascadeRule.ALL):
                # Broker failure -> Topics with no remaining routing brokers
                if current_type == "Broker":
                    for topic_id, brokers in self.graph._routing.items():
                        if current_id in brokers:
                            # Check if topic still has active brokers
                            active_brokers = [
                                b for b in brokers 
                                if b not in failed_set and self.graph.is_active(b)
                            ]
                            
                            if not active_brokers:
                                # Topic has no routing path — mark as failed
                                if topic_id not in failed_set:
                                    if self._rng.random() < scenario.cascade_probability:
                                        failed_set.add(topic_id)
                                        self.graph.fail_component(topic_id)
                                        cascade_sequence.append(CascadeEvent(
                                            component_id=topic_id,
                                            component_type="Topic",
                                            cause=f"no_active_brokers:{current_id}",
                                            depth=depth + 1
                                        ))
                                        queue.append((topic_id, depth + 1))
                
                # Application failure -> check for publisher/subscriber starvation
                if current_type == "Application":
                    publishes_to, subscribes_from = self.graph.get_app_topics(current_id)
                    
                    # Publisher starvation
                    for topic_id in publishes_to:
                        active_publishers = self.graph.get_publishers(topic_id)
                        if not active_publishers and topic_id not in failed_set:
                            if self._rng.random() < scenario.cascade_probability:
                                failed_set.add(topic_id)
                                self.graph.fail_component(topic_id)
                                cascade_sequence.append(CascadeEvent(
                                    component_id=topic_id,
                                    component_type="Topic",
                                    cause=f"publisher_starvation:{current_id}",
                                    depth=depth + 1
                                ))
                                queue.append((topic_id, depth + 1))
                    
                    # Subscriber starvation
                    for topic_id in subscribes_from:
                        active_subscribers = self.graph.get_subscribers(topic_id)
                        if not active_subscribers and topic_id not in failed_set:
                            if self._rng.random() < scenario.cascade_probability:
                                failed_set.add(topic_id)
                                self.graph.fail_component(topic_id)
                                cascade_sequence.append(CascadeEvent(
                                    component_id=topic_id,
                                    component_type="Topic",
                                    cause=f"subscriber_starvation:{current_id}",
                                    depth=depth + 1
                                ))
                                queue.append((topic_id, depth + 1))
            
            # === Network Cascade (Node -> Connected Nodes) ===
            if scenario.cascade_rule in (CascadeRule.NETWORK, CascadeRule.ALL):
                if current_type == "Node":
                    connected = self.graph.get_connected_nodes(current_id)
                    for neighbor_id in connected:
                        if neighbor_id not in failed_set:
                            # Check if neighbor becomes isolated (no remaining connections)
                            neighbor_connections = self.graph.get_connected_nodes(neighbor_id)
                            remaining = [
                                n for n in neighbor_connections 
                                if n not in failed_set and n != current_id
                            ]
                            if not remaining:
                                # Neighbor is isolated — mark as partitioned
                                if self._rng.random() < scenario.cascade_probability:
                                    failed_set.add(neighbor_id)
                                    self.graph.fail_component(neighbor_id)
                                    cascade_sequence.append(CascadeEvent(
                                        component_id=neighbor_id,
                                        component_type="Node",
                                        cause=f"network_partition:{current_id}",
                                        depth=depth + 1
                                    ))
                                    queue.append((neighbor_id, depth + 1))
        
        return max_depth
    
    def _calculate_impact(
        self,
        target_id: str,
        failed_set: Set[str]
    ) -> ImpactMetrics:
        """
        Calculate impact metrics after failure cascade.
        
        Three orthogonal dimensions:
        
        1. Reachability Loss — fraction of deliverable pub-sub paths broken.
           A path (publisher → topic → subscriber) is deliverable only when
           publisher, subscriber, AND at least one routing broker are all active.
        
        2. Fragmentation — normalized increase in weakly-connected components
           of the active subgraph. Measures topology disruption rather than
           simple component loss (which would overlap with reachability).
        
        3. Throughput Loss — QoS-weighted fraction of topic capacity lost.
           Higher-weight topics (e.g. safety-critical with high reliability
           QoS) contribute more to the loss than low-priority topics.
        """
        # === Reachability Loss (broker-aware) ===
        remaining_paths = len(self.graph.get_pub_sub_paths(active_only=True))
        
        if self._initial_paths > 0:
            reachability_loss = 1.0 - (remaining_paths / self._initial_paths)
        else:
            reachability_loss = 0.0
        
        # === Fragmentation (connected components) ===
        final_cc = self.graph.count_active_connected_components()
        initial_cc = self._initial_connected_components
        
        # Normalize: how many new disconnected islands were created,
        # relative to the maximum possible fragmentation
        if self._initial_components > 1:
            # Max new components = initial_components - 1 (each node becomes its own island)
            max_new_cc = self._initial_components - 1
            new_cc = max(0, final_cc - initial_cc)
            fragmentation = min(1.0, new_cc / max_new_cc)
        else:
            fragmentation = 0.0
        
        # === Throughput Loss (QoS-weighted) ===
        total_weight = self._initial_total_weight
        lost_weight = 0.0
        affected_topics = 0
        
        for topic_id, topic_info in self.graph.topics.items():
            topic_weight = getattr(topic_info, 'weight', 1.0)
            
            publishers = self.graph.get_publishers(topic_id)
            brokers = self.graph.get_routing_brokers(topic_id)
            subscribers = self.graph.get_subscribers(topic_id)
            
            if not publishers or not brokers or not subscribers:
                lost_weight += topic_weight
                affected_topics += 1
        
        if total_weight > 0:
            throughput_loss = lost_weight / total_weight
        else:
            throughput_loss = 0.0
        
        # === Infrastructure stats ===
        remaining_active = len([
            c for c in self.graph.components.values()
            if c.type in ("Application", "Broker", "Node")
            and c.state == ComponentState.ACTIVE
        ])
        failed_count = self._initial_components - remaining_active
        
        # === Affected Entities ===
        affected_pubs: Set[str] = set()
        affected_subs: Set[str] = set()
        
        for topic_id in self.graph.topics:
            publishers = self.graph.get_publishers(topic_id)
            brokers = self.graph.get_routing_brokers(topic_id)
            subscribers = self.graph.get_subscribers(topic_id)
            
            # Topic is affected if any part of the delivery chain is broken
            if not publishers or not brokers or not subscribers:
                # Track all parties on the broken topic
                all_pubs = self.graph._publishers.get(topic_id, [])
                all_subs = self.graph._subscribers.get(topic_id, [])
                affected_pubs.update(all_pubs)
                affected_subs.update(all_subs)
        
        # === Cascade by Type ===
        cascade_by_type: Dict[str, int] = defaultdict(int)
        for comp_id in failed_set:
            if comp_id == target_id:
                continue
            comp = self.graph.components.get(comp_id)
            if comp:
                cascade_by_type[comp.type] += 1
        
        return ImpactMetrics(
            initial_paths=self._initial_paths,
            remaining_paths=remaining_paths,
            reachability_loss=reachability_loss,
            initial_components=self._initial_components,
            failed_components=failed_count,
            initial_connected_components=initial_cc,
            final_connected_components=final_cc,
            fragmentation=fragmentation,
            initial_throughput=total_weight,
            remaining_throughput=total_weight - lost_weight,
            throughput_loss=throughput_loss,
            affected_topics=affected_topics,
            affected_subscribers=len(affected_subs),
            affected_publishers=len(affected_pubs),
            cascade_by_type=dict(cascade_by_type),
        )
    
    def _calculate_layer_impacts(self, failed_set: Set[str]) -> Dict[str, float]:
        """Calculate impact per analysis layer."""
        layer_impacts = {}
        
        layers = ["app", "infra", "mw", "system"]
        
        for layer in layers:
            layer_comps = set(self.graph.get_components_by_layer(layer))
            if not layer_comps:
                layer_impacts[layer] = 0.0
                continue
            
            # Compute impact as fraction of layer components affected
            affected = failed_set & layer_comps
            layer_impacts[layer] = len(affected) / len(layer_comps)
        
        return layer_impacts
    
    def _get_related_components(self, target_id: str, target_type: str) -> List[str]:
        """Determine directly related components for context in results."""
        related = []
        if target_type == "Application":
            lib_ids = self.graph.get_library_usage().get(target_id, [])
            for lid in lib_ids:
                lcomp = self.graph.components.get(lid)
                name = lcomp.properties.get("name", lid) if lcomp else lid
                if lcomp and "version" in lcomp.properties:
                    name += f" ({lcomp.properties['version']})"
                related.append(f"Uses Lib: {name}")
        elif target_type == "Node":
            hosted_ids = self.graph.get_node_allocations().get(target_id, [])
            for hid in hosted_ids:
                hcomp = self.graph.components.get(hid)
                name = hcomp.properties.get("name", hid) if hcomp else hid
                related.append(f"Hosts: {name}")
        elif target_type == "Broker":
            topic_ids = self.graph.get_broker_routing().get(target_id, [])
            for tid in topic_ids:
                topic = self.graph.topics.get(tid)
                name = topic.name if topic else tid
                related.append(f"Routes: {name}")
        return related
    
    def _empty_result(self, scenario: FailureScenario, reason: str) -> FailureResult:
        """Create an empty result for failed simulations."""
        return FailureResult(
            target_id=scenario.target_id,
            target_type="Unknown",
            scenario=reason,
            impact=ImpactMetrics(),
        )