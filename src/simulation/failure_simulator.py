"""
Failure Simulator

Simulates component failures and their cascading effects on the system.
Works directly on RAW structural relationships without DEPENDS_ON.

Impact Metrics:
- Reachability Loss: Percentage of broken pub-sub paths
- Infrastructure Fragmentation: Increase in disconnected components
- Throughput Loss: Reduction in message capacity
- Component Cascade: Number of dependent failures

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict

from .simulation_graph import SimulationGraph, ComponentState, RelationType


class FailureMode(Enum):
    """Types of failure modes."""
    CRASH = "crash"           # Complete component failure
    DEGRADED = "degraded"     # Partial failure (reduced capacity)
    NETWORK = "network"       # Network partition
    OVERLOAD = "overload"     # Resource exhaustion


class CascadeRule(Enum):
    """Rules for cascade propagation."""
    PHYSICAL = "physical"     # Node failure -> hosted components fail
    LOGICAL = "logical"       # App failure -> dependent apps affected
    NETWORK = "network"       # Network partition propagation
    ALL = "all"               # All cascade rules


@dataclass
class FailureScenario:
    """Configuration for a failure simulation."""
    target_id: str
    description: str = ""
    
    # Failure settings
    failure_mode: FailureMode = FailureMode.CRASH
    cascade_rule: CascadeRule = CascadeRule.ALL
    
    # Cascade parameters
    cascade_threshold: float = 0.5    # Weight threshold for cascade
    cascade_probability: float = 0.7  # Probability of cascade propagation
    max_cascade_depth: int = 5        # Maximum cascade depth
    
    # Analysis layer
    layer: str = "complete"  # application, infrastructure, complete
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class ImpactMetrics:
    """Impact metrics from a failure simulation."""
    # Reachability
    initial_paths: int = 0
    remaining_paths: int = 0
    reachability_loss: float = 0.0
    
    # Infrastructure
    initial_components: int = 0
    remaining_components: int = 0
    fragmentation: float = 0.0
    
    # Throughput (estimated)
    throughput_loss: float = 0.0
    affected_topics: int = 0
    
    # Cascade
    cascade_count: int = 0
    cascade_by_type: Dict[str, int] = field(default_factory=dict)
    
    @property
    def composite_impact(self) -> float:
        """
        Composite impact score combining all metrics.
        
        Formula: Impact = 0.4 * ReachLoss + 0.3 * Fragmentation + 0.3 * ThroughputLoss
        """
        return (
            0.4 * self.reachability_loss +
            0.3 * self.fragmentation +
            0.3 * self.throughput_loss
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reachability": {
                "initial_paths": self.initial_paths,
                "remaining_paths": self.remaining_paths,
                "loss_percent": round(self.reachability_loss * 100, 2),
            },
            "infrastructure": {
                "initial_components": self.initial_components,
                "remaining_components": self.remaining_components,
                "fragmentation_percent": round(self.fragmentation * 100, 2),
            },
            "throughput": {
                "loss_percent": round(self.throughput_loss * 100, 2),
                "affected_topics": self.affected_topics,
            },
            "cascade": {
                "total_count": self.cascade_count,
                "by_type": self.cascade_by_type,
            },
            "composite_impact": round(self.composite_impact, 4),
        }


@dataclass
class FailureResult:
    """Result of a failure simulation."""
    scenario: str
    target_id: str
    target_type: str
    
    # Cascaded failures
    cascaded_failures: List[str] = field(default_factory=list)
    cascade_sequence: List[Tuple[str, str, str]] = field(default_factory=list)  # (failed, cause, depth)
    
    # Impact metrics
    impact: ImpactMetrics = field(default_factory=ImpactMetrics)
    
    # Per-layer breakdown
    app_layer_impact: float = 0.0
    infra_layer_impact: float = 0.0
    
    # Analysis duration
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "target": {
                "id": self.target_id,
                "type": self.target_type,
            },
            "cascade": {
                "count": len(self.cascaded_failures),
                "failures": self.cascaded_failures[:20],  # Limit for readability
                "sequence": [
                    {"failed": f, "cause": c, "depth": d}
                    for f, c, d in self.cascade_sequence[:20]
                ],
            },
            "impact": self.impact.to_dict(),
            "layer_breakdown": {
                "application": round(self.app_layer_impact, 4),
                "infrastructure": round(self.infra_layer_impact, 4),
            },
            "duration_ms": round(self.duration_ms, 2),
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattened dict for CSV/DataFrame export."""
        return {
            "node_id": self.target_id,
            "node_type": self.target_type,
            "reachability_loss": round(self.impact.reachability_loss, 4),
            "fragmentation": round(self.impact.fragmentation, 4),
            "throughput_loss": round(self.impact.throughput_loss, 4),
            "composite_impact": round(self.impact.composite_impact, 4),
            "cascade_count": len(self.cascaded_failures),
            "app_layer_impact": round(self.app_layer_impact, 4),
            "infra_layer_impact": round(self.infra_layer_impact, 4),
        }


class FailureSimulator:
    """
    Simulates component failures and cascade effects.
    
    Uses RAW structural relationships to model failure propagation:
    - Physical cascade: Node failure -> hosted apps/brokers fail
    - Logical cascade: App failure -> dependent subscribers affected
    - Network cascade: Network partition propagation
    """
    
    def __init__(self, graph: SimulationGraph):
        """
        Initialize the failure simulator.
        
        Args:
            graph: SimulationGraph instance with raw structural relationships
        """
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        
        # Random generator
        self._rng = random.Random()
        
        # Cache initial state metrics
        self._initial_paths = len(graph.get_pub_sub_paths())
        self._initial_components = len([
            c for c in graph.components.values()
            if c.type in ("Application", "Broker", "Node")
        ])
    
    def simulate(self, scenario: FailureScenario) -> FailureResult:
        """
        Run a failure simulation.
        
        Args:
            scenario: Configuration for the failure simulation
            
        Returns:
            FailureResult with cascade and impact analysis
        """
        import time
        start_time = time.time()
        
        # Reset graph state
        self.graph.reset()
        
        # Set random seed if provided
        if scenario.seed is not None:
            self._rng = random.Random(scenario.seed)
        
        target = scenario.target_id
        
        # Validate target
        if target not in self.graph.graph:
            return self._empty_result(scenario, "Target not found")
        
        target_type = self.graph.graph.nodes[target].get("type", "Unknown")
        
        # === Phase 1: Initial Failure ===
        self.graph.fail_component(target)
        failed_set = {target}
        cascade_sequence = [(target, "initial", 0)]
        
        # === Phase 2: Cascade Propagation ===
        cascaded = self._propagate_cascade(
            target, scenario, failed_set, cascade_sequence
        )
        
        # === Phase 3: Impact Calculation ===
        impact = self._calculate_impact(target, failed_set, scenario)
        
        # === Phase 4: Layer-Specific Impact ===
        app_impact, infra_impact = self._calculate_layer_impacts(failed_set)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return FailureResult(
            scenario=scenario.description or f"Failure: {target}",
            target_id=target,
            target_type=target_type,
            cascaded_failures=list(cascaded),
            cascade_sequence=cascade_sequence,
            impact=impact,
            app_layer_impact=app_impact,
            infra_layer_impact=infra_impact,
            duration_ms=duration_ms,
        )
    
    def simulate_exhaustive(
        self, 
        scenario_template: FailureScenario,
        component_types: Optional[List[str]] = None
    ) -> List[FailureResult]:
        """
        Run failure simulation for all components.
        
        Args:
            scenario_template: Template scenario (target_id will be overridden)
            component_types: Filter to specific component types (None = all)
            
        Returns:
            List of FailureResult for each component
        """
        results = []
        
        # Determine targets based on layer
        if scenario_template.layer == "application":
            component_types = component_types or ["Application", "Broker"]
        elif scenario_template.layer == "infrastructure":
            component_types = component_types or ["Node"]
        else:
            component_types = component_types or ["Application", "Broker", "Node"]
        
        targets = [
            comp_id for comp_id, comp in self.graph.components.items()
            if comp.type in component_types
        ]
        
        for target_id in targets:
            scenario = FailureScenario(
                target_id=target_id,
                description=f"Exhaustive: {target_id}",
                failure_mode=scenario_template.failure_mode,
                cascade_rule=scenario_template.cascade_rule,
                cascade_threshold=scenario_template.cascade_threshold,
                cascade_probability=scenario_template.cascade_probability,
                max_cascade_depth=scenario_template.max_cascade_depth,
                layer=scenario_template.layer,
            )
            results.append(self.simulate(scenario))
        
        # Sort by impact (highest first)
        results.sort(key=lambda r: r.impact.composite_impact, reverse=True)
        
        return results
    
    def _propagate_cascade(
        self,
        initial_target: str,
        scenario: FailureScenario,
        failed_set: Set[str],
        cascade_sequence: List[Tuple[str, str, int]]
    ) -> List[str]:
        """Propagate cascade failures from initial target."""
        cascaded = []
        queue = [(initial_target, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= scenario.max_cascade_depth:
                continue
            
            current_type = self.graph.graph.nodes[current].get("type", "Unknown")
            
            # === Physical Cascade (Node -> Hosted Components) ===
            if (scenario.cascade_rule in (CascadeRule.PHYSICAL, CascadeRule.ALL) 
                and current_type == "Node"):
                hosted = self.graph.get_hosted_components(current)
                for comp in hosted:
                    if comp not in failed_set:
                        failed_set.add(comp)
                        cascaded.append(comp)
                        cascade_sequence.append((comp, f"hosted_on:{current}", depth + 1))
                        self.graph.fail_component(comp)
                        queue.append((comp, depth + 1))
            
            # === Logical Cascade (Publisher -> Topic -> Broker) ===
            if (scenario.cascade_rule in (CascadeRule.LOGICAL, CascadeRule.ALL)
                and current_type in ("Application", "Broker")):
                
                # Get topics this component interacts with
                if current_type == "Application":
                    publishes, subscribes = self.graph.get_app_topics(current)
                    topics = set(publishes + subscribes)
                else:
                    # Broker - get all routed topics
                    topics = set()
                    for src, tgt, data in self.graph.graph.out_edges(current, data=True):
                        if data.get("relation") == RelationType.ROUTES.value:
                            topics.add(tgt)
                
                for topic_id in topics:
                    # Check if topic becomes unreachable
                    topic_weight = self.graph.topics.get(topic_id, {})
                    if isinstance(topic_weight, dict):
                        weight = topic_weight.get("weight", 1.0)
                    else:
                        weight = getattr(topic_weight, "weight", 1.0)
                    
                    if weight >= scenario.cascade_threshold:
                        # Check broker availability for this topic
                        brokers = self.graph.get_routing_brokers(topic_id)
                        active_brokers = [b for b in brokers if self.graph.is_active(b)]
                        
                        if not active_brokers:
                            # Topic becomes unreachable - affect subscribers
                            subscribers = self.graph.get_subscribers(topic_id)
                            for sub in subscribers:
                                if sub not in failed_set:
                                    # Probabilistic cascade
                                    if self._rng.random() < scenario.cascade_probability:
                                        # Mark as degraded rather than failed
                                        self.graph.set_state(sub, ComponentState.DEGRADED)
            
            # === Network Cascade (Node -> Connected Nodes) ===
            if (scenario.cascade_rule in (CascadeRule.NETWORK, CascadeRule.ALL)
                and current_type == "Node"):
                
                for src, tgt, data in self.graph.graph.out_edges(current, data=True):
                    if data.get("relation") == RelationType.CONNECTS_TO.value:
                        neighbor = tgt
                        if neighbor not in failed_set:
                            edge_weight = data.get("weight", 1.0)
                            
                            if edge_weight >= scenario.cascade_threshold:
                                if self._rng.random() < scenario.cascade_probability:
                                    failed_set.add(neighbor)
                                    cascaded.append(neighbor)
                                    cascade_sequence.append((neighbor, f"network:{current}", depth + 1))
                                    self.graph.fail_component(neighbor)
                                    queue.append((neighbor, depth + 1))
        
        return cascaded
    
    def _calculate_impact(
        self,
        target: str,
        failed_set: Set[str],
        scenario: FailureScenario
    ) -> ImpactMetrics:
        """Calculate impact metrics after failure propagation."""
        
        # === Reachability Loss ===
        # How many pub-sub paths are broken?
        initial_paths = self._initial_paths
        remaining_paths = len(self.graph.get_pub_sub_paths(active_only=True))
        
        if initial_paths > 0:
            reachability_loss = 1.0 - (remaining_paths / initial_paths)
        else:
            reachability_loss = 0.0
        
        # === Infrastructure Fragmentation ===
        # How fragmented did the infrastructure become?
        initial_active = self._initial_components
        remaining_active = len([
            c for c in self.graph.components.values()
            if c.type in ("Application", "Broker", "Node")
            and self.graph.is_active(c.id)
        ])
        
        if initial_active > 1:
            fragmentation = (initial_active - remaining_active) / (initial_active - 1)
        else:
            fragmentation = 0.0
        
        # === Throughput Loss ===
        # Estimated based on broker and topic availability
        affected_topics = 0
        total_topics = len(self.graph.topics)
        
        for topic_id in self.graph.topics:
            brokers = self.graph.get_routing_brokers(topic_id)
            active_brokers = [b for b in brokers if self.graph.is_active(b)]
            
            if not active_brokers:
                affected_topics += 1
        
        if total_topics > 0:
            throughput_loss = affected_topics / total_topics
        else:
            throughput_loss = 0.0
        
        # === Cascade Statistics ===
        cascade_by_type = defaultdict(int)
        for comp_id in failed_set:
            if comp_id == target:
                continue
            comp_type = self.graph.components.get(comp_id, {})
            if hasattr(comp_type, 'type'):
                cascade_by_type[comp_type.type] += 1
            else:
                cascade_by_type["Unknown"] += 1
        
        return ImpactMetrics(
            initial_paths=initial_paths,
            remaining_paths=remaining_paths,
            reachability_loss=reachability_loss,
            initial_components=initial_active,
            remaining_components=remaining_active,
            fragmentation=fragmentation,
            throughput_loss=throughput_loss,
            affected_topics=affected_topics,
            cascade_count=len(failed_set) - 1,  # Exclude initial target
            cascade_by_type=dict(cascade_by_type),
        )
    
    def _calculate_layer_impacts(
        self, 
        failed_set: Set[str]
    ) -> Tuple[float, float]:
        """Calculate per-layer impact scores."""
        
        # Application layer impact
        app_components = set(self.graph.get_components_by_type("Application"))
        app_components.update(self.graph.get_components_by_type("Broker"))
        
        app_failed = app_components & failed_set
        app_impact = len(app_failed) / len(app_components) if app_components else 0.0
        
        # Infrastructure layer impact
        infra_components = set(self.graph.get_components_by_type("Node"))
        infra_failed = infra_components & failed_set
        infra_impact = len(infra_failed) / len(infra_components) if infra_components else 0.0
        
        return app_impact, infra_impact
    
    def _empty_result(self, scenario: FailureScenario, reason: str) -> FailureResult:
        """Create empty result for error cases."""
        return FailureResult(
            scenario=f"{scenario.description} ({reason})",
            target_id=scenario.target_id,
            target_type="Unknown",
        )


class BatchFailureSimulator:
    """
    Runs batch failure simulations for dataset generation and validation.
    """
    
    def __init__(self, graph: SimulationGraph):
        self.graph = graph
        self.simulator = FailureSimulator(graph)
    
    def generate_impact_dataset(
        self,
        layer: str = "complete",
        cascade_threshold: float = 0.5,
        cascade_probability: float = 0.7
    ) -> Dict[str, float]:
        """
        Generate impact scores for all components.
        
        Returns:
            Dict mapping component_id to composite_impact score
        """
        scenario_template = FailureScenario(
            target_id="",  # Will be overridden
            layer=layer,
            cascade_threshold=cascade_threshold,
            cascade_probability=cascade_probability,
        )
        
        results = self.simulator.simulate_exhaustive(scenario_template)
        
        return {
            r.target_id: r.impact.composite_impact
            for r in results
        }
    
    def rank_components_by_impact(
        self,
        layer: str = "complete"
    ) -> List[Tuple[str, str, float]]:
        """
        Rank components by their failure impact.
        
        Returns:
            List of (component_id, component_type, impact_score) sorted by impact
        """
        impact_scores = self.generate_impact_dataset(layer=layer)
        
        ranked = [
            (comp_id, self.graph.components[comp_id].type, score)
            for comp_id, score in impact_scores.items()
        ]
        
        return sorted(ranked, key=lambda x: x[2], reverse=True)