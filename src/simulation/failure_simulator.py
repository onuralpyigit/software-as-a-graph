"""
Failure Simulator

Simulates component failures and their cascading effects on the system.
Works directly on RAW structural relationships without DEPENDS_ON derivation.

Impact Metrics:
    - Reachability Loss: Percentage of broken pub-sub paths
    - Infrastructure Fragmentation: Increase in disconnected components
    - Throughput Loss: Reduction in message delivery capacity
    - Cascade Count: Number of components affected by cascade

Cascade Rules:
    - Physical: Node failure -> hosted components fail (RUNS_ON)
    - Logical: Broker failure -> topics become unreachable
    - Network: Network partition via CONNECTS_TO
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
    """Types of component failure modes."""
    CRASH = "crash"           # Complete failure - component stops
    DEGRADED = "degraded"     # Partial failure - reduced capacity
    PARTITION = "partition"   # Network partition - unreachable
    OVERLOAD = "overload"     # Resource exhaustion


class CascadeRule(Enum):
    """Rules governing failure cascade propagation."""
    PHYSICAL = "physical"     # Node failure cascades to hosted components
    LOGICAL = "logical"       # Broker failure affects topic routing
    NETWORK = "network"       # Network partition propagation
    ALL = "all"               # All cascade rules applied


@dataclass
class FailureScenario:
    """Configuration for a failure simulation."""
    target_id: str
    description: str = ""
    
    # Failure settings
    failure_mode: FailureMode = FailureMode.CRASH
    cascade_rule: CascadeRule = CascadeRule.ALL
    
    # Cascade parameters
    cascade_probability: float = 1.0    # Probability of cascade propagation
    max_cascade_depth: int = 10         # Maximum cascade depth
    
    # Layer filter (which components to consider)
    layer: str = "system"  # app, infra, mw-app, mw-infra, system
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class ImpactMetrics:
    """
    Impact metrics from a failure simulation.
    
    All metrics are normalized to [0, 1] range for comparison.
    """
    # Reachability (pub-sub path connectivity)
    initial_paths: int = 0
    remaining_paths: int = 0
    reachability_loss: float = 0.0
    
    # Infrastructure (component availability)
    initial_components: int = 0
    failed_components: int = 0
    fragmentation: float = 0.0
    
    # Throughput (message delivery capacity)
    initial_throughput: float = 1.0
    remaining_throughput: float = 1.0
    throughput_loss: float = 0.0
    
    # Affected entities
    affected_topics: int = 0
    affected_subscribers: int = 0
    affected_publishers: int = 0
    
    # Cascade
    cascade_count: int = 0
    cascade_depth: int = 0
    cascade_by_type: Dict[str, int] = field(default_factory=dict)
    
    @property
    def composite_impact(self) -> float:
        """
        Composite impact score combining all metrics.
        
        Formula: I(v) = w_r * reachability + w_f * fragmentation + w_t * throughput
        """
        w_r, w_f, w_t = 0.4, 0.3, 0.3
        return (
            w_r * self.reachability_loss +
            w_f * self.fragmentation +
            w_t * self.throughput_loss
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
                "failed_components": self.failed_components,
                "fragmentation_percent": round(self.fragmentation * 100, 2),
            },
            "throughput": {
                "loss_percent": round(self.throughput_loss * 100, 2),
            },
            "affected": {
                "topics": self.affected_topics,
                "subscribers": self.affected_subscribers,
                "publishers": self.affected_publishers,
            },
            "cascade": {
                "count": self.cascade_count,
                "depth": self.cascade_depth,
                "by_type": self.cascade_by_type,
            },
            "composite_impact": round(self.composite_impact, 4),
        }


@dataclass
class CascadeEvent:
    """Record of a cascade propagation event."""
    component_id: str
    component_type: str
    cause: str
    depth: int


@dataclass
class FailureResult:
    """Result of a failure simulation."""
    target_id: str
    target_type: str
    scenario: str
    impact: ImpactMetrics
    
    # Cascade details
    cascaded_failures: List[str] = field(default_factory=list)
    cascade_sequence: List[CascadeEvent] = field(default_factory=list)
    
    # Per-layer impact
    layer_impacts: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "scenario": self.scenario,
            "impact": self.impact.to_dict(),
            "cascaded_failures": self.cascaded_failures,
            "cascade_sequence": [
                {"id": e.component_id, "type": e.component_type, "cause": e.cause, "depth": e.depth}
                for e in self.cascade_sequence
            ],
            "layer_impacts": {k: round(v, 4) for k, v in self.layer_impacts.items()},
        }


class FailureSimulator:
    """
    Simulates component failures and cascade propagation.
    
    Works on RAW structural relationships:
        - RUNS_ON: Physical cascade (node -> hosted components)
        - ROUTES: Logical cascade (broker -> topics)
        - CONNECTS_TO: Network cascade (node -> connected nodes)
    
    Example:
        >>> graph = SimulationGraph(uri="bolt://localhost:7687")
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
        
        # Baseline metrics (computed once)
        self._initial_paths = 0
        self._initial_components = 0
    
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
        
        # Validate target
        if scenario.target_id not in self.graph.components:
            return self._empty_result(scenario, f"Target '{scenario.target_id}' not found")
        
        target_comp = self.graph.components[scenario.target_id]
        
        self.logger.info(f"Simulating failure: {scenario.target_id} ({target_comp.type})")
        
        # Capture initial state
        self._initial_paths = len(self.graph.get_pub_sub_paths(active_only=True))
        self._initial_components = len([
            c for c in self.graph.components.values()
            if c.type in ("Application", "Broker", "Node") and c.state == ComponentState.ACTIVE
        ])
        
        # Fail the target
        self.graph.fail_component(scenario.target_id)
        failed_set = {scenario.target_id}
        cascade_sequence = [CascadeEvent(
            component_id=scenario.target_id,
            component_type=target_comp.type,
            cause="initial_failure",
            depth=0
        )]
        
        # Propagate cascade
        max_depth = self._propagate_cascade(
            scenario, 
            scenario.target_id, 
            failed_set, 
            cascade_sequence
        )
        
        # Calculate impact metrics
        impact = self._calculate_impact(scenario.target_id, failed_set)
        impact.cascade_count = len(failed_set) - 1
        impact.cascade_depth = max_depth
        
        # Calculate per-layer impacts
        layer_impacts = self._calculate_layer_impacts(failed_set)
        
        return FailureResult(
            target_id=scenario.target_id,
            target_type=target_comp.type,
            scenario=scenario.description or f"Failure: {scenario.target_id}",
            impact=impact,
            cascaded_failures=[c for c in failed_set if c != scenario.target_id],
            cascade_sequence=cascade_sequence,
            layer_impacts=layer_impacts,
        )
    
    def simulate_exhaustive(
        self,
        scenario_template: Optional[FailureScenario] = None,
        layer: str = "system"
    ) -> List[FailureResult]:
        """
        Run failure simulation for all components in a layer.
        
        Args:
            scenario_template: Base scenario configuration
            layer: Layer to analyze
            
        Returns:
            List of FailureResult sorted by impact (highest first)
        """
        results = []
        
        # Get components for the layer
        component_ids = self.graph.get_components_by_layer(layer)
        
        self.logger.info(f"Running exhaustive failure analysis: {len(component_ids)} components")
        
        for comp_id in component_ids:
            scenario = FailureScenario(
                target_id=comp_id,
                description=f"Exhaustive failure: {comp_id}",
                layer=layer,
                cascade_rule=scenario_template.cascade_rule if scenario_template else CascadeRule.ALL,
                cascade_probability=scenario_template.cascade_probability if scenario_template else 1.0,
                max_cascade_depth=scenario_template.max_cascade_depth if scenario_template else 10,
            )
            
            result = self.simulate(scenario)
            results.append(result)
        
        # Sort by composite impact (highest first)
        results.sort(key=lambda r: r.impact.composite_impact, reverse=True)
        
        return results
    
    def _propagate_cascade(
        self,
        scenario: FailureScenario,
        initial_target: str,
        failed_set: Set[str],
        cascade_sequence: List[CascadeEvent]
    ) -> int:
        """
        Propagate failure cascade from the initial target.
        
        Returns:
            Maximum cascade depth reached
        """
        max_depth = 0
        queue: List[Tuple[str, int]] = [(initial_target, 0)]
        
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
            
            # === Logical Cascade (Broker -> Topics become unreachable) ===
            if scenario.cascade_rule in (CascadeRule.LOGICAL, CascadeRule.ALL):
                if current_type == "Broker":
                    # Check all topics routed through this broker
                    for topic_id, brokers in self.graph._routing.items():
                        if current_id in brokers:
                            # Check if topic still has active brokers
                            active_brokers = [
                                b for b in brokers 
                                if b not in failed_set and self.graph.is_active(b)
                            ]
                            
                            if not active_brokers:
                                # Topic becomes unreachable - affect dependent apps
                                # (We don't fail apps, but track impact)
                                pass
                
                if current_type == "Application":
                    # Application failure may cascade to dependent subscribers
                    # if this app is a critical publisher
                    publishes_to, _ = self.graph.get_app_topics(current_id)
                    for topic_id in publishes_to:
                        publishers = self.graph.get_publishers(topic_id)
                        if not publishers:
                            # No more publishers for this topic
                            # Subscribers are effectively starved
                            pass
            
            # === Network Cascade (Node -> Connected Nodes) ===
            if scenario.cascade_rule in (CascadeRule.NETWORK, CascadeRule.ALL):
                if current_type == "Node":
                    # Check if this node is critical for network connectivity
                    connected = self.graph.get_connected_nodes(current_id)
                    # In a partition scenario, connected nodes might become isolated
                    # but we typically don't fail them unless specified
        
        return max_depth
    
    def _calculate_impact(
        self,
        target_id: str,
        failed_set: Set[str]
    ) -> ImpactMetrics:
        """Calculate impact metrics after failure cascade."""
        
        # === Reachability Loss ===
        remaining_paths = len(self.graph.get_pub_sub_paths(active_only=True))
        
        if self._initial_paths > 0:
            reachability_loss = 1.0 - (remaining_paths / self._initial_paths)
        else:
            reachability_loss = 0.0
        
        # === Infrastructure Fragmentation ===
        remaining_active = len([
            c for c in self.graph.components.values()
            if c.type in ("Application", "Broker", "Node")
            and c.state == ComponentState.ACTIVE
        ])
        
        failed_count = self._initial_components - remaining_active
        
        if self._initial_components > 1:
            fragmentation = failed_count / (self._initial_components - 1)
        else:
            fragmentation = 0.0
        
        # === Throughput Loss ===
        affected_topics = 0
        total_topics = len(self.graph.topics)
        
        for topic_id in self.graph.topics:
            # Check if topic has both publishers and brokers
            publishers = self.graph.get_publishers(topic_id)
            brokers = self.graph.get_routing_brokers(topic_id)
            subscribers = self.graph.get_subscribers(topic_id)
            
            if not publishers or not brokers or not subscribers:
                affected_topics += 1
        
        if total_topics > 0:
            throughput_loss = affected_topics / total_topics
        else:
            throughput_loss = 0.0
        
        # === Affected Entities ===
        affected_pubs = set()
        affected_subs = set()
        
        for topic_id in self.graph.topics:
            publishers = self.graph._publishers.get(topic_id, [])
            subscribers = self.graph._subscribers.get(topic_id, [])
            brokers = self.graph.get_routing_brokers(topic_id)
            
            # Topic is affected if it lost routing capability
            if not brokers:
                affected_pubs.update(publishers)
                affected_subs.update(subscribers)
        
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
            fragmentation=fragmentation,
            throughput_loss=throughput_loss,
            affected_topics=affected_topics,
            affected_subscribers=len(affected_subs),
            affected_publishers=len(affected_pubs),
            cascade_by_type=dict(cascade_by_type),
        )
    
    def _calculate_layer_impacts(self, failed_set: Set[str]) -> Dict[str, float]:
        """Calculate impact per analysis layer."""
        layer_impacts = {}
        
        layers = ["app", "infra", "mw-app", "mw-infra", "system"]
        
        for layer in layers:
            layer_comps = set(self.graph.get_components_by_layer(layer))
            if not layer_comps:
                layer_impacts[layer] = 0.0
                continue
            
            # Compute impact as fraction of layer components affected
            affected = failed_set & layer_comps
            layer_impacts[layer] = len(affected) / len(layer_comps)
        
        return layer_impacts
    
    def _empty_result(self, scenario: FailureScenario, reason: str) -> FailureResult:
        """Create an empty result for failed simulations."""
        return FailureResult(
            target_id=scenario.target_id,
            target_type="Unknown",
            scenario=reason,
            impact=ImpactMetrics(),
        )