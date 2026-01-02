"""
Failure Simulator - Version 5.0

Simulates component failures and measures impact on the pub-sub system.

Uses ORIGINAL edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
NOT derived DEPENDS_ON relationships.

Features:
- Single component failure simulation
- Cascade failure propagation
- Full campaign (all components)
- Impact scoring by reachability loss
- Layer-specific failure analysis

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Tuple


from .simulation_graph import (
    SimulationGraph,
    Component,
    ComponentType,
    EdgeType,
    ComponentStatus,
)


# =============================================================================
# Enums
# =============================================================================

class FailureMode(Enum):
    """Types of component failures."""
    CRASH = "crash"              # Complete failure
    DEGRADED = "degraded"        # Partial functionality
    PARTITION = "partition"      # Network isolation
    
    @property
    def impact_factor(self) -> float:
        """Impact multiplier for this failure mode."""
        return {
            FailureMode.CRASH: 1.0,
            FailureMode.DEGRADED: 0.5,
            FailureMode.PARTITION: 0.8,
        }[self]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FailureResult:
    """Result of a single failure simulation."""
    component_id: str
    component_type: ComponentType
    failure_mode: FailureMode
    
    # Impact metrics
    directly_affected: Set[str] = field(default_factory=set)
    cascade_affected: Set[str] = field(default_factory=set)
    paths_broken: int = 0
    total_paths: int = 0
    
    # Computed scores
    impact_score: float = 0.0
    reachability_loss: float = 0.0
    cascade_extent: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_affected(self) -> int:
        """Total components affected including cascade."""
        return len(self.directly_affected) + len(self.cascade_affected)
    
    @property
    def all_affected(self) -> Set[str]:
        """All affected component IDs."""
        return self.directly_affected | self.cascade_affected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "failure_mode": self.failure_mode.value,
            "directly_affected": list(self.directly_affected),
            "cascade_affected": list(self.cascade_affected),
            "total_affected": self.total_affected,
            "paths_broken": self.paths_broken,
            "total_paths": self.total_paths,
            "impact_score": round(self.impact_score, 6),
            "reachability_loss": round(self.reachability_loss, 6),
            "cascade_extent": round(self.cascade_extent, 6),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LayerFailureResult:
    """Failure results aggregated by layer."""
    layer: str
    layer_name: str
    results: List[FailureResult] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.results)
    
    @property
    def avg_impact(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.impact_score for r in self.results) / len(self.results)
    
    @property
    def max_impact(self) -> float:
        if not self.results:
            return 0.0
        return max(r.impact_score for r in self.results)
    
    def get_critical(self, threshold: float = 0.3) -> List[FailureResult]:
        """Get results with impact above threshold."""
        return [r for r in self.results if r.impact_score >= threshold]
    
    def ranked_by_impact(self) -> List[Tuple[str, float]]:
        """Get components ranked by impact."""
        return sorted(
            [(r.component_id, r.impact_score) for r in self.results],
            key=lambda x: x[1],
            reverse=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "count": self.count,
            "avg_impact": round(self.avg_impact, 6),
            "max_impact": round(self.max_impact, 6),
            "critical_count": len(self.get_critical()),
            "ranked": self.ranked_by_impact()[:10],
        }


@dataclass
class CampaignResult:
    """Result of a failure campaign (all components)."""
    results: List[FailureResult] = field(default_factory=list)
    by_layer: Dict[str, LayerFailureResult] = field(default_factory=dict)
    by_type: Dict[ComponentType, List[FailureResult]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    
    @property
    def total_simulations(self) -> int:
        return len(self.results)
    
    def ranked_by_impact(self) -> List[Tuple[str, float]]:
        """Get all components ranked by impact."""
        return sorted(
            [(r.component_id, r.impact_score) for r in self.results],
            key=lambda x: x[1],
            reverse=True
        )
    
    def get_critical(self, threshold: float = 0.3) -> List[FailureResult]:
        """Get results with impact above threshold."""
        return [r for r in self.results if r.impact_score >= threshold]
    
    def get_result(self, component_id: str) -> Optional[FailureResult]:
        """Get result for a specific component."""
        for r in self.results:
            if r.component_id == component_id:
                return r
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_simulations": self.total_simulations,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "ranked_top_20": self.ranked_by_impact()[:20],
            "critical_count": len(self.get_critical()),
            "by_layer": {k: v.to_dict() for k, v in self.by_layer.items()},
            "by_type": {
                k.value: len(v) for k, v in self.by_type.items()
            },
        }


# =============================================================================
# Failure Simulator
# =============================================================================

class FailureSimulator:
    """
    Simulates component failures in a pub-sub system.
    
    Uses original edge types to calculate impact based on
    message path disruption, not derived DEPENDS_ON relationships.
    
    Example:
        graph = SimulationGraph.from_json("system.json")
        simulator = FailureSimulator(cascade=True)
        
        # Single failure
        result = simulator.simulate_failure(graph, "broker_1")
        print(f"Impact: {result.impact_score:.4f}")
        
        # Full campaign
        campaign = simulator.simulate_all(graph)
        for comp_id, impact in campaign.ranked_by_impact()[:10]:
            print(f"{comp_id}: {impact:.4f}")
    """
    
    # Impact score weights
    WEIGHT_REACHABILITY = 0.6
    WEIGHT_CASCADE = 0.4
    
    def __init__(
        self,
        seed: Optional[int] = None,
        cascade: bool = True,
        cascade_threshold: float = 0.5,
        critical_threshold: float = 0.3,
    ):
        """
        Initialize the failure simulator.
        
        Args:
            seed: Random seed for reproducibility
            cascade: Enable cascade failure propagation
            cascade_threshold: Dependency fraction to trigger cascade
            critical_threshold: Impact threshold for critical classification
        """
        self.rng = random.Random(seed)
        self.cascade = cascade
        self.cascade_threshold = cascade_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
    
    # =========================================================================
    # Single Failure Simulation
    # =========================================================================
    
    def simulate_failure(
        self,
        graph: SimulationGraph,
        component_id: str,
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> FailureResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: The simulation graph
            component_id: Component to fail
            failure_mode: Type of failure
        
        Returns:
            FailureResult with impact metrics
        """
        comp = graph.get_component(component_id)
        if not comp:
            raise ValueError(f"Component not found: {component_id}")
        
        # Get baseline paths
        total_paths = len(graph.get_message_paths())
        paths_through = len(graph.get_paths_through_component(component_id))
        
        # Calculate direct impact
        directly_affected = self._calculate_direct_impact(graph, component_id, comp.type)
        
        # Calculate cascade if enabled
        cascade_affected = set()
        if self.cascade:
            cascade_affected = self._propagate_cascade(
                graph, {component_id} | directly_affected
            )
        
        # Calculate impact score
        paths_broken = paths_through
        reachability_loss = paths_broken / total_paths if total_paths > 0 else 0.0
        
        total_components = len(graph.components) - 1  # Exclude failed component
        cascade_extent = len(cascade_affected) / total_components if total_components > 0 else 0.0
        
        # Apply failure mode factor
        impact_factor = failure_mode.impact_factor
        
        impact_score = impact_factor * (
            self.WEIGHT_REACHABILITY * reachability_loss +
            self.WEIGHT_CASCADE * cascade_extent
        )
        
        return FailureResult(
            component_id=component_id,
            component_type=comp.type,
            failure_mode=failure_mode,
            directly_affected=directly_affected,
            cascade_affected=cascade_affected,
            paths_broken=paths_broken,
            total_paths=total_paths,
            impact_score=min(1.0, impact_score),
            reachability_loss=reachability_loss,
            cascade_extent=cascade_extent,
        )
    
    def _calculate_direct_impact(
        self,
        graph: SimulationGraph,
        component_id: str,
        comp_type: ComponentType,
    ) -> Set[str]:
        """
        Calculate components directly affected by a failure.
        
        Uses original edge types to trace impact through the system.
        """
        affected = set()
        
        if comp_type == ComponentType.APPLICATION:
            # App failure affects topics it publishes to
            for topic_id in graph.get_topics_published_by(component_id):
                # Subscribers lose messages from this publisher
                for sub in graph.get_subscribers(topic_id):
                    if sub != component_id:
                        affected.add(sub)
        
        elif comp_type == ComponentType.TOPIC:
            # Topic failure affects all pub/sub relationships
            for sub in graph.get_subscribers(component_id):
                affected.add(sub)
            for pub in graph.get_publishers(component_id):
                affected.add(pub)
        
        elif comp_type == ComponentType.BROKER:
            # Broker failure affects all routed topics
            for topic_id in graph.get_topics_routed_by(component_id):
                affected.add(topic_id)
                for sub in graph.get_subscribers(topic_id):
                    affected.add(sub)
                for pub in graph.get_publishers(topic_id):
                    affected.add(pub)
        
        elif comp_type == ComponentType.NODE:
            # Node failure affects all components running on it
            for comp_id in graph.get_components_on_node(component_id):
                affected.add(comp_id)
                # Recursively get their impact
                comp = graph.get_component(comp_id)
                if comp:
                    affected.update(
                        self._calculate_direct_impact(graph, comp_id, comp.type)
                    )
        
        return affected
    
    def _propagate_cascade(
        self,
        graph: SimulationGraph,
        failed: Set[str],
    ) -> Set[str]:
        """
        Propagate cascade failures through the system.
        
        A component fails in cascade if more than cascade_threshold
        of its dependencies have failed.
        """
        cascade_failures = set()
        visited = set(failed)
        
        # Iterate until no new failures
        while True:
            new_failures = set()
            
            for comp_id, comp in graph.components.items():
                if comp_id in visited:
                    continue
                
                # Calculate what fraction of dependencies failed
                dependencies = self._get_dependencies(graph, comp_id, comp.type)
                if not dependencies:
                    continue
                
                failed_deps = len(dependencies & (failed | cascade_failures))
                dep_ratio = failed_deps / len(dependencies)
                
                if dep_ratio >= self.cascade_threshold:
                    new_failures.add(comp_id)
            
            if not new_failures:
                break
            
            cascade_failures.update(new_failures)
            visited.update(new_failures)
        
        return cascade_failures
    
    def _get_dependencies(
        self,
        graph: SimulationGraph,
        component_id: str,
        comp_type: ComponentType,
    ) -> Set[str]:
        """Get components that this component depends on."""
        dependencies = set()
        
        if comp_type == ComponentType.APPLICATION:
            # Application depends on topics it subscribes to
            dependencies.update(graph.get_topics_subscribed_by(component_id))
            # And the node it runs on
            node = graph.get_node_for_component(component_id)
            if node:
                dependencies.add(node)
        
        elif comp_type == ComponentType.TOPIC:
            # Topic depends on its broker
            broker = graph.get_broker_for_topic(component_id)
            if broker:
                dependencies.add(broker)
        
        elif comp_type == ComponentType.BROKER:
            # Broker depends on its node
            node = graph.get_node_for_component(component_id)
            if node:
                dependencies.add(node)
        
        return dependencies
    
    # =========================================================================
    # Batch Simulation
    # =========================================================================
    
    def simulate_batch(
        self,
        graph: SimulationGraph,
        component_ids: List[str],
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> FailureResult:
        """
        Simulate simultaneous failure of multiple components.
        
        Args:
            graph: The simulation graph
            component_ids: Components to fail
            failure_mode: Type of failure
        
        Returns:
            FailureResult for the combined failure
        """
        if not component_ids:
            raise ValueError("No components specified")
        
        total_paths = len(graph.get_message_paths())
        
        # Calculate combined impact
        all_directly_affected = set()
        all_paths_broken = set()
        
        for comp_id in component_ids:
            comp = graph.get_component(comp_id)
            if comp:
                affected = self._calculate_direct_impact(graph, comp_id, comp.type)
                all_directly_affected.update(affected)
                
                paths = graph.get_paths_through_component(comp_id)
                all_paths_broken.update(paths)
        
        # Remove the failed components from affected
        all_directly_affected -= set(component_ids)
        
        # Cascade
        cascade_affected = set()
        if self.cascade:
            cascade_affected = self._propagate_cascade(
                graph, set(component_ids) | all_directly_affected
            )
        
        # Calculate scores
        paths_broken = len(all_paths_broken)
        reachability_loss = paths_broken / total_paths if total_paths > 0 else 0.0
        
        total_components = len(graph.components) - len(component_ids)
        cascade_extent = len(cascade_affected) / total_components if total_components > 0 else 0.0
        
        impact_score = (
            self.WEIGHT_REACHABILITY * reachability_loss +
            self.WEIGHT_CASCADE * cascade_extent
        )
        
        # Use first component for type
        first_comp = graph.get_component(component_ids[0])
        
        return FailureResult(
            component_id=",".join(component_ids),
            component_type=first_comp.type if first_comp else ComponentType.APPLICATION,
            failure_mode=failure_mode,
            directly_affected=all_directly_affected,
            cascade_affected=cascade_affected,
            paths_broken=paths_broken,
            total_paths=total_paths,
            impact_score=min(1.0, impact_score),
            reachability_loss=reachability_loss,
            cascade_extent=cascade_extent,
        )
    
    # =========================================================================
    # Full Campaign
    # =========================================================================
    
    def simulate_all(
        self,
        graph: SimulationGraph,
        component_types: Optional[List[ComponentType]] = None,
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> CampaignResult:
        """
        Simulate failure of every component.
        
        Args:
            graph: The simulation graph
            component_types: Types to simulate (None = all)
            failure_mode: Type of failure
        
        Returns:
            CampaignResult with all results
        """
        import time
        start_time = time.time()
        
        results = []
        by_type: Dict[ComponentType, List[FailureResult]] = {t: [] for t in ComponentType}
        
        # Get components to simulate
        if component_types:
            components = [
                c for c in graph.components.values()
                if c.type in component_types
            ]
        else:
            components = list(graph.components.values())
        
        self.logger.info(f"Running failure campaign on {len(components)} components")
        
        for comp in components:
            result = self.simulate_failure(graph, comp.id, failure_mode)
            results.append(result)
            by_type[comp.type].append(result)
        
        # Group by layer
        by_layer = self._group_by_layer(graph, results)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return CampaignResult(
            results=results,
            by_layer=by_layer,
            by_type=by_type,
            duration_ms=duration_ms,
        )
    
    def _group_by_layer(
        self,
        graph: SimulationGraph,
        results: List[FailureResult],
    ) -> Dict[str, LayerFailureResult]:
        """Group results by layer."""
        by_layer = {}
        
        for layer_key, layer_def in SimulationGraph.LAYER_DEFINITIONS.items():
            layer_types = set(layer_def["component_types"])
            
            layer_results = [
                r for r in results
                if r.component_type in layer_types
            ]
            
            by_layer[layer_key] = LayerFailureResult(
                layer=layer_key,
                layer_name=layer_def["name"],
                results=layer_results,
            )
        
        return by_layer
    
    # =========================================================================
    # Type-Specific Simulation
    # =========================================================================
    
    def simulate_type(
        self,
        graph: SimulationGraph,
        component_type: ComponentType,
        failure_mode: FailureMode = FailureMode.CRASH,
    ) -> List[FailureResult]:
        """
        Simulate failure of all components of a specific type.
        
        Args:
            graph: The simulation graph
            component_type: Type to simulate
            failure_mode: Type of failure
        
        Returns:
            List of FailureResult for each component
        """
        results = []
        
        for comp in graph.get_components_by_type(component_type):
            result = self.simulate_failure(graph, comp.id, failure_mode)
            results.append(result)
        
        return results


# =============================================================================
# Factory Functions
# =============================================================================

def simulate_single_failure(
    graph: SimulationGraph,
    component_id: str,
    cascade: bool = True,
    seed: Optional[int] = None,
) -> FailureResult:
    """
    Quick function to simulate a single failure.
    
    Args:
        graph: Simulation graph
        component_id: Component to fail
        cascade: Enable cascade propagation
        seed: Random seed
    
    Returns:
        FailureResult
    """
    simulator = FailureSimulator(seed=seed, cascade=cascade)
    return simulator.simulate_failure(graph, component_id)


def simulate_all_components(
    graph: SimulationGraph,
    cascade: bool = True,
    seed: Optional[int] = None,
) -> CampaignResult:
    """
    Quick function to simulate all component failures.
    
    Args:
        graph: Simulation graph
        cascade: Enable cascade propagation
        seed: Random seed
    
    Returns:
        CampaignResult
    """
    simulator = FailureSimulator(seed=seed, cascade=cascade)
    return simulator.simulate_all(graph)
