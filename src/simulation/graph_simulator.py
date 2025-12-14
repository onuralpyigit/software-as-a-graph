#!/usr/bin/env python3
"""
Graph Simulator - Failure Impact Simulation
============================================

Simulates component failures in pub-sub systems and measures impact
using DEPENDS_ON relationships (app-to-app, app-to-broker, node-to-node,
node-to-broker).

The simulator calculates impact scores based on reachability loss,
which can be used to validate criticality predictions from the analyzer.

Key Metrics:
  - Reachability Loss: Pairs that can no longer communicate
  - Impact Score: Proportion of system affected by failure
  - Resilience Score: 1 - Impact Score
  - Cascade Depth: How far failures propagate

Usage:
    from src.simulation import GraphSimulator
    from src.analysis import GraphAnalyzer
    
    # Build dependency graph
    analyzer = GraphAnalyzer()
    analyzer.load_from_file('system.json')
    analyzer.derive_depends_on()
    graph = analyzer.build_dependency_graph()
    
    # Simulate failures
    simulator = GraphSimulator()
    result = simulator.simulate_failure(graph, 'component_id')
    print(f"Impact: {result.impact_score:.2%}")

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import random

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class FailureMode(Enum):
    """How a component fails"""
    CRASH = "crash"              # Complete shutdown
    DEGRADED = "degraded"        # Partial functionality loss
    NETWORK = "network"          # Network partition


class SimulationMode(Enum):
    """Type of simulation"""
    SINGLE = "single"            # Single component failure
    MULTIPLE = "multiple"        # Multiple simultaneous failures
    CASCADE = "cascade"          # Cascading failure propagation
    EXHAUSTIVE = "exhaustive"    # Test all components


@dataclass
class FailureEvent:
    """Records a single failure event"""
    component: str
    component_type: str
    failure_mode: FailureMode
    timestamp: datetime
    is_cascade: bool = False
    cascade_depth: int = 0
    cause: str = "primary"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'component_type': self.component_type,
            'failure_mode': self.failure_mode.value,
            'timestamp': self.timestamp.isoformat(),
            'is_cascade': self.is_cascade,
            'cascade_depth': self.cascade_depth,
            'cause': self.cause
        }


@dataclass
class SimulationResult:
    """Complete results from a failure simulation"""
    simulation_id: str
    simulation_mode: SimulationMode
    start_time: datetime
    end_time: datetime
    
    # Failed components
    failed_components: List[str]
    cascade_failures: List[str]
    failure_events: List[FailureEvent]
    
    # Impact metrics
    impact_score: float              # 0.0 to 1.0 (affected / total)
    resilience_score: float          # 1 - impact_score
    
    # Reachability analysis
    original_reachability: int       # Original reachable pairs
    remaining_reachability: int      # Remaining reachable pairs
    reachability_loss: int           # Lost reachable pairs
    reachability_loss_pct: float     # Percentage loss
    
    # Affected components (not failed, but impacted)
    affected_components: List[str]
    isolated_components: List[str]
    
    # Connectivity
    original_components: int         # Connected components before
    resulting_components: int        # Connected components after
    fragmentation: int               # Increase in components
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'simulation_id': self.simulation_id,
            'simulation_mode': self.simulation_mode.value,
            'duration_ms': (self.end_time - self.start_time).total_seconds() * 1000,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'failed_components': self.failed_components,
            'cascade_failures': self.cascade_failures,
            'failure_events': [e.to_dict() for e in self.failure_events],
            'impact_metrics': {
                'impact_score': round(self.impact_score, 4),
                'resilience_score': round(self.resilience_score, 4),
            },
            'reachability': {
                'original': self.original_reachability,
                'remaining': self.remaining_reachability,
                'lost': self.reachability_loss,
                'loss_percentage': round(self.reachability_loss_pct, 2)
            },
            'affected_components': self.affected_components,
            'isolated_components': self.isolated_components,
            'connectivity': {
                'original_components': self.original_components,
                'resulting_components': self.resulting_components,
                'fragmentation': self.fragmentation
            },
            'statistics': self.statistics
        }


@dataclass
class BatchSimulationResult:
    """Results from running multiple simulations"""
    total_simulations: int
    results: List[SimulationResult]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_simulations': self.total_simulations,
            'summary': self.summary,
            'results': [r.to_dict() for r in self.results]
        }
    
    def get_impact_ranking(self) -> List[Tuple[str, float]]:
        """Get components ranked by impact score"""
        impacts = []
        for result in self.results:
            if len(result.failed_components) == 1:
                impacts.append((result.failed_components[0], result.impact_score))
        return sorted(impacts, key=lambda x: x[1], reverse=True)


# ============================================================================
# Graph Simulator
# ============================================================================

class GraphSimulator:
    """
    Simulates failures in pub-sub systems and measures impact.
    
    Uses DEPENDS_ON relationships to determine:
    - Which components are affected by failures
    - How failures cascade through the system
    - Impact on system reachability
    """
    
    def __init__(self,
                 cascade_threshold: float = 0.7,
                 cascade_probability: float = 0.5,
                 max_cascade_depth: int = 5,
                 seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            cascade_threshold: Dependency loss ratio to trigger cascade (0-1)
            cascade_probability: Base probability of cascade occurrence (0-1)
            max_cascade_depth: Maximum depth of cascade propagation
            seed: Random seed for reproducibility
        """
        self.cascade_threshold = cascade_threshold
        self.cascade_probability = cascade_probability
        self.max_cascade_depth = max_cascade_depth
        
        if seed is not None:
            random.seed(seed)
        
        self._simulation_counter = 0
        self.logger = logging.getLogger('graph_simulator')
    
    # =========================================================================
    # Core Simulation Methods
    # =========================================================================
    
    def simulate_failure(self,
                        graph: nx.DiGraph,
                        component: str,
                        failure_mode: FailureMode = FailureMode.CRASH,
                        enable_cascade: bool = False) -> SimulationResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON edges
            component: Component ID to fail
            failure_mode: How the component fails
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with impact metrics
        """
        if component not in graph:
            raise ValueError(f"Component '{component}' not found in graph")
        
        self._simulation_counter += 1
        start_time = datetime.now()
        
        # Calculate original reachability
        original_reach = self._calculate_reachability(graph)
        original_cc = nx.number_weakly_connected_components(graph)
        
        # Create failed graph
        failed_graph = graph.copy()
        failed_components = [component]
        cascade_failures = []
        failure_events = []
        
        # Record primary failure
        comp_type = graph.nodes[component].get('type', 'Unknown')
        failure_events.append(FailureEvent(
            component=component,
            component_type=comp_type,
            failure_mode=failure_mode,
            timestamp=start_time,
            is_cascade=False,
            cascade_depth=0,
            cause="primary"
        ))
        
        # Remove failed component
        failed_graph.remove_node(component)
        
        # Simulate cascades if enabled
        if enable_cascade:
            cascade_failures, cascade_events = self._simulate_cascade(
                graph, failed_graph, failed_components, failure_mode
            )
            failed_components.extend(cascade_failures)
            failure_events.extend(cascade_events)
        
        # Analyze impact
        end_time = datetime.now()
        return self._build_result(
            graph, failed_graph, original_reach, original_cc,
            failed_components, cascade_failures, failure_events,
            SimulationMode.CASCADE if enable_cascade else SimulationMode.SINGLE,
            start_time, end_time
        )
    
    def simulate_multiple_failures(self,
                                   graph: nx.DiGraph,
                                   components: List[str],
                                   failure_mode: FailureMode = FailureMode.CRASH,
                                   enable_cascade: bool = False) -> SimulationResult:
        """
        Simulate simultaneous failure of multiple components.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON edges
            components: List of component IDs to fail
            failure_mode: How the components fail
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with impact metrics
        """
        # Validate components
        for comp in components:
            if comp not in graph:
                raise ValueError(f"Component '{comp}' not found in graph")
        
        self._simulation_counter += 1
        start_time = datetime.now()
        
        # Calculate original reachability
        original_reach = self._calculate_reachability(graph)
        original_cc = nx.number_weakly_connected_components(graph)
        
        # Create failed graph
        failed_graph = graph.copy()
        failed_components = list(components)
        cascade_failures = []
        failure_events = []
        
        # Record primary failures
        for comp in components:
            comp_type = graph.nodes[comp].get('type', 'Unknown')
            failure_events.append(FailureEvent(
                component=comp,
                component_type=comp_type,
                failure_mode=failure_mode,
                timestamp=start_time,
                is_cascade=False,
                cascade_depth=0,
                cause="primary"
            ))
            failed_graph.remove_node(comp)
        
        # Simulate cascades if enabled
        if enable_cascade:
            cascade_failures, cascade_events = self._simulate_cascade(
                graph, failed_graph, failed_components, failure_mode
            )
            failed_components.extend(cascade_failures)
            failure_events.extend(cascade_events)
        
        # Analyze impact
        end_time = datetime.now()
        return self._build_result(
            graph, failed_graph, original_reach, original_cc,
            failed_components, cascade_failures, failure_events,
            SimulationMode.MULTIPLE,
            start_time, end_time
        )
    
    def simulate_all_single_failures(self,
                                     graph: nx.DiGraph,
                                     component_types: Optional[List[str]] = None,
                                     enable_cascade: bool = False) -> BatchSimulationResult:
        """
        Simulate failure of each component individually.
        
        This is useful for:
        - Identifying most critical components
        - Validating criticality predictions
        - Generating impact scores for correlation analysis
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON edges
            component_types: Filter by types (e.g., ['Application', 'Broker'])
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            BatchSimulationResult with all individual results
        """
        results = []
        
        # Get components to test
        components = []
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            if component_types is None or node_type in component_types:
                components.append(node)
        
        self.logger.info(f"Running exhaustive simulation on {len(components)} components")
        
        for component in components:
            result = self.simulate_failure(
                graph, component,
                failure_mode=FailureMode.CRASH,
                enable_cascade=enable_cascade
            )
            results.append(result)
        
        # Build summary
        summary = self._build_batch_summary(results)
        
        return BatchSimulationResult(
            total_simulations=len(results),
            results=results,
            summary=summary
        )
    
    # =========================================================================
    # Cascade Simulation
    # =========================================================================
    
    def _simulate_cascade(self,
                         original_graph: nx.DiGraph,
                         failed_graph: nx.DiGraph,
                         initial_failures: List[str],
                         failure_mode: FailureMode) -> Tuple[List[str], List[FailureEvent]]:
        """
        Simulate cascading failures through DEPENDS_ON relationships.
        
        A component may fail in cascade if:
        1. It has lost a significant portion of its dependencies
        2. Random probability check passes
        
        Args:
            original_graph: Original graph
            failed_graph: Graph with initial failures removed
            initial_failures: Components that initially failed
            failure_mode: How components fail
        
        Returns:
            Tuple of (cascade_failures, cascade_events)
        """
        cascade_failures = []
        cascade_events = []
        failed_set = set(initial_failures)
        
        for depth in range(1, self.max_cascade_depth + 1):
            new_failures = []
            
            for node in list(failed_graph.nodes()):
                if node in failed_set:
                    continue
                
                # Check if this node should fail due to cascade
                if self._should_cascade(original_graph, failed_graph, node, failed_set):
                    new_failures.append(node)
                    comp_type = original_graph.nodes[node].get('type', 'Unknown')
                    
                    cascade_events.append(FailureEvent(
                        component=node,
                        component_type=comp_type,
                        failure_mode=failure_mode,
                        timestamp=datetime.now(),
                        is_cascade=True,
                        cascade_depth=depth,
                        cause=f"cascade_depth_{depth}"
                    ))
            
            if not new_failures:
                break
            
            # Apply new failures
            for node in new_failures:
                failed_graph.remove_node(node)
                failed_set.add(node)
                cascade_failures.append(node)
        
        return cascade_failures, cascade_events
    
    def _should_cascade(self,
                       original_graph: nx.DiGraph,
                       failed_graph: nx.DiGraph,
                       node: str,
                       failed_set: Set[str]) -> bool:
        """
        Determine if a component should fail due to cascade.
        
        Uses dependency loss ratio and probability to decide.
        """
        # Get original dependencies (predecessors in DEPENDS_ON graph)
        # In our graph, if A depends on B, edge is A -> B
        # So predecessors of a node are things that depend ON it
        # And successors are things it DEPENDS ON
        
        original_deps = set(original_graph.successors(node))
        if not original_deps:
            return False
        
        # Calculate how many dependencies were lost
        remaining_deps = original_deps - failed_set
        if node in failed_graph:
            remaining_deps = remaining_deps & set(failed_graph.nodes())
        
        lost_ratio = 1.0 - len(remaining_deps) / len(original_deps)
        
        # Fail if lost too many dependencies
        if lost_ratio >= self.cascade_threshold:
            return True
        
        # Also apply random probability based on loss
        return random.random() < (lost_ratio * self.cascade_probability)
    
    # =========================================================================
    # Impact Analysis
    # =========================================================================
    
    def _calculate_reachability(self, graph: nx.DiGraph) -> Set[Tuple[str, str]]:
        """
        Calculate all reachable pairs in the graph.
        
        Returns:
            Set of (source, target) pairs where target is reachable from source
        """
        reachable = set()
        for source in graph.nodes():
            try:
                descendants = nx.descendants(graph, source)
                for target in descendants:
                    if source != target:
                        reachable.add((source, target))
            except nx.NetworkXError:
                pass
        return reachable
    
    def _find_affected_components(self,
                                  original_graph: nx.DiGraph,
                                  failed_graph: nx.DiGraph,
                                  failed_set: Set[str]) -> List[str]:
        """
        Find components that are affected but not failed.
        
        A component is affected if its connectivity changed.
        """
        affected = []
        
        for node in failed_graph.nodes():
            if node in failed_set:
                continue
            
            # Check if any neighbor was lost
            original_neighbors = (
                set(original_graph.predecessors(node)) | 
                set(original_graph.successors(node))
            )
            current_neighbors = (
                set(failed_graph.predecessors(node)) | 
                set(failed_graph.successors(node))
            )
            
            if original_neighbors != current_neighbors:
                affected.append(node)
        
        return affected
    
    def _find_isolated_components(self, graph: nx.DiGraph) -> List[str]:
        """Find components with no connections."""
        isolated = []
        for node in graph.nodes():
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                isolated.append(node)
        return isolated
    
    def _build_result(self,
                     original_graph: nx.DiGraph,
                     failed_graph: nx.DiGraph,
                     original_reach: Set[Tuple[str, str]],
                     original_cc: int,
                     failed_components: List[str],
                     cascade_failures: List[str],
                     failure_events: List[FailureEvent],
                     simulation_mode: SimulationMode,
                     start_time: datetime,
                     end_time: datetime) -> SimulationResult:
        """Build complete simulation result."""
        
        # Calculate new reachability
        new_reach = self._calculate_reachability(failed_graph)
        lost_reach = original_reach - new_reach
        
        # Find affected and isolated
        failed_set = set(failed_components)
        affected = self._find_affected_components(original_graph, failed_graph, failed_set)
        isolated = self._find_isolated_components(failed_graph)
        
        # Connectivity
        new_cc = nx.number_weakly_connected_components(failed_graph) if len(failed_graph) > 0 else 0
        
        # Calculate impact score
        # Impact = (failed + affected) / total_original
        total_original = original_graph.number_of_nodes()
        total_affected = len(failed_components) + len(affected)
        impact_score = total_affected / total_original if total_original > 0 else 0.0
        
        # Reachability loss percentage
        reach_loss_pct = (len(lost_reach) / len(original_reach) * 100) if original_reach else 0.0
        
        # Statistics by type
        stats = self._compute_statistics(original_graph, failed_components, cascade_failures)
        
        return SimulationResult(
            simulation_id=f"sim_{self._simulation_counter}",
            simulation_mode=simulation_mode,
            start_time=start_time,
            end_time=end_time,
            failed_components=failed_components,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            impact_score=impact_score,
            resilience_score=1.0 - impact_score,
            original_reachability=len(original_reach),
            remaining_reachability=len(new_reach),
            reachability_loss=len(lost_reach),
            reachability_loss_pct=reach_loss_pct,
            affected_components=affected,
            isolated_components=isolated,
            original_components=original_cc,
            resulting_components=new_cc,
            fragmentation=new_cc - original_cc,
            statistics=stats
        )
    
    def _compute_statistics(self,
                           graph: nx.DiGraph,
                           failed_components: List[str],
                           cascade_failures: List[str]) -> Dict[str, Any]:
        """Compute failure statistics by type."""
        by_type = defaultdict(int)
        for comp in failed_components:
            if comp in graph.nodes():
                comp_type = graph.nodes[comp].get('type', 'Unknown')
                by_type[comp_type] += 1
        
        return {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'failed_count': len(failed_components),
            'cascade_count': len(cascade_failures),
            'primary_failures': len(failed_components) - len(cascade_failures),
            'by_type': dict(by_type)
        }
    
    def _build_batch_summary(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Build summary statistics for batch simulation."""
        if not results:
            return {}
        
        impact_scores = [r.impact_score for r in results]
        reach_losses = [r.reachability_loss_pct for r in results]
        cascade_counts = [len(r.cascade_failures) for r in results]
        
        # Find most impactful
        max_impact_result = max(results, key=lambda r: r.impact_score)
        
        return {
            'impact_score': {
                'min': round(min(impact_scores), 4),
                'max': round(max(impact_scores), 4),
                'mean': round(sum(impact_scores) / len(impact_scores), 4)
            },
            'reachability_loss': {
                'min': round(min(reach_losses), 2),
                'max': round(max(reach_losses), 2),
                'mean': round(sum(reach_losses) / len(reach_losses), 2)
            },
            'cascade_failures': {
                'total': sum(cascade_counts),
                'max': max(cascade_counts),
                'simulations_with_cascade': sum(1 for c in cascade_counts if c > 0)
            },
            'most_impactful': {
                'component': max_impact_result.failed_components[0] if max_impact_result.failed_components else None,
                'impact_score': round(max_impact_result.impact_score, 4),
                'reachability_loss_pct': round(max_impact_result.reachability_loss_pct, 2)
            }
        }
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self, result: SimulationResult) -> Dict[str, Any]:
        """
        Generate a human-readable report from simulation result.
        
        Args:
            result: SimulationResult to report on
        
        Returns:
            Dictionary with report sections
        """
        severity = self._classify_severity(result.impact_score)
        
        return {
            'summary': {
                'simulation_id': result.simulation_id,
                'mode': result.simulation_mode.value,
                'severity': severity,
                'impact_score': f"{result.impact_score:.1%}",
                'resilience_score': f"{result.resilience_score:.1%}",
                'reachability_loss': f"{result.reachability_loss_pct:.1f}%"
            },
            'failures': {
                'primary': len(result.failed_components) - len(result.cascade_failures),
                'cascade': len(result.cascade_failures),
                'total': len(result.failed_components),
                'components': result.failed_components
            },
            'impact': {
                'affected_components': len(result.affected_components),
                'isolated_components': len(result.isolated_components),
                'fragmentation': result.fragmentation
            },
            'recommendations': self._generate_recommendations(result)
        }
    
    def _classify_severity(self, impact_score: float) -> str:
        """Classify impact severity."""
        if impact_score >= 0.7:
            return "CRITICAL"
        elif impact_score >= 0.5:
            return "HIGH"
        elif impact_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, result: SimulationResult) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        if result.impact_score >= 0.7:
            recommendations.append(
                f"CRITICAL: Failure causes {result.impact_score:.0%} impact. "
                "Consider adding redundancy or failover mechanisms."
            )
        
        if len(result.cascade_failures) > 0:
            recommendations.append(
                f"CASCADE RISK: {len(result.cascade_failures)} cascade failures. "
                "Consider circuit breakers to contain failure propagation."
            )
        
        if len(result.isolated_components) > 0:
            recommendations.append(
                f"ISOLATION: {len(result.isolated_components)} components isolated. "
                "Review topology for redundant paths."
            )
        
        if result.fragmentation > 0:
            recommendations.append(
                f"FRAGMENTATION: System split into {result.fragmentation} additional components. "
                "Add backup connections for resilience."
            )
        
        if result.reachability_loss_pct > 50:
            recommendations.append(
                f"REACHABILITY: {result.reachability_loss_pct:.0f}% loss. "
                "Critical paths need redundancy."
            )
        
        if not recommendations:
            recommendations.append("Failure impact is within acceptable bounds.")
        
        return recommendations


# ============================================================================
# Convenience Functions
# ============================================================================

def simulate_single_failure(graph: nx.DiGraph,
                           component: str,
                           enable_cascade: bool = False) -> SimulationResult:
    """
    Convenience function to simulate a single failure.
    
    Args:
        graph: NetworkX DiGraph with DEPENDS_ON edges
        component: Component ID to fail
        enable_cascade: Whether to simulate cascading failures
    
    Returns:
        SimulationResult
    """
    simulator = GraphSimulator()
    return simulator.simulate_failure(graph, component, enable_cascade=enable_cascade)


def simulate_and_rank(graph: nx.DiGraph,
                     component_types: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """
    Simulate all single failures and rank by impact.
    
    Args:
        graph: NetworkX DiGraph with DEPENDS_ON edges
        component_types: Filter by component types
    
    Returns:
        List of (component_id, impact_score) sorted by impact descending
    """
    simulator = GraphSimulator()
    batch_result = simulator.simulate_all_single_failures(graph, component_types)
    return batch_result.get_impact_ranking()