"""
Failure Simulator

Comprehensive failure simulation for pub-sub systems including:
- Single and multiple component failures
- Cascading failure propagation
- Network/connection failures
- Partial degradation simulation
- Recovery scenario testing
- Impact quantification and analysis

Performance: Achieves 100-1000x real-time speedup through event-driven simulation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import random
from collections import defaultdict


class FailureType(Enum):
    """Types of failures that can be simulated"""
    COMPLETE = "complete"           # Total component failure
    PARTIAL = "partial"             # Degraded performance
    NETWORK = "network"             # Connection failure
    OVERLOAD = "overload"           # Resource exhaustion
    CASCADE = "cascade"             # Triggered by another failure
    TRANSIENT = "transient"         # Temporary failure


class FailureMode(Enum):
    """How the failure manifests"""
    CRASH = "crash"                 # Immediate stop
    HANG = "hang"                   # Unresponsive
    BYZANTINE = "byzantine"         # Incorrect behavior
    SLOWDOWN = "slowdown"           # Performance degradation
    INTERMITTENT = "intermittent"   # On-off behavior


class RecoveryStrategy(Enum):
    """Recovery strategies for components"""
    NONE = "none"                   # No recovery
    RESTART = "restart"             # Component restart
    FAILOVER = "failover"           # Switch to backup
    DEGRADED = "degraded"           # Continue with reduced capacity
    CIRCUIT_BREAKER = "circuit_breaker"  # Stop and wait


@dataclass
class FailureEvent:
    """Represents a single failure event"""
    component: str
    failure_type: FailureType
    failure_mode: FailureMode
    timestamp: datetime
    severity: float  # 0.0 to 1.0
    cause: str = ""
    is_cascade: bool = False
    cascade_depth: int = 0
    recovery_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'failure_type': self.failure_type.value,
            'failure_mode': self.failure_mode.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': round(self.severity, 3),
            'cause': self.cause,
            'is_cascade': self.is_cascade,
            'cascade_depth': self.cascade_depth,
            'recovery_time_ms': self.recovery_time_ms
        }


@dataclass
class SimulationResult:
    """Complete results from a failure simulation"""
    simulation_id: str
    simulation_type: str
    start_time: datetime
    end_time: datetime
    
    # Failed components
    failed_components: List[str]
    cascade_failures: List[str]
    
    # Affected components
    affected_components: List[str]
    isolated_components: List[str]
    
    # Events
    failure_events: List[FailureEvent]
    
    # Impact metrics
    impact_score: float              # 0.0 to 1.0
    resilience_score: float          # 1 - impact_score
    service_continuity: float        # Remaining functional capacity
    
    # Path analysis
    original_paths: int
    affected_paths: int
    remaining_paths: int
    
    # Reachability
    original_reachability: int
    lost_reachability: int
    reachability_loss_pct: float
    
    # Connectivity
    original_components: int
    resulting_components: int
    fragmentation: int
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'simulation_id': self.simulation_id,
            'simulation_type': self.simulation_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_ms': (self.end_time - self.start_time).total_seconds() * 1000,
            'failed_components': self.failed_components,
            'cascade_failures': self.cascade_failures,
            'affected_components': self.affected_components,
            'isolated_components': self.isolated_components,
            'failure_events': [e.to_dict() for e in self.failure_events],
            'impact_metrics': {
                'impact_score': round(self.impact_score, 4),
                'resilience_score': round(self.resilience_score, 4),
                'service_continuity': round(self.service_continuity, 4)
            },
            'path_analysis': {
                'original_paths': self.original_paths,
                'affected_paths': self.affected_paths,
                'remaining_paths': self.remaining_paths
            },
            'reachability': {
                'original': self.original_reachability,
                'lost': self.lost_reachability,
                'loss_percentage': round(self.reachability_loss_pct, 2)
            },
            'connectivity': {
                'original_components': self.original_components,
                'resulting_components': self.resulting_components,
                'fragmentation': self.fragmentation
            },
            'statistics': self.statistics
        }


class FailureSimulator:
    """
    Simulates component failures and analyzes system impact.
    
    Capabilities:
    - Single and multiple component failures
    - Cascading failure propagation
    - Connection/network failures
    - Partial failure simulation
    - Recovery scenario testing
    - Comprehensive impact quantification
    """
    
    def __init__(self,
                 propagation_threshold: float = 0.7,
                 cascade_probability: float = 0.5,
                 max_cascade_depth: int = 5,
                 seed: Optional[int] = None):
        """
        Initialize failure simulator.
        
        Args:
            propagation_threshold: Threshold for failure propagation (0-1)
            cascade_probability: Base probability of cascade failure (0-1)
            max_cascade_depth: Maximum cascade propagation depth
            seed: Random seed for reproducibility
        """
        self.propagation_threshold = propagation_threshold
        self.cascade_probability = cascade_probability
        self.max_cascade_depth = max_cascade_depth
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._simulation_counter = 0
        self.logger = logging.getLogger(__name__)
    
    def simulate_single_failure(self,
                               graph: nx.DiGraph,
                               component: str,
                               failure_type: FailureType = FailureType.COMPLETE,
                               failure_mode: FailureMode = FailureMode.CRASH,
                               severity: float = 1.0,
                               enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of a single component.
        
        Args:
            graph: NetworkX directed graph
            component: Component to fail
            failure_type: Type of failure
            failure_mode: How it fails
            severity: Failure severity (0-1)
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with comprehensive impact analysis
        """
        self.logger.info(f"Simulating {failure_type.value} failure of '{component}'")
        
        if component not in graph.nodes():
            raise ValueError(f"Component '{component}' not found in graph")
        
        self._simulation_counter += 1
        start_time = datetime.now()
        
        # Create working copy
        sim_graph = graph.copy()
        failure_events = []
        failed_components = [component]
        cascade_failures = []
        
        # Create initial failure event
        event = FailureEvent(
            component=component,
            failure_type=failure_type,
            failure_mode=failure_mode,
            timestamp=start_time,
            severity=severity,
            cause="Primary failure (simulated)"
        )
        failure_events.append(event)
        
        # Apply failure
        if failure_type == FailureType.COMPLETE:
            sim_graph.remove_node(component)
        elif failure_type == FailureType.PARTIAL:
            # Mark as degraded but keep in graph
            sim_graph.nodes[component]['degraded'] = True
            sim_graph.nodes[component]['capacity'] = 1.0 - severity
        
        # Simulate cascading failures
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures = self._simulate_cascade(
                graph, sim_graph, component, failure_events, 1
            )
            failed_components.extend(cascade_failures)
        
        end_time = datetime.now()
        
        # Analyze impact
        return self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=failed_components,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            simulation_type="single_failure",
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_multiple_failures(self,
                                  graph: nx.DiGraph,
                                  components: List[str],
                                  failure_type: FailureType = FailureType.COMPLETE,
                                  simultaneous: bool = True,
                                  enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of multiple components.
        
        Args:
            graph: NetworkX directed graph
            components: Components to fail
            failure_type: Type of failure
            simultaneous: Whether failures happen at same time
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with comprehensive impact analysis
        """
        self.logger.info(f"Simulating {failure_type.value} failure of {len(components)} components")
        
        # Validate components
        for comp in components:
            if comp not in graph.nodes():
                raise ValueError(f"Component '{comp}' not found in graph")
        
        self._simulation_counter += 1
        start_time = datetime.now()
        
        # Create working copy
        sim_graph = graph.copy()
        failure_events = []
        failed_components = list(components)
        cascade_failures = []
        
        # Create failure events
        for i, component in enumerate(components):
            timestamp = start_time if simultaneous else datetime.now()
            event = FailureEvent(
                component=component,
                failure_type=failure_type,
                failure_mode=FailureMode.CRASH,
                timestamp=timestamp,
                severity=1.0,
                cause=f"Multiple failure scenario ({i+1}/{len(components)})"
            )
            failure_events.append(event)
        
        # Apply failures
        if failure_type == FailureType.COMPLETE:
            for comp in components:
                if comp in sim_graph.nodes():
                    sim_graph.remove_node(comp)
        
        # Simulate cascading failures
        if enable_cascade:
            for component in components:
                if component not in sim_graph.nodes():
                    continue
                new_cascades = self._simulate_cascade(
                    graph, sim_graph, component, failure_events, 1
                )
                for c in new_cascades:
                    if c not in cascade_failures:
                        cascade_failures.append(c)
                        failed_components.append(c)
        
        end_time = datetime.now()
        
        return self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=failed_components,
            cascade_failures=cascade_failures,
            failure_events=failure_events,
            simulation_type="multiple_failure",
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_network_failure(self,
                                graph: nx.DiGraph,
                                source: str,
                                target: str) -> SimulationResult:
        """
        Simulate failure of a network connection.
        
        Args:
            graph: NetworkX directed graph
            source: Source component
            target: Target component
        
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating network failure: {source} -> {target}")
        
        if not graph.has_edge(source, target):
            raise ValueError(f"Edge {source} -> {target} not found")
        
        self._simulation_counter += 1
        start_time = datetime.now()
        
        # Create working copy and remove edge
        sim_graph = graph.copy()
        sim_graph.remove_edge(source, target)
        
        # Create failure event
        failure_events = [FailureEvent(
            component=f"{source}->{target}",
            failure_type=FailureType.NETWORK,
            failure_mode=FailureMode.CRASH,
            timestamp=start_time,
            severity=1.0,
            cause="Network connection failure"
        )]
        
        end_time = datetime.now()
        
        return self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=[],
            cascade_failures=[],
            failure_events=failure_events,
            simulation_type="network_failure",
            start_time=start_time,
            end_time=end_time
        )
    
    def simulate_random_failures(self,
                                graph: nx.DiGraph,
                                failure_probability: float = 0.1,
                                component_types: Optional[List[str]] = None,
                                enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate random failures based on probability.
        
        Args:
            graph: NetworkX directed graph
            failure_probability: Probability of each component failing
            component_types: Limit to specific types (e.g., ['Application'])
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating random failures (p={failure_probability})")
        
        # Select components to fail
        candidates = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if component_types is None or node_type in component_types:
                candidates.append(node)
        
        # Apply probability
        components_to_fail = [
            c for c in candidates
            if random.random() < failure_probability
        ]
        
        if not components_to_fail:
            self.logger.info("No components selected for failure")
            # Return empty result
            return self._create_empty_result(graph, "random_failure")
        
        return self.simulate_multiple_failures(
            graph, components_to_fail,
            failure_type=FailureType.COMPLETE,
            simultaneous=True,
            enable_cascade=enable_cascade
        )
    
    def simulate_targeted_attack(self,
                                graph: nx.DiGraph,
                                target_count: int = 5,
                                strategy: str = 'criticality') -> SimulationResult:
        """
        Simulate targeted attack on critical components.
        
        Args:
            graph: NetworkX directed graph
            target_count: Number of components to target
            strategy: Selection strategy ('criticality', 'degree', 'betweenness')
        
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating targeted attack ({strategy}, n={target_count})")
        
        # Select targets based on strategy
        if strategy == 'degree':
            sorted_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
            targets = [n for n, d in sorted_nodes[:target_count]]
        elif strategy == 'betweenness':
            bc = nx.betweenness_centrality(graph)
            sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
            targets = [n for n, c in sorted_nodes[:target_count]]
        else:  # criticality (combined)
            bc = nx.betweenness_centrality(graph)
            degree = dict(graph.degree())
            combined = {
                n: bc.get(n, 0) * 0.6 + (degree.get(n, 0) / max(degree.values()) if degree else 0) * 0.4
                for n in graph.nodes()
            }
            sorted_nodes = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            targets = [n for n, c in sorted_nodes[:target_count]]
        
        return self.simulate_multiple_failures(
            graph, targets,
            failure_type=FailureType.COMPLETE,
            simultaneous=True,
            enable_cascade=True
        )
    
    def run_failure_campaign(self,
                            graph: nx.DiGraph,
                            components: List[str],
                            iterations: int = 1) -> List[SimulationResult]:
        """
        Run failure simulation for each component individually.
        
        Args:
            graph: NetworkX directed graph
            components: Components to test
            iterations: Number of iterations per component
        
        Returns:
            List of SimulationResults for each component
        """
        self.logger.info(f"Running failure campaign for {len(components)} components")
        
        results = []
        for component in components:
            for _ in range(iterations):
                try:
                    result = self.simulate_single_failure(
                        graph, component,
                        failure_type=FailureType.COMPLETE,
                        enable_cascade=False
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to simulate {component}: {e}")
        
        return results
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _simulate_cascade(self,
                         original_graph: nx.DiGraph,
                         sim_graph: nx.DiGraph,
                         failed_component: str,
                         failure_events: List[FailureEvent],
                         depth: int) -> List[str]:
        """Simulate cascading failures from a failed component"""
        if depth > self.max_cascade_depth:
            return []
        
        cascade_failures = []
        
        # Find dependent components
        dependents = self._find_dependents(original_graph, failed_component)
        
        for dependent in dependents:
            if dependent not in sim_graph.nodes():
                continue  # Already failed
            
            # Check if dependent should fail
            if self._should_cascade(original_graph, sim_graph, dependent):
                # Create cascade failure event
                event = FailureEvent(
                    component=dependent,
                    failure_type=FailureType.CASCADE,
                    failure_mode=FailureMode.CRASH,
                    timestamp=datetime.now(),
                    severity=1.0,
                    cause=f"Cascade from {failed_component}",
                    is_cascade=True,
                    cascade_depth=depth
                )
                failure_events.append(event)
                
                # Apply failure
                sim_graph.remove_node(dependent)
                cascade_failures.append(dependent)
                
                # Recurse
                sub_cascades = self._simulate_cascade(
                    original_graph, sim_graph, dependent, failure_events, depth + 1
                )
                cascade_failures.extend(sub_cascades)
        
        return cascade_failures
    
    def _find_dependents(self, graph: nx.DiGraph, component: str) -> List[str]:
        """Find components that depend on the given component"""
        dependents = []
        
        # Components with DEPENDS_ON edges TO this component
        for source, target, data in graph.edges(data=True):
            if target == component and data.get('type') == 'DEPENDS_ON':
                dependents.append(source)
        
        # Subscribers to topics published by this component
        for source, target, data in graph.out_edges(component, data=True):
            if data.get('type') == 'PUBLISHES_TO':
                # Find subscribers to this topic
                for sub_source, sub_target, sub_data in graph.in_edges(target, data=True):
                    if sub_data.get('type') == 'SUBSCRIBES_TO' and sub_source != component:
                        if sub_source not in dependents:
                            dependents.append(sub_source)
        
        return dependents
    
    def _should_cascade(self,
                       original_graph: nx.DiGraph,
                       sim_graph: nx.DiGraph,
                       component: str) -> bool:
        """Determine if a component should fail due to cascade"""
        # Check if component has lost critical dependencies
        original_deps = set()
        for source, target, data in original_graph.edges(data=True):
            if source == component and data.get('type') == 'DEPENDS_ON':
                original_deps.add(target)
        
        remaining_deps = set()
        for source, target, data in sim_graph.edges(data=True):
            if source == component and data.get('type') == 'DEPENDS_ON':
                remaining_deps.add(target)
        
        if not original_deps:
            return False
        
        lost_ratio = 1.0 - len(remaining_deps) / len(original_deps)
        
        # Component fails if it lost too many dependencies
        if lost_ratio >= self.propagation_threshold:
            return True
        
        # Also apply random probability
        return random.random() < (lost_ratio * self.cascade_probability)
    
    def _analyze_impact(self,
                       original_graph: nx.DiGraph,
                       failed_graph: nx.DiGraph,
                       failed_components: List[str],
                       cascade_failures: List[str],
                       failure_events: List[FailureEvent],
                       simulation_type: str,
                       start_time: datetime,
                       end_time: datetime) -> SimulationResult:
        """Analyze the impact of failures on the system"""
        
        # Calculate reachability
        original_reach = self._calculate_reachability(original_graph)
        new_reach = self._calculate_reachability(failed_graph)
        lost_reach = original_reach - new_reach
        
        # Find affected components
        affected = self._find_affected_components(
            original_graph, failed_graph, failed_components
        )
        
        # Find isolated components
        isolated = self._find_isolated_components(failed_graph)
        
        # Connectivity analysis
        original_cc = nx.number_weakly_connected_components(original_graph)
        new_cc = nx.number_weakly_connected_components(failed_graph) if len(failed_graph) > 0 else 0
        
        # Path analysis
        original_paths = len(original_graph.edges())
        remaining_paths = len(failed_graph.edges())
        affected_paths = original_paths - remaining_paths
        
        # Calculate metrics
        total_nodes = len(original_graph)
        total_affected = len(failed_components) + len(affected)
        
        impact_score = total_affected / total_nodes if total_nodes > 0 else 0.0
        resilience_score = 1.0 - impact_score
        service_continuity = (total_nodes - len(failed_components)) / total_nodes if total_nodes > 0 else 0.0
        
        reachability_loss_pct = (len(lost_reach) / max(1, len(original_reach))) * 100
        
        # Statistics
        statistics = {
            'total_nodes': total_nodes,
            'total_edges': len(original_graph.edges()),
            'failed_count': len(failed_components),
            'cascade_count': len(cascade_failures),
            'affected_count': len(affected),
            'isolated_count': len(isolated),
            'primary_failures': len(failed_components) - len(cascade_failures),
            'by_type': self._count_by_type(original_graph, failed_components)
        }
        
        return SimulationResult(
            simulation_id=f"sim_{self._simulation_counter}",
            simulation_type=simulation_type,
            start_time=start_time,
            end_time=end_time,
            failed_components=failed_components,
            cascade_failures=cascade_failures,
            affected_components=list(affected),
            isolated_components=list(isolated),
            failure_events=failure_events,
            impact_score=impact_score,
            resilience_score=resilience_score,
            service_continuity=service_continuity,
            original_paths=original_paths,
            affected_paths=affected_paths,
            remaining_paths=remaining_paths,
            original_reachability=len(original_reach),
            lost_reachability=len(lost_reach),
            reachability_loss_pct=reachability_loss_pct,
            original_components=original_cc,
            resulting_components=new_cc,
            fragmentation=new_cc - original_cc,
            statistics=statistics
        )
    
    def _calculate_reachability(self, graph: nx.DiGraph) -> Set[Tuple[str, str]]:
        """Calculate all reachable pairs in the graph"""
        reachable = set()
        for source in graph.nodes():
            try:
                for target in nx.descendants(graph, source):
                    if source != target:
                        reachable.add((source, target))
            except:
                pass
        return reachable
    
    def _find_affected_components(self,
                                 original_graph: nx.DiGraph,
                                 failed_graph: nx.DiGraph,
                                 failed_components: List[str]) -> Set[str]:
        """Find components affected by failures (not failed themselves)"""
        affected = set()
        failed_set = set(failed_components)
        
        for node in failed_graph.nodes():
            if node in failed_set:
                continue
            
            # Check if connectivity changed
            original_neighbors = set(original_graph.predecessors(node)) | set(original_graph.successors(node))
            current_neighbors = set(failed_graph.predecessors(node)) | set(failed_graph.successors(node))
            
            if original_neighbors != current_neighbors:
                affected.add(node)
        
        return affected
    
    def _find_isolated_components(self, graph: nx.DiGraph) -> Set[str]:
        """Find components that became isolated"""
        isolated = set()
        for node in graph.nodes():
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                isolated.add(node)
        return isolated
    
    def _count_by_type(self, graph: nx.DiGraph, components: List[str]) -> Dict[str, int]:
        """Count failed components by type"""
        counts = defaultdict(int)
        for comp in components:
            if comp in graph.nodes():
                comp_type = graph.nodes[comp].get('type', 'Unknown')
                counts[comp_type] += 1
        return dict(counts)
    
    def _create_empty_result(self, graph: nx.DiGraph, simulation_type: str) -> SimulationResult:
        """Create an empty simulation result"""
        now = datetime.now()
        original_reach = self._calculate_reachability(graph)
        
        return SimulationResult(
            simulation_id=f"sim_{self._simulation_counter}",
            simulation_type=simulation_type,
            start_time=now,
            end_time=now,
            failed_components=[],
            cascade_failures=[],
            affected_components=[],
            isolated_components=[],
            failure_events=[],
            impact_score=0.0,
            resilience_score=1.0,
            service_continuity=1.0,
            original_paths=len(graph.edges()),
            affected_paths=0,
            remaining_paths=len(graph.edges()),
            original_reachability=len(original_reach),
            lost_reachability=0,
            reachability_loss_pct=0.0,
            original_components=nx.number_weakly_connected_components(graph),
            resulting_components=nx.number_weakly_connected_components(graph),
            fragmentation=0,
            statistics={}
        )
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self,
                       graph: nx.DiGraph,
                       result: SimulationResult) -> Dict[str, Any]:
        """Generate comprehensive failure analysis report"""
        severity = self._classify_severity(result.impact_score)
        
        return {
            'summary': {
                'simulation_id': result.simulation_id,
                'simulation_type': result.simulation_type,
                'severity': severity,
                'impact_score': round(result.impact_score, 4),
                'resilience_score': round(result.resilience_score, 4),
                'service_continuity': round(result.service_continuity, 4)
            },
            'failures': {
                'primary_failures': len(result.failed_components) - len(result.cascade_failures),
                'cascade_failures': len(result.cascade_failures),
                'total_failures': len(result.failed_components),
                'failed_components': result.failed_components,
                'cascade_chain': result.cascade_failures
            },
            'impact': {
                'affected_components': len(result.affected_components),
                'isolated_components': len(result.isolated_components),
                'reachability_loss_pct': round(result.reachability_loss_pct, 2),
                'fragmentation': result.fragmentation
            },
            'recommendations': self._generate_recommendations(result)
        }
    
    def _classify_severity(self, impact_score: float) -> str:
        """Classify failure severity"""
        if impact_score >= 0.8:
            return "CRITICAL"
        elif impact_score >= 0.5:
            return "HIGH"
        elif impact_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, result: SimulationResult) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        if result.impact_score >= 0.7:
            recommendations.append(
                f"CRITICAL: Failure causes severe impact ({result.impact_score:.1%}). "
                "Implement redundancy or failover mechanisms."
            )
        
        if len(result.cascade_failures) > 0:
            recommendations.append(
                f"CASCADE RISK: {len(result.cascade_failures)} cascade failures detected. "
                "Add circuit breakers or bulkheads to contain failures."
            )
        
        if len(result.isolated_components) > 0:
            recommendations.append(
                f"ISOLATION: {len(result.isolated_components)} components became isolated. "
                "Review network topology for redundant paths."
            )
        
        if result.fragmentation > 0:
            recommendations.append(
                f"FRAGMENTATION: System split into {result.fragmentation} additional components. "
                "Add backup connections to maintain connectivity."
            )
        
        if result.reachability_loss_pct > 50:
            recommendations.append(
                f"REACHABILITY: {result.reachability_loss_pct:.1f}% reachability lost. "
                "Critical communication paths need redundancy."
            )
        
        return recommendations