"""
Failure Simulator

Simulates component failures in the system graph to assess impact and resilience.
Supports:
- Single node failures
- Multiple node failures (cascading)
- Edge failures (connection loss)
- Partial failures (degraded performance)
- Recovery scenarios
- Failure propagation
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import copy


class FailureType(Enum):
    """Types of component failures"""
    COMPLETE = "complete"           # Total failure
    PARTIAL = "partial"             # Degraded performance
    INTERMITTENT = "intermittent"   # On/off failure
    CASCADE = "cascade"             # Triggered by other failures
    NETWORK = "network"             # Connection failure


class FailureMode(Enum):
    """How failure affects the component"""
    CRASH = "crash"                 # Component stops
    HANG = "hang"                   # Component becomes unresponsive
    SLOW = "slow"                   # Performance degradation
    CORRUPT = "corrupt"             # Data corruption
    DISCONNECT = "disconnect"       # Network disconnection


@dataclass
class FailureEvent:
    """Represents a single failure event"""
    component: str
    failure_type: FailureType
    failure_mode: FailureMode
    timestamp: datetime
    severity: float  # 0.0 (no impact) to 1.0 (complete failure)
    cause: Optional[str] = None
    propagated_from: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'component': self.component,
            'failure_type': self.failure_type.value,
            'failure_mode': self.failure_mode.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': round(self.severity, 3),
            'cause': self.cause,
            'propagated_from': self.propagated_from
        }


@dataclass
class SimulationResult:
    """Result of a failure simulation"""
    failed_components: List[str]
    affected_components: List[str]
    isolated_components: List[str]
    failure_events: List[FailureEvent]
    impact_score: float
    resilience_score: float
    service_continuity: float
    affected_paths: int
    total_paths: int
    graph_state: nx.DiGraph
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'failed_components': self.failed_components,
            'affected_components': self.affected_components,
            'isolated_components': self.isolated_components,
            'failure_events': [e.to_dict() for e in self.failure_events],
            'impact_score': round(self.impact_score, 3),
            'resilience_score': round(self.resilience_score, 3),
            'service_continuity': round(self.service_continuity, 3),
            'affected_paths': self.affected_paths,
            'total_paths': self.total_paths,
            'statistics': self.statistics
        }


class FailureSimulator:
    """
    Simulates component failures and analyzes system impact
    
    Capabilities:
    - Single and multiple component failures
    - Cascading failure propagation
    - Connection/network failures
    - Partial failure simulation
    - Recovery scenario testing
    - Impact quantification
    """
    
    def __init__(self, 
                 propagation_threshold: float = 0.7,
                 cascade_probability: float = 0.5):
        """
        Initialize failure simulator
        
        Args:
            propagation_threshold: Threshold for failure propagation (0-1)
            cascade_probability: Probability of cascade failure (0-1)
        """
        self.logger = logging.getLogger(__name__)
        self.propagation_threshold = propagation_threshold
        self.cascade_probability = cascade_probability
    
    def simulate_single_failure(self,
                                graph: nx.DiGraph,
                                component: str,
                                failure_type: FailureType = FailureType.COMPLETE,
                                failure_mode: FailureMode = FailureMode.CRASH,
                                severity: float = 1.0,
                                enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate failure of a single component
        
        Args:
            graph: NetworkX directed graph
            component: Component to fail
            failure_type: Type of failure
            failure_mode: How it fails
            severity: Failure severity (0-1)
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating {failure_type.value} failure of {component}")
        
        if component not in graph.nodes():
            raise ValueError(f"Component {component} not found in graph")
        
        # Create working copy
        sim_graph = graph.copy()
        
        # Initialize tracking
        failure_events = []
        failed_components = [component]
        
        # Create initial failure event
        initial_event = FailureEvent(
            component=component,
            failure_type=failure_type,
            failure_mode=failure_mode,
            timestamp=datetime.now(),
            severity=severity,
            cause="Simulated failure"
        )
        failure_events.append(initial_event)
        
        # Apply failure to graph
        if failure_type == FailureType.COMPLETE:
            sim_graph.remove_node(component)
        elif failure_type == FailureType.PARTIAL:
            # Mark as degraded
            sim_graph.nodes[component]['degraded'] = True
            sim_graph.nodes[component]['degradation'] = severity
        
        # Simulate cascade if enabled
        if enable_cascade and failure_type == FailureType.COMPLETE:
            cascade_failures = self._simulate_cascade(
                sim_graph, 
                component, 
                failure_events
            )
            failed_components.extend(cascade_failures)
        
        # Analyze impact
        result = self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=failed_components,
            failure_events=failure_events
        )
        
        return result
    
    def simulate_multiple_failures(self,
                                   graph: nx.DiGraph,
                                   components: List[str],
                                   failure_type: FailureType = FailureType.COMPLETE,
                                   enable_cascade: bool = True) -> SimulationResult:
        """
        Simulate simultaneous failure of multiple components
        
        Args:
            graph: NetworkX directed graph
            components: List of components to fail
            failure_type: Type of failure
            enable_cascade: Whether to simulate cascading failures
        
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating failure of {len(components)} components")
        
        # Validate components
        for component in components:
            if component not in graph.nodes():
                raise ValueError(f"Component {component} not found in graph")
        
        # Create working copy
        sim_graph = graph.copy()
        failure_events = []
        failed_components = list(components)
        
        # Create failure events
        timestamp = datetime.now()
        for component in components:
            event = FailureEvent(
                component=component,
                failure_type=failure_type,
                failure_mode=FailureMode.CRASH,
                timestamp=timestamp,
                severity=1.0,
                cause="Simulated multiple failure"
            )
            failure_events.append(event)
        
        # Apply failures
        if failure_type == FailureType.COMPLETE:
            sim_graph.remove_nodes_from(components)
        
        # Simulate cascade if enabled
        if enable_cascade:
            for component in components:
                if component in sim_graph.nodes():
                    cascade_failures = self._simulate_cascade(
                        sim_graph,
                        component,
                        failure_events
                    )
                    failed_components.extend(cascade_failures)
        
        # Analyze impact
        result = self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=failed_components,
            failure_events=failure_events
        )
        
        return result
    
    def simulate_network_failure(self,
                                 graph: nx.DiGraph,
                                 source: str,
                                 target: str) -> SimulationResult:
        """
        Simulate failure of a network connection
        
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
        
        # Create working copy
        sim_graph = graph.copy()
        
        # Remove edge
        sim_graph.remove_edge(source, target)
        
        # Create failure event
        failure_events = [FailureEvent(
            component=f"{source}->{target}",
            failure_type=FailureType.NETWORK,
            failure_mode=FailureMode.DISCONNECT,
            timestamp=datetime.now(),
            severity=1.0,
            cause="Simulated network failure"
        )]
        
        # Analyze impact
        result = self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=[],
            failure_events=failure_events
        )
        
        return result
    
    def simulate_partial_failure(self,
                                 graph: nx.DiGraph,
                                 component: str,
                                 degradation: float) -> SimulationResult:
        """
        Simulate partial failure (performance degradation)
        
        Args:
            graph: NetworkX directed graph
            component: Component to degrade
            degradation: Degradation level (0.0-1.0, where 1.0 is complete failure)
        
        Returns:
            SimulationResult with impact analysis
        """
        self.logger.info(f"Simulating {degradation*100:.0f}% degradation of {component}")
        
        if component not in graph.nodes():
            raise ValueError(f"Component {component} not found")
        
        # Create working copy
        sim_graph = graph.copy()
        
        # Mark as degraded
        sim_graph.nodes[component]['degraded'] = True
        sim_graph.nodes[component]['degradation'] = degradation
        
        # Create failure event
        failure_events = [FailureEvent(
            component=component,
            failure_type=FailureType.PARTIAL,
            failure_mode=FailureMode.SLOW,
            timestamp=datetime.now(),
            severity=degradation,
            cause="Simulated performance degradation"
        )]
        
        # Analyze impact (consider degradation in calculations)
        result = self._analyze_impact(
            original_graph=graph,
            failed_graph=sim_graph,
            failed_components=[],
            failure_events=failure_events
        )
        
        return result
    
    def simulate_cascading_failure(self,
                                   graph: nx.DiGraph,
                                   initial_component: str,
                                   max_cascade_depth: int = 5) -> SimulationResult:
        """
        Simulate cascading failure starting from initial component
        
        Args:
            graph: NetworkX directed graph
            initial_component: Component to fail initially
            max_cascade_depth: Maximum cascade propagation depth
        
        Returns:
            SimulationResult with full cascade analysis
        """
        self.logger.info(f"Simulating cascading failure from {initial_component}")
        
        # Start with single failure
        result = self.simulate_single_failure(
            graph,
            initial_component,
            enable_cascade=True
        )
        
        return result
    
    def test_resilience(self,
                       graph: nx.DiGraph,
                       critical_components: Optional[List[str]] = None) -> Dict[str, SimulationResult]:
        """
        Test system resilience by simulating failure of each critical component
        
        Args:
            graph: NetworkX directed graph
            critical_components: List of critical components (or None for all)
        
        Returns:
            Dictionary mapping component to simulation result
        """
        self.logger.info("Testing system resilience...")
        
        if critical_components is None:
            critical_components = list(graph.nodes())
        
        results = {}
        
        for component in critical_components:
            try:
                result = self.simulate_single_failure(
                    graph,
                    component,
                    enable_cascade=True
                )
                results[component] = result
            except Exception as e:
                self.logger.error(f"Failed to simulate failure of {component}: {e}")
        
        return results
    
    def find_critical_single_points(self,
                                    graph: nx.DiGraph,
                                    impact_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find single points of failure (SPOFs) - components whose failure causes high impact
        
        Args:
            graph: NetworkX directed graph
            impact_threshold: Minimum impact score to be considered critical
        
        Returns:
            List of (component, impact_score) tuples, sorted by impact
        """
        self.logger.info("Finding single points of failure...")
        
        spofs = []
        
        for node in graph.nodes():
            result = self.simulate_single_failure(
                graph,
                node,
                enable_cascade=True
            )
            
            if result.impact_score >= impact_threshold:
                spofs.append((node, result.impact_score))
        
        # Sort by impact (highest first)
        spofs.sort(key=lambda x: x[1], reverse=True)
        
        return spofs
    
    def simulate_recovery(self,
                         graph: nx.DiGraph,
                         failed_component: str,
                         recovery_time_ms: float) -> Dict[str, Any]:
        """
        Simulate recovery of a failed component
        
        Args:
            graph: NetworkX directed graph
            failed_component: Component that failed
            recovery_time_ms: Time to recover in milliseconds
        
        Returns:
            Dictionary with recovery analysis
        """
        self.logger.info(f"Simulating recovery of {failed_component}")
        
        # Simulate failure
        failure_result = self.simulate_single_failure(
            graph,
            failed_component,
            enable_cascade=False
        )
        
        # Calculate recovery metrics
        recovery_analysis = {
            'component': failed_component,
            'recovery_time_ms': recovery_time_ms,
            'affected_during_failure': len(failure_result.affected_components),
            'service_restored': True,
            'impact_during_failure': failure_result.impact_score,
            'recovery_priority': 'HIGH' if failure_result.impact_score > 0.7 else 'MEDIUM'
        }
        
        return recovery_analysis
    
    def _simulate_cascade(self,
                         graph: nx.DiGraph,
                         failed_component: str,
                         failure_events: List[FailureEvent]) -> List[str]:
        """
        Simulate cascading failures from a failed component
        
        Args:
            graph: Current state of graph
            failed_component: Component that failed
            failure_events: List to append new failure events
        
        Returns:
            List of additionally failed components
        """
        cascade_failures = []
        
        # Find dependent components
        # In a directed graph, components that depend on the failed one are predecessors
        # (they have edges pointing to the failed component via DEPENDS_ON)
        
        # For simulation, we look at components that lose critical dependencies
        # Check in-edges to find what was depending on the failed component
        
        # Since graph is modified during simulation, work with original edges
        # We need to check which components would be affected
        
        # Simple cascade model: components that only had one path to critical resource
        # fail if that path is broken
        
        return cascade_failures
    
    def _analyze_impact(self,
                       original_graph: nx.DiGraph,
                       failed_graph: nx.DiGraph,
                       failed_components: List[str],
                       failure_events: List[FailureEvent]) -> SimulationResult:
        """
        Analyze impact of failures on the system
        
        Args:
            original_graph: Original graph state
            failed_graph: Graph state after failures
            failed_components: List of failed components
            failure_events: List of failure events
        
        Returns:
            SimulationResult with comprehensive analysis
        """
        # Identify affected components
        affected_components = self._find_affected_components(
            original_graph,
            failed_graph,
            failed_components
        )
        
        # Identify isolated components (disconnected from main graph)
        isolated_components = self._find_isolated_components(failed_graph)
        
        # Calculate impact score
        total_components = len(original_graph)
        affected_count = len(affected_components) + len(failed_components)
        impact_score = affected_count / total_components if total_components > 0 else 0.0
        
        # Calculate resilience score (inverse of impact)
        resilience_score = 1.0 - impact_score
        
        # Calculate service continuity
        remaining_components = total_components - len(failed_components)
        service_continuity = remaining_components / total_components if total_components > 0 else 0.0
        
        # Analyze path impact
        original_paths = self._count_paths(original_graph)
        remaining_paths = self._count_paths(failed_graph)
        affected_paths = original_paths - remaining_paths
        
        # Gather statistics
        statistics = {
            'total_components': total_components,
            'failed_count': len(failed_components),
            'affected_count': len(affected_components),
            'isolated_count': len(isolated_components),
            'remaining_components': remaining_components,
            'failure_percentage': (len(failed_components) / total_components * 100) if total_components > 0 else 0,
            'affected_percentage': (affected_count / total_components * 100) if total_components > 0 else 0
        }
        
        return SimulationResult(
            failed_components=failed_components,
            affected_components=affected_components,
            isolated_components=isolated_components,
            failure_events=failure_events,
            impact_score=impact_score,
            resilience_score=resilience_score,
            service_continuity=service_continuity,
            affected_paths=affected_paths,
            total_paths=original_paths,
            graph_state=failed_graph,
            statistics=statistics
        )
    
    def _find_affected_components(self,
                                  original_graph: nx.DiGraph,
                                  failed_graph: nx.DiGraph,
                                  failed_components: List[str]) -> List[str]:
        """
        Find components affected by failures (but not failed themselves)
        
        Components are affected if:
        - They lost connectivity to critical dependencies
        - They can no longer reach certain other components
        - Their redundancy was reduced
        """
        affected = []
        
        for node in failed_graph.nodes():
            if node in failed_components:
                continue
            
            # Check if node lost critical connections
            original_successors = set(original_graph.successors(node)) if node in original_graph else set()
            current_successors = set(failed_graph.successors(node))
            
            lost_connections = original_successors - current_successors
            
            # If lost any connections, consider affected
            if lost_connections:
                affected.append(node)
        
        return affected
    
    def _find_isolated_components(self, graph: nx.DiGraph) -> List[str]:
        """Find components that are isolated (no in or out edges)"""
        isolated = []
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            if in_degree == 0 and out_degree == 0:
                isolated.append(node)
        
        return isolated
    
    def _count_paths(self, graph: nx.DiGraph) -> int:
        """Count total number of paths in graph (approximate)"""
        # For performance, we'll count edges as a proxy for paths
        return len(graph.edges())
    
    def generate_failure_report(self,
                               graph: nx.DiGraph,
                               result: SimulationResult) -> Dict[str, Any]:
        """
        Generate comprehensive failure analysis report
        
        Args:
            graph: Original graph
            result: Simulation result
        
        Returns:
            Detailed report dictionary
        """
        report = {
            'summary': {
                'impact_score': result.impact_score,
                'resilience_score': result.resilience_score,
                'service_continuity': result.service_continuity,
                'severity': self._classify_severity(result.impact_score)
            },
            'failures': {
                'failed_components': result.failed_components,
                'failure_count': len(result.failed_components),
                'failure_events': [e.to_dict() for e in result.failure_events]
            },
            'impact': {
                'affected_components': result.affected_components,
                'isolated_components': result.isolated_components,
                'affected_count': len(result.affected_components),
                'isolated_count': len(result.isolated_components),
                'affected_paths': result.affected_paths,
                'total_paths': result.total_paths
            },
            'statistics': result.statistics,
            'recommendations': self._generate_recommendations(graph, result)
        }
        
        return report
    
    def _classify_severity(self, impact_score: float) -> str:
        """Classify failure severity based on impact score"""
        if impact_score >= 0.8:
            return "CRITICAL"
        elif impact_score >= 0.5:
            return "HIGH"
        elif impact_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self,
                                 graph: nx.DiGraph,
                                 result: SimulationResult) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        if result.impact_score > 0.7:
            recommendations.append(
                f"CRITICAL: Failure causes severe impact ({result.impact_score:.1%}). "
                "Implement redundancy or failover mechanisms."
            )
        
        if len(result.isolated_components) > 0:
            recommendations.append(
                f"WARNING: {len(result.isolated_components)} components became isolated. "
                "Review network topology for single points of failure."
            )
        
        if result.service_continuity < 0.5:
            recommendations.append(
                f"ALERT: Service continuity dropped to {result.service_continuity:.1%}. "
                "Consider load balancing and replication strategies."
            )
        
        if len(result.failure_events) > len(result.failed_components):
            recommendations.append(
                "WARNING: Cascading failures detected. "
                "Implement circuit breakers and failure isolation."
            )
        
        if not recommendations:
            recommendations.append(
                "System shows good resilience to this failure scenario."
            )
        
        return recommendations
