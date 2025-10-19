"""
Impact Calculator

Provides quantitative impact assessment for system changes and failures.
Calculates business, technical, and operational impacts with configurable metrics.

Supports:
- Business impact (revenue, users, transactions)
- Technical impact (performance, availability, latency)
- Operational impact (SLA violations, incident response)
- Cost impact (downtime cost, recovery cost)
- Cascading impact analysis
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging


class ImpactDimension(Enum):
    """Dimensions of impact assessment"""
    BUSINESS = "business"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    REPUTATION = "reputation"


@dataclass
class ImpactMetrics:
    """Container for impact metrics"""
    business_impact: float          # 0-1 scale
    technical_impact: float         # 0-1 scale
    operational_impact: float       # 0-1 scale
    financial_impact_usd: float     # Dollar amount
    affected_users: int             # Number of users
    affected_transactions: int      # Number of transactions
    sla_violations: int             # Number of SLA breaches
    availability_loss: float        # Percentage
    performance_degradation: float  # Percentage
    recovery_time_estimate_hours: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'business_impact': round(self.business_impact, 3),
            'technical_impact': round(self.technical_impact, 3),
            'operational_impact': round(self.operational_impact, 3),
            'financial_impact_usd': round(self.financial_impact_usd, 2),
            'affected_users': self.affected_users,
            'affected_transactions': self.affected_transactions,
            'sla_violations': self.sla_violations,
            'availability_loss': round(self.availability_loss, 2),
            'performance_degradation': round(self.performance_degradation, 2),
            'recovery_time_estimate_hours': round(self.recovery_time_estimate_hours, 2)
        }


@dataclass
class ComponentImpact:
    """Impact assessment for a single component"""
    component: str
    component_type: str
    direct_impact: float            # Direct impact of this component
    indirect_impact: float          # Impact through dependencies
    total_impact: float             # Combined impact
    metrics: ImpactMetrics
    affected_downstream: List[str]  # Components affected downstream
    affected_upstream: List[str]    # Components affected upstream
    criticality_level: str          # LOW, MEDIUM, HIGH, CRITICAL
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'direct_impact': round(self.direct_impact, 3),
            'indirect_impact': round(self.indirect_impact, 3),
            'total_impact': round(self.total_impact, 3),
            'metrics': self.metrics.to_dict(),
            'affected_downstream': self.affected_downstream,
            'affected_upstream': self.affected_upstream,
            'criticality_level': self.criticality_level
        }


class ImpactCalculator:
    """
    Calculates quantitative impact of component failures and changes
    
    Provides multiple impact dimensions:
    - Business impact (users, revenue, transactions)
    - Technical impact (performance, availability)
    - Operational impact (SLAs, incident response)
    - Financial impact (downtime costs)
    
    Uses configurable cost models and impact factors.
    """
    
    def __init__(self,
                 hourly_downtime_cost_usd: float = 10000.0,
                 user_base_size: int = 100000,
                 avg_transaction_value_usd: float = 50.0,
                 sla_availability_target: float = 0.999):
        """
        Initialize impact calculator
        
        Args:
            hourly_downtime_cost_usd: Cost per hour of downtime
            user_base_size: Total number of users
            avg_transaction_value_usd: Average transaction value
            sla_availability_target: SLA availability target (e.g., 0.999 = 99.9%)
        """
        self.logger = logging.getLogger(__name__)
        self.hourly_downtime_cost = hourly_downtime_cost_usd
        self.user_base_size = user_base_size
        self.avg_transaction_value = avg_transaction_value_usd
        self.sla_target = sla_availability_target
    
    def calculate_component_impact(self,
                                   graph: nx.DiGraph,
                                   component: str,
                                   failure_severity: float = 1.0) -> ComponentImpact:
        """
        Calculate impact of a component failure
        
        Args:
            graph: NetworkX directed graph
            component: Component to analyze
            failure_severity: Severity of failure (0.0-1.0)
        
        Returns:
            ComponentImpact with detailed metrics
        """
        self.logger.info(f"Calculating impact for {component}")
        
        if component not in graph.nodes():
            raise ValueError(f"Component {component} not found in graph")
        
        node_data = graph.nodes[component]
        component_type = node_data.get('type', 'Unknown')
        
        # Calculate direct impact
        direct_impact = self._calculate_direct_impact(graph, component, node_data)
        
        # Calculate indirect impact through dependencies
        indirect_impact = self._calculate_indirect_impact(graph, component)
        
        # Total impact
        total_impact = direct_impact * failure_severity + indirect_impact * failure_severity
        
        # Find affected components
        affected_downstream = self._find_downstream_affected(graph, component)
        affected_upstream = self._find_upstream_affected(graph, component)
        
        # Calculate detailed metrics
        metrics = self._calculate_impact_metrics(
            graph,
            component,
            node_data,
            total_impact,
            len(affected_downstream) + len(affected_upstream)
        )
        
        # Classify criticality
        criticality = self._classify_criticality(total_impact)
        
        return ComponentImpact(
            component=component,
            component_type=component_type,
            direct_impact=direct_impact,
            indirect_impact=indirect_impact,
            total_impact=total_impact,
            metrics=metrics,
            affected_downstream=affected_downstream,
            affected_upstream=affected_upstream,
            criticality_level=criticality
        )
    
    def calculate_multi_component_impact(self,
                                        graph: nx.DiGraph,
                                        components: List[str]) -> Dict[str, ComponentImpact]:
        """
        Calculate combined impact of multiple component failures
        
        Args:
            graph: NetworkX directed graph
            components: List of components to analyze
        
        Returns:
            Dictionary mapping component to impact
        """
        self.logger.info(f"Calculating impact for {len(components)} components")
        
        impacts = {}
        
        for component in components:
            try:
                impact = self.calculate_component_impact(graph, component)
                impacts[component] = impact
            except Exception as e:
                self.logger.error(f"Failed to calculate impact for {component}: {e}")
        
        return impacts
    
    def calculate_business_impact(self,
                                 graph: nx.DiGraph,
                                 component: str,
                                 downtime_hours: float = 1.0) -> Dict[str, Any]:
        """
        Calculate business impact in business terms
        
        Args:
            graph: NetworkX directed graph
            component: Component to analyze
            downtime_hours: Expected downtime in hours
        
        Returns:
            Dictionary with business metrics
        """
        impact = self.calculate_component_impact(graph, component)
        
        # Calculate business metrics
        affected_user_percentage = impact.total_impact
        affected_users = int(self.user_base_size * affected_user_percentage)
        
        # Estimate transaction loss
        avg_transactions_per_hour = 1000  # Could be configured
        lost_transactions = int(
            avg_transactions_per_hour * downtime_hours * affected_user_percentage
        )
        revenue_loss = lost_transactions * self.avg_transaction_value
        
        # Calculate costs
        downtime_cost = self.hourly_downtime_cost * downtime_hours
        total_financial_impact = downtime_cost + revenue_loss
        
        return {
            'component': component,
            'business_metrics': {
                'affected_users': affected_users,
                'affected_user_percentage': round(affected_user_percentage * 100, 2),
                'lost_transactions': lost_transactions,
                'revenue_loss_usd': round(revenue_loss, 2),
                'downtime_cost_usd': round(downtime_cost, 2),
                'total_financial_impact_usd': round(total_financial_impact, 2)
            },
            'downtime_hours': downtime_hours,
            'impact_level': impact.criticality_level
        }
    
    def calculate_sla_impact(self,
                           graph: nx.DiGraph,
                           component: str,
                           downtime_minutes: float) -> Dict[str, Any]:
        """
        Calculate SLA impact and violations
        
        Args:
            graph: NetworkX directed graph
            component: Component to analyze
            downtime_minutes: Downtime in minutes
        
        Returns:
            Dictionary with SLA metrics
        """
        impact = self.calculate_component_impact(graph, component)
        
        # Calculate availability impact
        minutes_per_month = 30 * 24 * 60
        availability_loss = downtime_minutes / minutes_per_month
        
        # Check SLA violation
        current_availability = 1.0 - availability_loss
        sla_violated = current_availability < self.sla_target
        
        # Calculate SLA breach severity
        if sla_violated:
            breach_severity = (self.sla_target - current_availability) / self.sla_target
        else:
            breach_severity = 0.0
        
        return {
            'component': component,
            'sla_metrics': {
                'downtime_minutes': downtime_minutes,
                'availability_loss_percentage': round(availability_loss * 100, 4),
                'current_availability': round(current_availability * 100, 4),
                'sla_target': round(self.sla_target * 100, 4),
                'sla_violated': sla_violated,
                'breach_severity': round(breach_severity, 3),
                'nines': self._calculate_nines(current_availability)
            },
            'impact_level': impact.criticality_level
        }
    
    def calculate_cascading_impact(self,
                                  graph: nx.DiGraph,
                                  initial_component: str,
                                  propagation_factor: float = 0.5) -> Dict[str, Any]:
        """
        Calculate cascading impact through dependencies
        
        Args:
            graph: NetworkX directed graph
            initial_component: Starting point of failure
            propagation_factor: How much impact propagates (0.0-1.0)
        
        Returns:
            Dictionary with cascading analysis
        """
        self.logger.info(f"Calculating cascading impact from {initial_component}")
        
        # Get initial impact
        initial_impact = self.calculate_component_impact(graph, initial_component)
        
        # Track propagation
        impact_wave = {initial_component: initial_impact.total_impact}
        visited = {initial_component}
        
        # BFS to propagate impact
        queue = [(initial_component, initial_impact.total_impact)]
        
        while queue:
            current, current_impact = queue.pop(0)
            
            # Find dependents (predecessors in dependency graph)
            for dependent in graph.predecessors(current):
                if dependent in visited:
                    continue
                
                visited.add(dependent)
                
                # Propagate impact with decay
                propagated_impact = current_impact * propagation_factor
                impact_wave[dependent] = propagated_impact
                
                if propagated_impact > 0.1:  # Continue if significant
                    queue.append((dependent, propagated_impact))
        
        # Calculate total cascading impact
        total_cascade_impact = sum(impact_wave.values())
        
        return {
            'initial_component': initial_component,
            'initial_impact': initial_impact.total_impact,
            'affected_components': len(impact_wave),
            'impact_wave': {k: round(v, 3) for k, v in impact_wave.items()},
            'total_cascade_impact': round(total_cascade_impact, 3),
            'cascade_factor': round(total_cascade_impact / initial_impact.total_impact, 2),
            'propagation_depth': self._calculate_propagation_depth(impact_wave)
        }
    
    def rank_components_by_impact(self,
                                  graph: nx.DiGraph,
                                  top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Rank all components by their potential impact
        
        Args:
            graph: NetworkX directed graph
            top_n: Number of top components to return
        
        Returns:
            List of (component, impact_score) tuples
        """
        self.logger.info("Ranking components by impact...")
        
        impacts = []
        
        for node in graph.nodes():
            try:
                impact = self.calculate_component_impact(graph, node)
                impacts.append((node, impact.total_impact))
            except Exception as e:
                self.logger.warning(f"Could not calculate impact for {node}: {e}")
        
        # Sort by impact (highest first)
        impacts.sort(key=lambda x: x[1], reverse=True)
        
        return impacts[:top_n]
    
    def estimate_recovery_impact(self,
                                graph: nx.DiGraph,
                                component: str,
                                recovery_time_hours: float) -> Dict[str, Any]:
        """
        Estimate impact during recovery period
        
        Args:
            graph: NetworkX directed graph
            component: Component being recovered
            recovery_time_hours: Time to recover
        
        Returns:
            Dictionary with recovery impact analysis
        """
        # Calculate normal impact
        impact = self.calculate_component_impact(graph, component)
        
        # Calculate impact during recovery
        business_impact = self.calculate_business_impact(
            graph,
            component,
            downtime_hours=recovery_time_hours
        )
        
        # Calculate progressive recovery (assumes linear recovery)
        time_windows = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100% of recovery time
        recovery_timeline = []
        
        for window in time_windows:
            elapsed_time = recovery_time_hours * window
            remaining_impact = impact.total_impact * (1.0 - window)
            
            recovery_timeline.append({
                'elapsed_hours': round(elapsed_time, 2),
                'recovery_percentage': round(window * 100, 1),
                'remaining_impact': round(remaining_impact, 3),
                'service_level': round((1.0 - remaining_impact) * 100, 1)
            })
        
        return {
            'component': component,
            'recovery_time_hours': recovery_time_hours,
            'total_impact': impact.total_impact,
            'financial_impact': business_impact['business_metrics']['total_financial_impact_usd'],
            'recovery_timeline': recovery_timeline,
            'full_recovery_eta': recovery_time_hours
        }
    
    def _calculate_direct_impact(self,
                                 graph: nx.DiGraph,
                                 component: str,
                                 node_data: Dict) -> float:
        """Calculate direct impact of component based on its properties"""
        
        # Base impact from node degree
        degree = graph.degree(component)
        total_nodes = len(graph)
        degree_impact = degree / (2 * total_nodes) if total_nodes > 0 else 0
        
        # Adjust by component type
        component_type = node_data.get('type', 'Unknown')
        type_multiplier = {
            'Broker': 1.5,
            'Application': 1.0,
            'Topic': 1.2,
            'Node': 1.3
        }.get(component_type, 1.0)
        
        # Adjust by criticality score if available
        criticality = node_data.get('criticality_score', 0.5)
        
        # Combine factors
        direct_impact = degree_impact * type_multiplier * (0.5 + criticality)
        
        return min(1.0, direct_impact)
    
    def _calculate_indirect_impact(self,
                                   graph: nx.DiGraph,
                                   component: str) -> float:
        """Calculate indirect impact through dependencies"""
        
        # Count dependent components
        dependents = list(graph.predecessors(component))
        
        if not dependents:
            return 0.0
        
        # Impact increases with number of dependents
        total_nodes = len(graph)
        indirect_impact = len(dependents) / total_nodes if total_nodes > 0 else 0
        
        return min(1.0, indirect_impact * 0.5)  # Scale down indirect impact
    
    def _find_downstream_affected(self,
                                  graph: nx.DiGraph,
                                  component: str) -> List[str]:
        """Find components affected downstream (that depend on this component)"""
        
        # BFS to find all reachable nodes through predecessor edges
        affected = []
        visited = set()
        queue = [component]
        
        while queue:
            current = queue.pop(0)
            
            for predecessor in graph.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    affected.append(predecessor)
                    queue.append(predecessor)
        
        return affected
    
    def _find_upstream_affected(self,
                               graph: nx.DiGraph,
                               component: str) -> List[str]:
        """Find components affected upstream (that this component depends on)"""
        
        # BFS to find all reachable nodes through successor edges
        affected = []
        visited = set()
        queue = [component]
        
        while queue:
            current = queue.pop(0)
            
            for successor in graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    affected.append(successor)
                    queue.append(successor)
        
        return affected
    
    def _calculate_impact_metrics(self,
                                  graph: nx.DiGraph,
                                  component: str,
                                  node_data: Dict,
                                  total_impact: float,
                                  affected_count: int) -> ImpactMetrics:
        """Calculate detailed impact metrics"""
        
        # Business impact
        business_impact = total_impact
        
        # Technical impact (based on connectivity)
        degree = graph.degree(component)
        avg_degree = sum(dict(graph.degree()).values()) / len(graph) if len(graph) > 0 else 1
        technical_impact = min(1.0, degree / (avg_degree * 2))
        
        # Operational impact
        operational_impact = total_impact * 0.8
        
        # Financial impact
        estimated_downtime_hours = 1.0  # Default assumption
        financial_impact = self.hourly_downtime_cost * estimated_downtime_hours * total_impact
        
        # Affected users
        affected_users = int(self.user_base_size * total_impact)
        
        # Affected transactions (estimate)
        transactions_per_hour = 1000
        affected_transactions = int(transactions_per_hour * total_impact)
        
        # SLA violations (if impact is high)
        sla_violations = 1 if total_impact > 0.5 else 0
        
        # Availability loss
        availability_loss = total_impact * 100  # Percentage
        
        # Performance degradation
        performance_degradation = total_impact * 80  # Percentage
        
        # Recovery time estimate
        recovery_time = self._estimate_recovery_time(component, node_data, total_impact)
        
        return ImpactMetrics(
            business_impact=business_impact,
            technical_impact=technical_impact,
            operational_impact=operational_impact,
            financial_impact_usd=financial_impact,
            affected_users=affected_users,
            affected_transactions=affected_transactions,
            sla_violations=sla_violations,
            availability_loss=availability_loss,
            performance_degradation=performance_degradation,
            recovery_time_estimate_hours=recovery_time
        )
    
    def _estimate_recovery_time(self,
                               component: str,
                               node_data: Dict,
                               impact: float) -> float:
        """Estimate recovery time in hours"""
        
        # Base recovery time by component type
        component_type = node_data.get('type', 'Unknown')
        base_time = {
            'Application': 0.5,
            'Broker': 1.0,
            'Topic': 0.25,
            'Node': 2.0
        }.get(component_type, 1.0)
        
        # Scale by impact
        recovery_time = base_time * (1 + impact)
        
        return recovery_time
    
    def _classify_criticality(self, impact: float) -> str:
        """Classify criticality level based on impact score"""
        if impact >= 0.8:
            return "CRITICAL"
        elif impact >= 0.5:
            return "HIGH"
        elif impact >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_nines(self, availability: float) -> str:
        """Calculate 'nines' of availability (e.g., 99.9% = 3 nines)"""
        if availability >= 0.99999:
            return "5 nines (99.999%)"
        elif availability >= 0.9999:
            return "4 nines (99.99%)"
        elif availability >= 0.999:
            return "3 nines (99.9%)"
        elif availability >= 0.99:
            return "2 nines (99%)"
        else:
            return f"{availability*100:.2f}%"
    
    def _calculate_propagation_depth(self, impact_wave: Dict[str, float]) -> int:
        """Calculate depth of impact propagation"""
        return len(impact_wave)
