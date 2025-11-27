#!/usr/bin/env python3
"""
Graph Analysis Validation Framework

Comprehensive validation system for pub-sub system graph analysis results.
Implements a multi-faceted validation approach that combines:

1. Structural Validation: Ensures graph integrity and correctness
2. Criticality Validation: Validates criticality scores against failure simulations
3. Statistical Validation: Correlation analysis between predictions and outcomes
4. Cross-Validation: K-fold validation for robust performance metrics
5. Baseline Comparison: Compares against random and simple heuristic baselines

Target Metrics (from research methodology):
- Spearman correlation > 0.7 with historical/simulated failures
- F1-score > 0.9 for critical component identification
- Precision >= 0.9, Recall >= 0.85

Author: Software-as-a-Graph Framework
"""

import argparse
import json
import logging
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be limited.")


# ============================================================================
# Configuration and Constants
# ============================================================================

# Target validation thresholds from research methodology
TARGET_SPEARMAN_CORRELATION = 0.7
TARGET_F1_SCORE = 0.9
TARGET_PRECISION = 0.9
TARGET_RECALL = 0.85

# Default criticality scoring weights
DEFAULT_ALPHA = 0.4  # Betweenness centrality weight
DEFAULT_BETA = 0.3   # Articulation point weight
DEFAULT_GAMMA = 0.3  # Impact score weight

# Criticality level thresholds
CRITICALITY_THRESHOLDS = {
    'CRITICAL': 0.8,
    'HIGH': 0.6,
    'MEDIUM': 0.4,
    'LOW': 0.2,
    'MINIMAL': 0.0
}


class CriticalityLevel(Enum):
    """Criticality classification levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class ValidationMetric(Enum):
    """Validation metric types"""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KENDALL_TAU = "kendall_tau"
    MEAN_ABSOLUTE_ERROR = "mae"
    ROOT_MEAN_SQUARE_ERROR = "rmse"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CriticalityScore:
    """Composite criticality score for a component"""
    component: str
    component_type: str
    betweenness_centrality_norm: float
    is_articulation_point: bool
    impact_score: float
    composite_score: float
    criticality_level: CriticalityLevel
    components_affected: int = 0
    reachability_loss_percentage: float = 0.0


@dataclass
class FailureSimulationResult:
    """Result of simulating a component failure"""
    failed_component: str
    components_affected: List[str]
    reachability_loss: float
    impact_score: float
    cascading_failures: List[str] = field(default_factory=list)
    recovery_complexity: float = 0.0


@dataclass
class ValidationResult:
    """Overall validation results"""
    timestamp: str
    graph_summary: Dict[str, Any]
    structural_validation: Dict[str, Any]
    criticality_validation: Dict[str, Any]
    simulation_validation: Dict[str, Any]
    statistical_validation: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    cross_validation: Dict[str, Any]
    recommendations: List[str]
    overall_quality: str
    target_metrics_met: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with optional file output"""
    logger = logging.getLogger('graph_validation')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


logger = logging.getLogger('graph_validation')


# ============================================================================
# Graph Loading and Building
# ============================================================================

def load_graph_from_json(filepath: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Load a graph from JSON file format
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Tuple of (NetworkX graph, raw data dict)
    """
    logger.info(f"Loading graph from {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = build_graph_from_data(data)
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, data


def build_graph_from_data(data: Dict[str, Any]) -> nx.DiGraph:
    """
    Build NetworkX graph from JSON data structure
    
    Expected format with: nodes, applications, topics, brokers, relationships
    """
    G = nx.DiGraph()
    
    # Add infrastructure nodes
    for node in data.get('nodes', []):
        G.add_node(node['id'], type='Node', **{k: v for k, v in node.items() if k != 'id'})
    
    # Add applications
    for app in data.get('applications', []):
        G.add_node(app['id'], type='Application', **{k: v for k, v in app.items() if k != 'id'})
    
    # Add topics
    for topic in data.get('topics', []):
        G.add_node(topic['id'], type='Topic', **{k: v for k, v in topic.items() if k != 'id'})
    
    # Add brokers
    for broker in data.get('brokers', []):
        G.add_node(broker['id'], type='Broker', **{k: v for k, v in broker.items() if k != 'id'})
    
    # Add relationships
    relationships = data.get('relationships', {})
    
    for rel in relationships.get('runs_on', []):
        G.add_edge(rel['from'], rel['to'], type='RUNS_ON')
    
    for rel in relationships.get('publishes_to', []):
        qos = rel.get('qos', {})
        G.add_edge(rel['from'], rel['to'], type='PUBLISHES_TO', **qos)
    
    for rel in relationships.get('subscribes_to', []):
        qos = rel.get('qos', {})
        G.add_edge(rel['from'], rel['to'], type='SUBSCRIBES_TO', **qos)
    
    for rel in relationships.get('routes', []):
        G.add_edge(rel['from'], rel['to'], type='ROUTES')
    
    return G


def generate_sample_graph(
    num_nodes: int = 10,
    num_apps: int = 20,
    num_topics: int = 15,
    num_brokers: int = 3,
    seed: int = 42
) -> nx.DiGraph:
    """
    Generate a sample pub-sub graph for testing
    
    Args:
        num_nodes: Number of infrastructure nodes
        num_apps: Number of applications
        num_topics: Number of topics
        num_brokers: Number of brokers
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    G = nx.DiGraph()
    
    # Create nodes
    nodes = [f"node_{i}" for i in range(num_nodes)]
    apps = [f"app_{i}" for i in range(num_apps)]
    topics = [f"topic_{i}" for i in range(num_topics)]
    brokers = [f"broker_{i}" for i in range(num_brokers)]
    
    for node in nodes:
        G.add_node(node, type='Node', resources={'cpu': random.randint(4, 32)})
    
    for app in apps:
        G.add_node(app, type='Application', criticality=random.choice(['high', 'medium', 'low']))
    
    for topic in topics:
        qos = {
            'reliability': random.choice(['RELIABLE', 'BEST_EFFORT']),
            'durability': random.choice(['TRANSIENT_LOCAL', 'VOLATILE']),
            'history_depth': random.randint(1, 100)
        }
        G.add_node(topic, type='Topic', **qos)
    
    for broker in brokers:
        G.add_node(broker, type='Broker', capacity=random.randint(100, 1000))
    
    # Create RUNS_ON relationships (apps -> nodes)
    for app in apps:
        node = random.choice(nodes)
        G.add_edge(app, node, type='RUNS_ON')
    
    # Create PUBLISHES_TO relationships (apps -> topics)
    for app in apps:
        num_pubs = random.randint(1, 3)
        pub_topics = random.sample(topics, min(num_pubs, len(topics)))
        for topic in pub_topics:
            G.add_edge(app, topic, type='PUBLISHES_TO', 
                      message_rate=random.randint(10, 1000))
    
    # Create SUBSCRIBES_TO relationships (apps -> topics)
    for app in apps:
        num_subs = random.randint(1, 4)
        sub_topics = random.sample(topics, min(num_subs, len(topics)))
        for topic in sub_topics:
            G.add_edge(app, topic, type='SUBSCRIBES_TO')
    
    # Create ROUTES relationships (brokers -> topics)
    for topic in topics:
        broker = random.choice(brokers)
        G.add_edge(broker, topic, type='ROUTES')
    
    logger.info(f"Generated sample graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ============================================================================
# Structural Validation
# ============================================================================

class StructuralValidator:
    """Validates graph structural integrity"""
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all structural validations"""
        logger.info("Running structural validation...")
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Run validation checks
        checks = [
            ('node_types', self._validate_node_types()),
            ('edge_types', self._validate_edge_types()),
            ('connectivity', self._validate_connectivity()),
            ('reference_integrity', self._validate_reference_integrity()),
            ('orphan_detection', self._validate_no_orphans()),
            ('cycle_detection', self._validate_cycles()),
        ]
        
        for check_name, check_result in checks:
            results['checks'][check_name] = check_result
            if not check_result.get('passed', True):
                results['errors'].extend(check_result.get('errors', []))
            results['warnings'].extend(check_result.get('warnings', []))
        
        results['is_valid'] = len(results['errors']) == 0
        results['total_errors'] = len(results['errors'])
        results['total_warnings'] = len(results['warnings'])
        
        logger.info(f"Structural validation: {results['total_errors']} errors, {results['total_warnings']} warnings")
        
        return results
    
    def _validate_node_types(self) -> Dict[str, Any]:
        """Validate that all nodes have valid types"""
        valid_types = {'Node', 'Application', 'Topic', 'Broker'}
        invalid_nodes = []
        type_counts = defaultdict(int)
        
        for node, data in self.G.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            type_counts[node_type] += 1
            if node_type not in valid_types:
                invalid_nodes.append((node, node_type))
        
        return {
            'passed': len(invalid_nodes) == 0,
            'type_distribution': dict(type_counts),
            'invalid_nodes': invalid_nodes,
            'errors': [f"Invalid node type '{t}' for node '{n}'" for n, t in invalid_nodes]
        }
    
    def _validate_edge_types(self) -> Dict[str, Any]:
        """Validate that all edges have valid types"""
        valid_types = {'RUNS_ON', 'PUBLISHES_TO', 'SUBSCRIBES_TO', 'ROUTES', 'DEPENDS_ON'}
        invalid_edges = []
        type_counts = defaultdict(int)
        
        for source, target, data in self.G.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            type_counts[edge_type] += 1
            if edge_type not in valid_types:
                invalid_edges.append((source, target, edge_type))
        
        return {
            'passed': len(invalid_edges) == 0,
            'type_distribution': dict(type_counts),
            'invalid_edges': invalid_edges,
            'errors': [f"Invalid edge type '{t}' for edge {s}->{e}" for s, e, t in invalid_edges]
        }
    
    def _validate_connectivity(self) -> Dict[str, Any]:
        """Check graph connectivity"""
        is_weakly_connected = nx.is_weakly_connected(self.G)
        num_components = nx.number_weakly_connected_components(self.G)
        
        warnings = []
        if not is_weakly_connected:
            warnings.append(f"Graph has {num_components} disconnected components")
        
        return {
            'passed': True,  # Not a failure, just informational
            'is_weakly_connected': is_weakly_connected,
            'num_components': num_components,
            'warnings': warnings
        }
    
    def _validate_reference_integrity(self) -> Dict[str, Any]:
        """Validate all edge references point to existing nodes"""
        errors = []
        
        for source, target in self.G.edges():
            if source not in self.G.nodes():
                errors.append(f"Edge references non-existent source: {source}")
            if target not in self.G.nodes():
                errors.append(f"Edge references non-existent target: {target}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_no_orphans(self) -> Dict[str, Any]:
        """Check for orphaned components"""
        warnings = []
        orphans = {
            'applications': [],
            'topics': [],
            'brokers': []
        }
        
        for node, data in self.G.nodes(data=True):
            node_type = data.get('type')
            degree = self.G.degree(node)
            
            if degree == 0:
                if node_type == 'Application':
                    orphans['applications'].append(node)
                elif node_type == 'Topic':
                    orphans['topics'].append(node)
                elif node_type == 'Broker':
                    orphans['brokers'].append(node)
        
        for category, items in orphans.items():
            if items:
                warnings.append(f"{len(items)} orphaned {category}: {items[:5]}...")
        
        return {
            'passed': True,
            'orphans': orphans,
            'total_orphans': sum(len(v) for v in orphans.values()),
            'warnings': warnings
        }
    
    def _validate_cycles(self) -> Dict[str, Any]:
        """Detect and report cycles in the graph"""
        try:
            cycles = list(nx.simple_cycles(self.G))
            has_cycles = len(cycles) > 0
            
            return {
                'passed': True,  # Cycles aren't necessarily errors
                'has_cycles': has_cycles,
                'num_cycles': len(cycles),
                'sample_cycles': cycles[:5] if cycles else [],
                'warnings': [f"Graph contains {len(cycles)} cycles"] if has_cycles else []
            }
        except Exception as e:
            return {
                'passed': True,
                'has_cycles': False,
                'error': str(e)
            }


# ============================================================================
# Criticality Scoring
# ============================================================================

class CriticalityCalculator:
    """
    Calculates composite criticality scores using the formula:
    C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
    """
    
    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def calculate_all_scores(self, graph: nx.DiGraph) -> Dict[str, CriticalityScore]:
        """Calculate criticality scores for all nodes"""
        logger.info("Calculating criticality scores...")
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        max_bc = max(betweenness.values()) if betweenness else 1.0
        if max_bc == 0:
            max_bc = 1.0
        
        # Find articulation points (using undirected version)
        undirected = graph.to_undirected()
        try:
            articulation_points = set(nx.articulation_points(undirected))
        except:
            articulation_points = set()
        
        scores = {}
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Normalized betweenness centrality
            bc_norm = betweenness.get(node, 0.0) / max_bc
            
            # Articulation point indicator
            is_ap = node in articulation_points
            ap_indicator = 1.0 if is_ap else 0.0
            
            # Impact score (based on reachability loss)
            impact, affected, reachability_loss = self._calculate_impact(graph, node)
            
            # Composite score
            composite = (self.alpha * bc_norm) + (self.beta * ap_indicator) + (self.gamma * impact)
            
            # Classification
            level = self._classify_level(composite)
            
            scores[node] = CriticalityScore(
                component=node,
                component_type=node_type,
                betweenness_centrality_norm=bc_norm,
                is_articulation_point=is_ap,
                impact_score=impact,
                composite_score=composite,
                criticality_level=level,
                components_affected=affected,
                reachability_loss_percentage=reachability_loss
            )
        
        logger.info(f"Calculated criticality for {len(scores)} nodes")
        return scores
    
    def _calculate_impact(self, graph: nx.DiGraph, node: str) -> Tuple[float, int, float]:
        """
        Calculate impact score based on reachability loss
        
        Returns: (impact_score, components_affected, reachability_loss_percentage)
        """
        # Calculate original reachability
        original_reachable = set()
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target and nx.has_path(graph, source, target):
                    original_reachable.add((source, target))
        
        # Create graph without the node
        G_removed = graph.copy()
        G_removed.remove_node(node)
        
        # Calculate new reachability
        new_reachable = set()
        for source in G_removed.nodes():
            for target in G_removed.nodes():
                if source != target:
                    try:
                        if nx.has_path(G_removed, source, target):
                            new_reachable.add((source, target))
                    except:
                        pass
        
        # Calculate loss
        lost_reachability = original_reachable - new_reachable
        components_affected = len(set(pair[0] for pair in lost_reachability) | 
                                  set(pair[1] for pair in lost_reachability))
        
        if len(original_reachable) > 0:
            reachability_loss_pct = (len(lost_reachability) / len(original_reachable)) * 100
        else:
            reachability_loss_pct = 0.0
        
        # Normalize impact score to 0-1
        impact_score = min(1.0, components_affected / max(1, len(graph.nodes()) - 1))
        
        return impact_score, components_affected, reachability_loss_pct
    
    def _classify_level(self, score: float) -> CriticalityLevel:
        """Classify criticality based on composite score"""
        if score >= CRITICALITY_THRESHOLDS['CRITICAL']:
            return CriticalityLevel.CRITICAL
        elif score >= CRITICALITY_THRESHOLDS['HIGH']:
            return CriticalityLevel.HIGH
        elif score >= CRITICALITY_THRESHOLDS['MEDIUM']:
            return CriticalityLevel.MEDIUM
        elif score >= CRITICALITY_THRESHOLDS['LOW']:
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.MINIMAL


# ============================================================================
# Failure Simulation
# ============================================================================

class FailureSimulator:
    """Simulates component failures and measures impact"""
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.original_reachability = self._calculate_reachability(graph)
    
    def simulate_failure(self, component: str) -> FailureSimulationResult:
        """Simulate failure of a single component"""
        if component not in self.G.nodes():
            raise ValueError(f"Component {component} not found in graph")
        
        # Create graph without the component
        G_failed = self.G.copy()
        G_failed.remove_node(component)
        
        # Calculate impact
        new_reachability = self._calculate_reachability(G_failed)
        lost_pairs = self.original_reachability - new_reachability
        
        affected_components = list(set(
            pair[0] for pair in lost_pairs
        ) | set(
            pair[1] for pair in lost_pairs
        ))
        
        reachability_loss = len(lost_pairs) / max(1, len(self.original_reachability))
        impact_score = min(1.0, len(affected_components) / max(1, len(self.G.nodes()) - 1))
        
        return FailureSimulationResult(
            failed_component=component,
            components_affected=affected_components,
            reachability_loss=reachability_loss,
            impact_score=impact_score
        )
    
    def simulate_all_failures(self) -> Dict[str, FailureSimulationResult]:
        """Simulate failure of each component"""
        logger.info("Running failure simulations for all components...")
        
        results = {}
        total = len(self.G.nodes())
        
        for i, component in enumerate(self.G.nodes()):
            if (i + 1) % 10 == 0:
                logger.debug(f"Simulating failure {i+1}/{total}")
            results[component] = self.simulate_failure(component)
        
        logger.info(f"Completed {total} failure simulations")
        return results
    
    def _calculate_reachability(self, graph: nx.DiGraph) -> Set[Tuple[str, str]]:
        """Calculate all reachable pairs in the graph"""
        reachable = set()
        for source in graph.nodes():
            try:
                descendants = nx.descendants(graph, source)
                for target in descendants:
                    reachable.add((source, target))
            except:
                pass
        return reachable


# ============================================================================
# Validation Metrics Calculator
# ============================================================================

class ValidationMetricsCalculator:
    """Calculates validation metrics (precision, recall, F1, correlation)"""
    
    def __init__(self):
        pass
    
    def calculate_classification_metrics(
        self,
        predicted_critical: Set[str],
        actual_critical: Set[str],
        total_components: int
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 score for critical component identification
        
        Args:
            predicted_critical: Set of components predicted as critical
            actual_critical: Set of actually critical components
            total_components: Total number of components
        
        Returns:
            Dictionary with precision, recall, F1, accuracy
        """
        true_positives = len(predicted_critical & actual_critical)
        false_positives = len(predicted_critical - actual_critical)
        false_negatives = len(actual_critical - predicted_critical)
        true_negatives = total_components - len(predicted_critical | actual_critical)
        
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * (precision * recall) / max(0.001, precision + recall)
        accuracy = (true_positives + true_negatives) / max(1, total_components)
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4)
        }
    
    def calculate_rank_correlation(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate Spearman and Kendall correlation between predicted and actual rankings
        
        Args:
            predicted_scores: Component -> predicted criticality score
            actual_scores: Component -> actual impact score (from simulation)
        
        Returns:
            Dictionary with correlation coefficients and p-values
        """
        # Get common components
        common = set(predicted_scores.keys()) & set(actual_scores.keys())
        if len(common) < 3:
            return {
                'error': 'Insufficient data for correlation',
                'common_components': len(common)
            }
        
        predicted = [predicted_scores[c] for c in common]
        actual = [actual_scores[c] for c in common]
        
        result = {
            'n_samples': len(common)
        }
        
        if SCIPY_AVAILABLE:
            # Spearman correlation
            spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
            result['spearman_correlation'] = round(spearman_corr, 4)
            result['spearman_p_value'] = round(spearman_p, 6)
            
            # Kendall Tau correlation
            kendall_corr, kendall_p = stats.kendalltau(predicted, actual)
            result['kendall_tau'] = round(kendall_corr, 4)
            result['kendall_p_value'] = round(kendall_p, 6)
        else:
            # Simple rank-based correlation without scipy
            pred_ranks = self._rank_scores(predicted)
            actual_ranks = self._rank_scores(actual)
            
            n = len(pred_ranks)
            d_squared_sum = sum((pred_ranks[i] - actual_ranks[i]) ** 2 for i in range(n))
            spearman_corr = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
            
            result['spearman_correlation'] = round(spearman_corr, 4)
            result['spearman_p_value'] = None  # Cannot calculate without scipy
        
        return result
    
    def calculate_regression_metrics(
        self,
        predicted: List[float],
        actual: List[float]
    ) -> Dict[str, float]:
        """Calculate MAE and RMSE"""
        if len(predicted) != len(actual) or len(predicted) == 0:
            return {'error': 'Invalid input data'}
        
        errors = [abs(p - a) for p, a in zip(predicted, actual)]
        squared_errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]
        
        mae = sum(errors) / len(errors)
        rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
        
        return {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4)
        }
    
    def _rank_scores(self, scores: List[float]) -> List[int]:
        """Convert scores to ranks"""
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks


# ============================================================================
# Baseline Comparators
# ============================================================================

class BaselineComparator:
    """Compares analysis results against baseline methods"""
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
    
    def compare_all_baselines(
        self,
        predicted_critical: Set[str],
        actual_critical: Set[str],
        criticality_scores: Dict[str, CriticalityScore]
    ) -> Dict[str, Any]:
        """Compare against all baseline methods"""
        logger.info("Computing baseline comparisons...")
        
        metrics_calc = ValidationMetricsCalculator()
        total = len(self.G.nodes())
        
        results = {}
        
        # Random baseline
        random_critical = self._random_baseline(len(predicted_critical))
        results['random'] = metrics_calc.calculate_classification_metrics(
            random_critical, actual_critical, total
        )
        
        # Degree-based baseline
        degree_critical = self._degree_baseline(len(predicted_critical))
        results['degree_based'] = metrics_calc.calculate_classification_metrics(
            degree_critical, actual_critical, total
        )
        
        # PageRank baseline
        pagerank_critical = self._pagerank_baseline(len(predicted_critical))
        results['pagerank'] = metrics_calc.calculate_classification_metrics(
            pagerank_critical, actual_critical, total
        )
        
        # Our method
        results['composite_scoring'] = metrics_calc.calculate_classification_metrics(
            predicted_critical, actual_critical, total
        )
        
        # Calculate improvement over baselines
        our_f1 = results['composite_scoring']['f1_score']
        results['improvement_over_random'] = round(our_f1 - results['random']['f1_score'], 4)
        results['improvement_over_degree'] = round(our_f1 - results['degree_based']['f1_score'], 4)
        results['improvement_over_pagerank'] = round(our_f1 - results['pagerank']['f1_score'], 4)
        
        return results
    
    def _random_baseline(self, k: int) -> Set[str]:
        """Random selection baseline"""
        nodes = list(self.G.nodes())
        return set(random.sample(nodes, min(k, len(nodes))))
    
    def _degree_baseline(self, k: int) -> Set[str]:
        """Degree-based selection (simple heuristic)"""
        degrees = dict(self.G.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)
        return set(sorted_nodes[:k])
    
    def _pagerank_baseline(self, k: int) -> Set[str]:
        """PageRank-based selection"""
        try:
            pagerank = nx.pagerank(self.G)
            sorted_nodes = sorted(pagerank.keys(), key=lambda n: pagerank[n], reverse=True)
            return set(sorted_nodes[:k])
        except:
            # Fallback to degree if PageRank fails
            return self._degree_baseline(k)


# ============================================================================
# Cross-Validation
# ============================================================================

class CrossValidator:
    """K-fold cross-validation for robustness assessment"""
    
    def __init__(self, k_folds: int = 5):
        self.k_folds = k_folds
    
    def cross_validate(
        self,
        graph: nx.DiGraph,
        criticality_calculator: CriticalityCalculator
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on graph subsets
        
        Splits nodes into k folds and validates predictions on each fold
        """
        logger.info(f"Running {self.k_folds}-fold cross-validation...")
        
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        
        fold_size = len(nodes) // self.k_folds
        fold_results = []
        
        for fold in range(self.k_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.k_folds - 1 else len(nodes)
            
            test_nodes = set(nodes[start_idx:end_idx])
            train_nodes = set(nodes) - test_nodes
            
            # Create subgraph for training
            train_graph = graph.subgraph(train_nodes).copy()
            
            if train_graph.number_of_nodes() < 5:
                continue
            
            # Calculate scores on training set
            train_scores = criticality_calculator.calculate_all_scores(train_graph)
            
            # Validate on test set using failure simulation
            simulator = FailureSimulator(graph)
            
            predicted_scores = {}
            actual_scores = {}
            
            for node in test_nodes:
                if node in train_graph.nodes():
                    continue
                
                # Predict based on training patterns
                neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
                neighbor_scores = [
                    train_scores[n].composite_score 
                    for n in neighbors if n in train_scores
                ]
                predicted = sum(neighbor_scores) / max(1, len(neighbor_scores))
                predicted_scores[node] = predicted
                
                # Get actual impact from simulation
                sim_result = simulator.simulate_failure(node)
                actual_scores[node] = sim_result.impact_score
            
            if len(predicted_scores) < 3:
                continue
            
            # Calculate metrics for this fold
            metrics_calc = ValidationMetricsCalculator()
            correlation = metrics_calc.calculate_rank_correlation(
                predicted_scores, actual_scores
            )
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train_nodes),
                'test_size': len(test_nodes),
                'spearman_correlation': correlation.get('spearman_correlation', 0)
            })
        
        if not fold_results:
            return {'error': 'Insufficient data for cross-validation'}
        
        correlations = [r['spearman_correlation'] for r in fold_results if r.get('spearman_correlation')]
        
        return {
            'k_folds': self.k_folds,
            'fold_results': fold_results,
            'mean_correlation': round(statistics.mean(correlations), 4) if correlations else 0,
            'std_correlation': round(statistics.stdev(correlations), 4) if len(correlations) > 1 else 0,
            'min_correlation': round(min(correlations), 4) if correlations else 0,
            'max_correlation': round(max(correlations), 4) if correlations else 0
        }


# ============================================================================
# Main Validation Engine
# ============================================================================

class GraphValidationEngine:
    """
    Main validation engine that orchestrates all validation components
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        critical_threshold: float = 0.6
    ):
        self.G = graph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.critical_threshold = critical_threshold
        
        # Initialize components
        self.structural_validator = StructuralValidator(graph)
        self.criticality_calculator = CriticalityCalculator(alpha, beta, gamma)
        self.failure_simulator = FailureSimulator(graph)
        self.metrics_calculator = ValidationMetricsCalculator()
        self.baseline_comparator = BaselineComparator(graph)
        self.cross_validator = CrossValidator()
    
    def run_full_validation(self) -> ValidationResult:
        """Run complete validation pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Full Graph Analysis Validation")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. Graph summary
        logger.info("\n[1/6] Analyzing graph structure...")
        graph_summary = self._get_graph_summary()
        
        # 2. Structural validation
        logger.info("\n[2/6] Running structural validation...")
        structural_results = self.structural_validator.validate_all()
        
        # 3. Calculate criticality scores
        logger.info("\n[3/6] Calculating criticality scores...")
        criticality_scores = self.criticality_calculator.calculate_all_scores(self.G)
        
        # 4. Run failure simulations
        logger.info("\n[4/6] Running failure simulations...")
        simulation_results = self.failure_simulator.simulate_all_failures()
        
        # 5. Validate criticality against simulations
        logger.info("\n[5/6] Validating criticality predictions...")
        criticality_validation = self._validate_criticality(
            criticality_scores, simulation_results
        )
        
        # 6. Statistical validation
        logger.info("\n[6/6] Computing statistical validation...")
        statistical_validation = self._compute_statistical_validation(
            criticality_scores, simulation_results
        )
        
        # Baseline comparison
        logger.info("\nComparing against baselines...")
        predicted_critical = {
            node for node, score in criticality_scores.items()
            if score.composite_score >= self.critical_threshold
        }
        actual_critical = {
            node for node, result in simulation_results.items()
            if result.impact_score >= self.critical_threshold
        }
        baseline_comparison = self.baseline_comparator.compare_all_baselines(
            predicted_critical, actual_critical, criticality_scores
        )
        
        # Cross-validation
        logger.info("\nRunning cross-validation...")
        cross_validation = self.cross_validator.cross_validate(
            self.G, self.criticality_calculator
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            structural_results, criticality_validation, statistical_validation
        )
        
        # Check target metrics
        target_metrics_met = {
            'spearman_correlation_target': statistical_validation.get(
                'spearman_correlation', 0
            ) >= TARGET_SPEARMAN_CORRELATION,
            'f1_score_target': criticality_validation.get('f1_score', 0) >= TARGET_F1_SCORE,
            'precision_target': criticality_validation.get('precision', 0) >= TARGET_PRECISION,
            'recall_target': criticality_validation.get('recall', 0) >= TARGET_RECALL
        }
        
        # Determine overall quality
        metrics_met = sum(target_metrics_met.values())
        if metrics_met == 4:
            overall_quality = "EXCELLENT"
        elif metrics_met >= 3:
            overall_quality = "GOOD"
        elif metrics_met >= 2:
            overall_quality = "ACCEPTABLE"
        else:
            overall_quality = "NEEDS_IMPROVEMENT"
        
        elapsed = time.time() - start_time
        logger.info(f"\nValidation completed in {elapsed:.2f}s")
        logger.info(f"Overall quality: {overall_quality}")
        
        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            graph_summary=graph_summary,
            structural_validation=structural_results,
            criticality_validation=criticality_validation,
            simulation_validation={
                'total_simulations': len(simulation_results),
                'avg_impact': round(
                    sum(r.impact_score for r in simulation_results.values()) / 
                    max(1, len(simulation_results)), 4
                ),
                'max_impact': round(
                    max(r.impact_score for r in simulation_results.values()) 
                    if simulation_results else 0, 4
                ),
                'high_impact_components': [
                    node for node, result in simulation_results.items()
                    if result.impact_score >= 0.5
                ][:10]
            },
            statistical_validation=statistical_validation,
            baseline_comparison=baseline_comparison,
            cross_validation=cross_validation,
            recommendations=recommendations,
            overall_quality=overall_quality,
            target_metrics_met=target_metrics_met
        )
    
    def _get_graph_summary(self) -> Dict[str, Any]:
        """Get graph summary statistics"""
        node_types = defaultdict(int)
        for _, data in self.G.nodes(data=True):
            node_types[data.get('type', 'Unknown')] += 1
        
        edge_types = defaultdict(int)
        for _, _, data in self.G.edges(data=True):
            edge_types[data.get('type', 'Unknown')] += 1
        
        degrees = [d for _, d in self.G.degree()]
        
        return {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'is_connected': nx.is_weakly_connected(self.G),
            'num_components': nx.number_weakly_connected_components(self.G),
            'avg_degree': round(sum(degrees) / max(1, len(degrees)), 2),
            'density': round(nx.density(self.G), 4)
        }
    
    def _validate_criticality(
        self,
        criticality_scores: Dict[str, CriticalityScore],
        simulation_results: Dict[str, FailureSimulationResult]
    ) -> Dict[str, Any]:
        """Validate criticality predictions against simulation results"""
        predicted_critical = {
            node for node, score in criticality_scores.items()
            if score.composite_score >= self.critical_threshold
        }
        
        actual_critical = {
            node for node, result in simulation_results.items()
            if result.impact_score >= self.critical_threshold
        }
        
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predicted_critical, actual_critical, len(self.G.nodes())
        )
        
        # Add detailed breakdown
        metrics['predicted_critical_count'] = len(predicted_critical)
        metrics['actual_critical_count'] = len(actual_critical)
        metrics['threshold_used'] = self.critical_threshold
        
        return metrics
    
    def _compute_statistical_validation(
        self,
        criticality_scores: Dict[str, CriticalityScore],
        simulation_results: Dict[str, FailureSimulationResult]
    ) -> Dict[str, Any]:
        """Compute statistical correlation between predictions and actuals"""
        predicted = {
            node: score.composite_score 
            for node, score in criticality_scores.items()
        }
        actual = {
            node: result.impact_score 
            for node, result in simulation_results.items()
        }
        
        correlation = self.metrics_calculator.calculate_rank_correlation(
            predicted, actual
        )
        
        # Add regression metrics
        common = set(predicted.keys()) & set(actual.keys())
        pred_list = [predicted[c] for c in common]
        actual_list = [actual[c] for c in common]
        
        regression = self.metrics_calculator.calculate_regression_metrics(
            pred_list, actual_list
        )
        
        return {**correlation, **regression}
    
    def _generate_recommendations(
        self,
        structural: Dict[str, Any],
        criticality: Dict[str, Any],
        statistical: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Structural recommendations
        if structural.get('total_errors', 0) > 0:
            recommendations.append(
                f"Fix {structural['total_errors']} structural errors in the graph"
            )
        
        if not structural.get('checks', {}).get('connectivity', {}).get('is_weakly_connected', True):
            recommendations.append(
                "Consider adding connections between disconnected components"
            )
        
        # Criticality recommendations
        precision = criticality.get('precision', 0)
        recall = criticality.get('recall', 0)
        
        if precision < TARGET_PRECISION:
            recommendations.append(
                f"Precision ({precision:.2%}) below target ({TARGET_PRECISION:.0%}). "
                "Consider adjusting critical threshold or alpha weight."
            )
        
        if recall < TARGET_RECALL:
            recommendations.append(
                f"Recall ({recall:.2%}) below target ({TARGET_RECALL:.0%}). "
                "Consider lowering critical threshold or increasing gamma weight."
            )
        
        # Statistical recommendations
        spearman = statistical.get('spearman_correlation', 0)
        
        if spearman < TARGET_SPEARMAN_CORRELATION:
            recommendations.append(
                f"Spearman correlation ({spearman:.3f}) below target ({TARGET_SPEARMAN_CORRELATION}). "
                "Consider tuning alpha/beta/gamma weights or incorporating additional metrics."
            )
        
        if not recommendations:
            recommendations.append(
                "All target metrics met. The analysis methodology is performing well."
            )
        
        return recommendations


# ============================================================================
# Report Generation
# ============================================================================

def generate_validation_report(result: ValidationResult, output_path: Optional[str] = None) -> str:
    """Generate human-readable validation report"""
    lines = []
    
    lines.append("=" * 70)
    lines.append("GRAPH ANALYSIS VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {result.timestamp}")
    lines.append(f"Overall Quality: {result.overall_quality}")
    lines.append("")
    
    # Graph Summary
    lines.append("-" * 70)
    lines.append("GRAPH SUMMARY")
    lines.append("-" * 70)
    gs = result.graph_summary
    lines.append(f"  Total Nodes: {gs['total_nodes']}")
    lines.append(f"  Total Edges: {gs['total_edges']}")
    lines.append(f"  Connected: {gs['is_connected']}")
    lines.append(f"  Density: {gs['density']}")
    lines.append(f"  Node Types: {gs['node_types']}")
    lines.append("")
    
    # Structural Validation
    lines.append("-" * 70)
    lines.append("STRUCTURAL VALIDATION")
    lines.append("-" * 70)
    sv = result.structural_validation
    lines.append(f"  Valid: {sv['is_valid']}")
    lines.append(f"  Errors: {sv['total_errors']}")
    lines.append(f"  Warnings: {sv['total_warnings']}")
    lines.append("")
    
    # Criticality Validation
    lines.append("-" * 70)
    lines.append("CRITICALITY VALIDATION (Target: P≥0.9, R≥0.85, F1≥0.9)")
    lines.append("-" * 70)
    cv = result.criticality_validation
    lines.append(f"  Precision: {cv['precision']:.4f} {'✓' if cv['precision'] >= TARGET_PRECISION else '✗'}")
    lines.append(f"  Recall: {cv['recall']:.4f} {'✓' if cv['recall'] >= TARGET_RECALL else '✗'}")
    lines.append(f"  F1 Score: {cv['f1_score']:.4f} {'✓' if cv['f1_score'] >= TARGET_F1_SCORE else '✗'}")
    lines.append(f"  Accuracy: {cv['accuracy']:.4f}")
    lines.append(f"  Predicted Critical: {cv['predicted_critical_count']}")
    lines.append(f"  Actual Critical: {cv['actual_critical_count']}")
    lines.append("")
    
    # Statistical Validation
    lines.append("-" * 70)
    lines.append(f"STATISTICAL VALIDATION (Target: Spearman ≥ {TARGET_SPEARMAN_CORRELATION})")
    lines.append("-" * 70)
    stv = result.statistical_validation
    spearman = stv.get('spearman_correlation', 0)
    lines.append(f"  Spearman Correlation: {spearman:.4f} {'✓' if spearman >= TARGET_SPEARMAN_CORRELATION else '✗'}")
    if 'kendall_tau' in stv:
        lines.append(f"  Kendall Tau: {stv['kendall_tau']:.4f}")
    lines.append(f"  MAE: {stv.get('mae', 'N/A')}")
    lines.append(f"  RMSE: {stv.get('rmse', 'N/A')}")
    lines.append("")
    
    # Baseline Comparison
    lines.append("-" * 70)
    lines.append("BASELINE COMPARISON")
    lines.append("-" * 70)
    bc = result.baseline_comparison
    lines.append(f"  Random Baseline F1: {bc['random']['f1_score']:.4f}")
    lines.append(f"  Degree-Based F1: {bc['degree_based']['f1_score']:.4f}")
    lines.append(f"  PageRank F1: {bc['pagerank']['f1_score']:.4f}")
    lines.append(f"  Our Method F1: {bc['composite_scoring']['f1_score']:.4f}")
    lines.append(f"  Improvement over Random: +{bc['improvement_over_random']:.4f}")
    lines.append(f"  Improvement over Degree: +{bc['improvement_over_degree']:.4f}")
    lines.append("")
    
    # Cross-Validation
    lines.append("-" * 70)
    lines.append("CROSS-VALIDATION")
    lines.append("-" * 70)
    xv = result.cross_validation
    if 'error' not in xv:
        lines.append(f"  K-Folds: {xv['k_folds']}")
        lines.append(f"  Mean Correlation: {xv['mean_correlation']:.4f}")
        lines.append(f"  Std Correlation: {xv['std_correlation']:.4f}")
    else:
        lines.append(f"  {xv['error']}")
    lines.append("")
    
    # Target Metrics Summary
    lines.append("-" * 70)
    lines.append("TARGET METRICS SUMMARY")
    lines.append("-" * 70)
    for metric, met in result.target_metrics_met.items():
        status = "✓ MET" if met else "✗ NOT MET"
        lines.append(f"  {metric}: {status}")
    lines.append("")
    
    # Recommendations
    lines.append("-" * 70)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 70)
    for i, rec in enumerate(result.recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Validate graph analysis results for pub-sub systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a graph from JSON file
  python validate_graph.py --input system.json
  
  # Use sample graph for testing
  python validate_graph.py --sample --nodes 50 --apps 100 --topics 75
  
  # Custom criticality weights
  python validate_graph.py --input system.json --alpha 0.5 --beta 0.25 --gamma 0.25
  
  # Save results to files
  python validate_graph.py --input system.json --output-json results.json --output-report report.txt
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input graph JSON file'
    )
    input_group.add_argument(
        '--sample',
        action='store_true',
        help='Generate and validate a sample graph'
    )
    
    # Sample graph options
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes for sample graph')
    parser.add_argument('--apps', type=int, default=20, help='Number of applications for sample graph')
    parser.add_argument('--topics', type=int, default=15, help='Number of topics for sample graph')
    parser.add_argument('--brokers', type=int, default=3, help='Number of brokers for sample graph')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Criticality parameters
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f'Weight for betweenness centrality (default: {DEFAULT_ALPHA})')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA,
                        help=f'Weight for articulation point (default: {DEFAULT_BETA})')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'Weight for impact score (default: {DEFAULT_GAMMA})')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Critical threshold for classification (default: 0.6)')
    
    # Output options
    parser.add_argument('--output-json', '-o', type=str, help='Output JSON file for results')
    parser.add_argument('--output-report', '-r', type=str, help='Output text file for report')
    
    # Misc options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress console output')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    setup_logging(args.verbose, args.log_file)
    logger.setLevel(log_level)
    
    # Load or generate graph
    if args.sample:
        logger.info("Generating sample graph...")
        G = generate_sample_graph(
            num_nodes=args.nodes,
            num_apps=args.apps,
            num_topics=args.topics,
            num_brokers=args.brokers,
            seed=args.seed
        )
        raw_data = None
    else:
        logger.info(f"Loading graph from {args.input}...")
        G, raw_data = load_graph_from_json(args.input)
    
    # Validate weights sum to 1.0
    weight_sum = args.alpha + args.beta + args.gamma
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
        args.alpha /= weight_sum
        args.beta /= weight_sum
        args.gamma /= weight_sum
    
    # Create validation engine
    engine = GraphValidationEngine(
        graph=G,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        critical_threshold=args.threshold
    )
    
    # Run validation
    result = engine.run_full_validation()
    
    # Generate report
    report = generate_validation_report(result, args.output_report)
    
    if not args.quiet:
        print("\n" + report)
    
    # Save JSON results if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved to {args.output_json}")
    
    # Return exit code based on quality
    if result.overall_quality in ['EXCELLENT', 'GOOD']:
        return 0
    elif result.overall_quality == 'ACCEPTABLE':
        return 1
    else:
        return 2


if __name__ == '__main__':
    sys.exit(main())
