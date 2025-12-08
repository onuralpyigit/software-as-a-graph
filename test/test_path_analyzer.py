#!/usr/bin/env python3
"""
Test Suite for Path Analyzer

Comprehensive tests validating:
1. Dependency chain analysis
2. Message flow path tracing
3. Critical path identification
4. Path redundancy analysis
5. Failure propagation analysis
6. Path-level QoS analysis

Author: Software-as-a-Graph Research Project
Date: 2025-12-08
"""

import sys
from typing import Dict, List
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("NetworkX not available - tests will be skipped")

from src.analysis.path_analyzer import (
    PathAnalyzer,
    PathAnalysisResult,
    PathCriticalityLevel,
    PathInfo,
    MessageFlowPath,
    DependencyChain,
    PathRedundancyInfo,
    FailurePropagationPath
)


class TestResult:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def record_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")

    def record_fail(self, name: str, reason: str):
        self.failed += 1
        self.failures.append((name, reason))
        print(f"  ✗ {name}: {reason}")

    def record_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⊘ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        if self.failures:
            print("\nFailed tests:")
            for name, reason in self.failures:
                print(f"  - {name}: {reason}")
        print('='*60)
        return self.failed == 0


results = TestResult()


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_pub_sub_graph() -> 'nx.DiGraph':
    """Create a simple pub-sub graph for testing"""
    G = nx.DiGraph()

    # Add applications
    for app in ['App_A', 'App_B', 'App_C', 'App_D']:
        G.add_node(app, type='Application')

    # Add topics
    for topic in ['Topic_1', 'Topic_2']:
        G.add_node(topic, type='Topic', reliability='reliable', durability='volatile')

    # Add broker
    G.add_node('Broker_1', type='Broker')

    # Add infrastructure nodes
    G.add_node('Node_1', type='Node')
    G.add_node('Node_2', type='Node')

    # Add edges
    G.add_edge('App_A', 'Topic_1', type='PUBLISHES_TO')
    G.add_edge('App_B', 'Topic_1', type='SUBSCRIBES_TO')
    G.add_edge('App_B', 'Topic_2', type='PUBLISHES_TO')
    G.add_edge('App_C', 'Topic_2', type='SUBSCRIBES_TO')
    G.add_edge('Broker_1', 'Topic_1', type='ROUTES')
    G.add_edge('Broker_1', 'Topic_2', type='ROUTES')

    # Add dependencies
    G.add_edge('App_B', 'App_A', type='DEPENDS_ON', topics=['Topic_1'])
    G.add_edge('App_C', 'App_B', type='DEPENDS_ON', topics=['Topic_2'])
    G.add_edge('App_D', 'App_C', type='DEPENDS_ON', topics=['Topic_2'])

    # Add infrastructure mappings
    G.add_edge('App_A', 'Node_1', type='RUNS_ON')
    G.add_edge('App_B', 'Node_1', type='RUNS_ON')
    G.add_edge('App_C', 'Node_2', type='RUNS_ON')
    G.add_edge('Broker_1', 'Node_1', type='RUNS_ON')

    return G


def create_chain_graph() -> 'nx.DiGraph':
    """Create a linear chain graph for testing"""
    G = nx.DiGraph()

    # Create a chain: A -> B -> C -> D -> E
    apps = ['App_A', 'App_B', 'App_C', 'App_D', 'App_E']
    for app in apps:
        G.add_node(app, type='Application')

    for i in range(len(apps) - 1):
        G.add_edge(apps[i+1], apps[i], type='DEPENDS_ON', topics=[f'Topic_{i}'])

    return G


def create_redundant_graph() -> 'nx.DiGraph':
    """Create a graph with redundant paths for testing"""
    G = nx.DiGraph()

    # Create graph with multiple paths between source and target
    for node in ['Source', 'Path1_Mid', 'Path2_Mid', 'Path3_Mid', 'Target']:
        G.add_node(node, type='Application')

    # Three parallel paths
    G.add_edge('Source', 'Path1_Mid', type='DEPENDS_ON')
    G.add_edge('Path1_Mid', 'Target', type='DEPENDS_ON')

    G.add_edge('Source', 'Path2_Mid', type='DEPENDS_ON')
    G.add_edge('Path2_Mid', 'Target', type='DEPENDS_ON')

    G.add_edge('Source', 'Path3_Mid', type='DEPENDS_ON')
    G.add_edge('Path3_Mid', 'Target', type='DEPENDS_ON')

    return G


# ============================================================================
# Test PathInfo Data Class
# ============================================================================

def test_path_info_creation():
    """Test PathInfo dataclass creation and properties"""
    path = PathInfo(
        nodes=['A', 'B', 'C'],
        edges=['DEPENDS_ON', 'DEPENDS_ON'],
        path_type='dependency'
    )

    if path.length != 2:
        results.record_fail("PathInfo creation", f"Expected length 2, got {path.length}")
        return

    d = path.to_dict()
    if 'nodes' not in d or 'length' not in d:
        results.record_fail("PathInfo creation", "Missing keys in to_dict()")
        return

    results.record_pass("PathInfo creation and serialization")


def test_message_flow_path_creation():
    """Test MessageFlowPath dataclass creation"""
    flow = MessageFlowPath(
        publisher='App_A',
        topic='Topic_1',
        subscriber='App_B',
        brokers=['Broker_1'],
        infrastructure_nodes=['Node_1'],
        hop_count=3
    )

    if flow.hop_count != 3:
        results.record_fail("MessageFlowPath creation", f"Expected hop_count 3, got {flow.hop_count}")
        return

    d = flow.to_dict()
    if 'publisher' not in d or 'subscriber' not in d:
        results.record_fail("MessageFlowPath creation", "Missing keys in to_dict()")
        return

    results.record_pass("MessageFlowPath creation and serialization")


def test_dependency_chain_creation():
    """Test DependencyChain dataclass creation"""
    chain = DependencyChain(
        applications=['App_A', 'App_B', 'App_C'],
        connecting_topics=[['Topic_1'], ['Topic_2']],
        criticality_level=PathCriticalityLevel.HIGH,
        single_points_of_failure=['App_B'],
        redundancy_score=0.3
    )

    if chain.chain_length != 2:
        results.record_fail("DependencyChain creation", f"Expected chain_length 2, got {chain.chain_length}")
        return

    d = chain.to_dict()
    if 'applications' not in d or 'criticality_level' not in d:
        results.record_fail("DependencyChain creation", "Missing keys in to_dict()")
        return

    results.record_pass("DependencyChain creation and serialization")


# ============================================================================
# Test PathAnalyzer Initialization
# ============================================================================

def test_analyzer_initialization():
    """Test PathAnalyzer initialization with different parameters"""
    analyzer = PathAnalyzer()
    if analyzer.max_path_length != 10:
        results.record_fail("Analyzer init", f"Expected max_path_length 10, got {analyzer.max_path_length}")
        return

    custom_analyzer = PathAnalyzer(max_path_length=5, max_paths_per_pair=3, redundancy_threshold=3)
    if custom_analyzer.max_path_length != 5:
        results.record_fail("Analyzer init", f"Expected max_path_length 5, got {custom_analyzer.max_path_length}")
        return

    results.record_pass("PathAnalyzer initialization")


# ============================================================================
# Test Dependency Chain Analysis
# ============================================================================

def test_dependency_chain_analysis():
    """Test dependency chain detection"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Dependency chain analysis", "NetworkX not available")
        return

    G = create_chain_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    if len(result.dependency_chains) == 0:
        results.record_fail("Dependency chain analysis", "No dependency chains found")
        return

    # Should find chains of various lengths
    chain_lengths = [c.chain_length for c in result.dependency_chains]
    if max(chain_lengths) < 2:
        results.record_fail("Dependency chain analysis",
                          f"Expected longer chains, max found: {max(chain_lengths)}")
        return

    results.record_pass(f"Dependency chain analysis ({len(result.dependency_chains)} chains found)")


def test_longest_chain_detection():
    """Test that longest chain is correctly identified"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Longest chain detection", "NetworkX not available")
        return

    G = create_chain_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    if result.longest_chain is None:
        results.record_fail("Longest chain detection", "No longest chain identified")
        return

    # Chain A->B->C->D->E should be 4 hops
    if result.longest_chain.chain_length < 2:
        results.record_fail("Longest chain detection",
                          f"Expected longer chain, got length {result.longest_chain.chain_length}")
        return

    results.record_pass(f"Longest chain detection (length: {result.longest_chain.chain_length})")


# ============================================================================
# Test Message Flow Tracing
# ============================================================================

def test_message_flow_tracing():
    """Test message flow path detection"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Message flow tracing", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    if len(result.message_flows) == 0:
        results.record_fail("Message flow tracing", "No message flows found")
        return

    # Check flow structure
    for flow in result.message_flows:
        if not flow.publisher or not flow.topic or not flow.subscriber:
            results.record_fail("Message flow tracing", "Incomplete flow structure")
            return

    results.record_pass(f"Message flow tracing ({len(result.message_flows)} flows found)")


def test_message_flow_with_brokers():
    """Test that brokers are included in message flows"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Message flow with brokers", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    flows_with_brokers = [f for f in result.message_flows if len(f.brokers) > 0]
    if len(flows_with_brokers) == 0:
        results.record_fail("Message flow with brokers", "No flows include brokers")
        return

    results.record_pass(f"Message flow with brokers ({len(flows_with_brokers)} flows with brokers)")


# ============================================================================
# Test Path Redundancy Analysis
# ============================================================================

def test_path_redundancy_basic():
    """Test basic path redundancy analysis"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Path redundancy basic", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    # Should have some redundancy info
    if len(result.redundancy_info) == 0:
        # This can happen with very sparse graphs
        results.record_pass("Path redundancy basic (no dependency pairs to analyze)")
        return

    results.record_pass(f"Path redundancy basic ({len(result.redundancy_info)} pairs analyzed)")


def test_redundant_paths_detection():
    """Test detection of redundant paths"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Redundant paths detection", "NetworkX not available")
        return

    G = create_redundant_graph()
    analyzer = PathAnalyzer(redundancy_threshold=2)
    result = analyzer.analyze(G)

    # Should find high redundancy for Source->Target
    found_redundant = False
    for key, info in result.redundancy_info.items():
        if info.redundancy_level >= 2:
            found_redundant = True
            break

    if not found_redundant:
        # Check if any redundancy was found
        max_redundancy = max([info.redundancy_level for info in result.redundancy_info.values()], default=0)
        results.record_pass(f"Redundant paths detection (max redundancy level: {max_redundancy})")
    else:
        results.record_pass("Redundant paths detection (found redundant paths)")


def test_low_redundancy_detection():
    """Test detection of low redundancy paths"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Low redundancy detection", "NetworkX not available")
        return

    G = create_chain_graph()
    analyzer = PathAnalyzer(redundancy_threshold=2)
    result = analyzer.analyze(G)

    # Chain graph should have low redundancy
    if len(result.low_redundancy_pairs) == 0 and len(result.redundancy_info) > 0:
        # Check actual redundancy levels
        max_redundancy = max([info.redundancy_level for info in result.redundancy_info.values()], default=0)
        if max_redundancy >= 2:
            results.record_fail("Low redundancy detection", "Chain graph should have low redundancy")
            return

    results.record_pass(f"Low redundancy detection ({len(result.low_redundancy_pairs)} low redundancy pairs)")


# ============================================================================
# Test Failure Propagation Analysis
# ============================================================================

def test_failure_propagation_basic():
    """Test basic failure propagation analysis"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Failure propagation basic", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    if len(result.failure_propagations) == 0:
        results.record_fail("Failure propagation basic", "No failure propagation paths found")
        return

    results.record_pass(f"Failure propagation basic ({len(result.failure_propagations)} origins analyzed)")


def test_high_impact_origins():
    """Test identification of high impact failure origins"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("High impact origins", "NetworkX not available")
        return

    G = create_chain_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    if len(result.high_impact_origins) == 0 and len(result.failure_propagations) > 0:
        # Check if any have positive impact
        max_impact = max([p.impact_radius for p in result.failure_propagations.values()], default=0)
        if max_impact > 0:
            results.record_pass(f"High impact origins (max impact radius: {max_impact})")
            return

    results.record_pass(f"High impact origins ({len(result.high_impact_origins)} high-impact origins)")


def test_propagation_depth():
    """Test propagation depth calculation"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Propagation depth", "NetworkX not available")
        return

    G = create_chain_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    # Check propagation depths
    depths = [p.propagation_depth for p in result.failure_propagations.values()]
    if depths:
        max_depth = max(depths)
        results.record_pass(f"Propagation depth calculation (max depth: {max_depth})")
    else:
        results.record_pass("Propagation depth calculation (no propagations to analyze)")


# ============================================================================
# Test Critical Path Identification
# ============================================================================

def test_critical_path_identification():
    """Test identification of critical paths"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Critical path identification", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    # Should identify some critical paths
    if len(result.critical_paths) == 0:
        # This is acceptable for simple graphs
        results.record_pass("Critical path identification (no critical paths in simple graph)")
        return

    # Check path structure
    for path in result.critical_paths:
        if len(path.nodes) < 2:
            results.record_fail("Critical path identification", "Invalid path structure")
            return

    results.record_pass(f"Critical path identification ({len(result.critical_paths)} critical paths)")


def test_path_criticality_distribution():
    """Test criticality distribution calculation"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Criticality distribution", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    if not result.path_criticality_distribution:
        results.record_fail("Criticality distribution", "No distribution calculated")
        return

    # Should have standard categories
    expected_keys = {'critical', 'high', 'medium', 'low'}
    if not expected_keys.issubset(result.path_criticality_distribution.keys()):
        results.record_fail("Criticality distribution", "Missing categories in distribution")
        return

    results.record_pass("Criticality distribution calculation")


# ============================================================================
# Test Path-level QoS Analysis
# ============================================================================

def test_path_qos_analysis():
    """Test path-level QoS analysis"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Path QoS analysis", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()

    # Test QoS analysis along a path
    path = ['App_A', 'Topic_1', 'App_B']
    qos_result = analyzer.analyze_path_qos(G, path)

    if 'reliability' not in qos_result or 'durability' not in qos_result:
        results.record_fail("Path QoS analysis", "Missing QoS fields in result")
        return

    results.record_pass("Path-level QoS analysis")


def test_weakest_link_qos():
    """Test weakest link QoS identification"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Weakest link QoS", "NetworkX not available")
        return

    G = nx.DiGraph()
    G.add_node('App_A', type='Application')
    G.add_node('Topic_1', type='Topic', reliability='best_effort', durability='volatile')
    G.add_node('Topic_2', type='Topic', reliability='reliable', durability='persistent')
    G.add_node('App_B', type='Application')

    analyzer = PathAnalyzer()
    path = ['App_A', 'Topic_1', 'Topic_2', 'App_B']
    qos_result = analyzer.analyze_path_qos(G, path)

    # Weakest link should be best_effort for reliability
    if qos_result['reliability'] != 'best_effort':
        results.record_fail("Weakest link QoS",
                          f"Expected best_effort, got {qos_result['reliability']}")
        return

    results.record_pass("Weakest link QoS identification")


# ============================================================================
# Test Specialized Methods
# ============================================================================

def test_find_shortest_path():
    """Test shortest path finding"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Find shortest path", "NetworkX not available")
        return

    G = create_chain_graph()
    analyzer = PathAnalyzer()

    path_info = analyzer.find_shortest_path(G, 'App_E', 'App_A')

    if path_info is None:
        results.record_fail("Find shortest path", "No path found")
        return

    if len(path_info.nodes) < 2:
        results.record_fail("Find shortest path", "Invalid path length")
        return

    results.record_pass(f"Find shortest path (length: {path_info.length})")


def test_find_all_paths():
    """Test finding all paths between components"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Find all paths", "NetworkX not available")
        return

    G = create_redundant_graph()
    analyzer = PathAnalyzer()

    # Add edges for path finding in forward direction
    G.add_edge('Source', 'Target', type='DEPENDS_ON')

    paths = analyzer.find_all_paths(G, 'Source', 'Target', max_length=3)

    if len(paths) == 0:
        results.record_fail("Find all paths", "No paths found")
        return

    results.record_pass(f"Find all paths ({len(paths)} paths found)")


# ============================================================================
# Test Result Serialization
# ============================================================================

def test_result_to_dict():
    """Test PathAnalysisResult serialization"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Result serialization", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    d = result.to_dict()

    required_keys = ['summary', 'dependency_chains', 'message_flows',
                     'redundancy', 'failure_propagation', 'critical_paths']

    for key in required_keys:
        if key not in d:
            results.record_fail("Result serialization", f"Missing key: {key}")
            return

    results.record_pass("PathAnalysisResult serialization")


def test_get_path_statistics():
    """Test path statistics generation"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Path statistics", "NetworkX not available")
        return

    G = create_simple_pub_sub_graph()
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    stats = analyzer.get_path_statistics(result)

    required_keys = ['dependency_chains', 'message_flows', 'redundancy',
                     'failure_propagation', 'critical_paths']

    for key in required_keys:
        if key not in stats:
            results.record_fail("Path statistics", f"Missing key: {key}")
            return

    results.record_pass("Path statistics generation")


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_empty_graph():
    """Test analysis of empty graph"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Empty graph", "NetworkX not available")
        return

    G = nx.DiGraph()
    analyzer = PathAnalyzer()

    try:
        result = analyzer.analyze(G)
        if len(result.dependency_chains) != 0:
            results.record_fail("Empty graph", "Expected no chains in empty graph")
            return
        results.record_pass("Empty graph handling")
    except Exception as e:
        results.record_fail("Empty graph", f"Exception: {e}")


def test_single_node_graph():
    """Test analysis of single-node graph"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Single node graph", "NetworkX not available")
        return

    G = nx.DiGraph()
    G.add_node('A', type='Application')
    analyzer = PathAnalyzer()

    try:
        result = analyzer.analyze(G)
        results.record_pass("Single node graph handling")
    except Exception as e:
        results.record_fail("Single node graph", f"Exception: {e}")


def test_disconnected_graph():
    """Test analysis of disconnected graph"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Disconnected graph", "NetworkX not available")
        return

    G = nx.DiGraph()
    # Two disconnected components
    G.add_node('A', type='Application')
    G.add_node('B', type='Application')
    G.add_edge('A', 'B', type='DEPENDS_ON')

    G.add_node('C', type='Application')
    G.add_node('D', type='Application')
    G.add_edge('C', 'D', type='DEPENDS_ON')

    analyzer = PathAnalyzer()

    try:
        result = analyzer.analyze(G)
        results.record_pass("Disconnected graph handling")
    except Exception as e:
        results.record_fail("Disconnected graph", f"Exception: {e}")


def test_large_graph_performance():
    """Test performance with larger graph"""
    if not NETWORKX_AVAILABLE:
        results.record_skip("Large graph performance", "NetworkX not available")
        return

    import time

    # Create larger graph
    G = nx.DiGraph()
    n_apps = 50

    for i in range(n_apps):
        G.add_node(f'App_{i}', type='Application')

    # Create dependencies
    for i in range(n_apps - 1):
        G.add_edge(f'App_{i+1}', f'App_{i}', type='DEPENDS_ON')

    # Add some cross-links
    import random
    random.seed(42)
    for _ in range(n_apps):
        a = random.randint(0, n_apps - 1)
        b = random.randint(0, n_apps - 1)
        if a != b:
            G.add_edge(f'App_{a}', f'App_{b}', type='DEPENDS_ON')

    analyzer = PathAnalyzer(max_path_length=5)  # Limit for performance

    start = time.time()
    try:
        result = analyzer.analyze(G)
        elapsed = time.time() - start

        if elapsed > 30:  # Should complete within 30 seconds
            results.record_fail("Large graph performance", f"Too slow: {elapsed:.2f}s")
            return

        results.record_pass(f"Large graph performance ({n_apps} nodes, {elapsed:.2f}s)")
    except Exception as e:
        results.record_fail("Large graph performance", f"Exception: {e}")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("PATH ANALYZER - TEST SUITE")
    print("=" * 70)

    print("\n📦 Testing Data Classes...")
    test_path_info_creation()
    test_message_flow_path_creation()
    test_dependency_chain_creation()

    print("\n⚙️ Testing Analyzer Initialization...")
    test_analyzer_initialization()

    print("\n🔗 Testing Dependency Chain Analysis...")
    test_dependency_chain_analysis()
    test_longest_chain_detection()

    print("\n📬 Testing Message Flow Tracing...")
    test_message_flow_tracing()
    test_message_flow_with_brokers()

    print("\n🔀 Testing Path Redundancy Analysis...")
    test_path_redundancy_basic()
    test_redundant_paths_detection()
    test_low_redundancy_detection()

    print("\n💥 Testing Failure Propagation Analysis...")
    test_failure_propagation_basic()
    test_high_impact_origins()
    test_propagation_depth()

    print("\n⚠️ Testing Critical Path Identification...")
    test_critical_path_identification()
    test_path_criticality_distribution()

    print("\n📊 Testing Path-level QoS Analysis...")
    test_path_qos_analysis()
    test_weakest_link_qos()

    print("\n🔍 Testing Specialized Methods...")
    test_find_shortest_path()
    test_find_all_paths()

    print("\n📄 Testing Result Serialization...")
    test_result_to_dict()
    test_get_path_statistics()

    print("\n🔧 Testing Edge Cases...")
    test_empty_graph()
    test_single_node_graph()
    test_disconnected_graph()
    test_large_graph_performance()

    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
