#!/usr/bin/env python3
"""
Test Suite for Graph Analysis Scripts

Tests the analyze_graph.py script with various scenarios including:
- Small test graphs
- Graphs with articulation points
- Graphs with bridges
- Disconnected graphs
- Various graph topologies

Usage:
    python test_graph_analyzer.py
    python test_graph_analyzer.py --verbose
    python test_graph_analyzer.py --test-neo4j
"""

import argparse
import json
import sys
import tempfile
import os
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

# Test graph generators
def generate_simple_test_graph():
    """Generate a simple connected graph"""
    return {
        "nodes": [
            {"id": "node1", "name": "node1"},
            {"id": "node2", "name": "node2"}
        ],
        "applications": [
            {"id": "app1", "name": "app1", "type": "PRODUCER"},
            {"id": "app2", "name": "app2", "type": "PROSUMER"},
            {"id": "app3", "name": "app3", "type": "CONSUMER"}
        ],
        "topics": [
            {"id": "topic1", "name": "topic1"},
            {"id": "topic2", "name": "topic2"}
        ],
        "brokers": [
            {"id": "broker1", "name": "broker1"}
        ],
        "relationships": {
            "runs_on": [
                {"from": "app1", "to": "node1"},
                {"from": "app2", "to": "node1"},
                {"from": "app3", "to": "node2"},
                {"from": "broker1", "to": "node1"}
            ],
            "publishes_to": [
                {"from": "app1", "to": "topic1"},
                {"from": "app2", "to": "topic2"}
            ],
            "subscribes_to": [
                {"from": "app2", "to": "topic1"},
                {"from": "app3", "to": "topic2"}
            ],
            "routes": [
                {"from": "broker1", "to": "topic1"},
                {"from": "broker1", "to": "topic2"}
            ]
        }
    }


def generate_articulation_point_graph():
    """Generate a graph with clear articulation points"""
    return {
        "nodes": [
            {"id": "node1", "name": "node1"},
            {"id": "node2", "name": "node2"},
            {"id": "node3", "name": "node3"}
        ],
        "applications": [
            {"id": "app1", "name": "app1", "type": "PRODUCER"},
            {"id": "app2", "name": "app2", "type": "PROSUMER"},
            {"id": "app3", "name": "app3", "type": "PROSUMER"},
            {"id": "app4", "name": "app4", "type": "PRODUCER"},
            {"id": "app5", "name": "app5", "type": "CONSUMER"}
        ],
        "topics": [
            {"id": "topic_central", "name": "topic_central"},
            {"id": "topic_a", "name": "topic_a"},
            {"id": "topic_b", "name": "topic_b"}
        ],
        "brokers": [],
        "relationships": {
            "runs_on": [
                {"from": "app1", "to": "node1"},
                {"from": "app2", "to": "node1"},
                {"from": "app3", "to": "node2"},
                {"from": "app4", "to": "node3"},
                {"from": "app5", "to": "node3"}
            ],
            "publishes_to": [
                {"from": "app1", "to": "topic_a"},
                {"from": "app2", "to": "topic_central"},
                {"from": "app3", "to": "topic_central"},
                {"from": "app4", "to": "topic_b"}
            ],
            "subscribes_to": [
                {"from": "app2", "to": "topic_a"},
                {"from": "app3", "to": "topic_a"},
                {"from": "app3", "to": "topic_b"},
                {"from": "app5", "to": "topic_b"}
            ],
            "routes": []
        }
    }


def generate_bridge_graph():
    """Generate a graph with bridge edges"""
    return {
        "nodes": [
            {"id": "node1", "name": "node1"},
            {"id": "node2", "name": "node2"}
        ],
        "applications": [
            {"id": "app_cluster1_a", "name": "app_cluster1_a", "type": "PRODUCER"},
            {"id": "app_cluster1_b", "name": "app_cluster1_b", "type": "PROSUMER"},
            {"id": "app_cluster2_a", "name": "app_cluster2_a", "type": "PROSUMER"},
            {"id": "app_cluster2_b", "name": "app_cluster2_b", "type": "CONSUMER"}
        ],
        "topics": [
            {"id": "topic_cluster1", "name": "topic_cluster1"},
            {"id": "topic_bridge", "name": "topic_bridge"},
            {"id": "topic_cluster2", "name": "topic_cluster2"}
        ],
        "brokers": [],
        "relationships": {
            "runs_on": [
                {"from": "app_cluster1_a", "to": "node1"},
                {"from": "app_cluster1_b", "to": "node1"},
                {"from": "app_cluster2_a", "to": "node2"},
                {"from": "app_cluster2_b", "to": "node2"}
            ],
            "publishes_to": [
                {"from": "app_cluster1_a", "to": "topic_cluster1"},
                {"from": "app_cluster1_b", "to": "topic_bridge"},
                {"from": "app_cluster2_a", "to": "topic_cluster2"}
            ],
            "subscribes_to": [
                {"from": "app_cluster1_b", "to": "topic_cluster1"},
                {"from": "app_cluster2_a", "to": "topic_bridge"},
                {"from": "app_cluster2_b", "to": "topic_cluster2"}
            ],
            "routes": []
        }
    }


def generate_disconnected_graph():
    """Generate a disconnected graph"""
    return {
        "nodes": [
            {"id": "node1", "name": "node1"},
            {"id": "node2", "name": "node2"}
        ],
        "applications": [
            {"id": "app1", "name": "app1", "type": "PRODUCER"},
            {"id": "app2", "name": "app2", "type": "CONSUMER"},
            {"id": "app3", "name": "app3", "type": ""},
        ],
        "topics": [
            {"id": "topic1", "name": "topic1"},
            {"id": "topic2", "name": "topic2"}
        ],
        "brokers": [],
        "relationships": {
            "runs_on": [
                {"from": "app1", "to": "node1"},
                {"from": "app2", "to": "node1"},
                {"from": "app3", "to": "node2"}
            ],
            "publishes_to": [
                {"from": "app1", "to": "topic1"}
            ],
            "subscribes_to": [
                {"from": "app2", "to": "topic1"}
            ],
            "routes": []
        }
    }


def generate_star_topology_graph():
    """Generate a star topology (one central hub)"""
    return {
        "nodes": [{"id": f"node{i}", "name": f"node{i}"} for i in range(1, 6)],
        "applications": [{"id": f"app{i}", "name": f"app{i}", "type": "PROSUMER"} for i in range(1, 11)],
        "topics": [
            {"id": "topic_central", "name": "topic_central"}
        ] + [{"id": f"topic{i}", "name": f"topic{i}"} for i in range(1, 6)],
        "brokers": [{"id": "broker_central", "name": "broker_central"}],
        "relationships": {
            "runs_on": [
                {"from": f"app{i}", "to": f"node{(i-1)//2+1}"} for i in range(1, 11)
            ] + [{"from": "broker_central", "to": "node1"}],
            "publishes_to": [
                {"from": f"app{i}", "to": "topic_central"} for i in range(1, 6)
            ],
            "subscribes_to": [
                {"from": f"app{i}", "to": "topic_central"} for i in range(6, 11)
            ],
            "routes": [
                {"from": "broker_central", "to": "topic_central"}
            ] + [{"from": "broker_central", "to": f"topic{i}"} for i in range(1, 6)]
        }
    }


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(test_name):
    """Print test header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}TEST: {test_name}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.ENDC}")


def run_analysis_test(test_name, graph_data, expected_features):
    """
    Run a single analysis test
    
    Args:
        test_name: Name of the test
        graph_data: Graph data dictionary
        expected_features: Dictionary of expected features to verify
    
    Returns:
        True if test passes, False otherwise
    """
    print_test_header(test_name)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(graph_data, f, indent=2)
        temp_file = f.name
    
    try:
        # Run analysis
        print_info(f"Running analysis on {temp_file}...")
        result = subprocess.run(
            ['python3', '../analyze_graph.py', '--input', temp_file, '--export-json', 
             '--output', temp_file.replace('.json', '_result')],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print_error(f"Analysis failed with return code {result.returncode}")
            print(result.stderr)
            return False
        
        # Load results
        result_file = temp_file.replace('.json', '_result.json')
        if not os.path.exists(result_file):
            print_error("Result file not created")
            return False
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Verify expected features
        all_passed = True
        
        if 'min_nodes' in expected_features:
            actual = results['structure']['nodes']
            expected = expected_features['min_nodes']
            if actual >= expected:
                print_success(f"Node count: {actual} >= {expected}")
            else:
                print_error(f"Node count: {actual} < {expected}")
                all_passed = False
        
        if 'has_articulation_points' in expected_features:
            actual = results['node_analysis']['statistics']['articulation_points_count']
            if expected_features['has_articulation_points']:
                if actual > 0:
                    print_success(f"Found {actual} articulation points (expected)")
                else:
                    print_error("No articulation points found (expected some)")
                    all_passed = False
            else:
                if actual == 0:
                    print_success("No articulation points (as expected)")
                else:
                    print_error(f"Found {actual} articulation points (expected 0)")
                    all_passed = False
        
        if 'has_bridges' in expected_features:
            actual = results['edge_analysis']['statistics']['bridge_count']
            if expected_features['has_bridges']:
                if actual > 0:
                    print_success(f"Found {actual} bridges (expected)")
                else:
                    print_error("No bridges found (expected some)")
                    all_passed = False
            else:
                if actual == 0:
                    print_success("No bridges (as expected)")
                else:
                    print_error(f"Found {actual} bridges (expected 0)")
                    all_passed = False
        
        if 'is_connected' in expected_features:
            actual = results['structure']['is_weakly_connected']
            expected = expected_features['is_connected']
            if actual == expected:
                print_success(f"Connected: {actual} (as expected)")
            else:
                print_error(f"Connected: {actual} (expected {expected})")
                all_passed = False
        
        if 'min_recommendations' in expected_features:
            actual = len(results['recommendations'])
            expected = expected_features['min_recommendations']
            if actual >= expected:
                print_success(f"Recommendations: {actual} >= {expected}")
            else:
                print_error(f"Recommendations: {actual} < {expected}")
                all_passed = False
        
        # Cleanup
        os.unlink(temp_file)
        os.unlink(result_file)
        
        if all_passed:
            print_success(f"Test '{test_name}' PASSED")
        else:
            print_error(f"Test '{test_name}' FAILED")
        
        return all_passed
        
    except subprocess.TimeoutExpired:
        print_error("Analysis timed out")
        os.unlink(temp_file)
        return False
    except Exception as e:
        print_error(f"Test error: {e}")
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False


def run_all_tests(verbose=False):
    """Run all tests"""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("GRAPH ANALYSIS TEST SUITE")
    print(f"{'='*70}{Colors.ENDC}\n")
    
    tests = [
        {
            'name': 'Simple Connected Graph',
            'generator': generate_simple_test_graph,
            'expected': {
                'min_nodes': 6,
                'is_connected': True,
                'min_recommendations': 0
            }
        },
        {
            'name': 'Articulation Point Detection',
            'generator': generate_articulation_point_graph,
            'expected': {
                'min_nodes': 8,
                'has_articulation_points': True,
                'is_connected': True,
                'min_recommendations': 1
            }
        },
        {
            'name': 'Bridge Edge Detection',
            'generator': generate_bridge_graph,
            'expected': {
                'min_nodes': 7,
                'has_bridges': True,
                'is_connected': True,
                'min_recommendations': 1
            }
        },
        {
            'name': 'Disconnected Graph',
            'generator': generate_disconnected_graph,
            'expected': {
                'min_nodes': 5,
                'is_connected': False,
                'min_recommendations': 1
            }
        },
        {
            'name': 'Star Topology',
            'generator': generate_star_topology_graph,
            'expected': {
                'min_nodes': 15,
                'has_articulation_points': True,
                'is_connected': True,
                'min_recommendations': 1
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_spec in tests:
        graph_data = test_spec['generator']()
        if run_analysis_test(test_spec['name'], graph_data, test_spec['expected']):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}{Colors.ENDC}")
    print(f"Total tests: {passed + failed}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.ENDC}")
    else:
        print(f"Failed: {failed}")
    
    success_rate = (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    return failed == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test suite for graph analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--test', '-t', type=str,
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    # Check if analyze_graph.py exists
    if not os.path.exists('../analyze_graph.py'):
        print_error("analyze_graph.py not found in current directory")
        return 1
    
    try:
        success = run_all_tests(verbose=args.verbose)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130


if __name__ == '__main__':
    sys.exit(main())
