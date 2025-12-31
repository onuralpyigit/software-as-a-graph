#!/usr/bin/env python3
"""
Simple Test Runner
===================

Runs the test suite without pytest (uses unittest discovery).
For full functionality, install pytest: pip install pytest

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Quick tests only
    python run_tests.py --verbose    # Verbose output
    python run_tests.py --module core  # Run specific module tests
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Callable
from dataclasses import dataclass


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Terminal Output
# =============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @classmethod
    def disable(cls):
        for attr in ['GREEN', 'RED', 'YELLOW', 'BLUE', 'CYAN', 'BOLD', 'END']:
            setattr(cls, attr, '')


def print_header(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_section(text: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'-'*40}{Colors.END}")


def print_pass(text: str):
    print(f"  {Colors.GREEN}✓{Colors.END} {text}")


def print_fail(text: str):
    print(f"  {Colors.RED}✗{Colors.END} {text}")


def print_skip(text: str):
    print(f"  {Colors.YELLOW}○{Colors.END} {text}")


# =============================================================================
# Test Results
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    time_ms: float
    error: str = ""


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """Simple test runner"""
    
    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results: List[TestResult] = []
    
    def run_test(self, name: str, test_fn: Callable, is_slow: bool = False) -> bool:
        """Run a single test"""
        if is_slow and self.quick:
            print_skip(f"{name} (slow)")
            return True
        
        start = time.time()
        try:
            test_fn()
            elapsed = (time.time() - start) * 1000
            self.results.append(TestResult(name, True, elapsed))
            print_pass(f"{name} ({elapsed:.0f}ms)")
            return True
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            error = str(e)
            self.results.append(TestResult(name, False, elapsed, error))
            print_fail(f"{name}: {error}")
            if self.verbose:
                traceback.print_exc()
            return False
    
    def summary(self):
        """Print test summary"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.time_ms for r in self.results)
        
        print_header("TEST SUMMARY")
        
        print(f"  Total:  {len(self.results)}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
        if failed:
            print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
        print(f"  Time:   {total_time:.0f}ms")
        
        if failed:
            print(f"\n{Colors.RED}Failed Tests:{Colors.END}")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
            return False
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! ✓{Colors.END}")
            return True


# =============================================================================
# Test Definitions
# =============================================================================

def get_test_graph():
    """Get a test graph"""
    from src.core import generate_graph
    from src.simulation import SimulationGraph
    
    data = generate_graph(scale="small", scenario="iot", seed=42)
    return SimulationGraph.from_dict(data)


# Core Tests
def test_generate_graph():
    from src.core import generate_graph
    graph = generate_graph(scale="small", scenario="iot", seed=42)
    assert len(graph["applications"]) > 0

def test_all_scenarios():
    from src.core import generate_graph
    for scenario in ["iot", "financial", "healthcare", "smart_city"]:
        graph = generate_graph(scale="small", scenario=scenario, seed=42)
        assert len(graph["applications"]) > 0

# Simulation Tests
def test_simulation_graph():
    from src.simulation import SimulationGraph
    from src.core import generate_graph
    data = generate_graph(scale="small", scenario="iot", seed=42)
    graph = SimulationGraph.from_dict(data)
    assert len(graph.components) > 0

def test_failure_simulator():
    from src.simulation import FailureSimulator
    graph = get_test_graph()
    sim = FailureSimulator(seed=42)
    comp_id = list(graph.components.keys())[0]
    result = sim.simulate_failure(graph, comp_id)
    assert result.impact is not None

def test_failure_campaign():
    from src.simulation import FailureSimulator
    graph = get_test_graph()
    sim = FailureSimulator(seed=42)
    batch = sim.simulate_all_failures(graph)
    assert len(batch.results) > 0

def test_event_simulator():
    from src.simulation import EventSimulator
    graph = get_test_graph()
    sim = EventSimulator(seed=42)
    result = sim.simulate(graph, duration_ms=1000, message_rate=50)
    assert result.metrics.messages_published > 0

# Validation Tests
def test_spearman():
    from src.validation.metrics import spearman
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    rho, p = spearman(x, y)
    assert rho > 0.99

def test_graph_analyzer():
    from src.validation import GraphAnalyzer
    graph = get_test_graph()
    analyzer = GraphAnalyzer(graph)
    scores = analyzer.composite_score()
    assert len(scores) > 0

def test_validation_pipeline():
    from src.validation import ValidationPipeline
    graph = get_test_graph()
    pipeline = ValidationPipeline(seed=42)
    result = pipeline.run(graph)
    assert result.validation is not None

def test_compare_methods():
    from src.validation import ValidationPipeline
    graph = get_test_graph()
    pipeline = ValidationPipeline(seed=42)
    results = pipeline.compare_methods(graph)
    assert len(results) >= 3

# Analysis Tests
def test_criticality_level():
    from src.analysis import CriticalityLevel
    assert CriticalityLevel.CRITICAL.numeric > CriticalityLevel.HIGH.numeric
    assert CriticalityLevel.HIGH.numeric > CriticalityLevel.LOW.numeric

def test_box_plot_classifier():
    from src.analysis import BoxPlotClassifier
    classifier = BoxPlotClassifier(k_factor=1.5)
    scores = list(range(1, 101))
    stats = classifier.calculate_stats(scores)
    assert stats.count == 100
    assert stats.iqr == stats.q3 - stats.q1

def test_classify_items():
    from src.analysis import BoxPlotClassifier
    classifier = BoxPlotClassifier()
    items = [
        {"id": f"A{i}", "type": "App", "score": i * 0.1}
        for i in range(1, 11)
    ]
    result = classifier.classify(items, metric_name="test")
    assert len(result.items) == 10
    assert result.stats.count == 10

def test_classify_score():
    from src.analysis import BoxPlotClassifier, BoxPlotStats, CriticalityLevel
    classifier = BoxPlotClassifier()
    stats = BoxPlotStats(
        min_val=0, q1=25, median=50, q3=75, max_val=100,
        iqr=50, lower_fence=-50, upper_fence=150,
        mean=50, std_dev=25, count=100, k_factor=1.5,
    )
    level, is_outlier = classifier.classify_score(200, stats)
    assert level == CriticalityLevel.CRITICAL
    assert is_outlier is True

def test_antipattern_enums():
    from src.analysis import AntiPatternType, PatternSeverity
    assert AntiPatternType.GOD_TOPIC.value == "god_topic"
    assert PatternSeverity.CRITICAL.value == "critical"

# Visualization Tests
def test_graph_renderer():
    from src.visualization import GraphRenderer
    graph = get_test_graph()
    renderer = GraphRenderer()
    html = renderer.render(graph)
    assert "<!DOCTYPE html>" in html

def test_multi_layer():
    from src.visualization import GraphRenderer
    graph = get_test_graph()
    renderer = GraphRenderer()
    html = renderer.render_multi_layer(graph)
    assert len(html) > 1000

def test_dashboard():
    from src.visualization import DashboardGenerator
    graph = get_test_graph()
    generator = DashboardGenerator()
    html = generator.generate(graph)
    assert "<!DOCTYPE html>" in html

# Integration Tests
def test_full_pipeline():
    from src.validation import ValidationPipeline
    from src.visualization import DashboardGenerator
    
    graph = get_test_graph()
    pipeline = ValidationPipeline(seed=42)
    result = pipeline.run(graph)
    
    criticality = {k: {"score": v, "level": "high"} 
                   for k, v in result.predicted_scores.items()}
    
    generator = DashboardGenerator()
    html = generator.generate(graph, criticality=criticality, 
                              validation=result.validation.to_dict())
    assert len(html) > 10000


# =============================================================================
# Test Suite
# =============================================================================

TESTS = [
    # (name, function, is_slow)
    ("core.generate_graph", test_generate_graph, False),
    ("core.all_scenarios", test_all_scenarios, False),
    ("simulation.graph", test_simulation_graph, False),
    ("simulation.failure", test_failure_simulator, False),
    ("simulation.campaign", test_failure_campaign, True),
    ("simulation.events", test_event_simulator, False),
    ("validation.spearman", test_spearman, False),
    ("validation.analyzer", test_graph_analyzer, False),
    ("validation.pipeline", test_validation_pipeline, True),
    ("validation.compare", test_compare_methods, True),
    ("analysis.criticality_level", test_criticality_level, False),
    ("analysis.box_plot_classifier", test_box_plot_classifier, False),
    ("analysis.classify_items", test_classify_items, False),
    ("analysis.classify_score", test_classify_score, False),
    ("analysis.antipattern_enums", test_antipattern_enums, False),
    ("visualization.renderer", test_graph_renderer, False),
    ("visualization.multi_layer", test_multi_layer, False),
    ("visualization.dashboard", test_dashboard, False),
    ("integration.full", test_full_pipeline, True),
]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run test suite")
    parser.add_argument("--quick", "-q", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--module", "-m", type=str, help="Run specific module tests")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    
    args = parser.parse_args()
    
    if args.no_color:
        Colors.disable()
    
    print_header("Software-as-a-Graph Test Suite")
    
    runner = TestRunner(verbose=args.verbose, quick=args.quick)
    
    # Filter tests by module if specified
    tests = TESTS
    if args.module:
        tests = [(n, f, s) for n, f, s in TESTS if n.startswith(args.module)]
        if not tests:
            print(f"No tests found for module: {args.module}")
            return 1
    
    # Group tests by module
    modules = {}
    for name, fn, is_slow in tests:
        module = name.split(".")[0]
        if module not in modules:
            modules[module] = []
        modules[module].append((name, fn, is_slow))
    
    # Run tests
    for module, module_tests in modules.items():
        print_section(f"Testing: {module}")
        for name, fn, is_slow in module_tests:
            runner.run_test(name, fn, is_slow)
    
    # Summary
    success = runner.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
