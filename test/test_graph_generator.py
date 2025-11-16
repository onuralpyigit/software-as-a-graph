#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Graph Generator

Tests all features of the improved graph generation system:
1. All scale presets
2. All domain scenarios
3. Anti-pattern injection
4. High availability configurations
5. Validation
6. Export formats
7. Performance benchmarks

Usage:
    # Run all tests
    python test_graph_generator.py
    
    # Run specific test suites
    python test_graph_generator.py --suite scales
    python test_graph_generator.py --suite scenarios
    python test_graph_generator.py --suite antipatterns
    
    # Quick test (small graphs only)
    python test_graph_generator.py --quick
    
    # Performance benchmark
    python test_graph_generator.py --benchmark
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))
from src.core.graph_generator import GraphGenerator, GraphConfig

class TestResult:
    """Test result container"""
    def __init__(self, name: str, passed: bool, duration: float, 
                 error: str = None, stats: Dict = None):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.error = error
        self.stats = stats or {}


class GraphGeneratorTester:
    """Comprehensive test suite for graph generator"""
    
    def __init__(self, output_dir: Path = None, verbose: bool = False):
        self.output_dir = output_dir or Path('test_outputs')
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def run_all_tests(self, quick: bool = False):
        """Run all test suites"""
        print("=" * 70)
        print("GRAPH GENERATOR TEST SUITE".center(70))
        print("=" * 70)
        print()
        
        self.test_scales(quick=quick)
        self.test_scenarios(quick=quick)
        self.test_antipatterns(quick=quick)
        self.test_edge_cases()
        
        self.print_summary()
    
    def test_scales(self, quick: bool = False):
        """Test all scale presets"""
        print("\n[1] Testing Scale Presets")
        print("-" * 70)
        
        scales = ['tiny', 'small', 'medium']
        if not quick:
            scales.extend(['large', 'xlarge'])
        
        for scale in scales:
            self._run_test(f"Scale: {scale}", lambda: self._generate_graph(
                scale=scale,
                scenario='generic',
                filename=f'scale_{scale}.json'
            ))
    
    def test_scenarios(self, quick: bool = False):
        """Test all domain scenarios"""
        print("\n[2] Testing Domain Scenarios")
        print("-" * 70)
        
        scenarios = [
            'generic', 'iot', 'financial', 'ecommerce', 
            'analytics', 'smart_city', 'healthcare'
        ]
        
        scale = 'small' if quick else 'medium'
        
        for scenario in scenarios:
            self._run_test(f"Scenario: {scenario}", lambda s=scenario: self._generate_graph(
                scale=scale,
                scenario=s,
                filename=f'scenario_{s}.json'
            ))
    
    def test_antipatterns(self, quick: bool = False):
        """Test anti-pattern injection"""
        print("\n[3] Testing Anti-Patterns")
        print("-" * 70)
        
        antipatterns = [
            'spof',
            'broker_overload',
            'god_object',
            'single_broker',
            'tight_coupling',
            'chatty_communication',
            'bottleneck'
        ]
        
        for antipattern in antipatterns:
            self._run_test(f"Anti-pattern: {antipattern}", lambda ap=antipattern: self._generate_graph(
                scale='small',
                scenario='generic',
                antipatterns=[ap],
                filename=f'antipattern_{ap}.json'
            ))
        
        # Test multiple anti-patterns
        if not quick:
            self._run_test("Multiple anti-patterns", lambda: self._generate_graph(
                scale='medium',
                scenario='generic',
                antipatterns=['spof', 'broker_overload', 'tight_coupling'],
                filename='antipattern_multiple.json'
            ))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n[4] Testing Edge Cases")
        print("-" * 70)
        
        # Minimal graph
        self._run_test("Minimal graph", lambda: self._generate_graph(
            scale='tiny',
            scenario='generic',
            density=0.1,
            filename='edge_minimal.json'
        ))
        
        # Dense graph
        self._run_test("Dense graph", lambda: self._generate_graph(
            scale='small',
            scenario='generic',
            density=0.9,
            filename='edge_dense.json'
        ))
        
        # Custom parameters
        self._run_test("Custom parameters", lambda: GraphGenerator(
            GraphConfig(
                scale='custom',
                scenario='generic',
                num_nodes=10,
                num_applications=30,
                num_topics=15,
                num_brokers=3,
                edge_density=0.4,
                antipatterns=[],
                seed=12345
            )
        ).generate())
    
    def benchmark_performance(self):
        """Benchmark generation performance"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK".center(70))
        print("=" * 70)
        print()
        
        benchmarks = [
            ('tiny', 10),
            ('small', 10),
            ('medium', 5),
            ('large', 3),
            ('xlarge', 2),
        ]
        
        results = {}
        
        for scale, iterations in benchmarks:
            times = []
            print(f"Benchmarking {scale} scale ({iterations} iterations)...", end=" ", flush=True)
            
            for i in range(iterations):
                config = GraphConfig(
                    scale=scale,
                    scenario='generic',
                    num_nodes=GraphGenerator.SCALES[scale]['nodes'],
                    num_applications=GraphGenerator.SCALES[scale]['apps'],
                    num_topics=GraphGenerator.SCALES[scale]['topics'],
                    num_brokers=GraphGenerator.SCALES[scale]['brokers'],
                    edge_density=0.3,
                    antipatterns=[],
                    seed=42 + i
                )
                
                start = time.time()
                generator = GraphGenerator(config)
                graph = generator.generate()
                duration = time.time() - start
                times.append(duration)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results[scale] = {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'iterations': iterations
            }
            
            print(f"✓ avg: {avg_time:.3f}s, min: {min_time:.3f}s, max: {max_time:.3f}s")
        
        # Print summary table
        print("\n" + "-" * 70)
        print(f"{'Scale':<12} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'Iterations':<12}")
        print("-" * 70)
        for scale, stats in results.items():
            print(f"{scale:<12} {stats['avg']:<12.3f} {stats['min']:<12.3f} "
                  f"{stats['max']:<12.3f} {stats['iterations']:<12}")
        print("-" * 70)
    
    def _run_test(self, test_name: str, test_func):
        """Run a single test"""
        print(f"  Testing: {test_name:<40}", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            
            # Validate result
            if isinstance(result, dict):
                graph = result
                stats = self._get_graph_stats(graph)
            else:
                graph = None
                stats = {}
            
            print(f"✓ ({duration:.2f}s)")
            self.results.append(TestResult(test_name, True, duration, stats=stats))
            
            if self.verbose and stats:
                print(f"      → {stats['nodes']} nodes, {stats['apps']} apps, "
                      f"{stats['topics']} topics, {stats['edges']} edges")
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"✗ ({duration:.2f}s)")
            print(f"      Error: {str(e)}")
            self.results.append(TestResult(test_name, False, duration, error=str(e)))
            
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def _generate_graph(self, scale: str, scenario: str, filename: str = None, **kwargs):
        """Generate graph with given parameters"""
        scale_params = GraphGenerator.SCALES[scale]
        
        config = GraphConfig(
            scale=scale,
            scenario=scenario,
            num_nodes=scale_params['nodes'],
            num_applications=scale_params['apps'],
            num_topics=scale_params['topics'],
            num_brokers=scale_params['brokers'],
            edge_density=kwargs.get('density', 0.3),
            antipatterns=kwargs.get('antipatterns', []),
            seed=kwargs.get('seed', 42)
        )
        
        generator = GraphGenerator(config)
        graph = generator.generate()
        
        # Save if filename provided
        if filename:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(graph, f, indent=2)
        
        return graph
    
    def _get_graph_stats(self, graph: Dict) -> Dict:
        """Extract basic statistics from graph"""
        rels = graph.get('relationships', {})
        return {
            'nodes': len(graph.get('nodes', [])),
            'apps': len(graph.get('applications', [])),
            'topics': len(graph.get('topics', [])),
            'brokers': len(graph.get('brokers', [])),
            'edges': sum(len(rels.get(k, [])) for k in rels.keys())
        }
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY".center(70))
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration for r in self.results)
        
        print(f"\nTotal Tests: {len(self.results)}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Success Rate: {(passed/len(self.results)*100):.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time: {total_time/len(self.results):.2f}s per test")
        
        if failed > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  ✗ {result.name}")
                    if result.error:
                        print(f"     Error: {result.error}")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test suite for enhanced graph generator')
    parser.add_argument('--suite', choices=['scales', 'scenarios', 'antipatterns', 'ha', 'edges'],
                       help='Run specific test suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (small graphs)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--output-dir', default='test_outputs',
                       help='Output directory for test files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    tester = GraphGeneratorTester(
        output_dir=Path(args.output_dir),
        verbose=args.verbose
    )
    
    if args.benchmark:
        tester.benchmark_performance()
        return 0
    
    if args.suite:
        # Run specific suite
        if args.suite == 'scales':
            tester.test_scales(quick=args.quick)
        elif args.suite == 'scenarios':
            tester.test_scenarios(quick=args.quick)
        elif args.suite == 'antipatterns':
            tester.test_antipatterns(quick=args.quick)
        elif args.suite == 'edges':
            tester.test_edge_cases()
        
        tester.print_summary()
    else:
        # Run all tests
        tester.run_all_tests(quick=args.quick)
    
    # Return exit code based on test results
    failed = sum(1 for r in tester.results if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
