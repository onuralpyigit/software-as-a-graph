#!/usr/bin/env python3
"""
Comprehensive Test Suite for Graph Analysis Scripts

This script tests all three enhanced scripts (analyze_graph.py, simulate_graph.py,
visualize_graph.py) with various scenarios to ensure they work correctly.

Features:
- Generates test data
- Tests all scripts with different options
- Validates outputs
- Reports results
- Cleanup options

Usage:
    # Run all tests
    python test_graph_analysis.py
    
    # Run specific test
    python test_graph_analysis.py --test analyze
    
    # Quick test (subset)
    python test_graph_analysis.py --quick
    
    # Verbose output
    python test_graph_analysis.py --verbose
    
    # Keep test data
    python test_graph_analysis.py --keep-data
"""

import argparse
import subprocess
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")


def print_test(text: str):
    """Print test description"""
    print(f"\n{Colors.OKBLUE}▶ {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_failure(text: str):
    """Print failure message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def generate_test_graph(size: str = "small") -> Dict:
    """
    Generate test graph data
    
    Args:
        size: Graph size (small, medium, large)
    
    Returns:
        Dictionary containing graph data
    """
    if size == "small":
        num_apps = 10
        num_topics = 5
        num_brokers = 2
        num_nodes = 3
    elif size == "medium":
        num_apps = 30
        num_topics = 15
        num_brokers = 3
        num_nodes = 5
    else:  # large
        num_apps = 50
        num_topics = 25
        num_brokers = 5
        num_nodes = 8
    
    graph = {
        "applications": [],
        "topics": [],
        "brokers": [],
        "nodes": [],
        "publishes": [],
        "subscribes": [],
        "broker_routes": {}
    }
    
    # Generate nodes
    for i in range(num_nodes):
        graph["nodes"].append({
            "id": f"N{i+1}",
            "name": f"Node{i+1}",
            "ip_address": f"192.168.1.{i+1}",
            "zone": f"Zone{(i % 2) + 1}",
            "cpu_cores": 4,
            "memory_gb": 16
        })
    
    # Generate brokers
    for i in range(num_brokers):
        graph["brokers"].append({
            "id": f"B{i+1}",
            "name": f"Broker{i+1}",
            "node_id": f"N{(i % num_nodes) + 1}",
            "port": 7400 + i,
            "max_connections": 1000
        })
    
    # Generate topics
    for i in range(num_topics):
        graph["topics"].append({
            "id": f"T{i+1}",
            "name": f"Topic{i+1}",
            "type": ["sensor_data", "commands", "status", "telemetry"][i % 4],
            "qos_policy": {
                "durability": "TRANSIENT_LOCAL" if i % 2 == 0 else "VOLATILE",
                "reliability": "RELIABLE" if i % 3 != 0 else "BEST_EFFORT",
                "deadline_ms": 1000 if i % 2 == 0 else None
            }
        })
    
    # Generate applications
    for i in range(num_apps):
        graph["applications"].append({
            "id": f"A{i+1}",
            "name": f"Application{i+1}",
            "node_id": f"N{(i % num_nodes) + 1}",
            "broker_id": f"B{(i % num_brokers) + 1}",
            "criticality": ["LOW", "MEDIUM", "HIGH", "CRITICAL"][i % 4],
            "replicas": 1 if i % 5 == 0 else 0
        })
    
    # Generate publishes relationships
    for i in range(num_apps):
        # Each app publishes to 1-2 topics
        num_pubs = 1 if i % 2 == 0 else 2
        for j in range(num_pubs):
            topic_idx = (i + j) % num_topics
            graph["publishes"].append({
                "application_id": f"A{i+1}",
                "topic_id": f"T{topic_idx+1}",
                "rate_hz": 10.0
            })
    
    # Generate subscribes relationships
    for i in range(num_apps):
        # Each app subscribes to 1-3 topics
        num_subs = (i % 3) + 1
        for j in range(num_subs):
            topic_idx = (i + j + 2) % num_topics
            graph["subscribes"].append({
                "application_id": f"A{i+1}",
                "topic_id": f"T{topic_idx+1}"
            })
    
    # Generate broker routes
    for i, broker in enumerate(graph["brokers"]):
        broker_id = broker["id"]
        # Each broker routes subset of topics
        topics_per_broker = num_topics // num_brokers + 1
        start_idx = i * topics_per_broker
        end_idx = min(start_idx + topics_per_broker, num_topics)
        
        graph["broker_routes"][broker_id] = [
            f"T{j+1}" for j in range(start_idx, end_idx)
        ]
    
    return graph


class TestRunner:
    """Test runner for enhanced scripts"""
    
    def __init__(self, verbose: bool = False, keep_data: bool = False):
        """
        Initialize test runner
        
        Args:
            verbose: Enable verbose output
            keep_data: Keep test data after completion
        """
        self.verbose = verbose
        self.keep_data = keep_data
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_graph_analysis_"))
        self.results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "tests": []
        }
        
        print(f"Test directory: {self.test_dir}")
    
    def cleanup(self):
        """Clean up test directory"""
        if not self.keep_data and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"\nCleaned up test directory: {self.test_dir}")
        elif self.keep_data:
            print(f"\nTest data kept in: {self.test_dir}")
    
    def run_command(self, cmd: List[str], test_name: str) -> Tuple[bool, str, str]:
        """
        Run command and capture output
        
        Args:
            cmd: Command list
            test_name: Test name for reporting
        
        Returns:
            Tuple of (success, stdout, stderr)
        """
        if self.verbose:
            print(f"  Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            success = result.returncode == 0
            
            if self.verbose or not success:
                if result.stdout:
                    print(f"  STDOUT: {result.stdout[:500]}")
                if result.stderr:
                    print(f"  STDERR: {result.stderr[:500]}")
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check_file_exists(self, filepath: Path, description: str) -> bool:
        """
        Check if file exists and log result
        
        Args:
            filepath: Path to check
            description: File description
        
        Returns:
            True if file exists
        """
        exists = filepath.exists()
        if exists:
            size = filepath.stat().st_size
            if self.verbose:
                print(f"  ✓ {description}: {filepath.name} ({size} bytes)")
        else:
            print_failure(f"{description} not found: {filepath.name}")
        return exists
    
    def record_test(self, name: str, passed: bool, message: str = ""):
        """Record test result"""
        self.results["tests"].append({
            "name": name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if passed:
            self.results["passed"] += 1
            print_success(f"{name}: PASSED")
        else:
            self.results["failed"] += 1
            print_failure(f"{name}: FAILED - {message}")
    
    def test_analyze_basic(self) -> bool:
        """Test basic analyze_graph.py functionality"""
        print_test("Test: Basic analysis with JSON export")
        
        # Generate test data
        graph_data = generate_test_graph("small")
        input_file = self.test_dir / "test_graph.json"
        output_dir = self.test_dir / "analyze_basic"
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run analysis
        cmd = [
            "python", "../analyze_graph.py",
            "--input", str(input_file),
            "--output", str(output_dir),
            "--export-json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "analyze_basic")
        
        if not success:
            self.record_test("analyze_basic", False, "Command failed")
            return False
        
        # Check outputs
        json_file = output_dir / "analysis_results.json"
        if not self.check_file_exists(json_file, "JSON output"):
            self.record_test("analyze_basic", False, "JSON output missing")
            return False
        
        # Validate JSON content
        try:
            with open(json_file) as f:
                results = json.load(f)
            
            required_keys = ["graph_summary", "criticality_scores"]
            missing_keys = [k for k in required_keys if k not in results]
            
            if missing_keys:
                self.record_test("analyze_basic", False, f"Missing keys: {missing_keys}")
                return False
            
        except Exception as e:
            self.record_test("analyze_basic", False, f"Invalid JSON: {e}")
            return False
        
        self.record_test("analyze_basic", True)
        return True
    
    def test_analyze_all_formats(self) -> bool:
        """Test analysis with all export formats"""
        print_test("Test: Analysis with all export formats")
        
        # Generate test data
        graph_data = generate_test_graph("small")
        input_file = self.test_dir / "test_graph2.json"
        output_dir = self.test_dir / "analyze_all_formats"
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run analysis
        cmd = [
            "python", "../analyze_graph.py",
            "--input", str(input_file),
            "--output", str(output_dir),
            "--export-json", "--export-csv", "--export-html", "--export-md"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "analyze_all_formats")
        
        if not success:
            self.record_test("analyze_all_formats", False, "Command failed")
            return False
        
        # Check all output files
        files_to_check = [
            (output_dir / "analysis_results.json", "JSON"),
            (output_dir / "criticality_scores.csv", "CSV"),
            (output_dir / "analysis_report.html", "HTML"),
            (output_dir / "analysis_report.md", "Markdown")
        ]
        
        all_exist = True
        for filepath, description in files_to_check:
            if not self.check_file_exists(filepath, description):
                all_exist = False
        
        if not all_exist:
            self.record_test("analyze_all_formats", False, "Some outputs missing")
            return False
        
        self.record_test("analyze_all_formats", True)
        return True
    
    def test_analyze_with_simulation(self) -> bool:
        """Test analysis with failure simulation"""
        print_test("Test: Analysis with failure simulation")
        
        # Generate test data
        graph_data = generate_test_graph("medium")
        input_file = self.test_dir / "test_graph3.json"
        output_dir = self.test_dir / "analyze_simulation"
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run analysis with simulation
        cmd = [
            "python", "../analyze_graph.py",
            "--input", str(input_file),
            "--output", str(output_dir),
            "--simulate",
            "--export-json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "analyze_with_simulation")
        
        if not success:
            self.record_test("analyze_with_simulation", False, "Command failed")
            return False
        
        # Check output and verify simulation results
        json_file = output_dir / "analysis_results.json"
        if not self.check_file_exists(json_file, "JSON output"):
            self.record_test("analyze_with_simulation", False, "JSON output missing")
            return False
        
        try:
            with open(json_file) as f:
                results = json.load(f)
            
            if "failure_simulation" not in results:
                self.record_test("analyze_with_simulation", False, "No simulation results")
                return False
            
        except Exception as e:
            self.record_test("analyze_with_simulation", False, f"Invalid JSON: {e}")
            return False
        
        self.record_test("analyze_with_simulation", True)
        return True
    
    def test_simulate_basic(self) -> bool:
        """Test basic simulate_graph.py functionality"""
        print_test("Test: Basic simulation")
        
        # Generate test data
        graph_data = generate_test_graph("small")
        input_file = self.test_dir / "test_graph4.json"
        output_dir = self.test_dir / "simulate_basic"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run simulation
        cmd = [
            "python", "../simulate_graph.py",
            "--input", str(input_file),
            "--duration", "10",
            "--output", str(output_dir),
            "--export-json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "simulate_basic")
        
        if not success:
            self.record_test("simulate_basic", False, "Command failed")
            return False
        
        # Check output
        json_files = list(output_dir.glob("*_results.json"))
        if not json_files:
            self.record_test("simulate_basic", False, "No JSON output found")
            return False
        
        self.record_test("simulate_basic", True)
        return True
    
    def test_simulate_scenario(self) -> bool:
        """Test simulation with predefined scenario"""
        print_test("Test: Simulation with predefined scenario")
        
        # Generate test data
        graph_data = generate_test_graph("medium")
        input_file = self.test_dir / "test_graph5.json"
        output_dir = self.test_dir / "simulate_scenario"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run simulation with scenario
        cmd = [
            "python", "../simulate_graph.py",
            "--input", str(input_file),
            "--scenario", "single-app",
            "--output", str(output_dir),
            "--export-json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "simulate_scenario")
        
        if not success:
            self.record_test("simulate_scenario", False, "Command failed")
            return False
        
        # Check output
        json_files = list(output_dir.glob("*_results.json"))
        if not json_files:
            self.record_test("simulate_scenario", False, "No JSON output found")
            return False
        
        self.record_test("simulate_scenario", True)
        return True
    
    def test_visualize_basic(self) -> bool:
        """Test basic visualize_graph.py functionality"""
        print_test("Test: Basic visualization")
        
        # Generate test data
        graph_data = generate_test_graph("small")
        input_file = self.test_dir / "test_graph6.json"
        output_dir = self.test_dir / "visualize_basic"
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run visualization
        cmd = [
            "python", "../visualize_graph.py",
            "--input", str(input_file),
            "--output-dir", str(output_dir),
            "--complete",
            "--export-html"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "visualize_basic")
        
        if not success:
            self.record_test("visualize_basic", False, "Command failed")
            return False
        
        # Check output
        html_file = output_dir / "complete_system.html"
        if not self.check_file_exists(html_file, "HTML output"):
            self.record_test("visualize_basic", False, "HTML output missing")
            return False
        
        self.record_test("visualize_basic", True)
        return True
    
    def test_visualize_dashboard(self) -> bool:
        """Test dashboard visualization"""
        print_test("Test: Dashboard visualization")
        
        # Generate test data
        graph_data = generate_test_graph("medium")
        input_file = self.test_dir / "test_graph7.json"
        output_dir = self.test_dir / "visualize_dashboard"
        
        with open(input_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Run visualization
        cmd = [
            "python", "../visualize_graph.py",
            "--input", str(input_file),
            "--output-dir", str(output_dir),
            "--dashboard",
            "--run-analysis"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "visualize_dashboard")
        
        if not success:
            self.record_test("visualize_dashboard", False, "Command failed")
            return False
        
        # Check output
        dashboard_file = output_dir / "dashboard.html"
        if not self.check_file_exists(dashboard_file, "Dashboard HTML"):
            self.record_test("visualize_dashboard", False, "Dashboard missing")
            return False
        
        self.record_test("visualize_dashboard", True)
        return True
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid input"""
        print_test("Test: Error handling with invalid input")
        
        # Test with non-existent file
        cmd = [
            "python", "../analyze_graph.py",
            "--input", "/nonexistent/file.json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, "error_handling")
        
        # Should fail gracefully
        if success:
            self.record_test("error_handling", False, "Should have failed with invalid input")
            return False
        
        # Check for error message
        if "not found" not in stderr.lower() and "not found" not in stdout.lower():
            self.record_test("error_handling", False, "No proper error message")
            return False
        
        self.record_test("error_handling", True)
        return True
    
    def run_all_tests(self, quick: bool = False):
        """
        Run all tests
        
        Args:
            quick: Run only quick subset of tests
        """
        print_header("ENHANCED SCRIPTS TEST SUITE")
        
        start_time = time.time()
        
        # Test suite
        tests = [
            ("analyze_basic", self.test_analyze_basic, False),
            ("analyze_all_formats", self.test_analyze_all_formats, False),
            ("analyze_with_simulation", self.test_analyze_with_simulation, True),  # Slow
            ("simulate_basic", self.test_simulate_basic, False),
            ("simulate_scenario", self.test_simulate_scenario, True),  # Slow
            ("visualize_basic", self.test_visualize_basic, False),
            ("visualize_dashboard", self.test_visualize_dashboard, True),  # Slow
            ("error_handling", self.test_error_handling, False),
        ]
        
        # Filter tests if quick mode
        if quick:
            tests = [(name, func, is_slow) for name, func, is_slow in tests if not is_slow]
            print_warning("Running quick test suite (skipping slow tests)")
        
        # Run tests
        for name, test_func, is_slow in tests:
            try:
                test_func()
            except Exception as e:
                self.record_test(name, False, f"Exception: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Print summary
        elapsed = time.time() - start_time
        
        print_header("TEST SUMMARY")
        print(f"\nTotal Tests: {self.results['passed'] + self.results['failed']}")
        print_success(f"Passed: {self.results['passed']}")
        if self.results['failed'] > 0:
            print_failure(f"Failed: {self.results['failed']}")
        if self.results['skipped'] > 0:
            print_warning(f"Skipped: {self.results['skipped']}")
        
        print(f"\nExecution Time: {elapsed:.2f}s")
        
        # Detailed results
        if self.results['failed'] > 0:
            print(f"\n{Colors.FAIL}Failed Tests:{Colors.ENDC}")
            for test in self.results['tests']:
                if not test['passed']:
                    print(f"  ✗ {test['name']}: {test['message']}")
        
        success_rate = (self.results['passed'] / (self.results['passed'] + self.results['failed']) * 100 
                       if self.results['passed'] + self.results['failed'] > 0 else 0)
        
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Overall result
        if self.results['failed'] == 0:
            print_success("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print_failure(f"\n❌ {self.results['failed']} TEST(S) FAILED")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test suite for enhanced graph analysis scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--test', type=str,
                       choices=['analyze', 'simulate', 'visualize', 'all'],
                       default='all',
                       help='Which tests to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test suite (skip slow tests)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--keep-data', action='store_true',
                       help='Keep test data after completion')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose, keep_data=args.keep_data)
    
    try:
        # Run tests
        success = runner.run_all_tests(quick=args.quick)
        
        # Cleanup
        runner.cleanup()
        
        # Exit with appropriate code
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print_warning("\n\nTests interrupted by user")
        runner.cleanup()
        return 130
    except Exception as e:
        print_failure(f"\nTest suite error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        runner.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())
