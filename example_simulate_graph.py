#!/usr/bin/env python3
"""
Example: Using simulate_graph.py for Comprehensive System Simulation

This example demonstrates various simulation scenarios:
1. Basic traffic simulation
2. Single component failure
3. Cascading failures
4. Baseline comparison
5. Real-time monitoring
6. Comprehensive reporting
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print description"""
    print("\n" + "=" * 70)
    print(f"EXAMPLE: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
    else:
        print(f"✓ Command completed successfully")
    
    return result.returncode


def main():
    """Run example simulations"""
    
    # Check if test graph exists
    test_graph = Path("examples/test_system.json")
    
    if not test_graph.exists():
        print("Creating test system for examples...")
        
        # Create a simple test system
        import json
        import random
        random.seed(42)
        
        test_system = {
            "metadata": {"scenario": "Example Test System"},
            "nodes": [
                {"id": f"N{i}", "name": f"Node{i}",
                 "cpu_capacity": 16.0, "memory_gb": 32.0}
                for i in range(1, 6)
            ],
            "applications": [
                {"id": f"A{i}", "name": f"App{i}", 
                 "type": random.choice(["PRODUCER", "CONSUMER", "PROSUMER"])}
                for i in range(1, 21)
            ],
            "topics": [
                {"id": f"T{i}", "name": f"topic/data{i}",
                 "qos": {"history_depth": 10, "deadline_ms": 100}}
                for i in range(1, 11)
            ],
            "brokers": [
                {"id": f"B{i}", "name": f"Broker{i}"}
                for i in range(1, 3)
            ],
            "relationships": {
                "publishes_to": [
                    {"from": f"A{i}", "to": f"T{random.randint(1, 10)}",
                     "period_ms": random.choice([100, 200, 500]),
                     "msg_size": 1024}
                    for i in range(1, 21)
                ],
                "subscribes_to": [
                    {"from": f"A{i}", "to": f"T{j}"}
                    for i in range(1, 21)
                    for j in random.sample(range(1, 11), 2)
                ],
                "runs_on": [
                    {"from": f"A{i}", "to": f"N{((i-1) % 5) + 1}"}
                    for i in range(1, 21)
                ],
                "routes": [
                    {"from": f"B{((i-1) % 2) + 1}", "to": f"T{i}"}
                    for i in range(1, 11)
                ]
            }
        }
        
        test_graph.parent.mkdir(parents=True, exist_ok=True)
        with open(test_graph, 'w') as f:
            json.dump(test_system, f, indent=2)
        
        print(f"✓ Test system created: {test_graph}")
    
    examples = [
        {
            'description': '1. Basic Traffic Simulation (60s)',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '60']
        },
        {
            'description': '2. Single Component Failure',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '60',
                   '--fail-component', 'A1',
                   '--fail-type', 'complete',
                   '--fail-time', '30']
        },
        {
            'description': '3. Multiple Failures with Cascade',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '60',
                   '--fail-component', 'A1,B1',
                   '--enable-cascade']
        },
        {
            'description': '4. Broker Failure with Baseline',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '60',
                   '--baseline',
                   '--fail-component', 'B1',
                   '--fail-time', '30']
        },
        {
            'description': '5. Cascading Broker Scenario',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '90',
                   '--scenario', 'cascading-broker']
        },
        {
            'description': '6. Stress Test with Monitoring',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '120',
                   '--scenario', 'stress-test',
                   '--monitor',
                   '--monitor-interval', '10']
        },
        {
            'description': '7. Comprehensive Report (JSON + HTML)',
            'cmd': ['python', 'simulate_graph.py',
                   '--input', str(test_graph),
                   '--duration', '60',
                   '--scenario', 'multiple-simultaneous',
                   '--export-json',
                   '--export-html',
                   '--output', 'output/simulation_reports/']
        }
    ]
    
    print("\n" + "=" * 70)
    print("SIMULATE_GRAPH.PY - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print(f"\nThis script demonstrates {len(examples)} simulation scenarios.")
    print("Each example showcases different features and use cases.\n")
    
    # Ask user which examples to run
    print("Available examples:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['description']}")
    
    print(f"  A. Run all examples")
    print(f"  Q. Quit")
    
    choice = input("\nSelect example to run (1-{}, A, or Q): ".format(len(examples))).strip().upper()
    
    if choice == 'Q':
        print("Exiting...")
        return 0
    
    if choice == 'A':
        # Run all examples
        failed = 0
        for i, ex in enumerate(examples, 1):
            result = run_command(ex['cmd'], ex['description'])
            if result != 0:
                failed += 1
            
            # Wait for user to review
            if i < len(examples):
                input("\nPress Enter to continue to next example...")
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETE")
        print("=" * 70)
        print(f"Total: {len(examples)}, Successful: {len(examples) - failed}, Failed: {failed}")
        
        return 0 if failed == 0 else 1
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            ex = examples[idx]
            result = run_command(ex['cmd'], ex['description'])
            return result
        else:
            print(f"Invalid choice: {choice}")
            return 1
    except ValueError:
        print(f"Invalid choice: {choice}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
