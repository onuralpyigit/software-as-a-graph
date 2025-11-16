#!/usr/bin/env python3
"""
Minimal Verification Script

Quick script to verify the simulation system is working correctly.
This creates a simple test case and runs a basic simulation.

Run with: python verify_installation.py
"""

import sys
import json
import asyncio
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status message"""
    colors = {'info': BLUE, 'success': GREEN, 'error': RED, 'warning': YELLOW}
    color = colors.get(status, RESET)
    symbol = {'info': 'ℹ', 'success': '✓', 'error': '✗', 'warning': '⚠'}
    print(f"{color}{symbol.get(status, '•')} {message}{RESET}")


def check_imports():
    """Verify all required imports work"""
    print_status("Checking imports...", 'info')
    
    missing = []
    
    try:
        import networkx
        print_status("  networkx: OK", 'success')
    except ImportError:
        print_status("  networkx: MISSING", 'error')
        missing.append('networkx')
    
    try:
        import asyncio
        print_status("  asyncio: OK", 'success')
    except ImportError:
        print_status("  asyncio: MISSING", 'error')
        missing.append('asyncio')
    
    # Check src modules
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
        print_status("  LightweightDDSSimulator: OK", 'success')
    except ImportError as e:
        print_status(f"  LightweightDDSSimulator: MISSING ({e})", 'error')
        missing.append('LightweightDDSSimulator')
    
    try:
        from src.simulation.enhanced_failure_simulator import FailureSimulator
        print_status("  FailureSimulator: OK", 'success')
    except ImportError as e:
        print_status(f"  FailureSimulator: MISSING ({e})", 'error')
        missing.append('FailureSimulator')
    
    if missing:
        print_status(f"\nMissing dependencies: {', '.join(missing)}", 'error')
        print_status("\nInstall missing packages:", 'info')
        if 'networkx' in missing or 'asyncio' in missing:
            print("  pip install networkx")
        if 'LightweightDDSSimulator' in missing or 'FailureSimulator' in missing:
            print("  Ensure src/simulation/ directory contains all required files")
        return False
    
    return True


def create_minimal_graph():
    """Create minimal test graph"""
    return {
        'nodes': [
            {'id': 'N1', 'properties': {'name': 'TestNode'}}
        ],
        'applications': [
            {
                'id': 'Publisher',
                'node': 'N1',
                'properties': {
                    'name': 'TestPublisher',
                    'publish_topics': [['TestTopic', 2000, 256]],
                    'subscribe_topics': []
                }
            },
            {
                'id': 'Subscriber',
                'node': 'N1',
                'properties': {
                    'name': 'TestSubscriber',
                    'publish_topics': [],
                    'subscribe_topics': ['TestTopic']
                }
            }
        ],
        'brokers': [
            {
                'id': 'Broker1',
                'node': 'N1',
                'properties': {
                    'name': 'TestBroker',
                    'port': 7400,
                    'routing_delay_ms': 1.0
                }
            }
        ],
        'topics': [
            {
                'id': 'TestTopic',
                'broker': 'Broker1',
                'properties': {
                    'name': 'TestTopic',
                    'type': 'std_msgs/String',
                    'reliability': 'RELIABLE',
                    'deadline_ms': 10000,
                    'lifespan_ms': 30000
                }
            }
        ]
    }


async def run_basic_simulation():
    """Run a minimal simulation to verify functionality"""
    print_status("\nRunning basic simulation test...", 'info')
    
    try:
        from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
        
        # Create test graph
        graph_data = create_minimal_graph()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        print_status(f"  Created test graph: {temp_path}", 'info')
        
        # Create and load simulator
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        print_status("  Loaded graph into simulator", 'success')
        print_status(f"  Components: {len(simulator.nodes)} nodes, "
                    f"{len(simulator.applications)} apps, "
                    f"{len(simulator.brokers)} brokers, "
                    f"{len(simulator.topics)} topics", 'info')
        
        # Run short simulation
        print_status("  Running 5-second simulation...", 'info')
        results = await simulator.run_simulation(duration_seconds=5)
        
        # Verify results
        stats = results['global_stats']
        
        assert stats['messages_sent'] > 0, "No messages sent"
        assert stats['messages_delivered'] > 0, "No messages delivered"
        assert 0 <= stats['delivery_rate'] <= 1, "Invalid delivery rate"
        
        print_status("  Simulation completed successfully", 'success')
        print_status(f"  Messages: {stats['messages_sent']} sent, "
                    f"{stats['messages_delivered']} delivered "
                    f"({stats['delivery_rate']:.1%} rate)", 'success')
        print_status(f"  Latency: {stats['avg_latency_ms']:.2f}ms average", 'success')
        
        # Cleanup
        Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print_status(f"  Simulation failed: {e}", 'error')
        import traceback
        traceback.print_exc()
        return False


async def test_failure_injection():
    """Test failure injection capability"""
    print_status("\nTesting failure injection...", 'info')
    
    try:
        from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
        from src.simulation.enhanced_failure_simulator import (
            FailureSimulator, FailureType, ComponentType
        )
        
        # Create test graph
        graph_data = create_minimal_graph()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        # Setup
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        failure_sim = FailureSimulator()
        
        # Run simulation with failure
        sim_task = asyncio.create_task(simulator.run_simulation(duration_seconds=5))
        
        # Inject failure after 2 seconds
        await asyncio.sleep(2)
        print_status("  Injecting failure: Publisher", 'info')
        failure_sim.inject_failure(
            simulator,
            'Publisher',
            ComponentType.APPLICATION,
            FailureType.COMPLETE,
            severity=1.0,
            enable_cascade=False
        )
        
        # Wait for completion
        results = await sim_task
        
        # Verify failure was recorded
        assert len(failure_sim.failure_events) == 1, "Failure not recorded"
        assert 'Publisher' in failure_sim.active_failures, "Failure not active"
        
        print_status("  Failure injection successful", 'success')
        print_status(f"  Failure events: {len(failure_sim.failure_events)}", 'success')
        print_status(f"  Messages dropped: {results['global_stats']['messages_dropped']}", 'success')
        
        # Cleanup
        Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print_status(f"  Failure injection test failed: {e}", 'error')
        import traceback
        traceback.print_exc()
        return False


def print_summary(import_ok, sim_ok, fail_ok):
    """Print verification summary"""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    results = {
        'Import Check': import_ok,
        'Basic Simulation': sim_ok,
        'Failure Injection': fail_ok
    }
    
    for test, passed in results.items():
        status = 'success' if passed else 'error'
        print_status(f"{test}: {'PASS' if passed else 'FAIL'}", status)
    
    all_passed = all(results.values())
    
    print("="*60)
    
    if all_passed:
        print_status("\n✓ All tests passed! System is ready to use.", 'success')
        print_status("\nNext steps:", 'info')
        print("  1. Try quick examples: python quick_examples.py")
        print("  2. Run full tests: python test_simulation.py")
        print("  3. Read README.md for detailed usage")
    else:
        print_status("\n✗ Some tests failed. Please check errors above.", 'error')
        print_status("\nTroubleshooting:", 'info')
        print("  1. Ensure all dependencies installed: pip install networkx")
        print("  2. Check src/simulation/ directory exists with required files")
        print("  3. See README.md for more help")
    
    return 0 if all_passed else 1


async def main():
    """Main verification routine"""
    print("\n" + "="*60)
    print("SIMULATION SYSTEM - INSTALLATION VERIFICATION")
    print("="*60 + "\n")
    
    # Check imports
    import_ok = check_imports()
    
    if not import_ok:
        print_status("\nCannot proceed without required dependencies", 'error')
        return print_summary(import_ok, False, False)
    
    # Test basic simulation
    sim_ok = await run_basic_simulation()
    
    # Test failure injection
    fail_ok = await test_failure_injection()
    
    # Print summary
    return print_summary(import_ok, sim_ok, fail_ok)


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_status("\n\nVerification cancelled by user", 'warning')
        sys.exit(130)
    except Exception as e:
        print_status(f"\n\nUnexpected error: {e}", 'error')
        import traceback
        traceback.print_exc()
        sys.exit(1)
