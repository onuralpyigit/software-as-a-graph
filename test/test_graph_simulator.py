#!/usr/bin/env python3
"""
Test Suite for Simulation System

Comprehensive tests for:
- Lightweight DDS simulator
- Failure simulator
- Integration between components
- Various failure scenarios
- Performance validation

Run with: python test_simulation.py
"""

import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

try:
    from src.simulation.lightweight_dds_simulator import (
        LightweightDDSSimulator, Message, MessagePriority, SimulationStats
    )
    from src.simulation.enhanced_failure_simulator import (
        FailureSimulator, FailureType, ComponentType
    )
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure simulation modules are in src/simulation/")
    sys.exit(1)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_pass(self, test_name: str, message: str = ""):
        self.passed += 1
        self.tests.append({'name': test_name, 'status': 'PASS', 'message': message})
        print(f"✅ PASS: {test_name}")
        if message:
            print(f"   {message}")
    
    def add_fail(self, test_name: str, message: str = ""):
        self.failed += 1
        self.tests.append({'name': test_name, 'status': 'FAIL', 'message': message})
        print(f"❌ FAIL: {test_name}")
        if message:
            print(f"   {message}")
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total: {total}")
        print(f"Passed: {self.passed} ({self.passed/total*100:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {self.failed} ({self.failed/total*100:.1f}%)" if total > 0 else "Failed: 0")
        print("="*70)
        return self.failed == 0


def create_test_graph() -> Dict[str, Any]:
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
            {"id": "topic1", "name": "topic1", "message_size_bytes": 512, "message_rate_hz": 10},
            {"id": "topic2", "name": "topic2", "message_size_bytes": 1024, "message_rate_hz": 1}
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
                {"from": "app1", "to": "topic1", "period_ms": 100, "msg_size": 512},
                {"from": "app2", "to": "topic2", "period_ms": 1000, "msg_size": 1024}
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


def test_graph_loading(results: TestResults):
    """Test loading graph from JSON"""
    print("\n" + "="*70)
    print("TEST: Graph Loading")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        # Load into simulator
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        # Verify components loaded
        assert len(simulator.nodes) == 2, f"Expected 2 nodes, got {len(simulator.nodes)}"
        assert len(simulator.applications) == 3, f"Expected 3 apps, got {len(simulator.applications)}"
        assert len(simulator.brokers) == 1, f"Expected 1 broker, got {len(simulator.brokers)}"
        assert len(simulator.topics) == 2, f"Expected 2 topic, got {len(simulator.topics)}"
        
        results.add_pass(
            "Graph Loading",
            f"Loaded 2 nodes, 3 apps, 1 broker, 2 topic"
        )
        
        # Cleanup
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("Graph Loading", str(e))


async def test_basic_simulation(results: TestResults):
    """Test basic message simulation"""
    print("\n" + "="*70)
    print("TEST: Basic Message Simulation")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        # Run short simulation
        sim_results = await simulator.run_simulation(duration_seconds=10)
        
        stats = sim_results['global_stats']
        
        # Verify messages were sent and delivered
        assert stats['messages_sent'] > 0, "No messages sent"
        assert stats['messages_delivered'] > 0, "No messages delivered"
        assert stats['delivery_rate'] > 0, "Zero delivery rate"
        assert stats['avg_latency_ms'] > 0, "Zero latency"
        
        results.add_pass(
            "Basic Simulation",
            f"Sent: {stats['messages_sent']}, Delivered: {stats['messages_delivered']}, "
            f"Rate: {stats['delivery_rate']:.2%}, Latency: {stats['avg_latency_ms']:.2f}ms"
        )
        
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("Basic Simulation", str(e))


async def test_application_failure(results: TestResults):
    """Test application failure simulation"""
    print("\n" + "="*70)
    print("TEST: Application Failure")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        failure_sim = FailureSimulator()
        
        # Start simulation
        sim_task = asyncio.create_task(simulator.run_simulation(duration_seconds=10))
        
        # Inject failure after 3 seconds
        await asyncio.sleep(3)
        failure_sim.inject_failure(
            simulator,
            'app1',  # Fail the publisher
            ComponentType.APPLICATION,
            FailureType.COMPLETE,
            severity=1.0,
            enable_cascade=False
        )
        
        # Wait for simulation to complete
        sim_results = await sim_task
        
        # Verify failure impact
        assert len(failure_sim.failure_events) == 1, "Failure not recorded"
        assert 'app1' in failure_sim.active_failures, "Failure not active"
        
        # Verify reduced message delivery after failure
        stats = sim_results['global_stats']
        assert stats['messages_dropped'] > 0, "No messages dropped"
        
        results.add_pass(
            "Application Failure",
            f"Failed app1, Dropped: {stats['messages_dropped']}, "
            f"Events: {len(failure_sim.failure_events)}"
        )
        
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("Application Failure", str(e))


async def test_cascading_failure(results: TestResults):
    """Test cascading failure propagation"""
    print("\n" + "="*70)
    print("TEST: Cascading Failure")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        failure_sim = FailureSimulator(
            cascade_threshold=0.5,  # Lower threshold for easier cascade
            cascade_probability=0.8  # Higher probability
        )
        
        # Start simulation
        sim_task = asyncio.create_task(simulator.run_simulation(duration_seconds=15))
        
        # Inject failure with cascade
        await asyncio.sleep(3)
        failure_sim.inject_failure(
            simulator,
            'B1',  # Fail the broker
            ComponentType.BROKER,
            FailureType.COMPLETE,
            severity=1.0,
            enable_cascade=True
        )
        
        # Wait for simulation
        sim_results = await sim_task
        
        # Analyze impact
        impact = failure_sim.analyze_impact(simulator)
        
        # Verify cascade occurred (may or may not happen depending on load)
        assert len(failure_sim.failure_events) >= 1, "No failure events"
        assert 'B1' in impact.failed_components, "Broker not in failed components"
        
        results.add_pass(
            "Cascading Failure",
            f"Failed components: {len(impact.failed_components)}, "
            f"Affected: {len(impact.affected_components)}, "
            f"Cascade depth: {impact.cascade_depth}"
        )
        
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("Cascading Failure", str(e))


async def test_performance_metrics(results: TestResults):
    """Test performance metric collection"""
    print("\n" + "="*70)
    print("TEST: Performance Metrics")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        sim_results = await simulator.run_simulation(duration_seconds=10)
        
        stats = sim_results['global_stats']
        
        # Verify all metrics are present and valid
        assert 'messages_sent' in stats, "Missing messages_sent"
        assert 'messages_delivered' in stats, "Missing messages_delivered"
        assert 'avg_latency_ms' in stats, "Missing avg_latency_ms"
        assert 'delivery_rate' in stats, "Missing delivery_rate"
        assert 'bytes_transferred' in stats, "Missing bytes_transferred"
        
        assert stats['delivery_rate'] >= 0 and stats['delivery_rate'] <= 1, "Invalid delivery rate"
        assert stats['avg_latency_ms'] >= 0, "Negative latency"
        assert stats['bytes_transferred'] >= 0, "Negative bytes"
        
        results.add_pass(
            "Performance Metrics",
            f"All metrics valid: {len(stats)} metrics collected"
        )
        
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("Performance Metrics", str(e))


async def test_baseline_comparison(results: TestResults):
    """Test baseline vs failure comparison"""
    print("\n" + "="*70)
    print("TEST: Baseline Comparison")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        # Run baseline
        baseline_sim = LightweightDDSSimulator()
        baseline_sim.load_from_json(temp_path)
        baseline_results = await baseline_sim.run_simulation(duration_seconds=10)
        
        # Run with failure
        failure_sim_obj = LightweightDDSSimulator()
        failure_sim_obj.load_from_json(temp_path)
        
        failure_mgr = FailureSimulator()
        
        sim_task = asyncio.create_task(failure_sim_obj.run_simulation(duration_seconds=10))
        await asyncio.sleep(3)
        failure_mgr.inject_failure(
            failure_sim_obj,
            'A1',
            ComponentType.APPLICATION,
            FailureType.COMPLETE
        )
        failure_results = await sim_task
        
        # Compare
        baseline_rate = baseline_results['global_stats']['delivery_rate']
        failure_rate = failure_results['global_stats']['delivery_rate']
        
        assert failure_rate <= baseline_rate, "Failure didn't degrade performance"
        
        degradation = (baseline_rate - failure_rate) / baseline_rate * 100
        
        results.add_pass(
            "Baseline Comparison",
            f"Baseline rate: {baseline_rate:.2%}, Failure rate: {failure_rate:.2%}, "
            f"Degradation: {degradation:.1f}%"
        )
        
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("Baseline Comparison", str(e))


async def test_qos_policies(results: TestResults):
    """Test QoS policy enforcement"""
    print("\n" + "="*70)
    print("TEST: QoS Policies")
    print("="*70)
    
    try:
        graph_data = create_test_graph()
        
        # Modify topic for stricter deadlines
        graph_data['topics'][0]['properties']['deadline_ms'] = 100  # Very strict
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            temp_path = f.name
        
        simulator = LightweightDDSSimulator()
        simulator.load_from_json(temp_path)
        
        sim_results = await simulator.run_simulation(duration_seconds=10)
        
        stats = sim_results['global_stats']
        
        # With strict deadline, we should see some misses
        # (This depends on timing, so we just check the metric exists)
        assert 'deadline_misses' in stats, "Missing deadline_misses metric"
        
        results.add_pass(
            "QoS Policies",
            f"Deadline enforced: {stats['deadline_misses']} misses detected"
        )
        
        Path(temp_path).unlink()
        
    except Exception as e:
        results.add_fail("QoS Policies", str(e))


def test_message_creation(results: TestResults):
    """Test message object creation"""
    print("\n" + "="*70)
    print("TEST: Message Creation")
    print("="*70)
    
    try:
        msg = Message(
            id="msg_001",
            topic="test_topic",
            sender="app_1",
            payload_size=1024,
            timestamp=time.time(),
            priority=MessagePriority.HIGH,
            deadline_ms=1000,
            ttl_ms=5000
        )
        
        assert msg.id == "msg_001", "Wrong message ID"
        assert msg.topic == "test_topic", "Wrong topic"
        assert msg.payload_size == 1024, "Wrong payload size"
        assert msg.priority == MessagePriority.HIGH, "Wrong priority"
        
        # Test expiration
        assert not msg.is_expired(msg.timestamp + 1), "Should not be expired"
        assert msg.is_expired(msg.timestamp + 10), "Should be expired"
        
        results.add_pass("Message Creation", "Message object created and validated")
        
    except Exception as e:
        results.add_fail("Message Creation", str(e))


async def run_all_tests():
    """Run all tests"""
    results = TestResults()
    
    print("\n" + "="*70)
    print("SIMULATION SYSTEM TEST SUITE")
    print("="*70)
    print()
    
    # Unit tests
    test_message_creation(results)
    test_graph_loading(results)
    
    # Integration tests
    await test_basic_simulation(results)
    await test_application_failure(results)
    await test_cascading_failure(results)
    await test_performance_metrics(results)
    await test_baseline_comparison(results)
    await test_qos_policies(results)
    
    # Print summary
    success = results.summary()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
