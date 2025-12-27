"""
Tests for src.simulation module
================================

Tests SimulationGraph, FailureSimulator, and EventSimulator.
"""

import pytest


class TestSimulationGraph:
    """Tests for SimulationGraph class"""

    def test_create_from_dict(self, small_graph_data):
        """Create SimulationGraph from dictionary"""
        from src.simulation import SimulationGraph
        
        graph = SimulationGraph.from_dict(small_graph_data)
        
        assert graph is not None
        assert len(graph.components) > 0
        assert len(graph.connections) > 0

    def test_create_from_json(self, small_graph_file):
        """Create SimulationGraph from JSON file"""
        from src.simulation import SimulationGraph
        
        graph = SimulationGraph.from_json(small_graph_file)
        
        assert graph is not None
        assert len(graph.components) > 0

    def test_component_types(self, small_graph):
        """Check component types"""
        from src.simulation import ComponentType
        
        types = {c.type for c in small_graph.components.values()}
        
        assert ComponentType.APPLICATION in types
        assert ComponentType.TOPIC in types
        assert ComponentType.BROKER in types

    def test_get_publishers(self, small_graph):
        """Get publishers for a topic"""
        publishers = small_graph.get_publishers("T1")
        
        assert "A1" in publishers

    def test_get_subscribers(self, small_graph):
        """Get subscribers for a topic"""
        subscribers = small_graph.get_subscribers("T1")
        
        assert "A2" in subscribers or "A3" in subscribers

    def test_get_message_paths(self, small_graph):
        """Get message paths through system"""
        paths = small_graph.get_all_message_paths()
        
        assert len(paths) > 0

    def test_get_dependencies(self, medium_graph):
        """Get component dependencies"""
        # Get a broker
        brokers = [c for c in medium_graph.components.values() 
                   if c.type.value == "Broker"]
        
        if brokers:
            deps = medium_graph.get_dependencies(brokers[0].id)
            # Broker should have topic dependencies
            assert isinstance(deps, set)


class TestFailureSimulator:
    """Tests for FailureSimulator class"""

    def test_create_simulator(self):
        """Create failure simulator"""
        from src.simulation import FailureSimulator
        
        sim = FailureSimulator(seed=42)
        
        assert sim is not None

    def test_simulate_single_failure(self, medium_graph):
        """Simulate single component failure"""
        from src.simulation import FailureSimulator
        
        sim = FailureSimulator(seed=42)
        
        # Get a component
        comp_id = list(medium_graph.components.keys())[0]
        result = sim.simulate_failure(medium_graph, comp_id)
        
        assert result is not None
        assert result.impact is not None
        assert 0 <= result.impact.impact_score <= 1

    def test_simulate_with_cascade(self, medium_graph):
        """Simulate failure with cascade"""
        from src.simulation import FailureSimulator
        
        sim = FailureSimulator(cascade_threshold=0.5, seed=42)
        
        # Get a broker (likely to cause cascade)
        brokers = [c for c in medium_graph.components.keys() if c.startswith("B")]
        if brokers:
            result = sim.simulate_failure(
                medium_graph, brokers[0], enable_cascade=True
            )
            assert result is not None

    def test_simulate_without_cascade(self, medium_graph):
        """Simulate failure without cascade"""
        from src.simulation import FailureSimulator
        
        sim = FailureSimulator(seed=42)
        comp_id = list(medium_graph.components.keys())[0]
        
        result = sim.simulate_failure(
            medium_graph, comp_id, enable_cascade=False
        )
        
        assert result is not None
        assert len(result.cascade_failures) == 0

    @pytest.mark.slow
    def test_simulate_all_failures(self, medium_graph):
        """Simulate all component failures"""
        from src.simulation import FailureSimulator
        
        sim = FailureSimulator(seed=42)
        batch = sim.simulate_all_failures(medium_graph)
        
        assert batch is not None
        assert len(batch.results) == len(medium_graph.components)
        assert len(batch.critical_components) > 0

    def test_critical_components_sorted(self, medium_graph):
        """Critical components are sorted by impact"""
        from src.simulation import FailureSimulator
        
        sim = FailureSimulator(seed=42)
        batch = sim.simulate_all_failures(medium_graph)
        
        impacts = [score for _, score in batch.critical_components]
        assert impacts == sorted(impacts, reverse=True)


class TestEventSimulator:
    """Tests for EventSimulator class"""

    def test_create_simulator(self):
        """Create event simulator"""
        from src.simulation import EventSimulator
        
        sim = EventSimulator(seed=42)
        
        assert sim is not None

    def test_simulate_events(self, medium_graph):
        """Run event simulation"""
        from src.simulation import EventSimulator
        
        sim = EventSimulator(seed=42)
        result = sim.simulate(
            medium_graph,
            duration_ms=1000,
            message_rate=50,
        )
        
        assert result is not None
        assert result.metrics is not None
        assert result.metrics.messages_published > 0

    def test_simulation_metrics(self, medium_graph):
        """Check simulation metrics"""
        from src.simulation import EventSimulator
        
        sim = EventSimulator(seed=42)
        result = sim.simulate(medium_graph, duration_ms=1000, message_rate=50)
        
        metrics = result.metrics
        
        assert metrics.messages_published > 0
        assert metrics.messages_delivered >= 0
        assert metrics.delivery_rate() >= 0

    def test_simulation_latency(self, medium_graph):
        """Check latency metrics"""
        from src.simulation import EventSimulator
        
        sim = EventSimulator(seed=42)
        result = sim.simulate(medium_graph, duration_ms=2000, message_rate=100)
        
        assert result.metrics.avg_latency() >= 0
        assert result.metrics.p99_latency() >= 0

    def test_metrics_to_dict(self, medium_graph):
        """Convert metrics to dictionary"""
        from src.simulation import EventSimulator
        
        sim = EventSimulator(seed=42)
        result = sim.simulate(medium_graph, duration_ms=1000, message_rate=50)
        
        metrics_dict = result.metrics.to_dict()
        
        assert "messages" in metrics_dict
        assert "latency" in metrics_dict
        assert metrics_dict["messages"]["published"] > 0

    @pytest.mark.slow
    def test_long_simulation(self, medium_graph):
        """Run longer simulation"""
        from src.simulation import EventSimulator
        
        sim = EventSimulator(seed=42)
        result = sim.simulate(
            medium_graph,
            duration_ms=10000,
            message_rate=100,
        )
        
        assert result.metrics.messages_published > 500
