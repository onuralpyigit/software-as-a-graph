"""
Tests for the refactored simulation module.

Covers:
    - Event simulation (single source, all publishers)
    - Failure simulation (single target, exhaustive)
    - Layer-aware analysis
    - Component criticality classification
    - Edge criticality classification
    - Report generation
    - CLI argument parsing
"""

import sys
import json
import pytest
import importlib
from unittest.mock import MagicMock, patch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.models import GraphData, ComponentData, EdgeData
from src.domain.models.simulation.graph import SimulationGraph
from src.domain.models.simulation.metrics import (
    LayerMetrics,
    ComponentCriticality,
    EdgeCriticality,
    SimulationReport,
)
from src.domain.services.event_simulator import EventSimulator, EventScenario
from src.domain.services.failure_simulator import FailureSimulator, FailureScenario, ImpactMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def raw_graph_data():
    """Minimal pub-sub graph with all component types."""
    return GraphData(
        components=[
            ComponentData("App1", "Application"),
            ComponentData("App2", "Application"),
            ComponentData("App3", "Application"),
            ComponentData("Topic1", "Topic"),
            ComponentData("Topic2", "Topic"),
            ComponentData("Broker1", "Broker"),
            ComponentData("Node1", "Node"),
            ComponentData("Node2", "Node"),
            ComponentData("Lib1", "Library"),
        ],
        edges=[
            # App layer: pub/sub
            EdgeData("App1", "Topic1", "Application", "Topic", "PUBLISHES_TO", "PUBLISHES_TO"),
            EdgeData("App2", "Topic1", "Application", "Topic", "SUBSCRIBES_TO", "SUBSCRIBES_TO"),
            EdgeData("App1", "Topic2", "Application", "Topic", "PUBLISHES_TO", "PUBLISHES_TO"),
            EdgeData("App3", "Topic2", "Application", "Topic", "SUBSCRIBES_TO", "SUBSCRIBES_TO"),
            # Middleware: routing
            EdgeData("Broker1", "Topic1", "Broker", "Topic", "ROUTES", "ROUTES"),
            EdgeData("Broker1", "Topic2", "Broker", "Topic", "ROUTES", "ROUTES"),
            # Infrastructure: hosting
            EdgeData("App1", "Node1", "Application", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("App2", "Node1", "Application", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("App3", "Node2", "Application", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("Broker1", "Node1", "Broker", "Node", "RUNS_ON", "RUNS_ON"),
            # Infrastructure: connectivity
            EdgeData("Node1", "Node2", "Node", "Node", "CONNECTS_TO", "CONNECTS_TO"),
            # App layer: library usage
            EdgeData("App1", "Lib1", "Application", "Library", "USES", "USES"),
        ],
    )


@pytest.fixture
def sim_graph(raw_graph_data):
    """SimulationGraph instance from the fixture data."""
    return SimulationGraph(graph_data=raw_graph_data)


# =============================================================================
# Event Simulator Tests
# =============================================================================

class TestEventSimulator:
    """Tests for the EventSimulator domain service."""

    def test_single_publisher_reaches_subscriber(self, sim_graph):
        """App1 publishes to Topic1; App2 subscribes -> App2 should be reached."""
        sim = EventSimulator(sim_graph)
        result = sim.simulate(EventScenario("App1", "test", num_messages=10))

        assert "Topic1" in result.affected_topics
        assert "App2" in result.reached_subscribers
        assert result.metrics.messages_published > 0

    def test_multi_topic_publisher(self, sim_graph):
        """App1 publishes to both Topic1 and Topic2."""
        sim = EventSimulator(sim_graph)
        result = sim.simulate(EventScenario("App1", "test", num_messages=20))

        assert "Topic1" in result.affected_topics
        assert "Topic2" in result.affected_topics

    def test_nonexistent_source_returns_empty(self, sim_graph):
        """Simulating from a nonexistent app returns empty result."""
        sim = EventSimulator(sim_graph)
        result = sim.simulate(EventScenario("NoSuchApp", "test"))

        assert result.metrics.messages_published == 0

    def test_simulate_all_publishers(self, sim_graph):
        """Simulate all publishers and get per-publisher results."""
        sim = EventSimulator(sim_graph)
        scenario = EventScenario("", "all", num_messages=10, duration=5.0)
        results = sim.simulate_all_publishers(scenario)

        # App1 is the only publisher
        assert "App1" in results
        assert results["App1"].metrics.messages_published > 0

    def test_metrics_are_consistent(self, sim_graph):
        """published = delivered + dropped."""
        sim = EventSimulator(sim_graph)
        result = sim.simulate(EventScenario("App1", "test", num_messages=50))

        m = result.metrics
        assert m.messages_published == m.messages_delivered + m.messages_dropped

    def test_component_impacts_populated(self, sim_graph):
        """Component impacts dict should have entries for active components."""
        sim = EventSimulator(sim_graph)
        result = sim.simulate(EventScenario("App1", "test", num_messages=20))

        assert len(result.component_impacts) > 0
        # Publisher should have impact
        assert result.component_impacts.get("App1", 0) > 0


# =============================================================================
# Failure Simulator Tests
# =============================================================================

class TestFailureSimulator:
    """Tests for the FailureSimulator domain service."""

    def test_node_failure_cascades_to_hosted_apps(self, sim_graph):
        """Failing Node1 should cascade to App1, App2, Broker1 (all hosted)."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        assert "App1" in result.cascaded_failures
        assert "App2" in result.cascaded_failures
        assert "Broker1" in result.cascaded_failures
        assert result.impact.cascade_count >= 3

    def test_broker_failure_does_not_cascade_to_nodes(self, sim_graph):
        """Failing a broker should not cascade to infrastructure nodes."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        assert "Node1" not in result.cascaded_failures
        assert "Node2" not in result.cascaded_failures

    def test_nonexistent_target_returns_empty(self, sim_graph):
        """Targeting a nonexistent component returns empty result."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("NoSuchComp", "test"))

        assert result.impact.composite_impact == 0.0

    def test_exhaustive_covers_all_layer_components(self, sim_graph):
        """Exhaustive simulation should cover all analyzable components in layer."""
        sim = FailureSimulator(sim_graph)
        results = sim.simulate_exhaustive(layer="app")

        analyzed_ids = {r.target_id for r in results}
        expected_apps = {"App1", "App2", "App3"}
        assert expected_apps == analyzed_ids

    def test_exhaustive_sorted_by_impact(self, sim_graph):
        """Exhaustive results should be sorted by impact (descending)."""
        sim = FailureSimulator(sim_graph)
        results = sim.simulate_exhaustive(layer="system")

        impacts = [r.impact.composite_impact for r in results]
        assert impacts == sorted(impacts, reverse=True)

    def test_layer_impacts_present(self, sim_graph):
        """Failure result should include per-layer impact breakdown."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        assert "app" in result.layer_impacts
        assert "infra" in result.layer_impacts
        assert result.layer_impacts["infra"] > 0  # Node1 is infra

    def test_cascade_probability_zero_no_cascade(self, sim_graph):
        """With cascade_probability=0, no cascade should propagate."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario(
            "Node1", "test", cascade_probability=0.0,
        ))

        # Only the target itself fails, no cascade
        assert len(result.cascaded_failures) == 0

    def test_impact_metrics_normalized(self, sim_graph):
        """Impact metrics should be in [0, 1] range."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        assert 0 <= result.impact.reachability_loss <= 1
        assert 0 <= result.impact.fragmentation <= 1
        assert 0 <= result.impact.composite_impact <= 1


# =============================================================================
# Metrics Data Class Tests
# =============================================================================

class TestMetricsDataClasses:
    """Tests for metrics data classes."""

    def test_layer_metrics_to_dict(self):
        """LayerMetrics.to_dict() should produce valid structure."""
        m = LayerMetrics(
            layer="app",
            event_throughput=500,
            event_delivered=480,
            event_dropped=20,
            event_delivery_rate=96.0,
            event_drop_rate=4.0,
            total_components=10,
        )
        d = m.to_dict()

        assert d["layer"] == "app"
        assert d["event_metrics"]["throughput"] == 500
        assert d["event_metrics"]["delivery_rate_percent"] == 96.0
        assert d["total_components"] == 10

    def test_component_criticality_to_dict(self):
        """ComponentCriticality.to_dict() should produce valid structure."""
        c = ComponentCriticality(
            id="App1",
            type="Application",
            event_impact=0.8,
            failure_impact=0.6,
            combined_impact=0.68,
            level="critical",
            cascade_count=3,
        )
        d = c.to_dict()

        assert d["id"] == "App1"
        assert d["scores"]["combined_impact"] == 0.68
        assert d["level"] == "critical"
        assert d["metrics"]["cascade_count"] == 3

    def test_edge_criticality_to_dict(self):
        """EdgeCriticality.to_dict() should produce valid structure."""
        e = EdgeCriticality(
            source="App1",
            target="Topic1",
            relationship="PUBLISHES_TO",
            flow_impact=0.9,
            combined_impact=0.9,
            level="critical",
            messages_traversed=450,
        )
        d = e.to_dict()

        assert d["source"] == "App1"
        assert d["relationship"] == "PUBLISHES_TO"
        assert d["level"] == "critical"

    def test_simulation_report_to_dict(self):
        """SimulationReport.to_dict() should include all sections."""
        report = SimulationReport(
            timestamp="2025-01-01T00:00:00",
            graph_summary={"total_nodes": 10},
            layer_metrics={"app": LayerMetrics(layer="app", total_components=5)},
            component_criticality=[
                ComponentCriticality(id="App1", type="Application", level="critical"),
            ],
            recommendations=["HEALTHY: No issues"],
        )
        d = report.to_dict()

        assert "timestamp" in d
        assert "layer_metrics" in d
        assert d["layer_metrics"]["app"]["total_components"] == 5
        assert len(d["component_criticality"]) == 1
        assert len(d["recommendations"]) == 1

    def test_report_get_critical_components(self):
        """get_critical_components() filters correctly."""
        report = SimulationReport(
            timestamp="2025-01-01",
            graph_summary={},
            component_criticality=[
                ComponentCriticality(id="A", type="Application", level="critical"),
                ComponentCriticality(id="B", type="Application", level="high"),
                ComponentCriticality(id="C", type="Application", level="critical"),
            ],
        )
        critical = report.get_critical_components()
        assert len(critical) == 2
        assert all(c.level == "critical" for c in critical)


# =============================================================================
# SimulationGraph Layer Tests
# =============================================================================

class TestSimulationGraphLayers:
    """Tests for layer filtering on SimulationGraph."""

    def test_app_layer_components(self, sim_graph):
        """App layer should include Applications, Topics, Libraries."""
        comps = sim_graph.get_components_by_layer("app")
        types = {sim_graph.components[c].type for c in comps}
        assert "Application" in types
        assert "Topic" in types
        assert "Library" in types
        assert "Node" not in types

    def test_infra_layer_components(self, sim_graph):
        """Infra layer should include Nodes, Applications, Brokers."""
        comps = sim_graph.get_components_by_layer("infra")
        types = {sim_graph.components[c].type for c in comps}
        assert "Node" in types

    def test_mw_layer_components(self, sim_graph):
        """MW layer should include Brokers, Topics, Applications."""
        comps = sim_graph.get_components_by_layer("mw")
        types = {sim_graph.components[c].type for c in comps}
        assert "Broker" in types
        assert "Topic" in types

    def test_system_layer_includes_all(self, sim_graph):
        """System layer should include all component types."""
        comps = sim_graph.get_components_by_layer("system")
        assert len(comps) == len(sim_graph.components)

    def test_analyze_components_app_layer(self, sim_graph):
        """Analyze components for app layer should only be Applications."""
        analyze = sim_graph.get_analyze_components_by_layer("app")
        types = {sim_graph.components[c].type for c in analyze}
        assert types == {"Application"}

    def test_analyze_components_infra_layer(self, sim_graph):
        """Analyze components for infra layer should only be Nodes."""
        analyze = sim_graph.get_analyze_components_by_layer("infra")
        types = {sim_graph.components[c].type for c in analyze}
        assert types == {"Node"}

    def test_layer_relationships(self, sim_graph):
        """Each layer should define specific relationship types."""
        app_rels = sim_graph.get_layer_relationships("app")
        assert "PUBLISHES_TO" in app_rels
        assert "SUBSCRIBES_TO" in app_rels
        assert "RUNS_ON" not in app_rels

        infra_rels = sim_graph.get_layer_relationships("infra")
        assert "RUNS_ON" in infra_rels
        assert "CONNECTS_TO" in infra_rels


# =============================================================================
# CLI Smoke Tests
# =============================================================================

class TestCLI:
    """Smoke tests for CLI argument parsing and handler dispatch."""

    def test_event_single_source_args(self):
        """Parse event --source App1."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["event", "--source", "App1", "--messages", "50"])
        assert args.command == "event"
        assert args.source == "App1"
        assert args.messages == 50

    def test_event_all_args(self):
        """Parse event --all --layer mw."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["event", "--all", "--layer", "mw"])
        assert args.command == "event"
        assert args.all is True
        assert args.layer == "mw"

    def test_failure_target_args(self):
        """Parse failure --target Broker1."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["failure", "--target", "Broker1", "--cascade-prob", "0.8"])
        assert args.command == "failure"
        assert args.target == "Broker1"
        assert args.cascade_prob == 0.8

    def test_failure_exhaustive_args(self):
        """Parse failure --exhaustive --layer infra."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["failure", "--exhaustive", "--layer", "infra"])
        assert args.command == "failure"
        assert args.exhaustive is True
        assert args.layer == "infra"

    def test_report_args(self):
        """Parse report --layers app,mw --edges."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["report", "--layers", "app,mw", "--edges"])
        assert args.command == "report"
        assert args.layers == "app,mw"
        assert args.edges is True

    def test_classify_component_args(self):
        """Parse classify --layer system --top 20."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["classify", "--layer", "system", "--top", "20"])
        assert args.command == "classify"
        assert args.edges is False
        assert args.top == 20

    def test_classify_edge_args(self):
        """Parse classify --edges --layer app."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        args = parser.parse_args(["classify", "--edges", "--layer", "app"])
        assert args.command == "classify"
        assert args.edges is True

    def test_no_command_returns_error(self):
        """No subcommand should cause main() to return 1."""
        from bin.simulate_graph import build_parser
        parser = build_parser()
        # argparse will not populate args.command
        args = parser.parse_args([])
        assert args.command is None

    def test_cli_main_event_mocked(self):
        """End-to-end CLI test with mocked container."""
        mock_container = MagicMock()
        mock_display = MagicMock()
        mock_sim = MagicMock()

        mock_container.display_service.return_value = mock_display
        mock_container.simulation_service.return_value = mock_sim

        mock_event_result = MagicMock()
        mock_event_result.to_dict.return_value = {"source": "App1"}
        mock_sim.run_event_simulation.return_value = mock_event_result

        with patch.object(sys, "argv", ["simulate_graph.py", "event", "--source", "App1"]), \
             patch("src.application.container.Container", return_value=mock_container):
            # Need to reload module inside patch context for mock to take effect
            import bin.simulate_graph as simulate_graph
            importlib.reload(simulate_graph)
            ret = simulate_graph.main()

        assert ret == 0
        mock_sim.run_event_simulation.assert_called_once()
        mock_display.display_event_result.assert_called_once()
        mock_container.close.assert_called_once()

    def test_cli_main_report_mocked(self):
        """End-to-end CLI test for report command."""
        mock_container = MagicMock()
        mock_display = MagicMock()
        mock_sim = MagicMock()

        mock_container.display_service.return_value = mock_display
        mock_container.simulation_service.return_value = mock_sim

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"timestamp": "now"}
        mock_sim.generate_report.return_value = mock_report

        with patch.object(sys, "argv", ["simulate_graph.py", "report", "--layers", "app,infra"]), \
             patch("src.application.container.Container", return_value=mock_container):
            # Need to reload module inside patch context for mock to take effect
            import bin.simulate_graph as simulate_graph
            importlib.reload(simulate_graph)
            ret = simulate_graph.main()

        assert ret == 0
        mock_sim.generate_report.assert_called_once_with(
            layers=["app", "infra"], classify_edges=False,
        )
        mock_display.display_simulation_report.assert_called_once()