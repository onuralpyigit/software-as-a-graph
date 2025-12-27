"""
Test Configuration and Fixtures
================================

Shared pytest fixtures for testing the software-as-a-graph project.

Usage:
    pytest tests/                    # Run all tests
    pytest tests/ -v                 # Verbose output
    pytest tests/ -k "core"          # Run only core tests
    pytest tests/ --quick            # Quick subset
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Custom Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Skip slow tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests if --quick is specified"""
    if config.getoption("--quick"):
        skip_slow = pytest.mark.skip(reason="Skipped with --quick")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# =============================================================================
# Graph Data Fixtures
# =============================================================================

@pytest.fixture
def small_graph_data() -> Dict[str, Any]:
    """Small graph for quick tests"""
    return {
        "metadata": {
            "name": "test_small",
            "scenario": "test",
            "scale": "small",
        },
        "applications": [
            {"id": "A1", "name": "App1", "role": "publisher"},
            {"id": "A2", "name": "App2", "role": "subscriber"},
            {"id": "A3", "name": "App3", "role": "pubsub"},
        ],
        "topics": [
            {"id": "T1", "name": "Topic1"},
            {"id": "T2", "name": "Topic2"},
        ],
        "brokers": [
            {"id": "B1", "name": "Broker1"},
        ],
        "nodes": [
            {"id": "N1", "name": "Node1"},
        ],
        "relationships": {
            "publishes_to": [
                {"source": "A1", "target": "T1"},
                {"source": "A3", "target": "T2"},
            ],
            "subscribes_to": [
                {"source": "A2", "target": "T1"},
                {"source": "A3", "target": "T1"},
            ],
            "routes": [
                {"source": "B1", "target": "T1"},
                {"source": "B1", "target": "T2"},
            ],
            "runs_on": [
                {"source": "A1", "target": "N1"},
                {"source": "A2", "target": "N1"},
                {"source": "B1", "target": "N1"},
            ],
            "connects_to": [],
        },
    }


@pytest.fixture
def medium_graph_data() -> Dict[str, Any]:
    """Medium graph for standard tests"""
    from src.core import generate_graph
    return generate_graph(scale="small", scenario="iot", seed=42)


@pytest.fixture
def large_graph_data() -> Dict[str, Any]:
    """Large graph for stress tests"""
    from src.core import generate_graph
    return generate_graph(scale="medium", scenario="financial", seed=42)


# =============================================================================
# SimulationGraph Fixtures
# =============================================================================

@pytest.fixture
def small_graph(small_graph_data):
    """Small SimulationGraph instance"""
    from src.simulation import SimulationGraph
    return SimulationGraph.from_dict(small_graph_data)


@pytest.fixture
def medium_graph(medium_graph_data):
    """Medium SimulationGraph instance"""
    from src.simulation import SimulationGraph
    return SimulationGraph.from_dict(medium_graph_data)


@pytest.fixture
def large_graph(large_graph_data):
    """Large SimulationGraph instance"""
    from src.simulation import SimulationGraph
    return SimulationGraph.from_dict(large_graph_data)


# =============================================================================
# File Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_graph_file(small_graph_data, temp_dir) -> Path:
    """Small graph saved to temp file"""
    filepath = temp_dir / "small_graph.json"
    with open(filepath, 'w') as f:
        json.dump(small_graph_data, f)
    return filepath


@pytest.fixture
def medium_graph_file(medium_graph_data, temp_dir) -> Path:
    """Medium graph saved to temp file"""
    filepath = temp_dir / "medium_graph.json"
    with open(filepath, 'w') as f:
        json.dump(medium_graph_data, f)
    return filepath


# =============================================================================
# Analysis Fixtures
# =============================================================================

@pytest.fixture
def criticality_scores(medium_graph):
    """Pre-computed criticality scores"""
    from src.validation import GraphAnalyzer
    analyzer = GraphAnalyzer(medium_graph)
    composite = analyzer.composite_score()
    
    scores = sorted(composite.values())
    n = len(scores)
    p75 = scores[int(n * 0.75)] if n > 0 else 0
    p50 = scores[int(n * 0.50)] if n > 0 else 0
    
    result = {}
    for comp_id, score in composite.items():
        if score >= p75:
            level = "critical"
        elif score >= p50:
            level = "high"
        else:
            level = "low"
        result[comp_id] = {"score": score, "level": level}
    
    return result


# =============================================================================
# Validation Fixtures
# =============================================================================

@pytest.fixture
def validation_targets():
    """Default validation targets"""
    from src.validation import ValidationTargets
    return ValidationTargets(
        spearman=0.70,
        f1=0.90,
        precision=0.80,
        recall=0.80,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def assert_valid_html(html: str):
    """Assert that HTML is valid"""
    assert html.startswith("<!DOCTYPE html>") or html.startswith("<html")
    assert "</html>" in html
    assert "<body>" in html or "<body " in html


def assert_valid_json(data: Dict):
    """Assert that data is valid JSON-serializable"""
    json.dumps(data)  # Will raise if not serializable
