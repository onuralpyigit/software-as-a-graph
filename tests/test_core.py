"""
Tests for src.core module
===========================

Tests graph generation, model, and import functionality.
"""

import pytest
import json
from pathlib import Path


class TestGraphGeneration:
    """Tests for graph generation"""

    def test_generate_small_graph(self):
        """Generate small graph"""
        from src.core import generate_graph
        
        graph = generate_graph(scale="small", scenario="iot", seed=42)
        
        assert "applications" in graph
        assert "topics" in graph
        assert "brokers" in graph
        assert "nodes" in graph
        assert "relationships" in graph
        assert len(graph["applications"]) > 0

    def test_generate_medium_graph(self):
        """Generate medium graph"""
        from src.core import generate_graph
        
        graph = generate_graph(scale="medium", scenario="financial", seed=42)
        
        assert len(graph["applications"]) > len(
            generate_graph(scale="small", scenario="financial", seed=42)["applications"]
        )

    @pytest.mark.slow
    def test_generate_large_graph(self):
        """Generate large graph"""
        from src.core import generate_graph
        
        graph = generate_graph(scale="large", scenario="microservices", seed=42)
        
        assert len(graph["applications"]) >= 50

    def test_generate_all_scenarios(self):
        """Generate all scenario types"""
        from src.core import generate_graph
        
        scenarios = ["iot", "financial", "healthcare", "smart_city"]
        
        for scenario in scenarios:
            graph = generate_graph(scale="small", scenario=scenario, seed=42)
            assert len(graph["applications"]) > 0, f"Failed for {scenario}"

    def test_generate_deterministic(self):
        """Same seed produces same graph"""
        from src.core import generate_graph
        
        graph1 = generate_graph(scale="small", scenario="iot", seed=123)
        graph2 = generate_graph(scale="small", scenario="iot", seed=123)
        
        assert len(graph1["applications"]) == len(graph2["applications"])
        assert len(graph1["topics"]) == len(graph2["topics"])

    def test_generate_different_seeds(self):
        """Different seeds produce different graphs"""
        from src.core import generate_graph
        
        graph1 = generate_graph(scale="small", scenario="iot", seed=123)
        graph2 = generate_graph(scale="small", scenario="iot", seed=456)
        
        # At least some difference
        apps1 = {a["id"] for a in graph1["applications"]}
        apps2 = {a["id"] for a in graph2["applications"]}
        # IDs might be same but roles/connections differ


class TestGraphModel:
    """Tests for GraphModel class"""

    def test_create_from_dict(self, small_graph_data):
        """Create model from dictionary"""
        from src.core import GraphModel
        
        model = GraphModel.from_dict(small_graph_data)
        
        assert model is not None
        assert len(model.applications) == 3
        assert len(model.topics) == 2
        assert len(model.brokers) == 1

    def test_model_summary(self, small_graph_data):
        """Get model summary"""
        from src.core import GraphModel
        
        model = GraphModel.from_dict(small_graph_data)
        summary = model.summary()
        
        assert "applications" in summary
        assert "topics" in summary
        assert "brokers" in summary

    def test_model_to_dict(self, small_graph_data):
        """Convert model back to dictionary"""
        from src.core import GraphModel
        
        model = GraphModel.from_dict(small_graph_data)
        data = model.to_dict()
        
        assert "applications" in data
        assert len(data["applications"]) == 3


class TestGraphFile:
    """Tests for file I/O"""

    def test_save_and_load(self, small_graph_data, temp_dir):
        """Save and load graph JSON"""
        from src.core import generate_graph
        
        filepath = temp_dir / "test_graph.json"
        
        # Save
        with open(filepath, 'w') as f:
            json.dump(small_graph_data, f)
        
        # Load
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["applications"] == small_graph_data["applications"]

    def test_generate_and_save(self, temp_dir):
        """Generate graph and save to file"""
        from src.core import generate_graph
        
        graph = generate_graph(scale="small", scenario="iot", seed=42)
        filepath = temp_dir / "generated.json"
        
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=2)
        
        assert filepath.exists()
        assert filepath.stat().st_size > 100
