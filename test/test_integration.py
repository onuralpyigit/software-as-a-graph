"""
Integration Tests
==================

End-to-end tests for the complete pipeline.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestEndToEndPipeline:
    """End-to-end pipeline tests"""

    def test_full_pipeline(self, medium_graph):
        """Test complete pipeline flow"""
        from src.simulation import FailureSimulator, EventSimulator
        from src.validation import ValidationPipeline
        from src.visualization import DashboardGenerator
        
        # 1. Analyze and validate
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(medium_graph, analysis_method="composite")
        
        assert result.validation is not None
        assert result.validation.correlation.spearman > 0
        
        # 2. Simulate events
        event_sim = EventSimulator(seed=42)
        event_result = event_sim.simulate(medium_graph, duration_ms=2000, message_rate=50)
        
        assert event_result.metrics.messages_published > 0
        
        # 3. Generate dashboard
        criticality = {
            k: {"score": v, "level": "high" if v > 0.5 else "low"}
            for k, v in result.predicted_scores.items()
        }
        
        generator = DashboardGenerator()
        html = generator.generate(
            medium_graph,
            criticality=criticality,
            validation=result.validation.to_dict(),
            simulation=event_result.metrics.to_dict(),
        )
        
        assert len(html) > 10000

    def test_generate_to_validate_flow(self):
        """Test generate → validate flow"""
        from src.core import generate_graph
        from src.simulation import SimulationGraph
        from src.validation import ValidationPipeline
        
        # Generate
        graph_data = generate_graph(scale="small", scenario="iot", seed=42)
        graph = SimulationGraph.from_dict(graph_data)
        
        # Validate
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph)
        
        assert result.validation.status is not None

    @pytest.mark.slow
    def test_all_scenarios(self):
        """Test all scenario types"""
        from src.core import generate_graph
        from src.simulation import SimulationGraph
        from src.validation import ValidationPipeline
        
        scenarios = ["iot", "financial", "healthcare", "smart_city"]
        
        for scenario in scenarios:
            graph_data = generate_graph(scale="small", scenario=scenario, seed=42)
            graph = SimulationGraph.from_dict(graph_data)
            
            pipeline = ValidationPipeline(seed=42)
            result = pipeline.run(graph)
            
            assert result.validation is not None, f"Failed for {scenario}"
            assert result.validation.correlation.spearman is not None

    @pytest.mark.slow  
    def test_method_comparison(self, medium_graph):
        """Test comparing analysis methods"""
        from src.validation import ValidationPipeline
        
        pipeline = ValidationPipeline(seed=42)
        results = pipeline.compare_methods(medium_graph)
        
        assert len(results) >= 3
        
        # Check all methods have results
        for method, result in results.items():
            assert result.validation is not None
            assert result.validation.correlation.spearman is not None


class TestCLIIntegration:
    """Tests for CLI script integration"""

    def test_run_py_help(self):
        """Test run.py --help"""
        result = subprocess.run(
            [sys.executable, "run.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        assert "Software-as-a-Graph" in result.stdout

    def test_generate_graph_help(self):
        """Test generate_graph.py --help"""
        result = subprocess.run(
            [sys.executable, "generate_graph.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0

    def test_validate_graph_help(self):
        """Test validate_graph.py --help"""
        result = subprocess.run(
            [sys.executable, "validate_graph.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0

    def test_visualize_graph_help(self):
        """Test visualize_graph.py --help"""
        result = subprocess.run(
            [sys.executable, "visualize_graph.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0

    @pytest.mark.slow
    def test_run_quick_pipeline(self, temp_dir):
        """Test run.py --quick"""
        result = subprocess.run(
            [
                sys.executable, "run.py",
                "--quick",
                "--output", str(temp_dir),
                "--no-color",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=120,
        )
        
        assert result.returncode == 0
        assert "Pipeline Complete" in result.stdout
        
        # Check outputs exist
        assert (temp_dir / "dashboard.html").exists()

    @pytest.mark.slow
    def test_generate_and_validate(self, temp_dir):
        """Test generate → validate CLI flow"""
        # Generate
        graph_file = temp_dir / "test.json"
        gen_result = subprocess.run(
            [
                sys.executable, "generate_graph.py",
                "--scale", "small",
                "--scenario", "iot",
                "--output", str(graph_file),
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert gen_result.returncode == 0
        assert graph_file.exists()
        
        # Validate
        val_result = subprocess.run(
            [
                sys.executable, "validate_graph.py",
                "--input", str(graph_file),
                "--no-color",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60,
        )
        
        assert val_result.returncode == 0


class TestDataConsistency:
    """Tests for data consistency across modules"""

    def test_graph_roundtrip(self, small_graph_data, temp_dir):
        """Test graph data roundtrip"""
        from src.simulation import SimulationGraph
        import json
        
        # Create graph
        graph = SimulationGraph.from_dict(small_graph_data)
        
        # Save
        filepath = temp_dir / "graph.json"
        with open(filepath, 'w') as f:
            json.dump(small_graph_data, f)
        
        # Load
        graph2 = SimulationGraph.from_json(filepath)
        
        # Compare
        assert len(graph.components) == len(graph2.components)
        assert len(graph.connections) == len(graph2.connections)

    def test_validation_result_serialization(self, medium_graph):
        """Test validation result serialization"""
        from src.validation import ValidationPipeline
        import json
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(medium_graph)
        
        # Serialize
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        
        # Deserialize
        loaded = json.loads(json_str)
        
        assert loaded["validation"]["correlation"]["spearman"]["coefficient"] == \
               result.validation.correlation.spearman

    def test_criticality_consistency(self, medium_graph):
        """Test criticality scores are consistent"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        
        # Run twice
        scores1 = analyzer.composite_score()
        scores2 = analyzer.composite_score()
        
        # Should be identical
        for comp_id in scores1:
            assert abs(scores1[comp_id] - scores2[comp_id]) < 0.0001
