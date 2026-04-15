import json
import os
import sys
from pathlib import Path

# Fix path to import saag and bin
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bin.common.console import ConsoleDisplay

# Mock LayerAnalysisResult for testing purposes since we can't easily import the complex models
class MockStructural:
    def __init__(self, data):
        self.layer = data.get("layer", "system")
        self.nodes = data.get("nodes", 0)
        self.edges = data.get("edges", 0)
        self.density = data.get("density", 0.0)
        self.avg_degree = data.get("avg_degree", 0.0)
        self.avg_clustering = data.get("avg_clustering", 0.0)
        self.is_connected = data.get("is_connected", False)
        self.num_components = data.get("num_components", 0)
        self.num_articulation_points = data.get("num_articulation_points", 0)
        self.num_bridges = data.get("num_bridges", 0)
        self.connectivity_health = data.get("connectivity_health", "ROBUST")
        self.node_types = data.get("node_types", {})
        self.edge_types = data.get("edge_types", {})
        self.components = {} # Added to fix the error

class MockQuality:
    def __init__(self, data):
        self.components = []
        for c in data.get("components", []):
            self.components.append(MockComponent(c))
        self.classification_summary = MockClassificationSummary(data.get("classification_summary", {}))
        self.sensitivity = data.get("sensitivity", {})

class MockComponent:
    def __init__(self, data):
        self.id = data.get("id")
        self.type = data.get("type")
        self.scores = MockScores(data.get("scores", {}))
        self.levels = MockLevels(data.get("levels", {}))
        self.structural = MockCompStructural(data.get("structural", {}))

class MockCompStructural:
    def __init__(self, data):
        self.name = data.get("name", "Unknown")
        self.is_articulation_point = data.get("is_articulation_point", False)

class MockScores:
    def __init__(self, data):
        self.reliability = data.get("reliability", 0.0)
        self.maintainability = data.get("maintainability", 0.0)
        self.availability = data.get("availability", 0.0)
        self.vulnerability = data.get("vulnerability", 0.0)
        self.overall = data.get("overall", 0.0)

class MockLevels:
    def __init__(self, data):
        self.overall = MockLevel(data.get("overall", "minimal"))

class MockLevel:
    def __init__(self, name):
        self.name = name.upper()

class MockClassificationSummary:
    def __init__(self, data):
        self.total_components = data.get("total_components", 0)
        self.total_edges = data.get("total_edges", 0)
        self.component_distribution = data.get("component_distribution", {})
        self.edge_distribution = data.get("edge_distribution", {})

class MockResult:
    def __init__(self, data):
        self.layer = data.get("layer", "system")
        self.layer_name = data.get("layer_name", "System")
        self.description = data.get("description", "System layer analysis")
        self.structural = MockStructural(data.get("graph_summary", {}))
        self.quality = MockQuality(data.get("quality_analysis", {}))
        self.problems = [] # Simplified
        self.problem_summary = data.get("problem_summary", {})
        self.library_usage = data.get("library_usage", {})
        self.node_allocations = data.get("node_allocations", {})
        self.broker_routing = data.get("broker_routing", {})
        self.prediction = data.get("prediction")

def test():
    # Load sample data
    json_path = project_root / "output/scenario_01_autonomous_vehicle_results/analysis_results.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    app_data = data["layers"]["app"]
    
    # Take a subset of components for testing
    app_data["quality_analysis"]["components"] = app_data["quality_analysis"]["components"][:10]
    
    mock_result = MockResult(app_data)
    
    console = ConsoleDisplay()
    console.display_layer_result(mock_result)

if __name__ == "__main__":
    test()
