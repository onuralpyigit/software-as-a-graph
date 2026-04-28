
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from saag.core.criticality import CriticalityLevel
from saag.core.metrics import QualityScores, QualityLevels
from api.dependencies import (
    get_client,
    get_repository
)

client = TestClient(app)

@pytest.fixture
def mock_analysis_result():
    # Mocking a complex nested structure for LayerAnalysisResult
    result = MagicMock()
    result.layer = "system"
    result.layer_name = "System Layer"
    result.description = "Full system analysis"
    
    # Structural mock
    result.structural = MagicMock()
    result.structural.graph_summary.nodes = 1
    result.structural.graph_summary.edges = 0
    result.structural.graph_summary.density = 0.0
    result.structural.graph_summary.avg_degree = 0.0
    
    # Quality component mock
    comp = MagicMock()
    comp.id = "c1"
    comp.type = "Application"
    comp.structural = MagicMock()
    comp.structural.name = "App 1"
    comp.structural.betweenness = 0.1
    comp.structural.pagerank = 0.1
    comp.structural.closeness = 0.1
    comp.structural.eigenvector = 0.1
    comp.structural.clustering_coefficient = 0.1
    comp.structural.total_degree_raw = 1
    comp.structural.in_degree_raw = 0
    comp.structural.is_articulation_point = False
    comp.structural.is_isolated = False
    
    comp.levels = MagicMock(spec=QualityLevels)
    comp.levels.overall = CriticalityLevel.MEDIUM
    comp.levels.reliability = CriticalityLevel.LOW
    comp.levels.maintainability = CriticalityLevel.MEDIUM
    comp.levels.availability = CriticalityLevel.HIGH
    comp.levels.vulnerability = CriticalityLevel.MINIMAL
    
    comp.scores = MagicMock(spec=QualityScores)
    comp.scores.reliability = 0.2
    comp.scores.maintainability = 0.5
    comp.scores.availability = 0.8
    comp.scores.vulnerability = 0.1
    comp.scores.overall = 0.4
    
    comp.to_dict.return_value = {
        "id": "c1",
        "name": "App 1",
        "type": "Application",
        "criticality_level": "medium",
        "is_critical": False,
        "rmav_score": 0.4,
        "criticality_levels": {
            "reliability": "low",
            "maintainability": "medium",
            "availability": "high",
            "vulnerability": "minimal",
            "overall": "medium"
        },
        "scores": {
            "reliability": 0.2,
            "maintainability": 0.5,
            "availability": 0.8,
            "vulnerability": 0.1,
            "overall": 0.4
        },
        "structural": {
            "betweenness": 0.1,
            "pagerank": 0.1,
            "closeness": 0.1,
            "eigenvector": 0.1,
            "clustering_coefficient": 0.1,
            "total_degree_raw": 1,
            "in_degree_raw": 0,
            "out_degree_raw": 0,
            "is_articulation_point": False,
            "is_isolated": False
        }
    }
    
    result.quality = MagicMock()
    result.quality.components = [comp]
    result.quality.edges = []
    
    result.quality.classification_summary = MagicMock()
    result.quality.classification_summary.total_components = 1
    result.quality.classification_summary.component_distribution = {"medium": 1, "critical": 0, "high": 0}
    
    # Problem mock
    prob = MagicMock()
    prob.entity_id = "c1"
    prob.entity_type = "Application"
    prob.category = MagicMock()
    prob.category.value = "Maintainability"
    prob.severity = MagicMock()
    prob.severity.value = "HIGH"
    prob.name = "High Coupling"
    prob.description = "Component has too many dependencies"
    prob.recommendation = "Refactor to use interfaces"
    
    result.problems = [prob]
    
    return result

def test_analyze_full_system(mock_analysis_result):
    # Setup mock client
    mock_client_instance = MagicMock()
    
    # Client returns facades
    mock_analysis = MagicMock()
    mock_analysis.raw = mock_analysis_result.structural
    mock_layer_enum_structural = MagicMock()
    mock_layer_enum_structural.value = "system"
    mock_analysis.raw.layer = mock_layer_enum_structural
    
    mock_prediction = MagicMock()
    mock_prediction.all_components = mock_analysis_result.quality.components
    
    # Needs to be a string, not a mock, for pydantic validation
    mock_layer_enum = MagicMock()
    mock_layer_enum.value = "system"
    mock_prediction.raw.layer = mock_layer_enum
    mock_prediction.raw.edges = {}
    
    mock_client_instance.analyze.return_value = mock_analysis
    mock_client_instance.predict.return_value = mock_prediction
    mock_client_instance.detect_antipatterns.return_value = mock_analysis_result.problems
    
    # Override dependencies
    app.dependency_overrides[get_client] = lambda: mock_client_instance
    
    try:
        response = client.post("/api/v1/analysis/full", json={
            "credentials": {"uri": "bolt://localhost", "user": "n", "password": "p"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "analysis" in data
        mock_client_instance.analyze.assert_called_once_with(layer="system")
    finally:
        app.dependency_overrides = {}
