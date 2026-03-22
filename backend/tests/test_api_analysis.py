
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from src.core.criticality import CriticalityLevel
from src.core.metrics import QualityScores, QualityLevels
from api.dependencies import (
    get_analyze_graph_use_case, 
    get_predict_graph_use_case,
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
    # Setup mock use cases
    mock_analyze_uc = MagicMock()
    mock_analyze_uc.execute.return_value = mock_analysis_result.structural
    
    mock_predict_uc = MagicMock()
    mock_predict_uc.execute.return_value = (mock_analysis_result.quality, mock_analysis_result.problems)
    
    # Override dependencies
    app.dependency_overrides[get_analyze_graph_use_case] = lambda: mock_analyze_uc
    app.dependency_overrides[get_predict_graph_use_case] = lambda: mock_predict_uc
    
    try:
        response = client.post("/api/v1/analysis/full", json={
            "credentials": {"uri": "bolt://localhost", "user": "n", "password": "p"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "analysis" in data
        mock_analyze_uc.execute.assert_called_once_with("system")
    finally:
        app.dependency_overrides = {}
