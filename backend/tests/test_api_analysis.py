
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from src.core.criticality import CriticalityLevel
from src.core.metrics import QualityScores, QualityLevels
from api.dependencies import get_analysis_service

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
    comp.structural.name = "App 1"
    
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
    
    result.quality.components = [comp]
    result.quality.edges = []
    
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
    # Setup mock service
    mock_service = MagicMock()
    mock_service.analyze_layer.return_value = mock_analysis_result
    
    # Override dependency
    app.dependency_overrides[get_analysis_service] = lambda: mock_service
    
    try:
        # Call endpoint
        # Note: We still pass credentials in JSON, even though router doesn't use it anymore
        # Actually, the router DOESN'T take Neo4jCredentials anymore in the function signature!
        # It gets it via Depends(get_analysis_service) -> Depends(get_repository) -> Neo4jCredentials
        
        response = client.post("/api/v1/analysis/full", json={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        })
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["layer"] == "system"
        assert "analysis" in data
        assert data["analysis"]["summary"]["total_components"] == 1
        assert data["analysis"]["components"][0]["id"] == "c1"
        assert data["analysis"]["components"][0]["criticality_levels"]["overall"] == "medium"
        
        # Verify service called
        mock_service.analyze_layer.assert_called_once_with("system")
    finally:
        # Clean up overrides
        app.dependency_overrides = {}
