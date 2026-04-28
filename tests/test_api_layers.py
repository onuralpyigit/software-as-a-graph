
import pytest
from fastapi.testclient import TestClient
from api.main import app
from saag.core.layers import AnalysisLayer
from api.dependencies import get_client
from unittest.mock import MagicMock, patch

client = TestClient(app)

@pytest.fixture
def mock_client():
    mock = MagicMock()
    return mock

# Credentials for simulation requests
DUMMY_CREDS = {
    "credentials": {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "password",
        "database": "neo4j"
    }
}

# Minimal valid response structure for AnalysisEnvelope
VALID_ANALYSIS_RESPONSE = {
    "success": True,
    "layer": "system",
    "analysis": {
        "context": "Test Context",
        "description": "Test Description",
        "summary": {
            "total_components": 0,
            "critical_count": 0,
            "high_count": 0,
            "total_problems": 0,
            "critical_problems": 0,
            "components": {},
            "edges": {}
        },
        "stats": {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "avg_degree": 0.0
        },
        "components": [],
        "edges": [],
        "problems": []
    }
}

@pytest.mark.parametrize("layer_input, expected_canonical", [
    ("app", "app"),
    ("application", "app"),
    ("infra", "infra"),
    ("infrastructure", "infra"),
    ("mw", "mw"),
    ("middleware", "mw"),
    ("mw-app", "mw"),
    ("mw-infra", "mw"),
    ("system", "system"),
    ("complete", "system"),
    ("all", "system"),
])
def test_analysis_layer_resolution(layer_input, expected_canonical, mock_client):
    """Verify that various layer aliases resolve to the correct canonical layer in Analysis Router."""
    app.dependency_overrides[get_client] = lambda: mock_client
    
    with patch("api.routers.analysis._structural_analyze") as mock_sa:
        mock_sa.return_value = MagicMock()
        with patch("api.routers.analysis._predict") as mock_p:
            mock_p.return_value = MagicMock()
            with patch("api.routers.analysis._detect_antipatterns") as mock_da:
                mock_da.return_value = []
                with patch("api.routers.analysis.analysis_presenter.build_analysis_response") as mock_presenter:
                    mock_resp = VALID_ANALYSIS_RESPONSE.copy()
                    mock_resp["layer"] = expected_canonical
                    mock_presenter.return_value = mock_resp
                    
                    try:
                        response = client.post(f"/api/v1/analysis/layer/{layer_input}")
                        assert response.status_code == 200, f"Failed for {layer_input}: {response.text}"
                        
                        # Verify that _structural_analyze was called with the CANONICAL name
                        mock_sa.assert_called_with(mock_client, expected_canonical)
                    finally:
                        app.dependency_overrides = {}

@pytest.mark.parametrize("layer_input", ["invalid", "unknown", "midware"])
def test_analysis_layer_invalid(layer_input, mock_client):
    """Verify that invalid layer names return 400."""
    app.dependency_overrides[get_client] = lambda: mock_client
    try:
        response = client.post(f"/api/v1/analysis/layer/{layer_input}")
        assert response.status_code == 400
        assert "Unknown layer" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}

def test_simulation_failure_layer_resolution(mock_client):
    """Verify layer resolution in simulation failure endpoint."""
    app.dependency_overrides[get_client] = lambda: mock_client
    try:
        payload = {"target_id": "c1", "layer": "middleware", **DUMMY_CREDS}
        
        with patch("api.routers.simulation.simulation_presenter.format_failure_simulation_response") as mock_fmt:
            # Satisfy Pydantic FailureSimulationResponse
            mock_fmt.return_value = {
                "success": True, 
                "simulation_type": "single", 
                "result": {
                    "target_id": "c1", 
                    "target_type": "Application", 
                    "scenario": "test",
                    "impact": {
                        "reachability": {"loss_percent": 0.0},
                        "fragmentation": {"fragmentation_percent": 0.0},
                        "throughput": {"loss_percent": 0.0},
                        "flow_disruption": {"loss_percent": 0.0},
                        "cascade": {"count": 0, "depth": 0},
                        "affected": {"topics": 0, "publishers": 0, "subscribers": 0},
                        "composite_impact": 0.0,
                        "reliability": {"cascade_reach": 0.0, "weighted_cascade_impact": 0.0, "normalized_cascade_depth": 0.0, "reliability_impact": 0.0},
                        "maintainability": {"change_reach": 0.0, "weighted_change_impact": 0.0, "normalized_change_depth": 0.0, "maintainability_impact": 0.0},
                        "availability": {"weighted_reachability_loss": 0.0, "weighted_fragmentation": 0.0, "path_breaking_throughput_loss": 0.0, "availability_impact": 0.0, "ia_out": 0.0, "ia_in": 0.0},
                        "vulnerability": {"attack_reach": 0.0, "weighted_attack_impact": 0.0, "high_value_contamination": 0.0, "vulnerability_impact": 0.0}
                    },
                    "cascaded_failures": [],
                    "cascade_sequence": [],
                    "layer_impacts": {}
                }
            }
            
            response = client.post("/api/v1/simulation/failure", json=payload)
            assert response.status_code == 200, response.text
            mock_client.simulate.assert_called_with(target_id="c1", layer="mw", mode="single")
    finally:
        app.dependency_overrides = {}

def test_simulation_exhaustive_layer_resolution(mock_client):
    """Verify layer resolution in simulation exhaustive endpoint."""
    app.dependency_overrides[get_client] = lambda: mock_client
    try:
        payload = {"layer": "application", **DUMMY_CREDS}
        with patch("api.routers.simulation.simulation_presenter.format_exhaustive_simulation_response") as mock_fmt:
            mock_fmt.return_value = {
                "success": True, 
                "simulation_type": "exhaustive", 
                "layer": "app", 
                "summary": {
                    "total_components": 0, "avg_impact": 0.0, "max_impact": 0.0, "critical_count": 0, "high_count": 0, "medium_count": 0, "low_count": 0, "spof_count": 0
                }, 
                "results": []
            }
            
            response = client.post("/api/v1/simulation/exhaustive", json=payload)
            assert response.status_code == 200, response.text
            mock_client.simulate.assert_called_with(layer="app", mode="exhaustive")
    finally:
        app.dependency_overrides = {}

def test_simulation_report_layer_resolution(mock_client):
    """Verify layer resolution in simulation report endpoint."""
    app.dependency_overrides[get_client] = lambda: mock_client
    try:
        payload = {"layers": ["application", "infrastructure", "middleware", "complete"], **DUMMY_CREDS}
        with patch("api.routers.simulation.simulation_presenter.format_simulation_report_response") as mock_fmt:
            mock_fmt.return_value = {"success": True, "report": {"timestamp": "now", "layer_metrics": {}, "top_critical": []}}
            
            response = client.post("/api/v1/simulation/report", json=payload)
            assert response.status_code == 200, response.text
            mock_client.simulate.assert_called_with(layer="system", mode="report", layers=["app", "infra", "mw", "system"])
    finally:
        app.dependency_overrides = {}
