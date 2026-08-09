"""
Tests for the classification API router (POST /api/v1/classify).
"""
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from saag.core.models import GraphData, ComponentData, EdgeData

client = TestClient(app)

CREDENTIALS = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "pw"}


def _graph_data() -> GraphData:
    return GraphData(
        components=[
            ComponentData(id="App1", component_type="Application"),
            ComponentData(id="App2", component_type="Application"),
            ComponentData(id="App3", component_type="Application"),
        ],
        edges=[
            EdgeData(source_id="App1", target_id="App2", source_type="Application",
                     target_type="Application", relation_type="DEPENDS_ON",
                     dependency_type="app_to_app", weight=1.0),
            EdgeData(source_id="App2", target_id="App3", source_type="Application",
                     target_type="Application", relation_type="DEPENDS_ON",
                     dependency_type="app_to_app", weight=1.0),
        ],
    )


def test_classify_endpoint_returns_200_not_500():
    """Regression guard: the router used to call
    BoxPlotClassifier.classify_scores(values, item_type=..., metric_name=...),
    a method that does not exist (the real method is classify(data, ...),
    taking a list of {id, score} dicts, not a values dict) — every request to
    this endpoint raised AttributeError -> 500. Uses a real StructuralAnalyzer
    + BoxPlotClassifier, only the repository is faked, so this exercises the
    actual call the router makes rather than mocking it away.
    """
    mock_repo = MagicMock()
    mock_repo.get_graph_data.return_value = _graph_data()

    with patch("api.routers.classification.create_repository", return_value=mock_repo):
        response = client.post("/api/v1/classify", json=CREDENTIALS)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert set(body["classifications"]) == {"betweenness", "pagerank", "degree"}
    for metric_result in body["classifications"].values():
        assert set(metric_result["statistics"]) == {
            "min_val", "max_val", "median", "q1", "q3", "iqr", "upper_fence",
        }
        assert isinstance(metric_result["distribution"], dict)
        assert {c["id"] for c in metric_result["components"]} == {"App1", "App2", "App3"}
    assert {r["id"] for r in body["merged_ranking"]} == {"App1", "App2", "App3"}
