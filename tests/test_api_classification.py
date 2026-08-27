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


class TestStratifiedClassification:
    """`group_key` scores each node type against its own population.

    A single box-plot fence over a mixed population ranks every component against
    scales it does not share. Measured on the corpus, stratifying moves 62.8% of
    components to a different tier and changes CRITICAL/HIGH membership for 19.0%
    (results/tier_pooling_check.json), so this is a correctness property of the
    shipped tiers, not a reporting preference.
    """

    @staticmethod
    def _two_scale_population():
        """Apps on [0.0, 0.2], Brokers on [0.6, 0.9] — disjoint by construction."""
        apps = [
            {"id": f"a{i}", "score": 0.0 + 0.2 * i / 39.0, "type": "Application"}
            for i in range(40)
        ]
        brokers = [
            {"id": f"b{i}", "score": 0.6 + 0.3 * i / 19.0, "type": "Broker"}
            for i in range(20)
        ]
        return apps + brokers

    def test_pooling_hides_the_top_of_the_smaller_scale_population(self):
        from saag.analysis.classifier import BoxPlotClassifier

        data = self._two_scale_population()
        clf = BoxPlotClassifier()

        pooled = {i.id: i.level for i in clf.classify(data).items}
        grouped = {i.id: i.level for i in clf.classify(data, group_key="type").items}

        app_ids = [d["id"] for d in data if d["type"] == "Application"]
        pooled_app_levels = {pooled[i].value for i in app_ids}
        grouped_app_levels = {grouped[i].value for i in app_ids}

        # Pooled, the Brokers occupy the whole upper half and no Application can
        # reach HIGH regardless of how it ranks among Applications.
        assert "high" not in pooled_app_levels
        assert "high" in grouped_app_levels

    def test_grouping_preserves_every_item_and_reports_per_group_stats(self):
        from saag.analysis.classifier import BoxPlotClassifier

        data = self._two_scale_population()
        result = BoxPlotClassifier().classify(data, group_key="type")

        assert len(result.items) == len(data)
        assert sum(result.distribution.values()) == len(data)
        assert set(result.group_stats) == {"Application", "Broker"}
        # Each group's fence is derived from its own quartiles.
        assert result.group_stats["Application"].q3 < result.group_stats["Broker"].q1

    def test_ungrouped_and_undersized_items_fall_back_to_the_pooled_fence(self):
        from saag.analysis.classifier import BoxPlotClassifier

        data = self._two_scale_population()
        data.append({"id": "orphan", "score": 0.5})            # no type at all
        data.append({"id": "solo", "score": 0.5, "type": "Node"})  # group of one
        result = BoxPlotClassifier().classify(data, group_key="type")

        ids = {i.id for i in result.items}
        assert {"orphan", "solo"} <= ids
        # Too small for its own quartiles, so 'Node' gets no group statistics.
        assert "Node" not in result.group_stats

    def test_default_path_is_unchanged(self):
        from saag.analysis.classifier import BoxPlotClassifier

        data = self._two_scale_population()
        result = BoxPlotClassifier().classify(data)
        assert result.group_stats == {}
        assert "group_statistics" not in result.to_dict()
