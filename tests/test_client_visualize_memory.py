"""Client.visualize must simulate and validate on any repository, not only Neo4j.

It once built a SimulationService only when the repository had a Neo4j driver,
so an in-memory report silently lost its simulation and validation sections
(and showed the unmeasured validation as 0.000).
"""
import json
import logging
from pathlib import Path

from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository

MINI_ATM = Path(__file__).resolve().parent.parent / "examples" / "mini_atm.json"


def test_visualize_simulates_and_validates_on_memory_repository(tmp_path, caplog):
    repo = MemoryRepository()
    repo.save_graph(json.loads(MINI_ATM.read_text()), clear=True)
    repo.derive_dependencies()
    out = tmp_path / "report.html"

    with caplog.at_level(logging.ERROR):
        Client(repo=repo).visualize(output=str(out), layers=["system"])

    assert out.exists()
    failures = [r.getMessage() for r in caplog.records
                if "Simulation failed" in r.getMessage() or "Failed to validate" in r.getMessage()]
    assert failures == []


def test_report_spof_count_matches_spof_findings():
    """The report's SPOF card and badges count the catalog's SPOF findings.

    They once counted undirected articulation points only, so mini-ATM showed
    "SPOFs detected 0" next to a SPOF finding on the display (a directed AP).
    """
    from saag.analysis.service import AnalysisService
    from saag.prediction.service import PredictionService
    from saag.simulation.service import SimulationService
    from saag.visualization.collector import LayerDataCollector

    repo = MemoryRepository()
    repo.save_graph(json.loads(MINI_ATM.read_text()), clear=True)
    repo.derive_dependencies()
    data = LayerDataCollector(AnalysisService(repo), PredictionService(), SimulationService(repo), None, repo).collect_layer_data(
        "system", include_validation=False)

    spof_ids = {s["entity_id"] for s in data.anti_patterns if s["name"] == "Single Point of Failure (SPOF)"}
    assert spof_ids == {"A5"}  # the display: a directed articulation point, not an undirected one
    assert data.spof_count == 1
    assert {d.id for d in data.component_details if d.spof} == spof_ids

    # Each component lists the catalog IDs of the findings that name it. Real-time
    # findings carry entity_id and a display name, which the lists once ignored.
    lists = {d.id: d.anti_patterns for d in data.component_details}
    assert lists["A5"] == ["SPOF"]
    assert lists["T2"] == ["ISOLATED"]
    assert lists["A1"] == ["QOS_MISMATCH"]  # radar (w=0.15) publishes to tracker/A6 (w>=0.47)
    assert lists["A4"] == []


def test_edge_findings_attach_to_both_endpoints():
    """An Edge-scoped finding (u->v, or u<->v for a chatty pair) names both endpoints.

    At the mw layer mini-ATM has bridges A1->B1 and A4->B2; the brokers are the
    reported components there, so each must list BRIDGE_EDGE next to its own findings.
    """
    from saag.analysis.service import AnalysisService
    from saag.prediction.service import PredictionService
    from saag.simulation.service import SimulationService
    from saag.visualization.collector import LayerDataCollector

    repo = MemoryRepository()
    repo.save_graph(json.loads(MINI_ATM.read_text()), clear=True)
    repo.derive_dependencies()
    data = LayerDataCollector(AnalysisService(repo), PredictionService(), SimulationService(repo), None, repo
                              ).collect_layer_data("mw", include_validation=False)

    lists = {d.id: d.anti_patterns for d in data.component_details}
    assert "BRIDGE_EDGE" in lists["B1"]
    assert "BRIDGE_EDGE" in lists["B2"]
    assert "SPOF" in lists["B1"]
