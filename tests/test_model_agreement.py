"""reproduce/model_agreement.py: inter-modeler Jaccard on names, not IDs."""

import copy

import pytest

from reproduce.model_agreement import agreement

MODEL = {
    "applications": [{"id": "A0", "name": "device-service"}, {"id": "A1", "name": "core-data"}],
    "topics": [{"id": "T0", "name": "events.readings"}],
    "brokers": [{"id": "B0", "name": "bus"}],
    "nodes": [], "libraries": [],
    "relationships": {
        "publishes_to": [{"from": "A0", "to": "T0"}],
        "subscribes_to": [{"from": "A1", "to": "T0"}],
        "routes": [{"from": "B0", "to": "T0"}],
    },
}


def test_identical_models_agree_fully():
    r = agreement(MODEL, copy.deepcopy(MODEL))
    assert r["entity_jaccard"] == 1.0 and r["edge_jaccard"] == 1.0


def test_ids_do_not_need_to_match():
    other = copy.deepcopy(MODEL)
    for e in other["applications"]:
        e["id"] = "X" + e["id"]
    for rel in other["relationships"].values():
        for edge in rel:
            edge["from"] = "X" + edge["from"] if edge["from"].startswith("A") else edge["from"]
    assert agreement(MODEL, other)["edge_jaccard"] == 1.0


def test_disjoint_models_agree_nowhere():
    other = copy.deepcopy(MODEL)
    for etype in ("applications", "topics", "brokers"):
        for e in other[etype]:
            e["name"] += "-other"
    r = agreement(MODEL, other)
    assert r["entity_jaccard"] == 0.0 and r["edge_jaccard"] == 0.0


def test_alias_reconciles_a_renamed_entity():
    other = copy.deepcopy(MODEL)
    other["applications"][1]["name"] = "CoreDataService"
    other["relationships"]["subscribes_to"].append({"from": "A0", "to": "T0"})
    without = agreement(MODEL, other)
    assert without["entities"]["applications"]["jaccard"] == pytest.approx(1 / 3)
    r = agreement(MODEL, other, alias={"CoreDataService": "core-data"})
    assert r["entities"]["applications"]["jaccard"] == 1.0
    # One extra subscription in the second model: 1 shared of 2.
    assert r["edges"]["subscribes_to"]["jaccard"] == pytest.approx(0.5)
