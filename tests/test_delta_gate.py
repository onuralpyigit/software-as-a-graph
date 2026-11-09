"""Delta-aware CI gate (saag/analysis/delta_gate.py)."""
import copy
import json
from datetime import date
from pathlib import Path

import pytest

from saag.analysis.antipattern_detector import CATALOG
from saag.analysis.delta_gate import Waiver, delta_gate, finding_key, load_waivers

TODAY = date(2026, 11, 9)


def finding(pid, entity, ratio=None, entity_type="Component"):
    spec = CATALOG[pid]
    evidence = {} if ratio is None else {"fence_ratio": ratio}
    return {"entity_id": entity, "entity_type": entity_type, "category": spec.category,
            "severity": spec.severity, "name": spec.name, "description": "", "recommendation": "",
            "evidence": evidence}


def test_pre_existing_debt_does_not_block():
    base = [finding("SPOF", "A1"), finding("BOTTLENECK_EDGE", "A1->A2", 3.0, "Edge")]
    r = delta_gate(base, copy.deepcopy(base), today=TODAY)
    assert r.exit_code == 0 and r.carried == 2 and not r.blocking


def test_new_discrete_high_finding_blocks():
    r = delta_gate([], [finding("BRIDGE_EDGE", "A1->A2", entity_type="Edge")], today=TODAY)
    assert r.exit_code == 2 and len(r.blocking) == 1


def test_new_finding_on_another_entity_is_new():
    r = delta_gate([finding("SPOF", "A1")], [finding("SPOF", "A1"), finding("SPOF", "A2")], today=TODAY)
    assert [f["entity_id"] for f in r.blocking] == ["A2"]


def test_higher_severity_pattern_on_same_entity_blocks():
    # Severity is fixed per pattern, so worsening shows up as a new key.
    r = delta_gate([finding("HUB_AND_SPOKE", "A1")],
                   [finding("HUB_AND_SPOKE", "A1"), finding("FAILURE_HUB", "A1", 1.5)], today=TODAY)
    assert r.exit_code == 2 and r.blocking[0]["name"] == CATALOG["FAILURE_HUB"].name


@pytest.mark.parametrize("ratio,expected", [(1.0001, 1), (1.10, 1), (1.11, 2), (2.0, 2)])
def test_fence_margin(ratio, expected):
    r = delta_gate([], [finding("FAILURE_HUB", "A1", ratio)], margin=0.10, today=TODAY)
    assert r.exit_code == expected
    assert len(r.near_fence) == (1 if expected == 1 else 0)


def test_zero_margin_blocks_every_new_high_finding():
    r = delta_gate([], [finding("FAILURE_HUB", "A1", 1.0001)], margin=0.0, today=TODAY)
    assert r.exit_code == 2


def test_new_bottleneck_edge_warns_however_far_past_the_fence():
    r = delta_gate([], [finding("BOTTLENECK_EDGE", "A1->A2", 5.0, "Edge")], margin=0.0, today=TODAY)
    assert r.exit_code == 1 and len(r.new_advisory) == 1 and not r.blocking


def test_new_medium_is_advisory():
    r = delta_gate([], [finding("TOPIC_FANOUT", "T1")], today=TODAY)
    assert r.exit_code == 1 and len(r.new_advisory) == 1


def test_waiver_covers_new_finding_until_it_expires():
    new = [finding("SPOF", "B0")]
    live = Waiver("SPOF", "B0", date(2026, 12, 31), "sole legacy broker", "ops-lead")
    r = delta_gate([], new, waivers=[live], today=TODAY)
    assert r.exit_code == 0 and len(r.waived) == 1

    expired = Waiver("SPOF", "B0", date(2026, 11, 8), "sole legacy broker", "ops-lead")
    r = delta_gate([], new, waivers=[expired], today=TODAY)
    assert r.exit_code == 2 and r.expired_waivers == [expired]


def test_waiver_for_another_entity_does_not_apply():
    w = Waiver("SPOF", "B1", date(2027, 1, 1), "r", "a")
    assert delta_gate([], [finding("SPOF", "B0")], waivers=[w], today=TODAY).exit_code == 2


def test_resolved_findings_are_reported():
    r = delta_gate([finding("SPOF", "A1")], [], today=TODAY)
    assert r.exit_code == 0 and [f["entity_id"] for f in r.resolved] == ["A1"]


def test_negative_margin_rejected():
    with pytest.raises(ValueError):
        delta_gate([], [], margin=-0.1)


def test_load_waivers(tmp_path):
    p = tmp_path / "waivers.yaml"
    p.write_text("waivers:\n  - {pattern: SPOF, entity: B0, expires: 2027-01-01, "
                 "reason: sole legacy broker, approved_by: ops-lead}\n")
    assert load_waivers(p) == [Waiver("SPOF", "B0", date(2027, 1, 1), "sole legacy broker", "ops-lead")]


@pytest.mark.parametrize("entry,msg", [
    ({"pattern": "SPOF", "entity": "B0", "expires": "2027-01-01", "reason": "r"}, "approved_by"),
    ({"pattern": "NOPE", "entity": "B0", "expires": "2027-01-01", "reason": "r", "approved_by": "a"}, "unknown"),
])
def test_load_waivers_rejects_incomplete_entries(tmp_path, entry, msg):
    p = tmp_path / "waivers.json"
    p.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match=msg):
        load_waivers(p)


def _findings(topology):
    from saag.analysis.antipattern_detector import AntiPatternDetector
    from saag.analysis.service import AnalysisService
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.prediction.service import PredictionService

    repo = MemoryRepository()
    repo.save_graph(copy.deepcopy(topology), clear=True)
    quality = PredictionService().predict_quality(AnalysisService(repo).analyze_layer("system").structural)
    active = [p for p in CATALOG if p != "DEEP_PIPELINE"]
    return [p.to_dict() for p in AntiPatternDetector(active_patterns=active).detect(quality, layer="system")]


def test_unchanged_topology_passes_and_fence_findings_carry_a_ratio():
    topo = json.loads((Path(__file__).parent.parent / "data/scenarios/atm_system.json").read_text())
    base = _findings(topo)
    assert delta_gate(base, _findings(topo), today=TODAY).exit_code == 0
    bottlenecks = [f for f in base if finding_key(f)[0] == "BOTTLENECK_EDGE"]
    assert bottlenecks and all(f["evidence"]["fence_ratio"] > 1.0 for f in bottlenecks)
