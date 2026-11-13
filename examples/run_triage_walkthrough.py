#!/usr/bin/env python3
"""
Example script that walks through triage on the 15-component mini-ATM topology
and builds an interactive HTML page.

It shows the Triage Bridge (Pathway B's shortlist joined to Pathway A's
root-cause profile by component id, with a trained ranking and in cold start),
and then the two-tiered protocol on a small change: a non-blocking note from
closed-form references at pull-request time, the delta-aware gate, and a
targeted simulation at staging time.

The trained ranking uses a checkpoint from the git-ignored ``output/`` folder;
without it the page shows the cold-start bridge only.
"""

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag import Client
from saag.analysis.delta_gate import delta_gate
from saag.core.graph_io import build_graph_from_json
from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.fault_injector import RECOMMENDED_SEEDS, FaultInjector

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "triage_walkthrough.template.html"
CHECKPOINT = ROOT / "output/realworld_zeroshot/hgl_qos/seed_42"
TOP_K = 3

# The change under review: display starts publishing operator commands that
# tracker and conflict-detector consume. It closes a feedback loop.
CHANGE = {
    "topic": {"id": "T5", "name": "operator.commands", "size": 256, "frequency": 1.0,
              "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGH",
                      "deadline_ms": 1000, "history_depth": 10}},
    "publisher": "A5", "subscribers": ["A2", "A3"], "broker": "B2",
}


def r4(x):
    return None if x is None else round(float(x), 4)


def apply_change(topology: dict) -> dict:
    candidate = copy.deepcopy(topology)
    candidate["topics"].append(CHANGE["topic"])
    rel, tid = candidate["relationships"], CHANGE["topic"]["id"]
    rel["publishes_to"].append({"from": CHANGE["publisher"], "to": tid})
    rel["subscribes_to"] += [{"from": s, "to": tid} for s in CHANGE["subscribers"]]
    rel["routes"].append({"from": CHANGE["broker"], "to": tid})
    return candidate


def client_for(topology: dict) -> Client:
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    return Client(repo=repo)


def references(topology: dict) -> tuple:
    """Tier 1: the closed-form references, and how long they took."""
    from reproduce.idyn_rate_expansion import closed_forms
    from saag.prediction.structural_predictor import derive_flow_projection

    start = time.perf_counter()
    flow = derive_flow_projection(topology)
    indeg = {str(v): float(d) for v, d in flow.in_degree()}
    rate = closed_forms(topology)["Rate-I_dyn"]
    elapsed_ms = (time.perf_counter() - start) * 1000
    apps = [a["id"] for a in topology["applications"]]
    return {"indeg": {a: r4(indeg.get(a, 0.0)) for a in apps}, "rate": {a: r4(rate.get(a, 0.0)) for a in apps}}, elapsed_ms


def simulate(topology: dict) -> dict:
    """Tier 2: cascade impact of each application, five seeds."""
    records = FaultInjector(build_graph_from_json(topology), seeds=RECOMMENDED_SEEDS).run(
        node_types=["Application"]).records
    return {n: {"istar": r4(rec.impact_score), "std": r4(rec.impact_score_std)} for n, rec in records.items()}


def findings(diagnosis) -> list:
    return [p.to_dict() for p in (diagnosis.raw.problems or [])]


def brief(finding: dict) -> dict:
    return {"entity_id": finding["entity_id"], "name": finding["name"], "severity": finding["severity"],
            "fence_ratio": r4((finding.get("evidence") or {}).get("fence_ratio"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "triage_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    baseline = json.loads(TOPOLOGY.read_text())
    candidate = apply_change(baseline)

    # The bridge on the baseline: cold start (RM-ranked) and, when a checkpoint exists, GNN-ranked.
    base_client = client_for(baseline)
    base_diagnosis = base_client.diagnose(base_client.analyze(layer="system"), k=TOP_K)
    bridge = {"cold": base_diagnosis.triage.to_dict(), "trained": None}
    if CHECKPOINT.exists():
        prediction = base_client.predict(base_client.analyze(layer="app"), mode="gnn", gnn_checkpoint=str(CHECKPOINT))
        if prediction.raw.prediction_mode == "gnn_only":
            bridge["trained"] = base_client.triage(prediction, k=TOP_K).to_dict()

    cand_client = client_for(candidate)
    cand_diagnosis = cand_client.diagnose(cand_client.analyze(layer="system"), k=TOP_K)

    ref_base, _ = references(baseline)
    ref_cand, elapsed_ms = references(candidate)
    gate = delta_gate(findings(base_diagnosis), findings(cand_diagnosis))

    data = {
        "change": {"topic": CHANGE["topic"]["id"], "name": CHANGE["topic"]["name"], "publisher": CHANGE["publisher"],
                   "subscribers": CHANGE["subscribers"], "broker": CHANGE["broker"],
                   "rate": CHANGE["topic"]["frequency"]},
        "bridge": bridge,
        "bridge_after": cand_diagnosis.triage.to_dict(),
        "tier1": {"before": ref_base, "after": ref_cand, "ms": r4(elapsed_ms)},
        "absolute": {"before": base_diagnosis.raw.explanation.to_dict()["deployment_blocked"],
                     "after": cand_diagnosis.raw.explanation.to_dict()["deployment_blocked"],
                     "n_before": len(findings(base_diagnosis)), "n_after": len(findings(cand_diagnosis))},
        "gate": {"exit_code": gate.exit_code, "carried": gate.carried,
                 "blocking": [brief(f) for f in gate.blocking], "near_fence": [brief(f) for f in gate.near_fence],
                 "advisory": [brief(f) for f in gate.new_advisory], "resolved": [brief(f) for f in gate.resolved]},
        "tier2": {"before": simulate(baseline), "after": simulate(candidate)},
    }

    print(f"Tier 1 references on the candidate: {elapsed_ms:.2f} ms")
    print(f"Delta gate: exit {gate.exit_code}; {gate.carried} carried, {len(gate.blocking)} new blocking, "
          f"{len(gate.new_advisory)} new advisory")
    for app in sorted(data["tier2"]["after"]):
        before, after = data["tier2"]["before"][app]["istar"], data["tier2"]["after"][app]["istar"]
        print(f"  {app}  I* {before:.3f} -> {after:.3f}")

    write_page(TEMPLATE, args.output, data)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
