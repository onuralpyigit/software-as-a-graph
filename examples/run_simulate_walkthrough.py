#!/usr/bin/env python3
"""
Example script that walks through Step 5, Simulate (docs/simulation.md),
as a workflow on the 15-component mini-ATM topology and builds an interactive
HTML page.

run_simulator_walkthrough.py shows what each engine does to one failed
component. This script shows how simulation is used: producing a label
artifact and what its options change, the checks an artifact must pass before
it is trained on, and the oracle's run modes (exhaustive, pairwise, named
events, Monte Carlo and edge criticality).
"""

import argparse
import copy
import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from saag.core.graph_io import load_graph
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prediction.label_artifacts import (
    MIN_LABEL_RETEST, LabelArtifactError, check_stability, check_topology, select_labels)
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.fault_injector import RECOMMENDED_SEEDS, FaultInjector
from saag.simulation.service import SimulationService
from saag.testbed.hybrid import HybridProtocolConfig, run_hybrid_protocol

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "simulate_walkthrough.template.html"
SEEDS = list(RECOMMENDED_SEEDS)
LABELLED = ["Application", "Broker", "Library"]
THRESHOLDS = (0.1, 0.2, 0.5, 1.0)
EVENTS = {"both brokers": ["B1", "B2"], "sensor side": ["A1", "A2"], "both consumers": ["A5", "A6"]}
MONTE_CARLO = {"target": "A2", "cascade_probability": 0.5, "n_trials": 300}


def r4(x):
    return None if x is None else round(float(x), 4)


def labels(graph, **options) -> dict:
    records = FaultInjector(graph, seeds=SEEDS, **options).run(node_types=LABELLED).records
    return {n: {"score": r4(rec.impact_score), "std": r4(rec.impact_score_std)} for n, rec in records.items()}


def label_stage(graph) -> dict:
    """The canonical artifact, and how three options move the labels."""
    artifact = FaultInjector(graph, seeds=SEEDS).run(node_types=LABELLED).to_dict()
    cdf = FaultInjector(graph, seeds=SEEDS, failure_model="threshold_cdf").run(node_types=LABELLED)
    return {
        "artifact": artifact,
        "header": {k: artifact[k] for k in ("labeler", "schema_version", "graph_id", "topology_fingerprint",
                                            "seeds_used", "labeled_node_types", "unlabeled_node_ids",
                                            "degenerate_node_types", "label_stability")},
        "canonical": {n: {"score": r4(rec["impact_score"]), "std": r4(rec["impact_score_std"])}
                      for n, rec in artifact["records"].items()} if isinstance(artifact["records"], dict)
        else {rec["node_id"]: {"score": r4(rec["impact_score"]), "std": r4(rec["impact_score_std"])}
              for rec in artifact["records"]},
        "threshold": {str(t): labels(graph, propagation_threshold=t) for t in THRESHOLDS},
        "qos": {mode: labels(graph, qos_factor_mode=mode) for mode in FaultInjector.QOS_FACTOR_MODES},
        "cdf": {"labels": {n: {"score": r4(rec.impact_score), "std": r4(rec.impact_score_std),
                               "seeds": {str(s): r4(v) for s, v in rec.seed_impact_scores.items()}}
                           for n, rec in cdf.records.items()},
                "stability": cdf.label_stability},
    }


def safety_stage(artifact: dict, graph) -> list:
    """Offer the label checks one genuine artifact and four that must be refused."""
    other = load_graph(ROOT / "data/scenarios/atm_system_tiny.json")
    wrong_labeler = {**artifact, "labeler": "FailureSimulator"}
    unstable = copy.deepcopy(artifact)
    unstable["label_stability"]["test_retest_spearman"] = 0.41

    def attempt(what, check):
        try:
            check()
            return {"case": what, "accepted": True, "message": ""}
        except LabelArtifactError as exc:
            return {"case": what, "accepted": False, "message": str(exc)}

    return [
        attempt("The artifact as written, on the topology it was measured on",
                lambda: (select_labels(artifact), check_topology(artifact, graph), check_stability(artifact))),
        attempt("The same artifact offered for a different topology", lambda: check_topology(artifact, other)),
        attempt("An artifact written by the Validate-stage oracle", lambda: select_labels(wrong_labeler)),
        attempt("A label the artifact does not carry", lambda: select_labels(artifact, label="i_flow")),
        attempt("Labels that agree with themselves at rho = 0.41 across seeds", lambda: check_stability(unstable)),
    ]


def oracle_stage(topology: dict) -> dict:
    """The FailureSimulator's run modes through SimulationService."""
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    service = SimulationService(repo)

    single = {r.target_id: r4(r.impact.composite_impact) for r in service.run_failure_simulation_exhaustive(layer="system")}
    pairs = {r.target_id: r4(r.impact.composite_impact) for r in service.run_failure_simulation_pairwise(layer="system")}
    events = {name: r4(r.impact.composite_impact) for name, r in service.run_failure_events(EVENTS).items()}
    mc = service.run_failure_simulation_monte_carlo(
        MONTE_CARLO["target"], layer="system", cascade_probability=MONTE_CARLO["cascade_probability"],
        n_trials=MONTE_CARLO["n_trials"])
    hist, edges = np.histogram(mc.trial_impacts, bins=16)
    sweep = FailureSimulator(service._get_graph()).simulate_edge_removal_sweep()
    return {
        "single": single, "pairs": pairs, "events": events, "event_members": EVENTS,
        "monte_carlo": {**MONTE_CARLO, "mean": r4(mc.mean_impact), "std": r4(mc.std_impact),
                        "ci": [r4(v) for v in mc.ci_95], "hist": hist.tolist(), "edges": [r4(e) for e in edges],
                        "deterministic": single[MONTE_CARLO["target"]]},
        "edge_criticality": [{"source": e.source, "target": e.target, "relationship": e.relationship,
                              "impact": r4(e.combined_impact), "level": e.level, "evaluated": e.evaluated}
                             for e in sweep],
    }


def hybrid_stage() -> dict:
    """Run the 4-step Tier 1 + Tier 2 hybrid protocol on mini_atm (fast local mode)."""
    cfg = HybridProtocolConfig(
        duration_s=5.0,
        warmup_s=0.5,
        tier1_duration_s=6.0,
        tier1_seeds=(42, 123),
        tier1_label="i_flow",
        behavior_model="reactive",
        qos_semantics="dds",
        mode="local",
        top_k=3,
        n_disagreements=0,
        n_controls=2,
        repetitions=1,
        base_seed=42,
        fault_probes=({"mode": "crash"},),
    )
    with tempfile.TemporaryDirectory(prefix="saag_hybrid_walk_") as tmp:
        rep = run_hybrid_protocol(TOPOLOGY, Path(tmp), config=cfg)

    s0 = rep["step0_calibration"]
    s1 = rep["step1_tier1_screening"]
    s2 = rep["step2_shortlist"]
    crash = rep["step3_tier2_verification"]["probes"]["crash"]
    bh = s0["baseline_health"]["observed"]
    ranking = []
    for row in crash["operational_criticality"]["ranking"]:
        tid = row["target"]
        tdata = crash["per_target"].get(tid, {})
        ha = tdata.get("health_agreement") or {}
        oh = (tdata.get("health_summary") or {}).get("observed") or {}
        ranking.append({
            "target": tid,
            "type": row.get("target_type"),
            "is_control": bool(tdata.get("is_control", False)),
            "static_crit": (row.get("static_criticality") or {}).get("level") or "LOW",
            "static_prov": (row.get("static_criticality") or {}).get("provenance") or "default",
            "i_flow": r4(row.get("i_flow")),
            "i_delivery": r4(row.get("i_delivery")),
            "i_delivery_consumer": r4(row.get("i_delivery_consumer")),
            "i_timeliness": r4(row.get("i_timeliness")),
            "qiu_impact": r4(row.get("qiu_impact")),
            "op_crit_lost": int(row.get("lost_operational_critical_count", 0)),
            "counts": (oh.get("system_health") or {}).get("counts"),
            "health_agree": r4(ha.get("exact_state_agreement")),
        })
    return {
        "calibrated": bool(s0["calibrated"]),
        "n_tier1_candidates": s1["n_candidates"],
        "tier1_label": s1["label"],
        "baseline_counts": bh["system_health"]["counts"],
        "baseline_op_crit": bh["system_health"]["operational_critical"]["counts"],
        "baseline_qiu": r4(bh["product_in_use"]["overall_qiu_health_index"]),
        "shortlist": {
            "primary": s2["primary_targets"],
            "controls": s2["control_targets"],
        },
        "ranking": ranking,
        "tier1_vs_tier2": crash["operational_criticality"]["tier1_vs_tier2"],
        "replay_fidelity": crash["replay_fidelity"],
        "caveats": rep["caveats"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "simulate_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    topology = json.loads(TOPOLOGY.read_text())
    graph = load_graph(TOPOLOGY)
    label = label_stage(graph)
    safety = safety_stage(label.pop("artifact"), graph)
    oracle = oracle_stage(topology)
    hybrid = hybrid_stage()

    print(f"Label artifact: {label['header']['labeler']}, fingerprint {label['header']['topology_fingerprint']}, "
          f"retest rho {label['header']['label_stability']['test_retest_spearman']}")
    for s in safety:
        print(f"  {'accepted' if s['accepted'] else 'refused '}  {s['case']}")
    print(f"Oracle: {len(oracle['single'])} single failures, {len(oracle['pairs'])} pairs, "
          f"{len(oracle['edge_criticality'])} edges measured, Monte Carlo mean {oracle['monte_carlo']['mean']}")
    print(f"Hybrid Tier 1+2: {hybrid['n_tier1_candidates']} Tier-1 screened, "
          f"{len(hybrid['ranking'])} Tier-2 verified, baseline QiU={hybrid['baseline_qiu']}")

    write_page(TEMPLATE, args.output, {"label": label, "safety": safety, "oracle": oracle,
                                       "hybrid": hybrid,
                                       "min_retest": MIN_LABEL_RETEST, "thresholds": list(THRESHOLDS)})
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
