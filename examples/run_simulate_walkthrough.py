#!/usr/bin/env python3
"""
Example script that walks through Step 5, Simulate (docs/failure-simulation.md),
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

    print(f"Label artifact: {label['header']['labeler']}, fingerprint {label['header']['topology_fingerprint']}, "
          f"retest rho {label['header']['label_stability']['test_retest_spearman']}")
    for s in safety:
        print(f"  {'accepted' if s['accepted'] else 'refused '}  {s['case']}")
    print(f"Oracle: {len(oracle['single'])} single failures, {len(oracle['pairs'])} pairs, "
          f"{len(oracle['edge_criticality'])} edges measured, Monte Carlo mean {oracle['monte_carlo']['mean']}")

    write_page(TEMPLATE, args.output, {"label": label, "safety": safety, "oracle": oracle,
                                       "min_retest": MIN_LABEL_RETEST, "thresholds": list(THRESHOLDS)})
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
