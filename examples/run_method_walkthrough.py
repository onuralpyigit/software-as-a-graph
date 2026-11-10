#!/usr/bin/env python3
"""
Example script that runs every stage of the SaG pipeline on the 15-component
mini-ATM topology through the SDK (Model, Analyze, Predict, Diagnose with the
triage bridge, Simulate, Validate, Prescribe) and builds an interactive HTML
walkthrough of what each stage produced.

Stage 3 uses a trained HGT-QoS checkpoint from the git-ignored ``output/``
folder (see CHECKPOINT below; pass ``--checkpoint`` to use another).
"""

import argparse
import dataclasses
import json
import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag import Client
from saag.core.graph_io import load_graph
from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.fault_injector import RECOMMENDED_SEEDS, FaultInjector
from saag.simulation.service import SimulationService

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "sag_method_walkthrough.template.html"
CHECKPOINT = ROOT / "output/realworld_zeroshot/hgl_qos/seed_42"
PAGE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
             '<style>body{margin:0}[hidden]{display:none!important}</style>\n')

METRIC_KEYS = ["reverse_pagerank", "in_degree_raw", "out_degree_raw", "betweenness", "ap_c_directed", "cdi",
               "blast_radius", "dependency_weight_in", "dependency_weight_out", "clustering_coefficient"]
DIMENSIONS = ("fault_tolerance", "availability", "reliability", "maintainability", "overall")
TOP_K = 3


def r4(x):
    return None if x is None else round(float(x), 4)


def stage_model(repo) -> dict:
    """Stage 1: components with QoS-derived weights, raw edges and derived DEPENDS_ON edges."""
    graph = repo.get_graph_data(include_raw=True)
    return {
        "components": {c.id: {"type": c.component_type, "w": r4(c.weight)} for c in graph.components},
        "raw": [[e.source_id, e.target_id, e.relation_type, r4(e.weight)]
                for e in graph.edges if e.relation_type != "DEPENDS_ON"],
        "dep": [[e.source_id, e.target_id, e.dependency_type, r4(e.weight)]
                for e in graph.edges if e.relation_type == "DEPENDS_ON"],
    }


def stage_analyze(analysis) -> dict:
    """Stage 2: a few fields of the structural metric vector M(v), and the graph summary."""
    structural = analysis.raw.structural
    summary = dataclasses.asdict(structural.graph_summary)
    return {
        "keys": METRIC_KEYS,
        "metrics": {i: {k: r4(getattr(m, k)) for k in METRIC_KEYS} for i, m in structural.components.items()},
        "summary": {k: r4(v) for k, v in summary.items() if isinstance(v, (int, float))},
    }


def stage_predict(client, topology: dict, checkpoint: Path) -> tuple:
    """Stage 3 (Pathway B): the trained ranker on the app layer, closed-form references, and triage."""
    from reproduce.referee_round7 import raw_rankers

    prediction = client.predict(client.analyze(layer="app"), mode="gnn", gnn_checkpoint=str(checkpoint))
    if prediction.raw.prediction_mode != "gnn_only":
        sys.exit(f"GNN inference did not run (prediction_mode = {prediction.raw.prediction_mode}).")
    rankers = raw_rankers(topology)
    predict = {
        "mode": prediction.raw.prediction_mode,
        "gnn": {c.id: r4(c.rm_score) for c in prediction.all_components},
        "refs": {k: {a: r4(v) for a, v in rankers[k].items()} for k in ("Reach", "InDeg", "Analytic-I*")},
    }
    return predict, client.triage(prediction, k=TOP_K).to_dict()


def stage_diagnose(client, analysis) -> dict:
    """Stage 4 (Pathway A): RM dimension scores, anti-patterns, the report and RM-ranked triage."""
    diagnosis = client.diagnose(analysis, k=TOP_K)
    quality = diagnosis.raw
    report = quality.explanation.to_dict()
    return {
        "components": [{"id": c.id, "type": c.type, **{k: r4(getattr(c.scores, k)) for k in DIMENSIONS},
                        "level": c.levels.overall.value} for c in quality.components],
        "weights": {k: r4(getattr(quality.weights, k)) for k in ("r_alpha", "q_reliability", "q_maintainability")},
        "problems": [{k: p.to_dict()[k] for k in ("entity_id", "severity", "name", "category", "recommendation")}
                     for p in (quality.problems or [])],
        "report": {k: report[k] for k in ("total_components", "critical_count", "high_count",
                                          "deployment_blocked", "reason")},
        "explain": [{k: e[k] for k in ("component_id", "pattern", "level", "one_line", "top_risk")}
                    for e in report["component_explanations"]],
        "triage_rm": diagnosis.triage.to_dict(),
    }


def stage_simulate(repo) -> dict:
    """Stage 5: I*(v) from the Predict-stage labeler and I_comp(v) from the Validate-stage oracle."""
    labels = FaultInjector(load_graph(TOPOLOGY), seeds=RECOMMENDED_SEEDS).run(
        node_types=["Application", "Broker", "Library", "Node"]).records
    sweep = SimulationService(repo).run_failure_simulation_exhaustive(layer="system")
    return {
        "istar": {n: r4(rec.impact_score) for n, rec in labels.items()},
        "icomp": {r.target_id: {"comp": r4(r.impact.composite_impact), "IR": r4(r.impact.reliability_impact),
                                "IM": r4(r.impact.maintainability_impact)} for r in sweep},
    }


def stage_validate(client) -> dict:
    """Stage 6: Q(v) against I_comp(v), with the gate verdicts, for the system and app layers."""
    result = client.validate(layers=["system", "app"]).to_dict()

    def layer(name: str) -> dict:
        data = result["layers"][name]
        overall = data["validation_result"]["overall"]["metrics"]
        return {
            "passed": data["passed"], "summary": data["summary"], "gates": data["gates"],
            "n": data["validation_result"]["data_alignment"], "ci": overall["correlation"]["spearman_ci"],
            "strat": data["node_type_stratified"], "warnings": data["warnings"],
        }

    return {"targets": result["targets"], "system": layer("system"), "app": layer("app")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "sag_method_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="GNN checkpoint directory for Stage 3")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    if not args.checkpoint.exists():
        sys.exit(f"GNN checkpoint not found: {args.checkpoint}\nPass --checkpoint, or produce one first "
                 "(see reproduce/README.md).")

    topology = json.loads(TOPOLOGY.read_text())
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="system")

    data = {"model": stage_model(repo), "analyze": stage_analyze(analysis)}
    data["predict"], data["triage_gnn"] = stage_predict(client, topology, args.checkpoint)
    data["diagnose"] = stage_diagnose(client, analysis)
    data["simulate"] = stage_simulate(repo)
    data["validate"] = stage_validate(client)
    # Stage 7: candidate edits, each verified by counterfactual simulation.
    data["prescribe"] = client.prescribe(analysis, client.predict(analysis, mode="rm"), layer="system").to_dict()

    system, prescribe = data["validate"]["system"], data["prescribe"]
    print(f"1 Model      {len(data['model']['dep'])} derived DEPENDS_ON edges")
    print(f"3 Predict    top {TOP_K}: " + ", ".join(e["component_id"] for e in data["triage_gnn"]["entries"]))
    print(f"4 Diagnose   {data['diagnose']['report']['critical_count']} critical, "
          f"{data['diagnose']['report']['high_count']} high, {len(data['diagnose']['problems'])} anti-patterns")
    print(f"6 Validate   system rho = {system['summary']['spearman']:.3f} (n = {system['n']['matched_count']}), "
          f"layer passed: {system['passed']}")
    print(f"7 Prescribe  {prescribe['n_accepted_edits']} of {prescribe['n_candidate_edits']} edits accepted, "
          f"SRI {prescribe['original_sri']:.4f} -> {prescribe['mutated_sri']:.4f}")

    page = TEMPLATE.read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":"), default=str))
    args.output.write_text(PAGE_HEAD + page)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
