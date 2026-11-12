#!/usr/bin/env python3
"""
Example script that walks through anti-pattern detection (Step 4) and
Prescribe (Step 7, docs/prescription.md) on the smallest corpus scenario,
``data/scenarios/atm_system_tiny.json``, and builds an interactive HTML page.

It shows which of the 19 catalogue patterns fire, how findings become
candidate edits (topic splits, host reallocations, QoS upgrades), and how each
candidate is accepted or rejected by counterfactual simulation.
"""

import argparse
import collections
import json
import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag import Client
from saag.analysis.antipattern_detector import CATALOG
from saag.infrastructure.memory_repo import MemoryRepository

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = ROOT / "data/scenarios/atm_system_tiny.json"
TEMPLATE = SCRIPT_DIR / "prescribe_walkthrough.template.html"
#: Catalogue ids whose findings feed an automated operator (docs/prescription.md, section 2).
OPERATOR_OF = {"SPOF": "node_reallocation", "GOD_COMPONENT": "topic_split", "BOTTLENECK_EDGE": "topic_split",
               "FAILURE_HUB": "topic_split", "HUB_AND_SPOKE": "topic_split"}
SAMPLE = 6


def r6(x):
    return None if x is None else round(float(x), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "prescribe_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    topology = json.loads(TOPOLOGY.read_text())
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="system")
    prediction = client.predict(analysis, mode="rm")          # RM scores plus the anti-pattern scan
    quality = prediction.raw

    id_of = {spec.name: pid for pid, spec in CATALOG.items()}
    found = collections.defaultdict(list)
    for problem in quality.problems or []:
        found[id_of[problem.name]].append(problem.entity_id)
    catalog = [{"id": pid, "name": spec.name, "severity": str(getattr(spec.severity, "value", spec.severity)),
                "category": str(getattr(spec.category, "value", spec.category)), "description": spec.description,
                "recommendation": spec.recommendation, "count": len(found[pid]), "sample": found[pid][:SAMPLE],
                "operator": OPERATOR_OF.get(pid)} for pid, spec in CATALOG.items()]

    result = client.prescribe(analysis, prediction, layer="system").to_dict()
    names = {e["id"]: e["name"] for key in ("nodes", "brokers", "topics", "applications", "libraries")
             for e in topology[key]}
    # Verdicts come back in the order the candidates were compiled, so each one can be
    # matched to the edit it judged.
    proposed = result["candidate_policy"]
    details = ([("topic_split", e["topic"], e) for e in proposed["topic_splits"]]
               + [("node_reallocation", e["component"], e) for e in proposed["node_reallocations"]]
               + [("qos_upgrade", e["topic"], e) for e in proposed["qos_upgrades"]])
    assert [(k, t) for k, t, _ in details] == [(e["kind"], e["target"]) for e in result["edit_verdicts"]]
    report = quality.explanation.to_dict()
    data = {
        "scenario": {"file": str(TOPOLOGY.relative_to(ROOT)), "scale": topology["metadata"]["scale"], "names": names},
        "catalog": catalog,
        "failed_patterns": list(getattr(quality, "failed_patterns", []) or []),
        "levels": quality.classification_summary.component_distribution,
        "report": {k: report[k] for k in ("critical_count", "high_count", "deployment_blocked", "reason")},
        "edits": [{"kind": e["kind"], "target": e["target"], "accepted": e["accepted"], "reason": e["reason"],
                   "kappa": e["kappa"], "worst_delta": r6(e["worst_delta"]), "worst_margin": r6(e["worst_margin"]),
                   "per_threshold": {t: {k: r6(v) for k, v in row.items()} for t, row in e["per_threshold"].items()},
                   "detail": detail}
                  for e, (_, _, detail) in zip(result["edit_verdicts"], details)],
        "candidate_policy": proposed, "policy": result["policy"],
        "applied": result["applied_changes"], "accepted": result["accepted"],
        "sri": [result["original_sri"], result["mutated_sri"]],
        "metrics": {"before": result["original_metrics"], "after": result["mutated_metrics"]},
    }

    fired = [c for c in catalog if c["count"]]
    print(f"{len(fired)} of {len(catalog)} catalogue patterns fired:")
    for c in fired:
        print(f"  {c['id']:<20} {c['severity']:<9} {c['count']}")
    kinds = collections.Counter((e["kind"], e["accepted"]) for e in data["edits"])
    for kind in sorted({k for k, _ in kinds}):
        print(f"  {kind:<18} {kinds[(kind, True)]} accepted, {kinds[(kind, False)]} rejected")
    print(f"SRI {data['sri'][0]:.4f} -> {data['sri'][1]:.4f}")

    write_page(TEMPLATE, args.output, data)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
