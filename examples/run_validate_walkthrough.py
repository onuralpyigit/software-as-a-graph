#!/usr/bin/env python3
"""
Example script that walks through Step 6, Validate (docs/validation.md), on the
15-component mini-ATM topology and builds an interactive HTML page.

It follows the validation of the quality model's Q(v) against the
FailureSimulator's impact step by step: aligning the two populations, rank
correlation, its bootstrap uncertainty, how well the critical components are
caught, stratification by type, the dimension-level comparisons, the gates and
the system health indices.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import spearmanr

from saag import Client
from saag.infrastructure.memory_repo import MemoryRepository
from saag.validation.models import RELEASE_GATES, REPORTED_GATES

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "validate_walkthrough.template.html"
LAYERS = ("system", "app")
BOOTSTRAP_DRAWS = 2000
DIMENSIONS = ("reliability", "maintainability", "availability")


def r4(x):
    return None if x is None else round(float(x), 4)


def bootstrap_rhos(pairs: list, draws: int = BOOTSTRAP_DRAWS, seed: int = 42) -> list:
    """Spearman rho on resampled component sets, for the picture of the interval.

    The engine reports its own interval; this redraws the distribution behind
    one so that its width can be seen. A draw with no variation is skipped.
    """
    rng, x, y = np.random.default_rng(seed), np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    out = []
    for _ in range(draws):
        idx = rng.integers(0, len(pairs), len(pairs))
        if len(set(x[idx])) > 1 and len(set(y[idx])) > 1:
            out.append(float(spearmanr(x[idx], y[idx])[0]))
    return out


def layer_view(result, summary: dict) -> dict:
    overall = summary["validation_result"]["overall"]["metrics"]
    comparisons = [{"id": c.id, "type": c.type, "predicted": r4(c.predicted), "actual": r4(c.actual),
                    "cls": c.classification} for c in result.comparisons]
    rhos = bootstrap_rhos([(c["predicted"], c["actual"]) for c in comparisons])
    hist, edges = np.histogram(rhos, bins=20, range=(-1.0, 1.0))
    return {
        "name": summary["layer_name"], "passed": summary["passed"], "gates": summary["gates"],
        "counts": summary["validation_result"]["data_alignment"], "warnings": summary["warnings"],
        "comparisons": comparisons,
        "correlation": {k: r4(overall["correlation"][k]) for k in ("spearman", "kendall", "pearson", "spearman_p_value")},
        "ci": [r4(v) for v in overall["correlation"]["spearman_ci"]],
        "bootstrap": {"hist": hist.tolist(), "edges": [r4(e) for e in edges], "n": len(rhos),
                      "lo": r4(np.percentile(rhos, 2.5)), "hi": r4(np.percentile(rhos, 97.5))},
        "classification": {**{k: r4(overall["classification"][k]) for k in ("precision", "recall", "f1_score", "accuracy")},
                           "confusion": overall["classification"]["confusion_matrix"]},
        "top5": overall["ranking"]["top_5_agreement"],
        "values": {"spearman": r4(result.spearman), "overlap_at_q3": r4(result.f1_score),
                   "top5_overlap": r4(result.top_5_overlap), "predictive_gain": r4(result.predictive_gain),
                   "kappa_cta": r4(result.dimensional_validation.get("maintainability", {}).get("weighted_kappa_cta")),
                   "bottleneck_precision": r4(result.dimensional_validation.get("maintainability", {})
                                              .get("bottleneck_precision"))},
        "by_type": {t: {"n": v["sample_size"], "spearman": r4(v["metrics"]["correlation"]["spearman"])}
                    for t, v in summary["validation_result"]["by_type"].items()},
        "stratified": summary["node_type_stratified"],
        "dimensions": {d: {"points": [[p[0], r4(p[1]), r4(p[2])] for p in pts],
                           "spearman": r4(result.dimensional_validation[d]["spearman"]),
                           "truth": result.dimensional_validation[d]["ground_truth"],
                           "ci": [r4(v) for v in result.confidence_intervals[d]]}
                       for d, pts in result.dimensional_scatter.items() if d in DIMENSIONS},
        "composite": {k: (r4(v) if isinstance(v, (int, float)) else v)
                      for k, v in result.dimensional_validation["composite"].items() if k != "system_health"},
        "composite_ci": [r4(v) for v in result.confidence_intervals["composite"]],
        "health": result.system_health,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "validate_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    repo = MemoryRepository()
    repo.save_graph(json.loads(TOPOLOGY.read_text()), clear=True)
    repo.derive_dependencies()
    validation = Client(repo=repo).validate(layers=list(LAYERS))
    summary = validation.to_dict()

    data = {
        "targets": summary["targets"], "release_gates": list(RELEASE_GATES), "reported_gates": list(REPORTED_GATES),
        "layers": {name: layer_view(validation.raw.layers[name], summary["layers"][name]) for name in LAYERS},
    }
    for name, layer in data["layers"].items():
        gates = ", ".join(f"{g}={'n/m' if v is None else 'pass' if v else 'fail'}" for g, v in layer["gates"].items())
        print(f"{name:<7} n={layer['counts']['matched_count']:<3} rho={layer['correlation']['spearman']:+.3f} "
              f"CI={layer['ci']}  passed={layer['passed']}\n        {gates}")

    write_page(TEMPLATE, args.output, data)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
