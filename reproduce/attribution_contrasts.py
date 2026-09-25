#!/usr/bin/env python3
"""
reproduce/attribution_contrasts.py — where the learned engines' accuracy comes from
===================================================================================

PREREGISTRATION.md Amendment 7 (post hoc, exploratory). Reads the one-sweep
artifact of ``make rq-attribution`` and reports the five contrasts of the
manuscript's attribution table, Holm-corrected across the five:

  * gl_full_qos16_cap    vs gl_full_qos16_nfmask   QoS node columns, edge channel held
  * gl_full_qos16_nfmask vs gl_full_cap            QoS edge channel, node features held
  * tab_gbm_qos          vs tab_gbm                QoS node columns, no graph model
  * gl_full_cap          vs tab_gbm                neural vs trees, QoS-off features
  * gl_full_qos16_cap    vs tab_gbm_qos            neural vs trees, QoS-on features

plus each arm's median within-fold seed spread. Statistics are
``loso_significance.compare`` (two-sided Wilcoxon over folds, fold-bootstrap CI),
so the numbers are computed exactly as the registered tables' are.

Usage:
    PYTHONPATH=. python reproduce/attribution_contrasts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp
from reproduce.loso_significance import compare, holm

CONTRASTS = (
    ("gl_full_qos16_cap", "gl_full_qos16_nfmask", "QoS node columns, edge channel held"),
    ("gl_full_qos16_nfmask", "gl_full_cap", "QoS edge channel, node features held"),
    ("tab_gbm_qos", "tab_gbm", "QoS node columns, no graph model"),
    ("gl_full_cap", "tab_gbm", "neural vs trees, QoS-off features"),
    ("gl_full_qos16_cap", "tab_gbm_qos", "neural vs trees, QoS-on features"),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("results/loso_attribution_cpu.json"))
    p.add_argument("--output", type=Path, default=Path("results/attribution_contrasts.json"))
    args = p.parse_args()

    artifact = json.loads(args.input.read_text())
    table = artifact["comparison_table"]
    missing = sorted({v for a, b, _ in CONTRASTS for v in (a, b)} - set(table))
    if missing:
        print(f"Error: {args.input} lacks {missing}", file=sys.stderr)
        return 2

    results = []
    for a, b, quantity in CONTRASTS:
        r = compare(table, a, b)
        r["quantity"] = quantity
        results.append(r)
    holm(results)

    seed_spread = {
        v: float(np.median([f["std_metrics"]["spearman_rho"] for f in rows["folds"]]))
        for v, rows in artifact["per_variant_results"].items()
    }

    out = {
        "provenance": stamp(input=str(args.input)),
        "input_provenance": artifact.get("provenance"),
        "note": "Amendment 7: post hoc and exploratory; Holm across these five contrasts only.",
        "mean_rho": {v: float(t["mean_rho"]) for v, t in table.items()},
        "median_seed_sd": seed_spread,
        "contrasts": results,
    }
    args.output.write_text(json.dumps(out, indent=2))

    for r in results:
        lo, hi = r["delta_ci95"]
        print(f"{r['label']:>12s} vs {r['baseline_label']:<12s} {r['mean_delta']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] {r['wins']:>2d}/{r['n_folds']}  W={r['W']:.1f}  "
              f"p={r['p']:.4f}  p_holm={r['p_holm']:.3f}   {r['quantity']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
