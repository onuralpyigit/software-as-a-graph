#!/usr/bin/env python3
"""
reproduce/topo_ap_sensitivity.py — Topo / Topo-QoS with their articulation term restored
=======================================================================================

The closed-form baselines are specified as ``0.6 * BT + 0.4 * AP``. In the
evaluated LOSO path the AP term is always zero: the cached
``structural_metrics.json`` carries no ``ap_c_score``, so
``reproduce.main_table._parse_structural_metrics`` writes
``articulation_point = 0.0`` for every node, and that key takes precedence over
the ``ap_c_score`` that ``_saag_structural_features`` does compute. The
published scores are therefore rank-equivalent to betweenness and QoS-weighted
betweenness.

The registered comparator is left exactly as it ran (PREREGISTRATION.md,
Amendment 5). This script scores both forms on the same twelve LOSO holdouts,
Application population, against the same I*(v) labels, and checks that the
"as run" column reproduces the published per-fold values.

Usage:
    PYTHONPATH=. python reproduce/topo_ap_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import compute_inductive_metrics, load_scenario_bundle  # noqa: E402
from reproduce._provenance import stamp  # noqa: E402
from reproduce.main_table import _compute_topo_baseline_scores, _load_scenario_data  # noqa: E402


def _with_ap(struct: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Copy of ``struct`` whose articulation term reads the computed ap_c_score."""
    return {
        nid: {**m, "articulation_point": float(m.get("ap_c_score", 0.0))}
        for nid, m in struct.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    ap.add_argument("--published", type=Path,
                    default=Path("results/loso_all_variants_v5.json"))
    ap.add_argument("--output", type=Path, default=Path("results/topo_ap_sensitivity.json"))
    args = ap.parse_args()

    published = {}
    if args.published.exists():
        pv = json.loads(args.published.read_text())["per_variant_results"]
        for v in ("topo_baseline", "topo_qos"):
            published[v] = {f["holdout_id"]: f["mean_metrics"]["spearman_rho"]
                            for f in pv[v]["folds"]}

    rows: Dict[str, Dict[str, Any]] = {}
    for sdir in sorted(p for p in args.cache_dir.iterdir() if p.is_dir()):
        bundle = load_scenario_bundle(sdir)
        if bundle is None:
            continue
        graph, struct, *_ = _load_scenario_data(bundle.scenario_id, "projection", cache_dir=sdir)
        labels = {nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()}
        row: Dict[str, Any] = {
            "n_ap_nonzero": int(sum(1 for m in struct.values() if m.get("ap_c_score", 0.0) > 0)),
        }
        for variant, use_qos in (("topo_baseline", False), ("topo_qos", True)):
            for form, sm in (("as_run", struct), ("ap_restored", _with_ap(struct))):
                pred = _compute_topo_baseline_scores(graph, sm, use_qos=use_qos) or {}
                m = compute_inductive_metrics(pred, labels, bundle.graph, population="application")
                row[f"{variant}_{form}"] = float(m["spearman_rho"])
            if variant in published and bundle.scenario_id in published[variant]:
                row[f"{variant}_published"] = published[variant][bundle.scenario_id]
        rows[bundle.scenario_id] = row
        print(f"  {bundle.scenario_id:<28} AP>0: {row['n_ap_nonzero']:>3}   "
              f"Topo-QoS as run {row['topo_qos_as_run']:.4f}  AP restored {row['topo_qos_ap_restored']:.4f}")

    summary = {}
    for key in ("topo_baseline_as_run", "topo_baseline_ap_restored",
                "topo_qos_as_run", "topo_qos_ap_restored"):
        vals = [r[key] for r in rows.values()]
        summary[key] = float(np.mean(vals))
    for v in ("topo_baseline", "topo_qos"):
        diffs = [abs(r[f"{v}_as_run"] - r[f"{v}_published"]) for r in rows.values()
                 if f"{v}_published" in r]
        summary[f"{v}_max_abs_diff_vs_published"] = float(max(diffs)) if diffs else None
    print("\n  " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items() if v is not None))

    args.output.write_text(json.dumps({
        "note": __doc__.split("\n\n")[1],
        "population": "application",
        "per_scenario": rows,
        "summary": summary,
        "provenance": stamp(cache_dir=str(args.cache_dir)),
    }, indent=2))
    print(f"  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
