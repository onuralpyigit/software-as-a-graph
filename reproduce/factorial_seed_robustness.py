#!/usr/bin/env python3
"""
reproduce/factorial_seed_robustness.py — is the 2x2 interaction a seed artefact?
===============================================================================

GAT-N, the floor cell of the reported 2x2, has a median within-fold seed spread
of 0.298 against 0.024-0.114 for the other three learned arms. If a few
collapsed GAT-N seeds pull its fold means down, both main effects and the
typing x QoS interaction inherit that optimisation failure: any arm that fixes
it looks like a large simple effect, and two arms that both fix it look like
substitutes.

This script re-forms the fold score from the per-seed rhos already in the LOSO
artifact under three aggregations and recomputes the same three orthogonal
quantities with ``loso_significance.factorial``:

  * ``mean``          — the reported aggregation (reproduces Table 8);
  * ``median``        — robust to one or two collapsed seeds per fold;
  * ``trimmed``       — mean after dropping each arm's worst seed per fold.

No model is retrained, so it cannot say whether GAT-N is under-trained. It can
say whether the reported interaction depends on its worst seeds.

Usage:
    PYTHONPATH=. python reproduce/factorial_seed_robustness.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp
from reproduce.loso_significance import FACTORIAL_CELLS, factorial

AGGREGATIONS: Dict[str, Callable[[List[float]], float]] = {
    "mean": lambda xs: float(np.mean(xs)),
    "median": lambda xs: float(np.median(xs)),
    "trimmed": lambda xs: float(np.mean(sorted(xs)[1:])) if len(xs) > 1 else float(xs[0]),
}


def seed_rhos(artifact: Dict[str, Any], variant: str) -> Dict[str, List[float]]:
    """Per-fold list of per-seed Spearman rho for ``variant``."""
    folds = artifact["per_variant_results"][variant]["folds"]
    return {
        f["holdout_id"]: [s["spearman_rho"] for s in f["seed_metrics"]
                          if s.get("spearman_rho") is not None]
        for f in folds
    }


def table_for(artifact: Dict[str, Any], agg: Callable[[List[float]], float]) -> Dict[str, Any]:
    """A comparison_table-shaped dict holding only the four factorial cells."""
    return {
        v: {"per_fold": [{"holdout": h, "mean_rho": agg(xs)}
                         for h, xs in seed_rhos(artifact, v).items() if xs]}
        for v in FACTORIAL_CELLS.values()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--input", type=Path, default=Path("results/loso_all_variants_v5.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/factorial_seed_robustness_v5.json"))
    args = ap.parse_args()

    artifact = json.loads(args.input.read_text())
    spread = {
        v: float(np.median([np.std(xs) for xs in seed_rhos(artifact, v).values() if xs]))
        for v in FACTORIAL_CELLS.values()
    }

    results: Dict[str, Any] = {}
    print(f"\n  2x2 under alternative seed aggregations   ({args.input})")
    print("  " + "─" * 70)
    print(f"  {'aggregation':<12}{'quantity':<14}{'d rho':>9}{'wins':>8}{'p':>9}{'p_holm':>9}")
    for name, agg in AGGREGATIONS.items():
        rows = factorial(table_for(artifact, agg))
        results[name] = rows
        for r in rows:
            print(f"  {name:<12}{r['quantity']:<14}{r['mean_delta']:>+9.4f}"
                  f"{r['wins']:>5}/{r['n_folds']:<2}{r['p']:>9.4f}{r['p_holm']:>9.4f}")

    payload = {
        "input": str(args.input),
        "median_within_fold_seed_sd": spread,
        "aggregations": results,
        "provenance": stamp(input=str(args.input), aggregations=list(AGGREGATIONS)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n  median within-fold seed SD: "
          + ", ".join(f"{v} {s:.3f}" for v, s in spread.items()))
    print(f"  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
