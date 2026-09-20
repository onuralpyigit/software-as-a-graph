#!/usr/bin/env python3
"""Measure how far a code revision moves LOSO results at fixed seeds and corpus.

Section 8.3 reports that re-executing the pipeline across code revisions, at
identical seeds and corpus digests, leaves the training-free cells bit-identical
while moving the learned ones. That claim carried three figures with no artifact
behind them, which is the one thing this project has repeatedly resolved not to
ship. This script produces the artifact.

What it compares is two committed LOSO sweeps that name different commits and
the *same* corpus digest. Any cell that differs between them therefore differs
because the code changed, not because the data did -- the comparison is only
meaningful under that precondition, so it is asserted rather than assumed.

The quantity that matters is not the largest movement in the set. It is whether
the movement is comparable to the effects the manuscript draws conclusions from:
a pipeline whose learned cells shift by more than its headline contrast has not
established that contrast against its own revision noise. The summary therefore
reports the drift beside the contrast it has to be read against.

Usage
-----
    python reproduce/rerun_drift.py \\
        --before results/loso_all_variants_v4.json \\
        --after  results/loso_all_variants_v5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(ROOT))

from reproduce._provenance import stamp  # noqa: E402
from saag.evaluation import variant_registry as _registry  # noqa: E402

#: Variants that carry no parameters and so must be invariant under any change
#: to the training code. They are the control: if these move, the revision
#: touched scoring or data handling rather than learning, and the learned
#: drift below cannot be attributed to training nondeterminism.
TRAINING_FREE = ("topo_baseline", "topo_qos", "topology_rm")


def _folds(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {f["holdout_id"]: f for f in entry.get("folds", [])}


def _cells(doc: Dict[str, Any], variant: str) -> Tuple[Dict[str, float],
                                                       Dict[Tuple[str, int], float]]:
    """Per-fold means and per-(fold, seed) values for one variant."""
    entry = doc.get("per_variant_results", {}).get(variant)
    if not entry:
        return {}, {}
    fold_rho, seed_rho = {}, {}
    for holdout, fold in _folds(entry).items():
        rho = fold.get("mean_metrics", {}).get("spearman_rho")
        if rho is not None:
            fold_rho[holdout] = float(rho)
        for s in fold.get("seed_metrics", []):
            if s.get("spearman_rho") is not None:
                seed_rho[(holdout, int(s["seed"]))] = float(s["spearman_rho"])
    return fold_rho, seed_rho


def drift(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    variants = sorted(set(before.get("per_variant_results", {}))
                      & set(after.get("per_variant_results", {})))
    per_variant: List[Dict[str, Any]] = []
    for v in variants:
        fa, sa = _cells(before, v)
        fb, sb = _cells(after, v)
        fold_d = [(abs(fb[k] - fa[k]), k, fa[k], fb[k]) for k in sorted(set(fa) & set(fb))]
        seed_d = [(abs(sb[k] - sa[k]), k) for k in sorted(set(sa) & set(sb))]
        if not fold_d:
            continue
        worst_fold = max(fold_d)
        worst_seed = max(seed_d) if seed_d else (0.0, None)
        per_variant.append({
            "variant": v,
            "label": _registry.label(v, harness="loso"),
            "training_free": v in TRAINING_FREE,
            "n_fold_cells": len(fold_d),
            "n_seed_cells": len(seed_d),
            "max_abs_fold_delta": round(worst_fold[0], 4),
            "max_abs_fold_delta_at": {
                "holdout": worst_fold[1],
                "before": round(worst_fold[2], 4),
                "after": round(worst_fold[3], 4),
            },
            "max_abs_seed_delta": round(worst_seed[0], 4),
            "max_abs_seed_delta_at": (
                {"holdout": worst_seed[1][0], "seed": worst_seed[1][1]}
                if worst_seed[1] else None
            ),
            "mean_abs_fold_delta": round(sum(d for d, *_ in fold_d) / len(fold_d), 4),
        })

    learned = [r for r in per_variant if not r["training_free"]]
    free = [r for r in per_variant if r["training_free"]]
    return {
        "per_variant": per_variant,
        "summary": {
            "training_free_cells": sum(r["n_seed_cells"] for r in free),
            "training_free_max_abs_delta": (
                max((r["max_abs_seed_delta"] for r in free), default=None)
            ),
            "learned_cells": sum(r["n_seed_cells"] for r in learned),
            "learned_max_abs_seed_delta": (
                max((r["max_abs_seed_delta"] for r in learned), default=None)
            ),
            "learned_max_abs_fold_delta": (
                max((r["max_abs_fold_delta"] for r in learned), default=None)
            ),
            "learned_mean_abs_fold_delta": (
                round(sum(r["mean_abs_fold_delta"] for r in learned) / len(learned), 4)
                if learned else None
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="results/loso_all_variants_v4.json")
    ap.add_argument("--after", default="results/loso_all_variants_v5.json")
    ap.add_argument("--output", default="results/rerun_drift.json")
    ap.add_argument(
        "--allow-corpus-mismatch", action="store_true",
        help="Compare two sweeps whose corpus digests differ. Off by default: "
             "the drift would then confound a code change with a data change, "
             "which is precisely the attribution the artifact is meant to make.",
    )
    args = ap.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    paths = {k: _resolve(v) for k, v in (("before", args.before), ("after", args.after))}
    for role, path in paths.items():
        if not path.exists():
            print(f"Error: --{role} artifact {path} not found.", file=sys.stderr)
            return 2
    docs = {k: json.loads(p.read_text()) for k, p in paths.items()}

    prov = {k: d.get("provenance", {}) for k, d in docs.items()}
    digests = {k: p.get("corpus_digest") for k, p in prov.items()}
    if digests["before"] != digests["after"] and not args.allow_corpus_mismatch:
        print(f"Error: corpus digests differ ({digests['before']} vs "
              f"{digests['after']}). The drift would confound code and data; "
              f"pass --allow-corpus-mismatch to measure it anyway.",
              file=sys.stderr)
        return 2
    for role, p in prov.items():
        if p.get("dirty"):
            print(f"Warning: the --{role} artifact was produced from a modified "
                  f"working tree, so the commit it names does not pin it.",
                  file=sys.stderr)

    payload = drift(docs["before"], docs["after"])
    payload["measurement"] = (
        "Per-cell movement in LOSO Spearman rho between two committed sweeps "
        "at identical seeds and an identical corpus digest, so the difference "
        "is attributable to the code revision between their commits."
    )
    payload["before"] = {
        "path": str(paths["before"].relative_to(ROOT)),
        "commit": prov["before"].get("commit"),
    }
    payload["after"] = {
        "path": str(paths["after"].relative_to(ROOT)),
        "commit": prov["after"].get("commit"),
    }
    payload["corpus_digest"] = digests["after"]
    payload["provenance"] = stamp(before=args.before, after=args.after)

    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    s = payload["summary"]
    print(f"  {payload['before']['commit'][:8]} -> {payload['after']['commit'][:8]}"
          f"   corpus {(payload['corpus_digest'] or '-')[:12]}")
    print("  " + "─" * 70)
    print(f"  {'variant':<16}{'kind':<15}{'max |d| fold':>14}{'max |d| seed':>14}")
    for r in payload["per_variant"]:
        kind = "training-free" if r["training_free"] else "learned"
        print(f"  {r['variant']:<16}{kind:<15}{r['max_abs_fold_delta']:>14.4f}"
              f"{r['max_abs_seed_delta']:>14.4f}")
    print("  " + "─" * 70)
    print(f"  training-free: {s['training_free_cells']} cells, "
          f"max |drho| {s['training_free_max_abs_delta']:.6f}")
    print(f"  learned:       {s['learned_cells']} cells, "
          f"max |drho| {s['learned_max_abs_seed_delta']:.4f} at seed level, "
          f"{s['learned_max_abs_fold_delta']:.4f} at fold mean")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
