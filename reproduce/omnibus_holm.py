#!/usr/bin/env python3
"""
reproduce/omnibus_holm.py — one Holm correction over every registered contrast
==============================================================================

Each registration in ``docs/research/jss/PREREGISTRATION.md`` Holm-corrects its
own contrasts and nothing else. The amendments were issued in sequence, and
some were conditioned on an earlier amendment's result: Amendment 6 exists
because of Amendment 2's outcome. Correcting inside each family therefore
does not bound the error accumulated across the sequence. This script pools
every contrast that any registration designated as decision-bearing and applies
a single Holm correction to the pooled set.

The family (``REGISTERED_FAMILY``) is exactly the registered contrasts:

  * plan (2026-09-06)  — HGT-QoS and HGT vs Topo-QoS            (2)
  * Amendment 2        — matched 2x2 quantities + three controls (6)
  * Amendment 5        — SaG-Hybrid vs Topo-QoS and vs HGT-QoS   (2)
  * Amendment 6        — SaG-Hybrid-GAT vs Topo-QoS and vs GAT-N-QoS16-C (2)

Amendments 1, 3 and 4 register no contrast. Amendment 1 changes the selection
budget, Amendment 3 re-baselines and withdraws, and Amendment 4 adds analyses
that are exploratory by its own statement. Exploratory contrasts are left out
on purpose: pooling them would test a family nobody registered.

The per-fold p-values are not re-computed here; they are read from the
significance artifacts that ``reproduce/loso_significance.py`` writes, so the
omnibus correction cannot disagree with the tests it corrects.

Usage:
    PYTHONPATH=. python reproduce/omnibus_holm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(ROOT))

from reproduce._provenance import stamp  # noqa: E402
from reproduce.loso_significance import holm  # noqa: E402

#: (registration, artifact, block) — every block listed is taken whole.
REGISTERED_FAMILY = [
    ("plan", "loso_significance_v5.json", "preregistered"),
    ("amendment_2", "loso_significance_rq2_matched.json", "factorial"),
    ("amendment_2", "loso_significance_rq2_matched.json", "rq2_controls"),
    # The directionality control was registered with the other controls but run
    # later, in its own invocation (make rq-directionality).
    ("amendment_2", "loso_significance_directionality_cpu.json", "rq2_controls"),
    ("amendment_5", "loso_significance_hybrid_cpu.json", "hybrid"),
    ("amendment_6", "loso_significance_hybrid_gat_cpu.json", "hybrid_gat"),
]


def _label(row: Dict[str, Any]) -> str:
    if "quantity" in row:
        return row["label"]
    return f"{row['label']} vs {row['baseline_label']}"


def collect(results_dir: Path) -> List[Dict[str, Any]]:
    """The registered contrasts, each with its raw p and in-family Holm p."""
    family: List[Dict[str, Any]] = []
    for registration, artifact, block in REGISTERED_FAMILY:
        data = json.loads((results_dir / artifact).read_text(encoding="utf8"))
        for row in data[block]:
            family.append({
                "registration": registration,
                "artifact": artifact,
                "block": block,
                "contrast": _label(row),
                "mean_delta": row["mean_delta"],
                "p": row["p"],
                "p_holm_family": row.get("p_holm"),
            })
    # A registration's family can span sweeps: Amendment 2's controls ran in two
    # invocations. Each artifact's own p_holm saw only the rows it contained, so
    # Holm is re-applied within (registration, block) across every artifact. For
    # a family that lives in one artifact this reproduces its p_holm exactly.
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in family:
        groups.setdefault((r["registration"], r["block"]), []).append(r)
    for rows in groups.values():
        ps = [{"p": r["p"]} for r in rows]
        holm(ps)
        for r, q in zip(rows, ps):
            r["p_holm_family"] = q["p_holm"]
    return family


def omnibus(family: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Holm over the pooled family, stored as ``p_holm_omnibus``."""
    pooled = [dict(r) for r in family]
    holm(pooled)
    for r in pooled:
        r["p_holm_omnibus"] = r.pop("p_holm")
    return pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/omnibus_registered_holm.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows = omnibus(collect(args.results_dir))
    out = {
        "alpha": args.alpha,
        "m": len(rows),
        "contrasts": sorted(rows, key=lambda r: r["p"]),
        "provenance": stamp(family=[list(f) for f in REGISTERED_FAMILY]),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf8")

    print(f"Omnibus Holm over {len(rows)} registered contrasts (alpha = {args.alpha})")
    for r in out["contrasts"]:
        mark = "*" if r["p_holm_omnibus"] < args.alpha else " "
        print(f" {mark} {r['registration']:12s} {r['contrast']:40s} "
              f"p={r['p']:.5f}  family={r['p_holm_family'] or float('nan'):.4f}  "
              f"omnibus={r['p_holm_omnibus']:.4f}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
