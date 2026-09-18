#!/usr/bin/env python3
"""
reproduce/stratification_check.py — is any headline number an amalgamation artifact?

Every figure in Table 3 and Table 8 is an average over something: over seeds,
over scenarios, over node types, and over components whose failure reaches
nobody together with components whose failure reaches half the system. Simpson's
paradox is the case where the pooled figure and the per-stratum figures disagree
about the *direction* of an effect, and `saag/evaluation/metrics.py` already
carries two guards against it (`_per_type_rho`, `per_qos_tier_rho`) plus the
comment on ``EVAL_POPULATIONS`` refusing to pool Topic and Node with Application.

This script asks whether those guards are sufficient, by recomputing each
headline under both conventions and reporting where they part company:

  Axis 1  Scenarios pooled into one node set, vs. the mean of the twelve
          per-scenario correlations. Run raw and with labels rank-normalised
          inside each scenario, because max I*(v) spans ~3x and mean I*(v) ~24x
          across the cohort, and a pooled raw correlation partly measures which
          scenario a node came from.
  Axis 2  Components the oracle scores strictly positive, vs. all of them.
  Axis 3  F1 macro-averaged over scenarios vs. micro-averaged over pooled counts.
  Axis 4  The same pooling question on the five real systems.
  Axis 5  Application pooled with Library, vs. the two strata separately.

Inputs are the artifacts, not a re-run: `main_table.py` and
`realworld_zeroshot.py` both persist the per-node ``eval_points`` behind every
cell for exactly this purpose. Axis 5 needs a population the main table is not
scored on, so it reads a separate small artifact (see ``--applib``); it is
skipped when that file is absent.

Usage
-----
  python reproduce/stratification_check.py
  python reproduce/stratification_check.py --main-table results/main_table_f1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag.evaluation import variant_registry as _registry  # noqa: E402

_RESULTS = Path("results")

_VARIANTS = ["topo_baseline", "topo_qos", "gl", "gl_qos", "hgl", "hgl_qos"]
_LABEL = {v: _registry.label(v) for v in _VARIANTS}

#: Cut for the critical set, matching ``compute_inductive_metrics``' tau_frac.
_TAU_FRAC = 0.50


def _points(cell: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pts = cell.get("eval_points") or []
    return (
        np.array([p["pred"] for p in pts], dtype=np.float64),
        np.array([p["true"] for p in pts], dtype=np.float64),
    )


def _pooled_rho(cells: Sequence[Dict[str, Any]], normalise: bool) -> Optional[float]:
    """Spearman over every cell's points concatenated into one vector.

    ``normalise`` converts each cell's two vectors to within-cell ranks first,
    which removes the between-scenario level differences while preserving the
    within-scenario ordering the estimator is actually claiming to get right.
    """
    preds, trues = [], []
    for cell in cells:
        p, t = _points(cell)
        if len(p) < 3:
            continue
        if normalise:
            p, t = rankdata(p) / len(p), rankdata(t) / len(t)
        preds.append(p)
        trues.append(t)
    if not preds:
        return None
    rho = spearmanr(np.concatenate(preds), np.concatenate(trues)).correlation
    return None if np.isnan(rho) else float(rho)


def _confusion(p: np.ndarray, t: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """tp/fp/fn at the same relative cut on both vectors, or None if degenerate."""
    if len(p) < 3 or t.max() <= 0 or p.max() <= 0:
        return None
    true_c = t >= _TAU_FRAC * t.max()
    if true_c.all() or not true_c.any():
        return None
    pred_c = p >= _TAU_FRAC * p.max()
    return (
        int((pred_c & true_c).sum()),
        int((pred_c & ~true_c).sum()),
        int((~pred_c & true_c).sum()),
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def axis1_scenario_pooling(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_vs = defaultdict(list)
    for c in cells:
        by_vs[(c["variant"], c["seed"])].append(c)
    seeds = sorted({c["seed"] for c in cells})

    out: Dict[str, Any] = {"seeds": seeds, "per_variant": {}}
    for v in _VARIANTS:
        macro, raw, norm = [], [], []
        for s in seeds:
            group = by_vs.get((v, s), [])
            if not group:
                continue
            macro.append(float(np.mean([c["spearman_rho"] for c in group])))
            for acc, flag in ((raw, False), (norm, True)):
                val = _pooled_rho(group, flag)
                if val is not None:
                    acc.append(val)
        out["per_variant"][v] = {
            "macro": float(np.mean(macro)) if macro else None,
            "micro_raw": float(np.mean(raw)) if raw else None,
            "micro_rank_normalised": float(np.mean(norm)) if norm else None,
            "micro_raw_per_seed": raw,
        }

    def _order(key: str) -> List[str]:
        have = [v for v in _VARIANTS if out["per_variant"][v].get(key) is not None]
        return sorted(have, key=lambda v: -out["per_variant"][v][key])

    out["ranking"] = {k: _order(k) for k in ("macro", "micro_raw", "micro_rank_normalised")}
    out["ranking_changes_under_pooling"] = (
        out["ranking"]["macro"] != out["ranking"]["micro_raw"]
    )
    return out


def axis2_zero_stratum(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_variant: Dict[str, Any] = {}
    reversals: List[Dict[str, Any]] = []
    for v in _VARIANTS:
        pairs = [
            (c["spearman_rho"], c["spearman_rho_positive"], c)
            for c in cells
            if c["variant"] == v
            and isinstance(c.get("spearman_rho_positive"), (int, float))
            and isinstance(c.get("spearman_rho"), (int, float))
        ]
        if not pairs:
            continue
        pooled = np.array([x[0] for x in pairs])
        active = np.array([x[1] for x in pairs])
        flips = [
            {
                "scenario": c["scenario"], "seed": c["seed"], "variant": v,
                "rho": a, "rho_positive": b,
                "n_positive": c.get("n_positive"), "n": c.get("n_evaluated"),
            }
            for a, b, c in pairs if a > 0.1 and b < -0.1
        ]
        reversals.extend(flips)
        per_variant[v] = {
            "mean_rho": float(pooled.mean()),
            "mean_rho_positive": float(active.mean()),
            "collapse": float(active.mean() - pooled.mean()),
            "n_sign_reversals": len(flips),
            "n_cells": len(pairs),
        }
    reversals.sort(key=lambda r: r["rho"] - r["rho_positive"], reverse=True)
    return {"per_variant": per_variant, "reversals": reversals}


def axis3_f1_averaging(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_variant: Dict[str, Any] = {}
    for v in _VARIANTS:
        macro, tp = [], [0, 0, 0]
        for c in (c for c in cells if c["variant"] == v):
            conf = _confusion(*_points(c))
            if conf is None:
                continue
            macro.append(_f1(*conf))
            tp = [a + b for a, b in zip(tp, conf)]
        if not macro:
            continue
        per_variant[v] = {
            "macro_f1": float(np.mean(macro)),
            "micro_f1": _f1(*tp),
            "delta": _f1(*tp) - float(np.mean(macro)),
            "n_cells": len(macro),
        }
    return {"per_variant": per_variant}


def axis4_realworld(data: Dict[str, Any]) -> Dict[str, Any]:
    per_system = data.get("per_system", {})
    systems = sorted(s for s in per_system if per_system[s].get("eval_points"))
    fake_cells = [{"eval_points": per_system[s]["eval_points"]} for s in systems]

    macro = [per_system[s]["mean_rho"] for s in systems]
    out = {
        "systems": systems,
        "macro": float(np.mean(macro)) if macro else None,
        "micro_raw": _pooled_rho(fake_cells, False),
        "micro_rank_normalised": _pooled_rho(fake_cells, True),
        "zero_stratum": {},
    }

    refs = data.get("references", {})
    learned = data.get("label", data.get("variant", "learned"))
    rows = {learned: [(per_system[s]["mean_rho"], per_system[s].get("mean_rho_positive"))
                      for s in systems]}
    for name, per in refs.items():
        rows[name] = [(per.get(s, {}).get("rho"), per.get(s, {}).get("rho_positive"))
                      for s in systems]
    for name, pairs in rows.items():
        a = [x for x, _ in pairs if isinstance(x, (int, float))]
        b = [y for _, y in pairs if isinstance(y, (int, float))]
        if not a or not b:
            continue
        out["zero_stratum"][name] = {
            "mean_rho": float(np.mean(a)),
            "mean_rho_positive": float(np.mean(b)),
            "collapse": float(np.mean(b) - np.mean(a)),
            "n_systems_defined": len(b),
            "per_system": {
                s: {"rho": pairs[i][0], "rho_positive": pairs[i][1]}
                for i, s in enumerate(systems)
            },
        }
    return out


def axis5_type_pooling(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    cells = [c for c in json.loads(path.read_text())["cells"] if "error" not in c]
    rows, n_opposite, n_comparable = [], 0, 0
    for c in cells:
        types = c.get("per_node_type", {})
        app = types.get("Application", {}).get("rho")
        lib = types.get("Library", {}).get("rho")
        if not isinstance(app, (int, float)) or not isinstance(lib, (int, float)):
            continue
        n_comparable += 1
        pooled = c["spearman_rho"]
        opposite = (pooled > 0) != (lib > 0)
        n_opposite += opposite
        rows.append({
            "scenario": c["scenario"], "variant": c["variant"], "pooled": pooled,
            "Application": app, "Library": lib,
            "n_library": types["Library"].get("n"),
            "outside_both_strata": pooled > max(app, lib) or pooled < min(app, lib),
            "sign_differs_from_library": bool(opposite),
        })
    return {
        "rows": rows,
        "n_comparable": n_comparable,
        "n_sign_differs_from_library": int(n_opposite),
        "n_outside_both_strata": sum(r["outside_both_strata"] for r in rows),
    }


def render(report: Dict[str, Any]) -> str:
    L = _LABEL
    out: List[str] = ["# Stratification check — where pooling changes the answer", ""]

    a1 = report["axis1_scenario_pooling"]
    out += [
        "## 1. Pooling the twelve scenarios",
        "",
        "`macro` is the mean of the twelve per-scenario ρ (what Table 3 reports). "
        "`micro raw` pools every held-out node into one vector. `micro rank-norm` "
        "pools after ranking within each scenario, which strips the between-scenario "
        "level differences and keeps only the within-scenario ordering.",
        "",
        "| Variant | macro | micro raw | micro rank-norm |",
        "|---|---|---|---|",
    ]
    for v in _VARIANTS:
        e = a1["per_variant"].get(v, {})
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        out.append(f"| {L[v]} | {f(e.get('macro'))} | {f(e.get('micro_raw'))} "
                   f"| {f(e.get('micro_rank_normalised'))} |")
    out += [""]
    for key, name in (("macro", "macro"), ("micro_raw", "micro raw"),
                      ("micro_rank_normalised", "micro rank-norm")):
        out.append(f"- **{name}**: " + " > ".join(L[v] for v in a1["ranking"][key]))
    out += ["", f"Ranking changes under pooling: **{a1['ranking_changes_under_pooling']}**", ""]

    a2 = report["axis2_zero_stratum"]
    out += [
        "## 2. The zero-impact stratum",
        "",
        "`ρ⁺` is the same correlation restricted to components the oracle scores "
        "strictly positive — the ones whose failure actually reaches somebody. The "
        "gap is how much of the pooled ρ is separating inert components from active "
        "ones rather than ordering the active ones.",
        "",
        "| Variant | ρ | ρ⁺ | collapse | sign reversals |",
        "|---|---|---|---|---|",
    ]
    for v in _VARIANTS:
        e = a2["per_variant"].get(v)
        if not e:
            continue
        out.append(f"| {L[v]} | {e['mean_rho']:.3f} | {e['mean_rho_positive']:.3f} "
                   f"| {e['collapse']:+.3f} | {e['n_sign_reversals']}/{e['n_cells']} |")
    out += ["", "Worst single-cell reversals (ρ clearly positive, ρ⁺ clearly negative):", "",
            "| Scenario | Variant | seed | ρ | ρ⁺ | n⁺/n |", "|---|---|---|---|---|---|"]
    for r in a2["reversals"][:8]:
        out.append(f"| {r['scenario']} | {L.get(r['variant'], r['variant'])} | {r['seed']} "
                   f"| {r['rho']:+.3f} | {r['rho_positive']:+.3f} | {r['n_positive']}/{r['n']} |")
    out += [""]

    a3 = report["axis3_f1_averaging"]
    out += ["## 3. F1 averaging", "",
            "| Variant | macro-F1 | micro-F1 | Δ |", "|---|---|---|---|"]
    for v in _VARIANTS:
        e = a3["per_variant"].get(v)
        if not e:
            continue
        out.append(f"| {L[v]} | {e['macro_f1']:.3f} | {e['micro_f1']:.3f} | {e['delta']:+.3f} |")
    out += [""]

    a4 = report.get("axis4_realworld")
    if a4:
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        out += ["## 4. The five real systems", "",
                f"- macro (mean of five): {f(a4['macro'])}",
                f"- micro, pooled raw: {f(a4['micro_raw'])}",
                f"- micro, rank-normalised per system: {f(a4['micro_rank_normalised'])}",
                "", "| Predictor | ρ | ρ⁺ | collapse |", "|---|---|---|---|"]
        for name, e in a4["zero_stratum"].items():
            out.append(f"| {name} | {e['mean_rho']:+.3f} | {e['mean_rho_positive']:+.3f} "
                       f"| {e['collapse']:+.3f} |")
        out += [""]

    a5 = report.get("axis5_type_pooling")
    if a5:
        out += ["## 5. Application pooled with Library", "",
                f"Pooled ρ has the opposite sign to the Library stratum in "
                f"**{a5['n_sign_differs_from_library']}/{a5['n_comparable']}** cells where "
                f"both strata are defined; it falls outside the range of both strata in "
                f"**{a5['n_outside_both_strata']}/{a5['n_comparable']}**.", "",
                "| Scenario | Variant | pooled | Application | Library | n(Lib) |",
                "|---|---|---|---|---|---|"]
        for r in a5["rows"]:
            out.append(f"| {r['scenario']} | {L.get(r['variant'], r['variant'])} "
                       f"| {r['pooled']:+.3f} | {r['Application']:+.3f} | {r['Library']:+.3f} "
                       f"| {r['n_library']} |")
        out += [""]
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-table", type=Path, default=_RESULTS / "main_table.json")
    p.add_argument("--realworld", type=Path, default=_RESULTS / "realworld_zeroshot.json")
    p.add_argument("--applib", type=Path, default=_RESULTS / "applib_population.json",
                   help="main_table.py output scored with --eval-population app_lib; "
                        "axis 5 is skipped when absent.")
    p.add_argument("--output", type=Path, default=_RESULTS / "stratification_check")
    args = p.parse_args()

    if not args.main_table.exists():
        print(f"  Not found: {args.main_table}")
        return 1
    cells = [c for c in json.loads(args.main_table.read_text())["cells"] if "error" not in c]
    if not any(c.get("eval_points") for c in cells):
        print(f"  {args.main_table} carries no eval_points; re-run reproduce/main_table.py.")
        return 1

    report: Dict[str, Any] = {
        "axis1_scenario_pooling": axis1_scenario_pooling(cells),
        "axis2_zero_stratum": axis2_zero_stratum(cells),
        "axis3_f1_averaging": axis3_f1_averaging(cells),
        "sources": {"main_table": str(args.main_table)},
    }
    if args.realworld.exists():
        report["axis4_realworld"] = axis4_realworld(json.loads(args.realworld.read_text()))
        report["sources"]["realworld"] = str(args.realworld)
    a5 = axis5_type_pooling(args.applib)
    if a5:
        report["axis5_type_pooling"] = a5
        report["sources"]["applib"] = str(args.applib)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".md").write_text(render(report))
    print(f"  Saved {args.output.with_suffix('.json')}")
    print(f"  Saved {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
