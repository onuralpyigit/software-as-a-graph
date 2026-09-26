#!/usr/bin/env python3
"""
reproduce/dependency_graph_contrasts.py — does learning on the dependency graph beat counting it?
================================================================================================

PREREGISTRATION.md Amendment 9. Reads the one-sweep artifact of
``make rq-dependency-graph`` (the four learned arms on the DEPENDS_ON projection)
and reports the registered family of twelve contrasts, Holm-corrected within
itself and nowhere else:

  * each of GAT-P, GAT-P-QoS, Hybrid-GAT-P, HGT-P-QoS  vs  InDeg
  * each                                               vs  Reach
  * each                                               vs  its native counterpart

InDeg and Reach (Amendment 7) are recomputed per fold here, on the LOSO cache's
labels -- the labels every learned arm is scored against -- and checked against
``results/tf_baselines.json``. Native counterparts are read from their own clean
artifacts (NATIVE_SOURCES). Statistics are ``training_free_suite.paired`` (two-sided
Wilcoxon over folds, fold-bootstrap 95% CI, B = 2,000) and ``.holm``, the pair
Amendment 7's family used. Zero-shot over the five system models is descriptive,
and the decision rules D1-D3, M and Z are evaluated as registered.

Usage:
    PYTHONPATH=. python reproduce/dependency_graph_contrasts.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp
from reproduce.training_free_suite import (
    FOLDS, PUBLISHED_GBM_FEAT, SYSTEMS, holm, indeg, paired, reach, score,
)
from saag.evaluation import variant_registry as registry

#: Amendment 9 arm -> its native counterpart.
ARMS = {
    "gl_proj_cap": "gl_full_cap",
    "gl_proj_qos16_cap": "gl_full_qos16_cap",
    "gl_proj_qos16_indeg_prior": "gl_qos16_prior",
    "hgl_proj_qos": "hgl_qos",
}
#: Where each native counterpart's clean LOSO / zero-shot rows live.
NATIVE_SOURCES = {
    "gl_full_cap": ("results/loso_attribution_cpu.json",
                    "results/realworld_zeroshot_gl_full_cap_attribution.json"),
    "gl_full_qos16_cap": ("results/loso_attribution_cpu.json",
                          "results/realworld_zeroshot_gl_full_qos16_cap_attribution.json"),
    "gl_qos16_prior": ("results/loso_hybrid_gat_cpu.json",
                       "results/realworld_zeroshot_gl_qos16_prior_cpu.json"),
    "hgl_qos": ("results/loso_directionality_cpu.json",
                "results/realworld_zeroshot_hgl_qos_directionality.json"),
}
#: Re-run in the new invocation to show the routing left the native path alone,
#: each against the artifact it must reproduce.
REPRODUCED = {
    "topo_qos": "results/loso_attribution_cpu.json",
    "gl_full_qos16_cap": "results/loso_attribution_cpu.json",
}
#: Published five-system means the Z rule is stated against (Amendment 7).
REACH_SYSTEMS_PUBLISHED = 0.938


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _loso_rows(artifact: Dict[str, Any], variant: str) -> Dict[str, Dict[str, Optional[float]]]:
    """{holdout: {rho, rho_active}} from a loso_all_variants artifact."""
    folds = artifact["per_variant_results"][variant]["folds"]
    return {
        f["holdout_id"]: {
            "rho": f["mean_metrics"].get("spearman_rho"),
            "rho_active": f["mean_metrics"].get("spearman_rho_positive"),
        }
        for f in folds
    }


def _counts_on_cache(cache_root: Path, ids: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """InDeg and Reach scored against each cache's own I*(v) labels."""
    from cli.loso_evaluate import load_scenario_bundle
    from saag.prediction.structural_predictor import derive_flow_projection

    out: Dict[str, Dict[str, Any]] = {}
    for sid in ids:
        bundle = load_scenario_bundle(cache_root / sid)
        flow = derive_flow_projection(json.loads((cache_root / sid / "topology.json").read_text()))
        impact = {n: float(d.get("composite", 0.0)) for n, d in bundle.simulation.items()}
        out[sid] = {"InDeg": score(indeg(flow), impact, bundle.graph),
                    "Reach": score(reach(flow), impact, bundle.graph)}
    return out


def _zeroshot(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    d = _load(str(path))
    per = {sid: {"rho": s.get("mean_rho"), "rho_active": s.get("mean_rho_positive")}
           for sid, s in d["per_system"].items()}
    rhos = [v["rho"] for v in per.values() if v["rho"] is not None]
    return {"per_system": per, "mean_rho": float(np.mean(rhos)) if rhos else None,
            "provenance": d.get("provenance")}


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("results/loso_dependency_graph_cpu.json"))
    p.add_argument("--zeroshot-pattern", default="results/realworld_zeroshot_{v}_dependency_graph.json")
    p.add_argument("--probe", type=Path, default=Path("results/receptive_field_probe_dependency_graph.json"))
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--realworld-cache", type=Path, default=Path("output/realworld_cache"))
    p.add_argument("--output", type=Path, default=Path("results/dependency_graph_contrasts.json"))
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    artifact = _load(str(args.input))
    missing = sorted((set(ARMS) | set(REPRODUCED)) - set(artifact["per_variant_results"]))
    if missing:
        print(f"Error: {args.input} lacks {missing}", file=sys.stderr)
        return 2
    folds = list(FOLDS)

    # ── per-fold rows ────────────────────────────────────────────────────────
    rows: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        v: _loso_rows(artifact, v) for v in (*ARMS, *REPRODUCED)
    }
    native_prov: Dict[str, Any] = {}
    for v, (loso_path, _) in NATIVE_SOURCES.items():
        src = _load(loso_path)
        rows[f"{v}@published"] = _loso_rows(src, v)
        native_prov[v] = {"loso": loso_path, "provenance": src.get("provenance")}

    counts = _counts_on_cache(args.cache_dir, FOLDS)
    for c in ("InDeg", "Reach"):
        rows[c] = {sid: {"rho": counts[sid][c]["rho"], "rho_active": counts[sid][c]["rho_active"]}
                   for sid in folds}

    # Checks registered in Amendment 9: counts vs Amendment 7's artifact, and the
    # re-run native arms vs their earlier artifacts.
    tf = _load("results/tf_baselines.json")
    count_check = {
        c: max(abs(rows[c][sid]["rho"] - tf["per_fold"][FOLDS[sid]][c]["rho"]) for sid in folds)
        for c in ("InDeg", "Reach")
    }
    reproduced = {}
    for v, path in REPRODUCED.items():
        earlier = _loso_rows(_load(path), v)
        reproduced[v] = max(abs(rows[v][s]["rho"] - earlier[s]["rho"]) for s in folds)

    def col(v: str, key: str = "rho") -> List[Optional[float]]:
        return [rows[v][sid][key] for sid in folds]

    # ── registered family: 12 contrasts ──────────────────────────────────────
    contrasts: Dict[str, Dict[str, Any]] = {}
    for arm, native in ARMS.items():
        for ref, ref_key in (("InDeg", "InDeg"), ("Reach", "Reach"),
                             (registry.label(native, "loso"), f"{native}@published")):
            key = f"{registry.label(arm, 'loso')} vs {ref}"
            contrasts[key] = {"arm": arm, "reference": ref_key, **paired(col(arm), col(ref_key))}
    for k, ph in holm({k: c["p"] for k, c in contrasts.items()}).items():
        contrasts[k]["p_holm"] = ph

    # ── descriptive ──────────────────────────────────────────────────────────
    descriptive = {
        f"{registry.label(arm, 'loso')} vs {ref}": paired(col(arm), col(ref_key))
        for arm in ARMS
        for ref, ref_key in (("Topo-QoS", "topo_qos"),)
    }
    for arm in ARMS:
        descriptive[f"{registry.label(arm, 'loso')} vs GBM-Feat"] = paired(
            col(arm), [PUBLISHED_GBM_FEAT[FOLDS[s]] for s in folds])

    means = {
        v: {"label": (registry.label(v.split("@")[0], "loso") if v not in ("InDeg", "Reach") else v),
            "loso_mean_rho": _mean(col(v)), "loso_mean_rho_active": _mean(col(v, "rho_active"))}
        for v in rows
    }

    # ── zero-shot (descriptive) ──────────────────────────────────────────────
    sys_counts = _counts_on_cache(args.realworld_cache, SYSTEMS)
    zeroshot: Dict[str, Any] = {
        c: {"per_system": {sid: {"rho": sys_counts[sid][c]["rho"],
                                 "rho_active": sys_counts[sid][c]["rho_active"]} for sid in SYSTEMS},
            "mean_rho": _mean([sys_counts[sid][c]["rho"] for sid in SYSTEMS])}
        for c in ("InDeg", "Reach")
    }
    for arm, native in ARMS.items():
        zeroshot[arm] = _zeroshot(Path(args.zeroshot_pattern.format(v=arm)))
        zeroshot[f"{native}@published"] = _zeroshot(Path(NATIVE_SOURCES[native][1]))

    # ── decision rules, as registered ────────────────────────────────────────
    vs_indeg = {a: contrasts[f"{registry.label(a, 'loso')} vs InDeg"] for a in ARMS}
    vs_native = {a: contrasts[f"{registry.label(a, 'loso')} vs {registry.label(n, 'loso')}"]
                 for a, n in ARMS.items()}
    beats_indeg = [a for a, c in vs_indeg.items() if c["delta"] > 0 and c["p_holm"] < 0.05]
    differs_indeg = [a for a, c in vs_indeg.items() if c["p_holm"] < 0.05]
    zs_means = {a: (zeroshot[a] or {}).get("mean_rho") for a in ARMS}
    best_zs = max((a for a in ARMS if zs_means[a] is not None), key=lambda a: zs_means[a], default=None)
    decisions = {
        "D1": {"triggered": bool(beats_indeg), "arms": beats_indeg},
        "D2": {"triggered": not differs_indeg, "arms_differing": differs_indeg},
        "D3": {"triggered": all(c["delta"] < 0 for c in vs_indeg.values())},
        "M": {"triggered": any(c["delta"] > 0 and c["p_holm"] < 0.05 for c in vs_native.values()),
              "arms": [a for a, c in vs_native.items() if c["delta"] > 0 and c["p_holm"] < 0.05]},
        "Z": {"best_arm": best_zs, "best_mean": zs_means.get(best_zs) if best_zs else None,
              "threshold": REACH_SYSTEMS_PUBLISHED,
              "triggered": best_zs is not None and zs_means[best_zs] < REACH_SYSTEMS_PUBLISHED},
    }

    probe = _load(str(args.probe)) if args.probe.exists() else None
    out = {
        "provenance": stamp(input=str(args.input)),
        "input_provenance": artifact.get("provenance"),
        "native_sources": native_prov,
        "note": ("Amendment 9: exploratory family of 12, Holm within the family only; "
                 "not part of the manuscript omnibus."),
        "checks": {
            "counts_vs_tf_baselines_max_abs_diff": count_check,
            "counts_match_within_1e-3": all(v <= 1e-3 for v in count_check.values()),
            "reproduced_native_max_abs_diff": reproduced,
        },
        "per_fold": {v: rows[v] for v in rows},
        "means": means,
        "contrasts": contrasts,
        "descriptive": descriptive,
        "zeroshot": zeroshot,
        "decisions": decisions,
        "receptive_field_probe": probe and {
            "gradient_probe": probe.get("gradient_probe"),
            "edge_deletion_max_abs_delta": probe.get("edge_deletion_max_abs_delta"),
        },
    }
    args.output.write_text(json.dumps(out, indent=2))

    print(f"counts vs tf_baselines max |diff|: {count_check}")
    print(f"re-run native arms max |diff| vs earlier artifacts: {reproduced}")
    for v, m in means.items():
        print(f"  {m['label']:>14s} ({v:28s}) LOSO rho {m['loso_mean_rho']:.3f}")
    for k, c in contrasts.items():
        lo, hi = c["ci95"]
        print(f"{k:34s} {c['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] {c['won']:>2d}/{c['n']}  "
              f"p={c['p']:.4f}  p_holm={c['p_holm']:.3f}")
    for v, z in zeroshot.items():
        if z:
            print(f"zero-shot {v:30s} {z['mean_rho']:.3f}")
    print("decisions:", json.dumps(decisions))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
