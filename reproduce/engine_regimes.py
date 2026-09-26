#!/usr/bin/env python3
"""
reproduce/engine_regimes.py — where graph learning helps, and where it does not
===============================================================================

Post hoc and exploratory; trains nothing. Collects the per-fold LOSO results of
every learned and closed-form arm from the one-sweep artifacts the manuscript
already reports, joins them with per-scenario descriptors read from the corpus,
and reports:

  * per-fold rho and active-stratum rho for every arm;
  * regimes: folds grouped into terciles of the closed-form engine's accuracy
    (Topo-QoS rho), with each arm's mean rho and gain per tercile;
  * the descriptor x gain Spearman matrix over the twelve folds, with
    Benjamini-Hochberg q-values across the whole matrix;
  * whether HGT-QoS's receptive-field coverage of an Application
    (results/receptive_field_probe.json) explains its per-fold accuracy;
  * zero-shot per-system rho and active-stratum rho on the five system models.

Backs manuscript Section 8.2 (regime table) and the supplement's Engine Regimes
section.

Usage:
    PYTHONPATH=. python reproduce/engine_regimes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
if not (RESULTS / "loso_all_variants_v5.json").exists():
    alt = ROOT / "data" / "benchmarks"
    if (alt / "loso_all_variants_v5.json").exists():
        RESULTS = alt
CORPUS = ROOT / "data/scenarios"

#: arm -> (artifact, variant id). Each arm is read from the sweep the
#: manuscript reports it from, so per-fold means equal the printed means.
LOSO_ARMS = {
    "Topo": ("loso_hybrid_cpu.json", "topo_baseline"),
    "Topo-QoS": ("loso_rq2_matched.json", "topo_qos"),
    "GAT": ("loso_rq2_matched.json", "gl_full_cap"),
    "GAT-QoS": ("loso_rq2_matched.json", "gl_full_qos16_cap"),
    "HGT": ("loso_rq2_matched.json", "hgl"),
    "HGT-QoS": ("loso_rq2_matched.json", "hgl_qos"),
    "Hybrid-HGT": ("loso_hybrid_cpu.json", "hgl_qos_prior"),
    "Hybrid-GAT": ("loso_hybrid_gat_cpu.json", "gl_qos16_prior"),
    "HGT-QoS-U": ("loso_directionality_cpu.json", "hgl_qos_uni"),
    "GAT-w": ("loso_capacity_cpu.json", "gl_full_qos_cap"),
    "GAT-QoS-nf": ("loso_attribution_cpu.json", "gl_full_qos16_nfmask"),
    "GBM-Feat": ("loso_attribution_cpu.json", "tab_gbm"),
    "GBM-Feat-QoS": ("loso_attribution_cpu.json", "tab_gbm_qos"),
}

ZERO_SHOT_ARMS = {
    "GAT": "realworld_zeroshot_gl_full_cap_attribution.json",
    "GAT-QoS": "realworld_zeroshot_gl_full_qos16_cap_cpu.json",
    "GAT-w": "realworld_zeroshot_gl_full_qos_cap_capacity.json",
    "HGT-QoS": "realworld_zeroshot_hgl_qos_cpu.json",
    "HGT-QoS-U": "realworld_zeroshot_hgl_qos_uni_directionality.json",
    "Hybrid-HGT": "realworld_zeroshot_hgl_qos_prior_cpu.json",
    "Hybrid-GAT": "realworld_zeroshot_gl_qos16_prior_cpu.json",
    "GBM-Feat": "realworld_zeroshot_tab_gbm_attribution.json",
    "GBM-Feat-QoS": "realworld_zeroshot_tab_gbm_qos_attribution.json",
}
#: training-free references, read from the HGT-QoS zero-shot artifact
ZERO_SHOT_REFS = ("Topo", "Topo-QoS")

#: per-fold quantities correlated against the descriptors
GAINS = {
    "HGT-QoS - Topo-QoS": ("HGT-QoS", "Topo-QoS"),
    "GAT-QoS - Topo-QoS": ("GAT-QoS", "Topo-QoS"),
    "GBM-Feat - Topo-QoS": ("GBM-Feat", "Topo-QoS"),
    "HGT - GAT": ("HGT", "GAT"),
    "HGT-QoS - GAT-QoS": ("HGT-QoS", "GAT-QoS"),
    "Hybrid-HGT - HGT-QoS": ("Hybrid-HGT", "HGT-QoS"),
    "GAT-QoS - GAT": ("GAT-QoS", "GAT"),
    "GAT-QoS - GBM-Feat-QoS": ("GAT-QoS", "GBM-Feat-QoS"),
}

#: descriptor key in system_model_descriptives.json per LOSO fold
DESCRIPTIVE_NAME = {
    "atm_system": "ATM",
    "av_system": "AV System",
    "enterprise_system": "Enterprise",
    "financial_trading_system": "Financial Trading",
    "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "Enterprise Integration (ESB)",
    "industrial_scada_system": "Industrial SCADA",
    "iot_smart_city_system": "IoT Smart City",
    "logistics_fleet_system": "Logistics Fleet",
    "microservices_system": "Microservices",
    "realtime_gaming_system": "Real-Time Gaming",
    "telecom_ran_system": "Telecom RAN",
}

#: Row labels of the supplement's Engine Regimes tables.
FOLD_LABELS = {
    "atm_system": "ATM",
    "av_system": "AV System",
    "enterprise_system": "Enterprise",
    "financial_trading_system": "Financial Trading",
    "healthcare_system": "Healthcare",
    "hub_and_spoke_system": "ESB (Hub-and-Spoke)",
    "industrial_scada_system": "Industrial SCADA",
    "iot_smart_city_system": "IoT Smart City",
    "logistics_fleet_system": "Logistics Fleet",
    "microservices_system": "Microservices",
    "realtime_gaming_system": "Real-Time Gaming",
    "telecom_ran_system": "Telecom RAN",
}
SYSTEM_LABELS = {
    "realworld_autoware_ros2": "Autoware (ROS 2)",
    "realworld_edgex": "EdgeX Foundry",
    "realworld_homeassistant": "Home Assistant",
    "realworld_cloud_microservices": "Online Boutique (model)",
    "realworld_trainticket": "Train-Ticket (model)",
}
#: System models whose original is a publish-subscribe system; the other two are RPC systems.
PUBSUB_ORIGINALS = ("realworld_autoware_ros2", "realworld_edgex", "realworld_homeassistant")
#: The pure learned engines of the zero-shot comparison (no closed-form prior).
PURE_LEARNED = ("GAT", "GAT-QoS", "GAT-w", "HGT-QoS", "HGT-QoS-U")


def _folds(artifact: str, variant: str) -> dict:
    rows = json.loads((RESULTS / artifact).read_text())["per_variant_results"][variant]["folds"]
    return {
        f["holdout_id"]: {
            "rho": float(f["mean_metrics"]["spearman_rho"]),
            "rho_pos": float(f["mean_metrics"]["spearman_rho_positive"]),
        }
        for f in rows
    }


def _gini(xs: list) -> float:
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    return float(sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s)) if s else 0.0


def _descriptors(fold: str, descriptives: dict) -> dict:
    d = json.loads((CORPUS / f"{fold}.json").read_text())
    rel = d["relationships"]
    n_v = sum(len(d[k]) for k in ("applications", "topics", "brokers", "nodes", "libraries"))
    subs = Counter(r["to"] for r in rel["subscribes_to"])
    extra = descriptives[DESCRIPTIVE_NAME[fold]]
    return {
        "n_apps": len(d["applications"]),
        "n_topics": len(d["topics"]),
        "n_brokers": len(d["brokers"]),
        "n_hosts": len(d["nodes"]),
        "n_libraries": len(d["libraries"]),
        "edges_per_node": sum(len(v) for v in rel.values()) / n_v,
        "uses_per_app": len(rel["uses"]) / len(d["applications"]),
        "qos_profiles": len({json.dumps(t.get("qos", {}), sort_keys=True) for t in d["topics"]}),
        "subscriber_gini": _gini([subs.get(t["id"], 0) for t in d["topics"]]),
        "zero_share": float(extra["zero_share"]),
        "tie_fraction": float(extra["tie_fraction"]),
        "projection_density": float(extra["projection_density"]),
    }


def _bh(ps: list) -> list:
    """Benjamini-Hochberg q-values, in input order."""
    m = len(ps)
    order = np.argsort(ps)
    q = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, ps[i] * m / rank)
        q[i] = running
    return q.tolist()


def build() -> dict:
    """Everything the artifact holds except its provenance stamp."""
    per_fold = {arm: _folds(*src) for arm, src in LOSO_ARMS.items()}
    folds = sorted(per_fold["Topo-QoS"])
    for arm, rows in per_fold.items():
        if sorted(rows) != folds:
            raise ValueError(f"{arm} does not cover the twelve LOSO folds")
    means = {arm: float(np.mean([r[f]["rho"] for f in folds])) for arm, r in per_fold.items()}
    means_pos = {arm: float(np.mean([r[f]["rho_pos"] for f in folds])) for arm, r in per_fold.items()}

    # Regimes: terciles of the closed-form engine's per-fold accuracy.
    by_topo = sorted(folds, key=lambda f: per_fold["Topo-QoS"][f]["rho"])
    terciles = {"weak": by_topo[:4], "middle": by_topo[4:8], "strong": by_topo[8:]}
    regimes = {}
    for name, members in terciles.items():
        regimes[name] = {
            "folds": members,
            "mean_rho": {a: float(np.mean([per_fold[a][f]["rho"] for f in members])) for a in LOSO_ARMS},
            "gain_vs_topo_qos": {
                a: float(np.mean([per_fold[a][f]["rho"] - per_fold["Topo-QoS"][f]["rho"] for f in members]))
                for a in LOSO_ARMS if a != "Topo-QoS"
            },
            "wins_vs_topo_qos": {
                a: int(sum(per_fold[a][f]["rho"] > per_fold["Topo-QoS"][f]["rho"] for f in members))
                for a in LOSO_ARMS if a != "Topo-QoS"
            },
        }

    descriptives = json.loads((RESULTS / "system_model_descriptives.json").read_text())["loso"]
    descriptors = {f: _descriptors(f, descriptives) for f in folds}
    cells = []
    for gain, (a, b) in GAINS.items():
        ys = [per_fold[a][f]["rho"] - per_fold[b][f]["rho"] for f in folds]
        for desc in descriptors[folds[0]]:
            r, pv = spearmanr([descriptors[f][desc] for f in folds], ys)
            cells.append({"quantity": gain, "descriptor": desc, "rho": float(r), "p": float(pv)})
    for c, q in zip(cells, _bh([c["p"] for c in cells])):
        c["q_bh"] = q

    # Does HGT-QoS's receptive-field coverage of an Application explain its fold profile?
    rf = json.loads((RESULTS / "receptive_field_probe.json").read_text())["gradient_probe"]
    share = [rf[f]["hgl_qos"]["mean_rf_share"] for f in folds]
    coverage = {"hgl_qos_rf_share": dict(zip(folds, share))}
    for name, ys in (
        ("HGT-QoS rho", [per_fold["HGT-QoS"][f]["rho"] for f in folds]),
        ("HGT-QoS - Topo-QoS", [per_fold["HGT-QoS"][f]["rho"] - per_fold["Topo-QoS"][f]["rho"] for f in folds]),
        ("Hybrid-HGT - HGT-QoS", [per_fold["Hybrid-HGT"][f]["rho"] - per_fold["HGT-QoS"][f]["rho"] for f in folds]),
    ):
        r, pv = spearmanr(share, ys)
        coverage[name] = {"rho": float(r), "p": float(pv)}

    ref = json.loads((RESULTS / ZERO_SHOT_ARMS["HGT-QoS"]).read_text())["references"]
    zero_shot = {}
    for arm, artifact in ZERO_SHOT_ARMS.items():
        per_system = json.loads((RESULTS / artifact).read_text())["per_system"]
        zero_shot[arm] = {s: {"rho": float(v["mean_rho"]), "rho_pos": float(v["mean_rho_positive"])}
                          for s, v in per_system.items()}
    for arm in ZERO_SHOT_REFS:
        zero_shot[arm] = {s: {"rho": float(v["rho"]), "rho_pos": float(v["rho_positive"])}
                          for s, v in ref[arm].items()}
    systems = sorted(zero_shot["HGT-QoS"])
    zs_means = {a: float(np.mean([r[s]["rho"] for s in systems])) for a, r in zero_shot.items()}
    zs_means_pos = {a: float(np.mean([r[s]["rho_pos"] for s in systems])) for a, r in zero_shot.items()}

    return {
        "note": "Post hoc and exploratory; trains nothing. BH q-values span the whole descriptor x gain matrix.",
        "loso": {
            "folds": folds,
            "mean_rho": means,
            "mean_rho_pos": means_pos,
            "per_fold": per_fold,
            "regimes_by_topo_qos_tercile": regimes,
            "descriptors": descriptors,
            "correlations": cells,
            "n_tests": len(cells),
            "min_q_bh": min(c["q_bh"] for c in cells),
            "receptive_field_coverage": coverage,
        },
        "zero_shot": {
            "systems": systems,
            "mean_rho": zs_means,
            "mean_rho_pos": zs_means_pos,
            "per_system": zero_shot,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=RESULTS / "engine_regimes.json")
    args = p.parse_args()

    try:
        out = build()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    out = {"provenance": stamp(loso_arms={a: f"{s[0]}:{s[1]}" for a, s in LOSO_ARMS.items()},
                               zero_shot_arms=ZERO_SHOT_ARMS), **out}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    means, means_pos = out["loso"]["mean_rho"], out["loso"]["mean_rho_pos"]
    zs_means, zs_means_pos = out["zero_shot"]["mean_rho"], out["zero_shot"]["mean_rho_pos"]
    regimes, cells = out["loso"]["regimes_by_topo_qos_tercile"], out["loso"]["correlations"]
    coverage = out["loso"]["receptive_field_coverage"]

    print(f"{'arm':14s} {'LOSO rho':>9s} {'rho>0':>7s} {'ZS rho':>7s} {'ZS rho>0':>8s}")
    for arm in sorted(set(means) | set(zs_means), key=lambda a: -means.get(a, 0)):
        fmt = lambda d, k: f"{d[k]:.3f}" if k in d else "  -  "
        print(f"{arm:14s} {fmt(means, arm):>9s} {fmt(means_pos, arm):>7s} "
              f"{fmt(zs_means, arm):>7s} {fmt(zs_means_pos, arm):>8s}")
    for name, reg in regimes.items():
        print(f"{name:7s} {','.join(f.replace('_system', '') for f in reg['folds'])}: "
              + " ".join(f"{a}={reg['gain_vs_topo_qos'][a]:+.3f}"
                         for a in ("HGT-QoS", "GAT-QoS", "GBM-Feat", "Hybrid-HGT", "Hybrid-GAT")))
    top = sorted(cells, key=lambda c: c["p"])[:8]
    print(f"{len(cells)} correlations; smallest q_BH = {out['loso']['min_q_bh']:.3f}")
    for c in top:
        print(f"  {c['quantity']:24s} ~ {c['descriptor']:18s} rho={c['rho']:+.2f} p={c['p']:.3f} q={c['q_bh']:.2f}")
    print("RF coverage: " + ", ".join(f"{k} rho={v['rho']:+.2f} p={v['p']:.2f}"
                                       for k, v in coverage.items() if k != "hgl_qos_rf_share"))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
