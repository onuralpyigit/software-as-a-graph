#!/usr/bin/env python3
"""
reproduce/realworld_sigma_check.py — does the failure signature transfer?
=========================================================================

Section 7.2.3 claims the typed model's failure mode announces itself: on the
synthetic folds where HGT-QoS loses to Topo-QoS, the standard deviation of its
own predicted scores over the held-out Application population collapses toward
a constant. That quantity, written sigma-hat, needs no ground truth, so a
deployment could in principle refuse to trust a ranker before acting on it.

That claim was established on twelve folds drawn from one generator. This
script tests it where it actually matters -- on the five open-source systems,
which no part of the corpus produced -- by reusing the checkpoints
``reproduce/realworld_zeroshot.py`` already wrote. Inference only: no model is
retrained, so the sigma-hat reported here belongs to exactly the models whose
rho is reported there.

The test is directional and pre-stated: if the claim holds, the systems where
rho is lowest should carry the lowest sigma-hat, and the rank correlation
between sigma-hat and rho across the five systems should be positive.

With n = 5 systems no significance is claimed and none is reported; the
attainable two-sided floor for a rank correlation on five points is 0.0167 at
best and the sample cannot support an inference. This is a directional check on
held-out architectures, not a hypothesis test.

Usage
-----
    PYTHONPATH=. python reproduce/realworld_sigma_check.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import (  # noqa: E402
    _prepare_bundle_graph,
    discover_scenarios,
)
from saag.prediction.gnn_service import GNNService  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--realworld-cache", type=Path,
                   default=Path("output/realworld_cache"))
    p.add_argument("--ckpt-root", type=Path,
                   default=Path("output/realworld_zeroshot/hgl_qos"))
    p.add_argument("--zeroshot-result", type=Path,
                   default=RESULTS_DIR / "realworld_zeroshot.json")
    p.add_argument("--eval-population", default="application")
    p.add_argument("--output", type=Path,
                   default=RESULTS_DIR / "realworld_sigma_check.json")
    args = p.parse_args()

    if not args.zeroshot_result.exists():
        print(f"Error: {args.zeroshot_result} not found; run "
              "reproduce/realworld_zeroshot.py first.", file=sys.stderr)
        return 2
    zs = json.loads(args.zeroshot_result.read_text())

    seed_dirs = sorted(d for d in args.ckpt_root.glob("seed_*") if d.is_dir())
    if not seed_dirs:
        print(f"Error: no checkpoints under {args.ckpt_root}", file=sys.stderr)
        return 2

    real = discover_scenarios(args.realworld_cache, [], min_scenarios=1)
    use_qos = zs.get("variant", "hgl_qos") == "hgl_qos"

    # sigma-hat per (system, seed), then averaged over seeds -- matching how the
    # rho it is compared against was aggregated.
    per_system: Dict[str, Dict[str, Any]] = {}
    for b in real:
        graph, sm = _prepare_bundle_graph(b, use_qos)
        # The population sigma-hat is measured over must be the population rho
        # was scored on, or the two quantities describe different node sets.
        app_ids = {
            n for n, d in b.graph.nodes(data=True)
            if d.get("component_type", d.get("type")) == "Application"
        }
        sigmas, spreads = [], []
        for sd in seed_dirs:
            try:
                svc = GNNService.from_checkpoint(str(sd), graph=graph, layer="app")
                res = svc.predict(
                    graph=graph, structural_metrics=sm, rm_scores=b.rm,
                    eval_labels=b.simulation, mode="gnn", qos_enabled=use_qos,
                )
            except Exception as exc:                       # noqa: BLE001
                logger.error("  %s / %s failed: %s", b.scenario_id, sd.name, exc)
                continue
            vals = [float(ns.composite_score)
                    for nid, ns in res.node_scores.items() if nid in app_ids]
            if len(vals) > 1:
                sigmas.append(float(np.std(vals)))
        truth = [float(d.get("composite", 0.0))
                 for nid, d in b.simulation.items() if nid in app_ids]
        per_system[b.scenario_id] = {
            "n_seeds": len(sigmas),
            "n_app": len(app_ids),
            "sigma_hat": float(np.mean(sigmas)) if sigmas else float("nan"),
            "sigma_hat_sd": float(np.std(sigmas)) if sigmas else float("nan"),
            "truth_spread": float(np.std(truth)) if len(truth) > 1 else float("nan"),
            "rho": zs["per_system"].get(b.scenario_id, {}).get("mean_rho"),
        }

    sids = [s for s in sorted(per_system)
            if per_system[s]["n_seeds"] and per_system[s]["rho"] is not None]
    sig = np.array([per_system[s]["sigma_hat"] for s in sids])
    rho = np.array([per_system[s]["rho"] for s in sids])
    corr = float(spearmanr(sig, rho).statistic) if len(sids) > 2 else float("nan")

    # Two readings, and they disagree. As a *ranking* among real systems,
    # sigma-hat is tested by the correlation above. As an absolute *gate*, the
    # question is whether every real system falls below the spread the synthetic
    # folds showed -- in which case a threshold calibrated on the corpus rejects
    # them all, which is the right call at a mean rho of 0.125 even though it
    # discriminates between none of them.
    SYNTHETIC_MEDIAN_SIGMA = 0.155   # Section 7.2.3, twelve synthetic LOSO folds
    below = [s for s in sids if per_system[s]["sigma_hat"] < SYNTHETIC_MEDIAN_SIGMA]

    payload = {
        "variant": zs.get("variant"),
        "label": zs.get("label"),
        "oracle": zs.get("oracle"),
        "synthetic_median_sigma_reference": SYNTHETIC_MEDIAN_SIGMA,
        "n_below_synthetic_median": len(below),
        "systems_below_synthetic_median": below,
        "absolute_gate_note": (
            "The synthetic median is quoted from Section 7.2.3 and was measured "
            "on the pre-regeneration corpus, so this comparison is across "
            "corpora and is indicative rather than exact."
        ),
        "seeds": [d.name for d in seed_dirs],
        "eval_population": args.eval_population,
        "per_system": per_system,
        "spearman_sigma_vs_rho": corr,
        "claim_under_test": (
            "Section 7.2.3: sigma-hat (spread of the model's own predictions, "
            "label-free) collapses on the architectures where the model loses."
        ),
        "inference_note": (
            "n = 5 systems. No significance is claimed or reported; the sample "
            "cannot support one. Directional check only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n  sigma-hat vs rho on the five open-source systems "
          f"({zs.get('label')}, {len(seed_dirs)} seeds, inference only)")
    print("  " + "─" * 72)
    print(f"  {'system':<34}{'sigma_hat':>11}{'truth sd':>10}{'rho':>10}")
    for s in sorted(sids, key=lambda x: per_system[x]["sigma_hat"]):
        e = per_system[s]
        print(f"  {s:<34}{e['sigma_hat']:>11.4f}{e['truth_spread']:>10.4f}"
              f"{e['rho']:>10.4f}")
    print(f"\n  Spearman(sigma_hat, rho) over {len(sids)} systems = {corr:+.3f}")
    print("  Positive means low prediction spread accompanies poor ranking, "
          "which is\n  the direction Section 7.2.3 predicts. n = 5: directional, "
          "not a test.")
    print(f"\n  As an absolute gate: {len(below)} of {len(sids)} systems fall below the "
          f"synthetic\n  median sigma-hat of {SYNTHETIC_MEDIAN_SIGMA} quoted in Section 7.2.3 "
          "(cross-corpus, indicative).")
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
