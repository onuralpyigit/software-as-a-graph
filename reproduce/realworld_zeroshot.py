#!/usr/bin/env python3
"""
reproduce/realworld_zeroshot.py — the learned model on systems we did not generate
=================================================================================

Trains HGT-QoS once on the full synthetic corpus and scores it zero-shot on the
five open-source reference systems. Closes the gap Section 7.4 states plainly:
the real-world table reports only the deterministic explanation layer Q(v) and
the training-free topological baselines, so the typing claim of Section 7.2 has
never been tested outside our own generator.

Oracle
------
This is the part that cannot be shortcut. Table 9's ground truth is
``cli/validate_graph.py``'s composite cascade impact I_comp(v), produced by
``FailureSimulator`` — the Validate-stage oracle. The GNN is trained against
I*(v), produced by ``FaultInjector`` — the Predict-stage labeler. CLAUDE.md's
ground-truth contract forbids substituting one for the other within a stage
(``tests/test_groundtruth_contract.py``), so this script scores the learned
model against I*(v) and reports it as its **own table**, not as a column
appended to Table 9. Mixing the two oracles in one table would make the
learned column incomparable to the ones beside it while looking comparable.

Populate the real-world cache first, into a directory that is NOT
``output/loso_cache`` — ``discover_scenarios`` treats every directory it finds
there as a LOSO fold, so caching the real systems alongside the synthetic ones
would silently change the twelve-fold corpus behind every published number::

    CACHE_DIR=output/realworld_cache bash scripts/populate_loso_cache.sh \\
        realworld_autoware_ros2 realworld_cloud_microservices \\
        realworld_trainticket realworld_homeassistant realworld_edgex

Usage
-----
    PYTHONPATH=. python reproduce/realworld_zeroshot.py
    PYTHONPATH=. python reproduce/realworld_zeroshot.py --seeds 42,123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import (  # noqa: E402
    ScenarioBundle,
    _build_training_hetero,
    _build_validation_hetero,
    _prepare_bundle_graph,
    _select_val_bundle,
    compute_inductive_metrics,
    discover_scenarios,
)
from saag.evaluation import variant_registry as _registry  # noqa: E402
from saag.prediction.gnn_service import GNNService  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def train_once(
    bundles: List[ScenarioBundle],
    seed: int,
    ckpt_dir: Path,
    *,
    use_qos: bool,
    epochs: int,
    layers: int,
    rank_normalize_features: bool,
    rank_normalize_labels: bool,
) -> GNNService:
    """Train one HGT on the whole synthetic corpus.

    Mirrors the ``hgl_qos`` branch of ``cli.loso_evaluate.run_one_fold``: the
    largest scenario is the primary graph the splits are drawn on, one
    median-sized scenario is held out of the loss to drive early stopping, and
    the rest arrive through the inductive-graph channel. The difference is that
    nothing is held out for *testing* — the test set is the real-world corpus,
    which lives in a different cache entirely.
    """
    primary = max(bundles, key=lambda b: b.n_nodes)
    inductives = [b for b in bundles if b.scenario_id != primary.scenario_id]
    val_bundle = _select_val_bundle(inductives, "auto")
    if val_bundle is not None:
        inductives = [b for b in inductives if b.scenario_id != val_bundle.scenario_id]

    logger.info(
        "  seed %d: primary=%s (|V|=%d), inductive=%d, val=%s",
        seed, primary.scenario_id, primary.n_nodes, len(inductives),
        val_bundle.scenario_id if val_bundle else "(none)",
    )

    train_graph, train_sm = _prepare_bundle_graph(primary, use_qos)
    service = GNNService(
        checkpoint_dir=str(ckpt_dir),
        hidden_channels=64,
        num_heads=4,
        num_layers=layers,
        dropout=0.2,
        predict_edges=False,
    )
    service.train(
        graph=train_graph,
        structural_metrics=train_sm,
        simulation_results=primary.simulation,
        rm_scores=primary.rm,
        inductive_graphs=[
            _build_training_hetero(b, use_qos, rank_normalize_features)
            for b in inductives
        ],
        val_graph=(
            _build_validation_hetero(val_bundle, use_qos, rank_normalize_features)
            if val_bundle is not None else None
        ),
        seeds=[seed],
        num_epochs=epochs,
        lr=3e-4,
        patience=min(60, epochs),
        layer="app",
        qos_enabled=use_qos,
        rank_normalize_features=rank_normalize_features,
        rank_normalize_labels=rank_normalize_labels,
    )
    return service


def score(service: GNNService, bundle: ScenarioBundle, *, use_qos: bool,
          population: str) -> Dict[str, Any]:
    """Zero-shot predict on one real system and score against its I*(v) labels."""
    graph, sm = _prepare_bundle_graph(bundle, use_qos)
    result = service.predict(
        graph=graph,
        structural_metrics=sm,
        rm_scores=bundle.rm,
        eval_labels=bundle.simulation,
        mode="gnn",
        qos_enabled=use_qos,
    )
    pred = {nid: float(ns.composite_score) for nid, ns in result.node_scores.items()}
    true_impact = {
        nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()
    }
    return compute_inductive_metrics(
        pred, true_impact, bundle.graph, population=population
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-cache", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--realworld-cache", type=Path, default=Path("output/realworld_cache"))
    p.add_argument("--variant", default="hgl_qos", choices=["hgl_qos", "hgl"])
    p.add_argument("--seeds", default="42,123,456,789,2024")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--eval-population", default="application",
                   choices=["application", "app_lib", "labeled"])
    p.add_argument("--rank-normalize-features", action="store_true")
    p.add_argument("--rank-normalize-labels", action="store_true")
    p.add_argument("--workdir", type=Path, default=Path("output/realworld_zeroshot"))
    p.add_argument("--output", type=Path,
                   default=RESULTS_DIR / "realworld_zeroshot.json")
    args = p.parse_args()

    if not args.realworld_cache.exists():
        print(
            f"Error: {args.realworld_cache} not found. Populate it first:\n"
            f"  CACHE_DIR={args.realworld_cache} bash scripts/populate_loso_cache.sh "
            f"realworld_autoware_ros2 realworld_cloud_microservices "
            f"realworld_trainticket realworld_homeassistant realworld_edgex",
            file=sys.stderr,
        )
        return 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    use_qos = (args.variant == "hgl_qos")

    synthetic = discover_scenarios(args.synthetic_cache, [])
    real = discover_scenarios(args.realworld_cache, [], min_scenarios=1)
    logger.info("Synthetic training corpus: %d scenarios", len(synthetic))
    logger.info("Real-world evaluation corpus: %d systems", len(real))

    # A real system that leaked into the training cache would make this a
    # transductive evaluation while still being reported as zero-shot.
    overlap = {b.scenario_id for b in synthetic} & {b.scenario_id for b in real}
    if overlap:
        print(f"Error: {sorted(overlap)} appear in BOTH caches; "
              "this would not be a zero-shot evaluation.", file=sys.stderr)
        return 2

    per_system: Dict[str, List[Dict[str, Any]]] = {b.scenario_id: [] for b in real}
    t0 = time.time()
    for seed in seeds:
        ckpt = args.workdir / args.variant / f"seed_{seed}"
        ckpt.mkdir(parents=True, exist_ok=True)
        service = train_once(
            synthetic, seed, ckpt,
            use_qos=use_qos, epochs=args.epochs, layers=args.layers,
            rank_normalize_features=args.rank_normalize_features,
            rank_normalize_labels=args.rank_normalize_labels,
        )
        for b in real:
            try:
                m = score(service, b, use_qos=use_qos,
                          population=args.eval_population)
            except Exception as exc:                      # noqa: BLE001
                logger.error("  %s seed %d failed: %s", b.scenario_id, seed, exc,
                             exc_info=True)
                continue
            m["seed"] = seed
            per_system[b.scenario_id].append(m)
            logger.info("    %-32s rho=%.4f  F1@K=%.4f  n=%d",
                        b.scenario_id, m["spearman_rho"], m["f1_at_k"], m["n"])

    summary: Dict[str, Any] = {}
    for sid, runs in per_system.items():
        if not runs:
            summary[sid] = {"n_seeds": 0}
            continue
        rho = [r["spearman_rho"] for r in runs]
        f1 = [r["f1_at_k"] for r in runs]
        bundle = next(b for b in real if b.scenario_id == sid)
        summary[sid] = {
            "n_seeds": len(runs),
            "n_nodes": bundle.n_nodes,
            "n_evaluated": runs[0].get("n"),
            "mean_rho": float(np.mean(rho)),
            "std_rho": float(np.std(rho)),
            "mean_f1_at_k": float(np.mean(f1)),
            "std_f1_at_k": float(np.std(f1)),
            "label_stability": bundle.label_stability,
            "labeler": bundle.labeler,
        }

    scored = [s for s in summary.values() if s.get("n_seeds")]
    payload = {
        "variant": args.variant,
        "label": _registry.label(args.variant, harness="loso"),
        "oracle": "I*(v) / FaultInjector",
        "oracle_note": (
            "Scored against I*(v), the oracle the model was trained on. Table 9's "
            "Q(v) and Topo columns are scored against I_comp(v) (FailureSimulator) "
            "and are NOT directly comparable to these numbers."
        ),
        "training_corpus": sorted(b.scenario_id for b in synthetic),
        "eval_population": args.eval_population,
        "seeds": seeds,
        "epochs": args.epochs,
        "layers": args.layers,
        "rank_normalize_features": args.rank_normalize_features,
        "rank_normalize_labels": args.rank_normalize_labels,
        "elapsed_s": time.time() - t0,
        "per_system": summary,
        "mean_rho_across_systems": (
            float(np.mean([s["mean_rho"] for s in scored])) if scored else None
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n  {payload['label']} zero-shot on real systems "
          f"(oracle: {payload['oracle']})")
    print("  " + "─" * 74)
    print(f"  {'system':<34}{'|V|':>6}{'rho':>10}{'sd':>8}{'F1@K':>9}")
    for sid in sorted(summary):
        s = summary[sid]
        if not s.get("n_seeds"):
            print(f"  {sid:<34}{'—':>6}{'failed':>10}")
            continue
        print(f"  {sid:<34}{s['n_nodes']:>6}{s['mean_rho']:>10.4f}"
              f"{s['std_rho']:>8.4f}{s['mean_f1_at_k']:>9.4f}")
    if payload["mean_rho_across_systems"] is not None:
        print(f"  {'mean across systems':<34}{'':>6}"
              f"{payload['mean_rho_across_systems']:>10.4f}")
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
