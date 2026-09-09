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
          population: str, rank_normalize_features: bool = True) -> Dict[str, Any]:
    """Zero-shot predict on one real system and score against its I*(v) labels."""
    graph, sm = _prepare_bundle_graph(bundle, use_qos)
    result = service.predict(
        graph=graph,
        structural_metrics=sm,
        rm_scores=bundle.rm,
        eval_labels=bundle.simulation,
        mode="gnn",
        qos_enabled=use_qos,
        rank_normalize_features=rank_normalize_features,
    )
    pred = {nid: float(ns.composite_score) for nid, ns in result.node_scores.items()}
    true_impact = {
        nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()
    }
    m = compute_inductive_metrics(
        pred, true_impact, bundle.graph, population=population
    )

    # Topology prior for hybrid evaluation
    topo_pred = {
        nid: 0.6 * float(m_dict.get("betweenness_centrality", 0.0))
           + 0.4 * float(m_dict.get("ap_c_score", 0.0))
        for nid, m_dict in (bundle.structural or {}).items()
    }
    max_t = max(topo_pred.values()) if topo_pred and max(topo_pred.values()) > 0 else 1.0
    norm_t = {k: v / max_t for k, v in topo_pred.items()}
    max_g = max(pred.values()) if pred and max(pred.values()) > 0 else 1.0
    norm_g = {k: v / max_g for k, v in pred.items()}
    pred_hybrid = {k: 0.5 * norm_t.get(k, 0.0) + 0.5 * norm_g.get(k, 0.0) for k in pred}
    m_hybrid = compute_inductive_metrics(
        pred_hybrid, true_impact, bundle.graph, population=population
    )
    m["hybrid_spearman_rho"] = float(m_hybrid.get("spearman_rho", 0.0))
    m["hybrid_f1_at_k"] = float(m_hybrid.get("f1_at_k", 0.0))

    # Active strata (positive ground truth only, per tests/test_zero_exclusion.py)
    pos_impact = {nid: val for nid, val in true_impact.items() if val > 0}
    if len(pos_impact) >= 3:
        m_pos = compute_inductive_metrics(
            pred, pos_impact, bundle.graph, population=population
        )
        m["spearman_rho_positive"] = (
            float(m_pos.get("spearman_rho")) if m_pos.get("spearman_rho") is not None else None
        )
        m["n_positive"] = len(pos_impact)
    else:
        m["spearman_rho_positive"] = None
        m["n_positive"] = len(pos_impact)

    return m


def score_references(bundle: ScenarioBundle, *, population: str) -> Dict[str, Any]:
    """Score the training-free references against the SAME oracle and population.

    Without these the learned number is uninterpretable. The Q(v) figures in
    Section 7.4 are scored against I_comp(v) (FailureSimulator), so they cannot
    be set beside a model trained and scored on I*(v) — the comparison has to be
    rebuilt on one oracle, which is what this does.

    ``Topo`` is 0.6*betweenness + 0.4*articulation, the same combination
    ``reproduce.main_table._compute_topo_baseline_scores`` uses, read off the
    cached app-layer metrics (already the DEPENDS_ON projection, since the cache
    is built with ``analyze_graph.py --layer app``). ``Topo-QoS`` is the same
    combination with QoS-weighted betweenness, computed by the same function.

    Topo-QoS was previously omitted here on the grounds that the cache carried no
    QoS edge weights. It did not, but the cause was a defect rather than a
    property of the data: the architecture adapters emit ``weight: 1.0`` on every
    edge while the generator omits the key, and ``_project_topic_qos_onto_edges``
    guarded on key *absence*, so the QoS-derived weight was applied to generated
    topologies and skipped on transcribed ones. With that guard corrected the
    weights are present (48-66% of edges non-unit across the five systems) and
    the baseline is computable on the same footing as everywhere else.
    """
    true_impact = {
        nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()
    }
    out: Dict[str, Any] = {}

    rm_pred = {nid: float(v.get("overall", 0.0)) for nid, v in (bundle.rm or {}).items()}
    if rm_pred:
        out["RM"] = compute_inductive_metrics(
            rm_pred, true_impact, bundle.graph, population=population
        )

    topo_pred = {
        nid: 0.6 * float(m.get("betweenness_centrality", 0.0))
           + 0.4 * float(m.get("ap_c_score", 0.0))
        for nid, m in (bundle.structural or {}).items()
    }
    if topo_pred and any(v > 0 for v in topo_pred.values()):
        out["Topo"] = compute_inductive_metrics(
            topo_pred, true_impact, bundle.graph, population=population
        )

    # Topo-QoS via the canonical implementation, so this column and the synthetic
    # tables cannot drift apart. It returns None when the graph yields no signal,
    # and _qos_weighted_betweenness falls back to unweighted betweenness when no
    # QoS weights are present -- in which case Topo-QoS would equal Topo, so we
    # record whether the weights were actually there rather than leaving a
    # silently duplicated column.
    from reproduce.main_table import (
        _compute_topo_baseline_scores,
        _qos_weighted_betweenness,
    )
    n_weighted = sum(
        1 for _, _, d in bundle.graph.edges(data=True)
        if abs(float(d.get("qos_weight", d.get("weight", 1.0))) - 1.0) > 1e-9
    )
    topo_qos_pred = _compute_topo_baseline_scores(
        bundle.graph, bundle.structural, use_qos=True
    )
    app_ids = [n for n, d in bundle.graph.nodes(data=True)
               if d.get("type") == "Application"]
    scored = {v for nid, v in (topo_qos_pred or {}).items() if nid in set(app_ids)}
    if topo_qos_pred and len(scored) > 1:
        m = compute_inductive_metrics(
            topo_qos_pred, true_impact, bundle.graph, population=population
        )
        m["n_qos_weighted_edges"] = n_weighted
        out["Topo-QoS"] = m
    else:
        # Degenerate, and the reason is structural rather than incidental, so it
        # is recorded instead of emitted as a NaN column. ``Topo`` reads
        # betweenness off the cached *app-layer projection*; the QoS-weighted
        # variant recomputes it on the raw multigraph, where Applications never
        # route messages and betweenness is identically zero for all of them
        # (the degeneracy Section 6.2.2 gives as the reason topological
        # baselines run on the projection at all). Scoring Topo-QoS here needs
        # the derived DEPENDS_ON projection as a graph object; this cache stores
        # that layer only as precomputed scalars.
        out["Topo-QoS"] = {
            "unavailable": "qos_weighted_betweenness_degenerate_on_raw_multigraph",
            "n_qos_weighted_edges": n_weighted,
            "n_distinct_app_scores": len(scored),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-cache", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--realworld-cache", type=Path, default=Path("output/realworld_cache"))
    p.add_argument("--variant", default="hgl_qos", choices=["hgl_qos", "hgl"])
    p.add_argument("--seeds", default="42,123,456,789,2024")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--eval-population", default="application",
                   choices=["application", "app_lib", "labeled"])
    p.add_argument("--rank-normalize-features", action="store_true", default=True)
    p.add_argument("--no-rank-normalize-features", dest="rank_normalize_features", action="store_false")
    p.add_argument("--rank-normalize-labels", action="store_true", default=True)
    p.add_argument("--no-rank-normalize-labels", dest="rank_normalize_labels", action="store_false")
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
                          population=args.eval_population,
                          rank_normalize_features=args.rank_normalize_features)
            except Exception as exc:                      # noqa: BLE001
                logger.error("  %s seed %d failed: %s", b.scenario_id, seed, exc,
                             exc_info=True)
                continue
            m["seed"] = seed
            per_system[b.scenario_id].append(m)
            logger.info("    %-32s rho=%.4f  hybrid_rho=%.4f  F1@K=%.4f  n=%d",
                        b.scenario_id, m["spearman_rho"], m.get("hybrid_spearman_rho", 0.0),
                        m["f1_at_k"], m["n"])

    summary: Dict[str, Any] = {}
    for sid, runs in per_system.items():
        if not runs:
            summary[sid] = {"n_seeds": 0}
            continue
        rho = [r["spearman_rho"] for r in runs]
        hybrid_rho = [r.get("hybrid_spearman_rho", 0.0) for r in runs]
        f1 = [r["f1_at_k"] for r in runs]
        hybrid_f1 = [r.get("hybrid_f1_at_k", 0.0) for r in runs]
        pos_rho = [r["spearman_rho_positive"] for r in runs if r.get("spearman_rho_positive") is not None]

        bundle = next(b for b in real if b.scenario_id == sid)
        summary[sid] = {
            "n_seeds": len(runs),
            "n_nodes": bundle.n_nodes,
            "n_evaluated": runs[0].get("n"),
            "mean_rho": float(np.mean(rho)),
            "std_rho": float(np.std(rho)),
            "mean_hybrid_rho": float(np.mean(hybrid_rho)),
            "std_hybrid_rho": float(np.std(hybrid_rho)),
            "mean_f1_at_k": float(np.mean(f1)),
            "std_f1_at_k": float(np.std(f1)),
            "mean_hybrid_f1_at_k": float(np.mean(hybrid_f1)),
            "std_hybrid_f1_at_k": float(np.std(hybrid_f1)),
            "mean_rho_positive": float(np.mean(pos_rho)) if pos_rho else None,
            "std_rho_positive": float(np.std(pos_rho)) if pos_rho else None,
            "n_positive": runs[0].get("n_positive"),
            "label_stability": bundle.label_stability,
            "labeler": bundle.labeler,
        }

    # Training-free references on the same oracle, same population, same labels.
    references: Dict[str, Dict[str, Any]] = {}
    for b in real:
        for name, m in score_references(b, population=args.eval_population).items():
            # A reference that could not be computed records why; it is carried
            # through as-is rather than coerced to a number.
            if "spearman_rho" not in m:
                references.setdefault(name, {})[b.scenario_id] = dict(m)
                continue
            references.setdefault(name, {})[b.scenario_id] = {
                "rho": float(m["spearman_rho"]),
                "f1_at_k": float(m["f1_at_k"]),
            }
    ref_means = {}
    for name, per in references.items():
        rhos = [v["rho"] for v in per.values() if "rho" in v]
        ref_means[name] = float(np.mean(rhos)) if len(rhos) == len(per) else None

    scored = [s for s in summary.values() if s.get("n_seeds")]
    mean_rho_all = float(np.mean([s["mean_rho"] for s in scored])) if scored else None
    mean_hybrid_rho_all = float(np.mean([s["mean_hybrid_rho"] for s in scored])) if scored else None

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
        "references": references,
        "reference_mean_rho": ref_means,
        "reference_note": (
            "Training-free references scored against the same I*(v) labels, "
            "population and node set as the learned model. Topo-QoS is absent: "
            "it needs QoS-weighted betweenness on the projection graph, which "
            "this cache does not carry."
        ),
        "mean_rho_across_systems": mean_rho_all,
        "mean_hybrid_rho_across_systems": mean_hybrid_rho_all,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n  {payload['label']} zero-shot on real systems "
          f"(oracle: {payload['oracle']})")
    print("  " + "─" * 84)
    print(f"  {'system':<32}{'|V|':>5}{'GNN ρ':>9}{'sd':>7}{'Hybrid ρ':>10}{'sd':>7}{'F1@K':>8}")
    for sid in sorted(summary):
        s = summary[sid]
        if not s.get("n_seeds"):
            print(f"  {sid:<32}{'—':>5}{'failed':>9}")
            continue
        print(f"  {sid:<32}{s['n_nodes']:>5}{s['mean_rho']:>9.4f}"
              f"{s['std_rho']:>7.4f}{s['mean_hybrid_rho']:>10.4f}"
              f"{s['std_hybrid_rho']:>7.4f}{s['mean_f1_at_k']:>8.4f}")
    if mean_rho_all is not None:
        print(f"  {'mean across systems':<32}{'':>5}"
              f"{mean_rho_all:>9.4f}{'':>7}{mean_hybrid_rho_all:>10.4f}")

    if references:
        print(f"\n  Comparison Across Models (Application Stratum, Oracle: I*(v))")
        print("  " + "─" * 84)
        hdr = f"  {'system':<32}"
        for name in sorted(references):
            hdr += f"{name:>12}"
        hdr += f"{'HGT-QoS':>12}{'SaG-Hybrid':>12}"
        print(hdr)
        for sid in sorted(summary):
            row = f"  {sid:<32}"
            for name in sorted(references):
                v = references[name].get(sid)
                row += f"{v['rho']:>12.4f}" if v else f"{'—':>12}"
            s_ = summary[sid]
            if s_.get("n_seeds"):
                row += f"{s_['mean_rho']:>12.4f}{s_['mean_hybrid_rho']:>12.4f}"
            else:
                row += f"{'—':>12}{'—':>12}"
            print(row)
        row = f"  {'mean':<32}"
        for name in sorted(references):
            row += f"{ref_means[name]:>12.4f}"
        row += f"{mean_rho_all:>12.4f}{mean_hybrid_rho_all:>12.4f}"
        print(row)
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
