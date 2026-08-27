#!/usr/bin/env python3
"""
reproduce/ahp_sensitivity.py — AHP shrinkage (lambda) sensitivity sweep
======================================================================

Produces ``results/ahp_shrinkage_sweep.json``, the artifact that
``docs/structural-analysis.md`` §11.2 and §11.6 cite for the claim that Spearman
rho plateaus over lambda in [0.65, 0.75] and that the lambda = 0.70 default is
therefore not a tuned coincidence. That artifact did not previously exist —
``docs/internal/`` was empty — so the sensitivity claim rested on nothing
committed.

What the sweep varies
---------------------
lambda blends the stated AHP judgement toward a uniform prior:

    w_final(d) = lambda * w_AHP(d) + (1 - lambda) * (1 / n)

lambda = 1.0 is the raw judgement; lambda = 0.0 is equal weights, i.e. the
``--equal-weights`` baseline. The sweep therefore also answers the sharper
question a reviewer will ask: *does the AHP judgement buy anything at all over
weighting the four dimensions equally?*

Two things worth knowing when reading the output
------------------------------------------------
1. Spearman rho is rank-based, so it responds only to how the weights reorder
   components — not to the scale of the composite. A flat curve means the
   ranking is insensitive to the weighting, which is a robustness result and
   simultaneously an argument that the exact AHP numbers are not load-bearing.
2. ``AHPProcessor`` applies shrinkage to the **intra-dimension** vectors
   as well as the composite, so lambda moves every RM formula at once. The
   sweep reports the resulting weights alongside rho so the two can be read
   together.

Usage
-----
    PYTHONPATH=. python reproduce/ahp_sensitivity.py
    PYTHONPATH=. python reproduce/ahp_sensitivity.py --lambdas 0.0 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr

from saag.evaluation.metrics import resolve_eval_keys

logger = logging.getLogger("ahp_sensitivity")

DEFAULT_LAMBDAS = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
RESULTS_DIR = Path("results")


_STRUCTURAL_CACHE: Dict[Any, Any] = {}


def full_pipeline_structural(topology: Dict[str, Any], layer: str = "system"):
    """StructuralAnalysisResult for *topology*, over the whole typed graph.

    Memoised per (topology identity, layer, topic-weight constants). The RM
    sweeps vary only the *scoring* weights, which cannot change the structural
    analysis, so recomputing it per grid point is pure waste — the threshold
    sweep in particular recomputed an identical result once per threshold. The
    topic-weight triple is part of the key because ``reproduce/topic_weight_
    sensitivity.py`` *does* move it, and it propagates into every derived edge
    weight and therefore into the structural analysis itself; keying on the
    topology alone would silently serve that sweep one graph for every grid point.
    """
    from saag.core import models as _models

    key = (
        id(topology), layer,
        _models.TOPIC_QOS_WEIGHT_BETA,
        _models.TOPIC_SIZE_WEIGHT_ALPHA,
        _models.TOPIC_FREQ_WEIGHT_PSI,
    )
    if key not in _STRUCTURAL_CACHE:
        from saag.analysis.service import AnalysisService
        from saag.infrastructure.memory_repo import MemoryRepository

        repo = MemoryRepository()
        repo.save_graph(topology, clear=True)
        _STRUCTURAL_CACHE[key] = AnalysisService(repo).analyze_layer(layer).structural
    return _STRUCTURAL_CACHE[key]


def score_with_analyzer(
    topology: Dict[str, Any], analyzer: Any, layer: str = "system"
) -> Dict[str, float]:
    """Composite Q(v) for every component, scored by *analyzer* over the full graph.

    The shared entry point for every RM parameter sweep in ``reproduce/``. Using
    it rather than ``reproduce.main_table._compute_rm_from_structural`` matters:
    that helper scores features restricted to the Application+Library
    ``DEPENDS_ON`` projection, on which no Application is an articulation point
    and no incident edge is a bridge, so four of Availability's five terms vanish
    and A(v) — carrying w_R*(1-r_alpha) ~ 0.51 of the composite — is constant
    across Applications in six of the eight cached scenarios. A sweep run against
    that variant characterises a degenerate score rather than Q(v).
    """
    result = analyzer.analyze(full_pipeline_structural(topology, layer))
    return {
        str(getattr(c, "id", getattr(c, "component_id", ""))): float(c.scores.overall)
        for c in result.components
    }


def _score_components(topology: Dict[str, Any], lam: float, layer: str = "system") -> Dict[str, float]:
    """Composite Q(v) for every component under shrinkage factor *lam*.

    Runs the **full analysis pipeline** — MemoryRepository import, DEPENDS_ON
    derivation, StructuralAnalyzer over the whole typed graph, then RM scoring at
    *lam*. This is the RM baseline the manuscript defines in its Composite Quality
    Score section, and the same path that produced the cached
    ``quality_scores.json`` the LOSO harness consumes, so the sweep characterises
    the object it claims to.

    It deliberately does **not** use ``_compute_rm_from_structural``: that helper
    recomputes RM from features restricted to the Application+Library
    ``DEPENDS_ON`` projection, which exists to give the GNN a feature/label-aligned
    substrate. On that projection no Application is an articulation point and no
    incident edge is a bridge, so four of Availability's five terms vanish and
    A(v) collapses to 0.05*w(v) — a constant across Applications in six of the
    eight cached scenarios. Availability carries w_R*(1-r_alpha) ~ 0.51 of the
    composite, so sweeping lambda over that variant measures a degenerate score
    rather than Q(v). See :func:`_score_components_projection`.
    """
    from saag.analysis.analyzer import QualityAnalyzer

    return score_with_analyzer(
        topology, QualityAnalyzer(use_ahp=True, ahp_shrinkage=lam), layer)


def _score_components_projection(
    topology: Dict[str, Any], structural: Dict[str, Any], lam: float
) -> Dict[str, float]:
    """Q(v) recomputed on the Application+Library DEPENDS_ON projection.

    Reported as a secondary series only. See :func:`_score_components` for why
    this variant's Availability channel is degenerate; it is retained so the
    difference between the two substrates stays visible in the artifact instead
    of being silently dropped.
    """
    from reproduce.main_table import _compute_rm_from_structural
    from saag.analysis.analyzer import QualityAnalyzer

    scored = _compute_rm_from_structural(
        topology, structural,
        analyzer=QualityAnalyzer(use_ahp=True, ahp_shrinkage=lam),
    )
    return {nid: float(v.get("composite", 0.0)) for nid, v in scored.items()}


def _weights_snapshot(lam: float) -> Dict[str, Any]:
    """The weights actually shipped at this lambda, for both dimension levels.

    Recorded because the docs print pre-shrinkage AHP weights while the code
    ships post-shrinkage ones — shrinkage is applied to the intra-dimension
    vectors too, not only the composite.

    ``composite`` (q_reliability, q_maintainability) is NOT AHP-derived — it is
    a DECLARED constant (see ``QualityWeights`` docstring) and is therefore
    identical at every lambda. It is still recorded here so that invariance is
    visible in the artifact rather than merely asserted.
    """
    from saag.analysis.weight_calculator import AHPProcessor

    w = AHPProcessor(shrinkage_factor=lam).compute_weights()
    return {
        "composite": {
            "reliability": round(w.q_reliability, 4),
            "maintainability": round(w.q_maintainability, 4),
        },
        "fault_tolerance_terms": {
            "reverse_pagerank": round(w.ft_reverse_pagerank, 4),
            "in_degree": round(w.ft_in_degree, 4),
            "cdpot": round(w.ft_cdpot, 4),
        },
    }


def _load_topology(scenario: str) -> Dict[str, Any]:
    """Cache topology if present, else the scenario definition — same order as main_table."""
    from reproduce.main_table import _find_cache_dir, SCENARIOS_DIR

    cache_topo = _find_cache_dir(scenario) / "topology.json"
    path = cache_topo if cache_topo.exists() else SCENARIOS_DIR / f"{scenario}.json"
    return json.loads(path.read_text())


def run_sweep(scenarios: List[str], lambdas: List[float]) -> Dict[str, Any]:
    from reproduce.main_table import _load_scenario_data

    # Load features once; only the weighting changes across lambdas.
    loaded: Dict[str, Any] = {}
    for scenario in scenarios:
        try:
            topology = _load_topology(scenario)
            g, structural, simulation, _r, _gt = _load_scenario_data(scenario, substrate="projection")
            loaded[scenario] = (topology, structural, simulation, g)
        except Exception as exc:      # noqa: BLE001 - one bad scenario must not kill the sweep
            logger.warning("%s failed to load: %s", scenario, exc)

    def _rho(pred: Dict[str, float], truth: Dict[str, float], keys: List[str]):
        """Spearman rho on an explicit key set, or None if it is degenerate."""
        if len(keys) < 3:
            return None
        y_pred = np.array([pred[k] for k in keys])
        y_true = np.array([truth[k] for k in keys])
        if np.ptp(y_pred) == 0 or np.ptp(y_true) == 0:
            return None
        rho, _ = spearmanr(y_pred, y_true)
        return None if np.isnan(rho) else float(rho)

    rows: List[Dict[str, Any]] = []
    for lam in lambdas:
        per_scenario: Dict[str, float] = {}
        per_scenario_pooled: Dict[str, float] = {}
        per_scenario_projection: Dict[str, float] = {}
        for scenario, (topology, structural, simulation, graph) in loaded.items():
            try:
                pred = _score_components(topology, lam)
            except Exception as exc:      # noqa: BLE001
                logger.warning("%s @ lambda=%.2f failed: %s", scenario, lam, exc)
                continue
            try:
                pred_proj = _score_components_projection(topology, structural, lam)
            except Exception as exc:      # noqa: BLE001
                logger.warning("%s @ lambda=%.2f projection variant failed: %s",
                               scenario, lam, exc)
                pred_proj = {}

            truth = {k: float(v.get("composite", 0.0)) for k, v in simulation.items()}
            # Primary figure is scored on the Application population, matching
            # reproduce/main_table.py and cli/loso_evaluate.py. Pooling every node
            # type mixes populations with different scales and base rates, which
            # pushes this sweep's rho outside the range spanned by its own per-type
            # values (a Simpson's-paradox effect). The pooled figure is retained as
            # a secondary column so that effect stays visible rather than being
            # silently corrected away.
            app_keys = resolve_eval_keys(pred, truth, graph, population="application")
            pooled_keys = sorted(set(pred) & set(truth))

            rho_app = _rho(pred, truth, app_keys)
            if rho_app is not None:
                per_scenario[scenario] = rho_app
            rho_pooled = _rho(pred, truth, pooled_keys)
            if rho_pooled is not None:
                per_scenario_pooled[scenario] = rho_pooled

            if pred_proj:
                proj_keys = resolve_eval_keys(pred_proj, truth, graph, population="application")
                rho_proj = _rho(pred_proj, truth, proj_keys)
                if rho_proj is not None:
                    per_scenario_projection[scenario] = rho_proj

        rows.append({
            "lambda": lam,
            "eval_population": "application",
            "mean_rho": float(np.mean(list(per_scenario.values()))) if per_scenario else None,
            "std_rho": float(np.std(list(per_scenario.values()))) if per_scenario else None,
            "per_scenario_rho": {k: round(v, 4) for k, v in sorted(per_scenario.items())},
            "n_scenarios": len(per_scenario),
            "pooled_mean_rho": (
                float(np.mean(list(per_scenario_pooled.values())))
                if per_scenario_pooled else None),
            "pooled_per_scenario_rho": {
                k: round(v, 4) for k, v in sorted(per_scenario_pooled.items())},
            "projection_substrate_mean_rho": (
                float(np.mean(list(per_scenario_projection.values())))
                if per_scenario_projection else None),
            "projection_substrate_per_scenario_rho": {
                k: round(v, 4) for k, v in sorted(per_scenario_projection.items())},
            "weights": _weights_snapshot(lam),
        })
        mean = rows[-1]["mean_rho"]
        pooled = rows[-1]["pooled_mean_rho"]
        proj = rows[-1]["projection_substrate_mean_rho"]
        print(f"  lambda={lam:.2f}  mean_rho(app)={mean if mean is None else round(mean, 4)}  "
              f"pooled={pooled if pooled is None else round(pooled, 4)}  "
              f"projection={proj if proj is None else round(proj, 4)}  "
              f"({rows[-1]['n_scenarios']} scenarios)")

    defined = [r for r in rows if r["mean_rho"] is not None]
    best = max(defined, key=lambda r: r["mean_rho"]) if defined else None
    equal = next((r for r in defined if abs(r["lambda"]) < 1e-9), None)
    default = next((r for r in defined if abs(r["lambda"] - 0.7) < 1e-9), None)

    interpretation = {}
    if best and equal and default:
        rhos = [r["mean_rho"] for r in defined]
        spread = max(rhos) - min(rhos)
        # Is the curve monotone in lambda? If it is, there is no plateau to
        # appeal to, and the direction of that monotonicity decides whether the
        # AHP judgement helps or hurts.
        ordered = [r["mean_rho"] for r in sorted(defined, key=lambda r: r["lambda"])]
        monotone_decreasing = all(b <= a + 1e-9 for a, b in zip(ordered, ordered[1:]))
        monotone_increasing = all(b >= a - 1e-9 for a, b in zip(ordered, ordered[1:]))

        interpretation = {
            "best_lambda": best["lambda"],
            "best_mean_rho": round(best["mean_rho"], 4),
            "default_lambda_mean_rho": round(default["mean_rho"], 4),
            "uniform_intra_dim_mean_rho": round(equal["mean_rho"], 4),
            "ahp_lift_over_uniform_intra_dim": round(default["mean_rho"] - equal["mean_rho"], 4),
            "rho_spread_across_lambda": round(spread, 4),
            "monotone_decreasing_in_lambda": monotone_decreasing,
            "monotone_increasing_in_lambda": monotone_increasing,
            "plateau_in_065_075": (
                max(r["mean_rho"] for r in defined if 0.65 <= r["lambda"] <= 0.75)
                - min(r["mean_rho"] for r in defined if 0.65 <= r["lambda"] <= 0.75)
            ) < 0.01 if any(0.65 <= r["lambda"] <= 0.75 for r in defined) else None,
            "composite_weights_lambda_invariant": True,
            "note": (
                "Spearman rho is rank-based, so this measures only how the weighting "
                "reorders components. Read `monotone_decreasing_in_lambda` before "
                "quoting any plateau: if the curve is monotone there is no plateau to "
                "appeal to, and a negative `ahp_lift_over_uniform_intra_dim` means the "
                "stated AHP judgement ranks components *worse* than uniform intra-dimension "
                "weights on this cohort. Unlike the pre-RM-migration sweep, lambda=0 no "
                "longer means 'equal weights over four composite dimensions' — the "
                "composite (q_reliability=0.80, q_maintainability=0.20) is a DECLARED "
                "constant, not AHP-derived, so it is identical at every lambda "
                "(`composite_weights_lambda_invariant`). Lambda now only shrinks the "
                "intra-dimension vectors (fault-tolerance, maintainability, availability, "
                "impact) toward their own uniform priors. `mean_rho` is scored on the "
                "Application population via the full analysis pipeline — the RM baseline "
                "the manuscript defines. `pooled_mean_rho` pools every node type and is "
                "subject to the same Simpson's-paradox hazard the detection artifact "
                "records. `projection_substrate_mean_rho` recomputes RM from features "
                "restricted to the Application+Library DEPENDS_ON projection, where no "
                "Application is an articulation point and no incident edge is a bridge, "
                "so four of Availability's five terms vanish and A(v) — ~0.51 of the "
                "composite — is constant across Applications in six of eight scenarios. "
                "That series characterises a degenerate score, not Q(v), and must not be "
                "quoted as the RM baseline's sensitivity."
            ),
        }

    return {
        "lambdas": lambdas,
        "scenarios": scenarios,
        "rows": rows,
        "interpretation": interpretation,
    }


def parse_args():
    p = argparse.ArgumentParser(description="AHP shrinkage sensitivity sweep")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "ahp_shrinkage_sweep.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    from reproduce.main_table import ALL_SCENARIOS

    scenarios = args.scenarios or ALL_SCENARIOS
    print(f"AHP shrinkage sweep: {len(scenarios)} scenarios x {len(args.lambdas)} lambdas")
    report = run_sweep(scenarios, args.lambdas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    if report["interpretation"]:
        for k, v in report["interpretation"].items():
            if k != "note":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
