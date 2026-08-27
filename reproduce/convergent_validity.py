#!/usr/bin/env python3
"""
reproduce/convergent_validity.py — how much do the oracles agree?
=================================================================

Produces ``results/convergent_validity.json``.

The project has **three** simulation oracles and they measure different things:

``I*(v)``      ``FaultInjector`` — mean subscriber feed-loss fraction. Supplies
               GNN training labels and backs Tables 3/4.
``I_comp(v)``  ``FailureSimulator`` — 0.35*reachability + 0.25*fragmentation
               + 0.25*throughput + 0.15*flow_disruption. Backs the Validate-stage
               gates and the four-dimensional IR/IM/IA/IV decomposition.
``I_dyn(v)``   ``MessageFlowSimulator`` — the drop in delivered message rate that
               *surviving* consumers experience when v fails, measured by
               discrete-event simulation of actual traffic.

``docs/validation.md`` states the first two "correlate only loosely" but never
quantifies it, while the paper draws on both — the main table from I*, the
stratified and library analyses from I_comp. If a claim rests on one oracle and
its supporting argument on the other, the strength of that link is a number the
paper owes the reader. This script measures it.

Why the third oracle matters
----------------------------
I* and I_comp are both topological cascade engines over the same substrate, so
their agreement cannot rule out a shared construction artifact — the objection
that a topology-derived Q(v) is being validated against topology-derived labels.
I_dyn is behavioural: it observes message delivery under load rather than
reachability over edges. Agreement between I_dyn and I* is therefore evidence of
a different kind from agreement between I* and I_comp.

I_dyn is a **delivery-based, QoS-agnostic** measure in this study; see the
limitations in ``docs/failure-simulation.md`` for what it does not capture.

Interpretation
--------------
High agreement is a convergent-validity argument: differently-constructed
simulators rank components alike, so neither is likely to be an artifact of its
own construction. Low agreement bounds what either can support on its own, and
means results must name their oracle.

Usage
-----
    PYTHONPATH=. python reproduce/convergent_validity.py
    PYTHONPATH=. python reproduce/convergent_validity.py --scenarios atm_system
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import kendalltau, spearmanr

from saag.evaluation.metrics import resolve_eval_keys

logger = logging.getLogger("convergent_validity")

RESULTS_DIR = Path("results")


def _fault_injector_labels(
    scenario: str, seeds: List[int], qos: bool = True
) -> Dict[str, float]:
    """I*(v) — the labels that back the published tables."""
    from cli.loso_evaluate import _build_graph_from_json
    from saag.simulation.fault_injector import FaultInjector
    from reproduce.ahp_sensitivity import _load_topology

    graph = _build_graph_from_json(_load_topology(scenario))
    injector = FaultInjector(
        graph=graph,
        seeds=seeds,
        cascade_depth_limit=0,
        qos_factor_mode="ladder" if qos else "none",
    )
    result = injector.run(node_types=["Application", "Broker", "Library"])
    return {nid: float(rec.impact_score) for nid, rec in result.records.items()}


def _failure_simulator_labels(
    scenario: str, layer: str = "system", qos: bool = True
) -> Dict[str, float]:
    """I_comp(v) — the composite that backs the validation gates."""
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.simulation.failure_simulator import FailureSimulator
    from saag.simulation.graph import SimulationGraph
    from reproduce.ahp_sensitivity import _load_topology

    repo = MemoryRepository()
    repo.save_graph(_load_topology(scenario), clear=True)
    sim = FailureSimulator(
        SimulationGraph(repo.get_graph_data(include_raw=True)),
        qos_weighting=qos,
    )
    return {
        r.target_id: float(r.impact.composite_impact)
        for r in sim.simulate_exhaustive(layer=layer, seed=42)
    }


def _message_flow_labels(
    scenario: str,
    duration: float = 60.0,
    seed: int = 42,
    max_candidates: Optional[int] = None,
) -> Dict[str, float]:
    """I_dyn(v) — the delivery-rate loss surviving consumers actually suffer.

    One discrete-event run per candidate, so this is the expensive oracle. Only
    components that carry pub/sub traffic are scored: the engine cannot observe
    Brokers (ROUTES) or Nodes (RUNS_ON), and a component it cannot observe is
    omitted rather than recorded as impact 0.0.
    """
    from cli.loso_evaluate import _build_graph_from_json
    from saag.simulation.message_flow_simulator import MessageFlowSimulator
    from reproduce.ahp_sensitivity import _load_topology

    graph = _build_graph_from_json(_load_topology(scenario))

    probe = MessageFlowSimulator(graph=graph, duration=duration, seed=seed).run()
    candidates = list(probe.labeled_node_ids)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    labels: Dict[str, float] = {}
    for node in candidates:
        result = MessageFlowSimulator(
            graph=graph,
            duration=duration,
            fault_node=node,
            fault_time=duration / 2.0,
            seed=seed,
        ).run()
        event = result.fault_event
        if event is None:
            continue
        labels[node] = float(event.delivery_rate_before - event.delivery_rate_after)
    return labels


def _restrict(
    oracles: Dict[str, Dict[str, float]], scenario: str, population: str
) -> Dict[str, Dict[str, float]]:
    """Cut every oracle down to *population* before any pair is compared.

    Comparing oracles across pooled node types repeats the aggregation hazard the
    headline tables already guard against: a Broker's impact and an Application's
    impact are not on a common scale, so a pooled rank agreement can be carried
    (or destroyed) by the type mix rather than by the oracles. ``"labeled"``
    reproduces the pooled behaviour this script had before the flag existed.
    """
    if population == "labeled":
        return oracles
    from cli.loso_evaluate import _build_graph_from_json
    from reproduce.ahp_sensitivity import _load_topology

    graph = _build_graph_from_json(_load_topology(scenario))
    out: Dict[str, Dict[str, float]] = {}
    for name, scores in oracles.items():
        keys = resolve_eval_keys(scores, scores, graph, population=population)
        out[name] = {k: scores[k] for k in keys}
    return out


def _tie_robust_top(values: np.ndarray, k: int) -> set:
    """Indices of the top *k* values, plus every index tied with the k-th.

    ``np.argsort`` resolves ties by position, which is arbitrary; when several
    components share the k-th score, an index-ordered cut credits one oracle and
    penalises the other for what is actually the same measurement.
    """
    order = np.argsort(-values)
    cutoff = values[order[k - 1]]
    return set(np.flatnonzero(values >= cutoff).tolist())


def _pairwise(a_scores: Dict[str, float], b_scores: Dict[str, float]) -> Dict[str, Any]:
    """Rank agreement between one pair of oracles on the nodes they share."""
    common = sorted(set(a_scores) & set(b_scores))
    block: Dict[str, Any] = {"n_common": len(common)}
    if len(common) < 3:
        block["note"] = "insufficient overlap"
        return block

    a = np.array([a_scores[k] for k in common])
    b = np.array([b_scores[k] for k in common])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        block["note"] = "constant oracle on the shared node set"
        return block

    rho, p = spearmanr(a, b)
    tau, _ = kendalltau(a, b)

    # Do they agree on *who is critical*, not just on the ordering? Top-K set
    # overlap is the property a practitioner actually consumes.
    k = max(1, int(round(len(common) * 0.20)))
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    jaccard = len(top_a & top_b) / len(top_a | top_b)

    # ``np.argsort`` breaks ties by array index, so a plateau at the K-th value
    # is cut at an arbitrary point and the strict figure understates agreement by
    # however much of that plateau each oracle happened to keep. The tie-robust
    # set admits every element tied with the K-th, which is the honest reading of
    # "the top-K critical set" when the scores are not all distinct.
    rob_a = _tie_robust_top(a, k)
    rob_b = _tie_robust_top(b, k)
    jaccard_robust = len(rob_a & rob_b) / len(rob_a | rob_b)

    # What Jaccard would two *independent* rankings produce at this K? Without
    # this, a reader has no scale on which to judge 0.3 — the statistic's floor
    # is not 0 and its practical ceiling is not 1.
    n = len(common)
    exp_inter = (k * k) / n
    exp_union = 2 * k - exp_inter
    random_jaccard = exp_inter / exp_union if exp_union > 0 else float("nan")

    block.update({
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 6),
        "kendall_tau": round(float(tau), 4),
        "topk_jaccard": round(float(jaccard), 4),
        "topk_jaccard_tie_robust": round(float(jaccard_robust), 4),
        "topk_jaccard_random_baseline": round(float(random_jaccard), 4),
        "n_ties_at_cut_a": len(rob_a) - k,
        "n_ties_at_cut_b": len(rob_b) - k,
        "k": k,
        "scale_max_a": round(float(a.max()), 4),
        "scale_max_b": round(float(b.max()), 4),
    })
    return block


def compare(
    scenario: str,
    seeds: List[int],
    qos: bool = True,
    duration: float = 60.0,
    max_candidates: Optional[int] = None,
    skip_message_flow: bool = False,
    population: str = "application",
) -> Dict[str, Any]:
    """Pairwise agreement across the available oracles.

    ``qos=False`` reproduces the pre-QoS behaviour of the two topological
    engines, so the convergence claim — that weighting both oracles by the same
    w(t) should make them agree more — is testable rather than asserted. It does
    not apply to I_dyn, which is QoS-agnostic in this study.
    """
    oracles: Dict[str, Dict[str, float]] = {
        "i_star": _fault_injector_labels(scenario, seeds, qos=qos),
        "i_comp": _failure_simulator_labels(scenario, qos=qos),
    }
    if not skip_message_flow:
        oracles["i_dyn"] = _message_flow_labels(
            scenario, duration=duration, max_candidates=max_candidates)

    n_before = {name: len(scores) for name, scores in oracles.items()}
    oracles = _restrict(oracles, scenario, population)

    row: Dict[str, Any] = {
        "scenario": scenario,
        "eval_population": population,
        "n_per_oracle": {name: len(scores) for name, scores in oracles.items()},
        "n_per_oracle_unrestricted": n_before,
        "pairs": {},
    }
    for name_a, name_b in combinations(sorted(oracles), 2):
        row["pairs"][f"{name_a}__{name_b}"] = _pairwise(
            oracles[name_a], oracles[name_b])
    return row


def parse_args():
    p = argparse.ArgumentParser(description="Inter-oracle convergent validity")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "convergent_validity.json")
    p.add_argument(
        "--no-qos", action="store_true",
        help="Disable QoS weighting in both oracles, reproducing their pre-QoS "
             "behaviour. Run with and without to measure whether weighting both "
             "by the same w(t) actually makes them converge.",
    )
    p.add_argument(
        "--duration", type=float, default=60.0,
        help="Simulated seconds per message-flow run (default: 60).",
    )
    p.add_argument(
        "--max-candidates", type=int, default=None,
        help="Cap the number of components I_dyn scores. The message-flow oracle "
             "costs one discrete-event run per candidate, so large scenarios are "
             "dominated by it.",
    )
    p.add_argument(
        "--skip-message-flow", action="store_true",
        help="Compare only the two topological oracles, as this script did "
             "before I_dyn was added.",
    )
    p.add_argument(
        "--eval-population", default="application",
        choices=["application", "app_lib", "labeled"],
        help="Node population the oracles are compared on. 'application' "
             "(default) matches the population every headline table is scored "
             "on; 'labeled' pools all node types, which is what this script did "
             "before the flag existed.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    from reproduce.main_table import ALL_SCENARIOS

    scenarios = args.scenarios or ALL_SCENARIOS
    engines = "I*(v) [FaultInjector], I_comp(v) [FailureSimulator]"
    if not args.skip_message_flow:
        engines += ", I_dyn(v) [MessageFlowSimulator]"
    print(f"Convergent validity: {engines}")
    print(f"  {len(scenarios)} scenarios\n")

    rows = []
    for scenario in scenarios:
        try:
            row = compare(
                scenario, args.seeds, qos=not args.no_qos,
                duration=args.duration, max_candidates=args.max_candidates,
                skip_message_flow=args.skip_message_flow,
                population=args.eval_population,
            )
        except Exception as exc:      # noqa: BLE001
            logger.warning("%s failed: %s", scenario, exc)
            rows.append({"scenario": scenario, "error": str(exc)})
            continue
        rows.append(row)
        print(f"  {scenario}")
        for pair, block in row["pairs"].items():
            rho = block.get("spearman_rho")
            print(f"    {pair:22} rho={rho if rho is not None else '—':>8}  "
                  f"tau={block.get('kendall_tau', '—'):>8}  "
                  f"topK-Jaccard={block.get('topk_jaccard', '—'):>7}  "
                  f"tie-robust={block.get('topk_jaccard_tie_robust', '—'):>7}  "
                  f"n={block.get('n_common', 0)}")

    # Summarise each pair across scenarios; a pair is only as strong as the
    # scenarios it was actually measurable on.
    summary: Dict[str, Any] = {}
    for pair in sorted({p for r in rows for p in r.get("pairs", {})}):
        blocks = [r["pairs"][pair] for r in rows if pair in r.get("pairs", {})]
        def _vals(key):
            return [b[key] for b in blocks if isinstance(b.get(key), float)]

        rhos = _vals("spearman_rho")
        jac = _vals("topk_jaccard")
        jac_rob = _vals("topk_jaccard_tie_robust")
        jac_rand = _vals("topk_jaccard_random_baseline")
        summary[pair] = {
            "mean_spearman_rho": round(float(np.mean(rhos)), 4) if rhos else None,
            "min_spearman_rho": round(float(np.min(rhos)), 4) if rhos else None,
            "mean_topk_jaccard": round(float(np.mean(jac)), 4) if jac else None,
            "mean_topk_jaccard_tie_robust": (
                round(float(np.mean(jac_rob)), 4) if jac_rob else None),
            "mean_topk_jaccard_random_baseline": (
                round(float(np.mean(jac_rand)), 4) if jac_rand else None),
            "n_scenarios_measured": len(rhos),
        }

    # The ceiling any cross-oracle figure should be read against: how well does
    # a single oracle agree with *itself* across seeds? Without it, a reader
    # cannot tell construct divergence from measurement noise — and on this
    # corpus the self-agreement is far higher than the cross-oracle agreement,
    # so the gap is not noise.
    ceiling: Dict[str, Any] = {}
    stability_path = RESULTS_DIR / "label_stability.json"
    if stability_path.exists():
        try:
            stability = json.loads(stability_path.read_text()).get("summary", {})
            ceiling = {
                "source": str(stability_path),
                "test_retest_spearman_range": stability.get("test_retest_spearman_range"),
                "topk_jaccard_range": stability.get("topk_jaccard_range"),
                "note": (
                    "I*(v)'s own seed-to-seed agreement, from "
                    "reproduce/label_stability_check.py. Cross-oracle agreement "
                    "below this range is construct divergence, not label noise."
                ),
            }
        except Exception as exc:      # noqa: BLE001
            logger.warning("could not read %s: %s", stability_path, exc)

    report = {
        "seeds": args.seeds,
        "duration": args.duration,
        "eval_population": args.eval_population,
        "per_scenario": rows,
        "summary": summary,
        "self_agreement_ceiling": ceiling,
        "note": (
            "These are different quantities on different scales; only their rank "
            "agreement is meaningful. Read this before attributing a result measured "
            "against one oracle to evidence gathered against another — they are used "
            "in different sections of the paper. i_star and i_comp are both "
            "topological cascade engines over the same substrate, so their agreement "
            "cannot rule out a shared construction artifact; i_dyn is behavioural and "
            "carries independent evidence, but is delivery-based and QoS-agnostic."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    for pair, stats in summary.items():
        print(f"  {pair}: mean_rho={stats['mean_spearman_rho']} "
              f"min_rho={stats['min_spearman_rho']} "
              f"mean_jaccard={stats['mean_topk_jaccard']} "
              f"tie_robust={stats['mean_topk_jaccard_tie_robust']} "
              f"random={stats['mean_topk_jaccard_random_baseline']} "
              f"n={stats['n_scenarios_measured']}")


if __name__ == "__main__":
    main()
