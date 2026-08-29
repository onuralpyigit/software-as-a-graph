#!/usr/bin/env python3
"""
reproduce/topic_weight_sensitivity.py — sensitivity of results to (beta, alpha, psi)
===================================================================================

Produces ``results/topic_weight_sensitivity.json``.

What this answers
-----------------
The topic weight

    w(t) = beta * QoS(t) + alpha * SizeNorm(t) + psi * FreqNorm(t)

ships at ``(beta, alpha, psi) = (0.75, 0.15, 0.10)``
(``saag/core/models.py``). Unlike the inner QoS split
``(0.30, 0.40, 0.30)`` — which is AHP-derived, consistency-audited, and pinned by
``tests/test_ahp_shrinkage.py`` — the outer split is **DECLARED**: nobody elicited
it and nothing validated it. A reviewer is entitled to ask what rests on it.

This sweep answers that at three levels, cheapest first, because a claim about a
constant is only as good as how far downstream it is carried:

1. ``w_rank_rho`` — Spearman rho of the induced w(t) ordering against the shipped
   ordering, over every topic in the corpus. Pure algebra on the topic table; no
   pipeline involved.
2. ``topo_qos_rho`` — rho of the Topo-QoS structural baseline against I*(v),
   scored on the Application population. w(t) reaches this through the derived
   DEPENDS_ON edge weights, so it is the first place the constant can actually
   move a published number.
3. ``rm_rho`` — rho of the full-pipeline RM composite Q(v) against I*(v), same
   population. w(t) reaches this through both the derived edge weights and
   Availability's ``0.05 * w(v)`` term.

Levels 2 and 3 both re-derive the graph from a topology whose topic weights were
recomputed under the swept triple, so nothing is held fixed that the constant
would in reality move. The GNN variants are deliberately **not** retrained: the
argument is carried by the closed-form scorers, and a retrain per grid point
would cost more than it could add given the spread the baselines already show.

Grid
----
Includes the shipped triple, the two-term ``(0.85, 0.15, 0.00)`` form that
appeared in an earlier draft of the manuscript, the QoS-only corner
``(1, 0, 0)``, and a uniform prior ``(1/3, 1/3, 1/3)``. A flat curve across that
range is a robustness result and simultaneously an argument that the exact
triple is not load-bearing; a steep one would mean the constant needs elicitation
rather than declaration.

Usage
-----
    PYTHONPATH=. python reproduce/topic_weight_sensitivity.py
    PYTHONPATH=. python reproduce/topic_weight_sensitivity.py --scenarios av_system
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr

from saag.evaluation.metrics import resolve_eval_keys

logger = logging.getLogger("topic_weight_sensitivity")

RESULTS_DIR = Path("results")

#: (beta, alpha, psi). The shipped triple is listed first so it is unambiguous
#: which row is the reference the others are compared against.
DEFAULT_GRID: List[Tuple[float, float, float]] = [
    (0.75, 0.15, 0.10),   # shipped
    (0.90, 0.05, 0.05),
    (0.85, 0.15, 0.00),   # the two-term form printed in an earlier draft
    (1.00, 0.00, 0.00),   # QoS only
    (0.60, 0.20, 0.20),
    (0.50, 0.25, 0.25),
    (1 / 3, 1 / 3, 1 / 3),  # uniform prior
]


# ── Weight patching ───────────────────────────────────────────────────────────

class _TopicWeights:
    """Temporarily install a (beta, alpha, psi) triple process-wide.

    ``compute_topic_weight`` reads the three constants off its own module at call
    time, and every path that materialises a topic weight — ``MemoryRepository``
    on import, ``topic_weight_from_node_attrs`` in the DEPENDS_ON derivation --
    goes through it. Patching the module attributes is therefore enough to move
    the whole pipeline, and is why this sweep does not need a parallel
    implementation of the formula that could drift from the shipped one.
    """

    _NAMES = (
        "TOPIC_QOS_WEIGHT_BETA",
        "TOPIC_SIZE_WEIGHT_ALPHA",
        "TOPIC_FREQ_WEIGHT_PSI",
    )

    def __init__(self, triple: Tuple[float, float, float]) -> None:
        self.triple = triple

    def __enter__(self):
        from saag.core import models

        self._models = models
        self._saved = tuple(getattr(models, n) for n in self._NAMES)
        for name, value in zip(self._NAMES, self.triple):
            setattr(models, name, float(value))
        return self

    def __exit__(self, *exc):
        for name, value in zip(self._NAMES, self._saved):
            setattr(self._models, name, value)
        return False


# ── Level 1: the induced ordering of w(t) itself ──────────────────────────────

def _topic_terms(topologies: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three normalised terms for every topic in the corpus, unweighted.

    Reads SizeNorm/FreqNorm through ``compute_size_norm``/``compute_freq_norm``
    rather than reimplementing the envelope math inline, so this diagnostic
    cannot silently drift from the shipped formula the way an earlier version
    of it did (a hardcoded KiB/50.0 divisor and a bespoke frequency-fallback
    ladder, both retired -- see saag/core/models.py).
    """
    from saag.core.models import QoSPolicy, compute_size_norm, compute_freq_norm

    qos, size, freq = [], [], []
    for topology in topologies.values():
        for t in topology.get("topics", []):
            policy = QoSPolicy.from_dict(t.get("qos", {}))
            qos.append(policy.calculate_weight())

            raw_size = t.get("size", t.get("message_size", 1024)) or 1024
            size.append(compute_size_norm(raw_size))

            freq.append(compute_freq_norm(t.get("frequency")))
    return np.array(qos), np.array(size), np.array(freq)


# ── Levels 2 and 3: downstream rho against I*(v) ──────────────────────────────

def _topo_qos_rho(topology: Dict, truth: Dict[str, float]) -> Optional[float]:
    """rho(Topo-QoS, I*) on the Application population under current weights."""
    from cli.loso_evaluate import _build_graph_from_json
    from reproduce.main_table import (
        _compute_topo_baseline_scores, _derive_depends_on_edges,
        _saag_structural_features,
    )
    import networkx as nx

    deps = _derive_depends_on_edges(topology)
    app_ids = {a["id"] for a in topology.get("applications", [])}
    lib_ids = {lb["id"] for lb in topology.get("libraries", [])}
    allowed = app_ids | lib_ids

    dep_graph = nx.DiGraph()
    for nid in app_ids:
        dep_graph.add_node(nid, type="Application")
    for nid in lib_ids:
        dep_graph.add_node(nid, type="Library")
    for e in deps:
        src, dst = str(e["source"]), str(e["target"])
        if src in allowed and dst in allowed:
            dep_graph.add_edge(
                src, dst,
                weight=float(e.get("weight", 1.0)),
                qos_weight=float(e.get("qos_weight", 1.0)),
                type="DEPENDS_ON",
                dependency_type=e.get("type", "app_to_app"),
            )
    if dep_graph.number_of_nodes() == 0:
        return None

    structural = _saag_structural_features(topology)
    pred = _compute_topo_baseline_scores(dep_graph, structural, use_qos=True)
    if not pred:
        return None
    pred = {str(k): float(v) for k, v in pred.items()}

    graph = _build_graph_from_json(topology)
    return _rho(pred, truth, resolve_eval_keys(pred, truth, graph, "application"))


def _rm_rho(topology: Dict, truth: Dict[str, float], layer: str = "system") -> Optional[float]:
    """rho(Q(v), I*) on the Application population under current weights.

    Uses the same full-pipeline scorer as ``reproduce/ahp_sensitivity.py`` rather
    than the DEPENDS_ON-projection recomputation, whose Availability channel is
    degenerate on that substrate (see that module for the details).
    """
    from cli.loso_evaluate import _build_graph_from_json
    from reproduce.ahp_sensitivity import _score_components

    pred = _score_components(topology, lam=0.70, layer=layer)
    if not pred:
        return None
    graph = _build_graph_from_json(topology)
    return _rho(pred, truth, resolve_eval_keys(pred, truth, graph, "application"))


def _rho(pred: Dict[str, float], truth: Dict[str, float], keys: List[str]) -> Optional[float]:
    """Spearman rho on an explicit key set, or None when it is degenerate."""
    if len(keys) < 3:
        return None
    a = np.array([pred[k] for k in keys])
    b = np.array([truth[k] for k in keys])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    good = [v for v in vals if v is not None]
    return round(float(np.mean(good)), 4) if good else None


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(
    scenarios: List[str],
    grid: List[Tuple[float, float, float]],
    skip_downstream: bool = False,
) -> Dict[str, Any]:
    from reproduce.ahp_sensitivity import _load_topology
    from reproduce.main_table import _load_scenario_data

    topologies: Dict[str, Dict] = {}
    truths: Dict[str, Dict[str, float]] = {}
    for scenario in scenarios:
        try:
            topologies[scenario] = _load_topology(scenario)
            _g, _s, simulation, _r, _gt = _load_scenario_data(scenario, substrate="projection")
            truths[scenario] = {
                k: float(v.get("composite", 0.0)) for k, v in simulation.items()
            }
        except Exception as exc:      # noqa: BLE001 - one bad scenario must not kill the sweep
            logger.warning("%s failed to load: %s", scenario, exc)

    qos, size, freq = _topic_terms(topologies)
    n_topics = len(qos)
    shipped = DEFAULT_GRID[0]
    w_shipped = shipped[0] * qos + shipped[1] * size + shipped[2] * freq

    # How much of w(t)'s spread does each term actually supply at the shipped
    # triple? This is the structural half of the justification: a term whose
    # weighted contribution is an order of magnitude below another's cannot
    # reorder anything, whatever coefficient it carries.
    term_contributions = {
        "qos":       {"sd_raw": round(float(qos.std()), 4),
                      "sd_weighted": round(float((shipped[0] * qos).std()), 5)},
        "size_norm": {"sd_raw": round(float(size.std()), 4),
                      "sd_weighted": round(float((shipped[1] * size).std()), 5)},
        "freq_norm": {"sd_raw": round(float(freq.std()), 4),
                      "sd_weighted": round(float((shipped[2] * freq).std()), 5)},
    }

    rows: List[Dict[str, Any]] = []
    for triple in grid:
        beta, alpha, psi = triple
        w = beta * qos + alpha * size + psi * freq
        w_rank = (1.0 if np.ptp(w) == 0 or np.ptp(w_shipped) == 0
                  else float(spearmanr(w, w_shipped).statistic))

        row: Dict[str, Any] = {
            "beta": round(beta, 4), "alpha": round(alpha, 4), "psi": round(psi, 4),
            "is_shipped": tuple(round(x, 4) for x in triple) == tuple(round(x, 4) for x in shipped),
            "w_rank_rho_vs_shipped": round(w_rank, 4),
            "n_topics": n_topics,
        }

        if not skip_downstream:
            per_scenario: Dict[str, Dict[str, Optional[float]]] = {}
            with _TopicWeights(triple):
                for scenario, topology in topologies.items():
                    truth = truths.get(scenario, {})
                    if not truth:
                        continue
                    try:
                        tq = _topo_qos_rho(topology, truth)
                    except Exception as exc:      # noqa: BLE001
                        logger.warning("%s topo_qos @ %s failed: %s", scenario, triple, exc)
                        tq = None
                    try:
                        rm = _rm_rho(topology, truth)
                    except Exception as exc:      # noqa: BLE001
                        logger.warning("%s rm @ %s failed: %s", scenario, triple, exc)
                        rm = None
                    per_scenario[scenario] = {"topo_qos_rho": tq, "rm_rho": rm}

            row["eval_population"] = "application"
            row["per_scenario"] = {
                k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()}
                for k, v in sorted(per_scenario.items())
            }
            row["topo_qos_mean_rho"] = _mean([v["topo_qos_rho"] for v in per_scenario.values()])
            row["rm_mean_rho"] = _mean([v["rm_rho"] for v in per_scenario.values()])

        rows.append(row)
        print(f"  beta={beta:.2f} alpha={alpha:.2f} psi={psi:.2f}  "
              f"w_rank_rho={row['w_rank_rho_vs_shipped']:.4f}  "
              f"topo_qos={row.get('topo_qos_mean_rho')}  rm={row.get('rm_mean_rho')}")

    defined_w = [r["w_rank_rho_vs_shipped"] for r in rows]
    tq = [r["topo_qos_mean_rho"] for r in rows if r.get("topo_qos_mean_rho") is not None]
    rm = [r["rm_mean_rho"] for r in rows if r.get("rm_mean_rho") is not None]

    interpretation = {
        "min_w_rank_rho_vs_shipped": round(float(np.min(defined_w)), 4),
        "topo_qos_rho_spread": round(float(max(tq) - min(tq)), 4) if tq else None,
        "rm_rho_spread": round(float(max(rm) - min(rm)), 4) if rm else None,
        "term_contributions_at_shipped": term_contributions,
        "note": (
            "A small spread means the declared (beta, alpha, psi) is not "
            "load-bearing: every triple in the grid, including a uniform prior, "
            "induces essentially the same ordering and the same downstream rho. "
            "That is the structural justification for the constant — it is an "
            "argument that the choice does not matter, not a claim that the "
            "shipped value is optimal. Read `term_contributions_at_shipped` "
            "alongside it: SizeNorm is log-compressed into a narrow band, so its "
            "weighted spread is far below the QoS term's and it cannot reorder "
            "topics whatever alpha is set to. rho is scored on the Application "
            "population against I*(v), the primary oracle. The GNN variants are "
            "not retrained across the grid; this bounds the constant's effect on "
            "the closed-form scorers only."
        ),
    }

    return {
        "grid": [list(map(float, t)) for t in grid],
        "shipped": list(map(float, shipped)),
        "scenarios": sorted(topologies),
        "rows": rows,
        "interpretation": interpretation,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Topic-weight (beta, alpha, psi) sensitivity")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument(
        "--output", type=Path,
        default=RESULTS_DIR / "topic_weight_sensitivity.json",
    )
    p.add_argument(
        "--skip-downstream", action="store_true",
        help="Report only the w(t) ordering (level 1). Skips the pipeline reruns, "
             "which dominate the cost.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)
    from reproduce.main_table import ALL_SCENARIOS

    scenarios = args.scenarios or ALL_SCENARIOS
    print(f"Topic-weight sensitivity: {len(DEFAULT_GRID)} triples x {len(scenarios)} scenarios")

    report = run_sweep(scenarios, DEFAULT_GRID, skip_downstream=args.skip_downstream)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    for k, v in report["interpretation"].items():
        if k not in ("note", "term_contributions_at_shipped"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
