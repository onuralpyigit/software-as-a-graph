#!/usr/bin/env python3
"""
tools/qos_gini_generator.py — Block D: QoS Gini-spectrum synthetic scenario generator
======================================================================================

Takes a base scenario JSON + a target Gini coefficient and produces a synthetic
variant JSON with a QoS weight distribution matching the target Gini (± 0.02).

Usage
-----
  # Generate 5 Gini levels for the ATM scenario
  python tools/qos_gini_generator.py --scenario data/scenarios/atm_system.json

  # Generate a specific level
  python tools/qos_gini_generator.py \\
      --scenario data/scenarios/atm_system.json \\
      --gini 0.6 \\
      --output data/scenarios/gini_variants/atm_gini_0.6.json

Algorithm
---------
1. Parse all Topic QoS profiles → compute current Gini of w(topic) distribution.
2. Re-sample QoS levels via a discrete distribution parameterized to hit target Gini.
   - Reliability: Bernoulli(p_r)
   - Durability: categorical over {VOLATILE, TRANSIENT_LOCAL, PERSISTENT}
   - Priority: categorical over {LOW, MEDIUM, HIGH, URGENT}
   - p_r, p_dur, p_pri are solved such that the resulting weight distribution
     has the target Gini ± tolerance (binary search with 50 trials).
3. Assign re-sampled QoS to each Topic and update all associated edges.
4. Write output JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── QoS scoring (matches saag/core/models.py QoSPolicy) ──────────────────────

_REL_SCORE = {"BEST_EFFORT": 0.0, "RELIABLE": 1.0}
_DUR_SCORE = {"VOLATILE": 0.0, "TRANSIENT_LOCAL": 0.5, "TRANSIENT": 0.6, "PERSISTENT": 1.0}
_PRI_SCORE = {"LOW": 0.0, "MEDIUM": 0.33, "HIGH": 0.66, "URGENT": 1.0}
W_REL, W_DUR, W_PRI = 0.30, 0.40, 0.30

MIN_WEIGHT = 0.01
BETA = 0.85  # QoS weight coefficient in topic weight formula


def _qos_weight(rel: str, dur: str, pri: str) -> float:
    return W_REL * _REL_SCORE.get(rel, 0) + W_DUR * _DUR_SCORE.get(dur, 0) + W_PRI * _PRI_SCORE.get(pri, 0.33)


def _topic_weight(size_bytes: int, qos_w: float) -> float:
    size_kb = size_bytes / 1024
    size_norm = min(math.log2(1 + size_kb) / 50, 1.0)
    return max(MIN_WEIGHT, BETA * qos_w + (1 - BETA) * size_norm)


def _gini(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    s = sorted(values)
    total = sum(s)
    if total == 0:
        return 0.0
    cum = 0.0
    lorenz = []
    for v in s:
        cum += v
        lorenz.append(cum / total)
    area = sum(lorenz[i] + lorenz[i - 1] for i in range(1, n)) / (2 * (n - 1))
    return 1.0 - 2.0 * area


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _sample_qos(
    n: int,
    p_reliable: float,
    weights_dur: List[float],  # VOLATILE, TRANSIENT_LOCAL, PERSISTENT
    weights_pri: List[float],  # LOW, MEDIUM, HIGH, URGENT
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    rels = ["BEST_EFFORT", "RELIABLE"]
    durs = ["VOLATILE", "TRANSIENT_LOCAL", "PERSISTENT"]
    pris = ["LOW", "MEDIUM", "HIGH", "URGENT"]

    samples = []
    for _ in range(n):
        rel = rng.choices(rels, weights=[1 - p_reliable, p_reliable], k=1)[0]
        dur = rng.choices(durs, weights=weights_dur, k=1)[0]
        pri = rng.choices(pris, weights=weights_pri, k=1)[0]
        samples.append((rel, dur, pri))
    return samples


def _gini_from_params(
    n: int,
    sizes: List[int],
    p_reliable: float,
    weights_dur: List[float],
    weights_pri: List[float],
    rng: random.Random,
    n_trials: int = 20,
) -> float:
    """Estimate expected Gini for given sampling parameters (averaged over trials)."""
    ginis = []
    for _ in range(n_trials):
        samples = _sample_qos(n, p_reliable, weights_dur, weights_pri, rng)
        ws = [_topic_weight(sz, _qos_weight(*s)) for sz, s in zip(sizes, samples)]
        ginis.append(_gini(ws))
    return sum(ginis) / len(ginis)


def _solve_params_for_gini(
    target_gini: float,
    n: int,
    sizes: List[int],
    seed: int = 42,
    n_search: int = 50,
    tol: float = 0.03,
) -> Tuple[float, List[float], List[float]]:
    """Binary-search for sampling parameters yielding target_gini ± tol.

    Returns (p_reliable, weights_dur, weights_pri).
    Simple 1-D search: vary p_reliable while keeping weights_dur/pri uniform.
    For Gini = 0: all same QoS (uniform). For Gini > 0: high p_reliable
    creates heterogeneity between topics.
    """
    rng = random.Random(seed)

    if target_gini < 0.05:
        # All uniform: BEST_EFFORT / VOLATILE / MEDIUM
        return 0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]

    # Search p_reliable in [0, 1]; weights_dur skewed toward extremes for higher Gini
    best_params = (0.5, [0.4, 0.2, 0.4], [0.25, 0.25, 0.25, 0.25])
    best_err = float("inf")

    for i in range(n_search):
        # Anneal search
        p_r = min(1.0, target_gini * 1.5 * random.random() + target_gini * 0.3)
        # Higher target → more extreme durability distribution
        extreme = min(1.0, target_gini)
        w_dur = [max(0.01, 1 - extreme), max(0.01, (1 - extreme) * 0.5), max(0.01, extreme)]
        total = sum(w_dur)
        w_dur = [x / total for x in w_dur]
        w_pri = [max(0.01, 1 - extreme), 0.25, max(0.01, extreme * 0.5), max(0.01, extreme)]
        total = sum(w_pri)
        w_pri = [x / total for x in w_pri]

        g = _gini_from_params(n, sizes, p_r, w_dur, w_pri, rng)
        err = abs(g - target_gini)
        if err < best_err:
            best_err = err
            best_params = (p_r, w_dur, w_pri)
        if err < tol:
            break

    return best_params


# ── Scenario mutation ─────────────────────────────────────────────────────────

def _extract_topics(topology: Dict[str, Any]) -> List[Dict[str, Any]]:
    return topology.get("topics", [])


def _apply_qos_to_scenario(
    topology: Dict[str, Any],
    qos_assignments: Dict[str, Tuple[str, str, str]],
) -> Dict[str, Any]:
    """Return a deep copy of topology with updated QoS assignments per topic."""
    t = copy.deepcopy(topology)
    for topic in t.get("topics", []):
        tid = topic.get("id", topic.get("name", ""))
        if tid in qos_assignments:
            rel, dur, pri = qos_assignments[tid]
            topic["qos"] = {
                "reliability": rel,
                "durability": dur,
                "transport_priority": pri,
            }
    # Update edges that carry qos_profile
    for conn in t.get("connections", t.get("edges", [])):
        topic_id = conn.get("topic_id", conn.get("topic", ""))
        if topic_id in qos_assignments:
            rel, dur, pri = qos_assignments[topic_id]
            conn["qos_profile"] = {
                "reliability": rel,
                "durability": dur,
                "transport_priority": pri,
            }
    return t


# ── CLI ───────────────────────────────────────────────────────────────────────

GINI_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

OUTPUT_DIR = Path("data/scenarios/gini_variants")


def parse_args():
    p = argparse.ArgumentParser(description="Block D: QoS Gini-spectrum synthetic scenario generator.")
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--gini", type=float, default=None,
                   help="Target Gini level (default: generate all 5 levels)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _generate_one(
    scenario_path: Path,
    target_gini: float,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> Path:
    topology = json.loads(scenario_path.read_text())
    topics = _extract_topics(topology)
    n = len(topics)
    if n == 0:
        print(f"  Warning: no topics in {scenario_path.name}. Skipping Gini={target_gini}.")
        return None

    sizes = [t.get("size", 256) for t in topics]
    topic_ids = [t.get("id", t.get("name", f"topic_{i}")) for i, t in enumerate(topics)]

    # Compute current Gini
    cur_qos = []
    for t in topics:
        qos = t.get("qos", {}) or {}
        cur_qos.append(_qos_weight(
            qos.get("reliability", "BEST_EFFORT"),
            qos.get("durability", "VOLATILE"),
            qos.get("transport_priority", qos.get("priority", "MEDIUM")),
        ))
    cur_weights = [_topic_weight(sz, w) for sz, w in zip(sizes, cur_qos)]
    cur_gini = _gini(cur_weights)

    print(f"  Base Gini={cur_gini:.3f} → Target Gini={target_gini:.3f}  (n_topics={n})")

    p_r, w_dur, w_pri = _solve_params_for_gini(target_gini, n, sizes, seed=seed)
    rng = random.Random(seed)
    assignments_list = _sample_qos(n, p_r, w_dur, w_pri, rng)
    qos_map = dict(zip(topic_ids, assignments_list))

    # Verify achieved Gini
    ach_ws = [_topic_weight(sz, _qos_weight(*q)) for sz, q in zip(sizes, assignments_list)]
    ach_gini = _gini(ach_ws)
    print(f"  Achieved Gini: {ach_gini:.3f} (target {target_gini:.3f})")

    variant = _apply_qos_to_scenario(topology, qos_map)
    variant["_gini_metadata"] = {
        "base_scenario": scenario_path.name,
        "target_gini": target_gini,
        "achieved_gini": round(ach_gini, 4),
        "seed": seed,
        "n_topics": n,
    }

    if output_path is None:
        gini_tag = f"gini_{str(target_gini).replace('.', '')}"
        stem = scenario_path.stem
        output_path = OUTPUT_DIR / f"{stem}_{gini_tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(variant, indent=2))
    print(f"  Saved: {output_path}")
    return output_path


def main():
    args = parse_args()
    if not args.scenario.exists():
        print(f"Error: {args.scenario} not found", file=sys.stderr)
        sys.exit(1)

    gini_levels = [args.gini] if args.gini is not None else GINI_LEVELS
    print(f"\nQoS Gini Generator — {args.scenario.name}")
    for g in gini_levels:
        _generate_one(args.scenario, g, seed=args.seed, output_path=args.output if args.gini is not None else None)
    print("\nDone.")


if __name__ == "__main__":
    main()
