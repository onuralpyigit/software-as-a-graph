#!/usr/bin/env python3
"""
Control-arm experiment for the prescriptive pipeline: does per-edit verification do useful work?

`run_prescribe_all.py` reports what the *verified* policy achieves, but nothing to compare it to,
so the value of the acceptance filter is argued rather than measured. This script applies five
policies drawn from the same candidate set to the same scenario and scores each against the
unmutated graph with the same cascade oracle the filter uses:

    verified     edits the per-edit filter admitted (kappa * sigma_seed at every threshold)
    all          every candidate, i.e. no verification
    rejected     only the edits the filter withheld
    random       uniform random subsets of the candidates, sized like `verified`
    random_mix   random subsets with `verified`'s per-operator counts, which separates the
                 filter's choice of edits from its choice of operators

Outcomes per (threshold, seed), each paired against the baseline at the same point:

    fixed_rel    mean impact reduction over the baseline's components, relative to their mean
                 baseline impact. The filter's own statistic (`verifier.mean_reduction`), so a
                 policy is judged by the criterion that selected it.
    closed_rel   reduction in *total* impact summed over every component of each graph,
                 including the hosts a reallocation adds. The filter's statistic ignores those
                 hosts, because they have no baseline counterpart; this one does not.
    max_delta    reduction in the worst single-component impact.

Selection and evaluation use disjoint seeds: the filter admits edits on `--verify-seeds` and the
arms are scored on `--eval-seeds` (in-sample scores on the verification seeds are kept too).
"""

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.analysis.antipattern_detector import CATALOG
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prescription.evaluator import GraphEvaluator, repo_from_json
from saag.prescription.models import PrescriptionPolicy
from saag.prescription.mutator import apply_policy
from saag.prescription.service import PrescribeService
from saag.prescription.verifier import DEFAULT_SEEDS, DEFAULT_THRESHOLDS, EditVerifier
from reproduce._provenance import stamp
from reproduce.detection_validation import DEFAULT_EXCLUDED_PATTERNS
from reproduce.run_prescribe_all import SCENARIOS

DEFAULT_EVAL_SEEDS = (789, 2024, 31337)
OUTCOMES = ("fixed_rel", "closed_rel", "max_delta")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--verify-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--draws", type=int, default=20,
                   help="Random subsets per random arm (default 20).")
    p.add_argument("--rng-seed", type=int, default=20261009)
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"Scenario filenames to run, in order (default: all {len(SCENARIOS)}).")
    p.add_argument("--output", type=Path, default=Path("results/prescribe_controls.json"))
    p.add_argument("--resume", action="store_true",
                   help="Skip scenarios already present in --output.")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--heldout-only", action="store_true",
                   help="Score control arms on --eval-seeds only; only `verified` (and the baseline) "
                        "is also scored on --verify-seeds, which the additivity check needs. For "
                        "scenarios too large to score every arm in-sample.")
    return p.parse_args()


# ── Arm construction ──────────────────────────────────────────────────────────

def build_arms(
    accepted: Sequence[bool],
    kinds: Sequence[str],
    draws: int,
    rng: random.Random,
) -> Dict[str, List[List[int]]]:
    """Each arm is a list of draws; each draw is a sorted list of candidate indices."""
    n = len(accepted)
    verified = [i for i in range(n) if accepted[i]]
    rejected = [i for i in range(n) if not accepted[i]]
    arms: Dict[str, List[List[int]]] = {
        "verified": [verified],
        "all": [list(range(n))],
        "rejected": [rejected],
    }
    if not verified or not rejected:
        # With nothing admitted or nothing withheld, every random subset of
        # size |verified| is either empty or the verified set itself.
        return arms

    by_kind: Dict[str, List[int]] = {}
    for i, kind in enumerate(kinds):
        by_kind.setdefault(kind, []).append(i)
    quota = {k: sum(1 for i in verified if kinds[i] == k) for k in by_kind}

    arms["random"] = [sorted(rng.sample(range(n), len(verified))) for _ in range(draws)]
    arms["random_mix"] = [
        sorted(i for k, pool in by_kind.items() for i in rng.sample(pool, quota[k]))
        for _ in range(draws)
    ]
    return arms


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_point(base: Dict[str, float], mutated: Dict[str, float]) -> Dict[str, float]:
    """The three outcomes for one arm at one (threshold, seed); positive means better."""
    base_mean = statistics.fmean(base.values())
    base_total = sum(base.values())
    fixed = statistics.fmean(base[c] - mutated.get(c, 0.0) for c in base)
    return {
        # Absolute form of fixed_rel, in the filter's own units (additivity check only).
        "fixed_abs": fixed,
        "fixed_rel": fixed / base_mean if base_mean else 0.0,
        "closed_rel": (base_total - sum(mutated.values())) / base_total if base_total else 0.0,
        "max_delta": max(base.values()) - max(mutated.values()),
    }


_EDITS: List[Any] = []
_ORIGINAL: Dict[str, Any] = {}


def _init_worker(edits: List[Any], original_json: Dict[str, Any]) -> None:
    global _EDITS, _ORIGINAL
    _EDITS, _ORIGINAL = edits, original_json


def _impact_worker(task) -> Dict[str, float]:
    indices, threshold, seed = task
    policy = PrescriptionPolicy.from_edits([_EDITS[i] for i in indices])
    repo = repo_from_json(apply_policy(_ORIGINAL, policy))
    return GraphEvaluator("system").impact(repo, threshold=threshold, seed=seed)


def evaluate_arms(
    arms: Dict[str, List[List[int]]],
    edits: List[Any],
    original_json: Dict[str, Any],
    thresholds: Sequence[float],
    seeds: Sequence[int],
    jobs: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Score every draw of every arm at every (threshold, seed) against the baseline."""
    points = [(t, s) for t in thresholds for s in seeds]
    graphs = [([], "baseline", 0)] + [
        (idx, arm, d) for arm, draws in arms.items() for d, idx in enumerate(draws)
    ]
    tasks = [(idx, t, s) for idx, _, _ in graphs for t, s in points]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs, initializer=_init_worker, initargs=(edits, original_json),
    ) as pool:
        impacts = list(pool.map(_impact_worker, tasks, chunksize=1))

    by_graph = [impacts[g * len(points):(g + 1) * len(points)] for g in range(len(graphs))]
    baseline = dict(zip(points, by_graph[0]))

    results: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in arms}
    for (idx, arm, _), maps in zip(graphs[1:], by_graph[1:]):
        per_point = [
            {"threshold": t, "seed": s, **score_point(baseline[(t, s)], m)}
            for (t, s), m in zip(points, maps)
        ]
        results[arm].append({
            "n_edits": len(idx),
            "n_new_hosts": sum(1 for i in idx if edits[i].KIND == "node_reallocation"),
            "per_point": per_point,
        })
    return results


def summarise(draw: Dict[str, Any], seeds: Sequence[int]) -> Dict[str, float]:
    """Mean of each outcome over the given seeds and every threshold."""
    rows = [p for p in draw["per_point"] if p["seed"] in seeds]
    return {o: statistics.fmean(p[o] for p in rows) for o in OUTCOMES}


# ── Per-scenario driver ───────────────────────────────────────────────────────

def run_scenario(filename: str, name: str, args, rng: random.Random) -> Dict[str, Any]:
    original_json = json.loads((Path("data/scenarios") / filename).read_text())
    repo = MemoryRepository()
    repo.save_graph(original_json, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)

    # Same candidate compilation as run_prescribe_all.py (see the comments there).
    analysis = client.analyze(layer="system")
    prediction = client.predict(analysis, mode="rm", active_patterns=[
        pid for pid in CATALOG if pid not in DEFAULT_EXCLUDED_PATTERNS
    ])
    candidate_policy = PrescribeService(repo).compile_policy(analysis, prediction)
    edits = candidate_policy.edits()
    export = repo.export_json()

    verifier = EditVerifier(GraphEvaluator("system"), kappa=args.kappa,
                            seeds=args.verify_seeds, thresholds=args.thresholds)
    verdicts = verify_with_progress(verifier, repo, export, edits, args, filename)
    accepted = [v.accepted for v in verdicts]
    kinds = [e.KIND for e in edits]

    arms = build_arms(accepted, kinds, args.draws, rng)
    if args.heldout_only:
        scored = evaluate_arms(arms, edits, export, args.thresholds, args.eval_seeds, args.jobs)
        insample = evaluate_arms({"verified": arms["verified"]}, edits, export,
                                 args.thresholds, args.verify_seeds, args.jobs)
        scored["verified"][0]["per_point"] += insample["verified"][0]["per_point"]
    else:
        all_seeds = list(dict.fromkeys([*args.verify_seeds, *args.eval_seeds]))
        scored = evaluate_arms(arms, edits, export, args.thresholds, all_seeds, args.jobs)

    for draws in scored.values():
        for d in draws:
            d["heldout"] = summarise(d, args.eval_seeds)
            if any(p["seed"] in args.verify_seeds for p in d["per_point"]):
                d["insample"] = summarise(d, args.verify_seeds)

    return {
        "scenario": name,
        "file": filename,
        "config": _config(args),
        "n_candidates": len(edits),
        "n_accepted": sum(accepted),
        "candidates_by_operator": {k: kinds.count(k) for k in sorted(set(kinds))},
        "accepted_by_operator": {
            k: sum(1 for a, kk in zip(accepted, kinds) if a and kk == k) for k in sorted(set(kinds))
        },
        "additivity": additivity(verdicts, scored["verified"][0], args),
        "arms": scored,
        "contrasts": contrasts(scored),
        "verdicts": [
            {"kind": v.kind, "target": v.target, "accepted": v.accepted,
             "per_threshold": {t: s.to_dict() for t, s in v.per_threshold.items()}}
            for v in verdicts
        ],
    }


def verify_with_progress(verifier, repo, export, edits, args, filename) -> List[Any]:
    """`EditVerifier.verify`, with progress output and an on-disk cache of the verdicts.

    Verification dominates the cost on the largest scenario (hours), so it reports as it
    goes and its verdicts survive an interrupted run: a rerun with the same candidates and
    settings loads them instead of re-simulating.
    """
    from saag.prescription.models import EditVerdict, ThresholdStat
    from saag.prescription.verifier import _verify_single_edit_worker, judge

    cache = args.output.with_name(f"{args.output.stem}.{Path(filename).stem}.verdicts.json")
    key = {"targets": [[e.KIND, e.target] for e in edits], "kappa": args.kappa,
           "seeds": args.verify_seeds, "thresholds": args.thresholds}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["key"] == key:
            print(f"  loaded {len(saved['verdicts'])} cached verdicts from {cache}", flush=True)
            return [
                judge(v["kind"], v["target"],
                      {t: ThresholdStat(**s) for t, s in v["per_threshold"].items()},
                      args.kappa, len(args.thresholds), v.get("reason", "") if not v["per_threshold"] else "")
                for v in saved["verdicts"]
            ]

    baselines = verifier._baselines(repo)
    tasks = [(e, export, args.kappa, args.thresholds, args.verify_seeds, baselines, "system", None)
             for e in edits]
    verdicts: List[Optional[EditVerdict]] = [None] * len(edits)
    step = max(1, len(edits) // 20)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_verify_single_edit_worker, t): i for i, t in enumerate(tasks)}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            verdicts[futures[fut]] = fut.result()
            if n % step == 0 or n == len(edits):
                print(f"  verified {n}/{len(edits)} edits", flush=True)

    cache.write_text(json.dumps({"key": key, "verdicts": [
        {"kind": v.kind, "target": v.target, "accepted": v.accepted, "reason": v.reason,
         "per_threshold": {t: {"mean_delta": s.mean_delta, "sigma_seed": s.sigma_seed}
                           for t, s in v.per_threshold.items()}}
        for v in verdicts
    ]}))
    return verdicts


def _config(args) -> Dict[str, Any]:
    """The settings one scenario was run with; scenarios of one artifact may differ."""
    return {"kappa": args.kappa, "verify_seeds": args.verify_seeds,
            "eval_seeds": args.eval_seeds, "thresholds": args.thresholds,
            "draws": args.draws, "rng_seed": args.rng_seed,
            "heldout_only": args.heldout_only}


def additivity(verdicts, verified: Dict[str, Any], args) -> Optional[Dict[str, Any]]:
    """Sum of admitted edits' isolated effects vs. the effect of applying them together.

    Both are in the filter's absolute units (mean impact reduction over baseline components)
    on the verification seeds, so a ratio below 1 means the admitted edits interfere.
    """
    if not verified["n_edits"]:
        return None
    out = {}
    for t in args.thresholds:
        isolated = sum(v.per_threshold[str(t)].mean_delta for v in verdicts if v.accepted)
        joint = statistics.fmean(
            p["fixed_abs"] for p in verified["per_point"]
            if p["threshold"] == t and p["seed"] in args.verify_seeds
        )
        out[str(t)] = {"sum_of_isolated": isolated, "joint": joint,
                       "ratio": joint / isolated if isolated else None}
    return out


def contrasts(scored: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Held-out contrasts of `verified` against each control arm, per outcome."""
    v = scored["verified"][0]["heldout"]
    out: Dict[str, Any] = {}
    for arm in ("all", "rejected"):
        out[f"verified_minus_{arm}"] = {o: v[o] - scored[arm][0]["heldout"][o] for o in OUTCOMES}
    for arm in ("random", "random_mix"):
        if arm not in scored:
            continue
        draws = [d["heldout"] for d in scored[arm]]
        out[f"vs_{arm}"] = {
            o: {
                "random_median": statistics.median(d[o] for d in draws),
                "verified_minus_median": v[o] - statistics.median(d[o] for d in draws),
                # One-sided empirical p: share of draws at least as good as verified.
                "p_empirical": (1 + sum(d[o] >= v[o] for d in draws)) / (1 + len(draws)),
            }
            for o in OUTCOMES
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    records: List[Dict[str, Any]] = []
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text())
        records = prior.get("scenarios", [])
        # Scenarios written before per-scenario configs existed ran with the
        # artifact's top-level settings.
        for r in records:
            r.setdefault("config", {
                **{k: prior[k] for k in ("kappa", "verify_seeds", "eval_seeds",
                                         "thresholds", "draws", "rng_seed")},
                "heldout_only": False,
            })
    done = {r["file"] for r in records}

    items = ([(f, SCENARIOS[f]) for f in args.scenarios if f in SCENARIOS]
             if args.scenarios else list(SCENARIOS.items()))
    for i, (filename, name) in enumerate(items):
        if filename in done:
            continue
        # Per-scenario RNG so a resumed or reordered run draws the same subsets.
        rng = random.Random(f"{args.rng_seed}:{filename}")
        rec = run_scenario(filename, name, args, rng)
        records.append(rec)
        _report(rec)
        _write(args, records)
    print(f"\nWrote {args.output}")


def _report(rec: Dict[str, Any]) -> None:
    arms = rec["arms"]
    print(f"\n{rec['scenario']}: {rec['n_accepted']}/{rec['n_candidates']} admitted")
    print(f"  {'arm':<11} {'edits':>5} {'fixed_rel':>10} {'closed_rel':>11} {'max_delta':>10}")
    for arm, draws in arms.items():
        h = [d["heldout"] for d in draws]
        med = {o: statistics.median(x[o] for x in h) for o in OUTCOMES}
        tag = arm if len(draws) == 1 else f"{arm}(med)"
        print(f"  {tag:<11} {draws[0]['n_edits']:>5} {med['fixed_rel']:>+10.4f} "
              f"{med['closed_rel']:>+11.4f} {med['max_delta']:>+10.4f}")


def _write(args, records: List[Dict[str, Any]]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Settings can differ between scenarios (see --heldout-only); each record
    # carries its own under "config", and the stamp names the last invocation.
    args.output.write_text(json.dumps({
        "provenance": stamp(**_config(args)),
        "scenarios": records,
    }, indent=2))


if __name__ == "__main__":
    main()
