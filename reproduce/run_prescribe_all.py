#!/usr/bin/env python3
"""
Batch script to run prescriptive optimization across all seven benchmark scenarios.
Extracts baseline SRI, optimized SRI, delta improvement, and counts of applied refactoring operators.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.analysis.antipattern_detector import CATALOG
from saag.infrastructure.memory_repo import MemoryRepository
from reproduce.detection_validation import DEFAULT_EXCLUDED_PATTERNS

SCENARIOS = {
    "av_system.json": "Scenario 01 (Autonomous Vehicle)",
    "iot_smart_city_system.json": "Scenario 02 (IoT Smart City)",
    "financial_trading_system.json": "Scenario 03 (Financial Trading)",
    "healthcare_system.json": "Scenario 04 (Healthcare)",
    "hub_and_spoke_system.json": "Scenario 05 (Hub-and-Spoke)",
    "microservices_system.json": "Scenario 06 (Microservices Mesh)",
    "enterprise_system.json": "Scenario 07 (Hyper-Scale Enterprise)"
}

def parse_args():
    p = argparse.ArgumentParser(description="Prescriptive optimization across all benchmark scenarios.")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="Acceptance multiple: an edit is kept only when its mean impact reduction "
                        "exceeds kappa * sigma_seed at every propagation threshold (default 1.0).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Simulation seeds for the noise estimate (default: PrescribeService default).")
    p.add_argument("--thresholds", nargs="+", type=float, default=None,
                   help="Propagation thresholds the edit must clear at every value.")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"Scenario filenames to run (default: all {len(SCENARIOS)}).")
    p.add_argument("--output", type=Path, default=Path("results/prescribe_all.json"))
    p.add_argument("--resume", action="store_true",
                   help="Skip scenarios already present in --output and append to it. "
                        "The per-edit sweep is the expensive step here, so this matters for "
                        "recovering from an interrupted run without redoing finished scenarios.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Per-edit acceptance filter: kappa={args.kappa}"
          f"{'' if args.seeds is None else f', seeds={args.seeds}'}"
          f"{'' if args.thresholds is None else f', thresholds={args.thresholds}'}\n")
    print("| Scenario | Baseline SRI | Mutated SRI | Delta | Cand. | Acc. | Rej. | Splits | Reallocs | Upgrades | Remediated w/ ΔI | Mean ΔI% (§6.7) |")
    print("|----------|:------------:|:-----------:|:-----:|:-----:|:----:|:----:|:------:|:--------:|:--------:|:-----------------:|:---------------:|")

    # Load prior results for --resume: the per-edit verification sweep is the
    # expensive step (an exhaustive simulation per edit x threshold x seed), so
    # losing a completed scenario to an interrupted run is costly to redo.
    records = []
    done_files = set()
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text())
        records = prior.get("scenarios", [])
        done_files = {r["file"] for r in records}
        for r in records:
            mean_pct = r.get("mean_cascade_impact_reduction")
            print(f"| {r['scenario']} (resumed) | {r['original_sri']:.4f} | {r['mutated_sri']:.4f} | "
                  f"{r['sri_improvement']:+.4f} | {r['n_candidate_edits']} | {r['n_accepted_edits']} | "
                  f"{r['n_rejected_edits']} | - | - | - | - | "
                  f"{f'{mean_pct * 100:+.2f}%' if mean_pct is not None else 'n/a'} |")

    # --scenarios controls both selection and order (e.g. cheapest-first, to
    # bound how much work an interruption can lose), rather than being
    # filtered back into SCENARIOS' fixed declaration order.
    if args.scenarios:
        scenario_items = [(f, SCENARIOS[f]) for f in args.scenarios if f in SCENARIOS]
    else:
        scenario_items = list(SCENARIOS.items())

    all_reductions = [
        r["mean_cascade_impact_reduction"] for r in records
        if r.get("mean_cascade_impact_reduction") is not None
    ]
    for filename, name in scenario_items:
        if filename in done_files:
            continue
        json_path = Path("data/scenarios") / filename
        if not json_path.exists():
            print(f"Error: {json_path} not found.")
            continue

        repo = MemoryRepository()
        with open(json_path) as f:
            data = json.load(f)
        repo.save_graph(data, clear=True)
        repo.derive_dependencies()
        client = Client(repo=repo)

        # Analyze system layer
        analysis = client.analyze(layer="system")

        # Predict (RM + anti-patterns) so compile_policy() sees the same
        # CRITICAL/HIGH risk set Stage 6 is documented to consume. Without
        # this, the critical/spof/god sets it derives are all empty and only
        # Rule 1's risk-free branch (topic split) can ever fire -- Rules 2
        # and 3 (node reallocation, QoS upgrade) silently never trigger.
        # DEEP_PIPELINE is excluded: it enumerates every simple source-to-
        # sink path and does not terminate in practical time at these scales.
        prediction = client.predict(analysis, active_patterns=[
            pid for pid in CATALOG if pid not in DEFAULT_EXCLUDED_PATTERNS
        ])

        # Prescribe mutations
        prescribe_kwargs = {"kappa": args.kappa}
        if args.seeds is not None:
            prescribe_kwargs["seeds"] = args.seeds
        if args.thresholds is not None:
            prescribe_kwargs["thresholds"] = args.thresholds
        res = client.prescribe(
            analysis_result=analysis, prediction_result=prediction,
            layer="system", **prescribe_kwargs)

        policy = res.policy
        splits = len(policy.topic_splits)
        reallocs = len(policy.node_reallocations)
        upgrades = len(policy.qos_upgrades)

        # §6.7: mean cascade-impact reduction, restricted to remediated components with a stable
        # id across the mutation (node reallocations, QoS upgrades). Every edit counted here has
        # already passed the per-edit acceptance filter (§6.4), so a negative figure now means the
        # *interaction* between individually-verified edits regressed — not that an unverified edit
        # was applied. The candidate/accepted/rejected columns make the filter's action visible.
        n_with_delta = len(res.remediated_component_impact_deltas)
        mean_pct = res.mean_cascade_impact_reduction
        mean_pct_str = f"{mean_pct * 100:+.2f}%" if mean_pct is not None else "n/a"
        if mean_pct is not None:
            all_reductions.append(mean_pct)

        print(f"| {name} | {res.original_sri:.4f} | {res.mutated_sri:.4f} | {res.sri_improvement:+.4f} | "
              f"{len(res.edit_verdicts)} | {res.n_accepted} | {res.n_rejected} | "
              f"{splits} | {reallocs} | {upgrades} | {n_with_delta} | {mean_pct_str} |")

        records.append({
            "scenario": name,
            "file": filename,
            "original_sri": res.original_sri,
            "mutated_sri": res.mutated_sri,
            "sri_improvement": res.sri_improvement,
            "n_candidate_edits": len(res.edit_verdicts),
            "n_accepted_edits": res.n_accepted,
            "n_rejected_edits": res.n_rejected,
            "mean_cascade_impact_reduction": mean_pct,
            "edit_verdicts": [v.to_dict() for v in res.edit_verdicts],
        })

        # Save incrementally: the per-edit sweep above is the expensive step,
        # so a scenario finished here must not be lost to an interruption
        # before the next one completes.
        _write_output(args, records)

    n_cand = sum(r["n_candidate_edits"] for r in records)
    n_acc = sum(r["n_accepted_edits"] for r in records)
    regressions = [r for r in records
                   if r["mean_cascade_impact_reduction"] is not None
                   and r["mean_cascade_impact_reduction"] < 0]

    print(f"\nAcceptance filter: {n_acc}/{n_cand} candidate edits accepted across {len(records)} scenarios.")
    if all_reductions:
        overall = sum(all_reductions) / len(all_reductions)
        print(f"Mean §6.7 cascade-impact reduction across {len(all_reductions)} scenarios: {overall * 100:+.2f}%")
        worst = min(all_reductions)
        print(f"Worst per-scenario reduction: {worst * 100:+.2f}%  "
              f"({len(regressions)} of {len(records)} scenarios regressed)")

    _write_output(args, records)
    print(f"\nWrote {args.output}")


def _write_output(args, records) -> None:
    n_cand = sum(r["n_candidate_edits"] for r in records)
    n_acc = sum(r["n_accepted_edits"] for r in records)
    reductions = [r["mean_cascade_impact_reduction"] for r in records
                  if r["mean_cascade_impact_reduction"] is not None]
    regressions = [r for r in records
                   if r["mean_cascade_impact_reduction"] is not None
                   and r["mean_cascade_impact_reduction"] < 0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "kappa": args.kappa,
        "seeds": args.seeds,
        "thresholds": args.thresholds,
        "scenarios": records,
        "summary": {
            "n_candidate_edits": n_cand,
            "n_accepted_edits": n_acc,
            "mean_cascade_impact_reduction": (
                sum(reductions) / len(reductions) if reductions else None
            ),
            "n_scenarios_regressed": len(regressions),
        },
    }, indent=2))

if __name__ == "__main__":
    main()
