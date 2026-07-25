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
from saag.infrastructure.memory_repo import MemoryRepository

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
    p.add_argument("--output", type=Path, default=Path("results/prescribe_all.json"))
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Per-edit acceptance filter: kappa={args.kappa}"
          f"{'' if args.seeds is None else f', seeds={args.seeds}'}"
          f"{'' if args.thresholds is None else f', thresholds={args.thresholds}'}\n")
    print("| Scenario | Baseline SRI | Mutated SRI | Delta | Cand. | Acc. | Rej. | Splits | Reallocs | Upgrades | Remediated w/ ΔI | Mean ΔI% (§6.7) |")
    print("|----------|:------------:|:-----------:|:-----:|:-----:|:----:|:----:|:------:|:--------:|:--------:|:-----------------:|:---------------:|")

    all_reductions = []
    records = []
    for filename, name in SCENARIOS.items():
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

        # Prescribe mutations
        prescribe_kwargs = {"kappa": args.kappa}
        if args.seeds is not None:
            prescribe_kwargs["seeds"] = args.seeds
        if args.thresholds is not None:
            prescribe_kwargs["thresholds"] = args.thresholds
        res = client.prescribe(analysis_result=analysis, layer="system", **prescribe_kwargs)

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
                sum(all_reductions) / len(all_reductions) if all_reductions else None
            ),
            "n_scenarios_regressed": len(regressions),
        },
    }, indent=2))
    print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()
