#!/usr/bin/env python3
"""
Batch script to run prescriptive optimization across all seven benchmark scenarios.
Extracts baseline SRI, optimized SRI, delta improvement, and counts of applied refactoring operators.
"""

import sys
import json
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

def main():
    print("| Scenario | Baseline SRI | Mutated SRI | Delta | Splits | Reallocs | Upgrades | Remediated w/ ΔI | Mean ΔI% (§6.7) |")
    print("|----------|:------------:|:-----------:|:-----:|:------:|:--------:|:--------:|:-----------------:|:---------------:|")

    all_reductions = []
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
        res = client.prescribe(analysis_result=analysis, layer="system")

        policy = res.policy
        splits = len(policy.topic_splits)
        reallocs = len(policy.node_reallocations)
        upgrades = len(policy.qos_upgrades)

        # §6.7: mean cascade-impact reduction, restricted to remediated components with a stable
        # id across the mutation (node reallocations, QoS upgrades). Reported honestly per-scenario
        # since the current PrescribeService applies the full policy unconditionally (no per-edit
        # accept/reject filter, §6.4) and the result is a mixed one, not a uniform improvement.
        n_with_delta = len(res.remediated_component_impact_deltas)
        mean_pct = res.mean_cascade_impact_reduction
        mean_pct_str = f"{mean_pct * 100:+.2f}%" if mean_pct is not None else "n/a"
        if mean_pct is not None:
            all_reductions.append(mean_pct)

        print(f"| {name} | {res.original_sri:.4f} | {res.mutated_sri:.4f} | +{res.sri_improvement:.4f} | "
              f"{splits} | {reallocs} | {upgrades} | {n_with_delta} | {mean_pct_str} |")

    if all_reductions:
        overall = sum(all_reductions) / len(all_reductions)
        print(f"\nMean §6.7 cascade-impact reduction across {len(all_reductions)} scenarios: {overall * 100:+.2f}%")

if __name__ == "__main__":
    main()
