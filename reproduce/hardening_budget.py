"""
C2 — Hardening-budget experiment.

Operationalization: "hardening" a component means giving it a hot replica/failover so that,
if it fails, the failure no longer cascades to its dependents (I*(v) for a hardened v is
driven to ~0 by construction of the redundancy). What an operator with a budget of K
hardening actions wants to know is: which K components should I harden to eliminate the
largest share of total simulated cascade-impact risk in the system?

That quantity is the *risk-mass coverage* of the top-K selection:

    coverage(method) = sum(I*(v) for v in top_K_by_method) / sum(I*(v) for v in all Application nodes)

computed against the same independent FaultInjector ground truth used throughout this paper.
This metric needs only one FaultInjector pass per scenario (no graph surgery on the topology
that FaultInjector consumes), avoiding an earlier, rejected design: physically deleting the
top-K nodes from the graph before re-simulating. That first attempt was invalidated by its own
result — it made every method's "hardening" look harmful, because deleting a publisher node
destroys it as a message source for its subscribers (permanent, unbounded feed loss), which is
the opposite of what hardening (replication/failover) is supposed to model.

K matches the paper's rank-matched-binarization convention: K = |{v in Application : I*(v) > 0.5}|.
"""
import json
import random
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]  # repo root (reproduce/ is one level down)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from cli.simulate_graph import _load_graph
from saag.simulation.fault_injector import FaultInjector
from reversed_projection_ablation import build_projection

SCENARIOS = {
    "AV System": "av_system",
    "IoT Smart City": "iot_smart_city_system",
    "Financial Trading": "financial_trading_system",
    "Healthcare": "healthcare_system",
    "Hub-and-Spoke": "hub_and_spoke_system",
    "Microservices": "microservices_system",
    "Enterprise": "enterprise_system",
}
SEEDS = [42, 123, 456, 789, 2024]

LOSO_PREDICTIONS = json.loads(
    (ROOT / "output" / "loso" / "hgl_qos" / "inductive_predictions.json").read_text()
)


def top_k_betweenness(g: nx.DiGraph, apps, k: int):
    proj = build_projection(g, reversed_direction=False)
    n = proj.number_of_nodes()
    centrality = {v: len(nx.ancestors(proj, v)) / max(1, n - 1) for v in proj.nodes}
    return sorted(apps, key=lambda a: centrality.get(a, 0.0), reverse=True)[:k]


def top_k_hgl(stem, apps, k: int):
    preds = LOSO_PREDICTIONS.get(stem, {})
    ranked = sorted((a for a in apps if a in preds), key=lambda a: preds[a]["overall"], reverse=True)
    if len(ranked) < k:
        ranked += [a for a in apps if a not in ranked][: k - len(ranked)]
    return ranked[:k]


def main():
    rows = []
    for label, stem in SCENARIOS.items():
        g = _load_graph(ROOT / "data" / "scenarios" / f"{stem}.json")
        apps = [n for n, d in g.nodes(data=True) if d.get("type") == "Application"]

        injector = FaultInjector(graph=g, seeds=SEEDS)
        result = injector.run(node_ids=apps)
        impact = {nid: rec.impact_score for nid, rec in result.records.items()}
        total_risk = sum(impact.values())

        K = sum(1 for v in impact.values() if v > 0.5)
        if K == 0:
            K = max(1, round(0.1 * len(apps)))

        methods = {
            "HGL": top_k_hgl(stem, apps, K),
            "Betweenness": top_k_betweenness(g, apps, K),
            "Random": random.Random(42).sample(apps, K),
        }

        row = {"scenario": label, "K": K, "n_apps": len(apps), "total_risk": total_risk}
        for method, selected in methods.items():
            covered = sum(impact.get(a, 0.0) for a in selected)
            coverage_pct = 100.0 * covered / total_risk if total_risk > 0 else 0.0
            row[f"{method}_coverage_pct"] = coverage_pct
        rows.append(row)

        print(
            f"{label:20s} K={row['K']:3d}  "
            f"HGL={row['HGL_coverage_pct']:6.1f}%  "
            f"Betw={row['Betweenness_coverage_pct']:6.1f}%  "
            f"Rand={row['Random_coverage_pct']:6.1f}%"
        )

    summary = {}
    for method in ("HGL", "Betweenness", "Random"):
        vals = [r[f"{method}_coverage_pct"] for r in rows]
        summary[method] = sum(vals) / len(vals)

    print("\nMean simulated cascade-risk mass covered by top-K hardening, across 7 scenarios:")
    for method, val in summary.items():
        print(f"  {method:12s} {val:6.2f}%")

    out = {"seeds": SEEDS, "rows": rows, "mean_coverage_pct": summary}
    out_path = ROOT / "output" / "hardening_budget_experiment.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
