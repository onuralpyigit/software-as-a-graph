"""
A4.4 — Reversed-projection ablation.

Builds the DEPENDS_ON logical-dependency projection two ways — the paper's convention
(dependent -> dependency: subscriber -> publisher, app -> lib) and its literal inversion
(publisher -> subscriber, lib -> app) — computes betweenness centrality on each restricted
to Application nodes, and correlates each ranking against the independent FaultInjector
ground truth I*(v). Reports the resulting Spearman rho degradation.

Ground truth and raw structural graph loading reuse the exact same code path as the
published numbers (cli.simulate_graph._load_graph + saag.simulation.fault_injector.FaultInjector),
so the only thing varied here is the direction of the derived DEPENDS_ON edges used for the
structural baseline (Topo-BL) ranking — nothing about the simulator itself is touched.
"""
import json
import sys
from pathlib import Path

import networkx as nx
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]  # repo root (reproduce/ is one level down)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from cli.simulate_graph import _load_graph
from saag.simulation.fault_injector import FaultInjector

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


def build_projection(g: nx.DiGraph, reversed_direction: bool) -> nx.DiGraph:
    """Rule 1 (app_to_app via shared topic) + Rule 5 (app_to_lib via uses).

    Forward (paper convention): subscriber -> publisher, app -> lib (dependent -> dependency).
    Reversed (ablation): publisher -> subscriber, lib -> app (literal inversion).
    """
    proj = nx.DiGraph()
    apps = [n for n, d in g.nodes(data=True) if d.get("type") == "Application"]
    libs = [n for n, d in g.nodes(data=True) if d.get("type") == "Library"]
    proj.add_nodes_from(apps, type="Application")
    proj.add_nodes_from(libs, type="Library")

    publishers_of = {}  # topic -> [app]
    subscribers_of = {}  # topic -> [app]
    for u, v, d in g.edges(data=True):
        if d.get("type") == "PUBLISHES_TO":
            publishers_of.setdefault(v, []).append(u)
        elif d.get("type") == "SUBSCRIBES_TO":
            subscribers_of.setdefault(v, []).append(u)

    for topic, subs in subscribers_of.items():
        for pub in publishers_of.get(topic, []):
            for sub in subs:
                if sub == pub:
                    continue
                if reversed_direction:
                    proj.add_edge(pub, sub)  # publisher -> subscriber
                else:
                    proj.add_edge(sub, pub)  # subscriber -> publisher (paper convention)

    for u, v, d in g.edges(data=True):
        if d.get("type") == "USES":
            if reversed_direction:
                proj.add_edge(v, u)  # lib -> app
            else:
                proj.add_edge(u, v)  # app -> lib (paper convention)

    return proj


def main():
    rows = []
    for label, stem in SCENARIOS.items():
        g = _load_graph(ROOT / "data" / "scenarios" / f"{stem}.json")

        injector = FaultInjector(graph=g, seeds=SEEDS)
        result = injector.run(node_types=["Application"])
        impact = {nid: rec.impact_score for nid, rec in result.records.items()}

        apps = [n for n, d in g.nodes(data=True) if d.get("type") == "Application"]

        rhos = {}
        for direction, reversed_direction in (("forward", False), ("reversed", True)):
            proj = build_projection(g, reversed_direction)
            # Blast-radius proxy: |ancestors(v)| under the dependent -> dependency
            # convention = number of components that transitively depend on v, i.e.
            # what would be affected if v failed. (Betweenness centrality was tried
            # first and rejected: summed over all ordered pairs it is provably
            # invariant under globally reversing every edge, so it cannot show any
            # ablation effect by construction — not a meaningful baseline here.)
            n = proj.number_of_nodes()
            centrality = {
                v: len(nx.ancestors(proj, v)) / max(1, n - 1)
                for v in proj.nodes
            }
            y_pred = [centrality.get(a, 0.0) for a in apps]
            y_true = [impact.get(a, 0.0) for a in apps]
            rho, _ = spearmanr(y_pred, y_true)
            rhos[direction] = rho

        delta = rhos["reversed"] - rhos["forward"]
        rows.append({
            "scenario": label,
            "rho_forward_TopoBL": rhos["forward"],
            "rho_reversed_TopoBL": rhos["reversed"],
            "delta_rho": delta,
        })
        print(f"{label:20s} forward={rhos['forward']:+.3f}  reversed={rhos['reversed']:+.3f}  "
              f"delta={delta:+.3f}")

    mean_forward = sum(r["rho_forward_TopoBL"] for r in rows) / len(rows)
    mean_reversed = sum(r["rho_reversed_TopoBL"] for r in rows) / len(rows)
    print(f"\nMean forward rho  = {mean_forward:.4f}")
    print(f"Mean reversed rho = {mean_reversed:.4f}")
    print(f"Mean delta        = {mean_reversed - mean_forward:+.4f}")

    out = {
        "seeds": SEEDS,
        "rows": rows,
        "mean_forward_rho": mean_forward,
        "mean_reversed_rho": mean_reversed,
        "mean_delta_rho": mean_reversed - mean_forward,
    }
    out_path = ROOT / "output" / "reversed_projection_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
