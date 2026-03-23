"""
Step 4 — Failure & Event Simulation
=====================================
Demonstrates the three simulation modes supported by the framework:

  1. Single-target failure — simulate what happens when one component crashes
  2. Exhaustive failure    — every component is failed in turn (ground truth I(v))
  3. Event simulation     — discrete-event message flow from a publisher

The exhaustive mode also computes the four per-RMAV ground truths:
  IR(v)  Reliability  — cascade propagation dynamics
  IM(v)  Maintainability — change propagation via BFS on G^T
  IA(v)  Availability — QoS-weighted connectivity disruption
  IV(v)  Vulnerability — compromise propagation on G^T

Prerequisites:
  • Neo4j running with imported data (run examples/example_import.py first)

Run from the project root:
    python examples/example_simulation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.core import create_repository
from src.simulation import SimulationService


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_impact_table(results, label: str = "composite", top_n: int = 8) -> None:
    """Print ranked impact table for failure results."""
    ranked = sorted(results, key=lambda r: r.impact.composite_impact, reverse=True)
    print(f"\n  Top {top_n} components by composite impact I(v):")
    print(f"  {'Rank':<5} {'ID':<30} {'Type':<15} {'I(v)':>6} {'Reach':>7} {'Frag':>7} {'Casc':>6} {'Depth':>6}")
    print(f"  {'-'*90}")
    for i, r in enumerate(ranked[:top_n], 1):
        im = r.impact
        print(
            f"  {i:<5} {r.target_id[:29]:<30} {r.target_type[:14]:<15}"
            f" {im.composite_impact:>6.3f} {im.reachability_loss:>7.3f}"
            f" {im.fragmentation:>7.3f} {im.cascade_count:>6} {im.cascade_depth:>6}"
        )


def main():
    try:
        repo = create_repository()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        print("  Ensure Neo4j is running and import has been done.")
        return

    try:
        sim = SimulationService(repo)
        graph_data = repo.get_graph_data()
        all_comps = graph_data.components if hasattr(graph_data, "components") else []

        # ── 1. Single-target failure simulation ───────────────────────
        print_section("Mode 1: Single-target failure simulation")
        apps = [c for c in all_comps if c.component_type == "Application"]
        if apps:
            target = apps[0]
            print(f"  Failing component: {target.id}  (type={target.component_type})")
            result = sim.run_failure_simulation(target.id, layer="app")
            im = result.impact
            print(f"\n  Impact summary:")
            print(f"    Composite I(v)      : {im.composite_impact:.4f}")
            print(f"    Reachability loss   : {im.reachability_loss:.4f}  ({im.reachability_loss*100:.1f}%)")
            print(f"    Fragmentation       : {im.fragmentation:.4f}")
            print(f"    Throughput loss     : {im.throughput_loss:.4f}")
            print(f"    Flow disruption     : {im.flow_disruption:.4f}")
            print(f"    Cascade events      : {im.cascade_count}  (max depth={im.cascade_depth})")
            print(f"    Affected topics     : {im.affected_topics}")
            print(f"\n  Cascaded component IDs: {result.cascaded_failures[:10]}")
        else:
            print("  No Application components found — skipping.")

        # ── 2. Monte Carlo failure simulation ─────────────────────────
        print_section("Mode 2: Monte Carlo failure simulation  (n=50 trials)")
        if apps:
            target = apps[0]
            mc = sim.run_failure_simulation_monte_carlo(target.id, layer="app", n_trials=50)
            print(f"  Target     : {target.id}")
            print(f"  Trials     : {mc.n_trials}")
            print(f"  Mean I(v)  : {mc.mean_impact:.4f}  ± {mc.std_impact:.4f}")
            print(f"  95% CI     : [{mc.ci_95[0]:.4f}, {mc.ci_95[1]:.4f}]")

        # ── 3. Exhaustive failure simulation (ground truth) ───────────
        print_section("Mode 3: Exhaustive failure simulation (ground truth)")
        
        # ── Independence guarantee — the scientific foundation ────────────
        # NOTE: The criticality scores Q(v) you saw in example_analysis.py 
        # were computed purely from topological metrics (centrality, etc.).
        #
        # The impact scores I(v) we are about to compute are derived 
        # PURELY from cascade simulation rules—they have never seen Q(v).
        #
        # If Q(v) and I(v) correlate later in example_validation.py,
        # it proves that topology ALONE is a reliable predictor of 
        # runtime failure impact.
        # ──────────────────────────────────────────────────────────────────
        
        print("  Running exhaustive failure simulation for layer='app'...")
        print("  (This also computes IR(v), IM(v), IA(v), IV(v) ground truths)\n")
        exhaustive = sim.run_failure_simulation_exhaustive(layer="app")
        print(f"  Components simulated: {len(exhaustive)}")

        # Overall impact
        print_impact_table(exhaustive, top_n=8)

        # Per-dimension ground truths
        print(f"\n  Per-RMAV ground truths — top 5:")
        ranked = sorted(exhaustive, key=lambda r: r.impact.composite_impact, reverse=True)
        print(f"  {'ID':<28} {'IR(v)':>7} {'IM(v)':>7} {'IA(v)':>7} {'IV(v)':>7}")
        print(f"  {'-'*60}")
        for r in ranked[:5]:
            im = r.impact
            print(
                f"  {r.target_id[:27]:<28}"
                f" {im.reliability_impact:>7.4f}"
                f" {im.maintainability_impact:>7.4f}"
                f" {im.availability_impact:>7.4f}"
                f" {im.vulnerability_impact:>7.4f}"
            )

        # ── 4. Event simulation ───────────────────────────────────────
        print_section("Mode 4: Event (message-flow) simulation")
        if apps:
            source = apps[0]
            print(f"  Source publisher: {source.id}")
            ev = sim.run_event_simulation(source.id, num_messages=200, duration=10.0)
            m = ev.metrics
            print(f"\n  Messages published   : {m.messages_published}")
            print(f"  Messages delivered   : {m.messages_delivered}  ({m.delivery_rate:.1f}%)")
            print(f"  Messages dropped     : {m.messages_dropped}  ({m.drop_rate:.1f}%)")
            print(f"  Avg latency          : {m.avg_latency*1000:.2f} ms")
            print(f"  p99 latency          : {m.p99_latency*1000:.2f} ms")
            print(f"  Throughput           : {m.throughput:.1f} msg/s")
            print(f"  Reached subscribers  : {len(ev.reached_subscribers)}")
            print(f"  Brokers used         : {len(ev.brokers_used)}")
            print(f"  Successful flows     : {len(ev.successful_flows)}")

    finally:
        repo.close()

    print()
    print_section("Done")
    print("  Next step: run  examples/example_validation.py")


if __name__ == "__main__":
    main()