"""
Step 3 — Structural & Quality Analysis
=======================================
Demonstrates how to run the full RMAV quality analysis on graph data in Neo4j.

Computes per-component centrality metrics, then applies AHP-weighted formulas to
produce Reliability (R), Maintainability (M), Availability (A), Vulnerability (V)
and Overall (Q) scores.  Components are classified via the Box-Plot method.

Prerequisites:
  • Neo4j running with imported data (run examples/example_import.py first)

Run from the project root:
    python examples/example_analysis.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.core import create_repository
from src.analysis import AnalysisService
from src.explanation import CLIFormatter


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

LEVELS_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_component_table(components, top_n: int = 10) -> None:
    """Print a ranked table of the top-N components by overall score."""
    header = f"  {'Rank':<5} {'ID':<30} {'Type':<15} {'Q':>6} {'R':>6} {'M':>6} {'A':>6} {'V':>6}  {'Level'}"
    print(header)
    print(f"  {'-'*100}")
    for rank, c in enumerate(components[:top_n], 1):
        s = c.scores
        lv = c.levels.overall.value.upper()
        print(
            f"  {rank:<5} {c.id[:29]:<30} {c.type[:14]:<15}"
            f" {s.overall:>6.3f} {s.reliability:>6.3f}"
            f" {s.maintainability:>6.3f} {s.availability:>6.3f}"
            f" {s.vulnerability:>6.3f}  {lv}"
        )


def level_distribution(components) -> dict:
    dist: dict = {lv: 0 for lv in LEVELS_ORDER}
    for c in components:
        lv = c.levels.overall.value.upper()
        if lv in dist:
            dist[lv] += 1
    return dist


def interpret_critical_components(components):
    """Translate RMAV scores into engineering decisions."""
    print("\n  [Decision Support: Critical Component Interpretation]")
    found_critical = False
    for c in components:
        if c.levels.overall.value == "CRITICAL":
            found_critical = True
            reasons = []
            s = c.scores
            # Semantic interpretation of scores
            if s.availability > 0.70:
                reasons.append(f"structural SPOF — removing it disconnects the system")
            if s.reliability > 0.75:
                reasons.append(f"cascade amplifier — failures here broadcast to {c.structural.out_degree_raw} neighbors")
            if s.maintainability > 0.70:
                reasons.append(f"high-churn hub — changes here force {c.structural.in_degree_raw} other components to refactor")
            
            print(f"    • {c.id}: {' + '.join(reasons) if reasons else 'High combined risk score'}")
    
    if not found_critical:
        print("    ✅ No CRITICAL components found. System structural health is good.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    try:
        repo = create_repository()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        print("  Ensure Neo4j is running and import has been done.")
        return

    try:
        # AnalysisService can use AHP-derived weights (use_ahp=True) or defaults
        analyzer = AnalysisService(repo, use_ahp=False)

        # ── 1. Application layer ──────────────────────────────────────
        print_section("Layer: app  (Applications only)")
        app_res = analyzer.analyze_layer("app")
        comps = app_res.quality.components
        print(f"  Components analysed: {len(comps)}")
        dist = level_distribution(comps)
        print(f"  Criticality distribution: {dist}")
        
        # Replaced table with cards
        CLIFormatter.print_critical_report(app_res.quality, limit_top=3)

        # ── 2. System layer (all component types) ─────────────────────
        print_section("Layer: system  (All component types)")
        sys_res = analyzer.analyze_layer("system")
        all_comps = sys_res.quality.components
        print(f"  Components analysed: {len(all_comps)}")
        dist_all = level_distribution(all_comps)
        print(f"  Criticality distribution: {dist_all}")
        
        # Replaced table and interpretation with cards
        CLIFormatter.print_critical_report(sys_res.quality, problems=sys_res.problems, limit_top=5)

        # ── 3. Critical edges ─────────────────────────────────────────
        print_section("Top critical dependencies (edges)")
        crit_edges = [e for e in sys_res.quality.edges
                      if e.level.value.upper() in ("CRITICAL", "HIGH")]
        print(f"  CRITICAL/HIGH edges: {len(crit_edges)}")
        for e in crit_edges[:5]:
            print(
                f"    {e.source[:25]:<26} → {e.target[:25]:<26}"
                f"  Q={e.scores.overall:.3f}  [{e.level.value.upper()}]"
            )

        # ── 4. Problems / anti-patterns detected ──────────────────────
        print_section("Detected problems / anti-patterns")
        if sys_res.problems:
            for p in sys_res.problems[:8]:
                print(f"    [{p.severity.upper():<8}] {p.description}")
        else:
            print("    No problems detected.")

        # ── 5. Weight information ─────────────────────────────────────
        print_section("Active AHP weights")
        w = sys_res.quality.weights
        print(f"  Reliability:     RPR={w.r_reverse_pagerank:.2f}  w_in={w.r_w_in:.2f}  CDPot={w.r_cdpot:.2f}")
        print(f"  Maintainability: BT={w.m_betweenness:.2f}  w_out={w.m_w_out:.2f}  CouplingRisk={w.m_coupling_risk:.2f}  (1-CC)={w.m_clustering:.2f}")
        print(f"  Availability:    QSPOF={w.a_qspof:.2f}  BR={w.a_bridge_ratio:.2f}  AP_c_dir={w.a_ap_c_directed:.2f}  CDI={w.a_cdi:.2f}")
        print(f"  Vulnerability:   REV={w.v_reverse_eigenvector:.2f}  RCL={w.v_reverse_closeness:.2f}  QADS={w.v_qads:.2f}")
        print(f"  Overall:         R={w.q_reliability:.2f}  M={w.q_maintainability:.2f}  A={w.q_availability:.2f}  V={w.q_vulnerability:.2f}")

    finally:
        repo.close()

    print()
    print_section("Done")
    print("  Next step: run  examples/example_simulation.py")


if __name__ == "__main__":
    main()