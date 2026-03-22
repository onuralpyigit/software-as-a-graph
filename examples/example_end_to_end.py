"""
End-to-End Pipeline: Software-as-a-Graph
=========================================
Runs the complete 6-step analysis pipeline in a single script:

  Step 1  Generate   — Synthetic pub-sub topology (scale='tiny', reproducible seed)
  Step 2  Import     — Load into Neo4j (clears previous data)
  Step 3  Analyze    — RMAV quality scoring (R, M, A, V, Q) per component
  Step 4  Simulate   — Exhaustive failure simulation → I(v), IR(v), IM(v), IA(v), IV(v)
  Step 5  Validate   — Statistical comparison (Spearman, F1, per-dimension metrics)
  Step 6  Predict    — Train GNN criticality prediction model
  Step 7  Visualize  — Static HTML dashboard (self-contained research artefact)

Prerequisites:
  • Python 3.9+ virtual environment with requirements.txt installed
  • Neo4j 5.x running at bolt://localhost:7687  (user=neo4j, password=password)

Run from the project root:
    python examples/example_end_to_end.py

To use a different scale:
    python examples/example_end_to_end.py --scale small --layer app --seed 77
"""
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from tools.generation import generate_graph, SCALE_PRESETS
from src.core import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.visualization import VisualizationService
from src.prediction import GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict, extract_simulation_dict


# ──────────────────────────────────────────────
# CLI args
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Software-as-a-Graph — end-to-end pipeline")
    p.add_argument("--scale", default="tiny",
                   choices=list(SCALE_PRESETS.keys()),
                   help="Graph size preset (default: tiny)")
    p.add_argument("--layer", default="app",
                   choices=["app", "infra", "mw", "system"],
                   help="Primary analysis/validation layer (default: app)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--neo4j-uri",     default="bolt://localhost:7687")
    p.add_argument("--neo4j-user",    default="neo4j")
    p.add_argument("--neo4j-password",default="password")
    p.add_argument("--output-dir",    default=str(ROOT / "output"),
                   help="Directory for all output files")
    p.add_argument("--skip-viz", action="store_true",
                   help="Skip dashboard generation (faster)")
    return p.parse_args()


# ──────────────────────────────────────────────
# Pretty printing helpers
# ──────────────────────────────────────────────

STEP_TIMINGS: dict = {}


def step_header(n: int, title: str) -> None:
    print(f"\n{'━'*65}")
    print(f"  Step {n}/6 — {title}")
    print(f"{'━'*65}")
    STEP_TIMINGS[n] = time.time()


def step_done(n: int) -> None:
    elapsed = time.time() - STEP_TIMINGS.get(n, time.time())
    print(f"  ✓ Done  ({elapsed:.1f}s)")


def print_kv(key: str, value, width: int = 26) -> None:
    print(f"  {key:<{width}}: {value}")


def pass_fail(value: float, threshold: float) -> str:
    return "✓ PASS" if value >= threshold else "✗ FAIL"


# ──────────────────────────────────────────────
# Step functions
# ──────────────────────────────────────────────

def step1_generate(scale: str, seed: int, output_dir: Path) -> Path:
    """Generate a synthetic graph and save to JSON."""
    step_header(1, "Generate Synthetic Topology")

    graph_data = generate_graph(scale=scale, seed=seed)

    apps    = graph_data.get("applications", [])
    brokers = graph_data.get("brokers", [])
    topics  = graph_data.get("topics", [])
    nodes   = graph_data.get("nodes", [])

    print_kv("Scale preset",         scale)
    print_kv("Seed",                  seed)
    print_kv("Applications",          len(apps))
    print_kv("Brokers",               len(brokers))
    print_kv("Topics",                len(topics))
    print_kv("Infrastructure nodes",  len(nodes))

    qos_dist: dict = {}
    for t in topics:
        qos = t.get("qos_reliability", "?")
        qos_dist[qos] = qos_dist.get(qos, 0) + 1
    print_kv("Topic QoS distribution", qos_dist)

    out_path = output_dir / f"e2e_graph_{scale}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    print_kv("Saved JSON", out_path.name)

    step_done(1)
    return out_path


def step2_import(json_path: Path, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """Import graph JSON into Neo4j, return an open repository handle."""
    step_header(2, "Import into Neo4j")

    with open(json_path) as f:
        data = json.load(f)

    print_kv("Neo4j URI", neo4j_uri)
    print_kv("Graph file", json_path.name)

    repo = create_repository(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    repo.save_graph(data, clear=True)

    stats = repo.get_statistics()
    print_kv("Neo4j node count",         stats.get("node_count", "?"))
    print_kv("Relationship count",       stats.get("relationship_count", stats.get("total_count", "?")))

    step_done(2)
    return repo


def step3_analyze(repo, layer: str) -> dict:
    """Run RMAV quality analysis and return key results."""
    step_header(3, "RMAV Quality Analysis")

    analyzer = AnalysisService(repo, use_ahp=False)
    result   = analyzer.analyze_layer(layer)
    comps    = result.quality.components

    # Criticality distribution
    dist: dict = {}
    for c in comps:
        lv = c.levels.overall.value.upper()
        dist[lv] = dist.get(lv, 0) + 1

    print_kv("Layer",                 layer)
    print_kv("Components analysed",   len(comps))
    print_kv("Criticality dist.",     dist)

    # Top-5 table
    print(f"\n  Top 5 components by overall score Q(v):")
    print(f"  {'ID':<28} {'Type':<14} {'Q':>6} {'R':>6} {'M':>6} {'A':>6} {'V':>6}  Level")
    print(f"  {'-'*88}")
    for c in comps[:5]:
        s = c.scores
        print(
            f"  {c.id[:27]:<28} {c.type[:13]:<14}"
            f" {s.overall:>6.3f} {s.reliability:>6.3f}"
            f" {s.maintainability:>6.3f} {s.availability:>6.3f}"
            f" {s.vulnerability:>6.3f}  {c.levels.overall.value.upper()}"
        )

    # Critical edges
    crit_edges = [e for e in result.quality.edges
                  if e.level.value.upper() in ("CRITICAL", "HIGH")]
    print_kv("\nCritical/High edges",  len(crit_edges))

    step_done(3)
    return {"analysis_result": result, "top_components": comps[:5]}


def step4_simulate(repo, layer: str) -> list:
    """Run exhaustive failure simulation and return results list."""
    step_header(4, "Exhaustive Failure Simulation")

    sim_service = SimulationService(repo)

    print(f"  Simulating all components in layer='{layer}'...")
    print("  (Computes I(v), IR(v), IM(v), IA(v), IV(v) ground truths)")

    results = sim_service.run_failure_simulation_exhaustive(layer=layer)
    ranked  = sorted(results, key=lambda r: r.impact.composite_impact, reverse=True)

    print_kv("\nComponents simulated", len(results))

    # Summary stats
    avg_i   = sum(r.impact.composite_impact for r in results) / max(len(results), 1)
    max_i   = max(r.impact.composite_impact for r in results) if results else 0.0
    spof_n  = sum(1 for r in results if r.impact.fragmentation > 0.01)
    print_kv("Avg composite I(v)",    f"{avg_i:.4f}")
    print_kv("Max composite I(v)",    f"{max_i:.4f}  ({ranked[0].target_id if ranked else '-'})")
    print_kv("Components with SPOF",  spof_n)

    # Top-5 ground truths
    print(f"\n  Top 5 — composite impact I(v) and per-RMAV ground truths:")
    print(f"  {'ID':<28} {'I(v)':>7} {'IR(v)':>7} {'IM(v)':>7} {'IA(v)':>7} {'IV(v)':>7}")
    print(f"  {'-'*68}")
    for r in ranked[:5]:
        im = r.impact
        print(
            f"  {r.target_id[:27]:<28}"
            f" {im.composite_impact:>7.4f}"
            f" {im.reliability_impact:>7.4f}"
            f" {im.maintainability_impact:>7.4f}"
            f" {im.availability_impact:>7.4f}"
            f" {im.vulnerability_impact:>7.4f}"
        )

    step_done(4)
    return results


def step5_validate(repo, layer: str) -> dict:
    """Run statistical validation and return key metric scores."""
    step_header(5, "Statistical Validation")

    analysis   = AnalysisService(repo)
    simulation = SimulationService(repo)
    validator  = ValidationService(analysis, simulation, ndcg_k=10)

    layer_res = validator.validate_single_layer(layer)

    status = "PASS ✓" if layer_res.passed else "FAIL ✗"
    print_kv("Layer",            layer)
    print_kv("Overall status",   status)

    # Overall metrics
    vr = layer_res.validation_result
    if vr:
        ov = vr.overall
        pf_sp = pass_fail(ov.correlation.spearman, 0.75)
        pf_f1 = pass_fail(ov.classification.f1_score, 0.70)
        print(f"\n  Overall (Q vs I):")
        print(f"    Spearman ρ   : {ov.correlation.spearman:>7.4f}   {pf_sp}")
        print(f"    Kendall τ    : {ov.correlation.kendall:>7.4f}   (gap={abs(ov.correlation.spearman - ov.correlation.kendall):.4f})")
        print(f"    F1 Score     : {ov.classification.f1_score:>7.4f}   {pf_f1}")
        print(f"    Top-5 Overlap: {ov.ranking.top_5_overlap:>7.4f}")
        print(f"    RMSE         : {ov.error.rmse:>7.4f}")

    # Per-dimension summary row
    dv = layer_res.dimensional_validation
    if dv:
        print(f"\n  Per-dimension Spearman ρ:")
        for dim in ("reliability", "maintainability", "availability", "vulnerability"):
            if dim in dv:
                sp = dv[dim].get("spearman", 0.0)
                gt = dv[dim].get("ground_truth", "?")
                pf = pass_fail(sp, 0.65)
                print(f"    {dim.capitalize():<18} ρ={sp:>7.4f}  vs {gt:<8}  {pf}")

    # Warnings
    if layer_res.warnings:
        print(f"\n  Warnings ({len(layer_res.warnings)}):")
        for w in layer_res.warnings[:5]:
            print(f"    • {w}")

    step_done(5)

    return {
        "passed":   layer_res.passed,
        "spearman": vr.overall.correlation.spearman if vr else 0.0,
        "f1":       vr.overall.classification.f1_score if vr else 0.0,
    }


def step6_predict(repo, output_dir: Path, layer: str, analysis_summary: dict, sim_results: list) -> dict:
    """Train GNN and predict component criticality."""
    step_header(6, "GNN Criticality Prediction")

    result_analysis = analysis_summary.get("analysis_result")
    if not result_analysis:
        print("  [ERROR] Missing analysis results; skipping GNN prediction.")
        return {}

    structural_dict = extract_structural_metrics_dict(result_analysis.structural)
    rmav_dict       = extract_rmav_scores_dict(result_analysis.quality)
    simulation_dict = extract_simulation_dict(sim_results)

    gnn_service = GNNService(
        hidden_channels=32,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        predict_edges=True,
        checkpoint_dir=str(output_dir / "gnn_checkpoint"),
    )

    print(f"  Training GNN for layer='{layer}'...")
    train_result = gnn_service.train(
        graph=result_analysis.graph,
        structural_metrics=structural_dict,
        simulation_results=simulation_dict,
        rmav_scores=rmav_dict,
        num_epochs=15,
        lr=1e-3,
    )
    
    gnn_service.save()

    print_kv("\nTop Critical Components", min(5, len(train_result.top_critical_nodes(5))))
    for node in train_result.top_critical_nodes(3):
        print(f"    {node.component[:30]:<32} Score: {node.composite_score:.4f}  [{node.criticality_level}]")
        
    step_done(6)
    return {"trained": True}


def step7_visualize(repo, output_dir: Path, layer: str) -> Optional[str]:
    """Generate static HTML dashboard."""
    step_header(7, "Dashboard Visualization")

    analysis   = AnalysisService(repo)
    simulation = SimulationService(repo)
    validation = ValidationService(analysis, simulation)
    viz        = VisualizationService(
        analysis_service=analysis,
        simulation_service=simulation,
        validation_service=validation,
        repository=repo,
    )

    out_file = str(output_dir / "e2e_dashboard.html")
    print_kv("Layers",    f"{layer}, system")
    print_kv("Output",    out_file)

    path = viz.generate_dashboard(
        output_file=out_file,
        layers=[layer, "system"],
        include_network=True,
    )

    import os
    size_kb = os.path.getsize(path) / 1024
    print_kv("File size",  f"{size_kb:.1f} KB")
    print(f"\n  Open in browser:  xdg-open {os.path.abspath(path)}")

    step_done(6)
    return path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.time()

    print("\n" + "━" * 65)
    print("  Software-as-a-Graph — End-to-End Pipeline")
    print("━" * 65)
    print(f"  Scale  : {args.scale}")
    print(f"  Layer  : {args.layer}")
    print(f"  Seed   : {args.seed}")
    print(f"  Neo4j  : {args.neo4j_uri}")
    print(f"  Output : {output_dir}")

    repo = None
    try:
        # ── Steps 1–2 ────────────────────────────────────────────────
        json_path = step1_generate(args.scale, args.seed, output_dir)
        repo      = step2_import(
            json_path,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )

        # ── Steps 3–5 ────────────────────────────────────────────────
        analysis_summary  = step3_analyze(repo, args.layer)
        _sim_results      = step4_simulate(repo, args.layer)
        validation_summary = step5_validate(repo, args.layer)

        # ── Step 6 (Predict) ──────────────────────────────────────────
        _predict_summary  = step6_predict(repo, output_dir, args.layer, analysis_summary, _sim_results)

        # ── Step 7 (optional) ─────────────────────────────────────────
        if not args.skip_viz:
            step7_visualize(repo, output_dir, args.layer)
        else:
            print("\n  [Step 7 skipped — use --skip-viz=False to generate dashboard]")

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if repo is not None:
            repo.close()

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    print(f"\n{'━'*65}")
    print("  Pipeline Complete")
    print(f"{'━'*65}")
    print(f"  Total time   : {elapsed:.1f}s")
    if 'validation_summary' in dir() or 'validation_summary' in locals():
        passed = validation_summary.get("passed", False)
        sp     = validation_summary.get("spearman", 0.0)
        f1     = validation_summary.get("f1", 0.0)
        print(f"  Validation   : {'PASS ✓' if passed else 'FAIL ✗'}")
        print(f"  Spearman ρ   : {sp:.4f}")
        print(f"  F1 Score     : {f1:.4f}")
    print(f"  Output dir   : {output_dir.absolute()}")
    print()


if __name__ == "__main__":
    main()
