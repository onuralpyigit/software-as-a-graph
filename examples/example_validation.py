"""
Step 5 — Statistical Validation
================================
Demonstrates how to compare RMAV quality predictions against simulation
ground truths for each dimension.

The validation pipeline computes:
  • Overall   — Spearman ρ(Q, I), F1, Precision, Recall, NDCG@K, Top-5/10 overlap, RMSE
  • Reliability      — ρ(R, IR), CCR@5, CME
  • Maintainability  — ρ(M, IM), COCR@5, weighted-κ CTA, Bottleneck Precision
  • Availability     — ρ(A, IA), SPOF_F1, RRI
  • Vulnerability    — ρ(V, IV), AHCR@5, FTR, APAR, CDCC

Prerequisites:
  • Neo4j running with imported data (run examples/example_import.py first)

Run from the project root:
    python examples/example_validation.py

Interpreting the Results:
  • Spearman ρ ≥ 0.80: Excellent. The topology is a primary driver of risk.
  • Spearman ρ < 0.60: The system may have significant logic/QoS overrides
                      that topology alone cannot capture.
  • PASS: The model's predictive accuracy meets the safety-critical threshold.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.adapters import create_repository
from src.analysis import AnalysisService
from src.prediction import PredictionService
from src.simulation import SimulationService
from src.validation import ValidationService


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def pass_fail(value: float, threshold: float) -> str:
    return "✓ PASS" if value >= threshold else "✗ FAIL"


def print_overall_metrics(layer_result) -> None:
    vr = layer_result.validation_result
    if not vr:
        print("    (no validation result)")
        return
    ov = vr.overall
    corr = ov.correlation
    cls  = ov.classification
    rnk  = ov.ranking
    err  = ov.error

    print(f"    Spearman ρ     : {corr.spearman:>7.4f}   p={corr.spearman_p:.4f}  {pass_fail(corr.spearman, 0.75)}")
    print(f"    Kendall τ      : {corr.kendall:>7.4f}  (gap vs Spearman: {abs(corr.spearman - corr.kendall):.4f})")
    print(f"    F1 Score       : {cls.f1_score:>7.4f}  {pass_fail(cls.f1_score, 0.70)}")
    print(f"    Precision      : {cls.precision:>7.4f}")
    print(f"    Recall         : {cls.recall:>7.4f}")
    print(f"    Top-5 Overlap  : {rnk.top_5_overlap:>7.4f}")
    print(f"    Top-10 Overlap : {rnk.top_10_overlap:>7.4f}")
    print(f"    RMSE           : {err.rmse:>7.4f}")
    print(f"    Components     : {vr.predicted_count} predicted  |  {vr.actual_count} simulated  |  {vr.matched_count} matched")


def print_dimensional_metrics(dim_dict: dict) -> None:
    if not dim_dict:
        print("    (no per-dimension validation data)")
        return
    for dim, metrics in sorted(dim_dict.items()):
        gt = metrics.get("ground_truth", "?")
        sp = metrics.get("spearman", 0.0)
        n  = metrics.get("n", 0)
        print(f"\n    [{dim.upper():<16}]  ground truth={gt}  n={n}")
        print(f"      Spearman ρ     : {sp:>7.4f}  {pass_fail(sp, 0.70)}")

        # Reliability-specific
        if "ccr_5" in metrics:
            print(f"      CCR@5          : {metrics['ccr_5']:>7.4f}   (Cascade Capture Rate)")
            print(f"      CME            : {metrics['cme']:>7.4f}   (Cascade Magnitude Error — lower is better)")

        # Maintainability-specific
        if "cocr_5" in metrics:
            print(f"      COCR@5         : {metrics['cocr_5']:>7.4f}   (Change Overlap Capture Rate)")
            print(f"      κ_CTA          : {metrics['weighted_kappa_cta']:>7.4f}   (Weighted Coupling Tier Agreement)")
            print(f"      Bottleneck P   : {metrics['bottleneck_precision']:>7.4f}   (BT+w_out bottleneck precision)")

        # Availability-specific
        if "spof_f1" in metrics:
            print(f"      SPOF F1        : {metrics['spof_f1']:>7.4f}   P={metrics['spof_precision']:.4f}  R={metrics['spof_recall']:.4f}")
            print(f"      RRI            : {metrics['rri']:>7.4f}   (Robustness Rank Improvement)")

        # Vulnerability-specific
        if "ahcr_5" in metrics:
            print(f"      AHCR@5         : {metrics['ahcr_5']:>7.4f}   (Attack-Hit Capture Rate)")
            print(f"      FTR            : {metrics['ftr']:>7.4f}   (False Trust Rate — lower is better)")
            print(f"      APAR           : {metrics['apar']:>7.4f}   (Attack-Path Agreement Rate)")
            print(f"      CDCC           : {metrics['cdcc']:>7.4f}   (Cross-Dim Contamination Check; ~0 = orthogonal)")


def main():
    try:
        repo = create_repository()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        print("  Ensure Neo4j is running and import has been done.")
        return

    try:
        analysis   = AnalysisService(repo)
        simulation = SimulationService(repo)
        # ndcg_k=10 is the default ranking horizon for NDCG; adjust to suit your system size
        validation = ValidationService(analysis, PredictionService(), simulation, ndcg_k=10)

        # ── 1. Validate the application layer ─────────────────────────
        print_section("Validating layer: app")
        app_result = validation.validate_single_layer("app")
        status = "PASS ✓" if app_result.passed else "FAIL ✗"
        print(f"\n  Overall status: {status}")
        print(f"  Components: {app_result.predicted_components} predicted  |  {app_result.simulated_components} simulated")
        print()
        print_overall_metrics(app_result)

        # Per-dimension
        print_section("Per-dimension validation — app layer")
        print_dimensional_metrics(app_result.dimensional_validation)

        # ── 2. Validate the system layer ──────────────────────────────
        print_section("Validating layer: system")
        sys_result = validation.validate_single_layer("system")
        status_s = "PASS ✓" if sys_result.passed else "FAIL ✗"
        print(f"\n  Overall status: {status_s}")
        print_overall_metrics(sys_result)

        print("\n  [Scientific Interpretation]")
        rho = sys_result.overall.correlation.spearman
        if rho >= 0.85:
            print(f"    ✅ High Correlation ({rho:.3f}): The topological analysis is a")
            print("       valid 'digital twin' for predicting failure cascades.")
        elif rho >= 0.70:
            print(f"    ⚠️  Moderate Correlation ({rho:.3f}): Topology is a strong indicator,")
            print("       but non-topological factors (e.g. QoS) also play a role.")
        else:
            print(f"    ✗  Low Correlation ({rho:.3f}): Topology alone is insufficient")
            print("       to predict impact in this specific configuration.")

        # ── 3. Multi-layer pipeline ────────────────────────────────────
        print_section("Multi-layer pipeline: [app, mw, system]")
        pipeline = validation.validate_layers(["app", "mw", "system"])
        print(f"\n  All passed: {'YES ✓' if pipeline.all_passed else 'NO ✗'}")
        print(f"  Layers passed: {pipeline.layers_passed} / {len(pipeline.layers)}")
        print(f"\n  {'Layer':<10} {'Spearman':>9} {'F1':>7} {'Top-5':>7} {'Status'}")
        print(f"  {'-'*50}")
        for lname, lr in pipeline.layers.items():
            sp  = lr.spearman if lr.spearman is not None else 0.0
            f1  = lr.f1_score if lr.f1_score is not None else 0.0
            t5  = lr.top_5_overlap if lr.top_5_overlap is not None else 0.0
            st  = "PASS ✓" if lr.passed else "FAIL ✗"
            print(f"  {lname:<10} {sp:>9.4f} {f1:>7.4f} {t5:>7.4f}  {st}")

        # Warnings
        all_warnings = [w for lr in pipeline.layers.values() for w in (lr.warnings or [])]
        if all_warnings:
            print_section("Validation warnings")
            for w in all_warnings[:10]:
                print(f"  • {w}")

    finally:
        repo.close()

    print()
    print_section("Done")
    print("  Next step: run  examples/example_prediction.py")

if __name__ == "__main__":
    main()