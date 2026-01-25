"""
Validation Display Module

Terminal display formatting for validation results.
Reuses Colors and formatting utilities from analysis.display.

Provides:
    - Display functions for validation pipeline results
    - Display functions for layer-specific validation details
    - Status icons and metric coloring utilities
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import PipelineResult, LayerValidationResult
    from .metrics import ValidationTargets

# Reuse shared display utilities from analysis module
from ..analysis.display import (
    Colors,
    colored,
    print_header,
    print_subheader,
)


# =============================================================================
# Validation-Specific Utilities
# =============================================================================

def status_icon(passed: bool) -> str:
    """Get status icon for pass/fail."""
    if passed:
        return colored("✓", Colors.GREEN, bold=True)
    return colored("✗", Colors.RED, bold=True)


def status_text(passed: bool) -> str:
    """Get status text for pass/fail."""
    if passed:
        return colored("PASSED", Colors.GREEN, bold=True)
    return colored("FAILED", Colors.RED, bold=True)


def metric_color(value: float, target: float, higher_better: bool = True) -> str:
    """Get color for metric based on target."""
    if higher_better:
        passed = value >= target
    else:
        passed = value <= target
    
    if passed:
        return Colors.GREEN
    elif higher_better and value >= target * 0.9:
        return Colors.YELLOW
    elif not higher_better and value <= target * 1.1:
        return Colors.YELLOW
    return Colors.RED


# =============================================================================
# Pipeline Result Display
# =============================================================================

def display_pipeline_result(result: "PipelineResult") -> None:
    """Display the full pipeline result."""
    print_header("VALIDATION PIPELINE RESULTS", "═")
    
    print(f"\n  {colored('Timestamp:', Colors.CYAN)}      {result.timestamp}")
    print(f"  {colored('Components:', Colors.CYAN)}     {result.total_components}")
    
    # Overall status
    print(f"\n  {colored('Overall Status:', Colors.WHITE, bold=True)} {status_text(result.all_passed)}")
    print(f"  Layers Validated: {len(result.layers)}")
    print(f"  Layers Passed:    {result.layers_passed}")
    
    # Layer summary table
    print_subheader("Layer Summary")
    
    print(f"\n  {'Layer':<12} {'N':<6} {'Spearman':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Top-5':<10} {'Status':<10}")
    print(f"  {'-' * 78}")
    
    targets = result.targets
    for layer_name, layer_result in result.layers.items():
        spearman_color = metric_color(layer_result.spearman, targets.spearman)
        f1_color = metric_color(layer_result.f1_score, targets.f1_score)
        prec_color = metric_color(layer_result.precision, targets.precision)
        recall_color = metric_color(layer_result.recall, targets.recall)
        top5_color = metric_color(layer_result.top_5_overlap, targets.top_5_overlap)
        
        print(
            f"  {layer_name:<12} {layer_result.matched_components:<6} "
            f"{colored(f'{layer_result.spearman:>8.4f}', spearman_color)}   "
            f"{colored(f'{layer_result.f1_score:>8.4f}', f1_color)}   "
            f"{colored(f'{layer_result.precision:>8.4f}', prec_color)}   "
            f"{colored(f'{layer_result.recall:>8.4f}', recall_color)}   "
            f"{colored(f'{layer_result.top_5_overlap:>8.4f}', top5_color)}   "
            f"{status_text(layer_result.passed)}"
        )
    
    # Validation targets
    print_subheader("Validation Targets")
    print(f"\n  Spearman ρ ≥ {targets.spearman:.2f}  |  F1 ≥ {targets.f1_score:.2f}  |  "
          f"Precision ≥ {targets.precision:.2f}  |  Recall ≥ {targets.recall:.2f}  |  "
          f"Top-5 ≥ {targets.top_5_overlap:.2f}")
    
    # Cross-layer insights
    if result.cross_layer_insights:
        print_subheader("Cross-Layer Insights")
        for insight in result.cross_layer_insights:
            print(f"  • {insight}")
    
    # Detailed layer results
    for layer_name, layer_result in result.layers.items():
        display_layer_result(layer_result, targets)
    
    # Final summary
    print_header("FINAL SUMMARY", "-")
    
    if result.all_passed:
        print(f"\n  {colored('✓ All validation targets met!', Colors.GREEN, bold=True)}")
        print(f"\n  The graph analysis approach accurately predicts component criticality.")
        print(f"  Predicted quality scores correlate strongly with simulated failure impact.")
        print(f"\n  {colored('Conclusion:', Colors.CYAN)} The methodology is validated for production use.")
    else:
        print(f"\n  {colored('✗ Some validation targets not met.', Colors.RED, bold=True)}")
        print(f"\n  Review the metrics above to identify areas for improvement.")
        
        # Specific suggestions
        for layer_name, layer_result in result.layers.items():
            if not layer_result.passed:
                print(f"\n  {colored(f'Suggestions for {layer_name}:', Colors.YELLOW)}")
                
                if layer_result.spearman < targets.spearman:
                    print(f"    • Low Spearman correlation ({layer_result.spearman:.3f} < {targets.spearman})")
                    print(f"      Consider adjusting quality formula weights.")
                
                if layer_result.f1_score < targets.f1_score:
                    print(f"    • Low F1 score ({layer_result.f1_score:.3f} < {targets.f1_score})")
                    print(f"      Review classification thresholds.")
                
                if layer_result.precision < targets.precision:
                    print(f"    • Low precision ({layer_result.precision:.3f} < {targets.precision})")
                    print(f"      Model over-predicts critical components (too many false positives).")
                
                if layer_result.recall < targets.recall:
                    print(f"    • Low recall ({layer_result.recall:.3f} < {targets.recall})")
                    print(f"      Model misses critical components (too many false negatives).")
    
    print()


def display_layer_result(result: "LayerValidationResult", targets: "ValidationTargets") -> None:
    """Display detailed results for a single layer."""
    print_subheader(f"Layer: {result.layer_name} ({result.layer})")
    
    print(f"\n  {colored('Data Alignment:', Colors.CYAN)}")
    print(f"    Predicted Components:  {result.predicted_components}")
    print(f"    Simulated Components:  {result.simulated_components}")
    print(f"    Matched Components:    {result.matched_components}")
    
    if result.warnings:
        print(f"\n  {colored('Warnings:', Colors.YELLOW)}")
        for warning in result.warnings:
            print(f"    ⚠ {warning}")
    
    if not result.validation_result:
        return
    
    overall = result.validation_result.overall
    
    # Correlation metrics
    print(f"\n  {colored('Correlation Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    spearman_passed = overall.correlation.spearman >= targets.spearman
    print(f"    {'Spearman ρ':<15} {overall.correlation.spearman:>8.4f}     ≥{targets.spearman:<8.2f}   {status_icon(spearman_passed)}")
    
    pearson_passed = overall.correlation.pearson >= targets.pearson
    print(f"    {'Pearson r':<15} {overall.correlation.pearson:>8.4f}     ≥{targets.pearson:<8.2f}   {status_icon(pearson_passed)}")
    
    kendall_passed = overall.correlation.kendall >= targets.kendall
    print(f"    {'Kendall τ':<15} {overall.correlation.kendall:>8.4f}     ≥{targets.kendall:<8.2f}   {status_icon(kendall_passed)}")
    
    # Classification metrics
    print(f"\n  {colored('Classification Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    f1_passed = overall.classification.f1_score >= targets.f1_score
    print(f"    {'F1 Score':<15} {overall.classification.f1_score:>8.4f}     ≥{targets.f1_score:<8.2f}   {status_icon(f1_passed)}")
    
    prec_passed = overall.classification.precision >= targets.precision
    print(f"    {'Precision':<15} {overall.classification.precision:>8.4f}     ≥{targets.precision:<8.2f}   {status_icon(prec_passed)}")
    
    recall_passed = overall.classification.recall >= targets.recall
    print(f"    {'Recall':<15} {overall.classification.recall:>8.4f}     ≥{targets.recall:<8.2f}   {status_icon(recall_passed)}")
    
    print(f"    {'Accuracy':<15} {overall.classification.accuracy:>8.4f}")
    
    # Confusion matrix
    cm = overall.classification.confusion_matrix
    print(f"\n    {colored('Confusion Matrix:', Colors.GRAY)}")
    print(f"                            Actual Critical    Actual Non-Critical")
    print(f"      Pred Critical           {colored(str(cm['tp']), Colors.GREEN):>3}                {colored(str(cm['fp']), Colors.RED):>3}")
    print(f"      Pred Non-Critical       {colored(str(cm['fn']), Colors.RED):>3}                {colored(str(cm['tn']), Colors.GREEN):>3}")
    
    # Ranking metrics
    print(f"\n  {colored('Ranking Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    top5_passed = overall.ranking.top_5_overlap >= targets.top_5_overlap
    print(f"    {'Top-5 Overlap':<15} {overall.ranking.top_5_overlap:>8.4f}     ≥{targets.top_5_overlap:<8.2f}   {status_icon(top5_passed)}")
    
    top10_passed = overall.ranking.top_10_overlap >= targets.top_10_overlap
    print(f"    {'Top-10 Overlap':<15} {overall.ranking.top_10_overlap:>8.4f}     ≥{targets.top_10_overlap:<8.2f}   {status_icon(top10_passed)}")
    
    # Top-5 agreement
    if overall.ranking.top_5_predicted:
        # Helpers for name resolution
        names = result.component_names
        def get_name(comp_id: str) -> str:
            return f"{comp_id} ({names.get(comp_id, comp_id)})" if comp_id in names else comp_id

        print(f"\n    {colored('Top-5 Agreement:', Colors.GRAY)}")
        print(f"      Predicted:  {', '.join(get_name(c) for c in overall.ranking.top_5_predicted[:5])}")
        print(f"      Actual:     {', '.join(get_name(c) for c in overall.ranking.top_5_actual[:5])}")
        print(f"      Common:     {', '.join(get_name(c) for c in overall.ranking.top_5_common) or 'None'}")
    
    # Error metrics
    print(f"\n  {colored('Error Metrics:', Colors.CYAN)}")
    rmse_passed = overall.error.rmse <= targets.rmse_max
    print(f"    RMSE: {overall.error.rmse:.4f} {'≤' if rmse_passed else '>'} {targets.rmse_max} {status_icon(rmse_passed)}")
    print(f"    MAE:  {overall.error.mae:.4f}")
    print(f"    Max Error: {overall.error.max_error:.4f}")
    
    # Top mismatches (if available)
    if result.comparisons:
        # Helpers for name resolution
        names = result.component_names
        
        print(f"\n  {colored('Top Mismatches (by error):', Colors.CYAN)}")
        print(f"    {'Component':<35} {'Type':<12} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Class':<6}")
        print(f"    {'-' * 85}")
        
        for comp in result.comparisons[:5]:
            cls_color = Colors.GREEN if comp.classification in ("TP", "TN") else Colors.RED
            c_name = f"{comp.id} ({names.get(comp.id, comp.id)})" if comp.id in names else comp.id
            print(
                f"    {c_name:<35} {comp.type:<12} "
                f"{comp.predicted:>8.4f}   {comp.actual:>8.4f}   "
                f"{comp.error:>8.4f}   {colored(comp.classification, cls_color)}"
            )
    
    # Per-type breakdown
    if result.validation_result.by_type:
        print(f"\n  {colored('Per-Type Breakdown:', Colors.CYAN)}")
        print(f"    {'Type':<15} {'N':<6} {'Spearman':<10} {'F1':<10} {'Passed':<10}")
        print(f"    {'-' * 51}")
        
        for comp_type, type_result in result.validation_result.by_type.items():
            print(
                f"    {comp_type:<15} {type_result.sample_size:<6} "
                f"{type_result.correlation.spearman:>8.4f}   "
                f"{type_result.classification.f1_score:>8.4f}   "
                f"{status_icon(type_result.passed)}"
            )
