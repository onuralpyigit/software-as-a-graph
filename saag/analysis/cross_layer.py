"""
Cross-layer insight derivation.

Observations that only become visible by correlating criticality results across
two or more analysis layers. Requires RM levels, so it runs after the Predict
stage has enriched each ``LayerAnalysisResult`` — see
``saag.usecases.multi_layer_analysis.MultiLayerAnalysisUseCase``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from saag.core.criticality import CriticalityLevel

from .models import CrossLayerInsight, LayerAnalysisResult

#: A layer with more than this fraction of CRITICAL components is flagged.
CONCENTRATION_THRESHOLD = 0.30

#: Sort order for insight severities (lower sorts first).
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}


def compute_cross_layer_insights(
    results: Dict[str, LayerAnalysisResult],
) -> List[CrossLayerInsight]:
    """
    Derive insights that span multiple layers.

    Three insight types are produced:

    1. ``compound_critical``   — component is CRITICAL or HIGH in ≥2 distinct layers.
    2. ``systemic_spof``       — component is an articulation point in ≥2 distinct layers.
    3. ``layer_concentration`` — a layer where >30 % of analysed components are CRITICAL.

    Insights are sorted by severity, then by number of affected layers descending.
    """
    insights: List[CrossLayerInsight] = []
    insights += _per_component_insights(results)
    insights += _layer_concentration_insights(results)
    insights.sort(key=lambda i: (_SEVERITY_ORDER.get(i.severity, 9), -len(i.layers_affected)))
    return insights


def _per_component_insights(
    results: Dict[str, LayerAnalysisResult],
) -> List[CrossLayerInsight]:
    """Compound-criticality and systemic-SPOF signals for multi-layer components."""
    # component_id → {layer_name: ComponentQuality}
    comp_by_layer: Dict[str, Dict[str, Any]] = {}
    for layer_name, layer_result in results.items():
        for cq in layer_result.quality.components:
            comp_by_layer.setdefault(cq.id, {})[layer_name] = cq

    insights: List[CrossLayerInsight] = []
    for comp_id, layer_map in comp_by_layer.items():
        if len(layer_map) < 2:
            continue  # only appears in one layer — no cross-layer signal

        high_layers = sorted(
            lname for lname, cq in layer_map.items()
            if cq.levels.overall >= CriticalityLevel.HIGH
        )
        spof_layers = sorted(
            lname for lname, cq in layer_map.items()
            if cq.structural.is_articulation_point
        )

        # Name is identical across layers; take it from any available entry.
        sample_cq = next(iter(layer_map.values()))
        comp_name = getattr(sample_cq.structural, "name", comp_id)

        if len(high_layers) >= 2:
            any_critical = any(
                layer_map[l].levels.overall == CriticalityLevel.CRITICAL for l in high_layers
            )
            severity = "CRITICAL" if any_critical else "HIGH"
            insights.append(CrossLayerInsight(
                component_id=comp_id,
                csc_name=comp_name,
                insight_type="compound_critical",
                layers_affected=high_layers,
                severity=severity,
                description=(
                    f"{comp_name} is classified {severity} in "
                    f"{len(high_layers)} layers ({', '.join(high_layers)}), "
                    "indicating compound risk that spans architectural boundaries."
                ),
            ))

        if len(spof_layers) >= 2:
            insights.append(CrossLayerInsight(
                component_id=comp_id,
                csc_name=comp_name,
                insight_type="systemic_spof",
                layers_affected=spof_layers,
                severity="CRITICAL",
                description=(
                    f"{comp_name} is a structural articulation point in "
                    f"{len(spof_layers)} layers ({', '.join(spof_layers)}). "
                    "Its failure would disconnect subgraphs at multiple architectural levels."
                ),
            ))

    return insights


def _layer_concentration_insights(
    results: Dict[str, LayerAnalysisResult],
) -> List[CrossLayerInsight]:
    """Flag layers where CRITICAL components exceed CONCENTRATION_THRESHOLD."""
    insights: List[CrossLayerInsight] = []
    for layer_name, layer_result in results.items():
        components = layer_result.quality.components
        if not components:
            continue

        critical_count = sum(
            1 for cq in components if cq.levels.overall == CriticalityLevel.CRITICAL
        )
        fraction = critical_count / len(components)
        if fraction <= CONCENTRATION_THRESHOLD:
            continue

        insights.append(CrossLayerInsight(
            component_id="",
            csc_name="",
            insight_type="layer_concentration",
            layers_affected=[layer_name],
            severity="HIGH",
            description=(
                f"Layer '{layer_name}' has {critical_count}/{len(components)} "
                f"({fraction:.0%}) components classified as CRITICAL — "
                "high systemic risk concentration in this architectural tier."
            ),
        ))

    return insights
