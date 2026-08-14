"""
Per-Dimension Validation Specs

Each quality dimension is validated the same way: correlate the topology-derived
predictor against the simulation-derived ground truth for that dimension, then
compute a handful of dimension-specific "specialist" metrics.

Only the specialist metrics differ between dimensions, so the shared procedure
lives in ``ValidationService._validate_dimension`` and the differences are
declared here as data.
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from .metric_calculator import (
    calculate_bottleneck_precision,
    calculate_ccr_at_k, calculate_cme, calculate_cocr_at_k,
    calculate_spof_f1,
    calculate_weighted_kappa_cta,
)
from .models import ValidationTargets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DimensionInputs:
    """Everything a specialist function may need, uniform across dimensions.

    ``predicted``/``actual`` are restricted to this dimension's common ids, while
    ``predictions``/``ground_truths`` carry every dimension's dicts so a
    specialist can reach sideways (availability's DASA needs the directional
    ia_out/ia_in truths).
    """
    ids: List[str]                              # sorted common ids, len >= 3
    predicted: Dict[str, float]                 # {id: dimension score}
    actual: Dict[str, float]                    # {id: scaled dimension ground truth}
    components: Dict[str, Any]                  # {id: quality component}
    predictions: Dict[str, Dict[str, float]]    # keyed by dimension key
    ground_truths: Dict[str, Dict[str, float]]  # keyed by impact attribute name
    sim_results: List[Any]
    targets: ValidationTargets


@dataclass(frozen=True)
class DimensionSpec:
    """Declares how one quality dimension is validated."""
    key: str                  # dimension name, also the output dict key
    score_attr: str           # component.scores.<score_attr>
    impact_attr: str          # failure_result.impact.<impact_attr>
    ground_truth_label: str   # symbol used in the report, e.g. "IR(v)"
    specialists: Callable[[DimensionInputs], Dict[str, float]]


@dataclass(frozen=True)
class DimensionResult:
    """Outcome of validating one dimension."""
    spearman: float
    entry: Dict[str, Any]                              # -> dimensional_validation[key]
    scatter: List[Tuple[str, float, float, str]]       # -> dimensional_scatter[key]
    ci: Tuple[float, float]                            # -> confidence_intervals[key]


#: Structural metrics the availability specialists want that `StructuralMetrics`
#: does not currently produce — see the note in `_availability_specialists`.
#: `_structural` below is unused while that holds; both are kept so the fix
#: (populate these four fields, then call `_structural` + calculate_hsrr/
#: calculate_dasa/calculate_rri from metric_calculator again) is a small diff.
UNPOPULATED_STRUCTURAL_METRICS = ("ap_c_out", "ap_c_in", "qspof", "bridge_score")


def _structural(d: DimensionInputs, attr: str) -> Dict[str, float]:
    """Read a structural metric for every id, defaulting to 0.0 when absent."""
    return {
        cid: float(getattr(d.components[cid].structural, attr, 0.0) or 0.0)
        for cid in d.ids
        if cid in d.components
    }


def _reliability_specialists(d: DimensionInputs) -> Dict[str, float]:
    """CCR@5 (cascade capture) and CME (cascade magnitude error).

    Also emits the false-alarm diagnostic: components predicted highly
    reliability-critical that the cascade simulation ranks in its bottom half.
    """
    top_predicted = {c for c, _ in sorted(d.predicted.items(), key=lambda x: -x[1])[:5]}
    by_impact = sorted(d.actual.items(), key=lambda x: -x[1])
    bottom_half = {c for c, _ in by_impact[max(1, len(by_impact) // 2):]}
    false_alarms = top_predicted & bottom_half
    if false_alarms:
        logger.warning(
            "Reliability false alarms (HIGH R(v) but LOW IR(v)): %s", sorted(false_alarms)
        )

    return {
        "ccr_5": calculate_ccr_at_k(d.predicted, d.actual, k=5),
        "cme": calculate_cme(d.predicted, d.actual),
    }


def _maintainability_specialists(d: DimensionInputs) -> Dict[str, float]:
    """COCR@5, weighted-κ coupling tier agreement, and bottleneck precision."""
    betweenness = {
        cid: d.components[cid].structural.betweenness for cid in d.ids if cid in d.components
    }
    weight_out = {
        cid: d.components[cid].structural.dependency_weight_out
        for cid in d.ids if cid in d.components
    }
    return {
        "cocr_5": calculate_cocr_at_k(d.predicted, d.actual, k=5),
        "weighted_kappa_cta": calculate_weighted_kappa_cta(d.predicted, d.actual),
        "bottleneck_precision": calculate_bottleneck_precision(betweenness, weight_out, d.actual),
    }


def _availability_specialists(d: DimensionInputs) -> Dict[str, Any]:
    """SPOF-F1 (with its precision/recall), HSRR, DASA and RRI.

    Only SPOF-F1 is fully backed by the current analyzer: it uses
    `is_articulation_point`, which `StructuralMetrics` does populate. HSRR, DASA
    and RRI read the metrics named in `UNPOPULATED_STRUCTURAL_METRICS`
    (`qspof`, `ap_c_out`, `ap_c_in`, `bridge_score`) — none of these are fields
    `StructuralAnalyzer` ever writes to `StructuralMetrics`, so every predictor
    value the specialists would see is permanently 0.0 by omission, not by
    measurement. That does not evaluate to "no hidden SPOFs" or "no asymmetry";
    it evaluates to no measurement at all. Report None rather than a computed
    float, so "not yet measured" cannot be mistaken for "measured and
    near-zero". Flip this back to computing real values once the analyzer
    emits those four fields.
    """
    articulation = {
        cid: (1.0 if d.components[cid].structural.is_articulation_point else 0.0)
        for cid in d.ids if cid in d.components
    }
    spof = calculate_spof_f1(articulation, d.actual)
    return {
        "spof_f1": spof["f1"],
        "spof_precision": spof["precision"],
        "spof_recall": spof["recall"],
        "hsrr": None,
        "dasa": None,
        "rri": None,
    }


#: Validated in this order; ``key`` is a published output key (read by
#: saag/visualization/collector.py), so renaming one is a breaking change.
#: These are the two RM composite dimensions. Availability is a Reliability
#: sub-characteristic, not an independently-scored dimension, but it is still
#: worth validating on its own terms — see SUBCHARACTERISTIC_SPECS below.
DIMENSION_SPECS: Tuple[DimensionSpec, ...] = (
    DimensionSpec("reliability", "reliability", "reliability_impact", "IR(v)", _reliability_specialists),
    DimensionSpec("maintainability", "maintainability", "maintainability_impact", "IM(v)", _maintainability_specialists),
)

#: Sub-characteristic diagnostics: validated the same way as DIMENSION_SPECS
#: (correlate predictor vs. ground truth, run specialists) but reported
#: separately and excluded from the composite gates (I*, predictive_gain,
#: orthogonality) — including them there would double-count Reliability's
#: signal, since availability_impact also feeds reliability_impact's blend.
SUBCHARACTERISTIC_SPECS: Tuple[DimensionSpec, ...] = (
    DimensionSpec("availability", "availability", "availability_impact", "IA(v)", _availability_specialists),
)

#: Simulation impact fields that are scaled once and shared by the specs above.
#: The first three are the per-dimension/sub-characteristic ground truths; the
#: rest are auxiliary signals that individual specialists read.
GROUND_TRUTH_FIELDS: Tuple[str, ...] = (
    "composite_impact",
    "reliability_impact",
    "maintainability_impact",
    "availability_impact",
    "ia_out",
    "ia_in",
)
