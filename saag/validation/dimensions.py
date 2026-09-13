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
    specialist can reach sideways into another dimension's signal.
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


def _max_normalised(d: DimensionInputs, attr: str) -> Dict[str, float]:
    """Max-normalise one structural metric over the validated population.

    `calculate_bottleneck_precision`'s thresholds (BT > 0.60, w_out < 0.30) are
    calibrated for the normalised values `StructuralAnalyzer._compute_rm` scores
    against, not for raw betweenness and raw dependency_weight_out -- raw w_out is
    an unbounded sum of QoS weights, so the conjunction was very nearly
    unsatisfiable and BP read 0.0 by construction. This reproduces the analyzer's
    max-based scaling (`_normalize_max`) over the same population; it cannot reuse
    the analyzer's own table, which is built inside `_compute_rm` and never
    reaches the validation stage.
    """
    values = {
        cid: float(getattr(d.components[cid].structural, attr, 0.0) or 0.0)
        for cid in d.ids if cid in d.components
    }
    peak = max(values.values(), default=0.0)
    if peak <= 0.0:
        return {cid: 0.0 for cid in values}
    return {cid: v / peak for cid, v in values.items()}


def _maintainability_specialists(d: DimensionInputs) -> Dict[str, Any]:
    """COCR@5, weighted-κ coupling tier agreement, and bottleneck precision."""
    return {
        "cocr_5": calculate_cocr_at_k(d.predicted, d.actual, k=5),
        "weighted_kappa_cta": calculate_weighted_kappa_cta(d.predicted, d.actual),
        "bottleneck_precision": calculate_bottleneck_precision(
            _max_normalised(d, "betweenness"),
            _max_normalised(d, "dependency_weight_out"),
            d.actual,
        ),
    }


def _availability_specialists(d: DimensionInputs) -> Dict[str, Any]:
    """SPOF-F1 over the structural articulation points.

    HSRR, DASA and RRI used to be reported here as a hardcoded None. They read
    `qspof`, `ap_c_out`, `ap_c_in` and `bridge_score`, none of which
    `StructuralAnalyzer` ever writes to `StructuralMetrics`, so they were never
    measurable and nothing downstream consumed them; their calculators have been
    removed along with the keys. `is_articulation_point` is populated, so SPOF-F1
    is a real measurement and stays.
    """
    articulation = {
        cid: (1.0 if d.components[cid].structural.is_articulation_point else 0.0)
        for cid in d.ids if cid in d.components
    }
    return {"spof_f1": calculate_spof_f1(articulation, d.actual)["f1"]}


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

#: Simulation impact fields that are scaled once and shared by the specs above:
#: the composite plus the per-dimension/sub-characteristic ground truths.
#: ia_out/ia_in were scaled here for DASA and are no longer read by any
#: specialist; the simulator still emits them and the API still exposes them.
GROUND_TRUTH_FIELDS: Tuple[str, ...] = (
    "composite_impact",
    "reliability_impact",
    "maintainability_impact",
    "availability_impact",
)
