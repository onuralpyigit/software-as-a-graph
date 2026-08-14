"""
Layered SQuaRE Quality Model

Declares the four-layer measurement model that criticality is computed through, and
the provenance of each layer. This module is a *declaration*, not an engine: the
computation stays in saag/analysis/. What lives here is the layer contract, the
ISO/IEC 25019 domain priorities, and the derivation that turns them into RM
composite weights.

    Layer 0  Quality Measure Elements (ISO/IEC 25021)          MEASURED
             18 topological QMEs + 5 code QMEs, read off the artifact
                    |  deterministic, definitional
    Layer 1  Internal quality measures (ISO/IEC 25023)         DERIVED
             CQP, CDPot_enh, QSPOF, CouplingRisk_enh, MPCI, BR, ...
                    |  deterministic, AHP-weighted
    Layer 2  External quality attributes (ISO/IEC 25010:2023)  DERIVED, VALIDATED
             R = reliability (hierarchical: fault tolerance + availability)
             M = maintainability (modularity + modifiability)
                    |  per-attribute oracles: IR / IM
    Layer 3  Quality-in-use weighting (ISO/IEC 25019:2023)     DECLARED
             omega_domain -> effective RM weights

Layer 2 is the only layer with an oracle, and is where validation attaches
(saag/validation/dimensions.py DIMENSION_SPECS). Layer 3 is NOT a further
prediction stage; see the QIU_PROJECTION docstring for why it cannot be.

This is a two-characteristic model (Reliability, Maintainability), aligned with
ISO/IEC 25010:2023: Availability is scored as a sub-characteristic of Reliability
(saag.analysis.weight_calculator.QualityWeights.r_alpha), not as a peer dimension,
and Vulnerability/Security is not modelled here — see docs/criticality.md §4.3 for
the retirement rationale. The migration from the prior four-dimension RMAV model
was a re-parameterisation of the composite, not a re-derivation: at exact (unrounded)
values, w_R*r_alpha = 0.24/0.84, w_R*(1-r_alpha) = 0.43/0.84, w_M = 0.17/0.84 —
identical to the old Q(v) with the Vulnerability term dropped and renormalised.

Definitions D1-D4 and the full construct are in docs/criticality.md.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Dict, Optional, Tuple

from saag.analysis.weight_calculator import QualityWeights


class Provenance(Enum):
    """How a quantity in this model came to exist.

    The distinction is load-bearing: a DECLARED quantity must never be reported
    as though it were MEASURED. Serialised output that carries a Layer 3 result
    carries this tag with it.
    """

    #: Read directly off the artifact; no modelling step.
    MEASURED = "measured"
    #: Computed from measured inputs by a stated, deterministic formula.
    DERIVED = "derived"
    #: Asserted as a design judgement. Not fitted to, or validated against, any
    #: observation available in this project.
    DECLARED = "declared"


#: RM dimension order used by every vector in this module.
RM_ORDER: Tuple[str, str] = ("reliability", "maintainability")

#: ISO/IEC 25019:2023 quality-in-use characteristic order.
QIU_ORDER: Tuple[str, str, str] = ("beneficialness", "freedom_from_risk", "acceptability")

#: Projection from the RM external-quality vector onto ISO/IEC 25019:2023
#: quality-in-use harm. Rows are [Beneficialness, Freedom from risk,
#: Acceptability]; columns are [R, M]. Documented in docs/criticality.md §3.5.
#:
#: **This matrix cannot add ranking information, and that is a property of its
#: shape rather than of its coefficients.** Every row sums to 1.0, so for any
#: domain weights omega over the three characteristics::
#:
#:     Q_qiu = omega . (M s) = (M^T omega) . s        and   sum(M^T omega) = sum(omega) = 1
#:
#: A quality-in-use scalarisation is therefore *identical* to scoring the same
#: RM vector under a different composite weighting. It follows that:
#:
#: - There is no such thing as a "quality-in-use score" that ranks components
#:   differently from some RM weighting. Do not compute one and report it as an
#:   independent quantity.
#: - The matrix's real content is that it *derives* a weight vector from stated
#:   stakeholder priorities, which is what ``derive_rm_weights`` does with it.
#:
#: The coefficients themselves are DECLARED: nothing in this project observes
#: quality-in-use, so nothing here could have fitted them.
#:
#: Migration note (from the prior 4-D RMAV model, R/M/A/V columns): mechanically
#: folding A into R and dropping V — (0.35+0.40, 0.25), (0.10+0.40, 0.00),
#: (0.30+0.50, 0.00) renormalised — collapses rows 2 and 3 to identical values
#: (1.00, 0.00), making the matrix rank-1 in omega: the 3-D domain-priority
#: vector would reduce to a single degree of freedom, and domains with equal
#: Beneficialness but opposite risk/acceptability profiles (e.g. healthcare vs.
#: enterprise, both 0.40) would tie exactly. Row 1 needed no change — V's
#: coefficient there was already 0.00. Rows 2 and 3 are re-DECLARED below:
#: Maintainability's MTTR channel feeds freedom-from-risk (slow repair
#: prolongs hazard exposure) and its evolvability channel feeds acceptability
#: (perceived trust). This keeps the matrix rank-2 and every domain distinct;
#: see docs/criticality.md §3.5 for the worked comparison.
QIU_PROJECTION: Tuple[Tuple[float, float], ...] = (
    (0.75, 0.25),  # Beneficialness      <- R, M (mechanical fold of A into R)
    (0.80, 0.20),  # Freedom from risk   (re-declared)
    (0.60, 0.40),  # Acceptability       (re-declared)
)

#: Domain context vectors omega = [Beneficialness, Freedom from risk, Acceptability].
#:
#: docs/criticality.md §3.3 states these as *ordinal* constraints per deployment
#: domain (e.g. "omega_Risk >> omega_Ben" for ROS 2 / autonomous vehicle) rather
#: than as numbers. The values below are the smallest set of numeric assignments
#: that satisfy those stated orderings and sum to 1.0; they are DECLARED, and the
#: ordering — not the specific magnitude — is what the source table asserts.
#:
#: Keys are the values that actually appear in scenario ``metadata.domain``.
#: A domain absent from this table falls back to the static default weighting
#: rather than to an invented vector (see ``derive_rm_weights``).
DOMAIN_PRIORITIES: Dict[str, Tuple[float, float, float]] = {
    # Freedom from risk (health & life) dominates: a perception-path failure is
    # a life-safety hazard.  omega_Risk >> omega_Ben
    "autoware_ros2": (0.20, 0.65, 0.15),
    "av": (0.20, 0.65, 0.15),
    # Freedom from risk ~ Beneficialness > Acceptability
    "healthcare": (0.40, 0.45, 0.15),
    # Freedom from risk (economic) > Beneficialness
    "finance": (0.35, 0.50, 0.15),
    # Beneficialness (effectiveness) > Freedom from risk
    "air_traffic_management": (0.50, 0.30, 0.20),
    # Beneficialness (efficiency) > Acceptability
    "iot": (0.50, 0.20, 0.30),
    # Beneficialness ~ Acceptability
    "enterprise": (0.40, 0.20, 0.40),
    "microservices": (0.40, 0.20, 0.40),
    "cloud_microservices": (0.40, 0.20, 0.40),
    "trainticket_microservices": (0.40, 0.20, 0.40),
    # NOTE: 'hub-and-spoke' is deliberately absent. It names a topology class,
    # not a deployment domain, so it has no stakeholder priority and takes the
    # documented fallback.
}


class Layer(Enum):
    """The four layers, in computation order."""

    QME = 0
    INTERNAL = 1
    EXTERNAL = 2
    QIU_WEIGHTING = 3


#: Per-layer epistemic status, queryable rather than only prose. ``oracle`` names
#: the ground truth that validates the layer, or None where none exists.
LAYER_SPEC: Dict[Layer, Dict[str, object]] = {
    Layer.QME: {
        "standard": "ISO/IEC 25021",
        "provenance": Provenance.MEASURED,
        "oracle": None,
        "summary": "Quality measure elements read directly off the artifact.",
    },
    Layer.INTERNAL: {
        "standard": "ISO/IEC 25023 (internal measures)",
        "provenance": Provenance.DERIVED,
        "oracle": None,
        "summary": "Normalised composites over QMEs (CQP, CDPot_enh, QSPOF, ...).",
    },
    Layer.EXTERNAL: {
        "standard": "ISO/IEC 25010:2023",
        "provenance": Provenance.DERIVED,
        "oracle": "IR(v) / IM(v)",
        "summary": (
            "RM as estimates of two characteristics: Reliability (hierarchical: "
            "fault tolerance + availability sub-characteristics) and "
            "Maintainability (modularity+modifiability). "
            "The only layer with an oracle."
        ),
    },
    Layer.QIU_WEIGHTING: {
        "standard": "ISO/IEC 25019:2023",
        "provenance": Provenance.DECLARED,
        "oracle": None,
        "summary": (
            "Domain stakeholder priorities, projected onto an effective RM "
            "weighting. Not a prediction stage — see QIU_PROJECTION."
        ),
    },
}


def qiu_harm(scores: Tuple[float, float]) -> Dict[str, float]:
    """Project an RM vector onto the three quality-in-use harm characteristics.

    Provided for reporting and for the equivalence test that pins the collapse
    property. This is DECLARED output: it re-expresses an RM profile in
    stakeholder vocabulary and carries no information the profile did not.

    Args:
        scores: (R, M), each in [0, 1].

    Returns:
        Mapping over QIU_ORDER.
    """
    return {
        name: sum(row[i] * scores[i] for i in range(2))
        for name, row in zip(QIU_ORDER, QIU_PROJECTION)
    }


def effective_rm_weights(omega: Tuple[float, float, float]) -> Tuple[float, float]:
    """Collapse domain priorities onto RM composite weights: ``M^T omega``.

    This is the whole content of Layer 3. Scoring with these weights is
    *identical* to projecting onto quality-in-use harm and scalarising with
    ``omega`` — see QIU_PROJECTION.

    Args:
        omega: (Beneficialness, Freedom from risk, Acceptability), summing to 1.0.

    Returns:
        (q_reliability, q_maintainability).
    """
    return tuple(  # type: ignore[return-value]
        sum(QIU_PROJECTION[c][i] * omega[c] for c in range(3)) for i in range(2)
    )


def derive_rm_weights(
    domain: Optional[str],
    base: Optional[QualityWeights] = None,
) -> Tuple[QualityWeights, bool]:
    """Derive composite RM weights from a deployment domain's stakeholder priorities.

    Replaces the AHP weight judgement with a derivation traceable to declared
    domain priorities. The dimension-internal weights (ft_*, m_*, a_*) and
    r_alpha (the FT-vs-A split within Reliability) are untouched — only the two
    composite q_* weights change.

    Args:
        domain: A value from scenario ``metadata.domain``. None or unknown takes
            the fallback.
        base: Weights to derive from. Defaults to a fresh ``QualityWeights``.

    Returns:
        ``(weights, derived)`` — ``derived`` is False when the fallback was taken,
        in which case ``weights`` carries the unchanged static defaults.
    """
    base = base or QualityWeights()
    omega = DOMAIN_PRIORITIES.get((domain or "").strip().lower())
    if omega is None:
        return base, False

    q_r, q_m = effective_rm_weights(omega)
    return (
        replace(
            base,
            q_reliability=q_r,
            q_maintainability=q_m,
        ),
        True,
    )


#: Weight of Fault Tolerance within Reliability (R = r_alpha*FT + (1-r_alpha)*A).
#: Mirrors QualityWeights.r_alpha's default — see that field's docstring for
#: provenance ("algebraic aggregation of the retired 4-D AHP composite,
#: 0.24/0.67"). Kept here, not duplicated as a separate literal, so any code
#: that needs the constant without instantiating QualityWeights has one place
#: to import it from.
RELIABILITY_ALPHA: float = QualityWeights().r_alpha
