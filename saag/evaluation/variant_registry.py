"""Single source of truth for predictor-variant names, families and substrates.

Every table, figure and ``--help`` string that prints a variant name resolves it
here. Before this module the display labels lived in five hand-maintained dicts
(``reproduce/render_table.py``, ``render_results_figure.py``,
``render_stratified_figure.py``, ``loso_all_variants.py``,
``kfold_all_variants.py``) which had already drifted apart: the same ``hgl_qos``
was printed as "SaG" on Figure 5 and "HGL-QoS" in Table 3 of the same paper.

Internal ``variant_id`` strings are unchanged — they remain the keys of
``results/*.json`` and of ``output/loso_cache/`` — so nothing here affects a
published number. Only the printed string is owned by this module.

Naming scheme
-------------
Four families, with the substrate made explicit because RQ2's parity argument
rests on it:

    Structural baselines (training-free)   Topo | Topo-QoS | RM
    Homogeneous graph learning (GAT)       GAT  | GAT-QoS  | GAT-N | GAT-N-QoS
    Heterogeneous graph learning (HGT)     HGT  | HGT-QoS
    RQ2 confound controls                  GAT-N-C | GAT-N-QoS-C |
                                           GAT-N-QoS16-C | HGT-QoS-U

The ``-N`` infix marks the native multigraph; its absence marks the derived
Application--Library ``DEPENDS_ON`` projection. ``SaG`` is reserved for the
framework and is never a variant name.

The ``control`` family is deliberately separate rather than folded into the
homogeneous and heterogeneous ones. Its members are not manuscript columns: they
exist to hold one confound at a time constant so Section 7.2's attribution claim
can be tested, and every renderer selects columns by explicit ``include=`` list
or by family, so controls cannot leak into a published table by accident. The
:attr:`Variant.control_for` field records which confound each one isolates.

Harness-dependent substrate
---------------------------
``gl`` and ``gl_qos`` do not denote one substrate. ``reproduce/main_table.py``
runs them on the projection (see its ``substrate =`` line), while
``cli/loso_evaluate.py`` and ``cli/kfold_evaluate.py`` route only the ``topo_*``
variants through the projection and hand ``gl``/``gl_qos`` the native graph.
:func:`resolve` encodes that difference so no caller re-derives it, which is why
:func:`label` takes a ``harness``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

__all__ = [
    "Variant",
    "VARIANTS",
    "FAMILY_ORDER",
    "FAMILY_LABELS",
    "HARNESSES",
    "resolve",
    "label",
    "blurb",
    "order",
    "edge_dim",
    "hidden_for",
    "bidirectional_for",
]


@dataclass(frozen=True)
class Variant:
    """One predictor variant: its identity, where it sits, and what to call it."""

    #: Internal identifier. Also the ``results/*.json`` key — never rename.
    variant_id: str
    #: One of :data:`FAMILY_ORDER`.
    family: str
    #: ``"projection"``, ``"native"``, or ``"none"`` for the training-free scores.
    substrate: str
    #: Edge channel: ``"none"``, ``"weighted"``, ``"scalar"``, or ``"full16"``.
    qos: str
    #: Display label, e.g. ``"GAT-N-QoS"``.
    label: str
    #: One-line description, reused by ``--help`` text and docstrings.
    blurb: str
    #: Model width override. ``None`` means "whatever the harness's ``--hidden``
    #: says", which is what every reported variant uses. Only the RQ2 capacity
    #: controls set it, because matching HGT's parameter budget is the whole
    #: point of those arms.
    hidden_channels: Optional[int] = None
    #: HGT reverse-pass flag. ``None`` defers to ``NodeCriticalityGNN``'s own
    #: default (``True``). Only the directionality control sets it.
    use_bidirectional: Optional[bool] = None
    #: Which Section 7.2 confound this arm controls for, or ``None`` for the
    #: variants the manuscript actually reports as columns.
    control_for: Optional[str] = None


FAMILY_ORDER = ["structural", "homogeneous", "heterogeneous", "control"]

FAMILY_LABELS = {
    "structural": "Structural baselines (training-free)",
    "homogeneous": "Homogeneous graph learning (untyped GAT)",
    "heterogeneous": "Heterogeneous graph learning (typed HGT)",
    "control": "RQ2 confound controls (not manuscript columns)",
}

#: Harnesses that report variants. They differ in the substrate they give
#: ``gl``/``gl_qos`` — see the module docstring and :func:`resolve`.
HARNESSES = ("in_distribution", "loso", "kfold")

_VARIANT_LIST = [
    Variant(
        variant_id="topo_baseline",
        family="structural",
        substrate="projection",
        qos="none",
        label="Topo",
        blurb="unweighted betweenness + articulation scoring on the projection",
    ),
    Variant(
        variant_id="topo_qos",
        family="structural",
        substrate="projection",
        qos="weighted",
        label="Topo-QoS",
        blurb="QoS-weighted topological centrality on the projection",
    ),
    Variant(
        variant_id="topology_rm",
        family="structural",
        substrate="none",
        qos="none",
        label="RM",
        blurb="Reliability-Maintainability composite Q(v); no GNN",
    ),
    Variant(
        variant_id="gl",
        family="homogeneous",
        substrate="projection",
        qos="none",
        label="GAT",
        blurb="unweighted homogeneous GAT on the projection",
    ),
    Variant(
        variant_id="gl_qos",
        family="homogeneous",
        substrate="projection",
        qos="scalar",
        label="GAT-QoS",
        blurb="homogeneous GAT with scalar w(e) on the projection",
    ),
    Variant(
        variant_id="gl_full",
        family="homogeneous",
        substrate="native",
        qos="none",
        label="GAT-N",
        blurb="unweighted homogeneous GAT on the native multigraph",
    ),
    Variant(
        variant_id="gl_full_qos",
        family="homogeneous",
        substrate="native",
        qos="scalar",
        label="GAT-N-QoS",
        blurb="homogeneous GAT with scalar w(e) on the native multigraph",
    ),
    Variant(
        variant_id="hgl",
        family="heterogeneous",
        substrate="native",
        qos="none",
        label="HGT",
        blurb="QoS-masked Heterogeneous Graph Transformer on the native multigraph",
    ),
    Variant(
        variant_id="hgl_qos",
        family="heterogeneous",
        substrate="native",
        qos="full16",
        label="HGT-QoS",
        blurb="HGT with the full 16-D QoS edge encoding on the native multigraph",
    ),
    # ── RQ2 confound controls ────────────────────────────────────────────────
    # Section 7.2 attributed the typed-vs-untyped LOSO margin to "relational
    # typing rather than to substrate, training set, depth, or selection rule".
    # That list omitted three things the comparison did not hold constant, and
    # these arms hold each of them constant one at a time. Parameter counts are
    # measured against the LOSO primary graph (enterprise_system: 5 node types,
    # 10 relation triples), where HGT is 434,620; they are corpus-specific
    # because HGTConv's size depends on the relation set. See
    # tests/test_baselines.py::TestControlArmCapacityParity.
    Variant(
        variant_id="gl_full_cap",
        family="control",
        substrate="native",
        qos="none",
        label="GAT-N-C",
        blurb="GAT-N widened to 296 channels (437,496 params, 1.01x HGT); "
              "capacity control for the QoS-off replication of RQ2",
        hidden_channels=296,
        control_for="capacity",
    ),
    Variant(
        variant_id="gl_full_qos_cap",
        family="control",
        substrate="native",
        qos="scalar",
        label="GAT-N-QoS-C",
        blurb="GAT-N-QoS widened to 296 channels (439,272 params, 1.01x HGT's "
              "434,620, against 28,168 as published); capacity control for RQ2",
        hidden_channels=296,
        control_for="capacity",
    ),
    Variant(
        variant_id="gl_full_qos16_cap",
        family="control",
        substrate="native",
        qos="full16",
        label="GAT-N-QoS16-C",
        blurb="capacity-matched GAT-N reading all 16 edge-feature dims, the "
              "same channel HGT-QoS gets (429,992 params); edge-channel "
              "control for RQ2",
        hidden_channels=288,
        control_for="edge_channel",
    ),
    Variant(
        # qos stays "full16": this *is* a full-QoS HGT. The arm varies
        # directionality alone, and `qos` describes the edge channel.
        variant_id="hgl_qos_uni",
        family="control",
        substrate="native",
        qos="full16",
        label="HGT-QoS-U",
        blurb="HGT-QoS with the reverse HGTConv removed (330,895 params, so "
              "103,725 fewer); directionality control for RQ2",
        use_bidirectional=False,
        control_for="directionality",
    ),
]

VARIANTS: Dict[str, Variant] = {v.variant_id: v for v in _VARIANT_LIST}

#: Variants the LOSO and k-fold harnesses run on the native graph even though
#: their id spells the projection arm. Labelling only — never a data-lookup key.
_NATIVE_ALIASES = {"gl": "gl_full", "gl_qos": "gl_full_qos"}

#: Variants emphasised in LaTeX output (the proposed model).
_LATEX_BOLD = {"hgl_qos"}


def resolve(variant_id: str, harness: str = "in_distribution") -> str:
    """Return the variant whose *semantics* ``variant_id`` carries under ``harness``.

    Identity everywhere except LOSO and k-fold, where ``gl``/``gl_qos`` are run on
    the native multigraph and therefore describe ``gl_full``/``gl_full_qos``.
    Use this for display only; look data up by the original ``variant_id``.
    """
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness {harness!r}; expected one of {HARNESSES}")
    if harness in ("loso", "kfold"):
        return _NATIVE_ALIASES.get(variant_id, variant_id)
    return variant_id


def _lookup(variant_id: str, harness: str) -> Variant:
    resolved = resolve(variant_id, harness)
    try:
        return VARIANTS[resolved]
    except KeyError:
        raise KeyError(
            f"unknown variant {variant_id!r} (resolved to {resolved!r} under {harness!r})"
        ) from None


def label(
    variant_id: str,
    harness: str = "in_distribution",
    latex: bool = False,
) -> str:
    """Display label for ``variant_id`` as reported by ``harness``.

    With ``latex=True`` the label is wrapped in ``\\textsc{}``, or ``\\textbf{}``
    for the proposed model.
    """
    text = _lookup(variant_id, harness).label
    if not latex:
        return text
    macro = "textbf" if resolve(variant_id, harness) in _LATEX_BOLD else "textsc"
    return rf"\{macro}{{{text}}}"


def blurb(variant_id: str, harness: str = "in_distribution") -> str:
    """One-line description of ``variant_id`` as reported by ``harness``."""
    return _lookup(variant_id, harness).blurb


#: ``qos`` field -> the ``edge_dim`` a homogeneous GAT should be built with.
#: Consulted only on the homogeneous branch; HGT reads ``EDGE_FEATURE_DIM``
#: through its own encoder and ignores this. ``"weighted"`` maps to ``None``
#: because it describes the training-free Topo-QoS score, which builds no model.
_EDGE_DIM_BY_QOS = {"none": None, "weighted": None, "scalar": 1, "full16": 16}


def edge_dim(variant_id: str, harness: str = "in_distribution") -> Optional[int]:
    """GATConv ``edge_dim`` for ``variant_id``, or ``None`` for no edge channel.

    ``None`` also selects the ``"homo_unweighted"`` baseline class, so callers
    can branch on this one value instead of re-listing variant ids.
    """
    return _EDGE_DIM_BY_QOS[_lookup(variant_id, harness).qos]


def hidden_for(
    variant_id: str,
    default: int,
    harness: str = "in_distribution",
) -> int:
    """Model width for ``variant_id``, falling back to the harness's ``--hidden``.

    Returns ``default`` unchanged for every variant the manuscript reports, so
    threading this through a harness cannot move a published number. Only the
    capacity controls override it, and only because a matched parameter budget
    is what those arms exist to provide.
    """
    override = _lookup(variant_id, harness).hidden_channels
    return default if override is None else override


def bidirectional_for(variant_id: str, default: bool = True) -> bool:
    """Whether ``variant_id``'s HGT keeps its reverse pass.

    ``True`` for everything except the directionality control. The default is
    stated here as well as in ``NodeCriticalityGNN`` so that a flip in either
    place is visible as a disagreement rather than a silent change of model;
    ``tests/test_gnn_refactor.py`` pins both.
    """
    # Unknown ids (the structural scores) never build an HGT; tolerate them so
    # callers need no membership test before asking.
    variant = VARIANTS.get(resolve(variant_id, "loso"), None)
    if variant is None or variant.use_bidirectional is None:
        return default
    return variant.use_bidirectional


def order(
    family: Optional[str] = None,
    include: Optional[Iterable[str]] = None,
) -> List[str]:
    """Variant ids grouped by family, in :data:`FAMILY_ORDER`.

    ``family`` restricts the result to one family; ``include`` restricts it to a
    given set of ids (silently dropping ids the caller has no data for, which is
    how a renderer asks for "whichever of these columns exist").
    """
    if family is not None and family not in FAMILY_LABELS:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILY_ORDER}")
    wanted = None if include is None else set(include)
    return [
        v.variant_id
        for fam in FAMILY_ORDER
        if family is None or fam == family
        for v in _VARIANT_LIST
        if v.family == fam and (wanted is None or v.variant_id in wanted)
    ]
