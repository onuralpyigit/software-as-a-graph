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
A label is the architecture followed by what distinguishes it:

    Structural baselines (training-free)   Topo | Topo-QoS | RM
    Homogeneous graph learning (GAT)       GAT-S | GAT-S-w           (native)
                                           GAT-S-P | GAT-S-P-w       (projection)
    Heterogeneous graph learning (HGT)     HGT  | HGT-QoS
    Hybrid engines                         Hybrid-HGT | Hybrid-GAT
    RQ2 confound controls                  GAT | GAT-w | GAT-QoS | HGT-QoS-U

    -S     small GAT (28,168 parameters)
    -w     scalar QoS edge weight w(e)
    -QoS   on a GNN, the 16-D QoS edge vector; on Topo, QoS-weighted distances
    -P     the Application--Library DEPENDS_ON projection (default: native graph)
    Hybrid-X   engine X reading the Topo-QoS prior and correcting its logit

Unsuffixed GAT and GAT-QoS are the capacity-matched controls, at HGT's
parameter budget, so the matched 2x2 reads {GAT, HGT} x {-, -QoS}. ``SaG`` is
reserved for the framework and is never a variant name.

The labels changed on 2026-09-24. :data:`LEGACY_LABELS` maps each earlier label
to its current one, and :func:`relabel` applies it. Published artifacts and
PREREGISTRATION.md still carry the earlier labels.

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

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

__all__ = [
    "Variant",
    "VARIANTS",
    "LEGACY_LABELS",
    "relabel",
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
    "node_qos_for",
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
    #: Whether the QoS-derived *node* features are kept. ``None`` means they
    #: follow the edge channel (QoS on iff ``qos != "none"``), which is true of
    #: every arm except the two that decouple them.
    node_qos: Optional[bool] = None


FAMILY_ORDER = ["structural", "tabular", "homogeneous", "heterogeneous", "hybrid", "control"]

FAMILY_LABELS = {
    "structural": "Structural baselines (training-free)",
    "tabular": "Non-graph learned baseline (no message passing)",
    "homogeneous": "Homogeneous graph learning (untyped GAT)",
    "heterogeneous": "Heterogeneous graph learning (typed HGT)",
    "hybrid": "Hybrid engines (a learned engine correcting the closed-form score)",
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
        label="GAT-S-P",
        blurb="unweighted homogeneous GAT on the projection",
    ),
    Variant(
        variant_id="gl_qos",
        family="homogeneous",
        substrate="projection",
        qos="scalar",
        label="GAT-S-P-w",
        blurb="homogeneous GAT with scalar w(e) on the projection",
    ),
    Variant(
        variant_id="gl_full",
        family="homogeneous",
        substrate="native",
        qos="none",
        label="GAT-S",
        blurb="unweighted homogeneous GAT on the native multigraph",
    ),
    Variant(
        variant_id="gl_full_qos",
        family="homogeneous",
        substrate="native",
        qos="scalar",
        label="GAT-S-w",
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
        label="GAT",
        blurb="GAT-N widened to 296 channels (437,496 params, 1.01x HGT); "
              "capacity control for the QoS-off replication of RQ2",
        hidden_channels=296,
        control_for="capacity",
    ),
    Variant(
        variant_id="tab_gbm",
        family="tabular",
        substrate="native",
        qos="none",
        label="GBM-Feat",
        blurb="gradient boosting on the identical typed node features, no "
              "message passing; isolates whether the graph-learning gain is "
              "the aggregation or just the features",
    ),
    Variant(
        variant_id="tab_gbm_qos",
        family="tabular",
        substrate="native",
        qos="none",
        label="GBM-Feat-QoS",
        blurb="GBM-Feat reading the QoS-derived node features GAT-QoS reads; "
              "the non-graph counterpart of GAT-QoS rather than of GAT",
        node_qos=True,
    ),
    Variant(
        variant_id="gl_full_qos_cap",
        family="control",
        substrate="native",
        qos="scalar",
        label="GAT-w",
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
        label="GAT-QoS",
        blurb="capacity-matched GAT-N reading all 16 edge-feature dims, the "
              "same channel HGT-QoS gets (429,992 params); edge-channel "
              "control for RQ2",
        hidden_channels=288,
        control_for="edge_channel",
    ),
    Variant(
        variant_id="gl_full_qos16_nfmask",
        family="control",
        substrate="native",
        qos="full16",
        label="GAT-QoS-nf",
        blurb="GAT-QoS with its QoS-derived node features masked as in GAT and "
              "the 16-D QoS edge channel kept; separates the two QoS inputs",
        hidden_channels=288,
        control_for="node_qos",
        node_qos=False,
    ),
    Variant(
        # PREREGISTRATION.md Amendment 5. HGT-QoS reading the rank-normalised
        # Topo-QoS score as an extra input and learning a correction to its
        # logit. Opt-in: not part of the manuscript's default sweep.
        variant_id="hgl_qos_prior",
        family="hybrid",
        substrate="native",
        qos="full16",
        label="Hybrid-HGT",
        blurb="HGT-QoS learning a residual correction on the closed-form Topo-QoS score",
    ),
    Variant(
        # PREREGISTRATION.md Amendment 6. gl_full_qos16_cap with the same
        # Topo-QoS prior and logit correction as hgl_qos_prior. Opt-in.
        variant_id="gl_qos16_prior",
        family="hybrid",
        substrate="native",
        qos="full16",
        label="Hybrid-GAT",
        blurb="capacity-matched untyped GAT (16-D QoS) learning a residual "
              "correction on the closed-form Topo-QoS score",
        hidden_channels=288,
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

#: Earlier display labels -> current ones. Result artifacts written before the
#: 2026-09-24 relabelling embed the earlier strings (``label``, ``contrast``,
#: ``baseline_label``), as does PREREGISTRATION.md; this is the one place that
#: translates them. The in-distribution ``GAT``/``GAT-QoS`` of the old scheme are
#: deliberately absent: those strings are *current* labels of other variants, so
#: text using them cannot be relabelled mechanically.
LEGACY_LABELS: Dict[str, str] = {
    "GAT-N-QoS16-C": "GAT-QoS",
    "GAT-N-QoS-C": "GAT-w",
    "GAT-N-C": "GAT",
    "GAT-N-QoS": "GAT-S-w",
    "GAT-N": "GAT-S",
    "SaG-Hybrid-GAT": "Hybrid-GAT",
    "SaG-Hybrid": "Hybrid-HGT",
}

_LEGACY_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(k) for k in sorted(LEGACY_LABELS, key=len, reverse=True))
    + r")(?![\w-])"
)


def relabel(text: str) -> str:
    """Rewrite every earlier label inside ``text`` to its current label.

    Longest match first and on token boundaries, so ``GAT-N-QoS16-C`` becomes
    ``GAT-QoS`` rather than ``GAT-S-w16-C``. Current labels pass through unchanged.
    """
    return _LEGACY_RE.sub(lambda m: LEGACY_LABELS[m.group(1)], text)

#: The variant every reported Δρ is measured against, in every harness.
#:
#: It lives here, beside the variant identities, because four places need to
#: agree on it: the LOSO and k-fold comparison tables, the significance tests
#: that license those tables, and the renderer that typesets them. They did not
#: agree. Both harnesses computed "Δρ vs best baseline" as ``max`` over every
#: *other* row — which on the shipped LOSO artifact selected GAT-N-QoS, a
#: learned variant, and printed HGT-QoS's margin as +0.0346 under a heading
#: that says "baseline", with no interval. The pre-registered comparator is
#: Topo-QoS, where the same gap is +0.0851 with a 95% CI that includes zero.
#: A table and the test that licenses it must not quietly use different
#: reference points, and one name is how that is enforced.
PREREGISTERED_BASELINE = "topo_qos"

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


def node_qos_for(variant_id: str, harness: str = "in_distribution") -> bool:
    """Whether ``variant_id`` reads the QoS-derived node features.

    Follows the edge channel unless the variant decouples the two
    (:attr:`Variant.node_qos`).
    """
    variant = _lookup(variant_id, harness)
    if variant.node_qos is not None:
        return variant.node_qos
    return edge_dim(variant_id, harness) is not None


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
