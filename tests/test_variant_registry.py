"""Guards for saag/evaluation/variant_registry.py.

The registry exists because five hand-maintained label dicts drifted apart and
printed two different names for the same variant in one manuscript. These tests
pin the properties that made that possible.
"""

import pytest

from saag.evaluation.variant_registry import (
    FAMILY_LABELS,
    FAMILY_ORDER,
    HARNESSES,
    VARIANTS,
    label,
    order,
    resolve,
)


def _harness_variants():
    """Every variant id any harness can be asked to report on."""
    from reproduce.main_table import ALL_VARIANTS, CONTROL_VARIANTS
    from reproduce.loso_all_variants import ALL_VARIANTS as LOSO_VARIANTS

    return set(ALL_VARIANTS) | set(CONTROL_VARIANTS) | set(LOSO_VARIANTS)


def test_every_harness_variant_is_registered():
    """A variant a harness can emit must have a name to print."""
    missing = _harness_variants() - set(VARIANTS)
    assert not missing, f"unregistered variants: {sorted(missing)}"


@pytest.mark.parametrize("harness", HARNESSES)
def test_labels_are_unique_within_a_harness(harness):
    """Two columns of one table must never print the same name.

    Under LOSO/k-fold gl->gl_full and gl_qos->gl_full_qos, so the aliased pair is
    excluded: only ids a single harness actually reports are compared.
    """
    reported = set(VARIANTS) - ({"gl_full", "gl_full_qos"} if harness != "in_distribution" else set())
    labels = [label(v, harness) for v in sorted(reported)]
    assert len(labels) == len(set(labels)), f"duplicate labels under {harness}: {labels}"


def test_loso_and_kfold_report_gl_on_the_native_substrate():
    """Pins the substrate divergence between the harnesses.

    reproduce/main_table.py runs gl/gl_qos on the DEPENDS_ON projection, while
    cli/loso_evaluate.py and cli/kfold_evaluate.py route only topo_* through the
    projection and hand gl/gl_qos the native graph.
    """
    for harness in ("loso", "kfold"):
        assert resolve("gl", harness) == "gl_full"
        assert resolve("gl_qos", harness) == "gl_full_qos"
        assert label("gl_qos", harness) == "GAT-N-QoS"
        assert VARIANTS[resolve("gl_qos", harness)].substrate == "native"

    assert resolve("gl_qos", "in_distribution") == "gl_qos"
    assert label("gl_qos", "in_distribution") == "GAT-QoS"
    assert VARIANTS["gl_qos"].substrate == "projection"


def test_topo_variants_are_projection_based_in_every_harness():
    """The structural baselines are the one arm no harness disagrees about."""
    for harness in HARNESSES:
        for variant in ("topo_baseline", "topo_qos"):
            assert resolve(variant, harness) == variant
            assert VARIANTS[variant].substrate == "projection"


def test_families_are_declared():
    assert set(FAMILY_LABELS) == set(FAMILY_ORDER)
    for variant in VARIANTS.values():
        assert variant.family in FAMILY_ORDER


def test_order_groups_by_family():
    ids = order()
    assert set(ids) == set(VARIANTS)
    families = [VARIANTS[v].family for v in ids]
    assert families == sorted(families, key=FAMILY_ORDER.index), (
        f"order() interleaved families: {families}"
    )


def test_order_filters_to_available_columns():
    """A renderer asks for 'whichever of these exist' and gets them in order."""
    assert order(include=["hgl_qos", "topo_qos", "gl"]) == ["topo_qos", "gl", "hgl_qos"]
    assert order(family="heterogeneous") == ["hgl", "hgl_qos"]


def test_latex_labels_emphasise_only_the_proposed_model():
    assert label("hgl_qos", latex=True) == r"\textbf{HGT-QoS}"
    assert label("topo_baseline", latex=True) == r"\textsc{Topo}"


def test_unknown_inputs_raise():
    with pytest.raises(ValueError):
        resolve("hgl_qos", "not_a_harness")
    with pytest.raises(KeyError):
        label("no_such_variant")
