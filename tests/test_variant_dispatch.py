"""
tests/test_variant_dispatch.py — every variant id reaches the branch it names
=============================================================================

Both evaluation CLIs dispatch on the variant id with an if/elif chain. Until the
RQ2 control arms were added, the final branch was a bare ``else``, so *any*
unrecognised id — a typo, or a newly added variant whose branch someone forgot —
fell through to the Heterogeneous Graph Transformer and was trained, scored and
written to ``results/*.json`` under its own name. The run looked completely
normal; only the numbers were wrong.

These tests make the three membership tuples, the argparse ``choices`` list and
the registry agree, so that class of failure cannot come back silently.

Run:
    PYTHONPATH=. pytest tests/test_variant_dispatch.py -v
"""

from __future__ import annotations

import pytest

from saag.evaluation import variant_registry as registry

CLI_MODULES = ("cli.loso_evaluate", "cli.kfold_evaluate")


def _module(name):
    return pytest.importorskip(name, reason="evaluation CLI unavailable")


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_known_variants_partition_the_dispatch(module_name):
    """The three branch tuples are disjoint and cover KNOWN_VARIANTS exactly."""
    mod = _module(module_name)
    groups = (
        set(mod._STRUCTURAL_VARIANTS),
        set(mod._HOMOGENEOUS_VARIANTS),
        set(mod._HGT_VARIANTS),
        # Not every harness wires the non-graph arm; absent is an empty branch,
        # not a missing one.
        set(getattr(mod, "_TABULAR_VARIANTS", ())),
    )
    union = set().union(*groups)
    assert union == set(mod.KNOWN_VARIANTS)
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b), f"{a & b} is in two dispatch branches at once"


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_every_dispatchable_id_is_registered(module_name):
    """A branch for an id the registry does not know would print no label."""
    mod = _module(module_name)
    unknown = [v for v in mod.KNOWN_VARIANTS if v not in registry.VARIANTS]
    assert not unknown, f"dispatchable but unregistered: {unknown}"


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_choices_all_reach_a_branch(module_name):
    """The exact failure this file exists for: a choice with no branch.

    argparse would accept the id, the dispatch would fall through, and the HGT
    result would be written under the other variant's name.
    """
    import argparse
    import sys

    mod = _module(module_name)
    captured = {}
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kwargs):
        if args and args[0] in ("--variant", "--variants"):
            captured[args[0]] = kwargs.get("choices")
        return real_add(self, *args, **kwargs)

    # parse_args() reads sys.argv and takes no parameters, so the flags are
    # captured as the parser is built rather than by invoking it.
    argparse.ArgumentParser.add_argument = spy
    real_argv = sys.argv
    sys.argv = [module_name]
    try:
        try:
            mod.parse_args()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.add_argument = real_add
        sys.argv = real_argv

    assert captured, f"{module_name} exposes no --variant flag"
    for flag, choices in captured.items():
        assert choices, f"{flag} has no choices list to check"
        missing = [c for c in choices if c not in mod.KNOWN_VARIANTS]
        assert not missing, (
            f"{module_name} {flag} accepts {missing}, which no dispatch branch "
            "handles; those runs would silently execute the HGT branch."
        )


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_unknown_variant_raises_before_any_work(module_name):
    """The pre-flight check must fire outside the per-seed try/except.

    The dispatch sits inside a handler that logs and continues, so without a
    pre-flight guard a bad id produces a full set of nan folds and a plausible
    looking artifact rather than an error.
    """
    import inspect

    mod = _module(module_name)
    assert "definitely_not_a_variant" not in mod.KNOWN_VARIANTS
    entry = (
        mod.run_one_fold if module_name.endswith("loso_evaluate")
        else mod.run_one_scenario
    )

    # Fill every required parameter from the signature rather than by hand, so
    # adding a training hyperparameter does not silently turn this into a
    # TypeError test instead of a dispatch test. The values never get used: the
    # guard must fire before any of them is read.
    placeholders = {
        "bundles": [], "bundle": None, "holdout_idx": 0, "k": 5,
        "seeds": [42], "layer": "app", "workdir": None, "mode": "gnn",
        "variant": "definitely_not_a_variant",
    }
    kwargs = {}
    for name, param in inspect.signature(entry).parameters.items():
        if name in placeholders:
            kwargs[name] = placeholders[name]
        elif param.default is inspect.Parameter.empty:
            kwargs[name] = 1 if "epoch" in name or name in ("hidden", "heads", "layers") else 0.1

    with pytest.raises(ValueError, match="unrecognised variant"):
        entry(**kwargs)


def test_control_arms_are_not_manuscript_columns():
    """Controls must stay out of the canonical variant set.

    reproduce/main_table.py's ALL_VARIANTS defines the 7x6x5 matrix the
    manuscript reports. A control leaking into it would add a column to a
    published table.
    """
    main_table = pytest.importorskip("reproduce.main_table")
    controls = {
        v for v, spec in registry.VARIANTS.items() if spec.family == "control"
    }
    assert controls, "no control arms registered"
    assert not (controls & set(main_table.ALL_VARIANTS)), (
        "an RQ2 control arm is in ALL_VARIANTS and would become a table column"
    )
    assert controls <= set(main_table.CONTROL_VARIANTS)


def test_every_control_declares_what_it_controls_for():
    """An arm whose purpose is not recorded cannot be read off the table."""
    for variant_id, spec in registry.VARIANTS.items():
        if spec.family == "control":
            assert spec.control_for, f"{variant_id} declares no control_for"
        else:
            assert spec.control_for is None, (
                f"{variant_id} is not a control but declares control_for"
            )
