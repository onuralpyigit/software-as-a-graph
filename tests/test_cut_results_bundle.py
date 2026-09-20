"""
tests/test_cut_results_bundle.py
────────────────────────────────
Tests for reproduce/cut_results_bundle.py, the release-bundle assembler.

The bundle it replaces was assembled by hand and shipped
``detection_validation.json`` as a *crashed run*: valid JSON, every scenario
errored, ``n_scenarios_measured`` 0 and every summary metric ``null``. A copy
loop cannot tell that from a real artifact, so the guard that can is the thing
worth pinning.
"""

import json

import pytest

from reproduce.cut_results_bundle import EXTRA_ARTIFACTS, RENDERED, _empty_reason
from reproduce.reconcile_manuscript import FRESHNESS_TARGETS


def test_empty_reason_catches_zero_measured_scenarios():
    """The exact shape that shipped: a summary whose scenario count is zero."""
    crashed = {"summary": {"n_scenarios_measured": 0, "q_composite": {"mean_spearman_rho": None}}}
    assert _empty_reason(crashed) == "summary.n_scenarios_measured == 0"


def test_empty_reason_catches_all_scenarios_errored():
    """A run whose every row is an exception trace measured nothing."""
    crashed = {"per_scenario": [{"scenario": "av_system", "error": "TypeError: ..."},
                                {"scenario": "iot_smart_city_system", "error": "TypeError: ..."}]}
    reason = _empty_reason(crashed)
    assert reason is not None and "every per_scenario entry errored" in reason


def test_empty_reason_passes_a_partial_run():
    """One failed scenario out of several is a gap, not an empty artifact.

    Rejecting these would make the guard unusable: scenarios drop out for
    legitimate reasons (insufficient overlap) and the summary says so.
    """
    partial = {"summary": {"n_scenarios_measured": 7},
               "per_scenario": [{"scenario": "a", "error": "boom"}, {"scenario": "b", "n_scored": 40}]}
    assert _empty_reason(partial) is None


@pytest.mark.parametrize("payload", [None, [], "text", {}, {"summary": {}}])
def test_empty_reason_is_quiet_on_other_shapes(payload):
    """Most artifacts carry neither key; the guard must not invent a failure."""
    assert _empty_reason(payload) is None


def test_bundle_ships_every_artifact_the_reconciler_consumes():
    """The bundle's contents are derived, not mirrored.

    The last bundle omitted ten artifacts the manuscript cites. Deriving the
    list from the reconciler's registry is what prevents that, so a hand-written
    copy of it reappearing here should fail this test rather than pass silently.
    """
    shipped = set(FRESHNESS_TARGETS) | set(EXTRA_ARTIFACTS)
    assert "detection_validation_v3.json" in shipped, "§7.3's backing artifact"
    assert "main_table.json" in shipped, "Tables 3/5's backing artifact"
    assert not shipped & set(RENDERED), "rendered files are not artifacts"
