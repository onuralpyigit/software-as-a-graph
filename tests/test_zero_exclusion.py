"""
tests/test_zero_exclusion.py — the tied-at-zero sensitivity bound
=================================================================

On the transcribed open-source architectures, 40-66% of Applications carry a
ground-truth impact of exactly 0.0 (JSS Section 7.4). Spearman over a population
that heavily tied is dominated by midrank tie handling rather than by the
ordering under test, so a headline rho can be carried almost entirely by the two
rankings agreeing on which components are inert.

These pin the sensitivity bound that answers that objection:

  * rho over the strictly-positive subset is computed and reported alongside the
    full-population figure;
  * it never replaces it — zeros stay in the headline rho, because for a node
    the simulator actually injected, zero impact is a real measurement;
  * it is withheld rather than fabricated when the positive subset is too small.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── The oracle-agreement path (reproduce/convergent_validity.py) ──────────────

def test_pairwise_reports_zero_counts_and_positive_rho():
    from reproduce.convergent_validity import _pairwise

    # Ranks agree perfectly on the tied-zero block and disagree on the rest, so
    # the two figures must separate.
    a = {"z1": 0.0, "z2": 0.0, "z3": 0.0, "p1": 0.1, "p2": 0.2, "p3": 0.3}
    b = {"z1": 0.0, "z2": 0.0, "z3": 0.0, "p1": 0.3, "p2": 0.2, "p3": 0.1}

    block = _pairwise(a, b)

    assert block["n_zero_a"] == 3
    assert block["n_zero_b"] == 3
    assert block["n_both_positive"] == 3
    # Full population is lifted by the agreeing zero block; the positive subset
    # is perfectly reversed. If these were equal the bound would be vacuous.
    assert block["spearman_rho"] > block["spearman_rho_positive"]
    assert block["spearman_rho_positive"] == pytest.approx(-1.0)


def test_pairwise_keeps_zeros_in_the_headline_rho():
    """The zero-excluded figure is a bound, not a filter applied to the result."""
    from reproduce.convergent_validity import _pairwise

    a = {"z1": 0.0, "z2": 0.0, "p1": 0.1, "p2": 0.2, "p3": 0.3}
    b = {"z1": 0.0, "z2": 0.0, "p1": 0.1, "p2": 0.2, "p3": 0.3}

    block = _pairwise(a, b)
    # n_common counts every shared node, zeros included.
    assert block["n_common"] == 5
    assert block["n_both_positive"] == 3


def test_pairwise_withholds_positive_rho_on_a_tiny_subset():
    from reproduce.convergent_validity import _pairwise

    a = {"z1": 0.0, "z2": 0.0, "z3": 0.0, "z4": 0.0, "p1": 0.1, "p2": 0.2}
    b = {"z1": 0.0, "z2": 0.0, "z3": 0.0, "z4": 0.0, "p1": 0.1, "p2": 0.2}

    block = _pairwise(a, b)
    assert block["n_both_positive"] == 2
    assert block["spearman_rho_positive"] is None


# ── The zero-shot validation path (cli/validation/statistics.py) ──────────────

def _scores(pairs):
    """Build the NodeScores map run_statistical_tests consumes."""
    from cli.validation.scoring import NodeScores

    out = {}
    for i, (q, imp) in enumerate(pairs):
        ns = NodeScores(node_id=f"A{i}", node_type="Application")
        ns.Q = q
        ns.I = imp
        ns.degree_centrality = 0.0
        out[ns.node_id] = ns
    return out


def test_statistics_report_zero_exclusion_bound():
    from cli.validation.statistics import run_statistical_tests

    # Six inert components plus six active ones whose ordering Q gets backwards.
    pairs = [(0.9 - 0.1 * i, 0.0) for i in range(6)]
    pairs += [(0.1 * i, 0.6 - 0.1 * i) for i in range(6)]

    stat = run_statistical_tests(_scores(pairs), top_k=3, B=50)

    assert stat["n_zero_impact"] == 6
    assert stat["n_positive_impact"] == 6
    assert stat["spearman_rho_positive"] == pytest.approx(-1.0)
    assert stat["spearman_rho"] != stat["spearman_rho_positive"]


def test_statistics_withhold_positive_rho_when_all_labels_are_zero():
    from cli.validation.statistics import run_statistical_tests

    stat = run_statistical_tests(_scores([(0.5 - 0.05 * i, 0.0) for i in range(8)]),
                                 top_k=3, B=50)

    assert stat["n_positive_impact"] == 0
    assert stat["n_zero_impact"] == 8
    assert stat["spearman_rho_positive"] is None
