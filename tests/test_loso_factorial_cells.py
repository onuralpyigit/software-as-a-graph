"""
test_loso_factorial_cells.py
────────────────────────────
Pins the ``cells`` option of ``loso_significance.factorial``.

The matched 2x2 of PREREGISTRATION.md Amendment 2 (GAT-N-C, HGT, GAT-N-QoS16-C,
HGT-QoS) reuses the same three orthogonal quantities as the reported 2x2, over a
different set of arms. The default must stay the reported arms, and a custom
mapping must read exactly the arms it names.
"""

import pytest

from reproduce.loso_significance import FACTORIAL_CELLS, factorial

FOLDS = ["f1", "f2", "f3", "f4", "f5", "f6"]


def _arm(values):
    return {"per_fold": [{"holdout": f, "mean_rho": v} for f, v in zip(FOLDS, values)]}


@pytest.fixture
def table():
    # Reported arms: typing +0.2 without QoS, 0 with it -> interaction -0.2.
    # Matched arms: typing +0.1 at both Q levels -> interaction 0.
    return {
        "gl": _arm([0.30, 0.31, 0.32, 0.33, 0.34, 0.35]),
        "hgl": _arm([0.50, 0.51, 0.52, 0.53, 0.54, 0.55]),
        "gl_qos": _arm([0.60, 0.61, 0.62, 0.63, 0.64, 0.65]),
        "hgl_qos": _arm([0.60, 0.61, 0.62, 0.63, 0.64, 0.65]),
        "gl_full_cap": _arm([0.40, 0.41, 0.42, 0.43, 0.44, 0.45]),
        "gl_full_qos16_cap": _arm([0.50, 0.51, 0.52, 0.53, 0.54, 0.55]),
    }


def _interaction(rows):
    return next(r for r in rows if r["quantity"] == "interaction")["mean_delta"]


def test_default_cells_are_the_reported_arms(table):
    assert FACTORIAL_CELLS == {
        ("T0", "Q0"): "gl", ("T1", "Q0"): "hgl",
        ("T0", "Q1"): "gl_qos", ("T1", "Q1"): "hgl_qos",
    }
    assert factorial(table) == factorial(table, cells=FACTORIAL_CELLS)
    assert _interaction(factorial(table)) == pytest.approx(-0.2)


def test_custom_cells_read_only_the_named_arms(table):
    matched = {
        ("T0", "Q0"): "gl_full_cap", ("T1", "Q0"): "hgl",
        ("T0", "Q1"): "gl_full_qos16_cap", ("T1", "Q1"): "hgl_qos",
    }
    rows = factorial(table, cells=matched)
    assert _interaction(rows) == pytest.approx(0.0)
    typing = next(r for r in rows if r["quantity"] == "main_typing")["mean_delta"]
    assert typing == pytest.approx(0.1)


def test_missing_arm_yields_no_factorial(table):
    cells = dict(FACTORIAL_CELLS)
    cells[("T0", "Q0")] = "not_run"
    assert factorial(table, cells=cells) == []
