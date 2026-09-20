"""
tests/test_reconcile_guards.py
──────────────────────────────
Pins the reconciler's "I checked nothing" guards.

``_rows`` returns ``[]`` for a marker it cannot find, and a check that iterates
an empty list records no findings — so a table whose column count moved would
report as a clean run over rows nobody looked at. That is the precise failure
``reconcile_manuscript.py`` exists to prevent, and it has bitten this project
before (a stale hand-written label mirror silently skipped rows). The markers
these checks anchor on embed the column count, so they move whenever a column is
added: Table 7 grew a Δρ column, which is what prompted these tests.
"""

import pytest

import reproduce.reconcile_manuscript as R


@pytest.fixture
def moved_marker(monkeypatch):
    """Simulate Table 7 gaining a column without its check being updated."""
    original = R._tex

    def patched(name: str) -> str:
        return original(name).replace(
            r"\multicolumn{8}{l}{\textit{Training-free structural baselines}}",
            r"\multicolumn{9}{l}{\textit{Training-free structural baselines}}",
        )

    monkeypatch.setattr(R, "_tex", patched)


def test_table7_reports_a_skip_when_its_marker_moves(moved_marker):
    rep = R.Report()
    R.check_table7_loso(rep, "loso_all_variants_v5.json")
    assert rep.checked == 0
    assert rep.skipped, "a moved marker must surface as a skip, not as a clean pass"
    assert "tab:7" in rep.skipped[0]


def test_table7_checks_rows_against_the_live_manuscript():
    """The complement: with the marker intact, rows are actually checked."""
    rep = R.Report()
    R.check_table7_loso(rep, "loso_all_variants_v5.json")
    if rep.skipped:
        pytest.skip("loso_all_variants_v5.json absent (results/ is gitignored)")
    assert rep.checked > 0
    assert not rep.findings


def test_table7_delta_is_checked_against_the_significance_artifact():
    """The Δρ column is verified, not hand-typed.

    It must reconcile against loso_significance (which owns the pre-registered
    comparator) rather than against the variants artifact beside it, because the
    defect being guarded was a Δ measured against the wrong row.
    """
    rep = R.Report()
    R.check_table7_delta(rep)
    if rep.skipped:
        pytest.skip("loso_significance_v5.json absent (results/ is gitignored)")
    assert rep.checked > 0
    assert not rep.findings


def test_table9c_skips_loudly_when_the_table_is_absent(monkeypatch):
    """A table the manuscript does not carry is a skip, not a silent success."""
    original = R._tex
    monkeypatch.setattr(
        R, "_tex", lambda name: original(name).replace(r"\label{tab:9c}", r"\label{tab:9c-absent}")
    )
    rep = R.Report()
    R.check_table9c_active(rep)
    assert rep.checked == 0
    assert rep.skipped
