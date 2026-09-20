"""
tests/test_detection_validation.py
──────────────────────────────────
Tests for reproduce/detection_validation.py:
- Verifies evaluate() runs on a scenario without throwing keyword argument errors
- Verifies calculate_spof_f1 integration with ia_quantile=0.75
- Verifies output schema contains required predictor, catalog, and spof metrics
"""

from pathlib import Path
import pytest

from reproduce.detection_validation import (
    DEFAULT_EXCLUDED_PATTERNS,
    evaluate,
)


def test_evaluate_scenario_atm_system():
    """Verify detection validation runs cleanly on atm_system without calculate_spof_f1 errors."""
    row = evaluate(
        scenario="atm_system",
        layer="system",
        seed=42,
        threshold=0.2,
        excluded=DEFAULT_EXCLUDED_PATTERNS,
    )

    assert "error" not in row
    assert row["scenario"] == "atm_system"
    assert row["n_scored"] > 0
    assert "predictors" in row
    assert "q_composite" in row["predictors"]
    assert "catalog" in row
    assert "spof" in row

    spof = row["spof"]
    assert "f1" in spof
    assert "precision" in spof
    assert "recall" in spof
    assert isinstance(spof["f1"], float)
    assert 0.0 <= spof["f1"] <= 1.0
    assert "ia_threshold" in spof
    assert "ia_max" in spof


def test_icomp_istar_agreement_is_read_not_hardcoded(tmp_path, monkeypatch):
    """The cross-reference reports the artifact's value, whatever it is.

    A literal 0.4046 used to be baked into every detection artifact's ``note``.
    The figure moved when the oracle was recalibrated and the string did not, so
    the artifacts shipped a pointer to a measurement that existed in no
    convergent-validity file under any version.
    """
    import json

    from reproduce import detection_validation as dv

    (tmp_path / "convergent_validity.json").write_text(json.dumps(
        {"summary": {"i_comp__i_star": {"mean_spearman_rho": 0.1234,
                                        "n_scenarios_measured": 7}}}))
    monkeypatch.setattr(dv, "RESULTS_DIR", tmp_path)
    sentence = dv._icomp_istar_agreement()
    assert "0.1234" in sentence and "7 scenarios" in sentence
    assert "0.4046" not in sentence


def test_icomp_istar_agreement_says_so_when_unmeasured(tmp_path, monkeypatch):
    """An absent artifact yields an admission, never an invented number."""
    from reproduce import detection_validation as dv

    monkeypatch.setattr(dv, "RESULTS_DIR", tmp_path)
    sentence = dv._icomp_istar_agreement()
    assert "unmeasured" in sentence
    assert not any(ch.isdigit() for ch in sentence.replace("I_comp", "").replace("I*", ""))
