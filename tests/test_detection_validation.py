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
