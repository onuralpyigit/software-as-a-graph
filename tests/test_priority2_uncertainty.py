"""
Tests for Priority 2 pre-deployment uncertainty analysis:
- Latin Hypercube (LHS) and Sobol space-filling parameter designs
- Upper-tail risk summaries (mean, quantile, CVaR) and CVaR risk ranking
- Per-parameter sensitivity attribution (numeric Spearman & categorical eta_sq)
- Integration with runtime_labels and cli/simulate_graph.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from saag.simulation.runtime.uncertainty import (
    PARAMETER_RANGES,
    SAMPLING_METHODS,
    cvar_ranking,
    parameter_sensitivity,
    risk_summary,
    sample_design,
)

logging.disable(logging.WARNING)


# ── 1. Space-filling sampling designs (LHS & Sobol) ──────────────────────────

def test_lhs_stratifies_numeric_and_balances_categorical():
    n = 8
    design = sample_design(n, seed=7, ranges=PARAMETER_RANGES, method="lhs")
    assert len(design) == n
    # Reproducible for same seed
    assert design == sample_design(n, seed=7, ranges=PARAMETER_RANGES, method="lhs")

    # Numeric parameter: exactly one sample in each [i/n, (i+1)/n) stratum
    lo, hi = PARAMETER_RANGES["target_utilization"]
    strata = sorted(int(((d["target_utilization"] - lo) / (hi - lo)) * n) for d in design)
    assert strata == list(range(n))

    # Categorical parameter: evenly balanced (4 exponential, 4 uniform)
    cats = [d["service_distribution"] for d in design]
    assert cats.count("exponential") == 4
    assert cats.count("uniform") == 4


def test_sobol_design_deterministic_and_handles_non_power_of_two():
    for n in (3, 5, 8):
        a = sample_design(n, seed=11, ranges=PARAMETER_RANGES, method="sobol")
        b = sample_design(n, seed=11, ranges=PARAMETER_RANGES, method="sobol")
        assert a == b
        assert len(a) == n
        for d in a:
            lo, hi = PARAMETER_RANGES["target_utilization"]
            assert lo <= d["target_utilization"] <= hi
            assert d["service_distribution"] in PARAMETER_RANGES["service_distribution"]


def test_sample_design_validates_method_and_n():
    assert sample_design(0) == []
    with pytest.raises(ValueError, match="method"):
        sample_design(4, method="grid")
    with pytest.raises(ValueError, match="n must be"):
        sample_design(-1)
    assert set(SAMPLING_METHODS) == {"random", "lhs", "sobol"}


# ── 2. Risk summaries & CVaR ranking ─────────────────────────────────────────

def test_risk_summary_and_cvar_ranking():
    # Node A: steady 0.50 across all 10 draws (mean=0.50, cvar@0.9=0.50)
    # Node B: 0.40 on 9 draws, 1.40 on 1 draw (mean=0.50, cvar@0.9=1.40)
    # Node C: steady 0.20 across all 10 draws
    draws = [{"A": 0.50, "B": 0.40, "C": 0.20} for _ in range(9)]
    draws.append({"A": 0.50, "B": 1.40, "C": 0.20})

    summary = risk_summary(draws, alpha=0.9)
    assert summary["A"]["mean"] == pytest.approx(0.50)
    assert summary["B"]["mean"] == pytest.approx(0.50)
    assert summary["A"]["cvar"] == pytest.approx(0.50)
    assert summary["B"]["cvar"] == pytest.approx(1.40)
    assert summary["B"]["quantile"] == pytest.approx(0.50)

    # CVaR ranking puts tail-risky B ahead of steady A despite equal means
    assert cvar_ranking(draws, alpha=0.9) == ["B", "A", "C"]


# ── 3. Per-parameter sensitivity attribution ─────────────────────────────────

def test_parameter_sensitivity_identifies_numeric_and_categorical_drivers():
    design = sample_design(16, seed=42, ranges=PARAMETER_RANGES, method="lhs")
    # Synthetic impact driven primarily by target_utilization and service_distribution,
    # with zero dependence on staleness_periods, and burst_ratio marked inert.
    labels = []
    for s in design:
        u = s["target_utilization"]
        cat_bump = 0.08 if s["service_distribution"] == "exponential" else 0.0
        labels.append({
            "A0": 0.8 * u + cat_bump,
            "A1": 0.4 * u + 0.5 * cat_bump,
            "A2": 0.1,  # constant node
        })

    sens = parameter_sensitivity(design, labels, inert=["burst_ratio"])
    params = sens["parameters"]
    assert params["burst_ratio"] == {"kind": "inert", "score": 0.0}
    assert "burst_ratio" not in sens["top_drivers"]
    assert params["target_utilization"]["kind"] == "numeric"
    assert params["target_utilization"]["score"] > 0.75
    assert params["service_distribution"]["kind"] == "categorical"
    assert params["service_distribution"]["eta_sq"] > 0.05
    assert params["staleness_periods"]["score"] < params["target_utilization"]["score"]
    assert sens["top_drivers"][0] == "target_utilization"


# ── 4. End-to-end runtime_labels & CLI integration ───────────────────────────

def test_runtime_labels_parameter_stability_includes_risk_and_sensitivity():
    pytest.importorskip("simpy")
    from saag.core.graph_io import load_graph
    from saag.simulation.runtime.labels import runtime_labels

    g = load_graph(Path(__file__).resolve().parents[1] / "data" / "scenarios" / "tiny_system.json")
    out = runtime_labels(
        g,
        seeds=[42],
        settings={"duration": 6.0},
        parameter_draws=4,
        parameter_sampling="sobol",
        risk_alpha=0.75,
    )
    ps = out["parameter_stability"]
    assert ps["sampling"] == "sobol"
    assert ps["risk_alpha"] == 0.75
    assert "i_flow" in ps["risk"]
    assert "i_flow" in ps["cvar_ranking"]
    assert "i_flow" in ps["sensitivity"]
    assert "burst_ratio" in ps["inert"]
    assert ps["sensitivity"]["i_flow"]["parameters"]["burst_ratio"]["kind"] == "inert"


def test_cli_parses_parameter_sampling():
    from cli.simulate_graph import _build_parser

    args = _build_parser().parse_args(
        ["runtime-sweep", "--parameter-draws", "6", "--parameter-sampling", "sobol"]
    )
    assert args.parameter_draws == 6
    assert args.parameter_sampling == "sobol"
