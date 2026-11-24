"""
Unit tests for:
  1. stalled_in_trace prime-second / multi-topic inter-emission gap fix (healthcare A4 regression)
  2. reproduce/calibration_drift_eval.py perturbation & target selection helpers
  3. reproduce/fidelity_significance_v2.py exact Wilcoxon, HL CI, TOST, and Holm analysis
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reproduce.calibration_drift_eval import (
    perturb_plan,
    pick_behavioral_target,
    resolve_scenario_path,
)
from reproduce.fidelity_significance_v2 import analyze_study2, holm_adjust
from saag.core.graph_io import load_graph
from saag.simulation.calibration import TelemetryTrace
from saag.testbed.deployment import build_deployment
from saag.testbed.study import simulate_arm
from saag.validation.fidelity import stalled_in_trace


def test_stalled_in_trace_respects_max_pre_fault_gap_on_synchronized_topics():
    """Multi-topic sub-1 Hz app locked to integer seconds (no tick at prime t=59) is NOT stalled."""
    events = []
    # Periods 2, 3, 4, 5, 8 s — last tick before 60.0 is t=58.0 (gap 2.0 s)
    for period in (2.0, 3.0, 4.0, 5.0, 8.0):
        t = period
        while t < 60.0:
            events.append({"kind": "publish", "t": t, "app": "A4", "topic": f"T{int(period)}"})
            t += period
    # Genuine stalled app stops at t=30.0
    for t in range(1, 30):
        events.append({"kind": "publish", "t": float(t), "app": "A_DEAD", "topic": "T_DEAD"})

    trace = TelemetryTrace(events=events, start_s=0.0, end_s=60.0)
    stalled = stalled_in_trace(trace, after=30.0)
    assert "A4" not in stalled
    assert "A_DEAD" in stalled


def test_healthcare_a4_not_falsely_stalled_under_unrelated_fault():
    """Regression check on healthcare_system.json: crashing N0 must not mark A4 stalled."""
    p = Path("data/scenarios/healthcare_system.json")
    if not p.exists():
        pytest.skip("healthcare_system.json not present")
    g = load_graph(p)
    plan = build_deployment(p, open_loop=True, qos_semantics="dds")
    res = simulate_arm(g, plan, "S2", "N0", 60.0, 30.0, 5.0, 42)
    assert "A4" not in res["stalled"]


def test_perturb_plan_shifts_rates_and_busiest_host_service_times():
    src = resolve_scenario_path("tiny_system")
    plan = build_deployment(src, open_loop=True, qos_semantics="dds")
    pert, meta = perturb_plan(plan, rate_frac=0.4, service_mult=2.0)

    assert len(meta["shifted_topics"]) >= 1
    for t, mult in meta["shifted_topics"].items():
        assert pert["topics"][t]["rate_hz"] == pytest.approx(plan["topics"][t]["rate_hz"] * mult)
    assert meta["target_host"] is not None
    for a in meta["shifted_apps"]:
        assert pert["apps"][a]["service_mean_s"] == pytest.approx(plan["apps"][a]["service_mean_s"] * 2.0)

    pair = pick_behavioral_target(plan)
    assert pair is None or (len(pair) == 2 and pair[0] != pair[1])


def test_study2_significance_analysis_detects_clear_effects_at_n13():
    """Verify H2.1 Better and all 4 Holm-corrected secondary tests pass on clear synthetic data."""
    scenarios = {}
    for i in range(13):
        scenarios[f"s{i}"] = {
            "kappa": 1.0,
            "latency_p50_err_S0": 0.42 + 0.01 * (i % 3),
            "latency_p50_err_S2": 0.22 + 0.01 * (i % 2),
            "infra_rho_S2": 0.96,
            "infra_stall_J_S2": 0.95,
            "perturbed_delivery_err_S2": 0.35,
            "perturbed_delivery_err_S3": 0.02,
            "omission_err_crash_ref": 0.25,
            "omission_err_S2": 0.04,
        }
    # Add 14th scaled scenario (excluded from H2.1 latency, included in H2.2-H2.4)
    scenarios["s_scaled"] = {
        "kappa": 0.74,
        "latency_p50_err_S0": 0.05,
        "latency_p50_err_S2": 0.99,  # must be ignored by H2.1 because kappa < 1
        "infra_rho_S2": 0.97,
        "infra_stall_J_S2": 0.94,
        "perturbed_delivery_err_S2": 0.30,
        "perturbed_delivery_err_S3": 0.01,
        "omission_err_crash_ref": 0.22,
        "omission_err_S2": 0.03,
    }
    res = analyze_study2({"scenarios": scenarios})
    assert res["H2.1_latency_p50"]["n"] == 13
    assert res["H2.1_latency_p50"]["outcome"] == "Better"
    assert res["H2.1_latency_p50"]["p_superiority"] < 0.001
    for key, row in res["secondary"].items():
        assert row["n"] == 14
        assert row["significant"] is True, f"{key} should be significant"


def test_holm_adjust_step_down_stops_after_first_nonsignificant():
    adj = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02}, alpha=0.05)
    # Sorted: a (0.01 <= 0.0167 sig), c (0.02 <= 0.025 sig), b (0.03 <= 0.05 sig)
    assert adj["a"]["significant"] is True
    assert adj["c"]["significant"] is True
    assert adj["b"]["significant"] is True

    adj2 = holm_adjust({"a": 0.02, "b": 0.021}, alpha=0.05)
    # Sorted: a (0.02 <= 0.025 sig), b (0.021 <= 0.05 sig)
    assert adj2["a"]["significant"] is True
    adj3 = holm_adjust({"a": 0.03, "b": 0.04}, alpha=0.05)
    # a fails 0.025 -> b must also be False even though 0.04 <= 0.05
    assert adj3["a"]["significant"] is False
    assert adj3["b"]["significant"] is False
