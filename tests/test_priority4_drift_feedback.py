"""
Tests for Priority 4 closed-loop drift detection & Tier-2 residual feedback:
- detect_drift (steady-state agreement vs rate/latency drift)
- TelemetryCalibrator.update_profile (incremental EMA recalibration)
- diagnose_fault_residuals + apply_residual_feedback (closing stall-set residuals)
"""

from __future__ import annotations

import logging

import networkx as nx
import pytest

pytest.importorskip("simpy")

from saag.simulation.calibration import TelemetryCalibrator, TelemetryTrace  # noqa: E402
from saag.simulation.message_flow_simulator import MessageFlowSimulator  # noqa: E402
from saag.simulation.models import RuntimeTelemetryProfile  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.validation.fidelity import (  # noqa: E402
    apply_residual_feedback,
    detect_drift,
    diagnose_fault_residuals,
    stalled_in_trace,
)

logging.disable(logging.WARNING)


def _graph(freq: float = 10.0, trigger_m: str = "all_of") -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("S1", type="Application", app_type="sensor")
    g.add_node("S2", type="Application", app_type="sensor")
    g.add_node("M", type="Application", app_type="processor", trigger=trigger_m)
    g.add_node("K", type="Application", app_type="actuator")
    for t in ("t1", "t2", "tout"):
        g.add_node(t, type="Topic", frequency=freq)
    g.add_edge("S1", "t1", type="PUBLISHES_TO")
    g.add_edge("S2", "t2", type="PUBLISHES_TO")
    g.add_edge("M", "t1", type="SUBSCRIBES_TO")
    g.add_edge("M", "t2", type="SUBSCRIBES_TO")
    g.add_edge("M", "tout", type="PUBLISHES_TO")
    g.add_edge("K", "tout", type="SUBSCRIBES_TO")
    return g


# ── 1. Steady-state drift detection ──────────────────────────────────────────

def test_detect_drift_distinguishes_matched_from_shifted_workload():
    g_base = _graph(freq=10.0)
    g_drifted = _graph(freq=20.0)

    r_sim = MessageFlowSimulator(
        g_base, duration=60.0, seed=1, service_distribution="uniform",
        rng_streams="per_process", record_events=True,
    ).run()
    r_same = MessageFlowSimulator(
        g_base, duration=60.0, seed=2, service_distribution="uniform",
        rng_streams="per_process", record_events=True,
    ).run()
    r_shift = MessageFlowSimulator(
        g_drifted, duration=60.0, seed=2, service_distribution="uniform",
        rng_streams="per_process", record_events=True,
    ).run()

    tr_sim = TelemetryTrace.from_records(r_sim.events, start_s=0.0, end_s=60.0)
    tr_same = TelemetryTrace.from_records(r_same.events, start_s=0.0, end_s=60.0)
    tr_shift = TelemetryTrace.from_records(r_shift.events, start_s=0.0, end_s=60.0)

    ok = detect_drift(tr_sim, tr_same)
    assert ok["drifted"] is False
    assert ok["drifted_metrics"] == []

    bad = detect_drift(tr_sim, tr_shift)
    assert bad["drifted"] is True
    assert "topic_rate" in bad["drifted_metrics"]
    assert "t1" in bad["drifted_topics"]


# ── 2. Incremental EMA recalibration ─────────────────────────────────────────

def test_update_profile_blends_via_ema_and_preserves_unobserved():
    prior = RuntimeTelemetryProfile(
        msg_rate_per_sec={"t1": 10.0, "t_old": 5.0},
        service_time_dist={"M": {"dist": "exponential", "mean_s": 0.02, "cv": 1.0, "n": 100}},
        trigger_policy={"M": "any_of"},
        provenance={"source": "initial", "update_count": 1},
    )
    g_new = _graph(freq=20.0)
    res = MessageFlowSimulator(
        g_new, duration=20.0, seed=3, rng_streams="per_process", record_events=True
    ).run()
    tr = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=20.0)

    cal = TelemetryCalibrator(infer_policies=False)
    updated = cal.update_profile(prior, tr, alpha=0.5, source="window_2")

    # t1 blended: 0.5 * ~20 + 0.5 * 10 = ~15
    assert updated.msg_rate_per_sec["t1"] == pytest.approx(15.0, rel=0.08)
    # Unobserved topic and policy preserved
    assert updated.msg_rate_per_sec["t_old"] == 5.0
    assert updated.trigger_policy["M"] == "any_of"
    assert updated.provenance["update_count"] == 2
    assert updated.provenance["updated_from"] == "initial"
    assert updated.service_time_dist["M"]["n"] > 100

    with pytest.raises(ValueError, match="alpha"):
        cal.update_profile(prior, tr, alpha=0.0)


# ── 3. Tier-2 residual diagnosis & closed-loop policy feedback ───────────────

def test_diagnose_and_apply_residual_feedback_closes_policy_mismatch():
    # Ground truth ("observed" Tier 2): M actually runs all_of, so crashing S1 stalls M
    g_obs = _graph(freq=10.0, trigger_m="all_of")
    r_obs = MessageFlowSimulator(
        g_obs,
        duration=30.0,
        seed=5,
        faults=[FaultSpec("S1", "crash", 10.0)],
        behavior_model="reactive",
        rng_streams="per_process",
        record_events=True,
    ).run()
    tr_obs = TelemetryTrace.from_records(r_obs.events, start_s=0.0, end_s=30.0)
    obs_stalled = stalled_in_trace(tr_obs, 10.0) - {"S1"}
    assert "M" in obs_stalled

    # Initial digital twin ("simulated" Tier 1) mistakenly thinks M is any_of -> M does NOT stall
    profile = RuntimeTelemetryProfile(trigger_policy={"M": "any_of"})
    r_sim_before = MessageFlowSimulator(
        g_obs,
        duration=30.0,
        seed=5,
        faults=[FaultSpec("S1", "crash", 10.0)],
        behavior_model="reactive",
        telemetry_profile=profile,
        rng_streams="per_process",
        record_events=True,
    ).run()
    tr_sim_before = TelemetryTrace.from_records(r_sim_before.events, start_s=0.0, end_s=30.0)
    sim_stalled_before = stalled_in_trace(tr_sim_before, 10.0) - {"S1"}
    assert "M" not in sim_stalled_before

    # Diagnose residual and apply feedback
    diag = diagnose_fault_residuals(
        simulated_impact={"S1": 0.2},
        observed_impact={"S1": 0.7},
        simulated_stalled=sim_stalled_before,
        observed_stalled=obs_stalled,
        observed_trace=tr_obs,
        current_profile=profile,
    )
    assert diag["needs_recalibration"] is True
    assert "M" in diag["suspect_policies"]
    assert diag["policy_updates"] == {"M": "all_of"}
    assert "S1" in diag["large_impact_residuals"]

    updated_profile = apply_residual_feedback(profile, diag)
    # Original profile is not mutated
    assert profile.trigger_policy["M"] == "any_of"
    assert updated_profile.trigger_policy["M"] == "all_of"

    # Re-run digital twin with updated profile: M now stalls, closing the residual!
    r_sim_after = MessageFlowSimulator(
        g_obs,
        duration=30.0,
        seed=5,
        faults=[FaultSpec("S1", "crash", 10.0)],
        behavior_model="reactive",
        telemetry_profile=updated_profile,
        rng_streams="per_process",
        record_events=True,
    ).run()
    tr_sim_after = TelemetryTrace.from_records(r_sim_after.events, start_s=0.0, end_s=30.0)
    sim_stalled_after = stalled_in_trace(tr_sim_after, 10.0) - {"S1"}
    assert sim_stalled_after == obs_stalled
