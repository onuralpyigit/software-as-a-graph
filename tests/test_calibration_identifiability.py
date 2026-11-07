"""
Identifiability: simulate at known parameters, calibrate the trace, get them back.

`tests/test_testbed.py` checks the round trip on a four-app chain at fixed
parameters. These tests run it on a corpus scenario under the runtime-label
settings and under parameters drawn from `PARAMETER_RANGES` -- the guesses a
post-deployment calibration would have to narrow. What the calibrator does not
fit (staleness window, burstiness) is recorded as unidentifiable, not tested.
"""

import random
import statistics
from pathlib import Path

import pytest

pytest.importorskip("simpy")

from saag.core.graph_io import load_graph  # noqa: E402
from saag.simulation.calibration import (  # noqa: E402
    TelemetryCalibrator,
    TelemetryTrace,
    infer_trigger_policy,
)
from saag.simulation.message_flow_simulator import MessageFlowSimulator  # noqa: E402
from saag.simulation.runtime.behavior import policy_for  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.simulation.runtime.labels import RUNTIME_LABEL_SETTINGS, runtime_labels  # noqa: E402
from saag.simulation.runtime.uncertainty import (  # noqa: E402
    PARAMETER_RANGES,
    inert_parameters,
    sample_settings,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {k: v for k, v in RUNTIME_LABEL_SETTINGS.items() if k not in ("duration", "estimator")}


def _simulate(graph, duration, faults=None, **extra):
    sim = MessageFlowSimulator(graph, duration=duration, seed=7, record_events=True,
                               faults=faults, **{**SETTINGS, **extra})
    res = sim.run()
    return sim, res, TelemetryTrace.from_records(res.events, start_s=0.0, end_s=duration)


@pytest.fixture(scope="module")
def atm():
    return load_graph(ROOT / "data" / "scenarios" / "atm_system.json")


@pytest.mark.slow
@pytest.mark.parametrize("draw", [0, 1], ids=["draw0", "draw1"])
def test_rates_and_service_times_come_back_under_drawn_parameters(atm, draw):
    rng = random.Random(3)
    for _ in range(draw + 1):
        drawn = sample_settings(rng)
    sim, res, trace = _simulate(atm, 60.0, **drawn)
    prof = TelemetryCalibrator(infer_policies=False).calibrate(trace)

    rate_err = [abs(r - sim.topic_aggregate_rate(t)) / sim.topic_aggregate_rate(t)
                for t, r in prof.msg_rate_per_sec.items() if sim.topic_aggregate_rate(t) > 0]
    mean_err = [abs(f["mean_s"] - res.service_time_s[a]) / res.service_time_s[a]
                for a, f in prof.service_time_dist.items() if res.service_time_s.get(a)]
    assert len(rate_err) >= 15 and len(mean_err) >= 15
    # Measured 2026-11-07: medians 0.025-0.034 (rates), 0.009-0.010 (means).
    assert statistics.median(rate_err) < 0.06
    assert statistics.median(mean_err) < 0.06
    # Every app within sampling noise: error in standard errors (CV / sqrt(n)).
    # Measured at most 1.8 here, and 3.6 across 30 apps on healthcare.
    z = [abs(f["mean_s"] - res.service_time_s[a]) / res.service_time_s[a] * f["n"] ** 0.5 / f["cv"]
         for a, f in prof.service_time_dist.items() if res.service_time_s.get(a)]
    assert max(z) < 4.0


@pytest.mark.slow
def test_deadline_drops_do_not_bias_the_service_fit(atm):
    """A message dropped for its deadline was served first. Before its
    saag.deadline_miss carried service_ms, the calibrator fitted only the fast
    survivors: on atm A11 (3855 misses, 2141 deliveries) the mean came back 45%
    low, against 7% for the worst app with QoS off."""
    rng = random.Random(3)
    sample_settings(rng)
    _, res, trace = _simulate(atm, 60.0, **sample_settings(rng))
    fits = TelemetryCalibrator(infer_policies=False).calibrate(trace).service_time_dist
    assert max(abs(f["mean_s"] - res.service_time_s[a]) / res.service_time_s[a]
               for a, f in fits.items()) < 0.15


@pytest.mark.slow
@pytest.mark.parametrize("dist, cv", [("exponential", 1.0), ("uniform", 0.4 / 12 ** 0.5)])
def test_service_variability_comes_back(atm, dist, cv):
    # uniform is mean x U(0.8, 1.2). The calibrator calls it lognormal, which the
    # simulator replays with the measured CV, so the first two moments survive.
    _, _, trace = _simulate(atm, 60.0, service_distribution=dist)
    fits = TelemetryCalibrator(infer_policies=False).calibrate(trace).service_time_dist.values()
    assert statistics.median(f["cv"] for f in fits) == pytest.approx(cv, rel=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("staleness_periods", [2.0, 6.0])
def test_inferred_trigger_policy_is_never_wrong(atm, staleness_periods):
    """Every app is down 15 s in one of eight staggered slots, so inputs go
    partly and fully stale. The calibrator assumes 3 periods of staleness; the
    true window is drawn from 2-6. It may abstain, but must not mislabel."""
    apps = sorted(n for n, a in atm.nodes(data=True) if a.get("type") == "Application")
    faults = [FaultSpec(a, "transient_crash", 20.0 + (i % 8) * 20.0, 15.0)
              for i, a in enumerate(apps)]
    _, _, trace = _simulate(atm, 200.0, faults, staleness_periods=staleness_periods)

    right = 0
    for app in apps:
        got, evidence = infer_trigger_policy(trace, app)
        inputs = trace.inputs_of(app)
        if got is None or not inputs:
            continue  # abstained, or input-less (periodic in effect)
        truth = policy_for(atm.nodes[app]).kind
        if len(inputs) == 1 and {got, truth} <= {"all_of", "any_of"}:
            truth = got  # one input: all_of and any_of are the same policy
        assert got == truth, (app, evidence)
        right += 1
    assert right >= 14  # measured 15 and 14 of 17 apps with inputs and outputs


def test_burst_ratio_is_inert_under_the_runtime_label_settings():
    g = load_graph(ROOT / "data" / "scenarios" / "tiny_system.json")
    runs = [_simulate(g, 10.0, burst_ratio=b)[1].events for b in (2.0, 6.0)]
    assert runs[0] == runs[1]
    assert inert_parameters(RUNTIME_LABEL_SETTINGS) == ["burst_ratio"]
    assert inert_parameters({**RUNTIME_LABEL_SETTINGS, "workload_model": "mmpp"}) == []
    assert inert_parameters({}) == ["burst_ratio", "staleness_periods"]


def test_parameter_stability_names_the_inert_draws():
    g = load_graph(ROOT / "data" / "scenarios" / "tiny_system.json")
    out = runtime_labels(g, seeds=[1], settings={"duration": 5.0}, candidates=["A9"],
                         parameter_draws=2)
    assert out["parameter_stability"]["inert"] == ["burst_ratio"]
    assert set(out["parameter_stability"]["settings"][0]) == set(PARAMETER_RANGES)
