"""Tooling for docs/research/fidelity/PREREGISTRATION.md. No test compares a simulator with Docker."""

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("simpy")

from saag.core.graph_io import load_graph  # noqa: E402
from saag.simulation import MessageFlowSimulator  # noqa: E402
from saag.simulation.calibration import TelemetryTrace  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.testbed import orchestrate, study  # noqa: E402
from saag.testbed.agent import run_host  # noqa: E402
from saag.testbed.deployment import build_deployment  # noqa: E402
from saag.testbed.runtime import LocalTransport, TelemetrySink  # noqa: E402
from saag.validation.fidelity import observed_impact  # noqa: E402
from tests.test_testbed import scenario  # noqa: E402

TINY = "data/scenarios/tiny_system.json"


def test_observed_impact_equals_the_engines_i_dyn():
    # atm, not tiny: at 1 Hz, whole-sample counting differs from the engine by up to ~0.03.
    path = Path("data/scenarios/atm_system.json")
    g = load_graph(path)
    plan = build_deployment(path)
    for target in sorted(a for a, s in plan["apps"].items() if s["publishes"])[:3]:
        res = MessageFlowSimulator(g, duration=30.0, seed=42, fault_node=target, fault_time=15.0,
                                   record_events=True).run()
        trace = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=30.0)
        got = observed_impact(trace, study.subscribers_of(plan), target, 15.0)
        assert got == pytest.approx(res.fault_event.i_dyn, abs=0.002), target


def test_scaling_keeps_utilisation_and_scales_deadlines(tmp_path):
    src = scenario(tmp_path)
    scaled = study.write_scaled(src, 0.5, tmp_path / "scaled.json")
    base, half = build_deployment(src), build_deployment(scaled)
    assert half["topics"]["t1"]["rate_hz"] == 10.0
    assert half["topics"]["t1"]["deadline_ms"] == 1000.0
    assert half["apps"]["P"]["service_mean_s"] == pytest.approx(2 * base["apps"]["P"]["service_mean_s"])
    assert json.loads(src.read_text())["topics"][0]["frequency"] == 20  # source untouched


def test_host_load_and_kappa(tmp_path):
    plan = build_deployment(scenario(tmp_path))
    loads = study.host_loads(plan)
    assert loads == {"H1": 20 + 2 * 40, "H2": 20 + 20 + 20}
    assert study.kappa_for(plan, 50.0) == pytest.approx(0.5)
    assert study.kappa_for(plan, 500.0) == 1.0


def test_target_selection_is_stable_and_capped():
    plan = build_deployment(Path("data/scenarios/av_system.json"))
    apps = [a for a in plan["apps"] if a.startswith("A")]
    a = study.select_targets(plan, apps)
    assert a == study.select_targets(plan, apps)
    assert len(a["publishers"]) == 30 and len(a["sinks"]) <= 5
    assert not set(a["publishers"]) & set(a["sinks"])


def test_rate_expansion_is_eq7(tmp_path):
    r = study.rate_expansion(build_deployment(scenario(tmp_path)))
    assert r == {"S": 20.0, "P": 20.0, "K": 0.0}


def test_open_loop_plan_keeps_publishing_after_an_input_loss(tmp_path):
    def p_after(open_loop):
        d = orchestrate.build(scenario(tmp_path, brokers=False), tmp_path / str(open_loop), 6.0,
                              qos_semantics="dds", open_loop=open_loop)
        orchestrate.run(d, 6.0, [FaultSpec("S", "crash", 2.0)], mode="local")
        t = orchestrate.load_run(d)
        return sum(1 for e in t.events if e["kind"] == "publish" and e["app"] == "P" and e["t"] > 5.0)
    assert p_after(True) > 0


def test_component_crash_stops_every_replica(tmp_path):
    data = json.loads(scenario(tmp_path).read_text())
    data["relationships"]["runs_on"].append({"from": "S", "to": "H2"})
    src = tmp_path / "rep_system.json"
    src.write_text(json.dumps(data))
    d = orchestrate.build(src, tmp_path / "rep", 6.0)
    orchestrate.run(d, 6.0, [FaultSpec("S", "crash", 2.0)], mode="local")
    t = orchestrate.load_run(d)
    assert not [e for e in t.events if e["kind"] == "publish" and e["app"] == "S" and e["t"] > 3.0]


def test_cpu_counts_only_the_measured_window():
    cpu = study.CpuSampler()
    cpu.samples = {"saag-H1": [(0.0, 150.0), (10.0, 40.0), (20.0, 40.0), (30.0, 150.0)], "saag-H2": [(0.0, 90.0)]}
    assert cpu.p95_by_container(5.0, 25.0) == {"saag-H1": 40.0, "saag-H2": None}


def test_hosts_send_nothing_before_the_common_start(tmp_path):
    plan = build_deployment(scenario(tmp_path), open_loop=True)
    control, trace = tmp_path / "control", tmp_path / "all.jsonl"
    control.mkdir()
    agent = threading.Thread(target=run_host, args=(plan, "all", LocalTransport(), TelemetrySink(trace), 0.3),
                             kwargs={"control_dir": control, "await_start": True})
    agent.start()
    start = orchestrate._start_together(tmp_path, ["all"])
    agent.join()
    times = [json.loads(line)["time_unix_s"] for line in trace.read_text().splitlines()]
    assert times and min(times) >= start


def test_a_host_that_never_starts_is_a_timeout_not_a_hang(tmp_path, monkeypatch):
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "ready-H1").touch()
    monkeypatch.setattr(orchestrate, "READY_TIMEOUT_S", 0.0)
    with pytest.raises(TimeoutError, match="H2"):
        orchestrate._start_together(tmp_path, ["H1", "H2"])


@pytest.mark.parametrize("arm", ["S0", "S1", "S2"])
def test_every_arm_runs_through_one_estimator(arm):
    path = Path(TINY)
    g, plan = load_graph(path), build_deployment(path)
    target = next(a for a, s in plan["apps"].items() if s["publishes"])
    v = study.simulate_arm(g, plan, arm, target, 20.0, 10.0, 2.0, 42)
    assert -1.0 <= v["impact"] <= 1.0 and isinstance(v["stalled"], list)


def test_significance_decisions():
    from reproduce.fidelity_significance import holm, instrument_ok, primary

    assert primary([0.1, 0.2, 0.15, 0.3, 0.12, 0.25, 0.18, 0.22], 0.05)["outcome"] == "Better"
    assert primary([-0.1, -0.2, -0.15, -0.3, -0.12, -0.25, -0.18, -0.22], 0.05)["outcome"] == "Worse"
    near = [0.001, -0.002, 0.003, -0.004, 0.005, -0.006, 0.007, -0.008, 0.009, -0.01]
    assert primary(near, 0.05)["outcome"] == "Equivalent"
    assert primary([0.3, -0.25, 0.2, -0.28, 0.1, -0.15, 0.22, -0.2], 0.05)["outcome"] == "Inconclusive"
    assert primary([0.1, 0.2, 0.3], 0.05)["outcome"] == "Descriptive"
    res = holm({"H2": [0.1] * 3})
    assert res["H2"]["p"] is None and res["H2"]["significant"] is None
    assert instrument_ok(0.02) and not instrument_ok(0.3) and not instrument_ok(None)


def test_study_runner_refuses_until_frozen():
    from reproduce.fidelity_study import require_frozen

    with pytest.raises(SystemExit):
        require_frozen()


def test_analysis_refuses_a_partial_study(tmp_path):
    from reproduce.fidelity_significance import main

    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "COMPLETE").write_text("x")
    with pytest.raises(SystemExit, match="simulate is not complete"):
        main(tmp_path, delta=0.05)


def test_calibrated_arm_reads_a_profile(tmp_path):
    d = orchestrate.build(scenario(tmp_path), tmp_path / "tb", 8.0, qos_semantics="dds", open_loop=True)
    orchestrate.run(d, 8.0, [], mode="local")
    orchestrate.calibrate(d)
    profile = orchestrate.profile_from_json(json.loads((d / "profile.json").read_text()))
    src = Path(json.loads((d / "deployment.json").read_text())["source_scenario"])
    v = study.simulate_arm(load_graph(src), build_deployment(src), "S3", "S", 10.0, 5.0, 1.0, 42, profile)
    assert v["impact"] > 0


@pytest.mark.slow
def test_whole_pipeline_on_the_in_process_transport(tmp_path, monkeypatch):
    """Sweep, arms, analysis, end to end. The chain fixture is not in the study population."""
    from concurrent.futures import ThreadPoolExecutor

    import reproduce.fidelity_significance as fs
    import reproduce.fidelity_study as fr

    src = scenario(tmp_path)
    out = tmp_path / "res"
    monkeypatch.setattr(fr, "OUT", out)
    monkeypatch.setattr(fr, "MODE", "local")
    monkeypatch.setattr(fr, "BASE_S", 6.0)
    monkeypatch.setattr(fr, "WARM_S", 1.0)
    monkeypatch.setattr(fr, "BASELINES", 2)
    monkeypatch.setattr(fr, "CONDITIONS", {"O-open": True, "O-react": False})
    monkeypatch.setattr(fr, "scenario_path", lambda name: src)
    monkeypatch.setattr(fr, "require_frozen", lambda: None)
    monkeypatch.setattr(fr, "REPS", 3)
    monkeypatch.setattr(fr, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(study, "SIM_SEEDS", (42, 123))
    monkeypatch.setattr(study, "select_targets", lambda plan, apps, **k: {
        "publishers": ["S", "P"], "sinks": ["K"], "brokers": [], "hosts": []})
    out.mkdir()
    (out / "capacity.json").write_text(json.dumps({"chain": {"kappa": 1.0, "included": True}}))
    (out / "sources.json").write_text(json.dumps({"chain": str(src)}))

    class A:
        workers = 2
    fr.cmd_sweep(A())
    fr.cmd_simulate(A())
    res = fs.main(out, delta=0.05, min_publishers=1)
    row = res["scenarios"]["chain"]
    assert row["n_publishers"] == 2 and row["lost_targets"] == []
    assert row["steady"]["S2"]["delivery_error"] is not None
    assert "o_react_descriptive_only" in res
    assert row["dds_status"] and "S0" in row["rho"] and "S2" in row["rho"]


def _scenario_row(s0, s2, null=0.0):
    return {"lost_scenario": False, "rho": {"S0": s0, "S1": s0, "S2": s2, "R": s0},
            "null_median_rho": null, "steady": {}}


def test_report_applies_the_decision_table_and_the_instrument_check():
    from reproduce.fidelity_significance import report

    rows = {f"s{i}": _scenario_row(0.5, 0.6 + i / 100) for i in range(8)}
    res = report(rows, {}, 0.05)
    assert res["H1"]["outcome"] == "Better" and res["n_scenarios"] == 8
    assert res["secondary"]["H3"]["significant"] is True
    rows = {f"s{i}": _scenario_row(0.5, 0.6 + i / 100, null=0.4) for i in range(8)}
    res = report(rows, {}, 0.05)
    assert "withheld" in res and "H1" not in res


def test_undefined_rho_is_not_coerced_and_lost_scenarios_leave_the_contrast():
    from reproduce.fidelity_significance import _rho, contrast

    assert _rho({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 3}) is None
    rows = {"x": _scenario_row(0.5, 0.6), "y": {**_scenario_row(0.5, 0.9), "lost_scenario": True},
            "z": {**_scenario_row(0.5, 0.6), "rho": {"S0": None, "S2": 0.6}}}
    assert contrast(rows, "S2", "S0") == pytest.approx([0.1])
