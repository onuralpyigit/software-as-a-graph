"""Tests for end-of-simulation operational health-status summary and derived criticality."""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("simpy")

from saag.simulation.calibration import TelemetryTrace, make_event  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.testbed.deployment import build_deployment  # noqa: E402
from saag.testbed.hybrid import HybridProtocolConfig, run_hybrid_protocol  # noqa: E402
from saag.validation.health import (  # noqa: E402
    COMPONENT_TYPES,
    CRITICALITY_LEVELS,
    HealthThresholds,
    compare_health_summaries,
    derive_operational_criticality,
    load_health_schema,
    summarize_system_health,
)

logging.disable(logging.WARNING)


def _validate_against_schema(payload: dict) -> None:
    """Validate *payload* against health_summary.schema.json (jsonschema if installed + strict checks)."""
    schema = load_health_schema()
    assert payload["schema_version"] == "1.0"
    assert set(payload.keys()) == set(schema["required"])
    assert set(payload["system_health"]["by_type"].keys()) == set(COMPONENT_TYPES)
    assert set(payload["system_health"]["by_criticality"].keys()) == set(CRITICALITY_LEVELS)
    assert set(payload["system_health"]["operational_critical"]["by_type"].keys()) == set(COMPONENT_TYPES)

    # Check count conservation
    sh = payload["system_health"]
    assert sh["counts"]["total"] == (
        sh["counts"]["healthy"] + sh["counts"]["degraded"] + sh["counts"]["lost"]
    )
    assert sh["counts"]["total"] == len(payload["components"])
    assert sum(b["counts"]["total"] for b in sh["by_type"].values()) == sh["counts"]["total"]
    assert sum(b["counts"]["total"] for b in sh["by_criticality"].values()) == sh["counts"]["total"]

    # Optional jsonschema Draft 2020-12 validation when available
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=payload, schema=schema)
    except ImportError:
        pass


def _switched_critical_scenario(tmp_path):
    """Two-host switched topology with CRITICAL/HIGH/LOW entities, broker, and NetworkDevice."""
    data = {
        "metadata": {"domain": "av"},
        "nodes": [
            {"id": "H1", "name": "H1"},
            {"id": "H2", "name": "H2"},
            {"id": "H3", "name": "H3"},
        ],
        "network_devices": [
            {"id": "SW1", "name": "SW1", "role": "switch"},
            {"id": "SW_ISO", "name": "SW_ISO", "role": "switch"},
        ],
        "brokers": [
            {"id": "BR1", "name": "BR1"},
            {"id": "BR2", "name": "BR2"},
        ],
        "topics": [
            {
                "id": "t_crit", "name": "t_crit", "frequency": 10, "size": 64,
                "criticality": "CRITICAL",
                "qos": {"reliability": "RELIABLE", "durability": "VOLATILE", "transport_priority": "HIGH"},
                "deadline_ms": 500, "history_depth": 10,
            },
            {
                "id": "t_low", "name": "t_low", "frequency": 10, "size": 64,
                "criticality": "LOW",
                "qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE"},
            },
        ],
        "applications": [
            {"id": "S", "name": "S", "app_type": "sensor", "criticality": "HIGH"},
            {"id": "P", "name": "P", "app_type": "processor", "criticality": "CRITICAL"},
            {"id": "K", "name": "K", "app_type": "actuator", "criticality": "LOW"},
        ],
        "libraries": [],
        "relationships": {
            "publishes_to": [{"from": "S", "to": "t_crit"}, {"from": "P", "to": "t_low"}],
            "subscribes_to": [{"from": "P", "to": "t_crit"}, {"from": "K", "to": "t_low"}],
            "routes": [
                {"from": "BR1", "to": "t_crit"},
                {"from": "BR2", "to": "t_low"},
            ],
            "runs_on": [
                {"from": "S", "to": "H1"},
                {"from": "BR1", "to": "H1"},
                {"from": "P", "to": "H2"},
                {"from": "P", "to": "H3"},   # Replicated processor
                {"from": "BR2", "to": "H3"},
                {"from": "K", "to": "H3"},
            ],
            "connects_to": [
                {"from": "H1", "to": "SW1"},
                {"from": "H2", "to": "SW1"},
                {"from": "H3", "to": "SW1"},
                {"from": "H3", "to": "SW_ISO"},
            ],
            "uses": [],
        },
    }
    p = tmp_path / "switched_crit.json"
    p.write_text(json.dumps(data))
    return p


def _synthetic_healthy_trace(plan, duration_s: float = 10.0) -> TelemetryTrace:
    """Generate steady 10 Hz publish/deliver/forward events for *plan*."""
    events = []
    dt = 0.1
    t = 0.05
    mid = 0
    while t < duration_s:
        mid += 1
        events.append(make_event(t, "saag.publish", "S", "t_crit", **{"messaging.message.id": mid}))
        events.append(make_event(t + 0.002, "saag.forward", "BR1", "t_crit", service_ms=1.0))
        events.append(make_event(t + 0.005, "saag.deliver", "P", "t_crit", latency_ms=5.0, service_ms=2.0))
        events.append(make_event(t, "saag.publish", "P", "t_low", **{"messaging.message.id": mid}))
        events.append(make_event(t + 0.002, "saag.forward", "BR2", "t_low", service_ms=1.0))
        events.append(make_event(t + 0.005, "saag.deliver", "K", "t_low", latency_ms=5.0, service_ms=2.0))
        t += dt
    return TelemetryTrace.from_records(events, start_s=0.0, end_s=duration_s)


def test_derive_operational_criticality_all_five_types(tmp_path):
    plan = build_deployment(_switched_critical_scenario(tmp_path), network_gating=True)

    # Default: declared-first for Applications
    crit = derive_operational_criticality(plan, escalate_app_from_topics=False)
    assert crit["t_crit"]["level"] == "CRITICAL"
    assert crit["t_crit"]["provenance"] == "declared"
    assert crit["S"]["level"] == "HIGH"
    assert crit["S"]["provenance"] == "declared"

    # Brokers inherit from routed topics
    assert crit["BR1"]["level"] == "CRITICAL"
    assert crit["BR1"]["provenance"] == "derived"
    assert "t_crit" in crit["BR1"]["derived_from"]
    assert crit["BR2"]["level"] == "LOW"

    # Nodes inherit from resident apps/brokers/topics
    assert crit["H1"]["level"] == "CRITICAL"
    assert crit["H2"]["level"] == "CRITICAL"
    assert crit["H3"]["level"] == "CRITICAL"  # hosts replica of CRITICAL P

    # NetworkDevice SW1 carries t_crit (H1 -> SW1 -> H2) => CRITICAL
    assert crit["SW1"]["level"] == "CRITICAL"
    assert crit["SW1"]["provenance"] == "derived"
    assert "t_crit" in crit["SW1"]["derived_from"]
    # SW_ISO carries no flows, falls back to adjacent host H3 => CRITICAL
    assert crit["SW_ISO"]["level"] == "CRITICAL"
    assert crit["SW_ISO"]["derived_from"] == ["H3"]

    # With escalate_app_from_topics=True, S (HIGH) escalates to CRITICAL via t_crit
    crit_esc = derive_operational_criticality(plan, escalate_app_from_topics=True)
    assert crit_esc["S"]["level"] == "CRITICAL"
    assert crit_esc["S"]["provenance"] == "derived"
    assert "t_crit" in crit_esc["S"]["derived_from"]


def test_healthy_baseline_summary_and_schema(tmp_path):
    plan = build_deployment(_switched_critical_scenario(tmp_path), network_gating=True)
    trace = _synthetic_healthy_trace(plan, duration_s=10.0)

    summary = summarize_system_health(trace, plan, warmup_s=1.0)
    _validate_against_schema(summary)

    sh = summary["system_health"]
    assert sh["counts"]["lost"] == 0
    assert sh["counts"]["degraded"] == 0
    assert sh["counts"]["healthy"] == len(summary["components"])
    assert sh["by_type"]["Node"]["counts"]["healthy"] == 3
    assert sh["by_type"]["Application"]["counts"]["healthy"] == 3
    assert sh["by_type"]["Topic"]["counts"]["healthy"] == 2
    assert sh["by_type"]["Broker"]["counts"]["healthy"] == 2
    assert sh["by_type"]["NetworkDevice"]["counts"]["healthy"] == 2

    piu = summary["product_in_use"]
    assert piu["domain_context"]["domain"] == "av"
    assert piu["domain_context"]["derived"] is True
    assert piu["domain_context"]["omega"]["freedom_from_risk"] == 0.65
    assert piu["freedom_from_risk"]["zero_tolerance_critical_loss"] is False
    assert piu["overall_qiu_health_index"] == 1.0


def test_app_crash_cascade_and_critical_loss(tmp_path):
    plan = build_deployment(_switched_critical_scenario(tmp_path), network_gating=True)
    # Pre-fault [0, 5): healthy; post-fault [5, 10]: S crashes, t_crit dies, P stalls, t_low dies
    events = []
    t = 0.05
    mid = 0
    while t < 5.0:
        mid += 1
        events.append(make_event(t, "saag.publish", "S", "t_crit", **{"messaging.message.id": mid}))
        events.append(make_event(t + 0.002, "saag.forward", "BR1", "t_crit", service_ms=1.0))
        events.append(make_event(t + 0.005, "saag.deliver", "P", "t_crit", latency_ms=5.0, service_ms=2.0))
        events.append(make_event(t, "saag.publish", "P", "t_low", **{"messaging.message.id": mid}))
        events.append(make_event(t + 0.002, "saag.forward", "BR2", "t_low", service_ms=1.0))
        events.append(make_event(t + 0.005, "saag.deliver", "K", "t_low", latency_ms=5.0, service_ms=2.0))
        t += 0.1
    trace = TelemetryTrace.from_records(events, start_s=0.0, end_s=10.0)

    summary = summarize_system_health(
        trace, plan, fault_time=5.0, warmup_s=0.5, faults=[FaultSpec("S", "crash", 5.0)],
    )
    _validate_against_schema(summary)

    comps = summary["components"]
    assert comps["S"]["state"] == "lost"
    assert comps["t_crit"]["state"] == "lost"
    assert comps["P"]["state"] == "lost"
    assert comps["t_low"]["state"] == "lost"
    assert comps["K"]["state"] == "lost"

    piu = summary["product_in_use"]
    assert piu["freedom_from_risk"]["zero_tolerance_critical_loss"] is True
    assert piu["freedom_from_risk"]["lost_critical_count"] >= 2  # P and t_crit
    assert piu["freedom_from_risk"]["critical_safety_exposure_s"] >= 4.0
    assert piu["overall_qiu_health_index"] < 0.6


def test_host_crash_with_replica_failover_and_cpu_saturation(tmp_path):
    plan = build_deployment(_switched_critical_scenario(tmp_path), network_gating=True)
    # H2 crashes at t=5; P fails over to H3 (takeover event) and keeps emitting/receiving
    base_tr = _synthetic_healthy_trace(plan, duration_s=10.0)
    events = [
        {"time_unix_s": e["t"], "name": f"saag.{e['kind']}", "attributes": {
            "service.name": e["app"], "messaging.destination.name": e["topic"],
            "saag.latency_ms": e["latency_ms"], "saag.service_ms": e["service_ms"],
        }}
        for e in base_tr.events
    ]
    events.append(make_event(5.2, "saag.state", "P", state="takeover", host="H3"))
    trace = TelemetryTrace.from_records(events, start_s=0.0, end_s=10.0)

    summary = summarize_system_health(
        trace, plan, fault_time=5.0, warmup_s=0.5,
        faults=[FaultSpec("H2", "crash", 5.0)],
        docker_cpu_p95={"saag-H1": 45.0, "saag-H3": 92.5},
    )
    _validate_against_schema(summary)

    comps = summary["components"]
    assert comps["H2"]["state"] == "lost"
    assert comps["H3"]["state"] == "degraded"
    assert any(r.startswith("cpu_saturated:") for r in comps["H3"]["reasons"])
    assert comps["P"]["state"] == "degraded"
    assert "redundancy_depleted" in comps["P"]["reasons"]
    assert comps["P"]["metrics"]["redundancy_initial"] == 1
    assert comps["P"]["metrics"]["redundancy_remaining"] == 0
    assert comps["P"]["metrics"]["failover_occurred"] is True
    assert "P" in summary["product_in_use"]["freedom_from_risk"]["n_minus_0_critical_ids"]
    assert summary["system_health"]["resilience"]["failover_takeover_count"] == 1


def test_network_device_crash_and_comparison(tmp_path):
    plan = build_deployment(_switched_critical_scenario(tmp_path), network_gating=True)
    trace = _synthetic_healthy_trace(plan, duration_s=10.0)

    s_healthy = summarize_system_health(trace, plan, warmup_s=1.0)
    s_net_fail = summarize_system_health(
        trace, plan, fault_time=5.0, warmup_s=1.0,
        faults=[FaultSpec("SW1", "crash", 5.0)],
    )
    _validate_against_schema(s_net_fail)
    assert s_net_fail["components"]["SW1"]["state"] == "lost"
    assert s_net_fail["components"]["H1"]["state"] == "lost"  # isolated under network_gating

    cmp_same = compare_health_summaries(s_healthy, s_healthy)
    assert cmp_same["exact_state_agreement"] == 1.0
    assert cmp_same["operational_critical_lost_jaccard"] == 1.0
    assert cmp_same["qiu_health_index_abs_error"] == 0.0

    cmp_diff = compare_health_summaries(s_healthy, s_net_fail)
    assert cmp_diff["exact_state_agreement"] < 1.0
    assert cmp_diff["operational_critical_lost_jaccard"] == 0.0


def test_hybrid_protocol_populates_health_summaries(tmp_path):
    scen = _switched_critical_scenario(tmp_path)
    out_dir = tmp_path / "hybrid_health_out"
    cfg = HybridProtocolConfig(
        duration_s=3.0,
        warmup_s=0.5,
        tier1_duration_s=4.0,
        tier1_seeds=(42,),
        behavior_model="reactive",
        qos_semantics="dds",
        mode="local",
        top_k=1,
        n_disagreements=0,
        n_controls=1,
        repetitions=1,
        base_seed=42,
        fault_probes=({"mode": "crash"},),
    )
    report = run_hybrid_protocol(scen, out_dir, config=cfg)

    bh = report["step0_calibration"]["baseline_health"]
    _validate_against_schema(bh["observed"])
    _validate_against_schema(bh["simulated"])
    assert 0.0 <= bh["comparison"]["exact_state_agreement"] <= 1.0

    crash_targets = report["step3_tier2_verification"]["probes"]["crash"]["per_target"]
    assert crash_targets
    for entry in crash_targets.values():
        _validate_against_schema(entry["health_summary"]["observed"])
        _validate_against_schema(entry["health_summary"]["simulated"])
        assert "exact_state_agreement" in entry["health_agreement"]


def test_tier2_runtime_emits_state_transitions_with_criticality_and_queue_peak(tmp_path):
    from saag.testbed import orchestrate

    scen = _switched_critical_scenario(tmp_path)
    tb_dir = tmp_path / "tb_obs"
    orchestrate.build(scen, tb_dir, duration_s=3.0, qos_semantics="dds")
    orchestrate.run(
        tb_dir, duration_s=3.0, faults=[FaultSpec("S", "crash", 1.5)], mode="local", seed=7
    )

    raw_lines = [
        json.loads(line)
        for line in (tb_dir / "telemetry" / "all.jsonl").read_text().splitlines()
        if line.strip()
    ]
    state_events = [r for r in raw_lines if r.get("name") == "saag.health"]
    assert state_events, "Tier-2 agents must emit saag.health transitions"
    s_states = [r["attributes"]["saag.state"] for r in state_events if r["attributes"].get("service.name") == "S"]
    assert "crashed" in s_states
    s_crits = {r["attributes"].get("saag.criticality") for r in state_events if r["attributes"].get("service.name") == "S"}
    assert "HIGH" in s_crits
    br1_crits = {r["attributes"].get("saag.criticality") for r in state_events if r["attributes"].get("service.name") == "BR1"}
    assert "CRITICAL" in br1_crits

    peak_events = [
        r for r in raw_lines
        if r.get("name") == "saag.status" and r["attributes"].get("saag.status") == "cache_samples_peak"
    ]
    assert peak_events, "Tier-2 AppRuntime must emit cache_samples_peak status events"


def test_tier2_observed_operational_criticality_and_orchestrate_health_fidelity(tmp_path):
    from saag.testbed import orchestrate
    from saag.validation.health import (
        compare_operational_criticality,
        format_health_report,
        format_operational_criticality_table,
    )

    scen = _switched_critical_scenario(tmp_path)
    tb_dir = tmp_path / "tb_health"
    orchestrate.build(scen, tb_dir, duration_s=3.0, qos_semantics="dds")
    orchestrate.run(
        tb_dir, duration_s=3.0, faults=[FaultSpec("S", "crash", 1.5)], mode="local", seed=11
    )

    h_rep = orchestrate.health(tb_dir, warmup_s=0.3)
    assert (tb_dir / "health_summary.json").exists()
    assert "operational_criticality" in h_rep
    s_oc = h_rep["operational_criticality"]["S"]
    assert s_oc["target"] == "S"
    assert s_oc["static_criticality"]["level"] == "HIGH"
    assert s_oc["i_flow"] > 0.5
    assert s_oc["i_delivery"] > 0.3
    assert s_oc["qiu_impact"] > 0.0

    # Formatters render cleanly
    txt = format_health_report(h_rep, only_unhealthy=True)
    assert "Tier-2 Operational Health Summary" in txt
    assert "Product-in-Use (ISO 25019)" in txt
    oc_tbl = format_operational_criticality_table([s_oc], sort_by="i_flow")
    assert "i_flow" in oc_tbl and "S" in oc_tbl

    # orchestrate.fidelity includes health_summary + operational_criticality
    fid = orchestrate.fidelity(tb_dir, calibrated=False, warmup_s=0.3)
    assert "health_summary" in fid
    assert "operational_criticality" in fid
    assert "S" in fid["operational_criticality"]["observed"]

    # compare_operational_criticality works on multi-target dicts
    cmp = compare_operational_criticality(
        {"A": {"i_flow": 0.9, "i_delivery": 0.8}, "B": {"i_flow": 0.5, "i_delivery": 0.4}, "C": {"i_flow": 0.1, "i_delivery": 0.1}},
        {"A": {"i_flow": 0.85, "i_delivery": 0.75}, "B": {"i_flow": 0.45, "i_delivery": 0.35}, "C": {"i_flow": 0.1, "i_delivery": 0.05}},
    )
    assert cmp["i_flow"]["spearman"] == 1.0
    assert cmp["i_delivery"]["mae"] == pytest.approx(0.05, abs=1e-6)


def test_hybrid_protocol_writes_tier2_operational_criticality_ranking_and_cli_flags(tmp_path):
    from cli.simulate_graph import _build_parser

    scen = _switched_critical_scenario(tmp_path)
    out_dir = tmp_path / "hybrid_t2_oc"
    cfg = HybridProtocolConfig(
        duration_s=3.0,
        warmup_s=0.5,
        tier1_duration_s=4.0,
        tier1_seeds=(42,),
        behavior_model="reactive",
        qos_semantics="dds",
        mode="local",
        top_k=2,
        n_disagreements=0,
        n_controls=1,
        repetitions=1,
        base_seed=42,
        fault_probes=({"mode": "crash"},),
    )
    report = run_hybrid_protocol(scen, out_dir, config=cfg)

    t2_dir = out_dir / "tier2_health"
    assert (t2_dir / "baseline.json").exists()
    assert (t2_dir / "operational_criticality_ranking.json").exists()

    crash_probe = report["step3_tier2_verification"]["probes"]["crash"]
    oc = crash_probe["operational_criticality"]
    assert oc["sort_by"] == "i_flow"
    assert len(oc["ranking"]) >= 2
    assert "i_flow" in oc["tier1_vs_tier2"]
    assert "i_delivery" in oc["replay_vs_tier2"]
    for target, entry in crash_probe["per_target"].items():
        assert "operational_criticality" in entry
        assert "observed" in entry["operational_criticality"]
        assert (t2_dir / f"crash_{target}.json").exists()

    # CLI parser accepts new testbed health and hybrid-verify --show-health flags
    parser = _build_parser()
    ns_tb = parser.parse_args(["testbed", "health", "--output", str(out_dir), "--only-critical", "--json"])
    assert ns_tb.action == "health"
    assert ns_tb.only_critical is True
    assert ns_tb.json is True
    ns_hv = parser.parse_args(["hybrid-verify", "--input", str(scen), "--output", str(out_dir), "--show-health"])
    assert ns_hv.show_health is True
