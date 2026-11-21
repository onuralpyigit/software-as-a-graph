"""Tests for the two-tiered hybrid runtime simulation protocol (saag/testbed/hybrid.py)."""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("simpy")

from saag.core.graph_io import load_graph  # noqa: E402
from saag.simulation.models import RuntimeTelemetryProfile  # noqa: E402
from saag.simulation.runtime.labels import runtime_labels  # noqa: E402
from saag.testbed import orchestrate  # noqa: E402
from saag.testbed.deployment import build_deployment  # noqa: E402
from saag.testbed.hybrid import (  # noqa: E402
    HybridProtocolConfig,
    assess_probe_parity,
    load_prediction_scores,
    parse_fault_probe,
    run_hybrid_protocol,
    select_shortlist,
)

logging.disable(logging.WARNING)
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")


def _chain_scenario(tmp_path):
    """S (sensor, 20 Hz) -> t1 -> P (processor) -> t2 -> K (actuator), on two hosts."""
    data = {
        "nodes": [{"id": "H1", "name": "H1"}, {"id": "H2", "name": "H2"}],
        "brokers": [{"id": "BR", "name": "BR"}],
        "topics": [
            {
                "id": "t1", "name": "t1", "frequency": 20, "size": 64,
                "qos": {
                    "reliability": "RELIABLE",
                    "durability": "VOLATILE",
                    "transport_priority": "HIGH",
                },
                "deadline_ms": 500, "history_depth": 10,
            },
            {
                "id": "t2", "name": "t2", "frequency": 20, "size": 64,
                "qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE"},
            },
        ],
        "applications": [
            {"id": "S", "name": "S", "app_type": "sensor"},
            {"id": "P", "name": "P", "app_type": "processor"},
            {"id": "K", "name": "K", "app_type": "actuator"},
        ],
        "libraries": [],
        "relationships": {
            "publishes_to": [{"from": "S", "to": "t1"}, {"from": "P", "to": "t2"}],
            "subscribes_to": [{"from": "P", "to": "t1"}, {"from": "K", "to": "t2"}],
            "routes": [{"from": "BR", "to": "t1"}, {"from": "BR", "to": "t2"}],
            "runs_on": [
                {"from": "S", "to": "H1"},
                {"from": "BR", "to": "H1"},
                {"from": "P", "to": "H2"},
                {"from": "K", "to": "H2"},
            ],
            "connects_to": [{"from": "H1", "to": "H2"}],
            "uses": [],
        },
    }
    p = tmp_path / "chain_system.json"
    p.write_text(json.dumps(data))
    return p


def test_runtime_labels_accepts_telemetry_profile_and_is_json_serializable(tmp_path):
    g = load_graph(_chain_scenario(tmp_path))
    prof = RuntimeTelemetryProfile(
        msg_rate_per_sec={"t1": 18.0, "t2": 18.0},
        provenance={"source": "unit-test-trace"},
    )
    out = runtime_labels(
        g,
        candidates=["S", "P"],
        seeds=[42],
        settings={"duration": 4.0, "qos_semantics": "dds"},
        telemetry_profile=prof,
    )
    assert out["calibrated"] is True
    assert out["calibration_source"] == "unit-test-trace"
    assert "telemetry_profile" not in out["settings"]
    # Must serialise cleanly to JSON
    dumped = json.dumps(out)
    assert '"calibrated": true' in dumped


def test_load_prediction_scores_all_formats(tmp_path):
    # 1. Flat dict
    assert load_prediction_scores({"A": 0.9, "B": 0.2, "flag": True}) == {"A": 0.9, "B": 0.2}

    # 2. saag-predict JSON (GNN preferred over RM)
    pred_gnn = {
        "layers": {
            "system": {
                "gnn": {
                    "node_scores": {
                        "A": {"composite_score": 0.88},
                        "B": {"composite_score": 0.12},
                    }
                },
                "rm": {"A": {"overall": 0.5}, "B": {"overall": 0.4}},
            }
        }
    }
    p_gnn = tmp_path / "pred_gnn.json"
    p_gnn.write_text(json.dumps(pred_gnn))
    assert load_prediction_scores(p_gnn, layer="system") == {"A": 0.88, "B": 0.12}

    # 3. saag-predict JSON (RM fallback when no GNN)
    pred_rm = {"layers": {"system": {"rm": {"A": {"overall": 0.71}, "B": {"overall": 0.33}}}}}
    assert load_prediction_scores(pred_rm) == {"A": 0.71, "B": 0.33}

    # 4. impact_scores.json
    imp = {"records": {"A": {"impact_score": 0.65}, "B": {"impact_score": 0.0}}}
    assert load_prediction_scores(imp) == {"A": 0.65, "B": 0.0}

    # 5. runtime_labels.json
    rl = {"records": {"A": {"i_flow": {"mean": 0.77}}, "B": {"i_flow": {"mean": 0.11}}}}
    assert load_prediction_scores(rl) == {"A": 0.77, "B": 0.11}


def test_parse_fault_probe_valid_and_invalid():
    assert parse_fault_probe("crash") == {"mode": "crash"}
    assert parse_fault_probe("slowdown:duration=10,k=3") == {
        "mode": "slowdown",
        "duration_s": 10.0,
        "params": {"k": 3},
    }
    assert parse_fault_probe("omission:duration_s=5.5,p=0.4") == {
        "mode": "omission",
        "duration_s": 5.5,
        "params": {"p": 0.4},
    }
    with pytest.raises(ValueError, match="unknown fault probe mode"):
        parse_fault_probe("nonexistent_fault")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_fault_probe("omission:badparam")


def test_select_shortlist_topk_disagreements_and_controls():
    records = {
        "A": {"node_type": "Application", "i_flow": {"mean": 0.90}, "ref_flow_policy": 0.9, "ref_delivery_first_order": 0.8},
        "B": {"node_type": "Application", "i_flow": {"mean": 0.80}, "ref_flow_policy": 0.7, "ref_delivery_first_order": 0.6},
        "C": {"node_type": "Application", "i_flow": {"mean": 0.70}, "ref_flow_policy": 0.6, "ref_delivery_first_order": 0.5},
        "D": {"node_type": "Application", "i_flow": {"mean": 0.10}, "ref_flow_policy": 0.1, "ref_delivery_first_order": 0.1},
        "SINK": {"node_type": "Application", "i_flow": {"mean": 0.02}, "ref_flow_policy": 0.0, "ref_delivery_first_order": 0.0},
    }
    # Predictions rank A highest, D artificially high (disagrees with Tier 1), C artificially low
    preds = {"A": 0.95, "D": 0.85, "B": 0.50, "SINK": 0.20, "C": 0.05}

    sl = select_shortlist(
        records,
        predictions=preds,
        label="i_flow",
        top_k=2,
        n_disagreements=1,
        n_controls=1,
        sinks=["SINK"],
    )
    assert sl["primary_targets"][:2] == ["A", "D"]
    # C has the largest rank gap among remaining candidates (Tier 1 rank 3rd vs pred rank last)
    assert "C" in sl["primary_targets"]
    assert sl["control_targets"] == ["SINK"]
    by_id = {t["node_id"]: t for t in sl["targets"]}
    assert "top_k_predicted" in by_id["A"]["reasons"]
    assert "disagreement" in by_id["C"]["reasons"]
    assert by_id["SINK"]["is_control"] is True
    assert sl["tier1_diagnostics"]["n_predictions_matched"] == 5

    # Without predictions: ranks by Tier 1 directly
    sl_no_pred = select_shortlist(records, predictions=None, label="i_flow", top_k=2, n_controls=1, sinks=["SINK"])
    assert sl_no_pred["primary_targets"] == ["A", "B"]
    assert sl_no_pred["control_targets"] == ["SINK"]


def test_assess_probe_parity_flags_unsupported_and_caveats(tmp_path):
    plan = build_deployment(_chain_scenario(tmp_path), qos_semantics="dds")

    # Local mode: link_flap and buffer_leak are unsupported; reactive + kappa < 1 trigger caveats
    parity_local = assess_probe_parity(
        [{"mode": "crash"}, {"mode": "buffer_leak"}, {"mode": "link_flap"}],
        plan,
        mode="local",
        kappa=0.5,
        behavior_model="reactive",
        shortlist_targets=["S", "P"],
    )
    assert [u["probe"]["mode"] for u in parity_local["unsupported"]] == ["buffer_leak", "link_flap"]
    codes = {c["code"] for c in parity_local["caveats"]}
    assert {
        "local_mode_no_dds_transport",
        "circular_reactive_policies",
        "rate_scaled_latency_caveat",
        "unsupported_probes",
    } <= codes

    # Docker mode with open_loop: link_flap supported (with rediscovery lag caveat),
    # slowdown on station-less sensor S flags stationless divergence, no circularity caveat
    parity_docker = assess_probe_parity(
        [{"mode": "link_flap"}, {"mode": "slowdown", "params": {"k": 2.0}}],
        plan,
        mode="docker",
        kappa=1.0,
        behavior_model="open_loop",
        shortlist_targets=["S", "P"],
    )
    assert parity_docker["unsupported"] == []
    docker_codes = {c["code"] for c in parity_docker["caveats"]}
    assert "link_flap_rediscovery_lag" in docker_codes
    assert "slowdown_stationless_divergence" in docker_codes
    assert "circular_reactive_policies" not in docker_codes


def test_simulate_like_respects_open_loop_plan(tmp_path):
    scen = _chain_scenario(tmp_path)
    tb_dir = tmp_path / "tb_open"
    orchestrate.build(scen, tb_dir, duration_s=3.0, qos_semantics="dds", open_loop=True)
    orchestrate.run(tb_dir, duration_s=3.0, faults=[], mode="local", seed=1)
    tr = orchestrate.simulate_like(tb_dir, calibrated=False)
    assert tr.events, "open_loop simulate_like must produce events"


def test_hybrid_protocol_end_to_end_local(tmp_path):
    scen = _chain_scenario(tmp_path)
    out_dir = tmp_path / "hybrid_out"
    preds = {"S": 0.95, "P": 0.60, "BR": 0.85, "K": 0.05}
    cfg = HybridProtocolConfig(
        duration_s=3.0,
        warmup_s=0.5,
        tier1_duration_s=4.0,
        tier1_seeds=(42, 123),
        tier1_label="i_flow",
        behavior_model="reactive",
        qos_semantics="dds",
        mode="local",
        top_k=2,
        n_disagreements=1,
        n_controls=1,
        repetitions=1,
        base_seed=42,
        fault_probes=(
            {"mode": "crash"},
            {"mode": "omission", "duration_s": 1.0, "params": {"p": 0.5}},
            {"mode": "buffer_leak", "params": {"beta": 1.0}},
        ),
    )

    report = run_hybrid_protocol(scen, out_dir, config=cfg, predictions=preds)

    assert report["report_kind"] == "tier2_advisory_hybrid_verification"
    assert (out_dir / "hybrid_report.json").exists()
    assert (out_dir / "tier1_runtime_labels.json").exists()
    assert (out_dir / "shortlist.json").exists()
    assert (out_dir / "testbed" / "profile.json").exists()

    # Step 0 calibrated and produced steady-state comparison
    assert report["step0_calibration"]["calibrated"] is True
    assert report["step0_calibration"]["steady_state"]["metrics"]["topic_rate"]["n_compared"] >= 2

    # Step 1 ran calibrated Tier 1 sweep
    assert report["step1_tier1_screening"]["calibrated"] is True
    assert report["step1_tier1_screening"]["n_candidates"] >= 3

    # Step 2 selected primary targets + 1 disjoint negative control; K (predicted 0.05,
    # Tier-1 i_flow ~ 1.0 as sole sink) is surfaced as a prediction/Tier-1 disagreement.
    primary = report["step2_shortlist"]["primary_targets"]
    controls = report["step2_shortlist"]["control_targets"]
    assert len(primary) >= 2
    assert len(controls) == 1
    assert set(primary).isdisjoint(controls)
    by_id = {t["node_id"]: t for t in report["step2_shortlist"]["targets"]}
    assert "disagreement" in by_id["K"]["reasons"]

    # Step 3 verified crash and omission, skipped unsupported buffer_leak
    probes = report["step3_tier2_verification"]["probes"]
    assert "crash" in probes
    assert any(pid.startswith("omission") for pid in probes)
    unsupported = report["step3_tier2_verification"]["unsupported_probes"]
    assert [u["probe"]["mode"] for u in unsupported] == ["buffer_leak"]

    # Crash probe has primary targets and separate controls
    crash_rep = probes["crash"]
    assert crash_rep["n_primary_targets"] >= 2
    assert set(crash_rep["controls"]) == set(controls)
    for t_entry in crash_rep["per_target"].values():
        assert 0.0 <= t_entry["stall_jaccard_mean"] <= 1.0
        assert "dds_status" in t_entry


def test_cli_hybrid_verify_and_runtime_sweep_profile_parse():
    from cli.simulate_graph import _build_parser

    parser = _build_parser()
    rs_args = parser.parse_args([
        "runtime-sweep", "--input", "data/scenarios/tiny_system.json",
        "--profile", "output/testbed/tiny/profile.json",
    ])
    assert rs_args.profile == "output/testbed/tiny/profile.json"

    hv_args = parser.parse_args([
        "hybrid-verify",
        "--input", "data/scenarios/tiny_system.json",
        "--output", "output/hybrid/tiny",
        "--predictions", "output/predictions.json",
        "--mode", "docker",
        "--open-loop",
        "--fault-probe", "crash",
        "--fault-probe", "omission:duration=10,p=0.5",
        "--repetitions", "2",
        "--top-k", "4",
    ])
    assert hv_args.subcommand == "hybrid-verify"
    assert hv_args.mode == "docker"
    assert hv_args.open_loop is True
    assert hv_args.fault_probe == ["crash", "omission:duration=10,p=0.5"]
    assert hv_args.repetitions == 2
    assert hv_args.top_k == 4
