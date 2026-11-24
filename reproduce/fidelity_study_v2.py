#!/usr/bin/env python3
"""
Runner for Study 2 (docs/research/fidelity/PREREGISTRATION_V2.md).

Targets the independent rows of Table 7 across n=13 full-rate (kappa=1) scenarios
plus 1 scaled OSS scenario (n=14 for non-latency contrasts):
  - H2.1: Steady-state p50 latency fidelity (S0 vs S2, kappa=1)
  - H2.2: Infrastructure (broker + host) crash ranking & stall sets
  - H2.3: Perturbed-counterpart S2M calibration (S2 vs S3)
  - H2.4: Gray-failure (omission p=0.5) impact fidelity + 10 s partition rediscovery lag

Refuses to run Docker sweeps until PREREGISTRATION_V2.md has '**Status: FROZEN'
unless --allow-unfrozen is passed for local/pilot dry runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reproduce.calibration_drift_eval import (  # noqa: E402
    _run_sim,
    _trim,
    perturb_plan,
    resolve_scenario_path,
)
from reproduce.fidelity_significance_v2 import analyze_study2  # noqa: E402
from saag.core.graph_io import load_graph  # noqa: E402
from saag.simulation.calibration import TelemetryCalibrator  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.testbed import orchestrate, study  # noqa: E402
from saag.testbed.collect import load_run  # noqa: E402
from saag.testbed.compose import write_testbed  # noqa: E402
from saag.testbed.deployment import build_deployment  # noqa: E402
from saag.validation.fidelity import (  # noqa: E402
    compare_steady_state,
    observed_impact,
    stalled_in_trace,
)

logging.disable(logging.WARNING)

PREREG_V2 = ROOT / "docs" / "research" / "fidelity" / "PREREGISTRATION_V2.md"
OUT_V2 = ROOT / "results" / "fidelity_study_v2"
STUDY2_SEED = 20261124

STUDY2_SCENARIOS: Dict[str, float] = {
    "iot_smart_city": 1.0,
    "healthcare": 1.0,
    "hub_and_spoke": 1.0,
    "microservices": 1.0,
    "telecom_ran": 1.0,
    "realtime_gaming": 1.0,
    "logistics_fleet": 1.0,
    "atm": 1.0,
    "industrial_scada": 1.0,
    "realworld_autoware_ros2": 1.0,
    "realworld_edgex": 1.0,
    "realworld_homeassistant": 1.0,
    "realworld_trainticket": 1.0,
    "realworld_cloud_microservices": 0.7437469347719471,
}


def require_frozen(allow_unfrozen: bool = False) -> None:
    if allow_unfrozen:
        return
    if not any(l.startswith("**Status: FROZEN") for l in PREREG_V2.read_text().splitlines()):
        raise SystemExit(
            f"{PREREG_V2.name} is not frozen; freeze it before running Study 2 "
            "(or pass --allow-unfrozen for local dry-runs)."
        )


def top_fanout_publishers(plan: Mapping[str, Any], k: int = 3) -> List[str]:
    """Select top-k publishing applications by declared first-order rate expansion (deterministic)."""
    scores = study.rate_expansion(plan)
    pubs = [a for a, s in plan["apps"].items() if s.get("publishes")]
    pubs.sort(key=lambda a: (-scores.get(a, 0.0), a))
    return pubs[:k]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def run_scenario_v2(name: str, kappa: float, mode: str, duration_s: float, warmup_s: float,
                    reps: int, out_dir: Path) -> Dict[str, Any]:
    t0 = time.time()
    src = resolve_scenario_path(name)
    scaled_src = study.write_scaled(src, kappa, out_dir / "scaled" / f"{name}_k{kappa:.4g}.json")
    graph = load_graph(scaled_src)
    dur = duration_s / kappa
    warm = warmup_s / kappa
    fault_s = dur / 2.0

    plan = build_deployment(scaled_src, open_loop=True, qos_semantics="dds")
    plan["source_scenario"] = str(scaled_src)
    tb_nom = write_testbed(plan, out_dir / name / "tb_nominal", dur, seed=STUDY2_SEED)

    # ── H2.1: Steady-state p50/p95 latency (S0 vs S2) ──
    sim_s0 = _trim(study.simulate_fault_free(graph, plan, "S0", dur, seed=42), warm)
    sim_s2 = _trim(study.simulate_fault_free(graph, plan, "S2", dur, seed=42), warm)
    s0_p50, s2_p50, s2_p95 = [], [], []
    for r in range(reps):
        orchestrate.run(tb_nom, dur, [], mode=mode, seed=STUDY2_SEED + 10 + r)
        obs = _trim(load_run(tb_nom), warm)
        c0 = compare_steady_state(sim_s0, obs)
        c2 = compare_steady_state(sim_s2, obs)
        if c0["metrics"]["latency_p50"]["median_rel_error"] is not None:
            s0_p50.append(c0["metrics"]["latency_p50"]["median_rel_error"])
        if c2["metrics"]["latency_p50"]["median_rel_error"] is not None:
            s2_p50.append(c2["metrics"]["latency_p50"]["median_rel_error"])
        if c2["metrics"]["latency_p95"]["median_rel_error"] is not None:
            s2_p95.append(c2["metrics"]["latency_p95"]["median_rel_error"])

    # ── H2.2: Infrastructure (broker + host) crashes ──
    infra_targets = sorted(plan["brokers"]) + sorted(plan["hosts"])
    subs = study.subscribers_of(plan)
    obs_imp: Dict[str, float] = {}
    sim_imp: Dict[str, float] = {}
    stall_js: List[float] = []
    for tgt in infra_targets:
        if mode == "local" and tgt in plan["hosts"]:
            continue  # local mode cannot docker-kill hosts
        faults = [FaultSpec(tgt, "crash", fault_s)]
        failed = study._failed_with(plan, tgt)
        obs_vals = []
        obs_stalled_union: set = set()
        for r in range(reps):
            orchestrate.run(tb_nom, dur, faults, mode=mode, seed=STUDY2_SEED + 100 + r)
            tr_o = load_run(tb_nom)
            obs_vals.append(observed_impact(tr_o, subs, tgt, fault_s, warmup_s=warm, failed=failed))
            obs_stalled_union |= (stalled_in_trace(tr_o, fault_s) - {tgt})
        obs_imp[tgt] = statistics.mean(obs_vals)
        sim_r = study.simulate_arm(graph, plan, "S2", tgt, dur, fault_s, warm, seed=42)
        sim_imp[tgt] = sim_r["impact"]
        stall_js.append(jaccard(set(sim_r["stalled"]), obs_stalled_union))

    infra_rho = None
    if len(obs_imp) >= 3:
        keys = sorted(obs_imp)
        r_val = spearmanr([sim_imp[k] for k in keys], [obs_imp[k] for k in keys]).correlation
        infra_rho = None if r_val != r_val else float(r_val)

    # ── H2.3: Perturbed-counterpart calibration (S2 vs S3) ──
    plan_pert, _ = perturb_plan(plan)
    tb_pert = write_testbed(plan_pert, out_dir / name / "tb_perturbed", dur, seed=STUDY2_SEED + 200)
    orchestrate.run(tb_pert, dur, [], mode=mode, seed=STUDY2_SEED + 201)
    obs_cal = _trim(load_run(tb_pert), warm)
    prof_s3 = TelemetryCalibrator(warmup_s=0.0, infer_policies=False).calibrate(obs_cal, source="study2_cal")
    sim_s3 = _trim(_run_sim(graph, plan, dur, seed=42, profile=prof_s3), warm)

    orchestrate.run(tb_pert, dur, [], mode=mode, seed=STUDY2_SEED + 202)
    obs_eval = _trim(load_run(tb_pert), warm)
    pert_s2_deliv = compare_steady_state(sim_s2, obs_eval)["metrics"]["delivery_rate"]["median_rel_error"]
    pert_s3_deliv = compare_steady_state(sim_s3, obs_eval)["metrics"]["delivery_rate"]["median_rel_error"]

    # ── H2.4: Gray-failure omission (p=0.5) on top-3 publishers ──
    top3 = top_fanout_publishers(plan, k=3)
    err_s2_om, err_crash_om = [], []
    for pub in top3:
        faults_om = [FaultSpec(pub, "omission", fault_s, duration_s=dur - fault_s, params={"p": 0.5})]
        orchestrate.run(tb_nom, dur, faults_om, mode=mode, seed=STUDY2_SEED + 300)
        tr_o = load_run(tb_nom)
        imp_o = observed_impact(tr_o, subs, pub, fault_s, warmup_s=warm)

        tr_s_om = _run_sim(graph, plan, dur, seed=42, faults=faults_om)
        imp_s_om = observed_impact(tr_s_om, subs, pub, fault_s, warmup_s=warm)

        imp_crash = study.simulate_arm(graph, plan, "S2", pub, dur, fault_s, warm, seed=42)["impact"]
        err_s2_om.append(abs(imp_s_om - imp_o))
        err_crash_om.append(abs(imp_crash - imp_o))

    return {
        "scenario": name,
        "kappa": kappa,
        "wall_s": round(time.time() - t0, 2),
        "latency_p50_err_S0": statistics.median(s0_p50) if s0_p50 else None,
        "latency_p50_err_S2": statistics.median(s2_p50) if s2_p50 else None,
        "latency_p95_err_S2": statistics.median(s2_p95) if s2_p95 else None,
        "infra_n_targets": len(obs_imp),
        "infra_rho_S2": infra_rho,
        "infra_stall_J_S2": statistics.median(stall_js) if stall_js else None,
        "perturbed_delivery_err_S2": pert_s2_deliv,
        "perturbed_delivery_err_S3": pert_s3_deliv,
        "omission_err_S2": statistics.median(err_s2_om) if err_s2_om else None,
        "omission_err_crash_ref": statistics.median(err_crash_om) if err_crash_om else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", default=",".join(STUDY2_SCENARIOS.keys()))
    ap.add_argument("--mode", choices=("local", "docker"), default="docker")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", type=Path, default=OUT_V2)
    ap.add_argument("--allow-unfrozen", action="store_true")
    args = ap.parse_args()

    require_frozen(allow_unfrozen=args.allow_unfrozen)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary: Dict[str, Any] = json.loads(summary_path.read_text()) if summary_path.exists() else {"scenarios": {}}

    for name in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        if name in summary["scenarios"]:
            print(f"[skip] {name} already recorded in {summary_path}", flush=True)
            continue
        kappa = STUDY2_SCENARIOS.get(name, 1.0)
        row = run_scenario_v2(name, kappa, args.mode, args.duration, args.warmup, args.reps, args.output)
        summary["scenarios"][name] = row
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[done] {name}: {json.dumps(row)}", flush=True)

    analysis = analyze_study2(summary)
    (args.output / "analysis_v2.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
