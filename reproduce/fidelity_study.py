#!/usr/bin/env python3
"""
Runner for docs/research/fidelity/PREREGISTRATION.md.

    ramp      §8  capacity ramp on a pilot scenario (fault-free, no simulator)
    noise     §8  repeated crash sweep on a pilot scenario; δ from I_obs alone
    capacity  §2  60 s fault-free inclusion check of each candidate scenario
    sweep     §4  the study's runs             (refused until the file is FROZEN)
    simulate  §5  the simulator arms           (refused until the file is FROZEN)

The analysis (``fidelity_significance.py``) refuses to read a partial sweep.

Serial by design: one Docker system at a time. Completed runs are skipped on
resume, keyed by a fingerprint of the run's definition.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saag.core.graph_io import load_graph  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.testbed import orchestrate, study  # noqa: E402
from saag.testbed.collect import load_run  # noqa: E402
from saag.testbed.deployment import build_deployment  # noqa: E402
from saag.validation.fidelity import observed_impact, stalled_in_trace, summarize  # noqa: E402

PREREG = ROOT / "docs" / "research" / "fidelity" / "PREREGISTRATION.md"
OUT = ROOT / "results" / "fidelity_study"
CANDIDATES = ("av", "iot_smart_city", "financial_trading", "healthcare", "hub_and_spoke",
              "microservices", "enterprise", "telecom_ran", "industrial_scada", "realtime_gaming",
              "logistics_fleet", "atm")
STUDY_SEED = 20261105
REPS = 3
BASELINES = 5
MAX_ATTEMPTS = 3
BASE_S = 60.0
WARM_S = 5.0
CONDITIONS = {"O-open": True, "O-react": False}


def scenario_path(name: str) -> Path:
    return ROOT / "data" / "scenarios" / f"{name}_system.json"


def require_frozen() -> None:
    if not any(l.startswith("**Status: FROZEN") for l in PREREG.read_text().splitlines()):
        raise SystemExit(f"{PREREG.name} is not frozen; only the pilot may run (§8).")


CPU = Path("/sys/devices/system/cpu")
#: §4: the clocks the pilot was measured on.
PINNED = {"governor": "performance", "no_turbo": "1", "p_core_khz": 1900000}


#: Wall-clock time after which no new run starts (``sweep --hours``); None = no limit.
SESSION_END = None


class SessionOver(Exception):
    """The session's time budget is spent; every finished run is saved and the next session resumes."""


def check_session() -> None:
    if SESSION_END is not None and time.time() >= SESSION_END:
        raise SessionOver


def require_pinned() -> None:
    """§4: refuse to run a measured scenario unless the CPU clocks are pinned as in the pilot."""
    if MODE == "local":
        return
    gov = (CPU / "cpu0/cpufreq/scaling_governor").read_text().strip()
    turbo = (CPU / "intel_pstate/no_turbo").read_text().strip()
    mx = int((CPU / "cpu0/cpufreq/scaling_max_freq").read_text())
    mn = int((CPU / "cpu0/cpufreq/scaling_min_freq").read_text())
    if (gov, turbo) != (PINNED["governor"], PINNED["no_turbo"]) or mx != mn or mx != PINNED["p_core_khz"]:
        raise SystemExit(f"CPU clocks are not pinned (governor={gov}, no_turbo={turbo}, "
                         f"min/max={mn}/{mx} kHz); see PREREGISTRATION.md §4")


def prepare(name: str, kappa: float, tag: str, open_loop: bool, duration: float) -> Path:
    src = study.write_scaled(scenario_path(name), kappa, OUT / tag / f"{name}_k{kappa:.6g}_system.json")
    return orchestrate.build(src, OUT / tag / f"tb_{name}", duration, qos_semantics="dds", open_loop=open_loop)


MODE = "docker"


def measured_run(tb: Path, duration: float, faults, seed: int) -> dict:
    if MODE == "local":
        return {"meta": orchestrate.run(tb, duration, faults, mode="local", seed=seed), "cpu_p95": {}}
    with study.CpuSampler() as cpu:
        meta = orchestrate.run(tb, duration, faults, mode="docker", seed=seed)
    start = meta["start_unix"]
    return {"meta": meta, "cpu_p95": cpu.p95_by_container(start + WARM_S, start + duration)}


def archive(tb: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wb") as out:
        for f in sorted((tb / "telemetry").glob("*.jsonl")):
            with open(f, "rb") as fh:
                shutil.copyfileobj(fh, out)


def passes(att: dict, cpu: dict) -> bool:
    peak = max((v for v in cpu.values() if v is not None), default=None)
    return (att["median_rate_error"] is not None and att["median_rate_error"] <= 0.05
            and peak is not None and peak <= 80.0)


def cmd_ramp(a) -> None:
    res = []
    for m in [float(x) for x in a.mults.split(",")]:
        tb = prepare(a.scenario, m, "pilot", True, a.duration)
        plan = json.loads((tb / "deployment.json").read_text())
        r = measured_run(tb, a.duration, [], STUDY_SEED)
        att = study.attainment(load_run(tb), plan, WARM_S, a.duration)
        row = {"multiplier": m, "peak_host_load": max(study.host_loads(plan).values()),
               "attainment": att, "cpu_p95": r["cpu_p95"], "passed": passes(att, r["cpu_p95"])}
        res.append(row)
        print(json.dumps({k: row[k] for k in ("multiplier", "peak_host_load", "passed")},), 
              round(att["median_rate_error"] or -1, 4), max(r["cpu_p95"].values(), default=None), flush=True)
        (OUT / "pilot").mkdir(parents=True, exist_ok=True)
        (OUT / "pilot" / f"ramp_{a.scenario}.json").write_text(json.dumps(res, indent=2))
        if not row["passed"]:
            break


def attempt_run(tb, plan, duration, target, mid, kappa, key_seed):
    """§10: rerun an invalid run up to twice, each with a new seed; every failure is kept."""
    failures = []
    for attempt in range(MAX_ATTEMPTS):
        seed = study.run_seed(key_seed, str(attempt))
        faults = [FaultSpec(target, "crash", mid)] if target else []
        try:
            r = measured_run(tb, duration, faults, seed)
            trace = load_run(tb)
            why = study.run_validity(tb, plan, trace, target,
                                     r["meta"]["start_unix"], mid if target else None, WARM_S / kappa)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            r, trace, why = None, None, repr(exc)
        if why is None:
            return r, trace, seed, failures
        failures.append({"attempt": attempt, "seed": seed, "reason": why})
    return None, None, None, failures


def sweep_one(name, cond, kappa, tag, targets, reps, trace_root):
    """I_obs per target and repetition, with resume. Comparison with a simulator happens elsewhere."""
    duration = BASE_S / kappa
    tb = prepare(name, kappa, tag, CONDITIONS[cond], duration)
    plan = json.loads((tb / "deployment.json").read_text())
    subs = study.subscribers_of(plan)
    store = OUT / tag / f"iobs_{name}_{cond}.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    for rep in range(reps):
        for target in targets:
            check_session()
            key_seed = study.run_seed(STUDY_SEED, name, cond, target, str(rep))
            key = hashlib.sha1(f"{name}|{cond}|{kappa}|{target}|{rep}|{key_seed}".encode()).hexdigest()[:12]
            if key in done:
                continue
            mid = duration / 2.0
            r, trace, seed, failures = attempt_run(tb, plan, duration, target, mid, kappa, key_seed)
            row = {"target": target, "rep": rep, "seed": seed, "failures": failures, "valid": r is not None}
            if r is not None:
                row.update(
                    start_unix=r["meta"]["start_unix"], duration_s=duration, fault_s=mid,
                    trace=f"{key}.jsonl.gz",
                    i_obs=observed_impact(trace, subs, target, mid, warmup_s=WARM_S / kappa,
                                          failed=study._failed_with(plan, target)),
                    stalled=sorted(stalled_in_trace(trace, mid) - {target}),
                    pre_fault=study.attainment(trace, plan, WARM_S / kappa, mid),
                    cpu_p95=r["cpu_p95"])
                archive(tb, trace_root / row["trace"])
            done[key] = row
            store.write_text(json.dumps(done, indent=1))
            print(name, cond, target, rep, row.get("i_obs"), flush=True)
    return done


def baselines(name, cond, kappa, tag, trace_root):
    """§4: fault-free runs per condition, and for O-open one more that only calibration may read."""
    duration = BASE_S / kappa
    tb = prepare(name, kappa, tag, CONDITIONS[cond], duration)
    plan = json.loads((tb / "deployment.json").read_text())
    store = OUT / tag / f"baseline_{name}_{cond}.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    for i in range(BASELINES):
        check_session()
        if str(i) in done:
            continue
        key_seed = study.run_seed(STUDY_SEED, name, cond, "baseline", str(i))
        r, trace, seed, failures = attempt_run(tb, plan, duration, None, None, kappa, key_seed)
        row = {"seed": seed, "failures": failures, "valid": r is not None}
        if r is not None:
            row.update(start_unix=r["meta"]["start_unix"], duration_s=duration,
                       trace=f"baseline_{cond}_{i}.jsonl.gz",
                       attainment=study.attainment(trace, plan, WARM_S / kappa, duration),
                       cpu_p95=r["cpu_p95"])
            archive(tb, trace_root / row["trace"])
        done[str(i)] = row
        store.write_text(json.dumps(done, indent=1))
    profile = OUT / tag / f"profile_{name}.json"
    if cond == "O-open" and not profile.exists():
        r, _, _, failures = attempt_run(tb, plan, duration, None, None, kappa,
                                        study.run_seed(STUDY_SEED, name, cond, "calibration"))
        if r is None:
            raise RuntimeError(f"calibration run failed on {name}: {failures}")
        orchestrate.calibrate(tb, warmup_s=WARM_S / kappa)
        shutil.copy(tb / "profile.json", profile)


def by_target(done: dict) -> dict:
    out = {}
    for r in done.values():
        if r.get("valid", True):
            out.setdefault(r["target"], {})[r["rep"]] = r["i_obs"]
    return out


def leave_one_out(table: dict) -> list:
    """ρ of each repetition against the mean of the others."""
    reps = sorted({r for v in table.values() for r in v})
    ts = sorted(t for t, v in table.items() if set(v) == set(reps))
    rhos = []
    for r in reps:
        x = [table[t][r] for t in ts]
        y = [statistics.mean(table[t][o] for o in reps if o != r) for t in ts]
        rho = spearmanr(x, y).correlation
        if rho == rho:
            rhos.append(float(rho))
    return rhos


def cmd_noise(a) -> None:
    path = scenario_path(a.scenario)
    plan = build_deployment(path, open_loop=True)
    apps = [n for n, d in load_graph(path).nodes(data=True) if d.get("type") == "Application"]
    targets = study.select_targets(plan, apps)["publishers"]
    done = sweep_one(a.scenario, "O-open", a.kappa, "pilot", targets, REPS,
                     OUT / "pilot" / "traces" / a.scenario)
    rhos = leave_one_out(by_target(done))
    sd = statistics.stdev(rhos) if len(rhos) > 1 else None
    summary = {"scenario": a.scenario, "kappa": a.kappa, "n_targets": len(targets),
               "loo_rho": rhos, "ceiling": statistics.mean(rhos) if rhos else None, "loo_rho_sd": sd}
    (OUT / "pilot" / f"noise_{a.scenario}.json").write_text(json.dumps(summary, indent=2))
    print(summary)


def cmd_capacity(a) -> None:
    res = {}
    for name in CANDIDATES:
        plan = build_deployment(scenario_path(name), open_loop=True)
        kappa = study.kappa_for(plan, a.capacity)
        row = {"kappa": kappa}
        if kappa < 0.2:
            row.update(included=False, reason="kappa below 0.2")
        else:
            duration = BASE_S / kappa
            tb = prepare(name, kappa, "capacity", True, duration)
            scaled = json.loads((tb / "deployment.json").read_text())
            r = measured_run(tb, duration, [], STUDY_SEED)
            att = study.attainment(load_run(tb), scaled, WARM_S / kappa, duration)
            row.update(attainment=att, cpu_p95=r["cpu_p95"], included=passes(att, r["cpu_p95"]))
        res[name] = row
        print(name, row["kappa"], row["included"], flush=True)
        (OUT / "capacity.json").write_text(json.dumps(res, indent=2))


def included() -> dict:
    cap = json.loads((OUT / "capacity.json").read_text())
    return {n: r for n, r in cap.items() if r["included"]}


def targets_for(name: str) -> dict:
    """Chosen on the unscaled plan: scaling changes rates, never who publishes."""
    path = scenario_path(name)
    plan = build_deployment(path, open_loop=True)
    apps = [n for n, d in load_graph(path).nodes(data=True) if d.get("type") == "Application"]
    return study.select_targets(plan, apps)


def cmd_sweep(a) -> None:
    """§4, in sessions: ``--hours`` stops before a run would start past the budget; rerun to resume."""
    global SESSION_END
    require_frozen()
    hours = getattr(a, "hours", None)
    SESSION_END = time.time() + hours * 3600.0 if hours else None
    log = OUT / "sweep" / "sessions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    started, outcome = time.time(), "complete"
    try:
        _sweep()
    except SessionOver:
        outcome = "budget reached"
        print("session budget reached; finished runs are saved, run `sweep` again to resume", flush=True)
    finally:
        with open(log, "a") as fh:
            fh.write(json.dumps({"start_unix": started, "end_unix": time.time(),
                                 "hours": getattr(a, "hours", None), "outcome": outcome}) + "\n")


def _sweep() -> None:
    for name, row in included().items():
        require_pinned()
        t = targets_for(name)
        order = t["publishers"] + t["sinks"] + t["brokers"] + t["hosts"]
        rng = random.Random(study.run_seed(STUDY_SEED, name, "order"))
        rng.shuffle(order)
        for cond in CONDITIONS:
            root = OUT / "sweep" / "traces" / name
            baselines(name, cond, row["kappa"], "sweep", root)
            sweep_one(name, cond, row["kappa"], "sweep", order, REPS, root)
    (OUT / "sweep" / "COMPLETE").write_text("all runs finished\n")


def _arm_job(job: dict) -> dict:
    """One worker's task; simulator only, nothing observed is read."""
    logging.disable(logging.WARNING)
    path, plan = Path(job["scenario"]), job["plan"]
    profile = None
    if job["arm"] == "S3":
        profile = orchestrate.profile_from_json(json.loads(Path(job["profile"]).read_text()))
    g = load_graph(path)
    d, mid, warm = job["duration"], job["duration"] / 2.0, job["warm"]
    if job["target"] is None:
        trace = study.simulate_fault_free(g, plan, job["arm"], d, job["seed"], profile)
        return {**job["id"], "steady": trace_summary(trace, warm, d)}
    return {**job["id"], **study.simulate_arm(g, plan, job["arm"], job["target"], d, mid, warm,
                                              job["seed"], profile)}


def trace_summary(trace, warm: float, end: float) -> dict:
    """Keep what §7 compares; `summarize` is the same function the observed side uses."""
    return summarize(trace.window(warm, end))


def cmd_simulate(a) -> None:
    require_frozen()
    jobs = []
    for name, row in included().items():
        kappa, t = row["kappa"], targets_for(name)
        apps, infra = t["publishers"] + t["sinks"], t["brokers"] + t["hosts"]
        duration, warm = BASE_S / kappa, WARM_S / kappa
        for cond, arms in (("O-open", ("S0", "S1", "S2", "S3")), ("O-react", ("S0", "S4"))):
            tb = prepare(name, kappa, "simulate", CONDITIONS[cond], duration)
            plan = json.loads((tb / "deployment.json").read_text())
            scaled = plan["source_scenario"]
            prof = str(OUT / "sweep" / f"profile_{name}.json")
            for arm in arms:
                scored = apps + infra if study.ARMS[arm].get("deployment_model") == "stations" else apps
                for seed in study.SIM_SEEDS:
                    for target in scored + [None]:
                        jobs.append({"scenario": scaled, "plan": plan, "arm": arm, "target": target,
                                     "duration": duration, "warm": warm, "seed": seed, "profile": prof,
                                     "id": {"scenario_name": name, "cond": cond, "arm": arm,
                                            "target": target, "seed": seed}})
            if cond == "O-open" and kappa < 1.0:      # §7 scaling check
                com = build_deployment(scenario_path(name), qos_semantics="dds", open_loop=True)
                for seed in study.SIM_SEEDS:
                    for target in apps:
                        jobs.append({"scenario": str(scenario_path(name)), "plan": com, "arm": "S0",
                                     "target": target, "duration": BASE_S, "warm": WARM_S, "seed": seed,
                                     "profile": prof,
                                     "id": {"scenario_name": name, "cond": "committed", "arm": "S0",
                                            "target": target, "seed": seed}})
    store = OUT / "simulate" / "arms.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    key = lambda i: "|".join(str(i[k]) for k in ("scenario_name", "cond", "arm", "target", "seed"))  # noqa: E731
    done = {key(json.loads(l)) for l in store.read_text().splitlines()} if store.exists() else set()
    todo = [j for j in jobs if key(j["id"]) not in done]
    print(f"{len(todo)} of {len(jobs)} simulator runs to do", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool, open(store, "a") as out:
        for row in pool.map(_arm_job, todo, chunksize=4):
            out.write(json.dumps(row) + "\n")
            out.flush()
    (OUT / "simulate" / "COMPLETE").write_text("all runs finished\n")


def main() -> None:
    logging.disable(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ramp")
    r.add_argument("scenario")
    r.add_argument("--mults", default="1,2,5,10,20,50,100")
    r.add_argument("--duration", type=float, default=40.0)
    r.set_defaults(fn=cmd_ramp)
    n = sub.add_parser("noise")
    n.add_argument("scenario")
    n.add_argument("--kappa", type=float, default=1.0)
    n.set_defaults(fn=cmd_noise)
    c = sub.add_parser("capacity")
    c.add_argument("--capacity", type=float, required=True, help="B_h, messages/s on the busiest host")
    c.set_defaults(fn=cmd_capacity)
    w = sub.add_parser("sweep")
    w.add_argument("--hours", type=float, default=None,
                   help="session budget: start no new run after this many hours (resume by rerunning)")
    w.set_defaults(fn=cmd_sweep)
    m = sub.add_parser("simulate")
    m.add_argument("--workers", type=int, default=4)
    m.set_defaults(fn=cmd_simulate)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
