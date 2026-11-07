"""
Building blocks of the fidelity study (docs/research/fidelity/PREREGISTRATION.md).

Rate scaling, per-host load, target selection, the simulator arms and a Docker
run that logs per-container CPU. The orchestration lives in
``reproduce/fidelity_study.py``.
"""

from __future__ import annotations

import copy
import gzip
import json
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from saag.simulation._stats import percentile, derive_seed
from saag.simulation.calibration import TelemetryTrace
from saag.simulation.runtime.faults import FaultSpec
from saag.validation.fidelity import observed_impact, stalled_in_trace

SELECTION_SEED = 20261105
SIM_SEEDS = (42, 123, 456, 789, 2024)

#: §5 of the pre-registration. S3 is S2 plus a telemetry profile; S4 is O-react only.
_S2 = {"behavior_model": "open_loop", "deployment_model": "stations", "qos_semantics": "dds",
       "rng_streams": "per_process", "replica_aware": True}
ARMS: Dict[str, Dict[str, Any]] = {
    "S0": {},
    "S1": {"qos_semantics": "dds"},
    "S2": _S2,
    "S3": _S2,
    "S4": {**_S2, "behavior_model": "reactive"},
}


def scale_scenario(topology: Mapping[str, Any], kappa: float) -> Dict[str, Any]:
    """Multiply topic rates by *kappa* and divide deadlines by it.

    Service times need no change: the plan and the simulators both derive them
    from the rates and the target utilisation, so utilisation is preserved.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    out = copy.deepcopy(dict(topology))
    for t in out.get("topics", []):
        if t.get("frequency") is not None:
            t["frequency"] = float(t["frequency"]) * kappa
        for holder in (t, t.get("qos") or {}):
            if holder.get("deadline_ms") is not None:
                holder["deadline_ms"] = float(holder["deadline_ms"]) / kappa
    return out


def write_scaled(source: Path, kappa: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scale_scenario(json.loads(Path(source).read_text()), kappa)))
    return out


def host_loads(plan: Mapping[str, Any]) -> Dict[str, float]:
    """Messages each host handles per second, with every active instance on its first host.

    An application counts what it receives and emits. A topic's lowest-id
    router counts it twice: ingress and egress.
    """
    topics = plan["topics"]
    loads = {h: 0.0 for h in plan["hosts"]}
    for spec in plan["apps"].values():
        loads[spec["hosts"][0]] += (sum(topics[t]["rate_hz"] for t in spec["subscribes"])
                                    + sum(spec["per_publisher_rate"].values()))
    for b, spec in plan["brokers"].items():
        host = next((h for h, r in plan["hosts"].items() if b in r["brokers"]), None)
        if host is not None:
            loads[host] += sum(2.0 * topics[t]["rate_hz"] for t in spec["routes"]
                               if topics[t]["routers"][0] == b)
    return loads


def kappa_for(plan: Mapping[str, Any], capacity: float) -> float:
    """§2: the largest scale, at most 1, that keeps the busiest host within *capacity*."""
    peak = max(host_loads(plan).values(), default=0.0)
    return 1.0 if peak <= capacity else capacity / peak


def subscribers_of(plan: Mapping[str, Any]) -> Dict[str, List[str]]:
    subs: Dict[str, List[str]] = {t: [] for t in plan["topics"]}
    for a, spec in plan["apps"].items():
        for t in spec["subscribes"]:
            subs[t].append(a)
    return subs


def select_targets(plan: Mapping[str, Any], app_ids: List[str], cap: int = 30, sinks: int = 5,
                   seed: int = SELECTION_SEED) -> Dict[str, List[str]]:
    """§4: stratified publishers (sole vs shared publisher), sinks, all brokers and hosts."""
    apps = {a: plan["apps"][a] for a in sorted(app_ids) if a in plan["apps"]}
    sole = [a for a, s in apps.items()
            if s["publishes"] and any(len(plan["topics"][t]["publishers"]) == 1 for t in s["publishes"])]
    shared = [a for a, s in apps.items() if s["publishes"] and a not in sole]
    sink = [a for a, s in apps.items() if not s["publishes"]]
    rng = random.Random(seed)
    total = len(sole) + len(shared)
    if total <= cap:
        chosen = sole + shared
    else:
        n_sole = round(cap * len(sole) / total)
        chosen = rng.sample(sole, n_sole) + rng.sample(shared, cap - n_sole)
    return {"publishers": sorted(chosen),
            "sinks": sorted(rng.sample(sink, min(sinks, len(sink)))),
            "brokers": sorted(plan["brokers"]),
            "hosts": sorted(plan["hosts"])}


def rate_expansion(plan: Mapping[str, Any]) -> Dict[str, float]:
    """Arm R, Eq. 7: the sum over v's topics of r_t / |pub(t)| * |sub(t) minus v|."""
    subs = subscribers_of(plan)
    return {a: sum(plan["topics"][t]["rate_hz"] / len(plan["topics"][t]["publishers"])
                   * len([s for s in subs[t] if s != a]) for t in spec["publishes"])
            for a, spec in plan["apps"].items()}


def _simulate(graph: Any, plan: Mapping[str, Any], arm: str, duration_s: float, seed: int,
              faults_kw: Mapping[str, Any], profile: Any, dds_status: bool) -> TelemetryTrace:
    from saag.simulation.message_flow_simulator import MessageFlowSimulator

    settings = {**ARMS[arm], **faults_kw}
    if arm == "S3":
        settings["telemetry_profile"] = profile
    if dds_status:
        settings["dds_status"] = True
    res = MessageFlowSimulator(graph, duration=duration_s, seed=seed, record_events=True,
                               target_utilization=plan["target_utilization"], **settings).run()
    return TelemetryTrace.from_records(res.events, start_s=0.0, end_s=duration_s)


def simulate_fault_run(graph: Any, plan: Mapping[str, Any], arm: str, target: str, duration_s: float,
                       fault_s: float, seed: int, profile: Any = None,
                       dds_status: bool = False) -> TelemetryTrace:
    """One seed of one arm with *target* crashing at *fault_s*."""
    # S0 and S1 keep the published entry point; stations need the fault library.
    if ARMS[arm].get("deployment_model") == "stations":
        kw: Dict[str, Any] = {"faults": [FaultSpec(target, "crash", fault_s)]}
    else:
        kw = {"fault_node": target, "fault_time": fault_s}
    return _simulate(graph, plan, arm, duration_s, seed, kw, profile, dds_status)


def simulate_fault_free(graph: Any, plan: Mapping[str, Any], arm: str, duration_s: float, seed: int,
                        profile: Any = None) -> TelemetryTrace:
    return _simulate(graph, plan, arm, duration_s, seed, {}, profile, False)


def simulate_arm(graph: Any, plan: Mapping[str, Any], arm: str, target: str, duration_s: float,
                 fault_s: float, warmup_s: float, seed: int, profile: Any = None) -> Dict[str, Any]:
    """One seed of one arm: impact through the observed-impact estimator, and what stalled."""
    trace = simulate_fault_run(graph, plan, arm, target, duration_s, fault_s, seed, profile)
    return {"impact": observed_impact(trace, subscribers_of(plan), target, fault_s, warmup_s=warmup_s,
                                      failed=_failed_with(plan, target)),
            "stalled": sorted(stalled_in_trace(trace, fault_s) - {target})}


def _failed_with(plan: Mapping[str, Any], target: str) -> List[str]:
    """Components a crash of *target* removes outright: a host takes its single-hosted residents."""
    if target not in plan["hosts"]:
        return [target]
    r = plan["hosts"][target]
    return [a for a in r["apps"] if len(plan["apps"][a]["hosts"]) == 1] + list(r["brokers"])


def run_seed(study_seed: int, *key: str) -> int:
    return int(derive_seed(study_seed, "|".join(key)))


def load_archived(path: Path, start_s: float, end_s: float) -> TelemetryTrace:
    """Read a trace written by the runner's archive step, on run-relative time."""
    def records():
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r["time_unix_s"] = float(r["time_unix_s"]) - start_s
                    yield r
    return TelemetryTrace.from_records(records(), start_s=0.0, end_s=end_s)


def run_validity(tb: Path, plan: Mapping[str, Any], trace: TelemetryTrace, target: Optional[str],
                 start_unix: float, fault_s: Optional[float], warmup_s: float) -> Optional[str]:
    """§10: why a run is invalid, or None. Checks each host's own log and the pre-fault rates."""
    gone = {target} if target in plan["hosts"] else set()
    end = fault_s if fault_s is not None else trace.end_s
    in_process = (Path(tb) / "telemetry" / "all.jsonl").exists()
    for host in () if in_process else plan["hosts"]:
        log = Path(tb) / "telemetry" / f"{host}.jsonl"
        if not log.exists() or not log.stat().st_size:
            return f"missing trace: {host}"
        if host in gone:
            continue
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 4096))
            last = fh.read().splitlines()[-1]
        try:
            t = float(json.loads(last)["time_unix_s"]) - start_unix
        except ValueError:      # the final line of a killed writer may be cut short
            continue
        if t < trace.end_s - max(5.0, 0.1 * trace.end_s):
            return f"host stopped early: {host} at {t:.1f}s"
    err = attainment(trace, plan, warmup_s, end)["median_rate_error"]
    if err is None or err > 0.10:
        return f"pre-fault rate error {err}"
    return None


class CpuSampler:
    """Polls ``docker stats``; 100% is one core, the limit that matters to a GIL-bound agent."""

    def __init__(self, period_s: float = 0.0) -> None:
        self.period_s = period_s
        self.samples: Dict[str, List[Tuple[float, float]]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            done = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}}"],
                                  capture_output=True, text=True)
            for line in done.stdout.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].startswith("saag-"):
                    self.samples.setdefault(parts[0], []).append((time.time(), float(parts[1].rstrip("%"))))
            self._stop.wait(self.period_s)

    def __enter__(self) -> "CpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def p95_by_container(self, since: float, until: float) -> Dict[str, Optional[float]]:
        """Only samples inside the window count: startup and shutdown are not the load."""
        out: Dict[str, Optional[float]] = {}
        for c, rows in self.samples.items():
            kept = [v for t, v in rows if since <= t <= until]
            out[c] = percentile(kept, 95) if kept else None
        return out


def attainment(trace: TelemetryTrace, plan: Mapping[str, Any], warmup_s: float,
               end_s: Optional[float] = None) -> Dict[str, Any]:
    """Capacity only: declared rates against emitted rates. No simulator is involved."""
    end = end_s if end_s is not None else trace.end_s
    span = max(end - warmup_s, 1e-9)
    counts: Dict[str, int] = {}
    deliver = 0
    for e in trace.events:
        if warmup_s <= e["t"] < end:
            if e["kind"] == "publish":
                counts[e["topic"]] = counts.get(e["topic"], 0) + 1
            elif e["kind"] == "deliver":
                deliver += 1
    errs = sorted(abs(counts.get(t, 0) / span - s["rate_hz"]) / s["rate_hz"]
                  for t, s in plan["topics"].items() if s["publishers"] and s["rate_hz"])
    subs = subscribers_of(plan)
    expected = sum(s["rate_hz"] * len(subs[t]) for t, s in plan["topics"].items() if s["publishers"])
    return {"median_rate_error": percentile(errs, 50) if errs else None,
            "max_rate_error": errs[-1] if errs else None,
            "delivered_per_s": deliver / span, "declared_deliveries_per_s": expected,
            "wall_clock": time.time()}
