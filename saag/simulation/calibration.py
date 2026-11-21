"""
Telemetry traces and the calibrator that turns them into simulator parameters.

Post-deployment (or on the emulated testbed, ``saag/testbed/``), the runtime
parameters that pre-deployment runs have to guess can be measured. This module
defines the trace format the testbed agents write and `TelemetryCalibrator`,
which maps a trace onto `RuntimeTelemetryProfile` for
`MessageFlowSimulator(telemetry_profile=...)`.

Trace format
────────────
JSON lines, one event per line, shaped like an OpenTelemetry log record::

    {"time_unix_s": 12.345, "name": "saag.deliver",
     "attributes": {"service.name": "B", "messaging.destination.name": "t1",
                    "messaging.message.id": 17, "saag.publisher": "A",
                    "saag.created_unix_s": 12.301, "saag.latency_ms": 44.0,
                    "saag.service_ms": 6.1}}

Event names: ``saag.publish`` (an emission), ``saag.deliver`` (a sample that
met its deadline and was processed), ``saag.deadline_miss`` (processed, then
dropped as late; it carries ``saag.service_ms`` too), ``saag.drop``
(queue overflow) and ``saag.state`` (a component's state change; attribute
``saag.state``). Times are seconds on one clock shared by all agents.

What is calibrated, and what is not
───────────────────────────────────
Only quantities the trace measures directly are mapped:

* ``msg_rate_per_sec[t]``    emissions on t per second, all publishers.
* ``service_time_dist[a]``   mean and coefficient of variation of a's measured
                             per-message service time, with the family
                             (exponential when CV is near 1, else lognormal).
* ``trigger_policy[a]``      inferred from how a's emissions respond to its
                             inputs going stale (`infer_trigger_policy`); left
                             unset when the trace never shows an input outage,
                             because then the policy is not identifiable.

Nothing is written to ``edge_failure_correlation`` (a cascade-propagation
probability in `FailureSimulator`, not a drop rate) or to
``custom_starvation_bounds`` (no trace event measures a starvation tolerance).
"""

from __future__ import annotations

import json
import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import RuntimeTelemetryProfile

EVENT_NAMES = ("saag.publish", "saag.deliver", "saag.deadline_miss", "saag.drop", "saag.state",
               "saag.status")


def make_event(t: float, name: str, app: str, topic: Optional[str] = None, **attrs: Any) -> Dict[str, Any]:
    """Build one trace record (used by the testbed agents and by tests)."""
    a: Dict[str, Any] = {"service.name": app}
    if topic is not None:
        a["messaging.destination.name"] = topic
    for k, v in attrs.items():
        a[k if "." in k else f"saag.{k}"] = v
    return {"time_unix_s": t, "name": name, "attributes": a}


@dataclass
class TelemetryTrace:
    """Normalised events, indexed for the calibrator and the fidelity report."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], start_s: Optional[float] = None,
                     end_s: Optional[float] = None) -> "TelemetryTrace":
        """Parse records; *start_s* / *end_s* default to the first / last event.

        Pass the run's real bounds when known: a system that fell silent leaves
        no events at the end, and a trace ending at its last event cannot show
        the silence.
        """
        events = []
        for r in records:
            name = r.get("name")
            if name not in EVENT_NAMES:
                continue
            a = r.get("attributes", {})
            events.append({
                "t": float(r["time_unix_s"]),
                "kind": name.split(".", 1)[1],
                "app": a.get("service.name"),
                "topic": a.get("messaging.destination.name"),
                "publisher": a.get("saag.publisher"),
                "latency_ms": a.get("saag.latency_ms"),
                "service_ms": a.get("saag.service_ms"),
                "state": a.get("saag.state"),
                # saag.status: a DDS standard status count (runtime/dds_status).
                "status": a.get("saag.status"),
                "count": a.get("saag.count"),
                "entity": a.get("saag.entity"),
                "source": a.get("saag.source"),
                "instance": a.get("saag.instance"),
            })
        events.sort(key=lambda e: e["t"])
        start = start_s if start_s is not None else (events[0]["t"] if events else 0.0)
        end = end_s if end_s is not None else (events[-1]["t"] if events else 0.0)
        return cls(events=events, start_s=start, end_s=end)

    @classmethod
    def load(cls, paths: Iterable[Path]) -> "TelemetryTrace":
        def records():
            for p in paths:
                with open(p) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
        return cls.from_records(records())

    def window(self, start: float, end: float) -> "TelemetryTrace":
        ev = [e for e in self.events if start <= e["t"] < end]
        return TelemetryTrace(events=ev, start_s=start, end_s=end)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def times(self, kind: str, app: Optional[str] = None, topic: Optional[str] = None) -> List[float]:
        return [e["t"] for e in self.events if e["kind"] == kind
                and (app is None or e["app"] == app) and (topic is None or e["topic"] == topic)]

    def status_counts(self, until: Optional[float] = None,
                      prefer: Sequence[str] = ("cyclone", "simulator", "agent"),
                      exclude_instances: Iterable[str] = ()) -> Dict[str, Dict[str, int]]:
        """Latest value of every DDS status per entity, as of *until* (default: end).

        Keys follow `saag.simulation.runtime.dds_status.entity_key`. Several
        sources may report one status (the testbed's agent count and Cyclone's
        own); the first of *prefer* present wins. A replicated component's
        instances are combined by taking the maximum, since only the active one
        is counting. *exclude_instances* drops reports from instances that died
        (``"A"`` or ``"A@H1"``): a dead entity's last report is stale, which
        matters for gauges.
        """
        excluded = set(exclude_instances)
        latest: Dict[Tuple[str, str, str, str], Tuple[float, int]] = {}
        for e in self.events:
            if e["kind"] != "status" or (until is not None and e["t"] > until):
                continue
            src = e.get("source") or "agent"
            inst = e.get("instance") or e["app"]
            if inst in excluded:
                continue
            key = (e["status"], f"{e['entity']}|{e['topic']}|{e['app']}", src, inst)
            latest[key] = (e["t"], int(e["count"] or 0))
        by_src: Dict[Tuple[str, str, str], int] = {}
        for (status, ent, src, _inst), (_, value) in latest.items():
            k = (status, ent, src)
            by_src[k] = max(by_src.get(k, 0), value)
        out: Dict[str, Dict[str, int]] = defaultdict(dict)
        for status, ent in sorted({(st, en) for st, en, _ in by_src}):
            for src in prefer:
                if (status, ent, src) in by_src:
                    out[status][ent] = by_src[(status, ent, src)]
                    break
        return dict(out)

    def apps(self) -> List[str]:
        return sorted({e["app"] for e in self.events if e["app"]})

    def inputs_of(self, app: str) -> List[str]:
        return sorted({e["topic"] for e in self.events
                       if e["kind"] == "deliver" and e["app"] == app and e["topic"]})

    def outputs_of(self, app: str) -> List[str]:
        return sorted({e["topic"] for e in self.events
                       if e["kind"] == "publish" and e["app"] == app and e["topic"]})


# ── Trigger-policy inference ─────────────────────────────────────────────────

def _fresh(times: Sequence[float], t: float, window: float) -> bool:
    i = bisect_right(times, t)
    return i > 0 and t - times[i - 1] <= window


#: Emitting shares strictly between these are not evidence for either policy.
AMBIGUOUS_SHARE = (0.2, 0.8)


def _median_gap(times: Sequence[float]) -> float:
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return statistics.median(gaps) if gaps else math.inf


def infer_trigger_policy(
    trace: TelemetryTrace, app: str, step_s: float = 0.1, periods: float = 3.0,
    min_expected: float = 3.0,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Classify *app*'s trigger policy from how its output responds to stale inputs.

    Time is split into three input states -- all inputs fresh, some stale, all
    stale -- on a grid of *step_s*. In each state the app's realised emission
    rate is compared with its active rate (one over its median inter-emission
    gap): ``emitting_share = emissions in state / (time in state x active rate)``.
    A periodic source keeps emitting with every input stale; ``any_of`` stops
    only when all are stale; ``all_of`` stops as soon as one is.

    A rate, not "emitted within the last few gaps": for a slow output (a third
    of a hertz is common when publishers share a topic) such a window outlasts
    the stall itself and makes a stalled app look alive.

    A state counts as evidence only when it was long enough to expect at least
    *min_expected* emissions; shares strictly inside `AMBIGUOUS_SHARE` are not
    evidence either way. Returns ``(None, evidence)`` when the trace never puts
    the app in a discriminating state.
    """
    inputs = trace.inputs_of(app)
    outputs = trace.outputs_of(app)
    if not outputs:
        return None, {"reason": "no emissions"}
    if not inputs:
        return "periodic", {"reason": "no inputs"}
    deliveries = {t: trace.times("deliver", app, t) for t in inputs}
    emits = trace.times("publish", app)
    # Windows from the median inter-event gap, not count / run length: the
    # latter includes the very outages being classified.
    win_in = {t: periods * _median_gap(ts) for t, ts in deliveries.items()}
    gap_out = _median_gap(emits)
    if not math.isfinite(gap_out):
        return None, {"reason": "too few emissions"}

    # After a replica takeover the new active instance emits on freshness it
    # tracked without recording it; skip one input window after each takeover.
    takeovers = [e["t"] for e in trace.events
                 if e["kind"] == "state" and e["app"] == app and e["state"] == "takeover"]
    blind = max(win_in.values())
    start = trace.start_s + blind

    def fresh_at(t: float) -> Optional[int]:
        if t < start or any(tk <= t < tk + blind for tk in takeovers):
            return None
        return sum(1 for tp in inputs if _fresh(deliveries[tp], t, win_in[tp]))

    def _bucket(fresh: int) -> str:
        return "all_fresh" if fresh == len(inputs) else "all_stale" if fresh == 0 else "some_stale"

    time_in = {"all_fresh": 0.0, "some_stale": 0.0, "all_stale": 0.0}
    time_by_fresh: Dict[int, float] = defaultdict(float)
    t = start
    while t < trace.end_s:
        fc = fresh_at(t)
        if fc is not None:
            time_in[_bucket(fc)] += step_s
            if 0 < fc < len(inputs):
                time_by_fresh[fc] += step_s
        t += step_s
    emitted = {k: 0 for k in time_in}
    emit_by_fresh: Dict[int, int] = defaultdict(int)
    for te in emits:
        fc = fresh_at(te)
        if fc is not None:
            emitted[_bucket(fc)] += 1
            if 0 < fc < len(inputs):
                emit_by_fresh[fc] += 1
    expected = {k: v / gap_out for k, v in time_in.items()}
    share = {k: (emitted[k] / expected[k] if expected[k] >= min_expected else None)
             for k in time_in}
    evidence = {"seconds": {k: round(v, 3) for k, v in time_in.items()},
                "expected_emissions": {k: round(v, 2) for k, v in expected.items()},
                "emitting_share": share}

    def _infer_k_of_n() -> Optional[Tuple[str, Dict[str, Any]]]:
        obs_shares = {
            f: emit_by_fresh[f] / (time_by_fresh[f] / gap_out)
            for f in sorted(time_by_fresh)
            if (time_by_fresh[f] / gap_out) >= min_expected
        }
        if not obs_shares:
            return None
        matches = []
        for k_cand in range(2, len(inputs) + 1):
            if (all(abs(s - min(1.0, f / k_cand)) <= 0.15 for f, s in obs_shares.items())
                    and any(f < k_cand and s < hi for f, s in obs_shares.items())):
                matches.append(k_cand)
        if len(matches) == 1:
            return f"k_of_n:{matches[0]}", {**evidence, "fresh_count_share": obs_shares}
        return None

    lo, hi = AMBIGUOUS_SHARE
    if share["all_stale"] is not None:
        if share["all_stale"] >= hi:
            return "periodic", evidence
        if share["all_stale"] > lo:
            return None, {**evidence, "reason": "ambiguous emitting share with all inputs stale"}
        if share["some_stale"] is not None:
            if share["some_stale"] <= lo:
                return "all_of", evidence
            if share["some_stale"] >= hi:
                return "any_of", evidence
            k_match = _infer_k_of_n()
            if k_match is not None:
                return k_match
            return None, {**evidence, "reason": "ambiguous emitting share with some inputs stale"}
        if len(inputs) == 1:
            # One input: all_of and any_of coincide; report the simpler.
            return "any_of", evidence
        return None, {**evidence, "reason": "no partial outage observed"}
    if share["some_stale"] is not None:
        if share["some_stale"] <= lo:
            return "all_of", evidence
        if lo < share["some_stale"] < hi:
            k_match = _infer_k_of_n()
            if k_match is not None:
                return k_match
    return None, {**evidence, "reason": "no input outage observed"}


# ── Calibrator ───────────────────────────────────────────────────────────────

def fit_service_time(samples_s: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Mean, CV and family of measured service times (None when too few)."""
    xs = [x for x in samples_s if x is not None and x > 0]
    if len(xs) < 5:
        return None
    mean = statistics.fmean(xs)
    cv = statistics.stdev(xs) / mean if mean > 0 else 0.0
    dist = "exponential" if 0.7 <= cv <= 1.3 else "lognormal"
    return {"dist": dist, "mean_s": mean, "cv": cv, "n": len(xs)}


class TelemetryCalibrator:
    """Map an observed `TelemetryTrace` onto a `RuntimeTelemetryProfile`."""

    def __init__(self, warmup_s: float = 0.0, infer_policies: bool = True) -> None:
        self.warmup_s = warmup_s
        self.infer_policies = infer_policies

    def calibrate(self, trace: TelemetryTrace, source: str = "trace") -> RuntimeTelemetryProfile:
        tr = trace.window(trace.start_s + self.warmup_s, trace.end_s + 1e-9) if self.warmup_s else trace
        prof = RuntimeTelemetryProfile()
        dur = tr.duration_s
        pub_times: Dict[str, List[float]] = defaultdict(list)
        service: Dict[str, List[float]] = defaultdict(list)
        for e in tr.events:
            if e["kind"] == "publish" and e["topic"]:
                pub_times[e["topic"]].append(e["t"])
            # Deadline drops were served too; leaving them out fits only the
            # fast survivors (45% low on atm A11).
            elif e["kind"] in ("deliver", "deadline_miss") and e["service_ms"] is not None:
                service[e["app"]].append(float(e["service_ms"]) / 1000.0)
        if dur > 0:
            rates: Dict[str, float] = {}
            for t, ts in sorted(pub_times.items()):
                active_dur = dur
                if len(ts) >= 3:
                    med_gap = _median_gap(ts)
                    if math.isfinite(med_gap) and med_gap > 0:
                        outage_thresh = max(5.0, 10.0 * med_gap)
                        gaps = [ts[0] - tr.start_s, *(b - a for a, b in zip(ts, ts[1:])), tr.end_s - ts[-1]]
                        silent = sum(g - med_gap for g in gaps if g > outage_thresh)
                        active_dur = max(1e-9, dur - silent)
                rates[t] = len(ts) / active_dur
            prof.msg_rate_per_sec = rates
        for app, xs in sorted(service.items()):
            fit = fit_service_time(xs)
            if fit is not None:
                prof.service_time_dist[app] = fit
        if self.infer_policies:
            for app in tr.apps():
                policy, ev = infer_trigger_policy(tr, app)
                prof.trigger_evidence[app] = ev
                if policy is not None:
                    prof.trigger_policy[app] = policy
        prof.provenance = {
            "source": source, "window_s": [tr.start_s, tr.end_s], "n_events": len(tr.events),
            "warmup_s": self.warmup_s,
        }
        return prof
