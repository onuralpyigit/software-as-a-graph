"""
Faults on the testbed: `FaultSpec` -> timed actions on containers and processes.

The same `FaultSpec`s that drive `MessageFlowSimulator` drive the testbed, so a
fidelity comparison injects the identical fault in both.

========================  =====================================================
fault                     action
========================  =====================================================
Node CRASH                ``docker kill`` the host's container
Node TRANSIENT_CRASH      ``docker kill``, then ``docker start`` (agents restart
                          their residents fresh; telemetry is appended)
Node FREEZE               ``docker pause`` / ``docker unpause``
Node SLOWDOWN             a SLOWDOWN control file for every resident, as for
                          an application fault (``docker update --cpus`` did
                          not slow a service that sleeps)
Application / Broker      control file ``<control>/<id>.json`` read by the
CRASH, TRANSIENT_CRASH,   agent (crashed / frozen / slowdown / omission /
FREEZE, SLOWDOWN,         babbling), reset when the fault ends
OMISSION, BABBLING
LINK_FLAP a|b             a partition (below) during each bad period, from the
                          same Gilbert-Elliott schedule the simulator draws
========================  =====================================================

A *partition* is a set of host pairs that cannot talk: each container drops
ingress traffic from its blocked peers' addresses with ``tc``, unicast and
multicast alike, so DDS discovery and liveliness see the cut too. Without
``network_gating`` the blocked pairs are exactly the flapping links' host
endpoints. With it, a pair is blocked when CONNECTS_TO no longer connects the
two hosts once cut links and crashed Nodes / NetworkDevices are removed --
`MessageFlowSimulator(network_gating=True)`'s reachability -- which also makes a
NetworkDevice crash real.

BUFFER_LEAK and REDELIVERY_STORM are not emulated; `plan_actions` reports them
in ``unsupported`` instead of silently skipping them, as it does any fault on a
network vertex that has no container.

With ``local=True`` (the whole scenario in one process over `LocalTransport`,
no containers) a host fault becomes the same control-file fault on every
component resident on that host, and LINK_FLAP is unsupported.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from saag.core.network import Fabric, link_key
from saag.simulation._stats import derive_seed
from saag.simulation.runtime.faults import FaultSpec, RuntimeFaultMode

from .compose import container_name

M = RuntimeFaultMode
#: Time of a partition the topology imposes before any fault: applied as soon as
#: chaos starts, which is before traffic does.
BEFORE_START = float("-inf")


@dataclass
class Action:
    t: float                                   # seconds after run start
    kind: str                                  # "docker" | "control" | "partition"
    argv: Optional[List[str]] = None
    target: Optional[str] = None
    control: Optional[Dict[str, Any]] = None


@dataclass
class ChaosPlan:
    actions: List[Action] = field(default_factory=list)
    unsupported: List[Dict[str, Any]] = field(default_factory=list)


def _control(t: float, target: str, **state) -> Action:
    return Action(t=t, kind="control", target=target, control=state)


def _residents(plan: Dict[str, Any], host: str) -> List[str]:
    """Control names on *host*: a replicated application by its instance there."""
    h = plan["hosts"][host]
    return [f"{a}@{host}" if len(plan["apps"][a].get("hosts", [])) > 1 else a
            for a in h["apps"]] + h["brokers"]


def _flap_periods(f: FaultSpec, i: int, seed: int, duration_s: float) -> List[Tuple[float, float]]:
    """The bad periods of a LINK_FLAP, drawn as the simulator draws them."""
    p = f.params
    rng = random.Random(derive_seed(seed, f"chaos|{i}|{f.target}"))
    good, bad = float(p.get("mean_good_s", 2.0)), float(p.get("mean_bad_s", 0.5))
    t, stop = f.start_s, f.end_s if f.end_s is not None else duration_s
    out = []
    while True:
        t += rng.expovariate(1.0 / good)
        if t >= stop:
            return out
        cut = t
        t += rng.expovariate(1.0 / bad)
        out.append((cut, min(t, stop)))
        if t >= stop:
            return out


def _partitions(events: List[Tuple[float, str, Any]], plan: Dict[str, Any]) -> List[Action]:
    """Network events -> one partition action each time the blocked pairs change."""
    net = plan.get("network") or {}
    gating = bool(net.get("gating"))
    hosts = sorted(plan["hosts"])
    fabric = Fabric(net.get("vertices", []), [(a, b, None) for a, b in plan.get("links", [])])
    gated = [h for h in hosts if h in fabric.vertices]
    cut: Set[frozenset] = set()
    down: Set[str] = set()
    out: List[Action] = []
    last: Dict[str, List[str]] = {}

    def blocked() -> Dict[str, List[str]]:
        pairs: Set[Tuple[str, str]] = set()
        if gating:
            island = fabric.islands(down, cut)
            live = [h for h in gated if h not in down]
            pairs = {(a, b) for a in live for b in live
                     if a != b and island.get(a) != island.get(b)}
        else:
            for link in cut:
                a, b = sorted(link)
                if a in plan["hosts"] and b in plan["hosts"]:
                    pairs |= {(a, b), (b, a)}
        by_host: Dict[str, List[str]] = {}
        for a, b in sorted(pairs):
            by_host.setdefault(a, []).append(b)
        return by_host

    # A topology that already separates hosts is partitioned from the start.
    for t, op, arg in [(BEFORE_START, "init", None)] + sorted(events, key=lambda e: e[0]):
        if op == "cut":
            cut.add(link_key(*arg))
        elif op == "heal":
            cut.discard(link_key(*arg))
        elif op == "down":
            down.add(arg)
        elif op == "up":
            down.discard(arg)
        now = blocked()
        if now != last:
            out.append(Action(t, "partition", control={"blocked": now}))
            last = now
    return out


def plan_actions(faults: List[FaultSpec], plan: Dict[str, Any], duration_s: float,
                 seed: int = 0, local: bool = False) -> ChaosPlan:
    hosts = set(plan["hosts"])
    net = plan.get("network") or {}
    gating = bool(net.get("gating"))
    # Network vertices with no container: switches, routers, empty Nodes.
    bare = set(net.get("vertices", [])) - hosts
    out = ChaosPlan()
    if local:
        expanded = []
        for f in faults:
            if f.target in hosts:
                expanded += [FaultSpec(c, f.mode, f.start_s, f.duration_s, f.params)
                             for c in _residents(plan, f.target)]
            elif f.mode == M.LINK_FLAP or f.target in bare:
                out.unsupported.append({**f.to_dict(), "reason": "no network in local mode"})
            else:
                expanded.append(f)
        faults, hosts, bare = expanded, set(), set()
    events: List[Tuple[float, str, Any]] = []
    for i, f in enumerate(faults):
        end = f.end_s if f.end_s is not None else None
        p = f.params
        if f.mode == M.LINK_FLAP:
            link = tuple(f.target.split("|", 1))
            if not gating and not set(link) <= hosts:
                out.unsupported.append({**f.to_dict(), "reason":
                                        "a link to a container-less vertex needs network_gating"})
                continue
            for cut, heal in _flap_periods(f, i, seed, duration_s):
                events += [(cut, "cut", link), (heal, "heal", link)]
            continue
        if f.target in bare:
            if gating and f.mode in (M.CRASH, M.TRANSIENT_CRASH):
                events.append((f.start_s, "down", f.target))
                if f.mode == M.TRANSIENT_CRASH:
                    events.append((end, "up", f.target))
            else:
                out.unsupported.append({**f.to_dict(), "reason": "network vertex without a container"
                                        + ("" if gating else "; crashes need network_gating")})
            continue
        if f.target in hosts:
            c = container_name(f.target)
            if f.mode in (M.CRASH, M.TRANSIENT_CRASH):
                out.actions.append(Action(f.start_s, "docker", ["docker", "kill", c]))
                events.append((f.start_s, "down", f.target))
                if f.mode == M.TRANSIENT_CRASH:
                    out.actions.append(Action(end, "docker", ["docker", "start", c]))
                    events.append((end, "up", f.target))
            elif f.mode == M.FREEZE:
                out.actions.append(Action(f.start_s, "docker", ["docker", "pause", c]))
                out.actions.append(Action(end, "docker", ["docker", "unpause", c]))
            elif f.mode == M.SLOWDOWN:
                k = float(p.get("k", 4.0))
                for name in _residents(plan, f.target):
                    out.actions.append(_control(f.start_s, name, slowdown=k))
                    if end is not None:
                        out.actions.append(_control(end, name))
            else:
                out.unsupported.append({**f.to_dict(), "reason": "host-level mode not emulated"})
            continue
        state = {
            M.CRASH: {"crashed": True}, M.TRANSIENT_CRASH: {"crashed": True},
            M.FREEZE: {"frozen": True},
            M.SLOWDOWN: {"slowdown": float(p.get("k", 4.0))},
            M.OMISSION: {"omission": float(p.get("p", 0.5))},
            M.BABBLING: {"babbling": float(p.get("k", 5.0))},
        }.get(f.mode)
        if state is None:
            out.unsupported.append({**f.to_dict(), "reason": "not emulated on the testbed"})
            continue
        out.actions.append(_control(f.start_s, f.target, **state))
        if end is not None:
            out.actions.append(_control(end, f.target))
    if hosts:
        # After the docker actions of the same instant: a restarted host gets its rules back.
        out.actions += _partitions(events, plan)
    out.actions.sort(key=lambda a: a.t)
    return out


class _Partitioner:
    """Applies partition actions as per-container ingress drop rules, changing only what changed."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.applied: Dict[str, Optional[Tuple[str, ...]]] = {}
        self.ips: Dict[str, Optional[str]] = {}

    def _ip(self, host: str) -> Optional[str]:
        if host not in self.ips:
            res = self.runner(["docker", "inspect", "-f",
                               "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                               container_name(host)], capture_output=True, text=True)
            ip = (getattr(res, "stdout", "") or "").strip()
            self.ips[host] = ip or None
        return self.ips[host]

    def restarted(self, container: str) -> None:
        """A restarted container lost its rules and may have a new address: redo every rule."""
        for h in list(self.ips):
            if container_name(h) == container:
                del self.ips[h]
        self.applied = {h: None for h in self.applied}

    def apply(self, blocked: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        log = []
        for host in sorted(set(blocked) | set(self.applied)):
            want = tuple(blocked.get(host, ()))
            if self.applied.get(host, ()) == want:
                continue
            cmd = "tc qdisc del dev eth0 ingress 2>/dev/null; true"
            if want:
                rules = ["tc qdisc add dev eth0 handle ffff: ingress"]
                rules += [f"tc filter add dev eth0 parent ffff: protocol ip u32 match ip src {ip}/32 action drop"
                          for ip in map(self._ip, want) if ip]
                cmd += "; " + " && ".join(rules)
            res = self.runner(["docker", "exec", container_name(host), "sh", "-c", cmd],
                              capture_output=True, text=True)
            self.applied[host] = want
            log.append({"partition": host, "blocked": list(want),
                        "rc": getattr(res, "returncode", None)})
        return log


def execute(plan: ChaosPlan, control_dir: Path, t0: float,
            runner=subprocess.run, sleep=time.sleep, clock=time.time) -> List[Dict[str, Any]]:
    """Apply the actions at their times (relative to *t0*); returns an action log."""
    log = []
    net = _Partitioner(runner)
    for a in plan.actions:
        delay = t0 + a.t - clock()
        if delay > 0:
            sleep(delay)
        if a.kind == "control":
            (Path(control_dir) / f"{a.target}.json").write_text(json.dumps(a.control))
            log.append({"t": clock() - t0, "control": a.target, "state": a.control})
        elif a.kind == "partition":
            log += [{"t": clock() - t0, **e} for e in net.apply(a.control["blocked"])]
        else:
            res = runner(a.argv, capture_output=True, text=True)
            if a.argv[:2] == ["docker", "start"]:
                net.restarted(a.argv[2])
            log.append({"t": clock() - t0, "argv": a.argv, "rc": getattr(res, "returncode", None)})
    return log
