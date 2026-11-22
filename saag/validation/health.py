"""
End-of-simulation operational health-status summary and derived criticality.

Evaluates system runtime health across all five component types (``Node``,
``Application``, ``Topic``, ``Broker``, ``NetworkDevice``), classifies each
entity as ``healthy``, ``degraded``, or ``lost``, derives operational
criticality for infrastructure entities from the topics/applications/flows they
support, and computes ISO/IEC 25019:2023 Product-in-Use health indices.

Operates strictly on ``TelemetryTrace`` + deployment ``plan`` (tier-agnostic),
so Tier 1 (`MessageFlowSimulator`) and Tier 2 (`saag/testbed`) runs are
evaluated identically and compared side-by-side via `compare_health_summaries`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from saag.core.network import Fabric, link_key, place
from saag.core.quality_model import DOMAIN_PRIORITIES, QIU_ORDER
from saag.simulation.calibration import TelemetryTrace
from saag.validation.fidelity import stalled_in_trace

COMPONENT_TYPES: Tuple[str, ...] = ("Node", "Application", "Topic", "Broker", "NetworkDevice")
CRITICALITY_LEVELS: Tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
CRITICALITY_RANK: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
CRITICALITY_WEIGHT: Dict[str, float] = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "CRITICAL": 4.0}
DEFAULT_OMEGA: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "health_summary.schema.json"


@dataclass(frozen=True)
class HealthThresholds:
    """Configurable thresholds for end-of-simulation health classification."""

    healthy_attainment_min: float = 0.90
    deadline_miss_rate_max: float = 0.05
    cpu_p95_max: float = 80.0
    tail_window_s: float = 10.0
    operational_critical_levels: Tuple[str, ...] = ("CRITICAL", "HIGH")
    escalate_app_from_topics: bool = False


def load_health_schema() -> Dict[str, Any]:
    """Load the Draft 2020-12 JSON Schema for health-status summaries."""
    return json.loads(SCHEMA_PATH.read_text())


# ── Criticality helpers ──────────────────────────────────────────────────────

def _norm_crit(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = val.value if hasattr(val, "value") else str(val)
    s = s.strip().upper()
    return s if s in CRITICALITY_RANK else None


def _max_crit_with_sources(
    candidates: Iterable[Tuple[str, str]],
) -> Tuple[Optional[str], List[str]]:
    """Return ``(highest_level, sorted_driving_source_ids)`` from ``(level, source_id)`` pairs."""
    best_rank = -1
    best_level: Optional[str] = None
    sources: Set[str] = set()
    for lvl, src in candidates:
        r = CRITICALITY_RANK.get(lvl, -1)
        if r > best_rank:
            best_rank = r
            best_level = lvl
            sources = {src}
        elif r == best_rank and r >= 0:
            sources.add(src)
    return best_level, sorted(sources)


def _broker_host_map(plan: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for b, spec in plan.get("brokers", {}).items():
        if isinstance(spec, Mapping) and spec.get("host"):
            out[b] = str(spec["host"])
            continue
        host = next((h for h, r in plan.get("hosts", {}).items() if b in r.get("brokers", ())), None)
        out[b] = host
    return out


def _network_devices_in_plan(plan: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if isinstance(plan.get("network_devices"), Mapping):
        return {str(k): dict(v) for k, v in plan["network_devices"].items()}
    net = plan.get("network") or {}
    if isinstance(net.get("devices"), Mapping):
        return {str(k): dict(v) for k, v in net["devices"].items()}
    hosts = set(plan.get("hosts", {}))
    return {
        str(v): {"role": "switch", "criticality": None}
        for v in sorted(set(net.get("vertices", ())) - hosts)
    }


def derive_operational_criticality(
    plan: Mapping[str, Any],
    *,
    escalate_app_from_topics: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Resolve operational criticality for every entity in *plan*.

    - ``Topic``: uses declared criticality (fallback ``"LOW"``).
    - ``Application``: uses declared criticality; derives from published/subscribed
      topics when undeclared (or always takes ``max(declared, topics)`` when
      ``escalate_app_from_topics=True``).
    - ``Broker``: ``max(declared, max_criticality(routed_topics))``.
    - ``Node``: ``max(declared, resident_apps, resident_brokers, resident_app_topics)``.
    - ``NetworkDevice``: ``max(declared, criticality of flows traversing the device
      on shortest/ECMP paths)``, falling back to adjacent ``Node``s when no flows
      cross it.
    """
    result: Dict[str, Dict[str, Any]] = {}

    def _entry(level: str, provenance: str, derived_from: Sequence[str] = ()) -> Dict[str, Any]:
        return {
            "level": level,
            "weight": CRITICALITY_WEIGHT[level],
            "provenance": provenance,
            "derived_from": list(derived_from),
        }

    # 1. Topics
    topic_crit: Dict[str, str] = {}
    for t, spec in sorted(plan.get("topics", {}).items()):
        decl = _norm_crit(spec.get("criticality"))
        if decl is not None:
            topic_crit[t] = decl
            result[t] = _entry(decl, "declared")
        else:
            topic_crit[t] = "LOW"
            result[t] = _entry("LOW", "default")

    # 2. Applications
    app_crit: Dict[str, str] = {}
    for a, spec in sorted(plan.get("apps", {}).items()):
        decl = _norm_crit(spec.get("criticality"))
        rel_topics = sorted(set(spec.get("publishes", ())) | set(spec.get("subscribes", ())))
        t_lvl, t_srcs = _max_crit_with_sources(
            (topic_crit[t], t) for t in rel_topics if t in topic_crit
        )
        if decl is not None and not escalate_app_from_topics:
            app_crit[a] = decl
            result[a] = _entry(decl, "declared")
        elif decl is not None and escalate_app_from_topics:
            if t_lvl is not None and CRITICALITY_RANK[t_lvl] > CRITICALITY_RANK[decl]:
                app_crit[a] = t_lvl
                result[a] = _entry(t_lvl, "derived", t_srcs)
            else:
                app_crit[a] = decl
                result[a] = _entry(decl, "declared")
        elif t_lvl is not None:
            app_crit[a] = t_lvl
            result[a] = _entry(t_lvl, "derived", t_srcs)
        else:
            app_crit[a] = "LOW"
            result[a] = _entry("LOW", "default")

    # 3. Brokers (derived from routed topics)
    broker_crit: Dict[str, str] = {}
    for b, spec in sorted(plan.get("brokers", {}).items()):
        decl = _norm_crit(spec.get("criticality"))
        routed = [t for t in spec.get("routes", ()) if t in topic_crit]
        r_lvl, r_srcs = _max_crit_with_sources((topic_crit[t], t) for t in routed)
        if decl is not None and (r_lvl is None or CRITICALITY_RANK[decl] > CRITICALITY_RANK[r_lvl]):
            broker_crit[b] = decl
            result[b] = _entry(decl, "declared")
        elif r_lvl is not None:
            broker_crit[b] = r_lvl
            result[b] = _entry(r_lvl, "derived", r_srcs)
        else:
            broker_crit[b] = "LOW"
            result[b] = _entry("LOW", "default")

    # 4. Nodes / Hosts (derived from resident apps, resident brokers, and resident app topics)
    host_crit_decl = plan.get("host_criticality") or {}
    node_crit: Dict[str, str] = {}
    for h, spec in sorted(plan.get("hosts", {}).items()):
        decl = _norm_crit(spec.get("criticality") if spec.get("criticality") is not None else host_crit_decl.get(h))
        cands: List[Tuple[str, str]] = []
        for a in spec.get("apps", ()):
            if a in app_crit:
                cands.append((app_crit[a], a))
            a_spec = plan.get("apps", {}).get(a, {})
            for t in set(a_spec.get("publishes", ())) | set(a_spec.get("subscribes", ())):
                if t in topic_crit:
                    cands.append((topic_crit[t], t))
        for b in spec.get("brokers", ()):
            if b in broker_crit:
                cands.append((broker_crit[b], b))
        d_lvl, d_srcs = _max_crit_with_sources(cands)
        if decl is not None and (d_lvl is None or CRITICALITY_RANK[decl] > CRITICALITY_RANK[d_lvl]):
            node_crit[h] = decl
            result[h] = _entry(decl, "declared")
        elif d_lvl is not None:
            node_crit[h] = d_lvl
            result[h] = _entry(d_lvl, "derived", d_srcs)
        else:
            node_crit[h] = "LOW"
            result[h] = _entry("LOW", "default")

    # 5. NetworkDevices (derived from carried pub-sub flows via ECMP shortest paths)
    devices = _network_devices_in_plan(plan)
    if devices:
        net = plan.get("network") or {}
        vertices = set(net.get("vertices", ())) | set(plan.get("hosts", ())) | set(devices)
        links = [(a, b, None) for a, b in plan.get("links", ())]
        fabric = Fabric(vertices, links)
        b_host = _broker_host_map(plan)
        subs_by_topic: Dict[str, List[str]] = defaultdict(list)
        for a, a_spec in plan.get("apps", {}).items():
            for t in a_spec.get("subscribes", ()):
                subs_by_topic[t].append(a)

        device_flow_cands: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        dag_cache: Dict[str, Any] = {}
        for t, t_spec in plan.get("topics", {}).items():
            pubs = list(t_spec.get("publishers", ()))
            subs = subs_by_topic.get(t, [])
            if not pubs or not subs:
                continue
            routers = list(t_spec.get("routers", ()))
            broker_h = place([b_host[r] for r in routers if b_host.get(r)]) if routers else None
            for p in pubs:
                p_hosts = plan.get("apps", {}).get(p, {}).get("hosts", ())
                ph = place(p_hosts)
                if not ph:
                    continue
                for s in subs:
                    if s == p:
                        continue
                    s_hosts = plan.get("apps", {}).get(s, {}).get("hosts", ())
                    sh = place(s_hosts)
                    if not sh:
                        continue
                    flow_lvl = max(
                        (topic_crit.get(t, "LOW"), app_crit.get(p, "LOW"), app_crit.get(s, "LOW")),
                        key=lambda lv: CRITICALITY_RANK[lv],
                    )
                    legs = [(ph, broker_h), (broker_h, sh)] if broker_h else [(ph, sh)]
                    for src, dst in legs:
                        if not src or not dst or src == dst:
                            continue
                        shares = fabric.ecmp_shares(src, dst, down=set(), cut=set(), _cache=dag_cache)
                        if not shares:
                            continue
                        touched = {u for u, _ in shares} | {v for _, v in shares}
                        for d in touched & set(devices):
                            device_flow_cands[d].append((flow_lvl, t))

        for d, d_spec in sorted(devices.items()):
            decl = _norm_crit(d_spec.get("criticality"))
            cands = device_flow_cands.get(d, [])
            if not cands:
                # Fallback: directly connected hosts
                cands = [(node_crit[n], n) for n in sorted(fabric.adj.get(d, ())) if n in node_crit]
            d_lvl, d_srcs = _max_crit_with_sources(cands)
            if decl is not None and (d_lvl is None or CRITICALITY_RANK[decl] > CRITICALITY_RANK[d_lvl]):
                result[d] = _entry(decl, "declared")
            elif d_lvl is not None:
                result[d] = _entry(d_lvl, "derived", d_srcs)
            else:
                result[d] = _entry("LOW", "default")

    return result


# ── Fault parsing helpers ────────────────────────────────────────────────────

def _fault_dict(f: Any) -> Dict[str, Any]:
    if isinstance(f, Mapping):
        return dict(f)
    if hasattr(f, "to_dict"):
        return f.to_dict()
    return {
        "target": getattr(f, "target", ""),
        "mode": getattr(getattr(f, "mode", ""), "value", str(getattr(f, "mode", ""))),
        "start_s": float(getattr(f, "start_s", 0.0)),
        "duration_s": getattr(f, "duration_s", None),
        "params": dict(getattr(f, "params", {}) or {}),
    }


def _active_at(f: Mapping[str, Any], t: float) -> bool:
    start = float(f.get("start_s", 0.0))
    dur = f.get("duration_s")
    if t < start:
        return False
    if dur is None:
        return True
    return t <= start + float(dur) + 1e-9


# ── Aggregation helpers ──────────────────────────────────────────────────────

def _counts_and_shares(states: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, Optional[float]]]:
    total = len(states)
    h = sum(1 for s in states if s == "healthy")
    d = sum(1 for s in states if s == "degraded")
    lo = sum(1 for s in states if s == "lost")
    counts = {"total": total, "healthy": h, "degraded": d, "lost": lo}
    if total == 0:
        shares: Dict[str, Optional[float]] = {"healthy": None, "degraded": None, "lost": None}
    else:
        shares = {
            "healthy": round(h / total, 6),
            "degraded": round(d / total, 6),
            "lost": round(lo / total, 6),
        }
    return counts, shares


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve_domain_omega(domain: Optional[str]) -> Tuple[Tuple[float, float, float], bool]:
    key = (domain or "").strip().lower()
    if key in DOMAIN_PRIORITIES:
        return DOMAIN_PRIORITIES[key], True
    return DEFAULT_OMEGA, False


# ── Main health evaluator ────────────────────────────────────────────────────

def summarize_system_health(
    trace: TelemetryTrace,
    plan: Mapping[str, Any],
    *,
    fault_time: Optional[float] = None,
    warmup_s: float = 0.0,
    faults: Iterable[Any] = (),
    docker_cpu_p95: Optional[Mapping[str, Optional[float]]] = None,
    thresholds: Optional[HealthThresholds] = None,
) -> Dict[str, Any]:
    """Compute the end-of-simulation operational health-status summary.

    Returns a JSON-serializable dict conforming to
    ``saag/validation/schemas/health_summary.schema.json``.
    """
    th = thresholds or HealthThresholds()
    crit_map = derive_operational_criticality(
        plan, escalate_app_from_topics=th.escalate_app_from_topics
    )
    op_crit_levels = set(th.operational_critical_levels)

    end_s = trace.end_s
    start_s = trace.start_s
    if fault_time is not None and end_s > fault_time:
        post_span = end_s - fault_time
        tail_start = max(fault_time, end_s - min(th.tail_window_s, post_span))
    else:
        active_span = max(0.0, end_s - warmup_s)
        tail_start = max(warmup_s, end_s - min(th.tail_window_s, active_span))
    tail_span = max(end_s - tail_start, 1e-9)

    has_pre_window = fault_time is not None and (fault_time - warmup_s) >= 0.5
    pre_span = (fault_time - warmup_s) if has_pre_window and fault_time is not None else 0.0

    # Parse active faults at end_s
    fault_list = [_fault_dict(f) for f in faults]
    initial_fault_targets: Set[str] = {str(f["target"]) for f in fault_list if f.get("target")}
    crashed_at_end: Set[str] = set()
    frozen_at_end: Set[str] = set()
    degraded_fault_at_end: Dict[str, List[str]] = defaultdict(list)
    flapping_links_at_end: Set[frozenset] = set()
    for f in fault_list:
        if not _active_at(f, end_s):
            continue
        mode = str(f.get("mode", "")).lower()
        target = str(f.get("target", ""))
        if mode in ("crash", "transient_crash"):
            crashed_at_end.add(target)
        elif mode == "freeze":
            frozen_at_end.add(target)
        elif mode in ("slowdown", "omission", "babbling", "buffer_leak", "redelivery_storm"):
            degraded_fault_at_end[target].append(f"fault:{mode}")
        elif mode == "link_flap" and "|" in target:
            a, b = target.split("|", 1)
            flapping_links_at_end.add(link_key(a, b))

    # Index events in pre-window, tail-window, and full run
    pre_pub_app: Dict[str, int] = defaultdict(int)
    pre_pub_topic: Dict[str, int] = defaultdict(int)
    pre_deliv_app: Dict[str, int] = defaultdict(int)
    pre_deliv_topic: Dict[str, int] = defaultdict(int)
    pre_fwd_broker: Dict[str, int] = defaultdict(int)

    tail_pub_app: Dict[str, int] = defaultdict(int)
    tail_pub_topic: Dict[str, int] = defaultdict(int)
    tail_deliv_app: Dict[str, int] = defaultdict(int)
    tail_deliv_topic: Dict[str, int] = defaultdict(int)
    tail_dm_app: Dict[str, int] = defaultdict(int)
    tail_dm_topic: Dict[str, int] = defaultdict(int)
    tail_drop_comp: Dict[str, int] = defaultdict(int)
    tail_drop_topic: Dict[str, int] = defaultdict(int)
    tail_fwd_broker: Dict[str, int] = defaultdict(int)

    takeover_apps: Set[str] = set()
    takeover_count = 0

    for e in trace.events:
        t = e["t"]
        kind = e["kind"]
        app = e["app"]
        topic = e["topic"]

        if kind == "state" and e.get("state") == "takeover":
            takeover_count += 1
            if app:
                takeover_apps.add(app)

        if has_pre_window and warmup_s <= t < fault_time:  # type: ignore[operator]
            if kind == "publish":
                if app:
                    pre_pub_app[app] += 1
                if topic:
                    pre_pub_topic[topic] += 1
            elif kind == "deliver":
                if app:
                    pre_deliv_app[app] += 1
                if topic:
                    pre_deliv_topic[topic] += 1
            elif kind == "forward" and app:
                pre_fwd_broker[app] += 1

        if tail_start <= t <= end_s:
            if kind == "publish":
                if app:
                    tail_pub_app[app] += 1
                if topic:
                    tail_pub_topic[topic] += 1
            elif kind == "deliver":
                if app:
                    tail_deliv_app[app] += 1
                if topic:
                    tail_deliv_topic[topic] += 1
            elif kind == "deadline_miss":
                if app:
                    tail_dm_app[app] += 1
                if topic:
                    tail_dm_topic[topic] += 1
            elif kind == "drop":
                if app:
                    tail_drop_comp[app] += 1
                if topic:
                    tail_drop_topic[topic] += 1
            elif kind == "forward" and app:
                tail_fwd_broker[app] += 1

    # DDS status counts (post-fault delta when fault_time is set, else full run)
    end_status = trace.status_counts()
    pre_status = trace.status_counts(until=fault_time) if fault_time is not None else {}

    def _status_delta(status_name: str, comp: Optional[str] = None, topic: Optional[str] = None) -> int:
        end_map = end_status.get(status_name, {})
        pre_map = pre_status.get(status_name, {})
        total = 0
        for key, val in end_map.items():
            _, k_topic, k_comp = key.split("|", 2)
            if comp is not None and k_comp != comp:
                continue
            if topic is not None and k_topic != topic:
                continue
            delta = val - pre_map.get(key, 0) if fault_time is not None else val
            if delta > 0:
                total += delta
        return total

    stalled_apps: Set[str] = (
        stalled_in_trace(trace, fault_time) if fault_time is not None else set()
    )

    # Network topology & islands at end_s
    hosts_spec = plan.get("hosts", {})
    apps_spec = plan.get("apps", {})
    topics_spec = plan.get("topics", {})
    brokers_spec = plan.get("brokers", {})
    devices_spec = _network_devices_in_plan(plan)
    b_host = _broker_host_map(plan)
    net_cfg = plan.get("network") or {}
    network_gating = bool(net_cfg.get("gating"))

    all_vertices = set(net_cfg.get("vertices", ())) | set(hosts_spec) | set(devices_spec)
    fabric = Fabric(all_vertices, [(a, b, None) for a, b in plan.get("links", ())])
    down_net_vertices = (set(hosts_spec) | set(devices_spec)) & (crashed_at_end | frozen_at_end)
    base_islands = fabric.islands()
    end_islands = fabric.islands(down=down_net_vertices, cut=flapping_links_at_end)

    # Helper to look up container CPU p95
    def _cpu_for_host(h: str) -> Optional[float]:
        if not docker_cpu_p95:
            return None
        for k in (h, f"saag-{h}"):
            if k in docker_cpu_p95 and docker_cpu_p95[k] is not None:
                return round(float(docker_cpu_p95[k]), 3)
        return None

    subs_by_topic: Dict[str, List[str]] = defaultdict(list)
    for a, spec in apps_spec.items():
        for t in spec.get("subscribes", ()):
            subs_by_topic[t].append(a)

    components: Dict[str, Dict[str, Any]] = {}

    # ── 1. Evaluate Applications ─────────────────────────────────────────────
    for a, spec in sorted(apps_spec.items()):
        app_hosts = list(spec.get("hosts", ()))
        live_hosts = [h for h in app_hosts if h not in crashed_at_end and h not in frozen_at_end]
        red_init = max(0, len(app_hosts) - 1) if app_hosts else 0
        red_rem = max(0, len(live_hosts) - 1) if live_hosts else 0

        # Baseline rates
        decl_pub_rate = sum(float(r) for r in spec.get("per_publisher_rate", {}).values())
        decl_deliv_rate = sum(
            float(topics_spec[t]["rate_hz"]) for t in spec.get("subscribes", ()) if t in topics_spec
        )
        base_pub_rate = (
            (pre_pub_app[a] / pre_span) if (has_pre_window and pre_pub_app[a] > 0) else decl_pub_rate
        )
        base_deliv_rate = (
            (pre_deliv_app[a] / pre_span)
            if (has_pre_window and pre_deliv_app[a] > 0)
            else decl_deliv_rate
        )

        tail_pub_rate = tail_pub_app[a] / tail_span
        tail_deliv_rate = tail_deliv_app[a] / tail_span
        em_att = (tail_pub_rate / base_pub_rate) if base_pub_rate > 0 else None
        del_att = (tail_deliv_rate / base_deliv_rate) if base_deliv_rate > 0 else None

        # Deadline misses (both engine drop and DDS status)
        dds_dm = _status_delta("requested_deadline_missed", comp=a) + _status_delta(
            "offered_deadline_missed", comp=a
        )
        dm_total = tail_dm_app[a] + dds_dm
        denom_dm = tail_deliv_app[a] + tail_dm_app[a] + (tail_pub_app[a] if dds_dm and not tail_deliv_app[a] else 0)
        dm_rate = (dm_total / (denom_dm + dds_dm)) if (denom_dm + dds_dm) > 0 else 0.0
        drops = tail_drop_comp[a] + _status_delta("cache_replaced_dropped_samples", comp=a) + _status_delta("sample_rejected", comp=a)

        reasons: List[str] = []
        is_lost = False
        if a in crashed_at_end or (app_hosts and not live_hosts):
            is_lost = True
            reasons.append("crashed" if a in crashed_at_end else "all_hosts_down")
        elif a in frozen_at_end:
            is_lost = True
            reasons.append("frozen")
        elif a in stalled_apps and base_pub_rate > 0:
            is_lost = True
            reasons.append("stalled")
        elif (
            (em_att is not None and em_att == 0.0 and (del_att is None or del_att == 0.0))
            or (em_att is None and del_att is not None and del_att == 0.0)
        ):
            is_lost = True
            reasons.append("zero_tail_activity")

        if is_lost:
            state = "lost"
            h_score = 0.0
        else:
            if a in degraded_fault_at_end:
                reasons.extend(degraded_fault_at_end[a])
            if em_att is not None and em_att < th.healthy_attainment_min:
                reasons.append(f"emission_below_threshold:{em_att:.3f}")
            if del_att is not None and del_att < th.healthy_attainment_min:
                reasons.append(f"delivery_below_threshold:{del_att:.3f}")
            if dm_rate > th.deadline_miss_rate_max:
                reasons.append(f"deadline_miss_rate:{dm_rate:.3f}")
            if drops > 0:
                reasons.append(f"drops:{drops}")
            if red_init > 0 and red_rem < red_init:
                reasons.append("redundancy_depleted")

            if reasons:
                state = "degraded"
                att_vals = [_clamp01(v) for v in (em_att, del_att) if v is not None]
                base_score = min(att_vals) if att_vals else 0.75
                if dm_rate > th.deadline_miss_rate_max:
                    base_score = min(base_score, _clamp01(1.0 - dm_rate))
                if base_score >= 0.99:
                    base_score = 0.85 if "redundancy_depleted" in reasons else 0.90
                h_score = round(max(0.05, base_score), 6)
            else:
                state = "healthy"
                h_score = 1.0

        components[a] = {
            "id": a,
            "type": "Application",
            "state": state,
            "health_score": h_score,
            "criticality": crit_map[a],
            "reasons": sorted(set(reasons)),
            "metrics": {
                "delivery_attainment": round(del_att, 6) if del_att is not None else None,
                "emission_attainment": round(em_att, 6) if em_att is not None else None,
                "deadline_miss_rate": round(dm_rate, 6),
                "drop_count": int(drops),
                "redundancy_remaining": red_rem,
                "redundancy_initial": red_init,
                "failover_occurred": a in takeover_apps,
                "cpu_p95": None,
            },
        }

    # ── 2. Evaluate Brokers ──────────────────────────────────────────────────
    for b, spec in sorted(brokers_spec.items()):
        host = b_host.get(b)
        routed = list(spec.get("routes", ()))
        decl_fwd = sum(float(topics_spec[t]["rate_hz"]) for t in routed if t in topics_spec)
        base_fwd = (pre_fwd_broker[b] / pre_span) if (has_pre_window and pre_fwd_broker[b] > 0) else decl_fwd
        tail_fwd = tail_fwd_broker[b] / tail_span
        # Only score forwarding attainment if saag.forward events were actually recorded
        has_fwd_telemetry = (pre_fwd_broker[b] + tail_fwd_broker[b]) > 0
        fwd_att = (tail_fwd / base_fwd) if (has_fwd_telemetry and base_fwd > 0) else None
        drops = tail_drop_comp[b]

        reasons = []
        if b in crashed_at_end or (host is not None and host in crashed_at_end):
            state, h_score = "lost", 0.0
            reasons.append("crashed" if b in crashed_at_end else "host_down")
        elif b in frozen_at_end or (host is not None and host in frozen_at_end):
            state, h_score = "lost", 0.0
            reasons.append("frozen")
        else:
            if b in degraded_fault_at_end:
                reasons.extend(degraded_fault_at_end[b])
            if fwd_att is not None and fwd_att < th.healthy_attainment_min:
                reasons.append(f"forward_below_threshold:{fwd_att:.3f}")
            if drops > 0:
                reasons.append(f"drops:{drops}")
            if reasons:
                state = "degraded"
                h_score = round(max(0.05, _clamp01(fwd_att if fwd_att is not None else 0.75)), 6)
            else:
                state, h_score = "healthy", 1.0

        components[b] = {
            "id": b,
            "type": "Broker",
            "state": state,
            "health_score": h_score,
            "criticality": crit_map[b],
            "reasons": sorted(set(reasons)),
            "metrics": {
                "delivery_attainment": round(fwd_att, 6) if fwd_att is not None else None,
                "emission_attainment": None,
                "deadline_miss_rate": None,
                "drop_count": int(drops),
                "redundancy_remaining": None,
                "redundancy_initial": None,
                "failover_occurred": False,
                "cpu_p95": None,
            },
        }

    # ── 3. Evaluate Topics ───────────────────────────────────────────────────
    for t, spec in sorted(topics_spec.items()):
        pubs = list(spec.get("publishers", ()))
        subs = subs_by_topic.get(t, [])
        routers = list(spec.get("routers", ()))

        live_pubs = [p for p in pubs if p in components and components[p]["state"] != "lost"]
        live_subs = [s for s in subs if s in components and components[s]["state"] != "lost"]
        live_routers = [r for r in routers if r in components and components[r]["state"] != "lost"]
        red_init = max(0, len(routers) - 1) if routers else (max(0, len(pubs) - 1) if pubs else 0)
        red_rem = max(0, len(live_routers) - 1) if routers else (max(0, len(live_pubs) - 1) if pubs else 0)

        decl_pub = float(spec.get("rate_hz", 0.0))
        decl_deliv = decl_pub * len(subs)
        base_pub = (pre_pub_topic[t] / pre_span) if (has_pre_window and pre_pub_topic[t] > 0) else decl_pub
        base_deliv = (
            (pre_deliv_topic[t] / pre_span) if (has_pre_window and pre_deliv_topic[t] > 0) else decl_deliv
        )
        em_att = ((tail_pub_topic[t] / tail_span) / base_pub) if base_pub > 0 else None
        del_att = ((tail_deliv_topic[t] / tail_span) / base_deliv) if base_deliv > 0 else None

        dds_dm = _status_delta("requested_deadline_missed", topic=t) + _status_delta(
            "offered_deadline_missed", topic=t
        )
        dm_total = tail_dm_topic[t] + dds_dm
        denom_dm = tail_deliv_topic[t] + tail_dm_topic[t] + dds_dm
        dm_rate = (dm_total / denom_dm) if denom_dm > 0 else 0.0
        drops = (
            tail_drop_topic[t]
            + _status_delta("cache_replaced_dropped_samples", topic=t)
            + _status_delta("sample_rejected", topic=t)
            + _status_delta("sample_lost", topic=t)
        )

        reasons = []
        is_lost = False
        if pubs and not live_pubs:
            is_lost = True
            reasons.append("orphaned_no_live_publishers")
        elif routers and not live_routers:
            is_lost = True
            reasons.append("no_live_brokers")
        elif subs and not live_subs:
            is_lost = True
            reasons.append("no_live_subscribers")
        elif (
            (base_deliv > 0 and del_att == 0.0)
            or (base_deliv == 0 and base_pub > 0 and em_att == 0.0)
        ):
            is_lost = True
            reasons.append("zero_tail_delivery")

        if is_lost:
            state, h_score = "lost", 0.0
        else:
            if len(live_pubs) < len(pubs):
                reasons.append(f"partial_publisher_loss:{len(live_pubs)}/{len(pubs)}")
            if routers and len(live_routers) < len(routers):
                reasons.append(f"broker_redundancy_depleted:{len(live_routers)}/{len(routers)}")
            if del_att is not None and del_att < th.healthy_attainment_min:
                reasons.append(f"delivery_below_threshold:{del_att:.3f}")
            if em_att is not None and em_att < th.healthy_attainment_min:
                reasons.append(f"emission_below_threshold:{em_att:.3f}")
            if dm_rate > th.deadline_miss_rate_max:
                reasons.append(f"deadline_miss_rate:{dm_rate:.3f}")
            if drops > 0:
                reasons.append(f"drops:{drops}")

            if reasons:
                state = "degraded"
                att_vals = [_clamp01(v) for v in (del_att, em_att) if v is not None]
                base_score = min(att_vals) if att_vals else 0.80
                if dm_rate > th.deadline_miss_rate_max:
                    base_score = min(base_score, _clamp01(1.0 - dm_rate))
                if base_score >= 0.99:
                    base_score = 0.85
                h_score = round(max(0.05, base_score), 6)
            else:
                state, h_score = "healthy", 1.0

        components[t] = {
            "id": t,
            "type": "Topic",
            "state": state,
            "health_score": h_score,
            "criticality": crit_map[t],
            "reasons": sorted(set(reasons)),
            "metrics": {
                "delivery_attainment": round(del_att, 6) if del_att is not None else None,
                "emission_attainment": round(em_att, 6) if em_att is not None else None,
                "deadline_miss_rate": round(dm_rate, 6),
                "drop_count": int(drops),
                "redundancy_remaining": red_rem,
                "redundancy_initial": red_init,
                "failover_occurred": bool(routers and len(live_routers) < len(routers) and live_routers),
                "cpu_p95": None,
            },
        }

    # ── 4. Evaluate Nodes (Hosts) ────────────────────────────────────────────
    live_host_ids = [h for h in hosts_spec if h not in crashed_at_end and h not in frozen_at_end]
    for h, spec in sorted(hosts_spec.items()):
        cpu = _cpu_for_host(h)
        reasons = []
        if h in crashed_at_end:
            state, h_score = "lost", 0.0
            reasons.append("crashed")
        elif h in frozen_at_end:
            state, h_score = "lost", 0.0
            reasons.append("frozen")
        else:
            # Check network partitioning if network_gating is active and there are peers
            if network_gating and len(live_host_ids) > 1 and h in base_islands:
                peers_before = {
                    p for p in live_host_ids if p != h and base_islands.get(p) == base_islands[h]
                }
                peers_after = {
                    p for p in peers_before if end_islands.get(p) == end_islands.get(h)
                }
                if peers_before and not peers_after:
                    reasons.append("network_isolated")
                elif len(peers_after) < len(peers_before):
                    reasons.append(f"partially_partitioned:{len(peers_after)}/{len(peers_before)}")

            if "network_isolated" in reasons:
                state, h_score = "lost", 0.0
            else:
                if h in degraded_fault_at_end:
                    reasons.extend(degraded_fault_at_end[h])
                if cpu is not None and cpu > th.cpu_p95_max:
                    reasons.append(f"cpu_saturated:{cpu:.1f}%")
                residents = list(spec.get("apps", ())) + list(spec.get("brokers", ()))
                bad_residents = [
                    r for r in residents if r in components and components[r]["state"] != "healthy"
                ]
                if bad_residents:
                    reasons.append(f"degraded_or_lost_residents:{len(bad_residents)}")
                if reasons:
                    state = "degraded"
                    res_scores = [components[r]["health_score"] for r in residents if r in components]
                    base_score = (sum(res_scores) / len(res_scores)) if res_scores else 0.75
                    if cpu is not None and cpu > th.cpu_p95_max:
                        base_score = min(base_score, 0.70)
                    h_score = round(max(0.05, min(0.95, base_score)), 6)
                else:
                    state, h_score = "healthy", 1.0

        components[h] = {
            "id": h,
            "type": "Node",
            "state": state,
            "health_score": h_score,
            "criticality": crit_map[h],
            "reasons": sorted(set(reasons)),
            "metrics": {
                "delivery_attainment": None,
                "emission_attainment": None,
                "deadline_miss_rate": None,
                "drop_count": 0,
                "redundancy_remaining": None,
                "redundancy_initial": None,
                "failover_occurred": False,
                "cpu_p95": cpu,
            },
        }

    # ── 5. Evaluate NetworkDevices ───────────────────────────────────────────
    for d in sorted(devices_spec):
        reasons = []
        if d in crashed_at_end:
            state, h_score = "lost", 0.0
            reasons.append("crashed")
        elif d in frozen_at_end:
            state, h_score = "lost", 0.0
            reasons.append("frozen")
        else:
            flapping = [lk for lk in flapping_links_at_end if d in lk]
            if flapping:
                state, h_score = "degraded", 0.5
                reasons.append(f"incident_links_flapping:{len(flapping)}")
            elif d in degraded_fault_at_end:
                state, h_score = "degraded", 0.75
                reasons.extend(degraded_fault_at_end[d])
            else:
                state, h_score = "healthy", 1.0

        components[d] = {
            "id": d,
            "type": "NetworkDevice",
            "state": state,
            "health_score": h_score,
            "criticality": crit_map[d],
            "reasons": sorted(set(reasons)),
            "metrics": {
                "delivery_attainment": None,
                "emission_attainment": None,
                "deadline_miss_rate": None,
                "drop_count": 0,
                "redundancy_remaining": None,
                "redundancy_initial": None,
                "failover_occurred": False,
                "cpu_p95": None,
            },
        }

    # ── System-level aggregations ────────────────────────────────────────────
    all_recs = list(components.values())
    sys_counts, sys_shares = _counts_and_shares([r["state"] for r in all_recs])

    by_type: Dict[str, Any] = {}
    for ctype in COMPONENT_TYPES:
        recs = [r for r in all_recs if r["type"] == ctype]
        c_cnt, c_shr = _counts_and_shares([r["state"] for r in recs])
        by_crit_t: Dict[str, Any] = {}
        for lv in CRITICALITY_LEVELS:
            lv_recs = [r for r in recs if r["criticality"]["level"] == lv]
            lc, ls = _counts_and_shares([r["state"] for r in lv_recs])
            by_crit_t[lv] = {"counts": lc, "shares": ls}
        by_type[ctype] = {
            "counts": c_cnt,
            "shares": c_shr,
            "ids": {
                "healthy": sorted(r["id"] for r in recs if r["state"] == "healthy"),
                "degraded": sorted(r["id"] for r in recs if r["state"] == "degraded"),
                "lost": sorted(r["id"] for r in recs if r["state"] == "lost"),
            },
            "by_criticality": by_crit_t,
        }

    by_criticality: Dict[str, Any] = {}
    for lv in CRITICALITY_LEVELS:
        lv_recs = [r for r in all_recs if r["criticality"]["level"] == lv]
        lc, ls = _counts_and_shares([r["state"] for r in lv_recs])
        by_criticality[lv] = {"counts": lc, "shares": ls}

    op_recs = [r for r in all_recs if r["criticality"]["level"] in op_crit_levels]
    op_counts, op_shares = _counts_and_shares([r["state"] for r in op_recs])
    op_by_type: Dict[str, Any] = {}
    for ctype in COMPONENT_TYPES:
        t_recs = [r for r in op_recs if r["type"] == ctype]
        tc, ts = _counts_and_shares([r["state"] for r in t_recs])
        op_by_type[ctype] = {"counts": tc, "shares": ts}

    # Resilience metrics: time-integrated availability over 1.0 s bins
    ti_avail, op_crit_exposure_s = _time_integrated_metrics(
        trace, plan, crit_map, op_crit_levels, warmup_s=warmup_s
    )
    total_base_deliv = sum(
        (pre_deliv_topic[t] / pre_span) if (has_pre_window and pre_deliv_topic[t] > 0)
        else (float(topics_spec[t]["rate_hz"]) * len(subs_by_topic.get(t, [])))
        for t in topics_spec
    )
    total_tail_deliv = sum(tail_deliv_topic.values()) / tail_span
    e2e_att = round(total_tail_deliv / total_base_deliv, 6) if total_base_deliv > 0 else None

    blast_radius = sum(
        1 for r in all_recs if r["state"] != "healthy" and r["id"] not in initial_fault_targets
    )
    cpu_vals = [v for v in (docker_cpu_p95 or {}).values() if v is not None]
    peak_cpu = round(max(float(v) for v in cpu_vals), 3) if cpu_vals else None

    # ── ISO/IEC 25019:2023 Product-in-Use metrics ────────────────────────────
    # 1. Beneficialness
    sinks = [
        a for a, s in apps_spec.items() if s.get("subscribes") and not s.get("publishes")
    ] or list(apps_spec)
    sink_w_sum = sum(components[a]["criticality"]["weight"] for a in sinks if a in components)
    crit_mission_cont = (
        round(
            sum(
                components[a]["criticality"]["weight"] * components[a]["health_score"]
                for a in sinks if a in components
            ) / sink_w_sum,
            6,
        )
        if sink_w_sum > 0
        else None
    )
    all_w_sum = sum(r["criticality"]["weight"] for r in all_recs)
    crit_weighted_health = (
        round(sum(r["criticality"]["weight"] * r["health_score"] for r in all_recs) / all_w_sum, 6)
        if all_w_sum > 0
        else 1.0
    )
    op_topics = [r for r in op_recs if r["type"] == "Topic"]
    crit_topic_timely = (
        round(
            sum(
                1 for r in op_topics
                if r["state"] == "healthy"
                and (r["metrics"]["deadline_miss_rate"] or 0.0) <= th.deadline_miss_rate_max
            ) / len(op_topics),
            6,
        )
        if op_topics
        else None
    )
    ben_parts = [v for v in (crit_mission_cont, crit_weighted_health, crit_topic_timely) if v is not None]
    ben_score = round(_clamp01(sum(ben_parts) / len(ben_parts)), 6) if ben_parts else 1.0

    # 2. Freedom from Risk
    lost_strict_crit = sum(
        1 for r in all_recs if r["criticality"]["level"] == "CRITICAL" and r["state"] == "lost"
    )
    lost_op_crit = op_counts["lost"]
    op_redundant = [
        r for r in op_recs if (r["metrics"]["redundancy_initial"] or 0) > 0
    ]
    n_minus_0_ids = sorted(
        r["id"] for r in op_redundant if (r["metrics"]["redundancy_remaining"] or 0) == 0
    )
    red_depl_rate = (
        round(len(n_minus_0_ids) / len(op_redundant), 6) if op_redundant else None
    )
    if op_counts["total"] > 0:
        op_lost_shr = op_counts["lost"] / op_counts["total"]
        op_deg_shr = op_counts["degraded"] / op_counts["total"]
        risk_score = round(_clamp01(1.0 - op_lost_shr - 0.5 * op_deg_shr), 6)
    else:
        risk_score = round(crit_weighted_health, 6)

    # 3. Acceptability
    op_sla_entities = [r for r in op_recs if r["type"] in ("Application", "Topic")]
    sla_breaches = [
        r for r in op_sla_entities
        if r["state"] == "lost"
        or (r["metrics"]["deadline_miss_rate"] or 0.0) > th.deadline_miss_rate_max
        or r["metrics"]["drop_count"] > 0
    ]
    sla_compliance = (
        round(1.0 - len(sla_breaches) / len(op_sla_entities), 6) if op_sla_entities else None
    )
    op_ids = {r["id"] for r in op_recs}
    dds_breaches: Dict[str, int] = {}
    for st_name in (
        "requested_deadline_missed",
        "offered_deadline_missed",
        "sample_lost",
        "sample_rejected",
        "cache_replaced_dropped_samples",
        "requested_incompatible_qos",
    ):
        cnt = 0
        for key, val in end_status.get(st_name, {}).items():
            _, k_topic, k_comp = key.split("|", 2)
            if k_comp in op_ids or k_topic in op_ids:
                delta = val - pre_status.get(st_name, {}).get(key, 0) if fault_time is not None else val
                if delta > 0:
                    cnt += delta
        dds_breaches[st_name] = int(cnt)
    acc_score = round(_clamp01(sla_compliance if sla_compliance is not None else 1.0), 6)

    domain = plan.get("domain")
    omega_tuple, omega_derived = _resolve_domain_omega(domain)
    omega_dict = {name: round(val, 6) for name, val in zip(QIU_ORDER, omega_tuple)}
    overall_qiu = round(
        _clamp01(
            omega_tuple[0] * ben_score
            + omega_tuple[1] * risk_score
            + omega_tuple[2] * acc_score
        ),
        6,
    )

    return {
        "schema_version": "1.0",
        "metadata": {
            "scenario": str(plan.get("scenario", "unknown")),
            "domain": domain,
            "window_s": [round(start_s, 6), round(end_s, 6)],
            "tail_window_s": [round(tail_start, 6), round(end_s, 6)],
            "fault_time_s": round(fault_time, 6) if fault_time is not None else None,
            "thresholds": {
                "healthy_attainment_min": th.healthy_attainment_min,
                "deadline_miss_rate_max": th.deadline_miss_rate_max,
                "cpu_p95_max": th.cpu_p95_max,
                "tail_window_s": th.tail_window_s,
                "operational_critical_levels": list(th.operational_critical_levels),
            },
            "criticality_policy": {
                "escalate_app_from_topics": th.escalate_app_from_topics,
                "weights": dict(CRITICALITY_WEIGHT),
            },
        },
        "system_health": {
            "counts": sys_counts,
            "shares": sys_shares,
            "by_type": by_type,
            "by_criticality": by_criticality,
            "operational_critical": {
                "levels": list(th.operational_critical_levels),
                "counts": op_counts,
                "shares": op_shares,
                "by_type": op_by_type,
            },
            "resilience": {
                "time_integrated_availability": ti_avail,
                "end_to_end_delivery_attainment": e2e_att,
                "blast_radius_count": int(blast_radius),
                "failover_takeover_count": int(takeover_count),
                "peak_cpu_p95": peak_cpu,
            },
        },
        "product_in_use": {
            "beneficialness": {
                "score": ben_score,
                "critical_mission_continuity": crit_mission_cont,
                "criticality_weighted_health_index": crit_weighted_health,
                "critical_topic_timeliness_rate": crit_topic_timely,
            },
            "freedom_from_risk": {
                "score": risk_score,
                "zero_tolerance_critical_loss": bool(lost_op_crit > 0),
                "lost_critical_count": int(lost_strict_crit),
                "lost_operational_critical_count": int(lost_op_crit),
                "critical_safety_exposure_s": round(op_crit_exposure_s, 3),
                "critical_redundancy_depletion_rate": red_depl_rate,
                "n_minus_0_critical_ids": n_minus_0_ids,
            },
            "acceptability": {
                "score": acc_score,
                "critical_sla_compliance_rate": sla_compliance,
                "critical_sla_breach_count": len(sla_breaches),
                "dds_status_breaches": dds_breaches,
            },
            "domain_context": {
                "domain": domain,
                "omega": omega_dict,
                "derived": omega_derived,
            },
            "overall_qiu_health_index": overall_qiu,
        },
        "components": components,
    }


def _time_integrated_metrics(
    trace: TelemetryTrace,
    plan: Mapping[str, Any],
    crit_map: Mapping[str, Mapping[str, Any]],
    op_crit_levels: Set[str],
    *,
    warmup_s: float = 0.0,
    bin_s: float = 1.0,
) -> Tuple[Optional[float], float]:
    """Compute time-integrated topic availability and operational-critical exposure seconds."""
    span = trace.end_s - warmup_s
    topics = plan.get("topics", {})
    active_topics = [
        t for t, s in topics.items() if s.get("publishers") and float(s.get("rate_hz", 0.0)) > 0
    ]
    if span < bin_s or not active_topics:
        return None, 0.0

    n_bins = int(math.floor(span / bin_s))
    if n_bins <= 0:
        return None, 0.0

    seen_in_bin: List[Set[str]] = [set() for _ in range(n_bins)]
    for e in trace.events:
        if e["kind"] not in ("publish", "deliver") or not e["topic"]:
            continue
        if e["t"] < warmup_s or e["t"] >= warmup_s + n_bins * bin_s:
            continue
        idx = int((e["t"] - warmup_s) // bin_s)
        if 0 <= idx < n_bins:
            seen_in_bin[idx].add(e["topic"])

    op_crit_topics = [
        t for t in active_topics
        if crit_map.get(t, {}).get("level") in op_crit_levels
        and float(topics[t].get("rate_hz", 0.0)) * bin_s >= 1.0
    ]
    checkable = [
        t for t in active_topics if float(topics[t].get("rate_hz", 0.0)) * bin_s >= 1.0
    ]
    if not checkable:
        return 1.0, 0.0

    avail_sum = 0.0
    exposure_s = 0.0
    for b_topics in seen_in_bin:
        avail_sum += sum(1 for t in checkable if t in b_topics) / len(checkable)
        if op_crit_topics and any(t not in b_topics for t in op_crit_topics):
            exposure_s += bin_s

    return round(avail_sum / n_bins, 6), exposure_s


# ── Tier-1 vs Tier-2 comparison ──────────────────────────────────────────────

def compare_health_summaries(
    simulated: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare Tier-1 simulated and Tier-2 observed health summaries."""
    s_comps = simulated.get("components", {})
    o_comps = observed.get("components", {})
    common = sorted(set(s_comps) & set(o_comps))

    exact_matches = sum(1 for c in common if s_comps[c]["state"] == o_comps[c]["state"])
    overall_acc = round(exact_matches / len(common), 6) if common else None

    by_type_acc: Dict[str, Optional[float]] = {}
    for ctype in COMPONENT_TYPES:
        t_common = [c for c in common if s_comps[c]["type"] == ctype]
        if t_common:
            m = sum(1 for c in t_common if s_comps[c]["state"] == o_comps[c]["state"])
            by_type_acc[ctype] = round(m / len(t_common), 6)
        else:
            by_type_acc[ctype] = None

    op_levels = set(
        observed.get("system_health", {}).get("operational_critical", {}).get("levels", ("CRITICAL", "HIGH"))
    )
    sim_lost_op = {
        c for c in common
        if s_comps[c]["state"] == "lost" and s_comps[c]["criticality"]["level"] in op_levels
    }
    obs_lost_op = {
        c for c in common
        if o_comps[c]["state"] == "lost" and o_comps[c]["criticality"]["level"] in op_levels
    }
    union_op = sim_lost_op | obs_lost_op
    op_lost_jaccard = round(len(sim_lost_op & obs_lost_op) / len(union_op), 6) if union_op else 1.0

    sim_qiu = float(simulated.get("product_in_use", {}).get("overall_qiu_health_index", 0.0))
    obs_qiu = float(observed.get("product_in_use", {}).get("overall_qiu_health_index", 0.0))
    sim_cwh = float(
        simulated.get("product_in_use", {}).get("beneficialness", {}).get("criticality_weighted_health_index", 0.0)
    )
    obs_cwh = float(
        observed.get("product_in_use", {}).get("beneficialness", {}).get("criticality_weighted_health_index", 0.0)
    )

    mismatches = [
        {
            "id": c,
            "type": s_comps[c]["type"],
            "criticality": s_comps[c]["criticality"]["level"],
            "simulated_state": s_comps[c]["state"],
            "observed_state": o_comps[c]["state"],
        }
        for c in common
        if s_comps[c]["state"] != o_comps[c]["state"]
    ]

    return {
        "n_compared": len(common),
        "exact_state_agreement": overall_acc,
        "by_type_state_agreement": by_type_acc,
        "operational_critical_lost_jaccard": op_lost_jaccard,
        "qiu_health_index_abs_error": round(abs(sim_qiu - obs_qiu), 6),
        "criticality_weighted_health_abs_error": round(abs(sim_cwh - obs_cwh), 6),
        "mismatches": mismatches[:20],
    }


# ── Tier-2 Operational Criticality Extraction & Comparison ───────────────────

def observed_operational_criticality(
    trace: TelemetryTrace,
    plan: Mapping[str, Any],
    target: str,
    fault_time: float,
    *,
    warmup_s: float = 0.0,
    failed: Optional[Iterable[str]] = None,
    fault_duration_s: Optional[float] = None,
    health_summary: Optional[Mapping[str, Any]] = None,
    baseline_qiu: float = 1.0,
) -> Dict[str, Any]:
    """Compute Tier-2 observed operational criticality labels from a faulted trace.

    Mirrors Tier 1's `saag.simulation.runtime.labels.labels_from_run` directly
    from observed/simulated `TelemetryTrace` events, plus Product-in-Use
    `qiu_impact` and the target's static/derived criticality provenance.
    """
    from saag.validation.fidelity import observed_impact

    apps_spec = plan.get("apps", {})
    topics_spec = plan.get("topics", {})
    hosts_spec = plan.get("hosts", {})

    if failed is not None:
        gone = set(failed)
    elif target in hosts_spec:
        r = hosts_spec[target]
        gone = {target} | {
            a for a in r.get("apps", ()) if len(apps_spec.get(a, {}).get("hosts", ())) <= 1
        } | set(r.get("brokers", ()))
    else:
        gone = {target}

    subs_map: Dict[str, List[str]] = {t: [] for t in topics_spec}
    for a, spec in apps_spec.items():
        for t in spec.get("subscribes", ()):
            if t in subs_map:
                subs_map[t].append(a)

    pre_span = max(fault_time - warmup_s, 1e-9)
    post_span = max(trace.end_s - fault_time, 1e-9)

    pre_deliv: Dict[str, int] = defaultdict(int)
    post_deliv: Dict[str, int] = defaultdict(int)
    dm_pre = dm_post = deliv_pre_tot = deliv_post_tot = 0

    for e in trace.events:
        t = e["t"]
        if t < warmup_s or t > trace.end_s:
            continue
        kind = e["kind"]
        app = e["app"]
        if kind == "deliver":
            # Bucket on creation time when latency_ms is present, matching observed_impact
            ct = (t - float(e["latency_ms"]) / 1000.0) if e["latency_ms"] is not None else t
            if ct < warmup_s:
                continue
            if ct < fault_time:
                if app:
                    pre_deliv[app] += 1
                if app not in gone:
                    deliv_pre_tot += 1
            else:
                if app:
                    post_deliv[app] += 1
                if app not in gone:
                    deliv_post_tot += 1
        elif kind == "deadline_miss" and app not in gone:
            if t < fault_time:
                dm_pre += 1
            else:
                dm_post += 1

    # 1. i_delivery (volume-weighted delivery loss of surviving readers)
    i_delivery = observed_impact(
        trace, subs_map, target, fault_time, warmup_s=warmup_s, failed=gone,
    )

    # 2. i_delivery_consumer (consumer-averaged delivery loss, unweighted by Hz)
    consumer_losses: List[float] = []
    for a in sorted(apps_spec):
        if a in gone or not apps_spec[a].get("subscribes"):
            continue
        r_pre = pre_deliv[a] / pre_span
        if r_pre > 0:
            r_post = post_deliv[a] / post_span
            consumer_losses.append(1.0 - r_post / r_pre)
    i_delivery_consumer = (
        sum(consumer_losses) / len(consumer_losses) if consumer_losses else 0.0
    )

    # 3. i_flow (criticality-weighted end-to-end loss at sinks)
    crit_map = derive_operational_criticality(plan)
    sinks = sorted(
        a for a, s in apps_spec.items() if s.get("subscribes") and not s.get("publishes")
    )
    flow_num = flow_den = 0.0
    for a in sinks:
        w = float(crit_map.get(a, {}).get("weight", 1.0))
        if a in gone:
            flow_num += w * 1.0
            flow_den += w
        else:
            r_pre = pre_deliv[a] / pre_span
            if r_pre > 0:
                r_post = post_deliv[a] / post_span
                flow_num += w * (1.0 - r_post / r_pre)
                flow_den += w
    i_flow = (flow_num / flow_den) if flow_den > 0 else 0.0

    # 4. i_timeliness (DDS requested_deadline_missed rate delta, or engine miss-share delta)
    i_timeliness: Optional[float]
    if plan.get("qos_semantics") == "dds":
        surviving_readers = [
            f"reader|{t}|{a}"
            for a, s in apps_spec.items()
            if a not in gone
            for t in s.get("subscribes", ())
        ]
        if surviving_readers:
            pre_st = trace.status_counts(until=fault_time).get("requested_deadline_missed", {})
            end_cut = max(fault_time, trace.end_s - 1.5) if (trace.end_s - fault_time) > 2.5 else trace.end_s
            end_st = trace.status_counts(until=end_cut).get("requested_deadline_missed", {})
            eff_post_span = max(end_cut - fault_time, 1e-9)
            pre_m = sum(pre_st.get(k, 0) for k in surviving_readers)
            post_m = sum(max(0, end_st.get(k, 0) - pre_st.get(k, 0)) for k in surviving_readers)
            i_timeliness = (post_m / (len(surviving_readers) * eff_post_span)) - (
                pre_m / (len(surviving_readers) * pre_span)
            )
        else:
            i_timeliness = 0.0
    else:
        pre_tot = dm_pre + deliv_pre_tot
        post_tot = dm_post + deliv_post_tot
        if pre_tot > 0 and post_tot > 0:
            i_timeliness = (dm_post / post_tot) - (dm_pre / pre_tot)
        else:
            i_timeliness = None

    # 5. i_recovery (seconds after bounded fault end until delivery regains 95% of pre-fault rate)
    i_recovery: Optional[float] = None
    if fault_duration_s is not None and deliv_pre_tot > 0:
        fault_end = fault_time + float(fault_duration_s)
        if trace.end_s > fault_end:
            pre_rate_tot = deliv_pre_tot / pre_span
            bin_s = 1.0
            n_bins = int(math.floor((trace.end_s - fault_end) / bin_s))
            if n_bins > 0:
                bins = [0] * n_bins
                for e in trace.events:
                    if e["kind"] == "deliver" and e["app"] not in gone and e["t"] >= fault_end:
                        idx = int((e["t"] - fault_end) // bin_s)
                        if 0 <= idx < n_bins:
                            bins[idx] += 1
                for idx, cnt in enumerate(bins):
                    if (cnt / bin_s) >= 0.95 * pre_rate_tot:
                        i_recovery = round(idx * bin_s, 3)
                        break

    # 6. Product-in-Use & health-derived operational criticality
    hs = health_summary
    if hs is None:
        hs = summarize_system_health(trace, plan, fault_time=fault_time, warmup_s=warmup_s)
    piu = hs.get("product_in_use", {})
    fault_qiu = float(piu.get("overall_qiu_health_index", 1.0))
    cwh = float(piu.get("beneficialness", {}).get("criticality_weighted_health_index", 1.0))
    ffr = piu.get("freedom_from_risk", {})

    return {
        "target": target,
        "target_type": hs.get("components", {}).get(target, {}).get("type"),
        "static_criticality": crit_map.get(target),
        "i_flow": round(float(i_flow), 6),
        "i_delivery": round(float(i_delivery), 6),
        "i_delivery_consumer": round(float(i_delivery_consumer), 6),
        "i_timeliness": round(float(i_timeliness), 6) if i_timeliness is not None else None,
        "i_recovery": i_recovery,
        "qiu_impact": round(float(baseline_qiu - fault_qiu), 6),
        "criticality_weighted_health_loss": round(float(1.0 - cwh), 6),
        "lost_operational_critical_count": int(ffr.get("lost_operational_critical_count", 0)),
        "zero_tolerance_critical_loss": bool(ffr.get("zero_tolerance_critical_loss", False)),
    }


def compare_operational_criticality(
    tier1_by_target: Mapping[str, Mapping[str, Any]],
    tier2_by_target: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str] = ("i_flow", "i_delivery", "i_delivery_consumer", "i_timeliness"),
    k_frac: float = 0.2,
) -> Dict[str, Any]:
    """Compare Tier-1 vs Tier-2 operational criticality rankings across labels."""
    from saag.validation.fidelity import compare_fault_response

    def _extract(rec: Mapping[str, Any], lbl: str) -> Optional[float]:
        v = rec.get(lbl)
        if isinstance(v, Mapping):
            v = v.get("mean")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    out: Dict[str, Any] = {}
    for lbl in labels:
        t1: Dict[str, float] = {}
        t2: Dict[str, float] = {}
        for cid in sorted(set(tier1_by_target) & set(tier2_by_target)):
            v1 = _extract(tier1_by_target[cid], lbl)
            v2 = _extract(tier2_by_target[cid], lbl)
            if v1 is not None and v2 is not None:
                t1[cid] = v1
                t2[cid] = v2
        cmp = compare_fault_response(t1, t2, k_frac=k_frac)
        errs = [abs(t1[c] - t2[c]) for c in t1]
        out[lbl] = {
            "n_common": len(t1),
            "spearman": round(cmp["spearman"], 6) if cmp.get("spearman") is not None else None,
            "topk_jaccard": round(cmp["topk_jaccard"], 6) if cmp.get("topk_jaccard") is not None else None,
            "mae": round(sum(errs) / len(errs), 6) if errs else None,
            "max_abs_error": round(max(errs), 6) if errs else None,
        }
    return out


# ── Human-readable Tier-2 Formatters ─────────────────────────────────────────

def format_health_report(
    summary: Mapping[str, Any],
    *,
    only_unhealthy: bool = False,
    only_critical: bool = False,
) -> str:
    """Render a `summarize_system_health` dict as a structured plain-text table."""
    meta = summary.get("metadata", {})
    sh = summary.get("system_health", {})
    piu = summary.get("product_in_use", {})
    ffr = piu.get("freedom_from_risk", {})
    ben = piu.get("beneficialness", {})
    acc = piu.get("acceptability", {})
    dom = piu.get("domain_context", {})
    op_levels = set(sh.get("operational_critical", {}).get("levels", ("CRITICAL", "HIGH")))

    lines: List[str] = []
    lines.append(
        f"Tier-2 Operational Health Summary — {meta.get('scenario', 'unknown')} "
        f"(domain={meta.get('domain') or 'default'}, tail={meta.get('tail_window_s')}, "
        f"fault_t={meta.get('fault_time_s')})"
    )
    cnt = sh.get("counts", {})
    op_cnt = sh.get("operational_critical", {}).get("counts", {})
    lines.append(
        f"  Overall: total={cnt.get('total', 0)}  healthy={cnt.get('healthy', 0)}  "
        f"degraded={cnt.get('degraded', 0)}  lost={cnt.get('lost', 0)}  |  "
        f"Op-Critical ({'/'.join(sorted(op_levels))}): "
        f"H={op_cnt.get('healthy', 0)} D={op_cnt.get('degraded', 0)} L={op_cnt.get('lost', 0)}"
    )
    lines.append(
        f"  Product-in-Use (ISO 25019): QiU={piu.get('overall_qiu_health_index', 0.0):.3f}  "
        f"[Ben={ben.get('score', 0.0):.3f}, Risk={ffr.get('score', 0.0):.3f}, Acc={acc.get('score', 0.0):.3f}]  "
        f"zero_tol_crit_loss={ffr.get('zero_tolerance_critical_loss')}  "
        f"N-0_crit={ffr.get('n_minus_0_critical_ids', [])}"
    )

    lines.append("")
    lines.append(f"  {'Type':14s} {'Total':>5s} {'Healthy':>7s} {'Degraded':>8s} {'Lost':>5s} {'OpCrit Lost':>11s}")
    lines.append("  " + "-" * 56)
    for ctype in COMPONENT_TYPES:
        tc = sh.get("by_type", {}).get(ctype, {}).get("counts", {})
        oc = sh.get("operational_critical", {}).get("by_type", {}).get(ctype, {}).get("counts", {})
        lines.append(
            f"  {ctype:14s} {tc.get('total', 0):5d} {tc.get('healthy', 0):7d} "
            f"{tc.get('degraded', 0):8d} {tc.get('lost', 0):5d} {oc.get('lost', 0):11d}"
        )

    comps = summary.get("components", {})
    rows = []
    for cid, rec in sorted(comps.items(), key=lambda kv: (kv[1]["type"], kv[0])):
        if only_unhealthy and rec["state"] == "healthy":
            continue
        if only_critical and rec["criticality"]["level"] not in op_levels:
            continue
        rows.append(rec)

    if rows:
        lines.append("")
        lines.append(
            f"  {'ID':16s} {'Type':13s} {'Crit (Prov)':18s} {'State':9s} {'Score':>6s}  Reasons"
        )
        lines.append("  " + "-" * 78)
        for r in rows:
            c = r["criticality"]
            crit_str = f"{c['level']} ({c['provenance']})"
            reasons_str = ", ".join(r.get("reasons") or ("-",))
            lines.append(
                f"  {r['id']:16s} {r['type']:13s} {crit_str:18s} "
                f"{r['state']:9s} {r['health_score']:6.3f}  {reasons_str}"
            )
    else:
        lines.append("\n  (No components matched the selected filter.)")

    if dom.get("derived"):
        om = dom.get("omega", {})
        lines.append(
            f"\n  Domain omega ({dom.get('domain')}): "
            f"Ben={om.get('beneficialness')}, Risk={om.get('freedom_from_risk')}, Acc={om.get('acceptability')}"
        )
    return "\n".join(lines)


def format_operational_criticality_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    sort_by: str = "i_flow",
) -> str:
    """Render Tier-2 observed operational criticality rankings as a plain-text table."""
    if not rows:
        return "  (No Tier-2 operational criticality observations.)"

    def _sort_key(r: Mapping[str, Any]) -> Tuple[float, str]:
        v = r.get(sort_by)
        val = float(v) if isinstance(v, (int, float)) else -1e9
        return (-val, str(r.get("target", "")))

    ordered = sorted(rows, key=_sort_key)
    lines = [
        f"  {'Target':12s} {'Type':11s} {'Static':8s} {'i_flow':>7s} {'i_deliv':>7s} "
        f"{'i_cons':>7s} {'i_time':>7s} {'qiu_imp':>7s} {'OpCritL':>7s}",
        "  " + "-" * 81,
    ]
    for r in ordered:
        sc = (r.get("static_criticality") or {}).get("level") or "-"
        ttype = str(r.get("target_type") or "-")
        fmt = lambda k: f"{r[k]:7.3f}" if isinstance(r.get(k), (int, float)) else f"{'N/A':>7s}"  # noqa: E731
        lines.append(
            f"  {str(r.get('target', ''))[:12]:12s} {ttype[:11]:11s} {sc:8s} "
            f"{fmt('i_flow')} {fmt('i_delivery')} {fmt('i_delivery_consumer')} "
            f"{fmt('i_timeliness')} {fmt('qiu_impact')} {int(r.get('lost_operational_critical_count', 0)):7d}"
        )
    return "\n".join(lines)
