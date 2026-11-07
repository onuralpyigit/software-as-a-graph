#!/usr/bin/env python3
"""
Analysis for docs/research/fidelity/PREREGISTRATION.md §6, §7 and §10.

    python reproduce/fidelity_significance.py --delta 0.05

Reads results/fidelity_study/ and refuses to run until the sweep and the
simulator arms are both complete, which is what keeps §10's blinding: this is
the only place a simulator is compared with an observed run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scipy.stats import norm, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saag.validation.fidelity import compare_dds_status, summarize  # noqa: E402


def hodges_lehmann(d: Sequence[float]) -> float:
    return float(statistics.median((a + b) / 2 for a, b in combinations_with_replacement(d, 2)))


def hl_ci(d: Sequence[float], alpha: float = 0.05) -> List[float]:
    walsh = sorted((a + b) / 2 for a, b in combinations_with_replacement(d, 2))
    n = len(d)
    z = norm.ppf(1 - alpha / 2)
    k = int(n * (n + 1) / 4 - z * (n * (n + 1) * (2 * n + 1) / 24) ** 0.5)
    k = max(k, 0)
    return [float(walsh[k]), float(walsh[-k - 1])]


def rank_biserial(d: Sequence[float]) -> float:
    nz = [x for x in d if x != 0]
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = {i: r + 1 for r, i in enumerate(order)}
    total = sum(ranks.values()) or 1
    return (sum(r for i, r in ranks.items() if nz[i] > 0) - sum(r for i, r in ranks.items() if nz[i] < 0)) / total


def _p(d: Sequence[float], alternative: str) -> float:
    if all(x == 0 for x in d):
        return 1.0
    return float(wilcoxon(d, alternative=alternative, method="exact", zero_method="wilcox").pvalue)


def primary(d: Sequence[float], delta: float, alpha: float = 0.05) -> Dict[str, object]:
    """H1: superiority, TOST and the §6 decision table. Fewer than 6 scenarios: descriptive only."""
    d = list(d)
    if not d:
        return {"n": 0, "outcome": "Descriptive"}
    out: Dict[str, object] = {"n": len(d), "hl": hodges_lehmann(d), "hl_ci": hl_ci(d),
                              "rank_biserial": rank_biserial(d)}
    if len(d) < 6:
        out["outcome"] = "Descriptive"
        return out
    sup = _p(d, "greater")
    tost = max(_p([x + delta for x in d], "greater"), _p([x - delta for x in d], "less"))
    two = _p(d, "two-sided")
    out.update(p_superiority=sup, p_tost=tost, p_two_sided=two)
    if sup < alpha:
        out["outcome"] = "Better"
    elif two < alpha and out["hl_ci"][1] < 0:
        out["outcome"] = "Worse"
    elif tost < alpha:
        out["outcome"] = "Equivalent"
    else:
        out["outcome"] = "Inconclusive"
    return out


def holm(family: Dict[str, Sequence[float]], alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """§7 secondary family; two-sided. Below n = 8 it is descriptive."""
    res: Dict[str, Dict[str, object]] = {}
    for k, d in family.items():
        if not d:
            res[k] = {"n": 0, "hl": None, "hl_ci": None, "p": None, "significant": None}
            continue
        res[k] = {"n": len(d), "hl": hodges_lehmann(d), "hl_ci": hl_ci(d),
                  "p": _p(d, "two-sided") if len(d) >= 8 else None, "significant": None}
    tested = sorted((k for k in res if res[k]["p"] is not None), key=lambda k: res[k]["p"])
    still = True
    for i, k in enumerate(tested):
        still = still and res[k]["p"] < alpha / (len(tested) - i)
        res[k]["significant"] = still
    return res


def instrument_ok(null_median_rho: Optional[float]) -> bool:
    """§7: shuffled labels must sit near zero, or the results are withheld."""
    return null_median_rho is not None and abs(null_median_rho) <= 0.10


# ── From raw results to per-scenario contrasts ───────────────────────────────

MIN_PUBLISHERS = 10
MAX_TARGET_LOSS = 0.10
PERMUTATIONS = 1000
REPS = 3


def _rho(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    """Spearman on the common keys; None when undefined (§10: never coerced to 0)."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).correlation)


def _median_rel_error(sim: Dict[str, float], obs: Dict[str, float]) -> Optional[float]:
    errs = [abs(sim[k] - obs[k]) / abs(obs[k]) for k in set(sim) & set(obs) if obs[k]]
    return statistics.median(errs) if errs else None


def load_observed(out: Path, name: str, cond: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    rows = json.loads((out / "sweep" / f"iobs_{name}_{cond}.json").read_text()).values()
    table: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        if r.get("valid"):
            table.setdefault(r["target"], {})[r["rep"]] = r
    return table


def load_arms(out: Path) -> Dict[tuple, Dict[str, Any]]:
    """(scenario, cond, arm) -> {"impact": {target: seed mean}, "stalled": ..., "steady": [...]}."""
    raw: Dict[tuple, Dict[str, Any]] = {}
    for line in (out / "simulate" / "arms.jsonl").read_text().splitlines():
        r = json.loads(line)
        slot = raw.setdefault((r["scenario_name"], r["cond"], r["arm"]),
                              {"impact": {}, "stalled": {}, "steady": []})
        if r["target"] is None:
            slot["steady"].append(r["steady"])
        else:
            slot["impact"].setdefault(r["target"], []).append(r["impact"])
            slot["stalled"].setdefault(r["target"], []).append(set(r["stalled"]))
    for slot in raw.values():
        slot["impact"] = {t: statistics.mean(v) for t, v in slot["impact"].items()}
    return raw


def analyse_scenario(name: str, cond: str, kappa: float, targets: Dict[str, List[str]],
                     observed: Dict[str, Dict[int, Dict[str, Any]]], arms: Dict[tuple, Dict[str, Any]],
                     reference: Dict[str, float], baselines: List[Dict[str, Any]],
                     rng: random.Random, min_publishers: int = MIN_PUBLISHERS) -> Dict[str, Any]:
    """Everything §7 reports for one scenario under one condition."""
    every = [t for group in targets.values() for t in group]
    ok = {t for t in every if len(observed.get(t, {})) == REPS}
    lost = sorted(set(every) - ok)
    pubs = [t for t in targets["publishers"] if t in ok]
    res: Dict[str, Any] = {"kappa": kappa, "lost_targets": lost, "n_publishers": len(pubs),
                           "lost_scenario": len(lost) > MAX_TARGET_LOSS * len(every) or len(pubs) < min_publishers}
    i_obs = {t: statistics.mean(observed[t][r]["i_obs"] for r in observed[t]) for t in ok}
    loo = []
    for r in range(REPS):
        x = {t: observed[t][r]["i_obs"] for t in pubs}
        y = {t: statistics.mean(observed[t][o]["i_obs"] for o in range(REPS) if o != r) for t in pubs}
        v = _rho(x, y)
        if v is not None:
            loo.append(v)
    res["ceiling"] = statistics.mean(loo) if loo else None
    obs_pub = {t: i_obs[t] for t in pubs}
    res["rho"] = {arm: _rho({t: slot["impact"][t] for t in pubs if t in slot["impact"]}, obs_pub)
                  for (s, c, arm), slot in arms.items() if s == name and c == cond}
    res["rho"]["R"] = _rho({t: reference[t] for t in pubs if t in reference}, obs_pub)
    s2 = arms.get((name, cond, "S2"))
    if s2:
        keys = [t for t in pubs if t in s2["impact"]]
        null = []
        shuffled = [s2["impact"][t] for t in keys]
        for _ in range(PERMUTATIONS):
            rng.shuffle(shuffled)
            v = _rho(dict(zip(keys, shuffled)), obs_pub)
            if v is not None:
                null.append(v)
        res["null_median_rho"] = statistics.median(null) if null else None
    # Delivery and latency against the fault-free baselines.
    res["steady"] = {}
    obs_steady = [b["summary"] for b in baselines if b.get("summary")]
    if obs_steady:
        for (s, c, arm), slot in arms.items():
            if s != name or c != cond or not slot["steady"]:
                continue
            row: Dict[str, Optional[float]] = {}
            for metric, out_key in (("delivery_rate", "delivery_error"), ("latency_p50", "latency_p50_error"),
                                    ("latency_p95", "latency_p95_error")):
                errs = [_median_rel_error(sim[metric], o[metric])
                        for sim, o in zip(slot["steady"], obs_steady)]
                errs = [e for e in errs if e is not None]
                row[out_key] = statistics.median(errs) if errs else None
            if kappa < 1.0:         # §2: latency counts only at kappa = 1
                row["latency_p50_error"] = row["latency_p95_error"] = None
            res["steady"][arm] = row
    # Brokers and hosts: no comparator exists, because S0 cannot observe them.
    res["infra"] = {}
    infra = [t for t in targets["brokers"] + targets["hosts"] if t in ok]
    for arm in ("S2", "S3"):
        slot = arms.get((name, cond, arm))
        if not slot or not infra:
            continue
        jac = []
        for t in infra:
            if slot["stalled"].get(t):
                a = slot["stalled"][t][0]
                b = set(observed[t][0].get("stalled", []))
                jac.append(len(a & b) / len(a | b) if a | b else 1.0)
        res["infra"][arm] = {
            "rho": _rho({t: slot["impact"][t] for t in infra if t in slot["impact"]},
                        {t: i_obs[t] for t in infra}),
            "stall_jaccard": statistics.mean(jac) if jac else None}
    res["sinks"] = {t: i_obs[t] for t in targets["sinks"] if t in ok}
    return res


def dds_status_sample(out: Path, name: str, cond: str, src: Path, plan: Dict[str, Any],
                      observed: Dict[str, Dict[int, Dict[str, Any]]], targets: Dict[str, List[str]],
                      rng: random.Random, k: int = 5) -> List[Dict[str, Any]]:
    """§7: `compare_dds_status` for S2 on 5 seeded faults, against each fault's first repetition."""
    from saag.core.graph_io import load_graph
    from saag.testbed.study import SIM_SEEDS, load_archived, simulate_fault_run

    pool = sorted(t for group in targets.values() for t in group if 0 in observed.get(t, {}))
    g = load_graph(src)
    res = []
    for t in rng.sample(pool, min(k, len(pool))):
        r = observed[t][0]
        sim = simulate_fault_run(g, plan, "S2", t, r["duration_s"], r["fault_s"], SIM_SEEDS[0], dds_status=True)
        obs = load_archived(out / "sweep" / "traces" / name / r["trace"], r["start_unix"], r["duration_s"])
        cmp = compare_dds_status(sim, obs, split_at=r["fault_s"])
        res.append({"target": t, "statuses": {s: v.get("after_split") for s, v in cmp["statuses"].items()}})
    return res


def contrast(scenarios: Dict[str, Dict[str, Any]], a: str, b: str) -> List[float]:
    """rho(a) - rho(b) per surviving scenario; one with an undefined rho leaves this contrast."""
    return [r["rho"][a] - r["rho"][b] for r in scenarios.values()
            if not r["lost_scenario"] and r["rho"].get(a) is not None and r["rho"].get(b) is not None]


def report(scenarios: Dict[str, Dict[str, Any]], react: Dict[str, Dict[str, Any]], delta: float) -> Dict[str, Any]:
    live = {n: r for n, r in scenarios.items() if not r["lost_scenario"]}
    nulls = [r["null_median_rho"] for r in live.values() if r.get("null_median_rho") is not None]
    null = statistics.median(nulls) if nulls else None
    out: Dict[str, Any] = {"n_scenarios": len(live), "delta": delta, "null_median_rho": null,
                           "instrument_ok": instrument_ok(null), "scenarios": scenarios,
                           "o_react_descriptive_only": react}
    if not out["instrument_ok"]:
        out["withheld"] = "§7 instrument check failed; find and report the cause before any result."
        return out
    out["H1"] = primary(contrast(scenarios, "S2", "S0"), delta)
    h5 = [r["steady"]["S2"]["delivery_error"] - r["steady"]["S3"]["delivery_error"]
          for r in live.values()
          if r["steady"].get("S2", {}).get("delivery_error") is not None
          and r["steady"].get("S3", {}).get("delivery_error") is not None]
    out["secondary"] = holm({
        "H2": [r["rho"]["S0"] - 0.70 for r in live.values() if r["rho"].get("S0") is not None],
        "H3": contrast(scenarios, "S2", "R"),
        "H4": contrast(scenarios, "S0", "R"),
        "H5": h5,
    })
    out["attribution_S1_minus_S0"] = contrast(scenarios, "S1", "S0")
    return out


def load_baselines(out: Path, name: str, cond: str) -> List[Dict[str, Any]]:
    from saag.testbed.study import load_archived

    rows = json.loads((out / "sweep" / f"baseline_{name}_{cond}.json").read_text())
    res = []
    warm_of = lambda r: 5.0 * r["duration_s"] / 60.0  # noqa: E731
    for r in rows.values():
        if r.get("valid"):
            trace = load_archived(out / "sweep" / "traces" / name / r["trace"], r["start_unix"], r["duration_s"])
            res.append({**r, "summary": summarize(trace.window(warm_of(r), r["duration_s"]))})
    return res


def main(out: Optional[Path] = None, delta: Optional[float] = None,
         min_publishers: int = MIN_PUBLISHERS) -> Dict[str, Any]:
    if delta is None:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--delta", type=float, required=True, help="δ from §8 of the pre-registration")
        delta = p.parse_args().delta
    from saag.core.graph_io import load_graph
    from saag.testbed import study
    from saag.testbed.deployment import build_deployment

    out = out or ROOT / "results" / "fidelity_study"
    for part in ("sweep", "simulate"):
        if not (out / part / "COMPLETE").exists():
            raise SystemExit(f"{part} is not complete; no comparison may be computed yet (§10).")
    cap = json.loads((out / "capacity.json").read_text())
    arms = load_arms(out)
    sources = json.loads((out / "sources.json").read_text()) if (out / "sources.json").exists() else {}
    rng = random.Random(study.SELECTION_SEED)
    tables: Dict[str, Dict[str, Dict[str, Any]]] = {"O-open": {}, "O-react": {}}
    for name, row in cap.items():
        if not row["included"]:
            continue
        src = Path(sources.get(name) or ROOT / "data" / "scenarios" / f"{name}_system.json")
        plan = build_deployment(src, open_loop=True)
        apps = [n for n, d in load_graph(src).nodes(data=True) if d.get("type") == "Application"]
        targets = study.select_targets(plan, apps)
        reference = study.rate_expansion(plan)
        for cond in tables:
            tables[cond][name] = analyse_scenario(
                name, cond, row["kappa"], targets, load_observed(out, name, cond), arms, reference,
                load_baselines(out, name, cond), rng, min_publishers)
        tables["O-open"][name]["dds_status"] = dds_status_sample(
            out, name, "O-open", src, plan, load_observed(out, name, "O-open"), targets, rng)
        committed = arms.get((name, "committed", "S0"))
        s0 = arms.get((name, "O-open", "S0"))
        if committed and s0:
            tables["O-open"][name]["scaling_rho"] = _rho(s0["impact"], committed["impact"])
    res = report(tables["O-open"], tables["O-react"], delta)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in main().items() if k not in ("scenarios", "o_react_descriptive_only")},
                     indent=2, default=str))
