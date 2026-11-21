#!/usr/bin/env python3
"""Validity report for finished scenarios of the fidelity sweep (PREREGISTRATION.md §10).

§10 "Blinding": no simulator is compared with I_obs until every run has finished,
and progress monitoring reads run validity only. This script therefore drops
``i_obs`` and ``stalled`` from every record before reading it, and reports only:

- runs and valid runs per condition, and reruns with their reasons (§10 "Failed run");
- targets with fewer than 3 valid repetitions (§10 "Lost target");
- whether the scenario is lost: more than 10% of targets lost, or fewer than
  10 publishing applications left (§10 "Lost scenario");
- pre-fault rate error, the run-validity bound (median ≤ 10%);
- CPU p95 per container against the 80% inclusion bound (§2, descriptive here);
- wall-clock time per run (median gap between run starts).

    PYTHONPATH=. python reproduce/fidelity_sweep_progress.py [scenario ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reproduce.fidelity_study as fs  # noqa: E402

BLINDED = {"i_obs", "stalled"}


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [{k: v for k, v in r.items() if k not in BLINDED} for r in json.loads(path.read_text()).values()]


def peak_cpu(r: dict) -> float | None:
    vals = [v for v in (r.get("cpu_p95") or {}).values() if v is not None]
    return max(vals) if vals else None


def report(name: str, sweep: Path) -> str:
    t = fs.targets_for(name)
    targets = t["publishers"] + t["sinks"] + t["brokers"] + t["hosts"]
    lines = [f"## {name}", "",
             f"Targets: {len(targets)} ({len(t['publishers'])} publishers, {len(t['sinks'])} sinks, "
             f"{len(t['brokers'])} brokers, {len(t['hosts'])} hosts); "
             f"expected fault runs per condition: {len(targets) * fs.REPS}.", "",
             "| Condition | Baselines valid | Fault runs | Valid | Reruns | Lost targets | "
             "Pre-fault rate error, median / max | Peak CPU p95 | s per run |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    lost_any, reasons = set(), []
    for cond in fs.CONDITIONS:
        base = records(sweep / f"baseline_{name}_{cond}.json")
        runs = records(sweep / f"iobs_{name}_{cond}.json")
        valid = [r for r in runs if r["valid"]]
        reasons += [f["reason"] for r in runs + base for f in r["failures"]]
        reps = dict.fromkeys(targets, 0)
        for r in valid:
            reps[r["target"]] += 1
        lost = {x for x, n in reps.items() if n < fs.REPS}
        lost_any |= lost
        errs = [e for r in valid if (e := (r.get("pre_fault") or {}).get("median_rate_error")) is not None]
        cpus = [c for c in map(peak_cpu, valid + base) if c is not None]
        starts = sorted(r["start_unix"] for r in valid)
        # Median gap between consecutive starts, so pauses between sessions do not count.
        pace = statistics.median(b - a for a, b in zip(starts, starts[1:])) if len(starts) > 1 else None
        err_s = f"{statistics.median(errs):.3f} / {max(errs):.3f}" if errs else "n/a"
        cpu_s = f"{max(cpus):.1f}%" if cpus else "n/a"
        pace_s = f"{pace:.0f}" if pace else "n/a"
        lines.append(f"| {cond} | {sum(r['valid'] for r in base)}/{len(base)} | {len(runs)} | {len(valid)} | "
                     f"{sum(len(r['failures']) for r in runs + base)} | {len(lost)} | {err_s} | {cpu_s} | {pace_s} |")
    pubs_left = len(set(t["publishers"]) - lost_any)
    lost_share = len(lost_any) / len(targets)
    verdict = "LOST: exclude and report" if lost_share > 0.10 or pubs_left < 10 else "retained"
    lines += ["",
              f"Lost targets (either condition; dropped from all arms): {len(lost_any)} of {len(targets)} "
              f"({lost_share:.0%}). Publishers left: {pubs_left}. **§10 verdict: {verdict}.**"]
    if reasons:
        lines += ["", "Rerun reasons: " + "; ".join(sorted(set(reasons)))]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenarios", nargs="*", help="default: every scenario marked done")
    a = p.parse_args()
    sweep = fs.OUT / "sweep"
    names = a.scenarios or sorted(x.name for x in (sweep / "done").glob("*"))
    print("\n".join(report(n, sweep) for n in names))


if __name__ == "__main__":
    main()
