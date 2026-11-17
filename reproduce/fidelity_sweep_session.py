#!/usr/bin/env python3
"""One budgeted session of the fidelity sweep (PREREGISTRATION.md, frozen 2026-11-16, 9f84aad9).

``fidelity_study.py sweep`` runs the whole study (about 62 machine-hours) in one
go. This script runs the same per-scenario steps (``baselines`` and
``sweep_one``, in both conditions, in the runner's seeded target order) for the
next unfinished scenarios of the frozen population, and stops at the end of
``--scenarios`` scenarios or when ``--hours`` is spent, whichever comes first.

- The budget is checked before each run starts, so no run is cut short.
- Completed runs are skipped on resume (same keys as the runner), so sessions
  and ``fidelity_study.py sweep`` can be mixed.
- A finished scenario is marked in ``sweep/done/<name>``; every session appends
  its outcome to ``sweep/sessions.jsonl``.
- Only the nine scenarios of §8 "Inclusion" are run, whatever ``capacity.json``
  says, and each must still be included there at κ = 1.

    PYTHONPATH=. python reproduce/fidelity_sweep_session.py --hours 6 --scenarios 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reproduce.fidelity_study as fs  # noqa: E402

# PREREGISTRATION.md §8 "Inclusion", in capacity.json order.
FROZEN = ["iot_smart_city", "healthcare", "hub_and_spoke", "microservices", "telecom_ran",
          "industrial_scada", "realtime_gaming", "logistics_fleet", "atm"]


class BudgetSpent(Exception):
    pass


def done_marker(name: str) -> Path:
    return fs.OUT / "sweep" / "done" / name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hours", type=float, default=6.0, help="time budget for this session")
    p.add_argument("--scenarios", type=int, default=1, help="scenarios to finish in this session")
    a = p.parse_args()

    deadline = time.time() + a.hours * 3600
    attempt = fs.attempt_run

    def budgeted_attempt(*args, **kw):
        if time.time() >= deadline:
            raise BudgetSpent()
        return attempt(*args, **kw)

    fs.attempt_run = budgeted_attempt
    fs.require_frozen()
    inc = fs.included()
    for n in FROZEN:
        if not inc.get(n, {}).get("included") or inc[n]["kappa"] != 1.0:
            raise SystemExit(f"{n}: capacity.json no longer matches the frozen §8 table")

    log = fs.OUT / "sweep" / "sessions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    start, finished, outcome = time.time(), [], "scenario limit reached"
    try:
        for name in [n for n in FROZEN if not done_marker(n).exists()][: a.scenarios]:
            t = fs.targets_for(name)
            order = t["publishers"] + t["sinks"] + t["brokers"] + t["hosts"]
            random.Random(fs.study.run_seed(fs.STUDY_SEED, name, "order")).shuffle(order)
            for cond in fs.CONDITIONS:
                root = fs.OUT / "sweep" / "traces" / name
                fs.baselines(name, cond, 1.0, "sweep", root)
                fs.sweep_one(name, cond, 1.0, "sweep", order, fs.REPS, root)
            done_marker(name).parent.mkdir(parents=True, exist_ok=True)
            done_marker(name).write_text(time.strftime("%F %T\n"))
            finished.append(name)
        if all(done_marker(n).exists() for n in FROZEN):
            outcome = "all scenarios finished"
    except BudgetSpent:
        outcome = "budget reached"
    except BaseException as exc:
        outcome = f"stopped: {exc!r}"
        raise
    finally:
        with log.open("a") as f:
            f.write(json.dumps({"start_unix": start, "end_unix": time.time(), "hours": a.hours,
                                "scenarios": a.scenarios, "finished": finished,
                                "outcome": outcome}) + "\n")
        print("SESSION", outcome, finished, flush=True)


if __name__ == "__main__":
    main()
