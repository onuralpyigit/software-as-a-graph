#!/usr/bin/env python3
"""
reproduce/cut_results_bundle.py — assemble the artifact bundle that ships with the paper
========================================================================================

The bundle is the copy of ``results/`` a reviewer or a reader actually receives.
``results/`` itself is gitignored, so the bundle is the only place the numbers in
the manuscript exist outside the author's disk.

The previous bundle was assembled by hand and shipped three defects that a copy
loop cannot see:

  * ``detection_validation.json`` was a **crashed run** — every scenario had
    errored with a ``TypeError``, ``n_scenarios_measured`` was 0 and every
    summary metric was ``null``. It looked exactly like a normal artifact.
  * The artifact the manuscript's §7.3 actually cites
    (``detection_validation_v3.json``) was **not in the bundle at all**, along
    with nine others — including both timing artifacts behind the abstract's
    headline latency figures.
  * ``convergent_validity.json`` was a superseded four-scenario run, so the
    bundle reported the higher, stale agreement figure.

Each of those is mechanically detectable, which is what this script does. It
takes the artifact list from ``reconcile_manuscript``'s own registries rather
than a hand-written mirror: a mirror is how the reconciler once reported a clean
run over rows it never checked, and a bundle listing its own contents would fail
the same way. Anything the manuscript consumes is shipped; nothing else is.

Usage:
    PYTHONPATH=. python reproduce/cut_results_bundle.py
    PYTHONPATH=. python reproduce/cut_results_bundle.py --out results/SaG_JSS_Results_20260920
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import corpus_digest
from reproduce.reconcile_manuscript import (
    CORPUS_INDEPENDENT_ARTIFACTS,
    FRESHNESS_TARGETS,
    PAIRED_TIMING_ARTIFACTS,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

#: Table 3/5 are checked against ``main_table.json`` by ``check_table5_indist``,
#: which names it as a default argument rather than in a registry, so it is the
#: one artifact this list has to add by hand.
EXTRA_ARTIFACTS = {"main_table.json": "Tables 3/5 in-distribution"}

#: Rendered outputs. These are derived from the artifacts above and are shipped
#: because the manuscript's tables are typeset from them; re-render before
#: cutting (``make -f reproduce/Makefile table3 table4``) or they carry retired
#: scenario labels, which is what happened to "Hub-and-Spoke" / "Enterprise
#: Integration (ESB)" in the last bundle.
RENDERED = [
    "table3_main_results.md",
    "table3_main_results.tex",
    "table3_main_results.csv",
    "table3_id_metrics.md",
    # table3_id_metrics.md's F1 column is top-K overlap (both sets have K
    # members, so P == R == F1 identically); its calibration note points here
    # for F1 proper, so the two ship together or the note dangles.
    "table3_identification_metrics.md",
    "table4_loso_results.md",
    "table4_loso_results.tex",
    "table5_per_type_metrics.md",
    "table_rq2_controls.md",
    "figure4_stratified_rho.md",
    "figure4_stratified_rho.png",
    "figure4_stratified_rho.pdf",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _empty_reason(payload: Any) -> Optional[str]:
    """Why this artifact measured nothing, or None if it measured something.

    A crashed run is not an absent file: it is a well-formed JSON document whose
    every field is ``null``. ``detection_validation.json`` shipped in exactly
    that state. The two shapes below are how the reproduce scripts record a run
    that produced no measurement.
    """
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("n_scenarios_measured") == 0:
        return "summary.n_scenarios_measured == 0"
    rows = payload.get("per_scenario")
    if isinstance(rows, list) and rows and all(
        isinstance(r, dict) and "error" in r for r in rows
    ):
        return f"every per_scenario entry errored ({rows[0]['error']!r})"
    return None


def _inspect(name: str, backs: str, corpus: Optional[str]) -> Tuple[Dict[str, Any], List[str]]:
    """One artifact's manifest entry, plus the reasons it must not be shipped."""
    path = RESULTS / name
    entry: Dict[str, Any] = {"backs": backs}
    errors: List[str] = []

    if not path.exists():
        return {**entry, "status": "missing"}, [f"{name} ({backs}) is not in results/"]

    payload: Any = None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        errors.append(f"{name} is not readable JSON")

    reason = _empty_reason(payload)
    if reason:
        errors.append(f"{name} measured nothing: {reason}")

    prov = (payload.get("provenance") or {}) if isinstance(payload, dict) else {}
    entry["commit"] = prov.get("commit")
    entry["dirty"] = prov.get("dirty")
    entry["corpus_digest"] = prov.get("corpus_digest")

    if prov.get("dirty"):
        errors.append(f"{name} was produced from a modified working tree "
                      f"(commit {str(prov.get('commit'))[:8]}) and cannot be regenerated")

    # The corpus-freshness rule cannot apply to artifacts that never read the
    # corpus; the reconciler exempts the same two sets and for the same reason.
    timing = name in PAIRED_TIMING_ARTIFACTS or name in set(PAIRED_TIMING_ARTIFACTS.values())
    exempt = name in CORPUS_INDEPENDENT_ARTIFACTS or timing
    if entry["corpus_digest"] is None:
        entry["status"] = ("exempt-timing" if timing else "exempt-self-generated") \
            if exempt else "unstamped"
    elif corpus is not None and entry["corpus_digest"] != corpus:
        entry["status"] = "stale"
        errors.append(f"{name} describes corpus {entry['corpus_digest'][:8]}, "
                      f"not the one on disk ({corpus[:8]})")
    else:
        entry["status"] = "stamped"

    entry["bytes"] = path.stat().st_size
    entry["sha256"] = _sha256(path)
    return entry, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="Bundle directory (default: results/SaG_JSS_Results_<YYYYmmdd_HHMM>).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing bundle directory.")
    args = ap.parse_args()

    out = args.out or RESULTS / f"SaG_JSS_Results_{datetime.now():%Y%m%d_%H%M}"
    if out.exists() and not args.force:
        print(f"Error: {out} already exists (use --force to overwrite).", file=sys.stderr)
        return 1

    corpus = corpus_digest()
    artifacts = {**FRESHNESS_TARGETS, **CORPUS_INDEPENDENT_ARTIFACTS, **EXTRA_ARTIFACTS}
    for timed, paired in PAIRED_TIMING_ARTIFACTS.items():
        artifacts.setdefault(timed, "wall-clock timing")
        artifacts.setdefault(paired, "wall-clock timing (paired)")

    print(f"Cutting bundle → {out}")
    print(f"  corpus digest : {corpus}")
    print(f"  artifacts     : {len(artifacts)} consumed by the manuscript\n")

    manifest_artifacts: Dict[str, Any] = {}
    errors: List[str] = []
    for name, backs in sorted(artifacts.items()):
        entry, errs = _inspect(name, backs, corpus)
        manifest_artifacts[name] = entry
        errors.extend(errs)
        mark = {"stamped": "ok",
                "exempt-timing": "ok (wall-clock, machine-bound)",
                "exempt-self-generated": "ok (corpus-independent)",
                "unstamped": "UNSTAMPED", "stale": "STALE", "missing": "MISSING"}[entry["status"]]
        print(f"  {name:38s} {mark:26s} {backs}")

    if errors:
        print("\nRefusing to cut the bundle:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1

    missing_rendered = [n for n in RENDERED if not (RESULTS / n).exists()]
    if missing_rendered:
        print("\nRefusing to cut the bundle — rendered tables absent "
              "(run `make -f reproduce/Makefile table3 table4 figure4`):", file=sys.stderr)
        for n in missing_rendered:
            print(f"  ✗ {n}", file=sys.stderr)
        return 1

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest_rendered: Dict[str, Any] = {}
    for name in list(artifacts) + RENDERED:
        shutil.copy2(RESULTS / name, out / name)
    for name in RENDERED:
        manifest_rendered[name] = {"bytes": (out / name).stat().st_size,
                                   "sha256": _sha256(out / name)}

    unstamped = sorted(n for n, e in manifest_artifacts.items() if e["status"] == "unstamped")
    manifest = {
        "bundle": out.name,
        "cut_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_digest": corpus,
        "artifacts": manifest_artifacts,
        "rendered": manifest_rendered,
        "unstamped_artifacts": unstamped,
        "note": ("Artifacts are the manuscript's backing data, taken from "
                 "reconcile_manuscript's registries. 'unstamped' artifacts carry no "
                 "provenance block, so their correspondence to this corpus is asserted "
                 "by the cut, not proved by the artifact — re-run them through "
                 "reproduce/_provenance.stamp to close that gap."),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n✓ {len(artifacts)} artifacts + {len(RENDERED)} rendered files → {out}")
    if unstamped:
        print(f"  {len(unstamped)} artifact(s) carry no provenance stamp: {', '.join(unstamped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
