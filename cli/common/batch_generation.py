#!/usr/bin/env python3
"""
bin/generate_datasets.py — Batch dataset generator for all scenario configs.

Generates one JSON dataset per scenario YAML found in data/, optionally
generates multi-seed variants for validation stability, and refreshes the
legacy data/system.json and data/dataset.json with current-format fields
(code_metrics, system_hierarchy).

Usage:
    python bin/generate_datasets.py [OPTIONS]

Options:
    --input-dir PATH      Directory containing scenario_*.yaml   [default: data/]
    --output-dir PATH     Output directory for generated JSON     [default: output/]
    --refresh-legacy      Regenerate data/system.json and data/dataset.json
    --multi-seed          Also generate per-seed variants for scenarios 01-06
    --seeds LIST          Comma-separated seeds for multi-seed   [default: 42,123,456,789,2024]
    --scenario PATTERN    Only generate matching scenario names (substring match)
    --force               Overwrite existing output files
    --manifest PATH       Write manifest JSON                    [default: output/dataset_manifest.json]
    --dry-run             Print plan without writing any files
    -v, --verbose         Verbose output per scenario
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
CYAN  = "\033[96m"; GREEN = "\033[92m"; RED   = "\033[91m"
AMBER = "\033[93m"; BOLD  = "\033[1m";  RESET = "\033[0m"
DIM   = "\033[2m"

def _hdr(msg: str) -> None:
    print(f"\n{CYAN}{BOLD}{'='*66}\n  {msg}\n{'='*66}{RESET}")

def _ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def _warn(msg: str) -> None: print(f"  {AMBER}⚠ {RESET}  {msg}")
def _err(msg: str)  -> None: print(f"  {RED}✗{RESET}  {msg}")
def _info(msg: str) -> None: print(f"  {DIM}·{RESET}  {msg}")
def _step(msg: str) -> None: print(f"     {msg}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DatasetRecord:
    scenario_name: str
    source_config: str            # relative path of YAML or "scale:<name>"
    output_path: str
    seed: int
    generation_mode: str          # "statistical" | "scale"
    counts: Dict[str, int] = field(default_factory=dict)
    edge_counts: Dict[str, int] = field(default_factory=dict)
    qos_distribution: Dict[str, Any] = field(default_factory=dict)
    criticality_distribution: Dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    elapsed_s: float = 0.0
    status: str = "ok"            # "ok" | "skipped" | "error"
    error: str = ""


@dataclass
class Manifest:
    generated_at: str
    project_root: str
    generator_version: str = "current"
    datasets: List[DatasetRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Topology summary helpers
# ---------------------------------------------------------------------------
def _count_nodes(data: Dict) -> Dict[str, int]:
    return {
        "applications": len(data.get("applications", [])),
        "brokers":      len(data.get("brokers",      [])),
        "topics":       len(data.get("topics",        [])),
        "nodes":        len(data.get("nodes",         [])),
        "libraries":    len(data.get("libraries",     [])),
    }


def _count_edges(data: Dict) -> Dict[str, int]:
    rels       = data.get("relationships", {})
    pub_edges  = rels.get("publishes_to", data.get("publish_edges", []))
    sub_edges  = rels.get("subscribes_to", data.get("subscribe_edges", []))
    runs_on    = rels.get("runs_on", data.get("runs_on", []))
    routes     = rels.get("routes", data.get("routes", []))
    lib_edges  = rels.get("uses", data.get("library_edges", []))
    return {
        "publish":   len(pub_edges),
        "subscribe": len(sub_edges),
        "runs_on":   len(runs_on),
        "routes":    len(routes),
        "library":   len(lib_edges),
        "total":     len(pub_edges) + len(sub_edges) + len(runs_on) + len(routes) + len(lib_edges),
    }


def _qos_dist(data: Dict) -> Dict[str, Any]:
    topics = data.get("topics", [])
    if not topics:
        return {}
    reliability: Dict[str, int] = {}
    durability:  Dict[str, int] = {}
    priority:    Dict[str, int] = {}
    for t in topics:
        qos = t.get("qos", {})
        r = qos.get("reliability", "UNKNOWN")
        d = qos.get("durability",  "UNKNOWN")
        p = qos.get("transport_priority", "UNKNOWN")
        reliability[r] = reliability.get(r, 0) + 1
        durability[d]  = durability.get(d, 0) + 1
        priority[p]    = priority.get(p, 0) + 1
    return {
        "reliability": dict(sorted(reliability.items())),
        "durability":  dict(sorted(durability.items())),
        "priority":    dict(sorted(priority.items())),
    }


def _criticality_dist(data: Dict) -> Dict[str, int]:
    apps = data.get("applications", [])
    dist: Dict[str, int] = {}
    for a in apps:
        c = str(a.get("criticality", "unknown"))
        dist[c] = dist.get(c, 0) + 1
    return dict(sorted(dist.items()))


def _print_topology_summary(record: DatasetRecord) -> None:
    c = record.counts
    e = record.edge_counts
    q = record.qos_distribution
    print(f"     {'nodes':<12} apps={c.get('applications',0):>4}  "
          f"brokers={c.get('brokers',0):>2}  "
          f"topics={c.get('topics',0):>3}  "
          f"infra={c.get('nodes',0):>3}  "
          f"libs={c.get('libraries',0):>3}")
    print(f"     {'edges':<12} pub={e.get('publish',0):>4}  "
          f"sub={e.get('subscribe',0):>4}  "
          f"routes={e.get('routes',0):>3}  "
          f"total={e.get('total',0):>4}")
    if q.get("reliability"):
        rel = "  ".join(f"{k}={v}" for k, v in q["reliability"].items())
        print(f"     {'QoS rely':<12} {rel}")
    if q.get("durability"):
        dur = "  ".join(f"{k}={v}" for k, v in q["durability"].items())
        print(f"     {'QoS dur':<12} {dur}")


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------
def _load_generation_tools():
    """Lazy import so startup is fast for --help / --dry-run."""
    from tools.generation import GenerationService, load_config, generate_graph
    from tools.generation.models import SCALE_PRESETS
    return GenerationService, load_config, generate_graph, SCALE_PRESETS


def generate_from_yaml(
    yaml_path: Path,
    output_path: Path,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> DatasetRecord:
    scenario_name = yaml_path.stem  # e.g. scenario_01_autonomous_vehicle
    record = DatasetRecord(
        scenario_name=scenario_name,
        source_config=str(yaml_path.relative_to(_PROJECT_ROOT)),
        output_path=str(output_path.relative_to(_PROJECT_ROOT)),
        seed=0,
        generation_mode="statistical",
    )

    if output_path.exists() and not force:
        _warn(f"{scenario_name:<40s}  already exists, skipping  (--force to overwrite)")
        record.status = "skipped"
        return record

    if dry_run:
        _info(f"{scenario_name:<40s}  [dry-run] would write → {output_path}")
        record.status = "skipped"
        return record

    try:
        GenerationService, load_config, _, _ = _load_generation_tools()
        t0 = time.perf_counter()

        config  = load_config(yaml_path)
        record.seed = config.seed
        service = GenerationService(config=config)
        data    = service.generate()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        elapsed = time.perf_counter() - t0
        record.elapsed_s               = round(elapsed, 3)
        record.counts                  = _count_nodes(data)
        record.edge_counts             = _count_edges(data)
        record.qos_distribution        = _qos_dist(data)
        record.criticality_distribution = _criticality_dist(data)
        record.generated_at            = datetime.now(timezone.utc).isoformat()

        _ok(f"{scenario_name:<40s}  seed={config.seed}  "
            f"apps={record.counts.get('applications',0):>4}  "
            f"edges={record.edge_counts.get('total',0):>5}  "
            f"({elapsed:.2f}s)")
        if verbose:
            _print_topology_summary(record)

    except Exception as exc:
        _err(f"{scenario_name:<40s}  FAILED: {exc}")
        record.status = "error"
        record.error  = str(exc)
        if verbose:
            import traceback; traceback.print_exc()

    return record


def generate_from_scale(
    scale: str,
    seed: int,
    output_path: Path,
    label: str,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> DatasetRecord:
    record = DatasetRecord(
        scenario_name=label,
        source_config=f"scale:{scale}",
        output_path=str(output_path.relative_to(_PROJECT_ROOT)),
        seed=seed,
        generation_mode="scale",
    )

    if output_path.exists() and not force:
        _warn(f"{label:<40s}  already exists, skipping  (--force to overwrite)")
        record.status = "skipped"
        return record

    if dry_run:
        _info(f"{label:<40s}  [dry-run] would write → {output_path}")
        record.status = "skipped"
        return record

    try:
        GenerationService, _, _, _ = _load_generation_tools()
        t0 = time.perf_counter()

        service = GenerationService(scale=scale, seed=seed)
        data    = service.generate()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        elapsed = time.perf_counter() - t0
        record.elapsed_s               = round(elapsed, 3)
        record.counts                  = _count_nodes(data)
        record.edge_counts             = _count_edges(data)
        record.qos_distribution        = _qos_dist(data)
        record.criticality_distribution = _criticality_dist(data)
        record.generated_at            = datetime.now(timezone.utc).isoformat()

        _ok(f"{label:<40s}  seed={seed}  "
            f"apps={record.counts.get('applications',0):>4}  "
            f"edges={record.edge_counts.get('total',0):>5}  "
            f"({elapsed:.2f}s)")
        if verbose:
            _print_topology_summary(record)

    except Exception as exc:
        _err(f"{label:<40s}  FAILED: {exc}")
        record.status = "error"
        record.error  = str(exc)
        if verbose:
            import traceback; traceback.print_exc()

    return record


# ---------------------------------------------------------------------------
# Multi-seed variant generation
# ---------------------------------------------------------------------------
_MULTI_SEED_SCENARIOS = [
    "scenario_01", "scenario_02", "scenario_03",
    "scenario_04", "scenario_05", "scenario_06",
]

def _should_multi_seed(scenario_name: str) -> bool:
    return any(scenario_name.startswith(p) for p in _MULTI_SEED_SCENARIOS)


def generate_multi_seed_variants(
    yaml_path: Path,
    output_dir: Path,
    seeds: List[int],
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> List[DatasetRecord]:
    records = []
    stem = yaml_path.stem  # e.g. scenario_01_autonomous_vehicle
    try:
        GenerationService, load_config, _, _ = _load_generation_tools()
        base_config = load_config(yaml_path)
    except Exception as exc:
        _err(f"Cannot load {yaml_path.name}: {exc}")
        return records

    seeds_dir = output_dir / "multi_seed" / stem
    for seed in seeds:
        out_path = seeds_dir / f"{stem}_seed{seed}.json"
        label    = f"{stem}_seed{seed}"

        if out_path.exists() and not force:
            _warn(f"  {label:<50s}  exists, skipping")
            r = DatasetRecord(
                scenario_name=label, source_config=str(yaml_path.relative_to(_PROJECT_ROOT)),
                output_path=str(out_path.relative_to(_PROJECT_ROOT)), seed=seed,
                generation_mode="statistical", status="skipped",
            )
            records.append(r)
            continue

        if dry_run:
            _info(f"  {label:<50s}  [dry-run]")
            r = DatasetRecord(
                scenario_name=label, source_config=str(yaml_path.relative_to(_PROJECT_ROOT)),
                output_path=str(out_path.relative_to(_PROJECT_ROOT)), seed=seed,
                generation_mode="statistical", status="skipped",
            )
            records.append(r)
            continue

        try:
            import copy
            patched = copy.deepcopy(base_config)
            patched.seed = seed
            t0 = time.perf_counter()
            service = GenerationService(config=patched)
            data    = service.generate()
            elapsed = time.perf_counter() - t0

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)

            r = DatasetRecord(
                scenario_name=label,
                source_config=str(yaml_path.relative_to(_PROJECT_ROOT)),
                output_path=str(out_path.relative_to(_PROJECT_ROOT)),
                seed=seed,
                generation_mode="statistical",
                counts=_count_nodes(data),
                edge_counts=_count_edges(data),
                qos_distribution=_qos_dist(data),
                criticality_distribution=_criticality_dist(data),
                generated_at=datetime.now(timezone.utc).isoformat(),
                elapsed_s=round(elapsed, 3),
            )
            records.append(r)
            _ok(f"  {label:<50s}  apps={r.counts.get('applications',0):>4}  "
                f"({elapsed:.2f}s)")
            if verbose:
                _print_topology_summary(r)

        except Exception as exc:
            _err(f"  {label:<50s}  FAILED: {exc}")
            r = DatasetRecord(
                scenario_name=label, source_config=str(yaml_path.relative_to(_PROJECT_ROOT)),
                output_path=str(out_path.relative_to(_PROJECT_ROOT)), seed=seed,
                generation_mode="statistical", status="error", error=str(exc),
            )
            records.append(r)

    return records


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def _print_summary(records: List[DatasetRecord]) -> None:
    ok_recs   = [r for r in records if r.status == "ok"]
    skip_recs = [r for r in records if r.status == "skipped"]
    err_recs  = [r for r in records if r.status == "error"]

    _hdr("Dataset Generation Summary")
    col_w = 44

    print(f"  {'SCENARIO':<{col_w}}  {'SEED':>6}  {'APPS':>5}  {'TOPICS':>6}  "
          f"{'EDGES':>6}  {'STATUS':>8}  {'TIME':>6}")
    print(f"  {'-'*col_w}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}")

    for r in records:
        status_str = (
            f"{GREEN}ok{RESET}"      if r.status == "ok" else
            f"{AMBER}skipped{RESET}" if r.status == "skipped" else
            f"{RED}error{RESET}"
        )
        apps   = r.counts.get("applications", "-")
        topics = r.counts.get("topics", "-")
        edges  = r.edge_counts.get("total", "-")
        t_str  = f"{r.elapsed_s:.2f}s" if r.elapsed_s else "-"

        print(f"  {r.scenario_name:<{col_w}}  {r.seed:>6}  {apps!s:>5}  "
              f"{topics!s:>6}  {edges!s:>6}  {status_str:>8}  {t_str:>6}")

    print()
    total_apps  = sum(r.counts.get("applications", 0) for r in ok_recs)
    total_edges = sum(r.edge_counts.get("total", 0) for r in ok_recs)
    total_time  = sum(r.elapsed_s for r in ok_recs)
    print(f"  {GREEN}Generated : {len(ok_recs):>3}{RESET}  "
          f"{AMBER}Skipped : {len(skip_recs):>3}{RESET}  "
          f"{RED}Errors : {len(err_recs):>3}{RESET}")
    print(f"  Total nodes generated : {total_apps:,} applications  "
          f"{total_edges:,} edges  in {total_time:.1f}s")


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------
def _write_manifest(records: List[DatasetRecord], manifest_path: Path) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(_PROJECT_ROOT),
        "generator_version": "current",
        "total_datasets": len(records),
        "by_status": {
            "ok":      sum(1 for r in records if r.status == "ok"),
            "skipped": sum(1 for r in records if r.status == "skipped"),
            "error":   sum(1 for r in records if r.status == "error"),
        },
        "datasets": [asdict(r) for r in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _ok(f"Manifest written → {manifest_path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_batch_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-dir",      type=Path, default=_PROJECT_ROOT / "data",
                   help="Directory containing scenario_*.yaml files")
    p.add_argument("--output-dir",     type=Path, default=_PROJECT_ROOT / "output",
                   help="Output directory for generated JSON files")
    p.add_argument("--refresh-legacy", action="store_true",
                   help="Regenerate data/system.json and data/dataset.json")
    p.add_argument("--multi-seed",     action="store_true",
                   help="Generate per-seed variants for scenarios 01-06")
    p.add_argument("--seeds",          type=str, default="42,123,456,789,2024",
                   help="Comma-separated seeds for multi-seed mode")
    p.add_argument("--scenario",       type=str, default=None,
                   help="Only generate matching scenario name (substring)")
    p.add_argument("--force",          action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--manifest",       type=Path,
                   default=_PROJECT_ROOT / "output" / "dataset_manifest.json",
                   help="Path to write dataset manifest JSON")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print plan without writing any files")
    p.add_argument("-v", "--verbose",  action="store_true",
                   help="Print topology details per scenario")


# ---------------------------------------------------------------------------
# Main Routine for Batch
# ---------------------------------------------------------------------------
def run_batch_generation(args: argparse.Namespace) -> int:
    records: List[DatasetRecord] = []

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    _hdr("SaG Dataset Generator")
    print(f"  Input dir  : {args.input_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Force      : {args.force}")
    print(f"  Multi-seed : {args.multi_seed}  seeds={seeds if args.multi_seed else 'n/a'}")
    print(f"  Dry-run    : {args.dry_run}")

    # ── 1. Discover scenario YAMLs ────────────────────────────────────────
    yaml_files: List[Path] = sorted(args.input_dir.glob("scenario_*.yaml"))
    if not yaml_files:
        _err(f"No scenario_*.yaml files found in {args.input_dir}")
        return 1

    if args.scenario:
        yaml_files = [f for f in yaml_files if args.scenario in f.stem]
        if not yaml_files:
            _err(f"No scenario files match '{args.scenario}'")
            return 1

    # Reorder: scenario_08 (smoke) first, then 01-07 in order
    smoke = [f for f in yaml_files if "scenario_08" in f.stem]
    rest  = [f for f in yaml_files if "scenario_08" not in f.stem]
    yaml_files = smoke + rest

    _hdr(f"Stage 1 / 3 — Scenario datasets  ({len(yaml_files)} scenarios)")

    for yaml_path in yaml_files:
        stem      = yaml_path.stem
        out_path  = args.output_dir / f"{stem}.json"
        record    = generate_from_yaml(
            yaml_path=yaml_path,
            output_path=out_path,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        records.append(record)

    # ── 2. Multi-seed variants ─────────────────────────────────────────────
    if args.multi_seed:
        _hdr(f"Stage 2 / 3 — Multi-seed variants  (seeds: {seeds})")
        ms_files = [f for f in sorted(args.input_dir.glob("scenario_*.yaml"))
                    if _should_multi_seed(f.stem)]
        if args.scenario:
            ms_files = [f for f in ms_files if args.scenario in f.stem]
        for yaml_path in ms_files:
            print(f"\n  {CYAN}{yaml_path.stem}{RESET}")
            sub = generate_multi_seed_variants(
                yaml_path=yaml_path,
                output_dir=args.output_dir,
                seeds=seeds,
                force=args.force,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            records.extend(sub)
    else:
        _hdr("Stage 2 / 3 — Multi-seed variants  (skipped — use --multi-seed to enable)")

    # ── 3. Legacy dataset refresh ──────────────────────────────────────────
    _hdr("Stage 3 / 3 — Legacy dataset refresh")

    if args.refresh_legacy:
        # system.json — medium scale, seed 42  (replaces stale version)
        legacy_system = _PROJECT_ROOT / "data" / "system.json"
        records.append(generate_from_scale(
            scale="medium", seed=42,
            output_path=legacy_system,
            label="data/system.json (medium, seed=42)",
            force=True,
            dry_run=args.dry_run,
            verbose=args.verbose,
        ))
        # dataset.json — small scale, seed 42  (replaces stale version)
        legacy_dataset = _PROJECT_ROOT / "data" / "dataset.json"
        records.append(generate_from_scale(
            scale="small", seed=42,
            output_path=legacy_dataset,
            label="data/dataset.json (small, seed=42)",
            force=True,
            dry_run=args.dry_run,
            verbose=args.verbose,
        ))
    else:
        _warn("Skipping legacy refresh  (pass --refresh-legacy to regenerate "
              "data/system.json and data/dataset.json with code_metrics + system_hierarchy)")

    # ── 4. Summary & manifest ──────────────────────────────────────────────
    _print_summary(records)

    if not args.dry_run:
        _write_manifest(records, args.manifest)

    err_count = sum(1 for r in records if r.status == "error")
    return 1 if err_count else 0
