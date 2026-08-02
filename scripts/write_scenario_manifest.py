#!/usr/bin/env python3
"""
scripts/write_scenario_manifest.py — provenance manifest for the scenario corpus.

Records, per committed dataset in data/scenarios/: the source YAML, its seed,
entity counts, and the canonical SHA-256 of the JSON. The manifest is what makes
corpus drift detectable — a dataset that no longer matches what its config
generates, or a LOSO cache that no longer matches the dataset it was built from,
shows up as a hash mismatch instead of silently changing published numbers.

Usage:
    PYTHONPATH=. python scripts/write_scenario_manifest.py
    PYTHONPATH=. python scripts/write_scenario_manifest.py --check   # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = _PROJECT_ROOT / "data" / "scenarios"
MANIFEST_PATH = SCENARIOS_DIR / "MANIFEST.json"

#: config stem → committed dataset stem, and the role that dataset plays.
#: Mirrors SCENARIO_SYSTEM_MAP in cli/common/batch_generation.py; roles are the
#: ones documented in docs/scenario.md.
CORPUS = {
    "scenario_01_autonomous_vehicle": ("av_system", "evaluation"),
    "scenario_02_iot_smart_city": ("iot_smart_city_system", "evaluation"),
    "scenario_03_financial_trading": ("financial_trading_system", "evaluation"),
    "scenario_04_healthcare": ("healthcare_system", "evaluation"),
    "scenario_05_hub_and_spoke": ("hub_and_spoke_system", "evaluation"),
    "scenario_06_microservices": ("microservices_system", "evaluation"),
    "scenario_07_enterprise_xlarge": ("enterprise_system", "evaluation"),
    "scenario_08_tiny_regression": ("tiny_system", "fixture"),
    "scenario_09_xlarge_stress": ("xlarge_system", "fixture"),
    "scenario_10_atm_system": ("atm_system", "case_study"),
    "scenario_11_integration_hub_migration": ("integration_hub_migration_system", "fixture"),
}

ENTITY_KEYS = ("applications", "topics", "brokers", "nodes", "libraries")


def canonical_sha256(data: dict) -> str:
    """SHA-256 of the JSON-canonical form (sorted keys, no whitespace).

    Same construction as tests/test_generation_service.py::_canonical_sha256, so
    the golden hash pinned there is directly comparable to the manifest entry.
    """
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _generator_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_manifest() -> dict:
    entries = {}
    for config_stem, (dataset_stem, role) in sorted(CORPUS.items()):
        dataset_path = SCENARIOS_DIR / f"{dataset_stem}.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset: {dataset_path}")
        data = json.loads(dataset_path.read_text())
        entries[dataset_stem] = {
            "config": f"{config_stem}.yaml",
            "role": role,
            "seed": (data.get("metadata") or {}).get("seed"),
            "domain": (data.get("metadata") or {}).get("domain"),
            "counts": {k: len(data.get(k) or []) for k in ENTITY_KEYS},
            "sha256": canonical_sha256(data),
        }

    evaluation = [e for e in entries.values() if e["role"] == "evaluation"]
    pooled = {k: sum(e["counts"][k] for e in evaluation) for k in ENTITY_KEYS}
    pooled["total"] = sum(pooled.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_commit": _generator_commit(),
        "regenerate_with": (
            "PYTHONPATH=. python cli/generate_graph.py batch "
            "--input-dir data/scenarios --output-dir <dir> --force"
        ),
        "pooled_evaluation_suite": pooled,
        "datasets": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify committed datasets against MANIFEST.json; write nothing.")
    args = ap.parse_args()

    current = build_manifest()

    if args.check:
        if not MANIFEST_PATH.exists():
            print(f"✗ No manifest at {MANIFEST_PATH}")
            return 1
        stored = json.loads(MANIFEST_PATH.read_text())
        failures = []
        for name, entry in current["datasets"].items():
            want = (stored.get("datasets") or {}).get(name)
            if want is None:
                failures.append(f"{name}: not in manifest")
            elif want["sha256"] != entry["sha256"]:
                failures.append(
                    f"{name}: sha256 {want['sha256'][:12]} (manifest) != "
                    f"{entry['sha256'][:12]} (on disk)"
                )
        for name in (stored.get("datasets") or {}):
            if name not in current["datasets"]:
                failures.append(f"{name}: in manifest but missing on disk")
        if failures:
            print("✗ Corpus does not match MANIFEST.json:")
            for f in failures:
                print(f"    {f}")
            return 1
        print(f"✓ {len(current['datasets'])} datasets match MANIFEST.json")
        return 0

    MANIFEST_PATH.write_text(json.dumps(current, indent=2) + "\n")
    print(f"✓ Wrote {MANIFEST_PATH.relative_to(_PROJECT_ROOT)} "
          f"({len(current['datasets'])} datasets, "
          f"pooled evaluation suite = {current['pooled_evaluation_suite']['total']} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
