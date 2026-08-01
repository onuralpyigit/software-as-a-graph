"""Corpus integrity: the committed datasets must be what their configs generate.

This guards a failure the repository has already had once. Published Table 3 / Table 4
numbers were computed on cached topologies that had silently diverged from
`data/scenarios/*.json`, so `make table3` reproduced the paper only because the stale
cache won the topology-source race in reproduce/main_table.py. Nothing detected it.

Two invariants close that gap:
  1. every committed dataset still regenerates byte-identically from its YAML config;
  2. MANIFEST.json agrees with what is on disk, so drift is a test failure rather than
     a quiet change in published numbers.
"""

import json
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent

from scripts.write_scenario_manifest import (  # noqa: E402
    CORPUS,
    MANIFEST_PATH,
    SCENARIOS_DIR,
    canonical_sha256,
)
from tools.generation import GenerationService, load_config  # noqa: E402

#: Pooled node population of the seven evaluation scenarios, as reported in
#: docs/research/jss/draft.md §7.1. A change here is a change to the paper.
_PAPER_POOLED_COUNTS = {
    "applications": 850,
    "topics": 375,
    "libraries": 165,
    "nodes": 119,
    "brokers": 36,
}


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.fail(
            f"{MANIFEST_PATH} is missing. Regenerate it with "
            "`PYTHONPATH=. python scripts/write_scenario_manifest.py`."
        )
    return json.loads(MANIFEST_PATH.read_text())


@pytest.mark.parametrize("config_stem,dataset_stem", [
    (c, d) for c, (d, _role) in sorted(CORPUS.items())
])
def test_dataset_regenerates_from_its_config(config_stem, dataset_stem):
    """The committed JSON must be exactly what its YAML produces today."""
    config_path = SCENARIOS_DIR / f"{config_stem}.yaml"
    dataset_path = SCENARIOS_DIR / f"{dataset_stem}.json"
    assert config_path.exists(), f"Missing config: {config_path}"
    assert dataset_path.exists(), f"Missing dataset: {dataset_path}"

    regenerated = GenerationService(config=load_config(config_path)).generate()
    committed = json.loads(dataset_path.read_text())

    assert canonical_sha256(regenerated) == canonical_sha256(committed), (
        f"{dataset_stem}.json no longer matches {config_stem}.yaml. Either the generator "
        f"changed behaviour or the dataset was hand-edited. Regenerate the corpus:\n"
        f"    PYTHONPATH=. python cli/generate_graph.py batch "
        f"--input-dir data/scenarios --output-dir data/scenarios --force\n"
        f"    PYTHONPATH=. python scripts/write_scenario_manifest.py\n"
        f"and rebuild the caches (`make -f reproduce/Makefile cache`) before trusting any "
        f"result computed against the old datasets."
    )


def test_manifest_matches_committed_datasets(manifest):
    """MANIFEST.json is the provenance record; it must not drift from the files."""
    mismatches = []
    for name, entry in manifest["datasets"].items():
        path = SCENARIOS_DIR / f"{name}.json"
        if not path.exists():
            mismatches.append(f"{name}: in manifest but missing on disk")
            continue
        actual = canonical_sha256(json.loads(path.read_text()))
        if actual != entry["sha256"]:
            mismatches.append(
                f"{name}: manifest {entry['sha256'][:12]} != on-disk {actual[:12]}"
            )
    for _config, (dataset, _role) in CORPUS.items():
        if dataset not in manifest["datasets"]:
            mismatches.append(f"{dataset}: on disk but absent from manifest")

    assert not mismatches, (
        "MANIFEST.json is out of date:\n  " + "\n  ".join(mismatches)
        + "\nRefresh it with `PYTHONPATH=. python scripts/write_scenario_manifest.py`."
    )


def test_evaluation_suite_matches_paper_population():
    """The seven evaluation scenarios must still pool to the population §7.1 reports."""
    pooled = dict.fromkeys(_PAPER_POOLED_COUNTS, 0)
    for _config, (dataset, role) in CORPUS.items():
        if role != "evaluation":
            continue
        data = json.loads((SCENARIOS_DIR / f"{dataset}.json").read_text())
        for key in pooled:
            pooled[key] += len(data.get(key) or [])

    assert pooled == _PAPER_POOLED_COUNTS, (
        f"Evaluation-suite population changed: {pooled} != {_PAPER_POOLED_COUNTS}. "
        "draft.md §7.1 states 1,545 pooled nodes with this breakdown; update the paper "
        "and this test together, never one alone."
    )
    assert sum(pooled.values()) == 1545
