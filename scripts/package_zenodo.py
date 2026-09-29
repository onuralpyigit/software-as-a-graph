#!/usr/bin/env python3
"""
scripts/package_zenodo.py — Build and optionally upload the Zenodo replication package
======================================================================================

Creates the distribution archives for Zenodo:
  1. SaG_JSS_Results_<stamp>.tar.gz      — Verified results bundle + MANIFEST.json (~1.5 MB)
  2. SaG_JSS_GNN_Checkpoints.tar.gz      — Pretrained PyG model checkpoints (~140 MB)
  3. SaG_JSS_Replication_Package.zip     — Full standalone replication package (code, data, Dockerfile, scripts)
  4. CHECKSUMS_SHA256.txt                — Cryptographic digests for all archives
  5. README_ZENODO.md                    — Manifest and unpacking instructions for Zenodo

Usage:
  # Package only (local build into dist/zenodo/):
  python scripts/package_zenodo.py

  # Test upload to Zenodo Sandbox:
  python scripts/package_zenodo.py --upload --sandbox --token <SANDBOX_TOKEN>

  # Upload to Zenodo (creates new draft or updates existing):
  python scripts/package_zenodo.py --upload --token <ZENODO_TOKEN>
  python scripts/package_zenodo.py --upload --token <ZENODO_TOKEN> --deposition-id <ID>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "zenodo"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "output" / "gnn_checkpoints"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_reconciled() -> None:
    print("▶ Step 1: Reconciling manuscript figures against artifacts...")
    cmd = [sys.executable, str(ROOT / "reproduce" / "reconcile_manuscript.py")]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print("✗ Manuscript reconciliation failed:", file=sys.stderr)
        print(res.stderr or res.stdout, file=sys.stderr)
        sys.exit(1)
    print("  ✓ Manuscript reconciliation verified (all figures match).")


def get_or_cut_bundle() -> Path:
    print("▶ Step 2: Verifying results bundle...")
    cmd = [sys.executable, str(ROOT / "reproduce" / "cut_results_bundle.py")]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print("✗ Results bundle cutting failed:", file=sys.stderr)
        print(res.stderr or res.stdout, file=sys.stderr)
        sys.exit(1)
    
    # Locate the newest SaG_JSS_Results_* directory
    bundles = sorted(RESULTS_DIR.glob("SaG_JSS_Results_*"))
    if not bundles:
        print("✗ No SaG_JSS_Results_* directory found in results/", file=sys.stderr)
        sys.exit(1)
    bundle_path = bundles[-1]
    print(f"  ✓ Latest bundle: {bundle_path.name}")
    return bundle_path


def package_results_bundle(bundle_dir: Path, out_dir: Path) -> Path:
    archive_name = f"{bundle_dir.name}.tar.gz"
    out_path = out_dir / archive_name
    print(f"▶ Step 3: Compressing results bundle → {archive_name}...")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created {out_path.name} ({size_mb:.2f} MB)")
    return out_path


def package_checkpoints(out_dir: Path) -> Optional[Path]:
    if not CHECKPOINTS_DIR.exists():
        print(f"  ⚠ Checkpoints directory {CHECKPOINTS_DIR} not found, skipping checkpoint archive.")
        return None
    archive_name = "SaG_JSS_GNN_Checkpoints.tar.gz"
    out_path = out_dir / archive_name
    print(f"▶ Step 4: Compressing GNN model checkpoints → {archive_name}...")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(CHECKPOINTS_DIR, arcname="gnn_checkpoints")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created {out_path.name} ({size_mb:.2f} MB)")
    return out_path


def package_full_replication(bundle_dir: Path, out_dir: Path) -> Path:
    archive_name = "SaG_JSS_Replication_Package_v1.0.0.zip"
    out_path = out_dir / archive_name
    print(f"▶ Step 5: Assembling complete standalone replication package → {archive_name}...")

    # Inclusion list relative to ROOT
    includes = [
        "saag",
        "tools",
        "cli",
        "api",
        "data/scenarios",
        "data/benchmarks",
        "data/dataset.json",
        "data/system.json",
        "data/graph_config.yaml",
        "reproduce",
        "docs/research/jss",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml",
        "LICENSE",
        "README.md",
        ".zenodo.json",
    ]

    # Exclude patterns
    def should_exclude(rel: str) -> bool:
        parts = Path(rel).parts
        if any(p in (".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", "vendor") for p in parts):
            return True
        if any(p.endswith((".pyc", ".pyo", ".log", ".aux", ".out", ".spl", ".synctex.gz")) for p in parts):
            return True
        return False

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add included trees and files
        for inc in includes:
            src = ROOT / inc
            if not src.exists():
                continue
            if src.is_file():
                if not should_exclude(inc):
                    zipf.write(src, arcname=f"software-as-a-graph/{inc}")
            elif src.is_dir():
                for file_path in src.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(ROOT)
                        if not should_exclude(str(rel)):
                            zipf.write(file_path, arcname=f"software-as-a-graph/{rel}")

        # Add the results bundle
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(bundle_dir)
                zipf.write(file_path, arcname=f"software-as-a-graph/results/{bundle_dir.name}/{rel}")

        # Add standalone REPLICATION_README
        readme_content = f"""# Software-as-a-Graph: Replication Package (v1.0.0)

Journal of Systems and Software (JSS) — Special Issue VSI:AI4MSS
Manuscript: "Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish-Subscribe Systems?"
Authors: Ibrahim Onuralp Yigit, Feza Buzluca (Istanbul Technical University)

## Quick Verification
1. Verify empirical results against manuscript numbers:
   python reproduce/reconcile_manuscript.py

2. Run Block-0 test suite (deterministic gate):
   make -f reproduce/Makefile block0

3. Docker replication (recommended):
   docker build -t sag-jss -f reproduce/Dockerfile .
   docker run --rm sag-jss make -f reproduce/Makefile block0

See `reproduce/README.md` and `docs/research/jss/experiments/README.md` for detailed instructions.
"""
        zipf.writestr("software-as-a-graph/README_REPLICATION.md", readme_content)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created {out_path.name} ({size_mb:.2f} MB)")
    return out_path


def generate_checksums(files: List[Path], out_dir: Path) -> Path:
    print("▶ Step 6: Generating cryptographic SHA-256 manifest...")
    manifest_path = out_dir / "CHECKSUMS_SHA256.txt"
    lines = []
    for f in files:
        digest = sha256_file(f)
        lines.append(f"{digest}  {f.name}")
    content = "\n".join(lines) + "\n"
    manifest_path.write_text(content)
    print(f"  ✓ Manifest written to {manifest_path.name}")
    return manifest_path


def generate_zenodo_doc(files: List[Path], bundle_name: str, out_dir: Path) -> Path:
    readme_path = out_dir / "README_ZENODO.md"
    content = f"""# Zenodo Replication Package — Software-as-a-Graph

**Title:** Software-as-a-Graph: Replication Package (Datasets, Generator Configurations, Simulation Harnesses, Model Checkpoints, and Analysis Scripts)  
**Journal:** Journal of Systems and Software (Special Issue: VSI:AI4MSS)  
**Authors:** Ibrahim Onuralp Yigit, Feza Buzluca (*Istanbul Technical University*)  
**GitHub Repository:** [https://github.com/onuralpyigit/software-as-a-graph](https://github.com/onuralpyigit/software-as-a-graph)  
**Release Date:** {datetime.now():%Y-%m-%d}

---

## Contents of this Deposit

This deposit contains the complete empirical replication package backing all findings, tables, and figures reported in the manuscript:

| File | Description | Size |
|---|---|---|
| `SaG_JSS_Replication_Package_v1.0.0.zip` | Full standalone replication package (source code, datasets, scenario configs, reproduction scripts, Dockerfile, and the verified results bundle). | ~35 MB |
| `SaG_JSS_GNN_Checkpoints.tar.gz` | Trained PyTorch Geometric model checkpoints for all architectures (HGT-QoS, HGT, GAT, GAT-QoS, etc.) across 12 scenario topologies and 5 random seeds (42, 123, 456, 789, 2024). | ~140 MB |
| `{bundle_name}.tar.gz` | Standalone results bundle containing all 36 empirical JSON result artifacts backing reported tables, 12 rendered LaTeX/CSV/Markdown tables, figures, and `MANIFEST.json`. | ~1.5 MB |
| `CHECKSUMS_SHA256.txt` | SHA-256 cryptographic checksums for all files in this deposit. | <1 KB |

---

## Verification & Replication

### 1. Mechanical Verification (Instant)
Verify that all 1,601 numbers reported in the manuscript match the empirical artifacts:
```bash
python reproduce/reconcile_manuscript.py
# Expected output: OK — 1601 figures match their artifacts.
```

### 2. Sanity Check Gate (10 seconds)
```bash
make -f reproduce/Makefile block0
# Expected output: 46 passed. ✓ Block 0 PASSED — W1 audit gate cleared.
```

### 3. Containerized Replication via Docker
```bash
docker build -t sag-jss -f reproduce/Dockerfile .
docker run --rm sag-jss make -f reproduce/Makefile block0
```

Refer to `reproduce/README.md` and `docs/research/jss/experiments/README.md` for in-depth reproduction documentation.
"""
    readme_path.write_text(content)
    print(f"  ✓ Zenodo documentation written to {readme_path.name}")
    return readme_path


def upload_to_zenodo(
    files: List[Path],
    token: str,
    sandbox: bool = False,
    deposition_id: Optional[str] = None,
    publish: bool = False,
) -> None:
    try:
        import requests
    except ImportError:
        print("✗ The 'requests' package is required for Zenodo API upload. Please run: pip install requests", file=sys.stderr)
        sys.exit(1)

    base_url = "https://sandbox.zenodo.org/api/deposit/depositions" if sandbox else "https://zenodo.org/api/deposit/depositions"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"\n▶ Uploading to Zenodo ({'Sandbox' if sandbox else 'Production'})...")

    # Load metadata from .zenodo.json
    zenodo_json_path = ROOT / ".zenodo.json"
    if not zenodo_json_path.exists():
        print("✗ .zenodo.json not found in repository root.", file=sys.stderr)
        sys.exit(1)
    raw_meta = json.loads(zenodo_json_path.read_text())
    payload_metadata = {"metadata": raw_meta}

    if deposition_id:
        print(f"  Using existing deposition: {deposition_id}")
        r = requests.get(f"{base_url}/{deposition_id}", headers=headers)
        if r.status_code != 200:
            print(f"✗ Failed to access deposition {deposition_id}: {r.status_code} {r.text}", file=sys.stderr)
            sys.exit(1)
        depo = r.json()
    else:
        print("  Creating new draft deposition...")
        r = requests.post(base_url, headers=headers, json=payload_metadata)
        if r.status_code not in (200, 201):
            print(f"✗ Failed to create deposition: {r.status_code} {r.text}", file=sys.stderr)
            sys.exit(1)
        depo = r.json()
        depo_id = depo["id"]
        print(f"  ✓ Created draft deposition ID: {depo_id}")

    depo_id = depo["id"]
    bucket_url = depo.get("links", {}).get("bucket")

    # Upload files using Bucket API
    if bucket_url:
        for file_path in files:
            print(f"  Uploading {file_path.name} ({file_path.stat().st_size / (1024*1024):.2f} MB)...")
            with file_path.open("rb") as fp:
                ur = requests.put(f"{bucket_url}/{file_path.name}", data=fp, headers=headers)
                if ur.status_code not in (200, 201):
                    print(f"  ✗ Failed to upload {file_path.name}: {ur.status_code} {ur.text}", file=sys.stderr)
                    sys.exit(1)
            print(f"  ✓ Uploaded {file_path.name}")
    else:
        # Fallback to legacy files API
        files_url = f"{base_url}/{depo_id}/files"
        for file_path in files:
            print(f"  Uploading {file_path.name} (legacy API)...")
            with file_path.open("rb") as fp:
                ur = requests.post(files_url, data={"name": file_path.name}, files={"file": fp}, headers=headers)
                if ur.status_code not in (200, 201):
                    print(f"  ✗ Failed to upload {file_path.name}: {ur.status_code} {ur.text}", file=sys.stderr)
                    sys.exit(1)
            print(f"  ✓ Uploaded {file_path.name}")

    # Update metadata
    print("  Updating deposition metadata...")
    mr = requests.put(f"{base_url}/{depo_id}", headers=headers, json=payload_metadata)
    if mr.status_code not in (200, 201):
        print(f"  ⚠ Metadata update returned: {mr.status_code} {mr.text}")
    else:
        print("  ✓ Metadata successfully applied.")

    depo_url = depo.get("links", {}).get("html", f"https://zenodo.org/deposit/{depo_id}")
    doi = depo.get("metadata", {}).get("prereserve_doi", {}).get("doi") or depo.get("doi")
    print("\n" + "=" * 70)
    print(f"🎉 Deposition ready on Zenodo!")
    print(f"  Deposition ID : {depo_id}")
    print(f"  URL           : {depo_url}")
    if doi:
        print(f"  Reserved DOI  : {doi}")
    print("=" * 70)

    if publish:
        confirm = input("\n⚠ Are you sure you want to PUBLISH? This is irreversible! (yes/no): ")
        if confirm.strip().lower() == "yes":
            pub_r = requests.post(f"{base_url}/{depo_id}/actions/publish", headers=headers)
            if pub_r.status_code in (200, 201, 202):
                pub_data = pub_r.json()
                print(f"✓ PUBLISHED! Permanent DOI: {pub_data.get('doi')}")
            else:
                print(f"✗ Publish failed: {pub_r.status_code} {pub_r.text}", file=sys.stderr)
        else:
            print("  Publish aborted. Record remains in Draft state.")
    else:
        print("\nNote: Deposition is saved in DRAFT mode so you can review it before publishing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-verify", action="store_true", help="Skip manuscript reconciliation check.")
    parser.add_argument("--out-dir", type=Path, default=DIST, help="Output directory for archives.")
    parser.add_argument("--upload", action="store_true", help="Upload the generated packages to Zenodo.")
    parser.add_argument("--token", type=str, default=os.getenv("ZENODO_TOKEN") or os.getenv("ZENODO_ACCESS_TOKEN"),
                        help="Zenodo Personal Access Token.")
    parser.add_argument("--sandbox", action="store_true", help="Target Zenodo Sandbox instead of production.")
    parser.add_argument("--deposition-id", type=str, default=None, help="Existing draft deposition ID to update.")
    parser.add_argument("--publish", action="store_true", help="Publish the deposition immediately after upload.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_verify:
        ensure_reconciled()

    bundle_dir = get_or_cut_bundle()

    packages: List[Path] = []
    # 1. Results bundle
    pkg_results = package_results_bundle(bundle_dir, out_dir)
    packages.append(pkg_results)

    # 2. Checkpoints
    pkg_ckpts = package_checkpoints(out_dir)
    if pkg_ckpts:
        packages.append(pkg_ckpts)

    # 3. Full replication archive
    pkg_full = package_full_replication(bundle_dir, out_dir)
    packages.append(pkg_full)

    # 4. Checksums
    checksum_file = generate_checksums(packages, out_dir)
    packages.append(checksum_file)

    # 5. Zenodo Doc
    zenodo_doc = generate_zenodo_doc(packages, bundle_dir.name, out_dir)

    print("\n✓ Zenodo packaging complete! Generated files:")
    for p in packages:
        size = f"({p.stat().st_size / (1024*1024):.2f} MB)" if p.stat().st_size > 1024 * 1024 else f"({p.stat().st_size} bytes)"
        print(f"  • {p.name:45s} {size}")
    print(f"  • {zenodo_doc.name}")

    if args.upload:
        if not args.token:
            print("\n✗ Upload requested but no Zenodo token provided. Pass --token or set ZENODO_TOKEN.", file=sys.stderr)
            return 1
        upload_to_zenodo(
            files=[pkg_results, pkg_ckpts, pkg_full, checksum_file] if pkg_ckpts else [pkg_results, pkg_full, checksum_file],
            token=args.token,
            sandbox=args.sandbox,
            deposition_id=args.deposition_id,
            publish=args.publish,
        )
    else:
        print("\nTo upload to Zenodo:")
        print("  Option A (Web UI): Upload the files in 'dist/zenodo/' to your deposit at https://zenodo.org/deposit")
        print("  Option B (CLI):    python scripts/package_zenodo.py --upload --token <YOUR_TOKEN>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
