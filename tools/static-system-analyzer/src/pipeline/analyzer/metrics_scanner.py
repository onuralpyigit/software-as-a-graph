"""
Code metrics scanner module (ck - Code Metrics for Java).

Uses the 'ck' tool (https://github.com/mauricioaniche/ck) to compute
object-oriented metrics for Java projects. A single tool provides all
four required metric categories:

1. Size       – LOC, CLOC, class/method/field counts
2. Complexity – WMC (Weighted Methods per Class = Σ CCN)
3. Cohesion   – LCOM (Lack of Cohesion of Methods)
4. Coupling   – CBO, RFC, Fan-in, Fan-out

ck is a standalone JAR; no server, no configuration, fully offline.
"""

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger import log_info, log_warning, log_error, log_debug
from common.shared_store import (
    SHARED_DIR_NAME,
    ensure_platform_link,
    shared_path,
)

# Path to the ck JAR inside the Docker container (set in Dockerfile)
CK_JAR_PATH = os.environ.get("CK_JAR_PATH", "/opt/ck/ck.jar")


@dataclass
class MetricsConfig:
    """Configuration for code metrics scanning.

    Attributes:
        platform_name: Platform name for grouping projects.
        source_dir: Root directory containing cloned project sources.
        output_dir: Directory where _metrics.json files will be written.
        logs_dir: Directory for log files.
        log_level: Logging level.
    """
    platform_name: str
    source_dir: str
    output_dir: str
    logs_dir: str
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# ck runner
# ---------------------------------------------------------------------------

def _run_ck(source_path: str) -> Optional[List[Dict[str, str]]]:
    """Run ck on a Java project directory and return parsed class.csv rows.

    ck command:
        java -jar ck.jar <source> false 0 true <output_dir>

    Args:
        source_path: Path to Java project source directory.

    Returns:
        List of dicts (one per class) parsed from class.csv, or None on
        failure.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ck_out_")
    try:
        # ck treats the last arg as a file prefix; trailing '/' ensures
        # output goes INTO the directory (e.g. /tmp/ck_out_xxx/class.csv)
        ck_output_prefix = tmp_dir.rstrip("/") + "/"

        cmd = [
            "java", "-jar", CK_JAR_PATH,
            source_path,       # source directory
            "false",            # use_jars
            "0",                # max_at_once (0 = unlimited)
            "true",             # variables_and_fields
            ck_output_prefix,   # output file prefix (trailing '/')
        ]

        log_info(f"Running ck: java -jar ck.jar {source_path} ...")
        log_debug(f"ck command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per project
        )

        if result.returncode != 0:
            log_warning(f"ck exited with code {result.returncode} for: {source_path}")
            if result.stderr:
                log_warning(f"ck stderr (last 500 chars): {result.stderr[-500:]}")

        # Find class.csv in the output directory
        class_csv = os.path.join(tmp_dir, "class.csv")
        if not os.path.isfile(class_csv):
            log_warning(f"ck did not produce class.csv for: {source_path}")
            if result.stdout:
                log_debug(f"ck stdout: {result.stdout[-300:]}")
            return None

        rows = _parse_class_csv(class_csv)
        log_info(f"ck produced metrics for {len(rows)} classes.")
        return rows

    except subprocess.TimeoutExpired:
        log_error(f"ck timed out for: {source_path}")
        return None
    except FileNotFoundError:
        log_error(f"java or ck.jar not found. Ensure JRE is installed and "
                  f"CK_JAR_PATH={CK_JAR_PATH} exists.")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_class_csv(csv_path: str) -> List[Dict[str, str]]:
    """Parse ck class.csv into a list of row dicts.

    Args:
        csv_path: Path to class.csv file.

    Returns:
        List of dicts keyed by CSV column name.
    """
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except (IOError, csv.Error) as e:
        log_error(f"Failed to parse class.csv: {e}")
    return rows


# ---------------------------------------------------------------------------
# Metric summarisation helpers
# ---------------------------------------------------------------------------

def _safe_int(value: str, default: int = 0) -> int:
    """Safely parse an integer from a CSV string value."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value: str, default: float = 0.0) -> float:
    """Safely parse a float from a CSV string value."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _summarize_size(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute size metrics from ck class.csv rows.

    Metrics:
        total_loc             – Total lines of code across all classes
        total_classes         – Number of classes analysed
        total_methods         – Sum of totalMethodsQty
        total_fields          – Sum of totalFieldsQty

    Args:
        rows: Parsed class.csv rows.

    Returns:
        Size metrics dictionary.
    """
    total_loc = sum(_safe_int(r.get("loc")) for r in rows)
    total_methods = sum(_safe_int(r.get("totalMethodsQty")) for r in rows)
    total_fields = sum(_safe_int(r.get("totalFieldsQty")) for r in rows)
    total_classes = len(rows)

    return {
        "total_loc": total_loc,
        "total_classes": total_classes,
        "total_methods": total_methods,
        "total_fields": total_fields,
    }


def _summarize_complexity(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute complexity metrics from ck class.csv rows.

    Metrics:
        total_wmc   – Sum of WMC (= sum of cyclomatic complexity of all methods)
        avg_wmc     – Average WMC per class
        max_wmc     – Maximum WMC among classes
        high_complexity_classes – Classes with WMC ≥ 50 (top 20)

    Args:
        rows: Parsed class.csv rows.

    Returns:
        Complexity metrics dictionary.
    """
    n = len(rows)
    if n == 0:
        return {
            "total_wmc": 0, "avg_wmc": 0.0, "max_wmc": 0,
            "high_complexity_classes": [],
        }

    wmcs = [_safe_int(r.get("wmc")) for r in rows]

    # High-complexity classes (WMC >= 50)
    high = []
    for r in sorted(rows, key=lambda r: _safe_int(r.get("wmc")), reverse=True):
        wmc = _safe_int(r.get("wmc"))
        if wmc >= 50:
            high.append({
                "class": r.get("class", "unknown"),
                "file": r.get("file", "unknown"),
                "wmc": wmc,
                "loc": _safe_int(r.get("loc")),
            })
        if len(high) >= 20:
            break

    return {
        "total_wmc": sum(wmcs),
        "avg_wmc": round(sum(wmcs) / n, 2),
        "max_wmc": max(wmcs),
        "high_complexity_classes": high,
    }


def _summarize_coupling(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute coupling metrics from ck class.csv rows.

    Metrics:
        avg_cbo     – Average Coupling Between Objects
        max_cbo     – Maximum CBO
        avg_rfc     – Average Response For a Class
        max_rfc     – Maximum RFC
        avg_fanin   – Average Fan-in (incoming dependencies)
        max_fanin   – Maximum Fan-in
        avg_fanout  – Average Fan-out (outgoing dependencies)
        max_fanout  – Maximum Fan-out
        high_coupling_classes – Classes with CBO ≥ 30 (top 20)

    Args:
        rows: Parsed class.csv rows.

    Returns:
        Coupling metrics dictionary.
    """
    n = len(rows)
    if n == 0:
        return {
            "avg_cbo": 0.0, "max_cbo": 0,
            "avg_rfc": 0.0, "max_rfc": 0,
            "avg_fanin": 0.0, "max_fanin": 0,
            "avg_fanout": 0.0, "max_fanout": 0,
            "high_coupling_classes": [],
        }

    cbos = [_safe_int(r.get("cbo")) for r in rows]
    rfcs = [_safe_int(r.get("rfc")) for r in rows]
    fanins = [_safe_int(r.get("fanin")) for r in rows]
    fanouts = [_safe_int(r.get("fanout")) for r in rows]

    # High-coupling classes (CBO >= 30)
    high = []
    for r in sorted(rows, key=lambda r: _safe_int(r.get("cbo")), reverse=True):
        cbo = _safe_int(r.get("cbo"))
        if cbo >= 30:
            high.append({
                "class": r.get("class", "unknown"),
                "file": r.get("file", "unknown"),
                "cbo": cbo,
                "rfc": _safe_int(r.get("rfc")),
                "fanin": _safe_int(r.get("fanin")),
                "fanout": _safe_int(r.get("fanout")),
            })
        if len(high) >= 20:
            break

    return {
        "avg_cbo": round(sum(cbos) / n, 2),
        "max_cbo": max(cbos),
        "avg_rfc": round(sum(rfcs) / n, 2),
        "max_rfc": max(rfcs),
        "avg_fanin": round(sum(fanins) / n, 2),
        "max_fanin": max(fanins),
        "avg_fanout": round(sum(fanouts) / n, 2),
        "max_fanout": max(fanouts),
        "high_coupling_classes": high,
    }


def _summarize_cohesion(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute cohesion metrics from ck class.csv rows.

    ck computes LCOM (Lack of Cohesion of Methods) per class.
    Higher LCOM → lower cohesion → class may be doing too many things.

    Metrics:
        avg_lcom              – Average LCOM across classes
        max_lcom              – Maximum LCOM
        low_cohesion_classes  – Classes with LCOM ≥ 100 (top 20)

    Args:
        rows: Parsed class.csv rows.

    Returns:
        Cohesion metrics dictionary.
    """
    n = len(rows)
    if n == 0:
        return {
            "avg_lcom": 0.0, "max_lcom": 0,
            "low_cohesion_classes": [],
        }

    lcoms = [_safe_float(r.get("lcom")) for r in rows]

    # Low-cohesion classes (LCOM >= 100)
    low = []
    for r in sorted(rows, key=lambda r: _safe_float(r.get("lcom")), reverse=True):
        lcom_val = _safe_float(r.get("lcom"))
        if lcom_val >= 100:
            low.append({
                "class": r.get("class", "unknown"),
                "file": r.get("file", "unknown"),
                "lcom": lcom_val,
                "totalMethodsQty": _safe_int(r.get("totalMethodsQty")),
                "totalFieldsQty": _safe_int(r.get("totalFieldsQty")),
            })
        if len(low) >= 20:
            break

    return {
        "avg_lcom": round(sum(lcoms) / n, 2),
        "max_lcom": max(lcoms) if lcoms else 0,
        "low_cohesion_classes": low,
    }


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def _write_metrics_json(
    output_dir: str,
    versioned_name: str,
    metrics: Optional[Dict[str, Any]],
    platform: Optional[str] = None,
) -> str:
    """Write code metrics to a per-application JSON file.

    When ``platform`` is provided, the JSON is written once into the shared
    store (``<output_dir>/../_shared/<versioned_name>_metrics.json``) and a
    relative symlink is created under the per-platform directory. Without
    ``platform`` the legacy direct-write behaviour is preserved.

    Args:
        output_dir: Per-platform output directory.
        versioned_name: ``<app_name>_<version>`` identifier (e.g.
            ``common_lib_1.0.0``). Used as the filename stem to avoid
            collisions between different versions of the same app.
        metrics: Metric dictionary, or None if scan failed.
        platform: Platform name; enables the shared-store layout.

    Returns:
        Path to the written (or linked) JSON file.
    """
    artifact = f"{versioned_name}_metrics.json"
    data = metrics if metrics is not None else {}

    if platform:
        base_dir = Path(output_dir).parent
        shared_target = shared_path(base_dir, artifact)
        shared_target.parent.mkdir(parents=True, exist_ok=True)
        with shared_target.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        link_path = ensure_platform_link(base_dir, platform, artifact)
        log_info(f"Metrics results written to: {shared_target} (linked: {link_path})")
        return str(link_path)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, artifact)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log_info(f"Metrics results written to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_project(
    config: MetricsConfig,
    app_name: str,
    app_source_path: str,
    versioned_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Scan a single Java project with ck and produce size / complexity / cohesion / coupling metrics.

    Steps:
        1. Reuse existing shared metrics when present (dedup by versioned_name)
        2. Run ck → class.csv
        3. Parse and summarise size, complexity, cohesion, coupling
        4. Write _metrics.json (shared store + per-platform symlink)

    Args:
        config: Metrics configuration.
        app_name: Application name (used for logs and legacy filename).
        app_source_path: Path to application source code.
        versioned_name: ``<app_name>_<version>``; used for the dedup key and
            the on-disk filename. Falls back to ``app_name`` for callers
            that have not yet been updated.

    Returns:
        Dictionary with size / complexity / cohesion / coupling, or None on failure.
    """
    versioned_name = versioned_name or app_name
    platform = config.platform_name

    # Dedup: reuse shared metrics if present
    base_dir = Path(config.output_dir).parent
    artifact = f"{versioned_name}_metrics.json"
    shared_target = shared_path(base_dir, artifact)
    if platform and shared_target.exists():
        log_info(f"Reusing shared metrics for {versioned_name} ({shared_target})")
        ensure_platform_link(base_dir, platform, artifact)
        try:
            with shared_target.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            return cached if cached else None
        except (OSError, json.JSONDecodeError) as exc:
            log_warning(f"Could not load shared metrics {shared_target}: {exc}")

    log_info(f"Scanning project: {app_name} ({app_source_path})")

    rows = _run_ck(app_source_path)

    if rows is None or len(rows) == 0:
        log_warning(f"ck produced no class data for: {app_name}")
        _write_metrics_json(config.output_dir, versioned_name, None, platform=platform)
        return None

    metrics: Dict[str, Any] = {
        "size": _summarize_size(rows),
        "complexity": _summarize_complexity(rows),
        "cohesion": _summarize_cohesion(rows),
        "coupling": _summarize_coupling(rows),
    }

    log_info(
        f"  size: {metrics['size']['total_loc']} LOC, "
        f"{metrics['size']['total_classes']} classes, "
        f"{metrics['size']['total_methods']} methods"
    )
    log_info(
        f"  complexity: avg_wmc={metrics['complexity']['avg_wmc']}, "
        f"max_wmc={metrics['complexity']['max_wmc']}"
    )
    log_info(
        f"  cohesion: avg_lcom={metrics['cohesion']['avg_lcom']}, "
        f"max_lcom={metrics['cohesion']['max_lcom']}"
    )
    log_info(
        f"  coupling: avg_cbo={metrics['coupling']['avg_cbo']}, "
        f"max_cbo={metrics['coupling']['max_cbo']}, "
        f"avg_fanin={metrics['coupling']['avg_fanin']}, "
        f"avg_fanout={metrics['coupling']['avg_fanout']}"
    )

    _write_metrics_json(config.output_dir, versioned_name, metrics, platform=platform)
    return metrics


def scan_all_projects(config: MetricsConfig) -> Dict[str, Optional[Dict[str, Any]]]:
    """Scan all application directories under the platform source directory.

    Discovers project folders under source_dir/<platform_name>/ and scans
    each one with ck.

    Args:
        config: Metrics configuration.

    Returns:
        Dictionary mapping app_name → metrics dict (or None on failure).
    """
    platform_source = Path(config.source_dir) / config.platform_name
    if not platform_source.is_dir():
        log_error(f"Platform source directory not found: {platform_source}")
        return {}

    results: Dict[str, Optional[Dict[str, Any]]] = {}
    project_dirs = sorted([
        d for d in platform_source.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    log_info(f"Found {len(project_dirs)} project directories to scan.")

    for project_dir in project_dirs:
        # Extract base app name (remove version suffix: app_name_1.0.0 → app_name)
        parts = project_dir.name.rsplit("_", 1)
        if len(parts) == 2 and _is_version_string(parts[1]):
            app_name = parts[0]
        else:
            app_name = project_dir.name

        versioned_name = project_dir.name
        metrics = scan_project(config, app_name, str(project_dir), versioned_name=versioned_name)
        results[versioned_name] = metrics

    success_count = sum(1 for v in results.values() if v is not None)
    log_info(f"Metrics scanning complete. {success_count}/{len(results)} successful.")
    return results


def _is_version_string(s: str) -> bool:
    """Check if a string looks like a version number (e.g. '1.0.0', '2.3').

    Args:
        s: String to check.

    Returns:
        True if the string appears to be a version number.
    """
    parts = s.split(".")
    return len(parts) >= 2 and all(p.isdigit() for p in parts)
