"""
enrich_dataset.py
=================
Fetches SonarQube static analysis metrics for every application in the
graph-model dataset and writes an enriched JSON file that Step 1
(Graph Model Construction) can consume directly.

Pipeline position:

    dataset.json  +  SonarQube Web API
            │                │
            └────────────────┘
                    │
            [enrich_dataset.py]
                    │
            enriched_dataset.json
                    │
            Step 1: Graph Model Construction
                    │
            Step 2-6: Analysis → Validation → Visualisation

The enriched dataset is *backward-compatible*: all original fields are
preserved unchanged.  SonarQube data is added as a ``static_analysis``
block on each application node and as a ``gnn_features`` block (indices
18-23 of the extended feature vector) for direct consumption by the GNN
training pipeline.

Usage
-----
# Real SonarQube instance
python enrich_dataset.py \\
    --input  input/dataset.json \\
    --output input/enriched_dataset.json \\
    --sonar-url  http://localhost:9000 \\
    --sonar-token <your-token> \\
    --key-map  config/sonar_key_map.json

# Dry-run with synthetic data (no SonarQube required)
python enrich_dataset.py \\
    --input  input/dataset.json \\
    --output input/enriched_dataset.json \\
    --dry-run

Dependencies: requests, numpy  (pip install requests numpy)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore  # caught at runtime if --dry-run not set

logger = logging.getLogger("enrich_dataset")

# ---------------------------------------------------------------------------
# 1. SonarQube metric keys and their RMAV mapping
# ---------------------------------------------------------------------------

#: Ordered list of metric keys fetched from the SonarQube API.
#: The order here defines the canonical gnn_features indices 18-23.
SONAR_METRIC_KEYS: List[str] = [
    "complexity",                 # index 18 → R – Reliability
    "coverage",                   # index 19 → R – Reliability
    "sqale_debt_ratio",           # index 20 → M – Maintainability
    "bugs",                       # index 21 → R – Reliability (defect count)
    "vulnerabilities",            # index 22 → V – Vulnerability
    "duplicated_lines_density",   # index 23 → M – Maintainability
]

#: Human-readable metadata for each metric (used in reports and schema docs).
METRIC_META: Dict[str, dict] = {
    "complexity": {
        "label": "Cyclomatic Complexity",
        "rmav_dimension": "R",
        "normalization": "minmax",
        "higher_is_worse": True,
    },
    "coverage": {
        "label": "Code Coverage (%)",
        "rmav_dimension": "R",
        "normalization": "direct",      # already 0–100, mapped to 0–1
        "higher_is_worse": False,       # low coverage → high risk
    },
    "sqale_debt_ratio": {
        "label": "Technical Debt Ratio (%)",
        "rmav_dimension": "M",
        "normalization": "direct",
        "higher_is_worse": True,
    },
    "bugs": {
        "label": "Bug Count",
        "rmav_dimension": "R",
        "normalization": "minmax",
        "higher_is_worse": True,
    },
    "vulnerabilities": {
        "label": "Vulnerability Count",
        "rmav_dimension": "V",
        "normalization": "minmax",
        "higher_is_worse": True,
    },
    "duplicated_lines_density": {
        "label": "Duplicated Lines Density (%)",
        "rmav_dimension": "M",
        "normalization": "direct",
        "higher_is_worse": True,
    },
}


# ---------------------------------------------------------------------------
# 2. Data classes
# ---------------------------------------------------------------------------

@dataclass
class RawSonarMetrics:
    """Raw (un-normalised) values exactly as returned by the SonarQube API."""
    component_key: str
    complexity: float = 0.0
    coverage: float = 0.0
    sqale_debt_ratio: float = 0.0
    bugs: float = 0.0
    vulnerabilities: float = 0.0
    duplicated_lines_density: float = 0.0
    fetch_error: Optional[str] = None  # non-None if the API call failed


@dataclass
class NormalisedSonarMetrics:
    """
    Normalised metrics in [0, 1], ready for injection into the graph model.

    All values are normalised so that 1.0 = worst (highest risk) and
    0.0 = best, *regardless of whether the raw metric is "higher-is-better"
    or "higher-is-worse"*.  This convention matches the existing topological
    metric normalisation in Step 2.
    """
    component_key: str
    complexity_norm: float = 0.0          # index 18
    coverage_risk_norm: float = 0.0       # index 19  (1 − coverage/100)
    debt_ratio_norm: float = 0.0          # index 20
    bugs_norm: float = 0.0                # index 21
    vulnerabilities_norm: float = 0.0     # index 22
    duplication_norm: float = 0.0         # index 23

    def as_gnn_feature_slice(self) -> List[float]:
        """Return indices 18-23 of the extended GNN feature vector."""
        return [
            self.complexity_norm,
            self.coverage_risk_norm,
            self.debt_ratio_norm,
            self.bugs_norm,
            self.vulnerabilities_norm,
            self.duplication_norm,
        ]


# ---------------------------------------------------------------------------
# 3. SonarQube client
# ---------------------------------------------------------------------------

class SonarQubeClient:
    """
    Thin wrapper around the SonarQube Web API.

    Only the ``/api/measures/component`` endpoint is used, which is
    available in SonarQube Community Edition (v7.9+).
    """

    # Maximum component keys per bulk request (SonarQube default page size)
    _BULK_LIMIT = 100

    def __init__(self, base_url: str, token: str, timeout: int = 30):
        if requests is None:
            raise ImportError("The 'requests' package is required. "
                              "Install it with: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (token, "")   # token-based auth: token as username
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_metrics(
        self,
        component_keys: List[str],
        metric_keys: Optional[List[str]] = None,
    ) -> Dict[str, RawSonarMetrics]:
        """
        Fetch metrics for a list of SonarQube component keys.

        Returns a dict keyed by component_key.  Missing components receive
        a ``RawSonarMetrics`` with ``fetch_error`` set.
        """
        if metric_keys is None:
            metric_keys = SONAR_METRIC_KEYS

        results: Dict[str, RawSonarMetrics] = {}

        # SonarQube's components/search endpoint accepts a comma-separated list
        # of component keys.  Batch in chunks to stay within URL-length limits.
        for chunk_start in range(0, len(component_keys), self._BULK_LIMIT):
            chunk = component_keys[chunk_start: chunk_start + self._BULK_LIMIT]
            self._fetch_chunk(chunk, metric_keys, results)

        # Fill in any component keys that produced no result
        for key in component_keys:
            if key not in results:
                results[key] = RawSonarMetrics(
                    component_key=key,
                    fetch_error="Component not found in SonarQube",
                )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_chunk(
        self,
        component_keys: List[str],
        metric_keys: List[str],
        results: Dict[str, RawSonarMetrics],
    ) -> None:
        """Fetch one batch of components and populate *results* in place."""
        url = f"{self.base_url}/api/measures/components"
        params = {
            "componentKeys": ",".join(component_keys),
            "metricKeys": ",".join(metric_keys),
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network error, auth failure, etc.
            error_msg = str(exc)
            logger.error("SonarQube API error for chunk %s: %s",
                         component_keys[:3], error_msg)
            for key in component_keys:
                results[key] = RawSonarMetrics(
                    component_key=key, fetch_error=error_msg
                )
            return

        for component in payload.get("components", []):
            key = component["key"]
            raw = RawSonarMetrics(component_key=key)
            for measure in component.get("measures", []):
                metric = measure["metric"]
                value_str = measure.get("value", "0")
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    value = 0.0
                if hasattr(raw, metric):
                    setattr(raw, metric, value)
            results[key] = raw


# ---------------------------------------------------------------------------
# 4. Dry-run synthetic data generator
# ---------------------------------------------------------------------------

class DryRunClient:
    """
    Generates synthetic SonarQube metrics using a seeded RNG.

    Distributions are designed to be realistic:
    - Complexity:    log-normal (most apps simple, some very complex)
    - Coverage:      beta-like (commonly 40-80 %)
    - Debt ratio:    uniform-low (0-15 %)
    - Bugs:          Poisson(λ=3)
    - Vulnerabilities: Poisson(λ=1)
    - Duplication:   uniform (0-20 %)
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def fetch_metrics(
        self,
        component_keys: List[str],
        metric_keys: Optional[List[str]] = None,  # ignored; always returns all 6
    ) -> Dict[str, RawSonarMetrics]:
        results: Dict[str, RawSonarMetrics] = {}
        for key in component_keys:
            results[key] = RawSonarMetrics(
                component_key=key,
                complexity=float(self._rng.lognormal(mean=2.5, sigma=0.8)),
                coverage=float(np.clip(self._rng.normal(60, 18), 0, 100)),
                sqale_debt_ratio=float(np.clip(self._rng.exponential(scale=4), 0, 50)),
                bugs=float(self._rng.poisson(lam=3)),
                vulnerabilities=float(self._rng.poisson(lam=1)),
                duplicated_lines_density=float(np.clip(
                    self._rng.uniform(0, 20), 0, 100
                )),
            )
        return results


# ---------------------------------------------------------------------------
# 5. Normalisation
# ---------------------------------------------------------------------------

class MetricNormaliser:
    """
    Fits min-max scalers over all application metrics and transforms them
    to [0, 1] risk scores (higher = more risky in every dimension).
    """

    def __init__(self, raw_metrics: Dict[str, RawSonarMetrics]):
        self._raw = raw_metrics
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}
        self._fit()

    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Compute per-metric min and max over all non-error components."""
        valid = [m for m in self._raw.values() if m.fetch_error is None]
        for key in SONAR_METRIC_KEYS:
            values = [getattr(m, key, 0.0) for m in valid]
            self._min[key] = min(values) if values else 0.0
            self._max[key] = max(values) if values else 1.0

    def _minmax(self, value: float, key: str) -> float:
        lo, hi = self._min[key], self._max[key]
        if math.isclose(lo, hi):
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def _direct_pct(self, value: float) -> float:
        """Normalise a percentage (0-100) to (0-1)."""
        return float(np.clip(value / 100.0, 0.0, 1.0))

    # ------------------------------------------------------------------

    def transform(self, raw: RawSonarMetrics) -> NormalisedSonarMetrics:
        """
        Convert a single ``RawSonarMetrics`` to ``NormalisedSonarMetrics``.

        If ``raw.fetch_error`` is set, all scores are 0.0 (unknown → neutral).
        """
        if raw.fetch_error is not None:
            return NormalisedSonarMetrics(component_key=raw.component_key)

        # coverage risk: low coverage = high risk, so invert
        coverage_risk = 1.0 - self._direct_pct(raw.coverage)

        return NormalisedSonarMetrics(
            component_key=raw.component_key,
            complexity_norm=self._minmax(raw.complexity, "complexity"),
            coverage_risk_norm=coverage_risk,
            debt_ratio_norm=self._direct_pct(raw.sqale_debt_ratio),
            bugs_norm=self._minmax(raw.bugs, "bugs"),
            vulnerabilities_norm=self._minmax(raw.vulnerabilities, "vulnerabilities"),
            duplication_norm=self._direct_pct(raw.duplicated_lines_density),
        )

    def transform_all(self) -> Dict[str, NormalisedSonarMetrics]:
        return {key: self.transform(raw) for key, raw in self._raw.items()}


# ---------------------------------------------------------------------------
# 6. Key mapping (dataset app ID ↔ SonarQube component key)
# ---------------------------------------------------------------------------

def load_key_map(path: Optional[str], applications: List[dict]) -> Dict[str, str]:
    """
    Return a dict mapping  dataset app ID  →  SonarQube component key.

    If *path* is provided, load from a JSON file of the form:
        { "A0": "my.project:app-0", "A1": "my.project:app-1", ... }

    If *path* is None (dry-run mode), auto-generate identity mapping
        { "A0": "A0", "A1": "A1", ... }
    which works because DryRunClient accepts arbitrary keys.
    """
    if path is None:
        return {app["id"]: app["id"] for app in applications}

    key_map_path = Path(path)
    if not key_map_path.exists():
        raise FileNotFoundError(
            f"Key map file not found: {key_map_path}\n"
            "Create a JSON file mapping dataset app IDs to SonarQube "
            "component keys, e.g.:\n"
            '{ "A0": "com.example:service-alpha", "A1": "com.example:service-beta" }'
        )
    with key_map_path.open() as f:
        return json.load(f)


def build_reverse_key_map(key_map: Dict[str, str]) -> Dict[str, str]:
    """Reverse: SonarQube component key → dataset app ID."""
    reverse: Dict[str, str] = {}
    for app_id, sonar_key in key_map.items():
        if sonar_key in reverse:
            logger.warning(
                "Duplicate SonarQube key '%s' mapped from both '%s' and '%s'",
                sonar_key, reverse[sonar_key], app_id,
            )
        reverse[sonar_key] = app_id
    return reverse


# ---------------------------------------------------------------------------
# 7. Dataset enrichment
# ---------------------------------------------------------------------------

def enrich_dataset(
    dataset: dict,
    normalised: Dict[str, NormalisedSonarMetrics],
    raw: Dict[str, RawSonarMetrics],
    key_map: Dict[str, str],
) -> dict:
    """
    Return a new dataset dict with ``static_analysis`` and ``gnn_features``
    blocks added to each application node.

    Schema added to each application
    ---------------------------------
    {
      "id": "A0",
      "name": "App-0",
      ...original fields...,

      "static_analysis": {
        "source": "sonarqube",
        "component_key": "com.example:service-alpha",
        "fetch_error": null,
        "raw": {
          "complexity": 14.0,
          "coverage": 72.3,
          "sqale_debt_ratio": 3.1,
          "bugs": 2.0,
          "vulnerabilities": 0.0,
          "duplicated_lines_density": 4.5
        },
        "normalised": {
          "complexity_norm": 0.42,
          "coverage_risk_norm": 0.28,
          "debt_ratio_norm": 0.06,
          "bugs_norm": 0.33,
          "vulnerabilities_norm": 0.0,
          "duplication_norm": 0.045
        },
        "rmav_hints": {
          "R": ["complexity_norm", "coverage_risk_norm", "bugs_norm"],
          "M": ["debt_ratio_norm", "duplication_norm"],
          "V": ["vulnerabilities_norm"]
        }
      },

      "gnn_features": {
        "description": "Indices 18-23 of the extended 24-dim feature vector",
        "values": [0.42, 0.28, 0.06, 0.33, 0.0, 0.045]
      }
    }
    """
    import copy
    enriched = copy.deepcopy(dataset)

    enriched_count = 0
    error_count = 0

    for app in enriched.get("applications", []):
        app_id = app["id"]
        sonar_key = key_map.get(app_id)

        if sonar_key is None:
            logger.warning("No SonarQube key mapped for app '%s' — skipping", app_id)
            app["static_analysis"] = _missing_sa_block(app_id, "No key mapping")
            app["gnn_features"] = _zero_gnn_block()
            continue

        norm = normalised.get(sonar_key)
        raw_m = raw.get(sonar_key)

        if norm is None or raw_m is None:
            logger.warning("No metrics returned for '%s' (%s)", app_id, sonar_key)
            app["static_analysis"] = _missing_sa_block(app_id, "No metrics returned")
            app["gnn_features"] = _zero_gnn_block()
            error_count += 1
            continue

        app["static_analysis"] = {
            "source": "sonarqube",
            "component_key": sonar_key,
            "fetch_error": raw_m.fetch_error,
            "raw": {
                "complexity": raw_m.complexity,
                "coverage": raw_m.coverage,
                "sqale_debt_ratio": raw_m.sqale_debt_ratio,
                "bugs": raw_m.bugs,
                "vulnerabilities": raw_m.vulnerabilities,
                "duplicated_lines_density": raw_m.duplicated_lines_density,
            },
            "normalised": {
                "complexity_norm": round(norm.complexity_norm, 6),
                "coverage_risk_norm": round(norm.coverage_risk_norm, 6),
                "debt_ratio_norm": round(norm.debt_ratio_norm, 6),
                "bugs_norm": round(norm.bugs_norm, 6),
                "vulnerabilities_norm": round(norm.vulnerabilities_norm, 6),
                "duplication_norm": round(norm.duplication_norm, 6),
            },
            "rmav_hints": {
                "R": ["complexity_norm", "coverage_risk_norm", "bugs_norm"],
                "M": ["debt_ratio_norm", "duplication_norm"],
                "V": ["vulnerabilities_norm"],
            },
        }

        app["gnn_features"] = {
            "description": "Indices 18-23 of the extended 24-dim feature vector",
            "indices": {
                18: "complexity_norm",
                19: "coverage_risk_norm",
                20: "debt_ratio_norm",
                21: "bugs_norm",
                22: "vulnerabilities_norm",
                23: "duplication_norm",
            },
            "values": [round(v, 6) for v in norm.as_gnn_feature_slice()],
        }

        if raw_m.fetch_error:
            error_count += 1
        else:
            enriched_count += 1

    # Stamp the metadata block so downstream tools can detect enrichment
    enriched.setdefault("metadata", {})["static_analysis"] = {
        "source": "sonarqube",
        "metric_keys": SONAR_METRIC_KEYS,
        "gnn_feature_indices": "18-23",
        "enriched_count": enriched_count,
        "error_count": error_count,
        "metric_metadata": METRIC_META,
    }

    logger.info(
        "Enrichment complete: %d applications enriched, %d errors",
        enriched_count, error_count,
    )
    return enriched


# ---------------------------------------------------------------------------
# 8. Helper builders for missing/error cases
# ---------------------------------------------------------------------------

def _missing_sa_block(app_id: str, reason: str) -> dict:
    return {
        "source": "sonarqube",
        "component_key": None,
        "fetch_error": reason,
        "raw": {k: 0.0 for k in SONAR_METRIC_KEYS},
        "normalised": {
            "complexity_norm": 0.0,
            "coverage_risk_norm": 0.0,
            "debt_ratio_norm": 0.0,
            "bugs_norm": 0.0,
            "vulnerabilities_norm": 0.0,
            "duplication_norm": 0.0,
        },
        "rmav_hints": {
            "R": ["complexity_norm", "coverage_risk_norm", "bugs_norm"],
            "M": ["debt_ratio_norm", "duplication_norm"],
            "V": ["vulnerabilities_norm"],
        },
    }


def _zero_gnn_block() -> dict:
    return {
        "description": "Indices 18-23 of the extended 24-dim feature vector",
        "indices": {
            18: "complexity_norm",
            19: "coverage_risk_norm",
            20: "debt_ratio_norm",
            21: "bugs_norm",
            22: "vulnerabilities_norm",
            23: "duplication_norm",
        },
        "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


# ---------------------------------------------------------------------------
# 9. Validation & reporting
# ---------------------------------------------------------------------------

def print_enrichment_report(
    dataset: dict,
    normalised: Dict[str, NormalisedSonarMetrics],
    key_map: Dict[str, str],
) -> None:
    """Print a summary table to stdout for quick visual inspection."""
    print("\n" + "=" * 72)
    print("  SONARQUBE ENRICHMENT REPORT")
    print("=" * 72)
    print(
        f"{'App ID':<8} {'SQ Key':<28} "
        f"{'Cpx':>5} {'Cov%':>6} {'Debt%':>6} "
        f"{'Bugs':>5} {'Vuln':>5} {'Dup%':>5}"
    )
    print("-" * 72)

    for app in dataset.get("applications", []):
        app_id = app["id"]
        sonar_key = key_map.get(app_id, "—")
        norm = normalised.get(sonar_key)

        if norm is None:
            row = f"{app_id:<8} {sonar_key:<28}" + " " * 34 + " (missing)"
        else:
            sa = app.get("static_analysis", {})
            raw = sa.get("raw", {})
            row = (
                f"{app_id:<8} {sonar_key:<28} "
                f"{raw.get('complexity', 0):>5.0f} "
                f"{raw.get('coverage', 0):>6.1f} "
                f"{raw.get('sqale_debt_ratio', 0):>6.1f} "
                f"{raw.get('bugs', 0):>5.0f} "
                f"{raw.get('vulnerabilities', 0):>5.0f} "
                f"{raw.get('duplicated_lines_density', 0):>5.1f}"
            )
        print(row)

    meta = dataset.get("metadata", {}).get("static_analysis", {})
    print("=" * 72)
    print(
        f"  Enriched: {meta.get('enriched_count', '?')}  |  "
        f"Errors: {meta.get('error_count', '?')}"
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# 10. CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enrich_dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--input", "-i",
        default="input/dataset.json",
        help="Path to the original dataset JSON (default: input/dataset.json)",
    )
    io.add_argument(
        "--output", "-o",
        default="input/enriched_dataset.json",
        help="Path to write the enriched dataset (default: input/enriched_dataset.json)",
    )

    sq = parser.add_argument_group("SonarQube connection")
    sq.add_argument(
        "--sonar-url",
        default="http://localhost:9000",
        help="SonarQube base URL (default: http://localhost:9000)",
    )
    sq.add_argument(
        "--sonar-token",
        default=None,
        help="SonarQube user/project token (or set SONAR_TOKEN env var)",
    )
    sq.add_argument(
        "--key-map",
        default=None,
        metavar="PATH",
        help=(
            "JSON file mapping dataset app IDs to SonarQube component keys. "
            "Required when not using --dry-run. "
            "Example: config/sonar_key_map.json"
        ),
    )
    sq.add_argument(
        "--timeout",
        type=int, default=30,
        help="HTTP timeout for SonarQube API calls in seconds (default: 30)",
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate synthetic SonarQube metrics instead of calling the API. "
            "Useful for testing the pipeline without a running SonarQube instance."
        ),
    )
    misc.add_argument(
        "--seed",
        type=int, default=42,
        help="RNG seed for --dry-run synthetic data (default: 42)",
    )
    misc.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress the enrichment summary table",
    )
    misc.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load dataset ──────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    logger.info("Loading dataset from %s", input_path)
    with input_path.open() as f:
        dataset = json.load(f)

    applications = dataset.get("applications", [])
    logger.info("Dataset contains %d applications", len(applications))

    # ── Build key map ─────────────────────────────────────────────────────
    try:
        key_map = load_key_map(
            path=None if args.dry_run else args.key_map,
            applications=applications,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    sonar_component_keys = list(key_map.values())

    # ── Fetch metrics ─────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY-RUN mode: generating synthetic SonarQube metrics (seed=%d)", args.seed)
        client = DryRunClient(seed=args.seed)
    else:
        token = args.sonar_token
        if token is None:
            import os
            token = os.environ.get("SONAR_TOKEN")
        if token is None:
            logger.error(
                "No SonarQube token provided. "
                "Use --sonar-token or set the SONAR_TOKEN environment variable."
            )
            return 1
        logger.info("Connecting to SonarQube at %s", args.sonar_url)
        client = SonarQubeClient(
            base_url=args.sonar_url,
            token=token,
            timeout=args.timeout,
        )

    logger.info("Fetching metrics for %d components…", len(sonar_component_keys))
    t0 = time.monotonic()
    raw_metrics = client.fetch_metrics(sonar_component_keys)
    elapsed = time.monotonic() - t0
    logger.info("Metrics fetched in %.2f s", elapsed)

    # ── Normalise ─────────────────────────────────────────────────────────
    logger.info("Normalising metrics across all applications…")
    normaliser = MetricNormaliser(raw_metrics)
    normalised_metrics = normaliser.transform_all()

    # ── Enrich dataset ────────────────────────────────────────────────────
    logger.info("Injecting static analysis data into dataset…")
    enriched = enrich_dataset(dataset, normalised_metrics, raw_metrics, key_map)

    # ── Write output ──────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(enriched, f, indent=2)
    logger.info("Enriched dataset written to %s", output_path)

    # ── Report ────────────────────────────────────────────────────────────
    if not args.no_report:
        print_enrichment_report(enriched, normalised_metrics, key_map)

    return 0


if __name__ == "__main__":
    sys.exit(main())
